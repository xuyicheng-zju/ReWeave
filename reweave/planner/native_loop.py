"""Native decision sessions driving the shared executable W/C/L runtime."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import stat
import time
import uuid
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from jsonschema import ValidationError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from reweave.memory.notes import planner_memory_guidance
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    AGENT_EXECUTION_FAILURE,
    API_ERROR,
    PLANNER_TIMEOUT,
    PlannerResult,
)
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.planner.codex_connection import codex_connection_config
from reweave.planner.cursor_connection import cursor_model_spec
from reweave.planner.decisions import DECISION_INSTRUCTIONS, decision_error
from reweave.planner.native_context import (
    EpisodeRecord,
    baseline_context,
    prepare_native,
)
from reweave.planner.providers import (
    ClaudeDecisionSession,
    CodexDecisionSession,
    CursorDecisionSession,
    NativeCompletedInvalidDecision,
    NativeProtocolError,
    NativeSessionError,
    NativeTransportError,
    NativeTurnFailed,
    NativeTurnRecoveryRequired,
)
from reweave.runtime.adaptation import HarnessAdaptation
from reweave.runtime.capability import (
    CandidateExecutionFailure,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
    ProgramDeadlineExceeded,
)
from reweave.runtime.kernel import async_timeout, validate_json
from reweave.runtime.model_budget import (
    ModelBudgetExceeded,
    ModelCallBudget,
    ModelServiceBudgetExceeded,
    ModelServiceFailure,
)
from reweave.utils.persistence import atomic_json, atomic_text


def _read_recovery_control(path, identity):
    """Read only the fixed Host control file, with bounded identity-bound data."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        attributes = os.fstat(descriptor)
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_size > 16384:
            raise ValueError("Native recovery control must be a bounded regular file")
        raw = os.read(descriptor, 16385)
    finally:
        os.close(descriptor)
    if len(raw) > 16384:
        raise ValueError("Native recovery control exceeds its bound")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "action",
        "session_id",
        "request_id",
        "turn_id",
        "nonce",
    }:
        raise ValueError("Native recovery control has unexpected fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["action"] != "resume"
        or any(value[key] != expected for key, expected in identity.items())
        or not isinstance(value["nonce"], str)
        or not re.fullmatch("[0-9a-f]{32}", value["nonce"])
    ):
        raise ValueError("Native recovery control differs from the blocked identity")
    return value


class NativeBundlePlanner:
    """Keep task control in L; a native CLI owns each individual decision turn."""

    controller_request_unit = "native_turn"
    baseline_context = staticmethod(baseline_context)

    def __init__(
        self,
        *,
        output_dir,
        provider="codex",
        codex_auth_mode="auth",
        model=None,
        reasoning_effort="medium",
        service_tier="default",
        adaptation_mode="static",
        online_workflows=False,
        session_factory=None,
        timeout_s=None,
    ):
        if provider not in {"codex", "claude", "cursor"}:
            raise ValueError("No native session adapter is installed for this provider")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError(
                "Native reasoning_effort must be low, medium, high or xhigh"
            )
        if service_tier not in {"default", "fast"}:
            raise ValueError("Native service_tier must be default or fast")
        if adaptation_mode not in {"static", "adaptive"}:
            raise ValueError("adaptation_mode must be static or adaptive")
        self.output_dir = Path(output_dir)
        self.provider = provider
        self.adaptation_mode = adaptation_mode
        if type(online_workflows) is not bool:
            raise ValueError("online_workflows must be a boolean")
        self.online_workflows = online_workflows
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort
        self.codex_auth_mode = None
        if provider == "codex":
            codex_connection_config(codex_auth_mode)
            self.codex_auth_mode = codex_auth_mode
            self.model = CodexDecisionSession.resolve_model(model)
            default_factory = partial(
                CodexDecisionSession,
                model=self.model,
                codex_auth_mode=codex_auth_mode,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
            )
        elif provider == "claude":
            self.model = ClaudeDecisionSession.resolve_model(model)
            default_factory = partial(
                ClaudeDecisionSession,
                model=self.model,
                reasoning_effort=reasoning_effort,
                editable_paths=(
                    ("harness/task-family/**",) if adaptation_mode == "adaptive" else ()
                ),
            )
        else:
            self.model = CursorDecisionSession.resolve_model(model)
            self._cursor_spec = cursor_model_spec(self.model)
            if service_tier == "fast" and not self._cursor_spec["fast"]:
                raise ValueError(
                    "Cursor fast tier requires a fast model id such as grok-4.7-xhigh-fast"
                )
            # A catalog id that names its effort overrides the requested effort.
            self.reasoning_effort = (
                self._cursor_spec["reasoning_effort"] or reasoning_effort
            )
            default_factory = partial(
                CursorDecisionSession,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                service_tier=service_tier,
            )
        self.session_factory = (
            default_factory if session_factory is None else session_factory
        )
        self._bundle_configuration = None
        self.timeout_s = timeout_s

    def runtime_configuration(self):
        configuration = self._provider_configuration()
        if self.online_workflows:
            configuration["online_workflows"] = True
        return configuration

    def _provider_configuration(self):
        """Record selected settings; native handshake metadata stays in its event log."""
        if self.provider == "claude":
            return {
                "provider": "claude",
                "model": self.model,
                "model_provider": "anthropic",
                "protocol": "claude-agent-sdk",
                "thinking": "disabled",
                "effort": self.reasoning_effort,
                "underlying_model_attempts": "unavailable",
                "permission_mode": "dontAsk",
                "tools": "read-only"
                if self.adaptation_mode == "static"
                else "read, edit harness/task-family",
                "native_recovery": {
                    "revision": "claude-agent-sdk-v1",
                    "automatic_failed_turn_continuations": 0,
                    "unconfirmed_prompt_behavior": "do_not_replay",
                    "blocked_behavior": "park_until_operator_stops",
                    "budget_and_deadline_reset": False,
                },
            }
        if self.provider == "cursor":
            return {
                "provider": "cursor",
                "model": self.model,
                "model_parameters": {
                    "model": self._cursor_spec["model"],
                    "reasoning_effort": self.reasoning_effort,
                    "fast": self._cursor_spec["fast"],
                    "context": self._cursor_spec["context"],
                    "mode": "ask",
                },
                "authentication": "cursor_login",
                "billing": "subscription",
                "protocol": "cursor-acp",
                "executable": "agent",
                "service_tier": self.service_tier,
                "underlying_model_attempts": "unavailable",
                "approval_policy": "reject",
                "native_recovery": {
                    "revision": "cursor-acp-v1",
                    "automatic_failed_turn_continuations": 0,
                    "unconfirmed_prompt_behavior": "do_not_replay",
                    "blocked_behavior": "wait_for_identity_bound_host_resume",
                    "resume_control": "native/recovery/resume.json",
                    "budget_and_deadline_reset": False,
                },
            }
        connection = codex_connection_config(self.codex_auth_mode)
        return {
            "provider": "codex",
            "model": self.model,
            "model_provider": connection["model_provider"],
            "codex_auth_mode": self.codex_auth_mode,
            "base_url": connection.get(
                f"model_providers.{connection['model_provider']}.base_url"
            ),
            "service_tier": self.service_tier,
            "protocol": "stdio-jsonrpc",
            "underlying_model_attempts": "unavailable",
            "approval_policy": "never",
            "sandbox": "workspace-write",
            "native_recovery": {
                "revision": "same-session-recovery-v1",
                "automatic_failed_turn_continuations": CodexDecisionSession.MAX_CONTINUATIONS,
                "continuation_charges_controller_attempt": True,
                "transport_reconnect_charges_controller_attempt": False,
                "blocked_behavior": "wait_for_identity_bound_host_resume",
                "resume_control": "native/recovery/resume.json",
                "budget_and_deadline_reset": False,
            },
        }

    def sampling_settings(self):
        return {"reasoning_effort": self.reasoning_effort}

    def bind_bundle(self, bundle, *, episode_limits):
        self._bundle_configuration = (bundle, episode_limits)

    def solve(
        self,
        *,
        system_prompt,
        user_message,
        toolkit,
        max_turns,
        input_queue=None,
        dashboard_interaction=None,
    ):
        if self._bundle_configuration is None:
            from reweave.evolution.bundle import (
                AgentBundle,
                RuntimeConfiguration,
                baseline_loop,
            )
            from reweave.runtime.kernel import EpisodeLimits

            physical = toolkit.control_step_budget()
            runtime = RuntimeConfiguration(
                revision_id=uuid.uuid4().hex,
                dependencies={},
                tool_api=toolkit.get_tools_contract_spec(),
                models=self.runtime_configuration(),
                memory=(),
                evaluator={"authority": "toolkit official evaluator"},
                environment={"adapter": type(toolkit).__module__},
                configuration={
                    "controller_request_unit": "native_turn",
                    "adaptation_mode": self.adaptation_mode,
                },
                wait_steps=0,
                physical_steps=physical["limit"] if physical else None,
            )
            self.bind_bundle(
                AgentBundle(
                    record_id=uuid.uuid4().hex,
                    runtime=runtime,
                    context=baseline_context(),
                    loop=baseline_loop(),
                ),
                episode_limits=EpisodeLimits(
                    program_calls=None, gateway_calls=None, wall_time_s=self.timeout_s
                ),
            )
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError(
                "Native episode execution does not accept interactive session replacement"
            )
        return asyncio.run(self._solve(system_prompt, user_message, toolkit, max_turns))

    async def _solve(self, system_prompt, user_message, toolkit, max_turns):
        bundle, limits = self._bundle_configuration
        directory = toolkit.state.output_dir / "native"
        workspace = directory / "workspace"
        record = EpisodeRecord(directory / "records")
        adaptation = HarnessAdaptation(
            bundle,
            candidate_directory=workspace / "harness",
            audit_directory=directory / "adaptation",
            mode=self.adaptation_mode,
            online_workflows=self.online_workflows,
        )
        runtime = BundleRuntime(
            bundle,
            toolkit,
            user_message,
            episode_limits=limits,
            adaptation=adaptation,
            online_workflows=self.online_workflows,
        )
        session = self.session_factory(directory / "provider")
        notes = getattr(toolkit, "episode_notes", None)
        closeout_s = limits.note_closeout_s if notes is not None else None
        budget = ModelCallBudget(
            max_turns - int(closeout_s is not None), directory / "native_turns.jsonl"
        )
        bind_budget = getattr(toolkit, "bind_model_budget", None)
        if bind_budget is not None:
            bind_budget(budget)
        prompt_path = workspace / "AGENTS.md"
        native_instructions = (
            DECISION_INSTRUCTIONS + "\n"
            "Physical intermediate observations are archived; the next decision receives the current observation once. Explicit image reads are retained. "
            "Image-return upper bounds for this adapter: "
            + json.dumps(toolkit.native_image_contract())
            + ". The complete next request may contain at most ten distinct images. "
            "Return task tool proposals as structured JSON; arguments_json contains the exact JSON object for each tool. The Host alone executes them. "
            "Decide directly from the supplied observation messages, images, tool feedback and task experience already in this session. "
            "Do not browse the repository, reread episode history, discover development skills or run a coding preflight for ordinary decisions. "
            "Use native file tools only for a necessary targeted harness revision or a specific missing historical fact; do not repeat reads of unchanged source. "
            "Keep the decision explanation brief; return structured proposals without routine progress narration. "
            "The complete task instructions and tool catalog are in this working directory's AGENTS.md. "
            "Use native session history for previous observations and accepted feedback; "
            "available_tools lists the exact currently allowed names. "
            "Harness source is in harness/. AGENTS.md is Host-owned and read-only. "
            + (
                "You may edit only task-family/ in this private W/C/L source; global/ is read-only and task-specific/ is Host-owned. Future calls activate settled revisions; "
                "an executing L changes only by returning {status: checkpoint, state: {...}}. "
                if self.adaptation_mode == "adaptive"
                else "C/L and global assets are fixed; write episode-private W through write_workflow. "
                if self.online_workflows
                else "This condition measures a static harness; do not modify source. "
            )
            + "Never invoke robot or model services from the native shell. All task effects use the returned proposals."
        )
        if self.online_workflows:
            from reweave.runtime.online_workflow import ONLINE_WORKFLOW_INSTRUCTIONS

            native_instructions += "\n\n" + ONLINE_WORKFLOW_INSTRUCTIONS
        guidance = planner_memory_guidance(
            toolkit, adaptation_mode=self.adaptation_mode
        )
        if guidance:
            native_instructions += "\n\n" + guidance
        instructions = system_prompt + "\n\n" + native_instructions
        # This is the initial session/root contract, not an API conversation.
        # Subsequent C inputs and native turns remain in their original records.
        planner_context_path = (
            toolkit.state.output_dir / "harness_context/planner_context.json"
        )
        planner_context = {
            "backend": "native",
            **self.runtime_configuration(),
            "reasoning_effort": self.sampling_settings()["reasoning_effort"],
            "model_settings": self.sampling_settings(),
            "controller_request_unit": self.controller_request_unit,
            "adaptation_mode": self.adaptation_mode,
            **({"online_workflows": True} if self.online_workflows else {}),
            "system_prompt": system_prompt,
            "effective_instructions": instructions,
            "task_message": user_message,
            "tools": runtime.specs(),
        }
        atomic_json(planner_context_path, planner_context)
        previous_guidance = None
        session_started = False
        request_lock = asyncio.Lock()

        recovery_directory = directory / "recovery"
        consumed_resumes = set()

        def check_wait_boundary():
            runtime.kernel.raise_if_failed()
            remaining = runtime.kernel.remaining_wall_time()
            if remaining is not None and remaining <= 0:
                scope = (
                    "episode"
                    if runtime.kernel._deadline is not None
                    and runtime.kernel._deadline <= time.monotonic()
                    else "program"
                )
                raise ProgramDeadlineExceeded(
                    "Native recovery exceeded the original deadline", scope=scope
                )

        async def wait_for_resume(error, request_id, *, starting=False):
            """Park the live episode; only an identity-bound Host control can retry."""
            if isinstance(error, NativeProtocolError):
                raise ModelServiceFailure(
                    f"Native protocol violation cannot resume; close without replay: {error}",
                    retryable=False,
                ) from error
            state_reader = getattr(session, "recovery_status", None)
            try:
                recovery = (
                    state_reader(request_id)
                    if state_reader is not None and not starting
                    else {}
                )
            except NativeSessionError:
                recovery = {}
            turn_id = recovery.get("turn_id")
            from reweave.evolution.infrastructure.campaign_slots import campaign_root

            campaign = campaign_root(directory)
            policy_path = (
                campaign / "scheduler/online/resource_policy.json" if campaign else None
            )
            policy = (
                json.loads(policy_path.read_text())
                if policy_path and policy_path.exists()
                else {}
            )
            wait_s = policy.get("native_failed_wait_s", 300 if campaign else None)
            if wait_s is not None and (
                type(wait_s) not in (int, float) or not 0 < wait_s <= 3600
            ):
                raise ValueError("Invalid native infrastructure wait limit")
            # Submitted/unknown turns retain their uncertainty boundary. Only a
            # failed provider turn or pre-decision startup may expire here.
            wait_deadline = (
                time.monotonic() + wait_s
                if wait_s
                and (
                    starting
                    or (
                        recovery.get("phase")
                        in {"failed", "recovery_blocked", "recovery_prepared"}
                        and recovery.get("reason") != "unsettled_native_effects"
                    )
                )
                else None
            )
            identity = {
                "session_id": session.session_id,
                "request_id": request_id,
                "turn_id": turn_id,
            }
            control = recovery_directory / "resume.json"
            blocked = {
                "schema_version": 1,
                "status": "blocked",
                **identity,
                # Not a model turn in progress: the Host stopped and waits for an
                # operator control bound to this exact request identity.
                "state": "parked_operator_required",
                "parked_at": time.time(),
                "error_type": type(error).__name__,
                "reason": getattr(error, "reason", str(error))[:1000],
                # The provider's own typed error (e.g. a bare Bad Request), unabridged
                # up to a bound, so the operator sees the cause and not a status.
                "provider_error": json.dumps(
                    getattr(error, "error", None), default=str
                )[:2000],
                "request_phase": recovery.get("phase"),
                # authorize_recovery admits only a verified failed turn; any other
                # phase needs the episode stopped, never a resume control.
                "operator_resumable": recovery.get("phase")
                in {"failed", "recovery_blocked"},
                "continuations": recovery.get("continuations"),
                "controller_attempts": budget.controller_attempts,
                "controller_limit": budget.maximum,
                "original_deadline": episode_deadline,
                "active_deadline": runtime.kernel.deadline(),
                "deadline_clock": "monotonic",
                "resume_control": str(control),
                "automatic_sampling": False,
                "automatic_replay": False,
                "infrastructure_wait_limit_s": wait_s if wait_deadline else None,
            }
            atomic_json(recovery_directory / "blocked.json", blocked)
            runtime.kernel._event("native_recovery_blocked", **blocked)
            last_rejected = None
            while True:
                check_wait_boundary()
                if wait_deadline is not None and time.monotonic() >= wait_deadline:
                    runtime.kernel._event(
                        "native_recovery_wait_expired",
                        **identity,
                        wait_s=wait_s,
                        physical_replay=False,
                    )
                    raise ModelServiceFailure(
                        "Known-failed native service wait expired; close this episode and use only the original infrastructure retry allowance",
                        retryable=True,
                    )
                try:
                    value = _read_recovery_control(control, identity)
                except (ValueError, OSError) as invalid:
                    rejection = (type(invalid).__name__, str(invalid))
                    if rejection != last_rejected:
                        runtime.kernel._event(
                            "native_recovery_control_rejected",
                            **identity,
                            error_type=type(invalid).__name__,
                            reason=str(invalid)[:1000],
                        )
                        last_rejected = rejection
                    value = None
                if value is not None:
                    last_rejected = None
                    nonce = value["nonce"]
                    receipt = recovery_directory / "resume_receipts" / f"{nonce}.json"
                    if nonce not in consumed_resumes and not receipt.exists():
                        # Consume durably before authorization. A process crash
                        # cannot turn an old operator command into another retry.
                        consumed_resumes.add(nonce)
                        outcome = {**value, "status": "requested"}
                        atomic_json(receipt, outcome)
                        try:
                            authorize = getattr(session, "authorize_recovery", None)
                            if not starting and authorize is not None:
                                try:
                                    authorization = await connected(
                                        lambda: authorize(
                                            request_id,
                                            turn_id,
                                            nonce,
                                            session_id=session.session_id,
                                        ),
                                        request_id,
                                    )
                                except NativeTurnRecoveryRequired:
                                    authorization = {"will_submit": True}
                            else:
                                authorization = {"will_submit": not starting}
                        except NativeSessionError as denied:
                            outcome.update(
                                status="rejected",
                                error_type=type(denied).__name__,
                                reason=str(denied)[:1000],
                            )
                            atomic_json(receipt, outcome)
                            runtime.kernel._event(
                                "native_recovery_resume_rejected", **outcome
                            )
                        else:
                            check_wait_boundary()
                            outcome.update(
                                status="authorized",
                                will_submit=authorization.get("will_submit", False),
                            )
                            atomic_json(receipt, outcome)
                            atomic_json(
                                recovery_directory / "blocked.json",
                                {**blocked, "status": "resuming", "nonce": nonce},
                            )
                            runtime.kernel._event("native_recovery_resumed", **outcome)
                            return authorization
                await asyncio.sleep(0.25)

        async def connected(operation, request_id):
            reconnects = 0
            while True:
                try:
                    check_wait_boundary()
                    return await operation()
                except NativeTransportError as exc:
                    # Provider state decides whether to resume, read a cached result,
                    # or report unknown submission. Never create a replacement turn.
                    reconnects += 1
                    runtime.kernel._event(
                        "native_connection_wait",
                        request_id=request_id,
                        reconnects=reconnects,
                        error=str(exc),
                    )
                    await asyncio.sleep(min(2 ** min(reconnects, 5), 30))

        async def request(context, program_call_id):
            nonlocal previous_guidance, session_started
            async with request_lock:
                identity = uuid.uuid4().hex
                context_guidance = context["instructions"]
                if context_guidance.startswith(system_prompt):
                    context_guidance = context_guidance[len(system_prompt) :].lstrip()
                guidance_text = (
                    "# Episode instructions\n\n"
                    + instructions
                    + "\n\n# Interpreting task constraints\n\n"
                    + "Follow explicit task rules and Host execution contracts. "
                    "Treat evolved guidance and experience as revisable advice, not "
                    "an independent source of task prohibitions. Distinguish a "
                    "prohibited outcome from actions that may cause it; preserve "
                    "the rule's scope and conditions. Do not infer permission from "
                    "an absent checker or relax an explicit prohibition. When "
                    "guidance conflicts with authoritative rules, follow those "
                    "rules and report the discrepancy in available feedback. "
                    "Unresolved ambiguity remains uncertainty, not a new rule."
                    + "\n\n# Context guidance and task experience\n\n"
                    + context_guidance
                    + "\n\n# Task\n\n"
                    + user_message
                    + "\n\n# Tool catalog\n\n```json\n"
                    + json.dumps(
                        context["tools"], ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n```\n"
                )
                guidance_changed = guidance_text != previous_guidance
                if guidance_changed:
                    atomic_text(prompt_path, guidance_text)
                    prompt_path.chmod(0o444)
                delivery_context = context
                if not session_started:
                    # The fixed task is already in AGENTS.md. Remove its exact
                    # initial copy, retaining all observation and tool feedback.
                    delivery_context = {
                        **context,
                        "messages": json.loads(json.dumps(context["messages"])),
                    }
                    for message in delivery_context["messages"]:
                        message["parts"] = [
                            part
                            for part in message["parts"]
                            if not (
                                part.get("part_kind") == "user-prompt"
                                and part.get("content") == user_message
                            )
                        ]
                if not session_started:
                    planner_context["effective_instructions"] = guidance_text
                    atomic_json(planner_context_path, planner_context)
                    while True:
                        try:
                            await connected(
                                lambda: session.start(
                                    guidance_text,
                                    workspace,
                                ),
                                "session_start",
                            )
                        except NativeSessionError as exc:
                            await wait_for_resume(exc, identity, starting=True)
                        else:
                            session_started = True
                            break
                text, images = prepare_native(
                    delivery_context,
                    record.directory / "images",
                    limits.request_limits,
                    include_tools=False,
                )
                if guidance_changed and previous_guidance is not None:
                    text = (
                        "Host updated AGENTS.md at this settled call boundary. Apply this guidance delta:\n"
                        + "".join(
                            difflib.unified_diff(
                                previous_guidance.splitlines(keepends=True),
                                guidance_text.splitlines(keepends=True),
                                fromfile="previous/AGENTS.md",
                                tofile="current/AGENTS.md",
                            )
                        )
                        + "\n\n"
                        + text
                    )
                specs = {item["name"]: item for item in context["tools"]}
                schema = {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "arguments_json": {"type": "string"},
                                },
                                "required": ["name", "arguments_json"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["text", "tool_calls"],
                    "additionalProperties": False,
                }
                record.append("Prepared observation", identity, text, images)
                decision_started = time.monotonic() if self.online_workflows else None
                if context.get("request_kind") == "note_closeout":
                    budget.maximum = min(max_turns, budget.controller_attempts + 1)
                elif notes is not None and notes.has_writes:
                    budget.maximum = max_turns
                token = budget.controller_kind.set(
                    context.get("request_kind", "planner")
                )
                try:
                    charge_turn = True
                    while True:
                        check_wait_boundary()
                        try:
                            if charge_turn:
                                runtime.kernel.charge_workflow("model")
                            with (
                                budget.request("controller", request_id=identity)
                                if charge_turn
                                else nullcontext()
                            ):
                                decision = await connected(
                                    lambda: session.decide(
                                        text, images, schema, identity
                                    ),
                                    identity,
                                )
                        except NativeTurnRecoveryRequired as exc:
                            # A new provider turn is a new charged attempt. Pure
                            # transport reconnects stay inside the prior context.
                            charge_turn = True
                            runtime.kernel._event(
                                "native_turn_continuation",
                                request_id=identity,
                                session_id=session.session_id,
                                continuation=exc.continuation,
                                reason=exc.reason,
                            )
                            check_wait_boundary()
                            delay_s = min(2 ** min(exc.continuation, 5), 30)
                            remaining = runtime.kernel.remaining_wall_time()
                            if remaining is not None:
                                delay_s = min(delay_s, max(0.0, remaining))
                            runtime.kernel._event(
                                "native_continuation_backoff",
                                request_id=identity,
                                continuation=exc.continuation,
                                delay_s=delay_s,
                            )
                            await asyncio.sleep(delay_s)
                            check_wait_boundary()
                        except NativeCompletedInvalidDecision as exc:
                            runtime.kernel._event(
                                "native_completed_invalid_decision",
                                request_id=identity,
                                session_id=session.session_id,
                                reason=str(exc)[:1000],
                                physical_replay=False,
                            )
                            raise CandidateExecutionFailure(
                                "The model completed a decision without a valid executable proposal"
                            ) from exc
                        except NativeSessionError as exc:
                            if budget.controller_attempts >= budget.maximum:
                                raise ModelServiceBudgetExceeded(
                                    "controller attempt cost cap exhausted after service failure; no continuation allowance remains",
                                    retryable=False,
                                ) from exc
                            if type(exc) is NativeTurnFailed and not callable(
                                getattr(session, "authorize_recovery", None)
                            ):
                                raise ModelServiceFailure(
                                    "Provider turn ended with an API error and has no recovery interface; close using the original infrastructure retry allowance: "
                                    + json.dumps(exc.error, default=str)[:2000],
                                    retryable=True,
                                ) from exc
                            authorization = await wait_for_resume(exc, identity)
                            if "result" in authorization:
                                decision = authorization["result"]
                                break
                            charge_turn = authorization.get("will_submit", False)
                        else:
                            break
                finally:
                    budget.controller_kind.reset(token)
                previous_guidance = guidance_text
                recovery_path = recovery_directory / "blocked.json"
                if recovery_path.exists():
                    recovery_record = json.loads(recovery_path.read_text())
                    if recovery_record.get("request_id") == identity:
                        atomic_json(
                            recovery_path, {**recovery_record, "status": "completed"}
                        )
                record.append(
                    "Completed native turn",
                    identity,
                    json.dumps(decision, ensure_ascii=False),
                )
                calls, parts = [], [TextPart(decision["text"])]
                batch_problem = decision_error(
                    [call["name"] for call in decision["tool_calls"]],
                    image_contract=toolkit.native_image_contract(),
                    image_limit=min(10, limits.request_limits.max_images or 10),
                )
                for index, proposal in enumerate(decision["tool_calls"]):
                    call = {
                        **proposal,
                        "id": f"call_{identity}_{index}",
                        "provider_id": f"{identity}_{index}",
                    }
                    problem = batch_problem
                    try:
                        call["arguments"] = json.loads(proposal["arguments_json"])
                    except json.JSONDecodeError:
                        call["arguments"] = proposal["arguments_json"]
                        problem = (
                            problem
                            or "Tool arguments are not complete JSON; the original was not repaired"
                        )
                    if call["name"] not in specs:
                        problem = (
                            problem or f"Unknown or unavailable tool: {call['name']}"
                        )
                    else:
                        try:
                            validate_json(
                                specs[call["name"]]["input_schema"], call["arguments"]
                            )
                        except ValidationError as exc:
                            problem = problem or exc.message
                    if problem:
                        call["validation_error"] = problem
                        parts.append(TextPart(json.dumps({"rejected_proposal": call})))
                        runtime.kernel._event(
                            "model_call_rejected",
                            request_id=identity,
                            call=call,
                            execution_performed=False,
                        )
                    else:
                        parts.append(
                            ToolCallPart(
                                call["name"],
                                call["arguments"],
                                tool_call_id=call["provider_id"],
                            )
                        )
                    calls.append(call)
                message = ModelMessagesTypeAdapter.dump_python(
                    [
                        ModelResponse(
                            parts=parts,
                            metadata={
                                "request_id": identity,
                                "program_call_id": program_call_id,
                                "tool_call_ids": {
                                    call["provider_id"]: call["id"] for call in calls
                                },
                            },
                        )
                    ],
                    mode="json",
                )[0]
                review = getattr(toolkit, "review_planner_decision", None)
                if review is not None:
                    verdict = await review(
                        {
                            "request_id": identity,
                            "program_call_id": program_call_id,
                            "response": message,
                        }
                    )
                    if verdict is not None:
                        runtime.kernel._event(
                            "model_decision_reviewed",
                            request_id=identity,
                            program_call_id=program_call_id,
                            verdict=verdict,
                        )
                        if not verdict["accepted"]:
                            for call in calls:
                                call["validation_error"] = (
                                    "Operator rejected this decision: "
                                    + verdict["reason"]
                                )
                runtime.messages.append(
                    {
                        "role": "assistant",
                        "content": decision,
                        "request_id": identity,
                        "program_call_id": program_call_id,
                        "native_session_id": session.session_id,
                    }
                )
                runtime.kernel._event(
                    "model_response",
                    request_id=identity,
                    program_call_id=program_call_id,
                    controller_request_unit="native_turn",
                    response=message,
                )
                if self.online_workflows:
                    runtime.kernel._event(
                        "native_decision_timing",
                        request_id=identity,
                        elapsed_s=time.monotonic() - decision_started,
                    )
                return {
                    "message": message,
                    "text": decision["text"],
                    "tool_calls": calls,
                }

        error = termination = None
        retryable = None
        cancelled = False
        episode_deadline = runtime.kernel._deadline
        if closeout_s is not None and episode_deadline is not None:
            # Reserve within the original absolute deadline, without extending
            # the episode or excluding elapsed action/model settlement time.
            runtime.kernel._deadline = episode_deadline - closeout_s
        try:
            await runtime.run(request, system_prompt, model_budget=budget)
        except PhysicalOutcomeUnknown as exc:
            error, termination = str(exc), ACTION_OUTCOME_UNKNOWN
        except asyncio.CancelledError:
            cancelled = True
            # External cancellation is not exhaustion of the agent's time budget.
            # Unknown physical effects are settled separately by the kernel.
            error, termination, retryable = "Native episode cancelled", API_ERROR, False
        except ProgramDeadlineExceeded as exc:
            error, termination = (
                str(exc),
                PLANNER_TIMEOUT if exc.scope == "episode" else AGENT_EXECUTION_FAILURE,
            )
        except (CandidateExecutionFailure, ModelBudgetExceeded) as exc:
            error, termination = str(exc), AGENT_EXECUTION_FAILURE
        except EnvironmentServiceFailure as exc:
            error, termination, retryable = str(exc), API_ERROR, exc.retryable
        except ModelServiceFailure as exc:
            error, termination, retryable = str(exc), API_ERROR, exc.retryable
        except Exception as exc:  # noqa: BLE001 - preserve infrastructure outcome and evidence
            error, termination, retryable = str(exc), API_ERROR, False
        finally:
            runtime.kernel._deadline = episode_deadline
        try:
            if closeout_s is not None:
                closeout = {"status": "not_written"}
                if notes.has_writes:
                    closeout = {"status": "already_written"}
                elif cancelled or termination in {API_ERROR, ACTION_OUTCOME_UNKNOWN}:
                    closeout["reason"] = "cancelled_or_unsafe_to_request_again"
                elif budget.controller_attempts >= max_turns or not getattr(
                    runtime, "execution", None
                ):
                    closeout["reason"] = "budget_exhausted_or_not_started"
                else:
                    try:
                        async with async_timeout(closeout_s):
                            written = await runtime.execution.closeout_notes(
                                {
                                    "termination_class": termination,
                                    "error": error,
                                }
                            )
                        closeout = {"status": "written", **written}
                    except asyncio.CancelledError:
                        closeout["reason"] = "closeout_cancelled"
                    except Exception as exc:  # noqa: BLE001 - notes never rewrite the physical result
                        closeout.update(reason=type(exc).__name__, error=str(exc))
                atomic_json(notes.root / "closeout.json", closeout)
                runtime.kernel._event("planner_notes_closeout", **closeout)
        finally:
            recovery_path = recovery_directory / "blocked.json"
            if recovery_path.exists():
                recovery_record = json.loads(recovery_path.read_text())
                if recovery_record.get("status") in {"blocked", "resuming"}:
                    atomic_json(
                        recovery_path,
                        {
                            **recovery_record,
                            "status": "terminated",
                            "termination_class": termination,
                            "cancelled": cancelled,
                        },
                    )
                    runtime.kernel._event(
                        "native_recovery_wait_ended",
                        request_id=recovery_record["request_id"],
                        termination_class=termination,
                        cancelled=cancelled,
                    )
            atomic_json(directory / "adaptation" / "receipt.json", adaptation.receipt())
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001 - retain the primary episode outcome
                record.append("Native cleanup failed", "close", str(exc))
                if error is None:
                    error, termination, retryable = str(exc), API_ERROR, False
        execution = getattr(runtime, "execution", None)
        root = execution.root if execution is not None else None
        return PlannerResult(
            finish_result=root.finish_result if root else None,
            messages=runtime.messages,
            stats={
                "requests": None,
                "model_calls": None,
                "visible_turns": budget.controller_attempts,
                "controller_request_unit": "native_turn",
                "adaptation_mode": self.adaptation_mode,
                **({"online_workflows": True} if self.online_workflows else {}),
                "cost_usd": None,
            },
            error=error,
            termination_class=termination,
            api_error_retryable=retryable,
        )
