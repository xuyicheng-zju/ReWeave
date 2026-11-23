# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Structured lifetimes for callable code from immutable plugin packages."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from reweave.evolution.bundle import ModelContext, NativeModelContext
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.runtime.capability import (
    CandidateExecutionFailure,
    ModelInputRejected,
    PhysicalBudget,
    PhysicalOutcomeUnknown,
)
from reweave.runtime.model_input import validate_native_request


async def build_context(kernel, program, context, context_call, tools, *, native=False):
    """Execute C and validate immutable selected tools on every decision path."""
    schema = NativeModelContext if native else ModelContext
    prepared = schema.model_validate(
        await kernel.run_plugin(program, context, context_call)
    ).model_dump(mode="json")
    if native:
        try:
            validate_native_request(prepared, kernel._episode_limits.request_limits)
        except ModelInputRejected as error:
            kernel._event(
                "context_rejected", input=context, output=prepared, error=str(error)
            )
            raise
    catalog = {spec["name"]: spec for spec in tools}
    names = [spec.get("name") for spec in prepared["tools"]]
    if (
        any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or any(spec != catalog.get(spec.get("name")) for spec in prepared["tools"])
    ):
        raise PermissionError(
            "context builder changed a tool interface or exposed forbidden tools"
        )
    return prepared


@dataclass
class PendingCall:
    """A model suggestion and the conversation receiving its tool return."""

    call: dict[str, Any]
    in_plan_conversation: bool
    # The first call's identity scopes ordering to this one model decision.
    decision_id: str
    not_executed_reason: str | None = None


@dataclass
class ExecutionContext:
    """One invocation's private conversation and explicitly attenuated authority."""

    identity: str
    plugin: str
    task: str
    permissions: frozenset[str]
    parent: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, PendingCall] = field(default_factory=dict)
    context_state: dict[str, Any] = field(default_factory=dict)
    # Last known unsent input failure; available to C until valid delivery succeeds.
    context_rejection: dict[str, Any] | None = None
    history_position: int = 0
    # Native messages remain scoped to their program; settled operation delivery
    # is shared by the one provider session below.
    delivered_message_position: int = 0
    # Pending message index -> bridge.history index. This is transient native
    # cross-call delivery bookkeeping, not a second trace or source of feedback.
    feedback_history_positions: dict[int, int] = field(default_factory=dict)
    finish_result: dict[str, Any] | None = None
    # W and its child tasks retain the registry under which the workflow started.
    workflow_revision_id: str | None = None


class ProgramExecution:
    """Own tasks and evidence, without prescribing an agent's algorithm."""

    def __init__(self, bridge, model_request, base_instructions, *, model_budget=None):
        self.bridge = bridge
        self.kernel = bridge.kernel
        self.model_request = model_request
        self.base_instructions = base_instructions
        self.model_budget = model_budget
        self.current: ContextVar[ExecutionContext] = ContextVar("program_execution")
        self.sessions: dict[str, ExecutionContext] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.root: ExecutionContext | None = None
        self._native_plan_lock = asyncio.Lock()
        self._native_history_position = 0

    def permissions(self):
        allowed = self.current.get().permissions
        restriction = self.kernel.authority.get()
        return allowed if restriction is None else allowed & restriction

    def specs(self):
        allowed = self.permissions()
        revision_id = self.current.get().workflow_revision_id
        catalog = (
            self.bridge.specs()
            if revision_id is None
            else self.bridge.specs(revision_id)
        )
        return [spec for spec in catalog if spec["name"] in allowed]

    def terminal(self):
        """Read the current outcome without cloning observation or history."""
        return bool(
            self.current.get().finish_result is not None
            or self.bridge.toolkit.solved()
            or self.bridge.toolkit.episode_ended()
        )

    def snapshot(self, *, history_start=0):
        """Detached public inputs: observation is the last state-changing feedback;
        last_tool_result and history include reads, failures and cancelled outcomes.
        A snapshot is settled evidence, not live sensing during an active action.
        """
        session = self.current.get()
        budget = self.bridge.toolkit.control_step_budget()
        return json.loads(
            json.dumps(
                {
                    "program_call_id": session.identity,
                    "parent_program_call_id": session.parent,
                    "task_language": session.task,
                    "terminal": self.terminal(),
                    "observation": self.bridge.observation,
                    "history": self.bridge.history[history_start:],
                    "last_tool_result": self.bridge.history[-1]
                    if self.bridge.history
                    else None,
                    "tools": self.specs(),
                    "epoch": self.kernel.epoch,
                    "remaining_wall_time_s": self.kernel.remaining_wall_time(),
                    "physical_budget": None
                    if budget is None
                    else PhysicalBudget.model_validate(budget).model_dump(mode="json"),
                    "controller_budget": self.model_budget.snapshot()
                    if self.model_budget is not None
                    else None,
                    "request_limits": self.kernel._episode_limits.request_limits.model_dump(
                        mode="json"
                    ),
                }
            )
        )

    async def run(self):
        session = ExecutionContext(
            uuid.uuid4().hex,
            "main",
            self.bridge.task,
            frozenset(spec["name"] for spec in self.bridge.specs())
            | {"model:plan", "task:start", "task:join", "task:cancel"},
        )
        self.root = session
        self.sessions[session.identity] = session
        adaptation = self.bridge.adaptation
        if adaptation is None:
            return await self._run(session, self.bridge.bundle.loop.program, None)
        while True:
            with adaptation.bind("L") as binding:
                with self.kernel.event_scope(revision_id=binding.revision.revision_id):
                    result = await self._run(
                        session, binding.program, None, loop_state=binding.state
                    )
            if not (isinstance(result, dict) and result.get("status") == "checkpoint"):
                return result
            if (
                session.finish_result is not None
                or self.bridge.toolkit.solved()
                or self.bridge.toolkit.episode_ended()
            ):
                return result
            # The old Python stack and all owned children have returned before the handoff.
            await self._stage_edits(session)
            adaptation.checkpoint(result["state"])

    async def _run(self, session, program, payload, *, loop_state=None):
        token = self.current.set(session)
        self.kernel._event(
            "program_started",
            program_call_id=session.identity,
            parent_program_call_id=session.parent,
            plugin=session.plugin,
            task=session.task,
            permissions=sorted(session.permissions),
        )
        if not session.messages:
            session.messages.append(
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": session.task}],
                }
            )
        failure = None
        try:
            with self.kernel.event_scope(
                program_call_id=session.identity,
                parent_program_call_id=session.parent,
                feedback_source="program",
            ):
                source_input = self.snapshot() if payload is None else payload
                if session.parent is None and self.bridge.adaptation is not None:
                    source_input["state"] = loop_state or {}
                result = await self.kernel.run_plugin(
                    program,
                    source_input,
                    self.invoke,
                )
                checkpoint = (
                    self.bridge.adaptation is not None
                    and isinstance(result, dict)
                    and result.get("status") == "checkpoint"
                )
                if checkpoint:
                    if session.parent is not None:
                        raise CandidateExecutionFailure(
                            "only the root L may return a cooperative checkpoint"
                        )
                    if "state" not in result:
                        raise CandidateExecutionFailure(
                            "cooperative L checkpoint requires explicit JSON state"
                        )
                    self.bridge.adaptation._copy_state(result["state"])
                if session.parent is None and not checkpoint and not self.terminal():
                    raise CandidateExecutionFailure(
                        "root Loop returned while the episode was nonterminal; "
                        "use feedback to replan and continue until the environment ends"
                    )
        except BaseException as error:
            failure = error
            self.kernel.raise_failure(
                "program_failed",
                error,
                program_call_id=session.identity,
            )
        finally:
            try:
                await self._drain(session)
            except BaseException as cleanup_error:
                if failure is not None and not isinstance(
                    cleanup_error, PhysicalOutcomeUnknown
                ):
                    failure.add_note(
                        f"Program cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}",
                    )
                    if failure is cleanup_error:
                        raise
                    raise failure from cleanup_error
                raise
            finally:
                self.current.reset(token)
        self.kernel._event(
            "program_completed", program_call_id=session.identity, result=result
        )
        return result

    async def _drain(self, session):
        children = [
            self.tasks[key]
            for key, child in self.sessions.items()
            if child.parent == session.identity and key in self.tasks
        ]
        for child in children:
            if not child.done():
                child.cancel()
        joined = asyncio.gather(*children, return_exceptions=True)
        interrupted = False
        while not joined.done():
            try:
                await asyncio.shield(joined)
            except asyncio.CancelledError:
                interrupted = True
        outcomes = joined.result()
        for outcome in outcomes:
            if isinstance(outcome, PhysicalOutcomeUnknown):
                raise outcome
        if interrupted:
            raise asyncio.CancelledError

    @asynccontextmanager
    async def workflow_scope(self, workflow_id):
        """A W invocation owns its children, even when the outer loop continues."""
        parent = self.current.get()
        session = ExecutionContext(
            uuid.uuid4().hex,
            "workflow:" + workflow_id,
            parent.task,
            parent.permissions,
            parent.identity,
            workflow_revision_id=self.kernel._event_context.get().get("revision_id"),
        )
        self.sessions[session.identity] = session
        token = self.current.set(session)
        try:
            with self.kernel.event_scope(
                program_call_id=session.identity,
                parent_program_call_id=parent.identity,
                feedback_source="program",
            ):
                yield
        finally:
            try:
                await self._drain(session)
            finally:
                self.current.reset(token)

    async def start(self, values):
        from reweave.evolution.bundle import PythonProgram
        from reweave.runtime.source import validate_program_source

        parent = self.current.get()
        package = self.kernel.program_context.get().model_dump(mode="json")
        package["entrypoint"] = values["entrypoint"]
        program = PythonProgram.model_validate(package)
        validate_program_source(program)
        permitted = self.permissions()
        requested = values.get("permissions")
        if parent.parent is None:
            safe = frozenset(
                name
                for name in permitted
                if not name.startswith("workflow_")
                and name != "finish"
                and self.kernel._operations.get(name, {}).get("effect")
                not in {"physical", "write"}
            )
            permissions = safe if requested is None else frozenset(requested)
            if not permissions <= safe:
                raise PermissionError(
                    "root Loop tasks cannot receive physical or Workflow execution "
                    "authority; use a runtime.plan() tool call"
                )
        else:
            permissions = permitted if requested is None else frozenset(requested)
        if not permissions <= permitted:
            raise PermissionError("task permissions exceed caller authority")
        session = ExecutionContext(
            uuid.uuid4().hex,
            values["entrypoint"],
            parent.task,
            permissions,
            parent.identity,
            workflow_revision_id=parent.workflow_revision_id,
        )
        self.sessions[session.identity] = session
        self.tasks[session.identity] = asyncio.create_task(
            self._run(session, program, values["payload"])
        )
        self.kernel._event(
            "task_started",
            program_call_id=session.identity,
            parent_program_call_id=parent.identity,
            entrypoint=values["entrypoint"],
        )
        return {"handle": session.identity}

    def _child(self, handle):
        if isinstance(handle, dict):
            handle = handle["handle"]
        if (
            handle not in self.tasks
            or self.sessions[handle].parent != self.current.get().identity
        ):
            raise PermissionError("task handle does not belong to caller")
        return self.tasks[handle]

    async def join(self, values):
        return await self._child(values["handle"])

    async def cancel(self, values):
        handle = values.get("handle")
        if handle is None:
            await self.kernel.cancel()
        else:
            task = self._child(handle)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return {"cancel_acknowledged": True}

    async def _stage_edits(self, session):
        adaptation = self.bridge.adaptation
        if adaptation is None:
            return
        try:
            await run_blocking(
                adaptation.stage,
                reason="Edits observed at a settled decision/checkpoint boundary",
                evidence=(f"program:{session.identity}",),
            )
        except (ValueError, KeyError, TypeError, SyntaxError, PermissionError) as error:
            # A rejected edit is public feedback in the same session, never a claim
            # that the new source ran and never an unlabelled old-code fallback.
            feedback = {
                "harness_edit_rejected": {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "active_revision_id": adaptation.active.revision_id,
                    "instruction": "The edit did not activate. This decision uses the explicitly recorded active C; repair the private candidate before its next capture.",
                }
            }
            session.messages.append(
                {
                    "kind": "request",
                    "parts": [
                        {"part_kind": "user-prompt", "content": json.dumps(feedback)}
                    ],
                }
            )
            self.kernel._event("harness_edit_rejected", **feedback)
        if session.parent is None:
            session.permissions = session.permissions | frozenset(
                spec["name"] for spec in self.bridge.specs()
            )

    async def plan(self, objective=None, *, tools=None, request_kind=None):
        # Every native root/local call uses the same provider session. Construct
        # its incremental view and commit delivery in one serialized boundary,
        # so concurrent calls cannot both select the same unsettled feedback.
        guard = (
            self._native_plan_lock
            if self.bridge.adaptation is not None
            else nullcontext()
        )
        async with guard:
            corrected_contexts = []
            while True:
                try:
                    return await self._plan_at_boundary(
                        objective, tools=tools, request_kind=request_kind
                    )
                except ModelInputRejected as error:
                    if (
                        self.bridge.adaptation is None
                        or self.bridge.adaptation.mode == "static"
                        or request_kind == "note_closeout"
                    ):
                        raise
                    current_context = self.bridge.adaptation.active.bundle.context
                    if current_context in corrected_contexts:
                        self.kernel._event(
                            "context_correction_stalled",
                            program_call_id=self.current.get().identity,
                            error=str(error),
                            reason="C remained unchanged after a correction and its input is still rejected.",
                        )
                        raise CandidateExecutionFailure(
                            "context correction made no executable C change and input is still rejected; "
                            "stop this known program failure for offline repair"
                        ) from error
                    corrected_contexts.append(current_context)
                    await self._correct_context(error)

    async def _correct_context(self, error):
        """Correct known unsent input using one budgeted, action-free native turn."""
        session = self.current.get()
        feedback = {
            "status": "model_input_rejected",
            "error": str(error),
            "instruction": (
                "The rejected request was not sent and no action was executed. "
                "Correct C's evidence selection before retrying this same decision. "
                "Do not reconstruct session history or blindly copy additional camera groups. "
                "In adaptive mode use native file tools for a targeted task-family C edit; "
                "static source remains read-only. C also receives context_rejection next time. "
                "This is a text-only correction turn. All task proposals from it are discarded; "
                "return no tool calls. Session, counters and delivery positions are preserved."
            ),
        }
        session.context_rejection = feedback
        message = {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": json.dumps(feedback)}],
        }
        session.messages.append(message)
        self.kernel._event(
            "model_input_rejected", program_call_id=session.identity, **feedback
        )
        correction = {
            "instructions": self.base_instructions,
            "messages": [message],
            "tools": [],
            "state": {},
            "request_kind": "context_correction",
        }
        response = await self.model_request(correction, session.identity)
        feedback["correction"] = response["text"]
        self.kernel._event(
            "context_correction",
            program_call_id=session.identity,
            text=response["text"],
            discarded_proposals=response["tool_calls"],
        )
        session.messages.append(
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": json.dumps(
                            {
                                "context_correction": response["text"],
                                "task_proposals_discarded": len(response["tool_calls"]),
                            }
                        ),
                    }
                ],
            }
        )

    async def _plan_at_boundary(self, objective=None, *, tools=None, request_kind=None):
        adaptation = self.bridge.adaptation
        session = self.current.get()
        if request_kind != "note_closeout" and (
            session.finish_result is not None
            or self.bridge.toolkit.solved()
            or self.bridge.toolkit.episode_ended()
        ):
            return {
                "terminal": True,
                "text": "The episode has ended; no further decision was requested.",
                "tool_calls": [],
            }
        if (
            adaptation is not None
            and request_kind != "note_closeout"
            and not session.pending
        ):
            await self._stage_edits(session)
        binding = adaptation.bind("C") if adaptation is not None else nullcontext(None)
        with binding as pinned:
            decision_revision_id = session.workflow_revision_id or (
                pinned.revision.revision_id if pinned is not None else None
            )
            with self.kernel.event_scope(
                **(
                    {"context_revision_id": pinned.revision.revision_id}
                    if pinned is not None
                    else {}
                )
            ):
                return await self._plan(
                    objective,
                    tools=tools,
                    request_kind=request_kind,
                    context_bundle=pinned.revision.bundle
                    if pinned is not None
                    else self.bridge.bundle,
                    revision_id=decision_revision_id,
                )

    async def _plan(
        self,
        objective=None,
        *,
        tools=None,
        request_kind=None,
        context_bundle,
        revision_id,
    ):
        """The only model entry: settled evidence -> C -> budgeted request."""
        from pydantic_ai.messages import (
            ModelMessagesTypeAdapter,
            ModelRequest,
            UserPromptPart,
        )

        from reweave.planner.model_messages import content_blocks_to_pydantic

        session = self.current.get()
        if "model:plan" not in self.permissions():
            raise PermissionError("model request exceeds caller authority")
        for identity, pending in list(session.pending.items()):
            if pending.not_executed_reason is not None:
                self.feedback(
                    {
                        "call_id": identity,
                        "result": {
                            "status": "not_executed",
                            "reason": pending.not_executed_reason,
                        },
                    }
                )
        terminal_response = {
            "terminal": True,
            "text": "The episode has ended; no further decision was requested.",
            "tool_calls": [],
        }
        if request_kind != "note_closeout" and self.terminal():
            return terminal_response
        preparation_started = time.perf_counter()
        if not session.messages:
            session.messages.append(
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": session.task}],
                }
            )
        # Direct W reads/actions have no provider tool-call envelope. Make their
        # settled feedback available through the same conversation/C path. Calls
        # executed from a model decision already have a paired envelope below.
        native = self.bridge.adaptation is not None
        history_start = (
            self._native_history_position if native else session.history_position
        )
        pending_feedback = {
            history_index
            for position, history_index in session.feedback_history_positions.items()
            if position >= session.delivered_message_position
        }
        for index in range(history_start, len(self.bridge.history)):
            if native and index in pending_feedback:
                # An earlier failed delivery or an actual tool return already
                # has this feedback queued in this call's messages.
                continue
            feedback = self.bridge._tool_result(
                "public_feedback",
                {
                    "history_index": index,
                    "feedback": self.bridge.history[index],
                },
            )
            text, images = content_blocks_to_pydantic(feedback.content_blocks)
            if native:
                session.feedback_history_positions[len(session.messages)] = index
            session.messages.extend(
                ModelMessagesTypeAdapter.dump_python(
                    [
                        ModelRequest(
                            [UserPromptPart([text, *images] if images else text)]
                        )
                    ],
                    mode="json",
                )
            )
        session.history_position = len(self.bridge.history)
        if objective is not None:
            session.messages.append(
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": objective}],
                }
            )
        message_end = len(session.messages)
        history_end = len(self.bridge.history)
        messages = (
            [
                message
                for position, message in enumerate(
                    session.messages[session.delivered_message_position : message_end],
                    start=session.delivered_message_position,
                )
                if message["kind"] == "request"
                and session.feedback_history_positions.get(
                    position, self._native_history_position
                )
                >= self._native_history_position
            ]
            if native
            else session.messages
        )
        context = {
            **self.snapshot(
                history_start=self._native_history_position if native else 0
            ),
            "base_instructions": self.base_instructions,
            "state": json.loads(json.dumps(session.context_state)),
            "assets": context_bundle.context.model_dump(mode="json"),
            "messages": json.loads(json.dumps(messages)),
            "objective": objective,
            "tool_effects": {
                name: value["effect"] for name, value in self.kernel._operations.items()
            },
        }
        if session.context_rejection is not None:
            context["context_rejection"] = json.loads(
                json.dumps(session.context_rejection)
            )
        if tools is not None:
            context["tools"] = tools
        # A fresh fact for this request, not another accumulated history. Even
        # an inherited C consuming only messages can select the current budget.
        # C's complete selected view and these original inputs are both traced.
        if not context["messages"] or context["messages"][-1]["kind"] != "request":
            context["messages"].append({"kind": "request", "parts": []})
        context["messages"][-1]["parts"].insert(
            0,
            {
                "part_kind": "user-prompt",
                "content": "Current runtime budget. Physical counts use the "
                "declared unit, not observation indices or tool-call counts; "
                "null means unknown. "
                + json.dumps(
                    {
                        "physical_budget": context["physical_budget"],
                        "controller_budget": context["controller_budget"],
                        "remaining_wall_time_s": context["remaining_wall_time_s"],
                    }
                ),
            },
        )

        async def context_call(method, name, arguments):
            operation = self.kernel._operations.get(name)
            if operation is None or operation["effect"] != "read":
                raise PermissionError(
                    "context builder cannot perform actions, spawn tasks or call models recursively; it may only retrieve read evidence"
                )
            return await self.invoke(method, name, arguments)

        snapshot_ready = time.perf_counter()
        if native:
            context = self.bridge.toolkit.prepare_model_context(context)
        observation_ready = time.perf_counter()
        try:
            prepared = await build_context(
                self.kernel,
                context_bundle.context.program,
                context,
                context_call,
                self.specs() if tools is None else tools,
                native=native,
            )
        except Exception as error:
            if not isinstance(error, ModelInputRejected):
                # Known request rejections already retain input/output above.
                self.kernel._event(
                    "context_build_failed",
                    input=context,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            raise
        context_ready = time.perf_counter()
        if native:
            finalize = getattr(self.bridge.toolkit, "finalize_model_context", None)
            if finalize is not None:
                prepared = finalize(prepared)
            try:
                self.bridge.toolkit.validate_model_context(prepared)
            except ModelInputRejected as error:
                self.kernel._event(
                    "context_rejected", input=context, output=prepared, error=str(error)
                )
                raise
        self.kernel._event(
            "context_built",
            program_call_id=session.identity,
            input=context,
            output=prepared,
            preparation_timings_s={
                "feedback_and_snapshot": snapshot_ready - preparation_started,
                "observation": observation_ready - snapshot_ready,
                "context_program": context_ready - observation_ready,
                "validation": time.perf_counter() - context_ready,
            },
        )
        # Ledger attribution only; all callers use the identical API and C.
        prepared["request_kind"] = request_kind or (
            "planner" if session.parent is None else "reason"
        )
        if request_kind != "note_closeout" and self.terminal():
            return terminal_response
        response = await self.model_request(prepared, session.identity)
        if native:
            session.delivered_message_position = message_end
            self._native_history_position = history_end
            session.feedback_history_positions = {
                position: history_index
                for position, history_index in session.feedback_history_positions.items()
                if position >= message_end
            }
        session.context_state = prepared["state"]
        session.context_rejection = None
        session.messages.append(response["message"])
        if "operator_feedback" in response:
            session.messages.append(
                {
                    "kind": "request",
                    "parts": [
                        {
                            "part_kind": "user-prompt",
                            "content": response["operator_feedback"],
                        }
                    ],
                }
            )
        for call in response["tool_calls"]:
            if revision_id is not None:
                call["revision_id"] = revision_id
                for token, binding in self.bridge.adaptation.continuations:
                    if (
                        call["name"] == "workflow_" + binding.workflow.workflow_id
                        and isinstance(call.get("arguments"), dict)
                        and call["arguments"].get("continuation_id") == token
                    ):
                        call["revision_id"] = binding.revision.revision_id
                        break
            session.pending[call["id"]] = PendingCall(
                call,
                "validation_error" not in call,
                response["tool_calls"][0]["id"],
            )
        return {key: value for key, value in response.items() if key != "message"}

    async def closeout_notes(self, outcome):
        """One text-only request through the same C, then retain its original prose."""
        from dataclasses import replace

        root = self.root
        if root is None:
            raise RuntimeError("cannot summarize an episode that never started")
        messages = json.loads(json.dumps(root.messages))
        pending_ids = {item.call["provider_id"] for item in root.pending.values()}
        # Send new labelled facts for unresolved provider returns. Native history
        # already contains the proposals; do not rewrite it or forge tool results.
        pending_facts = [
            {"unreturned_at_episode_end": part}
            for message in messages
            if message["kind"] == "response"
            for part in message["parts"]
            if part.get("part_kind") == "tool-call"
            and part.get("tool_call_id") in pending_ids
        ]
        if pending_facts:
            messages.append(
                {
                    "kind": "request",
                    "parts": [
                        {
                            "part_kind": "user-prompt",
                            "content": json.dumps(pending_facts),
                        }
                    ],
                }
            )
        session = replace(root, messages=messages, pending={})
        token = self.current.set(session)
        try:
            response = await self.plan(
                "The action phase has ended. Write the missing episode working note "
                "as concise Markdown text, without tool calls or further actions. "
                "Record the actual approaches and parameters, positive and negative "
                "observations and public references, the observed stopping point, "
                "uncertainty and the next discriminating test. Failure is not proof "
                "of impossibility; self-reported finish is not an official verdict. "
                "If nothing was learned, state that explicitly. This text will be "
                "retained as an unverified private task-family draft for later "
                "coding review. Online termination: " + json.dumps(outcome),
                tools=[],
                request_kind="note_closeout",
            )
            if response["tool_calls"] or not response["text"].strip():
                raise ValueError("note close-out requires nonblank text without tools")
            notes = self.bridge.toolkit.episode_notes
            result = await self.bridge.execute(
                "write_text_file",
                {
                    "path": str(notes.drafts / "task-family/episode-closeout.md"),
                    "content": response["text"],
                },
            )
            if result.result.get("written") is not True:
                raise RuntimeError("note close-out write failed: " + str(result.result))
            return result.result
        finally:
            self.current.reset(token)

    async def execute(self, call):
        from pydantic_ai.messages import (
            ModelMessagesTypeAdapter,
            ModelRequest,
            ToolReturnPart,
            UserPromptPart,
        )

        from reweave.planner.base import accepted_finish_result
        from reweave.planner.decisions import tool_failed
        from reweave.planner.model_messages import content_blocks_to_pydantic

        session = self.current.get()
        name, arguments = call["name"], call["arguments"]
        identity = call.get("id")
        if identity is not None and identity not in session.pending:
            raise ValueError("tool call identity is not pending in this conversation")
        pending = session.pending.get(identity)
        requested = pending.call if pending is not None else None
        if requested is not None and requested == call:
            first = next(
                key
                for key, item in session.pending.items()
                if item.decision_id == pending.decision_id
            )
            if identity != first:
                raise PermissionError(
                    "tool calls must execute in the order returned by runtime.plan(); "
                    "settle the earlier call before executing this one"
                )
            if pending.not_executed_reason is not None:
                return self.feedback(
                    {
                        "call_id": identity,
                        "result": {
                            "status": "not_executed",
                            "reason": pending.not_executed_reason,
                        },
                    }
                )
        if (
            requested is not None
            and requested == call
            and "validation_error" in requested
        ):
            result = self.feedback(
                {
                    "call_id": identity,
                    "result": {
                        "error": requested["validation_error"],
                        "instruction": "Choose an available tool and supply valid arguments; nothing was executed.",
                    },
                }
            )
            self._skip_decision_tail(pending)
            return result
        if session.parent is None and (
            pending is None or not pending.in_plan_conversation or requested != call
        ):
            raise PermissionError(
                "root Loop may execute only an unchanged tool call returned by "
                "runtime.plan()"
            )
        if name not in self.permissions():
            raise PermissionError("tool exceeds execution authority")
        operation = self.kernel._operations.get(name)
        if (
            pending is not None
            or name.startswith("workflow_")
            or operation is not None
            and operation["effect"] == "physical"
        ) and self.terminal():
            result = {
                "status": "not_executed",
                "execution": "not_executed",
                "reason": "environment episode ended",
            }
            if pending is not None:
                self._skip_decision_tail(pending, reason=result["reason"])
                return self.feedback({"call_id": identity, "result": result})
            return result
        settlement_token = self.bridge.settled_operation.set(None)
        try:
            with self.kernel.event_scope(
                planner_call_id=identity,
                feedback_source="model" if requested == call else "program",
            ):
                result = await self.bridge.execute(
                    name, arguments, revision_id=call.get("revision_id")
                )
            history_index = self.bridge.settled_operation.get()
        finally:
            self.bridge.settled_operation.reset(settlement_token)
        self.bridge.record_tool(call, result, requested)
        content, images = content_blocks_to_pydantic(result.content_blocks)
        if identity is not None and identity in session.pending:
            pending = session.pending.pop(identity)
            original = pending.call
            if original != call:
                content = json.dumps(
                    {
                        "requested": original,
                        "executed": call,
                        "result": content,
                    }
                )
            parts = (
                [
                    ToolReturnPart(
                        original["name"], content, tool_call_id=original["provider_id"]
                    )
                ]
                if pending.in_plan_conversation
                else [
                    UserPromptPart(
                        json.dumps(
                            {
                                "rejected_call": original,
                                "executed": call,
                                "result": content,
                            }
                        )
                    )
                ]
            )
            if images:
                parts.append(UserPromptPart(images))
        else:
            parts = [
                UserPromptPart(
                    json.dumps(
                        {
                            "executed": call,
                            "result": content,
                            "feedback_source": "program",
                            "program_call_id": session.identity,
                        }
                    )
                )
            ]
            if images:
                parts.append(UserPromptPart(images))
        if (
            self.bridge.adaptation is not None
            and name in self.kernel._operations
            and history_index is not None
        ):
            # A fixed tool's ACK and its settled operation are the same feedback.
            # Another program may deliver that operation to the shared native
            # session before this program asks for its next decision.
            session.feedback_history_positions[len(session.messages)] = history_index
        session.messages.extend(
            ModelMessagesTypeAdapter.dump_python([ModelRequest(parts)], mode="json")
        )
        session.history_position = len(self.bridge.history)
        if requested == call and tool_failed(result.result):
            self._skip_decision_tail(pending)
        elif requested == call and self.terminal():
            self._skip_decision_tail(pending, reason="environment episode ended")
        if name == "finish":
            session.finish_result = accepted_finish_result(result.result)
        return (
            self.bridge._text_view(result.result)
            if not isinstance(result.result, dict)
            else result.result
        )

    def _skip_decision_tail(self, failed, *, reason=None):
        for remaining in self.current.get().pending.values():
            if remaining.decision_id != failed.decision_id:
                continue
            remaining.not_executed_reason = reason or (
                f"Tool {failed.call['name']!r} failed or was cancelled; "
                "replan from its feedback."
            )
            self.kernel._event(
                "model_call_not_executed",
                planner_call_id=remaining.call["id"],
                call=remaining.call,
                reason=remaining.not_executed_reason,
                execution_performed=False,
            )

    def feedback(self, values):
        from pydantic_ai.messages import (
            ModelMessagesTypeAdapter,
            ModelRequest,
            ToolReturnPart,
            UserPromptPart,
        )

        session = self.current.get()
        identity = values["call_id"]
        if identity not in session.pending:
            raise ValueError("feedback identity is not pending in this conversation")
        pending = session.pending.pop(identity)
        requested = pending.call
        result = {
            "source": "program_feedback",
            "execution_performed": False,
            "feedback": values["result"],
        }
        session.messages.extend(
            ModelMessagesTypeAdapter.dump_python(
                [
                    ModelRequest(
                        [
                            ToolReturnPart(
                                requested["name"],
                                result,
                                tool_call_id=requested["provider_id"],
                            )
                            if pending.in_plan_conversation
                            else UserPromptPart(
                                json.dumps(
                                    {"rejected_call": requested, "result": result}
                                )
                            )
                        ]
                    )
                ],
                mode="json",
            )
        )
        self.bridge.record_feedback(requested, result)
        self.kernel._event("program_feedback", planner_call_id=identity, result=result)
        return result

    async def invoke(self, method, name, arguments):
        self.kernel.raise_if_failed()
        values = arguments or {}
        if method == "control":
            if name == "episode:snapshot":
                return self.snapshot()
            if name == "tool:execute":
                return await self.execute(values["call"])
            if name == "tool:feedback":
                return self.feedback(values)
            if name.startswith("workflow:"):
                return await self.execute(
                    {
                        "name": "workflow_" + name.removeprefix("workflow:"),
                        "arguments": values,
                    }
                )
            if name.startswith("task:"):
                if name not in self.permissions():
                    raise PermissionError("task operation exceeds caller authority")
                return await {
                    "task:start": self.start,
                    "task:join": self.join,
                    "task:cancel": self.cancel,
                }[name](values)
        if name not in self.permissions():
            raise PermissionError("operation exceeds caller authority")
        operation = self.kernel._operations.get(name)
        if (
            operation is None
            or {
                "read": "observe",
                "physical": "act",
                "model": "control",
                "task": "control",
            }.get(operation["effect"])
            != method
        ):
            raise PermissionError("runtime method does not match operation effect")
        if self.current.get().parent is None and operation["effect"] == "physical":
            raise PermissionError(
                "root Loop physical execution requires a tool call returned by "
                "runtime.plan()"
            )
        if operation["effect"] == "physical" and self.terminal():
            return {
                "status": "not_executed",
                "execution": "not_executed",
                "reason": "environment episode ended",
            }
        return await self.kernel.invoke(name, values, epoch=self.kernel.epoch)
