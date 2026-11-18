# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Fixed execution gateway for immutable W/C/L bundles.

Replaceable plugins request operations; only this kernel dispatches effects. A
workflow cannot register code, access an author, or select its own success label.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaError
from pydantic import Field, model_validator
from referencing import Registry

from reweave.evolution.bundle import (
    AgentBundle,
    Asset,
    ProgramLimits,
    RequestLimits,
    WorkflowResult,
    WorkflowSpec,
)
from reweave.runtime.capability import (
    CandidateExecutionFailure,
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
    ProgramDeadlineExceeded,
)
from reweave.runtime.exception_notes import add_exception_note
from reweave.runtime.model_budget import ModelBudgetExceeded, ModelServiceFailure

try:
    from asyncio import timeout as async_timeout
except ImportError:  # Python 3.10
    from async_timeout import timeout as async_timeout
from reweave.runtime.sandbox import ProgramSandbox
from reweave.runtime.source import validate_program_source


class EpisodeLimits(Asset):
    """Fixed total limits shared by L, workflows and the main controller."""

    program_calls: int | None = Field(
        gt=0,
        strict=True,
        description="Total callable invocations; null disables this additional episode cap.",
    )
    gateway_calls: int | None = Field(
        gt=0,
        strict=True,
        description="Total public operation dispatches; null disables this additional episode cap.",
    )
    wall_time_s: float | None = Field(
        gt=0,
        strict=True,
        description="Total episode wall-clock budget, including API waits, scheduling and workflows; null disables this time cap.",
    )
    request_limits: RequestLimits = Field(
        default_factory=RequestLimits,
        description="Deployment-specific body budgets shared by every root and local controller request; frozen in the runtime configuration.",
    )
    note_closeout_s: float | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="Text-only close-out timeout and one reserved shared Planner request if no private note was written. With a finite episode deadline, reserve this time within it; without one, this only bounds close-out. Null leaves note timing to the Planner. No additional requests or physical actions are granted.",
    )

    @model_validator(mode="after")
    def check_note_reserve(self):
        if (
            self.note_closeout_s is not None
            and self.wall_time_s is not None
            and self.note_closeout_s >= self.wall_time_s
        ):
            raise ValueError(
                "note close-out reserve must be smaller than episode wall time"
            )
        return self


Gateway = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
EventSink = Callable[[dict[str, Any]], None]


class WorkflowContinuationError(ValueError):
    """Known, pre-execution rejection of caller-owned continuation identity."""


def validate_json(schema: dict[str, Any], value: Any) -> None:
    """No external URI resolution, even when a schema validator supports it."""
    # Frozen tuples serialize back to ordinary JSON arrays before validation.
    schema = json.loads(json.dumps(schema))
    Draft202012Validator(schema, registry=Registry()).validate(
        json.loads(json.dumps(value))
    )


class EpisodeKernel:
    """One episode's private state and fixed authorities, never a search archive."""

    def __init__(
        self,
        bundle: AgentBundle,
        *,
        operations: dict[str, dict[str, Any]],
        gateway: Gateway,
        cancel: Callable[[], Awaitable[None]],
        emit: EventSink,
        episode_limits: EpisodeLimits,
        sandbox: ProgramSandbox | None = None,
        workflow_scope: Callable[[str], AbstractAsyncContextManager] | None = None,
        snapshot: Callable[[], dict[str, Any]] | None = None,
        episode_ended: Callable[[], bool] | None = None,
    ):
        self._bundle = AgentBundle.model_validate_json(bundle.model_dump_json())
        self._identity = self._bundle.record_id
        self._operations = json.loads(json.dumps(operations))
        self._gateway, self._cancel, self._emit = gateway, cancel, emit
        self._sandbox = sandbox or ProgramSandbox()
        self._workflow_scope = workflow_scope or (lambda _: nullcontext())
        self._snapshot = snapshot
        # Whole-environment terminal authority, independent of program context.
        # None declares that this Kernel has no environment terminal source.
        self._episode_ended = episode_ended
        self._episode_limits = episode_limits
        self._deadline = (
            time.monotonic() + episode_limits.wall_time_s
            if episode_limits.wall_time_s is not None
            else None
        )
        self._program_calls = 0
        self._fatal: PhysicalOutcomeUnknown | None = None
        self._evidence_error: BaseException | None = None
        self._service_failure: BaseException | None = None
        self.program_deadline: ContextVar[float | None] = ContextVar(
            "program_deadline", default=None
        )
        self.budget_frames: ContextVar[tuple] = ContextVar(
            "workflow_budget_frames", default=()
        )
        self.program_context: ContextVar[Any] = ContextVar("executing_program")
        self.authority: ContextVar[frozenset[str] | None] = ContextVar(
            "workflow_authority", default=None
        )
        self._calls = 0
        self._workflows = {item.workflow_id: item for item in self._bundle.workflows}
        self._epoch = 0
        self._cancel_pending = 0
        self._dispatch = asyncio.Lock()
        self._continuations: dict[str, dict[str, Any]] = {}
        self._event_context: ContextVar[dict[str, Any]] = ContextVar(
            "kernel_event_context", default={}
        )
        for workflow in self._workflows.values():
            for permission in workflow.permissions:
                operation = self._operations.get(permission)
                if operation is None or operation["effect"] not in {
                    "read",
                    "physical",
                    "model",
                    "task",
                }:
                    raise ValueError(
                        f"workflow requires unknown/non-public operation {permission!r}"
                    )
            if any(
                self._operations[name]["effect"] != "read"
                for name in workflow.observations
            ):
                raise ValueError("workflow observations include a non-read operation")
            validate_program_source(workflow.program)
        validate_program_source(self._bundle.loop.program)
        validate_program_source(self._bundle.context.program)

    def charge_workflow(self, effect):
        frames = self.budget_frames.get()
        for limits, counts in frames:
            maximum = {
                "runtime": limits.runtime_calls,
                "physical": limits.physical_calls,
                "model": limits.model_calls,
            }[effect]
            if maximum is not None and counts[effect] >= maximum:
                raise RuntimeError(f"workflow {effect} budget exhausted")
        for _, counts in frames:
            counts[effect] += 1

    def raise_if_failed(self):
        if self._fatal is not None:
            raise self._fatal
        if self._evidence_error is not None:
            raise self._evidence_error
        if self._service_failure is not None:
            raise self._service_failure

    def deadline(self) -> float | None:
        """Earliest active episode/program deadline, inherited by local tasks."""
        active = self.program_deadline.get()
        deadlines = [value for value in (self._deadline, active) if value is not None]
        return min(deadlines) if deadlines else None

    def remaining_wall_time(self, *, episode_only: bool = False) -> float | None:
        """Remaining active time; null explicitly means no applicable deadline."""
        deadline = self._deadline if episode_only else self.deadline()
        return max(0.0, deadline - time.monotonic()) if deadline is not None else None

    @property
    def epoch(self) -> int:
        return self._epoch

    def catalog(self, bundle: AgentBundle | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": item.workflow_id,
                "description": item.description,
                "applicability": item.applicability,
                "input_schema": item.input_schema,
                "output_schema": item.output_schema,
            }
            for item in (
                bundle.workflows if bundle is not None else self._workflows.values()
            )
        )

    @contextmanager
    def event_scope(self, **identities: Any) -> Iterator[None]:
        """Explicit causal identities, inherited across async gateway dispatch."""
        token = self._event_context.set({**self._event_context.get(), **identities})
        try:
            yield
        finally:
            self._event_context.reset(token)

    def _event(self, kind: str, **values: Any) -> None:
        try:
            self._emit(
                {
                    "kind": kind,
                    "bundle_id": self._identity,
                    "epoch": self._epoch,
                    "wall_time_s": time.time(),
                    "monotonic_s": time.monotonic(),
                    **self._event_context.get(),
                    **values,
                }
            )
        except BaseException as error:
            if self._evidence_error is None:
                self._evidence_error = error
            raise

    def raise_failure(self, kind: str, error: BaseException, **values: Any) -> None:
        """Record a failure without replacing its outcome with an audit error.

        An already confirmed unknown physical outcome owns the final classification,
        including when cleanup subsequently raises a different exception.
        """
        outcome = self._fatal or self._service_failure or error
        if outcome is not error:
            add_exception_note(
                outcome, f"Subsequent {kind}: {type(error).__name__}: {error}"
            )
        try:
            self._event(
                kind, error=type(outcome).__name__, detail=str(outcome), **values
            )
        except BaseException as evidence_error:
            add_exception_note(outcome, f"{kind} evidence failed: {evidence_error}")
            if outcome is evidence_error:
                raise
            raise outcome from evidence_error
        if outcome is not error:
            raise outcome from error
        raise outcome

    async def _run_program(self, program, limits, payload, invoke, timing=None):
        """Record every rejected worker RPC, including errors caught by plugin code."""
        from reweave.runtime.model_budget import ModelServiceFailure

        external_s = 0.0

        async def recorded_invoke(method, name, arguments):
            nonlocal external_s
            # A previous failed audit is a Host failure, even if the plugin caught
            # the original RPC error. Never dispatch another operation afterwards.
            self.raise_if_failed()
            started = time.monotonic()
            try:
                return await invoke(method, name, arguments)
            except BaseException as error:
                if isinstance(
                    error,
                    (
                        ModelServiceFailure,
                        EnvironmentInvalid,
                        EnvironmentServiceFailure,
                    ),
                ):
                    self._service_failure = error
                self.raise_failure(
                    "program_rpc_failed",
                    error,
                    entrypoint=program.entrypoint,
                    method=method,
                    name=name,
                    arguments=arguments,
                )
            finally:
                external_s += time.monotonic() - started

        deadlines = [
            value
            for value in (
                self.deadline(),
                time.monotonic() + limits.wall_time_s
                if limits.wall_time_s is not None and self._deadline is not None
                else None,
            )
            if value is not None
        ]
        token = self.program_deadline.set(min(deadlines) if deadlines else None)
        started = time.monotonic()
        try:
            result = await self._sandbox.run_program(
                program,
                limits,
                payload,
                recorded_invoke,
                input_schema={"type": "object"},
                exclude_runtime_time=self._deadline is None,
            )
            self.raise_if_failed()
            return result
        finally:
            if timing is not None:
                timing["elapsed_s"] = (
                    time.monotonic()
                    - started
                    - (external_s if self._deadline is None else 0.0)
                )
            self.program_deadline.reset(token)

    async def run_plugin(self, program, payload, invoke):
        """Execute a complete plugin without imposing an orchestration algorithm."""
        self.raise_if_failed()
        self._program_calls += 1
        if (
            self._episode_limits.program_calls is not None
            and self._program_calls > self._episode_limits.program_calls
        ):
            raise RuntimeError("fixed episode program-call budget exhausted")
        remaining = self.remaining_wall_time()
        if remaining is not None and remaining <= 0:
            raise ProgramDeadlineExceeded(
                "fixed execution deadline exhausted",
                scope="episode"
                if self.remaining_wall_time(episode_only=True) == 0
                else "program",
            )
        limits = ProgramLimits(
            runtime_calls=self._episode_limits.gateway_calls,
            physical_calls=self._episode_limits.gateway_calls,
            model_calls=self._episode_limits.gateway_calls,
            wall_time_s=remaining,
        )
        token = self.program_context.set(program)
        try:
            async with async_timeout(remaining):
                return await self._run_program(program, limits, payload, invoke)
        except TimeoutError as error:
            self.raise_if_failed()
            if self.remaining_wall_time() != 0:
                raise
            raise ProgramDeadlineExceeded(
                "fixed execution deadline exhausted",
                scope="episode"
                if self.remaining_wall_time(episode_only=True) == 0
                else "program",
            ) from error
        finally:
            self.program_context.reset(token)

    async def invoke(
        self, name: str, arguments: dict[str, Any], *, epoch: int
    ) -> dict[str, Any]:
        self.raise_if_failed()
        operation = self._operations.get(name)
        if operation is None:
            raise PermissionError(f"operation is not in the frozen catalog: {name}")
        validate_json(operation["input_schema"], arguments)
        self.charge_workflow("runtime")
        lock = (
            self._dispatch
            if operation["effect"] in {"read", "physical", "control", "write"}
            else nullcontext()
        )
        async with lock:
            self.raise_if_failed()
            if epoch != self._epoch or self._cancel_pending:
                raise RuntimeError("stale action rejected at dispatch")
            if operation["effect"] == "physical":
                if self._episode_ended is not None and self._episode_ended():
                    return {
                        "status": "not_executed",
                        "execution": "not_executed",
                        "reason": "environment episode ended",
                    }
                self.charge_workflow("physical")
            self._calls += 1
            if (
                self._episode_limits.gateway_calls is not None
                and self._calls > self._episode_limits.gateway_calls
            ) or self.remaining_wall_time(episode_only=True) == 0:
                raise RuntimeError("fixed episode gateway budget exhausted")
            action_id = uuid.uuid4().hex
            self._event(
                "action_submitted",
                action_id=action_id,
                effect=operation["effect"],
                operation=name,
                arguments=arguments,
            )
            try:
                with self.event_scope(action_id=action_id):
                    async with async_timeout(self.remaining_wall_time()):
                        result = await self._gateway(name, arguments)
            except BaseException as error:
                if isinstance(error, PhysicalOutcomeUnknown):
                    self._fatal = error
                self.raise_failure(
                    "action_failed", error, action_id=action_id, operation=name
                )
            self._event(
                "action_completed",
                action_id=action_id,
                effect=operation["effect"],
                operation=name,
                result=result,
            )
            return result

    async def cancel(self) -> None:
        self._cancel_pending += 1
        self._event("cancel_requested")
        try:
            await self._cancel()
            # Kernel never acknowledges before the real physical adapter does.
            async with self._dispatch:
                self._epoch += 1
                self._continuations.clear()
                self._event("cancel_acknowledged")
        finally:
            self._cancel_pending -= 1

    async def workflow(
        self,
        workflow_id: str,
        arguments: dict[str, Any],
        *,
        observation: dict[str, Any],
        task_language: str,
        epoch: int,
        continuation_id: str | None = None,
        ancestors: tuple[str, ...] = (),
        budget_frames: tuple[tuple[ProgramLimits, dict[str, int]], ...] = (),
        workflows: tuple[WorkflowSpec, ...] | None = None,
        recover_program_errors: bool = False,
    ) -> WorkflowResult:
        self.raise_if_failed()
        invocation_id = uuid.uuid4().hex
        parent = self._event_context.get().get("workflow_invocation_id")
        with self.event_scope(
            workflow_invocation_id=invocation_id,
            parent_workflow_invocation_id=parent,
            workflow_id=workflow_id,
        ):
            self._event(
                "workflow_started", arguments=arguments, continuation_id=continuation_id
            )
            try:
                async with self._workflow_scope(workflow_id):
                    result = await self._workflow(
                        workflow_id,
                        arguments,
                        observation=observation,
                        task_language=task_language,
                        epoch=epoch,
                        continuation_id=continuation_id,
                        ancestors=ancestors,
                        budget_frames=budget_frames,
                        workflows=workflows,
                    )
            except WorkflowContinuationError as error:
                # Rejected before consuming state or executing a program. This
                # caller protocol mistake has a known zero-physical outcome;
                # execution/transport failures still poison the episode below.
                result = WorkflowResult(
                    status="failed",
                    summary=str(error),
                    output={
                        "error": "invalid_continuation",
                        "execution_performed": False,
                    },
                )
                self._event(
                    "workflow_continuation_rejected",
                    continuation_id=continuation_id,
                    execution_performed=False,
                    error=str(error),
                )
            except BaseException as error:
                if (
                    recover_program_errors
                    and isinstance(
                        error, (ValueError, TypeError, RuntimeError, JsonSchemaError)
                    )
                    and not isinstance(
                        error,
                        (
                            CandidateExecutionFailure,
                            EnvironmentInvalid,
                            EnvironmentServiceFailure,
                            ModelBudgetExceeded,
                            ModelServiceFailure,
                        ),
                    )
                ):
                    self.raise_if_failed()
                    self._event(
                        "workflow_failed",
                        error=type(error).__name__,
                        detail=str(error),
                        recoverable=True,
                    )
                    result = WorkflowResult(
                        status="failed",
                        summary="Online W failed; inspect current state before revising or retrying.",
                        output={
                            "error": str(error),
                            "error_type": type(error).__name__,
                            "physical_effects_may_have_occurred": True,
                        },
                    )
                else:
                    self.raise_failure("workflow_failed", error)
            self._event("workflow_completed", result=result.model_dump(mode="json"))
            return result

    async def _workflow(
        self,
        workflow_id: str,
        arguments: dict[str, Any],
        *,
        observation: dict[str, Any],
        task_language: str,
        epoch: int,
        continuation_id: str | None,
        ancestors: tuple[str, ...],
        budget_frames: tuple[tuple[ProgramLimits, dict[str, int]], ...],
        workflows: tuple[WorkflowSpec, ...] | None,
    ) -> WorkflowResult:
        if epoch != self._epoch:
            raise RuntimeError("stale workflow invocation")
        registry = (
            {item.workflow_id: item for item in workflows}
            if workflows is not None
            else self._workflows
        )
        if workflow_id in ancestors:
            raise ValueError("recursive workflow invocation")
        continuation_state = {}
        counters = {"runtime": 0, "physical": 0, "model": 0}
        if continuation_id is not None:
            previous = self._continuations.get(continuation_id)
            if (
                previous is None
                or previous["workflow_id"] != workflow_id
                or previous["ancestors"] != ancestors
                or not Draft202012Validator({"const": previous["arguments"]}).is_valid(
                    arguments
                )
            ):
                raise WorkflowContinuationError(
                    "Continuation is unknown, consumed, or belongs to different workflow/arguments/ancestry. Resume with the yielded resume_arguments unchanged; changing even max_steps or tolerance requires a fresh call without continuation_id. Limits remain cumulative. No program or physical operation was executed by this rejected call."
                )
            # Continuations execute the original complete dependency registry.
            registry = previous["registry"]
        item = registry.get(workflow_id)
        if item is None:
            raise PermissionError("workflow is absent from the bound registry")
        remaining_wall = item.limits.wall_time_s
        validate_json(item.input_schema, arguments)
        if continuation_id is not None:
            continuation_state = previous["state"]
            counters = previous["counters"]
            remaining_wall = previous["remaining_wall"]
            self._continuations.pop(continuation_id)
        if remaining_wall is not None and remaining_wall <= 0:
            raise RuntimeError("workflow execution time budget exhausted")
        inherited_frames = budget_frames or self.budget_frames.get()
        frames = (*inherited_frames, (item.limits, counters))

        async def invoke(
            method: str, name: str, values: dict[str, Any]
        ) -> dict[str, Any]:
            if name.startswith("workflow:"):
                self.charge_workflow("runtime")
                child = name.removeprefix("workflow:")
                if (
                    child not in {d.workflow_id for d in item.dependencies}
                    or method != "control"
                ):
                    raise PermissionError("undeclared workflow dependency")
                if (
                    not isinstance(values, dict)
                    or set(values) - {"arguments", "continuation_id"}
                    or "arguments" not in values
                ):
                    raise ValueError(
                        "workflow dependency call requires arguments and optional continuation_id"
                    )
                result = await self.workflow(
                    child,
                    values["arguments"],
                    continuation_id=values.get("continuation_id"),
                    observation=observation,
                    task_language=task_language,
                    epoch=epoch,
                    ancestors=(*ancestors, workflow_id),
                    budget_frames=frames,
                    workflows=tuple(registry.values()),
                )
                return result.model_dump(mode="json")
            snapshot_request = method == "control" and name == "episode:snapshot"
            if not snapshot_request:
                if name not in item.permissions:
                    raise PermissionError(
                        "workflow operation exceeds declared permissions"
                    )
                effect = self._operations[name]["effect"]
                allowed_method = {
                    "read": "observe",
                    "physical": "act",
                    "model": "control",
                    "task": "control",
                }[effect]
                if method != allowed_method:
                    raise PermissionError(
                        "runtime method does not match operation effect"
                    )
            restriction = frozenset(item.permissions) | frozenset(
                "workflow_" + d.workflow_id for d in item.dependencies
            )
            parent_authority = self.authority.get()
            program_token = self.program_context.set(item.program)
            token = self.authority.set(
                restriction
                if parent_authority is None
                else restriction & parent_authority
            )
            try:
                if snapshot_request:
                    self.charge_workflow("runtime")
                    if self._snapshot is None:
                        raise RuntimeError(
                            "workflow snapshot provider is not configured"
                        )
                    return self._snapshot()
                return await self.invoke(name, values, epoch=epoch)
            finally:
                self.authority.reset(token)
                self.program_context.reset(program_token)

        payload = {
            **(
                self._snapshot()
                if self._snapshot
                else {"observation": observation, "task_language": task_language}
            ),
            "arguments": arguments,
            "state": continuation_state,
        }
        timing = {}
        frame_token = self.budget_frames.set(frames)
        try:
            result = WorkflowResult.model_validate(
                await asyncio.wait_for(
                    self._run_program(
                        item.program,
                        item.limits.model_copy(update={"wall_time_s": remaining_wall}),
                        payload,
                        invoke,
                        timing,
                    ),
                    timeout=self.remaining_wall_time(),
                )
            )
        finally:
            self.budget_frames.reset(frame_token)
        validate_json(item.output_schema, result.output)
        if epoch != self._epoch:
            raise RuntimeError("workflow result belongs to a cancelled generation")
        if result.continuation_id is not None or result.resume_arguments is not None:
            raise ValueError(
                "workflow cannot mint its own continuation token or arguments"
            )
        if result.status == "yielded":
            continuation_id = continuation_id or uuid.uuid4().hex
            self._continuations[continuation_id] = {
                "workflow_id": workflow_id,
                "arguments": json.loads(json.dumps(arguments)),
                "ancestors": ancestors,
                "state": result.model_dump(mode="json")["state"],
                "counters": counters,
                "remaining_wall": remaining_wall - timing["elapsed_s"]
                if remaining_wall is not None
                else None,
                "registry": registry,
            }
            result = result.model_copy(
                update={
                    "continuation_id": continuation_id,
                    "resume_arguments": json.loads(json.dumps(arguments)),
                }
            )
        elif continuation_id is not None:
            self._continuations.pop(continuation_id, None)
        return result
