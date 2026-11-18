# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Episode-local state machine for the currently executing robot tool."""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from reweave.robots.execution_trace import (
    EXECUTION_OUTCOMES,
    ExecutionOutcome,
    ExecutionTrace,
)
from reweave.robots.vla_execution import VLAActionCursor, VLAChunkPrediction

ExecutionPhase: TypeAlias = Literal["idle", "running"]


@dataclass(frozen=True, slots=True)
class ActiveExecutionSnapshot:
    """Detached view of the episode-local execution state.

    ``execution_id`` identifies one planner-visible state-changing tool call.
    ``tool_name`` is that call's registered toolkit name. Every completed
    call closes its execution, including an early return from an optional check.
    Workflow and scheduler identities belong to the kernel event log; this
    state machine owns only the actual physical tool invocation.
    """

    phase: ExecutionPhase
    execution_id: str | None
    tool_name: str | None
    arguments: dict[str, Any] | None


class ActiveExecution:
    """Own the live continuation between a tool boundary and control steps.

    This object does not generate or execute repairs. It enforces the current
    runtime lifecycle and writes every transition into the run's existing
    :class:`ExecutionTrace`, which remains the single persistent trace.
    """

    def __init__(self, trace: ExecutionTrace) -> None:
        self._trace = trace
        self._phase: ExecutionPhase = "idle"
        self._execution_id: str | None = None
        self._tool_name: str | None = None
        self._arguments: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @property
    def snapshot(self) -> ActiveExecutionSnapshot:
        """Return the current state without exposing mutable runtime data."""
        with self._lock:
            return ActiveExecutionSnapshot(
                phase=self._phase,
                execution_id=self._execution_id,
                tool_name=self._tool_name,
                arguments=copy.deepcopy(self._arguments),
            )

    def _require_execution(self) -> str:
        execution_id = self._execution_id
        if execution_id is None or self._phase == "idle":
            raise RuntimeError("no robot execution is active")
        return execution_id

    def public_control_steps(self) -> list[dict[str, Any]]:
        """Complete public action evidence from this episode, without critic values."""
        return self._trace.public_control_steps()

    def begin(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Start one planner-visible state-changing tool continuation."""
        name = str(tool_name).strip()
        if not name:
            raise ValueError("active execution tool_name must be non-empty")
        with self._lock:
            if self._phase != "idle":
                raise RuntimeError(
                    "cannot begin a robot execution while another continuation "
                    f"is {self._phase}"
                )
            execution_id = uuid.uuid4().hex
            execution_arguments = copy.deepcopy(arguments or {})
            self._trace.record_execution_started(
                execution_id=execution_id,
                tool_name=name,
                arguments=execution_arguments,
            )
            self._phase = "running"
            self._execution_id = execution_id
            self._tool_name = name
            self._arguments = execution_arguments
            return execution_id

    def record_forward(
        self,
        prediction: VLAChunkPrediction | VLAActionCursor,
        *,
        instruction: str,
    ) -> None:
        """Bind one VLA forward to the currently running continuation."""
        with self._lock:
            execution_id = self._require_execution()
            if self._phase != "running":
                raise RuntimeError(
                    "cannot start a VLA forward outside a running invocation"
                )
            self._trace.record_forward(
                prediction,
                instruction=instruction,
                execution_ref=execution_id,
            )

    def discard_open_suffix(self, *, reason: str) -> int:
        """Close the active forward at a tool boundary, failure, or cancellation."""
        with self._lock:
            execution_id = self._require_execution()
            return self._trace.record_open_suffix_discarded(
                execution_ref=execution_id,
                reason=reason,
            )

    def finish(self, outcome: ExecutionOutcome) -> str:
        """Finish the active tool and return its stable execution id."""
        if outcome not in EXECUTION_OUTCOMES:
            raise ValueError(f"unsupported active execution outcome: {outcome!r}")
        with self._lock:
            execution_id = self._require_execution()
            self._trace.record_execution_finished(
                execution_ref=execution_id,
                outcome=outcome,
            )
            self._phase = "idle"
            self._execution_id = None
            self._tool_name = None
            self._arguments = None
            return execution_id
