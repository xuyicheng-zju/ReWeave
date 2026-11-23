# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Append-only causal trace for VLA forwards and environment control steps."""

from __future__ import annotations

import copy
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from reweave.robots.vla_execution import (
    EvidenceScalar,
    VLAActionCursor,
    VLAChunkPrediction,
)

ExecutionOutcome: TypeAlias = Literal[
    "completed",
    "succeeded",
    "interrupted",
    "failed",
    "cancelled",
]
# Complete outcome vocabulary persisted by ``active_execution_finished``.
EXECUTION_OUTCOMES = frozenset(
    {
        "completed",
        "succeeded",
        "interrupted",
        "failed",
        "cancelled",
    }
)


def _serialize_evidence(
    values: Mapping[str, EvidenceScalar],
    *,
    plane: str,
) -> dict[str, EvidenceScalar]:
    """Normalize one evidence plane before it enters the trusted trace."""
    result: dict[str, EvidenceScalar] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{plane} evidence names must be non-empty")
        if plane == "public" and name.startswith("privileged."):
            raise ValueError(
                f"public evidence cannot use the privileged namespace: {name!r}"
            )
        if plane == "critic_private" and not name.startswith("privileged."):
            raise ValueError(
                f"critic-private evidence must use privileged.* names: {name!r}"
            )
        if isinstance(raw_value, (np.bool_, bool)):
            value: EvidenceScalar = bool(raw_value)
        elif isinstance(raw_value, (np.integer, int)):
            value = int(raw_value)
        elif isinstance(raw_value, (np.floating, float)):
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"{plane} evidence {name!r} must be finite")
        elif isinstance(raw_value, str):
            value = raw_value
        else:
            raise TypeError(
                f"{plane} evidence {name!r} has unsupported type "
                f"{type(raw_value).__name__}"
            )
        result[name] = value
    return result


@dataclass(slots=True)
class _ForwardProgress:
    """Mutable trace-owned execution position for one recorded VLA forward."""

    execution_ref: str
    execution_horizon_steps: int
    physical_call: dict[str, Any]
    prediction_start_index: int
    next_action_index: int = 0
    suffix_closed: bool = False


@dataclass(slots=True)
class _ExecutionProgress:
    """Trace validation state for one planner-visible tool continuation."""

    tool_name: str
    physical_call: dict[str, Any]
    phase: str = "running"


class ExecutionTrace:
    """Own the ordered execution events for one Harness run.

    ``states.json`` records planner-visible tool boundaries. This trace is a
    separate authority for the faster execution plane: VLA forwards,
    environment control steps and discarded suffixes. Passing no path keeps the same
    semantics in memory for focused tests.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._events: list[dict[str, Any]] = []
        self._forward_progress: dict[str, _ForwardProgress] = {}
        self._execution_progress: dict[str, _ExecutionProgress] = {}
        self._active_execution_ref: str | None = None
        self._control_records: list[dict[str, Any]] = []
        self._telemetry_steps: set[int] = set()
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("")

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Detached snapshot of events recorded so far, including nested values."""
        with self._lock:
            return tuple(copy.deepcopy(self._events))

    def _append(self, event: dict[str, Any]) -> None:
        event["event_index"] = len(self._events)
        self._events.append(event)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def public_control_steps(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                {
                    **{
                        key: value
                        for key, value in event.items()
                        if key
                        not in {
                            "critic_private_evidence",
                            "physical_features",
                            "evidence_event_indices",
                            "physical_features_event_index",
                        }
                    },
                    "evidence_event_indices": {
                        "public_evidence": event["evidence_event_indices"][
                            "public_evidence"
                        ],
                    },
                }
                for event in merge_control_telemetry(self._events)
                if event["event_type"] == "control_step"
            ]
            return json.loads(json.dumps(rows))

    def record_execution_started(
        self,
        *,
        execution_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Record the only planner-visible tool continuation currently active."""
        identifier = str(execution_id).strip()
        name = str(tool_name).strip()
        if not identifier or not name:
            raise ValueError("execution_id and tool_name must be non-empty")
        try:
            serialized_arguments = json.loads(json.dumps(arguments))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "active execution arguments must be JSON serializable"
            ) from exc
        if not isinstance(serialized_arguments, dict):
            raise ValueError("active execution arguments must be an object")
        with self._lock:
            if identifier in self._execution_progress:
                raise ValueError(f"duplicate active execution id: {identifier!r}")
            if self._active_execution_ref is not None:
                raise ValueError(
                    "another active execution has not finished: "
                    f"{self._active_execution_ref!r}"
                )
            self._execution_progress[identifier] = _ExecutionProgress(
                tool_name=name,
                physical_call={
                    "physical_call_ref": identifier,
                    "tool_name": name,
                    "arguments": serialized_arguments,
                },
            )
            self._active_execution_ref = identifier
            self._append(
                {
                    "event_type": "active_execution_started",
                    "execution_id": identifier,
                    "tool_name": name,
                    "arguments": serialized_arguments,
                }
            )

    def _execution(
        self,
        execution_ref: str,
        *,
        phases: set[str],
    ) -> _ExecutionProgress:
        progress = self._execution_progress.get(execution_ref)
        if progress is None:
            raise ValueError(
                f"event references unknown active execution: {execution_ref!r}"
            )
        if self._active_execution_ref != execution_ref:
            raise ValueError(
                f"execution is not active: {execution_ref!r}; active="
                f"{self._active_execution_ref!r}"
            )
        if progress.phase not in phases:
            raise ValueError(
                f"execution {execution_ref!r} is {progress.phase}, expected "
                f"one of {sorted(phases)}"
            )
        return progress

    def record_forward(
        self,
        prediction: VLAChunkPrediction | VLAActionCursor,
        *,
        instruction: str,
        execution_ref: str,
    ) -> None:
        """Record one heavy VLA inference before any generated action runs."""
        prompt = str(instruction).strip()
        if not prompt:
            raise ValueError("VLA forward instruction must be non-empty")
        if (
            isinstance(prediction, VLAActionCursor)
            and not prediction.inference_performed
        ):
            raise ValueError(
                "record_forward requires the first action of a new VLA forward"
            )
        with self._lock:
            self._execution(execution_ref, phases={"running"})
            if prediction.forward_id in self._forward_progress:
                raise ValueError(f"duplicate VLA forward_id: {prediction.forward_id!r}")
            open_forward_ids = sorted(
                forward_id
                for forward_id, progress in self._forward_progress.items()
                if not progress.suffix_closed
            )
            if open_forward_ids:
                raise ValueError(
                    "new VLA forward requires the previous action suffix to be "
                    f"executed or discarded; open forwards: {open_forward_ids}"
                )
            self._forward_progress[prediction.forward_id] = _ForwardProgress(
                execution_ref=execution_ref,
                execution_horizon_steps=prediction.execution_horizon_steps,
                prediction_start_index=prediction.prediction_start_index,
                physical_call=dict(
                    self._execution_progress[execution_ref].physical_call
                ),
            )
            self._append(
                {
                    "event_type": "vla_forward",
                    "forward_id": prediction.forward_id,
                    "execution_ref": execution_ref,
                    "instruction": prompt,
                    "physical_call_ref": self._forward_progress[
                        prediction.forward_id
                    ].physical_call["physical_call_ref"],
                    "tool_name": self._forward_progress[
                        prediction.forward_id
                    ].physical_call["tool_name"],
                    "prediction_horizon_steps": (prediction.prediction_horizon_steps),
                    "execution_horizon_steps": (prediction.execution_horizon_steps),
                    "inference_elapsed_s": prediction.inference_elapsed_s,
                    "prediction_start_index": prediction.prediction_start_index,
                }
            )

    def _control_progress(
        self, action: VLAActionCursor, *, execution_ref: str | None = None
    ) -> _ForwardProgress:
        """Validate a cursor while holding the trace lock; do not advance it."""
        progress = self._forward_progress.get(action.forward_id)
        if progress is None:
            raise ValueError(
                "control step references an unrecorded VLA forward: "
                f"{action.forward_id!r}"
            )
        if execution_ref is not None and progress.execution_ref != execution_ref:
            raise ValueError("control step belongs to another tool execution")
        self._execution(progress.execution_ref, phases={"running"})
        if progress.suffix_closed:
            raise ValueError(
                "control step references a VLA forward whose suffix is closed: "
                f"{action.forward_id!r}"
            )
        if action.execution_horizon_steps != progress.execution_horizon_steps:
            raise ValueError(
                "control step execution horizon differs from its recorded "
                f"forward: {action.execution_horizon_steps} != "
                f"{progress.execution_horizon_steps}"
            )
        if action.prediction_start_index != progress.prediction_start_index:
            raise ValueError(
                "control step source index differs from its recorded forward"
            )
        if action.action_index != progress.next_action_index:
            raise ValueError(
                "VLA actions must execute in cursor order: expected index "
                f"{progress.next_action_index}, got {action.action_index}"
            )
        return progress

    def validate_control_step(
        self, action: VLAActionCursor, *, execution_ref: str
    ) -> None:
        """Reject invalid/replayed cursors before the owner submits physical I/O.

        The active execution serializes submission and settlement. Validation
        does not consume the cursor; only an acknowledgement or unknown outcome
        advances it.
        """
        with self._lock:
            self._control_progress(action, execution_ref=execution_ref)

    def record_control_step(
        self,
        action: VLAActionCursor,
        *,
        terminated: object,
        truncated: object,
        public_evidence: Mapping[str, EvidenceScalar] | None = None,
        critic_private_evidence: Mapping[str, EvidenceScalar] | None = None,
    ) -> int:
        """Record one VLA action and its trusted post-action evidence."""
        public = _serialize_evidence(public_evidence or {}, plane="public")
        private = _serialize_evidence(
            critic_private_evidence or {},
            plane="critic_private",
        )
        duplicate_features = set(public) & set(private)
        if duplicate_features:
            raise ValueError(
                "control-step evidence cannot occur in both planes: "
                f"{sorted(duplicate_features)}"
            )
        with self._lock:
            progress = self._control_progress(action)
            progress.next_action_index += 1
            if progress.next_action_index == progress.execution_horizon_steps:
                progress.suffix_closed = True
            event = {
                "event_type": "control_step",
                "control_step_index": len(self._control_records),
                "forward_ref": action.forward_id,
                "execution_ref": progress.execution_ref,
                "physical_call_ref": progress.physical_call["physical_call_ref"],
                "tool_name": progress.physical_call["tool_name"],
                "action_index": action.action_index,
                "source_action_index": action.prediction_start_index
                + action.action_index,
                "action": action.action.tolist(),
                "terminated": bool(np.asarray(terminated).any()),
                "truncated": bool(np.asarray(truncated).any()),
            }
            if public:
                event["public_evidence"] = public
            if private:
                event["critic_private_evidence"] = private
            self._append(event)
            self._control_records.append(event)
            return event["control_step_index"]

    def record_scripted_control_step(
        self,
        *,
        execution_ref: str,
        action_index: int,
        action: object,
        terminated: object,
        truncated: object,
        public_evidence: Mapping[str, EvidenceScalar] | None = None,
    ) -> int:
        """Record one non-VLA controller action in its active tool execution."""
        public = _serialize_evidence(public_evidence or {}, plane="public")
        index = int(action_index)
        if index < 0:
            raise ValueError("scripted action index must be non-negative")
        serialized_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.isfinite(serialized_action).all():
            raise ValueError("scripted action must contain only finite values")
        with self._lock:
            progress = self._execution(execution_ref, phases={"running"})
            event = {
                "event_type": "control_step",
                "control_step_index": len(self._control_records),
                "execution_ref": execution_ref,
                "physical_call_ref": progress.physical_call["physical_call_ref"],
                "tool_name": progress.physical_call["tool_name"],
                "forward_ref": None,
                "action_index": index,
                "action": serialized_action.tolist(),
                "terminated": bool(np.asarray(terminated).any()),
                "truncated": bool(np.asarray(truncated).any()),
            }
            if public:
                event["public_evidence"] = public
            self._append(event)
            self._control_records.append(event)
            return event["control_step_index"]

    def record_action_outcome_unknown(
        self,
        *,
        execution_ref: str,
        error: Exception,
        action: VLAActionCursor | None = None,
        requested_action=None,
        action_index: int | None = None,
    ) -> None:
        """Separate an unacknowledged action from the provably unexecuted suffix."""
        with self._lock:
            owner = self._execution(execution_ref, phases={"running"})
            if action is not None:
                progress = self._control_progress(action, execution_ref=execution_ref)
                progress.next_action_index += 1
                progress.suffix_closed = (
                    progress.next_action_index == progress.execution_horizon_steps
                )
            self._append(
                {
                    "event_type": "action_outcome_unknown",
                    "execution_ref": execution_ref,
                    "physical_call_ref": owner.physical_call["physical_call_ref"],
                    "forward_ref": action.forward_id if action is not None else None,
                    "action_index": action.action_index
                    if action is not None
                    else action_index,
                    "source_action_index": action.prediction_start_index
                    + action.action_index
                    if action is not None
                    else None,
                    "requested_action": np.asarray(
                        action.action if action is not None else requested_action
                    ).tolist(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    def record_reset_state(self, *, physical_features, error=None):
        """Record an initial or subsequent reset boundary without charging an action."""
        with self._lock:
            kind = (
                "reset_state"
                if any(e["event_type"] == "initial_state" for e in self._events)
                else "initial_state"
            )
            self._append(
                {
                    "event_type": kind,
                    "physical_features": physical_features,
                    "error": error,
                }
            )

    def record_telemetry(
        self,
        control_step_index: int,
        *,
        public_evidence=None,
        critic_private_evidence=None,
        error: str | None = None,
        physical_features=None,
        action_wall_time_s=None,
    ) -> None:
        """Append post-action processing evidence to an already committed action."""
        if action_wall_time_s is not None and (
            isinstance(action_wall_time_s, bool)
            or not isinstance(action_wall_time_s, (int, float))
            or not math.isfinite(action_wall_time_s)
            or action_wall_time_s < 0
        ):
            raise ValueError(
                "action acknowledgement time must be finite and nonnegative"
            )
        public = _serialize_evidence(public_evidence or {}, plane="public")
        private = _serialize_evidence(
            critic_private_evidence or {}, plane="critic_private"
        )
        with self._lock:
            if (
                isinstance(control_step_index, bool)
                or not isinstance(control_step_index, int)
                or not 0 <= control_step_index < len(self._control_records)
            ):
                raise ValueError("telemetry references an unrecorded physical action")
            if control_step_index in self._telemetry_steps:
                raise ValueError("duplicate telemetry for a physical action")
            control = self._control_records[control_step_index]
            self._telemetry_steps.add(control_step_index)
            self._append(
                {
                    "event_type": "control_telemetry",
                    "physical_features": physical_features or {},
                    "action_wall_time_s": action_wall_time_s,
                    "control_step_index": control_step_index,
                    "execution_ref": control["execution_ref"],
                    "control_event_index": control["event_index"],
                    "public_evidence": public,
                    "critic_private_evidence": private,
                    "error": error,
                }
            )

    def record_action_suffix_discarded(
        self,
        action: VLAActionCursor,
        *,
        reason: str,
    ) -> int:
        """Close and record the unexecuted suffix after ``action``.

        Environment termination closes the current forward. Cancellation and
        tool-boundary cleanup close all owned suffixes through `discard_open_suffix`.
        The returned count is zero when the action already exhausted its
        forward, in which case no suffix event is appended.
        """
        reasons = {"episode_end"}
        if reason not in reasons:
            raise ValueError(
                f"unsupported action suffix discard reason {reason!r}; "
                f"expected one of {sorted(reasons)}"
            )

        with self._lock:
            progress = self._forward_progress.get(action.forward_id)
            if progress is None:
                raise ValueError(
                    "suffix discard references an unrecorded VLA forward: "
                    f"{action.forward_id!r}"
                )
            self._execution(progress.execution_ref, phases={"running"})
            expected_next = action.action_index + 1
            if progress.next_action_index != expected_next:
                raise ValueError(
                    "suffix discard must follow its last executed action: "
                    f"expected next index {expected_next}, trace has "
                    f"{progress.next_action_index}"
                )
            discarded = progress.execution_horizon_steps - expected_next
            progress.suffix_closed = True
            progress.next_action_index = progress.execution_horizon_steps
            if discarded:
                self._append(
                    {
                        "event_type": "action_suffix_discarded",
                        "forward_ref": action.forward_id,
                        "first_discarded_action_index": expected_next,
                        "discarded_action_steps": discarded,
                        "reason": reason,
                        "cause_ref": None,
                    }
                )
            return discarded

    def record_open_suffix_discarded(
        self,
        *,
        execution_ref: str,
        reason: str,
    ) -> int:
        """Close the current forward at a tool boundary, failure, or cancellation.

        Unlike :meth:`record_action_suffix_discarded`, this transition may
        happen between control steps, so the first unexecuted index comes from
        trace-owned forward progress rather than from a supplied action.
        """
        reasons = {
            "tool_failed",
            "tool_cancelled",
            "tool_boundary",
        }
        if reason not in reasons:
            raise ValueError(
                f"unsupported execution suffix discard reason {reason!r}; "
                f"expected one of {sorted(reasons)}"
            )
        with self._lock:
            self._execution(execution_ref, phases={"running"})
            open_forwards = [
                (forward_id, progress)
                for forward_id, progress in self._forward_progress.items()
                if progress.execution_ref == execution_ref
                and not progress.suffix_closed
            ]
            if len(open_forwards) > 1:
                raise RuntimeError(
                    "one active execution owns multiple open VLA forwards: "
                    f"{[forward_id for forward_id, _ in open_forwards]}"
                )
            if not open_forwards:
                return 0
            forward_id, progress = open_forwards[0]
            first_discarded = progress.next_action_index
            discarded = progress.execution_horizon_steps - first_discarded
            progress.suffix_closed = True
            progress.next_action_index = progress.execution_horizon_steps
            if discarded:
                self._append(
                    {
                        "event_type": "action_suffix_discarded",
                        "forward_ref": forward_id,
                        "first_discarded_action_index": first_discarded,
                        "discarded_action_steps": discarded,
                        "reason": reason,
                        "cause_ref": execution_ref,
                    }
                )
            return discarded

    def record_execution_finished(
        self,
        *,
        execution_ref: str,
        outcome: ExecutionOutcome,
    ) -> None:
        """Close a planner-visible continuation after all suffixes are closed."""
        if outcome not in EXECUTION_OUTCOMES:
            raise ValueError(
                f"unsupported active execution outcome {outcome!r}; "
                f"expected one of {sorted(EXECUTION_OUTCOMES)}"
            )
        with self._lock:
            execution = self._execution(
                execution_ref,
                phases={"running"},
            )
            open_forward_ids = sorted(
                forward_id
                for forward_id, progress in self._forward_progress.items()
                if progress.execution_ref == execution_ref
                and not progress.suffix_closed
            )
            if open_forward_ids:
                raise ValueError(
                    "active execution cannot finish with an open action suffix: "
                    f"{open_forward_ids}"
                )
            execution.phase = "finished"
            self._active_execution_ref = None
            self._append(
                {
                    "event_type": "active_execution_finished",
                    "execution_ref": execution_ref,
                    "outcome": outcome,
                }
            )


def merge_control_telemetry(events):
    """Join disjoint per-key evidence, preserving every actual source event.

    Derived ``evidence_event_indices`` maps each plane's key to its original
    event index. Identical overlapping values retain the control source; a
    conflicting value or duplicate telemetry is rejected. Originals never change.
    """
    rows = json.loads(json.dumps(events))
    controls = {}
    planes = ("public_evidence", "critic_private_evidence")
    for row in rows:
        if row["event_type"] == "control_step":
            index = row["control_step_index"]
            if index in controls:
                raise ValueError("duplicate control step identity")
            controls[index] = row
            row["physical_features_event_index"] = (
                row["event_index"] if row.get("physical_features") else None
            )
            row["evidence_event_indices"] = {
                plane: dict.fromkeys(row.get(plane, {}), row["event_index"])
                for plane in planes
            }
        elif row["event_type"] == "control_telemetry":
            control = controls.get(row["control_step_index"])
            if (
                control is None
                or control["execution_ref"] != row["execution_ref"]
                or control["event_index"] != row["control_event_index"]
            ):
                raise ValueError("telemetry has no matching preceding physical action")
            if "telemetry_event_index" in control:
                raise ValueError("duplicate telemetry for a physical action")
            for plane in planes:
                original = control.get(plane, {})
                added = row.get(plane, {})
                if not isinstance(original, dict) or not isinstance(added, dict):
                    raise ValueError("control evidence planes must be dictionaries")
                conflicts = {
                    key
                    for key in original.keys() & added.keys()
                    if original[key] != added[key]
                    or type(original[key]) is not type(added[key])
                }
                if conflicts:
                    raise ValueError(
                        f"telemetry would overwrite conflicting {plane} evidence: {sorted(conflicts)}"
                    )
                merged = {**added, **original}
                control[plane] = _serialize_evidence(
                    merged,
                    plane="public" if plane == "public_evidence" else "critic_private",
                )
                control["evidence_event_indices"][plane].update(
                    {key: row["event_index"] for key in added if key not in original}
                )
            original_features = control.get("physical_features", {})
            telemetry_features = row.get("physical_features", {})
            if (
                original_features
                and telemetry_features
                and original_features != telemetry_features
            ):
                raise ValueError(
                    "telemetry would overwrite conflicting physical feature plane"
                )
            if telemetry_features and not original_features:
                control["physical_features"] = telemetry_features
                control["physical_features_event_index"] = row["event_index"]
            if set(control["public_evidence"]) & set(
                control["critic_private_evidence"]
            ):
                raise ValueError("control-step evidence cannot occur in both planes")
            control["telemetry_event_index"] = row["event_index"]
            control["telemetry_error"] = row["error"]
            if row.get("action_wall_time_s") is not None:
                if control.get("action_wall_time_s") not in (
                    None,
                    row["action_wall_time_s"],
                ):
                    raise ValueError(
                        "telemetry would overwrite conflicting action timing"
                    )
                control["action_wall_time_s"] = row["action_wall_time_s"]
    return rows
