# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Typed contracts for one VLA forward and its executable actions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


def _exact_payload(
    payload: object,
    *,
    expected_fields: frozenset[str],
    contract_name: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"{contract_name} must be a mapping, got {type(payload).__name__}"
        )
    fields = set(payload)
    missing = sorted(expected_fields - fields)
    unknown = sorted(fields - expected_fields)
    if missing or unknown:
        raise RuntimeError(
            f"invalid {contract_name} fields: missing={missing}, unknown={unknown}"
        )
    return payload


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"{field_name} must be a positive integer, got {value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"{field_name} must be positive, got {parsed}")
    return parsed


def _nonnegative_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeError(f"{field_name} must be a finite number, got {value!r}")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise RuntimeError(
            f"{field_name} must be finite and nonnegative, got {parsed!r}"
        )
    return parsed


def _forward_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"forward_id must be a non-empty string, got {value!r}")
    return value


def _finite_actions(value: object, *, dimensions: int, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != dimensions or 0 in array.shape or not np.isfinite(array).all():
        raise RuntimeError(
            f"{field_name} must be a finite {dimensions}-D non-empty array; "
            f"got shape {array.shape}"
        )
    array = np.array(array, dtype=np.float32, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class VLAActionCursor:
    """One executable action and its position in the generating forward.

    ``prediction_horizon_steps`` is the full temporal horizon sampled by the
    policy. ``execution_horizon_steps`` is the contiguous window exposed for
    execution before another VLA forward is required. ``action_index`` is zero-based
    within that executable window. ``inference_performed`` is true only for
    the first action returned by a newly completed forward.
    """

    action: np.ndarray
    forward_id: str
    prediction_horizon_steps: int
    execution_horizon_steps: int
    action_index: int
    inference_performed: bool
    inference_elapsed_s: float
    # First executable sample in the original prediction (zero for prefix policies).
    prediction_start_index: int = 0

    _RPC_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "action",
            "forward_id",
            "prediction_horizon_steps",
            "execution_horizon_steps",
            "action_index",
            "inference_performed",
            "inference_elapsed_s",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _finite_actions(self.action, dimensions=1, field_name="action"),
        )
        object.__setattr__(self, "forward_id", _forward_id(self.forward_id))
        prediction_horizon = _positive_int(
            self.prediction_horizon_steps,
            field_name="prediction_horizon_steps",
        )
        execution_horizon = _positive_int(
            self.execution_horizon_steps,
            field_name="execution_horizon_steps",
        )
        if (
            type(self.prediction_start_index) is not int
            or self.prediction_start_index < 0
        ):
            raise RuntimeError("prediction_start_index must be a nonnegative integer")
        if self.prediction_start_index + execution_horizon > prediction_horizon:
            raise RuntimeError(
                "execution_horizon_steps cannot exceed "
                f"prediction_horizon_steps: {execution_horizon} > "
                f"{prediction_horizon}"
            )
        object.__setattr__(self, "prediction_horizon_steps", prediction_horizon)
        object.__setattr__(self, "execution_horizon_steps", execution_horizon)

        if isinstance(self.action_index, bool) or not isinstance(
            self.action_index, (int, np.integer)
        ):
            raise RuntimeError(
                f"action_index must be an integer, got {self.action_index!r}"
            )
        action_index = int(self.action_index)
        if not 0 <= action_index < execution_horizon:
            raise RuntimeError(
                "action_index must be within the executable horizon; "
                f"got {action_index} for horizon {execution_horizon}"
            )
        object.__setattr__(self, "action_index", action_index)

        if not isinstance(self.inference_performed, bool):
            raise RuntimeError(
                f"inference_performed must be boolean, got {self.inference_performed!r}"
            )
        if self.inference_performed != (action_index == 0):
            raise RuntimeError(
                "inference_performed must be true exactly for action_index 0"
            )
        elapsed = _nonnegative_float(
            self.inference_elapsed_s,
            field_name="inference_elapsed_s",
        )
        if not self.inference_performed and elapsed != 0:
            raise RuntimeError("buffered actions must report inference_elapsed_s=0")
        object.__setattr__(self, "inference_elapsed_s", elapsed)

    @property
    def remaining_action_steps(self) -> int:
        """Actions left in this forward after the current action."""
        return self.execution_horizon_steps - self.action_index - 1

    @classmethod
    def from_rpc(cls, payload: object) -> VLAActionCursor:
        """Validate the current single-action RPC response."""
        fields = _exact_payload(
            payload,
            expected_fields=cls._RPC_FIELDS,
            contract_name="VLA action response",
        )
        return cls(**{field: fields[field] for field in cls._RPC_FIELDS})


@dataclass(frozen=True, slots=True)
class VLAChunkPrediction:
    """One single-environment VLA forward exposed as a contiguous action window.

    ``actions`` has shape ``[execution_horizon_steps, action_dimension]``.
    The model may sample a longer ``prediction_horizon_steps`` internally.
    ``prediction_start_index`` identifies the first original sample in this
    local window. Existing ``vla.predict`` RPC responses expose a prefix (zero);
    G2's separate Forge protocol returns the full sequence and crops locally.
    """

    actions: np.ndarray
    forward_id: str
    prediction_horizon_steps: int
    execution_horizon_steps: int
    inference_elapsed_s: float
    # First executable sample in the original prediction (zero for prefix policies).
    prediction_start_index: int = 0

    _RPC_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "actions",
            "forward_id",
            "prediction_horizon_steps",
            "execution_horizon_steps",
            "inference_elapsed_s",
        }
    )

    def __post_init__(self) -> None:
        actions = _finite_actions(self.actions, dimensions=2, field_name="actions")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "forward_id", _forward_id(self.forward_id))
        prediction_horizon = _positive_int(
            self.prediction_horizon_steps,
            field_name="prediction_horizon_steps",
        )
        execution_horizon = _positive_int(
            self.execution_horizon_steps,
            field_name="execution_horizon_steps",
        )
        if execution_horizon != actions.shape[0]:
            raise RuntimeError(
                "execution_horizon_steps must equal the returned action count; "
                f"got horizon {execution_horizon}, actions {actions.shape[0]}"
            )
        if (
            type(self.prediction_start_index) is not int
            or self.prediction_start_index < 0
        ):
            raise RuntimeError("prediction_start_index must be a nonnegative integer")
        if self.prediction_start_index + execution_horizon > prediction_horizon:
            raise RuntimeError(
                "execution_horizon_steps cannot exceed "
                f"prediction_horizon_steps: {execution_horizon} > "
                f"{prediction_horizon}"
            )
        object.__setattr__(self, "prediction_horizon_steps", prediction_horizon)
        object.__setattr__(self, "execution_horizon_steps", execution_horizon)
        object.__setattr__(
            self,
            "inference_elapsed_s",
            _nonnegative_float(
                self.inference_elapsed_s,
                field_name="inference_elapsed_s",
            ),
        )

    @classmethod
    def from_rpc(cls, payload: object) -> VLAChunkPrediction:
        """Validate a single-environment batched ``vla.predict`` response."""
        fields = _exact_payload(
            payload,
            expected_fields=cls._RPC_FIELDS,
            contract_name="VLA chunk response",
        )
        batched_actions = _finite_actions(
            fields["actions"], dimensions=3, field_name="actions"
        )
        if batched_actions.shape[0] != 1:
            raise RuntimeError(
                "VLA chunk response is single-environment only; "
                f"got batch size {batched_actions.shape[0]}"
            )
        return cls(
            actions=batched_actions[0],
            forward_id=fields["forward_id"],
            prediction_horizon_steps=fields["prediction_horizon_steps"],
            execution_horizon_steps=fields["execution_horizon_steps"],
            inference_elapsed_s=fields["inference_elapsed_s"],
        )

    def iter_actions(self) -> Iterator[VLAActionCursor]:
        """Yield every action with its authoritative forward cursor."""
        for action_index, action in enumerate(self.actions):
            yield VLAActionCursor(
                action=action,
                forward_id=self.forward_id,
                prediction_horizon_steps=self.prediction_horizon_steps,
                execution_horizon_steps=self.execution_horizon_steps,
                action_index=action_index,
                prediction_start_index=self.prediction_start_index,
                inference_performed=action_index == 0,
                inference_elapsed_s=(
                    self.inference_elapsed_s if action_index == 0 else 0.0
                ),
            )


class VLAActionBuffer:
    """Own the unexecuted action suffix of one agent-side VLA forward.

    LIBERO returns the whole executable prefix to the agent, so this object is
    the authority for which generated action is next and how much of the same
    forward remains. ``discard_remaining`` is irreversible: once a runtime
    interruption removes a suffix, that buffer cannot yield another action.
    """

    def __init__(self, prediction: VLAChunkPrediction) -> None:
        self._prediction = prediction
        self._actions = tuple(prediction.iter_actions())
        self._next_action_index = 0
        self._pending_action: VLAActionCursor | None = None

    @property
    def forward_id(self) -> str:
        """Identifier of the VLA forward that created this buffer."""
        return self._prediction.forward_id

    @property
    def prediction(self) -> VLAChunkPrediction:
        """Immutable prediction contract backing this buffer."""
        return self._prediction

    @property
    def remaining_action_steps(self) -> int:
        """Number of actions that can still be executed from this buffer."""
        return len(self._actions) - self._next_action_index

    @property
    def action_dimension(self) -> int:
        """Action width of the owned forward."""
        return int(self._prediction.actions.shape[1])

    @property
    def remaining_actions(self) -> np.ndarray:
        """Return a read-only copy of the complete unexecuted suffix."""
        if self._pending_action is not None:
            raise RuntimeError("cannot inspect a buffer with a pending physical action")
        suffix = np.array(
            self._prediction.actions[self._next_action_index :],
            dtype=np.float32,
            copy=True,
        )
        suffix.setflags(write=False)
        return suffix

    def snapshot(self) -> dict[str, Any]:
        """Return the JSON-safe leftover suffix owned by this live chunk."""
        actions = self.remaining_actions
        return {
            "forward_id": self.forward_id,
            "remaining_action_steps": self.remaining_action_steps,
            "action_dimension": self.action_dimension,
            "actions": actions.tolist(),
        }

    def pop_next(self) -> VLAActionCursor | None:
        """Return the next action pending physical execution.

        The action remains part of the unexecuted suffix until
        :meth:`release_selected` settles its executed or uncertain outcome. This preserves
        exact suffix ownership when cancellation arrives after selection but
        before the physical control boundary.
        """
        if self._pending_action is not None:
            raise RuntimeError(
                "cannot select another VLA action before confirming the pending one"
            )
        if self._next_action_index == len(self._actions):
            return None
        action = self._actions[self._next_action_index]
        self._pending_action = action
        return action

    def release_selected(self, action: VLAActionCursor) -> None:
        """Release ownership after the Host records an executed or uncertain outcome."""
        if self._pending_action is None:
            raise RuntimeError("no VLA action is pending execution")
        if action is not self._pending_action:
            raise ValueError("released VLA action differs from the pending action")
        self._next_action_index += 1
        self._pending_action = None

    def discard_remaining(self) -> int:
        """Irreversibly remove and count the current unexecuted suffix."""
        discarded = self.remaining_action_steps
        self._next_action_index = len(self._actions)
        self._pending_action = None
        return discarded


# Complete scalar telemetry values recorded in public or simulator-private planes.
EvidenceScalar = bool | int | float | str


@dataclass(frozen=True, slots=True)
class VLAExecution:
    """Actual environment actions applied during one receding-horizon call."""

    executed_action_steps: int

    def __post_init__(self) -> None:
        value = self.executed_action_steps
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("executed_action_steps must be a nonnegative integer")
