# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Translate world-frame Cartesian deltas through the live OSC contract."""

from dataclasses import dataclass
from typing import Any

import numpy as np


def collect_cartesian_control_spec(env: Any) -> dict:
    """Read controller configuration, never task/object state, in the worker."""
    controller = env.env.robots[0].part_controllers["right"]
    if (
        controller.name != "OSC_POSE"
        or controller.impedance_mode != "fixed"
        or controller.input_type != "delta"
    ):
        raise ValueError(
            "LIBERO Cartesian tools require fixed-impedance delta OSC_POSE"
        )
    frame = controller.input_ref_frame
    if frame not in {"world", "base"}:
        raise ValueError(f"unsupported OSC reference frame: {frame}")
    return {
        "input_min": np.asarray(controller.input_min).tolist(),
        "input_max": np.asarray(controller.input_max).tolist(),
        "output_min": np.asarray(controller.output_min).tolist(),
        "output_max": np.asarray(controller.output_max).tolist(),
        "world_from_control": (
            np.eye(3) if frame == "world" else np.asarray(controller.origin_ori)
        ).tolist(),
    }


@dataclass(frozen=True)
class CartesianControl:
    """Live affine OSC map: inputs are normalized, outputs are meters/radians."""

    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    # Rotation taking controller-frame vectors to world-frame vectors.
    world_from_control: np.ndarray

    @classmethod
    def from_spec(cls, spec: dict) -> "CartesianControl":
        if set(spec) != set(cls.__dataclass_fields__):
            raise ValueError("incomplete or unknown Cartesian control specification")
        obj = cls(**{k: np.asarray(v, dtype=np.float64) for k, v in spec.items()})
        for key in ("input_min", "input_max", "output_min", "output_max"):
            array = getattr(obj, key)
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError(f"{key} must contain six finite OSC bounds")
        if not (
            np.all(obj.input_max > obj.input_min)
            and np.all(obj.output_max > obj.output_min)
        ):
            raise ValueError("OSC upper bounds must exceed lower bounds")
        if not (np.all(obj.output_min < 0) and np.all(obj.output_max > 0)):
            raise ValueError("OSC physical bounds must contain zero")
        rotation = obj.world_from_control
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("world_from_control must be a proper rotation matrix")
        return obj

    def to_spec(self) -> dict:
        """Persist the exact mapping used to encode a servo's actions."""
        return {
            name: getattr(self, name).tolist() for name in self.__dataclass_fields__
        }

    def action(self, translation, rotation, gripper: float) -> np.ndarray:
        """Invert the environment's scale_action; preserve world-frame units."""
        delta = np.asarray([*translation, *rotation], dtype=np.float64)
        if delta.shape != (6,) or not np.isfinite(delta).all():
            raise ValueError("Cartesian delta must have six finite components")
        if not np.isfinite(gripper) or not -1 <= gripper <= 1:
            raise ValueError("gripper must be finite and within [-1, 1]")
        delta[:3] = self.world_from_control.T @ delta[:3]
        delta[3:] = self.world_from_control.T @ delta[3:]
        delta = np.clip(delta, self.output_min, self.output_max)
        scaled = self.input_min + (delta - self.output_min) * (
            self.input_max - self.input_min
        ) / (self.output_max - self.output_min)
        return np.concatenate((scaled, [gripper])).astype(np.float32)
