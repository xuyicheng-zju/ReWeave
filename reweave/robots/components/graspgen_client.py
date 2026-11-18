# Copyright 2026 The ReWeave Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validated, inference-only client for a remote GraspGen-X server."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_RIGID_ATOL = 2e-3
_CHECKS_NOT_PERFORMED = {
    "reachability": "not_checked",
    "scene_collision": "not_checked",
    "path_collision": "not_checked",
}


def _validated_rigid_transform(value: Any, *, name: str) -> np.ndarray:
    object_values = np.asarray(value, dtype=object).reshape(-1)
    if any(isinstance(item, (bool, np.bool_)) for item in object_values):
        raise ValueError(f"{name} must contain numeric values, not booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values, not booleans")
    transform = raw.astype(np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=_RIGID_ATOL):
        raise ValueError(f"{name} must use homogeneous coordinates")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_RIGID_ATOL):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_RIGID_ATOL):
        raise ValueError(f"{name} rotation must be right-handed")
    return transform


def _validated_finite_triple(
    value: Any, *, name: str, positive: bool
) -> tuple[float, ...]:
    raw = np.asarray(value)
    object_values = np.asarray(value, dtype=object).reshape(-1)
    if any(isinstance(item, (bool, np.bool_)) for item in object_values):
        raise ValueError(f"{name} must contain numeric values, not booleans")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    values = raw.astype(np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain exactly three finite values")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return tuple(float(item) for item in values)


def _validated_moe_z_offsets(value: Any) -> tuple[float, ...]:
    """Validate the official GraspMoE approach-axis offset grid in centimetres."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("moe_z_offsets_cm must be a list or tuple")
    if not 1 <= len(value) <= 64:
        raise ValueError("moe_z_offsets_cm must contain between 1 and 64 values")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("moe_z_offsets_cm must contain numeric values, not booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError("moe_z_offsets_cm must contain numeric values")
    offsets = raw.astype(np.float64)
    if offsets.ndim != 1 or not np.isfinite(offsets).all():
        raise ValueError("moe_z_offsets_cm must contain finite values")
    return tuple(float(item) for item in offsets)


class GraspGenXSweepVolume(BaseModel):
    """Official geometry conditioning sent to GraspGen-X for one gripper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extents_open: tuple[float, float, float] = Field(
        description="Positive XYZ extents in metres of the fully open finger volume."
    )
    offset_open: tuple[float, float, float] = Field(
        description=(
            "XYZ centre in metres of the fully open finger volume in the canonical "
            "GraspGen-X gripper frame."
        )
    )
    extents_mid: tuple[float, float, float] = Field(
        description="Positive XYZ extents in metres of the half-open finger volume."
    )
    offset_mid: tuple[float, float, float] = Field(
        description=(
            "XYZ centre in metres of the half-open finger volume in the canonical "
            "GraspGen-X gripper frame."
        )
    )
    gripper_type: int = Field(
        strict=True,
        ge=0,
        le=2,
        description=(
            "GraspGen-X morphology code: 0 parallel two-finger, 1 revolute "
            "two-finger, or 2 revolute three-finger."
        ),
    )
    fingertip_depth: float = Field(
        strict=True,
        gt=0.0,
        description=(
            "Positive distance in metres from the canonical gripper root to the "
            "fingertip along its +Z approach axis."
        ),
    )

    @field_validator("extents_open", "extents_mid", mode="before")
    @classmethod
    def _positive_extents(cls, value: Any, info: Any) -> tuple[float, ...]:
        return _validated_finite_triple(value, name=info.field_name, positive=True)

    @field_validator("offset_open", "offset_mid", mode="before")
    @classmethod
    def _finite_offsets(cls, value: Any, info: Any) -> tuple[float, ...]:
        return _validated_finite_triple(value, name=info.field_name, positive=False)

    @field_validator("fingertip_depth")
    @classmethod
    def _finite_fingertip_depth(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fingertip_depth must be finite")
        return value


class GraspGenXGripper(BaseModel):
    """Bind one adapter arm to a model gripper and its TCP transform.

    ``grasp_frame_from_tcp`` maps target TCP coordinates into GraspGen-X's
    canonical grasp frame. A predicted public-frame grasp pose is therefore
    post-multiplied by this matrix to obtain the public-frame TCP pose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gripper_name: str = Field(
        min_length=1,
        description=(
            "Human-readable gripper asset label recorded in result provenance; "
            "the inference request is conditioned by sweep_volume, not this label."
        ),
    )
    sweep_volume: GraspGenXSweepVolume = Field(
        description=(
            "Official GraspGen-X open and half-open swept-volume conditioning "
            "for this arm's gripper."
        )
    )
    grasp_frame_from_tcp: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = Field(
        description=(
            "Rigid transform from this arm's target TCP frame into the canonical "
            "GraspGen-X grasp frame; post-multiply a predicted grasp pose by it "
            "to obtain the TCP pose."
        )
    )

    @field_validator("gripper_name")
    @classmethod
    def _nonempty_gripper_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("gripper_name cannot be blank")
        return normalized

    @field_validator("grasp_frame_from_tcp", mode="before")
    @classmethod
    def _rigid_grasp_frame_from_tcp(cls, value: Any) -> Any:
        transform = _validated_rigid_transform(value, name="grasp_frame_from_tcp")
        return tuple(tuple(float(item) for item in row) for row in transform)


class GraspGenXConfig(BaseModel):
    """Connection, sampling, and per-arm gripper bindings for GraspGen-X."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str = Field(
        description="Complete tcp://host:port address of the remote ZMQ server."
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        strict=True,
        description="Maximum seconds allowed for each ZMQ send or receive operation.",
    )
    num_grasps: int = Field(
        default=200,
        ge=1,
        strict=True,
        description="Raw grasp count requested from the GraspMoE object planner.",
    )
    max_points: int = Field(
        default=8192,
        ge=1,
        strict=True,
        description="Maximum observed object points sent after deterministic sampling.",
    )
    moe_z_offsets_cm: tuple[float, ...] = Field(
        default=(-8.0, -6.0, -4.0, -2.0, -1.0, 0.0),
        min_length=1,
        max_length=64,
        description=(
            "GraspMoE OBB offsets in centimetres along each candidate's approach "
            "axis; sent explicitly to infer_object."
        ),
    )
    grippers: dict[str, GraspGenXGripper] = Field(
        min_length=1,
        description=(
            "Per-arm sweep-volume conditioning, provenance label, and calibrated "
            "grasp-frame-to-TCP binding."
        ),
    )

    @field_validator("endpoint")
    @classmethod
    def _zmq_tcp_endpoint(cls, value: str) -> str:
        endpoint = value.strip()
        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint must contain a valid TCP port") from exc
        if (
            parsed.scheme != "tcp"
            or parsed.hostname is None
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("endpoint must have the form tcp://host:port")
        return endpoint

    @field_validator("timeout_s")
    @classmethod
    def _finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timeout_s must be finite")
        return value

    @field_validator("moe_z_offsets_cm", mode="before")
    @classmethod
    def _finite_moe_z_offsets(cls, value: Any) -> tuple[float, ...]:
        return _validated_moe_z_offsets(value)

    @field_validator("grippers")
    @classmethod
    def _valid_arm_names(
        cls, value: dict[str, GraspGenXGripper]
    ) -> dict[str, GraspGenXGripper]:
        if any(not isinstance(arm, str) or not arm.strip() for arm in value):
            raise ValueError("gripper arm names must be non-blank strings")
        if any(arm != arm.strip() for arm in value):
            raise ValueError("gripper arm names cannot have surrounding whitespace")
        return value


def load_graspgen_config(path: str | Path) -> GraspGenXConfig:
    """Load a strict GraspGen-X config from one YAML or JSON file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"GraspGen-X config not found: {config_path}")
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            payload = json.load(stream)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(stream)
        else:
            raise ValueError("GraspGen-X config path must end in .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError("GraspGen-X config must be a mapping")
    return GraspGenXConfig.model_validate(payload)


class GraspGenXClient:
    """Send independent sweep-conditioned GraspMoE requests to GraspGen-X."""

    def __init__(self, config: GraspGenXConfig) -> None:
        self.config = config

    def close(self) -> None:
        """Release client resources; each request already closes its own resources."""

    def check_ready(self) -> dict[str, Any]:
        """Check server health and sweep-conditioned object inference support."""

        health = self._request({"action": "health"})
        if health.get("status") != "ok":
            raise RuntimeError(
                f"GraspGen-X at {self.config.endpoint} is unhealthy: {health!r}"
            )
        metadata = self._request({"action": "metadata"})
        actions = metadata.get("actions")
        if not isinstance(actions, list) or "infer_object" not in actions:
            raise RuntimeError(
                "GraspGen-X metadata does not advertise sweep-conditioned "
                "'infer_object'"
            )
        return metadata

    def generate_grasps(
        self,
        points_m: np.ndarray,
        *,
        arm: str,
        frame: str,
        max_candidates: int | None = 5,
    ) -> dict[str, Any]:
        """Generate ranked TCP candidates without checking or executing motion."""

        if max_candidates is not None:
            if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
                raise TypeError("max_candidates must be an integer or None")
            if not 1 <= max_candidates <= 32:
                raise ValueError("max_candidates must lie in [1, 32]")
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError("arm must be a non-blank string")
        if arm not in self.config.grippers:
            raise ValueError(
                f"arm {arm!r} has no GraspGen-X gripper binding; "
                f"configured={sorted(self.config.grippers)}"
            )
        if not isinstance(frame, str) or not frame.strip():
            raise ValueError("frame must be a non-blank string")

        points = _validated_points(points_m)
        points = _deterministic_downsample(points, self.config.max_points)
        gripper = self.config.grippers[arm]
        response = self._request(
            {
                "action": "infer_object",
                "point_cloud": points,
                "sweep_volume_params": gripper.sweep_volume.model_dump(mode="json"),
                "planner": "graspmoe",
                "num_grasps": self.config.num_grasps,
                "moe_z_offsets_cm": list(self.config.moe_z_offsets_cm),
            }
        )
        grasps, scores, branch_tags = _validated_inference_response(response)
        order = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                branch_tags[index],
                tuple(float(value) for value in grasps[index].reshape(-1)),
            ),
        )
        if max_candidates is not None:
            order = order[:max_candidates]
        grasp_frame_from_tcp = np.asarray(
            gripper.grasp_frame_from_tcp, dtype=np.float64
        )
        candidates = []
        for rank, index in enumerate(order, start=1):
            grasp_pose = grasps[index]
            tcp_pose = grasp_pose @ grasp_frame_from_tcp
            quaternion = _rotation_to_quaternion_xyzw(tcp_pose[:3, :3])
            candidates.append(
                {
                    "candidate_id": f"grasp_{rank:02d}",
                    "score": float(scores[index]),
                    "branch_tag": branch_tags[index],
                    "position_m": [float(value) for value in tcp_pose[:3, 3]],
                    "quaternion_xyzw": [float(value) for value in quaternion],
                    "approach_direction": [float(value) for value in grasp_pose[:3, 2]],
                }
            )
        return {
            "model": "GraspGen-X",
            "planner": "graspmoe",
            "conditioning": {
                "asset_label": gripper.gripper_name,
                "sweep_volume": gripper.sweep_volume.model_dump(mode="json"),
                "moe_z_offsets_cm": list(self.config.moe_z_offsets_cm),
            },
            "arm": arm,
            "frame": frame,
            "observed_points": len(points),
            "candidates": candidates,
            "checks": dict(_CHECKS_NOT_PERFORMED),
            "scope": "grasp_pose_generation_only",
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "GraspGen-X client dependencies are missing; install ReWeave with "
                '`pip install -e ".[graspgen-x]"`'
            ) from exc

        context = zmq.Context()
        socket = None
        try:
            socket = context.socket(zmq.REQ)
            timeout_ms = max(1, math.ceil(self.config.timeout_s * 1000.0))
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.config.endpoint)
            socket.send(
                msgpack.packb(payload, use_bin_type=True, default=msgpack_numpy.encode)
            )
            packed = socket.recv()
        except zmq.error.Again as exc:
            raise TimeoutError(
                f"GraspGen-X at {self.config.endpoint} did not respond within "
                f"{self.config.timeout_s:g} seconds"
            ) from exc
        finally:
            if socket is not None:
                socket.close(linger=0)
            context.term()
        response = msgpack.unpackb(packed, raw=False, object_hook=msgpack_numpy.decode)
        if not isinstance(response, dict):
            raise RuntimeError("GraspGen-X returned a non-object response")
        if "error" in response:
            raise RuntimeError(f"GraspGen-X server error: {response['error']}")
        return response


def _validated_points(points_m: Any) -> np.ndarray:
    object_values = np.asarray(points_m, dtype=object).reshape(-1)
    if any(isinstance(item, (bool, np.bool_)) for item in object_values):
        raise ValueError("points_m must contain numeric values, not booleans")
    raw = np.asarray(points_m)
    if raw.dtype.kind not in "fiu":
        raise ValueError("points_m must contain numeric values, not booleans")
    points = raw.astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"points_m must be non-empty [N,3]; got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_m must be finite")
    return np.ascontiguousarray(points)


def _deterministic_downsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.floor(np.arange(maximum, dtype=np.float64) * len(points) / maximum)
    return np.ascontiguousarray(points[indices.astype(np.int64)])


def _validated_inference_response(
    response: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not {"grasps", "confidences", "branch_tags"}.issubset(response):
        raise RuntimeError(
            "GraspGen-X response lacks grasps, confidences, or branch_tags"
        )
    object_grasps = np.asarray(response["grasps"], dtype=object).reshape(-1)
    object_scores = np.asarray(response["confidences"], dtype=object).reshape(-1)
    if any(
        isinstance(item, (bool, np.bool_)) for item in (*object_grasps, *object_scores)
    ):
        raise RuntimeError(
            "GraspGen-X grasps and confidences must be numeric, not booleans"
        )
    raw_grasps = np.asarray(response["grasps"])
    raw_scores = np.asarray(response["confidences"])
    if raw_grasps.dtype.kind not in "fiu" or raw_scores.dtype.kind not in "fiu":
        raise RuntimeError("GraspGen-X grasps and confidences must be numeric")
    grasps = raw_grasps.astype(np.float64)
    scores = raw_scores.astype(np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise RuntimeError(f"GraspGen-X grasps must be [K,4,4]; got {grasps.shape}")
    if scores.ndim != 1 or len(scores) != len(grasps):
        raise RuntimeError("GraspGen-X returned mismatched grasps and confidences")
    raw_branch_tags = response["branch_tags"]
    if not isinstance(raw_branch_tags, list) or len(raw_branch_tags) != len(grasps):
        raise RuntimeError("GraspGen-X returned mismatched grasps and branch_tags")
    if any(not isinstance(tag, str) or not tag.strip() for tag in raw_branch_tags):
        raise RuntimeError("GraspGen-X branch_tags must be non-blank strings")
    branch_tags = tuple(tag.strip() for tag in raw_branch_tags)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError("GraspGen-X confidences must be finite values in [0,1]")
    validated = np.empty_like(grasps)
    for index, grasp in enumerate(grasps):
        try:
            validated[index] = _validated_rigid_transform(
                grasp, name=f"grasps[{index}]"
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid GraspGen-X response: {exc}") from exc
    return validated, scores, branch_tags


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to a normalized xyzw quaternion."""

    from scipy.spatial.transform import Rotation

    quaternion = Rotation.from_matrix(rotation).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


__all__ = [
    "GraspGenXClient",
    "GraspGenXConfig",
    "GraspGenXGripper",
    "GraspGenXSweepVolume",
    "load_graspgen_config",
]
