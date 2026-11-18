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

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from pydantic import ValidationError

from reweave.robots.components.graspgen_client import (
    GraspGenXClient,
    GraspGenXConfig,
    GraspGenXGripper,
    GraspGenXSweepVolume,
    load_graspgen_config,
)

msgpack = pytest.importorskip("msgpack")
msgpack_numpy = pytest.importorskip("msgpack_numpy")
zmq = pytest.importorskip("zmq")


def _sweep_volume() -> GraspGenXSweepVolume:
    return GraspGenXSweepVolume(
        extents_open=[0.08, 0.02, 0.06],
        offset_open=[0.0, 0.0, 0.05],
        extents_mid=[0.04, 0.02, 0.06],
        offset_mid=[0.0, 0.0, 0.05],
        gripper_type=0,
        fingertip_depth=0.08,
    )


def _gripper(transform: list[list[float]] | None = None) -> GraspGenXGripper:
    return GraspGenXGripper(
        gripper_name="franka_panda",
        sweep_volume=_sweep_volume(),
        grasp_frame_from_tcp=transform or np.eye(4).tolist(),
    )


def _config(endpoint: str, **overrides: Any) -> GraspGenXConfig:
    values = {
        "endpoint": endpoint,
        "timeout_s": 1.0,
        "num_grasps": 17,
        "max_points": 4,
        "grippers": {"right": _gripper()},
    }
    values.update(overrides)
    return GraspGenXConfig.model_validate(values)


def _serve(
    responses: list[dict[str, Any]],
    *,
    inspect: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, threading.Thread]:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")

    def run() -> None:
        try:
            for response in responses:
                request = msgpack.unpackb(
                    socket.recv(), raw=False, object_hook=msgpack_numpy.decode
                )
                if inspect is not None:
                    inspect(request)
                socket.send(
                    msgpack.packb(
                        response, use_bin_type=True, default=msgpack_numpy.encode
                    )
                )
        finally:
            socket.close(linger=0)
            context.term()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return f"tcp://127.0.0.1:{port}", thread


def _pose(*, x: float = 0.0, rotation: np.ndarray | None = None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def test_real_zmq_round_trip_uses_graspmoe_and_returns_tcp_candidates() -> None:
    requests: list[dict[str, Any]] = []
    rotation_z_90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    endpoint, server = _serve(
        [
            {
                "grasps": np.stack(
                    [_pose(x=0.2), _pose(x=0.1, rotation=rotation_z_90)]
                ),
                "confidences": np.asarray([0.25, 0.9], dtype=np.float32),
                "branch_tags": ["diffusion", "obb"],
                "timing": {"infer_ms": 1.0},
            }
        ],
        inspect=requests.append,
    )
    grasp_frame_from_tcp = np.eye(4)
    grasp_frame_from_tcp[2, 3] = 0.05
    config = _config(
        endpoint,
        grippers={"right": _gripper(grasp_frame_from_tcp.tolist())},
    )
    points = np.arange(18, dtype=np.float32).reshape(6, 3) / 100.0

    result = GraspGenXClient(config).generate_grasps(
        points, arm="right", frame="world", max_candidates=2
    )
    server.join(timeout=2.0)

    assert not server.is_alive()
    assert len(requests) == 1
    request = requests[0]
    assert request["action"] == "infer_object"
    assert request["planner"] == "graspmoe"
    assert request["num_grasps"] == 17
    assert request["moe_z_offsets_cm"] == [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0]
    assert request["sweep_volume_params"] == _sweep_volume().model_dump(mode="json")
    assert "gripper_name" not in request
    assert "grasp_threshold" not in request
    assert "topk_num_grasps" not in request
    np.testing.assert_array_equal(request["point_cloud"], points[[0, 1, 3, 4]])

    assert result == {
        "model": "GraspGen-X",
        "planner": "graspmoe",
        "conditioning": {
            "asset_label": "franka_panda",
            "sweep_volume": _sweep_volume().model_dump(mode="json"),
            "moe_z_offsets_cm": [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0],
        },
        "arm": "right",
        "frame": "world",
        "observed_points": 4,
        "candidates": [
            {
                "candidate_id": "grasp_01",
                "score": pytest.approx(0.9),
                "branch_tag": "obb",
                "position_m": pytest.approx([0.1, 0.0, 0.05]),
                "quaternion_xyzw": pytest.approx(
                    [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
                ),
                "approach_direction": [0.0, 0.0, 1.0],
            },
            {
                "candidate_id": "grasp_02",
                "score": pytest.approx(0.25),
                "branch_tag": "diffusion",
                "position_m": pytest.approx([0.2, 0.0, 0.05]),
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "approach_direction": [0.0, 0.0, 1.0],
            },
        ],
        "checks": {
            "reachability": "not_checked",
            "scene_collision": "not_checked",
            "path_collision": "not_checked",
        },
        "scope": "grasp_pose_generation_only",
    }


def test_check_ready_performs_health_then_metadata() -> None:
    actions: list[str] = []
    metadata = {"actions": ["health", "metadata", "infer_object"], "model": {}}
    endpoint, server = _serve(
        [{"status": "ok"}, metadata],
        inspect=lambda request: actions.append(request["action"]),
    )

    assert GraspGenXClient(_config(endpoint)).check_ready() == metadata
    server.join(timeout=2.0)
    assert actions == ["health", "metadata"]


def test_check_ready_rejects_server_with_only_removed_named_infer() -> None:
    endpoint, server = _serve(
        [
            {"status": "ok"},
            {"actions": ["health", "metadata", "infer"], "model": {}},
        ]
    )

    with pytest.raises(RuntimeError, match="infer_object"):
        GraspGenXClient(_config(endpoint)).check_ready()
    server.join(timeout=2.0)


def test_internal_unbounded_mode_returns_every_sorted_branch() -> None:
    endpoint, server = _serve(
        [
            {
                "grasps": np.stack([_pose(x=0.3), _pose(x=0.1), _pose(x=0.2)]),
                "confidences": np.asarray([0.2, 0.9, 0.5], dtype=np.float32),
                "branch_tags": ["diffusion", "obb", "diffusion"],
            }
        ]
    )

    result = GraspGenXClient(_config(endpoint)).generate_grasps(
        np.zeros((5, 3)), arm="right", frame="world", max_candidates=None
    )
    server.join(timeout=2.0)

    assert [candidate["branch_tag"] for candidate in result["candidates"]] == [
        "obb",
        "diffusion",
        "diffusion",
    ]
    assert [candidate["position_m"][0] for candidate in result["candidates"]] == (
        pytest.approx([0.1, 0.2, 0.3])
    )


@pytest.mark.parametrize(
    "response",
    [
        {"grasps": np.zeros((1, 4, 4), dtype=np.float32)},
        {
            "grasps": np.zeros((1, 3, 3), dtype=np.float32),
            "confidences": np.asarray([0.5], dtype=np.float32),
            "branch_tags": ["diffusion"],
        },
        {
            "grasps": np.zeros((1, 4, 4), dtype=np.float32),
            "confidences": np.asarray([0.5], dtype=np.float32),
        },
        {
            "grasps": np.stack([np.eye(4), np.eye(4)]),
            "confidences": np.asarray([0.5], dtype=np.float32),
            "branch_tags": ["diffusion", "obb"],
        },
        {
            "grasps": np.stack([np.eye(4)]),
            "confidences": np.asarray([1.2], dtype=np.float32),
            "branch_tags": ["diffusion"],
        },
        {
            "grasps": np.stack([np.eye(4)]),
            "confidences": np.asarray([0.5], dtype=np.float32),
            "branch_tags": [],
        },
        {
            "grasps": [[[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            "confidences": [0.5],
            "branch_tags": ["diffusion"],
        },
        {
            "grasps": np.stack([np.eye(4)]),
            "confidences": np.asarray([0.5], dtype=np.float32),
            "branch_tags": [1],
        },
    ],
)
def test_malformed_server_response_is_rejected(response: dict[str, Any]) -> None:
    endpoint, server = _serve([response])
    client = GraspGenXClient(_config(endpoint))

    with pytest.raises(RuntimeError, match="GraspGen-X"):
        client.generate_grasps(np.zeros((5, 3)), arm="right", frame="camera")
    server.join(timeout=2.0)


@pytest.mark.parametrize(
    "transform",
    [
        [[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    ],
)
def test_gripper_binding_rejects_non_rigid_transforms(
    transform: list[list[float]],
) -> None:
    with pytest.raises(ValidationError):
        _gripper(transform)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extents_open", [0.08, 0.02]),
        ("extents_open", [0.08, 0.0, 0.06]),
        ("extents_mid", [0.04, -0.02, 0.06]),
        ("offset_open", [0.0, float("nan"), 0.05]),
        ("offset_mid", [0.0, True, 0.05]),
        ("gripper_type", 3),
        ("gripper_type", True),
        ("fingertip_depth", 0.0),
        ("fingertip_depth", float("inf")),
    ],
)
def test_sweep_volume_rejects_invalid_geometry(field: str, value: Any) -> None:
    payload = _sweep_volume().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        GraspGenXSweepVolume.model_validate(payload)


def test_gripper_requires_sweep_volume_and_rejects_unknown_geometry_fields() -> None:
    payload = _gripper().model_dump(mode="python")
    payload.pop("sweep_volume")
    with pytest.raises(ValidationError):
        GraspGenXGripper.model_validate(payload)

    sweep_payload = _sweep_volume().model_dump(mode="python")
    sweep_payload["legacy_width"] = 0.08
    with pytest.raises(ValidationError):
        GraspGenXSweepVolume.model_validate(sweep_payload)


def test_input_validation_rejects_invalid_points_and_candidate_limit() -> None:
    client = GraspGenXClient(_config("tcp://127.0.0.1:5556"))

    with pytest.raises(ValueError, match="finite"):
        client.generate_grasps(
            np.asarray([[0.0, np.nan, 0.0]]), arm="right", frame="world"
        )
    with pytest.raises(ValueError, match="numeric"):
        client.generate_grasps(np.ones((2, 3), dtype=bool), arm="right", frame="world")
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        client.generate_grasps(
            np.zeros((2, 3)), arm="right", frame="world", max_candidates=33
        )
    with pytest.raises(TypeError, match="integer or None"):
        client.generate_grasps(
            np.zeros((2, 3)), arm="right", frame="world", max_candidates=True
        )


def test_timeout_sends_once_and_does_not_retry() -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    requests: list[dict[str, Any]] = []

    def receive_without_reply() -> None:
        try:
            requests.append(
                msgpack.unpackb(
                    socket.recv(), raw=False, object_hook=msgpack_numpy.decode
                )
            )
            time.sleep(0.1)
        finally:
            socket.close(linger=0)
            context.term()

    server = threading.Thread(target=receive_without_reply, daemon=True)
    server.start()
    client = GraspGenXClient(_config(f"tcp://127.0.0.1:{port}", timeout_s=0.02))

    with pytest.raises(TimeoutError, match="did not respond"):
        client.generate_grasps(np.zeros((5, 3)), arm="right", frame="camera")
    server.join(timeout=1.0)
    assert len(requests) == 1


def test_load_graspgen_config_supports_yaml_and_json(tmp_path: Path) -> None:
    payload = {
        "endpoint": "tcp://grasp-host:5556",
        "grippers": {
            "left": {
                "gripper_name": "unitree_g1",
                "sweep_volume": _sweep_volume().model_dump(mode="json"),
                "grasp_frame_from_tcp": np.eye(4).tolist(),
            }
        },
    }
    yaml_path = tmp_path / "graspgen.yaml"
    yaml_path.write_text(
        "endpoint: tcp://grasp-host:5556\n"
        "grippers:\n"
        "  left:\n"
        "    gripper_name: unitree_g1\n"
        "    sweep_volume:\n"
        "      extents_open: [0.08, 0.02, 0.06]\n"
        "      offset_open: [0.0, 0.0, 0.05]\n"
        "      extents_mid: [0.04, 0.02, 0.06]\n"
        "      offset_mid: [0.0, 0.0, 0.05]\n"
        "      gripper_type: 0\n"
        "      fingertip_depth: 0.08\n"
        "    grasp_frame_from_tcp:\n"
        "      - [1, 0, 0, 0]\n"
        "      - [0, 1, 0, 0]\n"
        "      - [0, 0, 1, 0]\n"
        "      - [0, 0, 0, 1]\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "graspgen.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_graspgen_config(yaml_path).grippers["left"].gripper_name == "unitree_g1"
    assert load_graspgen_config(json_path).endpoint == "tcp://grasp-host:5556"


def test_config_is_frozen_and_forbids_unknown_fields() -> None:
    config = _config("tcp://127.0.0.1:5556")
    with pytest.raises(ValidationError):
        GraspGenXConfig.model_validate(
            {
                **config.model_dump(),
                "legacy_host": "localhost",
            }
        )
    with pytest.raises(ValidationError):
        config.timeout_s = 2.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_s", float("inf")),
        ("timeout_s", "30"),
        ("num_grasps", True),
        ("num_grasps", "200"),
        ("max_points", False),
        ("max_points", "8192"),
    ],
)
def test_config_rejects_nonfinite_or_coerced_counts(field: str, value: Any) -> None:
    payload = _config("tcp://127.0.0.1:5556").model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        GraspGenXConfig.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [
        [],
        [0.0] * 65,
        [True],
        ["-8"],
        [float("nan")],
        [float("inf")],
        0.0,
    ],
)
def test_config_rejects_invalid_moe_z_offsets(value: Any) -> None:
    payload = _config("tcp://127.0.0.1:5556").model_dump()
    payload["moe_z_offsets_cm"] = value

    with pytest.raises(ValidationError):
        GraspGenXConfig.model_validate(payload)


def test_config_accepts_explicit_refined_moe_z_offset_grid() -> None:
    offsets = tuple(float(value) for value in range(-10, 1))

    config = _config(
        "tcp://127.0.0.1:5556",
        moe_z_offsets_cm=offsets,
    )

    assert config.moe_z_offsets_cm == offsets


def test_config_schema_describes_every_field() -> None:
    config_schema = GraspGenXConfig.model_json_schema()
    gripper_schema = GraspGenXGripper.model_json_schema()
    sweep_schema = GraspGenXSweepVolume.model_json_schema()

    assert all(
        field.get("description") for field in config_schema["properties"].values()
    )
    assert all(
        field.get("description") for field in gripper_schema["properties"].values()
    )
    assert all(
        field.get("description") for field in sweep_schema["properties"].values()
    )
