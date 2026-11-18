# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Physical units, terminal conditions and observation-coordinate regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reweave.benchmarks.libero.cartesian_control import CartesianControl
from reweave.benchmarks.libero.tools import TOOLS_SPEC, LiberoPrimitives, _mask_to_world
from reweave.tools.toolkit import ToolResult


def control_spec():
    return {
        "input_min": [-1] * 6,
        "input_max": [1] * 6,
        "output_min": [-0.05] * 3 + [-0.5] * 3,
        "output_max": [0.05] * 3 + [0.5] * 3,
        "world_from_control": np.eye(3).tolist(),
    }


def test_world_deltas_invert_actual_controller_scale_and_frame():
    spec = control_spec()
    spec["world_from_control"] = (
        Rotation.from_euler("z", 90, degrees=True).as_matrix().tolist()
    )
    control = CartesianControl.from_spec(spec)
    action = control.action([0.01, 0, 0], [0, 0, 0.08], 1)
    physical = action[:6] * np.array([0.05] * 3 + [0.5] * 3)
    assert control.world_from_control @ physical[:3] == pytest.approx([0.01, 0, 0])
    assert control.world_from_control @ physical[3:] == pytest.approx([0, 0, 0.08])
    assert action[5] == pytest.approx(0.16)


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_max", [0] * 6),
        ("input_min", [float("nan")] * 6),
        ("world_from_control", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ],
)
def test_invalid_controller_contract_fails_loudly(field, value):
    spec = control_spec()
    spec[field] = value
    with pytest.raises(ValueError):
        CartesianControl.from_spec(spec)


def primitives(*, stalled=False):
    p = object.__new__(LiberoPrimitives)
    p._cartesian_control = None
    p._last_obs_eef_pos = np.array([0.0, 0.0, 1.0])
    p._episode_monitors = SimpleNamespace(reset_tool_epoch=lambda: None)
    p._scripted_interruption_feedback = lambda proposal: {}
    p.rotation = Rotation.from_quat([1, 0, 0, 0])
    p.actions = []
    p.env = SimpleNamespace(
        terminated=False,
        truncated=False,
        get_cartesian_control_spec=control_spec,
        raw_obs=lambda: {"robot0_eef_quat": p.rotation.as_quat()},
    )

    def step(action, **kwargs):
        p.actions.append(action.copy())
        if not stalled:
            p._last_obs_eef_pos += action[:3] * 0.05
            p.rotation = Rotation.from_rotvec(action[3:6] * 0.5) * p.rotation

    p._scripted_control_step = step
    return p


def test_position_reached_does_not_skip_requested_yaw():
    p = primitives()
    result = p.move_to([0, 0, 1], target_yaw=0.4, gripper=1)
    assert result["target_reached"]
    assert result["steps_used"] == len(p.actions) > 0
    assert result["final_yaw"] == pytest.approx(0.4, abs=0.02)
    assert max(abs(a[5] * 0.5) for a in p.actions) <= 0.100001
    assert all(a[6] == 1 for a in p.actions)


def test_full_quaternion_servo_tracks_shortest_world_so3_residual():
    p = primitives()
    target = Rotation.from_euler("xyz", [35, -25, 70], degrees=True)
    result = p.move_pose(
        [0, 0, 1],
        target_quaternion_xyzw=target.as_quat().tolist(),
        quaternion_step=0.06,
        ori_tol=0.015,
        max_steps=120,
    )
    assert result["target_reached"]
    assert result["final_orientation_error_rad"] < 0.015
    assert len(p.actions) > 0
    assert max(np.linalg.norm(action[3:6] * 0.5) for action in p.actions) <= 0.06001
    residual = (target * p.rotation.inv()).magnitude()
    assert residual < 0.015


@pytest.mark.parametrize(
    "quaternion",
    ([0, 0, 0], [0, 0, 0, 2], [0, 0, float("nan"), 1]),
)
def test_full_quaternion_servo_rejects_invalid_targets_before_action(quaternion):
    p = primitives()
    with pytest.raises(ValueError, match="target_quaternion_xyzw"):
        p.move_pose([0, 0, 1], target_quaternion_xyzw=quaternion)
    assert not p.actions


def test_full_quaternion_servo_rejects_mixed_orientation_modes():
    p = primitives()
    with pytest.raises(ValueError, match="mutually exclusive"):
        p.move_pose(
            [0, 0, 1],
            target_yaw=0.1,
            target_quaternion_xyzw=[0, 0, 0, 1],
        )
    assert not p.actions


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("move_to", {"xyz": [0, 0, 1]}),
        ("move_pose", {"xyz": [0, 0, 1]}),
        ("rotate_wrist", {"target_yaw": 0}),
        ("rotate_pitch", {"target_pitch": 0}),
    ],
)
def test_already_reached_target_reports_zero_actual_steps(method, kwargs):
    p = primitives()
    r = getattr(p, method)(**kwargs)
    assert r["steps_used"] == 0 and not p.actions
    assert r["target_reached"] and r["stop_reason"] == "target_reached"


def test_budget_exhaustion_is_distinct_from_convergence():
    p = primitives(stalled=True)
    r = p.move_pose([0.1, 0, 1], max_steps=3)
    assert r["steps_used"] == len(p.actions) == 3
    assert not r["target_reached"] and r["stop_reason"] == "step_budget_exhausted"
    r = p.move_pose([0.1, 0, 1], max_steps=0)
    assert r["steps_used"] == 0 and not r["target_reached"]


def test_settled_motion_response_exposes_stall_without_changing_action_budget():
    stalled = primitives(stalled=True)
    result = stalled.move_to([0.1, 0, 1], max_steps=25)
    response = result["motion_response"]
    assert len(stalled.actions) == 25
    assert response["steps"] == 20
    assert response["path_length_m"] == response["target_distance_reduction_m"] == 0
    moving = primitives()
    result = moving.move_to([0.1, 0, 1], max_steps=25)
    assert result["target_reached"]
    assert result["motion_response"]["target_distance_reduction_m"] > 0.08
    assert result["motion_response"]["path_length_m"] > 0.08


def test_terminal_environment_receives_no_more_actions():
    p = primitives()
    p.env.truncated = True
    r = p.move_to([0.1, 0, 1])
    assert r["steps_used"] == 0 and not p.actions and r["stop_reason"] == "truncated"


def test_servo_rejects_removed_calibration_knob_and_ambiguous_rotation():
    p = primitives()
    with pytest.raises(TypeError):
        p.move_to([0, 0, 1], action_scale=0.1)
    with pytest.raises(ValueError):
        p.rotate_wrist(target_yaw=1, delta_yaw=1)
    with pytest.raises(ValueError):
        p.move_to([float("nan"), 0, 1])
    for name in ["move_to", "move_pose"]:
        assert (
            "action_scale"
            not in next(x for x in TOOLS_SPEC if x["name"] == name)["input_schema"][
                "properties"
            ]
        )


def test_centroid_uses_same_row_col_order_as_point_input():
    mask = np.zeros((4, 7), dtype=bool)
    mask[1, 5] = True
    r = _mask_to_world(mask, np.ones((4, 7, 3)), min_valid=1)
    assert r["centroid_pixel"] == [1, 5] and r["centroid_pixel_order"] == "row_col"


def test_image_labels_preserve_order_and_original_payloads():
    r = {
        "_image_bytes": b"policy",
        "_image_cam_bytes": b"calibration",
        "_image_labels": {
            "_image_bytes": "policy; not for projection",
            "_image_cam_bytes": "agentview_high.png; calibration",
        },
    }
    tr = ToolResult("observe", r)
    text = json.loads(tr.content_blocks[0]["text"])
    assert [x["image_number"] for x in text["image_inputs"]] == [1, 2]
    assert text["image_inputs"][1]["description"] == "agentview_high.png; calibration"
    assert len([x for x in tr.content_blocks if x["type"] == "image"]) == 2
    assert r["_image_bytes"] == b"policy"


def test_gripper_budget_is_counted_from_executed_actions():
    p = primitives()
    p.env.truncated = True
    result = p.set_gripper(gripper=1, steps=5)
    assert result["steps_used"] == 0 and not p.actions


def test_libero_author_sources_include_controller_protocol(tmp_path):
    from pathlib import Path
    from unittest.mock import patch

    from reweave.benchmarks.libero.toolkit import LiberoToolkit
    from reweave.tools.toolkit import Toolkit

    toolkit = object.__new__(LiberoToolkit)
    with patch.object(Toolkit, "harness_source_files", return_value={}):
        sources = toolkit.harness_source_files()
    for name in ("cartesian_control.py", "env_client.py", "env_server.py"):
        path = Path("reweave/benchmarks/libero") / name
        assert sources[path.as_posix()] == path.read_text()


def vla_primitives():
    from reweave.robots.vla_execution import VLAExecution

    p = object.__new__(LiberoPrimitives)
    p.env = SimpleNamespace(terminated=False, truncated=False)
    p._last_obs_eef_z = 1.0
    p._last_obs_gripper = 0.03
    p._begin_vla_tool = lambda: None
    p.chunk_calls = 0

    def chunk(*args, **kwargs):
        p.chunk_calls += 1
        p._last_obs_eef_z = 0.8 if p.chunk_calls == 1 else 0.9
        return VLAExecution(executed_action_steps=1)

    p._vlm_chunk = chunk
    return p


@pytest.mark.parametrize("tool", ["pi0_pick", "pi0_doubled"])
@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_vla_after_episode_end_does_not_report_phantom_chunks(tool, terminal):
    p = vla_primitives()
    setattr(p.env, terminal, True)
    result = getattr(p, tool)("target")
    assert result["chunks_used"] == p.chunk_calls == 0
    assert result["success"] is (terminal == "terminated")
    if tool == "pi0_doubled":
        assert result["contact_skill_executed"] is False


@pytest.mark.parametrize("budget", [0, -1, True, 1.2])
@pytest.mark.parametrize("tool", ["pi0_pick", "pi0_doubled"])
def test_vla_rejects_invalid_budget_before_inference(tool, budget):
    p = vla_primitives()
    with pytest.raises(ValueError, match="max_chunks"):
        getattr(p, tool)("target", max_chunks=budget)
    assert p.chunk_calls == 0


@pytest.mark.parametrize("descent_thresh, expected", [(0.10, False), (0.03, True)])
def test_pick_can_adjust_descent_when_starting_near_object(descent_thresh, expected):
    p = vla_primitives()
    samples = iter([0.96, 1.02])

    def chunk(*args, **kwargs):
        p.chunk_calls += 1
        p._last_obs_eef_z = next(samples)

    p._vlm_chunk = chunk
    result = p.pi0_pick("target", max_chunks=2, descent_thresh=descent_thresh)
    assert result["success"] is expected
    assert result["diagnostics"]["descent_thresh"] == descent_thresh


@pytest.mark.parametrize(
    "opening, expected", [(0.0, False), (0.01, True), (0.06, False)]
)
def test_pick_opening_bounds_reject_empty_or_open_gripper(opening, expected):
    p = vla_primitives()
    p._last_obs_gripper = opening
    result = p.pi0_pick("target", max_chunks=2, gripper_open_thresh=0.01)
    assert result["success"] is expected
    assert result["diagnostics"]["gripper_open_thresh"] == 0.01


def test_pick_defaults_match_explicit_upstream_defaults():
    implicit = vla_primitives().pi0_pick("target", max_chunks=2)
    explicit = vla_primitives().pi0_pick(
        "target", max_chunks=2, gripper_open_thresh=0.0, descent_thresh=0.10
    )
    assert implicit == explicit
    assert implicit["success"] is True


@pytest.mark.parametrize(
    "thresholds",
    [
        {"descent_thresh": -0.1},
        {"descent_thresh": float("nan")},
        {"gripper_open_thresh": float("inf")},
        {"gripper_open_thresh": -0.01},
        {"gripper_open_thresh": 0.06},
        {"gripper_open_thresh": 0.07},
        {"lift_thresh": 0.0},
    ],
)
def test_pick_invalid_thresholds_fail_before_inference(thresholds):
    p = vla_primitives()
    with pytest.raises(ValueError, match="pick thresholds"):
        p.pi0_pick("target", **thresholds)
    assert p.chunk_calls == 0


def test_pick_truncation_overrides_local_grasp_heuristic():
    p = vla_primitives()
    chunk = p._vlm_chunk

    def truncate(*args, **kwargs):
        result = chunk(*args, **kwargs)
        p.env.truncated = p.chunk_calls == 2
        return result

    p._vlm_chunk = truncate
    result = p.pi0_pick("target", max_chunks=2, descent_thresh=0.03)
    assert result["truncated"] is True
    assert result["success"] is False


@pytest.mark.parametrize("rows", [[12, 15], [-5, -1], [8, 2]])
def test_back_project_region_cannot_wrap_or_reverse_into_another_surface(rows):
    from reweave.benchmarks.libero.tools import back_project

    state = SimpleNamespace(
        get=lambda step: SimpleNamespace(
            step_idx=0, artifacts={"agentview_world_high.npz"}
        ),
        load=lambda name, step: np.ones((10, 10, 3)),
    )
    result = back_project(row_range=rows, col_range=[0, 5], state=state)
    assert "error" in result


def test_partial_observation_reports_capture_error_and_keeps_available_images(tmp_path):
    from reweave.benchmarks.libero.tools import dump_state, view_env_state
    from reweave.session import EnvState

    def render(**kwargs):
        raise RuntimeError("renderer unavailable")

    p = SimpleNamespace(
        env=SimpleNamespace(
            terminated=False,
            truncated=False,
            raw_obs=lambda: {
                "robot0_eef_pos": [0, 0, 1],
                "robot0_eef_quat": [1, 0, 0, 0],
                "robot0_gripper_qpos": [0.04, -0.04],
            },
            get_task_language=lambda: "open drawer",
            get_camera_meta=lambda *args, **kwargs: {},
            render_camera=render,
        ),
        _last_obs={"main_images": np.zeros((4, 4, 3), dtype=np.uint8)},
    )
    state = EnvState(tmp_path)
    record = dump_state(p, state)
    result = view_env_state(record.step_idx, state=state)
    assert result["_image_bytes"]
    assert len(result["observation_errors"]) >= 6
    assert any(
        "renderer unavailable" in error["error"]
        for error in result["observation_errors"]
    )
    assert state.load("observation_errors.json", step=0) == result["observation_errors"]
