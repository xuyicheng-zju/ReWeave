# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Public geometry and scripted controls use observed pixels and settled actions."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import (
    SimulatedEnvironmentRPC,
    SimulatedPolicyRPC,
    create_simulated_toolkit,
)
from reweave.benchmarks.robodojo import perception, tools
from reweave.benchmarks.robodojo.access import JOINT_ACTION_FIELDS
from reweave.robots.components.sam3_client import Sam3Result


def _segment_geometry(primitives, text_prompt, **kwargs):
    geometry, _mask = perception.segment_observation(
        primitives.env.last_obs,
        sam3_client=primitives.sam3_client,
        text_prompt=text_prompt,
        **kwargs,
    )
    return geometry


@pytest.mark.parametrize("camera", [name for name, _ in tools.CAMERA_ARTIFACTS])
def test_ring_mask_preserves_centroid_and_supplies_a_foreground_pixel(camera):
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    depth = np.full((3, 3), 2.0)
    depth[1, 1] = 10.0
    observation = {
        "vision": {
            camera: {
                "color": np.zeros((3, 3, 3), dtype=np.uint8),
                "depth": depth,
                "intrinsic_matrix": np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
                "extrinsic_matrix": np.eye(4),
            }
        }
    }
    primitives = SimpleNamespace(
        env=SimpleNamespace(last_obs=observation),
        model_request=lambda kind: nullcontext(),
        sam3_client=SimpleNamespace(
            segment=lambda *args, **kwargs: Sam3Result(
                found=True, mask=mask, score=0.9, box=[0, 0, 3, 3]
            )
        ),
    )

    result = _segment_geometry(primitives, "ring", camera=camera)

    assert result["centroid_rc"] == [1.0, 1.0]
    row, col = result["mask_point_rc"]
    assert type(row) is int and type(col) is int
    assert mask[row, col]
    assert (row - 1) ** 2 + (col - 1) ** 2 == 1
    point = perception.back_project_observation(
        primitives.env.last_obs, row, col, camera=camera
    )
    assert point["depth_m"] == 2.0
    assert point["world_xyz"][2] == -2.0


@pytest.mark.parametrize("mismatch", ["mask", "depth"])
def test_segmentation_rejects_mismatched_camera_pixel_grids(mismatch):
    mask = np.ones((3, 4) if mismatch == "mask" else (3, 3), dtype=bool)
    depth = np.ones((3, 4) if mismatch == "depth" else (3, 3))
    primitives = SimpleNamespace(
        env=SimpleNamespace(
            last_obs={
                "vision": {
                    "cam_head": {"color": np.zeros((3, 3, 3), np.uint8), "depth": depth}
                }
            }
        ),
        model_request=lambda kind: nullcontext(),
        sam3_client=SimpleNamespace(
            segment=lambda *args, **kwargs: Sam3Result(found=True, mask=mask)
        ),
    )

    with pytest.raises(ValueError, match="shape|grid"):
        _segment_geometry(primitives, "object")


@pytest.mark.parametrize("depth_name", ["distance_to_image_plane", "depth"])
@pytest.mark.parametrize("channel_axis", [False, True])
def test_back_project_uses_the_usd_pixel_axes_and_camera_to_world_transform(
    depth_name, channel_axis
):
    depth = np.full((3, 5), 2.0)
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = {
        depth_name: depth[..., None] if channel_axis else depth,
        "color": np.zeros((3, 5, 3), np.uint8),
        "intrinsic_matrix": np.array([[2, 0, 2], [0, 4, 1], [0, 0, 1]]),
        "extrinsic_matrix": transform,
    }
    primitives = SimpleNamespace(
        env=SimpleNamespace(last_obs={"vision": {"cam_head": camera}})
    )

    result = perception.back_project_observation(primitives.env.last_obs, row=0, col=4)

    assert result["pixel"] == [0, 4]
    assert result["depth_m"] == 2.0
    assert result["world_xyz"] == [9.5, 22.0, 28.0]


def test_rgb_only_segmentation_does_not_invent_depth_coordinates():
    mask = np.ones((3, 3), bool)
    primitives = SimpleNamespace(
        env=SimpleNamespace(
            last_obs={"vision": {"cam_head": {"color": np.zeros((3, 3, 3), np.uint8)}}}
        ),
        model_request=lambda kind: nullcontext(),
        sam3_client=SimpleNamespace(
            segment=lambda *args, **kwargs: Sam3Result(found=True, mask=mask)
        ),
    )

    result = _segment_geometry(primitives, "object")
    assert result["found"] is True and result["mask_point_rc"] == [1, 1]
    assert (
        "depth"
        in perception.back_project_observation(primitives.env.last_obs, row=1, col=1)[
            "error"
        ]
    )


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("gripper,expected", [(1, 0.0), (-1, 1.0)])
def test_set_gripper_holds_both_joint_targets_without_ee_ik(
    tmp_path, arm, gripper, expected
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    for index, current in enumerate(("left", "right")):
        rpc.observation["state"][f"{current}_arm_joint_state"] = np.arange(6) + index
        rpc.observation["state"][f"{current}_ee_joint_state"] = np.array(
            [0.3 + index / 2]
        )
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    actions = []

    def call(method, args=(), **kwargs):
        if method == "env.plan_position_path":
            pytest.fail("changing a gripper must not invoke IK")
        if method == "env.step":
            actions.append(args[0])
            assert set(args[0]) == {name for name, _ in JOINT_ACTION_FIELDS}
        return original(method, args=args, **kwargs)

    rpc.call = call
    try:
        result = toolkit.execute_tool("set_gripper", {"arm": arm, "gripper": gripper})
        assert not result.result.get("error"), result.result
        assert rpc.actions == len(actions) == 1
        assert toolkit.control_step_budget()["used"] == 1
        for current in ("left", "right"):
            np.testing.assert_array_equal(
                actions[0][f"{current}_arm_joint_state"],
                rpc.observation["state"][f"{current}_arm_joint_state"],
            )
            assert actions[0][f"{current}_ee_joint_state"] == (
                [expected]
                if current == arm
                else rpc.observation["state"][f"{current}_ee_joint_state"].tolist()
            )
        controls = [
            row
            for row in toolkit._execution_trace.events
            if row["event_type"] == "control_step"
        ]
        assert len(controls) == 1 and controls[0]["forward_ref"] is None
    finally:
        toolkit.close()


@pytest.mark.parametrize("gripper", [-1, 0, 1])
def test_move_to_at_current_xyz_still_executes_an_explicit_gripper_command(
    tmp_path, gripper
):
    rpc = SimulatedEnvironmentRPC(limit=1, terminal_after=99)
    rpc.observation["state"]["right_ee_joint_state"] = np.array(
        [1.0 if gripper > 0 else 0.0]
    )
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    actions = []

    def call(method, args=(), **kwargs):
        if method == "env.plan_position_path":
            pytest.fail("an already reached position does not require IK")
        if method == "env.step":
            actions.append(args[0])
        return original(method, args=args, **kwargs)

    rpc.call = call
    try:
        result = toolkit.execute_tool(
            "move_to",
            {
                "arm": "right",
                "xyz": [0.0, 0.0, 0.3],
                "gripper": gripper,
                "max_steps": 1,
            },
        )
        assert not result.result.get("error"), result.result
        assert result.result["capability_result"]["reached"] is True
        expected_steps = int(gripper != 0)
        assert rpc.actions == len(actions) == expected_steps
        assert result.result["capability_result"]["steps_used"] == expected_steps
        if gripper:
            assert actions[0]["right_ee_joint_state"] == [0.0 if gripper > 0 else 1.0]
        assert toolkit.control_step_budget()["used"] == expected_steps
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "cancel_during", ["move-observation", "move-ik", "gripper-observation"]
)
def test_cancellation_during_observation_or_ik_prevents_the_next_physical_submission(
    tmp_path, monkeypatch, cancel_during
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    observations = 0
    current_obs = tools._current_obs

    def observe(primitives):
        nonlocal observations
        value = current_obs(primitives)
        observations += 1
        if (cancel_during, observations) in {
            ("move-observation", 2),
            ("gripper-observation", 1),
        }:
            toolkit._active_operation.cancel_event.set()
        return value

    def call(method, *args, **kwargs):
        value = original(method, *args, **kwargs)
        if cancel_during == "move-ik" and method == "env.plan_position_path":
            toolkit._active_operation.cancel_event.set()
        return value

    monkeypatch.setattr(tools, "_current_obs", observe)
    rpc.call = call
    try:
        result = (
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
            if cancel_during == "gripper-observation"
            else toolkit.execute_tool(
                "move_to", {"arm": "right", "xyz": [0.1, 0.2, 0.3], "max_steps": 1}
            )
        )
        assert result.result["capability_result"]["code"] == "tool_cancelled"
        assert rpc.actions == toolkit.control_step_budget()["used"] == 0
        assert not any(
            row["event_type"] == "control_step"
            for row in toolkit._execution_trace.events
        )
    finally:
        toolkit.close()


def test_cancellation_during_policy_forward_discards_the_unexecuted_prediction(
    tmp_path,
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    policy = SimulatedPolicyRPC()
    toolkit, _, _ = create_simulated_toolkit(
        tmp_path / "episode", rpc=rpc, policy=policy
    )
    original = policy.call

    def call(*args, **kwargs):
        value = original(*args, **kwargs)
        toolkit._active_operation.cancel_event.set()
        return value

    policy.call = call
    try:
        result = toolkit.execute_tool("pi0_pick", {"max_chunks": 2})
        assert result.result["capability_result"]["code"] == "tool_cancelled"
        assert policy.calls == 1 and rpc.actions == 0
        events = toolkit._execution_trace.events
        assert not any(row["event_type"] == "control_step" for row in events)
        assert (
            sum(
                row["discarded_action_steps"]
                for row in events
                if row["event_type"] == "action_suffix_discarded"
            )
            == 4
        )
    finally:
        toolkit.close()
