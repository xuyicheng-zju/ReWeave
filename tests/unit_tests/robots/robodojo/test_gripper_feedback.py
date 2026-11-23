# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Measured gripper feedback is separate from native control/VLA command targets."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import (
    SimulatedEnvironmentRPC,
    SimulatedPolicyRPC,
    create_simulated_toolkit,
    simulated_observation,
)
from reweave.benchmarks.robodojo.access import (
    MEASURED_GRIPPER_KEYS,
    graph_features,
    public_observation,
    robot_state_features,
)
from reweave.benchmarks.robodojo.agent_support import (
    _obs_dict,
    _synchronize_tiled_camera_apertures,
)
from reweave.benchmarks.robodojo.prompt_bundle import system_prompt
from reweave.benchmarks.robodojo.tools import TOOLS_SPEC
from reweave.robots.components.pi05_vla_client import Pi05VLAClient
from tests.unit_tests.robots.robodojo import test_runtime_limits

agent_environment = test_runtime_limits.agent_environment


def test_tiled_camera_producer_synchronizes_square_pixel_aperture() -> None:
    vertical_aperture = 14.266

    def set_horizontal_aperture(value, maintain_square_pixels):
        nonlocal vertical_aperture
        assert value == 22.212 and maintain_square_pixels is True
        vertical_aperture = 22.212 * 480 / 640

    camera = SimpleNamespace(
        get_resolution=lambda: (640, 480),
        get_horizontal_aperture=lambda: 22.212,
        get_vertical_aperture=lambda: vertical_aperture,
        set_horizontal_aperture=set_horizontal_aperture,
    )

    def intrinsics(*_):
        fx = 640 * 10.0 / 22.212
        fy = 480 * 10.0 / vertical_aperture
        return np.asarray([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0, 0, 1]])

    env = SimpleNamespace(
        camera_manager=SimpleNamespace(
            cameras=[[camera]],
            camera_names=[["cam_head"]],
            get_camera_intrinsics=intrinsics,
        )
    )

    receipt = _synchronize_tiled_camera_apertures(env)

    assert vertical_aperture == pytest.approx(16.659)
    assert receipt == [
        {
            "env_index": 0,
            "camera": "cam_head",
            "resolution": [640, 480],
            "horizontal_aperture": 22.212,
            "previous_vertical_aperture": 14.266,
            "vertical_aperture": pytest.approx(16.659),
            "previous_intrinsic_matrix": [
                [pytest.approx(288.1325400684315), 0.0, 320.0],
                [0.0, pytest.approx(336.4643207626525), 240.0],
                [0.0, 0.0, 1.0],
            ],
            "intrinsic_matrix": [
                [pytest.approx(288.1325400684315), 0.0, 320.0],
                [0.0, pytest.approx(288.1325400684315), 240.0],
                [0.0, 0.0, 1.0],
            ],
            "source": "Replicator tiled render product square-pixel contract",
        }
    ]


def native_environment(*, sign=1, opening=0.6):
    raw = simulated_observation()
    raw["state"]["left_ee_joint_state"] = np.array([0.2])
    raw["state"]["right_ee_joint_state"] = np.array([0.8])
    raw["action"] = {"left_ee_joint_state": [0.2], "right_ee_joint_state": [0.8]}
    targets = [
        SimpleNamespace(
            type="target",
            ee_type="gripper",
            arm_name=f"{arm}_arm",
            gripper_scale=[-0.1, 0.4],
            gripper_move={"sign": sign},
        )
        for arm in ("left", "right")
    ]
    reads = []
    joint = -0.1 + opening * 0.5 if sign == 1 else 0.4 - opening * 0.5

    def actual(robot, *, env_idx_list):
        reads.append((robot, env_idx_list))
        assert any(robot is target for target in targets)
        return {0: np.array([joint, joint * -1])}

    manager = SimpleNamespace(
        robot_list=[SimpleNamespace(type="support"), *targets],
        get_end_effector_real_val=actual,
        control_manager=SimpleNamespace(prev_control=[deepcopy(raw["action"])]),
    )
    env = SimpleNamespace(
        task_name="put_bottles_into_dustbin",
        get_obs=lambda *, env_idx: raw,
        robot_manager=manager,
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(
                get_one_description=lambda: [raw["instruction"]]
            )
        ),
    )
    frames = []
    return env, raw, targets, reads, SimpleNamespace(record=frames.append), frames


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("opening", [-0.05, 0.0, 0.6, 1.0, 1.05])
def test_canonical_observation_adds_actual_opening_without_mutating_native_targets(
    sign, opening
):
    env, raw, targets, reads, recorder, frames = native_environment(
        sign=sign, opening=opening
    )
    before_targets = deepcopy(env.robot_manager.control_manager.prev_control)
    observation = _obs_dict(env, recorder)
    assert observation is not raw and observation["state"] is not raw["state"]
    assert frames == [observation]
    assert len(reads) == 2
    for target, read in zip(targets, reads):
        assert read[0] is target and read[1] == [0]
    for key in MEASURED_GRIPPER_KEYS:
        assert observation["state"][key] == pytest.approx([opening])
        assert key not in raw["state"]
    assert observation["state"]["left_ee_joint_state"] == [0.2]
    assert observation["state"]["right_ee_joint_state"] == [0.8]
    assert observation["action"] is raw["action"]
    assert raw["action"] == {
        "left_ee_joint_state": [0.2],
        "right_ee_joint_state": [0.8],
    }
    assert env.robot_manager.control_manager.prev_control == before_targets
    public = public_observation(observation)
    assert set(MEASURED_GRIPPER_KEYS) <= set(public["state"])
    assert "object_pose" not in public["state"] and "action" not in public
    vectors = robot_state_features(public)
    features = graph_features(vectors)
    assert len(vectors) == 8 and len(features) == 30
    for key in MEASURED_GRIPPER_KEYS:
        coordinate = features[f"robot.{key}.0"]
        assert coordinate["value"] == pytest.approx(opening)
        assert coordinate["unit"] == "dimensionless"
        assert "actual gripper joint" in coordinate["derivation"]
    assert "command-derived" in features["robot.left_ee_joint_state.0"]["derivation"]


@pytest.mark.parametrize(
    "defect",
    ["reader", "missing_arm", "bad_scale", "bad_sign", "nan_joint", "empty_joint"],
)
def test_new_native_observation_never_substitutes_command_for_missing_measurement(
    defect,
):
    env, raw, targets, _, recorder, frames = native_environment()
    if defect == "reader":

        def fail(*args, **kwargs):
            raise RuntimeError("measured joint read failed")

        env.robot_manager.get_end_effector_real_val = fail
    elif defect == "missing_arm":
        env.robot_manager.robot_list.remove(targets[1])
    elif defect == "bad_scale":
        targets[0].gripper_scale = [0.4, 0.4]
    elif defect == "bad_sign":
        targets[0].gripper_move["sign"] = 0
    elif defect == "nan_joint":
        env.robot_manager.get_end_effector_real_val = lambda *a, **k: {0: [np.nan]}
    else:
        env.robot_manager.get_end_effector_real_val = lambda *a, **k: {0: []}
    with pytest.raises(
        (ValueError, RuntimeError), match="RoboDojo|measured joint read failed"
    ):
        _obs_dict(env, recorder)
    assert not frames
    assert not set(MEASURED_GRIPPER_KEYS) & set(raw["state"])
    assert raw["state"]["left_ee_joint_state"] == [0.2]


def test_historical_receipt_stays_six_vectors_without_invented_measurements():
    values = robot_state_features(simulated_observation())
    original = deepcopy(values)
    assert len(graph_features(values)) == 28
    assert values == original and not set(MEASURED_GRIPPER_KEYS) & set(values)
    values[MEASURED_GRIPPER_KEYS[0]] = [0.6]
    with pytest.raises(ValueError, match="RoboDojo"):
        graph_features(values)
    with pytest.raises(ValueError, match="both measured"):
        robot_state_features({"state": values})


def test_measured_channels_do_not_change_native_fourteen_value_vla_input():
    env, _, _, _, recorder, _ = native_environment()
    public = public_observation(_obs_dict(env, recorder))
    baseline = dict(
        public,
        state={
            key: value
            for key, value in public["state"].items()
            if key not in MEASURED_GRIPPER_KEYS
        },
    )
    client = Pi05VLAClient(SimulatedPolicyRPC(), embodiment="robodojo")
    before, after = client.encode_obs(baseline), client.encode_obs(public)
    for key in before:
        np.testing.assert_equal(before[key], after[key])
    assert after["states"].shape == (1, 14)
    np.testing.assert_allclose(after["states"][0, -2:], [0.2, 0.8])
    assert not np.allclose(after["states"][0, -2:], [0.6, 0.6])


@pytest.mark.parametrize(
    "tool,arguments,selected_gripper",
    [
        ("set_gripper", {"arm": "right", "gripper": 1}, 0.0),
        (
            "move_joints",
            {
                "arm": "right",
                "joint_positions": [0.2] * 6,
                "tol_rad": 0.01,
                "max_steps": 1,
            },
            0.8,
        ),
        (
            "move_pose",
            {
                "arm": "right",
                "xyz": [0.1, 0, 0.3],
                "quaternion_wxyz": [1, 0, 0, 0],
                "tol_m": 0.005,
                "tol_rad": 0.01,
                "max_steps": 1,
            },
            0.8,
        ),
    ],
)
def test_public_measurement_survives_adapter_client_trace_and_planner_state_while_hold_keeps_target(
    tmp_path, agent_environment, tool, arguments, selected_gripper
):
    native, _, _, reads, recorder, _ = native_environment()
    server = agent_environment.__new__(agent_environment)
    server.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)])
    server.recorder = recorder
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    server.observations = rpc.observations
    server.get_status = lambda: {
        "step": rpc.actions,
        "step_limit": rpc.meta["max_episode_steps"],
    }
    rpc.observation = server.get_native_obs()
    original = rpc.call
    commands = []
    plans = []
    trajectory_packets = []

    def call(method, args=(), **kwargs):
        if method in {"env.get_obs", "env.reset"}:
            return server.get_obs()
        if method == "env.solve_ik_pose":
            pytest.fail("A native trajectory must not fall back to bare pose IK")
        if method in {"env.plan_pose_path", "env.plan_joint_path"}:
            assert kwargs["kwargs"]["arm"] == "right"
            if method == "env.plan_pose_path":
                poses = np.asarray(kwargs["kwargs"]["poses"])
                assert poses.ndim == 2 and poses.shape[1] == 7
                assert np.isfinite(poses).all()
            else:
                assert kwargs["kwargs"]["joints"] == arguments["joint_positions"]
            plan = {
                "status": "Success",
                "joint_positions": [
                    np.asarray(
                        rpc.observation["state"]["right_arm_joint_state"]
                    ).tolist(),
                    [0.2] * 6,
                ],
                "joint_velocities": [[0.1] * 6, [0.0] * 6],
                # The two original samples belong to one acknowledged control
                # window, matching this fixture's max_steps=1 contract.
                "planner_dt_s": 0.04,
                "control_period_s": 0.08,
            }
            plans.append(deepcopy(plan))
            return plan
        if method == "env.step_trajectory":
            assert not args
            trajectory_packets.append(deepcopy(kwargs["kwargs"]))
        if method == "env.step":
            commands.append(args[0])
            rpc.observation = server.get_native_obs()
        return original(method, args=args, **kwargs)

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool(tool, arguments).result
        assert not result.get("error")
        assert (
            rpc.actions == toolkit.control_step_budget()["used"] == len(commands) == 1
        )
        assert commands[0]["left_ee_joint_state"] == [0.2]
        assert commands[0]["right_ee_joint_state"] == [selected_gripper]
        if tool == "set_gripper":
            assert not plans and not trajectory_packets
        else:
            assert len(plans) == len(trajectory_packets) == 1
            packet = trajectory_packets[0]
            assert packet["arm"] == "right" and packet["gripper"] == 0
            assert packet["positions"] == plans[0]["joint_positions"]
            assert packet["velocities"] == plans[0]["joint_velocities"]
            assert result["capability_result"]["planned_actions"] == 1
            assert result["capability_result"]["steps_used"] == 1
        assert reads and all(
            0.6 != value for value in commands[0]["left_ee_joint_state"]
        )
        state = toolkit.execute_tool("view_env_state", {}).result["state"]["state"]
        assert state["left_gripper_measured_opening"] == pytest.approx([0.6])
        assert state["left_ee_joint_state"] == [0.2]
        telemetry = [
            row
            for row in toolkit._execution_trace.events
            if row["event_type"] in {"initial_state", "control_telemetry"}
        ]
        assert len(telemetry) == 2
        assert all(
            row["physical_features"]["left_gripper_measured_opening"]
            == pytest.approx([0.6])
            for row in telemetry
        )
        assert all(
            len(graph_features(row["physical_features"])) == 30 for row in telemetry
        )
    finally:
        toolkit.close()


def test_public_contract_distinguishes_command_and_measured_gripper_roles():
    prompt = system_prompt()
    description = next(
        spec["description"] for spec in TOOLS_SPEC if spec["name"] == "view_env_state"
    )
    assert "command-derived" in prompt and "*_gripper_measured_opening" in prompt
    assert "command target" in prompt and "contact-blocked" in prompt
    assert (
        "command-derived" in description and "*_gripper_measured_opening" in description
    )
