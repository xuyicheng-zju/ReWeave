"""Official motion planner routing for position and joint atomic actions."""

from types import SimpleNamespace

import numpy as np
import pytest

from reweave.benchmarks.robodojo.agent_support import (
    _plan_joint_path,
    _plan_position_path,
)


class NativePlanner:
    dt = 0.004

    def __init__(self):
        self.calls = []
        self.criteria = None
        self.fail = False
        self.status = None
        self.motion_planner = SimpleNamespace(plan_cspace=self.raw_cspace)
        self.ik_solver = SimpleNamespace()
        self.scene_model = {"collision": "native"}
        self.robot_cfg = {"limits": "native"}

    def raw_cspace(self, *, current_state, goal_state):
        self.calls.append(("raw_cspace", current_state.copy(), goal_state.copy()))
        return SimpleNamespace(
            status=self.status, success=np.array([not self.fail]), planning_time=0.05
        )

    def _run_with_constraint(self, planner, axes, callback):
        assert planner is self.motion_planner
        previous = self.criteria
        self.criteria = axes
        try:
            return callback()
        finally:
            self.criteria = previous

    def plan_path(self, current, pose, *, real_robot_pose, constraint_pose=None):
        self.calls.append(
            ("plan_path", current.copy(), pose, real_robot_pose, constraint_pose)
        )

        def generate():
            assert self.criteria == [1, 1, 1, 0, 0, 0]
            return SimpleNamespace(
                status=self.status,
                success=np.array([not self.fail]),
                planning_time=0.05,
            )

        self._run_with_constraint(self.motion_planner, constraint_pose, generate)
        if self.fail:
            return {"status": "Fail"}
        return {
            "status": "Success",
            "position": np.linspace(current, current + 0.2, 11),
            "velocity": np.zeros((11, 6)),
        }

    def plan_joint(self, current, target):
        self.calls.append(("plan_joint", current.copy(), list(target)))
        self.motion_planner.plan_cspace(
            current_state=current, goal_state=np.array(target)
        )
        if self.fail:
            return {"status": "Fail"}
        return {
            "status": "Success",
            "position": np.linspace(current, target, 11),
            "velocity": np.zeros((11, 6)),
        }


def native_scene(planner):
    robot = SimpleNamespace(
        type="target",
        arm_name="right_arm",
        robot_name="x5",
        entity_origin_pose=[-0.3, -0.45, 0.765, 1, 0, 0, 0],
        arm_joints_name=[f"joint{i}" for i in range(1, 7)],
        arm_joint_indices=list(range(6)),
    )
    articulation = SimpleNamespace(
        data=SimpleNamespace(soft_joint_pos_limits=np.array([[[-3.2, 3.2]] * 6]))
    )
    measured = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    manager = SimpleNamespace(
        dt=0.004,
        robot_list=[robot],
        robot_key=[articulation],
        planner={"x5": planner},
        get_joint=lambda *args, **kwargs: {0: measured.copy()},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("single-shot IK must not be used for a motion trajectory")

    manager.solve_ik = forbidden
    env = SimpleNamespace(
        robot_manager=manager,
        obs_manager=SimpleNamespace(collect_interval=10),
        get_obs=lambda **kwargs: {
            "state": {"right_ee_pose": [0, 0, 0.2, 0.5, 0.5, 0.5, 0.5]}
        },
    )
    return env, measured


def test_position_plan_calls_official_path_once_with_xyz_only_criteria():
    planner = NativePlanner()
    env, measured = native_scene(planner)
    original = planner._run_with_constraint
    result = _plan_position_path(env, "right", [0.2, 0.1, 0.3])
    assert result["status"] == "Success"
    assert len(planner.calls) == 1
    call = planner.calls[0]
    assert call[0] == "plan_path"
    np.testing.assert_array_equal(call[1], measured)
    assert call[2] == [0.2, 0.1, 0.3, 0.5, 0.5, 0.5, 0.5]
    assert call[4] == [1, 1, 1, 0, 0, 0]
    assert planner.criteria is None and planner._run_with_constraint == original
    assert len(result["joint_positions"]) == 11
    assert result["planner_dt_s"] == 0.004 and result["control_period_s"] == 0.04
    assert planner.scene_model == {"collision": "native"}


def test_joint_plan_uses_measured_start_and_official_plan_cspace():
    planner = NativePlanner()
    env, measured = native_scene(planner)
    original = planner.motion_planner.plan_cspace
    target = [1, 0, -0.5, 1, 0.3, -0.2]
    result = _plan_joint_path(env, "right", target)
    assert result["status"] == "Success"
    assert [call[0] for call in planner.calls] == ["plan_joint", "raw_cspace"]
    np.testing.assert_array_equal(planner.calls[0][1], measured)
    assert planner.calls[0][2] == target
    assert planner.motion_planner.plan_cspace == original
    np.testing.assert_array_equal(result["joint_positions"][0], measured)
    np.testing.assert_array_equal(result["joint_positions"][-1], target)
    assert result["joint_velocities"] == np.zeros((11, 6)).tolist()


@pytest.mark.parametrize("kind", ["position", "joint"])
@pytest.mark.parametrize(
    "status", [None, "IK_FAIL", "INVALID_START_STATE_WORLD_COLLISION"]
)
def test_official_failure_keeps_actual_status_without_fallback_or_guessed_reason(
    kind, status
):
    planner = NativePlanner()
    planner.fail, planner.status = True, status
    env, _ = native_scene(planner)
    result = (
        _plan_position_path(env, "right", [0.2, 0.1, 0.3])
        if kind == "position"
        else _plan_joint_path(env, "right", [0] * 6)
    )
    assert result["status"] == "Fail"
    assert result["planner_status"] == status
    assert result["reason_known"] is (status is not None)
    assert len([call for call in planner.calls if call[0].startswith("plan_")]) == 1
    assert planner.criteria is None


def test_joint_goal_outside_native_limits_is_rejected_before_planning():
    planner = NativePlanner()
    env, _ = native_scene(planner)
    result = _plan_joint_path(env, "right", [4, 0, 0, 0, 0, 0])
    assert result["status"] == "Fail" and "articulation limits" in result["error"]
    assert planner.calls == []


@pytest.mark.parametrize(
    "positions", [np.full((11, 6), np.nan), np.ones((11, 7)), np.full((11, 6), 4)]
)
def test_joint_trajectory_must_be_finite_with_native_dof_and_limits(positions):
    planner = NativePlanner()
    env, _ = native_scene(planner)
    planner.plan_joint = lambda *args: {
        "status": "Success",
        "position": positions,
        "velocity": np.zeros_like(positions),
    }
    with pytest.raises(ValueError, match="trajectory|articulation limits"):
        _plan_joint_path(env, "right", [0] * 6)


def test_native_exception_restores_official_hooks_and_does_not_search_alternatives():
    planner = NativePlanner()
    env, _ = native_scene(planner)
    original = planner.motion_planner.plan_cspace

    def failure(*args):
        raise RuntimeError("official native failure")

    planner.plan_joint = failure
    with pytest.raises(RuntimeError, match="official native failure"):
        _plan_joint_path(env, "right", [0] * 6)
    assert planner.motion_planner.plan_cspace == original


def queue_scene():
    from copy import deepcopy

    from reweave.benchmarks.robodojo.agent_support import _apply_trajectory_window

    planner = NativePlanner()
    env, _ = native_scene(planner)
    active = env.robot_manager.robot_list[0]
    active.gripper_name = "right_ee"
    other = SimpleNamespace(type="target", arm_name="left_arm", gripper_name="left_ee")
    manager = env.robot_manager
    manager.robot_list.append(other)
    recorded = []
    pending = []
    previous = {
        "left_arm_joint_state": {"position": [1.5] * 6},
        "left_ee_joint_state": {"position": [0.02, 0.02]},
        "right_arm_joint_state": {"position": [0] * 6},
        "right_ee_joint_state": {"position": [0.01, 0.01]},
    }
    queue = SimpleNamespace(
        prev_control=[previous],
        get_empty=lambda ids: [0] if not pending else [],
        push=lambda ids, sequences: pending.extend(sequences[0]),
    )
    manager.control_manager = queue
    manager.process_name = lambda value: value + "_joint_state"
    manager.get_end_effector_real_val = lambda *args, **kwargs: {
        0: np.array([0.01, 0.01])
    }

    def plan_ee(*, env_idx, arm_tag, result):
        assert env_idx == 0 and arm_tag == "right_arm"
        return [
            {
                "right_arm_joint_state": {
                    "position": position.copy(),
                    "velocity": velocity.copy(),
                }
            }
            for position, velocity in zip(
                result["position"], result["velocity"], strict=True
            )
        ]

    manager.plan_ee = plan_ee
    grip_calls = []

    def plan_grip(**kwargs):
        grip_calls.append(kwargs)
        return [
            {"right_ee_joint_state": {"position": [0.005, 0.005], "velocity": [0, 0]}}
            for _ in range(kwargs["control_step_num"])
        ]

    manager.plan_endeffector_joint = plan_grip

    def step(ids):
        assert ids == [0]
        frame = pending.pop(0)
        recorded.append(deepcopy(frame))
        previous.update(frame)

    env.step = step
    env.apply_target = lambda *args: pytest.fail("trajectory cannot use policy ramp")
    rewards, phases = [], []
    env.reward_manager = SimpleNamespace(step=lambda ids: rewards.append(ids))
    env.interact = True
    env.query_support_arm_traj = lambda **kwargs: phases.append("query")
    env.check_support_arm_stable = lambda **kwargs: phases.append("stable")
    env.take_action_cnt = [7]
    env.support_arm_action = [
        [{"support": {"position": [index]}} for index in range(10)]
    ]
    return env, recorded, grip_calls, rewards, phases, _apply_trajectory_window


@pytest.mark.parametrize("rows", [1, 7, 10])
def test_native_window_preserves_every_position_and_velocity_and_counts_one_ack(rows):
    env, recorded, grips, rewards, phases, execute = queue_scene()
    points = np.linspace(np.zeros(6), np.ones(6), rows)
    speeds = np.full((rows, 6), 0.123)
    assert execute(env, "right", points, speeds) == rows
    np.testing.assert_array_equal(
        [frame["right_arm_joint_state"]["position"] for frame in recorded], points
    )
    np.testing.assert_array_equal(
        [frame["right_arm_joint_state"]["velocity"] for frame in recorded], speeds
    )
    assert env.take_action_cnt == [8] and len(recorded) == rows
    assert len(env.support_arm_action[0]) == 10 - rows
    assert rewards == [[0]] and phases == ["query", "stable"]
    assert all(
        frame["left_arm_joint_state"]["position"] == [1.5] * 6 for frame in recorded
    )
    assert grips == []


def test_native_window_uses_official_gripper_planning_and_settled_hold_targets():
    env, recorded, grips, _, _, execute = queue_scene()
    execute(env, "right", np.zeros((3, 6)), np.zeros((3, 6)), 1)
    assert grips == [
        {
            "env_idx": 0,
            "arm_tag": "right_arm",
            "target_val": [0.0],
            "control_step_num": 3,
        }
    ]
    assert all(
        frame["right_ee_joint_state"]["position"] == [0.005, 0.005]
        for frame in recorded
    )


@pytest.mark.parametrize(
    "positions", [np.zeros((11, 6)), np.full((2, 6), np.nan), np.full((2, 6), 5)]
)
def test_invalid_native_window_is_rejected_before_counter_or_queue_changes(positions):
    env, recorded, grips, rewards, _, execute = queue_scene()
    with pytest.raises(ValueError):
        execute(env, "right", positions, np.zeros_like(positions))
    assert (
        env.take_action_cnt == [7] and recorded == [] and rewards == [] and grips == []
    )
    assert len(env.support_arm_action[0]) == 10


@pytest.mark.parametrize("kind", ["position", "joint"])
def test_planner_timestep_mismatch_is_explicit_before_native_planning(kind):
    planner = NativePlanner()
    env, _ = native_scene(planner)
    env.robot_manager.dt = 0.008
    with pytest.raises(ValueError, match="timestep differs"):
        if kind == "position":
            _plan_position_path(env, "right", [0, 0, 0.3])
        else:
            _plan_joint_path(env, "right", [0] * 6)
    assert planner.calls == []


@pytest.mark.parametrize(
    "fault", ["missing", "malformed", "wrong_substeps", "wrong_dt"]
)
def test_trajectory_lost_or_invalid_ack_blocks_both_action_paths_without_replay(
    tmp_path, fault
):
    from examples.robodojo_demo import SimulatedEnvironmentRPC
    from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient
    from reweave.runtime.capability import PhysicalOutcomeUnknown

    class WireRPC(SimulatedEnvironmentRPC):
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            result = super().call(method, args=args, kwargs=kwargs, timeout_s=timeout_s)
            if method == "env.step_trajectory" and fault in {
                "wrong_substeps",
                "wrong_dt",
            }:
                result[-1][
                    "trajectory_substeps"
                    if fault == "wrong_substeps"
                    else "trajectory_dt_s"
                ] = 99
            return result

    rpc = WireRPC(terminal_after=99, directory=tmp_path / "server")
    client = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    client.plan_joint_path("right", [0] * 6)
    if fault in {"missing", "malformed"}:
        rpc.acknowledgement = fault
    with pytest.raises(PhysicalOutcomeUnknown):
        client.step_trajectory("right", [[0] * 6], [[0] * 6])
    assert rpc.actions == 1 and client.elapsed_steps() == 0
    assert client.last_obs is None
    with pytest.raises(PhysicalOutcomeUnknown):
        client.step_trajectory("right", [[0] * 6], [[0] * 6])
    with pytest.raises(PhysicalOutcomeUnknown):
        client.step(np.zeros(14))
    assert rpc.actions == 1


def test_trajectory_ack_counts_one_window_and_preserves_policy_joint_and_ee_interfaces(
    tmp_path,
):
    from examples.robodojo_demo import SimulatedEnvironmentRPC
    from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient

    rpc = SimulatedEnvironmentRPC(
        limit=5, terminal_after=99, directory=tmp_path / "server"
    )
    client = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    client.plan_joint_path("right", [0] * 6)
    observation, _, _, info = client.step_trajectory("right", [[0] * 6], [[0.1] * 6])
    assert client.elapsed_steps() == rpc.actions == 1
    assert observation["physical_step"] == 1 and info["trajectory_substeps"] == 1
    client.step(np.zeros(14))
    ee_action = {
        name: rpc.observation["state"][name]
        for name in (
            "left_ee_pose",
            "right_ee_pose",
            "left_ee_joint_state",
            "right_ee_joint_state",
        )
    }
    client.step(ee_action)
    assert client.elapsed_steps() == rpc.actions == 3
