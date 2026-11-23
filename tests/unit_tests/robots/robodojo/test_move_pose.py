# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Full-pose control preserves rotation constraints and the shared ACK lifecycle."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import ValidationError, validate
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo import tools
from reweave.benchmarks.robodojo.agent_support import (
    _plan_pose_path,
    _solve_ik_pose,
)
from reweave.benchmarks.robodojo.env_facade import RoboDojoEnvFacade
from reweave.benchmarks.robodojo.prompt_bundle import system_prompt
from reweave.benchmarks.robodojo.robot_spec import ROBODOJO_DASHBOARD_SPEC
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.runtime_manifest import RuntimeAuthoringContract
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo, test_runtime_limits

completed_demo = test_demo.completed_demo
agent_environment = test_runtime_limits.agent_environment


def pose_arguments(arm="right", **values):
    return {
        "arm": arm,
        "xyz": [0.0, 0.0, 0.3],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "tol_m": 0.005,
        "tol_rad": 0.01,
        "max_steps": 3,
        **values,
    }


def install_pose_solver(
    rpc, *, reached=True, result=None, cancel=None, waypoint_count=1
):
    original = rpc.call
    targets, commands = [], []
    solved_arm = None

    def call(method, args=(), kwargs=None, **options):
        nonlocal solved_arm
        if method == "env.solve_ik_position":
            pytest.fail("full-pose control must not relax orientation")
        if method == "env.plan_pose_path":
            solved_arm = kwargs["arm"]
            targets.append(kwargs["poses"][-1])
            if cancel is not None:
                cancel()
            if result is not None and result["status"] == "Fail":
                return result
            joints = result["joint_value"] if result is not None else [0.2] * 6
            return {
                "status": "Success",
                "planner_dt_s": 0.04,
                "control_period_s": 0.04,
                "joint_positions": [joints] * waypoint_count,
                "joint_velocities": [[0.0] * len(joints)] * waypoint_count,
            }
        if method == "env.step":
            commands.append(args[0])
            if reached and targets:
                # Synthetic forward kinematics supplies measured full-pose progress.
                rpc.observation["state"][f"{solved_arm}_ee_pose"] = np.array(
                    targets[-1]
                )
        return original(method, args=args, kwargs=kwargs, **options)

    rpc.call = call
    return targets, commands


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("gripper", [-1, 0, 1])
def test_same_xyz_wrong_orientation_requires_full_pose_motion(tmp_path, arm, gripper):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    other = "left" if arm == "right" else "right"
    rpc.observation["state"][f"{arm}_ee_pose"][3:] = [0, 0, 0, 1]
    rpc.observation["state"][f"{arm}_ee_joint_state"] = np.array([0.4])
    rpc.observation["state"][f"{other}_arm_joint_state"] = np.arange(6) / 10
    rpc.observation["state"][f"{other}_ee_joint_state"] = np.array([0.6])
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    targets, commands = install_pose_solver(rpc)
    try:
        feedback = toolkit.execute_tool(
            "move_pose", pose_arguments(arm, gripper=gripper)
        ).result["capability_result"]
        assert feedback["reached"] and feedback["steps_used"] == rpc.actions == 1
        assert feedback["start_pose"][:3] == feedback["final_pose"][:3]
        assert feedback["start_pose"][3:] != feedback["final_pose"][3:]
        assert targets == [[0, 0, 0.3, 1, 0, 0, 0]]
        assert feedback["position_error_m"] == feedback["rotation_error_rad"] == 0
        np.testing.assert_array_equal(
            commands[0][f"{other}_arm_joint_state"], np.arange(6) / 10
        )
        assert commands[0][f"{other}_ee_joint_state"] == [0.6]
        assert commands[0][f"{arm}_ee_joint_state"] == [
            0.4 if gripper == 0 else 0.0 if gripper > 0 else 1.0
        ]
        assert toolkit.control_step_budget()["used"] == 1
    finally:
        toolkit.close()


@pytest.mark.parametrize("limit,max_steps,expected", [(5, 2, 1), (1, 3, 1)])
def test_pose_residual_obeys_tool_and_episode_limits(
    tmp_path, limit, max_steps, expected
):
    rpc = SimulatedEnvironmentRPC(limit=limit, terminal_after=99)
    rpc.observation["state"]["right_ee_pose"][3:] = [0, 0, 0, 1]
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    targets, _ = install_pose_solver(rpc, reached=False)
    try:
        feedback = toolkit.execute_tool(
            "move_pose", pose_arguments(max_steps=max_steps)
        ).result["capability_result"]
        assert feedback["reached"] is False
        assert feedback["position_error_m"] == 0
        assert feedback["rotation_error_rad"] == pytest.approx(np.pi)
        assert feedback["steps_used"] == rpc.actions == expected
        assert feedback["max_joint_error_rad"] == pytest.approx(0.2)
        assert feedback["ik_error"] is None
        assert len(targets) == 1
        assert toolkit.control_step_budget()["used"] == expected
    finally:
        toolkit.close()


def test_pose_returns_final_residual_without_terminal_target_replay(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=5, terminal_after=99)
    rpc.observation["state"]["right_ee_pose"][3:] = [0, 0, 0, 1]
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(
        rpc, reached=False, result={"status": "Success", "joint_value": [0.02] * 6}
    )
    try:
        feedback = toolkit.execute_tool(
            "move_pose", pose_arguments(max_steps=4)
        ).result["capability_result"]
        assert feedback["steps_used"] == rpc.actions == 1
        assert feedback["reached"] is False
        assert feedback["max_joint_error_rad"] == pytest.approx(0.02)
        assert feedback["ik_error"] is None
        assert toolkit.control_step_budget()["remaining"] == 4
    finally:
        toolkit.close()


@pytest.mark.parametrize("gripper,expected", [(0, 0), (-1, 1), (1, 1)])
def test_quaternion_sign_equivalence_and_explicit_gripper_at_reached_pose(
    tmp_path, gripper, expected
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    targets, _ = install_pose_solver(rpc)
    try:
        feedback = toolkit.execute_tool(
            "move_pose",
            pose_arguments(quaternion_wxyz=[-2, 0, 0, 0], gripper=gripper),
        ).result["capability_result"]
        assert feedback["reached"] and feedback["rotation_error_rad"] == 0
        assert feedback["steps_used"] == rpc.actions == expected
        assert targets == []
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "failure",
    [
        {"status": "Fail", "error": "unreachable"},
        {"status": "Success", "joint_value": [0] * 5},
    ],
)
def test_ik_failure_or_malformed_solution_submits_no_action(tmp_path, failure):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    targets, _ = install_pose_solver(rpc, result=failure)
    try:
        result = toolkit.execute_tool(
            "move_pose", pose_arguments(xyz=[0.1, 0, 0.3])
        ).result
        if failure["status"] == "Fail":
            assert result["capability_result"]["ik_error"] == "unreachable"
            assert result["capability_result"]["reached"] is False
        else:
            assert "invalid path" in result["error"]
        assert len(targets) == 1 and rpc.actions == 0
        assert toolkit.control_step_budget()["used"] == 0
    finally:
        toolkit.close()


@pytest.mark.parametrize("boundary", ["first_read", "second_read", "ik"])
def test_cancellation_during_observation_or_ik_prevents_pose_submission(
    tmp_path, monkeypatch, boundary
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)

    def cancel():
        toolkit._active_operation.cancel_event.set()

    install_pose_solver(rpc, cancel=cancel if boundary == "ik" else None)
    current_obs = tools._current_obs
    reads = 0

    def observe(primitives):
        nonlocal reads
        value = current_obs(primitives)
        reads += 1
        if (boundary, reads) in {("first_read", 1), ("second_read", 2)}:
            cancel()
        return value

    monkeypatch.setattr(tools, "_current_obs", observe)
    try:
        result = toolkit.execute_tool(
            "move_pose", pose_arguments(xyz=[0.1, 0, 0.3])
        ).result
        assert result["capability_result"]["code"] == "tool_cancelled"
        assert rpc.actions == toolkit.control_step_budget()["used"] == 0
    finally:
        toolkit.close()


def test_lost_pose_action_ack_is_unknown_and_cannot_replay(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99, acknowledgement="missing")
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc)
    try:
        with pytest.raises(PhysicalOutcomeUnknown) as raised:
            toolkit.execute_tool("move_pose", pose_arguments(xyz=[0.1, 0, 0.3]))
        with pytest.raises(PhysicalOutcomeUnknown) as repeated:
            toolkit.execute_tool("move_pose", pose_arguments(xyz=[0.1, 0, 0.3]))
        assert raised.value is repeated.value is toolkit.physical_outcome_error
        assert rpc.actions == 1 and toolkit.control_step_budget()["used"] == 0
        assert (
            sum(
                row["event_type"] == "action_outcome_unknown"
                for row in toolkit._execution_trace.events
            )
            == 1
        )
    finally:
        toolkit.close()


def test_pose_path_exceeding_max_steps_is_rejected_before_action(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc, waypoint_count=4)
    try:
        feedback = toolkit.execute_tool(
            "move_pose", pose_arguments(xyz=[0.1, 0, 0.3], max_steps=3)
        ).result["capability_result"]
        assert feedback["reached"] is False
        assert "requires 4 acknowledged actions" in feedback["ik_error"]
        assert feedback["planned_actions"] == 4
        assert rpc.actions == toolkit.control_step_budget()["used"] == 0
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "values",
    [
        {"xyz": [0, 0]},
        {"quaternion_wxyz": [0, 0, 0, 0]},
        {"quaternion_wxyz": [1, 0, float("nan"), 0]},
        {"tol_m": 0},
        {"tol_rad": -1},
        {"max_steps": 0},
        {"arm": "support"},
    ],
)
def test_invalid_pose_is_rejected_before_action(tmp_path, values):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool("move_pose", pose_arguments(**values)).result
        assert result.get("error") and rpc.actions == 0
    finally:
        toolkit.close()


def test_rpc_adapter_reaches_native_full_pose_solver_unchanged(
    agent_environment, tmp_path
):
    robot = SimpleNamespace(
        type="target",
        arm_name="right_arm",
        arm_joints_name=[f"joint{index}" for index in range(1, 7)],
        arm_joint_indices=list(range(6)),
    )
    articulation = SimpleNamespace(
        data=SimpleNamespace(soft_joint_pos_limits=np.asarray([[[-6.0, 6.0]] * 6]))
    )
    received = []

    def solve_ik(**kwargs):
        received.append(kwargs)
        return {"status": "Success", "joint_value": np.arange(6)}

    native = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[robot], robot_key=[articulation], solve_ik=solve_ik
        )
    )
    server = agent_environment.__new__(agent_environment)
    server.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)])
    from reweave.benchmarks.robodojo.observation_service import ObservationService

    server.observations = ObservationService(tmp_path)
    facade = RoboDojoEnvFacade(server)
    target = [0.12, -0.31, 0.9, 1, 1, 0, 0]
    result = facade._dispatch("env.solve_ik_pose", (), {"arm": "right", "pose": target})
    assert "env.solve_ik_pose" in facade._readonly_methods
    assert result["status"] == "Success"
    assert len(received) == 1 and received[0]["robot"] is robot
    assert received[0]["env_idx"] == 0
    np.testing.assert_allclose(
        received[0]["target_pose"], [0.12, -0.31, 0.9, 2**-0.5, 2**-0.5, 0, 0]
    )
    assert target == [0.12, -0.31, 0.9, 1, 1, 0, 0]


def test_native_ik_failure_does_not_try_another_orientation():
    robot = SimpleNamespace(type="target", arm_name="left_arm")
    targets = []

    def solve_ik(**kwargs):
        targets.append(kwargs["target_pose"])
        return {"status": "Fail", "error": "unreachable rotation"}

    env = SimpleNamespace(
        robot_manager=SimpleNamespace(robot_list=[robot], solve_ik=solve_ik)
    )
    result = _solve_ik_pose(env, "left", [0, 0, 0.3, 0, 0, 1, 0])
    assert result == {"status": "Fail", "error": "unreachable rotation"}
    assert targets == [[0, 0, 0.3, 0, 0, 1, 0]]


def test_native_ik_solution_outside_articulation_limits_is_rejected():
    robot = SimpleNamespace(
        type="target",
        arm_name="right_arm",
        arm_joints_name=[f"joint{index}" for index in range(1, 7)],
        arm_joint_indices=list(range(6)),
    )
    articulation = SimpleNamespace(
        data=SimpleNamespace(
            soft_joint_pos_limits=np.asarray(
                [[[-3.0, 3.0]] * 5 + [[-2.4, 2.4]]], dtype=np.float64
            )
        )
    )

    def solve_ik(**kwargs):
        return {
            "status": "Success",
            "joint_value": [0.0, 0.0, 0.0, 0.0, 0.0, -2.89],
        }

    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[robot], robot_key=[articulation], solve_ik=solve_ik
        )
    )
    result = _solve_ik_pose(env, "right", [0, 0, 0.3, 1, 0, 0, 0])

    assert result == {
        "status": "Fail",
        "error": "IK solution exceeds native articulation limits",
        "joints": ["joint6"],
    }


def test_native_ik_requires_authoritative_articulation_limits():
    robot = SimpleNamespace(
        type="target",
        arm_name="left_arm",
        arm_joints_name=[f"joint{index}" for index in range(1, 7)],
        arm_joint_indices=list(range(6)),
    )
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[robot],
            solve_ik=lambda **kwargs: {
                "status": "Success",
                "joint_value": [0.0] * 6,
            },
        )
    )

    with pytest.raises(ValueError, match="does not expose articulation limits"):
        _solve_ik_pose(env, "left", [0, 0, 0.3, 1, 0, 0, 0])


def configured_native_scene(planner):
    robot = SimpleNamespace(
        type="target",
        arm_name="right_arm",
        robot_name="x5",
        entity_origin_pose=[-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707],
        arm_joints_name=[f"joint{index}" for index in range(1, 7)],
        arm_joint_indices=list(range(6)),
    )
    articulation = SimpleNamespace(
        data=SimpleNamespace(soft_joint_pos_limits=np.asarray([[[-3.2, 3.2]] * 6]))
    )
    manager = SimpleNamespace(
        dt=0.004,
        robot_list=[robot],
        robot_key=[articulation],
        planner={"x5": planner},
        ik_solver={"x5": planner},
        get_joint=lambda robot, env_idx_list: {0: np.zeros(6)},
    )
    env = SimpleNamespace(
        robot_manager=manager, obs_manager=SimpleNamespace(collect_interval=10)
    )
    return env, robot


class ConfiguredPlanner:
    """A planner with defaults and no adapter obstacle/collision override API."""

    dt = 0.004

    def __init__(self):
        self.robot_cfg = {
            "collision_links": ["base_link", "link6"],
            "lock_joints": None,
        }
        self.scene_model = {"table": {"dimensions": [3, 3, 0.05]}}
        self.calls = []
        self.ik_solver = object()

    def _run_with_constraint(self, planner, constraint_pose, plan_fn):
        return plan_fn()

    def plan_path(self, current, target, *, real_robot_pose):
        self.calls.append((np.asarray(current).copy(), target, real_robot_pose))
        self._run_with_constraint(
            None,
            None,
            lambda: SimpleNamespace(
                status="Success", success=np.asarray([True]), planning_time=0.01
            ),
        )
        end = np.asarray(current) + 1
        return {
            "status": "Success",
            "position": np.stack((current, end)),
            "velocity": np.zeros((2, 6)),
        }


def test_native_pose_path_chains_official_plans_without_scene_or_collision_overrides():
    planner = ConfiguredPlanner()
    env, robot = configured_native_scene(planner)
    before = json.dumps([planner.robot_cfg, planner.scene_model])
    result = _plan_pose_path(
        env, "right", [[0, 0, 0.3, 1, 0, 0, 0], [0.01, 0, 0.3, 1, 0, 0, 0]]
    )
    assert result["status"] == "Success" and len(result["joint_positions"]) == 3
    assert result["planner_dt_s"] == 0.004 and result["control_period_s"] == 0.04
    assert result["planner_world"] == {
        "source": "RoboDojo configured CuRobo planner",
        "robot_root_pose_world": robot.entity_origin_pose,
    }
    assert json.dumps([planner.robot_cfg, planner.scene_model]) == before
    np.testing.assert_array_equal(planner.calls[0][0], np.zeros(6))
    np.testing.assert_array_equal(planner.calls[1][0], np.ones(6))
    assert all(call[2] == robot.entity_origin_pose for call in planner.calls)
    assert len(result["raw_planner_results"]) == 2
    assert "fixed_mount_world_exception" not in result
    assert "restored_sphere_collision_validation" not in result
    # This scene has no reward/table/camera metadata and this planner exposes no
    # set_world_obstacles, disable_link_collision or generated finger model.


@pytest.mark.parametrize("raises", [False, True])
def test_native_pose_failure_stays_official_failure_without_an_alternate_plan(raises):
    planner = ConfiguredPlanner()
    env, _ = configured_native_scene(planner)
    original = planner._run_with_constraint
    called = []

    def fail(*args, **kwargs):
        called.append((args, kwargs))
        if raises:
            raise RuntimeError("official planning failed")
        return {"status": "Fail"}

    planner.plan_path = fail
    if raises:
        with pytest.raises(RuntimeError, match="official planning failed"):
            _plan_pose_path(env, "right", [[0, 0, 0.3, 1, 0, 0, 0]])
    else:
        result = _plan_pose_path(env, "right", [[0, 0, 0.3, 1, 0, 0, 0]])
        assert result["status"] == "Fail"
    assert len(called) == 1 and planner._run_with_constraint == original


@pytest.mark.parametrize(
    "positions", [np.full((2, 6), np.nan), np.zeros((2, 7)), np.full((2, 6), 9)]
)
def test_official_plan_still_requires_finite_in_limit_six_joint_waypoints(positions):
    planner = ConfiguredPlanner()
    env, _ = configured_native_scene(planner)
    planner.plan_path = lambda *args, **kwargs: {
        "status": "Success",
        "position": positions,
        "velocity": np.zeros_like(positions),
    }
    with pytest.raises(ValueError, match="invalid trajectory|articulation limits"):
        _plan_pose_path(env, "right", [[0, 0, 0.3, 1, 0, 0, 0]])


def test_pose_contract_exposure_and_planner_dispatch(completed_demo, tmp_path):
    captured, _, _, _, _ = completed_demo
    bundle = AgentBundle.model_validate_json(
        (captured / "baseline/bundle.json").read_text()
    )
    rpc = SimulatedEnvironmentRPC(terminal_after=1)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc)
    limits = EpisodeLimits(program_calls=20, gateway_calls=20, wall_time_s=30)
    runtime = BundleRuntime(
        bundle, toolkit, "SIMULATED pose target", episode_limits=limits
    )
    contract = RuntimeAuthoringContract.from_runtime(runtime)
    public_contract = json.loads(contract.model_dump_json())
    tools = {item["name"]: item for item in public_contract["tools"]}
    capabilities = {
        item["name"]: item for item in public_contract["capability_catalog"]
    }
    for name in ("move_pose", "move_to", "view_env_state", "back_project"):
        assert tools[name]["description"] == capabilities[name]["description"]
    for name in ("move_pose", "move_to", "view_env_state"):
        description = tools[name]["description"]
        assert "native configured EE frame" in description
        assert "*_ee_pose" in description and "fingertip/pinch" in description
    assert "visible surface" in tools["back_project"]["description"]
    assert "EE target" in tools["back_project"]["description"]
    assert "native configured EE frame" in system_prompt()
    assert "visible surface" in system_prompt()
    tool = next(
        tool for tool in public_contract["tools"] if tool["name"] == "move_pose"
    )
    capability = next(
        item
        for item in public_contract["capability_catalog"]
        if item["name"] == "move_pose"
    )
    assert (
        contract.operations["move_pose"]["effect"] == capability["effect"] == "physical"
    )
    assert capability["input_schema"] == tool["input_schema"]
    assert "capability_result" in capability["output_schema"]["properties"]
    validate(pose_arguments(), tool["input_schema"])
    for malformed in (
        {"arm": "right"},
        pose_arguments(extra=True),
        pose_arguments(quaternion_wxyz=[1, 0, 0]),
    ):
        with pytest.raises(ValidationError):
            validate(malformed, tool["input_schema"])
    assert "move_pose" in ROBODOJO_DASHBOARD_SPEC["primitives"]
    assert toolkit.control_step_budget()["operation_costs"]["move_pose"] == {
        "upper_bound_formula": "max_steps or remaining_episode_budget",
    }
    (tmp_path / "authoring_contract.json").write_text(
        contract.model_dump_json(indent=2)
    )

    def model(messages, info):
        assert "move_pose" in {item.name for item in info.function_tools}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "move_pose", pose_arguments(xyz=[0.1, 0, 0.3]), "pose-motion"
                )
            ]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(bundle, episode_limits=limits)
    try:
        result = planner.solve(
            system_prompt="SIMULATED",
            user_message="Move pose",
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None and rpc.actions == 1
        events = [
            json.loads(line)
            for line in (tmp_path / "episode/harness_context/bundle_events.jsonl")
            .read_text()
            .splitlines()
        ]
        assert any(
            event["kind"] == "action_completed"
            and event.get("operation") == "move_pose"
            for event in events
        )
    finally:
        toolkit.close()


@pytest.mark.parametrize("raises", [False, True])
def test_optional_linear_path_uses_public_native_criterion_and_restores_defaults(
    monkeypatch, raises
):
    import sys

    criteria = object()
    calls, updates = [], []

    def linear_motion(**kwargs):
        calls.append(kwargs)
        return criteria

    monkeypatch.setitem(sys.modules, "curobo", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "curobo.types",
        SimpleNamespace(ToolPoseCriteria=SimpleNamespace(linear_motion=linear_motion)),
    )
    planner = ConfiguredPlanner()
    planner.tool_frames = ["link6"]
    planner._default_tool_pose_criteria = {"link6": object()}
    planner.motion_planner = SimpleNamespace(update_tool_pose_criteria=updates.append)
    env, _ = configured_native_scene(planner)
    original = planner.plan_path

    def plan(*args, **kwargs):
        assert updates[-1] == {"link6": criteria}
        if raises:
            raise RuntimeError("native linear planning failed")
        return original(*args, **kwargs)

    planner.plan_path = plan
    if raises:
        with pytest.raises(RuntimeError, match="native linear planning failed"):
            _plan_pose_path(env, "right", [[0, 0, 0.3, 1, 0, 0, 0]], linear_axis="x")
    else:
        assert (
            _plan_pose_path(env, "right", [[0, 0, 0.3, 1, 0, 0, 0]], linear_axis="x")[
                "status"
            ]
            == "Success"
        )
    assert calls == [
        {"axis": "x", "non_terminal_scale": 1.0, "project_distance_to_goal": True}
    ]
    assert updates == [{"link6": criteria}, planner._default_tool_pose_criteria]
