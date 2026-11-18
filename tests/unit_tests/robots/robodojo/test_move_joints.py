# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Explicit public joint targets restore configurations through the normal ACK path."""

import json

import numpy as np
import pytest
from jsonschema import ValidationError, validate
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo import tools
from reweave.benchmarks.robodojo.robot_spec import ROBODOJO_DASHBOARD_SPEC
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.runtime_manifest import RuntimeAuthoringContract
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo

completed_demo = test_demo.completed_demo


def joint_arguments(arm="right", **values):
    return {
        "arm": arm,
        "joint_positions": [0.0] * 6,
        "tol_rad": 0.01,
        "max_steps": 3,
        **values,
    }


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("gripper", [-1, 0, 1])
def test_same_xyz_different_joint_configuration_requires_acknowledged_motion(
    tmp_path, arm, gripper
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    other = "left" if arm == "right" else "right"
    rpc.observation["state"][f"{arm}_arm_joint_state"] = np.full(6, 0.4)
    rpc.observation["state"][f"{other}_arm_joint_state"] = np.arange(6) / 10
    rpc.observation["state"][f"{arm}_ee_joint_state"] = np.array([0.4])
    rpc.observation["state"][f"{other}_ee_joint_state"] = np.array([0.6])
    rpc.observation["state"][f"{arm}_ee_pose"][3:] = [
        np.cos(np.pi / 8),
        0,
        0,
        np.sin(np.pi / 8),
    ]
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    commands = []

    def call(method, args=(), **kwargs):
        if method == "env.solve_ik_position":
            pytest.fail("joint targets must not use pose IK")
        if method == "env.step":
            commands.append(args[0])
            # Synthetic forward kinematics restores orientation at unchanged XYZ.
            rpc.observation["state"][f"{arm}_arm_joint_state"] = np.asarray(
                args[0][f"{arm}_arm_joint_state"]
            )
            rpc.observation["state"][f"{arm}_ee_pose"][3:] = [1, 0, 0, 0]
        return original(method, args=args, **kwargs)

    rpc.call = call
    try:
        result = toolkit.execute_tool(
            "move_joints", joint_arguments(arm, gripper=gripper)
        ).result
        feedback = result["capability_result"]
        assert not result.get("error")
        assert feedback["reached"] and feedback["steps_used"] == rpc.actions == 1
        assert feedback["start_joint_positions"] == [0.4] * 6
        assert feedback["final_joint_positions"] == [0.0] * 6
        assert feedback["joint_errors_rad"] == [0.0] * 6
        assert feedback["max_joint_error_rad"] == 0
        np.testing.assert_array_equal(
            commands[0][f"{other}_arm_joint_state"], np.arange(6) / 10
        )
        assert commands[0][f"{other}_ee_joint_state"] == [0.6]
        assert commands[0][f"{arm}_ee_joint_state"] == [
            0.4 if gripper == 0 else 0.0 if gripper > 0 else 1.0
        ]
        trace = toolkit._execution_trace.events
        assert sum(row["event_type"] == "control_step" for row in trace) == 1
        assert not any(row["event_type"] == "action_outcome_unknown" for row in trace)
    finally:
        toolkit.close()


@pytest.mark.parametrize("limit,max_steps,expected", [(5, 2, 1), (1, 3, 1)])
def test_measured_residual_stops_at_tool_or_episode_allowance_without_tool_failure(
    tmp_path, limit, max_steps, expected
):
    rpc = SimulatedEnvironmentRPC(limit=limit, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool(
            "move_joints",
            joint_arguments(joint_positions=[0.2] * 6, max_steps=max_steps),
        ).result
        feedback = result["capability_result"]
        assert not result.get("error")
        assert feedback["reached"] is False
        assert feedback["max_joint_error_rad"] == pytest.approx(0.2)
        assert feedback["joint_errors_rad"] == [0.2] * 6
        assert feedback["steps_used"] == rpc.actions == expected
        assert toolkit.control_step_budget()["used"] == expected
    finally:
        toolkit.close()


def test_each_joint_action_uses_measured_progress_and_fresh_other_arm_state(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    commands = []

    def call(method, args=(), **kwargs):
        if method == "env.plan_joint_path":
            return {
                "status": "Success",
                "planner_dt_s": 0.04,
                "control_period_s": 0.04,
                "joint_positions": [[0.1] * 6, [0.2] * 6, [0.3] * 6],
                "joint_velocities": [[0.0] * 6] * 3,
            }
        if method == "env.step":
            commands.append(args[0])
            np.testing.assert_allclose(
                args[0]["left_arm_joint_state"], (len(commands) - 1) * 0.02
            )
            rpc.observation["state"]["right_arm_joint_state"] = np.full(
                6, len(commands) * 0.1
            )
            rpc.observation["state"]["left_arm_joint_state"] = np.full(
                6, len(commands) * 0.02
            )
        return original(method, args=args, **kwargs)

    rpc.call = call
    try:
        feedback = toolkit.execute_tool(
            "move_joints", joint_arguments(joint_positions=[0.3] * 6, max_steps=5)
        ).result["capability_result"]
        assert feedback["reached"] and feedback["steps_used"] == len(commands) == 3
        assert feedback["max_joint_error_rad"] < 0.01
        np.testing.assert_allclose(feedback["final_joint_positions"], 0.3)
    finally:
        toolkit.close()


@pytest.mark.parametrize("gripper,steps", [(0, 0), (-1, 1), (1, 1)])
def test_reached_joint_target_still_executes_an_explicit_gripper_command(
    tmp_path, gripper, steps
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        feedback = toolkit.execute_tool(
            "move_joints", joint_arguments(gripper=gripper)
        ).result["capability_result"]
        assert feedback["reached"] and feedback["steps_used"] == rpc.actions == steps
    finally:
        toolkit.close()


@pytest.mark.parametrize("read_number", [1, 2])
def test_cancellation_during_observation_prevents_joint_submission(
    tmp_path, monkeypatch, read_number
):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    current_obs = tools._current_obs
    observations = 0

    def observe(primitives):
        nonlocal observations
        value = current_obs(primitives)
        observations += 1
        if observations == read_number:
            toolkit._active_operation.cancel_event.set()
        return value

    monkeypatch.setattr(tools, "_current_obs", observe)
    try:
        result = toolkit.execute_tool(
            "move_joints", joint_arguments(joint_positions=[0.3] * 6)
        ).result
        assert result["capability_result"]["code"] == "tool_cancelled"
        assert rpc.actions == toolkit.control_step_budget()["used"] == 0
    finally:
        toolkit.close()


def test_lost_joint_action_acknowledgement_is_unknown_without_replay(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99, acknowledgement="missing")
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        with pytest.raises(PhysicalOutcomeUnknown) as raised:
            toolkit.execute_tool(
                "move_joints", joint_arguments(joint_positions=[0.2] * 6)
            )
        assert toolkit.physical_outcome_error is raised.value
        assert rpc.actions == 1 and toolkit.control_step_budget()["used"] == 0
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "values",
    [
        {"joint_positions": [0] * 5},
        {"joint_positions": [0] * 7},
        {"joint_positions": [0, 0, 0, 0, 0, float("nan")]},
        {"arm": "support"},
        {"tol_rad": 0},
        {"max_steps": 0},
    ],
)
def test_invalid_joint_target_is_rejected_before_physical_submission(tmp_path, values):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool("move_joints", joint_arguments(**values)).result
        assert result.get("error")
        assert rpc.actions == 0
    finally:
        toolkit.close()


def test_joint_tool_is_in_actual_frozen_authoring_contract_and_planner_execution(
    completed_demo, tmp_path
):
    captured, _, _, _, _ = completed_demo
    bundle = AgentBundle.model_validate_json(
        (captured / "baseline/bundle.json").read_text()
    )
    rpc = SimulatedEnvironmentRPC(terminal_after=1)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    limits = EpisodeLimits(program_calls=20, gateway_calls=20, wall_time_s=30)
    runtime = BundleRuntime(
        bundle, toolkit, "SIMULATED joint target", episode_limits=limits
    )
    contract = RuntimeAuthoringContract.from_runtime(runtime)
    public_contract = json.loads(contract.model_dump_json())
    tool = next(
        tool for tool in public_contract["tools"] if tool["name"] == "move_joints"
    )
    assert contract.operations["move_joints"]["effect"] == "physical"
    capability = next(
        item
        for item in public_contract["capability_catalog"]
        if item["name"] == "move_joints"
    )
    assert capability["effect"] == "physical"
    assert capability["input_schema"] == tool["input_schema"]
    assert "capability_result" in capability["output_schema"]["properties"]
    validate(joint_arguments(), tool["input_schema"])
    for malformed in (
        {"arm": "right"},
        joint_arguments(xyz=[0, 0, 0]),
        joint_arguments(joint_positions=[0] * 7),
    ):
        with pytest.raises(ValidationError):
            validate(malformed, tool["input_schema"])
    assert "move_joints" in ROBODOJO_DASHBOARD_SPEC["primitives"]
    assert toolkit.control_step_budget()["operation_costs"]["move_joints"] == {
        "upper_bound_formula": "max_steps or remaining_episode_budget",
    }
    (tmp_path / "authoring_contract.json").write_text(
        contract.model_dump_json(indent=2)
    )
    requests = []

    def model(messages, info):
        requests.append(messages)
        assert "move_joints" in {item.name for item in info.function_tools}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "move_joints",
                    joint_arguments(joint_positions=[0.2] * 6),
                    "joint-motion",
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
            user_message="Move joints",
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None
        assert len(requests) == rpc.actions == 1
        events = [
            json.loads(line)
            for line in (tmp_path / "episode/harness_context/bundle_events.jsonl")
            .read_text()
            .splitlines()
        ]
        assert any(
            event["kind"] == "action_completed"
            and event.get("operation") == "move_joints"
            for event in events
        )
    finally:
        toolkit.close()
