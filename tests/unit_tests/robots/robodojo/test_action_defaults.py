"""Optional per-tool action budgets preserve explicit limits and complete paths."""

from types import SimpleNamespace

import pytest
from jsonschema import validate

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo import tools
from tests.unit_tests.robots.robodojo.test_move_pose import (
    install_pose_solver,
    pose_arguments,
)


@pytest.mark.parametrize("name", ["move_to", "move_joints"])
@pytest.mark.parametrize("explicit_limit,expected", [(None, 23), (2, 0)])
def test_default_action_budget_can_exceed_old_twenty_steps_and_keeps_explicit_limit(
    tmp_path, name, explicit_limit, expected
):
    rpc = SimulatedEnvironmentRPC(limit=23, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    arguments = (
        {"arm": "right", "xyz": [0.1, 0, 0.3]}
        if name == "move_to"
        else {"arm": "right", "joint_positions": [0.2] * 6, "tol_rad": 0.01}
    )
    original = rpc.call

    def planned(method, args=(), kwargs=None, **options):
        if method in {"env.plan_position_path", "env.plan_joint_path"}:
            return {
                "status": "Success",
                "planner_dt_s": 0.04,
                "control_period_s": 0.04,
                "joint_positions": [[0.2] * 6] * 23,
                "joint_velocities": [[0.0] * 6] * 23,
            }
        return original(method, args=args, kwargs=kwargs, **options)

    rpc.call = planned
    if explicit_limit is not None:
        arguments["max_steps"] = explicit_limit
    try:
        feedback = toolkit.execute_tool(name, arguments).result["capability_result"]
        assert feedback["steps_used"] == rpc.actions == expected
        assert feedback["reached"] is False
        assert toolkit.control_step_budget()["remaining"] == 23 - expected
    finally:
        toolkit.close()


@pytest.mark.parametrize("max_steps", [None, 8])
def test_complete_pose_path_cannot_partially_consume_insufficient_remaining_budget(
    tmp_path, max_steps
):
    rpc = SimulatedEnvironmentRPC(limit=4, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc, waypoint_count=4)
    arguments = pose_arguments(xyz=[0.1, 0, 0.3])
    if max_steps is None:
        arguments.pop("max_steps")
    else:
        arguments["max_steps"] = max_steps
    try:
        toolkit.execute_tool("set_gripper", {"arm": "left", "gripper": -1})
        assert toolkit.control_step_budget()["remaining"] == 3
        feedback = toolkit.execute_tool("move_pose", arguments).result[
            "capability_result"
        ]
        assert feedback["planned_actions"] == 4
        assert feedback["steps_used"] == 0
        assert feedback["reached"] is False
        assert "requires 4 acknowledged actions" in feedback["ik_error"]
        assert "remaining episode budget=3" in feedback["ik_error"]
        assert rpc.actions == toolkit.control_step_budget()["used"] == 1
    finally:
        toolkit.close()


def test_optional_pose_budget_allows_a_complete_path_exactly_fitting_remaining_steps(
    tmp_path,
):
    rpc = SimulatedEnvironmentRPC(limit=3, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    _, commands = install_pose_solver(rpc, waypoint_count=3)
    rpc.after_action = lambda: rpc.observation["state"].__setitem__(
        "right_arm_joint_state", commands[-1]["right_arm_joint_state"]
    )
    arguments = pose_arguments(xyz=[0.1, 0, 0.3])
    arguments.pop("max_steps")
    try:
        feedback = toolkit.execute_tool("move_pose", arguments).result[
            "capability_result"
        ]
        assert feedback["planned_actions"] == feedback["steps_used"] == rpc.actions == 3
        assert feedback["ik_error"] is None
        assert feedback["reached"] is True
        assert toolkit.control_step_budget()["remaining"] == 0
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("move_to", {"arm": "right", "xyz": [0.1, 0, 0.3]}),
        (
            "move_pose",
            {
                "arm": "right",
                "xyz": [0.1, 0, 0.3],
                "quaternion_wxyz": [1, 0, 0, 0],
                "tol_m": 0.005,
                "tol_rad": 0.01,
            },
        ),
        (
            "move_joints",
            {"arm": "right", "joint_positions": [0.2] * 6, "tol_rad": 0.01},
        ),
    ],
)
def test_omitted_budget_is_valid_and_the_declared_upper_bound_has_a_default(
    tmp_path, name, arguments
):
    spec = next(spec for spec in tools.TOOLS_SPEC if spec["name"] == name)
    validate(arguments, spec["input_schema"])
    # Omitted means the remaining budget; an explicit value can be any positive integer.
    validate({**arguments, "max_steps": 1500}, spec["input_schema"])
    assert "default" not in spec["input_schema"]["properties"]["max_steps"]
    rpc = SimulatedEnvironmentRPC(limit=4, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        cost = toolkit.control_step_budget()["operation_costs"][name]
        assert cost["upper_bound_formula"] == "max_steps or remaining_episode_budget"
        assert "default_arguments" not in cost
        assert rpc.actions == 0
    finally:
        toolkit.close()


@pytest.mark.parametrize("max_steps,expected", [(100, 0), (250, 101), (2, 0)])
def test_complete_path_preserves_explicit_budget_above_default(max_steps, expected):
    # Pure local trajectory dispatch: no server, inference, real action or media.
    env = SimpleNamespace(
        last_obs={},
        terminated=False,
        truncated=False,
        count=0,
        steps_remaining=lambda: 203,
    )

    def step_trajectory(arm, positions, velocities, *, gripper):
        env.count += 1
        return {}, 0.0, False, {}

    env.step_trajectory = step_trajectory
    primitives = SimpleNamespace(env=env, _check_cancelled=lambda: None)
    plan = {
        "status": "Success",
        "planner_dt_s": 0.04,
        "control_period_s": 0.04,
        "joint_positions": [[0.2] * 6] * 101,
        "joint_velocities": [[0.0] * 6] * 101,
    }
    result = tools._execute_native_path(
        primitives, "right", plan, max_steps=max_steps, gripper=0
    )
    assert result["steps_used"] == env.count == expected
    assert result["planned_actions"] == 101
    assert (result["ik_error"] is None) is (expected == 101)


def test_default_pose_budget_is_the_remaining_budget_not_a_fixed_cap(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=250, terminal_after=9999)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc, waypoint_count=101)
    arguments = pose_arguments(xyz=[0.1, 0, 0.3])
    arguments.pop("max_steps")
    try:
        feedback = toolkit.execute_tool("move_pose", arguments).result[
            "capability_result"
        ]
        assert feedback["planned_actions"] == 101
        assert feedback["ik_error"] is None
        assert feedback["steps_used"] == rpc.actions == 101
    finally:
        toolkit.close()


def test_default_pose_budget_still_rejects_a_path_over_the_remaining_budget(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=100, terminal_after=9999)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    install_pose_solver(rpc, waypoint_count=101)
    arguments = pose_arguments(xyz=[0.1, 0, 0.3])
    arguments.pop("max_steps")
    try:
        feedback = toolkit.execute_tool("move_pose", arguments).result[
            "capability_result"
        ]
        assert feedback["steps_used"] == rpc.actions == 0
        assert "remaining episode budget=100" in feedback["ik_error"]
    finally:
        toolkit.close()


def test_dual_arm_motion_and_gripper_commands_reject_implicit_arm_and_intermediate_values():
    from reweave.benchmarks.robodojo import tools

    schemas = {tool["name"]: tool["input_schema"] for tool in tools.TOOLS_SPEC}
    assert "arm" in schemas["move_to"]["required"]
    assert schemas["set_gripper"]["properties"]["gripper"]["enum"] == [-1, 1]
    with pytest.raises(ValueError, match="gripper 1 .* or -1"):
        tools.set_gripper(None, None, "right", 0.5)
    with pytest.raises(TypeError):
        tools.move_to(None, None, [0.1, 0.0, 0.3])
