"""Policy/gripper controls retain native execution without authored stop priors."""

import numpy as np

from examples.robodojo_demo import (
    SimulatedEnvironmentRPC,
    SimulatedPolicyRPC,
    create_simulated_toolkit,
)


def test_pose_lift_and_command_closure_never_stop_a_policy_chunk(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=99, terminal_after=99)
    policy = SimulatedPolicyRPC()
    submitted = []
    original = rpc.call

    def after_action():
        # This satisfies the removed 6cm-descent/4cm-lift/closed-command rule
        # after action two; neither pose motion nor command closure is a verdict.
        for arm in ("left", "right"):
            rpc.observation["state"][f"{arm}_ee_pose"][2] = (
                0.2 if rpc.actions == 1 else 0.3
            )
            rpc.observation["state"][f"{arm}_ee_joint_state"][:] = 0

    rpc.after_action = after_action

    def call(method, args=(), kwargs=None, timeout_s=None):
        assert method not in {
            "env.plan_pose_path",
            "env.solve_ik_position",
            "env.solve_ik_pose",
        }
        if method == "env.step":
            submitted.append(args[0])
        return original(method, args, kwargs, timeout_s)

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(
        tmp_path / "episode", rpc=rpc, policy=policy
    )
    try:
        result = toolkit.execute_tool("pi0_pick", {"arm": "left", "max_chunks": 2})
        assert result.result["error"] is None
        capability = result.result["capability_result"]
        assert capability["chunks_used"] == policy.calls == 2
        assert capability["actions_used"] == rpc.actions == 8
        assert capability["reference_arm"] == "left"
        assert capability["stop_reason"] == "max_chunks"
        assert capability["terminated"] is capability["truncated"] is False
        assert "success" not in capability and "arms" not in capability
        assert toolkit.solved() is False
        # The unmodified 14-value policy controls both arms and both grippers,
        # independently of the tool's reference arm label.
        for action in submitted:
            np.testing.assert_array_equal(action["left_arm_joint_state"], np.zeros(6))
            np.testing.assert_array_equal(action["right_arm_joint_state"], np.zeros(6))
            assert (
                action["left_ee_joint_state"] == action["right_ee_joint_state"] == [0.0]
            )
        suffixes = [
            e
            for e in toolkit._execution_trace.events
            if e["event_type"] == "action_suffix_discarded"
        ]
        assert (
            not suffixes
        )  # No policy suffix was discarded by a pose/gripper heuristic.
    finally:
        toolkit.close()


def test_set_gripper_feedback_labels_the_command_without_claiming_aperture(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=99, terminal_after=99)
    submitted = []
    original = rpc.call

    def call(method, args=(), kwargs=None, timeout_s=None):
        assert method not in {
            "env.plan_pose_path",
            "env.solve_ik_position",
            "env.solve_ik_pose",
        }
        if method == "env.step":
            submitted.append(args[0])
        return original(method, args, kwargs, timeout_s)

    rpc.call = call
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert result.result["error"] is None
        capability = result.result["capability_result"]
        assert capability["commanded_gripper"] == "closed"
        assert "gripper" not in capability and "success" not in capability
        assert rpc.actions == len(submitted) == 1 and policy.calls == 0
        action = submitted[0]
        np.testing.assert_array_equal(action["left_arm_joint_state"], np.zeros(6))
        np.testing.assert_array_equal(action["right_arm_joint_state"], np.zeros(6))
        assert action["left_ee_joint_state"] == [1.0]
        assert action["right_ee_joint_state"] == [0.0]
        # The fixture deliberately acknowledges while retaining the old opening;
        # command acceptance must not be mislabeled as measured closure.
        assert toolkit._primitives.env.last_obs["state"]["right_ee_joint_state"] == [
            1.0
        ]
    finally:
        toolkit.close()
