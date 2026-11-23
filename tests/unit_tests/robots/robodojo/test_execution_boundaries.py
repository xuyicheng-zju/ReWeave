# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Acknowledgements, terminal boundaries and cancellation preserve action truth."""

import json

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.robots.execution_trace import merge_control_telemetry
from reweave.runtime.capability import PhysicalOutcomeUnknown


def events(toolkit):
    return toolkit._execution_trace.events


def test_official_terminal_in_chunk_stops_and_discards_suffix(tmp_path):
    toolkit, rpc, policy = create_simulated_toolkit(tmp_path / "episode")
    result = toolkit.execute_tool("pi0_pick", {"max_chunks": 3})
    assert not result.result.get("error"), result.result
    assert rpc.actions == 2 and policy.calls == 1
    capability = result.result["capability_result"]
    assert capability["stop_reason"] == "environment_terminal"
    assert capability["actions_used"] == 2
    assert "success" not in capability  # Only the official evaluator reports success.
    assert toolkit.solved() is True
    assert sum(row["event_type"] == "control_step" for row in events(toolkit)) == 2
    suffix = [
        row for row in events(toolkit) if row["event_type"] == "action_suffix_discarded"
    ]
    assert len(suffix) == 1 and suffix[0]["discarded_action_steps"] == 2
    assert toolkit._execution_controller.snapshot.phase == "idle"
    toolkit.close()


def test_cancellation_checked_before_each_action(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    rpc.after_action = lambda: toolkit._active_operation.cancel_event.set()
    result = toolkit.execute_tool("pi0_pick", {"max_chunks": 3})
    assert result.result["capability_result"]["code"] == "tool_cancelled"
    assert rpc.actions == 1 and policy.calls == 1
    assert sum(row["event_type"] == "control_step" for row in events(toolkit)) == 1
    suffix = [
        row for row in events(toolkit) if row["event_type"] == "action_suffix_discarded"
    ]
    assert sum(row["discarded_action_steps"] for row in suffix) == 3
    assert toolkit._execution_controller.snapshot.phase == "idle"
    toolkit.close()


@pytest.mark.parametrize(
    "acknowledgement", ["missing", "malformed", "counter", "terminal_type"]
)
def test_unknown_ack_permanently_blocks_replay_and_reset(tmp_path, acknowledgement):
    rpc = SimulatedEnvironmentRPC(acknowledgement=acknowledgement, terminal_after=99)
    original = rpc.call

    def call(method, *args, **kwargs):
        value = original(method, *args, **kwargs)
        if method == "env.step" and acknowledgement in {"counter", "terminal_type"}:
            obs, reward, done, info = value
            if acknowledgement == "counter":
                info["status"]["step"] += 1
            else:
                done = 1
            return obs, reward, done, info
        return value

    rpc.call = call
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    with pytest.raises(PhysicalOutcomeUnknown):
        toolkit.execute_tool("pi0_pick", {"max_chunks": 1})
    with pytest.raises(PhysicalOutcomeUnknown):
        toolkit.execute_tool("pi0_pick", {"max_chunks": 1})
    with pytest.raises(PhysicalOutcomeUnknown):
        toolkit._primitives.env.step(np.zeros(14))
    with pytest.raises(ValueError, match="cannot reset"):
        toolkit._primitives.env.reset()
    assert rpc.actions == 1 and policy.calls == 1
    assert not any(row["event_type"] == "control_step" for row in events(toolkit))
    assert (
        sum(row["event_type"] == "action_outcome_unknown" for row in events(toolkit))
        == 1
    )
    toolkit.close()


def test_move_to_reached_on_final_allowed_action(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=1, terminal_after=99)
    original = rpc.call

    def call(method, *args, **kwargs):
        if method == "env.step":
            rpc.observation["state"]["right_ee_pose"][:3] = [0.1, 0.2, 0.3]
        return original(method, *args, **kwargs)

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    result = toolkit.execute_tool(
        "move_to", {"arm": "right", "xyz": [0.1, 0.2, 0.3], "max_steps": 1}
    )
    assert (
        result.result["capability_result"]["reached"] is True
        and result.result["capability_result"]["steps_used"] == 1
    )
    assert toolkit.control_step_budget()["remaining"] == 0
    assert toolkit.episode_ended() is True
    result = toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
    assert result.result["status"] == "not_executed" and rpc.actions == 1
    assert "error" not in result.result
    toolkit.close()


def test_native_actions_require_active_tool_and_reset_cannot_replay(tmp_path):
    toolkit, rpc, _ = create_simulated_toolkit(tmp_path / "episode")
    env = toolkit._primitives.env
    with pytest.raises(RuntimeError, match="active tool"):
        env.step(np.zeros(14))
    with pytest.raises(RuntimeError, match="individually acknowledged"):
        env.chunk_step([np.zeros(14)])
    toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
    with pytest.raises(ValueError, match="cannot reset"):
        env.reset()
    assert rpc.actions == 1
    json.loads((tmp_path / "episode/execution_trace.jsonl").read_text().splitlines()[0])
    toolkit.close()


def test_scripted_action_metadata_survives_post_action_telemetry(tmp_path):
    toolkit, rpc, _ = create_simulated_toolkit(tmp_path / "episode")
    result = toolkit.execute_tool("set_gripper", {"arm": "left", "gripper": -1})
    assert not result.result.get("error"), result.result
    original = events(toolkit)
    original_bytes = json.dumps(original)
    rows = merge_control_telemetry(original)
    control = next(row for row in rows if row["event_type"] == "control_step")
    assert control["public_evidence"]["action_fields"] == (
        "left_arm_joint_state,right_arm_joint_state,left_ee_joint_state,right_ee_joint_state"
    )
    assert control["physical_features"]["left_ee_pose"]
    assert control["telemetry_event_index"] > control["event_index"]
    public = toolkit._execution_trace.public_control_steps()[0]["public_evidence"]
    assert public == control["public_evidence"]
    assert json.dumps(original) == original_bytes
    assert rpc.actions == 1
    toolkit.close()


def test_observation_failure_after_ack_preserves_committed_action(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    original = rpc.call

    def call(method, *args, **kwargs):
        value = original(method, *args, **kwargs)
        if method == "env.step":
            obs, reward, done, info = value
            malformed = {
                **obs,
                "state": {**obs["state"], "right_arm_joint_state": [0.0] * 5},
            }
            return malformed, reward, done, info
        return value

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    result = toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
    assert result.result["error"]
    assert rpc.actions == toolkit._primitives.env.elapsed_steps() == 1
    assert sum(row["event_type"] == "control_step" for row in events(toolkit)) == 1
    assert not any(
        row["event_type"] == "action_outcome_unknown" for row in events(toolkit)
    )
    with pytest.raises(ValueError, match="cannot reset"):
        toolkit._primitives.env.reset()
    assert toolkit._execution_controller.snapshot.phase == "idle"
    toolkit.close()


def test_physical_budget_truncates_policy_chunk_after_one_ack(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=1, terminal_after=99)
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    result = toolkit.execute_tool("pi0_pick", {"max_chunks": 3})
    assert result.result["error"] is None
    assert rpc.actions == 1 and policy.calls == 1
    assert result.result["capability_result"]["stop_reason"] == "physical_budget"
    assert result.result["capability_result"]["actions_used"] == 1
    assert toolkit.episode_ended() is True and toolkit.solved() is False
    suffix = [
        row for row in events(toolkit) if row["event_type"] == "action_suffix_discarded"
    ]
    assert sum(row["discarded_action_steps"] for row in suffix) == 3
    toolkit.close()


@pytest.mark.parametrize(
    "violation", ["duplicate", "closed_suffix", "unrecorded_forward"]
)
def test_vla_cursor_violation_rejected_before_any_additional_rpc(tmp_path, violation):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    active = toolkit._execution_controller
    active.begin("pi0_pick", {})
    prediction = toolkit._primitives.env.predict_ref(
        toolkit._primitives.env.current_observation_ref,
        toolkit._primitives.env.last_obs["instruction"],
    )
    if violation != "unrecorded_forward":
        active.record_forward(prediction, instruction="SIMULATED cursor rejection")
    cursors = list(prediction.iter_actions())
    submitted = 0
    if violation == "duplicate":
        toolkit._primitives.env.step(cursors[0].action, cursor=cursors[0])
        submitted = 1
    if violation == "closed_suffix":
        active.discard_open_suffix(reason="tool_boundary")
    try:
        with pytest.raises((RuntimeError, ValueError)):
            toolkit._primitives.env.step(cursors[0].action, cursor=cursors[0])
        assert rpc.actions == submitted
        assert toolkit._primitives.env.elapsed_steps() == submitted
        assert (
            sum(row["event_type"] == "control_step" for row in events(toolkit))
            == submitted
        )
    finally:
        active.discard_open_suffix(reason="tool_boundary")
        active.finish("failed")
        toolkit.close()


def test_foreign_execution_cursor_rejected_before_rpc(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    active = toolkit._execution_controller
    active.begin("pi0_pick", {"prompt": "SIMULATED previous execution"})
    prediction = toolkit._primitives.env.predict_ref(
        toolkit._primitives.env.current_observation_ref,
        toolkit._primitives.env.last_obs["instruction"],
    )
    active.record_forward(prediction, instruction="SIMULATED previous execution")
    cursor = next(prediction.iter_actions())
    active.discard_open_suffix(reason="tool_boundary")
    active.finish("completed")
    active.begin("pi0_pick", {"prompt": "SIMULATED current execution"})
    try:
        with pytest.raises(ValueError, match="another tool execution"):
            toolkit._primitives.env.step(cursor.action, cursor=cursor)
        assert rpc.actions == toolkit._primitives.env.elapsed_steps() == 0
        assert not any(row["event_type"] == "control_step" for row in events(toolkit))
    finally:
        active.finish("failed")
        toolkit.close()
