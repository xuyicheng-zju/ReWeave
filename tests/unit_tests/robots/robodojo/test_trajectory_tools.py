"""Every native sample is dispatched once; trace-sized arrays stay outside feedback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from reweave.benchmarks.robodojo import tools
from reweave.runtime.capability import PhysicalOutcomeUnknown


def trajectory_environment():
    packets = []
    env = SimpleNamespace(
        last_obs={}, terminated=False, truncated=False, steps_remaining=lambda: 100
    )

    def step(arm, positions, velocities, *, gripper):
        packets.append((arm, positions, velocities, gripper))
        return {}, 0.0, False, {}

    env.step_trajectory = step
    return env, packets


def path(length=23):
    positions = np.arange(length * 6, dtype=float).reshape(length, 6) / 1000
    velocities = -positions / 2
    return {
        "status": "Success",
        "planner_dt_s": 0.004,
        "control_period_s": 0.04,
        "joint_positions": positions.tolist(),
        "joint_velocities": velocities.tolist(),
    }


def test_all_position_velocity_rows_survive_windowing_with_no_padding_or_terminal_replay():
    env, packets = trajectory_environment()
    plan = path()
    primitives = SimpleNamespace(env=env, _check_cancelled=lambda: None)
    feedback = tools._execute_native_path(
        primitives, "left", plan, max_steps=20, gripper=-1
    )
    assert [len(packet[1]) for packet in packets] == [10, 10, 3]
    assert [row for packet in packets for row in packet[1]] == plan["joint_positions"]
    assert [row for packet in packets for row in packet[2]] == plan["joint_velocities"]
    assert all(packet[0] == "left" and packet[3] == -1 for packet in packets)
    assert feedback["steps_used"] == feedback["planned_actions"] == 3
    assert feedback["terminal_target_joints"] == plan["joint_positions"][-1]
    assert "joint_positions" not in feedback and "joint_velocities" not in feedback
    assert (
        "raw_planner_result" not in feedback and "raw_planner_results" not in feedback
    )
    assert len(json.dumps(feedback)) < 1500


@pytest.mark.parametrize(
    "kind,known", [(None, False), ("INVALID_START_STATE_JOINT_LIMITS", True)]
)
def test_native_failure_feedback_keeps_official_scalar_status_and_excludes_raw_details(
    kind, known
):
    env, packets = trajectory_environment()
    plan = {
        "status": "Fail",
        "error": "native planning failed",
        "planner_status": kind,
        "reason_known": known,
        "raw_planner_result": {"huge": "x" * 250000},
    }
    feedback = tools._execute_native_path(
        SimpleNamespace(env=env, _check_cancelled=lambda: None),
        "right",
        plan,
        max_steps=200,
        gripper=0,
    )
    assert feedback["planner_status"] == kind
    assert feedback["reason_known"] is known
    assert feedback["ik_error"] == "native planning failed"
    assert feedback["steps_used"] == 0 and not packets
    assert len(json.dumps(feedback)) < 1000
    assert "raw_planner_result" not in feedback


def test_unknown_trajectory_ack_propagates_without_another_window_or_replay():
    env, packets = trajectory_environment()
    original = env.step_trajectory
    failure = PhysicalOutcomeUnknown("trajectory outcome unknown")

    def unknown(*args, **kwargs):
        original(*args, **kwargs)
        raise failure

    env.step_trajectory = unknown
    with pytest.raises(PhysicalOutcomeUnknown) as error:
        tools._execute_native_path(
            SimpleNamespace(env=env, _check_cancelled=lambda: None),
            "right",
            path(),
            max_steps=200,
            gripper=0,
        )
    assert error.value is failure
    assert len(packets) == 1


def test_cancelled_window_prevents_subsequent_trajectory_dispatch():
    env, packets = trajectory_environment()

    def cancelled():
        if packets:
            raise RuntimeError("cancelled by test owner")

    with pytest.raises(RuntimeError, match="cancelled by test owner"):
        tools._execute_native_path(
            SimpleNamespace(env=env, _check_cancelled=cancelled),
            "right",
            path(),
            max_steps=200,
            gripper=0,
        )
    assert len(packets) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_velocities", [[0] * 5]),
        ("joint_positions", [[float("nan")] * 6]),
        ("planner_dt_s", 0),
    ],
)
def test_malformed_trajectory_submits_no_window(field, value):
    env, packets = trajectory_environment()
    plan = {**path(), field: value}
    with pytest.raises(ValueError, match="invalid path"):
        tools._execute_native_path(
            SimpleNamespace(env=env, _check_cancelled=lambda: None),
            "right",
            plan,
            max_steps=200,
            gripper=0,
        )
    assert not packets
