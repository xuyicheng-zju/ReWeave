# Copyright (c) 2026 RPent Contributors
"""Current native timeout settings and unknown outcomes remain authoritative."""

import pytest

from reweave.planner.base import ACTION_OUTCOME_UNKNOWN, API_ERROR, build_planner
from reweave.planner.providers import NativeTurnFailed
from reweave.runtime.capability import EnvironmentServiceFailure, PhysicalOutcomeUnknown
from tests.unit_tests.harness.planner.test_native_episode_end import (
    TerminalRobot,
    TerminalSession,
)


@pytest.mark.parametrize("timeout", [0, -1, None])
def test_disabled_timeout_keeps_the_existing_native_session(tmp_path, timeout):
    robot = TerminalRobot(tmp_path, "terminated")
    session = TerminalSession(tmp_path)
    planner = build_planner("native", output_dir=tmp_path, planner_timeout_s=timeout)
    assert planner.timeout_s is None
    planner.session_factory = lambda _: session
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=robot, max_turns=3
    )
    assert result.error is None, result.error
    assert len(session.requests) == robot.actions == 1 and session.closed


@pytest.mark.parametrize("cleanup", ["normal", "provider_error", "transport_error"])
def test_cleanup_cannot_relabel_a_latched_unknown_action(tmp_path, cleanup):
    robot = TerminalRobot(tmp_path, "terminated", unknown=True)

    class Session(TerminalSession):
        async def close(self):
            self.closed = True
            if cleanup == "provider_error":
                raise NativeTurnFailed("failed", {"message": "provider close failed"})
            if cleanup == "transport_error":
                raise RuntimeError("transport close failed")

    session = Session(tmp_path)
    planner = build_planner("native", output_dir=tmp_path, planner_timeout_s=0)
    planner.session_factory = lambda _: session
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=robot, max_turns=3
    )
    assert result.termination_class == ACTION_OUTCOME_UNKNOWN
    assert result.error == str(robot.error) and result.api_error_retryable is None
    assert robot.physical_outcome_error is robot.error and session.closed
    with pytest.raises(PhysicalOutcomeUnknown) as blocked:
        robot.execute_tool("end_episode", {})
    assert blocked.value is robot.error and robot.actions == len(session.requests) == 1


@pytest.mark.parametrize("retryable", [False, True])
def test_environment_failure_reaches_native_infrastructure_result(tmp_path, retryable):
    robot = TerminalRobot(tmp_path, "terminated")
    failure = EnvironmentServiceFailure("no action submitted", retryable=retryable)
    attempts = []

    def fail():
        attempts.append("not_submitted")
        raise failure

    spec = next(
        spec for spec in robot.get_tools_spec() if spec["name"] == "end_episode"
    )
    robot.add_tool("end_episode", spec, fail)
    session = TerminalSession(tmp_path)
    planner = build_planner("native", output_dir=tmp_path, planner_timeout_s=0)
    planner.session_factory = lambda _: session
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=robot, max_turns=3
    )
    assert result.termination_class == API_ERROR, result.error
    assert result.api_error_retryable is retryable
    assert result.error == str(failure)
    assert robot.actions == 0 and attempts == ["not_submitted"]
    assert len(session.requests) == 1 and session.closed
    with pytest.raises(EnvironmentServiceFailure) as blocked:
        robot.execute_tool("end_episode", {})
    assert blocked.value is failure
