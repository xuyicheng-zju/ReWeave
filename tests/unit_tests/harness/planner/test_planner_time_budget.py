# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Native turn cancellation closes its original session without replay."""

import asyncio

import pytest

from reweave.planner.base import ACTION_OUTCOME_UNKNOWN
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline


def test_native_deadline_cancels_and_drains_submitted_decision(tmp_path):
    sequence = []

    async def decide(messages, info):
        sequence.append("submitted")
        try:
            await asyncio.sleep(30)
        finally:
            sequence.append("drained")

    planner = ScriptedNativePlanner(decide)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=2
        ),
    )
    result = planner.solve(
        system_prompt="Inspect",
        user_message="Task",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.error and "deadline" in result.error.lower()
    assert sequence == ["submitted", "drained"]
    assert len(planner.scripted_sessions) == 1
    assert result.stats["visible_turns"] == 1


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_native_unknown_outcome_survives_session_cleanup(tmp_path, cleanup_error):
    robot = Robot(tmp_path)

    def unknown():
        raise PhysicalOutcomeUnknown("submitted action lost acknowledgement")

    robot.add_tool(
        "unknown",
        {
            "name": "unknown",
            "description": "Submit synthetic action",
            "input_schema": {"type": "object", "properties": {}},
        },
        unknown,
    )
    planner = ScriptedNativePlanner(
        lambda messages, info: {
            "text": "Act",
            "tool_calls": [{"name": "unknown", "arguments_json": "{}"}],
        }
    )
    original = planner.session_factory

    def factory(directory):
        session = original(directory)
        if cleanup_error:

            async def fail_close():
                raise RuntimeError("secondary cleanup failure")

            session.close = fail_close
        return session

    planner.session_factory = factory
    result = planner.solve(
        system_prompt="Act", user_message="Task", toolkit=robot, max_turns=3
    )
    assert result.termination_class == ACTION_OUTCOME_UNKNOWN
    assert "lost acknowledgement" in result.error
    assert result.api_error_retryable is None
    assert len(planner.scripted_sessions) == 1
