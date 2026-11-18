# Modified for the ReWeave namespace.
"""A cancelled caller must retain the operation's eventual result or failure."""

import asyncio
import json
import threading

import pytest

from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from reweave.tools.toolkit import ToolResult
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_bundle_runtime import Robot


@pytest.mark.parametrize("failure", [None, RuntimeError, PhysicalOutcomeUnknown])
def test_cancelled_gateway_retains_settlement_and_identity(tmp_path, failure):
    robot = Robot(tmp_path)
    robot.add_tool(
        "move",
        {"name": "move", "description": "move", "input_schema": {"type": "object"}},
        lambda: {},
    )
    started, release = threading.Event(), threading.Event()

    def execute(name, arguments):
        started.set()
        assert release.wait(5)
        if failure:
            raise failure("late operation failure")
        return ToolResult(name=name, result={"confirmed_steps": 3})

    robot.execute_tool = execute
    runtime = BundleRuntime(
        baseline(),
        robot,
        "test",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )

    async def exercise():
        task = asyncio.create_task(runtime.kernel.invoke("move", {}, epoch=0))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(
            PhysicalOutcomeUnknown
            if failure is PhysicalOutcomeUnknown
            else asyncio.CancelledError
        ):
            await task

    asyncio.run(exercise())
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    settled = [event for event in events if event["kind"] == "operation_settled"]
    assert len(settled) == 1
    assert settled[0]["cancellation_requested"]
    assert settled[0]["action_id"] == next(
        event["action_id"] for event in events if event["kind"] == "action_submitted"
    )
    if failure:
        assert settled[0]["error"]["type"] == failure.__name__
        assert "late operation failure" in settled[0]["error"]["traceback"]
    else:
        assert settled[0]["result"]["confirmed_steps"] == 3
        assert runtime.observation["confirmed_steps"] == 3
    assert not runtime._submitted
