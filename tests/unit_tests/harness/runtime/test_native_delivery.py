"""A shared native session receives settled feedback once across program calls."""

import asyncio
import json

import pytest

from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.execution import ExecutionContext, ProgramExecution
from reweave.runtime.kernel import EpisodeLimits
from tests.unit_tests.harness.runtime.test_adaptation import (
    Robot,
    controller,
    executable_bundle,
    proposal,
)


def completed():
    return {
        "message": {"kind": "response", "parts": []},
        "text": "CPU-only completed decision",
        "tool_calls": [],
    }


def history(marker):
    return {
        "operation": "inspect_scene",
        "status": "completed",
        "result": {"marker": marker},
    }


def setup(tmp_path, request, *, native=True, physical=False):
    initial = executable_bundle()
    manager = controller(tmp_path / "sources", initial)
    robot = Robot(tmp_path / "episode")
    if physical:

        def motion():
            robot.calls += 1
            return {"position": robot.calls}

        robot.add_tool(
            "physical_motion",
            {
                "name": "physical_motion",
                "description": "CPU fixture physical motion",
                "input_schema": {"type": "object", "properties": {}},
            },
            motion,
        )
    bridge = BundleRuntime(
        initial,
        robot,
        "Inspect current evidence",
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=60
        ),
        adaptation=manager if native else None,
    )
    execution = ProgramExecution(bridge, request, "Fixed task instructions")
    bridge.execution = execution
    root = ExecutionContext(
        "root",
        "main",
        "Inspect the public scene",
        frozenset(spec["name"] for spec in bridge.specs()) | {"model:plan"},
    )
    execution.root = root
    execution.sessions[root.identity] = root
    return robot, bridge, execution, root


def marker_count(context, marker):
    return json.dumps(context["messages"]).count(marker)


def test_root_workflow_root_share_one_settled_history_delivery(tmp_path):
    seen = []

    async def request(context, call_id):
        seen.append((context, call_id))
        return completed()

    robot, bridge, execution, root = setup(tmp_path, request)
    originals = [history("parent-evidence"), history("child-evidence")]

    async def run():
        token = execution.current.set(root)
        try:
            bridge.history.append(originals[0])
            await execution.plan()
            async with execution.workflow_scope("probe"):
                await execution.plan()
                bridge.history.append(originals[1])
                await execution.plan()
            await execution.plan()
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert [marker_count(context, "parent-evidence") for context, _ in seen] == [
        1,
        0,
        0,
        0,
    ]
    assert [marker_count(context, "child-evidence") for context, _ in seen] == [
        0,
        0,
        1,
        0,
    ]
    assert seen[0][1] == seen[3][1] == root.identity
    assert seen[1][1] == seen[2][1] != root.identity
    assert bridge.history == originals and robot.calls == 0
    assert bridge.kernel is execution.kernel


def test_concurrent_native_plans_select_feedback_after_previous_delivery(tmp_path):
    seen = []

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def request(context, _):
            seen.append(context)
            if len(seen) == 1:
                entered.set()
                await release.wait()
            return completed()

        robot, bridge, execution, root = setup(tmp_path, request)
        token = execution.current.set(root)
        try:
            bridge.history.append(history("first-evidence"))
            first = asyncio.create_task(execution.plan())
            await entered.wait()
            # A different operation may settle while the first decision is in
            # flight. It belongs to the next prepared native turn.
            bridge.history.append(history("later-evidence"))

            async def child():
                async with execution.workflow_scope("probe"):
                    await execution.plan()

            second = asyncio.create_task(child())
            await asyncio.sleep(0)
            assert len(seen) == 1
            events = [
                json.loads(line) for line in bridge._trace.read_text().splitlines()
            ]
            assert sum(event["kind"] == "context_built" for event in events) == 1
            release.set()
            await asyncio.gather(first, second)
            assert robot.calls == 0
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert [marker_count(context, "first-evidence") for context in seen] == [1, 0]
    assert [marker_count(context, "later-evidence") for context in seen] == [0, 1]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_delivery_retains_feedback_without_duplicating_preparation(
    tmp_path, failure
):
    seen = []

    async def request(context, _):
        seen.append(context)
        if len(seen) == 1:
            raise failure("CPU delivery did not complete")
        return completed()

    robot, bridge, execution, root = setup(tmp_path, request)

    async def run():
        token = execution.current.set(root)
        try:
            bridge.history.append(history("undelivered-evidence"))
            with pytest.raises(failure):
                await execution.plan()
            assert execution._native_history_position == 0
            assert root.delivered_message_position == 0
            await execution.plan()
            async with execution.workflow_scope("probe"):
                await execution.plan()
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert [marker_count(context, "undelivered-evidence") for context in seen] == [
        1,
        1,
        0,
    ]
    assert len(bridge.history) == 1 and robot.calls == 0


def test_sibling_delivery_consumes_pending_fixed_tool_ack_once(tmp_path):
    seen = []

    async def request(context, _):
        seen.append(context)
        return proposal(context, 0, "inspect_scene") if len(seen) == 1 else completed()

    robot, bridge, execution, root = setup(tmp_path, request)

    async def run():
        token = execution.current.set(root)
        try:
            decision = await execution.plan()
            await execution.execute(decision["tool_calls"][0])
            assert root.feedback_history_positions
            async with execution.workflow_scope("probe"):
                await execution.plan()
            await execution.plan()
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert "position" in json.dumps(seen[1]["messages"])
    assert "position" not in json.dumps(seen[2]["messages"])
    assert robot.calls == 1 and len(bridge.history) == 1
    assert any(
        part["part_kind"] == "tool-return"
        for message in root.messages
        for part in message["parts"]
    )


def test_lock_rejected_action_keeps_feedback_without_claiming_previous_ack(tmp_path):
    seen = []

    async def request(context, _):
        seen.append(context)
        return (
            proposal(context, 0, "physical_motion") if len(seen) == 1 else completed()
        )

    robot, bridge, execution, root = setup(tmp_path, request, physical=True)
    assert bridge.kernel._operations["physical_motion"]["effect"] == "physical"
    assert bridge.kernel._episode_ended is not None

    async def run():
        token = execution.current.set(root)
        sentinel = bridge.settled_operation.set(0)
        try:
            bridge.history.append(history("previous-completed-operation"))
            decision = await execution.plan()
            await bridge.kernel._dispatch.acquire()
            action = asyncio.create_task(execution.execute(decision["tool_calls"][0]))
            await asyncio.sleep(0)
            robot.terminal = True
            bridge.kernel._dispatch.release()
            outcome = await action
            assert outcome["status"] == "not_executed"
            assert root.feedback_history_positions == {}
            assert bridge.settled_operation.get() == 0
            await execution.plan(request_kind="note_closeout", tools=[])
        finally:
            bridge.settled_operation.reset(sentinel)
            execution.current.reset(token)

    asyncio.run(run())
    assert "not_executed" in json.dumps(seen[1]["messages"])
    assert "previous-completed-operation" not in json.dumps(seen[1]["messages"])
    assert len(bridge.history) == 1 and robot.calls == 0


def test_api_programs_keep_independent_full_conversations(tmp_path):
    seen = []

    async def request(context, call_id):
        seen.append((context, call_id))
        return completed()

    robot, bridge, execution, root = setup(tmp_path, request, native=False)

    async def run():
        token = execution.current.set(root)
        try:
            bridge.history.append(history("independent-api-evidence"))
            await execution.plan()
            async with execution.workflow_scope("probe"):
                await execution.plan()
            await execution.plan()
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert [
        marker_count(context, "independent-api-evidence") for context, _ in seen
    ] == [1, 1, 1]
    assert robot.calls == 0
