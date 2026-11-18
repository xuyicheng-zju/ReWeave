# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Real plugin planning over synthetic tools; never a physical robot rollout."""

import json

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.planner.base import AGENT_EXECUTION_FAILURE
from reweave.runtime.kernel import EpisodeLimits
from reweave.tools.toolkit import readonly
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_bundle_runtime import (
    Robot,
    terminal_response,
)
from tests.unit_tests.harness.evolution.test_plugin_execution import run


@pytest.mark.parametrize(
    "scenario,count,executed",
    [
        ("normal", 1, 1),
        ("normal", 5, 5),
        ("error", 5, 2),
        ("cancel", 5, 2),
        ("residual", 5, 5),
        ("structured_status", 5, 5),
        ("over_limit", 6, 0),
        ("finish_order", 2, 0),
    ],
)
def test_plugin_decision_batches_replan_only_on_explicit_failure(
    tmp_path,
    scenario,
    count,
    executed,
):
    robot = Robot(tmp_path)
    seen = []

    @readonly
    def inspect():
        robot.calls += 1
        if robot.calls == 2:
            if scenario == "error":
                return {"error": "synthetic tool failure"}
            if scenario == "cancel":
                return {"interrupted": True, "active_execution_outcome": "cancelled"}
            if scenario == "residual":
                return {
                    "target_reached": False,
                    "tracking": {"max_joint_error_rad": 0.8},
                }
            if scenario == "structured_status":
                return {"status": {"step": 431, "step_limit": 1100}}
        return {"position": robot.calls}

    spec, _ = robot._tools["inspect_scene"]
    robot.add_tool("inspect_scene", spec, inspect)

    def model(messages, info):
        seen.append(messages)
        assert "1–5 tool calls" in info.instructions
        if len(seen) == 1:
            names = ["inspect_scene"] * count
            if scenario == "finish_order":
                names[0] = "finish"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name,
                        {"status": "success", "summary": "invalid order"}
                        if name == "finish"
                        else {},
                        tool_call_id=f"call-{i}",
                    )
                    for i, name in enumerate(names)
                ]
            )
        assert robot.calls == executed
        if scenario in {"error", "cancel", "over_limit", "finish_order"}:
            assert "not_executed" in str(messages)
        return terminal_response()

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=100,
            gateway_calls=100,
            wall_time_s=30,
        ),
    )
    result = planner.solve(
        system_prompt="Test serial decisions.",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    assert len(seen) == 2 and robot.calls == executed
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    submitted = [
        event
        for event in events
        if event["kind"] == "action_submitted" and event["operation"] == "inspect_scene"
    ]
    assert len(submitted) == executed
    if scenario in {"over_limit", "finish_order"}:
        rejected = [event for event in events if event["kind"] == "model_call_rejected"]
        assert len(rejected) == count


def test_fixed_runtime_rejects_reordered_calls_from_replacement_loop(tmp_path):
    robot = Robot(tmp_path)

    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart("inspect_scene", {}, tool_call_id=f"call-{i}")
                for i in range(2)
            ]
        )

    result = run(
        tmp_path,
        """def run(runtime, context):
    decision = runtime.plan()
    for call in reversed(decision["tool_calls"]):
        runtime.execute(call)
    return {"completed": True}
""",
        model,
        robot=robot,
        terminal=False,
    )

    assert result.termination_class == AGENT_EXECUTION_FAILURE
    assert "order returned by runtime.plan()" in result.error
    assert robot.calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        {"error": "synthetic tool failure"},
        {"interrupted": True},
        {"active_execution_outcome": "cancelled"},
        {"status": "failed"},
    ],
)
@pytest.mark.parametrize("execute_remainder", [False, True])
def test_fixed_runtime_settles_failed_batch_before_replanning(
    tmp_path, failure, execute_remainder
):
    robot = Robot(tmp_path)
    seen = []

    @readonly
    def inspect():
        robot.calls += 1
        return failure

    spec, _ = robot._tools["inspect_scene"]
    robot.add_tool("inspect_scene", spec, inspect)

    def model(messages, info):
        seen.append(messages)
        if len(seen) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("inspect_scene", {}, tool_call_id=f"call-{i}")
                    for i in range(3)
                ]
            )
        assert robot.calls == 1
        assert str(messages).count("not_executed") == 2
        return terminal_response()

    result = run(
        tmp_path,
        """def run(runtime, context):
    decision = runtime.plan()
    calls = decision["tool_calls"]
    runtime.execute(calls[0])
"""
        + (
            "    for call in calls[1:]:\n        runtime.execute(call)\n"
            if execute_remainder
            else ""
        )
        + """    decision = runtime.plan()
    runtime.execute(decision["tool_calls"][0])
    return {"completed": True}
""",
        model,
        robot=robot,
        terminal=False,
    )

    assert result.error is None, result.error
    assert len(seen) == 2 and robot.calls == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    skipped = [event for event in events if event["kind"] == "model_call_not_executed"]
    assert len(skipped) == 2
    assert all(event["execution_performed"] is False for event in skipped)


@pytest.mark.parametrize("invalid", ["arguments", "tool"])
def test_invalid_first_call_blocks_legal_remainder_for_replacement_loop(
    tmp_path, invalid
):
    robot = Robot(tmp_path)
    seen = []

    @readonly
    def inspect(mandatory):
        robot.calls += 1
        return {"observed": mandatory}

    robot.add_tool(
        "inspect_scene",
        {
            "name": "inspect_scene",
            "description": "Inspect a synthetic scene.",
            "input_schema": {
                "type": "object",
                "properties": {"mandatory": {"type": "string"}},
                "required": ["mandatory"],
            },
        },
        inspect,
    )

    def model(messages, info):
        seen.append(messages)
        if len(seen) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "inspect_scene" if invalid == "arguments" else "unknown_tool",
                        {},
                        tool_call_id="invalid-call",
                    ),
                    ToolCallPart(
                        "inspect_scene",
                        {"mandatory": "valid"},
                        tool_call_id="valid-call",
                    ),
                ]
            )
        assert "not_executed" in str(messages)
        return terminal_response()

    result = run(
        tmp_path,
        """def run(runtime, context):
    for call in runtime.plan()["tool_calls"]:
        runtime.execute(call)
    runtime.execute(runtime.plan()["tool_calls"][0])
    return {"completed": True}
""",
        model,
        robot=robot,
        terminal=False,
    )

    assert result.error is None, result.error
    assert len(seen) == 2
    assert robot.calls == 0
