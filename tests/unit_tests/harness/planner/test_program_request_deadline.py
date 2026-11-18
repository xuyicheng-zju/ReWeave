# Copyright (c) 2026 RPent Contributors
"""Native model/program failures retain settled actions and actual turn costs."""

import asyncio

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.planner.base import AGENT_EXECUTION_FAILURE
from reweave.planner.providers import NativeTurnFailed
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import PhysicalRobot, ScriptedNativePlanner, rows
from tests.unit_tests.harness.evolution.test_author_interaction import candidate


@pytest.mark.parametrize("mode", ["provider_refusal", "short_program", "cpu_deadline"])
def test_native_nested_failure_keeps_acknowledged_action_and_turn_cost(tmp_path, mode):
    body = (
        "    while True:\n        pass\n"
        if mode == "cpu_deadline"
        else '    runtime.plan("Inspect current evidence")\n'
    )
    source = (
        "def run(runtime, context):\n"
        + body
        + '    return {"status": "completed", "summary": "checked", "output": {}}\n'
    )
    loop = 'def run(runtime, context):\n    runtime.execute(runtime.plan()["tool_calls"][0])\n    runtime.execute(runtime.plan()["tool_calls"][0])\n    return {}\n'
    value = candidate(source, loop=loop).model_dump(mode="json")
    item = value["workflows"][0]
    item["permissions"], item["observations"] = ["model:plan"], []
    item["limits"].update(model_calls=10, runtime_calls=20, wall_time_s=4.0)
    robot = PhysicalRobot(tmp_path)
    submitted = []

    async def decision(messages, info):
        submitted.append(messages)
        if len(submitted) == 1:
            return {
                "text": "Move once",
                "tool_calls": [{"name": "move_once", "arguments_json": "{}"}],
            }
        if len(submitted) == 2:
            return {
                "text": "Inspect in workflow",
                "tool_calls": [
                    {"name": "workflow_probe", "arguments_json": '{"arguments":{}}'}
                ],
            }
        if mode == "provider_refusal":
            raise NativeTurnFailed(
                "failed", {"message": "native provider refused input"}
            )
        await asyncio.sleep(20)
        pytest.fail("The program deadline must cancel this turn")

    planner = ScriptedNativePlanner(decision)
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=20
        ),
    )
    result = planner.solve(
        system_prompt="Inspect",
        user_message="Move once, then inspect.",
        toolkit=robot,
        max_turns=6,
    )
    assert robot.moves == 1
    if mode == "provider_refusal":
        assert result.termination_class == "api_error"
        assert result.api_error_retryable is True
        assert not (tmp_path / "native/recovery/blocked.json").exists()
        assert "native provider refused input" in result.error
    else:
        assert result.termination_class == AGENT_EXECUTION_FAILURE

    events = rows(tmp_path / "harness_context/bundle_events.jsonl")
    assert (
        len(
            [
                e
                for e in events
                if e["kind"] == "action_completed" and e["operation"] == "move_once"
            ]
        )
        == 1
    )
    ledger = rows(tmp_path / "native/native_turns.jsonl")
    assert len(ledger) == len(submitted) == (2 if mode == "cpu_deadline" else 3)
    assert all(row["wall_time_s"] >= 0 for row in ledger)
