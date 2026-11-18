# Modified for the ReWeave namespace.
"""Plan-confirmed Runtime calls survive complete episode archiving."""

import asyncio
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.evolution.evidence.records import archive_episode, inventory_episode
from reweave.evolution.evidence.trace import (
    build_episode_context,
    query_trace,
    resolve_trace_reference,
)
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import PhysicalRobot, RecordingSink, ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import baseline


def test_plan_confirmed_loop_calls_keep_planner_and_program_provenance(tmp_path):
    source = tmp_path / "episode"
    source.mkdir()
    robot = PhysicalRobot(source)
    planned = [
        "move_once",
        "inspect_scene",
        "move_once",
        "inspect_scene",
        "read_terminal_state",
    ]
    requests = []

    def model(messages, _info):
        index = len(requests)
        requests.append(messages)
        name = planned[index]
        arguments = {}
        return ModelResponse(
            parts=[ToolCallPart(name, arguments, tool_call_id=f"planned_{index}")]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=RecordingSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=30
        ),
    )
    result = asyncio.run(
        planner._solve(
            system_prompt="Exercise program feedback.",
            user_message="Inspect after moving.",
            toolkit=robot,
            max_turns=6,
        )
    )
    assert result.error is None
    assert robot.moves == robot.calls == 2
    messages = [m for m in result.messages if m["role"] == "tool"]
    assert len(requests) == len(messages) == 5
    assert all(m["tool_call_id"] for m in messages)
    assert all(m["provider_tool_call_id"] for m in messages)
    # The tiny actuator has no simulator trace writer; these are genuinely zero
    # simulator control records, while its actual kernel action events are retained.
    (source / "execution_trace.jsonl").write_text("")
    (source / "transcript_program.json").write_text(
        json.dumps(
            {
                "messages": result.messages,
                "official_success": False,
                "termination_class": "physical_failure",
            }
        )
    )
    (source / "harness_context/capability_catalog.json").write_text(
        json.dumps(robot.get_tools_contract_spec())
    )
    before = inventory_episode(source)
    workspace = tmp_path / "author"
    evidence = archive_episode(source, workspace / ".archive/episodes")
    build_episode_context(workspace, evidence)
    timeline = query_trace(
        workspace, evidence_id=evidence.evidence_id, view="timeline"
    )["data"]
    program = timeline["planner_calls"]
    assert len(program) == 5
    assert [
        resolve_trace_reference(workspace, row["feedback"]) for row in program
    ] == messages
    assert all(row["program_call_id"] for row in program)
    assert len({row["feedback"]["pointer"] for row in program}) == 5
    assert timeline["program_feedback"] == []
    assert inventory_episode(source) == before
    evidence.verify(workspace / ".archive/episodes" / evidence.evidence_id)
