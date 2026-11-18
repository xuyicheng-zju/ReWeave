# Modified for the ReWeave namespace.
"""Real failure shapes, explicit selection and end-to-end candidate input ownership."""

import base64
import copy
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from reweave.evolution.bundle import AgentBundle, RequestLimits
from reweave.planner.base import AGENT_EXECUTION_FAILURE
from reweave.runtime.capability import CandidateExecutionFailure
from reweave.runtime.kernel import EpisodeLimits
from reweave.runtime.model_input import image_inventory, select_images, validate_request
from tests.native_fixture import PhysicalRobot, ScriptedNativePlanner, rows
from tests.unit_tests.harness.evolution.test_author_interaction import candidate


def recorded_shape(condition):
    """Keep original nesting and duplicate identities; substitute synthetic pixels."""
    record = json.loads(
        (Path(__file__).parent / "data/request_image_shapes.json").read_text()
    )[condition]
    identities = {}
    state = {}
    for image in record["state_images"]:
        identity = image["image_id"]
        if identity not in identities:
            stream = io.BytesIO()
            Image.new("RGB", (2, 2), (len(identities), 20, 30)).save(
                stream, format="PNG"
            )
            identities[identity] = base64.b64encode(stream.getvalue()).decode()
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in image["path"].split("/")[1:]
        ]
        container = state
        for index, part in enumerate(parts):
            key = int(part) if isinstance(container, list) else part
            if isinstance(container, list):
                while len(container) <= key:
                    container.append(None)
            if index == len(parts) - 1:
                container[key] = {
                    "name": identity + ".png",
                    "media_type": "image/png",
                    "data_base64": identities[identity],
                }
            else:
                if (
                    container[key]
                    if isinstance(container, list)
                    else container.get(key)
                ) is None:
                    container[key] = [] if parts[index + 1].isdigit() else {}
                container = container[key]
    return state, record


@pytest.mark.parametrize("condition,count,unique", [("002", 168, 87), ("003", 84, 42)])
def test_original_recursive_shape_exceeds_budget_even_when_one_shape_deduplicates_below_it(
    tmp_path, condition, count, unique
):
    state, record = recorded_shape(condition)
    state["image_evidence"] = "Existing caller evidence must not be overwritten."
    original = copy.deepcopy(state)
    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter,
        ModelRequest,
        UserPromptPart,
    )

    from reweave.planner.bundle_runtime import BundleRuntime
    from reweave.planner.model_messages import content_blocks_to_pydantic
    from tests.native_robot import Robot
    from tests.unit_tests.harness.evolution.fixtures import baseline

    runtime = BundleRuntime(
        baseline(),
        Robot(tmp_path),
        "Compare evidence",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    converted = runtime._tool_result("recorded_input", state)
    text, images = content_blocks_to_pydantic(converted.content_blocks)
    messages = ModelMessagesTypeAdapter.dump_python(
        [ModelRequest([UserPromptPart([text, *images])])], mode="json"
    )
    inventory = image_inventory(messages)
    assert len(inventory) == count
    assert len({image.data for image in images}) == unique
    assert {item["source_pointer"] for item in json.loads(text)["image_inputs"]} == {
        item["path"] for item in record["state_images"]
    }
    with pytest.raises(CandidateExecutionFailure, match="images budget"):
        validate_request(
            {"images": count, "input_bytes": 1}, RequestLimits(max_images=50)
        )
    paths = [inventory[0]["path"], inventory[-1]["path"]]
    selected, audit = select_images(
        messages,
        {"paths": paths, "reason": "Compare recorded acquisition with current view."},
    )
    assert len(image_inventory(selected)) == 2
    assert audit["source_images"] == count and audit["selected_images"] == 2
    assert len(audit["images"]) == count and state == original
    assert original["image_evidence"] in selected[0]["parts"][0]["content"][0]
    assert "omitted" not in json.dumps(original)


def test_deployment_image_and_byte_budgets_reject_oversized_requests():
    state, _ = recorded_shape("003")
    with pytest.raises(CandidateExecutionFailure, match="missing evidence"):
        select_images(state, {"paths": ["/absent"], "reason": "needed"})
    validate_request({"images": 50, "input_bytes": 1000}, RequestLimits(max_images=50))
    with pytest.raises(CandidateExecutionFailure, match="images budget"):
        validate_request(
            {"images": 51, "input_bytes": 1000}, RequestLimits(max_images=50)
        )
    validate_request(
        {"images": 168, "input_bytes": 1000}, RequestLimits(max_images=None)
    )
    with pytest.raises(CandidateExecutionFailure, match="input_bytes budget"):
        validate_request(
            {"images": 1, "input_bytes": 1001}, RequestLimits(max_input_bytes=1000)
        )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("selection", [False, True])
def test_root_and_nested_plan_share_C_selection_and_reject_original_shape_without_action_replay(
    tmp_path, nested, selection
):
    state, _ = recorded_shape("003")
    from reweave.tools.toolkit import readonly
    from tests.unit_tests.harness.evolution.test_plugin_execution import program

    class EvidenceRobot(PhysicalRobot):
        def __init__(self, path):
            super().__init__(path)
            self.add_tool(
                "read_evidence",
                {
                    "name": "read_evidence",
                    "description": "Recorded-shape fixture",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                self.read_evidence,
            )

        @readonly
        def read_evidence(self):
            return state

    inquiry = '    runtime.observe("read_evidence", {})\n    runtime.plan("Compare acquisition and current evidence")\n'
    workflow_source = (
        "def run(runtime, context):\n"
        + inquiry
        + '    return {"status": "completed", "summary": "checked", "output": {}}\n'
    )
    loop_source = 'def run(runtime, context):\n    runtime.execute(runtime.plan()["tool_calls"][0])\n'
    loop_source += (
        '    runtime.execute(runtime.plan()["tool_calls"][0])\n' if nested else inquiry
    )
    loop_source += '    runtime.execute(runtime.plan("Decide completion")["tool_calls"][0])\n    return {}\n'
    value = candidate(workflow_source, loop=loop_source).model_dump(mode="json")
    value["workflows"][0]["permissions"] = ["model:plan", "read_evidence"]
    value["workflows"][0]["observations"] = []
    value["workflows"][0]["limits"]["model_calls"] = 2
    if selection:
        value["context"]["program"] = program(
            """def build(runtime, context):
    messages = context["messages"]
    paths = ["/" + str(mi) + "/parts/" + str(pi) + "/content/" + str(ci)
             for mi, message in enumerate(messages)
             for pi, part in enumerate(message["parts"])
             if isinstance(part.get("content"), list)
             for ci, item in enumerate(part["content"])
             if isinstance(item, dict) and item.get("kind") == "binary"]
    return {"instructions": context["base_instructions"], "messages": messages, "tools": context["tools"],
            "image_selection": {"paths": list(dict.fromkeys([paths[0], paths[-1]])), "reason": "Fixture compares first and current evidence."} if paths else None}
""",
            "context:build",
        ).model_dump(mode="json")
    robot = EvidenceRobot(tmp_path)
    decisions = []

    def decide(messages, info):
        decisions.append(messages)
        if len(decisions) == 1:
            name, arguments = "move_once", {}
        elif nested and len(decisions) == 2:
            name, arguments = "workflow_probe", {"arguments": {}}
        else:
            # Correction turns are real native turns, but their proposals are
            # discarded while the oversized C remains invalid.
            name, arguments = "move_once", {}
        return {
            "text": "SIMULATED native input-budget test",
            "tool_calls": [{"name": name, "arguments_json": json.dumps(arguments)}],
        }

    planner = ScriptedNativePlanner(decide)
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            program_calls=100,
            gateway_calls=100,
            # Exercise rejection/correction and the finite controller budget.
            # Sandbox startup and audit I/O must not turn this into a clock test.
            wall_time_s=None,
            request_limits=RequestLimits(max_images=10),
        ),
    )
    result = planner.solve(
        system_prompt="Inspect",
        user_message="Move once then inspect.",
        toolkit=robot,
        max_turns=6,
    )
    assert robot.moves == 1
    assert result.termination_class == AGENT_EXECUTION_FAILURE
    events = rows(tmp_path / "harness_context/bundle_events.jsonl")
    moved = [
        row
        for row in events
        if row["kind"] == "action_completed" and row["operation"] == "move_once"
    ]
    assert len(moved) == 1
    corrections = [row for row in events if row["kind"] == "context_correction"]
    assert not corrections
    assert not any(row["kind"] == "context_correction_stalled" for row in events)
    assert result.stats["visible_turns"] < 6
    assert all(
        row["discarded_proposals"]
        and row["discarded_proposals"][0]["name"] == "move_once"
        for row in corrections
    )
    assert len(planner.scripted_sessions) == 1
    assert any(
        row["kind"] == "operation_settled" and row["operation"] == "read_evidence"
        for row in events
    )
    # Selection cannot evade the complete C input cap before image selection.
    assert any(
        "images" in str(row) and ("budget" in str(row) or "limit" in str(row))
        for row in events
    )
