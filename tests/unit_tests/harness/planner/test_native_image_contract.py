"""Single-request image construction, including the historical base64 failure."""

import base64
import copy
import io
import json
import random

import pytest
from PIL import Image

from reweave.evolution.bundle import RequestLimits
from reweave.planner.decisions import decision_error
from reweave.planner.native_context import prepare_native, prepare_observation
from reweave.runtime.model_input import image_inventory, unique_image_count
from tests.unit_tests.harness.planner.test_native_context import inputs


def frame(seed):
    stream = io.BytesIO()
    Image.frombytes("RGB", (64, 64), random.Random(seed).randbytes(64 * 64 * 3)).save(
        stream, "PNG"
    )
    return stream.getvalue()


def binary(data):
    return {
        "kind": "binary",
        "media_type": "image/png",
        "data": base64.urlsafe_b64encode(data).decode(),
    }


def observation(data):
    return {
        "_runtime_images": [
            {
                "name": str(i),
                "media_type": "image/png",
                "data_base64": base64.b64encode(raw).decode(),
            }
            for i, raw in enumerate(data)
        ]
    }


@pytest.mark.parametrize("url", [False, True])
def test_byte_dedup_handles_real_png_base64_alphabets_and_data_urls(tmp_path, url):
    raw = frame(1)
    assert base64.urlsafe_b64encode(raw) != base64.b64encode(raw)
    image = (
        {
            "kind": "image-url",
            "url": "data:image/png;base64," + base64.urlsafe_b64encode(raw).decode(),
        }
        if url
        else binary(raw)
    )
    context = inputs(
        observation=observation([raw]),
        messages=[
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": [image]}],
            }
        ],
    )
    prepared = prepare_observation(context)
    assert len(image_inventory(prepared["messages"])) == 1
    prepared["instructions"] = "Inspect"
    _, images = prepare_native(prepared, tmp_path, RequestLimits(max_images=1))
    assert [p.read_bytes() for p in images] == [raw]


def test_five_physical_calls_keep_ack_text_and_explicit_read_pixels(tmp_path):
    current = [frame(i) for i in range(3)]
    read = frame(99)
    messages = []
    for i in range(5):
        messages.append(
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "move",
                        "content": f"ack-{i}",
                        "tool_call_id": f"move-{i}",
                    },
                    {
                        "part_kind": "user-prompt",
                        "content": [binary(frame(10 + 3 * i + j)) for j in range(3)],
                    },
                ],
            }
        )
    messages[-1]["parts"].extend(
        [
            {
                "part_kind": "tool-return",
                "tool_name": "read_image",
                "content": "explicit historical image",
                "tool_call_id": "read-1",
            },
            {"part_kind": "user-prompt", "content": [binary(read)]},
        ]
    )
    source = inputs(observation=observation(current), messages=messages)
    source["tool_effects"] = {"move": "physical", "read_image": "read"}
    original = copy.deepcopy(source)
    prepared = prepare_observation(source)
    assert unique_image_count(prepared["messages"]) == 4
    assert unique_image_count(prepared["source_messages"]) == 16
    assert source == original
    assert all(f"ack-{i}" in json.dumps(prepared["messages"]) for i in range(5))
    prepared["instructions"] = "Inspect"
    _, paths = prepare_native(prepared, tmp_path, RequestLimits(max_images=4))
    assert [p.read_bytes() for p in paths] == [read, *current]
    assert prepare_observation(prepared) == prepared


def test_read_batches_are_rejected_before_execution_and_physical_batches_fit():
    contract = {
        "observation": 3,
        "reads": {"view_env_state": 3, "segment": 1, "new_reader": None},
    }
    assert decision_error(["move"] * 5, image_contract=contract) is None
    assert decision_error(["view_env_state"] * 2, image_contract=contract) is None
    assert "Nothing was executed" in decision_error(
        ["view_env_state"] * 3, image_contract=contract
    )
    assert "undeclared" in decision_error(["new_reader"], image_contract=contract)
    assert "Nothing was executed" in decision_error(
        ["segment"] * 2, image_contract=contract, image_limit=4
    )


def test_native_planner_rejects_entire_read_batch_before_any_gateway_call(tmp_path):
    from reweave.runtime.kernel import EpisodeLimits
    from reweave.tools.toolkit import readonly
    from tests.native_fixture import PhysicalRobot, ScriptedNativePlanner, rows
    from tests.unit_tests.harness.evolution.fixtures import baseline

    class Robot(PhysicalRobot):
        def native_image_contract(self):
            return {"observation": 3, "reads": {"view": 3}}

    robot = Robot(tmp_path)
    calls = []

    @readonly
    def view():
        calls.append("view")
        return {"_image_bytes": frame(20)}

    robot.add_tool(
        "view",
        {
            "name": "view",
            "description": "Recorded views",
            "input_schema": {"type": "object", "properties": {}},
        },
        view,
    )
    decisions = []

    def decide(messages, info):
        decisions.append(messages)
        if len(decisions) == 1:
            names = ["move_once", "view", "view", "view"]
            return {
                "text": "Synthetic over-budget batch",
                "tool_calls": [
                    {"name": name, "arguments_json": "{}"} for name in names
                ],
            }
        assert "Nothing was executed" in str(messages)
        return {
            "text": "Stop fixture",
            "tool_calls": [
                {
                    "name": "finish",
                    "arguments_json": json.dumps(
                        {"status": "failed", "summary": "fixture complete"}
                    ),
                }
            ],
        }

    planner = ScriptedNativePlanner(decide)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=None, gateway_calls=None, wall_time_s=None
        ),
    )
    result = planner.solve(
        system_prompt="Synthetic batch check",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None
    assert calls == [] and robot.moves == 0
    rejected = [
        event
        for event in rows(tmp_path / "harness_context/bundle_events.jsonl")
        if event["kind"] == "model_call_rejected"
    ]
    assert len(rejected) == 4


def test_workflow_custom_pixels_are_preserved_while_known_physical_repeats_are_referenced():
    physical, custom = frame(1), frame(2)
    context = inputs(
        observation=observation([physical]),
        messages=[
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "workflow_inspect",
                        "content": "Visual workflow result",
                        "tool_call_id": "w",
                    },
                    {
                        "part_kind": "user-prompt",
                        "content": [binary(physical), binary(custom)],
                    },
                ],
            }
        ],
    )
    context["tool_effects"] = {"move": "physical"}
    context["history"] = [{"operation": "move", "result": observation([physical])}]
    prepared = prepare_observation(context)
    assert unique_image_count(prepared["messages"]) == 2
    content = prepared["messages"][0]["parts"][1]["content"]
    assert isinstance(content[0], str) and content[1] == binary(custom)


@pytest.mark.parametrize(
    "feedback_source, expected", [("program", 3), ("model", 27), (None, 27)]
)
def test_read_origin_controls_default_pixels_without_erasing_evidence(
    feedback_source, expected
):
    """Eight W control reads are not eight requests to show cameras to the model."""
    messages = []
    for index in range(8):
        feedback = {
            "operation": "view_env_state",
            "result": {"aperture": index},
            "program_call_id": "workflow-child",
        }
        if feedback_source is not None:
            feedback["feedback_source"] = feedback_source
        messages.append(
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            json.dumps({"history_index": index, "feedback": feedback}),
                            *[binary(frame(3 * index + j + 10)) for j in range(3)],
                        ],
                    }
                ],
            }
        )
    original = inputs(
        observation=observation([frame(j) for j in range(3)]), messages=messages
    )
    original["tool_effects"] = {"view_env_state": "read"}
    original["program_call_id"] = "root"
    prepared = prepare_observation(original)
    assert unique_image_count(prepared["messages"]) == expected
    assert prepared["source_messages"] == messages
    assert unique_image_count(prepared["source_messages"]) == 24
    assert all(
        f'"aperture": {i}' in json.dumps(prepared["messages"]).replace('\\"', '"')
        for i in range(8)
    )


def test_historical_read_source_uses_call_ancestry_not_read_name():
    from reweave.evolution.validation.context_replay import recorded_feedback_sources

    context = inputs(
        observation=observation([frame(1)]),
        messages=[
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": json.dumps(
                            {
                                "history_index": i,
                                "feedback": {"operation": "view_env_state"},
                            }
                        ),
                    }
                    for i in range(4)
                ],
            }
        ],
    )
    events = [
        {
            "kind": "workflow_started",
            "program_call_id": "root",
            "planner_call_id": "parent-call",
        },
        *[
            {
                "kind": "operation_settled",
                "operation": "view_env_state",
                "program_call_id": "child",
                "parent_program_call_id": "root",
                "planner_call_id": call,
            }
            for call in ("parent-call", "nested-model-call", None)
        ],
        {"kind": "operation_settled", "operation": "view_env_state"},
    ]
    restored = recorded_feedback_sources(context, events)
    sources = [
        json.loads(part["content"])["feedback"].get("feedback_source")
        for part in restored["messages"][0]["parts"]
    ]
    assert sources == ["program", "model", "program", None]
    assert "feedback_source" not in json.dumps(context)
