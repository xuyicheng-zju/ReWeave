"""Native messages retain complete text and reference actual image files."""

import asyncio
import base64
import copy
import io
import json

import pytest
from PIL import Image
from pydantic_ai.messages import ModelMessagesTypeAdapter

from reweave.evolution.bundle import ProgramLimits, RequestLimits
from reweave.planner.native_context import (
    baseline_context,
    prepare_native,
    prepare_observation,
)
from reweave.runtime.capability import CandidateExecutionFailure
from reweave.runtime.sandbox import ProgramSandbox


def pixels(color):
    output = io.BytesIO()
    Image.new("RGB", (7, 5), color).save(output, "PNG")
    return output.getvalue()


def binary(data):
    return {
        "kind": "binary",
        "media_type": "image/png",
        "data": base64.b64encode(data).decode(),
    }


def inputs(*, observation, messages=None):
    return {
        "messages": messages or [{"kind": "request", "parts": []}],
        "observation": observation,
        "physical_budget": {"limit": 1100, "used": 100, "remaining": 1000},
        "remaining_wall_time_s": None,
        "objective": "Read the current scene",
        "assets": {"instructions": "", "skills": []},
        "base_instructions": "Task guidance",
        "tools": [],
    }


def build(value):
    namespace = {}
    exec(baseline_context().program.files[0].source, namespace)
    prepared = namespace["build"](None, prepare_observation(value))
    ModelMessagesTypeAdapter.validate_python(prepared["messages"])
    return prepared


def pointer(value, path):
    for token in path.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def observation_text(prepared):
    content = prepared["messages"][-1]["parts"][-1]["content"]
    return json.loads(content[0] if isinstance(content, list) else content)


@pytest.mark.parametrize("already_supplied", [False, True])
def test_observation_pixels_are_structured_once_with_resolvable_references(
    tmp_path, already_supplied
):
    data = [pixels(color) for color in ("red", "green", "blue")]
    frames = [
        {
            "name": f"camera-{index}",
            "media_type": "image/png",
            "data_base64": base64.b64encode(raw).decode(),
            "calibration_tag": f"calibration-{index}",
        }
        for index, raw in enumerate(data)
    ]
    messages = [{"kind": "request", "parts": []}]
    if already_supplied:
        messages[0]["parts"].append(
            {"part_kind": "user-prompt", "content": [binary(raw) for raw in data]}
        )
    log = "FULL_LOG_START:" + "完整反馈" * 30000 + ":FULL_LOG_END"
    source = inputs(
        observation={
            "state": {"position": 1},
            "nested": {"_runtime_images": frames},
            "log": log,
        },
        messages=messages,
    )
    original = copy.deepcopy(source)
    prepared = build(source)
    text, images = prepare_native(prepared, tmp_path, RequestLimits(max_images=3))
    assert source == original
    assert [path.read_bytes() for path in images] == data
    assert all(frame["data_base64"] not in text for frame in frames)
    delivered = json.loads(text)
    projected = observation_text(delivered)
    assert projected["observation"]["log"] == log
    assert projected["physical_budget"] == source["physical_budget"]
    references = projected["observation"]["nested"]["_runtime_images"]
    for index, reference in enumerate(references):
        assert reference["name"] == frames[index]["name"]
        assert reference["calibration_tag"] == frames[index]["calibration_tag"]
        assert pointer(delivered, reference["image_reference"]) == {
            "image_file": str(images[index])
        }


def test_same_pixels_reuse_one_file_without_removing_image_occurrence_references(
    tmp_path,
):
    raw = pixels("red")
    encoded = base64.b64encode(raw).decode()
    prepared = {
        "instructions": "Read both source references",
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            binary(raw),
                            {
                                "kind": "image-url",
                                "url": "data:image/png;base64," + encoded,
                            },
                        ],
                    }
                ],
            }
        ],
        "tools": [],
    }
    original = copy.deepcopy(prepared)
    text, images = prepare_native(prepared, tmp_path, RequestLimits())
    assert len(images) == 1 and images[0].read_bytes() == raw
    content = json.loads(text)["messages"][0]["parts"][0]["content"]
    assert content == [{"image_file": str(images[0])}] * 2
    assert prepared == original
    # The wire request contains one unique image, with two source references.
    _, selected = prepare_native(
        prepared, tmp_path / "one-image", RequestLimits(max_images=1)
    )
    assert len(selected) == 1


def test_literal_json_tool_text_is_not_guessed_or_rewritten(tmp_path):
    literal = json.dumps(
        {"_runtime_images": [{"data_base64": base64.b64encode(pixels("red")).decode()}]}
    )
    prepared = build(
        inputs(
            observation=None,
            messages=[
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": literal}],
                }
            ],
        )
    )
    text, images = prepare_native(prepared, tmp_path, RequestLimits())
    assert images == ()
    assert json.loads(text)["messages"][0]["parts"][0]["content"] == literal


def test_different_pixels_remain_distinct(tmp_path):
    data = [pixels("red"), pixels("blue")]
    prepared = build(
        inputs(
            observation={
                "_runtime_images": [
                    {
                        "name": "same-camera-name",
                        "media_type": "image/png",
                        "data_base64": base64.b64encode(raw).decode(),
                    }
                    for raw in data
                ]
            }
        )
    )
    _, images = prepare_native(prepared, tmp_path, RequestLimits())
    assert len(images) == 2
    assert [path.read_bytes() for path in images] == data


def test_baseline_image_separation_runs_through_actual_context_sandbox(tmp_path):
    raw = pixels("red")
    source = inputs(
        observation={
            "_runtime_images": [
                {
                    "name": "current-camera",
                    "media_type": "image/png",
                    "data_base64": base64.b64encode(raw).decode(),
                }
            ]
        }
    )

    async def forbidden_gateway(*args):
        pytest.fail("context image separation must not issue a Host request")

    prepared = asyncio.run(
        ProgramSandbox().run_program(
            baseline_context().program,
            ProgramLimits(
                runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=20
            ),
            prepare_observation(source),
            forbidden_gateway,
            input_schema={"type": "object"},
        )
    )
    ModelMessagesTypeAdapter.validate_python(prepared["messages"])
    text, images = prepare_native(prepared, tmp_path, RequestLimits())
    assert len(images) == 1 and images[0].read_bytes() == raw
    assert source["observation"]["_runtime_images"][0]["data_base64"] not in text


def test_catalog_reference_preserves_messages_pixels_and_full_request_budget(tmp_path):
    prepared = build(inputs(observation={"state": "unchanged"}))
    prepared["messages"][-1]["parts"].append(
        {"part_kind": "user-prompt", "content": [binary(pixels("blue"))]}
    )
    prepared["tools"] = [{"name": "inspect", "description": "Complete schema " * 100}]
    full, full_images = prepare_native(prepared, tmp_path / "full", RequestLimits())
    reference, reference_images = prepare_native(
        prepared, tmp_path / "reference", RequestLimits(), include_tools=False
    )
    delivered = json.loads(reference)
    assert delivered["available_tools"] == ["inspect"]
    assert delivered["tool_catalog"] == "unchanged"
    assert "tools" not in delivered
    assert len(reference.encode()) < len(full.encode())
    assert [path.read_bytes() for path in full_images] == [
        path.read_bytes() for path in reference_images
    ]
    # Only the materialized image filenames differ between these two requests.
    assert json.loads(full.replace(str(full_images[0]), "pixels")) == json.loads(
        json.dumps(
            {"messages": delivered["messages"], "tools": prepared["tools"]},
            ensure_ascii=False,
        ).replace(str(reference_images[0]), "pixels")
    )
    with pytest.raises(CandidateExecutionFailure, match="input_bytes budget"):
        prepare_native(
            prepared,
            tmp_path / "over-budget",
            RequestLimits(max_input_bytes=500),
            include_tools=False,
        )


@pytest.mark.parametrize("count", [0, 1, 3, 10])
def test_native_c_allows_free_camera_count_and_keyframe_reemphasis(tmp_path, count):
    frames = [binary(pixels((i, 0, 0))) for i in range(count)]
    context = {
        "instructions": "Relevant keyframe comparison",
        "tools": [],
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": ["Historical frame; not current geometry", *frames],
                    }
                ],
            }
        ],
    }
    text, paths = prepare_native(context, tmp_path, RequestLimits())
    assert len(paths) == count
    assert "Historical frame" in text


@pytest.mark.parametrize("hide_extra", [False, True])
def test_native_c_rejects_eleven_images_before_selection(tmp_path, hide_extra):
    frames = [binary(pixels((i, 0, 0))) for i in range(11)]
    context = {
        "instructions": "Oversized C",
        "tools": [],
        "messages": [
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": frames}],
            }
        ],
    }
    if hide_extra:
        context["image_selection"] = {"paths": [], "reason": "Hide over-budget sources"}
    with pytest.raises(CandidateExecutionFailure, match="11 > 10"):
        prepare_native(context, tmp_path, RequestLimits())
    assert not list(tmp_path.iterdir())
