# Modified for the ReWeave namespace.
"""One default incremental C leaves persistent session history with its provider."""

import asyncio
import base64
import copy

import pytest

from reweave.evolution.bundle import (
    ContextPackage,
    ProgramLimits,
    upgrade_context_bundle,
)
from reweave.planner.native_context import baseline_context
from reweave.runtime.capability import ModelInputRejected
from reweave.runtime.model_input import validate_native_images
from reweave.runtime.sandbox import ProgramSandbox
from tests.unit_tests.harness.evolution.fixtures import baseline


def build(messages, state=None):
    context = {
        "base_instructions": "BASE",
        "assets": {"instructions": "GUIDANCE", "skills": [{"text": "EXPERIENCE"}]},
        "messages": messages,
        "state": state or {},
        "tools": [{"name": "observe"}],
    }
    original = copy.deepcopy(context)

    async def no_calls(*args):
        raise AssertionError("baseline C must not call models or tools")

    result = asyncio.run(
        ProgramSandbox().run_program(
            ContextPackage().program,
            ProgramLimits(
                runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=10
            ),
            context,
            no_calls,
            input_schema={"type": "object"},
        )
    )
    assert context == original
    return result


def request(text, images=0):
    return {
        "kind": "request",
        "parts": [
            {
                "part_kind": "user-prompt",
                "content": [
                    text,
                    *[
                        {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(f"{text}-{index}".encode()).decode(),
                            "kind": "image-url",
                        }
                        for index in range(images)
                    ],
                ],
            }
        ],
    }


def test_baseline_is_one_native_context_program():
    assert baseline_context() == ContextPackage()
    assert "window_start" not in ContextPackage().program.files[0].source
    assert "len(messages) - 40" not in ContextPackage().program.files[0].source


def test_baseline_sends_only_supplied_incremental_input_and_retains_authored_state():
    first = build([request("current k", 3)], {"hypothesis": "check board"})
    following = build([request("current k+1", 3)], first["state"])
    assert first["messages"] == [request("current k", 3)]
    assert following["messages"] == [request("current k+1", 3)]
    assert following["state"] == {"hypothesis": "check board"}
    assert following["instructions"] == "BASE\nGUIDANCE\nEXPERIENCE"
    assert following["tools"] == [{"name": "observe"}]
    assert "image_selection" not in following


def test_baseline_never_installs_a_message_window_or_silently_drops_images():
    messages = [request(f"supplied-{index}", 1) for index in range(90)]
    result = build(messages)
    assert result["messages"] == messages
    assert result["state"] == {}
    assert "image_selection" not in result
    with pytest.raises(ModelInputRejected, match="10"):
        validate_native_images(result["messages"])


def test_explicit_upgrade_installs_current_incremental_c_without_mutating_parent():
    parent = baseline()
    value = parent.model_dump(mode="json")
    value["context"]["program"]["files"][0]["source"] = (
        'def build(runtime, context):\n    return {"messages": context["messages"][-40:], "state": {"window_start": 80}}\n'
    )
    parent = type(parent).model_validate(value)
    before = parent.model_dump(mode="json")
    upgraded = upgrade_context_bundle(parent, parent.runtime)
    assert upgraded.context.program == ContextPackage().program
    assert upgraded.record_id != parent.record_id
    assert upgraded.loop == parent.loop and upgraded.workflows == parent.workflows
    assert upgraded.runtime == parent.runtime
    assert parent.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="already uses"):
        upgrade_context_bundle(upgraded, upgraded.runtime)
