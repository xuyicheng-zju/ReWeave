"""Native authors receive the actual persistent-session input ownership contract."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.prompts import (
    author_instructions,
    author_reference_documents,
    development_guide,
    runtime_contract,
    runtime_interfaces,
)
from reweave.evolution.bundle import NativeModelContext
from reweave.planner.native_context import baseline_context
from tests.unit_tests.harness.evolution.fixtures import protocol


def test_native_author_knows_memory_and_free_evidence_choices_with_ten_image_cap():
    proposal = SimpleNamespace(
        scopes=("W", "C", "L"),
        purpose="mutation",
        task_protocol=protocol(),
    )
    instructions = " ".join(author_instructions(proposal).split())
    assert instructions.startswith("# Develop one complete Harness")
    references = " ".join(
        "\n".join(author_reference_documents(proposal, "python").values()).split()
    )
    assert "Online provider history stays native" in instructions
    interfaces = runtime_interfaces("native_turn")
    execution = runtime_contract("native_turn")
    assert "no mandatory camera triple" in execution
    assert "at most 10 images" in instructions
    assert (
        "Preserve fixed runtime, tool, observation and budget contracts" in instructions
    )
    assert "C shapes usable evidence" in instructions
    assert "experiment_access.json" not in references
    assert (
        "historical or transformed frames must carry their own source observation"
        in references
    )
    assert "do not relabel them as the current observation" in references
    assert "read_video" in development_guide(proposal, "python")
    assert "C freely organizes evidence" in interfaces["C"]
    assert "at most 10" in interfaces["C"]
    assert "do not blindly attach all three" in interfaces["C"]
    assert "context_rejection" in interfaces["C"]
    assert "text-only correction turn in the same session" in interfaces["C"]
    assert "80/40" not in interfaces["C"]
    assert "C owns selection" not in execution
    assert "Image selection belongs only to C" not in execution
    assert "same native provider session" in execution
    schema = NativeModelContext.model_json_schema()
    assert (
        "historical evidence" in schema["properties"]["image_selection"]["description"]
    )
    assert "10-image ceiling" in schema["properties"]["image_selection"]["description"]
    assert "no mandatory camera set" in schema["properties"]["messages"]["description"]


@pytest.mark.parametrize(
    "field", ["history_window", "session_id", "model_context_window"]
)
def test_native_transport_configuration_is_not_authored_context(field):
    value = {
        "instructions": "Synthetic",
        "messages": [{"kind": "request", "parts": []}],
        "tools": [],
        field: "not an authored field",
    }
    with pytest.raises(ValidationError, match=field):
        NativeModelContext.model_validate(value)


def test_unknown_online_backend_does_not_inherit_a_default_api_contract():
    with pytest.raises(ValueError, match="unknown online controller"):
        runtime_interfaces("unknown")


def test_native_baseline_c_does_not_implement_host_image_or_history_transport():
    source = baseline_context().program.files[0].source
    assert '"messages": context["messages"]' in source
    for host_mechanism in (
        "_runtime_images",
        "image_selection",
        "image_sources",
        "history",
    ):
        assert host_mechanism not in source
