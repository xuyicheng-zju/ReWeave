"""Parent fixtures are reusable engineering inputs, never inherited admission."""

import json

from reweave.evolution.authoring.fixtures import (
    SUITE,
    history_parent_interactions,
    seed_parent_interactions,
)
from reweave.evolution.evidence.records import artifact_record
from tests.unit_tests.harness.evolution.fixtures import baseline


def test_exact_parent_suite_is_reused_and_existing_edits_survive(tmp_path):
    parent = baseline().model_dump(mode="json")
    source = tmp_path / "prior/workspace" / SUITE
    source.parent.mkdir(parents=True)
    source.write_text('[{"name": "parent fixture"}]')
    fixture = artifact_record(source)
    workspace = tmp_path / "next/workspace"
    shared = {"bundle": parent, "interaction_suite": fixture}
    seed_parent_interactions(workspace, "unrelated-parent", [shared])
    assert not (workspace / SUITE).exists()
    seed_parent_interactions(workspace, parent["record_id"], [shared])
    assert (workspace / SUITE).read_bytes() == source.read_bytes()
    metadata = json.loads((workspace / "context/parent_interaction.json").read_text())
    assert metadata["source"] == fixture
    assert "no prior verdict" in metadata["meaning"]
    (workspace / SUITE).write_text('[{"name": "edited"}]')
    seed_parent_interactions(workspace, parent["record_id"], [shared])
    assert "edited" in (workspace / SUITE).read_text()


def test_single_task_history_reuses_only_parent_published_suite(tmp_path):
    parent = baseline().model_dump(mode="json")
    source = tmp_path / "prior/workspace" / SUITE
    source.parent.mkdir(parents=True)
    source.write_text('[{"name": "fixture"}]')
    receipt = source.parents[3] / "result.json"
    receipt.write_text(
        json.dumps(
            {
                "candidates": [{"bundle": parent}],
                "artifacts": [artifact_record(source)],
            }
        )
    )
    original = {**artifact_record(receipt), "pointer": ""}
    originals = {
        "navigation": {
            "versions": {
                "v1": {"bundle_id": parent["record_id"], "author_attempt": "A1"}
            },
            "attempts": {"A1": {"receipt": original}},
        }
    }
    workspace = tmp_path / "next/workspace"
    history_parent_interactions(workspace, parent["record_id"], originals)
    assert (workspace / SUITE).read_bytes() == source.read_bytes()


def test_analyst_corrections_keep_measurement_scope():
    from reweave.evolution.distillation.progress import NativeTopologyProgress

    analyst = object.__new__(NativeTopologyProgress)
    prompt = analyst._correction_prompt({"error_type": "ValueError"}, guidance="")
    assert "phase=interaction" not in prompt
    assert "repair the candidate" in prompt
    assert json.loads(prompt.split("\n\n", 1)[1]) == {"error_type": "ValueError"}
