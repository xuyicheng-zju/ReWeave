# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Online and offline source loading share one exact byte capture/parser."""

import pytest

from reweave.evolution.authoring import candidate
from tests.unit_tests.harness.evolution.fixtures import baseline, mutated


def test_capture_bundle_needs_complete_wcl_without_offline_proposal(tmp_path):
    initial = baseline()
    candidate.write_bundle_source(initial, tmp_path)
    inventory = {}
    captured = candidate.capture_bundle(
        tmp_path,
        initial.runtime,
        candidate_inventory=inventory,
        record_id=initial.record_id,
    )
    assert captured.bundle == initial
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "SKILL.md").exists()
    assert (
        captured.files["task-family/context/instructions.md"]
        == initial.context.instructions.encode()
    )
    assert inventory["task-family/context/instructions.md"]["links"] == 1
    with pytest.raises(TypeError):
        captured.files["task-family/context/instructions.md"] = b"replacement"


def test_capture_bundle_parses_and_snapshots_the_single_capture(tmp_path, monkeypatch):
    initial = baseline()
    changed = mutated(initial, "C")
    candidate.write_bundle_source(initial, tmp_path)
    original = candidate._capture_candidate
    calls = []

    def replace_after_capture(root, inventory):
        calls.append(root)
        captured = original(root, inventory)
        candidate.write_bundle_source(changed, root)
        return captured

    monkeypatch.setattr(candidate, "_capture_candidate", replace_after_capture)
    captured = candidate.capture_bundle(
        tmp_path, initial.runtime, record_id=initial.record_id
    )
    assert calls == [tmp_path]
    assert captured.bundle == initial
    assert (
        captured.files["task-family/context/instructions.md"]
        == initial.context.instructions.encode()
    )
    assert (
        tmp_path / "task-family/context/instructions.md"
    ).read_text() == changed.context.instructions


def test_capture_bundle_does_not_accept_editable_runtime_files(tmp_path):
    initial = baseline()
    candidate.write_bundle_source(initial, tmp_path)
    (tmp_path / "runtime.json").write_text(initial.runtime.model_dump_json())
    with pytest.raises(ValueError, match="undeclared/missing candidate files"):
        candidate.capture_bundle(tmp_path, initial.runtime, record_id=initial.record_id)
