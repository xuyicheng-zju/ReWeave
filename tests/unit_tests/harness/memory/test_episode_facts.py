"""Host facts and recipes retain one actual trace and never reconstruct actions."""

import json

import pytest

from reweave.memory.episode import record_episode


def episode(root, success):
    transcript = root / "transcript_task.json"
    transcript.write_text(
        json.dumps(
            {
                "official_success": success,
                "termination_class": "success" if success else "valid_failure",
                "bundle_id": "executed",
            }
        )
    )
    rows = [
        {
            "event_type": "control_step",
            "control_step_index": index,
            "action": [index, 0.123],
            "physical_call_ref": "call-1",
            "ack": "accepted",
        }
        for index in range(2)
    ]
    (root / "execution_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return transcript, rows


def test_success_exports_exact_matched_pair_idempotently(tmp_path):
    transcript, rows = episode(tmp_path, True)
    audit_path = record_episode(transcript)
    audit = json.loads(audit_path.read_text())
    recipe = audit_path.with_name(audit["recipe"])
    assert [json.loads(line) for line in recipe.read_text().splitlines()] == rows
    assert audit["owner"] == "Host" and audit["control_steps"] == 2
    original = (audit_path.read_bytes(), recipe.read_bytes())
    record_episode(transcript)
    assert original == (audit_path.read_bytes(), recipe.read_bytes())
    assert audit["sources"][transcript.name]["path"] == transcript.name


def test_failure_records_audit_without_winning_recipe(tmp_path):
    transcript, _ = episode(tmp_path, False)
    audit = json.loads(record_episode(transcript).read_text())
    assert audit["official_success"] is False and audit["recipe"] is None
    assert not (tmp_path / "task-specific/recipe.jsonl").exists()


@pytest.mark.parametrize("corruption", ["unknown", "missing_step"])
def test_unknown_or_incomplete_trace_cannot_publish_success(tmp_path, corruption):
    transcript, rows = episode(tmp_path, True)
    trace = tmp_path / "execution_trace.jsonl"
    if corruption == "unknown":
        with trace.open("a") as stream:
            stream.write(json.dumps({"event_type": "action_outcome_unknown"}) + "\n")
    else:
        rows[1]["control_step_index"] = 3
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        record_episode(transcript)
    assert not (tmp_path / "task-specific").exists()


def test_existing_episode_facts_cannot_be_replaced(tmp_path):
    transcript, _ = episode(tmp_path, True)
    audit = record_episode(transcript)
    before = audit.read_bytes()
    (tmp_path / "task-specific/recipe.jsonl").write_text("forged")
    with pytest.raises(ValueError, match="refusing to replace"):
        record_episode(transcript)
    assert audit.read_bytes() == before


def test_episode_relocation_preserves_fact_references_and_idempotence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    transcript, _ = episode(source, True)
    audit_path = record_episode(transcript)
    before = audit_path.read_bytes()
    moved = tmp_path / "rounds/round-00/rollout"
    moved.parent.mkdir(parents=True)
    source.rename(moved)
    audit_path = moved / "task-specific/audit.json"
    audit = json.loads(audit_path.read_text())
    for original in audit["sources"].values():
        path = audit_path.parent / audit["source_root"] / original["path"]
        assert path.stat().st_size == original["bytes"]
    assert record_episode(moved / transcript.name).read_bytes() == before
