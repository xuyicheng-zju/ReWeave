# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Text import freezes audited C content without creating a learning service."""

import os
import asyncio
import json

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import AgentBundle, ProgramLimits
from reweave.memory.__main__ import main
from reweave.memory.import_bundle import CorpusManifest, import_corpus, publish_import
from reweave.runtime.sandbox import ProgramSandbox
from tests.unit_tests.harness.evolution.fixtures import baseline


def source(path, text):
    path.write_text(text, encoding="utf-8")
    return {"path": path.name}


def inputs(tmp_path, *, evidence_count=1, scope="task-family"):
    manifest = {
        "texts": [
            {
                "skill_id": "reference_conditions",
                "scope": scope,
                "source": source(
                    tmp_path / "lesson.md",
                    "# Hypothesis\nFresh observations may contradict this account.\n",
                ),
                "evidence": [
                    source(
                        tmp_path / f"original-{index}.json",
                        json.dumps({"official_success": False}),
                    )
                    for index in range(evidence_count)
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    bundle_path = tmp_path / "parent.json"
    bundle_path.write_text(baseline().model_dump_json())
    return manifest, manifest_path, bundle_path


@pytest.mark.parametrize("scope", ["global", "task-family", "task-specific"])
def test_import_publishes_new_bundle_with_source_audit_and_default_c_consumption(
    tmp_path,
    scope,
):
    _, manifest_path, bundle_path = inputs(tmp_path, evidence_count=4, scope=scope)
    output = tmp_path / "imported.json"
    assert (
        main(
            [
                "import-bundle",
                "--manifest",
                str(manifest_path),
                "--bundle",
                str(bundle_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    imported = AgentBundle.model_validate_json(output.read_text())
    parent = baseline()
    assert imported.record_id != parent.record_id
    assert imported.context != parent.context
    assert imported.workflows == parent.workflows
    assert imported.loop == parent.loop
    assert imported.runtime == parent.runtime
    receipt = json.loads(output.with_name(output.name + ".receipt.json").read_text())
    assert receipt["input_bundle"]["record_id"] == parent.record_id
    assert receipt["output_bundle"]["record_id"] == imported.record_id
    assert receipt["texts"][0]["verification"] == "unverified"
    assert len(receipt["texts"][0]["evidence"]) == 4
    assert receipt["texts"][0]["scope"] == imported.context.skills[0].scope == scope

    async def no_calls(*args):
        raise AssertionError("C must not invoke a model or tool")

    result = asyncio.run(
        ProgramSandbox().run_program(
            imported.context.program,
            ProgramLimits(
                runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=10
            ),
            {
                "base_instructions": "BASE",
                "assets": {
                    "instructions": imported.context.instructions,
                    "skills": [
                        skill.model_dump(mode="json")
                        for skill in imported.context.skills
                    ],
                },
                "messages": [],
                "state": {},
                "tools": [],
                "request_limits": {"max_images": 50},
            },
            no_calls,
            input_schema={"type": "object"},
        )
    )
    assert imported.context.skills[0].text in result["instructions"]
    assert imported.context.skills[0].scope == scope
    assert str(tmp_path / "original-0.json") in result["instructions"]
    assert "unverified" in result["instructions"]
    assert (
        "official_success" not in result["instructions"]
    )  # Evidence is referenced, never replayed/inlined.


def test_unsupported_text_is_preserved_and_incremental_import_keeps_prior_assets(
    tmp_path,
):
    manifest, _, _ = inputs(tmp_path, evidence_count=0)
    first, inventory = import_corpus(
        baseline(), CorpusManifest.model_validate(manifest), source_root=tmp_path
    )
    assert inventory[0]["evidence"] == []
    assert inventory[0]["verification"] == "unverified"
    next_manifest = json.loads(json.dumps(manifest))
    next_manifest["texts"][0]["skill_id"] = "second_reference"
    second, _ = import_corpus(
        first, CorpusManifest.model_validate(next_manifest), source_root=tmp_path
    )
    assert second.context.skills[0] == first.context.skills[0]
    assert len(second.context.skills) == 2
    with pytest.raises(ValueError, match="already exist"):
        import_corpus(
            first, CorpusManifest.model_validate(manifest), source_root=tmp_path
        )


@pytest.mark.parametrize("failure", ["missing", "blank", "invalid_utf8"])
def test_bad_original_fails_without_publication(tmp_path, failure):
    manifest, manifest_path, bundle_path = inputs(tmp_path)
    if failure == "missing":
        (tmp_path / "original-0.json").unlink()
    else:
        raw = b" \n" if failure == "blank" else b"\xff"
        (tmp_path / "lesson.md").write_bytes(raw)
        manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "result.json"
    with pytest.raises((ValueError, OSError)):
        publish_import(manifest_path, bundle_path, output)
    assert not output.exists()
    assert not output.with_name(output.name + ".receipt.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_id",
        "old_scope",
        "unknown_scope",
        "unknown",
        "remote",
        "duplicate_evidence",
    ],
)
def test_manifest_rejects_ambiguous_or_removed_contracts(tmp_path, failure):
    manifest, _, _ = inputs(tmp_path)
    entry = manifest["texts"][0]
    if failure == "duplicate_id":
        manifest["texts"].append(entry)
    elif failure == "old_scope":
        entry["scope"] = "suite"
    elif failure == "unknown_scope":
        entry["scope"] = "unknown"
    elif failure == "unknown":
        entry["confidence"] = "verified"
    elif failure == "remote":
        entry["source"]["path"] = "https://example.org/lesson.md"
    else:
        entry["evidence"].append(entry["evidence"][0])
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(manifest)


@pytest.mark.parametrize("existing", ["bundle", "receipt"])
def test_import_never_overwrites_existing_publication(tmp_path, existing):
    _, manifest_path, bundle_path = inputs(tmp_path)
    output = tmp_path / "result.json"
    receipt_path = output.with_name(output.name + ".receipt.json")
    target = output if existing == "bundle" else receipt_path
    target.write_text("existing bytes")
    with pytest.raises(FileExistsError):
        publish_import(manifest_path, bundle_path, output)
    assert target.read_text() == "existing bytes"
    assert not (receipt_path if existing == "bundle" else output).exists()


def test_failed_bundle_publication_rolls_back_only_its_receipt(tmp_path, monkeypatch):
    _, manifest_path, bundle_path = inputs(tmp_path)
    output = tmp_path / "result.json"
    link = os.link

    def fail_bundle(source, path):
        if path == output:
            raise OSError("publication failed")
        link(source, path)

    monkeypatch.setattr(os, "link", fail_bundle)
    with pytest.raises(OSError, match="publication failed"):
        publish_import(manifest_path, bundle_path, output)
    assert not list(tmp_path.glob("result.json*"))


def test_import_reads_current_source_without_identity_gate(tmp_path):
    _, manifest_path, bundle_path = inputs(tmp_path)
    (tmp_path / "lesson.md").write_text("Current revised lesson")
    output = tmp_path / "current.json"
    publish_import(manifest_path, bundle_path, output)
    imported = AgentBundle.model_validate_json(output.read_text())
    assert "Current revised lesson" in imported.context.skills[0].text
