"""Author previews expose exact Host manifest failures without paid execution."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.tools import implementation_schema
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace


def implementation(topology):
    site = {
        "pointers": ["/context/instructions"],
        "explanation": "The deployed planner retains this original source behavior.",
    }
    return {
        "source_constraints": dict.fromkeys(topology.sources, site),
        "interfaces": {
            member.key: site
            for group in topology.mechanisms.values()
            for member in group.members
        },
    }


def isolated_preview(tmp_path, monkeypatch):
    request, workspace, _, topology, author = prepared_workspace(tmp_path)
    calls = []

    async def contracts(candidate, contract, *, case_results):
        calls.append("contracts")
        case_results.append({"scope": "C", "validation_status": "passed"})

    async def interactions(request, workspace, candidate, case_results):
        calls.append("interaction")
        case_results.append({"scope": "interaction", "validation_status": "passed"})

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts", contracts
    )
    monkeypatch.setattr(author, "_check_interactions", interactions)
    return request, workspace, topology, author, calls


@pytest.mark.parametrize("field", ["source_constraints", "interfaces"])
def test_interaction_preview_rejects_extra_manifest_keys_with_readable_report(
    tmp_path, monkeypatch, field
):
    request, workspace, topology, author, calls = isolated_preview(
        tmp_path, monkeypatch
    )
    raw = implementation(topology)
    extra = (
        "unrelated-source-condition"
        if field == "source_constraints"
        else "group-id-instead-of-interface"
    )
    raw[field][extra] = next(iter(raw[field].values()))
    atomic_json(workspace / ".scratch/shared_implementation.json", raw)

    with pytest.raises(ValueError, match="unexpected keys.*candidate_validation.json"):
        asyncio.run(author.validate_code(request, workspace, interactions=True))

    report = json.loads((workspace / "context/candidate_validation.json").read_text())
    assert report["error_type"] == "ValueError"
    assert field in report["error"] and extra in report["error"]
    assert '"missing_count":0' in report["error"]
    assert '"extra_count":1' in report["error"]
    assert "required_keys" in report["error"]
    assert [case["scope"] for case in report["contract_case_results"]] == [
        "C",
        "interaction",
    ]
    assert calls == ["contracts", "interaction"]


def test_interaction_preview_accepts_exact_manifest_and_real_candidate_locations(
    tmp_path, monkeypatch
):
    request, workspace, topology, author, calls = isolated_preview(
        tmp_path, monkeypatch
    )
    atomic_json(
        workspace / ".scratch/shared_implementation.json", implementation(topology)
    )
    result = asyncio.run(author.validate_code(request, workspace, interactions=True))
    assert result["structurally_valid"] and result["interaction_tested"]
    assert result["physical_execution_tested"] is False
    assert result["contract_cases"] == result["interaction_cases"] == 1
    assert calls == ["contracts", "interaction"]
    report = json.loads((workspace / "context/candidate_validation.json").read_text())
    assert "error" not in report


def test_preview_resolves_manifest_pointers_against_the_captured_candidate(
    tmp_path, monkeypatch
):
    request, workspace, topology, author, _ = isolated_preview(tmp_path, monkeypatch)
    raw = implementation(topology)
    raw["source_constraints"][next(iter(topology.sources))] = {
        "pointers": ["/context/absent"],
        "explanation": "This location deliberately does not exist in the candidate.",
    }
    atomic_json(workspace / ".scratch/shared_implementation.json", raw)
    with pytest.raises(ValueError, match="pointer does not resolve"):
        asyncio.run(author.validate_code(request, workspace, interactions=True))
    report = json.loads((workspace / "context/candidate_validation.json").read_text())
    assert report["error_type"] == "ValueError"
    assert "pointer does not resolve" in report["error"]


def test_early_interaction_preview_allows_manifest_not_yet_written(
    tmp_path, monkeypatch
):
    request, workspace, _, author, calls = isolated_preview(tmp_path, monkeypatch)
    assert not (workspace / ".scratch/shared_implementation.json").exists()
    result = asyncio.run(author.validate_code(request, workspace, interactions=True))
    assert result["interaction_tested"] and not result["physical_execution_tested"]
    assert calls == ["contracts", "interaction"]


def test_direct_interaction_preview_does_not_apply_aligned_manifest_requirements(
    tmp_path, monkeypatch
):
    request, workspace, _, author, calls = isolated_preview(tmp_path, monkeypatch)
    request = {**request, "method": "direct", "shared_topology": None}
    path = workspace / ".scratch/shared_implementation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("this is deliberately not a shared implementation JSON document")
    result = asyncio.run(author.validate_code(request, workspace, interactions=True))
    assert result["structurally_valid"] and result["interaction_tested"]
    assert result["physical_execution_tested"] is False
    assert calls == ["contracts", "interaction"]


def test_recovered_workspace_refreshes_exact_schema_and_preserves_author_files(
    tmp_path,
):
    request, workspace, _, topology, author = prepared_workspace(tmp_path)
    schema_path = workspace / "context/shared_implementation.schema.json"
    schema_path.write_text(json.dumps({"type": "object", "additionalProperties": True}))
    manifest_path = workspace / ".scratch/shared_implementation.json"
    raw = implementation(topology)
    raw["interfaces"]["author-work-in-progress"] = next(
        iter(raw["interfaces"].values())
    )
    atomic_json(manifest_path, raw)
    original_manifest = manifest_path.read_bytes()
    candidate_files = {
        path.relative_to(workspace): path.read_bytes()
        for path in (workspace / "candidate").rglob("*")
        if path.is_file()
    }

    recovered, _ = author.prepare(request, workspace.parent)

    assert recovered == workspace
    schema = json.loads(schema_path.read_text())
    assert schema == implementation_schema(topology)
    for field, keys in (
        ("source_constraints", set(topology.sources)),
        ("interfaces", set(implementation(topology)["interfaces"])),
    ):
        assert set(schema["properties"][field]["propertyNames"]["enum"]) == keys
        assert set(schema["properties"][field]["required"]) == keys
    assert manifest_path.read_bytes() == original_manifest
    assert candidate_files
    assert {
        path.relative_to(workspace): path.read_bytes()
        for path in (workspace / "candidate").rglob("*")
        if path.is_file()
    } == candidate_files
