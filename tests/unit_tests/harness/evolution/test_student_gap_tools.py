"""Current gap references are source scoped, paged and available without inference."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.student_gaps import teacher_unit_bank
from reweave.evolution.distillation.tools import SharedAuthorTools
from reweave.evolution.distillation.topology import occurrence_key
from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace


def gap_workspace(tmp_path):
    _, workspace, graphs, _, _ = prepared_workspace(tmp_path)
    bank = teacher_unit_bank(graphs)
    original = occurrence_key("alpha", "edge0")
    candidates = [unit for unit in bank["units"] if unit != original]
    fact = {"fluent": "phase", "arguments": ["object"], "values": ["held"]}
    variants = [
        {
            "query": "query-0",
            "unit": unit,
            "entity_mapping": {"object": f"object-{i}"},
            "matched_results": [fact],
            "uncovered_results": [fact] if i % 2 else [],
            "missing_entry": [fact] if i % 3 == 0 else [],
            "entry_conflicts": [fact] if i % 2 else [],
            "unbound_entry": [fact] if i % 3 == 1 else [],
            "match_kind": "partial" if i % 2 else "full",
        }
        for unit in candidates
        for i in range(10)
    ]
    report = {
        "student_bundle_id": "current-parent",
        "mechanisms": [
            {
                "id": "mechanism-0",
                "task": "alpha",
                "retained_node": "a",
                "frontier_node": "b",
                "record_ids": ["record-0"],
                "query_ids": ["query-0"],
                "candidate_units": candidates,
            }
        ],
        "gaps": [
            {
                "id": "query-0",
                "mechanism": "mechanism-0",
                "record_id": "record-0",
                "unit": original,
                "known": [],
                "desired_results": [fact],
            }
        ],
        "matches": {"mechanism-0": variants},
        "summary": {"failed_conditions": 1},
        "protected_outcomes": [],
        "unresolved": [],
    }
    (workspace / "context/student_gaps.json").write_text(json.dumps(report))
    (workspace / "context/teacher_unit_bank.json").write_text(json.dumps(bank))
    return workspace, bank, report, original


def call(tools, **arguments):
    return asyncio.run(tools.model_call("query_student_gap", arguments))


def test_catalog_is_conditional_and_mechanisms_are_compact(tmp_path):
    workspace, _, report, _ = gap_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
    names = {tool.name for tool in asyncio.run(tools.catalog())}
    assert {"query_student_gap", "read_original", "read_reference_graph"} <= names
    result = call(tools)
    assert result["student_bundle_id"] == "current-parent"
    assert result["mechanisms"][0]["candidate_count"] == 3
    assert "candidate_units" not in result["mechanisms"][0]
    assert "matches" not in result and result["complete"]
    assert result["summary"] == report["summary"]
    (workspace / "context/student_gaps.json").unlink()
    absent = SharedAuthorTools(workspace, tmp_path / "absent.jsonl")
    assert "query_student_gap" not in {
        tool.name for tool in asyncio.run(absent.catalog())
    }
    with pytest.raises(ValueError, match="unavailable"):
        call(absent)


def test_all_candidate_and_variant_pages_preserve_partial_conflicting_matches(tmp_path):
    workspace, bank, report, _ = gap_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
    offset, candidates = 0, []
    while offset is not None:
        result = call(tools, gap_id="mechanism-0", offset=offset, limit=1)
        candidates.extend(result["candidates"])
        offset = result["next_offset"]
    assert [row["unit"] for row in candidates] == report["mechanisms"][0][
        "candidate_units"
    ]
    assert all(row["variant_count"] == 10 for row in candidates)
    assert all(row["partial_count"] == row["full_count"] == 5 for row in candidates)
    assert all(row["entry_flag_counts"]["entry_conflicts"] == 5 for row in candidates)
    unit = candidates[0]["unit"]
    first = call(tools, gap_id="mechanism-0", unit=unit)
    assert len(first["matches"]) == first["next_offset"] == 8
    second = call(tools, gap_id="mechanism-0", unit=unit, offset=8)
    variants = first["matches"] + second["matches"]
    assert variants == [
        m for m in report["matches"]["mechanism-0"] if m["unit"] == unit
    ]
    assert second["complete"] and second["next_offset"] is None
    assert first["detail"] == bank["units"][unit]
    assert any(m["match_kind"] == "partial" and m["entry_conflicts"] for m in variants)


def test_original_unmatched_interface_and_any_admitted_unit_are_readable(tmp_path):
    workspace, bank, _, original = gap_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
    result = call(tools, gap_id="mechanism-0", unit=original)
    assert result["matches"] == [] and result["not_a_match"]
    assert result["detail"] == bank["units"][original]
    for identity, detail in bank["units"].items():
        assert call(tools, unit=identity)["detail"] == detail


@pytest.mark.parametrize(
    "arguments",
    [
        {"gap_id": "unseen"},
        {"unit": "unseen"},
        {"gap_id": "mechanism-0", "unit": "unseen"},
        {"gap_id": None},
        {"unit": 1},
        {"offset": -1},
        {"offset": True},
        {"offset": 0.5},
        {"limit": 0},
        {"limit": 65},
        {"limit": "8"},
        {"limit": False},
        {"top_k": 2},
    ],
)
def test_unknown_ids_and_malformed_page_arguments_are_rejected(tmp_path, arguments):
    workspace, _, _, _ = gap_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        asyncio.run(tools.model_call("query_student_gap", arguments))


@pytest.mark.parametrize("mutation", ["foreign_unit", "foreign_task", "foreign_query"])
def test_report_cannot_admit_external_sources_or_queries(tmp_path, mutation):
    workspace, bank, report, _ = gap_workspace(tmp_path)
    if mutation == "foreign_unit":
        bank["units"]["unseen::edge0"] = {"task": "unseen", "edge": "edge0"}
    elif mutation == "foreign_task":
        report["mechanisms"][0]["task"] = "unseen"
    else:
        report["matches"]["mechanism-0"][0]["query"] = "unseen"
    (workspace / "context/student_gaps.json").write_text(json.dumps(report))
    (workspace / "context/teacher_unit_bank.json").write_text(json.dumps(bank))
    with pytest.raises(ValueError):
        SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
