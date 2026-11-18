# Modified for the ReWeave namespace.
"""Global parent selection, historical source reuse and complete inheritance."""

import copy
import itertools
import json

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.candidate import read_candidate, read_references
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import Proposal, RoundDecision, VersionReference
from reweave.evolution.graph.projection import map_row
from reweave.evolution.graph.selection import (
    parent_statistics,
    policy_parent,
    select_parent,
)
from reweave.evolution.validation.admission import admit_candidate
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    author_candidate,
    baseline,
    mutated,
    protocol,
)
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def selection_case(rows, successes=None):
    versions = tuple(
        VersionReference(version_id=f"version-{i:03d}", bundle_id=f"bundle-{i:03d}")
        for i in range(len(rows))
    )
    episodes = {}
    cells = {}
    for i, (v, row) in enumerate(zip(versions, rows)):
        cells[v.version_id] = {}
        for seed, count in enumerate(row):
            receipt = f"receipt-{i:03d}-{seed:03d}"
            cells[v.version_id][str(seed)] = receipt
            episodes[receipt] = {
                "receipt_id": receipt,
                "bundle_id": v.bundle_id,
                "condition_id": str(seed),
                "progress": None if count is None else count / 10,
                "completed_nodes": None
                if count is None
                else [f"milestone-{j}" for j in range(count)],
                "peak_progress": 1,
                "official_success": bool(successes and seed in successes[i]),
            }
    return {
        "cells": cells,
        "episodes": episodes,
        "versions": {v.version_id: v.bundle_id for v in versions},
    }, versions


def test_parent_uses_all_seeds_terminal_progress_when_success_counts_tie():
    value, versions = selection_case(
        [[3, 3, 3, 3], [5, 0, 0, 0], [5, None, None, None]],
    )
    original = copy.deepcopy(value)
    for order in itertools.permutations(versions):
        assert select_parent(value, order) == versions[0]
    stats = parent_statistics(map_row(value, versions[2].version_id))
    assert stats["parent_mean_progress"] == 0.125
    assert stats["parent_unknown_conditions"] == 3
    assert value == original  # Unknown physical results remain null.


def test_parent_prioritizes_official_success_over_mean_topology_progress():
    value, versions = selection_case([[7, 7, 7], [10, 0, 0]], [[], [0]])
    # The 1/3-success candidate wins despite lower mean progress (1/3 vs 0.7).
    for order in itertools.permutations(versions):
        assert select_parent(value, order) == versions[1]


def test_equal_success_counts_use_retained_progress_to_resolve_the_frontier():
    value, versions = selection_case([[10, 2, 2], [10, 4, 1]], [[0], [0]])
    for order in itertools.permutations(versions):
        assert select_parent(value, order) == versions[1]


def test_confirmed_success_priority_does_not_fill_unknown_official_outcomes():
    value, versions = selection_case([[10, None, None], [7, 7, 7]], [[0], []])
    for condition in ("1", "2"):
        receipt = value["cells"][versions[0].version_id][condition]
        value["episodes"][receipt]["official_success"] = None
    assert select_parent(value, versions) == versions[0]
    stats = parent_statistics(map_row(value, versions[0].version_id))
    assert stats["parent_success_rate"] is None
    assert stats["parent_unknown_conditions"] == 2


def test_final_successful_ties_choose_smaller_implementation_before_recency():
    value, versions = selection_case([[10, 10], [10, 10]], [[0, 1], [0, 1]])
    sizes = {versions[0].version_id: 100, versions[1].version_id: 200}
    assert policy_parent(value, versions, protocol()) == versions[1]
    assert (
        policy_parent(
            value,
            versions,
            protocol(),
            purpose="final_selection",
            successful_sizes=sizes,
        )
        == versions[0]
    )
    sizes[versions[1].version_id] = 100
    assert select_parent(value, versions, successful_sizes=sizes) == versions[1]


@pytest.mark.parametrize(
    "rows,successes", [([[7, 7], [10, 10]], [[], [0, 1]]), ([[2, 2], [4, 4]], None)]
)
def test_final_size_never_overrides_success_or_failed_candidate_progress(
    rows, successes
):
    value, versions = selection_case(rows, successes)
    sizes = {versions[0].version_id: 1, versions[1].version_id: 10000}
    assert select_parent(value, versions, successful_sizes=sizes) == versions[1]


def test_final_size_does_not_change_unsuccessful_exact_ties():
    value, versions = selection_case([[4, 4], [4, 4]])
    sizes = {versions[0].version_id: 1, versions[1].version_id: 10000}
    assert select_parent(value, versions, successful_sizes=sizes) == versions[1]
    with pytest.raises(ValueError, match="only a final-selection"):
        policy_parent(value, versions, protocol(), successful_sizes=sizes)


def test_smaller_dev_success_cannot_override_feedback_validation_success():
    value, versions = selection_case([[10, 10], [10, 10]], [[0, 1], [0, 1]])
    value["validation_cells"] = {}
    for index, version in enumerate(versions):
        receipt = f"feedback-receipt-{index}"
        value["validation_cells"][version.version_id] = {"feedback": receipt}
        value["episodes"][receipt] = {
            "receipt_id": receipt,
            "bundle_id": version.bundle_id,
            "condition_id": "feedback",
            "completed_nodes": list(range(10 if index else 7)),
            "progress": 1.0 if index else 0.7,
            "official_success": bool(index),
        }
    sizes = {versions[0].version_id: 1, versions[1].version_id: 10000}
    assert select_parent(value, versions, successful_sizes=sizes) == versions[1]


def test_parent_ties_use_success_then_the_newest_version():
    value, versions = selection_case([[2, 2], [4, 0], [4, 0]], [[], [0], [0]])
    older, newer = versions[1], versions[2]
    assert select_parent(value, versions) == newer
    assert select_parent(value, (versions[0], newer, older)) == older
    # Repeated executable content is one candidate, represented by its newest version.
    alias = VersionReference(version_id="alias", bundle_id=older.bundle_id)
    value["cells"][alias.version_id] = copy.deepcopy(value["cells"][older.version_id])
    value["versions"][alias.version_id] = alias.bundle_id
    assert select_parent(value, (*versions, alias)) == alias
    assert select_parent(value, (alias, *versions)) == newer


def test_new_graph_rescores_old_versions_and_can_change_the_parent():
    value, versions = selection_case([[3, 3], [4, 0]])
    assert select_parent(value, versions) == versions[0]
    refreshed = copy.deepcopy(value)
    for cell in map_row(refreshed, versions[0].version_id).values():
        cell["progress"] = 0.1
        cell["completed_nodes"] = ["milestone-0"]
    assert select_parent(refreshed, versions) == versions[1]
    assert select_parent(value, versions) == versions[0]


def test_normalization_rounding_does_not_change_equal_progress_ties():
    value, versions = selection_case([[1, 1, 1], [3, 0, 0]])
    for order in itertools.permutations(versions):
        assert select_parent(value, order) == order[-1]


def bind_references(root, bundles):
    """Bind synthetic complete sources and their evaluation identities."""
    path = host_input(root, "proposal.json")
    proposal = json.loads(path.read_text())
    receipt = proposal["evidence"][0]
    proposal["references"], proposal["evidence"] = [], []
    for i, bundle in enumerate(bundles):
        if i == 0:
            proposal["references"].append(
                {"version_id": f"reference-{i:03d}", "bundle_id": bundle.record_id}
            )
        record = copy.deepcopy(receipt)
        record["record_id"] = f"reference-observation-{i:03d}"
        record["request"]["record_id"] = f"reference-evaluation-{i:03d}"
        record["request"]["bundle_id"] = bundle.record_id
        proposal["evidence"].append(record)
    proposal["parent"] = proposal["references"][0]
    atomic_json(path, Proposal.model_validate(proposal).model_dump(mode="json"))
    atomic_json(
        host_input(root, "references.json"),
        {bundles[0].record_id: bundles[0].model_dump(mode="json")},
    )


def composition(root, references):
    bind_references(root, references)
    value = references[0].model_dump(mode="json")
    value["record_id"] = "fixture-composition"
    value["loop"] = references[1].loop.model_dump(mode="json")
    candidate = AgentBundle.model_validate(value)
    plan = json.loads((root / "candidate/plan.json").read_text())
    author_candidate(candidate, root / "candidate")
    plan.update(
        scopes=["L"],
        summary="Combine complete known C and L components; synthetic composition only.",
    )
    atomic_json(root / "candidate/plan.json", plan)
    return candidate


def test_new_composition_of_unchanged_components_passes_real_admission(tmp_path):
    import asyncio

    root = workspace(tmp_path)
    references = (mutated(baseline(), "C"), mutated(baseline(), "L"))
    candidate = composition(root, references)
    assert candidate.novel_scopes(references) == frozenset()
    assert all(not candidate.same_harness(reference) for reference in references)
    cases = []
    assert asyncio.run(admit_candidate(root, case_results=cases)).same_harness(
        candidate
    )
    report = next(c for c in cases if c["scope"] == "interaction")
    assert report["coverage"]["C"] and report["coverage"]["L"]
    assert not report["physical_execution_tested"]


def test_returning_the_parent_unchanged_is_rejected(tmp_path):
    root = workspace(tmp_path)
    references = (mutated(baseline(), "C"), mutated(baseline(), "L"))
    composition(root, references)
    plan = (root / "candidate/plan.json").read_text()
    author_candidate(references[0], root / "candidate")
    (root / "candidate/plan.json").write_text(plan)
    with pytest.raises(ValueError, match="differ from every reference"):
        read_candidate(root)


def test_reference_inventory_rejects_a_missing_published_reference(tmp_path):
    root = workspace(tmp_path)
    refs = [baseline(), mutated(baseline(), "C")]
    bind_references(root, refs)
    path = host_input(root, "references.json")
    payload = json.loads(path.read_text())
    payload.pop(refs[0].record_id)
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="reference inventory"):
        read_references(root)


def test_reference_records_expose_actual_values_without_source_certification(tmp_path):
    root = workspace(tmp_path)
    refs = [baseline(), mutated(baseline(), "C")]
    bind_references(root, refs)
    path = host_input(root, "references.json")
    payload = json.loads(path.read_text())
    payload[refs[0].record_id]["context"]["instructions"] += " recorded change"
    payload[refs[0].record_id]["runtime"]["physical_steps"] += 1
    atomic_json(path, payload)
    actual = read_references(root)
    assert actual[0].context.instructions.endswith(" recorded change")
    assert actual[0].runtime.physical_steps == refs[0].runtime.physical_steps + 1


@pytest.mark.parametrize(
    "field,value",
    [("parent_version_id", "a" * 64), ("parent_bundle_id", "b" * 64), ("donors", [])],
)
def test_proposal_rejects_removed_roles(tmp_path, field, value):
    root = workspace(tmp_path)
    proposal = json.loads(host_input(root, "proposal.json").read_text())
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Proposal.model_validate({**proposal, field: value})


def test_decision_rejects_removed_roles(tmp_path):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    decision = {
        "record_id": "fixture-decision",
        "request_id": "fixture-graph-request",
        "analysis": proposal.graph,
        "graph": proposal.graph,
        "progress_map": proposal.progress_map,
        "references": proposal.references,
        "parent": proposal.parent,
        "width": 1,
        "candidate_cost": {
            "rollout_usd": None,
        },
        "alternatives": [{"width": 1.0}],
        "statistics": {},
        "reason": "Synthetic test",
    }
    RoundDecision.model_validate(decision)
    for model, payload in (
        (RoundDecision, decision),
        (Proposal, proposal.model_dump(mode="json")),
    ):
        missing = dict(payload)
        missing.pop("parent")
        with pytest.raises(ValidationError, match="Field required"):
            model.model_validate(missing)
        with pytest.raises(ValidationError, match="starting reference must be exactly"):
            model.model_validate(
                {
                    **payload,
                    "parent": VersionReference(version_id="c" * 64, bundle_id="d" * 64),
                }
            )
    with pytest.raises(ValidationError, match="at most 1 item"):
        RoundDecision.model_validate(
            {**decision, "references": proposal.references * 2}
        )
    for field in (
        "parent_version_id",
        "parent_bundle_id",
        "donors",
        "standard_cost_s",
    ):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            RoundDecision.model_validate({**decision, field: []})


@pytest.mark.parametrize("declared", [False, True])
def test_borrowed_workflow_loop_pair_requires_real_recovery(tmp_path, declared):
    import asyncio

    from reweave.evolution.validation.admission import check_interaction_suite
    from tests.unit_tests.harness.evolution.test_author_interaction import (
        OLD_GATE,
        candidate,
        contract,
        scenario,
    )

    root = workspace(tmp_path)
    references = (candidate(OLD_GATE), mutated(baseline(), "L"))
    combined = composition(root, references)
    assert not combined.novel_scopes(references)
    atomic_json(root / "context/gateway.json", contract())
    case = scenario()
    case["recovery"] = declared
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"}
    ]
    atomic_json(root / ".scratch/tests/interaction.json", [case])

    async def run():
        return await check_interaction_suite(
            root, combined, scenario_path=".scratch/tests/interaction.json"
        )

    if declared:
        reports = asyncio.run(run())
        assert reports[0]["validation_status"] == "passed"
        assert reports[0]["coverage"]["workflows"] == ["probe"]
    else:
        with pytest.raises(ValueError, match="requires a recovery=true"):
            asyncio.run(run())


@pytest.mark.parametrize("duplicate", [False, True])
def test_proposal_rejects_additional_starting_references(tmp_path, duplicate):
    root = workspace(tmp_path)
    proposal = json.loads(host_input(root, "proposal.json").read_text())
    extra = (
        proposal["references"][0]
        if duplicate
        else {"version_id": "other", "bundle_id": "other-content"}
    )
    proposal["references"].append(extra)
    with pytest.raises(ValidationError, match="at most 1 item"):
        Proposal.model_validate(proposal)
