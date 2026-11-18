"""A ten-point gate protects the newest frontier at equal official success."""

import copy

import pytest
from pydantic import ValidationError

from reweave.evolution.contracts import EvolutionProtocol, GraphAnalysisRequest
from reweave.evolution.graph.selection import current_round_versions, policy_parent
from reweave.evolution.search.matrix import best_version
from tests.unit_tests.harness.evolution.fixtures import protocol
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_references import selection_case


@pytest.mark.parametrize(
    "old,current,blocked", [(6, 6, False), (6, 5, False), (7, 5, False)]
)
def test_equal_sr_progress_gate_boundary(old, current, blocked):
    value, versions = selection_case([[old], [current]])
    audit = {}
    selected = policy_parent(
        value,
        versions,
        protocol(),
        current_versions=(versions[1].version_id,),
        audit=audit,
    )
    assert selected == (versions[1] if old == current or blocked else versions[0])
    assert audit["rollback_blocked"] is blocked


def test_subthreshold_history_does_not_hide_the_best_current_sibling():
    value, versions = selection_case([[6, 5], [1, 1], [5, 5]])
    original = copy.deepcopy(value)
    audit = {}
    assert (
        policy_parent(
            value,
            versions,
            protocol(),
            current_versions=tuple(v.version_id for v in versions[1:]),
            audit=audit,
        )
        == versions[2]
    )
    assert audit["rollback_blocked"] is True
    assert audit["progress_gap"] == pytest.approx(0.05)
    assert audit["current_round_best"] == versions[2].version_id
    assert value == original


def test_same_round_siblings_compare_directly_without_a_margin():
    value, versions = selection_case([[1, 1], [6, 5], [5, 5]])
    assert (
        policy_parent(
            value,
            versions,
            protocol(),
            current_versions=tuple(v.version_id for v in versions[1:]),
        )
        == versions[1]
    )


def test_greater_official_success_bypasses_progress_gate():
    value, versions = selection_case([[10, 0, 0], [7, 7, 7]], [[0], []])
    assert (
        policy_parent(
            value, versions, protocol(), current_versions=(versions[1].version_id,)
        )
        == versions[0]
    )


def test_missing_current_official_evidence_is_not_assumed_to_be_zero_sr():
    value, versions = selection_case([[6], [5]])
    value["episodes"][value["cells"][versions[1].version_id]["0"]][
        "official_success"
    ] = None
    assert (
        policy_parent(
            value, versions, protocol(), current_versions=(versions[1].version_id,)
        )
        == versions[0]
    )


@pytest.mark.parametrize("purpose", ["search", "final_selection"])
def test_graph_and_sr_only_selection_keep_partial_official_evidence(purpose):
    value, versions = selection_case([[0, 0, 0], [9, 9, 9]], [[], [0]])
    incomplete = value["episodes"][value["cells"][versions[1].version_id]["1"]]
    incomplete["official_success"] = None
    archive = {v.version_id: v for v in versions}

    def selections():
        rows = {
            v.bundle_id: tuple(
                value["episodes"][receipt]["official_success"]
                for receipt in value["cells"][v.version_id].values()
            )
            for v in versions
        }
        graph_parent = policy_parent(
            value,
            versions,
            protocol(),
            purpose=purpose,
            current_versions=(versions[1].version_id,),
        )
        return graph_parent.version_id, best_version(tuple(archive), archive, rows)

    # A settled partial row keeps its confirmed success and does not block search.
    assert selections() == (versions[1].version_id, versions[1].version_id)
    # Settling that missing result preserves the same shared official ranking.
    incomplete["official_success"] = False
    assert selections() == (versions[1].version_id, versions[1].version_id)


def test_sr_only_preserves_official_coverage_tie_without_topology_scores():
    value, versions = selection_case([[5, 5, 5], [5, 5, 5]], [[0], [0]])
    value["episodes"][value["cells"][versions[1].version_id]["1"]][
        "official_success"
    ] = None
    rows = {
        versions[0].bundle_id: (True, False, False),
        versions[1].bundle_id: (True, None, False),
    }
    archive = {v.version_id: v for v in versions}
    assert policy_parent(value, versions, protocol()) == versions[0]
    assert best_version(tuple(archive), archive, rows) == versions[0].version_id


def test_gate_is_search_only_final_archive_still_prefers_success_and_compactness():
    value, versions = selection_case([[6, 5], [5, 5]])
    assert (
        policy_parent(
            value,
            versions,
            protocol(),
            purpose="final_selection",
            current_versions=(versions[1].version_id,),
        )
        == versions[0]
    )
    value, versions = selection_case([[10], [10]], [[0], [0]])
    assert (
        policy_parent(
            value,
            versions,
            protocol(),
            purpose="final_selection",
            current_versions=(versions[1].version_id,),
            successful_sizes={versions[0].version_id: 10, versions[1].version_id: 20},
        )
        == versions[0]
    )


def test_current_batch_uses_proposal_rounds_not_names_or_publication_order():
    state = {
        "order": ["h0", "z", "latest-looking-name", "a"],
        "versions": {
            "h0": {"proposal_id": None},
            "z": {"proposal_id": "p2b"},
            "latest-looking-name": {"proposal_id": "p1"},
            "a": {"proposal_id": "p2a"},
        },
        "proposals": {
            "p1": {"round_index": 1},
            "p2a": {"round_index": 2},
            "p2b": {"round_index": 2},
        },
    }
    assert current_round_versions(state) == ("z", "a")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("inf"), float("nan")])
def test_gate_configuration_rejects_invalid_progress_scales(threshold):
    with pytest.raises(ValidationError):
        EvolutionProtocol.model_validate(
            {**protocol().model_dump(), "parent_rollback_min_progress": threshold}
        )


def test_request_freezes_and_validates_current_batch(tmp_path):
    request, *_ = analysis_request(tmp_path)
    raw = request.model_dump(mode="json")
    raw["current_round_versions"] = [request.versions[0].version_id]
    frozen = GraphAnalysisRequest.model_validate(raw)
    assert GraphAnalysisRequest.model_validate_json(frozen.model_dump_json()) == frozen
    for ids in (["absent"], [request.versions[0].version_id] * 2):
        with pytest.raises(ValidationError, match="current-round versions"):
            GraphAnalysisRequest.model_validate({**raw, "current_round_versions": ids})
