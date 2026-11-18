# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""One task backbone, optional physical branches and exact Harness provenance."""

import copy

import pytest
from pydantic import ValidationError

from reweave.evolution.contracts import SemanticGraph, VersionReference
from reweave.evolution.graph.projection import map_row, project_map
from reweave.evolution.graph.selection import select_parent
from tests.unit_tests.harness.evolution.test_semantic_graph import (
    case,
    first_cell,
    mapped,
    validate,
)


def rename_frames(payload, episode_id, replacements):
    """Preserve each real interval while changing its synthetic state interpretation."""
    spans = payload["episodes"][episode_id]
    for index, name in replacements.items():
        spans[index]["node_id"] = name
    edges = {}
    for edge in payload["edges"].values():
        for witness in edge["witnesses"]:
            path = payload["episodes"][witness["episode_id"]]
            source = path[witness["first_frame"]]["node_id"]
            target = path[witness["last_frame"]]["node_id"]
            key = source + "_" + target
            if key not in edges:
                edges[key] = {
                    **copy.deepcopy(edge),
                    "source": source,
                    "target": target,
                    "witnesses": [],
                }
            edges[key]["witnesses"].append(witness)
    payload["edges"] = edges


def test_first_analysis_can_establish_unobserved_required_backbone_without_success(
    tmp_path,
):
    request, raw, graph = case(tmp_path, paths=(("start", "held"),))
    result = mapped(request, raw, graph)
    assert not raw.episodes[0].official_success
    assert graph.nodes["open"].role == "milestone"
    assert not graph.nodes["open"].references
    assert graph.nodes["open"].uncertainty
    assert first_cell(result)["progress"] == 0.5
    assert result["milestone_nodes"] == ["held", "near", "open"]
    assert result["node_evidence"]["open"] == []
    assert "progress_points" not in result
    assert "z" not in first_cell(result)


def test_extra_optional_branch_nodes_do_not_improve_progress_parent_or_width(tmp_path):
    request, raw, graph = case(
        tmp_path,
        paths=(("start", "held", "open"), ("start", "held", "held", "held", "open")),
    )
    original = mapped(request, raw, graph)
    payload = graph.model_dump(mode="json")
    for name in ("grasp_variant", "alternate_servo"):
        payload["nodes"][name] = {
            **copy.deepcopy(payload["nodes"]["held"]),
            "role": "branch",
            "predecessors": ["held"],
            "meaning": "Optional physical configuration; same retained task achievement",
        }
    rename_frames(
        payload, raw.episodes[1].receipt_id, {2: "grasp_variant", 3: "alternate_servo"}
    )
    result = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert [
        next(iter(map_row(result, version).values()))["progress"]
        for version in result["cells"]
    ] == [0.75, 0.75]
    assert select_parent(result, request.versions) == select_parent(
        original, request.versions
    )


def test_new_required_milestone_reprojects_every_historical_harness(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "held"), ("start", "open")))
    old = mapped(request, raw, graph)
    payload = graph.model_dump(mode="json")
    payload["nodes"]["clearance"] = {
        "meaning": "New required physical achievement between holding and opening",
        "role": "milestone",
        "predecessors": ["held"],
        "references": [],
        "uncertainty": "Synthetic new backbone hypothesis; not directly observed",
    }
    payload["nodes"]["open"]["predecessors"] = ["clearance"]
    new = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert [
        next(iter(map_row(old, version).values()))["progress"]
        for version in old["cells"]
    ] == [
        0.5,
        0.75,
    ]
    assert [
        next(iter(map_row(new, version).values()))["progress"]
        for version in new["cells"]
    ] == [
        0.4,
        0.8,
    ]
    assert new["graph_id"] is old["graph_id"] is None
    assert new["milestone_nodes"] != old["milestone_nodes"]
    assert new["execution_graph_id"] == old["execution_graph_id"]


def test_all_success_modes_have_full_progress_including_released_goal(tmp_path):
    request, raw, graph = case(
        tmp_path, paths=(("start", "held", "goal"),) * 2, success=True
    )
    payload = graph.model_dump(mode="json")
    payload["nodes"]["released_goal"] = {
        **copy.deepcopy(payload["nodes"]["goal"]),
        "meaning": "Object correctly placed and gripper released; official success",
    }
    rename_frames(payload, raw.episodes[1].receipt_id, {2: "released_goal"})
    result = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert [
        next(iter(map_row(result, version).values()))["progress"]
        for version in result["cells"]
    ] == [1, 1]
    assert len(result["node_evidence"]["released_goal"]) == 1
    assert result["node_evidence"]["released_goal"][0]["completed_episode_ids"] == [
        raw.episodes[1].receipt_id
    ]


def test_node_evidence_distinguishes_arrival_and_same_episode_completion(tmp_path):
    request, raw, graph = case(
        tmp_path,
        paths=(("start", "held", "goal"), ("start", "held", "open")),
        success=(True, False),
    )
    alias = VersionReference(
        version_id="a" * 64, bundle_id=request.versions[0].bundle_id
    )
    result = project_map(raw, graph, (*request.versions, alias), request.protocol.dev)
    evidence = {item["bundle_id"]: item for item in result["node_evidence"]["held"]}
    assert len(evidence) == 2  # A version alias does not add another physical sample.
    completed = evidence[request.versions[0].bundle_id]
    failed = evidence[request.versions[1].bundle_id]
    assert (
        completed["completed_episode_ids"]
        == completed["reached_episode_ids"]
        == [raw.episodes[0].receipt_id]
    )
    assert failed["reached_episode_ids"] == [raw.episodes[1].receipt_id]
    assert failed["completed_episode_ids"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "cycle",
        "branch_predecessor",
        "unknown_predecessor",
        "missing_goal_ancestor",
        "duplicate",
    ],
)
def test_invalid_backbone_links_are_rejected(tmp_path, defect):
    _, _, graph = case(tmp_path)
    payload = graph.model_dump(mode="json")
    if defect == "cycle":
        payload["nodes"]["near"]["predecessors"] = ["open"]
    elif defect == "branch_predecessor":
        payload["nodes"]["held"]["predecessors"] = ["start"]
    elif defect == "unknown_predecessor":
        payload["nodes"]["held"]["predecessors"] = ["missing"]
    elif defect == "missing_goal_ancestor":
        payload["nodes"]["goal"]["predecessors"] = ["near"]
    else:
        payload["nodes"]["held"]["predecessors"] = ["near", "near"]
    with pytest.raises(ValidationError):
        SemanticGraph.model_validate(payload)


@pytest.mark.parametrize("field", ["progress_points", "achieved_points", "score"])
def test_separate_fact_scores_and_model_ratings_are_rejected(tmp_path, field):
    _, _, graph = case(tmp_path)
    payload = graph.model_dump(mode="json")
    if field == "progress_points":
        payload[field] = {"contact": "one point"}
    else:
        payload["nodes"]["held"][field] = (
            ["contact"] if field == "achieved_points" else 5
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SemanticGraph.model_validate(payload)


def test_unobserved_branch_rejected_and_backbone_uncertainty_required(tmp_path):
    request, _, graph = case(tmp_path)
    payload = graph.model_dump(mode="json")
    payload["nodes"]["unobserved_branch"] = {
        **payload["nodes"]["start"],
        "references": [],
    }
    with pytest.raises(ValueError, match="branch nodes require observed"):
        validate(request, SemanticGraph.model_validate(payload))
    payload["nodes"].pop("unobserved_branch")
    payload["nodes"]["near"]["uncertainty"] = ""
    with pytest.raises(ValueError, match="unobserved backbone"):
        validate(request, SemanticGraph.model_validate(payload))
