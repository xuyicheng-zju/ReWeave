# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Task progress excludes free background without discarding real failed work."""

import pytest
from pydantic import ValidationError

from reweave.evolution.contracts import SemanticGraph, StateSpan
from tests.unit_tests.harness.evolution.fixtures import artifact
from tests.unit_tests.harness.evolution.test_semantic_graph import (
    case,
    first_cell,
    mapped,
    validate,
)


def background_case(tmp_path, path):
    """Replace the synthetic free start with witnessed background intervals."""
    request, raw, graph = case(tmp_path, paths=(path,))
    payload = graph.model_dump(mode="json")
    del payload["nodes"]["start"]
    for spans in payload["episodes"].values():
        for span in spans:
            if span["node_id"] == "start":
                span.update(status="background", node_id=None)
    for edge in payload["edges"].values():
        if edge["source"] == "start":
            edge["source"] = None
        if edge["target"] == "start":
            edge["target"] = "held"
            for witness in edge["witnesses"]:
                witness["outcome"] = "not_achieved"
    return request, raw, SemanticGraph.model_validate(payload)


@pytest.mark.parametrize("length", [2, 20])
def test_background_and_wasted_attempts_are_known_zero_without_graph_nodes(
    tmp_path, length
):
    request, raw, graph = background_case(tmp_path, ("start",) * length)
    result = mapped(request, raw, graph)
    cell = first_cell(result)
    assert "start" not in graph.nodes and "start" not in result["node_evidence"]
    assert all(not arrivals for arrivals in result["node_evidence"].values())
    assert cell["progress"] == cell["peak_progress"] == 0
    assert cell["completed_nodes"] == cell["attained_nodes"] == []
    assert cell["terminal_state_known"] and cell["frontier"] is None
    assert sum(
        span["last_frame"] - span["first_frame"] + 1 for span in cell["state_spans"]
    ) == length and cell["path_changes"] == [None]
    assert all(s["status"] == "background" for s in cell["state_spans"])
    assert len(cell["transition_ids"]) == length - 1
    assert all(
        w["outcome"] == "not_achieved"
        for w in (result["transitions"][key] for key in cell["transition_ids"])
    )
    assert cell["usage"] == raw.episodes[0].usage.model_dump(mode="json")


def test_first_earned_node_needs_no_synthetic_start_and_regression_loses_progress(
    tmp_path,
):
    request, raw, graph = background_case(tmp_path, ("start", "held", "start"))
    result = mapped(request, raw, graph)
    cell = first_cell(result)
    assert graph.edges["start_held"].source is None
    assert cell["path_changes"] == [None, "held", None]
    assert cell["peak_progress"] == 0.5 and cell["progress"] == 0
    assert cell["terminal_state_known"]
    assert [
        w["outcome"]
        for w in (result["transitions"][key] for key in cell["transition_ids"])
    ] == ["achieved", "not_achieved"]
    assert result["node_evidence"]["held"][0]["reached_episode_ids"] == [
        raw.episodes[0].receipt_id
    ]


def test_unknown_terminal_is_not_converted_to_background_zero(tmp_path):
    request, raw, graph = background_case(tmp_path, ("start", "start"))
    payload = graph.model_dump(mode="json")
    spans = next(iter(payload["episodes"].values()))
    spans[-1].update(status="unknown", uncertainty="Final observation is missing")
    witness = payload["edges"]["start_start"]["witnesses"][0]
    witness["outcome"] = "unknown"
    cell = first_cell(mapped(request, raw, SemanticGraph.model_validate(payload)))
    assert cell["progress"] is None and cell["completed_nodes"] is None
    assert not cell["terminal_state_known"]
    assert [s["status"] for s in cell["state_spans"]] == ["background", "unknown"]


def test_unknown_start_cannot_be_used_as_known_background_mechanism(tmp_path):
    request, _, graph = background_case(tmp_path, ("start", "held"))
    payload = graph.model_dump(mode="json")
    next(iter(payload["episodes"].values()))[0].update(
        status="unknown", uncertainty="Missing initial evidence"
    )
    with pytest.raises(ValueError, match="cannot start from unknown"):
        validate(request, SemanticGraph.model_validate(payload))


def test_unsettled_physical_tail_overrides_even_known_background_assignment(tmp_path):
    request, raw, graph = background_case(tmp_path, ("start", "start"))
    raw = raw.model_copy(
        update={
            "episodes": (
                raw.episodes[0].model_copy(update={"terminal_state_known": False}),
            )
        }
    )
    request = request.model_copy(
        update={
            "graph": artifact(
                tmp_path, "unknown-tail.json", raw.model_dump(mode="json")
            )
        }
    )
    value = mapped(request, raw, graph)
    cell = first_cell(value)
    assert cell["progress"] is None and not cell["terminal_state_known"]


@pytest.mark.parametrize(
    "status,node_id,reason",
    [
        ("node", None, ""),
        ("background", "held", ""),
        ("unknown", "held", "Missing evidence"),
        ("unknown", None, ""),
    ],
)
def test_span_status_cannot_conflate_progress_background_and_unknown(
    status, node_id, reason
):
    with pytest.raises(ValidationError, match="semantic state interval"):
        StateSpan(
            first_frame=0,
            last_frame=0,
            status=status,
            node_id=node_id,
            uncertainty=reason,
        )


def test_span_requires_explicit_status_instead_of_guessing_from_null():
    with pytest.raises(ValidationError, match="status"):
        StateSpan.model_validate(
            {"first_frame": 0, "last_frame": 0, "node_id": None, "uncertainty": ""}
        )
