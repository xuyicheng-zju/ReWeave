"""The fragment method reaches the actual author without global graph search."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.engine import SharedEvolution
from reweave.evolution.distillation.fragments import fragment_transfer
from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator
from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace
from tests.unit_tests.harness.evolution.test_shared_topology import graphs_fixture
from tests.unit_tests.harness.evolution.test_shared_topology_integration import (
    shared_plan,
)


def test_inventory_contains_only_direct_edges_and_preserves_failures(tmp_path):
    graphs = graphs_fixture(tmp_path)
    value = fragment_transfer(graphs)
    assert len(value["inventory"]) == sum(len(g.edges) for g in graphs.values())
    for task, graph in graphs.items():
        for edge_id, edge in graph.edges.items():
            row = value["inventory"][f"{task}:{edge_id}"]
            assert row["witnesses"] == [
                w.model_dump(mode="json") for w in edge.witnesses
            ]
            assert row["effects"] == "unknown"
    assert all(not p["callable"] for p in value["proposals"])
    assert all(p["status"] == "synthesis_proposal" for p in value["proposals"])


def test_missing_encoding_is_not_silently_zero_coverage(tmp_path):
    graphs = graphs_fixture(tmp_path)
    graphs["alpha"] = graphs["alpha"].model_copy(update={"matching": None})
    with pytest.raises(ValueError, match="condition encoding"):
        fragment_transfer(graphs)


def test_engine_new_method_never_calls_legacy_matcher_and_recovers(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("global alignment was called")

    monkeypatch.setattr("reweave.evolution.graph.matching.match_graphs", forbidden)
    monkeypatch.setattr(
        "reweave.evolution.distillation.topology.match_graphs", forbidden
    )
    monkeypatch.setattr(
        "reweave.evolution.distillation.transfer.match_graphs", forbidden
    )
    graphs = graphs_fixture(tmp_path)
    p = shared_plan(graphs).model_copy(update={"method": "fragments"})
    author = Author()
    evaluator = Evaluator(lambda r, b: b.record_id != "initial")
    engine = SharedEvolution(tmp_path / "run", author, evaluator)
    result = asyncio.run(engine.run(p))
    assert author.requests
    for request in author.requests:
        assert request["fragment_transfer"]["schema"] == "topology-fragments/v2"
        assert request["fixed_alignments"] == []
        assert set(request["source_graphs"]) == set(graphs)
    before = len(evaluator.requests)
    assert asyncio.run(engine.run(p)) == result
    assert len(evaluator.requests) == before


def test_author_materializes_new_method_input(tmp_path):
    request, _, graphs, _, author = prepared_workspace(tmp_path)
    request["method"] = "fragments"
    request["fragment_transfer"] = fragment_transfer(graphs)
    workspace, _ = author.prepare(request, tmp_path / "fragments_author")
    context = json.loads((workspace / "context/input.json").read_text())
    assert context["fragment_transfer"] == "context/fragment_transfer.json"
    assert (
        json.loads((workspace / context["fragment_transfer"]).read_text())
        == request["fragment_transfer"]
    )
    assert json.loads((workspace / "context/alignments.json").read_text()) == []
    assert "completion_options" in (workspace / "AGENTS.md").read_text()


def test_cross_source_completion_options_keep_conditions_and_unknown_effects(tmp_path):
    from reweave.evolution.contracts import SemanticGraph
    from tests.unit_tests.harness.evolution.test_graph_retrieval import fact

    graphs = graphs_fixture(tmp_path, names=("alpha", "beta", "gamma"))
    for name, graph in list(graphs.items()):
        value = graph.model_dump(mode="json")
        value["matching"]["entities"]["object"]["conditions"] = []
        value["matching"]["interfaces"]["edge0"]["entry"] = (
            [fact("support", "yes")] if name == "beta" else []
        )
        if name == "gamma":
            value["matching"]["nodes"]["b"] = [fact("support", "yes")]
        graphs[name] = SemanticGraph.model_validate(value)
    result = fragment_transfer(graphs)
    proposal = next(
        p
        for p in result["proposals"]
        if p["query"] == "alpha:edge0" and p["reference"] == "beta:edge0"
    )
    assert proposal["relation"] == "requires_completion"
    assert proposal["missing_entry"] == [
        {"fluent": "support", "arguments": ["object"], "values": ["yes"]}
    ]
    completion = next(
        c for c in proposal["completion_options"] if c["fragment"] == "gamma:edge0"
    )
    assert completion["remaining_entry"] == []
    assert not completion["effects_checked"]
    assert not proposal["callable"]
    value = graphs["beta"].model_dump(mode="json")
    for witness in value["edges"]["edge0"]["witnesses"]:
        witness["outcome"] = "not_achieved"
    graphs["beta"] = SemanticGraph.model_validate(value)
    result = fragment_transfer(graphs)
    assert not any(p["reference"] == "beta:edge0" for p in result["proposals"])
    assert "beta:edge0" in result["inventory"]


def test_fragment_author_tools_cannot_fall_back_to_global_search(tmp_path):
    from reweave.evolution.distillation.tools import SharedAuthorTools

    request, _, graphs, _, author = prepared_workspace(tmp_path)
    request["method"] = "fragments"
    request["fragment_transfer"] = fragment_transfer(graphs)
    workspace, _ = author.prepare(request, tmp_path / "fragment_tools")
    tools = SharedAuthorTools(workspace, tmp_path / "audit.jsonl")
    catalog = {t.name for t in asyncio.run(tools.catalog())}
    assert "read_fragment_reuse" in catalog
    assert "retrieve_graphs" not in catalog
    result = asyncio.run(
        tools.model_call("read_fragment_reuse", {"query_task": "alpha"})
    )
    assert all(p["query"].startswith("alpha:") for p in result["proposals"])
    expected = [
        p
        for p in request["fragment_transfer"]["adaptation_proposals"]
        if p["query"].startswith("alpha:")
    ]
    assert result["total"] == len(expected)
    assert result["proposals"] == expected[:20]
    with pytest.raises(ValueError, match="invalid fragment page"):
        asyncio.run(tools.model_call("read_fragment_reuse", {"limit": 0}))
    with pytest.raises(ValueError, match="outside admitted"):
        asyncio.run(tools.model_call("read_fragment_reuse", {"query_task": "unseen"}))
    with pytest.raises(ValueError, match="use read_fragment_reuse"):
        asyncio.run(tools.model_call("retrieve_graphs", {}))


def test_missing_encoding_fails_before_any_evaluation(tmp_path):
    graphs = graphs_fixture(tmp_path)
    p = shared_plan(graphs).model_copy(update={"method": "fragments"})
    p = p.model_copy(
        update={
            "teachers": tuple(
                t.model_copy(
                    update={"graph": t.graph.model_copy(update={"matching": None})}
                )
                for t in p.teachers
            )
        }
    )
    author, evaluator = Author(), Evaluator(lambda r, b: True)
    with pytest.raises(ValueError, match="condition encoding"):
        asyncio.run(SharedEvolution(tmp_path / "unencoded", author, evaluator).run(p))
    assert not author.requests
    assert not evaluator.requests


def test_adaptation_pages_preserve_all_candidates(tmp_path):
    from reweave.evolution.distillation.tools import SharedAuthorTools

    request, _, graphs, _, author = prepared_workspace(tmp_path)
    request["method"] = "fragments"
    report = fragment_transfer(graphs)
    rows = [
        {"query": "alpha:edge0", "reference": "beta:edge0", "serial": i}
        for i in range(23)
    ]
    report["adaptation_proposals"] = rows
    request["fragment_transfer"] = report
    workspace, _ = author.prepare(request, tmp_path / "paged_author")
    tools = SharedAuthorTools(workspace, tmp_path / "paged_audit.jsonl")
    first = asyncio.run(tools.model_call("read_fragment_reuse", {"limit": 7}))
    second = asyncio.run(
        tools.model_call(
            "read_fragment_reuse", {"offset": first["next_offset"], "limit": 100}
        )
    )
    assert first["total"] == 23 and not first["complete"]
    assert second["complete"] and second["next_offset"] is None
    assert first["proposals"] + second["proposals"] == rows
