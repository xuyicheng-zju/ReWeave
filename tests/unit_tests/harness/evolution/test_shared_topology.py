"""Cross-task sharing must retain private constraints and exact original evidence."""

import pytest

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.distillation.engine import SharedTeacher
from reweave.evolution.distillation.topology import (
    SharedTopology,
    build_topology,
    feedback_splits,
    occurrence_key,
    reconstruct,
    revise_topology,
    validate_topology,
)
from reweave.evolution.distillation.transfer import aligned_references
from tests.unit_tests.harness.evolution.test_graph_retrieval import corpus_graph, fact
from tests.unit_tests.harness.evolution.test_shared import bundle


def graphs_fixture(tmp_path, names=("alpha", "beta")):
    result = {}
    for name in names:
        directory = tmp_path / name
        directory.mkdir()
        value = corpus_graph(directory).model_dump(mode="json")
        value["matching"]["interfaces"]["edge0"]["entry"] = [
            fact("grasp", "secure" if name == "alpha" else "loose")
        ]
        value["matching"]["entities"]["object"]["conditions"] = [
            fact("support", "yes" if name == "alpha" else "no")
        ]
        value["matching"]["uncertainties"] = ["private uncertainty: " + name]
        result[name] = SemanticGraph.model_validate(value)
    return result


def teachers(graphs):
    return tuple(
        SharedTeacher(
            task=name, bundle=bundle(name), learned_tasks=(name,), graph=graph
        )
        for name, graph in graphs.items()
    )


def initial_topology(graphs):
    refs, alignments = aligned_references(teachers(graphs), seconds=0.1)
    return build_topology(graphs, alignments), refs


def merge_pair(topology, graphs, edge="edge0"):
    return revise_topology(
        topology,
        graphs,
        merges=[
            {
                "left": occurrence_key("alpha", edge),
                "right": occurrence_key("beta", edge),
                "entity_mapping": {"object": "object"},
            }
        ],
        reason="share implementation, retain distinct guards",
    )


def test_private_guards_qualifications_and_full_evidence_roundtrip(tmp_path):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    topology = merge_pair(topology, graphs)
    assert any(len(group.members) > 1 for group in topology.mechanisms.values())
    for name, graph in graphs.items():
        assert reconstruct(topology, name) == graph
    # Source-only entry requirements did not become a common conjunction.
    group = next(
        g
        for g in topology.mechanisms.values()
        if {m.key for m in g.members} >= {occurrence_key(n, "edge0") for n in graphs}
    )
    assert not group.query.interfaces["step"].entry
    for source in topology.sources.values():
        assert source.interfaces["edge0"]["entry"].residual
    restored = SharedTopology.model_validate_json(topology.model_dump_json())
    validate_topology(restored, graphs)


def test_unmatched_and_single_task_mechanisms_have_retention_references(tmp_path):
    graphs = graphs_fixture(tmp_path, names=("alpha",))
    topology, refs = initial_topology(graphs)
    assert {edge for ref in refs for edge in ref.source_edges} == set(
        graphs["alpha"].edges
    )
    assert len(topology.mechanisms) == len(graphs["alpha"].edges)
    assert reconstruct(topology, "alpha") == graphs["alpha"]


@pytest.mark.parametrize(
    "mutation", ["guard", "dependency", "witness", "qualification", "missing_interface"]
)
def test_dropping_private_information_is_rejected(tmp_path, mutation):
    graphs = graphs_fixture(tmp_path)
    topology = merge_pair(initial_topology(graphs)[0], graphs)
    value = topology.model_dump(mode="json")
    source = value["sources"]["alpha"]
    if mutation == "guard":
        source["interfaces"]["edge0"]["entry"]["residual"] = []
    elif mutation == "dependency":
        source["skeleton"]["nodes"]["b"]["predecessors"] = []
    elif mutation == "witness":
        source["skeleton"]["edges"]["edge0"]["witnesses"][0]["uncertainty"] = (
            "rewritten history"
        )
    elif mutation == "qualification":
        source["entities"]["object"]["conditions"] = []
    else:
        del source["interfaces"]["edge0"]
    with pytest.raises((ValueError, KeyError)):
        validate_topology(SharedTopology.model_validate(value), graphs)


def test_pairwise_commonality_cannot_be_transitively_collapsed(tmp_path):
    graphs = graphs_fixture(tmp_path, names=("alpha", "beta", "gamma"))
    for name, graph in list(graphs.items()):
        value = graph.model_dump(mode="json")
        value["matching"]["entities"]["object"]["conditions"] = []
        value["matching"]["nodes"]["a"] = {
            "alpha": [fact("phase", "held"), fact("support", "yes")],
            "beta": [fact("phase", "held")],
            "gamma": [fact("support", "yes")],
        }[name]
        graphs[name] = SemanticGraph.model_validate(value)
    topology = build_topology(graphs, [])
    topology = merge_pair(topology, graphs)
    with pytest.raises(ValueError, match="jointly supported"):
        revise_topology(
            topology,
            graphs,
            merges=[
                {
                    "left": occurrence_key("alpha", "edge0"),
                    "right": occurrence_key("gamma", "edge0"),
                    "entity_mapping": {"object": "object"},
                }
            ],
        )


def test_role_cycle_cannot_reassign_a_shared_entity(tmp_path):
    graphs = graphs_fixture(tmp_path)
    for name, graph in list(graphs.items()):
        value = graph.model_dump(mode="json")
        value["matching"]["entities"]["other"] = {
            "description": "another physical object"
        }
        graphs[name] = SemanticGraph.model_validate(value)
    topology = merge_pair(initial_topology(graphs)[0], graphs)
    with pytest.raises(ValueError, match="cycle"):
        revise_topology(
            topology,
            graphs,
            merges=[
                {
                    "left": occurrence_key("alpha", "edge0"),
                    "right": occurrence_key("beta", "edge0"),
                    "entity_mapping": {"object": "other"},
                }
            ],
        )


def test_feedback_splits_only_implicated_usage_and_preserves_every_source(tmp_path):
    graphs = graphs_fixture(tmp_path)
    topology = merge_pair(initial_topology(graphs)[0], graphs)
    split = feedback_splits(
        topology, {"alpha": 0, "beta": 1}, {"alpha": 1, "beta": 1}, 0
    )
    assert split and all('"alpha"' in key for key in split)
    revised = revise_topology(
        topology, graphs, split=split, reason="measured source regression"
    )
    assert revised.revision == topology.revision + 1
    for name, graph in graphs.items():
        assert reconstruct(revised, name) == graph
    assert all(
        not ({m.task for m in g.members} == {"alpha", "beta"})
        for g in revised.mechanisms.values()
    )


def test_zero_proposal_budget_is_truthful_and_keeps_unique_capabilities(tmp_path):
    graphs = graphs_fixture(tmp_path)
    _, alignments = aligned_references(teachers(graphs), seconds=0.1)
    topology = build_topology(graphs, alignments, max_proposals=0)
    assert topology.search["examined"] == 0
    assert not topology.search["globally_optimal"]
    assert len(topology.mechanisms) == sum(len(g.edges) for g in graphs.values())
    validate_topology(topology, graphs)


def test_observed_transition_does_not_invent_required_progress_dependency(tmp_path):
    from reweave.evolution.distillation.topology import fragment

    graphs = graphs_fixture(tmp_path, names=("alpha",))
    value = graphs["alpha"].model_dump(mode="json")
    value["nodes"]["b"]["predecessors"] = []
    value["nodes"]["c"]["predecessors"] = ["a", "b"]
    graphs["alpha"] = SemanticGraph.model_validate(value)
    query = fragment(graphs["alpha"], "edge0")
    assert query.interfaces["step"].source == "before"
    assert query.dependencies == ()
    topology = build_topology(graphs, [])
    assert reconstruct(topology, "alpha") == graphs["alpha"]


def test_older_precomputed_references_cannot_omit_private_interfaces(tmp_path):
    from reweave.evolution.distillation.transfer import complete_source_references

    graphs = graphs_fixture(tmp_path)
    source_teachers = teachers(graphs)
    refs, _ = aligned_references(source_teachers, seconds=0.1)
    old = tuple(ref for ref in refs if ref.task == "alpha")
    complete = complete_source_references(source_teachers, old)
    for name, graph in graphs.items():
        assert {
            r.source_edges[0]
            for r in complete
            if r.task == name and len(r.source_edges) == 1
        } == set(graph.edges)
    assert complete_source_references(source_teachers, complete) == complete


def test_literal_alignment_keeps_real_interfaces_without_path_expansion(
    tmp_path, monkeypatch
):
    import reweave.evolution.distillation.transfer as transfer

    graphs = graphs_fixture(tmp_path)
    calls = []

    def match(query, reference, **kwargs):
        calls.append((query, reference))
        assert not any(key.startswith("witness_path/") for key in query.interfaces)
        assert not any(key.startswith("witness_path/") for key in reference.interfaces)
        return {"relations": [], "optimal": True}

    monkeypatch.setattr(transfer, "match_graphs", match)
    references, alignments = transfer.aligned_references(teachers(graphs), seconds=0.1)
    assert len(calls) == len(alignments) == 2
    assert all(len(r.source_edges) == 1 for r in references)
    assert len(references) == sum(len(g.edges) for g in graphs.values())
