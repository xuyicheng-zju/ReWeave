# Modified for the ReWeave namespace.
"""Assignment equivalence boundaries and admissible partial-search bounds."""

from itertools import product

import numpy as np
import pytest

from reweave.evolution.graph.alignment import GraphAlignment, ScopeMatch, assignment
from reweave.evolution.graph.matching import ConditionGraph, GraphQuery


def mappings(size, targets, partial=False):
    unset = object()
    options = [None, *range(targets)]
    if partial:
        options.append(unset)
    for values in product(options, repeat=size):
        selected = [v for v in values if v is not None and v is not unset]
        if len(set(selected)) == len(selected):
            yield {i: v for i, v in enumerate(values) if v is not unset}


def graph(edges):
    return GraphQuery(
        catalog={
            "flag": {
                "definition": "A context fact.",
                "arguments": [],
                "values": {"yes": "holds", "no": "does not hold"},
            }
        },
        entities={},
        nodes={
            str(i): [{"fluent": "flag", "arguments": [], "values": ["yes"]}]
            for i in range(3)
        },
        interfaces={
            str(i): {"source": str(a), "target": str(b)}
            for i, (a, b) in enumerate(edges)
        },
    )


def test_assignment_requires_all_rows_even_when_scipy_would_return_partial():
    assert assignment(np.zeros((2, 1))) is None
    assert assignment(np.array([[0, -np.inf], [0, -np.inf]])) is None
    assert assignment(np.empty((0, 0))) == (0, ())


@pytest.mark.parametrize(
    "query_edges,reference_edges",
    [
        ([(0, 1), (0, 2)], [(0, 1), (1, 2)]),
        ([(0, 1), (0, 1)], [(1, 0)]),
        ([(0, 0), (0, 1)], [(1, 1), (1, 2)]),
    ],
)
def test_every_topology_branch_bound_covers_independent_exact_completion(
    query_edges, reference_edges
):
    q, r = graph(query_edges), graph(reference_edges)
    engine = GraphAlignment(
        ConditionGraph.from_query(q), ConditionGraph.from_query(r, reference=True), 1000
    )
    complete = []
    # Independent exhaustive X/Z specification, without the assignment solver.
    for nodes in mappings(3, 3):
        best = None
        for edges in mappings(len(query_edges), len(reference_edges)):
            legal = all(
                j is None or (nodes[a], nodes[b]) == reference_edges[j]
                for i, j in edges.items()
                for a, b in [query_edges[i]]
            )
            required = all(
                nodes[a] is None
                or nodes[b] is None
                or any(
                    edges[i] is not None
                    for i, pair in enumerate(query_edges)
                    if pair == (a, b)
                )
                for a, b in set(query_edges)
            )
            if legal and required:
                score = sum(v is not None for v in nodes.values()) + sum(
                    v is not None for v in edges.values()
                )
                best = max(best or 0, score)
        complete.append((nodes, best))
    for nodes in mappings(3, 3, partial=True):
        feasible = [
            score
            for full, score in complete
            if score is not None and all(full[k] == v for k, v in nodes.items())
        ]
        bound = engine.bound({}, nodes)
        if feasible:
            assert bound is not None
            assert bound[0] + 1e-10 >= max(feasible)
        if len(nodes) == 3:
            assert (bound is None) == (not feasible)
            if feasible:
                assert bound[0] == max(feasible)
    result = engine.solve(5)
    assert result["optimal"]
    assert result["coverage"] * engine.mass == pytest.approx(
        max(s for _, s in complete if s is not None)
    )


def test_required_parallel_group_keeps_zero_reward_real_correspondence():
    engine = GraphAlignment(
        ConditionGraph.from_query(graph([(0, 1), (0, 1)])),
        ConditionGraph.from_query(graph([(0, 1)]), reference=True),
        1000,
    )
    selected = {0: 0, 1: 1, 2: None}
    result = engine.relation_assignment(np.zeros((2, 1)), selected)
    assert result is not None
    assert sum(j == 0 for _, j in result[1]) == 1
    assert engine.relation_assignment(np.full((2, 1), -np.inf), selected) is None
    assert (
        engine.relation_assignment(np.full((2, 1), -np.inf), {0: None, 1: 1, 2: None})
        is not None
    )


def test_selected_scope_nogood_requires_actual_binding_not_singleton_domain():
    engine = GraphAlignment(
        ConditionGraph.from_query(graph([])), ConditionGraph.from_query(graph([])), 1000
    )
    engine.domains = {"x": ("a",), "y": ("b",)}
    engine.state_scopes[0, 0] = ScopeMatch((), ((("x", "a"), ("y", "b")),))
    assert engine.object_domains({}, {0: 0}) == {"x": {"a"}, "y": {"b"}}
    assert engine.object_domains({"x": "a"}, {0: 0})["y"] == set()
    assert engine.object_domains({"x": "a", "y": "b"}, {0: 0}) is None
    assert engine.object_domains({"x": None}, {0: 0})["y"] == {"b"}


def test_scope_bound_cannot_use_entity_value_forbidden_by_another_selected_state():
    scope = ScopeMatch(((("reference_fact", (("x", "a"),)),),), ())
    assert scope.score({}, set(), {"x": {"a"}}) == 1
    assert scope.score({}, set(), {"x": set()}) == 0
