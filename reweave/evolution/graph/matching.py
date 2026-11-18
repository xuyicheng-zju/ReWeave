# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Maximum consistent coverage of a directed condition graph.

One object substitution and one state correspondence explain the whole query.
Known contradictions forbid a correspondence; absent information earns no
coverage. Typed connections between selected query states must be preserved.
Shared correspondences are searched with exact conditional assignments.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

from pydantic import Field, model_validator

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.graph.alignment import GraphAlignment
from reweave.evolution.graph.conditions import (
    GraphSemantics,
    InterfaceConditions,
    StateCondition,
    merge_catalogs,
)


class GraphTransition(InterfaceConditions):
    source: str | None = Field(
        description="Query state ID, or null for an edge-local known background entrance. A background entrance requires entry conditions and does not assert a shared progress state."
    )
    target: str = Field(
        description="Query state ID at the intended end of this directed interface."
    )


class GraphQuery(GraphSemantics):
    """A retrieval hypothesis; no program, witness or physical-success claim."""

    interfaces: dict[str, GraphTransition] = Field(
        default_factory=dict,
        description="Directed conditional interfaces with distinct IDs, including parallel alternatives. Empty requests no transition constraints.",
    )
    dependencies: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Directed retained-progress dependencies, compared through transitive closure. Empty requests no dependency constraints; it does not assert physical impossibility of extra reference links.",
    )

    @model_validator(mode="after")
    def validate_topology(self):
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("duplicate query dependency")
        for source, target in self.dependencies:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError("query dependency has an unknown endpoint")
        for interface in self.interfaces.values():
            if interface.target not in self.nodes or (
                interface.source is not None and interface.source not in self.nodes
            ):
                raise ValueError("query interface has an unknown endpoint")
            if interface.source is None and not interface.entry:
                raise ValueError("known background entrance requires entry conditions")
        dependency_closure(self.dependencies)
        self.validate_contexts(
            {key: (edge.source, edge.target) for key, edge in self.interfaces.items()}
        )
        return self


def dependency_closure(pairs):
    result = set(pairs)
    previous = None
    while previous != result:
        previous = result.copy()
        result |= {(a, d) for a, b in previous for c, d in previous if b == c}
    if any(a == b for a, b in result):
        raise ValueError("query dependencies must be acyclic")
    return sorted(result)


def semantic_query(
    graph: SemanticGraph, *, witness_paths=False
) -> tuple[GraphQuery, dict]:
    """Project the canonical graph; paths require actual episode continuity.

    Direct interfaces preserve all parallel alternatives, including failed ones.
    Optional reference paths use the first entrance and last exit as an envelope,
    not as a composed executable precondition. Only achieved, physically joined
    intermediates may extend a path; the last attempt may have failed.
    """
    if graph.matching is None:
        raise ValueError(
            "cross-task retrieval requires matching conditions on the canonical graph"
        )
    interfaces = {
        key: GraphTransition(
            source=edge.source,
            target=edge.target,
            **graph.matching.interfaces[key].model_dump(),
        )
        for key, edge in sorted(graph.edges.items())
    }
    paths = {
        key: ((key, 0),) if edge.witnesses else () for key, edge in graph.edges.items()
    }
    uncertainties = list(graph.matching.uncertainties)
    if witness_paths:
        episodes = defaultdict(list)
        for key, edge in graph.edges.items():
            for index, witness in enumerate(edge.witnesses):
                episodes[witness.episode_id].append((key, index, witness))

        def state_at(episode, frame):
            for span in graph.episodes[episode]:
                if span.first_frame <= frame <= span.last_frame:
                    return span.node_id if span.status == "node" else None
            return None

        chains = set()
        for episode, witnesses in episodes.items():
            successors = defaultdict(list)
            for item in witnesses:
                successors[item[2].first_frame].append(item)
            for first_key, first_index, first in witnesses:
                pending = [(first_key, first_index, first, ((first_key, first_index),))]
                seen = set()
                while pending:
                    key, index, witness, chain = pending.pop()
                    if (key, index) in seen:
                        continue
                    seen.add((key, index))
                    if len(chain) > 1:
                        chains.add(chain)
                    target = graph.edges[key].target
                    if (
                        witness.outcome != "achieved"
                        or state_at(episode, witness.last_frame) != target
                    ):
                        continue
                    for next_key, next_index, next_witness in successors[
                        witness.last_frame
                    ]:
                        if graph.edges[next_key].source == target:
                            try:
                                graph.matching.context(
                                    graph.matching.nodes[target],
                                    interfaces[key].exit,
                                    interfaces[next_key].entry,
                                )
                            except ValueError as error:
                                # The observed attempts remain in the graph.
                                # Only this inconsistent derived envelope is
                                # withheld; unknown join facts are permitted.
                                uncertainties.append(
                                    f"No path through {key}[{index}] -> {next_key}[{next_index}]: {error}"
                                )
                                continue
                            pending.append(
                                (
                                    next_key,
                                    next_index,
                                    next_witness,
                                    (*chain, (next_key, next_index)),
                                )
                            )
        for chain in sorted(chains):
            first, last = (interfaces[chain[index][0]] for index in (0, -1))
            key = f"witness_path/{len(paths)}"
            while key in interfaces:
                key += "/path"
            interfaces[key] = GraphTransition(
                source=first.source,
                target=last.target,
                entry=first.entry,
                exit=last.exit,
            )
            paths[key] = chain
    return GraphQuery(
        **graph.matching.model_dump(exclude={"interfaces", "uncertainties"}),
        uncertainties=tuple(dict.fromkeys(uncertainties)),
        interfaces=interfaces,
        dependencies=tuple(
            sorted(
                (predecessor, key)
                for key, node in graph.nodes.items()
                for predecessor in node.predecessors
            )
        ),
    ), paths


@dataclass
class ConditionGraph:
    """Solver view; tuple namespaces separate user IDs and entrance ports."""

    states: dict[tuple, tuple[StateCondition, ...]]
    relations: dict[tuple, tuple[tuple | None, tuple]]
    conditions: dict[tuple, StateCondition]
    contexts: dict[tuple, StateCondition]

    @classmethod
    def from_query(cls, query, *, reference=False):
        expand = query.context
        states = {("node", key): expand(value) for key, value in query.nodes.items()}
        relations = {
            ("dependency", source, target): (("node", source), ("node", target))
            for source, target in dependency_closure(query.dependencies)
        }
        conditions, contexts = {}, {}
        for key, interface in query.interfaces.items():
            source = (
                ("node", interface.source) if interface.source is not None else None
            )
            target = ("node", interface.target)
            # An interface refines its endpoint, so both sets of facts apply.
            # Inconsistent conjunctions reject the representation itself.
            entry = expand((*states.get(source, ()), *interface.entry))
            exit_conditions = expand((*states[target], *interface.exit))
            if source is None and reference:
                source = ("background", key)
                states[source] = entry
            relations["transition", key] = (source, target)
            for scope, values in (("entry", entry), ("exit", exit_conditions)):
                contexts.update(
                    {(scope, key, i): value for i, value in enumerate(values)}
                )
                # Endpoint facts can support an interface without diluting
                # its own requested conditions in the query denominator.
                requested = values if reference else expand(getattr(interface, scope))
                conditions.update(
                    {(scope, key, i): value for i, value in enumerate(requested)}
                )
        for key, values in states.items():
            facts = {("state", key, i): value for i, value in enumerate(values)}
            conditions.update(facts)
            contexts.update(facts)
        return cls(states, relations, conditions, contexts)


def match_graphs(
    query: GraphQuery, candidate: GraphQuery, *, seconds=10.0, max_candidates=100_000
) -> dict:
    """Return a feasible directed alignment and certified distance interval.

    Shared state/object correspondences are searched; conditional relation
    selection is solved exactly by assignment. Extra reference subgraphs are
    free, while known contradictions forbid a correspondence.
    """
    if not math.isfinite(seconds) or seconds <= 0 or max_candidates < 1:
        raise ValueError("matching resource limits must be positive and finite")
    started = time.perf_counter()
    merge_catalogs(query.catalog, candidate.catalog)
    alignment = GraphAlignment(
        ConditionGraph.from_query(query),
        ConditionGraph.from_query(candidate, reference=True),
        max_candidates,
    )
    prepared = time.perf_counter()
    result = alignment.solve(seconds)
    result["preparation_seconds"] = prepared - started
    result["seconds"] = time.perf_counter() - started
    return result
