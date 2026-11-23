# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Conditional graph alignment by bounded search and exact assignment.

Object/state correspondences carry the combinatorial coupling. Once fixed,
relation selection is a rectangular assignment with private omission columns.
Optimistic assignments bound unfinished branches without inventing support.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def assignment(weights):
    """Maximize finite entries, requiring every row to receive a column."""
    from scipy.optimize import linear_sum_assignment

    if not weights.shape[0]:
        return 0.0, ()
    if weights.shape[1] < weights.shape[0]:
        return None
    try:
        rows, columns = linear_sum_assignment(-weights)
    except ValueError as error:
        if "infeasible" not in str(error):
            raise
        return None
    if len(rows) != weights.shape[0]:
        return None
    return float(weights[rows, columns].sum()), tuple(zip(rows, columns, strict=True))


def bindings(left, right):
    """One fact correspondence must itself define a partial injection."""
    pairs = tuple(sorted(set(zip(left, right, strict=True))))
    if len({a for a, _ in pairs}) != len(pairs) or len({b for _, b in pairs}) != len(
        pairs
    ):
        return None
    return pairs


def possible(pairs, mapping, used):
    return all(mapping[a] == b if a in mapping else b not in used for a, b in pairs)


@dataclass(frozen=True)
class ScopeMatch:
    """Support alternatives per requested fact and forbidden full bindings."""

    supports: tuple
    conflicts: tuple

    def score(self, mapping, used, domains):
        if any(
            all(a in mapping and mapping[a] == b for a, b in pairs)
            for pairs in self.conflicts
        ):
            return -math.inf
        if not self.supports:
            return 1.0
        return sum(
            any(
                possible(pairs, mapping, used)
                and all(a in mapping or b in domains[a] for a, b in pairs)
                for _, pairs in alternatives
            )
            for alternatives in self.supports
        ) / len(self.supports)


class AlignmentBound(NamedTuple):
    """Relaxed reward and its independent state/relation assignment witnesses."""

    reward: float
    states: np.ndarray
    paths: np.ndarray
    state_pairs: tuple
    relation_pairs: tuple


class AlignmentSolution(NamedTuple):
    """One feasible complete correspondence, with omission explicitly recorded."""

    reward: float
    objects: dict
    nodes: dict
    bound: AlignmentBound | None


class GraphAlignment:
    """One search instance; all bounds refer to this fixed condition graph."""

    def __init__(self, query, reference, max_candidates):
        self.query, self.reference = query, reference
        self.states, self.targets = tuple(query.states), tuple(reference.states)
        self.edges, self.paths = tuple(query.relations), tuple(reference.relations)
        self.state_index = {key: i for i, key in enumerate(self.states)}
        self.target_index = {key: i for i, key in enumerate(self.targets)}
        self.objects = tuple(
            sorted({a for f in query.conditions.values() for a in f.arguments})
        )
        self.domains = {a: set() for a in self.objects}
        self.mass = len(self.states) + len(self.edges)
        self.candidates = 0
        self.max_candidates = max_candidates
        self.assignment_calls = 0
        self.search_nodes = 0
        self.pruned_nodes = 0
        self.qfacts = self.group(query.conditions)
        self.rfacts = self.group(reference.conditions)
        self.qcontexts = self.group(query.contexts)
        self.rcontexts = self.group(reference.contexts)
        self.state_scopes = {
            (i, j): self.scope(("state", a), ("state", b))
            for i, a in enumerate(self.states)
            for j, b in enumerate(self.targets)
        }
        self.edge_scopes = {}
        for i, a in enumerate(self.edges):
            for j, b in enumerate(self.paths):
                if a[0] != b[0]:
                    continue
                self.count()
                self.edge_scopes[i, j] = (
                    tuple(self.scope((s, a[1]), (s, b[1])) for s in ("entry", "exit"))
                    if a[0] == "transition"
                    else ()
                )
        groups = defaultdict(list)
        for i, key in enumerate(self.edges):
            groups[(key[0], *query.relations[key])].append(i)
        self.groups = tuple((key, tuple(rows)) for key, rows in groups.items())
        self.domains = {a: tuple(sorted(values)) for a, values in self.domains.items()}

    @staticmethod
    def group(facts):
        result = defaultdict(list)
        for key, fact in facts.items():
            result[key[:2]].append((key, fact))
        return result

    def count(self):
        self.candidates += 1
        if self.candidates > self.max_candidates:
            raise ValueError(
                "graph matching candidate budget exceeded; select a smaller query"
            )

    def scope(self, left_scope, right_scope):
        self.count()
        supports, conflicts = [], set()
        for _, left in self.qfacts[left_scope]:
            alternatives = []
            for key, right in self.rfacts[right_scope]:
                if left.fluent != right.fluent or not set(right.values) <= set(
                    left.values
                ):
                    continue
                pairs = bindings(left.arguments, right.arguments)
                if pairs is not None:
                    self.count()
                    alternatives.append((key, pairs))
                    for a, b in pairs:
                        self.domains[a].add(b)
            supports.append(tuple(alternatives))
        for _, left in self.qcontexts[left_scope]:
            for _, right in self.rcontexts[right_scope]:
                if left.fluent == right.fluent and not set(left.values) & set(
                    right.values
                ):
                    pairs = bindings(left.arguments, right.arguments)
                    if pairs is not None:
                        conflicts.add(pairs)
        for _ in conflicts:
            self.count()
        return ScopeMatch(tuple(supports), tuple(sorted(conflicts)))

    def object_domains(self, objects, nodes):
        """A selected state's last unresolved conflict binding is forbidden."""
        used = {b for b in objects.values() if b is not None}
        domains = {a: set(values) - used for a, values in self.domains.items()}
        for i, j in nodes.items():
            if j is None:
                continue
            for pairs in self.state_scopes[i, j].conflicts:
                if any(a in objects and objects[a] != b for a, b in pairs):
                    continue
                pending = [(a, b) for a, b in pairs if a not in objects]
                if not pending:
                    return None
                if len(pending) == 1:
                    a, b = pending[0]
                    domains[a].discard(b)
        return domains

    def state_weights(self, objects, nodes, domains):
        used_objects = {b for b in objects.values() if b is not None}
        used_nodes = {b for b in nodes.values() if b is not None}
        values = np.full((len(self.states), len(self.targets)), -np.inf)
        for (i, j), scope in self.state_scopes.items():
            if i in nodes:
                if nodes[i] != j:
                    continue
            elif j in used_nodes:
                continue
            values[i, j] = scope.score(objects, used_objects, domains)
        return values

    def path_weights(self, objects, states, domains):
        used = {b for b in objects.values() if b is not None}
        values = np.full((len(self.edges), len(self.paths)), -np.inf)
        for (i, j), scopes in self.edge_scopes.items():
            source, target = self.query.relations[self.edges[i]]
            start, end = self.reference.relations[self.paths[j]]
            # Equal query endpoints must stay equal; distinct ones stay distinct.
            if source is not None and (source == target) != (start == end):
                continue
            weights = [states[self.state_index[target], self.target_index[end]]]
            if source is not None:
                weights.append(
                    states[self.state_index[source], self.target_index[start]]
                )
            weights.extend(scope.score(objects, used, domains) for scope in scopes)
            values[i, j] = min(weights)
        return values

    def node_assignment(self, weights, nodes):
        n, m = weights.shape
        extended = np.full((n, m + n), -np.inf)
        extended[:, :m] = weights
        for i in range(n):
            if i not in nodes or nodes[i] is None:
                extended[i, m + i] = 0
        self.assignment_calls += 1
        return assignment(extended)

    def relation_assignment(self, weights, nodes):
        """Private dummy capacity encodes each group's lower bound exactly."""
        a, b = weights.shape
        dummies = []
        for (_, source, target), rows in self.groups:
            required = (
                source is not None
                and nodes.get(self.state_index[source]) is not None
                and nodes.get(self.state_index[target]) is not None
            )
            dummies.extend([rows] * (len(rows) - int(required)))
        extended = np.full((a, b + len(dummies)), -np.inf)
        extended[:, :b] = weights
        for column, rows in enumerate(dummies, b):
            extended[list(rows), column] = 0
        self.assignment_calls += 1
        return assignment(extended)

    def bound(self, objects, nodes):
        """Relax unfixed shared choices, preserving assignment capacities."""
        domains = self.object_domains(objects, nodes)
        if domains is None:
            return None
        states = self.state_weights(objects, nodes, domains)
        paths = self.path_weights(objects, states, domains)
        # A selected endpoint makes its typed connection mandatory if the other
        # endpoint is selected. Remove only values with no possible such path.
        for (_, source, target), rows in self.groups:
            if source is None:
                continue
            a, b = self.state_index[source], self.state_index[target]
            for endpoint, opposite, side in ((a, b, 0), (b, a, 1)):
                if nodes.get(opposite) is None:
                    continue
                allowed = set()
                for row in rows:
                    for column in np.flatnonzero(np.isfinite(paths[row])):
                        pair = self.reference.relations[self.paths[column]]
                        allowed.add(self.target_index[pair[side]])
                for value in range(len(self.targets)):
                    if value not in allowed:
                        states[endpoint, value] = -math.inf
        node_match = self.node_assignment(states, nodes)
        if node_match is None:
            return None
        paths = self.path_weights(objects, states, domains)
        edge_match = self.relation_assignment(paths, nodes)
        if edge_match is None:
            return None
        upper = min(float(self.mass), node_match[0] + edge_match[0])
        return AlignmentBound(upper, states, paths, node_match[1], edge_match[1])

    def complete_objects(self, objects, estimate):
        """Feasible identity proposal from the optimistic support witnesses."""
        votes = defaultdict(float)
        selected = [
            self.state_scopes[i, j]
            for i, j in estimate.state_pairs
            if j < len(self.targets)
        ]
        selected.extend(
            scope
            for i, j in estimate.relation_pairs
            if (i, j) in self.edge_scopes
            for scope in self.edge_scopes[i, j]
        )
        used = {b for b in objects.values() if b is not None}
        for scope in selected:
            for alternatives in scope.supports:
                for _, pairs in alternatives:
                    if possible(pairs, objects, used):
                        for a, b in pairs:
                            if a not in objects:
                                votes[a, b] += 1
        pending = [a for a in self.objects if a not in objects]
        choices = sorted({b for a in pending for b in self.domains[a]} - used)
        weights = np.full((len(pending), len(choices) + len(pending)), -np.inf)
        for i, a in enumerate(pending):
            for j, b in enumerate(choices):
                if votes[a, b] > 0:
                    weights[i, j] = votes[a, b]
            weights[i, len(choices) + i] = 0
        self.assignment_calls += 1
        _, matched = assignment(weights)
        return {
            **objects,
            **{
                pending[i]: choices[j] if j < len(choices) else None for i, j in matched
            },
        }

    def completion(self, objects, nodes, estimate):
        completed = self.complete_objects(objects, estimate)
        fixed = self.bound(completed, nodes)
        if fixed is None:
            return None
        selected = {
            i: j if j < len(self.targets) else None for i, j in fixed.state_pairs
        }
        exact = self.bound(completed, selected)
        if exact is None:
            return None
        return AlignmentSolution(exact.reward, completed, selected, exact)

    def solve(self, seconds):
        deadline = time.perf_counter() + seconds
        omitted_objects = dict.fromkeys(self.objects)
        omitted_nodes = dict.fromkeys(range(len(self.states)))
        best = AlignmentSolution(0.0, omitted_objects, omitted_nodes, None)
        stack = [(float(self.mass), {}, {})]
        tolerance = 1e-10
        while stack and time.perf_counter() < deadline:
            inherited, objects, nodes = stack.pop()
            if inherited <= best.reward + tolerance:
                self.pruned_nodes += 1
                continue
            self.search_nodes += 1
            estimate = self.bound(objects, nodes)
            if estimate is None or estimate.reward <= best.reward + tolerance:
                self.pruned_nodes += 1
                continue
            upper = estimate.reward
            if len(objects) == len(self.objects) and len(nodes) == len(self.states):
                best = AlignmentSolution(upper, objects, nodes, estimate)
                continue
            # A complete feasible certificate helps pruning; it never supplies a bound.
            proposal = self.completion(objects, nodes, estimate)
            if proposal is not None and proposal.reward > best.reward:
                best = proposal
            if upper <= best.reward + tolerance:
                self.pruned_nodes += 1
                continue
            if len(nodes) < len(self.states):
                pending = [i for i in range(len(self.states)) if i not in nodes]
                key = min(
                    pending, key=lambda i: (np.isfinite(estimate.states[i]).sum(), i)
                )
                choices = sorted(
                    (
                        j
                        for j in range(len(self.targets))
                        if np.isfinite(estimate.states[key, j])
                    ),
                    key=lambda j: -estimate.states[key, j],
                ) + [None]
                preferred = dict(estimate.state_pairs)[key]
                preferred = preferred if preferred < len(self.targets) else None
                choices.sort(key=lambda j: j != preferred)
                stack.extend(
                    (upper, objects, {**nodes, key: j}) for j in reversed(choices)
                )
            else:
                used = {b for b in objects.values() if b is not None}
                key = min(
                    (a for a in self.objects if a not in objects),
                    key=lambda a: (sum(b not in used for b in self.domains[a]), a),
                )
                choices = [b for b in self.domains[key] if b not in used] + [None]
                preferred = (
                    proposal.objects[key] if proposal is not None else choices[0]
                )
                choices.sort(key=lambda b: b != preferred)
                stack.extend(
                    (upper, {**objects, key: b}, nodes) for b in reversed(choices)
                )
        upper = max([best.reward, *(bound for bound, _, _ in stack)])
        return self.certificate(best, upper)

    def certificate(self, best, upper):
        score, objects, nodes, estimate = best
        states = {i: j for i, j in nodes.items() if j is not None}
        relations = {
            i: j
            for i, j in (estimate.relation_pairs if estimate is not None else ())
            if j < len(self.paths)
        }
        supported, reported_objects = set(), {}
        used = {b for b in objects.values() if b is not None}
        scopes = [
            (self.state_scopes[i, j], ("state", self.states[i]))
            for i, j in states.items()
        ]
        scopes.extend(
            (scope, (name, self.edges[i][1]))
            for i, j in relations.items()
            for name, scope in zip(
                ("entry", "exit"), self.edge_scopes[i, j], strict=False
            )
        )
        for scope, identity in scopes:
            for (key, _), alternatives in zip(
                self.qfacts[identity], scope.supports, strict=True
            ):
                pair = next(
                    (
                        pairs
                        for _, pairs in alternatives
                        if possible(pairs, objects, used)
                    ),
                    None,
                )
                if pair is not None:
                    supported.add(key)
                    reported_objects.update(pair)
        coverage = min(1.0, max(0.0, score / self.mass))
        gap = max(0.0, upper - score)
        return {
            "distance": 1 - coverage,
            "coverage": coverage,
            "query_mass": self.mass,
            "distance_lower_bound": max(0.0, 1 - upper / self.mass),
            "optimal": gap <= 1e-10,
            "solver_gap": gap / max(abs(score), 1e-12),
            "states": [
                {
                    "query": list(self.states[i]),
                    "reference": list(self.targets[j]),
                    "coverage": float(estimate.states[i, j]),
                }
                for i, j in sorted(states.items())
            ],
            "relations": [
                {
                    "query": list(self.edges[i]),
                    "reference": list(self.paths[j]),
                    "coverage": float(estimate.paths[i, j]),
                }
                for i, j in sorted(relations.items())
            ],
            "entity_mapping": reported_objects,
            "unmatched_entities": [
                a for a in self.objects if a not in reported_objects
            ],
            "unmatched_states": [
                list(a) for i, a in enumerate(self.states) if i not in states
            ],
            "unmatched_relations": [
                list(a) for i, a in enumerate(self.edges) if i not in relations
            ],
            "unsupported_conditions": [
                {
                    "scope": key[0],
                    "owner": key[1],
                    "condition": fact.model_dump(mode="json"),
                }
                for key, fact in self.query.conditions.items()
                if key not in supported
            ],
            "candidates": self.candidates,
            "search_nodes": self.search_nodes,
            "pruned_nodes": self.pruned_nodes,
            "assignment_calls": self.assignment_calls,
        }
