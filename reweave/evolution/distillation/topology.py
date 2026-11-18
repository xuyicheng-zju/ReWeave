"""Lossless, guarded factorization of source task topologies.

Only repeated encoded facts are factored. Every source graph must reconstruct
exactly, including private conditions, dependencies, witnesses and uncertainty.
Structural sharing is a synthesis hypothesis, never a physical-success claim.
"""

from __future__ import annotations

import copy
import json

from pydantic import BaseModel, ConfigDict, Field

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.graph.conditions import (
    StateCondition,
    merge_catalogs,
)
from reweave.evolution.graph.matching import (
    GraphQuery,
    dependency_closure,
    match_graphs,
    semantic_query,
)


class TopologyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopologyMember(TopologyModel):
    task: str
    edge: str
    roles: dict[str, str]

    @property
    def key(self):
        return occurrence_key(self.task, self.edge)


class SharedMechanism(TopologyModel):
    identity: str
    query: GraphQuery
    members: tuple[TopologyMember, ...] = Field(min_length=1)


class FactProjection(TopologyModel):
    mechanism: str | None = None
    scope: str | None = None
    roles: dict[str, str] = Field(default_factory=dict)
    residual: tuple[StateCondition, ...] = ()


class SourceProjection(TopologyModel):
    # No condition copy is hidden here: matching is reconstructed from bindings.
    skeleton: dict
    catalog: dict
    entities: dict
    uncertainties: tuple[str, ...]
    nodes: dict[str, FactProjection]
    interfaces: dict[str, dict[str, FactProjection]]


class SharedTopology(TopologyModel):
    revision: int = Field(default=0, ge=0)
    mechanisms: dict[str, SharedMechanism]
    sources: dict[str, SourceProjection]
    events: tuple[dict, ...] = ()
    search: dict = Field(default_factory=dict)


def occurrence_key(task, edge):
    return json.dumps([task, edge], separators=(",", ":"))


def fact_key(fact):
    return fact.fluent, tuple(fact.arguments), tuple(fact.values)


def remap(facts, roles):
    return tuple(
        fact.model_copy(update={"arguments": tuple(roles[a] for a in fact.arguments)})
        for fact in facts
    )


def fragment(graph, edge):
    """One literal interface; other graph edges never become an invented path."""
    semantics = graph.matching
    if semantics is None:
        raise ValueError("shared topology requires encoded source conditions")
    physical = graph.edges[edge]
    nodes = {"after": semantics.nodes[physical.target]}
    if physical.source is not None:
        nodes["before"] = semantics.nodes[physical.source]
    return GraphQuery(
        catalog=semantics.catalog,
        entities=semantics.entities,
        nodes=nodes,
        interfaces={
            "step": {
                "source": "before" if physical.source is not None else None,
                "target": "after",
                **semantics.interfaces[edge].model_dump(),
            }
        },
        dependencies=(("before", "after"),)
        if (physical.source, physical.target)
        in dependency_closure(
            (predecessor, node)
            for node, state in graph.nodes.items()
            for predecessor in state.predecessors
        )
        else (),
        uncertainties=semantics.uncertainties,
    )


def common_query(members, graphs):
    """A simultaneous core, with every noncommon clause left in its source delta."""
    queries = [(m, fragment(graphs[m.task], m.edge)) for m in members]
    shapes = {
        (q.interfaces["step"].source, q.interfaces["step"].target) for _, q in queries
    }
    if len(shapes) != 1:
        raise ValueError("shared interfaces have different directed endpoint roles")
    for member, query in queries:
        if set(member.roles) != set(query.entities) or len(
            set(member.roles.values())
        ) != len(member.roles):
            raise ValueError("each member needs an injective complete entity-role map")
    catalog = merge_catalogs(*(q.catalog for _, q in queries))

    def shared(scope):
        rows = []
        for member, query in queries:
            facts = (
                query.nodes[scope[5:]]
                if scope.startswith("node:")
                else getattr(query.interfaces["step"], scope)
            )
            rows.append({fact_key(f): f for f in remap(facts, member.roles)})
        keys = set(rows[0]).intersection(*(set(row) for row in rows[1:]))
        return tuple(rows[0][key] for key in sorted(keys))

    nodes = {name: shared("node:" + name) for name in queries[0][1].nodes}
    if any(not facts for facts in nodes.values()):
        raise ValueError("merge has no jointly supported condition at an endpoint")
    entry, exit_conditions = shared("entry"), shared("exit")
    source, target = next(iter(shapes))
    if source is None and not entry:
        raise ValueError("shared background entrance lost its explicit condition")
    facts = [f for rows in (*nodes.values(), entry, exit_conditions) for f in rows]
    used = {a for f in facts for a in f.arguments}
    return GraphQuery(
        catalog={
            key: value
            for key, value in catalog.items()
            if key in {f.fluent for f in facts}
        },
        entities={
            role: {"description": "shared physical or task-context role"}
            for role in sorted(used)
        },
        nodes=nodes,
        interfaces={
            "step": {
                "source": source,
                "target": target,
                "entry": entry,
                "exit": exit_conditions,
            }
        },
        dependencies=tuple(
            sorted(
                set(queries[0][1].dependencies).intersection(
                    *(set(q.dependencies) for _, q in queries[1:])
                )
            )
        ),
        uncertainties=(
            "Shared encoded core only; all source guards and qualifications remain in their projections.",
        ),
    )


def scope_facts(query, scope):
    return (
        query.nodes[scope[5:]]
        if scope.startswith("node:")
        else getattr(query.interfaces["step"], scope)
    )


def project_facts(binding, mechanisms):
    if binding.mechanism is None:
        if binding.scope is not None or binding.roles:
            raise ValueError("private projection cannot refer to shared roles")
        return binding.residual
    mechanism = mechanisms[binding.mechanism]
    inverse = {role: entity for entity, role in binding.roles.items()}
    if len(inverse) != len(binding.roles):
        raise ValueError("projection role map is not injective")
    core = remap(scope_facts(mechanism.query, binding.scope), inverse)
    return (*core, *binding.residual)


def reconstruct(topology, task):
    """Recover the full source, not just its shared or successful subgraph."""
    source = topology.sources[task]
    result = copy.deepcopy(source.skeleton)
    if "matching" in result:
        raise ValueError("source skeleton cannot hide a second condition graph")
    result["matching"] = {
        "catalog": source.catalog,
        "entities": source.entities,
        "uncertainties": source.uncertainties,
        "nodes": {
            key: project_facts(value, topology.mechanisms)
            for key, value in source.nodes.items()
        },
        "interfaces": {
            key: {
                scope: project_facts(value, topology.mechanisms)
                for scope, value in scopes.items()
            }
            for key, scopes in source.interfaces.items()
        },
    }
    return SemanticGraph.model_validate(result)


def assemble(graphs, groups, *, revision=0, events=(), search=None):
    """Factor known clauses while retaining every exact source residual."""
    mechanisms = {}
    membership = {}
    for members in sorted(groups, key=lambda group: sorted(m.key for m in group)):
        identity = f"mechanism_{len(mechanisms):05d}"
        members = tuple(sorted(members, key=lambda m: m.key))
        query = common_query(members, graphs)
        mechanisms[identity] = SharedMechanism(
            identity=identity, query=query, members=members
        )
        for member in members:
            if member.key in membership:
                raise ValueError("one literal source interface cannot have two owners")
            membership[member.key] = (identity, member)
    expected = {
        occurrence_key(task, edge)
        for task, graph in graphs.items()
        for edge in graph.edges
    }
    if set(membership) != expected:
        raise ValueError(
            "topology must retain every source interface, including unmatched ones"
        )
    sources = {}
    for task, graph in sorted(graphs.items()):
        semantics = graph.matching

        def binding(facts, owner=None, scope=None):
            if owner is None:
                return FactProjection(residual=facts)
            identity, member = owner
            core = remap(
                scope_facts(mechanisms[identity].query, scope),
                {v: k for k, v in member.roles.items()},
            )
            keys = {fact_key(f) for f in core}
            if not keys <= {fact_key(f) for f in facts}:
                raise ValueError("shared core is not supported by this source scope")
            return FactProjection(
                mechanism=identity,
                scope=scope,
                roles=member.roles,
                residual=tuple(f for f in facts if fact_key(f) not in keys),
            )

        nodes = {}
        interfaces = {}
        for edge, physical in sorted(graph.edges.items()):
            owner = membership[occurrence_key(task, edge)]
            for name, scope in (
                (physical.source, "node:before"),
                (physical.target, "node:after"),
            ):
                if name is not None and name not in nodes:
                    nodes[name] = binding(semantics.nodes[name], owner, scope)
            interfaces[edge] = {
                scope: binding(getattr(semantics.interfaces[edge], scope), owner, scope)
                for scope in ("entry", "exit")
            }
        for name, facts in semantics.nodes.items():
            nodes.setdefault(name, binding(facts))
        sources[task] = SourceProjection(
            skeleton=graph.model_dump(mode="json", exclude={"matching"}),
            catalog={
                key: value.model_dump(mode="json")
                for key, value in semantics.catalog.items()
            },
            entities={
                key: value.model_dump(mode="json")
                for key, value in semantics.entities.items()
            },
            uncertainties=semantics.uncertainties,
            nodes=nodes,
            interfaces=interfaces,
        )
    result = SharedTopology(
        revision=revision,
        mechanisms=mechanisms,
        sources=sources,
        events=tuple(events),
        search=search or {},
    )
    validate_topology(result, graphs)
    return result


def validate_topology(topology, graphs):
    if set(topology.sources) != set(graphs):
        raise ValueError("source partition changed during topology consolidation")
    members = [m for group in topology.mechanisms.values() for m in group.members]
    expected = {
        occurrence_key(task, edge)
        for task, graph in graphs.items()
        for edge in graph.edges
    }
    if (
        len({m.key for m in members}) != len(members)
        or {m.key for m in members} != expected
    ):
        raise ValueError("source interface coverage is not exact")
    for identity, mechanism in topology.mechanisms.items():
        if (
            mechanism.identity != identity
            or common_query(mechanism.members, graphs) != mechanism.query
        ):
            raise ValueError("shared core or multi-source role consistency changed")
    for task, graph in graphs.items():
        source = topology.sources[task]
        owners = {
            m.edge: (identity, m)
            for identity, group in topology.mechanisms.items()
            for m in group.members
            if m.task == task
        }
        for edge, scopes in source.interfaces.items():
            identity, member = owners[edge]
            if set(scopes) != {"entry", "exit"} or any(
                binding.mechanism != identity
                or binding.roles != member.roles
                or binding.scope != scope
                for scope, binding in scopes.items()
            ):
                raise ValueError(
                    "source interface binding changed its owner or role map"
                )
        for node, binding in source.nodes.items():
            if binding.mechanism is None:
                continue
            group = topology.mechanisms[binding.mechanism]
            if not any(
                m.task == task
                and m.roles == binding.roles
                and (
                    binding.scope == "node:before"
                    and graph.edges[m.edge].source == node
                    or binding.scope == "node:after"
                    and graph.edges[m.edge].target == node
                )
                for m in group.members
            ):
                raise ValueError("source node binding changed its usage position")
        if reconstruct(topology, task) != graph:
            raise ValueError(
                f"source constraints, dependencies or evidence lost: {task}"
            )


def initial_groups(graphs):
    return [
        [
            TopologyMember(
                task=task,
                edge=edge,
                roles={
                    entity: f"r{i}"
                    for i, entity in enumerate(sorted(graph.matching.entities))
                },
            )
        ]
        for task, graph in sorted(graphs.items())
        for edge in sorted(graph.edges)
    ]


def merge_groups(groups, left, right, entity_mapping, graphs):
    """A role-aware multiway merge, not transitive closure of pairwise similarity."""
    owners = {
        member.key: index for index, group in enumerate(groups) for member in group
    }
    a, b = owners[left], owners[right]
    ma = next(m for m in groups[a] if m.key == left)
    mb = next(m for m in groups[b] if m.key == right)
    if (
        not set(entity_mapping) <= ma.roles.keys()
        or not set(entity_mapping.values()) <= mb.roles.keys()
    ):
        raise ValueError("alignment refers to an unknown entity")
    if a == b:
        if any(
            ma.roles.get(entity) != mb.roles.get(other)
            for entity, other in entity_mapping.items()
        ):
            raise ValueError("inconsistent cycle of entity roles")
        return groups
    role_map = {}
    for entity, other in entity_mapping.items():
        if entity not in ma.roles or other not in mb.roles:
            raise ValueError("alignment refers to an unknown entity")
        role, target = ma.roles[entity], mb.roles[other]
        if role in role_map and role_map[role] != target:
            raise ValueError("inconsistent cycle of entity roles")
        role_map[role] = target
    if len(set(role_map.values())) != len(role_map):
        raise ValueError("joint role correspondence is not injective")
    occupied = {
        r for group in (groups[a], groups[b]) for m in group for r in m.roles.values()
    }
    for role in sorted(
        {r for m in groups[a] for r in m.roles.values()} - role_map.keys()
    ):
        fresh = f"r{len(occupied)}"
        while fresh in occupied:
            fresh += "_"
        occupied.add(fresh)
        role_map[role] = fresh
    merged = [
        *groups[b],
        *(
            m.model_copy(
                update={
                    "roles": {
                        entity: role_map[role] for entity, role in m.roles.items()
                    }
                }
            )
            for m in groups[a]
        ),
    ]
    common_query(merged, graphs)  # Simultaneous support, not just the last pair.
    return [g for index, g in enumerate(groups) if index not in {a, b}] + [merged]


def guard_variant_candidates(graphs, *, seconds):
    """Propose shared implementations without equating incompatible guards.

    This relaxed query is ONLY a hypothesis generator. Exact source conditions
    remain in the projections, and a merged core must be supported by all members.
    """
    queries = {}
    for name, graph in graphs.items():
        query, _ = semantic_query(graph)
        value = query.model_dump(mode="json")
        for entity in value["entities"].values():
            entity["conditions"] = []
        for interface in value["interfaces"].values():
            if interface["source"] is not None:
                interface["entry"] = []
            interface["exit"] = []
        queries[name] = GraphQuery.model_validate(value)
    return [
        {
            "query": left,
            "reference": right,
            "alignment": match_graphs(query, reference, seconds=seconds),
            "meaning": "guard-relaxed synthesis hypothesis, never physical equivalence",
        }
        for left, query in sorted(queries.items())
        for right, reference in sorted(queries.items())
        if left != right
    ]


def build_topology(graphs, alignments, *, max_proposals=4096, seconds=0.1):
    if any(graph.matching is None for graph in graphs.values()):
        raise ValueError("all shared sources require encoded conditions")
    merge_catalogs(*(g.matching.catalog for g in graphs.values()))
    groups = initial_groups(graphs)
    proposals = []
    variants = (
        guard_variant_candidates(graphs, seconds=seconds) if max_proposals else []
    )
    for pair in (*alignments, *variants):
        left, right = pair["query"], pair["reference"]
        if left not in graphs or right not in graphs or left == right:
            raise ValueError("alignment outside the admitted source partition")
        for relation in pair["alignment"]["relations"]:
            qa, rb = relation["query"], relation["reference"]
            if (
                qa[0] == rb[0] == "transition"
                and qa[1] in graphs[left].edges
                and rb[1] in graphs[right].edges
                and relation["coverage"] > 0
            ):
                proposals.append(
                    (
                        -relation["coverage"],
                        occurrence_key(left, qa[1]),
                        occurrence_key(right, rb[1]),
                        pair["alignment"]["entity_mapping"],
                    )
                )
    events = []
    for _, left, right, mapping in sorted(proposals, key=lambda row: row[:3])[
        :max_proposals
    ]:
        try:
            updated = merge_groups(groups, left, right, mapping, graphs)
        except ValueError as error:
            events.append(
                {
                    "operation": "merge_rejected",
                    "members": [left, right],
                    "reason": str(error),
                }
            )
        else:
            if updated is not groups:
                events.append({"operation": "merged", "members": [left, right]})
            groups = updated
    return assemble(
        graphs,
        groups,
        events=events,
        search={
            "proposals": len(proposals),
            "examined": min(len(proposals), max_proposals),
            "truncated": len(proposals) > max_proposals,
            "globally_optimal": False,
            "guard_variant_alignments": len(variants),
        },
    )


def revise_topology(topology, graphs, *, split=(), merges=(), reason="feedback"):
    """Apply reversible structure changes, always rechecking exact reconstruction."""
    validate_topology(topology, graphs)
    groups = [list(group.members) for group in topology.mechanisms.values()]
    known = {m.key for group in groups for m in group}
    if not set(split) <= known:
        raise ValueError("split refers to an unknown source interface")
    changed = False
    revised = []
    for group in groups:
        separate = [m for m in group if m.key in split and len(group) > 1]
        kept = [m for m in group if m not in separate]
        if kept:
            revised.append(kept)
        revised.extend([m] for m in separate)
        changed |= bool(separate)
    for proposal in merges:
        updated = merge_groups(
            revised,
            proposal["left"],
            proposal["right"],
            proposal["entity_mapping"],
            graphs,
        )
        changed |= updated is not revised
        revised = updated
    if not changed:
        return topology
    return assemble(
        graphs,
        revised,
        revision=topology.revision + 1,
        events=(
            *topology.events,
            {
                "operation": "revised",
                "reason": reason,
                "split": sorted(split),
                "merges": list(merges),
            },
        ),
        search=topology.search,
    )


def feedback_splits(
    topology,
    scores,
    anchors,
    tolerance,
    references=(),
    conditional_anchors=None,
    rates=None,
):
    failed = {
        task for task, floor in anchors.items() if scores[task] + tolerance < floor
    }
    selected = {
        m.key
        for group in topology.mechanisms.values()
        if len(group.members) > 1
        for m in group.members
        if m.task in failed
    }
    for ref in references:
        if (
            rates
            and conditional_anchors
            and rates[ref.key]["lower"] < conditional_anchors[ref.key]["upper"]
        ):
            selected.update(occurrence_key(ref.task, edge) for edge in ref.source_edges)
    known = {m.key for group in topology.mechanisms.values() for m in group.members}
    return selected & known
