"""Retrieve fixed teacher units from the current student's evidenced gaps."""

import json

from reweave.evolution.distillation.adaptation import anchor_binding, pack, unpack
from reweave.evolution.distillation.fragment_logic import Fact, entails, normalize
from reweave.evolution.distillation.topology import occurrence_key
from reweave.evolution.graph.matching import semantic_query


def facts(rows):
    return normalize(Fact(f.fluent, f.arguments, frozenset(f.values)) for f in rows)


def teacher_unit_bank(graphs):
    units = {}
    for task, graph in sorted(graphs.items()):
        if graph.matching is None:
            raise ValueError("student-gap retrieval requires encoded source graphs")
        q, _ = semantic_query(graph)
        for edge, interface in sorted(q.interfaces.items()):
            physical = graph.edges[edge]
            entry = facts(
                q.context((*q.nodes.get(interface.source, ()), *interface.entry))
            )
            result = facts((*q.nodes[interface.target], *interface.exit))
            effects = tuple(
                f
                for f in result
                if f.name != "official_success" and entails(entry, (f,)) != "supported"
            )
            units[occurrence_key(task, edge)] = {
                "task": task,
                "edge": edge,
                "source": physical.source,
                "target": physical.target,
                "entry": pack(entry),
                "result": pack(result),
                "effects": pack(effects),
                "has_achieved_witness": any(
                    w.outcome == "achieved" for w in physical.witnesses
                ),
                "witnesses": [w.model_dump(mode="json") for w in physical.witnesses],
                "references": [r.model_dump(mode="json") for r in physical.references],
                "program_ids": sorted(
                    {w.program_id for w in physical.witnesses if w.program_id}
                ),
            }
    return {"schema": "teacher-unit-bank/v1", "units": units}


def retrieve(goal, known, units):
    """Return every consistent nonempty result overlap, with honest applicability."""
    matches = []
    for identity, unit in units.items():
        if not unit["has_achieved_witness"]:
            continue
        effects = unpack(unit["effects"])
        bindings = {(): {}}
        for desired in goal:
            options = [anchor_binding(effect, desired) for effect in effects]
            options = [mapping for mapping in options if mapping is not None]
            joined = dict(bindings)
            for partial in bindings.values():
                for option in options:
                    mapping = dict(partial)
                    if any(a in mapping and mapping[a] != b for a, b in option.items()):
                        continue
                    mapping.update(option)
                    if len(set(mapping.values())) == len(mapping):
                        joined[tuple(sorted(mapping.items()))] = mapping
            bindings = joined
        if not goal:
            continue
        for _, mapping in sorted(bindings.items()):
            mapped = tuple(
                f.rename(mapping) for f in effects if all(a in mapping for a in f.args)
            )
            shared = tuple(f for f in goal if entails(mapped, (f,)) == "supported")
            if not shared:
                continue
            missing, conflicts, unbound = [], [], []
            for fact in unpack(unit["entry"]):
                if not all(a in mapping for a in fact.args):
                    unbound.append(fact)
                    continue
                renamed = fact.rename(mapping)
                status = entails(known, (renamed,))
                if status == "unknown":
                    missing.append(renamed)
                elif status == "contradicted":
                    conflicts.append(renamed)
            matches.append(
                {
                    "unit": identity,
                    "match_kind": "full" if len(shared) == len(goal) else "partial",
                    "entity_mapping": mapping,
                    "matched_results": pack(shared),
                    "uncovered_results": pack(
                        tuple(f for f in goal if f not in shared)
                    ),
                    "missing_entry": pack(missing),
                    "entry_conflicts": pack(conflicts),
                    "unbound_entry": pack(unbound),
                }
            )
    return matches


def build_gap_report(plan, bank, bundle_id, records, projection, *, query_graphs=None):
    """Current-student queries over a fixed library; Codex chooses references."""
    from reweave.evolution.distillation.engine import condition_key

    graphs = plan.source_graphs if query_graphs is None else query_graphs
    allowed = {
        condition_key(i): t.name
        for t in plan.sources
        if t.name in graphs
        for i in t.dev
    }
    if {condition_key(r.request.instance) for r in records} != allowed.keys() or len(
        records
    ) != len(allowed):
        raise ValueError("gap evidence must cover exactly the source development panel")
    cached_h0 = (
        {o.record.record_id: o.record for o in plan.initial_observations}
        if bundle_id == plan.initial.record_id
        else {}
    )
    units = bank["units"]
    if set(units) != {
        occurrence_key(t, e) for t, g in plan.source_graphs.items() for e in g.edges
    }:
        raise ValueError("teacher unit bank differs from fixed source graphs")
    query_units = units if query_graphs is None else teacher_unit_bank(graphs)["units"]
    observed = projection.get("topology_records", {})
    if set(observed) != {r.record_id for r in records}:
        raise ValueError(
            "gap projection must cover exactly the current student records"
        )
    gaps, mechanisms, matches, protected, unresolved = [], {}, {}, [], []
    cache = {}
    for record in records:
        item = observed[record.record_id]
        task, node = item["task"], item["node_id"]
        if task != allowed[
            condition_key(record.request.instance)
        ] or record.failure_origin not in ("none", "agent"):
            raise ValueError("only settled source development records may create gaps")
        if record.request.split != "dev":
            raise ValueError("test evidence cannot create student gaps")
        if (
            record.request.bundle_id != bundle_id
            and cached_h0.get(record.record_id) != record
        ):
            raise ValueError("stale bundle: gap evidence differs from current student")
        graph = graphs[task]
        if node is not None and (
            node not in graph.nodes
            or (not record.official_success and graph.nodes[node].role == "goal")
        ):
            raise ValueError("invalid retained source node")
        retained = set(graph.completed_nodes()[node]) if node is not None else set()
        protected.append(
            {
                "record_id": record.record_id,
                "task": task,
                "retained_nodes": sorted(retained),
                "official_success": record.official_success,
                "interpretation": "Retained outcomes and successful behavior, not proof every alternative executed.",
            }
        )
        if record.official_success:
            continue
        if node is None:
            unresolved.append(
                {
                    "record_id": record.record_id,
                    "task": task,
                    "retained_node": None,
                    "reason": "Unknown retained state; no physical frontier or failure cause is invented.",
                }
            )
            continue
        known = facts(graph.matching.context(graph.matching.nodes[node]))
        frontier = {
            key
            for key, n in graph.nodes.items()
            if n.role != "branch"
            and key not in retained
            and set(n.predecessors) <= retained
        }
        local = [
            key
            for key, u in query_units.items()
            if u["task"] == task and u["target"] in frontier
        ]
        if not local:
            unresolved.append(
                {
                    "record_id": record.record_id,
                    "task": task,
                    "retained_node": node,
                    "reason": "No direct interface into the evidenced frontier; original teachers remain available.",
                }
            )
        for identity in local:
            unit = query_units[identity]
            desired = tuple(
                f
                for f in unpack(unit["effects"])
                if entails(known, (f,)) != "supported"
            )
            mechanism = json.dumps([task, node, unit["target"]], separators=(",", ":"))
            group = mechanisms.setdefault(
                mechanism,
                {
                    "id": mechanism,
                    "task": task,
                    "retained_node": node,
                    "frontier_node": unit["target"],
                    "record_ids": [],
                    "query_ids": [],
                    "candidate_units": [],
                },
            )
            if record.record_id not in group["record_ids"]:
                group["record_ids"].append(record.record_id)
            query = f"{record.record_id}:{identity}"
            group["query_ids"].append(query)
            gaps.append(
                {
                    "id": query,
                    "mechanism": mechanism,
                    "record_id": record.record_id,
                    "task": task,
                    "unit": identity,
                    "retained_node": node,
                    "frontier_node": unit["target"],
                    "known": pack(known),
                    "desired_results": pack(desired),
                    "citations": item["citations"],
                    "projection_basis": item["basis"],
                    "meaning": "Uncompleted frontier obligation; incoming interfaces are alternatives, not inferred failure causes.",
                }
            )
            key = (desired, known)
            if key not in cache:
                cache[key] = retrieve(desired, known, units)
            for match in cache[key]:
                matches.setdefault(mechanism, []).append({"query": query, **match})
                if match["unit"] not in group["candidate_units"]:
                    group["candidate_units"].append(match["unit"])
    for group in mechanisms.values():
        for key in ("record_ids", "query_ids", "candidate_units"):
            group[key].sort()
    for key in matches:
        matches[key].sort(
            key=lambda m: (
                m["query"],
                m["unit"],
                json.dumps(m["entity_mapping"], sort_keys=True),
            )
        )
    return {
        "schema": "student-gaps/v1",
        "student_bundle_id": bundle_id,
        "record_ids": [r.record_id for r in records],
        "mechanisms": [mechanisms[k] for k in sorted(mechanisms)],
        "gaps": gaps,
        "matches": matches,
        "protected_outcomes": protected,
        "unresolved": unresolved,
        "summary": {
            "failed_conditions": sum(not r.official_success for r in records),
            "mechanisms": len(mechanisms),
            "case_queries": len(gaps),
            "candidate_units": len(
                {u for g in mechanisms.values() for u in g["candidate_units"]}
            ),
            "bank_units": len(units),
        },
        "evaluation_coordinates_excluded": ["official_success"],
        "interpretation": "Compact current-student query index; no forced plan, ranking, top-k or executable-transfer claim.",
    }
