"""Topology-local reuse proposals with exact source evidence and explicit limits.

Symbolic substitution suggests author edits; it never launches a fragment or
turns a failed transition into a successful standalone implementation.
"""

from __future__ import annotations

from reweave.evolution.distillation.fragment_logic import Fact, Fragment, compare
from reweave.evolution.graph.conditions import merge_catalogs
from reweave.evolution.graph.matching import semantic_query


def facts(values):
    return tuple(Fact(f.fluent, f.arguments, frozenset(f.values)) for f in values)


def serialize(values):
    return [
        {"fluent": f.name, "arguments": list(f.args), "values": sorted(f.values)}
        for f in values
    ]


def extract_fragments(graphs, *, entry_mode="local"):
    """Only literal interfaces. Intended outcomes retain achieved/failed witnesses."""
    if entry_mode not in {"local", "accumulated"}:
        raise ValueError("entry mode must be local or accumulated")
    merge_catalogs(*(g.matching.catalog for g in graphs.values() if g.matching))
    fragments, inventory = {}, {}
    for task, graph in sorted(graphs.items()):
        if graph.matching is None:
            raise ValueError(f"source topology needs condition encoding: {task}")
        query, _ = semantic_query(graph, witness_paths=False)
        rows = []
        for edge_id, interface in sorted(query.interfaces.items()):
            edge = graph.edges[edge_id]
            key = f"{task}:{edge_id}"
            entry = interface.entry
            if entry_mode == "accumulated" and interface.source is not None:
                entry = (*query.nodes[interface.source], *entry)
            pre = facts(query.context(entry))
            post = facts(
                query.context((*query.nodes[interface.target], *interface.exit))
            )
            positive = [w for w in edge.witnesses if w.outcome == "achieved"]
            fragment = Fragment(
                key,
                pre,
                post,
                writes=None,
                source=task,
                evidence=tuple(
                    f"{w.episode_id}:{w.first_frame}:{w.last_frame}:{w.outcome}"
                    for w in edge.witnesses
                ),
                assurance="observed" if positive else "candidate",
            )
            rows.append(fragment)
            inventory[key] = {
                "task": task,
                "edge": edge_id,
                "source_node": edge.source,
                "target_node": edge.target,
                "entry": serialize(pre),
                "intended_result": serialize(post),
                "declared_result": serialize(
                    facts((*query.nodes[interface.target], *interface.exit))
                ),
                "witnesses": [w.model_dump(mode="json") for w in edge.witnesses],
                "references": [r.model_dump(mode="json") for r in edge.references],
                "program_ids": sorted(
                    {w.program_id for w in edge.witnesses if w.program_id}
                ),
                "has_achieved_witness": bool(positive),
                "effects": "unknown",
                "standalone_entry": None,
            }
        fragments[task] = rows
    return fragments, inventory


def fragment_transfer(graphs, *, entry_mode="local"):
    """Exact local comparison plus condition-by-condition completion suggestions.

    Multi-source completions are synthesis proposals, not certified plans: the
    graph does not declare complete effects or isolated executable entry points.
    No fact deletion, semantic embedding verdict, global map or solver timeout.
    """
    bank, inventory = extract_fragments(graphs, entry_mode=entry_mode)
    proposals, pairs = [], []
    witnessed = [f for rows in bank.values() for f in rows if f.assurance == "observed"]
    for task, queries in bank.items():
        for other, references in bank.items():
            if task == other:
                continue
            direct, incomplete = set(), set()
            for query in queries:
                if not query.post:
                    continue
                for reference in references:
                    if reference.assurance != "observed":
                        continue
                    for match in compare(query, reference):
                        missing = match["debt"]
                        (incomplete if missing else direct).add(query.id)
                        completions = []
                        # A supplier is compared against one missing result.
                        # Preserve the supplier's own missing entry conditions.
                        for condition in missing:
                            need = Fragment(query.id, query.pre, (condition,))
                            objects = {
                                a
                                for f in (*query.pre, *query.post, *missing)
                                for a in f.args
                            }
                            for supplier in witnessed:
                                if supplier.id in {query.id, reference.id}:
                                    continue
                                for continuation in compare(need, supplier, objects):
                                    completions.append(
                                        {
                                            "condition": serialize((condition,))[0],
                                            "fragment": supplier.id,
                                            "entity_mapping": continuation["phi"],
                                            "remaining_entry": serialize(
                                                continuation["debt"]
                                            ),
                                            "effects_checked": False,
                                        }
                                    )
                        proposals.append(
                            {
                                "query": query.id,
                                "reference": reference.id,
                                "entity_mapping": match["phi"],
                                "relation": "direct"
                                if not missing
                                else "requires_completion",
                                "missing_entry": serialize(missing),
                                "completion_options": completions,
                                "status": "synthesis_proposal",
                                "callable": False,
                                "validation": "compose in shared C/W/L and evaluate on source development",
                            }
                        )
            pairs.append(
                {
                    "query": task,
                    "reference": other,
                    "direct": len(direct),
                    "requires_completion": len(incomplete - direct),
                    "query_interfaces": len(queries),
                }
            )
    from reweave.evolution.distillation.adaptation import adaptation_proposals

    adaptations, guidance = adaptation_proposals(bank, inventory)
    return {
        "schema": "topology-fragments/v2",
        "adaptation_proposals": adaptations,
        "guidance_metrics": guidance,
        "primary_guidance": "adaptation_proposals",
        "entry_mode": entry_mode,
        "complete": True,
        "scope": "literal source interfaces; per-fragment entity substitutions",
        "inventory": inventory,
        "pairs": pairs,
        "proposals": proposals,
        "limits": [
            "Symbolic relations concern annotated outcomes, not universal physical guarantees.",
            "Missing effects and standalone entries require author integration and physical validation.",
            "Unmatched and failed source interfaces remain in the inventory as evidence.",
        ],
    }
