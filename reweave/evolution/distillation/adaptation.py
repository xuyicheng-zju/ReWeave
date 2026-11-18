"""Outcome-anchored source suggestions for shared program synthesis."""

from reweave.evolution.distillation.fragment_logic import Fact, entails, normalize


def unpack(rows):
    return normalize(
        Fact(r["fluent"], tuple(r["arguments"]), frozenset(r["values"])) for r in rows
    )


def pack(rows):
    return [
        {"fluent": f.name, "arguments": list(f.args), "values": sorted(f.values)}
        for f in rows
    ]


def anchor_binding(source, query):
    if (
        source.name != query.name
        or len(source.args) != len(query.args)
        or not source.values <= query.values
    ):
        return None
    phi, reverse = {}, {}
    for a, b in zip(source.args, query.args):
        if (a in phi and phi[a] != b) or (b in reverse and reverse[b] != a):
            return None
        phi[a], reverse[b] = b, a
    return phi


def adaptation_proposals(bank, inventory):
    """Enumerate partial role bindings supported by at least one desired result.

    Context conflicts are reported, never silently weakened or treated as an
    executable repair plan. Unbound source roles remain explicitly source-local.
    """
    units = [u for rows in bank.values() for u in rows]
    declared_all = {u.id: unpack(inventory[u.id]["declared_result"]) for u in units}
    # Reserved evaluator coordinate in the shared catalog: an overall verdict
    # cannot supply an operation correspondence. It stays in original evidence.
    declared = {
        key: tuple(f for f in rows if f.name != "official_success")
        for key, rows in declared_all.items()
    }
    desired = {
        u.id: tuple(f for f in declared[u.id] if entails(u.pre, (f,)) != "supported")
        for u in units
    }
    proposals, covered, result_covered = [], set(), set()
    for q in units:
        if not desired[q.id]:
            continue
        for r in units:
            if r.source == q.source or r.assurance != "observed":
                continue
            bindings = {}
            for goal in desired[q.id]:
                for result in declared[r.id]:
                    phi = anchor_binding(result, goal)
                    if phi is not None:
                        bindings[tuple(sorted(phi.items()))] = phi
            for _, phi in sorted(bindings.items()):
                mapped = tuple(
                    f.rename(phi)
                    for f in declared[r.id]
                    if all(a in phi for a in f.args)
                )
                shared = tuple(
                    f for f in desired[q.id] if entails(mapped, (f,)) == "supported"
                )
                if not shared:
                    continue
                missing, conflicts, unbound = [], [], []
                for f in r.pre:
                    if not all(a in phi for a in f.args):
                        unbound.append(f)
                        continue
                    mapped_entry = f.rename(phi)
                    status = entails(q.pre, (mapped_entry,))
                    if status == "contradicted":
                        conflicts.append(mapped_entry)
                    elif status == "unknown":
                        missing.append(mapped_entry)
                uncovered = tuple(f for f in desired[q.id] if f not in shared)
                proposals.append(
                    {
                        "query": q.id,
                        "reference": r.id,
                        "entity_mapping": phi,
                        "mapping_scope": "shared_result_roles_only",
                        "relation": "adaptation",
                        "matched_results": pack(shared),
                        "uncovered_results": pack(uncovered),
                        "entry_additions": pack(missing),
                        "entry_conflicts": pack(conflicts),
                        "unbound_reference_entry": pack(unbound),
                        "reference_result": pack(declared_all[r.id]),
                        "status": "synthesis_proposal",
                        "callable": False,
                    }
                )
                covered.add(q.id)
                result_covered.update((q.id, f) for f in shared)
    eligible = sum(bool(v) for v in desired.values())
    denominator = sum(map(len, desired.values()))
    metrics = {
        "excluded_evaluation_coordinates": ["official_success"],
        "source_units": len(units),
        "eligible_source_units": eligible,
        "supported_source_units": len(covered),
        "guidance_coverage": len(covered) / eligible if eligible else None,
        "desired_results": denominator,
        "supported_results": len(result_covered),
        "result_support": len(result_covered) / denominator if denominator else None,
        "interpretation": "Source-only synthesis guidance availability; not transfer success or executable coverage.",
    }
    return proposals, metrics
