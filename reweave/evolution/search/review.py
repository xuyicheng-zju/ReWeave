# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Comparable retained-goal vectors and search diagnostics on one current graph."""

from __future__ import annotations

from reweave.evolution.graph.projection import map_row


def compare_versions(progress_map, before, after):
    """Keep gained and lost goals separate; unknown state is never an empty set."""
    left = map_row(progress_map, before)
    right = map_row(progress_map, after)
    if left.keys() != right.keys():
        raise ValueError("comparison requires the same development conditions")
    milestones = set(progress_map["milestone_nodes"])
    conditions = {}
    for key, old in left.items():
        new = right[key]
        known = (
            old["completed_nodes"] is not None
            and new["completed_nodes"] is not None
            and old.get("interpretation_id") == new.get("interpretation_id")
        )
        gained = (
            sorted(
                (set(new["completed_nodes"]) - set(old["completed_nodes"])) & milestones
            )
            if known
            else None
        )
        lost = (
            sorted(
                (set(old["completed_nodes"]) - set(new["completed_nodes"])) & milestones
            )
            if known
            else None
        )
        official_known = (
            old["official_success"] is not None and new["official_success"] is not None
        )
        same_achievement = (
            known
            and not gained
            and not lost
            and official_known
            and old["official_success"] == new["official_success"]
        )
        conditions[key] = {
            "before_receipt_id": old["receipt_id"],
            "after_receipt_id": new["receipt_id"],
            "gained_nodes": gained,
            "lost_nodes": lost,
            "official_gained": new["official_success"] and not old["official_success"]
            if official_known
            else None,
            "official_lost": old["official_success"] and not new["official_success"]
            if official_known
            else None,
            "physical_step_delta": new["usage"]["physical_steps"]
            - old["usage"]["physical_steps"]
            if same_achievement
            and old.get("usage") is not None
            and new.get("usage") is not None
            and old["usage"].get("physical_steps_complete", True)
            and new["usage"].get("physical_steps_complete", True)
            else None,
        }
    unknown = any(
        c["gained_nodes"] is None or c["official_gained"] is None
        for c in conditions.values()
    )
    gained = any(c["gained_nodes"] or c["official_gained"] for c in conditions.values())
    lost = any(c["lost_nodes"] or c["official_lost"] for c in conditions.values())
    status = (
        "unknown"
        if unknown
        else "tradeoff"
        if gained and lost
        else "improved"
        if gained
        else "regressed"
        if lost
        else "unchanged"
    )
    return {"status": status, "conditions": conditions}


def review_search(progress_map, versions, state, parent):
    """Reproject all historical mutations before diagnosing stagnation/tradeoffs.

    These diagnostics do not change scalar parent ranking, width, or round caps.
    Incomparable retained-goal vectors remain distinct complete implementations.
    """
    by_id = {v.version_id: v for v in versions}
    mutations = {}
    rounds = {}
    for identity in state.get("order", ()):
        value = state["versions"][identity]
        proposal = state.get("proposals", {}).get(value["proposal_id"])
        if proposal is None or identity not in by_id:
            continue
        before = proposal["parent"]["version_id"]
        if before not in by_id:
            continue
        comparison = compare_versions(progress_map, before, identity)
        mutations[identity] = comparison
        round_index = proposal["round_index"]
        rounds[round_index] = rounds.get(round_index, False) or comparison[
            "status"
        ] in {"improved", "unknown"}
    trailing = 0
    for round_index in sorted(rounds):
        trailing = 0 if rounds[round_index] else trailing + 1
    return {
        "graph_id": progress_map["graph_id"],
        "parent_version_id": parent.version_id,
        "mutations": mutations,
        "consecutive_without_observed_retained_gain": trailing,
        "meaning": "Node deltas require the same interpretation and development condition; first-observation nodes from different graphs are incomparable. Alternative goal nodes share the official-completion verdict. Unknown is not stagnation. Step deltas require equal observed goal sets, official outcome and complete counters; they are descriptive rollout costs, not causal efficiency. Scalar parent selection remains the published objective.",
    }
