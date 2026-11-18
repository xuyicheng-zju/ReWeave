# Modified for the ReWeave namespace.
"""Observe mutation exposure and matched official outcomes without causal claims."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.authoring.candidate import AuthorPlan
from reweave.evolution.search.matrix import evaluation_cells


def _matches(value, expected):
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(
            key in value and _matches(value[key], item)
            for key, item in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _matches(left, right)
                for left, right in zip(value, expected, strict=True)
            )
        )
    if isinstance(value, bool) or isinstance(expected, bool):
        return type(value) is type(expected) and value == expected
    return value == expected


def _event_matches(event, expectation):
    return event.get("kind") == expectation.kind and _matches(event, expectation.fields)


def observe_probes(record, probes):
    """Read original event positions; an effect must occur after activation."""
    original = next(
        (
            item
            for item in record.evidence
            if item.path.endswith("/harness_context/bundle_events.jsonl")
        ),
        None,
    )
    observations = [
        {
            "claim": probe.claim,
            "activation": None,
            "effect": None,
            "status": "unknown" if original is None else "not_observed",
        }
        for probe in probes
    ]
    if original is None or not probes:
        return observations
    original.verify()
    with Path(original.path).open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            for probe, observation in zip(probes, observations, strict=True):
                reference = {**original.model_dump(), "pointer": f"/{index}"}
                if observation["activation"] is None and _event_matches(
                    event, probe.activation
                ):
                    observation["activation"] = reference
                    observation["status"] = "activation_observed"
                elif (
                    observation["activation"] is not None
                    and observation["effect"] is None
                    and probe.effect is not None
                    and _event_matches(event, probe.effect)
                ):
                    observation["effect"] = reference
                    observation["status"] = "local_effect_observed"
    return observations


def review_mutation(proposal, candidate, records):
    """Use selected same-execution cells, preserving unknown and single-shot limits."""
    try:
        plan = AuthorPlan.model_validate_json(candidate.summary)
    except ValueError:
        # Imported historical/fixture summaries can be prose; never invent probes.
        plan = None
    probes = plan.probes if plan is not None else ()
    protocol = proposal.task_protocol
    bundles = (proposal.parent.bundle_id, candidate.bundle.record_id)
    current = tuple(
        r
        for r in records
        if r.request.execution == protocol.execution_id
        and r.request.bundle_id in bundles
    )
    cells = evaluation_cells(protocol, bundles, current)
    prior_receipts = {r.record_id for r in proposal.evidence}
    conditions = []
    for instance in protocol.instances:
        parent = cells.get((bundles[0], instance.record_id))
        child = cells.get((bundles[1], instance.record_id))
        comparable = (
            parent is not None
            and child is not None
            and (
                parent.request.conditions == child.request.conditions
                and parent.request.split == child.request.split
            )
        )
        measured = comparable and all(
            r.failure_origin in {"none", "agent"} for r in (parent, child)
        )
        before = parent.official_success if measured else None
        after = child.official_success if measured else None
        conditions.append(
            {
                "condition_id": instance.record_id,
                "parent_receipt_id": parent.record_id if parent else None,
                "candidate_receipt_id": child.record_id if child else None,
                "receipt_in_author_inputs": child.record_id in prior_receipts
                if child
                else None,
                "comparable": comparable,
                "official_before": before,
                "official_after": after,
                "official_change": "unknown"
                if not measured
                else "gained"
                if after and not before
                else "lost"
                if before and not after
                else "unchanged",
                "probes": observe_probes(child, probes)
                if child
                else [
                    {
                        "claim": probe.claim,
                        "activation": None,
                        "effect": None,
                        "status": "unknown",
                    }
                    for probe in probes
                ],
            }
        )
    return {
        "probes_declared": bool(probes),
        "conditions": conditions,
        "causal_improvement_established": False,
        "reliability_established": False,
        "meaning": "A missing activation signal leaves the intervention untested. A signal only establishes the declared observable event, not an internal branch or causality. Later local effects and matched official changes remain separate. Receipts already in the author's frozen inputs stay prior evidence; content aliases share measurements and never add independent trials. One observed rollout does not establish reliability; no extra physical trial is launched.",
    }
