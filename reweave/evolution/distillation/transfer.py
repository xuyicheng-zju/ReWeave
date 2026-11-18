"""Condition-local source references and conservative empirical transfer loss.

q is completion from the first evidenced entry into a fixed condition during a
source development rollout, within that rollout's original remaining budget.
It is not a counterfactual simulator restart or a claim about arbitrary states.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from reweave.evolution.bundle import Asset
from reweave.evolution.graph.matching import match_graphs, semantic_query


class MechanismReference(Asset):
    key: str = Field(min_length=1)
    task: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    exit: str = Field(min_length=1)
    # Full source Bundle remains available: these fragments are not standalone.
    source_bundle_id: str = Field(min_length=1)
    source_edges: tuple[str, ...] = ()
    source_witnesses: tuple[dict, ...] = ()
    citations: tuple[str, ...] = ()
    programs: dict = Field(default_factory=dict)
    conditions: dict = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class ConditionalObservation(Asset):
    reference: str
    record_id: str
    entry: Literal["observed", "absent", "unknown"]
    outcome: Literal["achieved", "not_achieved", "unknown"]
    citations: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ConditionalRate(Asset):
    lower: float = Field(ge=0, le=1, allow_inf_nan=False)
    upper: float = Field(ge=0, le=1, allow_inf_nan=False)
    observed_entries: int = Field(ge=0, strict=True)
    unknown_entries: int = Field(ge=0, strict=True)
    absent_entries: int = Field(ge=0, strict=True)
    known_outcomes: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def bounds(self):
        if self.lower > self.upper or self.known_outcomes > self.observed_entries:
            raise ValueError("invalid conditional evidence bounds")
        return self


def mechanism_reference(teacher, query, paths, identity):
    edges = tuple(edge for edge, _ in paths[identity])
    graph = teacher.graph
    interface = query.interfaces[identity]
    witnesses = tuple(
        graph.edges[edge].witnesses[index] for edge, index in paths[identity]
    )
    start, end = graph.edges[edges[0]].source, graph.edges[edges[-1]].target
    return MechanismReference(
        key=f"{teacher.task}:{identity}",
        task=teacher.task,
        entry=(graph.nodes[start].meaning + "; " if start else "")
        + graph.edges[edges[0]].entry_conditions,
        exit=graph.nodes[end].meaning + "; " + graph.edges[edges[-1]].exit_conditions,
        source_bundle_id=teacher.bundle.record_id,
        source_edges=edges,
        source_witnesses=tuple(w.model_dump(mode="json") for w in witnesses),
        programs={
            p: graph.programs[p].model_dump(mode="json")
            for w in witnesses
            if (p := w.program_id) is not None
        },
        conditions={
            "catalog": {k: v.model_dump(mode="json") for k, v in query.catalog.items()},
            "entities": {
                k: v.model_dump(mode="json") for k, v in query.entities.items()
            },
            "entry": [
                v.model_dump(mode="json")
                for v in query.context(
                    (
                        *query.nodes.get(interface.source, ()),
                        *interface.entry,
                    )
                )
            ],
            "exit": [
                v.model_dump(mode="json")
                for v in query.context(
                    (
                        *query.nodes[interface.target],
                        *interface.exit,
                    )
                )
            ],
        },
    )


def aligned_references(teachers, seconds, *, literal_only=False):
    """Align literal source interfaces without expanding derived path envelopes.

    Unmatched and singleton task capabilities never compete on average similarity.
    Failed attempts remain evidence and acquire no invented positive reference rate.
    """
    graphs = {}
    for teacher in teachers:
        if teacher.graph is None or teacher.graph.matching is None:
            raise ValueError("conditional alignment requires encoded source graphs")
        query, paths = semantic_query(teacher.graph)
        graphs[teacher.task] = (teacher, query, paths)
    alignments, selected = [], {}
    for name, (teacher, query, paths) in graphs.items():
        for edge in teacher.graph.edges:
            ref = mechanism_reference(teacher, query, paths, edge)
            selected[ref.key] = ref
        if literal_only:
            continue
        for target, (other, reference, other_paths) in graphs.items():
            if name == target:
                continue
            result = match_graphs(query, reference, seconds=seconds)
            alignments.append({"query": name, "reference": target, "alignment": result})
            for relation in result["relations"]:
                if (
                    relation["coverage"] > 0
                    and relation["reference"][0] == "transition"
                ):
                    identity = relation["reference"][1]
                    if other_paths[identity]:
                        ref = mechanism_reference(
                            other, reference, other_paths, identity
                        )
                        selected[ref.key] = ref
    counts = {
        r.task: sum(s.task == r.task for s in selected.values())
        for r in selected.values()
    }
    refs = tuple(
        r.model_copy(update={"weight": 1 / (len(counts) * counts[r.task])})
        for _, r in sorted(selected.items())
    )
    return refs, alignments


def complete_source_references(teachers, references):
    """An older precomputed alignment cannot omit a source-private interface."""
    selected = {r.key: r for r in references}
    if len(selected) != len(references):
        raise ValueError("duplicate source mechanism contract")
    added = False
    for teacher in teachers:
        if teacher.graph is None or teacher.graph.matching is None:
            continue
        query, paths = semantic_query(teacher.graph)
        for edge in teacher.graph.edges:
            if any(
                r.task == teacher.task and r.source_edges == (edge,)
                for r in selected.values()
            ):
                continue
            ref = mechanism_reference(teacher, query, paths, edge)
            if ref.key in selected:
                raise ValueError(
                    "source mechanism key conflicts with a literal interface"
                )
            selected[ref.key] = ref
            added = True
    if not added:
        return references
    counts = {
        r.task: sum(s.task == r.task for s in selected.values())
        for r in selected.values()
    }
    return tuple(
        r.model_copy(update={"weight": 1 / (len(counts) * counts[r.task])})
        for _, r in sorted(selected.items())
    )


def conditional_rates(references, observations, records, task_instances):
    """Require exhaustive reference/episode labels and evidence-bound citations.

    Unknown entry may be a failed opportunity. Unknown outcome never earns
    credit. Bounds expose uncertainty rather than turning it into success.
    """
    indexed = {r.record_id: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate physical receipt")
    expected = {
        (ref.key, r.record_id)
        for ref in references
        for r in records
        if r.request.instance.task in task_instances[ref.task]
    }
    supplied = [(o.reference, o.record_id) for o in observations]
    if len(supplied) != len(set(supplied)) or set(supplied) != expected:
        raise ValueError("conditional observations must cover every source opportunity")
    for obs in observations:
        record = indexed[obs.record_id]
        if record.request.split != "dev" or record.failure_origin in {
            "unknown",
            "infrastructure",
        }:
            raise ValueError("conditional evidence must be settled source development")
        paths = {f.path for f in record.evidence}
        if not set(obs.citations) <= paths:
            raise ValueError("conditional citation is not original episode evidence")
        if obs.entry != "observed" and obs.outcome != "unknown":
            raise ValueError("absent or unknown entry cannot assert an outcome")
    rates = {}
    for ref in references:
        group = [o for o in observations if o.reference == ref.key]
        reached = [o for o in group if o.entry == "observed"]
        unknown_entry = sum(o.entry == "unknown" for o in group)
        successes = sum(o.outcome == "achieved" for o in reached)
        unknown_exit = sum(o.outcome == "unknown" for o in reached)
        possible = len(reached) + unknown_entry
        rates[ref.key] = {
            "lower": successes / possible if possible else 0.0,
            "upper": (successes + unknown_exit + unknown_entry) / possible
            if possible
            else 1.0,
            "observed_entries": len(reached),
            "unknown_entries": unknown_entry,
            "absent_entries": sum(o.entry == "absent" for o in group),
            "known_outcomes": len(reached) - unknown_exit,
        }
    return rates


def transfer_loss(references, anchors, candidate):
    keys = {r.key for r in references}
    if set(anchors) != keys or set(candidate) != keys:
        raise ValueError("missing fixed conditional reference")
    drops = {
        key: max(0.0, anchors[key]["upper"] - candidate[key]["lower"]) for key in keys
    }
    return {
        "conditional_drop": drops,
        "conditional_loss": sum(
            sum(r.weight * drops[r.key] for r in references if r.task == task)
            / sum(r.weight for r in references if r.task == task)
            for task in {r.task for r in references}
        )
        / len({r.task for r in references})
        if references
        else 0.0,
        "conditional_rates": candidate,
        "conditional_complete": all(
            v["observed_entries"] > 0 and v["lower"] == v["upper"]
            for v in (*anchors.values(), *candidate.values())
        ),
    }
