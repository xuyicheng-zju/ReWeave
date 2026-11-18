"""Separate source-only native measurement of fixed conditional mechanisms."""

from __future__ import annotations

from pydantic import TypeAdapter

from reweave.evolution.contracts import EvaluationRecord
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.engine import development_task
from reweave.evolution.distillation.transfer import (
    ConditionalObservation,
    MechanismReference,
    conditional_rates,
)
from reweave.evolution.validation.admission import _read_workspace_file
from reweave.utils.persistence import atomic_json


class NativeConditionalAssessment(NativeSharedAuthor):
    source_queries = False

    instructions = """You are the independent conditional-outcome analyst, not the
student author. Read context/measurement.json and only its original source-dev
receipts. Do not edit C/W/L, run experiments, infer local success from official
whole-task success, or access held-out data. Treat mechanism entry and exit as
fixed; never weaken them to fit a student. Inspect original observations/images
and ordered action/decision evidence. Use the FIRST evidenced entry per mechanism
in each rollout, and assess completion before that rollout's original budget ends.
An absent entry requires evidence that the condition never occurred; incomplete
observability is unknown. Unknown/absent entry requires unknown outcome. Cite exact
original evidence file paths and explain the entry and outcome with trace positions
or image/frame identifiers. Do not count retry attempts as independent episodes.
For operation=measure, write candidate/measurement.json, one ConditionalObservation
for EVERY requested reference/record pair of the same source task, using the schema.
For operation=discover, write candidate/references.json: a list of fixed physical
MechanismReference contracts for each requested source task, grounded in the supplied
newly acquired harness's dev evidence. Preserve coupled dependencies by referring to
the complete source bundle. Include citations of original receipt evidence paths for each
contract. Use task-prefixed unique keys, empty programs and
source_edges unless a canonical graph supplies them. Do not assert counterfactual
physical outcomes or use task IDs as executable dispatch rules. These are measurement
contracts, not standalone executable skills. No other operation is authorized.
"""
    prompt = "Read context/measurement.json and produce only the requested measurement artifact."

    def prepare(self, request, directory):
        workspace, readable = super().prepare(request, directory)
        atomic_json(workspace / "context/measurement.json", request["measurement"])
        schema = (
            list[ConditionalObservation]
            if request["measurement"]["operation"] == "measure"
            else list[MechanismReference]
        )
        atomic_json(
            workspace / "context/measurement.schema.json",
            TypeAdapter(schema).json_schema(),
        )
        return workspace, readable

    async def admit(self, request, workspace, audit):
        measure = request["measurement"]
        if measure["operation"] == "discover":
            refs = TypeAdapter(list[MechanismReference]).validate_json(
                _read_workspace_file(workspace, "candidate/references.json")
            )
            teachers = {
                t["task"]: t["bundle"]["record_id"] for t in request["teachers"]
            }
            if {r.task for r in refs} != teachers.keys() or len(
                {r.key for r in refs}
            ) != len(refs):
                raise ValueError(
                    "discovery must cover every task with unique references"
                )
            evidence = {
                t["name"]: {
                    f["path"]
                    for record in request["feedback"]
                    if record["request"]["instance"]["task"]
                    in {i["task"] for i in t["dev"]}
                    for f in record["evidence"]
                }
                for t in request["tasks"]
            }
            for ref in refs:
                if not ref.citations or not set(ref.citations) <= evidence[ref.task]:
                    raise ValueError(
                        "discovered condition requires original source citations"
                    )
                if (
                    ref.source_bundle_id != teachers[ref.task]
                    or not ref.key.startswith(ref.task + ":")
                    or ref.programs
                    or ref.source_edges
                    or ref.source_witnesses
                ):
                    raise ValueError(
                        "discovery requires the exact complete source bundle"
                    )
            counts = {r.task: sum(s.task == r.task for s in refs) for r in refs}
            return {
                "references": [
                    r.model_copy(
                        update={"weight": 1 / (len(counts) * counts[r.task])}
                    ).model_dump(mode="json")
                    for r in refs
                ]
            }
        refs = TypeAdapter(list[MechanismReference]).validate_python(
            measure["references"]
        )
        observations = TypeAdapter(list[ConditionalObservation]).validate_json(
            _read_workspace_file(workspace, "candidate/measurement.json")
        )
        records = [EvaluationRecord.model_validate(r) for r in request["feedback"]]
        tasks = {t["name"]: {i["task"] for i in t["dev"]} for t in request["tasks"]}
        rates = conditional_rates(refs, observations, records, tasks)
        return {
            "rates": rates,
            "observations": [o.model_dump(mode="json") for o in observations],
        }


def measurement_request(plan, bundle, records, references=(), *, discover_tasks=()):
    # Common author workspace builder is reused, but NO candidate history,
    # previous labels or source outcomes bias this separate measurement session.
    tasks = tuple(
        t for t in plan.sources if not discover_tasks or t.name in discover_tasks
    )
    return {
        "round": 0,
        "mode": plan.mode,
        "method": "direct",
        "tasks": [development_task(plan, t) for t in tasks],
        "new_tasks": [],
        "parent": bundle.model_dump(mode="json"),
        "teachers": [
            {
                "task": t.name,
                "bundle": bundle.model_dump(mode="json"),
                "learned_tasks": [s.name for s in plan.sources],
                "graph": None,
                "evidence": [],
            }
            for t in tasks
        ]
        if discover_tasks
        else [],
        "anchors": {},
        "criteria": {},
        "history": [],
        "conditions": plan.conditions.model_dump(mode="json"),
        "alignment_seconds": plan.alignment_seconds,
        "feedback": [r.model_dump(mode="json") for r in records],
        "measurement": {
            "operation": "discover" if discover_tasks else "measure",
            "references": [r.model_dump(mode="json") for r in references],
        },
    }
