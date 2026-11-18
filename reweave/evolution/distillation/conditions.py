"""Source-only native encoding of missing physical matching conditions."""

from __future__ import annotations

from pydantic import TypeAdapter

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.engine import development_task
from reweave.evolution.graph.conditions import GraphSemantics, merge_catalogs
from reweave.evolution.validation.admission import _read_workspace_file
from reweave.utils.persistence import atomic_json


class NativeSourceConditions(NativeSharedAuthor):
    """Annotate existing source graphs without changing witnesses or programs."""

    source_queries = False

    instructions = """Encode physical matching conditions for the admitted SOURCE
teacher graphs. Read context/input.json, teachers and original source evidence.
Write candidate/conditions.json: a JSON object mapping every teacher task name to
GraphSemantics, using context/conditions.schema.json. Keep one compatible fluent
catalog across tasks, with explicit typed entities, finite value domains, state
conditions, and transition entry/exit conditions. Conditions must cover exactly
the original graph nodes and edges. Preserve uncertainty: absent evidence is
unknown, not success or contradiction. Entity names do not imply physical
properties. Do not invent task success or rewrite original topology, witnesses,
programs or results. No student construction, physical work, target evidence,
external retrieval or decision probes. Return annotations for every source task.
First compare meanings across ALL sources and choose one shared catalog. Reuse
one coordinate for the same physical property despite different task wording.
Keep physical state, action phase, and task completion as distinct meanings;
do not equate them merely to obtain matches. Preserve task-specific conditions
and record unresolved abstraction differences in uncertainties. Return the full
same catalog in every graph, including coordinates unused in that task.
"""
    prompt = "Encode source graph conditions into candidate/conditions.json only."

    def prepare(self, request, directory):
        if not request["teachers"] or any(
            t["graph"] is None for t in request["teachers"]
        ):
            raise ValueError(
                "source encoding requires existing canonical teacher graphs"
            )
        if request["method"] != "direct":
            raise ValueError(
                "source encoding uses raw evidence before any explicit alignment"
            )
        workspace, readable = super().prepare(request, directory)
        atomic_json(
            workspace / "context/conditions.schema.json",
            TypeAdapter(dict[str, GraphSemantics]).json_schema(),
        )
        return workspace, readable

    async def admit(self, request, workspace, audit):
        encoded = TypeAdapter(dict[str, GraphSemantics]).validate_json(
            _read_workspace_file(workspace, "candidate/conditions.json")
        )
        tasks = {teacher["task"] for teacher in request["teachers"]}
        if encoded.keys() != tasks:
            raise ValueError("condition encoding must cover exactly the source tasks")
        catalog = merge_catalogs(*(value.catalog for value in encoded.values()))
        graphs = {}
        for teacher in request["teachers"]:
            if teacher["graph"] is None:
                raise ValueError("cannot encode a missing canonical source graph")
            original = SemanticGraph.model_validate(teacher["graph"])
            payload = original.model_dump(mode="json")
            payload["matching"] = (
                encoded[teacher["task"]]
                .model_copy(update={"catalog": catalog})
                .model_dump(mode="json")
            )
            annotated = SemanticGraph.model_validate(payload)
            if original.model_dump(exclude={"matching"}) != annotated.model_dump(
                exclude={"matching"}
            ):
                raise ValueError("source encoding changed original graph evidence")
            graphs[teacher["task"]] = annotated.model_dump(mode="json")
        atomic_json(
            audit / "source_condition_validation.json",
            {
                "source_tasks": sorted(tasks),
                "originals_preserved": True,
                "physical_validation": False,
            },
        )
        return {"graphs": graphs}


def encoding_request(plan):
    """No unseen task metadata or held-out evidence enters the encoding job."""
    return {
        "round": 0,
        "mode": "consolidation",
        "method": "direct",
        "tasks": [development_task(plan, t) for t in plan.sources],
        "new_tasks": [],
        "teachers": [t.model_dump(mode="json") for t in plan.teachers],
        "parent": plan.initial.model_dump(mode="json"),
        "anchors": {},
        "criteria": {},
        "conditions": plan.conditions.model_dump(mode="json"),
        "alignment_seconds": plan.alignment_seconds,
        "feedback": [],
        "history": [],
    }
