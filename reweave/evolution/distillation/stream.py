"""Joint continual stages over one shared Harness and a growing frozen graph bank."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.bundle import AgentBundle, new_record_id
from reweave.evolution.contracts import EvaluationRecord, Instance, SemanticGraph
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedInitialObservation,
    SharedPlan,
    SharedTask,
    SharedTeacher,
    bind_json,
    condition_key,
)
from reweave.utils.persistence import atomic_json


def stage_task(task, new):
    seeds = (11, 12, 13) if new else (0,)
    template = task.dev[0]
    return task.model_copy(
        update={
            "test": (),
            "dev": tuple(
                Instance(
                    record_id=new_record_id(),
                    task=template.task,
                    initialization={
                        **template.initialization,
                        "seed": seed,
                        "reset_state_id": seed,
                    },
                )
                for seed in seeds
            ),
        }
    )


async def run_stream(
    directory,
    initial_plan,
    tasks,
    stages,
    author,
    evaluator,
    assessor=None,
    task_topology=None,
    *,
    initial_checkpoint,
    task_runtime_manifests,
    infrastructure_retries=0,
):
    """Each stage jointly updates one student; there is no per-new-task search."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_name = {t.name: t for t in tasks}
    seen = [t.name for t in initial_plan.sources]
    exposed = [name for stage in stages for name in stage]
    if (
        not stages
        or any(not stage for stage in stages)
        or len(set(exposed)) != len(exposed)
        or set(exposed) & set(seen)
        or set(exposed) | set(seen) != by_name.keys()
    ):
        raise ValueError(
            "joint stages must expose every new task exactly once after the initial sources"
        )
    current = json.loads(Path(initial_checkpoint).read_text())
    initial = AgentBundle.model_validate(current["bundle"])
    if initial.runtime != initial_plan.initial.runtime or set(
        current["source_graphs"]
    ) != set(seen):
        raise ValueError(
            "initial shared checkpoint or fixed graph bank differs from the declared source profile"
        )
    bind_json(
        directory / "stream.json",
        {
            "initial_checkpoint": str(initial_checkpoint),
            "initial_plan": initial_plan.model_dump(mode="json"),
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "stages": stages,
        },
    )
    checkpoints = []
    previous_update = None
    previous_plan = None
    for stage_index, stage in enumerate(stages, 1):
        parent = AgentBundle.model_validate(current["bundle"])
        update = directory / "updates" / f"{stage_index:03d}"
        path = update / "plan.json"
        if path.exists():
            plan = SharedPlan.model_validate_json(path.read_text())
            if (
                plan.initial != parent
                or plan.initial_learned_tasks != tuple(seen)
                or plan.new_tasks != tuple(stage)
            ):
                raise ValueError(
                    "joint stage recovery changed the parent, old task bank, or new task pair"
                )
        else:
            graphs = {
                name: SemanticGraph.model_validate(value)
                for name, value in current["source_graphs"].items()
            }
            value = initial_plan.model_dump(mode="json")
            value.update(
                record_id=new_record_id(),
                mode="continual",
                sources=[
                    stage_task(by_name[name], name in stage).model_dump(mode="json")
                    for name in (*seen, *stage)
                ],
                unseen=[],
                initial=parent.model_dump(mode="json"),
                initial_learned_tasks=seen,
                new_tasks=list(stage),
                teachers=[
                    SharedTeacher(
                        task=name,
                        bundle=parent,
                        learned_tasks=tuple(seen),
                        graph=graphs[name],
                    ).model_dump(mode="json")
                    for name in seen
                ],
                first_acquired_scores={},
                conditional_retention=False,
                teacher_retention=True,
                sequential_updates=True,
                checkpoint_selection="min_loss",
                fixed_topology_scoring=False,
                initial_observations=[],
                initial_topology_score=None,
                require_cached_h0=False,
                mechanism_references=[],
                alignment_records=[],
                first_conditional_rates={},
                source_graphs=current["source_graphs"],
                loss_threshold=initial_plan.loss_threshold,
            )
            if previous_update is not None:
                allowed = {
                    condition_key(i): t.name
                    for t in (SharedTask.model_validate(v) for v in value["sources"])
                    for i in t.dev
                }
                reused = {
                    condition_key(o.record.request.instance): o
                    for o in previous_plan.initial_observations
                    if o.record.request.bundle_id == parent.record_id
                    and condition_key(o.record.request.instance) in allowed
                }
                for record_path in previous_update.rglob("shared_result.json"):
                    record = EvaluationRecord.model_validate_json(
                        record_path.read_text()
                    )
                    key = condition_key(record.request.instance)
                    if (
                        record.request.bundle_id != parent.record_id
                        or record.request.split != "dev"
                        or record.failure_origin not in {"none", "agent"}
                        or key not in allowed
                    ):
                        continue
                    if key in reused and reused[key].record != record:
                        raise ValueError(
                            "multiple physical results exist for one unchanged stage-entry condition"
                        )
                    from reweave.evolution.distillation.graph_initialization import (
                        artifact,
                    )

                    reused[key] = SharedInitialObservation(
                        bundle=parent,
                        runtime=task_runtime_manifests[allowed[key]],
                        record=record,
                        source=artifact(record_path),
                    )
                value["initial_observations"] = [
                    o.model_dump(mode="json") for o in reused.values()
                ]
            plan = SharedPlan.model_validate(value)
        current = await SharedEvolution(
            update,
            author,
            evaluator,
            assessor,
            task_topology,
            infrastructure_retries=infrastructure_retries,
        ).run(plan)
        for name in seen:
            if current["source_graphs"][name] != plan.source_graphs[name].model_dump(
                mode="json"
            ):
                raise ValueError("joint learning changed an old task's frozen topology")
        previous_update = update
        previous_plan = plan
        seen.extend(stage)
        checkpoints.append(
            {
                "stage": stage_index,
                "new_tasks": list(stage),
                "tasks": list(seen),
                "processed_targets": len(seen) - len(initial_plan.sources),
                "update": str(update.relative_to(directory)),
                "frozen": current,
            }
        )
        atomic_json(directory / "checkpoints.json", checkpoints)
    return checkpoints


def work_report(directory: Path):
    """Aggregate originals once, including rejected candidates and retention work.

    This is execution work inside this run. Historical teacher-acquisition work
    is deliberately separate and cannot be silently treated as zero.
    """
    import json

    from reweave.evolution.authoring.author import (
        cumulative_usage,
        with_admission_usage,
    )
    from reweave.evolution.contracts import EvaluationRecord

    records = [
        EvaluationRecord.model_validate_json(p.read_text())
        for p in directory.rglob("shared_result.json")
    ]
    authors = [
        with_admission_usage(cumulative_usage(p), p)
        for p in directory.glob("**/audit")
        if (p.parent / "policy.json").is_file()
        and (p.parent / "request.json").is_file()
    ]
    unclosed = sum(
        not (p.parent / "shared_result.json").exists()
        for p in directory.rglob("shared_started.json")
    )
    steps_complete = all(r.usage.physical_steps_complete for r in records)
    return {
        "completed_rollouts": len(records),
        "development_rollouts": sum(r.request.split == "dev" for r in records),
        "test_rollouts": sum(r.request.split == "test" for r in records),
        "physical_steps": sum(r.usage.physical_steps for r in records),
        "physical_steps_complete": steps_complete and unclosed == 0,
        "development_steps": sum(
            r.usage.physical_steps for r in records if r.request.split == "dev"
        ),
        "development_work_s": sum(
            r.usage.charged_wall_time_s for r in records if r.request.split == "dev"
        ),
        "rollout_work_s": sum(r.usage.charged_wall_time_s for r in records),
        "offline_work_s": sum(a.charged_wall_time_s for a in authors)
        + sum(
            json.loads(p.read_text())["seconds"]
            for p in directory.rglob("alignment_work.json")
        )
        + sum(
            json.loads(p.read_text())["seconds"]
            for p in directory.glob("**/topology_work/*.json")
        ),
        "topology_work_s": sum(
            json.loads(p.read_text())["seconds"]
            for p in directory.glob("**/topology_work/*.json")
        ),
        "source_query_work_s": sum(
            json.loads(line)["elapsed_s"]
            for p in directory.glob("**/source_tools.jsonl")
            for line in p.read_text().splitlines()
        ),
        "source_query_note": "Source query time is contained in native author wall time; do not add it again.",
        "offline_work_complete": all(a.wall_time_complete for a in authors),
        "cost_usd": sum(a.cost_usd for a in [*authors, *(r.usage for r in records)])
        if all(a.cost_usd is not None for a in [*authors, *(r.usage for r in records)])
        else None,
        "unclosed_requests": unclosed,
        "teacher_acquisition_work": None,
        "teacher_acquisition_note": "Add original source development receipts before claiming end-to-end cost; not included in this run-local total.",
    }
