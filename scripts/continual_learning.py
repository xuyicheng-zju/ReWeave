"""Learn successive task batches starting from the selected H0."""

import argparse
import asyncio
import fcntl
from pathlib import Path

from reweave.cli.distill import SharedConfiguration, execute
from reweave.cli.harness import Harness, TaskKnowledge, author_policy, encode_knowledge
from reweave.evolution.bundle import AgentBundle, new_record_id
from reweave.evolution.contracts import EvaluationRecord, SemanticGraph
from reweave.evolution.distillation.engine import (
    SharedPlan,
    SharedTask,
    SharedTeacher,
    bind_json,
)
from reweave.evolution.distillation.stream import stage_task
from reweave.evolution.infrastructure.campaign_slots import (
    campaign_root,
    enabled,
    online_slots,
)
from reweave.utils.gpu_slots import decode_slot


def resource_pool(output, devices):
    pool = campaign_root(output)
    if pool is None:
        bind_json(
            output / "scheduler/online/resource_policy.json",
            {
                "authorized_devices": list(devices),
                "slots_per_gpu": 1,
            },
        )
        bind_json(output / "scheduler_policy.json", {"mode": "episode_pool"})
        bind_json(output / "campaign_state.json", {})
        pool = output
    available = {
        decode_slot(slot)[0] for slot in online_slots(pool / "scheduler/online")
    }
    if not enabled(pool) or not available.intersection(devices):
        raise ValueError(
            f"the episode pool has no active slot for the selected H0 devices: {pool}"
        )


async def prepare(h0, policy, output, stages, rounds, reset_id):
    tasks = {name: item.task for name, item in h0.knowledge.items()}
    incoming = {}
    for index in (index for stage in stages for index in stage):
        name = f"task_{index}"
        if name in tasks or name in incoming:
            raise ValueError(f"choose each new task once: {name}")
        incoming[name] = SharedTask(
            name=name, family=h0.suite, dev=(h0.instance(index, 0, 0),)
        )
    panels = {}
    if reset_id is not None:
        panel = []
        for name, task in (tasks | incoming).items():
            used = (*task.dev, *stage_task(task, new=name in incoming).dev)
            if any(item.initialization["reset_state_id"] == reset_id for item in used):
                raise ValueError(
                    f"test reset {reset_id} overlaps development for {name}"
                )
            index = task.dev[0].initialization["task"]
            panel.append(
                task.model_copy(
                    update={"test": (h0.instance(index, reset_id, reset_id),)}
                )
            )
        panels["held_out"] = tuple(panel)
    graphs = {name: item.graph for name, item in h0.knowledge.items()}
    if any(graph.matching is None for graph in graphs.values()):
        _, graphs = await encode_knowledge(
            h0,
            {name: (h0.bundle, item) for name, item in h0.knowledge.items()},
            policy,
            output / "encoding",
        )
    manifest = h0.runner.runtime_manifest
    plan = SharedPlan(
        record_id=new_record_id(),
        mode="consolidation" if tasks else "baseline",
        sources=tuple(tasks.values()),
        initial=h0.bundle,
        source_graphs=graphs,
        teachers=tuple(
            SharedTeacher(
                task=name, bundle=h0.bundle, learned_tasks=tuple(tasks), graph=graph
            )
            for name, graph in graphs.items()
        ),
        execution_id=manifest.execution_id,
        manifest_id=new_record_id(),
        conditions=manifest.conditions,
        test_conditions=manifest.conditions,
        max_rounds=0,
        conditional_retention=False,
    )
    initial = output / "initial.json"
    bind_json(
        initial,
        {
            "bundle": h0.bundle.model_dump(mode="json"),
            "source_graphs": {
                name: graph.model_dump(mode="json") for name, graph in graphs.items()
            },
        },
    )
    return SharedConfiguration(
        plan=plan,
        runner=h0.runner,
        test_runner=h0.runner,
        author=policy,
        resource_workspace=output,
        physical_repository=Path(__file__).resolve().parents[1],
        stream_tasks=tuple((tasks | incoming).values()),
        stages=tuple(tuple(f"task_{index}" for index in stage) for stage in stages),
        stream_initial=initial,
        stream_rounds=rounds,
        evaluation_panels=panels,
    )


def publish_checkpoints(h0, config, checkpoints, directory):
    knowledge = {
        name: TaskKnowledge(
            task=item.task,
            graph=config.plan.source_graphs[name],
            evidence=item.evidence,
        )
        for name, item in h0.knowledge.items()
    }
    for entry in checkpoints:
        update = directory / entry["update"]
        plan = SharedPlan.model_validate_json((update / "plan.json").read_text())
        frozen = entry["frozen"]
        evidence = {
            record.record_id: record
            for path in update.rglob("shared_result.json")
            if (
                record := EvaluationRecord.model_validate_json(path.read_text())
            ).request.split
            == "dev"
        }
        for task in plan.sources:
            if task.name in entry["new_tasks"]:
                graph = SemanticGraph.model_validate(frozen["source_graphs"][task.name])
                knowledge[task.name] = TaskKnowledge(
                    task=task,
                    graph=graph,
                    evidence=tuple(evidence[identity] for identity in graph.episodes),
                )
        yield Harness(
            suite=h0.suite,
            runner=h0.runner,
            bundle=AgentBundle.model_validate(frozen["bundle"]),
            knowledge=knowledge,
        ).publish(update)


async def learn(args, h0, policy, output):
    config_path = output / "configuration.json"
    if config_path.is_file():
        config = SharedConfiguration.model_validate_json(config_path.read_text())
    else:
        config = await prepare(
            h0,
            policy,
            output,
            args.stage,
            args.rounds,
            args.test_reset_id,
        )
        bind_json(config_path, config.model_dump(mode="json"))
    if args.prepare_only:
        print(config_path)
        return
    stream = output / "stream"
    checkpoints = await execute(config, stream, "stream")
    for path in publish_checkpoints(h0, config, checkpoints, stream):
        print(path)
    if config.evaluation_panels:
        await execute(config, stream, "stream-test")
        print(stream / "stage_test_report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h0", type=Path, required=True, help="The starting harness.json."
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        action="append",
        required=True,
        help="Task indices in one new batch. Repeat in arrival order.",
    )
    parser.add_argument("--author-policy", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--test-reset-id", type=int, help="Evaluate each stage on this held-out reset."
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.rounds < 1
        or (args.test_reset_id is not None and args.test_reset_id < 0)
        or any(index < 0 for stage in args.stage for index in stage)
    ):
        parser.error("supply positive rounds and nonnegative task and reset indices")
    source = args.h0.resolve()
    h0 = Harness.read(source)
    policy = author_policy(args.author_policy)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_json(
            output / "inputs.json",
            {
                "h0": str(source),
                "bundle": h0.bundle.record_id,
                "stages": args.stage,
                "rounds": args.rounds,
                "test_reset_id": args.test_reset_id,
                "author": policy.model_dump(mode="json"),
            },
        )
        resource_pool(output, h0.runner.devices)
        asyncio.run(learn(args, h0, policy, output))


if __name__ == "__main__":
    main()
