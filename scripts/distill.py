"""Merge the selected source Harnesses into one shared Harness."""

import argparse
import asyncio
import fcntl
import json
from pathlib import Path

from reweave.cli.harness import Harness, TaskKnowledge, author_policy, encode_knowledge
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.engine import (
    SharedTeacher,
    bind_json,
    development_task,
)
from reweave.evolution.distillation.topology import SharedTopology, build_topology
from reweave.evolution.distillation.transfer import aligned_references


async def merge(harnesses, policy, output):
    parent = harnesses[0]
    sources = {}
    runtime = parent.bundle.runtime.model_dump(exclude={"revision_id"})
    for harness in harnesses:
        if (
            harness.suite != parent.suite
            or harness.bundle.runtime.model_dump(exclude={"revision_id"}) != runtime
        ):
            raise ValueError("selected Harnesses must share one suite and runtime")
        if not harness.knowledge:
            raise ValueError(
                "each source Harness needs a task topology produced by optimization or continual learning"
            )
        for name, knowledge in harness.knowledge.items():
            if name in sources:
                raise ValueError(f"choose one source Harness for {name}")
            sources[name] = (harness.bundle, knowledge)
    plan, graphs = await encode_knowledge(parent, sources, policy, output / "encoding")
    teachers = tuple(
        SharedTeacher(
            task=name,
            bundle=bundle,
            learned_tasks=(name,),
            graph=graphs[name],
            evidence=tuple(
                record
                for record in item.evidence
                if record.request.bundle_id == bundle.record_id
            ),
        )
        for name, (bundle, item) in sources.items()
    )
    contract_path = output / "alignment.json"
    if contract_path.is_file():
        alignments = json.loads(contract_path.read_text())
    else:
        _, alignments = aligned_references(teachers, plan.alignment_seconds)
        bind_json(contract_path, alignments)
    topology_path = output / "topology.json"
    if topology_path.is_file():
        topology = SharedTopology.model_validate_json(topology_path.read_text())
    else:
        topology = build_topology(graphs, alignments, seconds=plan.alignment_seconds)
        bind_json(topology_path, topology.model_dump(mode="json"))
    request = {
        "round": 1,
        "mode": "consolidation",
        "method": "aligned",
        "tasks": [development_task(plan, task) for task in plan.sources],
        "new_tasks": [],
        "teachers": [teacher.model_dump(mode="json") for teacher in teachers],
        "source_graphs": {
            name: graph.model_dump(mode="json") for name, graph in graphs.items()
        },
        "shared_topology": topology.model_dump(mode="json"),
        "fixed_alignments": alignments,
        "parent": parent.bundle.model_dump(mode="json"),
        "anchors": {},
        "criteria": {},
        "conditions": plan.conditions.model_dump(mode="json"),
        "alignment_seconds": plan.alignment_seconds,
        "feedback": [
            record.model_dump(mode="json")
            for _, item in sources.values()
            for record in item.evidence
        ],
        "history": [],
    }
    author = NativeSharedAuthor(
        bundle_loader=None,
        policy=policy,
        evidence_workspace=output,
        runtime_manifest=parent.runner.runtime_manifest,
    )
    result = await author(request, output / "merge")
    knowledge = {
        name: TaskKnowledge(task=item.task, graph=graphs[name], evidence=item.evidence)
        for name, (_, item) in sources.items()
    }
    return Harness(
        suite=parent.suite,
        runner=parent.runner,
        bundle=AgentBundle.model_validate(result["bundle"]),
        knowledge=knowledge,
    ).publish(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        type=Path,
        action="append",
        required=True,
        help="A source Harness. Repeat for at least two inputs. The first is the starting implementation.",
    )
    parser.add_argument("--author-policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.harness) < 2:
        parser.error("supply at least two --harness inputs")
    inputs = [path.resolve() for path in args.harness]
    harnesses = [Harness.read(path) for path in inputs]
    policy = author_policy(args.author_policy)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_json(
            output / "inputs.json",
            {
                "harnesses": [
                    (str(path), harness.bundle.record_id)
                    for path, harness in zip(inputs, harnesses, strict=True)
                ],
                "author": policy.model_dump(mode="json"),
            },
        )
        print(asyncio.run(merge(harnesses, policy, output)))


if __name__ == "__main__":
    main()
