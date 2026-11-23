"""Evolve the selected Harness once from explicitly selected trajectories."""

import argparse
import asyncio
import fcntl
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from reweave.cli.harness import Harness, TaskKnowledge, author_policy
from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvolutionHistory,
    EvolutionProtocol,
    GraphAnalysisRequest,
    GraphPolicy,
    GraphSnapshot,
    Proposal,
    RoundDecision,
    SearchCost,
    VersionReference,
)
from reweave.evolution.distillation.engine import SharedTask, bind_json, condition_key
from reweave.evolution.distillation.graph_initialization import artifact
from reweave.evolution.evidence.trajectory import build_graph, episode_from_record
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.projection import analysis_graph_receipts, project_map
from reweave.utils.persistence import atomic_text


def prepare(harness, trajectories, policy, output):
    records = tuple(
        EvaluationRecord.model_validate_json(path.read_text()) for path in trajectories
    )
    task = records[0].request.instance.task
    if (
        len({record.record_id for record in records}) != len(records)
        or any(
            record.request.instance.task != task
            or record.request.split != "dev"
            or record.failure_origin not in {"none", "agent"}
            for record in records
        )
        or not task.startswith(f"libero:{harness.suite}:")
    ):
        raise ValueError(
            "select distinct settled development trajectories from one task in the Harness suite"
        )
    source_bundles = {harness.bundle.record_id: harness.bundle}
    for path, record in zip(trajectories, records, strict=True):
        record.verify()
        bundle = AgentBundle.model_validate_json(
            path.with_name("bundle.json").read_text()
        )
        if bundle.record_id != record.request.bundle_id:
            raise ValueError(f"trajectory and source Harness differ: {path}")
        if source_bundles.setdefault(bundle.record_id, bundle) != bundle:
            raise ValueError(
                f"source Harness changed under an existing identity: {path}"
            )

    request_path = output / "analysis_request.json"
    if request_path.exists():
        return GraphAnalysisRequest.model_validate_json(
            request_path.read_text()
        ), source_bundles
    versions = tuple(
        VersionReference(version_id=f"H{index}", bundle_id=bundle_id)
        for index, bundle_id in enumerate(source_bundles)
    )
    manifest = harness.runner.runtime_manifest
    operation = uuid5(NAMESPACE_URL, output.as_uri()).hex
    pool = f"{operation}-conditions"
    instances, cells, derived, episodes, originals = {}, {}, [], [], []
    for index, (record, original) in enumerate(zip(records, trajectories, strict=True)):
        instance = instances.setdefault(
            condition_key(record.request.instance), record.request.instance
        )
        bound = record.model_copy(
            update={
                "request": record.request.model_copy(
                    update={"instance": instance, "manifest_id": pool}
                )
            }
        )
        path = output / "evidence" / f"trajectory-{index:03d}.json"
        bind_json(path, bound.model_dump(mode="json"))
        episode = episode_from_record(bound, path, versions)
        if not episode.frames:
            raise ValueError(
                f"trajectory has no recorded physical observations: {original}"
            )
        derived.append(bound)
        episodes.append(episode)
        cells[(bound.request.bundle_id, instance.record_id)] = bound.record_id
        originals.extend((artifact(original), artifact(path), *record.evidence))

    bundle_entries = []
    for index, bundle in enumerate(source_bundles.values()):
        path = output / "evidence" / f"harness-{index:03d}.json"
        bind_json(path, bundle.model_dump(mode="json"))
        entry = artifact(path)
        bundle_entries.append({"bundle": entry.model_dump(mode="json")})
        originals.append(entry)
    history = output / "history.md"
    atomic_text(
        history,
        "# Selected trajectories\n\nThe operator chose the starting Harness and these exact development trajectories for one offline mutation. The candidate has no measured outcome.\n",
    )
    index_path = output / "history_originals.json"
    bind_json(
        index_path,
        {
            "files": {
                item.path: item.model_dump(exclude={"path"}) for item in originals
            },
            "navigation": {
                "bundles": bundle_entries,
                "episodes": [row.model_dump(mode="json") for row in derived],
            },
        },
    )
    protocol = EvolutionProtocol(
        record_id=f"{operation}-protocol",
        execution_id=manifest.execution_id,
        manifest_id=pool,
        graph=GraphPolicy(),
        dev=tuple(instances.values()),
        max_rounds=1,
        execution=manifest.conditions,
        infrastructure_retries=0,
        max_authors_per_round=1,
        width_mode="fixed",
        target_success_rate=1.0,
    )
    graph_path = output / "execution_graph.json"
    bind_json(
        graph_path,
        build_graph(episodes, protocol.graph, record_id=pool).model_dump(mode="json"),
    )
    request = GraphAnalysisRequest(
        record_id=f"{operation}-analysis",
        round_index=0,
        previous_analysis=None,
        protocol=protocol,
        graph=artifact(graph_path),
        history=EvolutionHistory(
            summary=artifact(history), originals=artifact(index_path)
        ),
        evidence=tuple(derived),
        versions=versions,
        retrospective_receipts=tuple(cells.values()),
        policy=policy.model_dump(mode="json"),
    )
    bind_json(request_path, request.model_dump(mode="json"))
    return request, source_bundles


async def evolve(harness, trajectories, policy, output):
    request, bundles = prepare(harness, trajectories, policy, output)
    shared = {
        "bundle_loader": bundles.__getitem__,
        "policy": policy,
        "runtime_manifest": harness.runner.runtime_manifest,
        "evidence_workspace": output / "evidence",
    }
    analysis = output / "analysis"
    receipt = await NativeGraphAnalysis(
        **shared, native_runtime_root=output / "graph_session/runtime"
    )(request, analysis, resume=analysis.exists())
    graph = receipt.analysis.graph
    proposal_path = output / "proposal.json"
    if proposal_path.is_file():
        proposal = Proposal.model_validate_json(proposal_path.read_text())
    else:
        raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
        progress = project_map(
            raw,
            graph,
            request.versions,
            request.protocol.dev,
            graph_id=receipt.record_id + "/analysis/graph",
            cell_receipts=analysis_graph_receipts(request, raw),
            validation_conditions=request.protocol.validation.conditions
            if request.protocol.validation
            else (),
        )
        semantic_path = output / "semantic_graph.json"
        map_path = output / "progress_map.json"
        bind_json(semantic_path, graph.model_dump(mode="json"))
        bind_json(map_path, progress)
        parent = request.versions[0]
        decision = RoundDecision(
            record_id=f"{request.record_id}/decision",
            request_id=request.record_id,
            analysis=artifact(analysis / "result.json"),
            graph=artifact(semantic_path),
            progress_map=artifact(map_path),
            parent=parent,
            references=(parent,),
            width=1,
            candidate_cost=SearchCost(rollout_usd=None),
            alternatives=(),
            statistics={},
            reason="The operator selected the parent and trajectories.",
        )
        decision_path = output / "decision.json"
        bind_json(decision_path, decision.model_dump(mode="json"))
        proposal = Proposal(
            record_id=f"{request.record_id}/mutation",
            round_index=0,
            author_index=0,
            parent=parent,
            references=(parent,),
            scopes=request.protocol.mutation_scopes,
            task_protocol=request.protocol,
            history=request.history,
            evidence=request.evidence,
            graph=decision.graph,
            execution_graph=request.graph,
            progress_map=decision.progress_map,
            decision=artifact(decision_path),
            unmeasured_parent=artifact(output / "evidence/harness-000.json")
            if not any(
                row.request.bundle_id == parent.bundle_id for row in request.evidence
            )
            else None,
        )
        bind_json(proposal_path, proposal.model_dump(mode="json"))

    author = output / "author"
    result = await NativeBundleAuthor(**shared)(
        proposal, author, resume=author.exists()
    )
    index = request.protocol.dev[0].initialization["task"]
    name = f"task_{index}"
    knowledge = TaskKnowledge(
        task=SharedTask(name=name, family=harness.suite, dev=request.protocol.dev),
        graph=graph,
        evidence=request.evidence,
    )
    return Harness(
        suite=harness.suite,
        runner=harness.runner,
        bundle=result.candidates[0].bundle,
        knowledge={**harness.knowledge, name: knowledge},
    ).publish(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path, action="append")
    parser.add_argument("--author-policy", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.harness.resolve()
    harness = Harness.read(source)
    trajectories = tuple(path.resolve() for path in args.trajectory)
    policy = author_policy(args.author_policy)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_json(
            output / "inputs.json",
            {
                "harness": harness.bundle.record_id,
                "source": str(source),
                "trajectories": [str(path) for path in trajectories],
                "author": policy.model_dump(mode="json"),
            },
        )
        print(asyncio.run(evolve(harness, trajectories, policy, output)))


if __name__ == "__main__":
    main()
