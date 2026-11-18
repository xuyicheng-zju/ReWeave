"""The first executed version H0 supplies each new task's initial graph.

Uses the same physical scaffold, native graph analyst and semantic admission as
single-task evolution. Failure can initialize a graph; imaginary success cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvidenceFile,
    EvolutionHistory,
    EvolutionIncomplete,
    EvolutionProtocol,
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    GraphSnapshot,
    VersionReference,
)
from reweave.evolution.distillation.engine import (
    bind_json,
    condition_key,
    task_execution,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.evidence.trajectory import build_graph, episode_from_record
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.refinement import validate_refinement
from reweave.utils.persistence import atomic_text


def artifact(path):
    return EvidenceFile(**artifact_record(path))


def prepare_initial_graph_request(plan, task, h0, records, directory, policy):
    """Freeze one task's first-execution evidence, never substitute pure H0 or tests."""
    directory.mkdir(parents=True, exist_ok=True)
    if h0 != plan.initial:
        raise ValueError(
            "graph cold start requires the first-executed starting Harness"
        )
    if task.name not in plan.new_tasks or task not in plan.sources:
        raise ValueError("graph cold start requires an explicitly new source task")
    expected = {condition_key(i) for i in task.dev}
    execution_id, conditions = task_execution(plan, task)
    if (
        len(records) != len(task.dev)
        or {condition_key(r.request.instance) for r in records} != expected
    ):
        raise ValueError("one H0 execution per declared development condition")
    for record in records:
        if (
            record.request.bundle_id != h0.record_id
            or record.request.split != "dev"
            or record.request.conditions != conditions
            or record.request.execution != execution_id
            or record.request.manifest_id != plan.manifest_id
        ):
            raise ValueError(
                "cold-start receipt differs from the declared H0/task/dev execution"
            )
        if (
            record.failure_origin in {"unknown", "infrastructure"}
            or record.episode is None
        ):
            raise EvolutionIncomplete(
                "cold-start graph needs settled physical evidence; no automatic replay"
            )
        record.verify()
    identity = {
        "task": {
            "name": task.name,
            "family": task.family,
            "dev": [i.model_dump(mode="json") for i in task.dev],
        },
        "h0": h0.model_dump(mode="json"),
        "records": [r.model_dump(mode="json") for r in records],
        "graph_policy": plan.new_task_graph_policy.model_dump(mode="json"),
        "native_policy": policy.model_dump(mode="json"),
    }
    bind_json(directory / "inputs.json", identity)
    # This copy retains the physical receipt ID and is not another evaluation.
    bind_json(directory / "h0.json", h0.model_dump(mode="json"))
    receipts = []
    for index, record in enumerate(records):
        path = directory / f"evaluation-{index:03d}.json"
        bind_json(path, record.model_dump(mode="json"))
        receipts.append(path)
    versions = (VersionReference(version_id="H0", bundle_id=h0.record_id),)
    scaffold_path = directory / "execution_graph.json"
    if scaffold_path.exists():
        scaffold = GraphSnapshot.model_validate_json(scaffold_path.read_text())
    else:
        episodes = [
            episode_from_record(record, path, versions)
            for record, path in zip(records, receipts, strict=True)
        ]
        if any(not episode.frames for episode in episodes):
            raise EvolutionIncomplete(
                "H0 first execution has no recorded physical frames"
            )
        scaffold = build_graph(episodes, plan.new_task_graph_policy)
        bind_json(scaffold_path, scaffold.model_dump(mode="json"))
    if (
        {e.receipt_id for e in scaffold.episodes} != {r.record_id for r in records}
        or any(e.bundle_id != h0.record_id for e in scaffold.episodes)
        or scaffold.policy != plan.new_task_graph_policy
    ):
        raise ValueError("cold-start scaffold provenance changed")
    originals = {
        f.path: {"artifact_id": f.artifact_id, "bytes": f.bytes}
        for record in records
        for f in record.evidence
    }
    for path in (directory / "h0.json", *receipts):
        item = artifact(path)
        originals[item.path] = {"artifact_id": item.artifact_id, "bytes": item.bytes}
    bind_json(
        directory / "history_originals.json", {"files": originals, "navigation": {}}
    )
    summary = directory / "history.md"
    if not summary.exists():
        atomic_text(
            summary,
            "# New-task graph initialization\n\n"
            "First-round development evidence from the supplied starting Harness. Use its task instruction and "
            "physical evidence to establish the initial backbone, even after failure. "
            "Unobserved milestones remain explicit hypotheses; never invent a successful "
            "transition or test outcome. No student update or graph retrieval has run.\n",
        )
    request_path = directory / "analysis_request.json"
    if request_path.exists():
        request = GraphAnalysisRequest.model_validate_json(request_path.read_text())
    else:
        protocol = EvolutionProtocol(
            record_id=new_record_id(),
            execution_id=execution_id,
            manifest_id=plan.manifest_id,
            graph=plan.new_task_graph_policy,
            dev=task.dev,
            max_rounds=0,
            execution=conditions,
            infrastructure_retries=0,
            max_authors_per_round=1,
            width_mode="fixed",
            target_success_rate=1.0,
        )
        request = GraphAnalysisRequest(
            record_id=new_record_id(),
            round_index=0,
            previous_analysis=None,
            protocol=protocol,
            graph=artifact(scaffold_path),
            history=EvolutionHistory(
                summary=artifact(summary),
                originals=artifact(directory / "history_originals.json"),
            ),
            evidence=tuple(records),
            versions=versions,
            policy=policy.model_dump(mode="json"),
        )
        bind_json(request_path, request.model_dump(mode="json"))
    if (
        request.evidence != tuple(records)
        or request.versions != versions
        or request.protocol.dev != task.dev
        or request.graph != artifact(scaffold_path)
        or request.policy != policy.model_dump(mode="json")
    ):
        raise ValueError("saved graph initialization request changed")
    return request


class NativeTaskGraphInitializer:
    def __init__(
        self,
        *,
        policy,
        runtime_manifest,
        analyzer_factory=NativeGraphAnalysis,
        task_runtime_manifests=None,
    ):
        self.policy = policy
        self.runtime_manifest = runtime_manifest
        self.task_runtime_manifests = task_runtime_manifests or {}
        self.analyzer_factory = analyzer_factory

    async def __call__(self, plan, task, h0, records, directory: Path):
        request = prepare_initial_graph_request(
            plan, task, h0, records, directory, self.policy
        )
        native = directory / "analysis"
        # Include the ordinary policy/request records for shared work accounting.
        bind_json(native / "policy.json", self.policy.model_dump(mode="json"))
        bind_json(native / "request.json", request.model_dump(mode="json"))
        analyzer = self.analyzer_factory(
            bundle_loader=lambda key: h0 if key == h0.record_id else None,
            policy=self.policy,
            runtime_manifest=self.task_runtime_manifests.get(
                task.name, self.runtime_manifest
            ),
            evidence_workspace=directory / "evidence",
            native_runtime_root=native / "runtime",
        )
        started = native / "started.json"
        completed = directory / "receipt.json"
        if completed.exists():
            receipt = GraphAnalysisReceipt.model_validate_json(completed.read_text())
        else:
            resume = (
                started.exists()
                or (native / "workspace").exists()
                or (native / "audit/inputs").exists()
            )
            call = analyzer.recover if resume else analyzer
            receipt = await call(
                request,
                native,
                before_launch=lambda: bind_json(
                    started, {"request_id": request.record_id}
                ),
            )
            if receipt.request_id != request.record_id:
                raise ValueError("initial graph belongs to another analysis request")
            validate_refinement(request, receipt.analysis)
            bind_json(completed, receipt.model_dump(mode="json"))
        if receipt.request_id != request.record_id:
            raise ValueError("cached initial graph belongs to another request")
        validate_refinement(request, receipt.analysis)
        graph = receipt.analysis.graph
        bind_json(directory / "semantic_graph.json", graph.model_dump(mode="json"))
        # Source-only original paths needed by the subsequent shared author.
        originals = json.loads(Path(request.history.originals.path).read_text())[
            "files"
        ]
        result = {
            "task": task.name,
            "h0_bundle_id": h0.record_id,
            "graph": graph.model_dump(mode="json"),
            "scaffold": request.graph.model_dump(mode="json"),
            "receipt": artifact(completed).model_dump(mode="json"),
            "originals": [{"path": path, **item} for path, item in originals.items()],
            "official_success": all(r.official_success for r in records),
            "matching_ready": graph.matching is not None,
        }
        bind_json(directory / "initialization.json", result)
        return result
