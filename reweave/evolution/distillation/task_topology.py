"""Refresh new-task graphs and retrieve mechanisms from immutable old-task graphs."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvolutionHistory,
    EvolutionProtocol,
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    SemanticGraph,
    VersionReference,
)
from reweave.evolution.distillation.conditions import NativeSourceConditions
from reweave.evolution.distillation.engine import (
    bind_json,
    condition_key,
    development_task,
    task_execution,
)
from reweave.evolution.distillation.graph_initialization import (
    NativeTaskGraphInitializer,
    artifact,
)
from reweave.evolution.distillation.student_gaps import (
    build_gap_report,
    teacher_unit_bank,
)
from reweave.evolution.evidence.trajectory import build_graph, episode_from_record
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.conditions import merge_catalogs
from reweave.evolution.graph.refinement import validate_refinement
from reweave.utils.persistence import atomic_json


def update_request(plan, task, history, previous, directory, policy):
    """One analysis sees every admitted version of this new task, and no tests."""
    directory.mkdir(parents=True, exist_ok=True)
    allowed = {condition_key(i) for i in task.dev}
    records, bundles, versions, episodes, originals = [], {}, [], [], {}
    execution, conditions = task_execution(plan, task)
    for item in history:
        bundle = item["bundle"]
        group = [
            r for r in item["records"] if condition_key(r.request.instance) in allowed
        ]
        if (
            len(group) != len(allowed)
            or {condition_key(r.request.instance) for r in group} != allowed
        ):
            raise ValueError(
                "new-task topology needs exactly its declared development cells"
            )
        version = VersionReference(
            version_id=f"H{item['round']}", bundle_id=bundle.record_id
        )
        versions.append(version)
        bundles[bundle.record_id] = bundle
        bundle_path = directory / "bundles" / f"{bundle.record_id}.json"
        bind_json(bundle_path, bundle.model_dump(mode="json"))
        evidence = artifact(bundle_path)
        originals[evidence.path] = evidence.model_dump(exclude={"path"})
        for record in group:
            request = record.request
            if (
                request.bundle_id != bundle.record_id
                or request.split != "dev"
                or request.execution != execution
                or request.conditions != conditions
                or request.manifest_id != plan.manifest_id
                or record.failure_origin not in {"none", "agent"}
                or record.episode is None
            ):
                raise ValueError(
                    "dynamic graph evidence differs from the shared student's dev contract"
                )
            record.verify()
            path = directory / "records" / f"{record.record_id}.json"
            bind_json(path, record.model_dump(mode="json"))
            records.append(record)
            episodes.append(episode_from_record(record, path, (version,)))
            for evidence in (*record.evidence, artifact(path)):
                originals[evidence.path] = evidence.model_dump(exclude={"path"})
    graph_path = directory / "execution_graph.json"
    graph_id = (
        json.loads(graph_path.read_text())["record_id"]
        if graph_path.exists()
        else new_record_id()
    )
    scaffold = build_graph(episodes, plan.new_task_graph_policy, record_id=graph_id)
    bind_json(graph_path, scaffold.model_dump(mode="json"))
    originals_path = directory / "history_originals.json"
    bind_json(originals_path, {"files": originals, "navigation": {}})
    summary = directory / "history.md"
    if not summary.exists():
        summary.write_text(
            "# Joint shared-student task history\n\nRefine this new task's topology using all admitted shared-Harness versions. Preserve past evidence; incorporate the latest three development conditions. Old-task reference graphs remain immutable. No independent task specialist or held-out result exists in this request.\n"
        )
    path = directory / "analysis_request.json"
    if path.exists():
        request = GraphAnalysisRequest.model_validate_json(path.read_text())
    else:
        protocol = EvolutionProtocol(
            record_id=new_record_id(),
            execution_id=execution,
            manifest_id=plan.manifest_id,
            graph=plan.new_task_graph_policy,
            dev=task.dev,
            max_rounds=0,
            infrastructure_retries=0,
            target_success_rate=1.0,
            execution=conditions,
            max_authors_per_round=1,
            width_mode="fixed",
            author_wall_time_s=None,
        )
        request = GraphAnalysisRequest(
            record_id=new_record_id(),
            round_index=history[-1]["round"],
            previous_analysis=artifact(Path(previous["receipt"]["path"])),
            protocol=protocol,
            graph=artifact(graph_path),
            history=EvolutionHistory(
                summary=artifact(summary), originals=artifact(originals_path)
            ),
            evidence=tuple(records),
            versions=tuple(versions),
            current_round_versions=(versions[-1].version_id,),
            policy=policy.model_dump(mode="json"),
        )
        bind_json(path, request.model_dump(mode="json"))
    if (
        request.evidence != tuple(records)
        or request.versions != tuple(versions)
        or request.protocol.dev != task.dev
        or request.previous_analysis != artifact(Path(previous["receipt"]["path"]))
    ):
        raise ValueError("saved dynamic topology request changed")
    return request, bundles


class NativeQueryConditions(NativeSourceConditions):
    instructions = (
        NativeSourceConditions.instructions
        + """\nThese source graphs are the currently admitted new-task development graphs.
context/fixed_catalog.json contains immutable coordinate definitions from old tasks
and prior rounds. Reuse those names, ordered roles and value domains verbatim.
Add a coordinate only for a physically distinct property; never rename an old
coordinate or equate different abstractions to obtain a match. Annotate only the
new-task graphs. Do not edit the old task topologies or train another Harness.
"""
    )

    def prepare(self, request, directory):
        workspace, readable = super().prepare(request, directory)
        atomic_json(workspace / "context/fixed_catalog.json", request["fixed_catalog"])
        return workspace, readable

    async def admit(self, request, workspace, audit):
        result = await super().admit(request, workspace, audit)
        from reweave.evolution.graph.conditions import FluentDefinition

        fixed = {
            key: FluentDefinition.model_validate(value)
            for key, value in request["fixed_catalog"].items()
        }
        graphs = {
            name: SemanticGraph.model_validate(value)
            for name, value in result["graphs"].items()
        }
        catalog = merge_catalogs(fixed, *(g.matching.catalog for g in graphs.values()))
        return {
            "graphs": {
                name: graph.model_copy(
                    update={
                        "matching": graph.matching.model_copy(
                            update={"catalog": catalog}
                        )
                    }
                ).model_dump(mode="json")
                for name, graph in graphs.items()
            }
        }


class NativeTaskTopology:
    """One graph refresh per new task per shared version; old graphs are read-only."""

    def __init__(
        self,
        *,
        policy,
        runtime_manifest,
        task_runtime_manifests,
        analyzer_factory=NativeGraphAnalysis,
        encoder_factory=NativeQueryConditions,
    ):
        self.policy, self.runtime_manifest = policy, runtime_manifest
        self.task_runtime_manifests = task_runtime_manifests
        self.analyzer_factory, self.encoder_factory = analyzer_factory, encoder_factory

    async def __call__(self, plan, history, directory):
        directory = Path(directory)
        path = directory / "result.json"
        current = history[-1]
        fixed = {
            name: g.model_dump(mode="json") for name, g in plan.source_graphs.items()
        }
        if set(fixed) != set(plan.initial_learned_tasks):
            raise ValueError(
                "reference topology bank must contain exactly the old tasks"
            )
        if path.exists():
            result = json.loads(path.read_text())
            if (
                result["student_bundle_id"] != current["bundle"].record_id
                or result["fixed_graphs"] != fixed
                or result["current_record_ids"]
                != [
                    r.record_id
                    for r in current["records"]
                    if r.request.instance.task
                    in {t.dev[0].task for t in plan.sources if t.name in plan.new_tasks}
                ]
            ):
                raise ValueError(
                    "cached topology refresh differs from its shared student or fixed bank"
                )
            return result
        previous = history[-2]["task_topology"] if len(history) > 1 else None
        reports = {}
        for task in plan.sources:
            if task.name not in plan.new_tasks:
                continue
            location = directory / task.name
            group = [
                r
                for r in current["records"]
                if r.request.instance.task == task.dev[0].task
            ]
            if previous is None:
                initializer = NativeTaskGraphInitializer(
                    policy=self.policy,
                    runtime_manifest=self.runtime_manifest,
                    task_runtime_manifests=self.task_runtime_manifests,
                    analyzer_factory=self.analyzer_factory,
                )
                reports[task.name] = await initializer(
                    plan, task, current["bundle"], group, location
                )
                continue
            request, bundles = update_request(
                plan,
                task,
                history,
                previous["analyses"][task.name],
                location,
                self.policy,
            )
            native = location / "analysis"
            bind_json(native / "policy.json", self.policy.model_dump(mode="json"))
            bind_json(native / "request.json", request.model_dump(mode="json"))
            analyzer = self.analyzer_factory(
                bundle_loader=bundles.get,
                policy=self.policy,
                runtime_manifest=self.task_runtime_manifests[task.name],
                evidence_workspace=location / "evidence",
                native_runtime_root=native / "runtime",
            )
            completed = location / "receipt.json"
            if completed.exists():
                receipt = GraphAnalysisReceipt.model_validate_json(
                    completed.read_text()
                )
            else:
                started = native / "started.json"
                call = (
                    analyzer.recover
                    if started.exists()
                    or (native / "workspace").exists()
                    or (native / "audit/inputs").exists()
                    else analyzer
                )
                receipt = await call(
                    request,
                    native,
                    before_launch=lambda: bind_json(
                        started, {"request_id": request.record_id}
                    ),
                )
                validate_refinement(request, receipt.analysis)
                bind_json(completed, receipt.model_dump(mode="json"))
            if receipt.request_id != request.record_id:
                raise ValueError("task topology receipt answers another request")
            validate_refinement(request, receipt.analysis)
            reports[task.name] = {
                "task": task.name,
                "graph": receipt.analysis.graph.model_dump(mode="json"),
                "receipt": artifact(completed).model_dump(mode="json"),
            }
        bank = teacher_unit_bank(plan.source_graphs)
        catalogs = [g.matching.catalog for g in plan.source_graphs.values()]
        if previous is not None:
            catalogs.extend(
                SemanticGraph.model_validate(g).matching.catalog
                for g in previous["graphs"].values()
            )
        old_catalog = merge_catalogs(*catalogs)
        tasks = [t for t in plan.sources if t.name in plan.new_tasks]
        teacher_views = [
            {
                "task": t.name,
                "bundle": current["bundle"].model_dump(mode="json"),
                "learned_tasks": [],
                "graph": reports[t.name]["graph"],
                "evidence": [],
            }
            for t in tasks
        ]
        raw_graphs = {name: value["graph"] for name, value in reports.items()}
        request = {
            "round": current["round"],
            "mode": "continual",
            "method": "direct",
            "tasks": [development_task(plan, t) for t in tasks],
            "new_tasks": list(plan.new_tasks),
            "teachers": teacher_views,
            "parent": current["bundle"].model_dump(mode="json"),
            "anchors": {},
            "criteria": {},
            "conditions": plan.conditions.model_dump(mode="json"),
            "alignment_seconds": plan.alignment_seconds,
            "feedback": [
                r.model_dump(mode="json")
                for item in history
                for r in item["records"]
                if r.request.instance.task in {t.dev[0].task for t in tasks}
            ],
            "history": [],
            "fixed_catalog": {
                key: value.model_dump(mode="json") for key, value in old_catalog.items()
            },
            "source_graphs": raw_graphs,
        }
        encoder = self.encoder_factory(
            bundle_loader=None,
            policy=self.policy,
            evidence_workspace=directory / "encoding-evidence",
            runtime_manifest=self.runtime_manifest,
        )
        encoded = await encoder(request, directory / "encoding")
        graphs = {
            name: SemanticGraph.model_validate(value)
            for name, value in encoded["graphs"].items()
        }
        if set(graphs) != set(plan.new_tasks) or any(
            graph.model_dump(exclude={"matching"})
            != SemanticGraph.model_validate(raw_graphs[name]).model_dump(
                exclude={"matching"}
            )
            for name, graph in graphs.items()
        ):
            raise ValueError(
                "condition encoding changed a new-task topology or omitted a task"
            )
        records = [
            r
            for r in current["records"]
            if r.request.instance.task in {t.dev[0].task for t in tasks}
        ]
        details = {}
        for record in records:
            graph = graphs[
                next(
                    t.name
                    for t in tasks
                    if t.dev[0].task == record.request.instance.task
                )
            ]
            spans = graph.episodes[record.record_id]
            node = (
                max(spans, key=lambda span: span.last_frame).node_id if spans else None
            )
            details[record.record_id] = {
                "task": next(
                    t.name
                    for t in tasks
                    if t.dev[0].task == record.request.instance.task
                ),
                "node_id": None if record.official_success else node,
                "basis": "latest frame partition in this round's evidence-refined topology",
                "citations": [e.path for e in record.evidence],
            }
        report = build_gap_report(
            plan,
            bank,
            current["bundle"].record_id,
            records,
            {"topology_records": details},
            query_graphs=graphs,
        )
        if fixed != {
            name: g.model_dump(mode="json") for name, g in plan.source_graphs.items()
        }:
            raise ValueError("new-task topology refresh changed the old task bank")
        result = {
            "student_bundle_id": current["bundle"].record_id,
            "fixed_graphs": fixed,
            "graphs": {name: g.model_dump(mode="json") for name, g in graphs.items()},
            "analyses": reports,
            "teacher_unit_bank": bank,
            "student_gaps": report,
            "current_record_ids": [r.record_id for r in records],
        }
        bind_json(path, result)
        return result
