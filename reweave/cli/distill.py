"""Validate, run, freeze and evaluate shared C/W/L; never start model services."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.distillation.assessment import NativeConditionalAssessment
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.conditions import (
    NativeSourceConditions,
    encoding_request,
)
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedPlan,
    SharedTask,
    task_execution,
)
from reweave.evolution.distillation.stream import run_stream, work_report
from reweave.evolution.distillation.task_topology import NativeTaskTopology
from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator
from reweave.utils.config import get_repo_root
from reweave.utils.persistence import atomic_json


class SharedConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: SharedPlan
    runner: RolloutConfiguration
    test_runner: RolloutConfiguration
    author: AuthorPolicy
    resource_workspace: Path
    physical_repository: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stream_tasks: tuple[SharedTask, ...] = ()
    stages: tuple[tuple[str, ...], ...] = ()
    stream_initial: Path | None = Field(
        default=None,
        description="Existing frozen shared checkpoint before the first joint continual stage; never retrained by the stream.",
    )
    stream_rounds: int | None = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
        description="Search-round limit per continual stage, independent of the starting checkpoint.",
    )
    infrastructure_retries: int = Field(
        default=0,
        ge=0,
        description="Maximum closed infrastructure replacements per shared development condition; valid policy results are never repeated.",
    )
    evaluation_panels: dict[str, tuple[SharedTask, ...]] = Field(default_factory=dict)
    task_runners: dict[str, RolloutConfiguration] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    task_test_runners: dict[str, RolloutConfiguration] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    def tasks(self):
        return (
            *self.plan.sources,
            *self.plan.unseen,
            *self.stream_tasks,
            *(task for panel in self.evaluation_panels.values() for task in panel),
        )

    @model_validator(mode="after")
    def execution_matches(self):
        if (
            self.physical_repository is not None
            and not self.physical_repository.is_absolute()
        ):
            raise ValueError("physical_repository must be an absolute source directory")
        if self.runner.runtime_manifest.conditions != self.plan.conditions:
            raise ValueError("development runner differs from shared plan conditions")
        if self.test_runner.runtime_manifest.conditions != self.plan.test_conditions:
            raise ValueError("test runner differs from frozen test conditions")
        bound = {}
        environments = {}
        named_environments = {}
        for task in self.tasks():
            if task.name in bound and bound[task.name] != task.execution:
                raise ValueError("task execution differs across source/test panels")
            bound[task.name] = task.execution
            environment = task.dev[0].task
            if (
                task.name in named_environments
                and named_environments[task.name] != environment
            ):
                raise ValueError("named task changes environment across panels")
            named_environments[task.name] = environment
            if environment in environments and environments[environment] != task.name:
                raise ValueError("task runner aliases must name one environment once")
            environments[environment] = task.name
        required = {name for name, binding in bound.items() if binding is not None}
        if (
            set(self.task_runners) != required
            or set(self.task_test_runners) != required
        ):
            raise ValueError("task execution bindings require exact dev/test runners")
        for split, runners in (
            ("dev", self.task_runners),
            ("test", self.task_test_runners),
        ):
            for name, runner in runners.items():
                binding = bound[name]
                conditions = (
                    binding.test_conditions if split == "test" else binding.conditions
                )
                if (
                    runner.runtime_manifest.conditions != conditions
                    or runner.runtime_manifest.execution_id != binding.execution_id
                ):
                    raise ValueError("task runner differs from its execution binding")
        if self.author.author != "codex" or self.author.graph_retrieval is not None:
            raise ValueError("use a source-isolated native Codex author")
        if not self.resource_workspace.is_absolute():
            raise ValueError(
                "resource_workspace must identify an absolute existing pool"
            )
        if bool(self.stream_tasks) != bool(self.stages):
            raise ValueError("stream tasks and stages must be specified together")
        if bool(self.stages) != (self.stream_initial is not None):
            raise ValueError(
                "a continual stream requires its existing initial shared checkpoint"
            )
        from reweave.evolution.distillation.engine import condition_key

        keys = []
        development = {condition_key(i) for task in self.plan.sources for i in task.dev}
        for name, tasks in self.evaluation_panels.items():
            if not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError("invalid evaluation panel name")
            if len({task.name for task in tasks}) != len(tasks) or any(
                not task.test for task in tasks
            ):
                raise ValueError(
                    "fixed evaluation panels need distinct tasks and test conditions"
                )
            keys.extend(condition_key(i) for task in tasks for i in task.test)
        if len(keys) != len(set(keys)) or development & set(keys):
            raise ValueError(
                "fixed testing overlaps another panel or a development condition"
            )
        return self


class TaskRolloutEvaluator:
    """Route a frozen physical recipe without modifying requests or C/W/L."""

    def __init__(self, config, *, repository, test=False):
        self.default = RolloutEvaluator(
            config.test_runner if test else config.runner,
            repository=repository,
            workspace=config.resource_workspace,
        )
        runners = config.task_test_runners if test else config.task_runners
        tasks = {task.name: task for task in config.tasks()}
        self.evaluators = {
            tasks[name].dev[0].task: RolloutEvaluator(
                runner, repository=repository, workspace=config.resource_workspace
            )
            for name, runner in runners.items()
        }
        self.bindings = {
            task.dev[0].task: task_execution(
                config.plan, task, "test" if test else "dev"
            )
            for task in tasks.values()
        }

    def runner_for(self, request):
        expected = self.bindings.get(request.instance.task)
        if expected != (request.execution, request.conditions):
            raise ValueError("shared request differs from the task execution binding")
        return self.evaluators.get(request.instance.task, self.default)

    async def __call__(self, request, bundle, directory, *, before_launch):
        return await self.runner_for(request)(
            request, bundle, directory, before_launch=before_launch
        )

    async def recover(self, request, bundle, directory, *, before_launch):
        return await self.runner_for(request).recover(
            request, bundle, directory, before_launch=before_launch
        )

    async def verify_retry_closed(self, request, bundle, directory):
        return await self.runner_for(request).verify_retry_closed(
            request, bundle, directory
        )


async def execute(config, output, mode):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await _execute(config, output, mode)


async def _execute(config, output, mode):
    repository = config.physical_repository or get_repo_root()
    if mode == "report":
        report = work_report(output)
        atomic_json(output / "work_report.json", report)
        return report
    if mode == "validate":
        return {
            "sources": len(config.plan.sources),
            "unseen": len(config.plan.unseen),
            "mode": config.plan.mode,
            "method": config.plan.method,
            "physical_work_started": False,
            "conditional_retention": config.plan.conditional_retention,
            "conditional_references_ready": bool(config.plan.mechanism_references),
            "live_runtime_verified": False,
        }
    if mode in {"prepare-fragments", "check-fragments"}:
        from reweave.evolution.distillation.engine import bind_json
        from reweave.evolution.distillation.preparation import (
            check_fragments,
            shared_encoding,
            source_graphs,
        )

        if mode == "check-fragments":
            report, _ = check_fragments(config.plan)
            atomic_json(output / "fragment_readiness.json", report)
            return report
        # Bind before model work: a different input cannot reuse native receipts.
        bind_json(output / "preparation_input.json", config.model_dump(mode="json"))
        published = output / "fragments.json"
        if published.exists():
            prepared = SharedConfiguration.model_validate_json(published.read_text())
            report, proposals = check_fragments(prepared.plan)
            bind_json(output / "fragment_transfer.json", proposals)
            report["configuration"] = str(published.resolve())
            report["reused_preparation"] = True
            atomic_json(output / "fragment_readiness.json", report)
            return report
        graphs = source_graphs(config.plan)
        value = config.model_dump(mode="json")
        value["plan"].update(
            method="fragments",
            topology_consolidation=True,
            source_graphs={n: g.model_dump(mode="json") for n, g in graphs.items()},
            alignment_records=[],
            mechanism_references=[],
        )
        for teacher in value["plan"]["teachers"]:
            teacher["graph"] = graphs[teacher["task"]].model_dump(mode="json")
        prepared = SharedConfiguration.model_validate(value)
        # Validate stopping policy before spending a model request.
        if prepared.plan.loss_threshold is None or prepared.plan.stop_on_source_success:
            raise ValueError("fragment experiment requires an explicit loss threshold")
        encoded_here = not shared_encoding(graphs)
        if encoded_here:
            await _execute(prepared, output / "encoding", "encode")
            prepared = SharedConfiguration.model_validate_json(
                (output / "encoding/fragments.json").read_text()
            )
        elif prepared.plan.conditional_retention:
            from reweave.evolution.distillation.transfer import (
                aligned_references,
                complete_source_references,
            )

            refs, _ = aligned_references(
                prepared.plan.teachers,
                prepared.plan.alignment_seconds,
                literal_only=True,
            )
            refs = complete_source_references(prepared.plan.teachers, refs)
            value = prepared.model_dump(mode="json")
            value["plan"]["mechanism_references"] = [
                r.model_dump(mode="json") for r in refs
            ]
            prepared = SharedConfiguration.model_validate(value)
        report, proposals = check_fragments(prepared.plan)
        bind_json(output / "fragments.json", prepared.model_dump(mode="json"))
        bind_json(output / "fragment_transfer.json", proposals)
        report["encoding_required"] = encoded_here
        report["configuration"] = str((output / "fragments.json").resolve())
        atomic_json(output / "fragment_readiness.json", report)
        return report
    if mode == "encode":
        if config.plan.mode == "baseline":
            return {
                "not_required": True,
                "source_tasks": 0,
                "native_work_started": False,
            }
        author = NativeSourceConditions(
            bundle_loader=None,
            policy=config.author,
            evidence_workspace=output,
            runtime_manifest=config.runner.runtime_manifest,
        )
        result = await author(
            encoding_request(config.plan), output / "source_conditions"
        )
        from reweave.evolution.distillation.engine import SharedTeacher
        from reweave.evolution.distillation.transfer import aligned_references

        annotated = tuple(
            SharedTeacher.model_validate(
                {
                    **t.model_dump(mode="json"),
                    "graph": result["graphs"][t.task],
                }
            )
            for t in config.plan.teachers
        )
        from reweave.evolution.distillation.engine import bind_json
        from reweave.evolution.distillation.transfer import MechanismReference

        alignment_path = output / "alignment_contract.json"
        if alignment_path.exists():
            fixed = json.loads(alignment_path.read_text())
            references = tuple(
                MechanismReference.model_validate(r) for r in fixed["references"]
            )
            alignments = fixed["alignments"]
        else:
            started = time.monotonic()
            references, alignments = aligned_references(
                annotated,
                config.plan.alignment_seconds,
                literal_only=config.plan.method == "fragments",
            )
            bind_json(
                alignment_path,
                {
                    "references": [r.model_dump(mode="json") for r in references],
                    "alignments": alignments,
                },
            )
            atomic_json(
                output / "alignment_work.json", {"seconds": time.monotonic() - started}
            )
        from reweave.evolution.distillation.transfer import complete_source_references

        references = complete_source_references(annotated, references)
        if config.plan.conditional_retention and not references:
            raise ValueError(
                "no positive witnessed mechanism alignment; formal run not ready"
            )
        methods = (
            ("fragments",)
            if config.plan.method == "fragments"
            else ("direct", "aligned")
        )
        for method in methods:
            value = config.model_dump(mode="json")
            value["plan"]["method"] = method
            if config.plan.conditional_retention:
                value["plan"]["mechanism_references"] = [
                    r.model_dump(mode="json") for r in references
                ]
                value["plan"]["alignment_records"] = alignments
            for teacher in value["plan"]["teachers"]:
                teacher["graph"] = result["graphs"][teacher["task"]]
            if value["plan"]["source_graphs"]:
                value["plan"]["source_graphs"] = result["graphs"]
            encoded = SharedConfiguration.model_validate(value)
            bind_json(output / f"{method}.json", encoded.model_dump(mode="json"))
        return {
            "source_tasks": len(result["graphs"]),
            "usage": result["usage"],
            "physical_work_started": False,
        }
    if mode == "run" and config.plan.method == "fragments":
        from reweave.evolution.distillation.preparation import check_fragments

        check_fragments(config.plan)
    if not config.resource_workspace.is_dir():
        raise ValueError(
            "existing resource workspace is required; this command does not create services"
        )
    from reweave.evolution.infrastructure.campaign_slots import campaign_root, enabled

    pool = campaign_root(config.resource_workspace)
    if pool is None or not enabled(pool):
        raise ValueError(
            "shared runs require the existing campaign admission pool, never private overlapping GPU locks"
        )
    evaluator = TaskRolloutEvaluator(
        config,
        repository=repository,
        test=mode in {"test", "stream-test", "baselines"},
    )
    author = NativeSharedAuthor(
        bundle_loader=None,
        policy=config.author,
        evidence_workspace=output,
        runtime_manifest=config.runner.runtime_manifest,
    )
    assessor = NativeConditionalAssessment(
        bundle_loader=None,
        policy=config.author,
        evidence_workspace=output,
        runtime_manifest=config.runner.runtime_manifest,
    )
    task_runtime_manifests = {
        task.name: config.task_runners.get(task.name, config.runner).runtime_manifest
        for task in config.tasks()
    }
    task_topology = NativeTaskTopology(
        policy=config.author,
        runtime_manifest=config.runner.runtime_manifest,
        task_runtime_manifests=task_runtime_manifests,
    )
    from reweave.evolution.distillation.progress import NativeTopologyProgress

    engine = SharedEvolution(
        output,
        author,
        evaluator,
        assessor,
        task_topology,
        infrastructure_retries=config.infrastructure_retries,
        topology_assessor=NativeTopologyProgress(
            bundle_loader=None,
            policy=config.author,
            evidence_workspace=output,
            runtime_manifest=config.runner.runtime_manifest,
        )
        if config.plan.fixed_topology_scoring
        else None,
    )
    if mode == "baselines":

        def aggregate(rows):
            return {
                t.name: {
                    "successes": sum(
                        r.official_success
                        for r in rows
                        if r.request.instance.task == t.dev[0].task
                    ),
                    "episodes": sum(
                        r.request.instance.task == t.dev[0].task for r in rows
                    ),
                    "settled": all(
                        r.failure_origin not in {"unknown", "infrastructure"}
                        for r in rows
                        if r.request.instance.task == t.dev[0].task
                    ),
                }
                for t in (*config.plan.sources, *config.plan.unseen)
                if any(r.request.instance.task == t.dev[0].task for r in rows)
            }

        h0 = await engine.evaluate(
            config.plan,
            config.plan.initial,
            (*config.plan.sources, *config.plan.unseen),
            output / "h0",
            split="test",
        )
        specialists = []
        for teacher in config.plan.teachers:
            task = next(t for t in config.plan.sources if t.name == teacher.task)
            specialists.extend(
                await engine.evaluate(
                    config.plan,
                    teacher.bundle,
                    (task,),
                    output / "specialists" / task.name,
                    split="test",
                )
            )
        result = {"h0": aggregate(h0), "specialists": aggregate(specialists)}
        atomic_json(output / "baseline_report.json", result)
        return result
    if mode == "run":
        return await engine.run(config.plan)
    if mode == "test":
        if config.evaluation_panels:
            return await engine.test_panels(config.plan, config.evaluation_panels)
        return await engine.test(config.plan)
    if mode == "stream":
        if not config.stages:
            raise ValueError("stream configuration is missing")
        return await run_stream(
            output,
            config.plan,
            config.stream_tasks,
            [list(s) for s in config.stages],
            author,
            evaluator,
            assessor,
            task_topology,
            initial_checkpoint=config.stream_initial,
            max_rounds=config.stream_rounds,
            task_runtime_manifests=task_runtime_manifests,
            infrastructure_retries=config.infrastructure_retries,
        )
    if mode == "stream-test":
        if not config.evaluation_panels:
            raise ValueError("continual tests require fixed evaluation panels")
        checkpoints = json.loads((output / "checkpoints.json").read_text())
        reports = []
        for checkpoint in checkpoints:
            update = output / checkpoint["update"]
            plan = SharedPlan.model_validate_json((update / "plan.json").read_text())
            report = await SharedEvolution(update, author, evaluator).test_panels(
                plan, config.evaluation_panels
            )
            reports.append(
                {
                    "stage": checkpoint["stage"],
                    "tasks": checkpoint["tasks"],
                    "report": report,
                }
            )
        atomic_json(output / "stage_test_report.json", reports)
        return reports
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "schema",
            "baselines",
            "validate",
            "encode",
            "prepare-fragments",
            "check-fragments",
            "run",
            "test",
            "stream",
            "stream-test",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "schema":
        print(json.dumps(SharedConfiguration.model_json_schema(), indent=2))
        return
    if args.config is None or args.output is None:
        parser.error("--config and --output are required")
    config = SharedConfiguration.model_validate_json(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "command.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = asyncio.run(execute(config, args.output.resolve(), args.mode))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
