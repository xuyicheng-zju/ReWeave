"""Source-only shared construction and continual updates over complete C/W/L.

The single-task GraphEvolution contract is unchanged. This outer search evaluates
one candidate on every development task, fixes source teachers for the entire
update, and freezes selection before any held-out execution.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from reweave.evolution.bundle import AgentBundle, Asset, new_record_id
from reweave.evolution.contracts import (
    EvaluationConditions,
    EvaluationRecord,
    EvaluationRequest,
    EvidenceFile,
    EvolutionIncomplete,
    GraphPolicy,
    Instance,
    SemanticGraph,
)
from reweave.evolution.distillation.metrics import (
    harness_delta_size,
    harness_size,
    retention,
)
from reweave.evolution.distillation.transfer import (
    ConditionalRate,
    MechanismReference,
    aligned_references,
    complete_source_references,
    transfer_loss,
)
from reweave.evolution.runtime_manifest import RuntimeManifest
from reweave.utils.persistence import atomic_json

# Source-size and synthetic sensitivity prior, not a measured convergence target.
DEFAULT_CONSOLIDATION_LOSS_THRESHOLD = 0.01
# Fixed-topology source8 development uses success, progress and positive size excess.
DEFAULT_FIXED_TOPOLOGY_LOSS_THRESHOLD = 0.1


def condition_key(instance: Instance) -> str:
    return json.dumps([instance.task, instance.initialization], sort_keys=True)


class SharedTaskExecution(Asset):
    """Host-owned task budgets, fixed before any shared candidate is evaluated."""

    execution_id: str = Field(min_length=1)
    conditions: EvaluationConditions
    test_conditions: EvaluationConditions

    @model_validator(mode="after")
    def static(self):
        if any(
            c.adaptation_mode != "static"
            for c in (self.conditions, self.test_conditions)
        ):
            raise ValueError("shared task execution requires static episodes")
        return self


class SharedTask(Asset):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    family: str
    dev: tuple[Instance, ...] = Field(min_length=1)
    test: tuple[Instance, ...] = ()
    execution: SharedTaskExecution | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def distinct(self):
        keys = [condition_key(i) for i in (*self.dev, *self.test)]
        if len(keys) != len(set(keys)):
            raise ValueError("development and test conditions must be distinct")
        if len({i.task for i in (*self.dev, *self.test)}) != 1:
            raise ValueError("each named task must use one environment task")
        return self


def task_execution(plan, task, split="dev"):
    binding = task.execution or plan
    return (
        binding.execution_id,
        binding.test_conditions if split == "test" else binding.conditions,
    )


def development_task(plan, task):
    """Expose only this source's development contract to native authors."""
    execution_id, conditions = task_execution(plan, task)
    return {
        "name": task.name,
        "family": task.family,
        "dev": [i.model_dump(mode="json") for i in task.dev],
        **(
            {
                "execution": {
                    "execution_id": execution_id,
                    "conditions": conditions.model_dump(mode="json"),
                }
            }
            if task.execution is not None
            else {}
        ),
    }


class SharedTeacher(Asset):
    task: str
    bundle: AgentBundle
    learned_tasks: tuple[str, ...]
    graph: SemanticGraph | None = None
    evidence: tuple[EvaluationRecord, ...] = ()


class SharedInitialObservation(Asset):
    """An unchanged physical H0 observation, with its actual original execution."""

    bundle: AgentBundle
    runtime: RuntimeManifest
    record: EvaluationRecord
    source: EvidenceFile


class SharedPlan(Asset):
    record_id: str
    mode: Literal["consolidation", "continual", "baseline"]
    method: Literal["aligned", "direct", "fragments"] = Field(
        default="aligned",
        description="Synthesis guidance in the shared engine. Direct is the persisted identifier for w/o topology: same author, objective and evaluation, with no graph/matching/gap inputs to synthesis. Independent fixed-topology measurement remains a common loss term when configured.",
    )
    sources: tuple[SharedTask, ...] = ()
    unseen: tuple[SharedTask, ...] = ()
    teachers: tuple[SharedTeacher, ...] = ()
    initial: AgentBundle
    size_reference: AgentBundle | None = None
    initialize_new_task_graphs: bool = True
    new_task_graph_policy: GraphPolicy = Field(default_factory=GraphPolicy)
    initial_learned_tasks: tuple[str, ...] = ()
    new_tasks: tuple[str, ...] = ()
    first_acquired_scores: dict[str, float] = Field(default_factory=dict)
    execution_id: str
    manifest_id: str
    conditions: EvaluationConditions
    test_conditions: EvaluationConditions
    max_rounds: int = Field(ge=0, strict=True)
    retention_tolerance: float = Field(default=0.0, ge=0, le=1)
    retention_weight: float = Field(default=1.0, ge=0)
    teacher_retention: bool = Field(default=True, exclude_if=lambda value: value)
    sequential_updates: bool = Field(default=False, exclude_if=lambda value: not value)
    checkpoint_selection: Literal["last", "min_loss"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Final checkpoint rule, independent of the update parent. Null preserves historical behavior; min_loss selects the lowest complete development loss including complexity.",
    )
    require_cached_h0: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="Require complete source-development H0 observations and fixed-topology scores; never launch a replacement H0 measurement.",
    )
    fixed_topology_scoring: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    topology_weight: float = Field(
        default=1.0, ge=0, exclude_if=lambda value: value == 1.0
    )
    initial_observations: tuple[SharedInitialObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    initial_topology_score: dict | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    size_weight_per_kb: float = Field(default=0.001, ge=0)
    max_size_kb: float | None = Field(default=None, gt=0)
    target_success_rate: float = Field(default=1.0, ge=0, le=1)
    conditional_retention: bool = True
    mechanism_references: tuple[MechanismReference, ...] = ()
    alignment_records: tuple[dict, ...] = ()
    first_conditional_rates: dict[str, dict] = Field(default_factory=dict)
    alignment_seconds: float = Field(default=1.0, gt=0)
    topology_consolidation: bool = True
    topology_max_proposals: int = Field(default=4096, ge=0, strict=True)
    source_graphs: dict[str, SemanticGraph] = Field(default_factory=dict)
    loss_threshold: float | None = Field(default=None, allow_inf_nan=False)
    # Retained for reading historical plans; new consolidation rejects this policy.
    stop_on_source_success: bool = False

    @model_validator(mode="after")
    def boundaries(self):
        if self.method == "fragments":
            if self.mode != "consolidation":
                raise ValueError(
                    "fragment distillation currently requires consolidation mode"
                )
            if self.alignment_records or any(
                len(r.source_edges) > 1 for r in self.mechanism_references
            ):
                raise ValueError(
                    "fragment distillation cannot inherit global or derived-path alignments"
                )
        if not self.teacher_retention and (
            self.conditional_retention or self.first_acquired_scores
        ):
            raise ValueError("disabled teacher retention cannot have teacher anchors")
        if (
            self.size_reference is not None
            and self.size_reference.runtime != self.initial.runtime
        ):
            raise ValueError("delta Harness reference must use the fixed runtime")
        source = {t.name: t for t in self.sources}
        unseen = {t.name: t for t in self.unseen}
        if self.mode == "baseline":
            if (
                self.sources
                or self.teachers
                or self.new_tasks
                or self.initial_learned_tasks
                or self.max_rounds
                or self.conditional_retention
                or self.mechanism_references
                or self.source_graphs
            ):
                raise ValueError(
                    "zero-source baseline cannot construct or inherit source knowledge"
                )
        elif not self.sources:
            raise ValueError("shared learning requires at least one development source")
        if len(source) != len(self.sources) or len(unseen) != len(self.unseen):
            raise ValueError("duplicate named task")
        if self.fixed_topology_scoring and set(self.source_graphs) != set(source):
            raise ValueError("fixed topology scoring requires every source graph")
        observed = set()
        allowed = {condition_key(i): t for t in self.sources for i in t.dev}
        for observation in self.initial_observations:
            record = observation.record
            key = condition_key(record.request.instance)
            if key not in allowed or key in observed:
                raise ValueError(
                    "H0 observations must be distinct source-dev conditions"
                )
            observed.add(key)
            conditions = task_execution(self, allowed[key])[1]
            if (
                not observation.bundle.same_harness(self.initial)
                or record.request.bundle_id != observation.bundle.record_id
                or record.request.execution != observation.runtime.execution_id
                or record.initialization != record.request.instance.initialization
                or record.request.conditions != conditions
                or observation.runtime.conditions != conditions
                or observation.runtime.runtime_configuration().models
                != self.initial.runtime.models
                or observation.runtime.tool_api != self.initial.runtime.tool_api
                or record.request.split != "dev"
                or record.failure_origin not in {"none", "agent"}
            ):
                raise ValueError("H0 observation differs from the fixed source profile")
        if self.require_cached_h0:
            if observed != allowed.keys():
                raise ValueError(
                    "cached H0 must cover every source development condition"
                )
            if self.fixed_topology_scoring and self.initial_topology_score is None:
                raise ValueError("cached H0 requires its fixed-topology score")
            if self.fixed_topology_scoring:
                score = self.initial_topology_score
                if set(score.get("topology_progress", {})) != source.keys() or set(
                    score.get("topology_records", {})
                ) != {o.record.record_id for o in self.initial_observations}:
                    raise ValueError(
                        "cached H0 topology must cover exactly the reused conditions"
                    )
        if source.keys() & unseen.keys():
            raise ValueError("unseen tasks must be disjoint from sources")
        if {t.family for t in self.sources} & {t.family for t in self.unseen}:
            raise ValueError(
                "unseen-family folds must exclude the complete target family"
            )
        if {i.task for t in self.sources for i in t.dev} & {
            i.task for t in self.unseen for i in t.dev
        }:
            raise ValueError("a new reset of a source task is not an unseen task")
        # Different aliases cannot hide an identical task/reset condition.
        keys = [
            condition_key(i)
            for t in (*self.sources, *self.unseen)
            for i in (*t.dev, *t.test)
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("task/reset overlap across partitions")
        if not set(self.initial_learned_tasks) <= source.keys():
            raise ValueError("initial harness learned unseen tasks")
        if not set(self.new_tasks) <= source.keys():
            raise ValueError("new tasks must be development sources")
        if len(set(self.new_tasks)) != len(self.new_tasks):
            raise ValueError("duplicate new task")
        if not self.first_acquired_scores.keys() <= source.keys() or any(
            not 0 <= value <= 1 for value in self.first_acquired_scores.values()
        ):
            raise ValueError("invalid first-acquisition retention anchors")
        if (
            self.conditions.adaptation_mode != "static"
            or self.test_conditions.adaptation_mode != "static"
        ):
            raise ValueError(
                "shared selection and frozen tests require static episodes"
            )
        names = [teacher.task for teacher in self.teachers]
        if len(names) != len(set(names)) or not set(names) <= source.keys():
            raise ValueError("one source teacher per named task")
        for teacher in self.teachers:
            if not set(teacher.learned_tasks) <= source.keys():
                raise ValueError("source teacher has unseen-task provenance")
            allowed = {condition_key(i) for i in source[teacher.task].dev}
            for record in teacher.evidence:
                if (
                    record.request.split != "dev"
                    or condition_key(record.request.instance) not in allowed
                ):
                    raise ValueError("teacher evidence must be source development only")
                if record.request.bundle_id != teacher.bundle.record_id:
                    raise ValueError("teacher evidence belongs to another bundle")
            if (
                self.method == "aligned"
                and self.mode == "consolidation"
                and (teacher.graph is None or teacher.graph.matching is None)
            ):
                raise ValueError(
                    "aligned consolidation requires canonical source graph conditions"
                )
        if self.mode == "consolidation" and set(names) != source.keys():
            raise ValueError("consolidation needs one teacher for every source task")
        if self.mode == "consolidation" and self.new_tasks:
            raise ValueError("new-task acquisition stopping belongs to continual mode")
        if self.mode == "continual":
            if self.method != "aligned" or not self.initialize_new_task_graphs:
                raise ValueError(
                    "continual learning requires aligned dynamic task topology"
                )
            old = set(self.initial_learned_tasks)
            if set(self.source_graphs) != old or any(
                g.matching is None for g in self.source_graphs.values()
            ):
                raise ValueError(
                    "continual learning requires the complete encoded fixed topology bank for old tasks"
                )
            if self.fixed_topology_scoring:
                raise ValueError(
                    "dynamic new-task topology cannot use fixed all-task progress scoring"
                )
            if (
                not self.new_tasks
                or old & set(self.new_tasks)
                or old | set(self.new_tasks) != source.keys()
            ):
                raise ValueError(
                    "continual update must partition old and newly exposed tasks"
                )
            if set(names) != old or any(
                t.bundle != self.initial for t in self.teachers
            ):
                raise ValueError(
                    "continual teachers must be the preceding shared harness, never new specialists"
                )
        if not self.source_graphs.keys() <= source.keys():
            raise ValueError("source graph bank must remain within development sources")
        if any(
            a.get("query") not in source or a.get("reference") not in source
            for a in self.alignment_records
        ):
            raise ValueError("alignment records must remain source-only")
        if self.mechanism_references:
            if len({r.key for r in self.mechanism_references}) != len(
                self.mechanism_references
            ):
                raise ValueError("duplicate mechanism reference")
            teacher_ids = {t.task: t.bundle.record_id for t in self.teachers}
            if any(
                r.task not in teacher_ids or r.source_bundle_id != teacher_ids[r.task]
                for r in self.mechanism_references
            ):
                raise ValueError(
                    "mechanism references require the current source teachers"
                )
        for rate in self.first_conditional_rates.values():
            ConditionalRate.model_validate(rate)
        if not self.first_conditional_rates.keys() <= {
            r.key for r in self.mechanism_references
        }:
            raise ValueError("conditional floors need their original references")
        return self


def bind_json(path: Path, value: dict) -> None:
    """Persist once; a changed plan/request cannot reuse a previous execution."""
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"published identity changed: {path.name}")
    else:
        atomic_json(path, value)


def task_scores(
    tasks: tuple[SharedTask, ...], records: list[EvaluationRecord]
) -> dict[str, float]:
    indexed = {condition_key(r.request.instance): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate evaluation condition")
    scores = {}
    for task in tasks:
        group = [indexed[condition_key(i)] for i in task.dev]
        if any(r.failure_origin in {"unknown", "infrastructure"} for r in group):
            raise EvolutionIncomplete(
                "unsettled or infrastructure results cannot select a shared harness"
            )
        scores[task.name] = sum(r.official_success for r in group) / len(group)
    return scores


def candidate_score(
    plan: SharedPlan, bundle: AgentBundle, scores: dict, anchors: dict
) -> dict:
    losses = retention(scores, anchors if plan.teacher_retention else {})
    size = harness_size(bundle)
    delta_size = harness_delta_size(bundle, plan.size_reference or plan.initial)
    teacher_deltas = [
        harness_delta_size(t.bundle, plan.size_reference or plan.initial)["size_kb"]
        for t in plan.teachers
    ]
    teacher_mean = sum(teacher_deltas) / len(teacher_deltas) if teacher_deltas else 0.0
    delta_excess = delta_size["size_kb"] - teacher_mean
    macro = sum(scores.values()) / len(scores)
    retained = all(
        drop <= plan.retention_tolerance + 1e-12
        for drop in losses["per_task_drop"].values()
    )
    capacity = plan.max_size_kb is None or size["size_kb"] <= plan.max_size_kb
    loss = (
        1
        - macro
        + plan.retention_weight * losses["mean_drop"]
        + plan.size_weight_per_kb
        * (max(0, delta_excess) if plan.mode == "continual" else delta_excess)
    )
    result = {
        "scores": scores,
        "macro_success": macro,
        **losses,
        "size": size,
        "delta_size": delta_size,
        "teacher_delta_mean_kb": teacher_mean,
        "delta_excess_kb": delta_excess,
        "eligible": retained and capacity,
        "loss": loss,
        "objective": 1 - loss,
    }
    if not plan.teacher_retention:
        result.update(
            retention_evaluated=False,
            per_task_drop={},
            mean_drop=None,
            worst_drop=None,
        )
    return result


class SharedEvolution:
    """Finite, durable outer search; injected adapters execute the real author/runner."""

    def __init__(
        self,
        directory: Path,
        author,
        evaluator,
        assessor=None,
        task_topology=None,
        topology_assessor=None,
        infrastructure_retries=0,
    ):
        if type(infrastructure_retries) is not int or infrastructure_retries < 0:
            raise ValueError("infrastructure_retries must be a nonnegative integer")
        self.directory = directory.resolve()
        self.author = author
        self.evaluator = evaluator
        self.assessor = assessor
        self.task_topology = task_topology
        self.topology_assessor = topology_assessor
        self.infrastructure_retries = infrastructure_retries

    async def evaluate(self, plan, bundle, tasks, directory, *, split="dev"):
        async def evaluate_cell(task, index, instance):
            execution_id, conditions = task_execution(plan, task, split)
            cell = directory / task.name / f"{index:03d}"
            request_path = cell / "shared_request.json"
            request = EvaluationRequest(
                record_id=new_record_id(),
                bundle_id=bundle.record_id,
                instance=instance,
                split=split,
                manifest_id=plan.manifest_id,
                execution=execution_id,
                conditions=conditions,
            )
            if request_path.exists():
                original = EvaluationRequest.model_validate_json(
                    request_path.read_text()
                )
                if original.model_dump(exclude={"record_id"}) != request.model_dump(
                    exclude={"record_id"}
                ):
                    raise ValueError("evaluation recovery identity changed")
                request = original
            bind_json(request_path, request.model_dump(mode="json"))
            for retry in range(self.infrastructure_retries + 1):
                attempt = cell if retry == 0 else cell / "retries" / f"{retry:03d}"
                bind_json(
                    attempt / "shared_request.json", request.model_dump(mode="json")
                )
                path = attempt / "shared_result.json"
                if path.exists():
                    record = EvaluationRecord.model_validate_json(path.read_text())
                else:
                    started = attempt / "shared_started.json"
                    if started.exists():
                        recover = getattr(self.evaluator, "recover", None)
                        if recover is None:
                            raise EvolutionIncomplete(
                                "started physical cell requires owner recovery; no replay"
                            )
                        call = recover
                    else:
                        call = self.evaluator
                    record = await call(
                        request,
                        bundle,
                        attempt,
                        before_launch=lambda: bind_json(
                            started, {"request_id": request.record_id}
                        ),
                    )
                    if record.request != request:
                        raise ValueError("runner returned an unrelated evaluation")
                    atomic_json(path, record.model_dump(mode="json"))
                if (
                    record.request != request
                    or record.initialization != instance.initialization
                ):
                    raise ValueError("saved evaluation identity mismatch")
                if (
                    record.failure_origin != "infrastructure"
                    or not record.infrastructure_retryable
                ):
                    return record
                if retry < self.infrastructure_retries:
                    verify = getattr(self.evaluator, "verify_retry_closed", None)
                    if verify is None:
                        raise EvolutionIncomplete(
                            "infrastructure retry requires original owner closure verification"
                        )
                    await verify(request, bundle, attempt)
            return record

        records = await asyncio.gather(
            *(
                evaluate_cell(task, index, instance)
                for task in tasks
                for index, instance in enumerate(
                    task.test if split == "test" else task.dev
                )
            ),
            return_exceptions=True,
        )
        # The campaign pool owns physical capacity. Settle every admitted cell
        # before propagating a peer failure; never cancel another live rollout.
        for record in records:
            if isinstance(record, BaseException):
                raise record
        return records

    async def run(self, plan: SharedPlan):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "shared.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return await self._run(plan)

    async def _run(self, plan):
        root = self.directory
        bind_json(root / "plan.json", plan.model_dump(mode="json"))
        frozen = root / "frozen.json"
        if frozen.exists():
            return json.loads(frozen.read_text())
        loss_threshold = (
            plan.loss_threshold
            if plan.loss_threshold is not None
            else DEFAULT_FIXED_TOPOLOGY_LOSS_THRESHOLD
        )
        if plan.mode == "consolidation" and (
            plan.loss_threshold is None or plan.stop_on_source_success
        ):
            raise ValueError(
                "shared consolidation requires an explicit loss_threshold and "
                "stop_on_source_success=False; prepare a new plan"
            )
        if plan.mode == "baseline":
            result = {
                "plan_id": plan.record_id,
                "bundle": plan.initial.model_dump(mode="json"),
                "round": 0,
                "score": None,
                "learned_tasks": [],
                "termination": "zero_source_baseline",
                "test_feedback_used": False,
            }
            bind_json(frozen, result)
            return result
        fragment_proposals = None
        if plan.method == "fragments":
            from reweave.evolution.distillation.fragments import fragment_transfer

            admitted_graphs = dict(plan.source_graphs)
            for teacher in plan.teachers:
                if teacher.graph is not None:
                    if (
                        teacher.task in admitted_graphs
                        and admitted_graphs[teacher.task] != teacher.graph
                    ):
                        raise ValueError("conflicting source graph provenance")
                    admitted_graphs[teacher.task] = teacher.graph
            if set(admitted_graphs) != {t.name for t in plan.sources}:
                raise ValueError("fragment distillation requires every source topology")
            # Reject missing encoding before any author/model or physical work.
            fragment_proposals = fragment_transfer(admitted_graphs)
            bind_json(root / "fragment_transfer.json", fragment_proposals)
        initial_records = None
        if plan.new_tasks and plan.initialize_new_task_graphs:
            if self.task_topology is None:
                raise ValueError(
                    "continual learning requires per-version dynamic new-task topology"
                )
            initial_records = await self.initial_evaluation(plan)
        if plan.conditional_retention and self.assessor is None:
            raise ValueError("conditional retention requires an independent assessor")
        references, alignments = (), []
        if plan.conditional_retention:
            contract_path = root / "mechanisms.json"
            if contract_path.exists():
                contract = json.loads(contract_path.read_text())
                references = tuple(
                    MechanismReference.model_validate(r) for r in contract["references"]
                )
                alignments = contract["alignments"]
                if complete_source_references(plan.teachers, references) != references:
                    raise ValueError(
                        "cached source contracts omit private interfaces; use a new run"
                    )
            else:
                if plan.mechanism_references:
                    references = plan.mechanism_references
                    alignments = list(plan.alignment_records)
                else:
                    started = time.monotonic()
                    references, alignments = aligned_references(
                        plan.teachers,
                        plan.alignment_seconds,
                        literal_only=plan.method == "fragments",
                    )
                    atomic_json(
                        root / "alignment_work.json",
                        {"seconds": time.monotonic() - started},
                    )
                references = complete_source_references(plan.teachers, references)
                if not references:
                    raise EvolutionIncomplete(
                        "no witnessed aligned mechanisms; cannot claim conditional distillation"
                    )
                bind_json(
                    contract_path,
                    {
                        "references": [r.model_dump(mode="json") for r in references],
                        "alignments": alignments,
                    },
                )
        from reweave.evolution.distillation.topology import (
            SharedTopology,
            build_topology,
            feedback_splits,
            revise_topology,
            validate_topology,
        )

        source_graphs = dict(plan.source_graphs)
        for teacher in plan.teachers:
            if teacher.graph is not None:
                if (
                    teacher.task in source_graphs
                    and source_graphs[teacher.task] != teacher.graph
                ):
                    raise ValueError("conflicting source graph provenance")
                source_graphs[teacher.task] = teacher.graph
        topology = None
        if (
            plan.method in {"aligned", "fragments"}
            and plan.mode == "consolidation"
            and plan.topology_consolidation
        ):
            if set(source_graphs) != {t.name for t in plan.sources}:
                raise ValueError("shared consolidation requires every source topology")
            topology_started = time.monotonic()
            try:
                topology_path = root / "topology_initial.json"
                if topology_path.exists():
                    topology = SharedTopology.model_validate_json(
                        topology_path.read_text()
                    )
                    validate_topology(topology, source_graphs)
                    saved_alignments = root / "topology_alignments.json"
                    if not alignments and saved_alignments.exists():
                        alignments = json.loads(saved_alignments.read_text())
                else:
                    if not alignments and plan.method == "aligned":
                        _, alignments = aligned_references(
                            plan.teachers, plan.alignment_seconds
                        )
                    bind_json(root / "topology_alignments.json", alignments)
                    topology = build_topology(
                        source_graphs,
                        alignments,
                        max_proposals=0
                        if plan.method == "fragments"
                        else plan.topology_max_proposals,
                        seconds=plan.alignment_seconds,
                    )
                    bind_json(topology_path, topology.model_dump(mode="json"))
            finally:
                atomic_json(
                    root / "topology_work" / f"{time.time_ns()}.json",
                    {
                        "seconds": time.monotonic() - topology_started,
                        "operation": "initialize_or_verify",
                    },
                )
        unit_bank = None
        if plan.method == "aligned" and plan.fixed_topology_scoring:
            from reweave.evolution.distillation.student_gaps import teacher_unit_bank

            unit_bank = teacher_unit_bank(source_graphs)
            bind_json(root / "teacher_unit_bank.json", unit_bank)
        by_name = {t.name: t for t in plan.sources}
        # Reference scores are measured in this runtime, on this dev pool. Never
        # inherit historical test success or assume a teacher's source score.
        anchors = dict(plan.first_acquired_scores)
        teacher_records = []
        conditional_anchors = {}
        for teacher in plan.teachers if plan.teacher_retention else ():
            task = by_name[teacher.task]
            if initial_records is not None and teacher.bundle == plan.initial:
                rows = [
                    r
                    for r in initial_records
                    if condition_key(r.request.instance)
                    in {condition_key(i) for i in task.dev}
                ]
            else:
                rows = await self.evaluate(
                    plan, teacher.bundle, (task,), root / "teachers" / task.name
                )
            score = task_scores((task,), rows)[task.name]
            anchors[task.name] = max(score, anchors.get(task.name, 0.0))
            teacher_records.extend(rows)
            relevant = tuple(r for r in references if r.task == task.name)
            if relevant:
                rates = await self.measure(
                    plan,
                    teacher.bundle,
                    rows,
                    relevant,
                    root / "teachers" / task.name / "measurement",
                )
                for key, value in rates.items():
                    floor = plan.first_conditional_rates.get(key)
                    if floor:
                        value = {**value, "upper": max(value["upper"], floor["upper"])}
                    conditional_anchors[key] = value
        bind_json(root / "anchors.json", anchors)
        if initial_records is None:
            initial_records = await self.initial_evaluation(plan)
        if plan.fixed_topology_scoring:
            bind_json(
                root / "initial_records.json",
                [r.model_dump(mode="json") for r in initial_records],
            )
        initial_score = candidate_score(
            plan, plan.initial, task_scores(plan.sources, initial_records), anchors
        )
        if plan.fixed_topology_scoring:
            await self.add_topology(
                plan,
                plan.initial,
                initial_records,
                initial_score,
                root / "initial/topology",
                cached=plan.initial_topology_score,
            )
        if references:
            local = await self.measure(
                plan,
                plan.initial,
                initial_records,
                references,
                root / "initial" / "measurement",
            )
            self.add_transfer(
                plan, initial_score, references, conditional_anchors, local
            )
        atomic_json(root / "initial" / "score.json", initial_score)
        history = [
            {
                "round": 0,
                "bundle": plan.initial,
                "score": initial_score,
                "records": initial_records,
                "topology": topology.model_dump(mode="json")
                if topology is not None
                else None,
            }
        ]

        def loss_target_reached(score):
            return (
                plan.checkpoint_selection == "min_loss" or score["eligible"]
            ) and score["loss"] < loss_threshold

        if plan.new_tasks and plan.initialize_new_task_graphs:
            history[0]["task_topology"] = await self.task_topology(
                plan, history, root / "initial/task_topologies"
            )
        feedback = list(initial_records)
        current = plan.initial
        rounds = 0 if loss_target_reached(initial_score) else plan.max_rounds
        for index in range(1, rounds + 1):
            round_root = root / "rounds" / f"{index:03d}"
            published = round_root / "request.json"
            saved_request = (
                json.loads(published.read_text()) if published.exists() else None
            )
            request = {
                "round": index,
                "mode": plan.mode,
                "method": plan.method,
                "tasks": [development_task(plan, t) for t in plan.sources],
                "new_tasks": list(plan.new_tasks),
                "source_graphs": {
                    name: graph.model_dump(mode="json")
                    for name, graph in source_graphs.items()
                },
                "shared_topology": topology.model_dump(mode="json")
                if topology is not None
                else None,
                "teachers": [t.model_dump(mode="json") for t in plan.teachers],
                "parent": current.model_dump(mode="json"),
                "size_reference": (plan.size_reference or plan.initial).model_dump(
                    mode="json"
                ),
                "anchors": anchors,
                "criteria": {
                    **(
                        {"checkpoint_selection": plan.checkpoint_selection}
                        if plan.checkpoint_selection is not None
                        else {}
                    ),
                    **({"sequential_updates": True} if plan.sequential_updates else {}),
                    **(
                        {
                            "fixed_topology_scoring": True,
                            "topology_weight": plan.topology_weight,
                            "sequential_updates": plan.sequential_updates,
                        }
                        if plan.fixed_topology_scoring
                        else {}
                    ),
                    **(
                        {"teacher_retention": False}
                        if not plan.teacher_retention
                        else {}
                    ),
                    "retention_tolerance": plan.retention_tolerance,
                    "retention_weight": plan.retention_weight,
                    "size_weight_per_kb": plan.size_weight_per_kb,
                    "max_size_kb": plan.max_size_kb,
                    "loss_threshold": loss_threshold,
                    "teacher_delta_mean_kb": initial_score["teacher_delta_mean_kb"],
                    "size_metric": "shared_delta_kb_minus_teacher_mean_delta_kb",
                    "positive_size_excess": plan.mode == "continual"
                    or plan.fixed_topology_scoring,
                    "stopping_rule": "loss_below_threshold_or_round_limit",
                },
                "conditions": plan.conditions.model_dump(mode="json"),
                "alignment_seconds": plan.alignment_seconds,
                "mechanisms": [r.model_dump(mode="json") for r in references],
                "fixed_alignments": alignments if plan.method == "aligned" else [],
                **(
                    {"fragment_transfer": fragment_proposals}
                    if fragment_proposals is not None
                    else {}
                ),
                "conditional_anchors": conditional_anchors,
                "feedback": [
                    r.model_dump(mode="json") for r in (*teacher_records, *feedback)
                ],
                "history": [
                    {"round": h["round"], "score": h["score"]} for h in history
                ],
            }
            if (
                unit_bank is not None
                and not (round_root / "author/result.json").exists()
            ):
                from reweave.evolution.distillation.student_gaps import (
                    build_gap_report,
                )

                student = next(
                    h
                    for h in reversed(history)
                    if h["bundle"].record_id == current.record_id
                )
                gap_report = build_gap_report(
                    plan,
                    unit_bank,
                    current.record_id,
                    student["records"],
                    student["score"],
                )
                bind_json(round_root / "student_gaps.json", gap_report)
                request["student_gaps"] = gap_report
                request["teacher_unit_bank"] = unit_bank
            elif unit_bank is not None:
                # Replay completed requests exactly; recovery never reauthors measured work.
                saved = json.loads((round_root / "request.json").read_text())
                for key in ("student_gaps", "teacher_unit_bank"):
                    if key in saved:
                        request[key] = saved[key]
            if plan.new_tasks and plan.initialize_new_task_graphs:
                student = next(
                    h
                    for h in reversed(history)
                    if h["bundle"].record_id == current.record_id
                )
                live = student["task_topology"]
                request["source_graphs"].update(live["graphs"])
                request["student_gaps"] = live["student_gaps"]
                request["teacher_unit_bank"] = live["teacher_unit_bank"]
                request["topology_refresh"] = {
                    "student_bundle_id": live["student_bundle_id"],
                    "new_tasks": list(live["graphs"]),
                    "old_tasks": list(source_graphs),
                    "current_record_ids": live["current_record_ids"],
                }
                if saved_request is not None:
                    saved_tasks = saved_request["topology_refresh"]["old_tasks"]
                    if sorted(saved_tasks) != sorted(source_graphs):
                        raise ValueError("old task membership changed during recovery")
                    request["topology_refresh"]["old_tasks"] = saved_tasks
            # Request deliberately excludes unseen task names/conditions and all test evidence.
            if plan.method == "direct":
                request["source_graphs"] = {}
                request["teachers"] = [
                    {**teacher, "graph": None} for teacher in request["teachers"]
                ]
                request["mechanisms"] = []
                request["conditional_anchors"] = {}
                request["history"] = [
                    {
                        **item,
                        "score": {
                            k: v
                            for k, v in item["score"].items()
                            if k != "topology_records"
                        },
                    }
                    for item in request["history"]
                ]
            if saved_request is not None:
                for field in ("loss_threshold", "stopping_rule"):
                    request["criteria"][field] = saved_request["criteria"][field]
            bind_json(published, request)
            authored = await self.author(request, round_root / "author")
            candidate = AgentBundle.model_validate(authored["bundle"])
            if topology is not None and authored.get("topology") is not None:
                topology = SharedTopology.model_validate(authored["topology"])
                validate_topology(topology, source_graphs)
            if candidate.runtime != plan.initial.runtime:
                raise ValueError("shared author changed the active runtime")
            rows = await self.evaluate(
                plan, candidate, plan.sources, round_root / "evaluation"
            )
            scores = task_scores(plan.sources, rows)
            score = candidate_score(plan, candidate, scores, anchors)
            if plan.fixed_topology_scoring:
                await self.add_topology(
                    plan, candidate, rows, score, round_root / "topology"
                )
            if references:
                local = await self.measure(
                    plan, candidate, rows, references, round_root / "measurement"
                )
                self.add_transfer(plan, score, references, conditional_anchors, local)
            atomic_json(round_root / "score.json", score)
            history.append(
                {
                    "round": index,
                    "bundle": candidate,
                    "score": score,
                    "records": rows,
                    "topology": topology.model_dump(mode="json")
                    if topology is not None
                    else None,
                }
            )
            if plan.new_tasks and plan.initialize_new_task_graphs:
                history[-1]["task_topology"] = await self.task_topology(
                    plan, history, round_root / "task_topologies"
                )
            feedback.extend(rows)
            if topology is not None:
                topology_started = time.monotonic()
                try:
                    split = feedback_splits(
                        topology,
                        scores,
                        anchors,
                        plan.retention_tolerance,
                        references,
                        conditional_anchors,
                        score.get("conditional_rates"),
                    )
                    topology = revise_topology(
                        topology,
                        source_graphs,
                        split=split,
                        reason=f"development feedback round {index}",
                    )
                    bind_json(
                        round_root / "topology_after.json",
                        topology.model_dump(mode="json"),
                    )
                finally:
                    atomic_json(
                        round_root / "topology_work" / f"{time.time_ns()}.json",
                        {
                            "seconds": time.monotonic() - topology_started,
                            "operation": "feedback_refinement",
                        },
                    )
            if loss_target_reached(score):
                break
            # Infeasible candidates may inform revision but cannot become the
            # published model or move the fixed retention reference.
            eligible = [h for h in history if h["score"]["eligible"]]
            current = (
                candidate
                if plan.sequential_updates
                else max(
                    eligible or history,
                    key=lambda h: (
                        h["score"]["objective"],
                        -h["score"]["delta_size"]["total_bytes"],
                        -h["round"],
                    ),
                )["bundle"]
            )
        if plan.checkpoint_selection == "min_loss":
            best = min(history, key=lambda h: (h["score"]["loss"], h["round"]))
        else:
            eligible = [h for h in history if h["score"]["eligible"]]
            if not eligible:
                atomic_json(
                    root / "rejected.json",
                    {
                        "reason": "no retention-feasible shared harness",
                        "anchors": anchors,
                    },
                )
                raise EvolutionIncomplete(
                    "no shared candidate meets source retention; no version published"
                )
            qualified_shared = [h for h in eligible if loss_target_reached(h["score"])]
            if qualified_shared:
                eligible = qualified_shared
            select_last = plan.checkpoint_selection == "last" or (
                plan.checkpoint_selection is None and plan.sequential_updates
            )
            if select_last:
                if not history[-1]["score"]["eligible"]:
                    raise EvolutionIncomplete(
                        "last shared candidate is infeasible; no historical fallback"
                    )
                best = history[-1]
            else:
                best = max(
                    eligible,
                    key=lambda h: (
                        h["score"]["objective"],
                        -h["score"]["delta_size"]["total_bytes"],
                        -h["round"],
                    ),
                )
        acquired = dict(plan.first_acquired_scores)
        for name, score in best["score"]["scores"].items():
            acquired.setdefault(name, score)
        retained_references = list(references)
        conditional_floors = dict(plan.first_conditional_rates)
        if references:
            for key, value in best["score"]["conditional_rates"].items():
                conditional_floors.setdefault(key, value)
            # Newly acquired task mechanisms become old-capability contracts in
            # the NEXT update; never retroactively change this round's objective.
            covered = {r.task for r in references}
            missing = tuple(t.name for t in plan.sources if t.name not in covered)
            if missing:
                from reweave.evolution.distillation.assessment import (
                    measurement_request,
                )

                rows = [
                    r
                    for r in best["records"]
                    if r.request.instance.task
                    in {
                        i.task for t in plan.sources if t.name in missing for i in t.dev
                    }
                ]
                discovered = await self.assessor(
                    measurement_request(
                        plan, best["bundle"], rows, discover_tasks=missing
                    ),
                    root / "acquired_mechanisms" / "author",
                )
                fresh = tuple(
                    MechanismReference.model_validate(r)
                    for r in discovered["references"]
                )
                if {r.task for r in fresh} != set(missing):
                    raise ValueError("discovered mechanisms omit an acquired task")
                retained_references.extend(fresh)
                conditional_floors.update(
                    await self.measure(
                        plan, best["bundle"], rows, fresh, root / "acquired_measurement"
                    )
                )
        result = {
            **(
                {
                    "checkpoint_selection": plan.checkpoint_selection,
                    "last_evaluated_round": history[-1]["round"],
                    "checkpoints": [
                        {
                            "round": h["round"],
                            "bundle_id": h["bundle"].record_id,
                            "loss": h["score"]["loss"],
                            "eligible": h["score"]["eligible"],
                        }
                        for h in history
                    ],
                }
                if plan.checkpoint_selection is not None
                else {}
            ),
            "shared_topology": best.get("topology"),
            "source_graphs": {
                name: graph.model_dump(mode="json")
                for name, graph in source_graphs.items()
            }
            | (best["task_topology"]["graphs"] if "task_topology" in best else {}),
            "task_topology": best.get("task_topology"),
            "termination": "loss_threshold_reached"
            if loss_target_reached(best["score"])
            else "round_limit",
            "mechanism_references": [
                r.model_copy(
                    update={"source_bundle_id": best["bundle"].record_id}
                ).model_dump(mode="json")
                for r in retained_references
            ],
            "first_conditional_rates": conditional_floors,
            "plan_id": plan.record_id,
            "bundle": best["bundle"].model_dump(mode="json"),
            "round": best["round"],
            "score": best["score"],
            "learned_tasks": [t.name for t in plan.sources],
            "first_acquired_scores": acquired,
            "initial_scores": initial_score["scores"],
            "new_task_acquisition": {
                name: next(
                    (
                        h["round"]
                        for h in history
                        if h["score"]["scores"][name] >= plan.target_success_rate
                    ),
                    None,
                )
                for name in plan.new_tasks
            },
            "max_rounds": plan.max_rounds,
            "rounds_executed": len(history) - 1,
            "loss_threshold": loss_threshold,
            "loss_threshold_reached": loss_target_reached(best["score"]),
            "test_feedback_used": False,
        }
        bind_json(frozen, result)
        return result

    async def initial_evaluation(self, plan):
        root = self.directory
        reused = {}
        for observation in plan.initial_observations:
            observation.source.verify()
            original = EvaluationRecord.model_validate_json(
                Path(observation.source.path).read_text()
            )
            if original != observation.record:
                raise ValueError("original H0 observation changed")
            reused[condition_key(original.request.instance)] = original
        missing = tuple(
            t.model_copy(
                update={
                    "dev": tuple(i for i in t.dev if condition_key(i) not in reused)
                }
            )
            for t in plan.sources
            if any(condition_key(i) not in reused for i in t.dev)
        )
        if plan.require_cached_h0 and missing:
            raise ValueError("cached H0 is incomplete; no fresh H0 rollout is allowed")
        fresh = await self.evaluate(plan, plan.initial, missing, root / "initial")
        indexed = {
            **reused,
            **{condition_key(r.request.instance): r for r in fresh},
        }
        initial_records = [
            indexed[condition_key(i)] for t in plan.sources for i in t.dev
        ]
        if reused:
            bind_json(
                root / "initial_reuse.json",
                {k: r.model_dump(mode="json") for k, r in reused.items()},
            )
        return initial_records

    async def add_topology(
        self, plan, bundle, records, score, directory, *, cached=None
    ):
        if cached is None:
            if self.topology_assessor is None:
                raise ValueError(
                    "fixed topology scoring requires an independent projector"
                )
            cached = await self.topology_assessor.score(
                plan, bundle, records, directory
            )
        bind_json(directory / "result.json", cached)
        score.update(cached)
        score["size_loss"] = plan.size_weight_per_kb * max(0, score["delta_excess_kb"])
        score["loss"] = (
            1
            - score["macro_success"]
            + plan.topology_weight * score["topology_loss"]
            + score["size_loss"]
        )
        score["objective"] = 1 - score["loss"]

    async def measure(self, plan, bundle, records, references, directory):
        from reweave.evolution.distillation.assessment import measurement_request

        request = measurement_request(plan, bundle, records, references)
        result = await self.assessor(request, directory / "author")
        if set(result["rates"]) != {r.key for r in references}:
            raise ValueError("assessor omitted a fixed condition")
        return {
            key: ConditionalRate.model_validate(value).model_dump(mode="json")
            for key, value in result["rates"].items()
        }

    @staticmethod
    def add_transfer(plan, score, references, anchors, rates):
        local = transfer_loss(references, anchors, rates)
        score.update(local)
        # Task drops remain HARD feasibility constraints. Eq. 9's soft penalty
        # is the conditional loss, replacing the prototype task-drop surrogate.
        score["loss"] = (
            1
            - score["macro_success"]
            + plan.retention_weight * local["conditional_loss"]
            + plan.size_weight_per_kb * score["delta_excess_kb"]
        )
        score["objective"] = 1 - score["loss"]

    async def test(self, plan):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "shared.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return await self._test(plan)

    async def _test(self, plan):
        root = self.directory
        bind_json(root / "plan.json", plan.model_dump(mode="json"))
        path = root / "frozen.json"
        if not path.exists():
            raise ValueError(
                "freeze a source-only shared harness before testing unseen tasks"
            )
        frozen = json.loads(path.read_text())
        if frozen["plan_id"] != plan.record_id:
            raise ValueError("frozen harness belongs to another plan")
        bundle = AgentBundle.model_validate(frozen["bundle"])
        results = {}
        for group, tasks in (("source", plan.sources), ("unseen", plan.unseen)):
            rows = await self.evaluate(
                plan, bundle, tasks, root / "test" / group, split="test"
            )
            results[group] = {
                task.name: {
                    "successes": sum(
                        r.official_success
                        for r in rows
                        if condition_key(r.request.instance)
                        in {condition_key(i) for i in task.test}
                    ),
                    "episodes": len(task.test),
                    "settled": all(
                        r.failure_origin not in {"unknown", "infrastructure"}
                        for r in rows
                        if condition_key(r.request.instance)
                        in {condition_key(i) for i in task.test}
                    ),
                }
                for task in tasks
            }
        atomic_json(root / "test_report.json", results)
        return results

    async def test_panels(self, plan, panels):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "shared.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bind_json(self.directory / "plan.json", plan.model_dump(mode="json"))
            bind_json(
                self.directory / "evaluation_panels.json",
                {
                    name: [task.model_dump(mode="json") for task in tasks]
                    for name, tasks in panels.items()
                },
            )
            path = self.directory / "frozen.json"
            if not path.exists():
                raise ValueError("freeze the harness before fixed-panel testing")
            frozen = json.loads(path.read_text())
            if frozen["plan_id"] != plan.record_id:
                raise ValueError("frozen harness belongs to another plan")
            bundle = AgentBundle.model_validate(frozen["bundle"])
            result = {"panels": {}, "families": {}}
            for name, tasks in panels.items():
                if not name.replace("_", "").replace("-", "").isalnum():
                    raise ValueError("invalid evaluation panel name")
                rows = await self.evaluate(
                    plan,
                    bundle,
                    tasks,
                    self.directory / "test" / "panels" / name,
                    split="test",
                )
                result["panels"][name] = {
                    task.name: {
                        "successes": sum(
                            r.official_success
                            for r in rows
                            if condition_key(r.request.instance)
                            in {condition_key(i) for i in task.test}
                        ),
                        "episodes": len(task.test),
                        "settled": all(
                            r.failure_origin not in {"unknown", "infrastructure"}
                            for r in rows
                            if condition_key(r.request.instance)
                            in {condition_key(i) for i in task.test}
                        ),
                    }
                    for task in tasks
                }
                result["families"].update({task.name: task.family for task in tasks})
            atomic_json(self.directory / "test_report.json", result)
            return result
