# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Shared immutable contracts for authoring, execution and evolution."""

from __future__ import annotations

from math import fsum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, model_validator

from reweave.evolution.bundle import (
    AgentBundle,
    Asset,
    PublishedRecord,
    RecordId,
    RuntimeConfiguration,
    Scope,
)
from reweave.evolution.evidence.archive_inputs import (
    archived_proposal,
    archived_proposal_value,
)
from reweave.evolution.evidence.records import EpisodeEvidence
from reweave.evolution.graph.conditions import GraphSemantics

BranchId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class EvolutionIncomplete(RuntimeError):
    """A launch cannot safely continue without explicit recovery or missing results."""


class Instance(PublishedRecord):
    """A complete task/reset condition, not a seed shared across different tasks."""

    task: str = Field(min_length=1, description="Environment-qualified task identity.")
    initialization: dict[str, Any] = Field(
        description="Full reset, seed, perturbation and initial-state condition."
    )


class EvaluationConditions(Asset):
    """Fixed full-agent budgets and sampling, independent of a candidate or seed."""

    physical_steps: int | None = Field(
        gt=0,
        strict=True,
        description="Total environment control steps including settling; null requires a finite episode deadline. Time-uncapped evaluations require a finite physical work allowance.",
    )
    wait_steps: int = Field(
        ge=0, strict=True, description="Settling steps before agent control."
    )
    model_requests: int = Field(
        gt=0,
        strict=True,
        description="Shared root/local controller allowance in controller_request_unit; excludes VLA/SAM inference. Native turns do not claim a count of hidden provider inference attempts.",
    )
    controller_request_unit: Literal["api_attempt", "native_turn"] = Field(
        description="api_attempt counts every actual API attempt including failures/retries; native_turn counts submitted provider turns, whose internal model attempts may be unreported.",
    )
    adaptation_mode: Literal["static", "adaptive"] = Field(
        description="Whether the episode may activate online W/C/L revisions. Static and adaptive episodes are distinct execution conditions.",
    )
    online_workflows: bool = Field(
        default=False,
        strict=True,
        exclude_if=lambda value: not value,
        description="Experimental online task-family W authoring. Disabled and omitted for historical/main conditions; enabling grants W authority only, independently of C/L adaptation.",
    )
    wall_time_s: float | None = Field(
        gt=0,
        strict=True,
        description="Total online episode wall-clock deadline, including API waits; null disables this time cap while request and physical limits remain in force.",
    )
    program_calls: int | None = Field(
        gt=0,
        strict=True,
        description="Total root, context and local callable invocations per episode; null disables this additional cap.",
    )
    gateway_calls: int | None = Field(
        gt=0,
        strict=True,
        description="Total operation dispatches across controller and workflows; null disables this additional cap.",
    )
    repetition: int = Field(
        ge=0,
        strict=True,
        description="Explicit repetition identity; never inferred from a retry.",
    )
    model_sampling: dict[str, Any] = Field(
        description="Actual exposed inference settings used by the runner; unreported native provider settings are not invented."
    )

    @model_validator(mode="after")
    def _bounded_work(self):
        if self.wall_time_s is None and self.physical_steps is None:
            raise ValueError(
                "time-uncapped evaluation requires a finite physical step budget"
            )
        return self


class GraphPolicy(Asset):
    """Frozen evidence scaffold and semantic analysis deadline; width uses fixed Host rules."""

    distance_bins_m: tuple[float, ...] = Field(
        default=(0.02, 0.10),
        description="Geometric EEF/entity distance boundaries in metres.",
    )
    joint_fraction_bins: tuple[float, ...] = Field(
        default=(0.25, 0.50, 0.75),
        description="Normalized finite joint range boundaries; never task progress.",
    )
    displacement_bin_m: float = Field(
        default=0.02,
        gt=0,
        description="Entity displacement from recorded reset boundary in metres.",
    )
    analysis_wall_time_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Optional end-to-end native graph analysis deadline in seconds; null disables the time cap.",
    )

    @model_validator(mode="after")
    def _bins(self):
        for bins in (self.distance_bins_m, self.joint_fraction_bins):
            if not bins or tuple(sorted(set(bins))) != bins or any(x < 0 for x in bins):
                raise ValueError(
                    "graph bins must be finite, unique, nonnegative and increasing"
                )
        return self


def same_execution_settings(
    actual: EvaluationConditions, expected: EvaluationConditions
) -> bool:
    """An independent repetition changes identity, never budgets or sampling."""
    return actual.model_dump(exclude={"repetition"}) == expected.model_dump(
        exclude={"repetition"}
    )


class FeedbackValidation(Asset):
    """A fixed second condition pool queried only after development qualification."""

    conditions: tuple[Instance, ...] = Field(
        min_length=1,
        description="Distinct task conditions disjoint from dev; observed failures feed subsequent evolution, so this is not a held-out test.",
    )
    target_success_rate: float = Field(
        gt=0,
        le=1,
        allow_inf_nan=False,
        description="Required official success fraction on this complete pool for the same development-qualified Bundle.",
    )


class SearchPolicy(Asset):
    """Finite-horizon Bayesian width: planning prices and the value of success.

    Success is valued in declared candidate costs (one author plus one rollout per
    development condition), so the value is invariant to the condition count; prices
    change width only when measured rollout dollars depart from the declared ones.
    """

    author_cost_usd: float = Field(
        default=8.0,
        gt=0,
        allow_inf_nan=False,
        description="Planning dollars per author session.",
    )
    rollout_cost_usd: float = Field(
        default=20.0,
        ge=0,
        allow_inf_nan=False,
        description="Planning dollars per development-condition rollout, used when complete measured dollars are unavailable.",
    )
    success_value: float = Field(
        default=50.0,
        gt=0,
        allow_inf_nan=False,
        description="Value of reaching the development target within the round budget, in declared candidate costs.",
    )
    prior_strength: float = Field(
        default=3.0,
        gt=0,
        allow_inf_nan=False,
        description="Beta concentration of the per-condition success rate; sibling outcome correlation is 1/(1+prior_strength).",
    )


class SearchCost(Asset):
    """Measured rollout dollars for width; missing measured dollars remain null."""

    rollout_usd: float | None = Field(
        ge=0,
        allow_inf_nan=False,
        description="Sum of median per-dev-condition dollars plus observed conditional validation dollars per evaluated content; retries included, null if incomplete. Null uses the declared planning price.",
    )


class EvolutionProtocol(PublishedRecord):
    """One task evaluated under a fixed, shared set of development conditions."""

    execution_id: RecordId = Field(
        description="Published complete evaluation execution configuration."
    )
    manifest_id: RecordId = Field(
        description="Published ordered physical condition inventory."
    )

    graph: GraphPolicy = Field(
        description="Required frozen physical graph and refinement policy; old matrix protocols are rejected."
    )
    search: SearchPolicy = Field(
        default_factory=SearchPolicy,
        description="Width planning prices and success value for the finite-horizon Bayesian width.",
    )

    dev: tuple[Instance, ...] = Field(
        min_length=1,
        description="Complete distinct conditions shared by every candidate.",
    )
    validation: FeedbackValidation | None = Field(
        default=None,
        description="Optional feedback validation gate after dev qualification; failures return to the shared history and physical graph without expanding the per-candidate dev set.",
    )
    max_rounds: int = Field(
        ge=0,
        strict=True,
        description="Maximum completed evolution rounds, excluding baseline evaluation; zero evaluates H0 only.",
    )
    execution: EvaluationConditions = Field(
        description="Complete frozen episode budgets and model sampling conditions."
    )
    execution_runtime: RuntimeConfiguration | None = Field(
        default=None,
        description="Published ordinary active execution configuration. A configuration revision retains historical Bundles and receipts without checking source identity.",
    )
    infrastructure_retries: int = Field(
        ge=0,
        strict=True,
        description="Explicit extra attempts only for runner-classified retryable infrastructure failures; all attempts are retained and charged. Settled missing outcomes remain null and cannot qualify, but do not stop graph analysis, later candidates or rounds. They never authorize extra retries or replay.",
    )

    max_authors_per_round: int = Field(
        default=1,
        gt=0,
        strict=True,
        description="Operator-controlled maximum independent author sessions; default one. Dynamic mode selects the actual width within this cap.",
    )
    width_mode: Literal["dynamic", "fixed"] = Field(
        default="dynamic",
        exclude_if=lambda value: value == "dynamic",
        description="Dynamic uses the finite-horizon Bayesian width; fixed launches exactly max_authors_per_round independent candidates in each mutation round. Both retain all candidates and their costs.",
    )
    mutation_scopes: tuple[Scope, ...] = Field(
        default=("W", "C", "L"),
        exclude_if=lambda value: value == ("W", "C", "L"),
        description="Author write authority, in W/C/L order. Unlisted components remain at H0 throughout this independent search. Restricted searches retain measured winners directly and skip unmeasured global source consolidation.",
    )
    success_consolidation: bool = Field(
        default=True,
        strict=True,
        exclude_if=lambda value: value,
        description="Whether to create a separate reusable-source artifact after measured success. This author work never changes the measured candidate or inherits its score. Component ablations disable it for every arm so post-search C/L extraction cannot change their edit scope or cost.",
    )
    evolution_mode: Literal["topology", "no_topology"] = Field(
        default="topology",
        exclude_if=lambda value: value == "topology",
        description="Topology uses graph analysis and progress. No-topology retains the same author and raw history, selecting search parents and final output globally by confirmed official successes, official coverage, then recency; complete rows reduce to SR with newest-version ties.",
    )
    topology_mode: Literal["evolving", "fixed_initial"] = Field(
        default="evolving", exclude_if=lambda value: value == "evolving"
    )
    parent_selection: Literal["global", "latest_evaluated"] = Field(
        default="global",
        exclude_if=lambda value: value == "global",
        description="Search parent only. Latest-evaluated uses the last chronological complete development result; final archive selection remains global.",
    )
    parent_rollback_min_progress: float = Field(
        default=0.10,
        ge=0,
        le=1,
        allow_inf_nan=False,
        exclude_if=lambda value: value == 0.10,
        description="At equal confirmed development success counts, an older-round search parent must exceed the best completely evaluated latest-round candidate's mean retained progress by at least this amount (0.10 = ten percentage points). Higher official success bypasses the gate; final archive selection is unaffected.",
    )
    author_wall_time_s: float | None = Field(
        default=None,
        gt=0,
        strict=True,
        allow_inf_nan=False,
        description="Optional Host-measured end-to-end author deadline, including preparation, provider waits, admission and same-session correction; null disables the time cap.",
    )
    target_success_rate: float = Field(
        gt=0,
        le=1,
        allow_inf_nan=False,
        description="Required mean official success on the complete development set.",
    )

    @model_validator(mode="after")
    def _conditions(self) -> EvolutionProtocol:
        if self.evolution_mode == "no_topology" and (
            self.max_authors_per_round != 1
            or self.mutation_scopes != ("W", "C", "L")
            or self.topology_mode != "evolving"
        ):
            raise ValueError(
                "no-topology keeps the full author and history, with width capped at one"
            )
        if self.parent_selection == "latest_evaluated" and (
            self.evolution_mode != "topology" or self.max_authors_per_round != 1
        ):
            raise ValueError(
                "latest-evaluated parent requires width capped at one and current topology projection"
            )
        if (
            self.width_mode == "dynamic"
            and self.max_authors_per_round != 1
            and (
                self.validation is not None
                or (len(self.dev) > 1 and self.target_success_rate < 1)
            )
        ):
            raise ValueError(
                "dynamic width requires development success to be the forecast completion event"
            )
        if not self.mutation_scopes or self.mutation_scopes != tuple(
            scope for scope in ("W", "C", "L") if scope in self.mutation_scopes
        ):
            raise ValueError(
                "mutation scopes must be a nonempty unique W/C/L subsequence"
            )
        if self.execution_runtime is not None and (
            self.execution_runtime.physical_steps,
            self.execution_runtime.wait_steps,
        ) != (self.execution.physical_steps, self.execution.wait_steps):
            raise ValueError("execution runtime horizon differs from episode budgets")
        if len({instance.task for instance in self.instances}) != 1:
            raise ValueError(
                "development and validation conditions must belong to one task"
            )
        if len({instance.record_id for instance in self.instances}) != len(
            self.instances
        ) or any(
            left.model_dump(exclude={"record_id"})
            == right.model_dump(exclude={"record_id"})
            for index, left in enumerate(self.instances)
            for right in self.instances[index + 1 :]
        ):
            raise ValueError(
                "duplicate or overlapping development/validation condition"
            )
        for instance in self.instances:
            key = "layout" if instance.task.startswith("robodojo:") else "seed"
            reset_id = instance.initialization.get(key)
            if type(reset_id) is not int or reset_id < 0:
                raise ValueError(
                    f"evaluation initialization requires a nonnegative integer {key}"
                )
        return self

    @property
    def instances(self) -> tuple[Instance, ...]:
        """Ordered physical condition inventory: development, then feedback validation."""
        return self.dev + (self.validation.conditions if self.validation else ())


class Usage(Asset):
    """Measured work; unreported provider cost remains unknown."""

    physical_steps: int = Field(
        ge=0,
        description="All observed simulator steps including settling and failed attempts.",
    )
    physical_steps_complete: bool = Field(
        default=True,
        strict=True,
        description="False means the step count is a measured lower bound; reserve the unobserved horizon.",
    )
    model_calls: int | None = Field(
        ge=0,
        description="All inference attempts, including controller, VLA and SAM, when fully observable; not the controller decision allowance.",
    )
    wall_time_s: float = Field(
        ge=0, description="Measured elapsed wall-clock execution time."
    )
    wall_time_complete: bool = Field(
        default=True,
        strict=True,
        description="False means elapsed time is a durable lower bound, not a complete measurement.",
    )
    wall_time_charge_s: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Host-reconciled conservative experiment charge after interruption, including downtime; never an exact elapsed-time measurement.",
    )
    cost_usd: float | None = Field(
        ge=0,
        description="Complete reported dollar cost, or null when any part is unknown.",
    )

    @model_validator(mode="after")
    def _timing_charge(self) -> Usage:
        if (
            self.wall_time_charge_s is not None
            and self.wall_time_charge_s < self.wall_time_s
        ):
            raise ValueError("reconciled time charge cannot be below measured time")
        return self

    @property
    def charged_wall_time_s(self) -> float:
        return (
            self.wall_time_s
            if self.wall_time_charge_s is None
            else self.wall_time_charge_s
        )


class EvidenceFile(Asset):
    """Retained original artifact with its ordinary published reference."""

    path: str = Field(
        min_length=1, description="Absolute path of the retained original artifact."
    )
    artifact_id: RecordId = Field(description="Saved original file record.")
    bytes: int = Field(ge=0, strict=True, description="Recorded original file size.")

    def verify(self) -> None:
        """Require the retained readable original."""
        source = Path(self.path)
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            raise ValueError(f"evidence is not a regular absolute file: {self.path}")


class EvaluationRequest(PublishedRecord):
    """Exact evaluation/cache identity, independent of its process attempt."""

    bundle_id: RecordId = Field(description="Immutable content being run.")
    instance: Instance = Field(description="Exact task and initialization.")
    split: Literal["dev", "validation", "test"] = Field(
        description="Development, gated feedback validation, or a separate frozen held-out test that cannot guide evolution."
    )
    manifest_id: RecordId = Field(
        description="Published ordered physical condition inventory."
    )
    execution: str = Field(
        min_length=1,
        description="Ordinary published execution configuration reference.",
    )
    conditions: EvaluationConditions = Field(
        description="Actual complete budget and model sampling conditions."
    )


class EvaluationRecord(PublishedRecord):
    """Runner-owned outcome. A candidate cannot label its own error as infrastructure."""

    request: EvaluationRequest = Field(description="Exact condition actually executed.")
    official_success: bool = Field(
        strict=True, description="Independent environment evaluator verdict."
    )
    failure_origin: Literal["none", "agent", "infrastructure", "unknown"] = Field(
        description="Runner-owned cause: unknown denotes an unacknowledged submitted action, scored as failure without automatic retry or assumed agent blame."
    )
    infrastructure_retryable: bool | None = Field(
        description="Runner-owned retry eligibility for infrastructure failures; null for every settled or unknown physical outcome."
    )
    usage: Usage = Field(description="Measured usage including failed work.")
    evidence: tuple[EvidenceFile, ...] = Field(
        min_length=1, description="Complete retained result inventory."
    )
    episode: EpisodeEvidence | None = Field(
        description="Complete decision archive, or null for incomplete runner evidence and explicitly synthetic protocol fixtures; original runner files remain in evidence."
    )
    initialization: dict[str, Any] = Field(
        description="Actual runner-confirmed task initialization, compared directly with the requested reset."
    )

    @model_validator(mode="after")
    def _outcome(self) -> EvaluationRecord:
        if self.official_success != (self.failure_origin == "none"):
            raise ValueError("official success and failure classification disagree")
        if (self.failure_origin == "infrastructure") != (
            self.infrastructure_retryable is not None
        ):
            raise ValueError(
                "only infrastructure failures require explicit retry eligibility"
            )
        if self.initialization != self.request.instance.initialization:
            raise ValueError("actual initialization differs from requested reset")
        if len({f.path for f in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence paths")
        return self

    def verify(self) -> None:
        for artifact in self.evidence:
            artifact.verify()


class EvolutionHistory(Asset):
    """Frozen shared memory given to every author in one evolution round."""

    summary: EvidenceFile = Field(
        description="Host-derived index of earlier mutations, parentage and observed development results. Complete author diagnoses and SKILL.md testimony remain in the indexed originals and may contain incorrect hypotheses."
    )
    originals: EvidenceFile = Field(
        description="Manifest of immutable author, evaluation, failure, and prior-history originals available for on-demand inspection."
    )

    def verify(self) -> None:
        """Verify both the eagerly supplied summary and the on-demand index."""
        self.summary.verify()
        self.originals.verify()


class VersionReference(Asset):
    """An exact historical version and its immutable executable content."""

    version_id: RecordId = Field(description="Version event in the frozen run archive.")
    bundle_id: RecordId = Field(
        description="Executable bundle content owned by that version."
    )


class OriginalReference(Asset):
    """Exact immutable file or location within JSON/JSONL evidence."""

    path: str = Field(
        description="Absolute original file path, bound to the evidence inventory."
    )
    artifact_id: RecordId = Field(description="Saved original file record.")
    bytes: int = Field(ge=0, strict=True, description="Recorded original file size.")
    pointer: str = Field(
        description="Empty for a complete code/media/evidence file; otherwise a JSON pointer. JSONL uses zero-based original line index as its first segment."
    )


class PhysicalFeature(Asset):
    """A measured scalar and its physical interpretation, independent of success."""

    value: bool | float | str | None = Field(
        description="Observed scalar; null denotes unavailable evidence."
    )
    unit: str = Field(
        description="SI unit, dimensionless, boolean, or text; never an implicit progress scale."
    )
    source: str = Field(
        description="Sensor, simulator variable or declared observable supplying the value."
    )
    derivation: str = Field(
        description="Measurement or present/past-only derivation used to obtain the value."
    )
    availability: Literal["observed", "unavailable"] = Field(
        description="Explicit missingness; unavailable is never false."
    )

    @model_validator(mode="after")
    def _availability(self):
        if (self.value is None) != (self.availability == "unavailable"):
            raise ValueError("feature value disagrees with availability")
        return self


class PhysicalFrame(Asset):
    """Initial state or an acknowledged post-action observation."""

    control_step: int | None = Field(
        ge=0,
        description="Original action index; null for an explicit initial/reset observation.",
    )
    execution_ref: str | None = Field(
        description="Original tool operation identity; null for initial state."
    )
    tool: str | None = Field(
        description="Fixed physical tool category, not an inferred action intent."
    )
    action_wall_time_s: float | None = Field(
        default=None,
        ge=0,
        description="Measured action RPC acknowledgement time in seconds; null for reset or unrecorded historical timing. Each non-reset frame costs one acknowledged physical step.",
    )
    action: tuple[float, ...] = Field(
        description="Actual acknowledged control vector; empty at reset."
    )
    feature_source: OriginalReference | None = Field(
        description="Exact original measurement dictionary for this frame; null means no recorded physical channels. Values and units remain in that original, not copied into each graph snapshot."
    )
    references: tuple[OriginalReference, ...] = Field(
        min_length=1, description="Exact original observation and action witnesses."
    )
    error: str | None = Field(
        description="Telemetry failure or missing physical interval. Breaks scaffold connectivity; semantic interpretation may use other original observations, but cannot bridge unacknowledged actions."
    )


class GraphEpisode(Asset):
    """One physical receipt, irrespective of how many authors produced its content."""

    receipt_id: RecordId = Field(
        description="Exact settled evaluation receipt identity; cache reuse counts once."
    )
    bundle_id: RecordId = Field(description="Executed complete Harness content.")
    condition_id: RecordId = Field(description="Frozen reset condition identity.")
    version_ids: tuple[RecordId, ...] = Field(
        description="All version events with this content before the graph freeze."
    )
    operation_starts: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Original execution IDs, tool names and input arguments with exact trace references; joined to action witnesses, never inferred from endpoints.",
    )
    operation_exits: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Original tool completion/unknown-action events with exact references; never synthesized into positive transitions.",
    )
    frames: tuple[PhysicalFrame, ...] = Field(
        description="All original initial/post-action frames, without temporal sampling."
    )
    feature_format: Literal["physical_scalars", "robodojo_proprioception"] = Field(
        default="physical_scalars",
        description="Original telemetry encoding selected from the receipt's robot contract: unit-bearing PhysicalFeature dictionaries, or RoboDojo's six native public state vectors projected to scalars only when read. Never inferred from value shapes.",
    )
    feature_names: tuple[str, ...] = Field(
        description="Sorted scalar channel names decoded from this episode's original frame measurements; navigation index only, never evidence that a channel is present in every frame."
    )
    terminal_state_known: bool = Field(
        default=True,
        description="False when an unacknowledged action leaves the final physical state unknown; last acknowledged observations remain historical evidence only.",
    )
    official_success: bool = Field(
        description="Valid evaluator receipt verdict; never a feature threshold."
    )
    exit_type: str = Field(
        description="Runner failure origin or success, preserving unknown physical outcomes."
    )
    usage: Usage = Field(
        description="Actual physical episode usage, not candidate author cost."
    )
    receipt: EvidenceFile = Field(description="Exact immutable evaluator receipt file.")


class GraphTransition(Asset):
    """An observed edge with its contiguous physical action witness."""

    source: RecordId = Field(
        description="Observed partition region before this action."
    )
    target: RecordId = Field(
        description="Observed partition region after this action; may equal source or revisit an earlier region."
    )
    episode_id: RecordId = Field(description="Physical receipt witnessing this edge.")
    frame_index: int = Field(
        ge=1,
        description="Post-action frame index in that episode; predecessor is index minus one.",
    )


class GraphSnapshot(PublishedRecord):
    """Host evidence scaffold; semantic meaning is owned by the shared SemanticGraph."""

    policy: GraphPolicy = Field(
        description="Frozen measurement bins for evidence navigation and the graph analysis deadline; not semantic state definitions."
    )
    episodes: tuple[GraphEpisode, ...] = Field(
        description="Deduplicated physical episodes with full frame/reference ranges."
    )
    nodes: dict[RecordId, dict[str, Any]] = Field(
        description="Only observed feature signatures, indexed by Host-allocated node numbers."
    )
    paths: dict[RecordId, tuple[RecordId | None, ...]] = Field(
        description="Original frame order per receipt; null breaks connectivity at missing telemetry."
    )
    transitions: tuple[GraphTransition, ...] = Field(
        description="Observed contiguous edges, including dwell, loops and regressions."
    )


class SemanticNode(Asset):
    """A meaningful, earned task-progress condition backed by original evidence."""

    meaning: str = Field(
        min_length=1,
        description="A positive, verifiable task-progress condition acquired through actual observation, grounding or action, with its task relevance and distinguishing conditions. No free reset facts, mere absence of success, activity labels or irrelevant failure states. For an unobserved milestone, define what would constitute this earned achievement without claiming it occurred. Never a future repair instruction.",
    )
    role: Literal["milestone", "branch", "goal"] = Field(
        description="All roles require meaningful, non-free task progress. milestone is necessary across valid routes; branch is an optional earned achievement that enables a useful route; goal is fixed-evaluator completion. Necessity is a revisable semantic hypothesis, not a numeric rating."
    )
    predecessors: tuple[str, ...] = Field(
        description="Immediate milestone node IDs whose achievements remain satisfied at this physical state. These links form the task backbone inside this same graph. A milestone also establishes itself; a branch adds no progress. Regressed branches link to the earlier achievements still retained, not the furthest node ever visited."
    )
    references: tuple[OriginalReference, ...] = Field(
        description="Original observations, grounded decisions, measurements or executed code supporting both the achieved condition and its acquisition. Empty only for an explicitly unobserved milestone or goal in the task backbone."
    )
    uncertainty: str = Field(
        description="Unverified aspects of this state interpretation, or an empty string when none are identified."
    )


class StateSpan(Asset):
    """An exhaustive, ordered assignment of original observations to semantic states."""

    first_frame: int = Field(
        ge=0,
        strict=True,
        description="Inclusive original GraphEpisode.frames index; never a resampled frame.",
    )
    last_frame: int = Field(
        ge=0,
        strict=True,
        description="Inclusive final frame with the same interpreted state.",
    )
    status: Literal["node", "background", "unknown"] = Field(
        description="node assigns an earned task-progress condition; background is known to retain no meaningful task progress and needs no graph node; unknown preserves insufficient evidence. All original frames remain covered."
    )
    node_id: str | None = Field(
        description="SemanticGraph.nodes key exactly when status=node; null for background or unknown."
    )
    uncertainty: str = Field(
        description="Reason for uncertain assignment; required for status=unknown. Background may explain non-progress without pretending it is unknown."
    )

    @model_validator(mode="after")
    def _range(self):
        if (
            self.last_frame < self.first_frame
            or (self.status == "node") != (self.node_id is not None)
            or (self.status == "unknown" and not self.uncertainty.strip())
        ):
            raise ValueError("invalid or unexplained semantic state interval")
        return self


class ExecutionSpan(Asset):
    """A real contiguous execution segment, preserving the actual method provenance."""

    episode_id: RecordId = Field(
        description="GraphEpisode receipt identity witnessing this segment."
    )
    first_frame: int = Field(
        ge=0,
        strict=True,
        description="Observed state immediately before the segment's first action.",
    )
    last_frame: int = Field(
        ge=1,
        strict=True,
        description="Post-action frame at the segment's end, strictly after first_frame.",
    )

    @model_validator(mode="after")
    def _range(self):
        if self.last_frame <= self.first_frame:
            raise ValueError("execution interval must include an action")
        return self


class TransitionWitness(ExecutionSpan):
    program_id: str | None = Field(
        description="Key in SemanticGraph.programs for the actual Harness fragment used in this attempt. Null means unresolved attribution, never an invented or deployable plugin."
    )
    bindings: dict[str, str | None] = Field(
        description="This attempt's observed entity for every edge role; null with uncertainty when unidentified. Not a task-ID routing rule."
    )
    outcome: Literal["achieved", "not_achieved", "unknown"] = Field(
        description="Evidence-supported local postcondition outcome, not the official task verdict. Actual final state remains in the episode projection."
    )
    uncertainty: str = Field(
        description="Unresolved outcome, program attribution or role-binding questions. Required when the program or local outcome is unknown."
    )

    @model_validator(mode="after")
    def _attribution(self):
        if (
            self.program_id is None
            or self.outcome == "unknown"
            or None in self.bindings.values()
        ) and not self.uncertainty.strip():
            raise ValueError(
                "unknown transition attribution or outcome needs uncertainty"
            )
        if any(
            not key.strip() or (value is not None and not value.strip())
            for key, value in self.bindings.items()
        ):
            raise ValueError("transition role bindings must be nonempty")
        return self


class UnclassifiedExecution(ExecutionSpan):
    reason: str = Field(
        min_length=1,
        description="Why this recorded interval cannot yet be assigned to a semantic mechanism; preserves evidence without inventing a method.",
    )


class HarnessProgram(Asset):
    """A historical implementation slice, not a claim of standalone executability."""

    bundle: OriginalReference = Field(
        description="Complete archived AgentBundle JSON (empty pointer) supplying the executed W/C/L code. Witnesses must come from this exact executable Bundle."
    )
    pointers: tuple[str, ...] = Field(
        min_length=1,
        description="Sorted unique JSON pointers selecting relevant W/C/L code, instructions, skills and dependencies inside that Bundle. Include coupled context/loop dependencies when necessary; do not copy their code into the graph.",
    )
    mechanism: str = Field(
        min_length=1,
        description="Observed decision/execution mechanism, including Planner evidence use, action selection, feedback and recovery when evidenced. May be realized by C/L without W. Separate recorded behavior from causal attribution to source instructions; not a proposed future modification.",
    )
    uncertainty: str = Field(
        description="Unverified causal attribution, dependency closure, isolation and transfer assumptions. Provenance alone never certifies an independently callable plugin."
    )

    @model_validator(mode="after")
    def _sources(self):
        if self.bundle.pointer or tuple(sorted(set(self.pointers))) != self.pointers:
            raise ValueError(
                "program requires a complete Bundle and sorted unique pointers"
            )
        if any(
            not p.startswith("/")
            or p.split("/")[1:2] not in (["workflows"], ["context"], ["loop"])
            for p in self.pointers
        ):
            raise ValueError("program pointers must select W/C/L assets")
        if any(
            right.startswith(left + "/")
            for left in self.pointers
            for right in self.pointers
            if left != right
        ):
            raise ValueError("program pointers must not duplicate a parent selection")
        return self


class SemanticEdge(Asset):
    """A conditional physical transition, realized by witnessed Harness programs."""

    source: str | None = Field(
        description="Earned task-progress node at the witnessed start, or null for known background before any retained progress; never an unknown start. No fabricated initial node is needed."
    )
    target: str = Field(
        description="Semantic postcondition node; failed witnesses need not reach it."
    )
    roles: dict[str, str] = Field(
        description="Object roles needed to describe this transition, with their task-independent meanings. Each witness supplies its actual bindings; an empty map is allowed when no roles are needed."
    )
    entry_conditions: str = Field(
        min_length=1,
        description="Evidence-supported physical conditions and required observation history at entry, beyond the source progress node. Include free environmental prerequisites here, not as progress nodes. Do not use future outcome labels.",
    )
    exit_conditions: str = Field(
        min_length=1,
        description="Intended acquired progress and continuation-relevant physical conditions. Failed attempts retain their actual exit in episode spans; equal target labels alone do not establish compatibility.",
    )
    explanation: str = Field(
        min_length=1,
        description="Evidence-based interpretation of outcome differences and physical preconditions, with causal uncertainty stated explicitly.",
    )
    references: tuple[OriginalReference, ...] = Field(
        min_length=1,
        description="Exact executed code, invocation, observation or image evidence supporting this semantic grouping.",
    )
    witnesses: tuple[TransitionWitness, ...] = Field(
        description="Assigned attempts with program and role bindings. Empty only for an inherited edge in first-observation analysis; the Host validates this against the previous graph. Host derives endpoints and costs from original intervals.",
    )


class SemanticGraph(Asset):
    """One semantic coordinate system and its admitted episode interpretations."""

    nodes: dict[str, SemanticNode] = Field(
        description="One task graph containing necessary milestone nodes, optional branch nodes and evaluator-defined goals. Establish a task backbone in the initial graph, even before any successful execution, and revise it with all subsequent evidence."
    )
    edges: dict[str, SemanticEdge] = Field(
        description="Conditional transitions with witnessed realizations. One edge may use many programs and one program may realize many edges; freely refine interfaces without inventing attempts."
    )
    programs: dict[str, HarnessProgram] = Field(
        description="Evidence-bound historical Harness fragments referenced by witnesses. Shared once per graph, including W/C/L dependencies; not a separate skill library or runtime registry."
    )
    episodes: dict[RecordId, tuple[StateSpan, ...]] = Field(
        description="Exhaustive frame partitions for every input physical episode, including all historical versions under this same graph."
    )
    unclassified: tuple[UnclassifiedExecution, ...] = Field(
        description="Explicit coverage of action intervals not assigned to semantic edges; no silent loss of attempts."
    )
    matching: GraphSemantics | None = Field(
        default=None,
        description="Structured conditions of this same canonical graph, required for cross-task retrieval. The graph analyst reuses the reference catalog and records scoped facts and uncertainty; topology, witnesses and programs remain owned by nodes/edges. Null means not yet encoded, never permission to fall back to text similarity.",
    )

    @model_validator(mode="after")
    def _structure(self):
        if self.matching is not None and (
            set(self.matching.nodes) != set(self.nodes)
            or set(self.matching.interfaces) != set(self.edges)
        ):
            raise ValueError(
                "matching conditions must cover exactly the canonical nodes and edges"
            )
        milestones = {k for k, n in self.nodes.items() if n.role == "milestone"}
        goals = {k for k, n in self.nodes.items() if n.role == "goal"}
        if not goals:
            raise ValueError("semantic task graph requires an evaluator goal node")
        for key, node in self.nodes.items():
            if (
                not key.strip()
                or len(set(node.predecessors)) != len(node.predecessors)
                or not set(node.predecessors) <= milestones
            ):
                raise ValueError("invalid semantic node or milestone predecessor")
        completed = self.completed_nodes()
        if any(completed[key] - {key} != milestones for key in goals):
            raise ValueError("every goal must complete the entire milestone backbone")
        for key, edge in self.edges.items():
            if (
                not key.strip()
                or (edge.source is not None and edge.source not in self.nodes)
                or edge.target not in self.nodes
            ):
                raise ValueError("semantic edge references an undeclared node")
            if any(
                not role.strip() or not meaning.strip()
                for role, meaning in edge.roles.items()
            ):
                raise ValueError("transition roles must have nonempty meanings")
            for witness in edge.witnesses:
                if (
                    witness.program_id is not None
                    and witness.program_id not in self.programs
                ):
                    raise ValueError(
                        "transition witness references an undeclared program"
                    )
                if set(witness.bindings) != set(edge.roles):
                    raise ValueError(
                        "transition witness must bind exactly its edge roles"
                    )
        used_programs = {
            w.program_id
            for e in self.edges.values()
            for w in e.witnesses
            if w.program_id is not None
        }
        if self.matching is not None:
            self.matching.validate_contexts(
                {key: (edge.source, edge.target) for key, edge in self.edges.items()}
            )
        if set(self.programs) != used_programs or any(
            not key.strip() for key in self.programs
        ):
            raise ValueError("graph programs require actual execution witnesses")
        for spans in self.episodes.values():
            cursor = 0
            for span in spans:
                if span.first_frame != cursor or (
                    span.node_id is not None and span.node_id not in self.nodes
                ):
                    raise ValueError(
                        "semantic state spans must tile original frames in order"
                    )
                cursor = span.last_frame + 1
        return self

    def original_references(self) -> tuple[OriginalReference, ...]:
        """One inventory of semantic evidence and implementation originals."""
        return tuple(
            dict.fromkeys(
                [
                    *(
                        ref
                        for item in (*self.nodes.values(), *self.edges.values())
                        for ref in item.references
                    ),
                    *(program.bundle for program in self.programs.values()),
                    *(
                        program.bundle.model_copy(update={"pointer": pointer})
                        for program in self.programs.values()
                        for pointer in program.pointers
                    ),
                ]
            )
        )

    def completed_nodes(self) -> dict[str, frozenset[str]]:
        """Derive retained backbone achievements from graph links, without point labels.

        Mechanism edges preserve loops and failed attempts. Only milestone
        predecessor links are acyclic; optional branches add no achievements.
        Alternative goal nodes each contribute one terminal achievement.
        """
        completed: dict[str, frozenset[str]] = {}
        active: set[str] = set()

        def visit(key):
            if key in active:
                raise ValueError("milestone backbone must be acyclic")
            if key not in completed:
                active.add(key)
                node = self.nodes[key]
                result = set().union(*(visit(p) for p in node.predecessors))
                if node.role != "branch":
                    result.add(key)
                completed[key] = frozenset(result)
                active.remove(key)
            return completed[key]

        for key in self.nodes:
            visit(key)
        return completed


class GraphAnalysisRequest(PublishedRecord):
    """Frozen inputs to one shared native search or final archive analysis."""

    purpose: Literal["search", "final_selection"] = Field(
        default="search",
        exclude_if=lambda value: value == "search",
        description="Final selection accounts for the settled archive without creating authors or physical trials, honoring the protocol's history projection rule.",
    )

    retrospective_receipts: tuple[RecordId, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Explicit physical receipts selected for an offline retrospective progress map. Original execution settings remain distinct; this grants no current-execution scores or physical execution authority.",
    )
    previous_analysis: EvidenceFile | None = Field(
        description="Prior round completion supplying the structured graph/evidence handoff to this round's independent native session; null only in round zero."
    )
    round_index: int = Field(
        ge=0,
        strict=True,
        description="Number of mutation rounds preceding these frozen inputs.",
    )
    protocol: EvolutionProtocol = Field(
        description="Host-private frozen protocol for request identity and validation; the analyst receives conditions but no search scheduling policy."
    )
    graph: EvidenceFile = Field(description="Immutable serialized GraphSnapshot.")
    history: EvolutionHistory = Field(
        description="Shared historical code, diagnostics and evidence originals."
    )
    evidence: tuple[EvaluationRecord, ...] = Field(
        min_length=1,
        description="All settled original evaluation receipts available before proposal.",
    )
    versions: tuple[VersionReference, ...] = Field(
        min_length=1,
        description="Eligible historical complete Harness versions, in stable order.",
    )
    current_round_versions: tuple[RecordId, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description="Host-frozen version IDs from the most recent published mutation round, including every sibling. Derived from proposal provenance, never version names; empty in historical requests without this provenance.",
    )
    policy: dict[str, Any] = Field(
        description="Exact native AuthorPolicy, independent of the online Planner."
    )

    @property
    def needs_forecast(self) -> bool:
        return (
            self.purpose == "search"
            and self.protocol.width_mode == "dynamic"
            and self.protocol.max_authors_per_round > 1
        )

    @model_validator(mode="after")
    def _mechanism_inputs(self):
        if len(set(self.current_round_versions)) != len(
            self.current_round_versions
        ) or not set(self.current_round_versions).issubset(
            v.version_id for v in self.versions
        ):
            raise ValueError("current-round versions must be unique eligible versions")
        if self.protocol.evolution_mode != "topology":
            raise ValueError(
                "trajectory reflection does not admit graph analysis requests"
            )
        if self.retrospective_receipts is not None:
            records = {record.record_id: record for record in self.evidence}
            if len(records) != len(self.evidence) or len(
                set(self.retrospective_receipts)
            ) != len(self.retrospective_receipts):
                raise ValueError("retrospective receipts must be unique")
            cells = set()
            bundles = {version.bundle_id for version in self.versions}
            for identity in self.retrospective_receipts:
                record = records.get(identity)
                if record is None:
                    raise ValueError("retrospective receipt is absent from evidence")
                request = record.request
                key = (request.bundle_id, request.instance.record_id)
                if (
                    request.split != "dev"
                    or request.manifest_id != self.protocol.manifest_id
                    or request.instance not in self.protocol.dev
                    or request.bundle_id not in bundles
                    or key in cells
                ):
                    raise ValueError(
                        "retrospective receipt differs from its declared task cell"
                    )
                cells.add(key)
        return self


class ProgressOutcome(Asset):
    """One possible all-condition terminal mean for a single complete candidate."""

    progress: float = Field(
        ge=0,
        le=100,
        strict=True,
        allow_inf_nan=False,
        description="Arithmetic mean terminal graph progress across the selected parent's full dev and already measured feedback validation conditions, equally weighted, after one independent author modifies one complete candidate with the same full historical graph evidence and frozen budgets. 0–100 on the submitted graph's scale; includes retained successes and regressions, never a per-seed ensemble or peak progress.",
    )
    probability: float = Field(
        gt=0,
        le=1,
        strict=True,
        allow_inf_nan=False,
        description="Prospective probability of this candidate-level mean outcome. Positive masses across the strictly increasing outcomes sum to one; subjective prediction, not official breakthrough probability or measured calibration.",
    )


class ProgressForecast(Asset):
    """An uncalibrated one-candidate outcome distribution, separate from measurements."""

    outcomes: tuple[ProgressOutcome, ...] = Field(
        min_length=1,
        description="Distribution of one complete child candidate's all-condition mean terminal graph progress after one independent author. Outcome progress values are strictly increasing without duplicates; positive probabilities sum to one within 1e-9. Not the best of multiple authors or per-seed conditional forecasts.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        strict=True,
        allow_inf_nan=False,
        description="Analyst confidence in the evidence supporting this distribution, 0–1, justified by graph witnesses and prior forecast errors. Audit only: not a utility discount, variance parameter, official breakthrough probability or measured calibration.",
    )
    rationale: str = Field(
        min_length=1,
        description="Evidence and uncertainty supporting regression, retention and improvement probabilities, including the distribution's tails. Cite current graph interfaces, actual continuations, original calls/media and failed prior repairs as relevant; explain achievable outcomes within one author budget without assigning modifications or author count.",
    )

    @model_validator(mode="after")
    def _distribution(self):
        if any(
            left.progress >= right.progress
            for left, right in zip(self.outcomes, self.outcomes[1:])
        ):
            raise ValueError("forecast outcome progress must be strictly increasing")
        if abs(fsum(outcome.probability for outcome in self.outcomes) - 1.0) > 1e-9:
            raise ValueError("forecast outcome probabilities must sum to one")
        return self

    @property
    def expected_progress(self) -> float:
        """Derived all-condition expected terminal progress, in percentage points."""
        return fsum(outcome.progress * outcome.probability for outcome in self.outcomes)


class GraphAnalysisResult(Asset):
    """Complete next graph and one-author forecast; Host alone selects search width."""

    graph: SemanticGraph = Field(
        description="Shared semantic graph replacing the prior interpretation, with all history reprojected."
    )
    changes: str = Field(
        min_length=1,
        description="Reasons and evidence for node/edge splits, merges and revised interpretations; no candidate prescriptions.",
    )
    forecast: ProgressForecast | None = Field(
        default=None,
        description="Prospective candidate distribution only when dynamic width can select multiple authors. Fixed-width and final analyses omit it. Historical forecasts remain readable.",
    )


class GraphAnalysisReceipt(PublishedRecord):
    """Native completion with exact analysis request, output and accounting."""

    request_id: RecordId = Field(
        description="Exact GraphAnalysisRequest answered by this session."
    )
    analysis: GraphAnalysisResult = Field(
        description="Complete proposed semantic graph, reasons for refinement and prospective one-author forecast."
    )
    session_id: str = Field(
        min_length=1,
        description="Native session identity preserved through corrections and recovery.",
    )
    usage: Usage = Field(
        description="Shared analysis work charged exactly once per round."
    )
    artifacts: tuple[EvidenceFile, ...] = Field(
        min_length=1, description="Complete native/input/output completion evidence."
    )


class RoundDecision(PublishedRecord):
    """Deterministic Host bridge from the shared semantic map to independent search."""

    request_id: RecordId = Field(description="Exact frozen graph analysis request.")
    analysis: EvidenceFile = Field(
        description="Verified native semantic refinement receipt."
    )
    graph: EvidenceFile = Field(
        description="Complete frozen SemanticGraph shared by all authors."
    )
    progress_map: EvidenceFile = Field(
        description="Host-derived versions × conditions × semantic progress, all using this graph."
    )
    references: tuple[VersionReference, ...] = Field(
        min_length=1,
        max_length=1,
        description="Exactly the complete parent Bundle used to start the candidate. All historical programs remain readable through graph witnesses and their source Bundles independently of this starting reference.",
    )
    parent: VersionReference = Field(
        description="Complete starting Harness selected under the protocol's search-parent rule."
    )
    width: int = Field(
        ge=1,
        strict=True,
        description="Actual independent author count. Fixed mode uses the protocol maximum; dynamic mode takes the smallest finite-horizon Bayesian maximizer up to that maximum.",
    )
    candidate_cost: SearchCost = Field(
        description="Historical complete rollout-group dollars; missing dollars remain null. Author price is fixed separately in protocol.search.",
    )
    alternatives: tuple[dict[str, float], ...] = Field(
        description="Dynamic allocation only: per integer K, success probability and finite-horizon value. Empty when width is fixed or capped at one; these are planning estimates, not measurements.",
    )
    statistics: dict[str, float | None] = Field(
        description="Forecast completion probability, Beta prior, remaining rounds, planning prices and parent graph statistics. Predictions are not physical measurements."
    )
    parent_selection_audit: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Current-round anchor, unrestricted archive winner and progress rollback gate outcome. Does not alter the immutable progress map.",
    )
    reason: str = Field(
        min_length=1,
        description="Exact deterministic parent selection and width calculation.",
    )
    search_review: dict[str, Any] = Field(
        default_factory=dict,
        description="Same-current-graph mutation goal vectors, observed regressions/tradeoffs and stagnation diagnostics. Historical decisions may lack these observations; they never change fixed search budgets or establish reliability.",
    )

    @model_validator(mode="after")
    def _references(self) -> RoundDecision:
        if self.references != (self.parent,):
            raise ValueError("starting reference must be exactly the parent")
        return self


class GraphRound(Asset):
    """A round persists before author proposals exist, enabling exact recovery."""

    request: GraphAnalysisRequest = Field(
        description="Immutable analysis inputs; all graph originals precede this freeze."
    )
    decision: EvidenceFile | None = Field(
        description="Immutable RoundDecision, null until analysis and Host selection finish."
    )
    outcome: EvidenceFile | None = Field(
        description="Settled independent author outcomes and actual charged work; subsequent analysis refines the graph."
    )
    construction_wall_time_s: float = Field(
        ge=0,
        description="Measured Host graph construction cost charged once, independently of analysis.",
    )


class FinalProjection(Asset):
    """Recoverable final archive analysis, outside the mutation-round budget."""

    request: GraphAnalysisRequest
    construction_wall_time_s: float = Field(ge=0)
    analysis: EvidenceFile | None = None
    graph: EvidenceFile | None = None
    progress_map: EvidenceFile | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.request.purpose != "final_selection":
            raise ValueError("final projection requires a final-selection request")
        if (
            len(
                {
                    value is None
                    for value in (self.analysis, self.graph, self.progress_map)
                }
            )
            != 1
        ):
            raise ValueError("final projection outputs must settle together")
        return self


class Proposal(PublishedRecord):
    """Frozen evidence for one mutation or post-success consolidation session."""

    purpose: Literal["mutation", "success_consolidation", "no_topology"] = Field(
        default="mutation",
        exclude_if=lambda value: value == "mutation",
        description="Trials edit only task-family. After a fixed-evaluator success, consolidation may extract global W/C/L and organize the complete three-layer publication without inheriting the winning score.",
    )

    graph: EvidenceFile | None = Field(
        default=None,
        description="Frozen shared SemanticGraph for mutation; null for success consolidation, which uses winning episode originals.",
    )
    unmeasured_parent: EvidenceFile | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Explicit complete seed Bundle for retrospective mutation when this latest Harness has no exact measured receipt. This supplies source identity only, never a physical witness or score.",
    )
    decision: EvidenceFile | None = Field(
        default=None,
        description="Host-only mutation scheduling provenance; null for success consolidation.",
    )
    execution_graph: EvidenceFile | None = Field(
        default=None,
        description="Complete underlying mutation evidence graph; null for success consolidation.",
    )
    progress_map: EvidenceFile | None = Field(
        default=None,
        description="Shared Host mutation projection; null for success consolidation.",
    )
    task_protocol: EvolutionProtocol = Field(
        description="Complete frozen author/evaluation protocol. A published execution-only revision may follow the graph decision's original protocol; its parent, references, width and historical map remain unchanged. Host exposes the current evaluation objective, tools and budgets to the author; search allocation remains Host-owned."
    )

    references: tuple[VersionReference, ...] = Field(
        min_length=1,
        max_length=1,
        description="Exactly the selected complete parent. Historical implementations are accessed through graph programs and their original source Bundles; there is no version donor selection.",
    )
    parent: VersionReference = Field(
        description="Complete Harness copied into candidate/ before this independent author starts."
    )
    round_index: int = Field(
        ge=0,
        strict=True,
        description="Zero-based evolution round shared by concurrently launched authors.",
    )
    author_index: int = Field(
        ge=0,
        strict=True,
        description="Stable independent author slot within this round.",
    )
    scopes: tuple[Scope, ...] = Field(
        description="Component write authority shared by every branch, equal to the task protocol's mutation_scopes."
    )
    history: EvolutionHistory = Field(
        description="Immutable shared summary and original-evidence index available at mutation creation."
    )
    evidence: tuple[EvaluationRecord, ...] = Field(
        min_length=1,
        description="Complete available development receipts, including historical failures.",
    )

    @model_validator(mode="after")
    def _inputs(self) -> Proposal:
        graph_inputs = (
            self.graph,
            self.decision,
            self.execution_graph,
            self.progress_map,
        )
        if self.purpose == "mutation" and any(item is None for item in graph_inputs):
            raise ValueError("mutation requires complete graph analysis inputs")
        if self.task_protocol.evolution_mode == "no_topology":
            if self.purpose not in {"no_topology", "success_consolidation"} or any(
                item is not None for item in graph_inputs
            ):
                raise ValueError("no-topology author cannot receive graph inputs")
        if (
            self.purpose == "no_topology"
            and self.task_protocol.evolution_mode != "no_topology"
        ):
            raise ValueError("no-topology proposal requires its own protocol")
        if self.purpose == "success_consolidation":
            if not self.task_protocol.success_consolidation:
                raise ValueError("success consolidation is disabled by this protocol")
            if any(item is not None for item in graph_inputs):
                raise ValueError(
                    "success consolidation uses measured episodes, not a search allocation"
                )
            if not any(
                r.official_success is True
                and r.request.bundle_id == self.parent.bundle_id
                for r in self.evidence
            ):
                raise ValueError(
                    "success consolidation requires exact-parent evaluator success"
                )
        if self.scopes != self.task_protocol.mutation_scopes:
            raise ValueError("author scopes differ from the task protocol")
        if self.purpose == "success_consolidation" and self.scopes != ("W", "C", "L"):
            raise ValueError(
                "restricted searches retain measured winners without consolidation"
            )
        # Infrastructure attempts and their retries share one evaluation request.
        if len({r.record_id for r in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evaluation evidence")
        # Historical budgets remain visible for diagnosis. The archive validates
        # their operator-published lineage; they are not paired measurements.
        available = {r.request.bundle_id for r in self.evidence}
        if self.unmeasured_parent is not None:
            if self.purpose != "mutation":
                raise ValueError("unmeasured seed is only available for mutation")
            self.unmeasured_parent.verify()
            seed = AgentBundle.model_validate_json(
                Path(self.unmeasured_parent.path).read_text()
            )
            if seed.record_id != self.parent.bundle_id:
                raise ValueError("unmeasured seed differs from the selected parent")
            available.add(seed.record_id)
        if self.references != (self.parent,):
            raise ValueError("starting reference must be exactly the parent")
        if not {r.bundle_id for r in self.references} <= available:
            raise ValueError("reference Harness lacks development evidence")
        return self


class Candidate(Asset):
    """One complete Harness owned by an independent author session."""

    branch_id: BranchId = Field(
        description="Workspace-local branch identifier; never a filesystem path.",
    )
    bundle: AgentBundle = Field(
        description="Complete admitted W/C/L executable content."
    )
    summary: str = Field(
        min_length=1,
        description="This branch's diagnosis, actual modification and validation.",
    )
    experience: str = Field(
        default="",
        description="Optional author-written candidate/SKILL.md with additional reusable lessons or unresolved questions. Empty means no additional note; diagnosis/change/checks remain in summary. Not an online C asset or a verified physical result; excluded from Bundle identity.",
    )


class AuthoredMutation(PublishedRecord):
    """One native session receipt containing exactly one admitted Harness."""

    proposal_id: RecordId = Field(
        description="Exact Proposal request that this native author receipt answers."
    )
    candidates: tuple[Candidate, ...] = Field(
        min_length=1,
        max_length=1,
        description="Exactly one complete candidate frozen by final Host admission.",
    )
    summary: str = Field(
        min_length=1, description="Diagnosis and chosen repair hypothesis."
    )
    session_id: str = Field(
        min_length=1, description="Native session dedicated to this Harness mutation."
    )
    usage: Usage = Field(
        description="Session and admission work, charged once for the mutation."
    )
    artifacts: tuple[EvidenceFile, ...] = Field(
        min_length=1, description="Complete native, output and validation originals."
    )


class BundleVersion(PublishedRecord):
    """Immutable version event; multiple events may name the same content."""

    bundle_id: RecordId = Field(
        description="Executable content, also the result matrix row identity."
    )
    proposal_id: str | None = Field(
        description="Producing mutation; null for the initial version."
    )
    branch_id: BranchId | None = Field(
        description="Branch within the producing mutation; null for the initial version."
    )

    @model_validator(mode="after")
    def _origin(self) -> BundleVersion:
        if (self.proposal_id is None) != (self.branch_id is None):
            raise ValueError("version origin requires both mutation and branch")
        return self


class AttemptFailure(Asset):
    """Workspace-relative reference to the latest orchestration failure."""

    path: str = Field(
        min_length=1, description="Workspace-relative immutable failure record."
    )

    @model_validator(mode="after")
    def _relative_path(self) -> AttemptFailure:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("attempt failure path must stay inside the workspace")
        return self


class AttemptState(Asset):
    """Durable lifecycle of one exact author or evaluation request."""

    result_directory: str | None = Field(
        default=None,
        description="Absolute location of a completed receipt stored outside the standard attempt tree, including imported evidence and run-owned round layouts; active locations belong to the workspace.",
    )
    kind: Literal["author", "graph_analysis", "evaluation"] = Field(
        description="Independent diagnosis-and-mutation session or one exact rollout condition."
    )
    key: str = Field(
        min_length=1,
        description="Stable mutation identity or evaluation identity plus infrastructure retry ordinal.",
    )
    request: dict[str, Any] = Field(
        description="Exact serialized Proposal or EvaluationRequest, selected by kind."
    )
    status: Literal[
        "prepared", "running", "completed", "failed", "cancelled", "timed_out"
    ] = Field(
        description="Host-owned durable attempt lifecycle, independent of official task success."
    )
    launches: int = Field(
        ge=0,
        strict=True,
        description="Actual launches registered by this adapter, including explicit recovery.",
    )
    deadline_at: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Absolute Host deadline for capped offline work; null for uncapped offline work and evaluation attempts.",
    )
    result_id: RecordId | None = Field(
        default=None,
        description="Published completed receipt ID; absent until durable publication.",
    )
    last_failure: AttemptFailure | None = Field(
        default=None,
        description="Latest orchestration error; previous originals remain archived.",
    )

    @model_validator(mode="after")
    def _lifecycle(self) -> AttemptState:
        if self.result_directory is not None and (
            self.status != "completed" or not Path(self.result_directory).is_absolute()
        ):
            raise ValueError(
                "imported result directory requires a completed absolute original"
            )
        request_type = {
            "author": Proposal,
            "graph_analysis": GraphAnalysisRequest,
            "evaluation": EvaluationRequest,
        }[self.kind]
        if self.kind == "author":
            archived_proposal(self.request)
        else:
            request_type.model_validate(self.request)
        if self.status in {"running", "completed"} and self.launches == 0:
            raise ValueError(f"{self.status} attempt requires a registered launch")
        if self.kind == "evaluation" and self.deadline_at is not None:
            raise ValueError("evaluation attempts use their frozen episode deadline")
        if (self.status == "completed") != (self.result_id is not None):
            raise ValueError("only a completed attempt may carry a result ID")
        return self


class BudgetRevision(Asset):
    """One explicit budget change preserving all other protocol fields."""

    previous: EvolutionProtocol = Field(
        description="Original protocol retained by immutable graph requests and receipts."
    )
    current: EvolutionProtocol = Field(
        description="Replacement protocol; increase or remove graph analysis wall time, remove author or online wall time, increase the finite search round limit, or change max_authors_per_round. Exactly one field changes; offline changes retain execution_id and existing frozen requests."
    )

    @model_validator(mode="after")
    def _change(self) -> BudgetRevision:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.previous.max_authors_per_round,
                self.current.max_authors_per_round,
            )
        ):
            raise ValueError("budget revision requires positive integer author caps")
        previous = self.previous.model_dump(mode="json")
        current = self.current.model_dump(mode="json")
        previous.pop("record_id")
        current.pop("record_id")
        execution_before = previous.pop("execution_id")
        execution_after = current.pop("execution_id")
        graph_before = previous["graph"].pop("analysis_wall_time_s")
        graph_after = current["graph"].pop("analysis_wall_time_s")
        author_before = previous.pop("author_wall_time_s")
        author_after = current.pop("author_wall_time_s")
        online_before = previous["execution"].pop("wall_time_s")
        online_after = current["execution"].pop("wall_time_s")
        rounds_before = previous.pop("max_rounds")
        rounds_after = current.pop("max_rounds")
        authors_before = previous.pop("max_authors_per_round")
        authors_after = current.pop("max_authors_per_round")
        authors_unchanged = authors_before == authors_after
        graph_increased = (
            authors_unchanged
            and author_before == author_after
            and graph_before is not None
            and (graph_after is None or graph_after > graph_before)
            and online_before == online_after
            and rounds_before == rounds_after
        )
        online_removed = (
            authors_unchanged
            and author_before == author_after
            and graph_after == graph_before
            and online_before is not None
            and online_after is None
            and rounds_before == rounds_after
        )
        rounds_increased = (
            authors_unchanged
            and author_before == author_after
            and graph_after == graph_before
            and online_before == online_after
            and type(rounds_after) is int
            and rounds_after > rounds_before
        )
        authors_changed = (
            author_before == author_after
            and type(authors_before) is int
            and type(authors_after) is int
            and authors_after > 0
            and authors_after != authors_before
            and graph_after == graph_before
            and online_before == online_after
            and rounds_before == rounds_after
        )
        author_removed = (
            authors_unchanged
            and author_before is not None
            and author_after is None
            and graph_before == graph_after
            and online_before == online_after
            and rounds_before == rounds_after
        )
        if (
            previous != current
            or (not online_removed and execution_before != execution_after)
            or (online_removed and execution_before == execution_after)
            or self.previous.record_id == self.current.record_id
            or not (
                graph_increased
                or author_removed
                or online_removed
                or rounds_increased
                or authors_changed
            )
        ):
            raise ValueError(
                "budget revision may only increase or remove graph wall time, remove author or online wall time, increase max_rounds, or change max_authors_per_round"
            )
        return self


class ExecutionRevision(Asset):
    """A published non-budget execution change; original evidence stays intact."""

    previous: EvolutionProtocol = Field(description="Original published protocol.")
    current: EvolutionProtocol = Field(
        description="Replacement with the actual active execution runtime and model sampling; episode budgets, task conditions and search/author policy remain fixed. Historical sampling is retained in original requests."
    )
    reason: str = Field(
        min_length=1,
        description="Operator explanation of the source, dependency, tool or environment change; this is provenance, not an approval gate.",
    )

    @model_validator(mode="after")
    def _change(self) -> ExecutionRevision:
        previous = self.previous.model_dump(mode="json")
        current = self.current.model_dump(mode="json")
        previous.pop("record_id")
        current.pop("record_id")
        previous.pop("execution_id")
        current.pop("execution_id")
        previous.pop("execution_runtime", None)
        current.pop("execution_runtime", None)
        previous["execution"].pop("model_sampling")
        current["execution"].pop("model_sampling")
        if (
            previous != current
            or self.current.execution_runtime is None
            or (
                self.previous.execution_runtime is not None
                and self.current.execution_runtime.model_dump(exclude={"revision_id"})
                == self.previous.execution_runtime.model_dump(exclude={"revision_id"})
            )
            or self.previous.record_id == self.current.record_id
            or self.previous.execution_id == self.current.execution_id
            or not self.reason.strip()
        ):
            raise ValueError(
                "execution revision must change only the published execution runtime"
            )
        return self


def execution_handoff_allowed(
    previous: EvolutionProtocol,
    current: EvolutionProtocol,
    revisions: tuple[ExecutionRevision | BudgetRevision, ...],
) -> bool:
    """Carry a frozen decision through runtime repairs or offline cap removal.

    Other budget changes do not authorize author input handoff. Frozen widths
    and original proposals are never rewritten by this provenance check.
    """
    if previous == current:
        return True
    before = previous.model_dump(
        exclude={
            "record_id",
            "execution_id",
            "execution_runtime",
            "max_authors_per_round",
            "author_wall_time_s",
        }
    )
    after = current.model_dump(
        exclude={
            "record_id",
            "execution_id",
            "execution_runtime",
            "max_authors_per_round",
            "author_wall_time_s",
        }
    )
    before["execution"].pop("model_sampling")
    after["execution"].pop("model_sampling")
    graph_before = before["graph"].pop("analysis_wall_time_s")
    graph_after = after["graph"].pop("analysis_wall_time_s")
    if graph_after != graph_before and not (
        graph_before is not None and graph_after is None
    ):
        return False
    if current.author_wall_time_s != previous.author_wall_time_s and not (
        previous.author_wall_time_s is not None and current.author_wall_time_s is None
    ):
        return False
    if before != after:
        return False
    outgoing = {
        revision.previous.record_id: revision.current
        for revision in revisions
        if isinstance(revision, ExecutionRevision)
        or revision.current.max_authors_per_round
        != revision.previous.max_authors_per_round
        or (
            revision.previous.graph.analysis_wall_time_s is not None
            and revision.current.graph.analysis_wall_time_s is None
        )
        or (
            revision.previous.author_wall_time_s is not None
            and revision.current.author_wall_time_s is None
        )
    }
    cursor = previous.record_id
    for _ in revisions:
        replacement = outgoing.get(cursor)
        if replacement is None:
            return False
        if replacement == current:
            return True
        cursor = replacement.record_id
    return False


class EvolutionState(Asset):
    """Durable archive with frozen proposals and exact attempt identities."""

    rounds: dict[int, GraphRound] = Field(
        default_factory=dict,
        description="Durable graph stages, including analysis completed before author creation.",
    )
    final_projections: dict[int, FinalProjection] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Final archive analyses in creation order, retaining earlier protocol revisions; absent in historical archives and H0-only evaluations.",
    )
    baseline: RecordId = Field(description="Initial executable content.")
    protocol: RecordId = Field(
        description="Frozen development conditions and termination contract."
    )
    budget_revisions: tuple[BudgetRevision, ...] = Field(
        default=(),
        description="Operator-published budget revisions. Together with execution_revisions they form one continuous protocol history ending at protocol; historical requests retain their original budgets and finite round limits. No consumed work is reset and no historical request is replayed.",
    )
    execution_revisions: tuple[ExecutionRevision, ...] = Field(
        default=(),
        description="Explicit non-budget execution changes. Together with budget_revisions these form one continuous protocol history ending at protocol. Historical Bundles, receipts, outcomes and native sessions are retained, with no automatic replay.",
    )
    versions: dict[RecordId, BundleVersion] = Field(
        description="Immutable version events indexed by their published record IDs."
    )
    order: tuple[RecordId, ...] = Field(
        description="Stable version creation order, independent of worker completion."
    )
    proposals: dict[
        str, Annotated[Proposal, BeforeValidator(archived_proposal_value)]
    ] = Field(
        description="Frozen author inputs; sole authority for which references were available, without claiming actual reuse or inherited ability."
    )
    attempts: dict[str, AttemptState] = Field(
        description="Durable native and rollout attempt lifecycle indexed by stable readable paths."
    )
    completed: bool = Field(
        strict=True, description="Whether a normal terminal report was published."
    )
    stop_reason: Literal["target_reached", "round_limit"] | None = Field(
        description="Normal terminal cause, otherwise null."
    )

    @model_validator(mode="after")
    def _archive(self) -> EvolutionState:
        protocols = {self.protocol}
        executions = set()
        revisions = (*self.budget_revisions, *self.execution_revisions)
        published_protocols = {}
        known = [
            *(
                protocol
                for revision in revisions
                for protocol in (revision.previous, revision.current)
            ),
            *(proposal.task_protocol for proposal in self.proposals.values()),
            *(record.request.protocol for record in self.rounds.values()),
            *(record.request.protocol for record in self.final_projections.values()),
        ]
        execution_values = {}
        condition_values = {}
        manifests = {}
        for protocol in known:
            existing = published_protocols.setdefault(protocol.record_id, protocol)
            if existing != protocol:
                raise ValueError(
                    "protocol record ID was reused for different actual inputs"
                )
            if (
                execution_values.setdefault(protocol.execution_id, protocol.execution)
                != protocol.execution
            ):
                raise ValueError(
                    "execution record ID was reused for different actual conditions"
                )
            if (
                manifests.setdefault(protocol.manifest_id, protocol.instances)
                != protocol.instances
            ):
                raise ValueError(
                    "condition inventory record ID was reused for different values"
                )
            for condition in protocol.instances:
                if (
                    condition_values.setdefault(condition.record_id, condition)
                    != condition
                ):
                    raise ValueError(
                        "condition record ID was reused for a different reset"
                    )
        incoming = {revision.current.record_id: revision for revision in revisions}
        if len(incoming) != len(revisions) or len(
            {revision.previous.record_id for revision in revisions}
        ) != len(revisions):
            raise ValueError("protocol revision history must not branch")
        cursor = self.protocol
        visited = set()
        while cursor in incoming and cursor not in visited:
            visited.add(cursor)
            revision = incoming[cursor]
            protocols.add(revision.previous.record_id)
            executions.update(
                (
                    revision.previous.execution_id,
                    revision.current.execution_id,
                )
            )
            cursor = revision.previous.record_id
        if len(visited) != len(revisions) or cursor in visited:
            raise ValueError("protocol revision chain must end at frozen protocol")
        if len(set(self.order)) != len(self.order) or set(self.order) != set(
            self.versions
        ):
            raise ValueError("version order must name every version exactly once")
        request_values = {}
        for identity, attempt in self.attempts.items():
            path = Path(identity)
            if (
                identity != attempt.key
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
            ):
                raise ValueError(
                    "attempt key must equal its stable relative path and exact request identity"
                )
            request = attempt.request
            if request_values.setdefault(request["record_id"], request) != request:
                raise ValueError(
                    "request record ID was reused for different actual inputs"
                )
            if attempt.kind == "evaluation":
                actual = EvaluationRequest.model_validate(request)
                if actual.execution in execution_values and not same_execution_settings(
                    actual.conditions, execution_values[actual.execution]
                ):
                    raise ValueError(
                        "evaluation receipt conditions differ from its published execution"
                    )
        roots = []
        for identity in self.order:
            version = self.versions[identity]
            if identity != version.record_id:
                raise ValueError("version key differs from its published record ID")
            if version.proposal_id is None:
                roots.append(version)
            else:
                proposal = self.proposals.get(version.proposal_id)
                if proposal is None:
                    raise ValueError("version has no producing proposal")
                previous = self.order[: self.order.index(identity)]
                if any(
                    ref not in previous
                    for ref in (r.version_id for r in proposal.references)
                ):
                    raise ValueError(
                        "reference versions must precede the authored version"
                    )
        if len(roots) > 1 or any(root.bundle_id != self.baseline for root in roots):
            raise ValueError("invalid initial version")
        for key, proposal in self.proposals.items():
            if key != proposal.record_id:
                raise ValueError("proposal identity changed")
            permitted = executions | {proposal.task_protocol.execution_id}
            if any(r.request.execution not in permitted for r in proposal.evidence):
                raise ValueError(
                    "proposal evidence has an unpublished execution budget"
                )
            for ref, bundle in [
                (r.version_id, r.bundle_id) for r in proposal.references
            ]:
                if ref not in self.versions or self.versions[ref].bundle_id != bundle:
                    raise ValueError(
                        "proposal reference version differs from its bundle"
                    )
        rounds: dict[int, list[Proposal]] = {}
        for proposal in self.proposals.values():
            rounds.setdefault(proposal.round_index, []).append(proposal)
        if sorted(rounds) != list(range(len(rounds))):
            raise ValueError("evolution rounds must be consecutive from zero")
        for proposals in rounds.values():
            if sorted(p.author_index for p in proposals) != list(range(len(proposals))):
                raise ValueError(
                    "author slots must be unique and consecutive from zero"
                )
            shared = [
                p.model_dump(
                    exclude={
                        "record_id",
                        "author_index",
                    }
                )
                for p in proposals
            ]
            if any(value != shared[0] for value in shared[1:]):
                raise ValueError("concurrent authors must share frozen round inputs")
        if sorted(self.rounds) != list(range(len(self.rounds))):
            raise ValueError("graph rounds must be consecutive from zero")
        for index, record in self.rounds.items():
            if (
                record.request.round_index != index
                or record.request.protocol.record_id not in protocols
            ):
                raise ValueError("graph round differs from its frozen task protocol")
            if (index == 0) != (record.request.previous_analysis is None):
                raise ValueError(
                    "graph round must retain the previous structured analysis handoff"
                )
            for proposal in rounds.get(index, ()):
                if (
                    record.decision != proposal.decision
                    or record.request.graph != proposal.execution_graph
                    or record.request.history != proposal.history
                    or not execution_handoff_allowed(
                        record.request.protocol,
                        proposal.task_protocol,
                        (*self.execution_revisions, *self.budget_revisions),
                    )
                ):
                    raise ValueError("author differs from its frozen graph round")
            if record.outcome is not None and record.decision is None:
                raise ValueError("settled graph round requires a decision")
        if self.completed != (self.stop_reason is not None):
            raise ValueError("terminal state requires an explicit stop reason")
        if sorted(self.final_projections) != list(range(len(self.final_projections))):
            raise ValueError("final projections must retain their creation order")
        if len(
            {
                stage.request.protocol.record_id
                for stage in self.final_projections.values()
            }
        ) != len(self.final_projections):
            raise ValueError("one protocol has at most one frozen final projection")
        for stage in self.final_projections.values():
            request = stage.request
            if (
                request.protocol.record_id not in protocols
                or request.round_index > len(self.rounds)
                or any(
                    record.outcome is None
                    for index, record in self.rounds.items()
                    if index < request.round_index
                )
                or tuple(v.version_id for v in request.versions)
                != self.order[: len(request.versions)]
                or any(
                    self.versions[v.version_id].bundle_id != v.bundle_id
                    for v in request.versions
                )
            ):
                raise ValueError(
                    "final projection must cover the settled complete archive"
                )
            if request.protocol.record_id == self.protocol and (
                request.round_index != len(self.rounds)
                or len(request.versions) != len(self.order)
            ):
                raise ValueError(
                    "current final projection must cover the complete archive"
                )
            if (
                self.completed
                and request.protocol.record_id == self.protocol
                and stage.progress_map is None
            ):
                raise ValueError("terminal state requires settled final projection")
        return self
