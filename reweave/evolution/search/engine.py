# Modified for the ReWeave namespace.
"""Physical-graph-guided offline evolution with immutable native sessions and receipts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

try:
    from asyncio import timeout as async_timeout
except ImportError:  # Python 3.10
    from async_timeout import timeout as async_timeout

from reweave.evolution.authoring.candidate import topology_feedback
from reweave.evolution.bundle import AgentBundle, Asset
from reweave.evolution.contracts import (
    AuthoredMutation,
    BundleVersion,
    EvaluationRecord,
    EvaluationRequest,
    EvidenceFile,
    EvolutionHistory,
    EvolutionIncomplete,
    EvolutionProtocol,
    EvolutionState,
    FinalProjection,
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    GraphRound,
    Proposal,
    RoundDecision,
    Usage,
    VersionReference,
    execution_handoff_allowed,
    same_execution_settings,
)
from reweave.evolution.evidence.archive_inputs import (
    archived_proposal,
    archived_proposal_value,
)
from reweave.evolution.evidence.records import artifact_record, artifact_records
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.campaign_slots import offline_admission
from reweave.evolution.infrastructure.persistence import attempt_directory
from reweave.evolution.search.matrix import (
    best_version,
    changes,
    passes,
    result_matrix,
    target_passed,
)
from reweave.utils.persistence import atomic_json, atomic_text


class AuthorWallTimeExceeded(TimeoutError):
    """One author branch exhausted its complete Host wall-clock allowance."""


def _usage_summary(usage: list[Usage], counted_attempts: int) -> dict[str, Any]:
    """Observed totals; a missing receipt never becomes zero known work."""
    return {
        "physical_steps": sum(item.physical_steps for item in usage),
        "physical_steps_complete": len(usage) == counted_attempts
        and all(item.physical_steps_complete for item in usage),
        "model_calls": sum(item.model_calls for item in usage)
        if len(usage) == counted_attempts
        and all(item.model_calls is not None for item in usage)
        else None,
        "known_model_calls": sum(item.model_calls or 0 for item in usage),
        "wall_time_s": sum(item.wall_time_s for item in usage),
        "charged_wall_time_s": sum(item.charged_wall_time_s for item in usage),
        "wall_time_complete": len(usage) == counted_attempts
        and all(item.wall_time_complete for item in usage),
        "cost_usd": sum(item.cost_usd for item in usage)
        if len(usage) == counted_attempts
        and usage
        and all(item.cost_usd is not None for item in usage)
        else None,
        "cost_complete": len(usage) == counted_attempts
        and bool(usage)
        and all(item.cost_usd is not None for item in usage),
    }


class EvolutionWorkspace:
    """Single-writer archive for version origins, proposals, and attempts."""

    def __init__(self, root: Path, baseline: AgentBundle, protocol: EvolutionProtocol):
        self.root = root.resolve()
        self.baseline = AgentBundle.model_validate_json(baseline.model_dump_json())
        self.protocol = EvolutionProtocol.model_validate_json(
            protocol.model_dump_json()
        )
        self.state_path = self.root / "evolution.json"
        if (protocol.execution.physical_steps, protocol.execution.wait_steps) != (
            baseline.runtime.physical_steps,
            baseline.runtime.wait_steps,
        ):
            raise ValueError("evolution horizon differs from the frozen baseline")

    def attempt_directory(self, key, attempt):
        """Host-owned location for one attempt, including explicit completed imports."""
        return attempt_directory(self.root, key, attempt)

    def require_configuration(self) -> Path:
        """Return the frozen run configuration or fail before launching work."""
        path = self.root / "configuration.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("evolution requires a frozen regular configuration.json")
        return path

    def execution_ids(self, state) -> frozenset[str]:
        """Published historical execution identities, never inferred from receipts."""
        archive = EvolutionState.model_validate(state)
        return frozenset(
            {
                self.protocol.execution_id,
                *(
                    protocol.execution_id
                    for revision in (
                        *archive.budget_revisions,
                        *archive.execution_revisions,
                    )
                    for protocol in (revision.previous, revision.current)
                ),
            }
        )

    @contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if self.state_path.exists():
                    state = EvolutionState.model_validate_json(
                        self.state_path.read_text()
                    ).model_dump(mode="json")
                    if (
                        state["baseline"] != self.baseline.record_id
                        or state["protocol"] != self.protocol.record_id
                    ):
                        raise ValueError(
                            "evolution baseline/protocol changed on resume"
                        )
                else:
                    state = EvolutionState(
                        baseline=self.baseline.record_id,
                        protocol=self.protocol.record_id,
                        versions={},
                        attempts={},
                        order=(),
                        proposals={},
                        stop_reason=None,
                        completed=False,
                    ).model_dump(mode="json")
                    self.save(state)
                yield state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save(self, state: dict[str, Any]) -> None:
        archive = EvolutionState.model_validate(state)
        for attempt in archive.attempts.values():
            if attempt.kind == "evaluation":
                request = EvaluationRequest.model_validate(attempt.request)
                if (
                    request.execution == self.protocol.execution_id
                    and not same_execution_settings(
                        request.conditions, self.protocol.execution
                    )
                ):
                    raise ValueError(
                        "evaluation conditions differ from the current published execution"
                    )
        validated = archive.model_dump(mode="json")
        # This mutable aggregate can contain hundreds of MB of already-published
        # records. Compact only this view; original request/receipt files retain
        # their existing bytes and IDs, with the same durable atomic publication.
        atomic_text(
            self.state_path,
            json.dumps(
                validated,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
        )

    def put_bundle(self, bundle: AgentBundle) -> str:
        if Path(bundle.record_id).name != bundle.record_id or bundle.record_id in {
            ".",
            "..",
        }:
            raise ValueError("bundle record ID must be one archive filename")
        path = self.root / "bundles" / f"{bundle.record_id}.json"
        value = bundle.model_dump(mode="json")
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("immutable bundle archive was modified")
        else:
            for original in sorted(path.parent.glob("*.json")):
                if original.name.startswith("."):
                    continue
                published = AgentBundle.model_validate_json(original.read_text())
                if published.same_harness(bundle) and published.runtime.model_dump(
                    exclude={"revision_id"}
                ) == bundle.runtime.model_dump(exclude={"revision_id"}):
                    return published.record_id
            atomic_json(path, value)
        return bundle.record_id

    def bundle(self, record_id: str) -> AgentBundle:
        if Path(record_id).name != record_id or record_id in {".", ".."}:
            raise ValueError("bundle record ID must be one archive filename")
        value = AgentBundle.model_validate_json(
            (self.root / "bundles" / f"{record_id}.json").read_text()
        )
        if value.record_id != record_id:
            raise ValueError("bundle archive record reference mismatch")
        return value

    def add_version(self, state: dict[str, Any], version: BundleVersion) -> str:
        self.bundle(version.bundle_id)
        value = version.model_dump(mode="json")
        identity = version.record_id
        existing = state["versions"].get(identity)
        if existing is not None and existing != value:
            raise ValueError("immutable bundle version changed")
        if existing is None or identity not in state["order"]:
            updated = {
                **state,
                "versions": {**state["versions"], identity: value},
                "order": [*state["order"], identity]
                if identity not in state["order"]
                else list(state["order"]),
            }
            self.save(updated)
            state.update(updated)
        return identity


class Author(Protocol):
    async def __call__(
        self, proposal: Proposal, directory: Path, *, before_launch: Callable[[], None]
    ) -> AuthoredMutation: ...

    def verify_result(
        self, proposal: Proposal, directory: Path
    ) -> AuthoredMutation: ...


class Evaluator(Protocol):
    async def __call__(
        self,
        request: EvaluationRequest,
        bundle: AgentBundle,
        directory: Path,
        *,
        before_launch: Callable[[], None],
    ) -> EvaluationRecord: ...


class EvolutionExecution:
    """Shared durable execution, accounting and evaluator transport only."""

    def __init__(
        self,
        workspace: EvolutionWorkspace,
        author: Author,
        evaluator: Evaluator,
        *,
        analyzer=None,
        recover: bool = False,
    ):
        self._active_attempts: dict[str, float] = {}
        self._evaluation_tasks: dict[
            str, asyncio.Task[tuple[EvaluationRecord, ...]]
        ] = {}
        self.workspace, self.author, self.evaluator = workspace, author, evaluator
        self.analyzer = analyzer
        self.recover = recover

    def _selected(self, key: str, attempt: dict[str, Any]) -> bool:
        """Whether a settled evaluation is shown to graph analysis and authors."""
        return True

    def accounting(self, state: dict[str, Any]) -> dict[str, Any]:
        usage = []
        groups = {"author": [], "evaluation": [], "graph_analysis": []}
        counts = {"author": 0, "evaluation": 0, "graph_analysis": 0}
        reserved_steps = 0
        reserved_wall = 0.0
        unbounded_reserved_wall = False
        unresolved_wall = False
        counted_attempts = 0
        for attempt_id, attempt in sorted(state["attempts"].items()):
            directory = self.workspace.attempt_directory(attempt_id, attempt)
            receipt = directory / "result.json"
            usage_path = directory / "audit/usage.json"
            if (
                attempt["launches"] == 0
                and not usage_path.exists()
                and attempt["status"] != "timed_out"
            ):
                continue
            counted_attempts += 1
            counts[attempt["kind"]] += 1
            measured = None
            if receipt.exists():
                receipt_type = {
                    "evaluation": EvaluationRecord,
                    "author": AuthoredMutation,
                    "graph_analysis": GraphAnalysisReceipt,
                }[attempt["kind"]]
                measured = receipt_type.model_validate_json(receipt.read_text()).usage
            elif usage_path.exists():
                measured = Usage.model_validate_json(usage_path.read_text())
            if attempt["status"] == "timed_out" and attempt["last_failure"]:
                failure = json.loads(
                    (self.workspace.root / attempt["last_failure"]["path"]).read_text()
                )
                elapsed = failure["wall_time_s"]
                if measured is None:
                    measured = Usage(
                        physical_steps=0,
                        model_calls=None,
                        wall_time_s=elapsed,
                        cost_usd=None,
                    )
                elif measured.wall_time_s < elapsed:
                    measured = measured.model_copy(update={"wall_time_s": elapsed})
            in_flight = attempt_id in self._active_attempts and not receipt.exists()
            if in_flight:
                elapsed = (
                    time.time()
                    - (attempt["deadline_at"] - self._native_deadline(attempt["kind"]))
                    if attempt["kind"] != "evaluation"
                    and attempt["deadline_at"] is not None
                    else time.monotonic() - self._active_attempts[attempt_id]
                )
                if measured is None:
                    measured = Usage(
                        physical_steps=0,
                        physical_steps_complete=attempt["kind"] != "evaluation",
                        model_calls=None,
                        wall_time_s=elapsed,
                        wall_time_complete=False,
                        cost_usd=None,
                    )
                limit = (
                    self.workspace.protocol.execution.wall_time_s
                    if attempt["kind"] == "evaluation"
                    else self._native_deadline(attempt["kind"])
                )
                if limit is None:
                    unbounded_reserved_wall = True
                else:
                    reserved_wall += (
                        max(0.0, attempt["deadline_at"] - time.time())
                        if attempt["kind"] != "evaluation"
                        else max(0.0, limit - elapsed)
                    )
            if measured is not None:
                usage.append(measured)
                groups[attempt["kind"]].append(measured)
            if not in_flight and (
                measured is None
                or (
                    not measured.wall_time_complete
                    and measured.wall_time_charge_s is None
                )
            ):
                unresolved_wall = True
            if attempt["kind"] == "evaluation" and (
                not receipt.exists() or not measured.physical_steps_complete
            ):
                horizon = self.workspace.protocol.execution.physical_steps
                if horizon is None:
                    reserved_steps = None
                elif reserved_steps is not None:
                    reserved_steps += max(
                        0,
                        horizon
                        - (measured.physical_steps if measured is not None else 0),
                    )
        graph_stages = [*state["rounds"].values()]
        graph_stages.extend(state.get("final_projections", {}).values())
        construction = [
            Usage(
                physical_steps=0,
                model_calls=0,
                wall_time_s=record["construction_wall_time_s"],
                cost_usd=0,
            )
            for record in graph_stages
        ]
        usage.extend(construction)
        groups["graph_construction"] = construction
        counts["graph_construction"] = len(construction)
        counted_attempts += len(construction)
        return {
            "graph_construction_wall_time_s": sum(
                r["construction_wall_time_s"] for r in graph_stages
            ),
            **_usage_summary(usage, counted_attempts),
            "reserved_physical_steps": reserved_steps,
            "reserved_wall_time_s": None if unbounded_reserved_wall else reserved_wall,
            "unresolved_wall_time": unresolved_wall,
            "by_kind": {
                kind: _usage_summary(values, counts[kind])
                for kind, values in groups.items()
            },
            "author_launches": sum(
                item["launches"]
                for item in state["attempts"].values()
                if item["kind"] == "author"
            ),
            "author_timeouts": sum(
                item["status"] == "timed_out"
                for item in state["attempts"].values()
                if item["kind"] == "author"
            ),
            "episode_launches": sum(
                item["launches"]
                for item in state["attempts"].values()
                if item["kind"] == "evaluation"
            ),
        }

    def _native_deadline(self, kind):
        return (
            self.workspace.protocol.graph.analysis_wall_time_s
            if kind == "graph_analysis"
            else self.workspace.protocol.author_wall_time_s
        )

    async def _attempt(
        self,
        state: dict[str, Any],
        kind: str,
        key: str,
        request: Asset,
        invoke: Callable[[Path, Callable[[], None]], Awaitable[Asset]],
        result_type: type[Asset],
        verify_result: Callable[[Asset, Path], Asset] | None = None,
        wall_time_s: float | None = None,
    ) -> Asset:
        if kind != "evaluation" and wall_time_s is None:
            wall_time_s = self._native_deadline(kind)
        if kind == "evaluation" and wall_time_s is not None:
            raise ValueError("evaluation attempts use the frozen episode deadline")
        attempt_path = Path(key)
        if attempt_path.is_absolute() or ".." in attempt_path.parts or not key:
            raise ValueError("attempt identity must be a relative readable path")
        request_value = request.model_dump(mode="json")
        matching = [
            identity
            for identity, attempt in state["attempts"].items()
            if attempt["kind"] == kind
            and (
                archived_proposal_value(attempt["request"])
                if kind == "author"
                else attempt["request"]
            )
            == request_value
            and (kind != "evaluation" or Path(attempt["key"]).name == attempt_path.name)
        ]
        if len(matching) > 1:
            raise ValueError("one exact request has multiple attempt identities")
        attempt_id = matching[0] if matching else attempt_path.as_posix()
        existing = state["attempts"].get(attempt_id)
        directory = self.workspace.attempt_directory(attempt_id, existing or {})
        receipt = directory / "result.json"
        recovering = existing is not None
        if existing is not None and (
            existing["kind"] != kind
            or existing["key"] != attempt_id
            or (
                archived_proposal_value(existing["request"])
                if kind == "author"
                else existing["request"]
            )
            != request_value
        ):
            raise ValueError(
                "readable attempt identity differs from its exact request identity"
            )
        if existing is None:
            existing = {
                "result_directory": None,
                "kind": kind,
                "key": key,
                "request": request_value,
                "status": "prepared",
                "launches": 0,
                "deadline_at": None,
                "result_id": None,
                "last_failure": None,
            }
            state["attempts"][attempt_id] = existing
            self.workspace.save(state)
        if (
            recovering
            and self.recover
            and existing["status"] != "completed"
            and kind != "evaluation"
            and wall_time_s is None
            and existing["deadline_at"] is not None
        ):
            # A published uncapped protocol permits this exact offline request
            # to resume. Preserve its native identity and original timeout.
            atomic_json(
                self.workspace.root
                / "audit/attempt_recoveries"
                / attempt_id
                / f"{uuid.uuid4().hex}.json",
                {
                    "timestamp": time.time(),
                    "previous_deadline_at": existing["deadline_at"],
                    "previous_status": existing["status"],
                    "deadline_at": None,
                    "protocol": self.workspace.protocol.record_id,
                    "reason": "Published offline time cap removal; recover the same request and native session.",
                },
            )
            existing["deadline_at"] = None
            if existing["status"] == "timed_out":
                existing["status"] = "cancelled"
            self.workspace.save(state)
        if existing["status"] == "timed_out":
            raise AuthorWallTimeExceeded(
                f"{kind} attempt {attempt_id} already exhausted its wall-clock deadline"
            )
        if not receipt.exists() and (
            existing["status"] == "completed" or existing["result_id"] is not None
        ):
            raise ValueError(
                f"completed attempt {attempt_id} is missing its immutable receipt; restore the original, never relaunch"
            )
        if receipt.exists():
            if existing["launches"] == 0:
                raise ValueError(
                    "adapter returned execution evidence without a registered launch"
                )
            result = (
                await run_blocking(verify_result, request, directory)
                if verify_result is not None
                else result_type.model_validate_json(receipt.read_text())
            )
            if not isinstance(result, result_type):
                raise TypeError("adapter verifier returned an invalid result type")
        else:
            async with offline_admission(directory, required=kind != "evaluation"):
                if (
                    kind != "evaluation"
                    and wall_time_s is not None
                    and existing["deadline_at"] is None
                    and existing["launches"] == 0
                    and existing["status"] == "prepared"
                ):
                    existing["deadline_at"] = time.time() + wall_time_s
                    self.workspace.save(state)
                directory.mkdir(parents=True, exist_ok=True)
                launched = False

                def before_launch() -> None:
                    nonlocal launched
                    if launched and kind == "evaluation":
                        raise RuntimeError(
                            "one evaluation adapter invocation cannot launch twice"
                        )
                    # Proven pre-spawn recovery reuses a physical registration;
                    # every author correction remains a measured native query.
                    if kind != "evaluation" or existing["launches"] == 0:
                        existing["launches"] += 1
                    existing["status"] = "running"
                    self.workspace.save(state)
                    self._active_attempts[attempt_id] = time.monotonic()
                    launched = True

                if recovering:
                    adapter = {
                        "author": self.author,
                        "graph_analysis": self.analyzer,
                        "evaluation": self.evaluator,
                    }[kind]
                    recovery = getattr(adapter, "recover", None)
                    if not self.recover or recovery is None:
                        raise EvolutionIncomplete(
                            f"attempt {attempt_id} needs explicit adapter recovery; not launching twice"
                        )
                    if (
                        kind != "evaluation"
                        and wall_time_s is not None
                        and existing["status"] in {"failed", "cancelled"}
                    ):
                        # A settled failure has stopped consuming work. Preserve all
                        # prior invocation time, excluding the operator's downtime.
                        failures = (
                            self.workspace.root / "audit/attempt_failures" / attempt_id
                        )
                        queue_only = set()
                        for path in (
                            self.workspace.root / "audit/offline_queue_reconciliation"
                        ).glob("*.json"):
                            correction = json.loads(path.read_text())
                            original = correction["original"]
                            if (
                                correction["attempt"] == attempt_id
                                and original["launches"] == 0
                                and original["request"] == existing["request"]
                                and original["last_failure"] is not None
                            ):
                                queue_only.add(original["last_failure"]["path"])
                        spent = sum(
                            json.loads(path.read_text())["invocation_wall_time_s"]
                            for path in failures.glob("*.json")
                            if path.relative_to(self.workspace.root).as_posix()
                            not in queue_only
                        )
                        remaining = max(0.0, wall_time_s - spent)
                        now = time.time()
                        atomic_json(
                            self.workspace.root
                            / "audit/attempt_recoveries"
                            / attempt_id
                            / f"{uuid.uuid4().hex}.json",
                            {
                                "timestamp": now,
                                "previous_deadline_at": existing["deadline_at"],
                                "prior_work_s": spent,
                                "remaining_work_s": remaining,
                                "deadline_at": now + remaining,
                            },
                        )
                        existing["deadline_at"] = now + remaining
                        self.workspace.save(state)
                started = time.monotonic()
                remaining_wall_time_s = (
                    max(0.0, existing["deadline_at"] - time.time())
                    if existing["deadline_at"] is not None
                    else None
                )
                if kind != "evaluation":
                    self._active_attempts[attempt_id] = time.monotonic()
                deadline = async_timeout(remaining_wall_time_s)
                try:
                    async with deadline:
                        if remaining_wall_time_s == 0:
                            await asyncio.sleep(0)
                        if recovering:
                            # This is adapter preparation, not confirmation that an old
                            # worker exited. The recovery adapter still owns that check.
                            existing["status"] = "prepared"
                            self.workspace.save(state)
                            result = (
                                await recovery(
                                    request, directory, before_launch=before_launch
                                )
                                if kind != "evaluation"
                                else await recovery(
                                    request,
                                    self.workspace.bundle(request.bundle_id),
                                    directory,
                                    before_launch=before_launch,
                                )
                            )
                        else:
                            result = await invoke(directory, before_launch)
                        if not isinstance(result, result_type):
                            raise TypeError("adapter returned an invalid result type")
                        if existing["launches"] == 0:
                            raise ValueError(
                                "adapter returned execution evidence without a registered launch"
                            )
                        atomic_json(receipt, result.model_dump(mode="json"))
                        await asyncio.sleep(0)
                except BaseException as error:
                    # Publish orchestration errors immediately. They are not model
                    # or physical outcomes and do not grant a retry by themselves.
                    expired = deadline.expired
                    timed_out = expired() if callable(expired) else expired
                    reported_error = (
                        AuthorWallTimeExceeded(
                            f"{kind} attempt {attempt_id} exceeded {wall_time_s:g}s end-to-end wall time"
                        )
                        if timed_out
                        else error
                    )
                    status = (
                        "timed_out"
                        if timed_out
                        else "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed"
                    )
                    # Evolution observations live outside adapter output inventories:
                    # an adapter may have sealed a valid receipt before raising.
                    failure_path = (
                        self.workspace.root
                        / "audit/attempt_failures"
                        / attempt_id
                        / f"{uuid.uuid4().hex}.json"
                    )
                    failure = {
                        "timestamp": time.time(),
                        "status": status,
                        "error_type": type(reported_error).__name__,
                        "error": str(reported_error),
                        "traceback": "".join(traceback.format_exception(error)),
                        "launches": existing["launches"],
                        "launch_registered_during_invocation": launched,
                        "invocation_wall_time_s": time.monotonic() - started,
                    }
                    if timed_out:
                        failure.update(
                            wall_time_s=time.time()
                            - (existing["deadline_at"] - wall_time_s),
                            wall_time_limit_s=wall_time_s,
                        )
                    atomic_json(failure_path, failure)
                    existing.update(
                        status=status,
                        last_failure={
                            "path": failure_path.relative_to(
                                self.workspace.root
                            ).as_posix(),
                        },
                    )
                    self.workspace.save(state)
                    if timed_out:
                        raise reported_error from error
                    raise
                finally:
                    self._active_attempts.pop(attempt_id, None)
        result_id = result.record_id
        if existing["status"] != "completed" or existing.get("result_id") != result_id:
            existing.update(status="completed", result_id=result_id)
            self.workspace.save(state)
        return result

    def _records(self, state) -> tuple[EvaluationRecord, ...]:
        """Read settled observations; consumers open only the originals they need."""
        records = []
        for identity, attempt in sorted(state["attempts"].items()):
            if attempt["kind"] != "evaluation" or attempt["status"] != "completed":
                continue
            if not self._selected(identity, attempt):
                continue
            path = self.workspace.attempt_directory(identity, attempt) / "result.json"
            result = EvaluationRecord.model_validate_json(path.read_text())
            if result.request.model_dump(mode="json") != attempt["request"]:
                raise ValueError("evaluation receipt request changed")
            records.append(result)
        return tuple(records)

    async def matrix(self, state, *, split="dev"):
        return result_matrix(
            self.workspace.protocol,
            (v["bundle_id"] for v in state["versions"].values()),
            await run_blocking(self._records, state),
            split=split,
            execution_ids=self.workspace.execution_ids(state),
        )

    async def _evaluate_bundle(
        self, state, bundle: AgentBundle, attempt_group: str, *, split="dev"
    ) -> tuple[EvaluationRecord, ...]:
        protocol = self.workspace.protocol
        conditions = protocol.dev if split == "dev" else protocol.validation.conditions
        manifest_id = protocol.manifest_id
        phase = "" if split == "dev" else "validation/"

        async def cell(condition_index, instance):
            request = EvaluationRequest(
                record_id=f"evaluation/{attempt_group}/{phase}condition-{condition_index + 1:03d}/{instance.record_id}",
                bundle_id=bundle.record_id,
                instance=instance,
                split=split,
                manifest_id=manifest_id,
                execution=self.workspace.protocol.execution_id,
                conditions=protocol.execution,
            )
            published = {}
            for attempt in state["attempts"].values():
                if attempt["kind"] != "evaluation":
                    continue
                original = EvaluationRequest.model_validate(attempt["request"])
                if original.model_dump(exclude={"record_id"}) == request.model_dump(
                    exclude={"record_id"}
                ):
                    published[original.record_id] = original
            if len(published) > 1:
                raise ValueError(
                    "one evaluation input has multiple published request records"
                )
            if published:
                request = next(iter(published.values()))
            first_key = f"evaluation/{attempt_group}/{phase}condition-{condition_index + 1:03d}/attempt-01"
            archived = state["attempts"].get(first_key)
            if (
                archived is not None
                and archived["request"]["execution"] != request.execution
                and not published
            ):
                # A separately published exact current-execution request is an
                # explicit repetition. Resume it by identity even when the
                # original group's first attempt belongs to an older runtime.
                # Without that request, retain historical work (including H0).
                original = EvaluationRequest.model_validate(archived["request"])
                if (
                    archived["status"] != "completed"
                    or original.execution not in self.workspace.execution_ids(state)
                    or original.model_dump(
                        exclude={"record_id", "execution", "conditions"}
                    )
                    != request.model_dump(
                        exclude={"record_id", "execution", "conditions"}
                    )
                ):
                    raise ValueError(
                        "historical evaluation is not settled under a published protocol revision"
                    )
                request = original

            async def invoke(directory, before_launch):
                return await self.evaluator(
                    request, bundle, directory, before_launch=before_launch
                )

            for retry in range(self.workspace.protocol.infrastructure_retries + 1):
                record = await self._attempt(
                    state,
                    "evaluation",
                    f"evaluation/{attempt_group}/{phase}condition-{condition_index + 1:03d}/attempt-{retry + 1:02d}",
                    request,
                    invoke,
                    EvaluationRecord,
                )
                if record.request != request:
                    raise ValueError("runner returned a different evaluation request")
                if record.failure_origin != "infrastructure":
                    return record
                if not record.infrastructure_retryable:
                    return record
            # The adapter returned settled receipts for every allowed attempt.
            # Exhaustion ends this cell's retries, not the finite search. Keep
            # its official outcome missing; unknown/unsettled work still blocks.
            return record

        # Gather does not abandon other workers when one cell fails.
        outcomes = await asyncio.gather(
            *(cell(index, instance) for index, instance in enumerate(conditions)),
            return_exceptions=True,
        )
        errors = [r for r in outcomes if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        if any(record.failure_origin == "unknown" for record in outcomes):
            raise EvolutionIncomplete(
                "unacknowledged physical outcome; reconcile original work before continuing"
            )
        # Qualification and search continuation are different decisions. A
        # settled infrastructure receipt stays null and cannot qualify, but it
        # supplies history for the next candidate without authorizing replay.
        return tuple(outcomes)

    async def _evaluate_candidate(self, state, bundle, attempt_group):
        """Pay for validation only after this exact content qualifies on dev."""
        records = await self._evaluate_bundle(state, bundle, attempt_group)
        protocol = self.workspace.protocol
        if protocol.validation is not None and passes(
            tuple(
                None
                if record.failure_origin == "infrastructure"
                else record.official_success
                for record in records
            ),
            protocol.target_success_rate,
        ):
            records += await self._evaluate_bundle(
                state, bundle, attempt_group, split="validation"
            )
        return records

    async def _evaluate_content(
        self, state, bundle: AgentBundle, attempt_group: str
    ) -> tuple[EvaluationRecord, ...]:
        """Evaluate content once while allowing its first author to start immediately."""
        task = self._evaluation_tasks.get(bundle.record_id)
        if task is None:
            task = asyncio.create_task(
                self._evaluate_candidate(state, bundle, attempt_group)
            )
            self._evaluation_tasks[bundle.record_id] = task
        try:
            return await task
        finally:
            if task.done() and self._evaluation_tasks.get(bundle.record_id) is task:
                del self._evaluation_tasks[bundle.record_id]

    async def _evaluate(self, state, version_id) -> tuple[EvaluationRecord, ...]:
        bundle = self.workspace.bundle(state["versions"][version_id]["bundle_id"])
        return await self._evaluate_content(
            state, bundle, f"V{state['order'].index(version_id):03d}"
        )


class GraphEvolution(EvolutionExecution):
    """Graph-selected rounds of independent sessions with immutable receipts."""

    def _history_summary(
        self,
        state: dict[str, Any],
        rows: dict[str, tuple[bool | None, ...]],
        results: dict[str, Asset],
        artifacts: dict[str, dict],
        round_index: int,
    ) -> tuple[str, dict[str, Any]]:
        """Index measured history; keep author testimony in its exact originals."""
        mutation_outcomes = ()
        if self.workspace.protocol.evolution_mode == "no_topology":
            mutation_outcomes = tuple(
                {"outcome": {"path": str(path)}}
                for index in sorted(
                    {p["round_index"] for p in state["proposals"].values()}
                )
                if (
                    path := self.workspace.root / "outcomes" / f"round-{index:06d}.json"
                ).exists()
            )
        versions = {
            identity: BundleVersion.model_validate(value)
            for identity, value in state["versions"].items()
        }
        labels = {
            identity: f"V{index:03d}" for index, identity in enumerate(state["order"])
        }
        attempt_labels = {
            identity: f"A{index:03d}"
            for index, identity in enumerate(state["attempts"])
        }
        records = tuple(
            result
            for result in results.values()
            if isinstance(result, EvaluationRecord)
        )
        validation_rows = result_matrix(
            self.workspace.protocol,
            (version.bundle_id for version in versions.values()),
            records,
            split="validation",
            execution_ids=self.workspace.execution_ids(state),
        )
        validation = self.workspace.protocol.validation

        def original(path: Path) -> dict[str, Any]:
            source = str(path.resolve())
            return {**artifacts[source], "path": source, "pointer": ""}

        navigation: dict[str, Any] = {
            "versions": {},
            "bundles": [],
            "attempts": {},
            "condition_index": [
                {
                    "label": f"C{index:03d}",
                    "condition_id": condition.record_id,
                    "split": "dev"
                    if index < len(self.workspace.protocol.dev)
                    else "validation",
                }
                for index, condition in enumerate(self.workspace.protocol.instances)
            ],
        }
        bundles = {}
        author_attempts = {}
        inquiries = []
        failures = []
        mutation_reviews = {}
        for prior_round in (*state["rounds"].values(), *mutation_outcomes):
            if prior_round["outcome"] is not None:
                outcome = json.loads(Path(prior_round["outcome"]["path"]).read_text())
                mutation_reviews.update(
                    (item["proposal_id"], item["mutation_review"])
                    for item in outcome["observations"]
                    if item.get("mutation_review") is not None
                )
        for attempt_id, attempt in state["attempts"].items():
            directory = self.workspace.attempt_directory(attempt_id, attempt)
            label = attempt_labels[attempt_id]
            result = results.get(attempt_id)
            entry = {
                "name": attempt_id,
                "kind": attempt["kind"],
                "status": attempt["status"],
                "launches": attempt["launches"],
                "receipt": original(directory / "result.json")
                if result is not None
                else None,
                "inquiry": None,
                "failures": [
                    original(path)
                    for path in sorted(
                        (
                            self.workspace.root / "audit/attempt_failures" / attempt_id
                        ).glob("*.json")
                    )
                ],
                "recoveries": [
                    original(path)
                    for path in sorted(
                        (
                            self.workspace.root
                            / "audit/attempt_recoveries"
                            / attempt_id
                        ).glob("*.json")
                    )
                ],
            }
            if isinstance(result, EvaluationRecord):
                entry.update(
                    bundle_id=result.request.bundle_id,
                    condition_id=result.request.instance.record_id,
                    evidence_id=result.episode.evidence_id if result.episode else None,
                    official_success=result.official_success,
                    failure_origin=result.failure_origin,
                    infrastructure_retryable=result.infrastructure_retryable,
                )
            elif isinstance(result, AuthoredMutation):
                author_attempts[attempt["request"]["record_id"]] = (label, result)
                marker = "/workspace/candidate/"
                entry["candidate_files"] = {
                    artifact.path.replace("\\", "/").split(marker, 1)[1]: original(
                        Path(artifact.path)
                    )
                    for artifact in result.artifacts
                    if marker in artifact.path.replace("\\", "/")
                }
            inquiry_path = directory / "audit/session_inquiry/result.json"
            if inquiry_path.is_file():
                entry["inquiry"] = original(inquiry_path)
                inquiries.append((label, json.loads(inquiry_path.read_text())))
            if attempt["last_failure"] is not None:
                path = self.workspace.root / attempt["last_failure"]["path"]
                entry["last_failure"] = original(path)
                failures.append((label, attempt, json.loads(path.read_text())))
            navigation["attempts"][label] = entry

        latest_feedback = []
        lines = [
            "# Shared Evolution History",
            "",
            f"This immutable view was materialized before evolution round {round_index + 1}. "
            "It indexes every earlier Harness mutation and observed development result. "
            "Author diagnoses and SKILL.md experience remain verbatim in their original receipts; "
            "read the indexed notes when relevant and compare hypotheses with subsequent official outcomes. "
            "Use history_originals.json for exact versions, source files, receipts and every evaluation attempt. "
            "V labels below are display names; the navigation supplies the corresponding version handles.",
            "",
            "## Conditions",
            "",
            "| Condition | Pool | Task | Initialization |",
            "| --- | --- | --- | --- |",
        ]
        for index, condition in enumerate(self.workspace.protocol.instances):
            initialization = json.dumps(
                condition.initialization,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).replace("|", "\\|")
            pool = "dev" if index < len(self.workspace.protocol.dev) else "validation"
            lines.append(
                f"| C{index:03d} | {pool} | `{condition.task}` | `{initialization}` |"
            )
        if validation is not None:
            lines.extend(
                [
                    "",
                    f"A candidate must first reach {self.workspace.protocol.target_success_rate:.0%} on development, "
                    f"then {validation.target_success_rate:.0%} on the complete validation pool. "
                    "Failed validation returns evidence to later authors and graph refinement; "
                    "this pool is not an independent held-out test.",
                ]
            )
        lines.extend(
            [
                "",
                "## Official results",
                "",
                "Unknown includes unmeasured conditions and unscored infrastructure outcomes; it is not failure. "
                "Each original attempt, including retries and incomplete runner evidence, remains in the navigation.",
                "",
                "| Version | "
                + " | ".join(
                    f"C{index:03d}"
                    for index in range(len(self.workspace.protocol.instances))
                )
                + " |",
                "| --- | "
                + " | ".join("---" for _ in self.workspace.protocol.instances)
                + " |",
            ]
        )
        for version_id in state["order"]:
            version = versions[version_id]
            outcomes = (*rows[version.bundle_id], *validation_rows[version.bundle_id])
            lines.append(
                f"| {labels[version_id]} | "
                + " | ".join(
                    "success" if result else "failure" if result is False else "unknown"
                    for result in outcomes
                )
                + " |"
            )
        lines.extend(["", "## Historical Harnesses", ""])
        for version_id in state["order"]:
            version = versions[version_id]
            row = rows[version.bundle_id]
            if version.bundle_id not in bundles:
                bundle_path = (
                    self.workspace.root / "bundles" / f"{version.bundle_id}.json"
                )
                bundles[version.bundle_id] = {
                    "bundle_id": version.bundle_id,
                    "bundle": original(bundle_path),
                    "evaluation_attempts": [
                        attempt_labels[identity]
                        for identity, result in results.items()
                        if isinstance(result, EvaluationRecord)
                        and result.request.bundle_id == version.bundle_id
                    ],
                }
            entry = {
                "label": labels[version_id],
                "bundle_id": version.bundle_id,
                "parent_version_id": None,
                "references": [],
                "author_attempt": None,
                "origin": None,
            }
            navigation["versions"][version_id] = entry
            lines.extend([f"### {labels[version_id]}", ""])
            if version.proposal_id is None:
                lines.append("- Origin: initial H0")
            else:
                proposal = archived_proposal(state["proposals"][version.proposal_id])
                author_label, receipt = author_attempts[version.proposal_id]
                candidate = next(
                    item
                    for item in receipt.candidates
                    if item.branch_id == version.branch_id
                )
                candidate_index = receipt.candidates.index(candidate)
                notes = {
                    field: {
                        **navigation["attempts"][author_label]["receipt"],
                        "pointer": f"/candidates/{candidate_index}/{field}",
                    }
                    for field in ("summary", "experience")
                }
                entry.update(
                    round_index=proposal.round_index,
                    parent_version_id=proposal.parent.version_id,
                    references=[ref.version_id for ref in proposal.references],
                    author_attempt=author_label,
                    author_notes=notes,
                    mutation_review=mutation_reviews.get(proposal.record_id),
                    origin={
                        "proposal_id": proposal.record_id,
                        "branch_id": version.branch_id,
                        "session_id": receipt.session_id,
                    },
                )
                feedback = [
                    item.model_dump(mode="json")
                    for item in topology_feedback(candidate.summary)
                ]
                if feedback:
                    entry["topology_feedback"] = feedback
                    if proposal.round_index == round_index - 1:
                        latest_feedback.append(
                            {
                                "version_id": version_id,
                                "label": labels[version_id],
                                "round_index": proposal.round_index,
                                "author_notes": notes,
                                "items": feedback,
                            }
                        )
                references = tuple(
                    self.workspace.bundle(ref.bundle_id) for ref in proposal.references
                )
                parent = self.workspace.bundle(proposal.parent.bundle_id)
                changed_scopes = sorted(candidate.bundle.novel_scopes((parent,)))
                novel_scopes = sorted(candidate.bundle.novel_scopes(references))
                comparisons = {
                    labels[ref.version_id]: changes(rows[ref.bundle_id], row)
                    for ref in proposal.references
                }
                lines.extend(
                    [
                        f"- Inherited parent: {labels[proposal.parent.version_id]}",
                        f"- Components changed from parent: {', '.join(changed_scopes)}",
                        "- Available reference set: "
                        + ", ".join(
                            labels[ref.version_id] for ref in proposal.references
                        ),
                        f"- Components novel to references: {', '.join(novel_scopes) or 'none (new composition)'}",
                        f"- Results against each reference: {json.dumps(comparisons, sort_keys=True)}",
                        f"- Mutation originals: {author_label} in history_originals.json",
                        "- Diagnosis and SKILL.md (unverified): navigation.versions for this version, author_notes.summary / author_notes.experience.",
                        "- Observed intervention exposure: navigation.versions for this version, mutation_review.",
                    ]
                )
            lines.append("")
        navigation["bundles"] = list(bundles.values())
        if latest_feedback:
            navigation["latest_topology_feedback"] = latest_feedback
            feedback_lines = [
                "## Latest author topology feedback",
                "",
                "These assessments concern the graph supplied to the preceding round. "
                "Check their evidence and subsequent outcomes; silence is not agreement. "
                "Older assessments remain in navigation.versions[*].topology_feedback.",
                "",
            ]
            for review in latest_feedback:
                for item in review["items"]:
                    feedback_lines.extend(
                        [
                            f"- **{item['stance']}** — {review['label']}, {item['target']}",
                            f"  Reason: {item['reason']}",
                            f"  Evidence: {len(item['evidence'])} original(s); exact references in navigation.versions for this version.",
                        ]
                    )
            feedback_lines.append("")
            lines[lines.index("## Conditions") : lines.index("## Conditions")] = (
                feedback_lines
            )
        lines.extend(["## Historical session inquiries (unverified testimony)", ""])
        for label, inquiry in inquiries:
            lines.extend(
                [
                    f"- {label}: completed={inquiry['completed']}. "
                    "Full answer and evidence are in history_originals.json; check against measured outcomes before reuse.",
                    "",
                    inquiry["question"]["question"],
                    "",
                ]
            )
        lines.extend(["## Orchestration failures and recoveries", ""])
        for label, attempt, failure in failures:
            lines.extend(
                [
                    f"- {label} ({attempt['kind']}): {failure['error_type']}. "
                    f"Current status: {attempt['status']}; launches: {attempt['launches']}. "
                    "All failure and recovery originals are in history_originals.json.",
                    "",
                    failure["error"],
                    "",
                ]
            )
        if not failures:
            lines.append("- None.")
        lines.extend(
            [
                "",
                "## Original evidence index",
                "",
                "[`history_originals.json`](history_originals.json) maps display versions to exact identities, "
                "complete bundles and author originals, and lists every evaluation, inquiry, failure and recovery attempt. "
                "Its files catalog also retains prior history snapshots and all validation and source originals. "
                "Read the indexed originals when needed; complete episodes and trace views are available through "
                "context/input.json and read_trace.",
                "",
            ]
        )
        return "\n".join(lines), navigation

    async def _freeze_history(self, state, rows, round_index, *, identity=None):
        versions = {
            k: BundleVersion.model_validate(v) for k, v in state["versions"].items()
        }
        identity = identity or f"round_{round_index:06d}"
        artifacts = {}
        results = {}
        published_paths = []
        for attempt_id, attempt in state["attempts"].items():
            directory = self.workspace.attempt_directory(attempt_id, attempt)
            inquiry = directory / "audit/session_inquiry/result.json"
            if inquiry.is_file():
                published_paths.append(inquiry)
            if attempt["status"] == "timed_out":
                private_roots = (
                    directory / "workspace/.scratch/runtime",
                    directory / "audit/session_inquiry/workspace",
                )
                for path in sorted(directory.rglob("*")):
                    if (
                        path.is_file()
                        and not path.name.startswith(".artifact_records")
                        and not path.is_symlink()
                        and not any(path.is_relative_to(root) for root in private_roots)
                    ):
                        published_paths.append(path)
                continue
            if attempt["status"] != "completed":
                raise EvolutionIncomplete(
                    "cannot select a new round with unresolved prior work"
                )
            receipt = directory / "result.json"
            model = {
                "evaluation": EvaluationRecord,
                "author": AuthoredMutation,
                "graph_analysis": GraphAnalysisReceipt,
            }[attempt["kind"]]
            result = model.model_validate_json(receipt.read_text())
            if isinstance(result, EvaluationRecord):
                if result.request.model_dump(mode="json") != attempt["request"]:
                    raise ValueError("evaluation receipt request changed")
            results[attempt_id] = result
            originals = (
                result.evidence
                if isinstance(result, EvaluationRecord)
                else result.artifacts
            )
            artifacts.update(
                {item.path: item.model_dump(mode="json") for item in originals}
            )
            published_paths.append(receipt)
        for category in ("attempt_failures", "attempt_recoveries"):
            for path in sorted(
                (self.workspace.root / "audit" / category).rglob("*.json")
            ):
                published_paths.append(path)
        for version in versions.values():
            path = self.workspace.root / "bundles" / f"{version.bundle_id}.json"
            published_paths.append(path)
        artifacts.update(
            (item["path"], item) for item in artifact_records(published_paths)
        )
        for prior in state["proposals"].values():
            history = EvolutionHistory.model_validate(prior["history"])
            for original in (history.summary, history.originals):
                artifacts[original.path] = original.model_dump(mode="json")
        summary_path = self.workspace.root / "history" / f"{identity}.md"
        originals_path = self.workspace.root / "history" / f"{identity}.originals.json"
        summary, navigation = self._history_summary(
            state, rows, results, artifacts, round_index
        )
        originals = {
            "files": {
                path: {"artifact_id": item["artifact_id"], "bytes": item["bytes"]}
                for path, item in artifacts.items()
            },
            "navigation": navigation,
        }
        if not summary_path.exists():
            atomic_text(summary_path, summary)
        if not originals_path.exists():
            atomic_json(originals_path, originals)
        history = EvolutionHistory(
            summary=EvidenceFile(**artifact_record(summary_path)),
            originals=EvidenceFile(**artifact_record(originals_path)),
        )
        return history

    def _graph(self, state):
        from reweave.evolution.evidence.trajectory import (
            build_graph,
            episode_from_record,
        )

        versions = tuple(
            VersionReference(version_id=k, bundle_id=state["versions"][k]["bundle_id"])
            for k in state["order"]
        )
        episodes = []
        for key, attempt in state["attempts"].items():
            if attempt["kind"] != "evaluation" or attempt["status"] != "completed":
                continue
            if not self._selected(key, attempt):
                continue
            path = self.workspace.attempt_directory(key, attempt) / "result.json"
            record = EvaluationRecord.model_validate_json(path.read_text())
            if record.failure_origin != "infrastructure" or record.episode is not None:
                episodes.append(episode_from_record(record, path, versions))
        return build_graph(episodes, self.workspace.protocol.graph)

    def _artifact(self, relative, value):
        path = self.workspace.root / relative
        serialized = (
            value.model_dump(mode="json") if isinstance(value, Asset) else value
        )
        if not path.exists():
            atomic_json(path, serialized)
        return EvidenceFile(**artifact_record(path))

    async def _analysis_request(self, state, rows, round_index, *, final_index=None):
        from reweave.evolution.graph.selection import current_round_versions

        started = time.monotonic()
        graph = await run_blocking(self._graph, state)
        final = final_index is not None
        name = f"final-{final_index:06d}" if final else f"round-{round_index:06d}"
        graph_file = self._artifact(f"graphs/{name}.json", graph)
        history = await self._freeze_history(
            state, rows, round_index, identity=name if final else None
        )
        previous = None
        if round_index:
            prior = GraphRound.model_validate(state["rounds"][str(round_index - 1)])
            prior.decision.verify()
            prior_decision = RoundDecision.model_validate_json(
                Path(prior.decision.path).read_text()
            )
            previous = prior_decision.analysis
        request = GraphAnalysisRequest(
            purpose="final_selection" if final else "search",
            previous_analysis=previous,
            record_id=f"final_{final_index:06d}_analysis"
            if final
            else f"round_{round_index:06d}_analysis",
            round_index=round_index,
            protocol=self.workspace.protocol,
            graph=graph_file,
            history=history,
            evidence=await run_blocking(self._records, state),
            versions=tuple(
                VersionReference(
                    version_id=k, bundle_id=state["versions"][k]["bundle_id"]
                )
                for k in state["order"]
            ),
            current_round_versions=current_round_versions(state),
            policy=self.analyzer.policy.model_dump(mode="json"),
        )
        return request, time.monotonic() - started

    async def _analyze(self, state, request, *, final_index=None):
        """Run one recoverable native analysis and deterministically project it."""
        from reweave.evolution.contracts import GraphSnapshot
        from reweave.evolution.graph.projection import project_map, selected_graph_receipts

        request.graph.verify()
        request.history.verify()

        async def invoke(directory, before_launch):
            return await self.analyzer(request, directory, before_launch=before_launch)

        final = final_index is not None
        result = await self._attempt(
            state,
            "graph_analysis",
            f"final-selection-{final_index:06d}/graph-analysis"
            if final
            else f"round-{request.round_index + 1:02d}/graph-analysis",
            request,
            invoke,
            GraphAnalysisReceipt,
            verify_result=self.analyzer.verify_result,
            wall_time_s=self.workspace.protocol.graph.analysis_wall_time_s,
        )
        attempt = next(
            k
            for k, a in state["attempts"].items()
            if a["kind"] == "graph_analysis"
            and a["request"] == request.model_dump(mode="json")
        )
        result_path = (
            self.workspace.attempt_directory(attempt, state["attempts"][attempt])
            / "result.json"
        )
        analysis_file = EvidenceFile(**artifact_record(result_path))
        evidence_graph = GraphSnapshot.model_validate_json(
            Path(request.graph.path).read_text()
        )
        name = (
            f"final-{final_index:06d}" if final else f"round-{request.round_index:06d}"
        )
        semantic_file = self._artifact(
            f"graphs/semantic-final-{final_index:06d}.json"
            if final
            else f"graphs/semantic-{request.round_index:06d}.json",
            result.analysis.graph,
        )
        progress_map = project_map(
            evidence_graph,
            result.analysis.graph,
            request.versions,
            request.protocol.dev,
            graph_id=result.record_id + "/analysis/graph",
            validation_conditions=request.protocol.validation.conditions
            if request.protocol.validation
            else (),
            cell_receipts=selected_graph_receipts(
                request.protocol,
                (version.bundle_id for version in request.versions),
                request.evidence,
                evidence_graph,
                execution_ids=self.workspace.execution_ids(state),
            ),
        )
        map_file = self._artifact(f"maps/{name}.json", progress_map)
        return result, analysis_file, semantic_file, map_file, progress_map

    async def _finalize_topology(self, state):
        """Include last-round outcomes before choosing the returned harness."""
        if not state["rounds"]:
            return
        stages = state.setdefault("final_projections", {})
        # A protocol revision must settle its original native request before a
        # new frozen history can be created. Never bypass unresolved work.
        for prior_key, value in tuple(stages.items()):
            if (
                value["request"]["protocol"]["record_id"]
                == self.workspace.protocol.record_id
                or value["progress_map"] is not None
            ):
                continue
            prior = FinalProjection.model_validate(value)
            _, analysis, graph, progress_map, _ = await self._analyze(
                state, prior.request, final_index=int(prior_key)
            )
            stages[prior_key] = prior.model_copy(
                update={
                    "analysis": analysis,
                    "graph": graph,
                    "progress_map": progress_map,
                }
            ).model_dump(mode="json")
            self.workspace.save(state)
        key = next(
            (
                k
                for k, stage in stages.items()
                if stage["request"]["protocol"]["record_id"]
                == self.workspace.protocol.record_id
            ),
            None,
        )
        if key is None:
            key = str(len(stages))
            request, elapsed = await self._analysis_request(
                state,
                await self.matrix(state),
                len(state["rounds"]),
                final_index=int(key),
            )
            stages[key] = FinalProjection(
                request=request, construction_wall_time_s=elapsed
            ).model_dump(mode="json")
            self.workspace.save(state)
        stage = FinalProjection.model_validate(stages[key])
        if stage.progress_map is not None:
            for evidence in (stage.analysis, stage.graph, stage.progress_map):
                evidence.verify()
            return
        _, analysis, graph, progress_map, _ = await self._analyze(
            state, stage.request, final_index=int(key)
        )
        stages[key] = stage.model_copy(
            update={"analysis": analysis, "graph": graph, "progress_map": progress_map}
        ).model_dump(mode="json")
        self.workspace.save(state)

    async def _proposals(self, state, rows, round_index):
        if self.workspace.protocol.evolution_mode == "no_topology":
            parent_id = best_version(
                state["order"],
                {
                    key: BundleVersion.model_validate(value)
                    for key, value in state["versions"].items()
                },
                rows,
            )
            parent = VersionReference(
                version_id=parent_id,
                bundle_id=state["versions"][parent_id]["bundle_id"],
            )
            proposal = Proposal(
                record_id=f"no-topology-round-{round_index + 1:02d}",
                purpose="no_topology",
                task_protocol=self.workspace.protocol,
                references=(parent,),
                parent=parent,
                round_index=round_index,
                author_index=0,
                scopes=self.workspace.protocol.mutation_scopes,
                history=await self._freeze_history(state, rows, round_index),
                evidence=await run_blocking(self._records, state),
            )
            state["proposals"][proposal.record_id] = proposal.model_dump(mode="json")
            self.workspace.save(state)
            return (proposal,)
        from reweave.evolution.graph.selection import decide_round, estimate_cost

        if self.analyzer is None:
            raise EvolutionIncomplete(
                "physical graph evolution requires a native graph analysis adapter"
            )
        key = str(round_index)
        if key not in state["rounds"]:
            request, elapsed = await self._analysis_request(state, rows, round_index)
            state["rounds"][key] = GraphRound(
                request=request,
                decision=None,
                outcome=None,
                construction_wall_time_s=elapsed,
            ).model_dump(mode="json")
            self.workspace.save(state)
        round_record = GraphRound.model_validate(state["rounds"][key])
        request = round_record.request
        if round_record.decision is not None:
            decision = RoundDecision.model_validate_json(
                Path(round_record.decision.path).read_text()
            )
            if decision.request_id != request.record_id:
                raise ValueError("round decision belongs to another graph request")
            proposals = tuple(
                archived_proposal(value)
                for value in state["proposals"].values()
                if value["round_index"] == round_index
            )
            if len(proposals) == decision.width:
                return tuple(
                    sorted(proposals, key=lambda proposal: proposal.author_index)
                )
        if round_record.decision is not None:
            decision_file = round_record.decision
            semantic_file = decision.graph
            map_file = decision.progress_map
        else:
            (
                result,
                analysis_file,
                semantic_file,
                map_file,
                progress_map,
            ) = await self._analyze(state, request)
            decision = decide_round(
                request,
                result,
                analysis_file,
                semantic_file,
                map_file,
                progress_map,
                estimate_cost(state, self.workspace.root, self.workspace.protocol),
                state,
            )
            decision_file = self._artifact(
                f"decisions/round-{round_index:06d}.json", decision
            )
            state["rounds"][key]["decision"] = decision_file.model_dump(mode="json")
            self.workspace.save(state)
        if decision.width > self.workspace.protocol.max_authors_per_round:
            raise EvolutionIncomplete(
                "frozen round width exceeds the reduced author cap; preserve the decision without creating additional authors"
            )
        proposal_protocol = request.protocol
        if (
            request.protocol.execution_id != self.workspace.protocol.execution_id
            or request.protocol.max_authors_per_round
            != self.workspace.protocol.max_authors_per_round
            or request.protocol.author_wall_time_s
            != self.workspace.protocol.author_wall_time_s
            or (
                request.protocol.graph.analysis_wall_time_s is not None
                and self.workspace.protocol.graph.analysis_wall_time_s is None
            )
        ):
            archive = EvolutionState.model_validate(state)
            if not execution_handoff_allowed(
                request.protocol,
                self.workspace.protocol,
                (*archive.execution_revisions, *archive.budget_revisions),
            ):
                raise ValueError(
                    "new authors require the graph protocol or its published execution revision"
                )
            proposal_protocol = self.workspace.protocol
        proposals = tuple(
            Proposal(
                record_id=f"round_{round_index:06d}_author_{index:03d}",
                round_index=round_index,
                author_index=index,
                parent=decision.parent,
                references=decision.references,
                scopes=proposal_protocol.mutation_scopes,
                evidence=request.evidence,
                history=request.history,
                graph=semantic_file,
                execution_graph=request.graph,
                progress_map=map_file,
                decision=decision_file,
                task_protocol=proposal_protocol,
            )
            for index in range(decision.width)
        )
        for proposal in proposals:
            state["proposals"].setdefault(
                proposal.record_id, proposal.model_dump(mode="json")
            )
        self.workspace.save(state)
        return proposals

    async def _settle_round(self, state, proposals):
        index = proposals[0].round_index
        analysis = None
        if self.workspace.protocol.evolution_mode == "no_topology":
            if (self.workspace.root / "outcomes" / f"round-{index:06d}.json").exists():
                return
        else:
            record = GraphRound.model_validate(state["rounds"][str(index)])
            if record.outcome is not None:
                return
            decision = RoundDecision.model_validate_json(
                Path(record.decision.path).read_text()
            )
            analysis = GraphAnalysisReceipt.model_validate_json(
                Path(decision.analysis.path).read_text()
            )
        rows = await self.matrix(state)
        validation_rows = await self.matrix(state, split="validation")
        from reweave.evolution.search.mutation_review import review_mutation

        records = await run_blocking(self._records, state)
        observations = []
        for proposal in proposals:
            attempt = next(
                a
                for a in state["attempts"].values()
                if a["kind"] == "author"
                and a["request"]["record_id"] == proposal.record_id
            )
            version = next(
                (
                    v
                    for v in state["versions"].values()
                    if v["proposal_id"] == proposal.record_id
                ),
                None,
            )
            success = None
            if version is not None:
                row = rows[version["bundle_id"]]
                if None not in row:
                    success = target_passed(
                        self.workspace.protocol,
                        row,
                        validation_rows[version["bundle_id"]],
                    )
            elif attempt["status"] == "timed_out":
                success = False
            mutation_review = None
            if version is not None:
                receipt = AuthoredMutation.model_validate_json(
                    (
                        self.workspace.attempt_directory(attempt["key"], attempt)
                        / "result.json"
                    ).read_text()
                )
                mutation_review = await run_blocking(
                    review_mutation,
                    proposal,
                    receipt.candidates[0].model_copy(
                        update={"bundle": self.workspace.bundle(version["bundle_id"])}
                    ),
                    records,
                )
            observations.append(
                {
                    "proposal_id": proposal.record_id,
                    "success": success,
                    "mutation_review": mutation_review,
                }
            )
        outcome = self._artifact(
            f"outcomes/round-{index:06d}.json",
            {
                "observations": observations,
                **(
                    {"graph_id": analysis.record_id + "/analysis/graph"}
                    if analysis is not None
                    else {}
                ),
            },
        )
        if analysis is not None:
            state["rounds"][str(index)]["outcome"] = outcome.model_dump(mode="json")
        self.workspace.save(state)

    async def _author(self, state, proposal):
        if (
            proposal.task_protocol.max_authors_per_round
            > self.workspace.protocol.max_authors_per_round
        ):
            decision = RoundDecision.model_validate_json(
                Path(proposal.decision.path).read_text()
            )
            if decision.width > self.workspace.protocol.max_authors_per_round:
                request_value = proposal.model_dump(mode="json")
                settled = any(
                    attempt["kind"] == "author"
                    and archived_proposal_value(attempt["request"]) == request_value
                    and (
                        self.workspace.attempt_directory(identity, attempt)
                        / "result.json"
                    ).is_file()
                    for identity, attempt in state["attempts"].items()
                )
                if not settled:
                    raise EvolutionIncomplete(
                        "frozen round width exceeds the reduced author cap; pending authors cannot launch or resume automatically"
                    )

        async def invoke(directory, before_launch):
            return await self.author(proposal, directory, before_launch=before_launch)

        result = await self._attempt(
            state,
            "author",
            f"round-{proposal.round_index + 1:02d}/author-{proposal.author_index + 1:02d}",
            proposal,
            invoke,
            AuthoredMutation,
            verify_result=self.author.verify_result,
            wall_time_s=self.workspace.protocol.author_wall_time_s,
        )
        if result.proposal_id != proposal.record_id:
            raise ValueError("author receipt answers a different proposal")
        return result

    async def _finish_round(self, state, proposals):
        async def branch(proposal):
            try:
                result = await self._author(state, proposal)
            except AuthorWallTimeExceeded:
                return None
            (candidate,) = result.candidates
            references = tuple(
                self.workspace.bundle(ref.bundle_id) for ref in proposal.references
            )
            candidate.bundle.assert_candidate(references, frozenset(proposal.scopes))
            if any(
                candidate.bundle.same_harness(reference) for reference in references
            ):
                raise ValueError("candidate must differ from every reference Harness")
            registered = self.workspace.bundle(
                self.workspace.put_bundle(candidate.bundle)
            )
            try:
                await self._evaluate_content(
                    state,
                    registered,
                    f"round-{proposal.round_index + 1:02d}-author-{proposal.author_index + 1:02d}",
                )
            except Exception as error:
                return result, registered, error
            return result, registered, None

        results = await asyncio.gather(
            *(branch(proposal) for proposal in proposals), return_exceptions=True
        )
        errors = []
        for proposal, outcome in zip(proposals, results, strict=True):
            if isinstance(outcome, BaseException):
                errors.append(outcome)
                continue
            if outcome is None:
                continue
            result, registered, evaluation_error = outcome
            (candidate,) = result.candidates
            self.workspace.add_version(
                state,
                BundleVersion(
                    record_id=f"version/{proposal.record_id}/{candidate.branch_id}",
                    bundle_id=registered.record_id,
                    proposal_id=proposal.record_id,
                    branch_id=candidate.branch_id,
                ),
            )
            if evaluation_error is not None:
                errors.append(evaluation_error)
        if errors:
            raise errors[0]
        await self._settle_round(state, proposals)

    async def _consolidate_successes(self, state):
        """Publish reusable source after actual success, with no inherited score.

        The existing native attempt journal owns interruption and same-session
        recovery. Consolidated content is an artifact, never a measured version.
        """
        if (
            not self.workspace.protocol.success_consolidation
            or self.workspace.protocol.mutation_scopes != ("W", "C", "L")
        ):
            # Consolidation extracts global C/L and could change frozen scopes.
            # Its output is unmeasured and never participates in final selection.
            return
        records = await run_blocking(self._records, state)
        protocol = self.workspace.protocol
        round_count = (
            len(state["proposals"])
            if protocol.evolution_mode == "no_topology"
            else len(state["rounds"])
        )
        bundles = tuple(v["bundle_id"] for v in state["versions"].values())
        executions = self.workspace.execution_ids(state)
        development = result_matrix(
            protocol, bundles, records, execution_ids=executions
        )
        validation = result_matrix(
            protocol, bundles, records, split="validation", execution_ids=executions
        )
        seen = set()
        for index, version_id in enumerate(state["order"]):
            version = state["versions"][version_id]
            if version["bundle_id"] in seen:
                continue
            seen.add(version["bundle_id"])
            if not target_passed(
                protocol,
                development[version["bundle_id"]],
                validation[version["bundle_id"]],
            ):
                continue
            if self.author is None:
                raise EvolutionIncomplete(
                    "successful harness requires a native consolidation author"
                )
            key = f"consolidation/V{index:03d}"
            archived = state["attempts"].get(key)
            if archived is not None:
                proposal = archived_proposal(archived["request"])
            else:
                history = await self._freeze_history(
                    state,
                    await self.matrix(state),
                    round_count,
                    identity=f"success_V{index:03d}",
                )
                parent = VersionReference(
                    version_id=version_id, bundle_id=version["bundle_id"]
                )
                proposal = Proposal(
                    record_id=f"success_V{index:03d}",
                    purpose="success_consolidation",
                    parent=parent,
                    references=(parent,),
                    scopes=("W", "C", "L"),
                    evidence=records,
                    history=history,
                    task_protocol=self.workspace.protocol,
                    round_index=round_count,
                    author_index=0,
                )

            async def invoke(directory, before_launch, proposal=proposal):
                return await self.author(
                    proposal, directory, before_launch=before_launch
                )

            result = await self._attempt(
                state,
                "author",
                key,
                proposal,
                invoke,
                AuthoredMutation,
                verify_result=self.author.verify_result,
                wall_time_s=self.workspace.protocol.author_wall_time_s,
            )
            if result.proposal_id != proposal.record_id:
                raise ValueError("consolidation receipt answers a different request")
            (candidate,) = result.candidates
            candidate.bundle.require_success_layers()
            candidate.bundle.assert_candidate(
                (self.workspace.bundle(proposal.parent.bundle_id),),
                frozenset(proposal.scopes),
            )
            published = self.workspace.put_bundle(candidate.bundle)
            publication = {
                "winning_bundle_id": proposal.parent.bundle_id,
                "organized_bundle_id": published,
                "author_receipt": str(
                    (
                        self.workspace.attempt_directory(key, state["attempts"][key])
                        / "result.json"
                    ).resolve()
                ),
                "official_success": None,
                "meaning": "Organized source passed synthetic admission. Physical success belongs only to the original winning episode.",
            }
            path = self.workspace.root / "publications" / f"V{index:03d}.json"
            if path.exists() and json.loads(path.read_text()) != publication:
                raise ValueError("successful harness publication changed")
            atomic_json(path, publication)

    async def run(self) -> dict[str, Any]:
        no_topology = self.workspace.protocol.evolution_mode == "no_topology"
        if no_topology and self.analyzer is not None:
            raise ValueError("w/o topology cannot receive an analyzer")
        self.workspace.require_configuration()
        with self.workspace.transaction() as state:
            self.workspace.put_bundle(self.workspace.baseline)
            initial = self.workspace.add_version(
                state,
                BundleVersion(
                    record_id="baseline",
                    bundle_id=self.workspace.baseline.record_id,
                    proposal_id=None,
                    branch_id=None,
                ),
            )
            from reweave.evolution.search.report import publish_report

            if state["completed"]:
                await self._consolidate_successes(state)
                records = await run_blocking(self._records, state)
                publish_report(self.workspace, state, records, self.accounting(state))
                return state
            try:
                await self._evaluate(state, initial)
                rounds = {}
                for value in state["proposals"].values():
                    p = archived_proposal(value)
                    rounds.setdefault(p.round_index, []).append(p)
                if rounds:
                    await self._finish_round(
                        state, sorted(rounds[max(rounds)], key=lambda p: p.author_index)
                    )
                reason = None
                while reason is None:
                    await self._consolidate_successes(state)
                    rows = await self.matrix(state)
                    validation_rows = await self.matrix(state, split="validation")
                    versions = {
                        k: BundleVersion.model_validate(v)
                        for k, v in state["versions"].items()
                    }
                    if any(
                        target_passed(
                            self.workspace.protocol,
                            rows[v.bundle_id],
                            validation_rows[v.bundle_id],
                        )
                        for v in versions.values()
                    ):
                        reason = "target_reached"
                        break
                    if len(rounds) >= self.workspace.protocol.max_rounds:
                        reason = "round_limit"
                        break
                    proposals = await self._proposals(state, rows, len(rounds))
                    rounds[len(rounds)] = proposals
                    await self._finish_round(state, proposals)
                if not no_topology:
                    await self._finalize_topology(state)
            except BaseException:
                records = await run_blocking(self._records, state)
                publish_report(self.workspace, state, records, self.accounting(state))
                raise
            state["completed"] = True
            state["stop_reason"] = reason
            # A report can be reconstructed after interruption without relaunching work.
            records = await run_blocking(self._records, state)
            publish_report(self.workspace, state, records, self.accounting(state))
            self.workspace.save(state)
            return EvolutionState.model_validate(state).model_dump(mode="json")
