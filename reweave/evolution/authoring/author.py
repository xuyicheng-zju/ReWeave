# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Native offline authors over an isolated W/C/L candidate and complete dev trace."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaError
from pydantic import TypeAdapter

from reweave.evolution.authoring.candidate import (
    author_plan_schema,
    author_write_roots,
    authoring_scope_contract,
    candidate_summary,
    host_input,
    initialize_candidate,
    output_inventory,
    read_proposal,
    staged_host_input,
    write_bundle_source,
    write_context,
    write_loop,
    write_workflow,
)
from reweave.evolution.authoring.config import (
    AuthorPolicy,
    connection_changed,
    recorded_sdk_cost_usd,
)
from reweave.evolution.authoring.fixtures import SUITE, history_parent_interactions
from reweave.evolution.authoring.prompts import (
    author_instructions,
    author_reference_documents,
    author_request,
    development_guide,
    runtime_contract,
    runtime_interfaces,
)
from reweave.evolution.authoring.references import AuthorReferences, history_originals
from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.authoring.workspace import (
    claude_workspace_options,
    native_output,
)
from reweave.evolution.bundle import AgentBundle, new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    EvidenceFile,
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    Proposal,
    RoundDecision,
    Usage,
)
from reweave.evolution.evidence.preparation import (
    _evidence_roots,
    _history_paths,
    _native_evidence_paths,
    author_visible_sources,
    bind_frozen,
    declared_originals,
    episode_gateway,
    fixed_gateway,
    history_index,
    publish_trace_context,
    request_protocol,
    retained_navigation,
    task_instructions,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.native_accounting import (
    complete_query,
    cumulative_usage,
    failed_initializations,
    reconcile_interrupted_checks,
    reconcile_interrupted_timing,
    with_admission_usage,
)
from reweave.evolution.infrastructure.native_accounting import (
    native_completion as native_completion,  # receipt writer shared with callers
)
from reweave.evolution.infrastructure.native_codex import CodexSession
from reweave.evolution.infrastructure.native_retry import (
    NATIVE_RETRY_LIMIT,
    NativeTransportFailure,
    is_retryable_native_error,
    native_retry_wait_s,
)
from reweave.evolution.infrastructure.session import (
    ClaudeSessionStore,
    native_event_field,
    serialize_native_event,
)
from reweave.evolution.validation.admission import admit_mutation
from reweave.utils.persistence import atomic_json, atomic_text

AUTHOR_VALIDATION_ERRORS = (
    ValueError,
    OSError,
    KeyError,
    SchemaError,
    JsonSchemaError,
    subprocess.TimeoutExpired,
    RuntimeError,
    TimeoutError,
)


class AuthorCorrectionRequired(RuntimeError):
    """A completed native query left invalid output that must be repaired."""


def _artifact(path: Path) -> EvidenceFile:
    return EvidenceFile(**artifact_record(path))


def _outside_evidence_roots(paths, roots):
    """Keep the same exact-file grants without scanning every root per file."""
    roots = frozenset(roots)
    return tuple(
        path for path in paths if path not in roots and roots.isdisjoint(path.parents)
    )


def _publication_paths(directory: Path) -> tuple[Path, ...]:
    """The complete authority of an authored result, excluding its own receipt."""
    return (
        directory / "native_result.json",
        *(
            (directory / "workspace" / SUITE,)
            if (directory / "workspace" / SUITE).is_file()
            else ()
        ),
        *(
            path
            for root in (
                directory / "audit",
                directory / "session",
                native_output(directory / "workspace"),
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and not path.is_relative_to(directory / "audit/session_inquiry/workspace")
        ),
    )


def _host_usage(wall_time_s: float, *, complete: bool = True) -> dict:
    """A Host-owned interval: measured time, no model call and no spend."""
    return Usage(
        physical_steps=0,
        model_calls=0,
        wall_time_s=wall_time_s,
        wall_time_complete=complete,
        cost_usd=0,
    ).model_dump(mode="json")


def system_read_roots() -> tuple[Path, ...]:
    """Runtime libraries are readable; neither home nor the repository root is."""
    return tuple(
        Path(p).resolve()
        for p in (
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/etc",
            "/System",
            "/dev",
            # /etc/resolv.conf links here on systemd hosts; without it DNS fails (EAI_AGAIN).
            "/run/systemd/resolve",
        )
        if Path(p).exists()
    )


def _prepared_evidence_access(request, workspace):
    prepared_access = host_input(workspace, "evidence_access.json")
    if prepared_access.is_file():
        access = json.loads(prepared_access.read_text())
        evidence_roots = {
            key: Path(root) for key, root in access["evidence_roots"].items()
        }
        granted = tuple(Path(path) for path in access["readable_files"])
    else:
        # Historical prepared workspaces keep their original read contract.
        evidence_roots = _evidence_roots(request)
        granted = (
            *_native_evidence_paths(request, evidence_roots),
            *_outside_evidence_roots(_history_paths(request), evidence_roots.values()),
        )
    return evidence_roots, granted


class NativeSession(CodexSession):
    """Shared native execution, recovery and receipts for authors and graph analysis."""

    def __init__(
        self,
        *,
        bundle_loader,
        policy: AuthorPolicy,
        evidence_workspace: Path,
        native_runtime_root=None,
        runtime_manifest=None,
    ):
        # Run-owned deterministic navigation, shared read-only across sessions.
        # Each settled episode is indexed once; this contains no model memory.
        self.evidence_workspace = evidence_workspace.resolve()
        self.native_runtime_root = native_runtime_root
        self.bundle_loader = bundle_loader
        self.policy = policy
        self.runtime_manifest = runtime_manifest

    def runtime_schema(self, controller_request_unit, scopes=("W", "C", "L")):
        """The fixed runtime and interaction authoring contract for every author adapter."""
        from reweave.evolution.bundle import (
            NativeModelContext,
            PluginCase,
            WorkflowResult,
        )
        from reweave.evolution.validation.interaction import InteractionScenario
        from reweave.runtime.capability import PhysicalBudget

        return {
            "model_context": NativeModelContext.model_json_schema(),
            "plugin_case": PluginCase.model_json_schema(),
            "workflow_result": WorkflowResult.model_json_schema(),
            "interaction_scenario": InteractionScenario.model_json_schema(),
            "physical_budget": TypeAdapter(PhysicalBudget | None).json_schema(),
            "interfaces": runtime_interfaces(controller_request_unit, scopes=scopes),
            **(
                {"authoring_authority": authoring_scope_contract(scopes)}
                if scopes != ("W", "C", "L")
                else {}
            ),
            "execution_contract": runtime_contract(controller_request_unit),
        }

    def _update_usage(self, directory: Path) -> Usage:
        audit = directory / "audit"
        usage = with_admission_usage(cumulative_usage(audit), audit)
        atomic_json(audit / "usage.json", usage.model_dump(mode="json"))
        return usage

    async def __call__(
        self,
        proposal: Proposal | GraphAnalysisRequest,
        directory: Path,
        *,
        resume: bool = False,
        before_launch: Callable[[], None] | None = None,
    ) -> AuthoredMutation | GraphAnalysisReceipt:
        return await self._run_native(
            proposal, directory, resume=resume, before_launch=before_launch
        )

    async def _run_native(
        self,
        request: Proposal | GraphAnalysisRequest,
        directory: Path,
        *,
        resume: bool,
        before_launch: Callable[[], None] | None,
    ) -> AuthoredMutation | GraphAnalysisReceipt:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".native.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "native author job is already owned by another process"
                ) from error
            try:
                timing = (time.monotonic(), time.time())
                self._verify_process_exit(directory)
                if (directory / "result.json").exists():
                    # Checking an immutable receipt must not append new audit
                    # entries or start another author query, even on rejection.
                    return await run_blocking(self.verify_result, request, directory)
                if resume:
                    audit = directory / "audit"
                    await run_blocking(reconcile_interrupted_checks, audit)
                    if (audit / "queries").exists():
                        self._update_usage(directory)
                    await run_blocking(reconcile_interrupted_timing, directory)
                while True:
                    try:
                        return await self._execute(
                            request,
                            directory,
                            resume=resume,
                            before_launch=before_launch,
                            timing_start=timing,
                        )
                    except AuthorCorrectionRequired:
                        # Same job lock, workspace, session and cumulative
                        # accounting. Invalid code is never a finished candidate.
                        pass
                    except NativeTransportFailure as failure:
                        if failure.task_directory != directory.resolve():
                            raise
                        await self._wait_for_transport_retry(directory, failure)
                    resume = True
                    timing = (time.monotonic(), time.time())
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    async def _wait_for_transport_retry(
        self, directory: Path, failure: NativeTransportFailure
    ) -> None:
        """Journal one bounded same-session retry and charge its backoff wait."""
        audit = directory / "audit"
        ledger = audit / "native_transport_retries.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        failures = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        if failures >= NATIVE_RETRY_LIMIT:
            raise failure.original
        attempt = failures + 1
        wait_s = native_retry_wait_s(attempt)
        identity = directory / "session/identity.json"
        event = {
            "event": "native_transport_retry_scheduled",
            "attempt": attempt,
            "maximum": NATIVE_RETRY_LIMIT,
            "error_type": type(failure.original).__name__,
            "error_bytes": len(str(failure.original).encode()),
            "retry_wait_s": wait_s,
            "same_session": identity.exists(),
            "session_id": json.loads(identity.read_text())["session_id"]
            if identity.exists()
            else None,
        }
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with (audit / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        retry_usage = audit / "retry_usage"
        retry_usage.mkdir(exist_ok=True)
        usage_path = retry_usage / f"{attempt:06d}.json"
        wait_started = time.monotonic()
        atomic_json(usage_path, _host_usage(0, complete=False))
        try:
            await asyncio.sleep(wait_s)
        finally:
            atomic_json(usage_path, _host_usage(time.monotonic() - wait_started))
            self._update_usage(directory)

    def _prepare_current_contract(self, protocol, context):
        """Publish the active fixed gateway and task rules for every author mode."""
        active_contract = (
            self.runtime_manifest.authoring_contract
            if self.runtime_manifest is not None
            else None
        )
        if protocol.execution_runtime is not None and active_contract is None:
            raise ValueError(
                "execution revision requires the active RuntimeManifest authoring contract"
            )
        if active_contract is not None:
            atomic_json(
                context / "gateway.json",
                {
                    "operations": active_contract.operations,
                    "tools": active_contract.tools,
                    **(
                        {"native_image_contract": active_contract.native_image_contract}
                        if active_contract.native_image_contract is not None
                        else {}
                    ),
                    "episode_limits": self.runtime_manifest.configuration["limits"][
                        "episode"
                    ],
                },
            )
            atomic_json(
                context / "capability_catalog.json", active_contract.capability_catalog
            )
        self._publish_task_rules(protocol, context)
        return active_contract

    def _publish_task_rules(self, protocol, context):
        configured_rules = (
            protocol.execution_runtime.configuration.get("task_rule_supplements")
            if protocol.execution_runtime is not None
            else None
        )
        if configured_rules is None and self.runtime_manifest is not None:
            configured_rules = self.runtime_manifest.configuration.get(
                "task_rule_supplements"
            )
        if configured_rules is not None:
            selected_tasks = sorted({instance.task for instance in protocol.instances})
            missing = [task for task in selected_tasks if task not in configured_rules]
            if missing:
                raise ValueError(
                    f"current task-rule configuration is missing protocol tasks: {missing}"
                )
            path = context / "current_task_rules.json"
            current = {
                "semantics": (
                    "Current published rules for this new authoring request. "
                    "Historical episode instructions remain unchanged."
                ),
                "values": [configured_rules[task] for task in selected_tasks],
            }
            if path.exists() and json.loads(path.read_text()) != current:
                prior = (
                    context / "task_contract_revisions" / (uuid.uuid4().hex + ".json")
                )
                prior.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, prior)
            atomic_json(path, current)

    def _prepare_evidence(self, request, workspace):
        context = workspace / "context"
        evidence_roots = _evidence_roots(request)
        protocol = request_protocol(request)
        active_contract = self._prepare_current_contract(protocol, context)
        originals = history_originals(request)
        if isinstance(request, Proposal):
            history_parent_interactions(workspace, request.parent.bundle_id, originals)
        references = AuthorReferences.create(
            workspace,
            registry=self.evidence_workspace / "author_references.json",
            evidence_roots=evidence_roots,
            originals=declared_originals(request, originals["files"]),
        )
        with references.batch():
            result = self._publish_evidence(
                request,
                workspace,
                references,
                evidence_roots,
                originals,
                active_contract,
                protocol,
            )
        # Reuse the exact permission inventory already read for preparation. This
        # private view grants no additional files and is bound to this request.
        atomic_json(
            workspace / ".host_inputs/evidence_access.json",
            {
                "evidence_roots": {
                    key: str(root) for key, root in evidence_roots.items()
                },
                "readable_files": [
                    str(path)
                    for path in (
                        *_native_evidence_paths(request, evidence_roots),
                        *_outside_evidence_roots(
                            (Path(path) for path in originals["files"]),
                            evidence_roots.values(),
                        ),
                    )
                ],
            },
        )
        return result

    def _publish_evidence(
        self,
        request,
        workspace,
        references,
        evidence_roots,
        originals,
        active_contract,
        protocol,
    ):
        context = workspace / "context"
        private = workspace / ".host_inputs"
        demonstrations = originals["navigation"].get("demonstrations", [])
        if demonstrations:
            atomic_json(
                context / "demonstrations.json",
                {"videos": references.encode(demonstrations)},
            )
        navigation = []
        executed_sources = {}
        source_inventory = {}
        instructions = {}
        for record in request.evidence:
            episode = record.episode
            if episode is None:
                navigation.append(retained_navigation(record, references))
                continue
            source_root = evidence_roots[episode.evidence_id]
            gateway = episode_gateway(record, source_root)
            sources = author_visible_sources(episode)
            # This source is an existing component of the published episode.
            # Repreparation after private inputs were committed must retain its
            # reference, even if the visible workspace was not yet published.
            source_id = f"{episode.evidence_id}/implementation"
            if source_id not in executed_sources:
                executed_sources[source_id] = {
                    "path": str(source_root / "harness_context/source"),
                    "files": sources,
                }
                source_inventory[references.identity("source", source_id)] = sorted(
                    sources
                )
            gateway = fixed_gateway(gateway, protocol.execution, source_root)
            if active_contract is None:
                bind_frozen(
                    context / "gateway.json",
                    gateway,
                    "historical evidence contains different frozen gateway conditions",
                )
            capability_catalog = json.loads(
                (source_root / "harness_context/capability_catalog.json").read_text()
            )
            if active_contract is None:
                bind_frozen(
                    context / "capability_catalog.json",
                    capability_catalog,
                    "historical evidence contains different capability catalogs",
                )
            trace_id, trace_index = publish_trace_context(
                self.evidence_workspace,
                episode,
                source_root,
                private,
                context / "trace",
                references,
            )
            for item in trace_index["task_instructions"]["values"]:
                key = (
                    item["official_text"],
                    item["task_rule_supplement"],
                    item["text"],
                )
                instructions.setdefault(key, []).append(record.record_id)
            navigation.append(
                {
                    "episode_id": record.record_id,
                    "bundle_id": record.request.bundle_id,
                    "executed_source": source_id,
                    "condition_id": record.request.instance.record_id,
                    "failure_origin": record.failure_origin,
                    "evidence_id": episode.evidence_id,
                    "trace_context": f"context/trace/{trace_id}/index.json",
                    "original_directory": str(source_root),
                    "official_success": record.official_success,
                    "recorded_outcome": trace_index["outcome"],
                }
            )
        implementation = {
            "executed_sources": "context/source_inventory.json",
            "meaning": "Full episode archives name their exact historical source snapshot. Retained-file-only records explicitly lack that snapshot. Current author contracts are the schemas, examples and Host tools in context/.",
        }
        atomic_json(private / "executed_sources.json", executed_sources)
        atomic_json(context / "source_inventory.json", source_inventory)
        atomic_json(
            context / "task_instructions.json",
            task_instructions(instructions, references),
        )
        summary = Path(request.history.summary.path).read_bytes().decode("utf-8")
        request.history.verify()
        history = history_index(originals, references)
        atomic_text(context / "EVOLUTION_HISTORY.md", summary)
        atomic_json(context / "history_originals.json", history)
        return navigation, implementation

    def _check_request(self, request, workspace: Path) -> None:
        # An ordinary record ID references saved input; it cannot replace the
        # actual request comparison required before reusing a cached operation.
        saved = self._read_request(workspace).model_dump(mode="json")
        if not Draft202012Validator({"const": saved}).is_valid(
            request.model_dump(mode="json")
        ):
            raise ValueError("native attempt input differs from its saved request")

    def _prepare_inputs(
        self, request: Proposal | GraphAnalysisRequest, directory: Path, *, resume: bool
    ) -> None:
        workspace = directory / "workspace"
        if workspace.exists() and not resume:
            raise ValueError(
                "existing native attempt requires explicit resume, never a new session"
            )
        if workspace.exists() or (directory / "audit/inputs").exists():
            self._check_request(request, workspace)
        was_prepared = workspace.exists()
        if not workspace.exists():
            staging = directory / ".scratch/runtime/preparation"
            if staging.exists():
                # A failed preparation is evidence, even without a native query.
                # Keep its exact tree outside the next author's visible inputs.
                preserved = directory / "audit/preparation_failures" / uuid.uuid4().hex
                preserved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, preserved)
            self._prepare(request, staging)
            private = staging / ".host_inputs"
            host_inputs = directory / "audit/inputs"
            if host_inputs.exists():
                shutil.rmtree(private)
            else:
                host_inputs.parent.mkdir(parents=True, exist_ok=True)
                os.replace(private, host_inputs)
            os.replace(staging, workspace)
        if was_prepared:
            self._publish_task_rules(request_protocol(request), workspace / "context")
        if (workspace / "context/current_task_rules.json").exists():
            input_path = workspace / "context/input.json"
            value = json.loads(input_path.read_text())
            value["current_task_rules"] = "context/current_task_rules.json"
            atomic_json(input_path, value)
        if self.runtime_manifest is not None:
            task = self.runtime_manifest.configuration.get("g2_task")
            if task is not None:
                path = workspace / "context/current_task.json"
                if path.exists() and json.loads(path.read_text()) != task:
                    prior = (
                        directory
                        / "audit/task_contract_revisions"
                        / (uuid.uuid4().hex + ".json")
                    )
                    prior.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, prior)
                atomic_json(path, task)
                input_path = workspace / "context/input.json"
                value = json.loads(input_path.read_text())
                value["current_task"] = "context/current_task.json"
                atomic_json(input_path, value)

    async def _execute(
        self,
        request: Proposal | GraphAnalysisRequest,
        directory: Path,
        *,
        resume: bool = False,
        before_launch: Callable[[], None] | None = None,
        timing_start: tuple[float, float] | None = None,
    ) -> AuthoredMutation | GraphAnalysisReceipt:
        """Run one author invocation and account any work before a query exists.

        Query and admission journals own their own intervals.  If an invocation
        fails before either journal creates a new entry, preserve one complete
        pre-query interval after all cleanup has finished.  Snapshotting paths
        per invocation matters on recovery because older query/check files may
        already exist.
        """
        started, started_at = timing_start or (time.monotonic(), time.time())
        audit = directory / "audit"

        def accounting_entries() -> set[Path]:
            return {
                path
                for child in ("preparation_usage", "queries", "checks")
                for path in (audit / child).glob("*.json")
            }

        entries_before = accounting_entries()
        try:
            return await self._execute_once(
                request,
                directory,
                resume=resume,
                before_launch=before_launch,
                timing_start=(started, started_at),
            )
        except BaseException as error:
            if accounting_entries() == entries_before:
                self._charge_unqueried_interval(directory, "prequery", error, started)
            raise

    def _charge_unqueried_interval(
        self, directory: Path, phase: str, error: BaseException, started: float
    ) -> None:
        """Charge a failed interval that no query or admission journal owns."""
        audit = directory / "audit"
        entries = audit / "preparation_usage"
        entries.mkdir(parents=True, exist_ok=True)
        usage_path = entries / f"{len(list(entries.glob('*.json'))):06d}.json"
        # begin_query is the sole model-launch boundary. The Host may have
        # initialized local clients/sessions, but no generative author call
        # exists before that callback.
        atomic_json(usage_path, _host_usage(time.monotonic() - started))
        self._update_usage(directory)
        with (audit / "phases.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "phase": phase,
                        "event": "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed",
                        "timestamp": time.time(),
                        "error_type": type(error).__name__,
                        "usage": usage_path.relative_to(directory).as_posix(),
                    }
                )
                + "\n"
            )

    async def _execute_once(
        self,
        request: Proposal | GraphAnalysisRequest,
        directory: Path,
        *,
        resume: bool = False,
        before_launch: Callable[[], None] | None = None,
        timing_start: tuple[float, float] | None = None,
    ) -> AuthoredMutation | GraphAnalysisReceipt:
        started, started_at = timing_start or (time.monotonic(), time.time())
        context_path = directory / "audit/context.json"
        if context_path.is_file():
            if connection_changed(
                json.loads(context_path.read_text())["policy"], self.policy
            ):
                raise ValueError(
                    "native author connection or model differs from its existing session"
                )
        elif (
            self.policy.author == "codex"
            and (directory / "session/identity.json").exists()
        ):
            raise ValueError(
                "existing native author session lacks its connection binding"
            )
        workspace = directory / "workspace"
        phases = directory / "audit/phases.jsonl"
        phases.parent.mkdir(parents=True, exist_ok=True)

        def phase(name: str, event: str, **details: Any) -> None:
            with phases.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "phase": name,
                            "event": event,
                            "timestamp": time.time(),
                            **details,
                        }
                    )
                    + "\n"
                )

        phase("preparation", "started")
        try:
            await run_blocking(self._prepare_inputs, request, directory, resume=resume)
        except BaseException as error:
            # The evolution attempt journal owns the complete exception. This
            # phase boundary also exists for direct NativeBundleAuthor callers.
            self._charge_unqueried_interval(directory, "preparation", error, started)
            raise
        phase("preparation", "completed")
        output = native_output(workspace)
        writable = (
            author_write_roots(request, output)
            if isinstance(request, Proposal)
            else (output,)
        )
        instructions, native_prompt = self._prompts(request, workspace)
        correction = await run_blocking(self._resume_correction, directory)
        if correction is not None:
            catalog = AuthorReferences.load(workspace)
            native_prompt = self._correction_prompt(catalog.encode(correction))
            catalog.save()
        native_receipt = directory / "native_result.json"
        if native_receipt.exists():
            return await self._publish(
                directory,
                timing_start=(started, started_at),
                before_launch=before_launch,
            )
        # Runtime libraries are readable; neither home nor the repository root is.
        evidence_roots, granted = _prepared_evidence_access(request, workspace)
        readable = (
            *system_read_roots(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            *granted,
            directory / "audit/corrections",
            directory / "audit/development_reports",
        )
        (directory / "audit/corrections").mkdir(parents=True, exist_ok=True)
        (directory / "audit/development_reports").mkdir(parents=True, exist_ok=True)
        audit = directory / "audit"
        audit.mkdir(exist_ok=True)
        events = audit / "events.jsonl"

        def record_event(value: Any) -> Any:
            serialized = serialize_native_event(value)
            with events.open("a") as stream:
                stream.write(json.dumps(serialized, ensure_ascii=False) + "\n")
            return serialized

        atomic_json(
            audit / "context.json",
            {
                "instructions": instructions,
                "policy": self.policy.model_dump(mode="json"),
                "request_id": request.record_id,
            },
        )
        queries = audit / "queries"
        queries.mkdir(exist_ok=True)
        query_path = queries / f"{len(list(queries.glob('*.json'))):06d}.json"

        def begin_query() -> None:
            if before_launch is not None:
                before_launch()
            phase("native_query", "started")
            atomic_json(
                query_path,
                Usage(
                    physical_steps=0,
                    model_calls=None,
                    wall_time_s=0,
                    wall_time_complete=False,
                    cost_usd=None,
                ).model_dump(mode="json"),
            )
            self._update_usage(directory)
            record(
                {
                    "event": "native_query_started",
                    "query": query_path.name,
                    "started_at": started_at,
                }
            )

        first_native_event = True

        def record(value: Any) -> None:
            nonlocal first_native_event
            serialized = record_event(value)
            if first_native_event:
                phase("native_query", "first_event")
                first_native_event = False
            # Keep every native event. Text deltas carry no new usage or state;
            # only the derived live wall-time lower bound would change. Unknown
            # payload fields still take the normal durable snapshot path.
            # Live wall time remains an explicit lower bound between messages;
            # completion/failure always commits the measured full duration.
            payload = (
                serialized.get("payload") if isinstance(serialized, dict) else None
            )
            text_delta = (
                self.policy.author == "codex"
                and isinstance(serialized, dict)
                and serialized.get("method") == "item/agentMessage/delta"
                and isinstance(payload, dict)
                and set(payload) == {"delta", "item_id", "thread_id", "turn_id"}
                and all(isinstance(item, str) for item in payload.values())
            )
            cursor_delta = (
                self.policy.author == "cursor"
                and isinstance(serialized, dict)
                and serialized.get("type") == "thinking"
                and serialized.get("subtype") == "delta"
                and set(serialized)
                <= {"type", "subtype", "text", "session_id", "timestamp_ms"}
            )
            if (
                native_event_field(value, "subtype") == "thinking_tokens"
                or text_delta
                or cursor_delta
            ):
                return
            current = Usage.model_validate_json(query_path.read_text()).model_copy(
                update={"wall_time_s": time.monotonic() - started}
            )
            atomic_json(query_path, current.model_dump(mode="json"))
            self._update_usage(directory)

        development = AuthorDevelopmentTools(
            workspace,
            audit / "development.jsonl",
            evidence_roots=evidence_roots,
            retrieval_config=self.policy.graph_retrieval,
        )
        try:
            phase("development_preflight", "started")
            await run_blocking(development.server.start)
            await run_blocking(
                development.preflight,
                readable=readable,
                writable=writable,
                control=directory / "launcher/development_preflight",
                backend="claude"
                if self.policy.author == "claude_code"
                else self.policy.author,
            )
            phase("development_preflight", "completed")
            if self.policy.author == "claude_code":
                import claude_agent_sdk

                from reweave.planner.codex_connection import claude_environment

                store = ClaudeSessionStore(directory / "session")
                identity_path = directory / "session/identity.json"
                identity = (
                    json.loads(identity_path.read_text())["session_id"]
                    if identity_path.exists()
                    else None
                )
                if identity is not None:
                    await run_blocking(store.bind, identity)
                    await run_blocking(store.checkpoint)
                elif resume and (
                    store.path.exists()
                    or any((workspace / ".scratch/runtime/claude").rglob("*.jsonl"))
                ):
                    raise ValueError(
                        "native session evidence exists without a recoverable identity"
                    )
                elif resume:
                    failed = failed_initializations(audit)
                    if failed:
                        record_event(
                            {
                                "event": "native_initialization_restarted",
                                "failed_queries": failed,
                                "usage_preserved": True,
                                "reason": "explicit recovery of terminal SDK initialization failures; no native journal or identity exists",
                            }
                        )
                workspace_options = claude_workspace_options(
                    workspace,
                    native_output(workspace),
                    directory / "launcher",
                    readable_roots=readable,
                    writable_roots=writable,
                    native_runtime_root=self.native_runtime_root,
                )
                workspace_options["env"] = {
                    **workspace_options.get("env", {}),
                    **claude_environment(),
                    "REWEAVE_AUTHOR_TOOLS_URL": development.server.url,
                }
                options = claude_agent_sdk.ClaudeAgentOptions(
                    cwd=workspace,
                    session_store=store,
                    resume=identity,
                    session_store_flush="eager",
                    model=self.policy.model,
                    effort=self.policy.reasoning_effort,
                    system_prompt={
                        "type": "preset",
                        "preset": "claude_code",
                        "append": instructions,
                    },
                    mcp_servers={
                        "author": {"type": "http", "url": development.server.url}
                    },
                    **workspace_options,
                )
                result = None
                begin_query()
                async for message in claude_agent_sdk.query(
                    prompt=native_prompt,
                    options=options,
                ):
                    record(message)
                    if (
                        type(message).__name__ == "SystemMessage"
                        and message.subtype == "init"
                    ):
                        atomic_json(
                            identity_path, {"session_id": message.data["session_id"]}
                        )
                    if type(message).__name__ == "MirrorErrorMessage":
                        raise RuntimeError("native transcript mirror failed")
                    if type(message).__name__ == "ResultMessage":
                        result = message
                        cost, complete = recorded_sdk_cost_usd(message)
                        measured = Usage(
                            physical_steps=0,
                            model_calls=message.num_turns,
                            wall_time_complete=False,
                            wall_time_s=time.monotonic() - started,
                            cost_usd=cost if complete else None,
                        )
                        atomic_json(query_path, measured.model_dump(mode="json"))
                if result is None:
                    raise RuntimeError("native author ended without a result")
                cost, complete = recorded_sdk_cost_usd(result)
                session_id = result.session_id
                usage = Usage(
                    physical_steps=0,
                    model_calls=result.num_turns,
                    wall_time_complete=False,
                    wall_time_s=time.monotonic() - started,
                    cost_usd=cost if complete else None,
                )
                atomic_json(query_path, usage.model_dump(mode="json"))
                usage = cumulative_usage(audit)
                self._update_usage(directory)
                if result.is_error:
                    raise RuntimeError(
                        "native author returned an error; inspect its full transcript"
                    )
                await run_blocking(complete_query, directory, usage, session_id, store)
            else:
                if self.policy.author == "cursor":
                    from reweave.evolution.infrastructure.native_cursor import (
                        cursor_query,
                    )

                    query = partial(cursor_query, self)
                else:
                    query = self._codex
                usage, session_id = await query(
                    workspace,
                    directory,
                    readable,
                    writable,
                    record,
                    started,
                    query_path,
                    begin_query,
                    instructions,
                    native_prompt,
                    development.server.url,
                )
        except BaseException as error:
            if not query_path.exists():
                record_event(
                    {
                        "event": "author_preflight_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(traceback.format_exception(error)),
                    }
                )
            if not native_receipt.exists() and query_path.exists():
                frames = traceback.extract_tb(error.__traceback__)
                record_event(
                    {
                        "event": "native_query_failed",
                        "query": query_path.name,
                        "phase": "sdk_initialization"
                        if any(
                            frame.name == "initialize"
                            and "claude_agent_sdk" in frame.filename
                            for frame in frames
                        )
                        else "native_query",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(traceback.format_exception(error)),
                    }
                )
            if (
                isinstance(error, Exception)
                and query_path.exists()
                and not native_receipt.exists()
                and is_retryable_native_error(error)
            ):
                raise NativeTransportFailure(directory, error) from error
            raise
        finally:
            try:
                await run_blocking(development.server.stop, timeout_s=None)
            finally:
                if query_path.exists():
                    phase("native_query", "ended")
                    measured = Usage.model_validate_json(
                        query_path.read_text()
                    ).model_copy(
                        update={
                            "wall_time_s": time.monotonic() - started,
                            "wall_time_complete": True,
                        }
                    )
                    atomic_json(query_path, measured.model_dump(mode="json"))
                    self._update_usage(directory)
        if not native_receipt.exists():
            raise RuntimeError(
                "native query ended without a verified completion receipt"
            )
        return await self._publish(directory, before_launch=before_launch)

    async def recover(
        self,
        request: Proposal | GraphAnalysisRequest,
        directory: Path,
        *,
        before_launch: Callable[[], None],
    ) -> AuthoredMutation | GraphAnalysisReceipt:
        """Resume the same job, or finish Host publication without another query."""
        return await self._run_native(
            request, directory, resume=True, before_launch=before_launch
        )

    def _resume_correction(self, directory: Path) -> dict | None:
        """Finish a durable rejection transition before resuming the same author."""
        repairs = sorted((directory / "audit/corrections").glob("*/feedback.json"))
        if not repairs:
            return None
        path = repairs[-1]
        feedback = json.loads(path.read_text())
        rejected = path.parent / "native_result.json"
        completion = json.loads(rejected.read_text())
        receipt = directory / "native_result.json"
        if receipt.exists():
            if receipt.read_bytes() != rejected.read_bytes():
                # A later completed query awaits its own independent admission.
                return None
            self._load_completion(directory)
            receipt.unlink()
        identity = json.loads((directory / "session/identity.json").read_text())
        if identity["session_id"] != completion["session_id"]:
            raise ValueError("correction cannot replace the native session")
        return {
            "complete_report": {
                **artifact_record(path),
                "pointer": "",
            },
            "error_type": feedback["error_type"],
            "contract_case_results": [
                {
                    key: case[key]
                    for key in (
                        "scope",
                        "workflow_id",
                        "case_index",
                        "validation_status",
                    )
                    if key in case
                }
                for case in feedback["contract_case_results"]
            ],
        }

    def _save_correction(
        self,
        directory: Path,
        check: Path,
        error: BaseException,
        *,
        case_results: list[dict[str, Any]],
    ) -> None:
        """Archive rejected bytes and full error before allowing further edits."""
        self._load_completion(directory)
        destination = directory / "audit/corrections" / check.stem
        staging = directory / ".scratch/runtime/correction"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        # Preserve invalid link/special entries in the bound output inventory.
        # Copy regular file bytes only; never follow an authored link or FIFO.
        root = native_output(directory / "workspace")
        for name, item in output_inventory(root).items():
            if item.get("directory") is True:
                (staging / "output" / name).mkdir(parents=True, exist_ok=True)
            elif "data_base64" in item:
                target = staging / "output" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
        shutil.copyfile(
            directory / "native_result.json", staging / "native_result.json"
        )
        atomic_json(
            staging / "feedback.json",
            {
                "check": check.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(error)),
                "contract_case_results": case_results,
                "candidate_files": sorted(
                    path.relative_to(root).as_posix()
                    for path in root.rglob("*")
                    if path.is_file()
                ),
            },
        )
        destination.parent.mkdir(exist_ok=True)
        os.replace(staging, destination)

    @staticmethod
    def _verify_process_exit(directory: Path) -> None:
        receipt = directory / "launcher/process.json"
        if not receipt.exists():
            return
        process = json.loads(receipt.read_text())
        # The trusted owner publishes this only after all native writers exit.
        # Its historical PID may now belong to another process.
        if process.get("descendants_drained") is True:
            return
        try:
            os.kill(process["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(
                "previous native process still exists; refusing concurrent resume/publication"
            )
        # Linux owns a timestamp supervisor. Its death alone cannot certify
        # that detached native tools have stopped writing (for example SIGKILL).
        if sys.platform == "linux" or "descendants_drained" in process:
            raise RuntimeError(
                "native descendants have no confirmed drain; refusing resume/publication"
            )

    def _load_completion(self, directory: Path) -> dict[str, Any]:
        self._verify_process_exit(directory)
        return json.loads((directory / "native_result.json").read_text())

    def _correction_prompt(self, feedback: dict, *, guidance: str | None = None) -> str:
        if guidance is None:
            guidance = (
                "Repair the named failure in the existing candidate and fixtures; "
                "do not restart the investigation or regenerate unaffected files. "
                "Use phase=interaction for runtime fixtures or phase=code for components. "
                "The Host repeats final admission. "
            )
        return (
            "Host final admission rejected the last output. Continue this same "
            "session and repair the candidate using the complete validation "
            "feedback file referenced below. The reference and compact receipt are authoritative; "
            "read only the error, traceback or case fields needed to correct this failure. "
            "Preserve the permitted scope and original evidence. "
            + guidance
            + " Do not bypass validators "
            "or report unexecuted checks as passed. Unmeasured physical outcomes "
            "remain limitations, not a requirement to fabricate proof or start a "
            "rollout. This is Host validation feedback, not new task evidence or "
            "permission to change the fixed runtime.\n\n"
            + json.dumps(feedback, ensure_ascii=False, indent=2)
        )


class NativeBundleAuthor(NativeSession):
    """Complete W/C/L mutation job using the common durable native session."""

    @staticmethod
    def _read_request(workspace):
        return read_proposal(workspace)

    def _prepare_inputs(self, request, directory, *, resume):
        if (
            request.task_protocol.evolution_mode == "no_topology"
            and self.policy.graph_retrieval is not None
        ):
            raise ValueError("w/o topology cannot load a graph retrieval corpus")
        super()._prepare_inputs(request, directory, resume=resume)
        if request.scopes != ("W", "C", "L"):
            context = directory / "workspace/context"
            authority = authoring_scope_contract(request.scopes)
            if any(
                json.loads((context / name).read_text()).get("authoring_authority")
                != authority
                for name in ("input.json", "runtime.schema.json")
            ):
                raise ValueError(
                    "restricted author workspace lacks its matching component authority; retain the original attempt for review"
                )

    def _prompts(self, request, workspace):
        instructions = author_instructions(request)
        native_prompt = author_request(request)
        path = workspace / "context/current_task_rules.json"
        if path.is_file():
            current = json.loads(path.read_text())
            block = (
                "Current published task contract for this new authoring request "
                "(historical episodes retain their originally recorded text):\n"
                + json.dumps(current, ensure_ascii=False, indent=2)
            )
            instructions += "\n\n" + block
            native_prompt += (
                "\n\nRead context/current_task_rules.json before diagnosing the "
                "historical evidence. It is the current published rule contract "
                "for this request; do not apply it retroactively to old episodes."
            )
        if (workspace / "context/demonstrations.json").exists():
            instructions += (
                "\nThis mutation is explicitly conditioned on the original expert videos at the read-only paths below. "
                "Choose how to inspect the original files yourself with native media tools, shell/Python and the native image viewer. "
                "No frames or viewing order have been preselected. "
                "Inspect the graph analysis's demonstration_review.json through its history file handle. "
                "In candidate/SKILL.md and the plan's normal changes/summary, explain the core solution, possible task misunderstandings, "
                "frame-backed lessons, their exact C/W/L changes, and what remains untested. "
                "Replace contradicted old advice directly; do not append mutually inconsistent rules. "
                "Preserve live visual grounding: never copy demonstration-specific coordinates or claim its success as this candidate's result."
                + "\nOriginal demonstration videos:\n"
                + (workspace / "context/demonstrations.json").read_text()
            )
        return instructions, native_prompt

    def _read_result(self, directory: Path) -> AuthoredMutation:
        return AuthoredMutation.model_validate_json(
            (directory / "result.json").read_text()
        )

    def verify_result(self, proposal: Proposal, directory: Path) -> AuthoredMutation:
        """Read a completed receipt without rechecking mutable session artifacts."""
        result = self._read_result(directory)
        if result.proposal_id != proposal.record_id:
            raise ValueError("published author receipt belongs to another proposal")
        self._check_request(proposal, directory / "workspace")
        return result

    def _prepare(self, request: Proposal, workspace: Path) -> tuple[Path, ...]:
        references = {
            ref.bundle_id: self.bundle_loader(ref.bundle_id)
            for ref in request.references
        }
        if any(bundle.record_id != identity for identity, bundle in references.items()):
            raise ValueError("reference Harness version differs")
        context = workspace / "context"
        context.mkdir(parents=True, exist_ok=True)
        request_name = "proposal.json"
        private = staged_host_input(workspace, request_name).parent
        private.mkdir(parents=True, exist_ok=True)
        atomic_json(private / request_name, request.model_dump(mode="json"))
        atomic_json(
            private / "references.json",
            {key: bundle.model_dump(mode="json") for key, bundle in references.items()},
        )
        navigation, implementation = self._prepare_evidence(request, workspace)
        catalog = AuthorReferences.load(workspace)
        with catalog.batch():
            return self._publish_author_inputs(
                request, workspace, references, navigation, implementation, catalog
            )

    def _publish_author_inputs(
        self, request, workspace, references, navigation, implementation, catalog
    ):
        context = workspace / "context"
        for ref in request.references:
            source = (
                context / "references" / catalog.identity("version", ref.version_id)
            )
            bundle = references[ref.bundle_id]
            write_bundle_source(bundle, source)
        protocol = request.task_protocol
        validation = protocol.validation
        inputs = {
            "request": request.record_id,
            "kind": request.purpose,
            "evaluation": {
                "task": protocol.dev[0].task,
                "dev": {
                    "condition_ids": [instance.record_id for instance in protocol.dev],
                    "target_success_rate": protocol.target_success_rate,
                },
                "validation": (
                    {
                        "condition_ids": [
                            instance.record_id for instance in validation.conditions
                        ],
                        "target_success_rate": validation.target_success_rate,
                    }
                    if validation is not None
                    else None
                ),
                "execution": protocol.execution.model_dump(mode="json"),
                "author_wall_time_s": protocol.author_wall_time_s,
                "meaning": "One complete Bundle must pass dev, then the configured feedback-validation pool. Historical per-condition best results cannot be combined into its score. Unmeasured validation is unknown. Graph progress guides search; the fixed evaluator determines success.",
            },
            "graph": "context/graph.json",
            "execution_graph": "context/execution_graph.json",
            "progress_map": "context/progress_map.json",
            "search_review": "context/search_review.json"
            if request.purpose == "mutation"
            else None,
            "parent": request.parent.model_dump(mode="json"),
            "program_evidence": {
                "semantic": {
                    "tool": "read_graph",
                    "layer": "semantic",
                    "view": "program",
                },
                "measured": {
                    "tool": "read_graph",
                    "layer": "map",
                    "view": "program",
                },
                "meaning": "The parent is a starting implementation, not an evidence filter. All historical programs remain available with source Bundles, conditions, witnesses, costs, failures and same-episode continuations. Program provenance is not proof of isolated causality or an independently executable plugin.",
            },
            "comparisons": {
                "tool": "read_graph",
                "layer": "map",
                "view": "comparison",
                "identity": catalog.identity("version", request.parent.version_id),
            },
            "references": [
                {
                    **ref.model_dump(mode="json"),
                    "path": f"context/references/{catalog.identity('version', ref.version_id)}",
                    "components": list(references[ref.bundle_id].components()),
                    "differs_from_parent": sorted(
                        references[ref.bundle_id].novel_scopes(
                            (references[request.parent.bundle_id],)
                        )
                    ),
                }
                for ref in request.references
            ],
            "history": {
                "summary": "context/EVOLUTION_HISTORY.md",
                "originals": "context/history_originals.json",
            },
            "round_index": request.round_index,
            "index_semantics": "Observed execution facts; author hypotheses remain unverified. History survives reference reselection.",
            "scopes": request.scopes,
            **(
                {"authoring_authority": authoring_scope_contract(request.scopes)}
                if request.scopes != ("W", "C", "L")
                else {}
            ),
            "index": navigation,
            "task_instructions": "context/task_instructions.json",
            "current_task_rules": (
                "context/current_task_rules.json"
                if (context / "current_task_rules.json").is_file()
                else None
            ),
            "candidate": "candidate",
            "schema": "context/plan.schema.json",
            "complete_originals": True,
            "implementation": implementation,
            "capability_catalog": "context/capability_catalog.json",
        }
        if request.task_protocol.evolution_mode == "no_topology":
            for key in (
                "graph",
                "execution_graph",
                "progress_map",
                "search_review",
                "comparisons",
            ):
                inputs.pop(key, None)
            inputs["program_evidence"] = {
                "tool": "read_source",
                "meaning": inputs["program_evidence"]["meaning"],
            }
            inputs["evaluation"]["meaning"] = inputs["evaluation"]["meaning"].replace(
                "Graph progress guides search; the fixed evaluator determines success.",
                "The fixed evaluator determines success.",
            )
            inputs["topology_disabled"] = True
        atomic_json(context / "input.json", catalog.encode(inputs))
        if request.purpose == "mutation":
            request.graph.verify()
            request.decision.verify()
            decision = RoundDecision.model_validate_json(
                Path(request.decision.path).read_text()
            )
            analysis = GraphAnalysisReceipt.model_validate_json(
                Path(decision.analysis.path).read_text()
            )
            atomic_json(
                context / "search_review.json", catalog.encode(decision.search_review)
            )
            shared_matches = (
                Path(decision.analysis.path).parent / "audit/graph_retrieval.json"
            )
            if any(Path(item.path) == shared_matches for item in analysis.artifacts):
                # Only this round's published analysis may supply shared retrieval.
                # Each author's later searches stay private and do not overwrite it.
                atomic_json(
                    context / "reference_matches.json",
                    json.loads(shared_matches.read_text()),
                )
            atomic_json(
                context / "graph.json",
                catalog.encode(json.loads(Path(request.graph.path).read_text())),
            )
            request.execution_graph.verify()
            request.progress_map.verify()
            for name, original in (
                ("execution_graph.json", request.execution_graph),
                ("progress_map.json", request.progress_map),
            ):
                atomic_json(
                    context / name,
                    catalog.encode(json.loads(Path(original.path).read_text())),
                )
        atomic_json(context / "bundle.schema.json", AgentBundle.model_json_schema())
        atomic_json(
            context / "plan.schema.json",
            author_plan_schema(topology=request.graph is not None),
        )
        from reweave.evolution.authoring.client import development_client
        from reweave.evolution.authoring.examples import author_examples
        from reweave.evolution.bundle import (
            ContextPackage,
            LoopPolicy,
            WorkflowSpec,
        )

        examples = author_examples()
        for scope, key, model, writer in (
            ("W", "workflow", WorkflowSpec, write_workflow),
            ("C", "context", ContextPackage, write_context),
            ("L", "loop", LoopPolicy, write_loop),
        ):
            example = examples[scope]
            root = context / "examples" / scope
            if scope == "W":
                root = root / example[key]["workflow_id"]
            value = model.model_validate(example.pop(key))
            if scope == "C":
                writer(root, value, context / "examples/global/context")
            else:
                writer(root, value)
            example["path"] = root.relative_to(workspace).as_posix()
            if request.scopes != ("W", "C", "L"):
                example["editable"] = scope in request.scopes
                example["usage"] = (
                    "Editable component interface example"
                    if scope in request.scopes
                    else "Read-only deployed interface example; do not modify or replace this component"
                )
        atomic_json(context / "examples.json", examples)
        (context / "DEVELOPMENT.md").write_text(
            development_guide(
                request,
                sys.executable,
                graph_retrieval=self.policy.graph_retrieval is not None,
            )
        )
        for name, document in author_reference_documents(
            request, sys.executable
        ).items():
            (context / name).write_text(document)
        client_source = context / "host_client.py"
        client_source.write_bytes(Path(__file__).with_name("client.py").read_bytes())
        (context / "dev.py").write_text(development_client())
        atomic_json(
            context / "runtime.schema.json",
            self.runtime_schema(
                request.task_protocol.execution.controller_request_unit,
                scopes=request.scopes,
            ),
        )
        initialize_candidate(
            workspace / "candidate", references[request.parent.bundle_id]
        )
        if request.purpose == "success_consolidation":
            atomic_json(
                workspace / "candidate/task-specific/evidence.json",
                [
                    r.model_dump(mode="json")
                    for r in request.evidence
                    if r.official_success is True
                    and r.request.bundle_id == request.parent.bundle_id
                ],
            )
        for directory in ("analysis", "tests", "media", "runtime"):
            (workspace / ".scratch" / directory).mkdir(parents=True, exist_ok=True)
        return (workspace / "candidate",)

    @staticmethod
    def _collect_result(
        directory,
        candidates,
        plan,
        usage,
        session_id,
        *,
        deferred_artifacts=(),
    ):
        workspace = directory / "workspace"
        # Native output and evidence survive rejection, not just admitted assets.
        artifacts = tuple(
            _artifact(path)
            for path in _publication_paths(directory)
            if path not in deferred_artifacts
        )

        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=read_proposal(workspace).record_id,
            candidates=candidates,
            summary=plan,
            session_id=session_id,
            usage=usage,
            artifacts=artifacts,
        )

    async def _publish(
        self,
        directory: Path,
        *,
        timing_start: tuple[float, float] | None = None,
        before_launch: Callable[[], None] | None = None,
    ) -> AuthoredMutation:
        # Admission measures the contract and interaction checks. Only the final
        # accounting/receipt commit follows this interval.
        checked_at, check_started_at = timing_start or (time.monotonic(), time.time())
        workspace, audit = directory / "workspace", directory / "audit"
        self._verify_process_exit(directory)
        checks = audit / "checks"
        checks.mkdir(exist_ok=True)
        await run_blocking(reconcile_interrupted_checks, audit)
        check_path = checks / f"{len(list(checks.glob('*.json'))):06d}.json"
        case_results: list[dict[str, Any]] = []
        evidence_path = checks / check_path.stem / "result.json"
        evidence = {
            "validation_status": "running",
            "contract_case_results": case_results,
        }
        evidence_written = False
        check_record = {
            "wall_time_s": 0,
            "wall_time_complete": False,
            "started_at": check_started_at,
            "validation_status": "running",
        }
        atomic_json(check_path, check_record)
        usage = cumulative_usage(audit)

        def record(value: Any) -> None:
            with (audit / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")

        try:
            completion = await run_blocking(self._load_completion, directory)
            # Query journals are the accounting authority. The native receipt's
            # usage is only its commit-time snapshot: checkpoint/cleanup can add
            # measured time later without rewriting that identity-bound receipt.
            if not any((audit / "queries").glob("*.json")):
                raise ValueError(
                    "completed author is missing native query usage evidence"
                )
            usage = cumulative_usage(audit)
            session_id = completion["session_id"]
            atomic_json(
                audit / "usage.json",
                with_admission_usage(usage, audit).model_dump(mode="json"),
            )
            try:
                candidates = await admit_mutation(workspace, case_results=case_results)
                record(
                    {
                        "event": "code_admission_passed",
                        "physical_execution_tested": False,
                    }
                )
            except AUTHOR_VALIDATION_ERRORS as error:
                task = asyncio.current_task()
                if (
                    task is not None and getattr(task, "cancelling", lambda: 0)()
                ) or isinstance(error.__context__, asyncio.CancelledError):
                    # run_blocking drains the worker and surfaces its real error.
                    # Cancellation still forbids starting a repair query: retain
                    # that worker failure as the cancellation's explicit cause.
                    raise asyncio.CancelledError(
                        "Author cancelled while final validation was settling"
                    ) from error
                await run_blocking(
                    self._save_correction,
                    directory,
                    check_path,
                    error,
                    case_results=case_results,
                )
                record(
                    {
                        "event": "admission_rejected",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "correction": f"corrections/{check_path.stem}/feedback.json",
                        "will_resume_same_session": True,
                    }
                )
                raise AuthorCorrectionRequired(str(error)) from error
            record({"event": "admission_passed", "physical_execution_tested": False})
            evidence["validation_status"] = "passed"
            await run_blocking(atomic_json, evidence_path, evidence)
            evidence_written = True
            deferred = (check_path, audit / "usage.json")
            result = await run_blocking(
                self._collect_result,
                directory,
                candidates,
                candidate_summary(workspace),
                usage,
                session_id,
                deferred_artifacts=deferred,
            )
        except BaseException as error:
            evidence_written = False
            evidence["validation_status"] = (
                "interrupted" if not isinstance(error, Exception) else "failed"
            )
            evidence["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
            raise
        finally:
            # Includes cancellation evidence produced by the production worker.
            # CancelledError propagates; it is not an author correction request.
            # Full case results can be large. Publish their separate original
            # inside the measured interval; the final accounting record stays small.
            # If collection failed, no published artifact references these new bytes.
            if not evidence_written:
                await run_blocking(atomic_json, evidence_path, evidence)
            evidence_record = await run_blocking(artifact_record, evidence_path)
            check_record.update(
                validation_status=evidence["validation_status"],
                evidence={
                    "path": evidence_path.relative_to(audit).as_posix(),
                    "artifact_id": evidence_record["artifact_id"],
                    "bytes": evidence_record["bytes"],
                },
                wall_time_s=time.monotonic() - checked_at,
                wall_time_complete=True,
            )
            atomic_json(check_path, check_record)
            usage = with_admission_usage(usage, audit)
            atomic_json(audit / "usage.json", usage.model_dump(mode="json"))
        # These were the only mutable artifacts during collection. Bind their
        # final bytes now; no audit or session writes may follow publication.
        accounting_artifacts = tuple(_artifact(path) for path in deferred)
        result = result.model_copy(
            update={
                "usage": usage,
                "artifacts": (*result.artifacts, *accounting_artifacts),
            }
        )
        atomic_json(directory / "result.json", result.model_dump(mode="json"))
        return result
