# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Episode-private source revisions activated at settled execution boundaries.

This module owns source capture and activation only. The existing Kernel remains
the authority for permissions, continuations, environment state and all budgets.
No Python stack is reloaded: callers bind immutable programs before execution.
"""

from __future__ import annotations

import difflib
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, JsonValue, TypeAdapter

from reweave.evolution.authoring.candidate import (
    CandidateCapture,
    capture_bundle,
    write_bundle_source,
    write_workflow,
)
from reweave.evolution.bundle import (
    AgentBundle,
    PythonProgram,
    Scope,
    WorkflowSpec,
    new_record_id,
)
from reweave.utils.persistence import atomic_json

_STATE = TypeAdapter(
    dict[str, JsonValue], config=ConfigDict(strict=True, allow_inf_nan=False)
)


def _changed_scopes(bundle: AgentBundle, previous: AgentBundle) -> tuple[Scope, ...]:
    """Compare actual components; source identities never control activation."""
    return tuple(
        scope
        for scope, name in (("W", "workflows"), ("C", "context"), ("L", "loop"))
        if getattr(bundle, name) != getattr(previous, name)
    )


def read_adaptation_receipt(path: Path) -> dict[str, Any]:
    """Read the actual revision chain and its artifact references, without hashes.

    This is evidence parsing, not source identity certification or a launch gate.
    Existing bundle IDs are recorded version references; no source is rehashed.
    """
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("mode") not in {
        "static",
        "adaptive",
    }:
        raise ValueError("adaptation receipt requires an explicit static/adaptive mode")
    revisions = receipt.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        raise ValueError("adaptation receipt requires its initial revision")
    identities = set()
    parent = None
    for revision in revisions:
        if not isinstance(revision, dict):
            raise ValueError("adaptation revision must be an object")
        identity = revision.get("revision_id")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in identities
            or revision.get("parent_revision_id") != parent
        ):
            raise ValueError("adaptation revision chain has invalid or duplicate links")
        identities.add(identity)
        parent = identity
        for name in ("bundle_path", "diff_path"):
            if (
                not isinstance(revision.get(name), str)
                or not Path(revision[name]).is_file()
            ):
                raise ValueError(f"adaptation revision has no readable {name}")
        directory = Path(revision["source_directory"])
        files = revision.get("source_files")
        if not isinstance(files, list) or any(
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (directory / name).is_file()
            for name in files
        ):
            raise ValueError("adaptation revision source inventory is incomplete")
    if receipt.get("initial_bundle_id") != revisions[0].get("bundle_id"):
        raise ValueError(
            "adaptation initial bundle reference differs from its revision"
        )
    if receipt.get("active_revision_id") not in identities or (
        receipt.get("pending_revision_id") is not None
        and receipt["pending_revision_id"] not in identities
    ):
        raise ValueError("adaptation active/pending revision reference is absent")
    activations = receipt.get("activations")
    if not isinstance(activations, list) or not activations:
        raise ValueError("adaptation receipt requires its initial activation")
    previous = None
    for activation in activations:
        if (
            not isinstance(activation, dict)
            or activation.get("revision_id") not in identities
            or activation.get("previous_revision_id") != previous
        ):
            raise ValueError("adaptation activation chain has invalid links")
        previous = activation["revision_id"]
    if previous != receipt["active_revision_id"] or (
        receipt["mode"] == "static"
        and not receipt.get("online_workflows", False)
        and any(
            item["revision_id"] != revisions[0]["revision_id"] for item in activations
        )
    ):
        raise ValueError(
            "adaptation activation does not describe the recorded mode/state"
        )
    if receipt.get("online_workflows", False) and receipt["mode"] == "static":
        initial = AgentBundle.model_validate_json(
            Path(revisions[0]["bundle_path"]).read_text()
        )
        active_ids = {item["revision_id"] for item in activations}
        for revision in revisions[1:]:
            if revision["revision_id"] not in active_ids:
                continue
            current = AgentBundle.model_validate_json(
                Path(revision["bundle_path"]).read_text()
            )
            current.assert_task_family_edit(initial)
            if set(_changed_scopes(current, initial)) - {"W"}:
                raise ValueError("online W authority does not permit static C/L edits")
    retained = receipt.get("retained_continuations")
    if not isinstance(retained, dict) or any(
        not isinstance(binding, dict) or binding.get("revision_id") not in identities
        for binding in retained.values()
    ):
        raise ValueError("adaptation retained continuation revision is absent")
    return receipt


@dataclass(frozen=True)
class HarnessRevision:
    """An immutable complete source snapshot, independent of activation status."""

    # Episode-local revision identity; repeated bundle content may have distinct revisions.
    revision_id: str
    # Previous captured revision, including a staged revision superseded before activation.
    parent_revision_id: str | None
    # Complete immutable W/C/L and the original fixed runtime.
    bundle: AgentBundle
    # Host-owned directory containing bundle.json, exact source files and diff.patch.
    directory: Path
    # Components changed from the previous captured revision, not an evaluator verdict.
    changed_scopes: tuple[Scope, ...]
    # Agent's explanation and public evidence references, preserved without certification.
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ProgramBinding:
    """The exact source used by one call or retained workflow continuation."""

    # Host invocation identity used only to settle this call, never a resource allowance.
    call_id: str
    scope: Scope
    revision: HarnessRevision
    program: PythonProgram
    # A W binding includes its original schema, permissions, limits and dependencies.
    workflow: WorkflowSpec | None = None
    # Detached cooperative L handoff, never a serialized Python stack.
    state: dict[str, Any] | None = None


class HarnessAdaptation:
    """Capture full edits and automatically activate them when execution settles.

    Keep one instance for the episode. C/W changes wait for active C/W calls;
    an unchanged running L keeps its old program. Changes to L additionally
    require ``checkpoint`` after the old L cooperatively returns and all calls
    settle. Pending proposals can explicitly bind their original revision.
    """

    def __init__(
        self,
        initial_bundle: AgentBundle,
        *,
        candidate_directory: Path,
        audit_directory: Path,
        mode: Literal["static", "adaptive"] = "adaptive",
        online_workflows: bool = False,
    ):
        if mode not in {"static", "adaptive"}:
            raise ValueError("adaptation mode must be static or adaptive")
        self.mode = mode
        if type(online_workflows) is not bool:
            raise ValueError("online_workflows must be a boolean")
        self.online_workflows = online_workflows
        self.candidate_directory = Path(candidate_directory)
        self.audit_directory = Path(audit_directory)
        if self.candidate_directory.is_symlink() or self.audit_directory.is_symlink():
            raise ValueError(
                "episode candidate and audit directories must not be links"
            )
        candidate = self.candidate_directory.resolve()
        audit = self.audit_directory.resolve()
        if candidate.is_relative_to(audit) or audit.is_relative_to(candidate):
            raise ValueError("candidate and audit directories must be separate")
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        if any(self.audit_directory.iterdir()):
            raise ValueError(
                "revision audit directory must be fresh; existing evidence is immutable"
            )
        self._lock = threading.RLock()
        self._revisions: list[HarnessRevision] = []
        self._files: dict[str, dict[str, bytes]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._calls: dict[str, ProgramBinding] = {}
        self._settlements = 0
        self._continuations: dict[str, ProgramBinding] = {}
        self._events: list[dict[str, Any]] = []
        self._pending: HarnessRevision | None = None
        self._checkpoint: dict[str, Any] | None = None
        # Never seed import validation from a serialized bundle or receipt.
        self._validated_capture: CandidateCapture | None = None
        initial = AgentBundle.model_validate_json(initial_bundle.model_dump_json())
        initial_id = uuid.uuid4().hex
        source = self.audit_directory / "revisions" / initial_id / "source"
        write_bundle_source(initial, source)
        files = {
            path.relative_to(source).as_posix(): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        }
        self._active = self._save_revision(
            initial, files, "Initial episode harness", (), revision_id=initial_id
        )
        self._activate_record(self._active, boundary="episode_start")
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        if not any(self.candidate_directory.iterdir()):
            write_bundle_source(initial, self.candidate_directory)

    @property
    def active(self) -> HarnessRevision:
        """Current revision for future calls; already-bound calls keep their source."""
        with self._lock:
            return self._active

    @property
    def pending(self) -> HarnessRevision | None:
        """Latest complete captured edit awaiting an automatic settled boundary."""
        with self._lock:
            return self._pending

    @property
    def continuations(self) -> tuple[tuple[str, ProgramBinding], ...]:
        """Retained original W contracts; Kernel alone owns resume state and counters."""
        with self._lock:
            return tuple(self._continuations.items())

    @property
    def loop_state(self) -> dict[str, Any] | None:
        """Detached state from the last cooperative checkpoint, if not yet resumed."""
        with self._lock:
            return (
                self._copy_state(self._checkpoint["state"])
                if self._checkpoint
                else None
            )

    def revision(self, revision_id: str) -> HarnessRevision:
        """Resolve a captured source revision for a cached proposal or continuation."""
        with self._lock:
            for item in self._revisions:
                if item.revision_id == revision_id:
                    return item
            raise KeyError(f"unknown harness revision: {revision_id}")

    def _event(self, kind: str, **values: Any) -> dict[str, Any]:
        event = {"kind": kind, "checked_at": time.time(), **values}
        with (self.audit_directory / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._events.append(event)
        return event

    def _failure(self, kind: str, error: BaseException, **values: Any) -> None:
        try:
            self._event(
                kind, error_type=type(error).__name__, error=str(error), **values
            )
        except BaseException as audit_error:
            error.add_note(
                f"Revision audit failed: {type(audit_error).__name__}: {audit_error}",
            )

    @staticmethod
    def _write_source(source, files):
        source.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for layer in ("global", "task-family", "task-specific"):
            (source / layer).mkdir(exist_ok=True)
        (source / "task-family/workflows").mkdir(exist_ok=True)
        (source / "task-family/context/skills").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _diff(previous, files, before, after):
        # Invalid UTF-8 is escaped in the diagnostic view; source snapshots retain exact bytes.
        return "".join(
            line
            for name in sorted(previous.keys() | files.keys())
            for line in difflib.unified_diff(
                previous.get(name, b"")
                .decode("utf-8", errors="backslashreplace")
                .splitlines(keepends=True),
                files.get(name, b"")
                .decode("utf-8", errors="backslashreplace")
                .splitlines(keepends=True),
                fromfile=f"{before}/{name}",
                tofile=f"{after}/{name}",
            )
        )

    def _failed_source(self, files):
        if not files:
            return None
        identity = uuid.uuid4().hex
        directory = self.audit_directory / "failed_edits" / identity
        self._write_source(directory / "source", files)
        previous = self._revisions[-1]
        diff = self._diff(
            self._files[previous.revision_id], files, previous.revision_id, identity
        )
        (directory / "diff.patch").write_text(diff, encoding="utf-8")
        return {
            "directory": str(directory),
            "source_files": sorted(files),
            "diff_path": str(directory / "diff.patch"),
        }

    def _save_revision(self, bundle, files, reason, evidence, *, revision_id=None):
        parent = self._revisions[-1] if self._revisions else None
        revision_id = revision_id or uuid.uuid4().hex
        if parent is not None and bundle.record_id == parent.bundle.record_id:
            bundle = AgentBundle.model_validate(
                {
                    **bundle.model_dump(mode="json"),
                    "record_id": new_record_id(),
                }
            )
        directory = self.audit_directory / "revisions" / revision_id
        source = directory / "source"
        self._write_source(source, files)
        atomic_json(directory / "bundle.json", bundle.model_dump(mode="json"))
        previous = self._files[parent.revision_id] if parent else {}
        diff = self._diff(
            previous, files, parent.revision_id if parent else "empty", revision_id
        )
        (directory / "diff.patch").write_text(diff, encoding="utf-8")
        revision = HarnessRevision(
            revision_id,
            parent.revision_id if parent else None,
            bundle,
            directory,
            _changed_scopes(bundle, parent.bundle) if parent else (),
            reason,
            tuple(evidence),
        )
        record = {
            "revision_id": revision_id,
            "parent_revision_id": revision.parent_revision_id,
            "bundle_id": bundle.record_id,
            "bundle_path": str(directory / "bundle.json"),
            "source_directory": str(source),
            "source_files": sorted(files),
            "diff_path": str(directory / "diff.patch"),
            "changed_scopes": list(revision.changed_scopes),
            "reason": reason,
            "evidence": list(evidence),
        }
        atomic_json(directory / "revision.json", record)
        self._revisions.append(revision)
        self._files[revision_id] = dict(files)
        self._records[revision_id] = record
        return revision

    def stage(self, *, reason: str, evidence: Sequence[str] = ()) -> HarnessRevision:
        """Capture complete private edits once and activate at the next safe boundary."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("revision reason must be nonblank")
        if (
            not isinstance(evidence, Sequence)
            or isinstance(evidence, str)
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            raise ValueError(
                "revision evidence must be a sequence of nonblank references"
            )
        evidence = tuple(evidence)
        with self._lock:
            files = {}
            captured_revision = None
            try:
                captured = capture_bundle(
                    self.candidate_directory,
                    self._active.bundle.runtime,
                    captured_files=files,
                    record_id=self._active.bundle.record_id,
                    validated_capture=self._validated_capture,
                )
                self._validated_capture = captured
                captured.bundle.assert_task_family_edit(self._active.bundle)
                previous = self._revisions[-1]

                def protected(values):
                    return {
                        k: v
                        for k, v in values.items()
                        if k.startswith(("global/", "task-specific/"))
                    }

                if protected(captured.files) != protected(
                    self._files[previous.revision_id]
                ):
                    raise ValueError(
                        "global and Host task-specific files are read-only during evolution"
                    )
                if self.mode == "static" and (
                    not self.online_workflows
                    or set(_changed_scopes(captured.bundle, self._active.bundle))
                    - {"W"}
                ):
                    if not _changed_scopes(captured.bundle, self._active.bundle):
                        return self._active
                    if _changed_scopes(captured.bundle, previous.bundle):
                        previous = self._save_revision(
                            captured.bundle, captured.files, reason, evidence
                        )
                    captured_revision = previous
                    self._event(
                        "stage_rejected",
                        revision_id=previous.revision_id,
                        mode=self.mode,
                    )
                    raise PermissionError(
                        "online harness editing is disabled in static mode"
                    )
                if not _changed_scopes(captured.bundle, previous.bundle):
                    self._activate_if_ready()
                    return previous
                captured_bundle = AgentBundle.model_validate(
                    {
                        **captured.bundle.model_dump(mode="json"),
                        "record_id": new_record_id(),
                    }
                )
                revision = self._save_revision(
                    captured_bundle, captured.files, reason, evidence
                )
                captured_revision = revision
                supersedes = self._pending.revision_id if self._pending else None
                self._pending = revision
                self._event(
                    "staged",
                    revision_id=revision.revision_id,
                    supersedes=supersedes,
                    files=sorted(files),
                )
                self._activate_if_ready()
                return revision
            except BaseException as error:
                failed_source = None
                if captured_revision is None:
                    try:
                        failed_source = self._failed_source(files)
                    except BaseException as audit_error:
                        error.add_note(
                            f"Failed-edit source audit failed: {audit_error}"
                        )
                self._failure(
                    "stage_failed",
                    error,
                    files=sorted(files),
                    failed_source=failed_source,
                    revision_id=captured_revision.revision_id
                    if captured_revision
                    else None,
                )
                raise

    def write_workflow(self, workflow: WorkflowSpec, *, reason: str) -> HarnessRevision:
        """Publish one ordinary W through the existing source/revision boundary.

        This Host tool grants only task-family W authority in a static episode.
        Native file edits and C/L activation still follow the configured mode.
        Existing continuations keep their immutable source and counters.
        """
        if not self.online_workflows:
            raise PermissionError("online W authoring is disabled")
        if workflow.layer != "task-family":
            raise PermissionError("online authoring may write only task-family W")
        with self._lock:
            # Capture pending native edits first; never overwrite an unrecorded edit.
            self.stage(reason="Capture private source before online W authoring")
            previous = self._pending or self._active
            existing = {item.workflow_id: item for item in previous.bundle.workflows}
            if (
                workflow.workflow_id in existing
                and existing[workflow.workflow_id].layer != "task-family"
            ):
                raise PermissionError("global W is read-only during online authoring")
            existing[workflow.workflow_id] = workflow
            AgentBundle.model_validate(
                {
                    **previous.bundle.model_dump(mode="json"),
                    "workflows": [
                        existing[name].model_dump(mode="json")
                        for name in sorted(existing)
                    ],
                }
            )
            root = self.candidate_directory / "task-family" / "workflows"
            target = root / workflow.workflow_id
            if any(
                path.is_symlink()
                for path in (self.candidate_directory, root.parent, root, target)
            ):
                raise ValueError("online W destination must not contain links")
            with tempfile.TemporaryDirectory(
                prefix="online-w-", dir=self.audit_directory
            ) as name:
                staging = Path(name)
                write_workflow(staging / "next", workflow)
                backup = staging / "previous"
                if target.exists():
                    target.rename(backup)
                (staging / "next").rename(target)
                try:
                    return self.stage(reason=reason)
                except BaseException:
                    target.rename(staging / "rejected")
                    if backup.exists():
                        backup.rename(target)
                    raise

    def _activate_record(self, revision, *, boundary):
        event = {
            "revision_id": revision.revision_id,
            "previous_revision_id": self._active.revision_id if self._events else None,
            "bundle_id": revision.bundle.record_id,
            "boundary": boundary,
            "checkpoint": self._checkpoint,
        }
        # Publish the complete activation before exposing its programs to a future call.
        atomic_json(self.audit_directory / "active.json", event)
        self._active = revision
        self._pending = None
        self._event("activated", **event)

    def _activate_if_ready(self):
        if self._pending is None:
            return
        changed = _changed_scopes(self._pending.bundle, self._active.bundle)
        if self._settlements or any(
            binding.scope != "L" for binding in self._calls.values()
        ):
            return
        if "L" in changed and (self._calls or self._checkpoint is None):
            return
        revision = self._pending
        try:
            self._activate_record(
                revision,
                boundary="loop_checkpoint" if "L" in changed else "settled_calls",
            )
        except BaseException as error:
            self._failure("activation_failed", error, revision_id=revision.revision_id)
            raise

    @staticmethod
    def _copy_state(state):
        return json.loads(json.dumps(_STATE.validate_python(state), allow_nan=False))

    def checkpoint(self, state: dict[str, Any]) -> HarnessRevision:
        """Record an explicit L return after all calls settle, then activate pending L.

        Host must call this only for a cooperative checkpoint returned by L, not
        an arbitrary interrupt. The next L binding receives this JSON state.
        """
        copied = self._copy_state(state)
        with self._lock:
            if self._calls or self._settlements:
                raise RuntimeError("loop checkpoint requires all calls to settle")
            checkpoint = {"revision_id": self._active.revision_id, "state": copied}
            self._event("loop_checkpoint", **checkpoint)
            self._checkpoint = checkpoint
            self._activate_if_ready()
            return self._active

    @contextmanager
    def bind(
        self,
        scope: Scope,
        *,
        workflow_id: str | None = None,
        continuation_id: str | None = None,
        revision_id: str | None = None,
    ) -> Iterator[ProgramBinding]:
        """Pin the source/contract for a call, including cached old proposals."""
        with self._lock:
            if scope not in {"W", "C", "L"}:
                raise ValueError("program scope must be W, C or L")
            if scope != "W" and (
                workflow_id is not None or continuation_id is not None
            ):
                raise ValueError("only W calls have workflow/continuation identities")
            previous = (
                self._continuations.get(continuation_id)
                if continuation_id is not None
                else None
            )
            if previous is not None:
                if workflow_id != previous.workflow.workflow_id or (
                    revision_id is not None
                    and revision_id != previous.revision.revision_id
                ):
                    raise ValueError(
                        "continuation source differs from its retained workflow revision"
                    )
                revision = previous.revision
            else:
                revision = (
                    self.revision(revision_id)
                    if revision_id is not None
                    else self._active
                )
            workflow = None
            if scope == "W":
                workflow = next(
                    (
                        item
                        for item in revision.bundle.workflows
                        if item.workflow_id == workflow_id
                    ),
                    None,
                )
                if workflow is None:
                    raise PermissionError(
                        f"workflow {workflow_id!r} is absent from revision {revision.revision_id}"
                    )
                program = workflow.program
            else:
                program = (
                    revision.bundle.context.program
                    if scope == "C"
                    else revision.bundle.loop.program
                )
            binding = ProgramBinding(
                uuid.uuid4().hex,
                scope,
                revision,
                program,
                workflow,
                self.loop_state if scope == "L" else None,
            )
            self._calls[binding.call_id] = binding
            if scope == "L":
                self._checkpoint = None
        failure = None
        try:
            yield binding
        except BaseException as error:
            failure = error
            raise
        finally:
            with self._lock:
                self._calls.pop(binding.call_id)
                try:
                    self._activate_if_ready()
                except BaseException as error:
                    if failure is None:
                        raise
                    failure.add_note(
                        f"Revision activation failed: {type(error).__name__}: {error}",
                    )
                    raise failure from error

    def retain_continuation(self, token: str, binding: ProgramBinding) -> None:
        """Retain the original W contract; Kernel still owns its state and budgets."""
        with self._lock:
            if (
                not isinstance(token, str)
                or not token
                or binding.scope != "W"
                or self.revision(binding.revision.revision_id) is not binding.revision
            ):
                raise ValueError(
                    "continuation requires a local workflow binding and nonempty token"
                )
            previous = self._continuations.get(token)
            if previous is not None and (
                previous.revision is not binding.revision
                or previous.workflow is not binding.workflow
            ):
                raise ValueError(
                    "continuation cannot change its retained workflow source"
                )
            self._event(
                "continuation_retained",
                continuation_id=token,
                revision_id=binding.revision.revision_id,
                workflow_id=binding.workflow.workflow_id,
            )
            self._continuations[token] = binding

    def consume_continuation(self, token: str) -> None:
        """Release a token only after its existing Kernel lifecycle consumes it."""
        with self._lock:
            binding = self._continuations.get(token)
            if binding is None:
                return
            self._event(
                "continuation_consumed",
                continuation_id=token,
                revision_id=binding.revision.revision_id,
            )
            del self._continuations[token]

    @contextmanager
    def settling(self) -> Iterator[None]:
        """Delay activation until an already-owned fixed tool/physical ACK settles."""
        with self._lock:
            self._settlements += 1
        failure = None
        try:
            yield
        except BaseException as error:
            failure = error
            raise
        finally:
            with self._lock:
                self._settlements -= 1
                try:
                    self._activate_if_ready()
                except BaseException as error:
                    if failure is None:
                        raise
                    failure.add_note(f"Revision activation failed: {error}")
                    raise failure from error

    def receipt(self) -> dict[str, Any]:
        """Export the one complete measured adaptation chain for episode evidence."""
        with self._lock:
            return json.loads(
                json.dumps(
                    {
                        "mode": self.mode,
                        **({"online_workflows": True} if self.online_workflows else {}),
                        "initial_bundle_id": self._revisions[0].bundle.record_id,
                        "active_revision_id": self._active.revision_id,
                        "pending_revision_id": self._pending.revision_id
                        if self._pending
                        else None,
                        "revisions": [
                            self._records[item.revision_id] for item in self._revisions
                        ],
                        "activations": [
                            event
                            for event in self._events
                            if event["kind"] == "activated"
                        ],
                        "events_path": str(self.audit_directory / "events.jsonl"),
                        "retained_continuations": {
                            token: {
                                "revision_id": binding.revision.revision_id,
                                "workflow_id": binding.workflow.workflow_id,
                            }
                            for token, binding in self._continuations.items()
                        },
                    },
                    allow_nan=False,
                )
            )
