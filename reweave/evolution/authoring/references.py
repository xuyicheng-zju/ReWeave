# Modified for the ReWeave namespace.
"""Readable, request-bound references at the author/Host boundary.

The run allocates append-only names; each workspace admits only the names it
actually receives. Published record IDs and original paths remain Host-owned.
Source code, explanations and evidence values are never rewritten as prose.
"""

from __future__ import annotations

import fcntl
import json
import re
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from reweave.evolution.contracts import (
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    OriginalReference,
    Proposal,
    RoundDecision,
)
from reweave.utils.persistence import atomic_json

_HANDLE = re.compile(r"([a-z][a-z_]*):([0-9]{3,})")
_REFERENCE_PATTERN = r"^file:[0-9]{3,}#(?:/.*)?$"
_IDENTITIES = {
    "graph_id": "graph",
    "execution_graph_id": "graph",
    "bundle_id": "bundle",
    "bundle_ids": "bundle",
    "version_id": "version",
    "version_ids": "version",
    "parent_version_id": "version",
    "condition_id": "condition",
    "condition_ids": "condition",
    "conditions": "condition",
    "receipt_id": "episode",
    "before_receipt_id": "episode",
    "after_receipt_id": "episode",
    "parent_receipt_id": "episode",
    "candidate_receipt_id": "episode",
    "episode_id": "episode",
    "episode_ids": "episode",
    "reached_episode_ids": "episode",
    "completed_episode_ids": "episode",
    "new_episode_ids": "episode",
    "pending_episode_ids": "episode",
    "projected_episode_ids": "episode",
    "evidence_id": "trace",
    "source_id": "source",
    "executed_source": "source",
    "transition_id": "transition",
    "transition_ids": "transition",
    "unresolved_transition_ids": "transition",
    "unattributed_transition_ids": "transition",
    "successor_id": "transition",
    "references": "version",
}
_TABLES = {
    "episodes": "episode",
    "paths": "episode",
    "versions": "version",
    "mutations": "version",
    "validation": "version",
    "cells": "version",
    "validation_cells": "version",
    "transitions": "transition",
    "conditional_interfaces": "interface",
    "executed_sources": "source",
}
_TEXT = {
    "source",
    "meaning",
    "mechanism",
    "explanation",
    "uncertainty",
    "entry_conditions",
    "exit_conditions",
    "instructions",
    "question",
    "answer",
    "rationale",
    "changes",
    "summary",
    "text",
    "description",
}
_LITERALS = {
    "matching",
    "catalog",
    "arguments",
    "parameters",
    "initialization",
    "model_sampling",
    "bindings",
    "roles",
    "features",
    "signature",
    "value",
    "result",
    "tool_proposals",
    "input_schema",
    "output_schema",
    "state_schema",
}
_REFERENCES = {
    "reference",
    "references",
    "evidence_refs",
    "feature_source",
    "bundle",
    "receipt",
    "complete_report",
    "original",
    "originals",
    "artifact",
    "artifacts",
    "source",
    "feedback",
    "request",
    "payload",
    "invocation",
    "arguments_reference",
    "executed_reference",
}


def history_originals(request: Proposal | GraphAnalysisRequest) -> dict:
    """Published originals plus the exact completed graph supplied to this job."""
    originals = json.loads(Path(request.history.originals.path).read_text())
    if set(originals) != {"files", "navigation"} or not all(
        isinstance(originals[field], dict) for field in ("files", "navigation")
    ):
        raise ValueError("history originals must contain files and navigation maps")
    analysis = None
    if isinstance(request, Proposal):
        if request.purpose == "mutation":
            request.decision.verify()
            analysis = RoundDecision.model_validate_json(
                Path(request.decision.path).read_text()
            ).analysis
    else:
        analysis = request.previous_analysis
    if analysis is not None:
        analysis.verify()
        receipt = GraphAnalysisReceipt.model_validate_json(
            Path(analysis.path).read_text()
        )
        originals["files"].update(
            {
                item.path: {"artifact_id": item.artifact_id, "bytes": item.bytes}
                for item in receipt.artifacts
            }
        )
        originals["files"][analysis.path] = {
            "artifact_id": analysis.artifact_id,
            "bytes": analysis.bytes,
        }
        # A published graph may still name originals in an earlier workspace.
        # Preserve its exact provenance when the current archive has new paths.
        for reference in receipt.analysis.graph.original_references():
            metadata = {"artifact_id": reference.artifact_id, "bytes": reference.bytes}
            previous = originals["files"].setdefault(reference.path, metadata)
            if previous != metadata:
                raise ValueError(
                    "published graph original differs from history inventory"
                )
    return originals


def _catalog_path(workspace: Path) -> Path:
    staging = workspace / ".host_inputs"
    return (
        staging if staging.is_dir() else workspace.parent / "audit/inputs"
    ) / "author_references.json"


class AuthorReferences:
    """One typed reference vocabulary shared by files, schemas and tools."""

    def __init__(self, workspace, registry, evidence_roots, entries, originals):
        self.workspace = Path(workspace)
        self.registry = Path(registry)
        self.evidence_roots = {key: Path(path) for key, path in evidence_roots.items()}
        self.entries = dict(entries)
        self.originals = dict(originals)
        self._global = None
        self._reverse = None
        self._counts = None
        self._allocation_owner = None

    @classmethod
    def create(cls, workspace, *, registry, evidence_roots, originals=None):
        workspace = Path(workspace)
        (workspace / ".host_inputs").mkdir(parents=True, exist_ok=True)
        if _catalog_path(workspace).exists():
            raise ValueError("author reference catalog already exists")
        result = cls(workspace, registry, evidence_roots, {}, originals or {})
        result.save()
        return result

    @classmethod
    def load(cls, workspace):
        value = json.loads(_catalog_path(Path(workspace)).read_text())
        return cls(
            workspace,
            value["registry"],
            value["evidence_roots"],
            value["entries"],
            value["originals"],
        )

    def save(self):
        path = _catalog_path(self.workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                previous = json.loads(path.read_text())["entries"]
                if any(
                    key in previous and previous[key] != value
                    for key, value in self.entries.items()
                ):
                    raise ValueError("an issued author reference cannot be rebound")
                self.entries = previous | self.entries
            atomic_json(
                path,
                {
                    "registry": str(self.registry),
                    "evidence_roots": {
                        key: str(root) for key, root in self.evidence_roots.items()
                    },
                    "entries": self.entries,
                    "originals": self.originals,
                },
            )

    @contextmanager
    def _allocation(self):
        if self._global is not None and self._allocation_owner == threading.get_ident():
            before = len(self._global)
            previous_entries = len(self.entries)
            try:
                yield
            except BaseException:
                while len(self._global) > before:
                    identity, value = self._global.popitem()
                    kind = identity.split(":")[0]
                    self._reverse.pop((kind, json.dumps(value, sort_keys=True)))
                    self._counts[kind] -= 1
                while len(self.entries) > previous_entries:
                    self.entries.popitem()
                raise
            return
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        with self.registry.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._global = (
                json.loads(self.registry.read_text()) if self.registry.exists() else {}
            )
            self._reverse = {
                (key.split(":")[0], json.dumps(value, sort_keys=True)): key
                for key, value in self._global.items()
            }
            self._counts = Counter(key.split(":")[0] for key in self._global)
            self._allocation_owner = threading.get_ident()
            before = len(self._global)
            previous_entries = len(self.entries)
            try:
                yield
                if len(self._global) != before:
                    atomic_json(self.registry, self._global)
                if len(self.entries) != previous_entries:
                    self.save()
            except BaseException:
                while len(self.entries) > previous_entries:
                    self.entries.popitem()
                raise
            finally:
                self._global = self._reverse = None
                self._counts = self._allocation_owner = None
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def batch(self):
        """Prepare references atomically, persisting the complete catalogs once."""
        with self._allocation():
            yield self

    def _register(self, kind, value):
        key = (kind, json.dumps(value, sort_keys=True))
        identity = self._reverse.get(key)
        if identity is None:
            count = self._counts[kind]
            identity = f"{kind}:{count + 1:03d}"
            self._global[identity] = value
            self._reverse[key] = identity
            self._counts[kind] += 1
        self.entries[identity] = value
        return identity

    def identity(self, kind, value):
        if not re.fullmatch(r"[a-z][a-z_]*", kind) or (
            not isinstance(value, str) or not value
        ):
            raise ValueError(
                "identity requires a domain name and a nonblank published record ID"
            )
        with self._allocation():
            return self._register(kind, value)

    def resolve(self, identity, *, kind=None):
        match = _HANDLE.fullmatch(identity)
        if identity not in self.entries:
            self.entries.update(
                json.loads(_catalog_path(self.workspace).read_text())["entries"]
            )
        if match is None or identity not in self.entries:
            raise ValueError(f"unknown author reference: {identity}")
        if kind is not None and match[1] != kind:
            raise ValueError(f"expected {kind} reference, received {identity}")
        return self.entries[identity]

    def _absolute(self, path):
        path = Path(path)
        if path.is_absolute():
            if ".." in path.parts:
                raise ValueError(
                    "original reference must not traverse parent directories"
                )
            return str(path)
        if path.parts[:2] != (".archive", "episodes") or len(path.parts) < 4:
            raise ValueError(
                "original reference requires an absolute or archived episode path"
            )
        root = self.evidence_roots.get(path.parts[2])
        if root is None or ".." in path.parts:
            raise ValueError("original reference is outside this request's episodes")
        return str(root.joinpath(*path.parts[3:]))

    def _file(self, path, artifact_id, byte_count):
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or type(byte_count) is not int
            or byte_count < 0
        ):
            raise ValueError(
                "original file requires its published artifact ID and byte count"
            )
        name = self._absolute(path)
        source = Path(name)
        if name in self.originals:
            if {"artifact_id": artifact_id, "bytes": byte_count} != self.originals[
                name
            ]:
                raise ValueError(
                    "original provenance differs from its frozen inventory"
                )
        elif not any(
            source.is_relative_to(root) for root in self.evidence_roots.values()
        ):
            audit = self.workspace.parent / "audit"
            reports = (
                audit / "development_reports",
                audit / "planner_probes",
                audit / "corrections",
            )
            if source != audit / "session_inquiry/result.json" and not any(
                source.is_relative_to(root) for root in reports
            ):
                raise ValueError("original file is outside this request's inventory")
        return self._register(
            "file", {"path": name, "artifact_id": artifact_id, "bytes": byte_count}
        )

    @staticmethod
    def _pointer(pointer):
        if (
            not isinstance(pointer, str)
            or (pointer and not pointer.startswith("/"))
            or re.search(r"~(?![01])", pointer)
        ):
            raise ValueError("original reference requires a valid JSON pointer")
        return pointer

    def reference(self, path, artifact_id, bytes, pointer=""):
        with self._allocation():
            return self._file(path, artifact_id, bytes) + "#" + self._pointer(pointer)

    def original(self, reference):
        if not isinstance(reference, str):
            raise ValueError("original reference must be a file handle")
        identity, _, pointer = reference.partition("#")
        value = self.resolve(identity, kind="file")
        return OriginalReference(**value, pointer=self._pointer(pointer))

    def trace_reference(self, reference):
        original = self.original(reference)
        path = Path(original.path)
        for identity, root in self.evidence_roots.items():
            if path.is_relative_to(root):
                return {
                    "path": str(
                        Path(".archive/episodes") / identity / path.relative_to(root)
                    ),
                    "artifact_id": original.artifact_id,
                    "bytes": original.bytes,
                    "pointer": original.pointer,
                }
        raise ValueError("reference is not an archived episode original")

    def encode(self, value, *, kind=None):
        with self._allocation():
            return self._walk(value, kind=kind, encode=True)

    def decode(self, value, *, kind=None):
        return self._walk(value, kind=kind, encode=False)

    def _walk(self, value, *, kind, encode, field=None):
        if field in _LITERALS:
            return value
        if field in _TEXT and not (
            isinstance(value, dict)
            and set(value)
            in (
                {"path", "artifact_id", "bytes"},
                {"path", "artifact_id", "bytes", "pointer"},
            )
        ):
            return value
        if isinstance(value, (list, tuple)):
            return [
                self._walk(
                    item,
                    kind=kind,
                    encode=encode,
                    field=None if isinstance(item, dict) else field,
                )
                for item in value
            ]
        if isinstance(value, str):
            if encode:
                return self._register(kind, value) if kind else value
            if (
                (field is None or field in _REFERENCES)
                and value.startswith("file:")
                and ("#" in value or _HANDLE.fullmatch(value))
            ):
                original = self.original(value)
                return original.model_dump(
                    mode="json", exclude=set() if "#" in value else {"pointer"}
                )
            if kind and _HANDLE.fullmatch(value):
                return self.resolve(value, kind=kind)
            if kind:
                raise ValueError(f"use an issued {kind} handle")
            return value
        if not isinstance(value, dict):
            return value
        fields = set(value)
        if fields in (
            {"path", "artifact_id", "bytes"},
            {"path", "artifact_id", "bytes", "pointer"},
        ):
            if not encode:
                raise ValueError(
                    "raw artifact references are not accepted by the author interface"
                )
            reference = self._file(value["path"], value["artifact_id"], value["bytes"])
            return (
                reference + "#" + self._pointer(value["pointer"])
                if "pointer" in value
                else reference
            )
        result = {}
        table_kind = kind or _TABLES.get(field)
        if field == "validation" and {"condition_ids", "target_success_rate"} & fields:
            table_kind = kind
        named_records = field in {
            "nodes",
            "edges",
            "programs",
            "node_evidence",
            "edge_statistics",
            "edge_interfaces",
        }
        transition = "frame_index" in value and "episode_id" in value
        scaffold = {"nodes", "paths", "policy", "episodes"} <= fields
        for key, item in value.items():
            new_key = key
            table_entry = table_kind and (encode or _HANDLE.fullmatch(key))
            if not encode and table_kind and not _HANDLE.fullmatch(key):
                raise ValueError(f"use an issued {table_kind} handle")
            if table_entry:
                new_key = (
                    self._register(table_kind, key)
                    if encode
                    else self.resolve(key, kind=table_kind)
                )
            child_kind = None if named_records else _IDENTITIES.get(key)
            if field == "edge_interfaces":
                child_kind = "interface"
            if transition and key in {"source", "target"}:
                child_kind = "partition"
            elif key == "nodes" and scaffold:
                child_kind = "partition"
            elif key == "node_id" and "signature" in value:
                child_kind = "partition"
            elif key == "target" and {"question", "evidence_refs"} <= fields:
                child_kind = "session"
            elif key == "path" and isinstance(item, (list, tuple)):
                child_kind = "partition"
            elif "version_id" in fields and key in {
                "cells",
                "validation_cells",
                "dev",
                "validation",
            }:
                child_kind = "condition"
            if table_entry and table_kind == "condition":
                child_kind = "episode" if isinstance(item, str) else None
            elif table_entry and field in {"cells", "validation_cells"}:
                child_kind = "condition"
            elif table_entry and field in {"versions", "validation"}:
                child_kind = (
                    "bundle"
                    if isinstance(item, str)
                    else "condition"
                    if field == "validation" and isinstance(item, dict)
                    else None
                )
            elif table_entry and field == "paths":
                child_kind = "partition"
            result[new_key] = self._walk(
                item,
                kind=child_kind,
                encode=encode,
                field=None
                if named_records or (transition and key == "source")
                else key,
            )
        return result

    def read_json(self, path):
        return self.decode(json.loads(Path(path).read_text()))

    @staticmethod
    def schema(schema):
        """Derive the author schema from the sole internal structural authority."""

        def visit(value, field=None):
            if isinstance(value, list):
                return [visit(item, field) for item in value]
            if not isinstance(value, dict):
                return value
            kind = _IDENTITIES.get(field) or _TABLES.get(field)
            result = {
                key: (
                    {name: visit(body, name) for name, body in item.items()}
                    if key == "properties"
                    else visit(item, field)
                )
                for key, item in value.items()
            }
            if kind and value.get("type") == "string":
                result["pattern"] = rf"^{kind}:[0-9]{{3,}}$"
            if (
                kind
                and value.get("type") == "object"
                and "additionalProperties" in value
                and "properties" not in value
            ):
                result["propertyNames"] = {
                    "type": "string",
                    "pattern": rf"^{kind}:[0-9]{{3,}}$",
                }
            return result

        result = visit(schema)
        for name, pattern, description in (
            (
                "OriginalReference",
                _REFERENCE_PATTERN,
                "Issued original file handle followed by # and its JSON pointer; an empty pointer selects the complete original. Copy the handle from current evidence tools or indices.",
            ),
            (
                "EvidenceFile",
                r"^file:[0-9]{3,}$",
                "Issued file handle. The Host retains the original path and published artifact record.",
            ),
        ):
            if name in result.get("$defs", {}):
                result["$defs"][name] = {
                    "type": "string",
                    "pattern": pattern,
                    "description": description,
                }
        return result
