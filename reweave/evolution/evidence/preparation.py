# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Author-visible evidence views over admitted episode archives and history."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reweave.evolution.authoring.references import AuthorReferences, history_originals
from reweave.evolution.contracts import (
    EvaluationConditions,
    GraphAnalysisRequest,
    GraphSnapshot,
    Proposal,
)
from reweave.evolution.evidence.trace import build_episode_context
from reweave.utils.persistence import atomic_json

_AUTHOR_SOURCE_PREFIXES = (
    "reweave/benchmarks/",
    "reweave/planner/",
    "reweave/runtime/",
    "reweave/robots/components/",
    "reweave/utils/rpc/",
)
_AUTHOR_SOURCE_FILES = frozenset(
    {
        "reweave/evolution/bundle.py",
        "reweave/session/__init__.py",
        "reweave/session/base.py",
    }
)
_SOURCE_PREFIX = "harness_context/source/"


def _author_source_files(files: dict[str, str]) -> dict[str, str]:
    """Expose online execution code, never the offline evolution implementation."""
    return {
        name: metadata
        for name, metadata in files.items()
        if name in _AUTHOR_SOURCE_FILES
        or any(name.startswith(prefix) for prefix in _AUTHOR_SOURCE_PREFIXES)
    }


def _evidence_roots(request: Proposal | GraphAnalysisRequest) -> dict[str, Path]:
    """Locate full episode archives without requiring pruned originals to exist."""
    roots: dict[str, Path] = {}
    for record in request.evidence:
        if record.episode is None:
            continue
        root = Path(
            os.path.commonpath(
                [str(Path(file.path).parent) for file in record.evidence]
            )
        )
        existing = roots.setdefault(record.episode.evidence_id, root)
        if existing != root:
            raise ValueError("one evidence identity resolves to multiple episode roots")
    return roots


def _history_paths(request: Proposal | GraphAnalysisRequest) -> tuple[Path, ...]:
    # The inventory grants read access, not a prerequisite to inspect every
    # historical artifact. Required originals are checked by their consumers.
    return tuple(Path(source) for source in history_originals(request)["files"])


def _native_evidence_paths(
    request: Proposal | GraphAnalysisRequest, evidence_roots: dict[str, Path]
) -> tuple[Path, ...]:
    """Expose episode originals to native readers without exposing source snapshots."""
    paths: set[Path] = set()
    for record in request.evidence:
        episode = record.episode
        if episode is None:
            # Retained files are admitted individually. They do not grant a
            # directory root or claim a missing control trace/source snapshot.
            paths.update(Path(file.path) for file in record.evidence)
            continue
        root = evidence_roots[episode.evidence_id]
        for name in episode.files:
            parts = Path(name).parts
            if parts[0] == "harness_context":
                if len(parts) < 2 or parts[1] == "source":
                    continue
                relative = Path(*parts[:2])
            else:
                relative = Path(parts[0])
            paths.add(root / relative)
    return tuple(sorted(paths))


def request_protocol(request: Proposal | GraphAnalysisRequest):
    return (
        request.protocol
        if isinstance(request, GraphAnalysisRequest)
        else request.task_protocol
    )


def declared_originals(request: Proposal | GraphAnalysisRequest, files: dict) -> dict:
    """Add every exact file named by the request or its execution graph."""
    files = dict(files)

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if {"path", "artifact_id", "bytes"} <= value.keys():
                files[value["path"]] = {
                    "artifact_id": value["artifact_id"],
                    "bytes": value["bytes"],
                }
            for item in value.values():
                visit(item)

    visit(request.model_dump(mode="json"))
    execution_graph = (
        request.graph
        if isinstance(request, GraphAnalysisRequest)
        else request.execution_graph
    )
    if execution_graph is not None:
        visit(
            GraphSnapshot.model_validate_json(
                Path(execution_graph.path).read_text()
            ).model_dump(mode="json")
        )
    return files


def retained_navigation(record, references: AuthorReferences) -> dict:
    """Navigation for a record whose complete episode archive was never retained."""
    return {
        "episode_id": record.record_id,
        "bundle_id": record.request.bundle_id,
        "condition_id": record.request.instance.record_id,
        "failure_origin": record.failure_origin,
        "official_success": record.official_success,
        "evidence_scope": "retained_files_only",
        "retained_files": references.encode(
            [file.model_dump(mode="json") for file in record.evidence]
        ),
        "unavailable": "Only the listed retained files are available. Do not assume a complete episode archive or reconstruct missing evidence.",
    }


def episode_gateway(record, source_root: Path) -> dict:
    """The frozen gateway contract actually used by the evaluated source bundle."""
    if "harness_context/bundle_contract.json" not in record.episode.files:
        raise ValueError("episode lacks the actual frozen bundle gateway contract")
    gateway = json.loads(
        (source_root / "harness_context/bundle_contract.json").read_text()
    )
    if gateway["bundle_id"] != record.request.bundle_id:
        raise ValueError("episode gateway differs from its evaluated source bundle")
    return gateway


def author_visible_sources(episode) -> dict[str, Any]:
    """Author-visible online implementation files of one executed snapshot."""
    archived = {
        name.removeprefix(_SOURCE_PREFIX): metadata
        for name, metadata in episode.files.items()
        if name.startswith(_SOURCE_PREFIX)
    }
    if not archived:
        raise ValueError(
            "historical evidence lacks its executed implementation sources"
        )
    sources = _author_source_files(archived)
    if not sources:
        raise ValueError(
            "historical evidence lacks author-visible online implementation sources"
        )
    return sources


def fixed_gateway(
    gateway: dict, execution: EvaluationConditions, source_root: Path
) -> dict:
    """Historical gateway tools under the currently published execution budgets."""
    fixed = {key: gateway[key] for key in ("operations", "episode_limits")}
    if "native_image_contract" in gateway:
        fixed["native_image_contract"] = gateway["native_image_contract"]
    # Historical episodes retain their original budgets. Development
    # and admission use the currently published execution conditions.
    fixed["episode_limits"] = {
        **fixed["episode_limits"],
        **execution.model_dump(
            include={"program_calls", "gateway_calls", "wall_time_s"}
        ),
    }
    planner_context = json.loads(
        (source_root / "harness_context/planner_context.json").read_text()
    )
    fixed["tools"] = json.loads(
        json.dumps(
            [
                spec
                for spec in planner_context["tools"]
                if spec["name"] in fixed["operations"]
            ]
        ).replace(str(source_root), "{{output_dir}}")
    )
    return fixed


def bind_frozen(path: Path, value, conflict: str) -> None:
    """All historical episodes of one request must share one frozen condition."""
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(conflict)
    atomic_json(path, value)


def publish_trace_context(
    evidence_workspace: Path,
    episode,
    root: Path,
    private: Path,
    trace_root: Path,
    references: AuthorReferences,
    transform: Callable[[Any], Any] = lambda value: value,
) -> tuple[str, dict]:
    """Index an episode once per run, then publish its referenced trace views.

    The raw run-owned index is linked into Host-private inputs; the author sees
    only encoded copies. Returns the trace handle and the raw trace index.
    """
    locks = evidence_workspace / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / episode.evidence_id).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        trace = build_episode_context(
            evidence_workspace, episode, root, include_visuals=False
        )
        relative = Path(trace["path"]).parent
        raw = private / "evidence" / relative
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(evidence_workspace / relative, raw, copy_function=os.link)
    trace_index = json.loads((private / "evidence" / trace["path"]).read_text())
    trace_id = references.identity("trace", episode.evidence_id)
    for original in sorted(raw.rglob("*.json")):
        value = json.loads(original.read_text())
        if original.name == "index.json":
            value["files"] = sorted(value["files"])
            value["contracts"].pop("executed_source", None)
            for media in value["media"]:
                media["path"] = str(
                    Path(media["source"]["path"]).relative_to(
                        Path(".archive/episodes") / episode.evidence_id
                    )
                )
        atomic_json(
            trace_root / trace_id / original.relative_to(raw),
            references.encode(transform(value)),
        )
    return trace_id, trace_index


def task_instructions(instructions: dict, references: AuthorReferences) -> list:
    return [
        {
            "official_text": official_text,
            "task_rule_supplement": task_rule_supplement,
            "text": text,
            "episode_ids": [
                references.identity("episode", episode) for episode in episodes
            ],
        }
        for (
            official_text,
            task_rule_supplement,
            text,
        ), episodes in instructions.items()
    ]


def history_index(originals: dict, references: AuthorReferences) -> dict:
    """Name every history original by handle; bundle originals read as bundle.json."""
    bundle_paths = {
        entry["bundle"]["path"] for entry in originals["navigation"].get("bundles", [])
    }
    # Allocate history handles together: each allocation persists the full catalog.
    handles = references.encode(
        [
            {"path": source, **metadata, "pointer": ""}
            for source, metadata in originals["files"].items()
        ]
    )
    return {
        "files": {
            handle: {
                "name": "bundle.json" if source in bundle_paths else Path(source).name
            }
            for source, handle in zip(originals["files"], handles, strict=True)
        },
        "navigation": references.encode(originals["navigation"]),
    }
