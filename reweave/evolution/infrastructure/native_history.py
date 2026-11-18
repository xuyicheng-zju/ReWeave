# Modified for the ReWeave namespace.
"""Capture exact native journal lineage and fork completed read-only inquiries.

Independent mutation authors never call the fork path. Inquiry copies only
recorded native journals, uses its own authentication and read-only tools, and
cannot publish candidate edits or physical actions.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.session import _codex_rows
from reweave.utils.persistence import atomic_json


def _rollout_path(journal: Path, thread_id: str, identity: str) -> Path:
    """The target journal, or an ancestor mirrored in its adjacent lineage directory."""
    if identity == thread_id:
        return journal
    return journal.parent / "lineage" / f"{identity}.jsonl"


def native_history(
    journal: Path, thread_id: str, sessions: Path | None = None
) -> dict[str, bytes]:
    """Read one complete native lineage, preserving every rollout's exact bytes.

    Mirrors own ancestor snapshots in their adjacent ``lineage`` directory.
    Capture instead resolves ancestors in the provider's native sessions tree.
    Neither source permits a missing, conflicting or cyclic ancestry edge.
    """
    history: dict[str, bytes] = {}
    current = journal
    identity = thread_id
    while True:
        if identity in history:
            raise ValueError("cyclic Codex native history lineage")
        if current.is_symlink() or not current.is_file():
            raise ValueError(f"missing native history rollout: {identity}")
        raw = current.read_bytes()
        rows = _codex_rows(raw, identity)
        history[identity] = raw
        metadata = [row["payload"] for row in rows if row.get("type") == "session_meta"]
        edges = [
            (meta.get("forked_from_id"), meta.get("forked_from_ordinal_exclusive"))
            for meta in metadata
        ]
        if any(edge != edges[0] for edge in edges[1:]):
            raise ValueError("conflicting Codex native history lineage metadata")
        ancestor, ordinal = edges[0]
        if ancestor is None:
            if ordinal is not None:
                raise ValueError("native history ordinal has no source rollout")
            break
        if not isinstance(ancestor, str) or str(UUID(ancestor)) != ancestor:
            raise ValueError("native history source ID is not a canonical UUID")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("native history source ordinal is invalid")
        identity = ancestor
        if sessions is None:
            current = _rollout_path(journal, thread_id, identity)
        else:
            candidates = list(sessions.rglob(f"*-{identity}.jsonl"))
            if not candidates:
                raise ValueError(f"missing native history source rollout: {identity}")
            if any(path.is_symlink() for path in candidates):
                raise ValueError("native history source rollout cannot be a symlink")
            if len({path.read_bytes() for path in candidates}) != 1:
                raise ValueError(
                    f"conflicting native history source rollouts: {identity}"
                )
            current = candidates[0]
    return history


# Raw bytes per private transport response, not a model or physical-work quota.
HISTORY_CHUNK_BYTES = 256 * 1024


def _history_path(directory: Path, thread_id: str, identity: str) -> Path:
    return _rollout_path(directory / "transcript.jsonl", thread_id, identity)


def _checkpoint_files(checkpoint: dict, thread_id: str) -> dict:
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"checkpoint_id", "session_id", "files"}
        or checkpoint["session_id"] != thread_id
        or not isinstance(checkpoint["files"], dict)
        or thread_id not in checkpoint["files"]
    ):
        raise ValueError(
            "native history checkpoint requires its target and file manifest"
        )
    if str(UUID(checkpoint["checkpoint_id"])) != checkpoint["checkpoint_id"]:
        raise ValueError("native history checkpoint ID is not a canonical UUID")
    for identity, file in checkpoint["files"].items():
        if not isinstance(identity, str) or str(UUID(identity)) != identity:
            raise ValueError("native history checkpoint ID is not a canonical UUID")
        if (
            not isinstance(file, dict)
            or set(file) != {"bytes"}
            or type(file["bytes"]) is not int
            or file["bytes"] < 0
        ):
            raise ValueError("native history file requires a nonnegative byte count")
    return checkpoint["files"]


class NativeHistorySource:
    """An immutable native closure served through bounded private byte reads.

    checkpoint.session_id names the target rollout; files maps every rollout ID
    in its ancestry to its complete byte count. A checkpoint ID identifies this
    transfer snapshot; received chunks are compared with actual staged bytes.
    """

    def __init__(self, journal: Path, thread_id: str):
        self._history = native_history(journal, thread_id)
        self.checkpoint = {
            "checkpoint_id": str(uuid4()),
            "session_id": thread_id,
            "files": {
                identity: {"bytes": len(raw)} for identity, raw in self._history.items()
            },
        }
        _checkpoint_files(self.checkpoint, thread_id)

    def handle(self, method: str, params: dict) -> dict:
        if method != "history_chunk" or set(params) != {"rollout_id", "offset"}:
            raise ValueError("unknown native history read")
        identity, offset = params["rollout_id"], params["offset"]
        raw = self._history.get(identity)
        if raw is None or type(offset) is not int or not 0 <= offset < len(raw):
            raise ValueError("native history chunk identity or offset is invalid")
        return {
            "rollout_id": identity,
            "offset": offset,
            "data": base64.b64encode(raw[offset : offset + HISTORY_CHUNK_BYTES]).decode(
                "ascii"
            ),
        }


def _prepare_history_receive(checkpoint, thread_id, directory, existing_journal):
    files = _checkpoint_files(checkpoint, thread_id)
    identity = checkpoint["checkpoint_id"]
    transfer = directory / identity
    if directory.is_symlink() or transfer.is_symlink():
        raise ValueError("native history transfer cannot be a symlink")
    transfer.mkdir(parents=True, exist_ok=True)
    manifest = transfer / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != checkpoint:
        raise ValueError("native history transfer checkpoint changed")
    atomic_json(manifest, checkpoint)
    history = transfer / "history"
    if history.is_symlink() or (history / "lineage").is_symlink():
        raise ValueError("native history transfer cannot be a symlink")
    for rollout, file in files.items():
        original = (
            None
            if existing_journal is None
            else _rollout_path(existing_journal, thread_id, rollout)
        )
        if original is not None and original.is_symlink():
            raise ValueError("native history checkpoint cannot be a symlink")
        target = _history_path(history, thread_id, rollout)
        if target.is_symlink():
            raise ValueError("native history transfer cannot be a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if (
                original is not None
                and original.exists()
                and original.stat().st_size <= file["bytes"]
            ):
                shutil.copyfile(original, target)
            else:
                target.touch(mode=0o600)
        if target.stat().st_size > file["bytes"]:
            raise ValueError("native history partial exceeds the checkpoint size")
    return transfer, history


def _reset_history_prefix(target):
    # Preserve a mismatched staged copy; published native journals stay untouched.
    rejected = target.with_name(target.name + ".rejected-" + uuid4().hex)
    target.rename(rejected)
    target.touch(mode=0o600)


def _existing_chunk(target, offset, size):
    with target.open("rb") as stream:
        stream.seek(offset)
        return stream.read(size)


def _append_history_chunk(target, offset, raw):
    with target.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        if stream.tell() != offset:
            raise ValueError(
                "native history chunk offset changed or was already received"
            )
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_received_history(history, checkpoint, thread_id):
    for identity, file in checkpoint["files"].items():
        path = _history_path(history, thread_id, identity)
        if path.stat().st_size != file["bytes"]:
            raise ValueError("native history received bytes differ from the checkpoint")
    journal = history / "transcript.jsonl"
    if set(native_history(journal, thread_id)) != set(checkpoint["files"]):
        raise ValueError("native history checkpoint contains unrelated rollouts")
    return journal


async def receive_history(
    checkpoint: dict,
    thread_id: str,
    directory: Path,
    request: Callable[[str, dict], Awaitable[dict]],
    *,
    existing_journal: Path | None = None,
) -> Path:
    """Receive bounded chunks, comparing actual staged bytes on interrupted reads.

    Completed checkpoint IDs read their cached result. Partial receives compare
    source bytes before reusing the local prefix; published mirrors stay intact.
    """
    transfer, history = await run_blocking(
        _prepare_history_receive, checkpoint, thread_id, directory, existing_journal
    )
    if (transfer / "received.json").exists():
        return await run_blocking(
            _verify_received_history, history, checkpoint, thread_id
        )
    for identity, file in checkpoint["files"].items():
        target = _history_path(history, thread_id, identity)
        offset = 0
        while offset < file["bytes"]:
            chunk = await request(
                "history_chunk", {"rollout_id": identity, "offset": offset}
            )
            if (
                not isinstance(chunk, dict)
                or set(chunk) != {"rollout_id", "offset", "data"}
                or chunk["rollout_id"] != identity
                or type(chunk["offset"]) is not int
                or chunk["offset"] != offset
                or not isinstance(chunk["data"], str)
                or len(chunk["data"]) > 4 * ((HISTORY_CHUNK_BYTES + 2) // 3)
            ):
                raise ValueError(
                    "native history chunk response is malformed or out of order"
                )
            raw = base64.b64decode(chunk["data"], validate=True)
            if len(raw) != min(HISTORY_CHUNK_BYTES, file["bytes"] - offset):
                raise ValueError("native history chunk has an incorrect byte count")
            existing = await run_blocking(_existing_chunk, target, offset, len(raw))
            if existing != raw[: len(existing)]:
                await run_blocking(_reset_history_prefix, target)
                offset = 0
                continue
            if len(existing) < len(raw):
                await run_blocking(
                    _append_history_chunk,
                    target,
                    offset + len(existing),
                    raw[len(existing) :],
                )
            offset += len(raw)
    journal = await run_blocking(
        _verify_received_history, history, checkpoint, thread_id
    )
    await run_blocking(atomic_json, transfer / "received.json", checkpoint)
    return journal
