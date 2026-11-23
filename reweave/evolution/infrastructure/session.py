# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Branch-owned native author journals, with provider-specific resume identities."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path, PureWindowsPath
from typing import Any


def native_event_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a provider event field from SDK root models, objects or raw JSON."""
    value = getattr(value, "root", value)
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def serialize_native_event(value: Any) -> Any:
    """Persist complete supported SDK values; unsupported objects fail visibly."""
    value = getattr(value, "root", value)
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_native_event(asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): serialize_native_event(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_native_event(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported SDK audit value: {type(value).__name__}")


def _validate_session_id(identity: str) -> None:
    # The SDK skips SessionStore materialization for non-UUID resume values.
    if re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        identity,
    ) is None:
        raise ValueError("Claude native session ID must be a UUID for durable resume")


def _key_identity(key: dict[str, Any]) -> tuple[str, str | None]:
    """Validate the SDK locator while keeping project relocation non-semantic."""
    if not isinstance(key, dict) or set(key) - {"project_key", "session_id", "subpath"}:
        raise ValueError("invalid Claude native session key")
    for field in ("project_key", "session_id"):
        if not isinstance(key.get(field), str) or not key[field].strip():
            raise ValueError(f"Claude native session key requires {field}")
    _validate_session_id(key["session_id"])
    subpath = key.get("subpath")
    if "subpath" in key and (not isinstance(subpath, str) or not subpath):
        raise ValueError("Claude native session subpath must be omitted or non-empty")
    if subpath is not None and (
        subpath.startswith(("/", "\\"))
        or PureWindowsPath(subpath).drive
        or any(part in {".", ".."} for part in re.split(r"[\\/]", subpath))
        or "\x00" in subpath
    ):
        raise ValueError(
            "Claude native subpath cannot be safely materialized for resume"
        )
    return key["session_id"], subpath


class ClaudeSessionStore:
    """SDK SessionStore backed by one lossless journal inside one author job.

    Project keys locate the CLI working directory and may change on relocation.
    The bound session ID and subpath own identity inside this private store.
    Every revision remains in the journal; load projects the latest UUID revision.
    """

    def __init__(self, directory: Path):
        if directory.is_symlink():
            raise ValueError("Claude session directory cannot be a symlink")
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "transcript.jsonl"
        self.error: str | None = None
        self.session_id: str | None = None

    def _rows(self) -> list[dict[str, Any]]:
        if self.path.is_symlink():
            raise ValueError("Claude native transcript cannot be a symlink")
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise ValueError(
                        "Claude native transcript has an incomplete final record"
                    )
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"key", "entry"}:
                    raise ValueError(
                        f"invalid Claude native transcript record {number}"
                    )
                identity, _subpath = _key_identity(row["key"])
                entry = row["entry"]
                if not isinstance(entry, dict):
                    raise ValueError(f"invalid Claude native transcript entry {number}")
                if "uuid" in entry and (
                    not isinstance(entry["uuid"], str) or not entry["uuid"]
                ):
                    raise ValueError(
                        f"invalid Claude native transcript UUID at record {number}"
                    )
                if self.session_id is not None and identity != self.session_id:
                    raise ValueError(
                        "Claude native transcript belongs to a different session"
                    )
                rows.append(row)
        return rows

    def bind(self, session_id: str) -> None:
        """Require all mirrored records to belong to the observed provider session."""
        _validate_session_id(session_id)
        if self.session_id is not None and self.session_id != session_id:
            raise ValueError("Claude native store is bound to a different session")
        self.session_id = session_id
        self._rows()

    async def append(self, key: dict[str, Any], entries: list[dict[str, Any]]) -> None:
        """Retain every received revision; SDK load alone performs UUID projection."""
        try:
            identity, _subpath = _key_identity(key)
            if self.session_id is not None and identity != self.session_id:
                raise ValueError("Claude native append belongs to a different session")
            if self.path.is_symlink():
                raise ValueError("Claude native transcript cannot be a symlink")
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) for entry in entries
            ):
                raise ValueError(
                    "Claude native append requires a list of entry objects"
                )
            payload = "".join(
                json.dumps(
                    {"key": key, "entry": entry}, ensure_ascii=False, allow_nan=False
                )
                + "\n"
                for entry in entries
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            self.error = str(exc)
            raise

    def _load(self, key: dict[str, Any]) -> list[dict[str, Any]] | None:
        requested = _key_identity(key)
        if self.session_id is not None and requested[0] != self.session_id:
            raise ValueError("Claude native load requested a different session")
        entries: list[dict[str, Any]] = []
        positions: dict[str, int] = {}
        for row in self._rows():
            if _key_identity(row["key"]) != requested:
                continue
            entry = row["entry"]
            identity = entry.get("uuid")
            if identity is not None and identity in positions:
                entries[positions[identity]] = entry
            else:
                if identity is not None:
                    positions[identity] = len(entries)
                entries.append(entry)
        if not entries and self.session_id is not None:
            raise ValueError("requested Claude native conversation is missing")
        return entries or None

    async def load(self, key: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Load exact latest revisions; a bound missing conversation cannot fall back."""
        return self._load(key)

    async def list_subkeys(self, key: dict[str, Any]) -> list[str]:
        identity, subpath = _key_identity(key)
        if subpath is not None:
            raise ValueError("list_subkeys requires a main-session key")
        if self.session_id is not None and identity != self.session_id:
            raise ValueError("Claude subkey lookup requested a different session")
        return sorted(
            {
                row["key"]["subpath"]
                for row in self._rows()
                if row["key"]["session_id"] == identity and "subpath" in row["key"]
            }
        )

    def checkpoint(self) -> dict[str, Any]:
        """Record the complete native journal size, including revisions and subagents."""
        if self.error:
            raise ValueError(
                f"Claude native transcript persistence failed: {self.error}"
            )
        rows = self._rows()
        if not rows:
            raise ValueError(
                "Claude returned without persisting its native conversation"
            )
        if self.session_id is not None:
            self._load({"session_id": self.session_id, "project_key": "checkpoint"})
        data = self.path.read_bytes()
        return {"bytes": len(data)}


def _codex_rows(raw: bytes, identity: str | None) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Codex native transcript has an incomplete final record")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid Codex native transcript record")
    identities = {
        row.get("payload", {}).get("id")
        for row in rows
        if row.get("type") == "session_meta"
    }
    if len(identities) != 1 or None in identities:
        raise ValueError("Codex native transcript requires one session identity")
    if identity is not None and identities != {identity}:
        raise ValueError("Codex native transcript belongs to a different session")
    return rows


class CodexSessionStore:
    """Mirror Codex's complete native JSONL, without translating provider records."""

    def __init__(self, directory: Path):
        if directory.is_symlink():
            raise ValueError("Codex session directory cannot be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "transcript.jsonl"
        self.session_id: str | None = None

    def _rows(self) -> list[dict[str, Any]]:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Codex native session transcript is missing")
        raw = self.path.read_bytes()
        return _codex_rows(raw, self.session_id)

    def bind(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Codex returned no native thread ID")
        if self.session_id is not None and self.session_id != session_id:
            raise ValueError("Codex native store is bound to a different session")
        self.session_id = session_id
        if self.path.exists():
            self._rows()

    def capture(self, native_path: Path) -> None:
        """Require the provider's original journal and retain its exact bytes.

        Resume uses the provider registry and session ID. The mirror records the
        current journal; provider metadata updates never gate continuation.
        """
        if native_path.is_symlink() or not native_path.is_file():
            raise ValueError("Codex native session log is missing")
        raw = native_path.read_bytes()
        rows = _codex_rows(raw, self.session_id)
        from reweave.evolution.infrastructure.native_history import native_history
        from reweave.utils.persistence import atomic_text

        identity = next(
            row["payload"]["id"] for row in rows if row.get("type") == "session_meta"
        )
        sessions = next(
            (parent for parent in native_path.parents if parent.name == "sessions"),
            None,
        )
        history = native_history(native_path, identity, sessions)
        for ancestor, snapshot in history.items():
            if ancestor == identity:
                continue
            destination = self.path.parent / "lineage" / f"{ancestor}.jsonl"
            if destination.exists() and destination.read_bytes() != snapshot:
                raise ValueError("native history ancestor snapshot changed")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_text(destination, snapshot.decode("utf-8"))
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def checkpoint(self) -> dict[str, Any]:
        self._rows()
        raw = self.path.read_bytes()
        return {"bytes": len(raw)}
