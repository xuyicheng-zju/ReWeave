# Modified for the ReWeave namespace.
"""Lossless episode originals and ordinary published artifact records."""

from __future__ import annotations

import base64
import binascii
import fcntl
import json
import os
import re
import shutil
import stat
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reweave.evolution.bundle import FrozenDict
from reweave.utils.persistence import atomic_json

# Required even when a transcript exists before an episode can be a complete
# decision archive. Initialization failures retain a partial runner inventory.
REQUIRED_DECISION_EVIDENCE = frozenset(
    {
        "execution_trace.jsonl",
        "harness_context/capability_catalog.json",
        "harness_context/planner_context.json",
    }
)


def artifact_record(path: Path, *, artifact_id: str | None = None) -> dict:
    """Allocate file metadata at publication, or retain its already-published ID.

    Readers use the saved record. Size describes the original file; it is never
    treated as content equality or used to derive the record ID.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be an available regular file: {path}")
    if artifact_id is not None:
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("published artifact ID must be nonblank")
        return {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
        }
    return _artifact_records((path,))[0]


def _artifact_records(paths):
    """Publish all new IDs in one directory together, before returning any ID."""
    parent = paths[0].parent
    for path in paths:
        if path.parent != parent or path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact must be an available regular file: {path}")
    index = parent / ".artifact_records"
    with (parent / ".artifact_records.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(index.read_text()) if index.exists() else {}
        changed = False
        for path in paths:
            if path.name not in records:
                records[path.name] = {
                    "artifact_id": uuid.uuid4().hex,
                    "bytes": path.stat().st_size,
                }
                changed = True
        if changed:
            atomic_json(index, records)
        return [{**records[path.name], "path": str(path.resolve())} for path in paths]


def artifact_records(paths):
    """Publish ordinary metadata once per directory, including failed attempts."""
    directories = defaultdict(list)
    for path in paths:
        directories[path.parent].append(path)
    return [
        record for group in directories.values() for record in _artifact_records(group)
    ]


def implementation_sources() -> dict[str, str]:
    """Repository Python implementation, including imported helper modules.

    Excludes credentials, runtime files and installed third-party dependencies.
    The latter remain identified by the execution environment contract.
    """
    repository = Path(__file__).resolve().parents[3]
    return {
        path.relative_to(repository).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((repository / "reweave").rglob("*.py"))
        if "__pycache__" not in path.parts
    }


@contextmanager
def _byte_store(directory: Path):
    """One locked index per publication batch, with exact byte comparisons."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".records.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = directory / "records.json"
        records = json.loads(index.read_text()) if index.exists() else []
        by_size = defaultdict(list)
        for record in records:
            path = directory / record["path"]
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"archived byte record is unavailable: {path}")
            by_size[record["bytes"]].append(record)
        initial_count = len(records)
        matched = {}

        def store(data, *, suffix):
            if not re.fullmatch(r"\.[a-z0-9]+", suffix):
                raise ValueError("artifact suffix must be a simple extension")
            if data in matched:
                return matched[data]
            for record in by_size[len(data)]:
                if (directory / record["path"]).read_bytes() == data:
                    matched[data] = record
                    return record
            identity = f"image-{len(records) + 1:06d}"
            path = directory / (identity + suffix)
            path.write_bytes(data)
            record = {"artifact_id": identity, "path": path.name, "bytes": len(data)}
            records.append(record)
            matched[data] = record
            return record

        try:
            yield store
        finally:
            # Keep every completed byte publication even if a later item fails.
            if len(records) != initial_count:
                atomic_json(index, records)


def store_bytes(data: bytes, directory: Path, *, suffix: str) -> dict:
    """Store one ordinary byte record, reusing an existing exact byte match."""
    if not re.fullmatch(r"\.[a-z0-9]+", suffix):
        raise ValueError("artifact suffix must be a simple extension")
    with _byte_store(directory) as store:
        return store(data, suffix=suffix)


def archive_record(
    value: dict, directory: Path, *, media_directory: Path | None = None
) -> dict:
    """Retain a journal record with each inline image string stored exactly once.

    Canonical Base64 image strings share the decoded observation file. Other
    strings, including invalid candidate input, retain their exact UTF-8 bytes.
    image_payloads maps RFC6901 pointers to these ordinary local files; null
    leaves are restored exactly. media_directory explicitly selects the shared
    observation store for a transcript in the episode root. This never changes
    live requests, selected images, resolution, or provider history.
    """
    if "image_payloads" in value:
        raise ValueError("image_payloads is reserved for the evidence journal")
    media_directory = media_directory or directory / "observations"
    media_prefix = media_directory.relative_to(directory).as_posix()
    if media_prefix not in {"observations", "harness_context/observations"}:
        raise ValueError("image store must be the journal or episode observation store")
    references = {}
    pending = defaultdict(list)
    images = {}

    def queue(data, directory, suffix, prefix, encoding=None):
        reference = {}
        pending[directory].append((data, suffix, prefix, encoding, reference))
        return reference

    def store_image(text, media_type):
        extension = (
            {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
            }.get(media_type)
            if isinstance(media_type, str)
            else None
        )
        if extension is not None:
            encoding = "base64url" if "-" in text or "_" in text else "base64"
            try:
                decoded = base64.b64decode(
                    text,
                    altchars=b"-_" if encoding == "base64url" else None,
                    validate=True,
                )
            except (ValueError, binascii.Error):
                pass
            else:
                # Re-encoding must reproduce the original string, including its
                # padding. Invalid/noncanonical inputs remain exact text evidence.
                encoded = (
                    base64.urlsafe_b64encode(decoded)
                    if encoding == "base64url"
                    else base64.b64encode(decoded)
                )
                if encoded.decode("ascii") == text:
                    return queue(
                        decoded, media_directory, extension, media_prefix, encoding
                    )
        return queue(
            text.encode("utf-8"), directory / "image_payloads", ".txt", "image_payloads"
        )

    def visit(item, pointer):
        if isinstance(item, dict):
            image_keys = set()
            if "media_type" in item:
                if "data_base64" in item:
                    image_keys.add("data_base64")
                if item.get("kind") == "binary" or item.get("type") == "base64":
                    image_keys.add("data")
            result = {}
            for key, child in item.items():
                position = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                if key in image_keys and isinstance(child, str):
                    media_type = item["media_type"]
                    image = (child, media_type if isinstance(media_type, str) else None)
                    if image not in images:
                        images[image] = store_image(child, media_type)
                    references[position] = images[image]
                    result[key] = None
                else:
                    result[key] = visit(child, position)
            return result
        if isinstance(item, (list, tuple)):
            return [
                visit(child, f"{pointer}/{index}") for index, child in enumerate(item)
            ]
        return item

    record = visit(value, "")
    # Never hold locks for two stores at once. Mixed canonical and literal image
    # payloads cannot deadlock when concurrent journals encounter them in reverse.
    for store_directory, items in pending.items():
        with _byte_store(store_directory) as store:
            for data, suffix, prefix, encoding, reference in items:
                stored = store(data, suffix=suffix)
                reference.update(stored, path=f"{prefix}/{stored['path']}")
                if encoding is not None:
                    reference["encoding"] = encoding
    if references:
        record["image_payloads"] = references
    return record


def restore_record(value, directory: Path):
    """Resolve the optional image-payload table of one journal record losslessly."""
    if not isinstance(value, dict) or "image_payloads" not in value:
        return value
    record = deepcopy(value)
    references = record.pop("image_payloads")
    if not isinstance(references, dict):
        raise ValueError("invalid journal image table")
    for pointer, reference in references.items():
        if not isinstance(reference, dict) or set(reference) not in (
            {"artifact_id", "path", "bytes"},
            {"artifact_id", "path", "bytes", "encoding"},
        ):
            raise ValueError("invalid journal image reference")
        binary = "encoding" in reference
        expected_path = (
            re.fullmatch(
                r"(?:harness_context/)?observations/"
                + re.escape(str(reference["artifact_id"]))
                + r"\.(?:png|jpg|webp|gif)",
                reference["path"],
            )
            if binary and isinstance(reference["path"], str)
            else reference["path"] == f"image_payloads/{reference['artifact_id']}.txt"
        )
        if (
            not isinstance(reference["artifact_id"], str)
            or not re.fullmatch(r"image-[0-9]{6,}", reference["artifact_id"])
            or not expected_path
            or (
                binary
                and (
                    not isinstance(reference["encoding"], str)
                    or reference["encoding"] not in {"base64", "base64url"}
                )
            )
            or type(reference["bytes"]) is not int
            or reference["bytes"] < 0
            or not isinstance(pointer, str)
            or not pointer.startswith("/")
            or re.search(r"~(?![01])", pointer)
        ):
            raise ValueError("invalid journal image reference")
        path = directory / reference["path"]
        if (
            path.resolve() != directory.resolve() / reference["path"]
            or not path.is_file()
        ):
            raise ValueError("journal image payload is missing or follows a link")
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer[1:].split("/")
        ]
        target = record
        for index, part in enumerate(parts):
            key = part
            if isinstance(target, list):
                if not part.isascii() or not part.isdecimal() or str(int(part)) != part:
                    raise ValueError("journal image array index must be canonical")
                key = int(part)
            if index < len(parts) - 1:
                target = target[key]
            else:
                if target[key] is not None:
                    raise ValueError("journal image reference must replace a null leaf")
                data = path.read_bytes()
                if len(data) != reference["bytes"]:
                    raise ValueError(
                        "journal image payload length differs from its reference"
                    )
                target[key] = (
                    (
                        base64.urlsafe_b64encode(data)
                        if reference["encoding"] == "base64url"
                        else base64.b64encode(data)
                    ).decode("ascii")
                    if binary
                    else data.decode("utf-8")
                )
    return record


def deduplicate_episode_media(directory: Path) -> dict[str, int]:
    """Share exact image bytes after the episode's writers have all stopped.

    Ordinary filenames and file modes remain unchanged. Comparison uses the
    actual bytes, with size only narrowing candidates; existing hardlinks are
    counted once. This operates before evidence publication, never on an active
    episode or on already-published historical originals.
    """
    candidates: dict[tuple[int, int], list[Path]] = {}
    seen = set()
    linked = saved = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"episode media must be an ordinary file: {path}")
        metadata = path.stat()
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen:
            continue
        group = candidates.setdefault(
            (metadata.st_size, stat.S_IMODE(metadata.st_mode)), []
        )
        data = None
        for previous in group:
            if data is None:
                data = path.read_bytes()
            if previous.read_bytes() != data:
                continue
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                os.link(previous, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            linked += 1
            if metadata.st_nlink == 1:
                saved += metadata.st_size
            break
        else:
            group.append(path)
            seen.add(identity)
    return {"files_linked": linked, "duplicate_bytes_removed": saved}


class EpisodeEvidence(BaseModel):
    """An immutable inventory of all files in one completed episode directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(
        min_length=1,
        description="Ordinary ID allocated once when this episode inventory is published.",
    )
    files: dict[str, dict] = Field(
        description="Every original file by episode-relative path, with its published artifact_id and byte count."
    )

    @field_validator("files")
    @classmethod
    def _relative_files(cls, files: dict[str, dict]) -> dict[str, dict]:
        for name, record in files.items():
            path = Path(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.as_posix() != name
                or set(record) != {"artifact_id", "bytes"}
                or not isinstance(record["artifact_id"], str)
                or not record["artifact_id"]
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
            ):
                raise ValueError("invalid episode evidence file reference")
        return FrozenDict({name: FrozenDict(record) for name, record in files.items()})

    def verify(self, directory: Path) -> None:
        """Require retained originals without deriving or checking source identities."""
        for name in self.files:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"episode evidence file is unavailable: {path}")


def inventory_episode(directory: Path) -> dict[str, dict]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"episode evidence directory is unavailable: {directory}")
    files, directories = {}, defaultdict(list)
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"episode evidence cannot follow symlinks: {path}")
        if path.is_file():
            if path.name in {
                "episode_evidence.json",
                ".artifact_records",
                ".artifact_records.lock",
            }:
                continue
            directories[path.parent].append(path)
        elif not path.is_dir():
            raise ValueError(f"unsupported evidence file type: {path}")
    for paths in directories.values():
        for path, record in zip(paths, _artifact_records(paths), strict=True):
            files[path.relative_to(directory).as_posix()] = {
                key: record[key] for key in ("artifact_id", "bytes")
            }
    return files


def inspect_episode(source: Path) -> EpisodeEvidence:
    """Publish one inventory for a completed episode; later reads retain its IDs."""
    # Inventory publication shares one ordinary ID across concurrent readers.
    with (source.parent / f".{source.name}.evidence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record_path = source / "episode_evidence.json"
        if record_path.exists():
            evidence = EpisodeEvidence.model_validate_json(record_path.read_text())
            evidence.verify(source)
            return evidence
        files = inventory_episode(source)
        missing = REQUIRED_DECISION_EVIDENCE - files.keys()
        if missing:
            raise ValueError(
                f"episode is missing required decision evidence: {sorted(missing)}"
            )
        transcripts = [
            name
            for name in files
            if Path(name).parent == Path(".")
            and name.startswith("transcript_")
            and name.endswith(".json")
        ]
        if len(transcripts) != 1:
            raise ValueError(
                "episode evidence requires exactly one complete transcript"
            )
        evidence = EpisodeEvidence(evidence_id=uuid.uuid4().hex, files=files)
        atomic_json(record_path, evidence.model_dump(mode="json"))
        return evidence


def archive_episode(source: Path, archive_root: Path) -> EpisodeEvidence:
    """Archive the episode once and retain its original inventory."""
    if archive_root.resolve().is_relative_to(source.resolve()):
        raise ValueError(
            "an evidence archive cannot be nested inside its source episode"
        )
    evidence = inspect_episode(source)
    destination = archive_root / evidence.evidence_id
    if not destination.exists():
        shutil.copytree(source, destination)
    evidence.verify(destination)
    return evidence
