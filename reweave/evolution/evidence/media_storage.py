"""Share exact media bytes after an episode process exits, before publication.

The CLI is ``python -m reweave.cli.archive_storage ATTEMPT_DIRECTORY``.
This does not launch, resume or contact an episode. Online code does not import it.
"""

from __future__ import annotations

import fcntl
import json
import stat
import time
from pathlib import Path

from reweave.evolution.evidence.records import deduplicate_episode_media
from reweave.utils.persistence import atomic_json

MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def storage_usage(directory: Path) -> dict[str, int]:
    """Report path bytes, unique inode bytes, and allocated filesystem blocks."""
    seen = set()
    result = {"path_bytes": 0, "unique_bytes": 0, "allocated_bytes": 0}
    for path in directory.rglob("*"):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            continue
        result["path_bytes"] += info.st_size
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            seen.add(identity)
            result["unique_bytes"] += info.st_size
            result["allocated_bytes"] += info.st_blocks * 512
    return result


def finalize_media(attempt: Path) -> dict:
    """Freeze only completed episode media, then reuse exact-byte deduplication.

    Published evidence is immutable and rejected. After an interruption, rerun
    this storage step: permission normalization and atomic hardlink replacement
    are idempotent, and no physical/model execution is reachable from this module.
    """
    attempt = Path(attempt).resolve()
    episode = attempt / "episode"
    if not episode.is_dir() or episode.is_symlink():
        raise ValueError("expected an attempt containing an ordinary episode directory")
    with (attempt / ".episode.evidence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (episode / "episode_evidence.json").exists():
            raise ValueError("already published episode evidence must not be modified")
        exit_receipt = json.loads((attempt / "exit.json").read_text())
        if exit_receipt.get("returncode") != 0:
            raise ValueError("episode writer must have exited successfully")
        transcripts = list(episode.glob("transcript_*.json"))
        if len(transcripts) != 1:
            raise ValueError("expected one completed episode transcript")
        transcript = json.loads(transcripts[0].read_text())
        if transcript.get("cleanup_errors") != []:
            raise ValueError("episode writers did not close cleanly")
        close_path = episode / "environment_owner_exit.json"
        if close_path.exists():
            close = json.loads(close_path.read_text())
            if (
                close.get("returncode") != 0
                or close.get("recording_close_receipt_verified") is not True
                or close.get("parent_closed") is not True
            ):
                raise ValueError("environment writer and videos must have closed")
        media = []
        aliases = {}
        for path in episode.rglob("*"):
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("episode images must be ordinary regular files")
            identity = (info.st_dev, info.st_ino)
            aliases[identity] = aliases.get(identity, 0) + 1
            media.append((path, info, identity))
        # Permission changes must not reach another run through an external link.
        for _, info, identity in media:
            if info.st_nlink != aliases[identity]:
                raise ValueError("episode image has a hardlink outside this episode")
        receipt_path = episode / "archive_storage.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["complete"]:
                return receipt
        else:
            receipt = {
                "complete": False,
                "before": storage_usage(episode),
                "storage_elapsed_s": 0.0,
                "image_contract": "original encoded bytes and original pixel grids",
            }
            atomic_json(receipt_path, receipt)
        started = time.monotonic()
        try:
            # Online observation images are writable while native snapshots are
            # already read-only. Once all writers exit, both are immutable media.
            for path, info, _ in media:
                path.chmod(stat.S_IMODE(info.st_mode) & ~0o222)
            deduplicate_episode_media(episode)
            receipt["after"] = storage_usage(episode)
            receipt["complete"] = True
        finally:
            receipt["storage_elapsed_s"] += time.monotonic() - started
            atomic_json(receipt_path, receipt)
        return receipt
