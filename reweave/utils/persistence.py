# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Durable atomic publication of UTF-8 text and JSON files."""

import json
import os
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any


@contextmanager
def atomic_path(path: Path, temporary: Path, *, publish=Path.replace, cleanup=Path.unlink):
    """Publish a staged file after its writer completes."""
    try:
        yield temporary
        publish(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            cleanup(temporary)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path: Path, value: str) -> None:
    """Publish complete UTF-8 text and durably synchronize its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".publish-", delete=False
    )
    with atomic_path(path, Path(stream.name)):
        with stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    sync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete JSON snapshot and durably synchronize its directory."""
    atomic_text(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
