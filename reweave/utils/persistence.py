# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Durable atomic publication of UTF-8 text and JSON files."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_text(path: Path, value: str) -> None:
    """Publish complete UTF-8 text and durably synchronize its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".publish-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete JSON snapshot and durably synchronize its directory."""
    atomic_text(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
