# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Durable atomic publication of UTF-8 text and JSON files."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_text(path: Path, value: str, *, exclusive: bool = False) -> None:
    """Publish UTF-8 text, optionally refusing to replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".publish-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            if exclusive:
                os.link(temporary, path)
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete JSON snapshot and durably synchronize its directory."""
    atomic_text(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
