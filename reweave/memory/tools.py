# Copyright 2026 The RPent Authors.
# Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-aware file-tool handlers for the common tools.

These wrap the shared IO in :mod:`reweave.tools.common` with two access
checks: robot isolation and a read-only corpus boundary. Each handler
only validates the path and checks access, then delegates the actual IO.
"""

from __future__ import annotations

from pathlib import Path

from reweave.memory.experience import EXPERIENCE_SCOPES
from reweave.tools import common
from reweave.tools.toolkit import readonly
from reweave.utils.config import get_repo_root

# Current published text/reference scopes; results contains immutable run evidence.
_READABLE_SCOPES = {*EXPERIENCE_SCOPES, "results"}

# Suffix appended to shared file-tool descriptions when registered through
# the memory access boundary.
MEMORY_BOUNDARY_NOTE = (
    " Reference corpora are read-only in every run mode. "
    "Memory for other robots and internal archives are unavailable."
)


def _resolve_memory_path(path: str) -> Path:
    """Resolve a file-tool path before memory boundary checks."""
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    return p.resolve()


def _classify_memory_path(resolved: Path, *, memory_root: Path) -> str:
    """Classify a resolved path as current, foreign, or non_memory."""
    if resolved.is_relative_to(memory_root):
        return "current"
    memory_namespace = get_repo_root() / "memory"
    if resolved.is_relative_to(memory_namespace):
        return "foreign"
    return "non_memory"


def _check_memory_access(
    path: str,
    *,
    memory_root: Path,
    access: str,
) -> None:
    """Keep published reference files read-only and hide private archives."""
    if not path:
        return
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = get_repo_root() / supplied
    resolved = supplied.resolve()
    if supplied.is_relative_to(memory_root) and not resolved.is_relative_to(
        memory_root
    ):
        raise PermissionError(f"memory path escapes the corpus: {path}")
    bucket = _classify_memory_path(resolved, memory_root=memory_root)
    if bucket == "non_memory":
        return
    if bucket == "foreign":
        raise PermissionError(f"access to another robot's memory is denied: {path}")
    parts = resolved.relative_to(memory_root).parts
    if not parts:
        if access == "read":
            return
        raise PermissionError(f"writing to memory is denied in this mode: {path}")

    top = parts[0]
    if access == "write":
        raise PermissionError(f"writing to memory is denied in this mode: {path}")

    if top in _READABLE_SCOPES or (len(parts) == 1 and top.endswith(".md")):
        return
    raise PermissionError(f"reading this memory path is denied: {path}")


@readonly
def read_text_file(
    path: str,
    *,
    memory_root: Path,
    max_chars: int | None = None,
    offset: int = 0,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="read",
    )
    return common.read_text_file(path, max_chars, offset=offset)


@readonly
def write_text_file(
    path: str,
    content: str,
    *,
    memory_root: Path,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="write",
    )
    return common.write_text_file(path, content)


@readonly
def list_dir(
    path: str = "",
    *,
    memory_root: Path,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="read",
    )
    return common.list_dir(path)
