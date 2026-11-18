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

"""Common physical agent tools."""

from __future__ import annotations

import os
from pathlib import Path

from reweave.tools.toolkit import readonly
from reweave.utils.config import get_repo_root
from reweave.utils.logging import get_output_dir

TOOLS_SPEC: list[dict] = [
    {
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for past recipe JSONLs, audit JSONs, "
            "and memory files. Reads the complete file by default. For explicit "
            "max_chars pagination, continue at next_offset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or repo-relative path",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Zero-based Unicode character offset; use the preceding page's next_offset.",
                },
                "max_chars": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "Optional characters per page; null (default) reads all remaining characters.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use this to save "
            "the working recipe JSONL and the final audit JSON at the end of "
            "a successful run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = {{output_dir}}. "
            "Use to inspect the working directory or discover existing resource files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: {{output_dir}}"},
            },
        },
    },
    {
        "name": "finish",
        "description": (
            "Call when the task is complete or unrecoverable. Halts the agent "
            "loop. Save any artifacts (recipe, audit) BEFORE calling finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Outcome, e.g. 'success', 'failure', or 'stuck'.",
                },
                "summary": {
                    "type": "string",
                    "description": "Short natural-language summary of the run.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    return p


# Low-level file IO; planner-facing access is wrapped by ReadOnlyCorpus.
@readonly
def read_text_file(
    path: str,
    max_chars: int | None = None,
    *,
    offset: int = 0,
) -> dict:
    """Read an explicit UTF-8 page without altering source contents."""
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if max_chars is not None and (type(max_chars) is not int or max_chars <= 0):
        raise ValueError("max_chars must be a positive integer or null")
    p = _resolve(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        original = p.read_bytes()
        text = original.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"error": f"cannot read UTF-8 file {p}: {exc}"}
    if offset > len(text):
        raise ValueError("offset exceeds the file's character count")
    end = len(text) if max_chars is None else min(offset + max_chars, len(text))
    return {
        "path": str(p),
        "size": len(text),
        "content": text[offset:end],
        "offset": offset,
        "next_offset": end if end < len(text) else None,
        "has_more": end < len(text),
    }


@readonly
def write_text_file(path: str, content: str) -> dict:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


@readonly
def list_dir(path: str = "") -> dict:
    # Default to the current output dir (so parallel agents see their own).
    p = _resolve(path) if path else get_output_dir()
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


@readonly
def finish(status: str, summary: str) -> dict:
    """Signal that the run is complete. Halts the agent loop.

    Planners stop only when the Host returns the ``_finish`` sentinel.
    A submitted finish call or successful transport alone is not acceptance.
    """
    return {"_finish": True, "status": status, "summary": summary}


TOOL_HANDLERS: dict = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "finish": finish,
}
