"""Identify native tool startup faults before output-format correction."""

import json

from reweave.evolution.infrastructure.session import (
    native_event_field,
    serialize_native_event,
)


class NativeToolStartupError(RuntimeError):
    """The native tool runtime could not start; model correction cannot fix it."""


def tool_startup_failure(event):
    if native_event_field(event, "method") not in {"item/completed", "error"}:
        return False
    text = json.dumps(serialize_native_event(event), ensure_ascii=False).lower()
    return "codex-code-mode-host" in text and any(
        marker in text
        for marker in (
            "permission denied",
            "os error 13",
            "no such file or directory",
            "failed to spawn",
        )
    )
