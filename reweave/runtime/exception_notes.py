# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Preserve exception notes on the Python 3.10."""

from __future__ import annotations


def add_exception_note(error: BaseException, note: str) -> None:
    """Attach an audit note without replacing the original failure."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        notes = getattr(error, "__notes__", None)
        if notes is None:
            notes = []
            error.__notes__ = notes
        notes.append(note)
