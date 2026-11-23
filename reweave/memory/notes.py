# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Planner-owned episode drafts and immutable originals for later C authoring."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from reweave.session.base import EnvState
    from reweave.tools.toolkit import Toolkit

from reweave.memory.experience import TASK_FAMILY_FORMAT
from reweave.tools import common
from reweave.tools.toolkit import readonly
from reweave.utils.persistence import atomic_json, atomic_text


class EpisodeNotes:
    """Own text only inside one episode's output; never update shared knowledge."""

    def __init__(self, episode_root: Path, state: EnvState) -> None:
        self.root = episode_root.resolve() / "planner_memory"
        self.drafts = self.root / "drafts"
        self._episode_root = episode_root.resolve()
        self._state = state
        self._lock = threading.Lock()

    def contains(self, path: Path) -> bool:
        return path.is_relative_to(self.root)

    @property
    def has_writes(self) -> bool:
        """A retained revision proves that this episode submitted a real note."""
        with self._lock:
            return any((self.root / "revisions").glob("*.json"))

    def _path(self, path: str, *, write: bool = False) -> Path:
        supplied = Path(path)
        if not supplied.is_absolute():
            supplied = self.root / supplied
        resolved = supplied.resolve()
        if not resolved.is_relative_to(self.drafts):
            raise PermissionError(
                "Planner memory access is confined to this episode's drafts"
            )
        if any(
            parent.is_symlink()
            for parent in [supplied, *supplied.parents]
            if parent.is_relative_to(self.root)
        ):
            raise PermissionError("Planner memory paths cannot contain symlinks")
        if write and (resolved.suffix != ".md" or resolved.name.startswith(".")):
            raise ValueError("Planner notes must be visible Markdown files")
        if write:
            parts = resolved.relative_to(self.drafts).parts
            if len(parts) < 2 or parts[0] != "task-family":
                raise ValueError(
                    "During evolution Planner notes may write only drafts/task-family/; "
                    "global is read-only and task-specific is Host-owned"
                )
        return resolved

    def _sources(self) -> list[dict]:
        sources = []
        for name in (
            "harness_context/planner_requests.jsonl",
            "harness_context/bundle_events.jsonl",
        ):
            path = self._episode_root / name
            if path.is_file():
                size = path.stat().st_size
                sources.append(
                    {
                        "path": name,
                        "prefix_bytes": size,
                    }
                )
        return sources

    @readonly
    def read_text_file(
        self,
        path: str,
        *,
        max_chars: int | None = None,
        offset: int = 0,
    ) -> dict:
        return common.read_text_file(
            str(self._path(path)),
            max_chars,
            offset=offset,
        )

    @readonly
    def list_dir(self, path: str = "") -> dict:
        target = (
            self.drafts
            if not path or Path(path).resolve() == self.root
            else self._path(path)
        )
        if not target.exists():
            return {"path": str(target), "count": 0, "files": []}
        return common.list_dir(str(target))

    @readonly
    def write_text_file(self, path: str, content: str) -> dict:
        """Keep every submitted version before replacing the readable draft."""
        target = self._path(path, write=True)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Planner notes must contain nonblank text")
        with self._lock:
            raw = content.encode("utf-8")
            revision = self.root / "revisions" / f"{uuid4().hex}.json"
            record = {
                "draft": target.relative_to(self.root).as_posix(),
                "text": content,
                "bytes": len(raw),
                "current_step": getattr(self._state, "latest_step", None),
                "public_sources": self._sources(),
                "verification": "unverified",
            }
            atomic_json(revision, record)
            atomic_text(target, content)
        return {
            "path": str(target),
            "revision": revision.relative_to(self._episode_root).as_posix(),
            "verification": "unverified",
            "written": True,
        }


def planner_memory_guidance(
    toolkit: Toolkit, *, adaptation_mode: str = "static"
) -> str:
    """Harness-style failed-attempt notes, kept private for later coding review."""
    if adaptation_mode not in {"static", "adaptive"}:
        raise ValueError("adaptation_mode must be static or adaptive")
    notes = getattr(toolkit, "episode_notes", None)
    if notes is None or "write_text_file" not in {
        item["name"] for item in toolkit.get_tools_spec()
    }:
        return ""
    source_guidance = (
        "Published corpus files remain fixed during this episode. You may revise "
        "this episode's private C skills in its harness source; changes activate "
        "at settled revision boundaries, while active calls retain their bound "
        "revision. Draft notes remain private Markdown evidence. "
        if adaptation_mode == "adaptive"
        else "Published corpus files and C skills are fixed during this episode. "
        "Only your draft text is writable. "
    )
    return (
        "Write useful experience online with write_text_file under "
        f"{notes.drafts}/task-family/working-notes.md; read your own draft only "
        "when needed with read_text_file/list_dir. Keep one note for this task, "
        "updating attempts, recovery and close-out sections instead of phase files. "
        "CLOSE OUT FAILED APPROACHES: before abandoning a failed approach or "
        "handing off, briefly record what was tried, actual parameters, observed "
        "result and public evidence references, revised hypotheses and uncertainty. "
        "Do not log every tool return or defer useful findings to offline review. "
        "BEFORE FINISH: update the same note with the latest observed outcome, "
        "positive/negative evidence and questions for the next agent; write a "
        "concise close-out if no note exists. Do this within existing budgets, "
        "without extra physical actions or an invented winning recipe. "
        "During evolution write only task-family experience. Existing global is "
        "read-only; the Host automatically records task-specific facts and traces. "
        "Cite those facts instead of copying an action log. Record observable "
        "applicability, counterexamples and uncertain causes; task/seed IDs are "
        "provenance, not dispatch keys. Re-ground current geometry. Historical "
        "success and notes never replace the fixed evaluator. "
        + source_guidance
        + "The existing offline author reviews, corrects and deduplicates these "
        "private revisions. After fixed-evaluator success it organizes the full "
        "Harness into global/task-family and Host-owned task-specific. Nothing is automatically "
        "merged or verified. Sibling episodes cannot see these drafts; subsequent "
        "batches receive them only after the whole round settles. Working notes "
        "may be plain Markdown; preserve the following format when editing a "
        "durable task-family C skill, with scope/id matching its directory. "
        "Use evidence actually received; label unknown reliability and untested "
        "ranges, distinguish causal hypotheses from observations, and reuse "
        "covered lessons rather than duplicate them.\n\n" + TASK_FAMILY_FORMAT
    )
