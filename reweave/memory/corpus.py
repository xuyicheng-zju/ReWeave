# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Read-only access to an explicitly supplied local reference corpus."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reweave.memory.notes import EpisodeNotes

from reweave.memory.experience import EXPERIENCE_SCOPES


class ReadOnlyCorpus:
    """Bind file tools without learning, publishing, or downloading knowledge."""

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self._root = Path(root).resolve()
        # local/none controls published access independently of episode drafts.
        self._enabled = enabled
        for removed in ("suite", "task_only"):
            if enabled and (self._root / removed).exists():
                raise ValueError(
                    f"removed corpus directory {removed!r}; explicitly prepare "
                    f"a fresh corpus with {'/'.join(EXPERIENCE_SCOPES)} scopes "
                    "and results evidence"
                )

    @property
    def root(self) -> Path:
        return self._root

    def get_common_tool_bindings(
        self,
        *,
        notes: EpisodeNotes | None = None,
        confined: bool = False,
    ) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        from reweave.memory import tools as memory_tools
        from reweave.tools import common
        from reweave.tools.toolkit import readonly

        handlers = {
            name: partial(getattr(memory_tools, name), memory_root=self._root)
            for name in ("read_text_file", "write_text_file", "list_dir")
        }
        if notes is not None or confined or not self._enabled:
            for name in handlers:
                published = handlers[name]
                episode = getattr(notes, name) if notes is not None else None

                def route(
                    *args, _published=published, _episode=episode, _name=name, **kwargs
                ):
                    value = kwargs.get("path") or (args[0] if args else "")
                    supplied = Path(value)
                    if (
                        notes is not None
                        and supplied.is_absolute()
                        and notes.contains(supplied)
                    ):
                        return _episode(*args, **kwargs)
                    resolved = memory_tools._resolve_memory_path(value)
                    if (
                        "planner_memory" in supplied.parts
                        or "planner_memory" in resolved.parts
                    ):
                        raise PermissionError(
                            "another episode's Planner notes are unavailable"
                        )
                    if not self._enabled and (
                        not value or resolved.is_relative_to(self._root)
                    ):
                        raise PermissionError(
                            "published memory is disabled by the none profile"
                        )
                    if confined:
                        if _name == "write_text_file":
                            raise PermissionError(
                                "writing is confined to this episode's Planner drafts"
                            )
                        if not value:
                            return _published(**{**kwargs, "path": str(self._root)})
                        guides = (
                            memory_tools.get_repo_root()
                            / "reweave/benchmarks/libero/guides"
                        )
                        if not resolved.is_relative_to(
                            self._root
                        ) and not resolved.is_relative_to(guides):
                            raise PermissionError(
                                "file reads are confined to published memory, fixed guides and this episode's Planner drafts"
                            )
                    return _published(*args, **kwargs)

                handlers[name] = readonly(route)
        bindings = {}
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            if name not in handlers:
                continue
            tool_spec = dict(spec)
            tool_spec["description"] += memory_tools.MEMORY_BOUNDARY_NOTE
            if notes is not None:
                tool_spec["description"] += (
                    " This episode also has Planner-owned Markdown drafts under "
                    "{{output_dir}}/planner_memory/drafts/task-family/; "
                    "only those notes may be updated; global is read-only and task-specific is Host-owned. "
                    "Every version is preserved for offline C review. "
                )
            bindings[name] = (tool_spec, handlers[name])
        return bindings
