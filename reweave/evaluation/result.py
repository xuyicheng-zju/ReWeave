# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Shared contracts for robot-specific run-result artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reweave.utils.persistence import atomic_path


@dataclass(frozen=True)
class RunFinalizationContext:
    """Structured runner state supplied to a robot result finalizer."""

    output_dir: Path
    robot_name: str
    task_desc: Mapping[str, Any]
    environment_success: bool | None
    termination_class: str
    agent_error: str | None
    elapsed_s: float
    planner: str
    model: str | None
    reasoning_effort: str
    max_turns: int
    planner_timeout_s: int | None
    finish_result: Mapping[str, Any] | None
    stats: Mapping[str, Any]


RunFinalizer = Callable[[RunFinalizationContext], Path | str | None]


def write_json_atomic(
    destination: Path | str,
    record: Mapping[str, Any],
) -> Path:
    """Write one JSON object atomically and return its destination path."""
    path = Path(destination)
    with atomic_path(path, path.with_name(f".{path.name}.tmp")) as temporary:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
    return path
