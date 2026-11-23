# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Shared contracts for robot-specific run-result artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
