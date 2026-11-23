# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Static robot-extension descriptor.

Lives in :mod:`reweave.robots` alongside
:class:`~reweave.robots.prompt_bundle.PromptBundle` so robots
and planners can both import it without pulling in
:mod:`reweave.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`reweave.tools.toolkit.Toolkit` and its robot subclasses —
``RobotSpec`` carries the robot identity, the prompt bundle, and runner hooks
that keep CLI orchestration independent of concrete robot implementations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reweave.dashboard.events import DashboardEventSink
from reweave.dashboard.spec import DashboardSpec
from reweave.evaluation.result import RunFinalizer
from reweave.robots.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from reweave.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`RobotSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class RobotSpec:
    """Robot-level (non-tool) extension points for Harness."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink, set[str] | None],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    dashboard: DashboardSpec | None = None
    finalize_run: RunFinalizer | None = None
