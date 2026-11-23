# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Lightweight types for robot-owned Dashboard configuration."""

from __future__ import annotations

from typing import Literal, TypedDict


class TaskFieldSpecRequired(TypedDict):
    name: str


class TaskFieldSpec(TaskFieldSpecRequired, total=False):
    kind: Literal["integer"]
    minimum: int
    choices: tuple[str, ...]
    suggestions: tuple[str, ...]


class TaskSpec(TypedDict):
    command: str
    usage: str
    fields: tuple[TaskFieldSpec, ...]
    display: str
    output_slug: str


class RuntimeComponentSpec(TypedDict):
    name: str
    label: str
    scope: Literal["shared", "unique"]


class FrameChannelSpec(TypedDict):
    name: str
    label: str
    artifact: str


class DashboardSpec(TypedDict):
    task: TaskSpec
    runtime_components: tuple[RuntimeComponentSpec, ...]
    frame_channels: tuple[FrameChannelSpec, ...]
    primitives: tuple[str, ...]
