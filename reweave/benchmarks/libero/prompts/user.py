# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""User prompt section bodies for a concrete LIBERO evaluation cell."""

from __future__ import annotations

CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
"""


MODE = """Use available images and tool feedback to solve task_language. Explicit
back_project/segment localization is optional unless the chosen action needs
geometry not already available; visually grounded VLA tools can act without it."""

BEGIN = """Start from the supplied observation. If the scene is not available, call
view_env_state(step=0). Choose the next justified operation; consult perception,
memory or guides only for a relevant uncertainty. Independent, fully specified
tool calls may share one response; do not batch feedback-dependent decisions."""
