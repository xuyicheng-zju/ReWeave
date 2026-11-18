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
