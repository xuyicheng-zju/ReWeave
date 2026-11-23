# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Local-memory variant of the single-attempt LIBERO evaluation prompt."""

from __future__ import annotations

from reweave.benchmarks.libero.prompts import system as base
from reweave.prompt.utils import Numbered, PromptNode

MEMORY_PROFILE = base.MEMORY_GUIDANCE

WORKFLOW_STEPS = base.WORKFLOW_STEPS


def system_prompt() -> PromptNode:
    """Assemble the local global/task-family/task-specific evaluation prompt."""
    return {
        "ROLE AND EVALUATION": base.ROLE_AND_EVALUATION,
        "MEMORY PROFILE — LOCAL GLOBAL + TASK-FAMILY + TASK-SPECIFIC": MEMORY_PROFILE,
        "MANIPULATION GUIDANCE": base.MANIPULATION_GUIDANCE,
        "RUNTIME": base.RUNTIME,
        "DECISION COST": base.DECISION_COST,
        "OPTIONAL TOOL REFERENCES": base.STEP_READ_GUIDES,
        "YOUR GOAL": base.GOAL,
        "RULES (NON-NEGOTIABLE)": base.RULES,
        "LOCALIZATION": base.LOCALIZATION,
        "PERCEPTION WHEN NEEDED": base.PERCEPTION_GUIDANCE,
        "WORKFLOW": Numbered(WORKFLOW_STEPS),
        "KEY HYPERPARAMETERS": base.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": base.OUTPUT_DISCIPLINE,
    }


__all__ = ["system_prompt", "WORKFLOW_STEPS"]
