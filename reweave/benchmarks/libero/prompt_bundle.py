# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""LIBERO prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from reweave.benchmarks.libero.prompts import explore as explore_parts
from reweave.benchmarks.libero.prompts import local_eval as local_eval_parts
from reweave.benchmarks.libero.prompts import system as system_parts
from reweave.benchmarks.libero.prompts import user as user_parts
from reweave.prompt.utils import Numbered, PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if (variables or {}).get("frozen_bundle"):
        return {
            "ROLE AND EVALUATION": system_parts.ROLE_AND_EVALUATION,
            "MANIPULATION GUIDANCE": system_parts.MANIPULATION_GUIDANCE,
            "RUNTIME": system_parts.RUNTIME,
            "DECISION COST": system_parts.DECISION_COST,
            "YOUR GOAL": system_parts.GOAL,
            "LOCALIZATION": system_parts.LOCALIZATION,
            "MEMORY": system_parts.MEMORY_GUIDANCE
            if (variables or {}).get("memory_profile") != "none"
            else "No external memory; filesystem tools are disabled.",
            "EXECUTION": "Choose among the fixed tools and registered workflows using their current descriptions. Use observed feedback to decide the next operation. Workflow completion is a local result; the environment decides task success. Code, tools, prompts and scheduling policy remain fixed throughout this episode.",
            "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
        }
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    if (variables or {}).get("memory_profile", "local") == "local":
        return local_eval_parts.system_prompt()
    filesystem_enabled = (variables or {}).get("memory_profile", "local") != "none"
    steps = system_parts.WORKFLOW_STEPS
    if not filesystem_enabled:
        steps = (
            *system_parts.EXECUTION_STEPS,
            "Continue until environment success or truncation. "
            "The Host archives evidence; filesystem tools are disabled.",
        )
    return {
        "ROLE AND EVALUATION": system_parts.ROLE_AND_EVALUATION,
        "MANIPULATION GUIDANCE": (system_parts.MANIPULATION_GUIDANCE),
        "RUNTIME": system_parts.RUNTIME,
        "DECISION COST": system_parts.DECISION_COST,
        "YOUR GOAL": system_parts.GOAL,
        "RULES (NON-NEGOTIABLE)": system_parts.RULES,
        "LOCALIZATION — how to get an object's world xyz WITHOUT GT coords": (
            system_parts.LOCALIZATION
        ),
        "PERCEPTION WHEN NEEDED": (system_parts.PERCEPTION_GUIDANCE),
        "OPTIONAL REFERENCES": {
            "MEMORY": system_parts.MEMORY_GUIDANCE,
            "GUIDES": system_parts.STEP_READ_GUIDES,
        }
        if filesystem_enabled
        else "No external memory; filesystem tools are disabled.",
        "WORKFLOW": Numbered(steps),
        "KEY HYPERPARAMETERS": system_parts.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    filesystem_enabled = (variables or {}).get("memory_profile", "local") != "none"
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": (
            user_parts.BEGIN
            if filesystem_enabled
            else user_parts.BEGIN + " "
            "No filesystem tools or external memory are available; the Host archives the run."
        ),
    }


__all__ = ["system_prompt", "user_prompt"]
