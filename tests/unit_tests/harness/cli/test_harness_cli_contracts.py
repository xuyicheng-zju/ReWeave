# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Offline checks for the public harness switches and Planner prompt contract."""

from __future__ import annotations

import argparse

import pytest

from reweave.benchmarks.libero import prompt_bundle as libero_prompts
from reweave.benchmarks.libero import robot_spec as libero_spec
from reweave.dashboard.events import NullDashboardEventSink
from reweave.prompt.utils import format_prompt
from reweave.tools.toolkit import Toolkit


@pytest.mark.parametrize(
    "removed_arguments",
    [
        ["--action-monitoring", "fixed_requery"],
        ["--action-monitoring", "runtime_refinement"],
        ["--online-modification"],
        ["--no-online-modification"],
        ["--task-harness", "harness.json"],
        ["--refinement-candidate", "candidate.json"],
        ["--online-author-mode", "episode_local"],
    ],
)
def test_robot_cli_rejects_retired_harness_contract(removed_arguments):
    parser = argparse.ArgumentParser()
    libero_spec._add_cli_args(parser, use_dashboard=True)

    with pytest.raises(SystemExit) as error:
        parser.parse_args(removed_arguments)

    assert error.value.code == 2


def test_memory_none_keeps_shared_tool_guidance_without_unavailable_file_workflow():
    variables = {
        "output_dir": "/offline/output",
        "recipe_tag": "task-s0",
        "memory_dir": "/offline/memory",
        "reference_tag": "task-s0",
    }
    no_memory_prompt = format_prompt(
        libero_prompts.system_prompt({"memory_profile": "none"}),
        variables=variables,
    )
    upstream_prompt = format_prompt(
        libero_prompts.system_prompt({"memory_profile": "local"}),
        variables=variables,
    )
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        filesystem_tools_enabled=False,
    )
    tool_names = {item["name"] for item in toolkit.get_tools_contract_spec()}

    from reweave.benchmarks.libero.prompts import system

    assert system.MANIPULATION_GUIDANCE in no_memory_prompt
    assert system.MANIPULATION_GUIDANCE in upstream_prompt
    assert "write_text_file" not in no_memory_prompt
    assert "/offline/memory" not in no_memory_prompt
    assert "filesystem tools are disabled" in no_memory_prompt
    assert "write_text_file" in upstream_prompt
    assert "mandatory all-object table" in no_memory_prompt
    assert tool_names.isdisjoint({"read_text_file", "write_text_file", "list_dir"})
    assert "finish" in tool_names


@pytest.mark.parametrize(
    "variables",
    [
        {"memory_profile": "local"},
        {"memory_profile": "none"},
        {"frozen_bundle": True},
        {"frozen_bundle": True, "memory_profile": "none"},
        {"mode": "explore"},
    ],
)
def test_libero_prompt_variants_allow_batching_without_perception_ritual(variables):
    from reweave.benchmarks.libero.prompts import system

    bindings = dict.fromkeys(
        (
            "suite",
            "task",
            "seed",
            "output_dir",
            "memory_dir",
            "recipe_tag",
            "reference_tag",
            "session_number",
            "session_max",
        ),
        "fixture",
    )
    rendered = format_prompt(
        libero_prompts.system_prompt(variables), variables=bindings
    )
    user = format_prompt(libero_prompts.user_prompt(variables), variables=bindings)
    assert system.DECISION_COST in rendered
    assert "multiple tool_calls" in rendered
    assert "without inspecting earlier results" in rendered
    assert "do not precommit an action" in rendered
    assert "localization is optional" in user
    assert "If the scene is not available" in user
    for retired in (
        "Issue one physical tool call at a time",
        "READ EACH AVAILABLE LOCAL MEMORY LAYER FIRST",
        "READ MEMORY FIRST",
        "use every layer that is available",
        "localize objects before motion",
        "FIRST-STEP ALGORITHM",
    ):
        assert retired not in rendered + user
    assert "terminated" in rendered
