# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Both entrypoints retain Planner errors independently of evaluator outcomes."""

import json
import sys
from types import SimpleNamespace

import pytest

from reweave.benchmarks.robodojo.robot_spec import finalize_run
from reweave.dashboard.state import ClaimedTask
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    AGENT_EXECUTION_FAILURE,
    API_ERROR,
    PLANNER_TIMEOUT,
    SUCCESS,
    PlannerResult,
)
from reweave.robots.prompt_bundle import PromptBundle
from reweave.robots.robot_spec import RobotSpec, RunConfig
from tests.unit_tests.harness.cli.test_main_contracts import _cli_module


@pytest.mark.parametrize("entrypoint", ["cli", "dashboard"])
@pytest.mark.parametrize("environment_success", [False, True])
@pytest.mark.parametrize(
    "planner_termination",
    [ACTION_OUTCOME_UNKNOWN, AGENT_EXECUTION_FAILURE, API_ERROR, PLANNER_TIMEOUT],
)
def test_entrypoints_finalize_raw_verdict_and_planner_failure_separately(
    tmp_path, monkeypatch, entrypoint, environment_success, planner_termination
):
    robot = "robodojo"
    task = {"task": "play_tic_tac_toe", "layout_set": 1, "layout": 0}
    lifecycle = []
    toolkit = SimpleNamespace(
        export_harness_context=lambda directory: None,
        finalize_episode_evaluation=lambda: None,
        solved=lambda: environment_success,
        close=lambda: lifecycle.append("close"),
    )
    spec = RobotSpec(
        name=robot,
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: None,
        parse_config=lambda args: RunConfig(
            recipe_tag="settled", output_dir=tmp_path, prompt_vars={}, task_desc=task
        ),
        init_runtime=lambda *args: ([], {}),
        finalize_run=finalize_run,
    )
    error = f"SIMULATED {planner_termination}"
    retryable = False if planner_termination == API_ERROR else None
    planner = SimpleNamespace(
        solve=lambda **kwargs: PlannerResult(
            termination_class=planner_termination,
            error=error,
            api_error_retryable=retryable,
        )
    )
    if entrypoint == "cli":
        cli = _cli_module()
        monkeypatch.setattr(cli, "enumerate_robots", lambda: (robot,))
        monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
        monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: toolkit)
        monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: planner)
        monkeypatch.setattr(
            sys, "argv", ["harness", "--robot", robot, "--planner", "native"]
        )
        assert cli.main() == 1
    else:
        from reweave.cli import dashboard

        monkeypatch.setattr(dashboard, "get_toolkit", lambda *args, **kwargs: toolkit)
        monkeypatch.setattr(dashboard, "build_planner", lambda *args, **kwargs: planner)
        args = SimpleNamespace(
            verbose=False,
            robot_name=robot,
            explore=False,
            planner="native",
            native_provider="codex",
            codex_auth_mode="auth",
            native_service_tier="default",
            adaptation_mode="static",
            model="SIMULATED",
            planner_timeout_s=0,
            reasoning_effort=None,
            max_turns=2,
        )
        state = SimpleNamespace(
            task_replacement_requested=False,
            emit=lambda event: None,
            bind_toolkit=lambda toolkit: None,
            unbind_toolkit=lambda toolkit: None,
        )
        assert (
            dashboard._run_dashboard_task(
                args=args,
                robot_spec=spec,
                state=state,
                claimed=ClaimedTask(number=1, request={}, output_dir=tmp_path),
                shared_primitives_kwargs={},
                unique_components=set(),
                session_root=tmp_path / "session",
            )
            == error
        )
    transcript = json.loads((tmp_path / "transcript_settled.json").read_text())
    unknown = planner_termination == ACTION_OUTCOME_UNKNOWN
    success = environment_success and not unknown
    termination = SUCCESS if success else planner_termination
    assert transcript["official_success"] is success
    assert transcript["environment_success"] is environment_success
    assert transcript["termination_class"] == termination
    assert transcript["planner_termination_class"] == planner_termination
    assert transcript["planner_api_error_retryable"] is retryable
    assert transcript["api_error_retryable"] is (
        retryable if termination == API_ERROR else None
    )
    assert transcript["error"] == error
    assert lifecycle == ["close"]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["environment_success"] is environment_success
    assert result["success"] is (None if unknown else success)
    assert result["valid"] is (not unknown)
