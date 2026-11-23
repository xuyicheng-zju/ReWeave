# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Unavailable environment verdicts override a stale agent termination in both CLIs."""

import json
import sys
from types import SimpleNamespace

import pytest

from reweave.dashboard.state import ClaimedTask
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    AGENT_EXECUTION_FAILURE,
    API_ERROR,
    PlannerResult,
)
from reweave.robots.prompt_bundle import PromptBundle
from reweave.robots.robot_spec import RobotSpec, RunConfig
from reweave.runtime.capability import EnvironmentInvalid, EnvironmentServiceFailure
from tests.unit_tests.harness.cli.test_main_contracts import _cli_module


@pytest.mark.parametrize("entrypoint", ["cli", "dashboard"])
@pytest.mark.parametrize(
    "prior_termination", [None, AGENT_EXECUTION_FAILURE, ACTION_OUTCOME_UNKNOWN]
)
@pytest.mark.parametrize("error_type", [EnvironmentInvalid, EnvironmentServiceFailure])
def test_cli_invalid_scene_preserves_null_verdict_and_infrastructure_origin(
    tmp_path, monkeypatch, entrypoint, prior_termination, error_type
):
    finalized = []
    lifecycle = []
    invalid = error_type("SIMULATED unavailable environment verdict")

    def solved():
        raise invalid

    toolkit = SimpleNamespace(
        export_harness_context=lambda directory: None,
        finalize_episode_evaluation=lambda: None,
        solved=solved,
        close=lambda: lifecycle.append("close"),
    )
    config = RunConfig(
        recipe_tag="invalid", output_dir=tmp_path, prompt_vars={}, task_desc={}
    )
    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: None,
        parse_config=lambda args: config,
        init_runtime=lambda *args: ([], {}),
        finalize_run=finalized.append,
    )
    planner = SimpleNamespace(
        solve=lambda **kwargs: PlannerResult(
            termination_class=prior_termination,
            error="candidate stopped" if prior_termination is not None else None,
        )
    )
    expected_error = (
        "candidate stopped"
        if prior_termination is not None
        else f"{error_type.__name__}: {invalid}"
    )
    if entrypoint == "cli":
        cli = _cli_module()
        monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
        monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
        monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: toolkit)
        monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: planner)
        monkeypatch.setattr(
            sys, "argv", ["harness", "--robot", "testrobot", "--planner", "native"]
        )
        assert cli.main() == 1
    else:
        from reweave.cli import dashboard

        monkeypatch.setattr(dashboard, "get_toolkit", lambda *args, **kwargs: toolkit)
        monkeypatch.setattr(dashboard, "build_planner", lambda *args, **kwargs: planner)
        args = SimpleNamespace(
            verbose=False,
            robot_name="testrobot",
            explore=False,
            planner="native",
            native_provider="codex",
            codex_auth_mode="auth",
            native_service_tier="default",
            adaptation_mode="static",
            model="offline",
            planner_timeout_s=10,
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
                robot_spec=robot_spec,
                state=state,
                claimed=ClaimedTask(number=1, request={}, output_dir=tmp_path),
                shared_primitives_kwargs={},
                unique_components=set(),
                session_root=tmp_path / "session",
            )
            == expected_error
        )
    transcript = json.loads((tmp_path / "transcript_invalid.json").read_text())
    assert transcript["termination_class"] == (
        ACTION_OUTCOME_UNKNOWN
        if prior_termination == ACTION_OUTCOME_UNKNOWN
        else API_ERROR
    )
    assert transcript["api_error_retryable"] is (
        None if prior_termination == ACTION_OUTCOME_UNKNOWN else False
    )
    assert transcript["error"] == expected_error
    assert transcript["official_success"] is False
    assert len(finalized) == 1
    assert finalized[0].environment_success is None
    assert finalized[0].termination_class == transcript["termination_class"]
    assert transcript["finalization_error"] == {
        "type": error_type.__name__,
        "message": str(invalid),
    }
    if prior_termination is not None:
        assert any(
            "subsequent episode error" in item for item in transcript["cleanup_errors"]
        )
    assert lifecycle == ["close"]
