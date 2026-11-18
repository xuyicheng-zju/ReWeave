# Attribution retained in NOTICE. Modified for the ReWeave namespace.
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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reweave.dashboard.session import DashboardSessionController
from reweave.dashboard.state import ClaimedTask
from reweave.evaluation import RunFinalizationContext
from reweave.planner.base import PlannerResult
from reweave.robots import PromptBundle, RunConfig
from reweave.runtime.capability import PhysicalOutcomeUnknown


class ScriptedState:
    def __init__(self, tasks: list[ClaimedTask] | None = None) -> None:
        self.tasks = list(tasks or [])
        self.ready_calls = 0
        self.failed_session: Exception | None = None
        self.completed: list[dict[str, Any]] = []
        self.task_replacement_requested = False

    def shared_services_ready(self) -> None:
        self.ready_calls += 1

    def fail_session(self, error: Exception) -> None:
        self.failed_session = error

    def wait_for_task(self) -> ClaimedTask | None:
        return self.tasks.pop(0) if self.tasks else None

    def complete_task(self, *, state: str, error: Any) -> None:
        self.completed.append({"state": state, "error": error})


class FakeDaemon:
    def __init__(
        self,
        name: str,
        stopped: list[str],
        *,
        stop_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.stopped = stopped
        self.stop_error = stop_error

    def stop(self) -> None:
        self.stopped.append(self.name)
        if self.stop_error is not None:
            raise self.stop_error


def _claimed_task(tmp_path: Path) -> ClaimedTask:
    return ClaimedTask(
        number=1,
        request={"mode": "pick", "seed": 1},
        output_dir=tmp_path / "task",
    )


def test_dashboard_session_shared_start_failure_marks_session_fatal(
    tmp_path: Path,
) -> None:
    state = ScriptedState([_claimed_task(tmp_path)])
    run_calls = 0

    def start_shared():
        raise RuntimeError("shared runtime failed")

    def run_task(claimed: ClaimedTask, shared: dict[str, Any]):
        nonlocal run_calls
        del claimed, shared
        run_calls += 1

    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=start_shared,
        run_task=run_task,
    )

    controller.run()

    assert isinstance(state.failed_session, RuntimeError)
    assert str(state.failed_session) == "shared runtime failed"
    assert state.ready_calls == 0
    assert run_calls == 0
    assert state.completed == []


@pytest.mark.parametrize(
    ("outcome", "expected_state", "expected_error"),
    [
        ("success", "succeeded", None),
        ("returned-error", "failed", "task failed"),
        ("raised-error", "failed", RuntimeError),
        ("replacement", "cancelled", "ignored after replacement"),
    ],
)
def test_dashboard_session_maps_task_outcomes(
    tmp_path: Path,
    outcome: str,
    expected_state: str,
    expected_error: Any,
) -> None:
    claimed = _claimed_task(tmp_path)
    state = ScriptedState([claimed])
    shared = {"model": object()}
    received: list[tuple[ClaimedTask, dict[str, Any]]] = []

    def run_task(task: ClaimedTask, shared_kwargs: dict[str, Any]):
        received.append((task, shared_kwargs))
        if outcome == "returned-error":
            return "task failed"
        if outcome == "raised-error":
            raise RuntimeError("task exploded")
        if outcome == "replacement":
            state.task_replacement_requested = True
            return "ignored after replacement"
        return None

    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=lambda: ([], shared),
        run_task=run_task,
    )

    controller.run()

    assert state.ready_calls == 1
    assert received == [(claimed, shared)]
    assert state.completed[0]["state"] == expected_state
    if expected_error is RuntimeError:
        assert isinstance(state.completed[0]["error"], RuntimeError)
        assert str(state.completed[0]["error"]) == "task exploded"
    else:
        assert state.completed[0]["error"] == expected_error


def test_dashboard_session_stops_shared_daemons_in_reverse_after_cleanup_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stopped: list[str] = []
    daemons = [
        FakeDaemon("first", stopped),
        FakeDaemon("second", stopped, stop_error=RuntimeError("stop failed")),
        FakeDaemon("third", stopped),
    ]
    state = ScriptedState()
    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=lambda: (daemons, {}),
        run_task=lambda claimed, shared: None,
    )

    controller.run()

    assert stopped == ["third", "second", "first"]
    assert "shared runtime cleanup failed: stop failed" in caplog.text


@pytest.mark.parametrize("reasoning_effort", ["low", "high"])
def test_dashboard_exploration_archives_without_memory_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str,
) -> None:
    from reweave.cli import dashboard as dashboard_cli

    finalized: list[RunFinalizationContext] = []
    planner_kwargs: dict[str, Any] = {}

    class FakeToolkit:
        evaluation_finalized = False

        def finalize_episode_evaluation(self) -> None:
            self.evaluation_finalized = True

        def solved(self) -> bool:
            return True

        def write_recipe(self, recipe_tag: str) -> str:
            return str(tmp_path / f"{recipe_tag}_recipe.jsonl")

        def close(self) -> None:
            assert self.evaluation_finalized is True

    class FakeState:
        task_replacement_requested = False

        def __init__(self) -> None:
            self.warnings: list[str] = []
            self.toolkit_lifecycle: list[str] = []

        def emit(self, event: Any) -> None:
            del event

        def begin_planner_session(self, **kwargs: Any) -> None:
            del kwargs

        def bind_toolkit(self, toolkit: Any) -> None:
            del toolkit
            self.toolkit_lifecycle.append("bound")

        def unbind_toolkit(self, toolkit: Any) -> None:
            del toolkit
            self.toolkit_lifecycle.append("unbound")

        def report_task_warning(self, warning: str) -> None:
            self.warnings.append(warning)

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            del kwargs
            return PlannerResult(
                finish_result={"status": "success"},
                messages=[],
                stats={},
            )

    output_dir = tmp_path / "task"
    run_config = RunConfig(
        recipe_tag="libero_s0",
        output_dir=output_dir,
        prompt_vars={},
        task_desc={"robot": "libero"},
    )
    robot_spec = SimpleNamespace(
        parse_config=lambda args: run_config,
        init_runtime=lambda *args: ([], {}),
        finalize_run=lambda context: finalized.append(context),
        prompts=PromptBundle(
            system=lambda variables: "system",
            user=lambda variables: "user",
        ),
    )
    args = SimpleNamespace(
        verbose=False,
        robot_name="libero",
        explore=True,
        explore_sessions=1,
        explore_attempts_per_session=2,
        planner="native",
        model="gpt-5.4",
        native_provider="codex",
        codex_auth_mode="auth",
        native_service_tier="default",
        adaptation_mode="static",
        planner_timeout_s=10,
        reasoning_effort=reasoning_effort,
        max_turns=2,
    )
    claimed = ClaimedTask(number=1, request={}, output_dir=output_dir)
    state = FakeState()
    monkeypatch.setattr(
        dashboard_cli, "get_toolkit", lambda *args, **kwargs: FakeToolkit()
    )

    def build_planner(*args: Any, **kwargs: Any) -> FakePlanner:
        planner_kwargs.update(kwargs)
        return FakePlanner()

    monkeypatch.setattr(dashboard_cli, "build_planner", build_planner)

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=state,
        claimed=claimed,
        shared_primitives_kwargs={},
        unique_components=set(),
        session_root=tmp_path / "session",
    )

    assert error is None
    assert state.toolkit_lifecycle == ["bound", "unbound"]
    assert state.warnings == []
    assert planner_kwargs["reasoning_effort"] == reasoning_effort
    assert planner_kwargs["native_provider"] == "codex"
    assert planner_kwargs["codex_auth_mode"] == "auth"
    assert planner_kwargs["native_service_tier"] == "default"
    assert planner_kwargs["adaptation_mode"] == "static"
    assert len(finalized) == 1
    assert finalized[0].environment_success is True
    assert finalized[0].termination_class == "success"
    assert finalized[0].reasoning_effort == reasoning_effort


@pytest.mark.parametrize(
    (
        "planner_failure",
        "evaluation_failure",
        "close_failure",
        "expected_error",
        "expected_termination",
        "cleanup_count",
    ),
    [
        (
            "raised-unknown",
            True,
            True,
            "PhysicalOutcomeUnknown: submitted action has no acknowledgement",
            "action_outcome_unknown",
            3,
        ),
        (
            "returned-unknown",
            True,
            True,
            "submitted action has no acknowledgement",
            "action_outcome_unknown",
            3,
        ),
        (
            "returned-agent-error",
            True,
            True,
            "program rejected the action",
            "api_error",
            3,
        ),
        (None, True, True, "RuntimeError: evaluator unavailable", "api_error", 3),
        (None, False, False, None, "physical_failure", 1),
    ],
)
def test_dashboard_preserves_primary_failure_when_finalization_and_unbind_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner_failure: str | None,
    evaluation_failure: bool,
    close_failure: bool,
    expected_error: str | None,
    expected_termination: str,
    cleanup_count: int,
) -> None:
    from reweave.cli import dashboard as dashboard_cli

    finalized: list[RunFinalizationContext] = []
    lifecycle: list[str] = []

    class FakeToolkit:
        def finalize_episode_evaluation(self) -> None:
            lifecycle.append("evaluate")
            if evaluation_failure:
                raise RuntimeError("evaluator unavailable")

        def solved(self) -> bool:
            return False

        def close(self) -> None:
            lifecycle.append("close")
            if close_failure:
                raise RuntimeError("close failed")

    class FakeState:
        task_replacement_requested = False

        def emit(self, event: Any) -> None:
            del event

        def bind_toolkit(self, toolkit: Any) -> None:
            del toolkit
            lifecycle.append("bind")

        def unbind_toolkit(self, toolkit: Any) -> None:
            del toolkit
            lifecycle.append("unbind")
            raise RuntimeError("unbind failed")

    def solve(**kwargs: Any) -> PlannerResult:
        if planner_failure == "raised-unknown":
            raise PhysicalOutcomeUnknown("submitted action has no acknowledgement")
        if planner_failure is not None:
            return PlannerResult(
                error=expected_error,
                termination_class=(
                    "agent_execution_failure"
                    if planner_failure == "returned-agent-error"
                    else expected_termination
                ),
            )
        return PlannerResult()

    def finalize_run(context: RunFinalizationContext) -> None:
        lifecycle.append("result")
        finalized.append(context)

    output_dir = tmp_path / "task"
    run_config = RunConfig(
        recipe_tag="audit", output_dir=output_dir, prompt_vars={}, task_desc={}
    )
    robot_spec = SimpleNamespace(
        parse_config=lambda args: run_config,
        init_runtime=lambda *args: (
            [FakeDaemon("first", lifecycle), FakeDaemon("second", lifecycle)],
            {},
        ),
        finalize_run=finalize_run,
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
    )
    args = SimpleNamespace(
        verbose=False,
        robot_name="testrobot",
        explore=False,
        planner="native",
        model="gpt-5.4",
        native_provider="codex",
        codex_auth_mode="auth",
        native_service_tier="default",
        adaptation_mode="static",
        planner_timeout_s=10,
        reasoning_effort="low",
        max_turns=2,
    )
    monkeypatch.setattr(
        dashboard_cli, "get_toolkit", lambda *args, **kwargs: FakeToolkit()
    )
    monkeypatch.setattr(
        dashboard_cli,
        "build_planner",
        lambda *args, **kwargs: SimpleNamespace(solve=solve),
    )

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=FakeState(),
        claimed=ClaimedTask(number=1, request={}, output_dir=output_dir),
        shared_primitives_kwargs={},
        unique_components=set(),
        session_root=tmp_path / "session",
    )

    transcript = json.loads((output_dir / "transcript_audit.json").read_text())
    assert error == (
        expected_error or "toolkit unbind failed: RuntimeError: unbind failed"
    )
    assert transcript["error"] == expected_error
    assert transcript["termination_class"] == expected_termination
    assert transcript["finalization_error"] == (
        {"type": "RuntimeError", "message": "evaluator unavailable"}
        if evaluation_failure
        else None
    )
    if planner_failure == "returned-agent-error":
        assert transcript["planner_termination_class"] == "agent_execution_failure"
        assert transcript["api_error_retryable"] is False
    if expected_termination == "action_outcome_unknown":
        assert transcript["planner_termination_class"] == "action_outcome_unknown"
        assert transcript["api_error_retryable"] is None
    assert transcript["official_success"] is False
    assert len(transcript["cleanup_errors"]) == cleanup_count
    assert (
        "toolkit unbind failed: RuntimeError: unbind failed"
        in transcript["cleanup_errors"]
    )
    if planner_failure is not None and evaluation_failure:
        assert transcript["cleanup_errors"][0] == (
            "episode evaluation finalization failed: RuntimeError: evaluator unavailable"
        )
    assert lifecycle == [
        "bind",
        "evaluate",
        "unbind",
        "close",
        "second",
        "first",
        "result",
    ]
    assert len(finalized) == 1
    assert finalized[0].agent_error == expected_error
    assert finalized[0].termination_class == expected_termination
    assert finalized[0].environment_success is (None if evaluation_failure else False)


@pytest.mark.parametrize(
    ("termination_class", "planner_error", "expected_sessions"),
    [
        ("action_outcome_unknown", "physical RPC timed out after submission", 1),
        ("api_error", "model service timed out", 1),
        ("planner_timeout", "planner deadline reached", 2),
    ],
)
def test_dashboard_continues_only_after_structured_planner_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_class: str,
    planner_error: str,
    expected_sessions: int,
) -> None:
    from reweave.cli import dashboard as dashboard_cli

    lifecycle: list[str] = []
    solve_calls = 0

    class FakeToolkit:
        def __init__(self) -> None:
            lifecycle.append("reset")

        def finalize_episode_evaluation(self) -> None:
            lifecycle.append("evaluate")

        def solved(self) -> bool:
            return False

        def close(self) -> None:
            lifecycle.append("close")

    class FakeState:
        task_replacement_requested = False

        def emit(self, event: Any) -> None:
            del event

        def begin_planner_session(self, **kwargs: Any) -> None:
            lifecycle.append("session")

        def bind_toolkit(self, toolkit: Any) -> None:
            lifecycle.append("bind")

        def unbind_toolkit(self, toolkit: Any) -> None:
            lifecycle.append("unbind")

    def solve(**kwargs: Any) -> PlannerResult:
        nonlocal solve_calls
        solve_calls += 1
        if solve_calls == 1:
            return PlannerResult(
                error=planner_error,
                termination_class=termination_class,
                api_error_retryable=(
                    False if termination_class == "api_error" else None
                ),
            )
        return PlannerResult()

    def init_runtime(*args: Any) -> tuple[list[Any], dict[str, Any]]:
        lifecycle.append("runtime")
        return [FakeDaemon("stop", lifecycle)], {}

    output_dir = tmp_path / "task"
    run_config = RunConfig(
        recipe_tag="sessions", output_dir=output_dir, prompt_vars={}, task_desc={}
    )
    robot_spec = SimpleNamespace(
        parse_config=lambda args: run_config,
        init_runtime=init_runtime,
        finalize_run=None,
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
    )
    args = SimpleNamespace(
        verbose=False,
        robot_name="libero",
        explore=True,
        explore_sessions=2,
        planner="native",
        model="gpt-5.4",
        native_provider="codex",
        codex_auth_mode="auth",
        native_service_tier="default",
        adaptation_mode="static",
        planner_timeout_s=10,
        reasoning_effort="low",
        max_turns=2,
    )
    monkeypatch.setattr(
        dashboard_cli, "get_toolkit", lambda *args, **kwargs: FakeToolkit()
    )
    monkeypatch.setattr(
        dashboard_cli,
        "build_planner",
        lambda *args, **kwargs: SimpleNamespace(solve=solve),
    )

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=FakeState(),
        claimed=ClaimedTask(number=1, request={}, output_dir=output_dir),
        shared_primitives_kwargs={},
        unique_components=set(),
        session_root=tmp_path / "session",
    )

    assert error == (planner_error if expected_sessions == 1 else None)
    assert solve_calls == expected_sessions
    assert lifecycle == [
        "runtime",
        *["session", "reset", "bind", "evaluate", "unbind", "close"]
        * expected_sessions,
        "stop",
    ]
    transcript = json.loads((output_dir / "transcript_sessions.json").read_text())
    assert transcript["termination_class"] == (
        termination_class if expected_sessions == 1 else "physical_failure"
    )
    assert transcript["error"] == (planner_error if expected_sessions == 1 else None)
