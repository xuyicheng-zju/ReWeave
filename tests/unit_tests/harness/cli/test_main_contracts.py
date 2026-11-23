# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ROBOT_HELP_CASES = [
    ("libero", ("--suite", "--task", "--libero-type")),
]


class ConfigCaptured(Exception):
    """Stop the CLI immediately after argument validation in routing tests."""


def _cli_module():
    from reweave.cli import main as cli

    return cli


def _run_cli_help(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "reweave.cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _capture_validated_args(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> tuple[str, Any]:
    cli = _cli_module()
    captured: dict[str, Any] = {}

    def add_cli_args(parser, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=1)

    def parse_config(args) -> None:
        captured["args"] = args
        raise ConfigCaptured

    def get_robot_spec(name: str):
        captured["robot_name"] = name
        return SimpleNamespace(
            add_cli_args=add_cli_args,
            parse_config=parse_config,
        )

    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero", "robodojo"))
    monkeypatch.setattr(cli, "get_robot_spec", get_robot_spec)
    monkeypatch.setattr(sys, "argv", ["harness", *argv])

    with pytest.raises(ConfigCaptured):
        cli.main()
    return captured["robot_name"], captured["args"]


def test_public_cli_help_runs_from_source_checkout_without_a_robot_runtime() -> None:
    result = _run_cli_help("--help")

    assert result.returncode == 0, result.stderr
    assert "Harness: Agentic Infrastructure for the Physical World" in result.stdout
    for option in (
        "--robot",
        "--planner",
        "--output-dir",
        "--dashboard",
        "--interactive",
    ):
        assert option in result.stdout
    for option in REMOVED_AUTHOR_LIMITS:
        assert option not in result.stdout


@pytest.mark.parametrize(
    ("robot_name", "robot_options"),
    ROBOT_HELP_CASES,
)
def test_public_robot_specific_help_exits_cleanly_and_includes_extension_options(
    robot_name: str,
    robot_options: tuple[str, ...],
) -> None:
    result = _run_cli_help("--robot", robot_name, "--help")

    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in robot_options)


def test_shared_cli_defaults_reach_robot_config_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_name, args = _capture_validated_args(
        monkeypatch,
        ["--robot", "libero"],
    )

    assert robot_name == "libero"
    assert args.robot_name == "libero"
    assert args.planner == "native"
    assert args.model == "gpt-5.4"
    assert args.reasoning_effort == "medium"
    assert args.native_service_tier == "default"
    assert args.max_turns == 100
    for option in REMOVED_AUTHOR_LIMITS:
        assert not hasattr(args, option.removeprefix("--").replace("-", "_"))
    assert args.dashboard is False
    assert args.dashboard_host == "127.0.0.1"
    assert args.dashboard_port == 0
    assert args.dashboard_language == "en"
    assert args.memory_profile == "local"


REMOVED_AUTHOR_LIMITS = (
    "--claude-code-author-max-budget-usd",
    "--offline-author-max-budget-usd",
    "--offline-author-max-turns",
    "--offline-author-timeout-s",
    "--runtime-author-timeout-s",
)


@pytest.mark.parametrize("option", REMOVED_AUTHOR_LIMITS)
def test_removed_author_limits_are_rejected_before_launch(option):
    result = _run_cli_help(
        "--robot", "libero", "--suite", "libero_goal", "--task", "0", option, "5"
    )

    assert result.returncode == 2
    assert f"unrecognized arguments: {option} 5" in result.stderr


@pytest.mark.parametrize(
    "option,value",
    [
        ("--base-url", "https://example.invalid"),
        ("--max-tokens", "123"),
        ("--claude-code-max-budget-usd", "23"),
        ("--no-images", None),
        ("--no-thinking", None),
    ],
)
def test_removed_api_and_direct_cli_options_are_rejected(option, value):
    extra = [option] + ([value] if value else [])
    result = _run_cli_help(
        "--robot", "libero", "--suite", "libero_goal", "--task", "0", *extra
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    "extra",
    [
        ("--env", "libero"),
        ("--auto-merge-memory",),
        ("--no-auto-merge-memory",),
    ],
)
def test_removed_memory_and_robot_alias_options_are_rejected(extra):
    result = _run_cli_help(
        "--robot", "libero", "--suite", "libero_goal", "--task", "0", *extra
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--robot", "libero", "--dashboard", "--interactive"],
            "cannot be used together",
        ),
        (["--robot", "robodojo", "--explore"], "supported only for LIBERO"),
        (
            ["--robot", "libero", "--explore", "--memory-profile", "hf"],
            "invalid choice",
        ),
        (
            ["--robot", "libero", "--explore", "--explore-sessions", "0"],
            "greater than 0",
        ),
        (
            [
                "--robot",
                "libero",
                "--memory-profile",
                "hf",
                "--memory-dir",
                "/offline/memory",
            ],
            "invalid choice",
        ),
    ],
)
def test_shared_cli_validation_stops_before_robot_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    cli = _cli_module()
    parse_called = False

    def add_cli_args(parser, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=1)

    def parse_config(args) -> None:
        nonlocal parse_called
        del args
        parse_called = True

    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero", "robodojo"))
    monkeypatch.setattr(
        cli,
        "get_robot_spec",
        lambda name: SimpleNamespace(
            name=name,
            add_cli_args=add_cli_args,
            parse_config=parse_config,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["harness", *argv])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
    assert parse_called is False


def test_handoff_message_lists_prior_attempts_deterministically(tmp_path: Path) -> None:
    cli = _cli_module()
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "attempt_10_failed.json").write_text("{}")
    (attempts / "attempt_02_failed.json").write_text("{}")
    (attempts / "unrelated.json").write_text("{}")

    message = cli._handoff_message(tmp_path, session_number=2, session_max=4)

    assert "agent 2 of up to 4" in message
    assert "2 attempt(s)" in message
    assert message.index("attempt_02_failed.json") < message.index(
        "attempt_10_failed.json"
    )
    assert "unrelated.json" not in message
    assert "run-local attempt archives" in message


def test_full_cli_exploration_archives_run_without_memory_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli_module()
    from reweave.planner.base import PlannerResult
    from reweave.robots.prompt_bundle import PromptBundle
    from reweave.robots.robot_spec import RobotSpec, RunConfig
    from reweave.tools.toolkit import ToolResult

    calls: dict[str, Any] = {}

    class FakeDaemon:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeToolkit:
        def finalize_episode_evaluation(self) -> None:
            pass

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.closed = False

        def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
            self.calls.append((name, args))
            return ToolResult(
                name,
                {
                    "_finish": True,
                    "status": args["status"],
                    "summary": args["summary"],
                },
            )

        def export_harness_context(self, output_dir) -> None:
            pass

        def close(self) -> None:
            self.closed = True

        def solved(self) -> bool:
            return True

        def write_recipe(self, recipe_tag: str) -> str:
            calls["write_recipe"] = recipe_tag
            return str(tmp_path / f"{recipe_tag}_recipe.jsonl")

    class ScriptedPlanner:
        def solve(
            self,
            *,
            system_prompt: str,
            user_message: str,
            toolkit: FakeToolkit,
            max_turns: int,
            input_queue: Any = None,
            dashboard_interaction: Any = None,
        ) -> PlannerResult:
            calls["solve"] = {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "max_turns": max_turns,
                "input_queue": input_queue,
                "dashboard_interaction": dashboard_interaction,
            }
            finish = toolkit.execute_tool(
                "finish",
                {"status": "success", "summary": "simulated task complete"},
            )
            return PlannerResult(
                finish_result=finish.result,
                messages=[{"role": "assistant", "content": "finished offline"}],
                stats={
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "tool_calls": 1,
                },
            )

    daemon = FakeDaemon()
    toolkit = FakeToolkit()
    planner = ScriptedPlanner()

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=1)
        parser.add_argument("--explore-attempts-per-session", type=int, default=2)

    def parse_config(args: Any) -> RunConfig:
        return RunConfig(
            recipe_tag="libero_s0",
            output_dir=Path(args.output_dir),
            prompt_vars={"memory_dir": args.memory_dir},
            task_desc={"robot": "libero"},
        )

    def init_runtime(*args: Any) -> tuple[list[FakeDaemon], dict[str, str]]:
        calls["init_runtime"] = args
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return [daemon], {"runtime": "simulated"}

    robot_spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "simulated system prompt",
            user=lambda variables: "simulated user task",
        ),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
    )

    def build_planner(*args: Any, **kwargs: Any) -> ScriptedPlanner:
        calls["build_planner"] = (args, kwargs)
        return planner

    def get_toolkit(*args: Any, **kwargs: Any) -> FakeToolkit:
        calls["get_toolkit"] = (args, kwargs)
        return toolkit

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in (
        "REWEAVE_CLAUDE_CODE_AUTHOR_MODEL",
        "REWEAVE_CODEX_AUTHOR_MODEL",
        "CODEX_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "build_planner", build_planner)
    monkeypatch.setattr(cli, "get_toolkit", get_toolkit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--robot",
            "libero",
            "--explore",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
            "--max-turns",
            "4",
        ],
    )

    assert cli.main() == 0

    assert calls["solve"] == {
        "system_prompt": "simulated system prompt\n",
        "user_message": "simulated user task\n",
        "max_turns": 4,
        "input_queue": None,
        "dashboard_interaction": None,
    }
    assert toolkit.calls == [
        ("finish", {"status": "success", "summary": "simulated task complete"})
    ]
    assert toolkit.closed is True
    assert daemon.stopped is True
    assert calls["get_toolkit"][1]["primitives_kwargs"] == {"runtime": "simulated"}
    assert calls["get_toolkit"][1]["mode"] == "exploration"
    assert calls["get_toolkit"][1]["attempts_per_session"] == 2
    assert calls["write_recipe"] == "libero_s0"

    transcript = json.loads((tmp_path / "transcript_libero_s0.json").read_text())
    assert transcript["robot"] == "libero"
    assert "runtime_author" not in transcript["agent_run_contract"]
    assert "runtime_authors" not in calls["build_planner"][1]
    assert transcript["finish"] == {
        "_finish": True,
        "status": "success",
        "summary": "simulated task complete",
    }
    assert transcript["stats"]["tool_calls"] == 1
    assert transcript["messages"] == [
        {"role": "assistant", "content": "finished offline"}
    ]
    assert transcript["termination_class"] == "success"
    assert transcript["official_success"] is True


@pytest.mark.parametrize(
    "option",
    [
        "--runtime-authors",
        "--claude-code-author-model",
        "--codex-author-model",
        "--offline-author",
    ],
)
def test_cli_rejects_removed_split_author_flags(option):
    result = _run_cli_help(
        "--robot", "libero", "--suite", "libero_goal", "--task", "0", option, "codex"
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    ("cleanup_source", "official_success", "planner_error"),
    [
        (None, False, None),
        ("daemon", True, None),
        ("toolkit", True, None),
        ("daemon", False, "planner unavailable"),
        (None, True, "planner unavailable"),
    ],
)
def test_full_cli_calls_robot_result_finalizer_without_robot_special_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_source: str | None,
    official_success: bool,
    planner_error: str | None,
) -> None:
    cli = _cli_module()
    from reweave.evaluation.result import RunFinalizationContext
    from reweave.utils.persistence import atomic_text
    from reweave.planner.base import PlannerResult
    from reweave.robots.prompt_bundle import PromptBundle
    from reweave.robots.robot_spec import RobotSpec, RunConfig

    captured: list[RunFinalizationContext] = []
    stopped: list[str] = []

    class FakeDaemon:
        stopped = False

        def __init__(self, name: str) -> None:
            self.name = name

        def stop(self) -> None:
            self.stopped = True
            stopped.append(self.name)
            if cleanup_source == "daemon" and self.name == "second":
                raise RuntimeError("daemon stop failed")

    class FakeToolkit:
        memory = SimpleNamespace()
        closed = False

        def finalize_episode_evaluation(self) -> None:
            pass

        def solved(self) -> bool:
            return official_success

        def export_harness_context(self, output_dir) -> None:
            pass

        def close(self) -> None:
            self.closed = True
            if cleanup_source == "toolkit":
                raise RuntimeError("toolkit close failed")

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            del kwargs
            return PlannerResult(
                finish_result={"status": "success", "summary": "planner claim"},
                messages=[],
                stats={"tool_calls": 0},
                error=planner_error,
                termination_class="api_error" if planner_error else None,
                api_error_retryable=False if planner_error else None,
            )

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--task-name", required=True)
        parser.add_argument("--seed", type=int, default=0)

    def parse_config(args: Any) -> RunConfig:
        return RunConfig(
            recipe_tag=f"{args.task_name}_s{args.seed}",
            output_dir=Path(args.output_dir),
            prompt_vars={"memory_dir": args.memory_dir},
            task_desc={"task_name": args.task_name, "seed": args.seed},
        )

    def finalize_run(context: RunFinalizationContext) -> Path:
        captured.append(context)
        assert robot_toolkit.closed is True
        destination = context.output_dir / "result.json"
        atomic_text(
            destination,
            json.dumps(
                {
                    "robot": context.robot_name,
                    "task": dict(context.task_desc),
                    "success": context.environment_success,
                    "planner_claim": context.finish_result,
                },
                indent=2,
            )
            + "\n",
        )
        return destination

    daemons = [FakeDaemon(name) for name in ("first", "second", "third")]
    robot_toolkit = FakeToolkit()
    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(
            system=lambda variables: "system",
            user=lambda variables: "task",
        ),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=lambda *args: (daemons, {"runtime": "simulated"}),
        finalize_run=finalize_run,
    )

    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: FakePlanner())
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: robot_toolkit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--robot",
            "testrobot",
            "--task-name",
            "OpenDrawer",
            "--seed",
            "1",
            "--planner",
            "native",
            "--model",
            "gpt-5.5",
            "--reasoning-effort",
            "xhigh",
            "--planner-timeout-s",
            "1800",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert cli.main() == (1 if cleanup_source or planner_error else 0)

    assert len(captured) == 1
    context = captured[0]
    assert context.robot_name == "testrobot"
    assert context.environment_success is official_success
    assert context.agent_error == planner_error
    assert context.planner == "native"
    assert context.model == "gpt-5.5"
    assert context.reasoning_effort == "xhigh"
    assert context.max_turns == 100
    assert context.planner_timeout_s == 1800
    assert context.stats == {"tool_calls": 0}
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "robot": "testrobot",
        "task": {"task_name": "OpenDrawer", "seed": 1},
        "success": official_success,
        "planner_claim": {"status": "success", "summary": "planner claim"},
    }
    assert robot_toolkit.closed is True
    assert all(daemon.stopped for daemon in daemons)
    assert stopped == ["third", "second", "first"]
    transcript = json.loads((tmp_path / "transcript_OpenDrawer_s1.json").read_text())
    assert transcript["error"] == planner_error
    assert transcript["official_success"] is official_success
    expected_termination = (
        "success"
        if official_success
        else "api_error"
        if planner_error
        else "physical_failure"
    )
    assert transcript["termination_class"] == expected_termination
    assert len(transcript["cleanup_errors"]) == bool(cleanup_source)


@pytest.mark.parametrize(
    ("planner_failure", "expected_error", "expected_termination", "cleanup_count"),
    [
        (
            "raised-unknown",
            "PhysicalOutcomeUnknown: submitted action has no acknowledgement",
            "action_outcome_unknown",
            2,
        ),
        (
            "returned-unknown",
            "submitted action has no acknowledgement",
            "action_outcome_unknown",
            2,
        ),
        (
            "returned-agent-error",
            "program rejected the action",
            "api_error",
            2,
        ),
        (None, "RuntimeError: evaluator unavailable", "api_error", 2),
    ],
)
def test_cli_preserves_primary_failure_when_evaluation_and_cleanup_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner_failure: str | None,
    expected_error: str,
    expected_termination: str,
    cleanup_count: int,
) -> None:
    from reweave.evaluation.result import RunFinalizationContext
    from reweave.planner.base import PlannerResult
    from reweave.robots.prompt_bundle import PromptBundle
    from reweave.robots.robot_spec import RobotSpec, RunConfig
    from reweave.runtime.capability import PhysicalOutcomeUnknown

    cli = _cli_module()
    finalized: list[RunFinalizationContext] = []
    cleanup: list[str] = []

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

    def finalize_evaluation() -> None:
        raise RuntimeError("evaluator unavailable")

    def close() -> None:
        cleanup.append("toolkit")
        raise RuntimeError("close failed")

    toolkit = SimpleNamespace(
        export_harness_context=lambda output_dir: None,
        solved=lambda: False,
        finalize_episode_evaluation=finalize_evaluation,
        close=close,
    )
    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: None,
        parse_config=lambda args: RunConfig(
            recipe_tag="unknown", output_dir=tmp_path, prompt_vars={}, task_desc={}
        ),
        init_runtime=lambda *args: (
            [SimpleNamespace(stop=lambda: cleanup.append("daemon"))],
            {},
        ),
        finalize_run=lambda context: finalized.append(context),
    )
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: toolkit)
    monkeypatch.setattr(
        cli, "build_planner", lambda *args, **kwargs: SimpleNamespace(solve=solve)
    )
    monkeypatch.setattr(
        sys, "argv", ["harness", "--robot", "testrobot", "--planner", "native"]
    )

    assert cli.main() == 1

    transcript = json.loads((tmp_path / "transcript_unknown.json").read_text())
    assert transcript["termination_class"] == expected_termination
    assert transcript["finalization_error"] == {
        "type": "RuntimeError",
        "message": "evaluator unavailable",
    }
    if planner_failure == "returned-agent-error":
        assert transcript["planner_termination_class"] == "agent_execution_failure"
        assert transcript["api_error_retryable"] is False
    assert transcript["error"] == expected_error
    assert transcript["official_success"] is False
    assert len(transcript["cleanup_errors"]) == cleanup_count
    assert transcript["cleanup_errors"][-1] == (
        "toolkit cleanup failed: RuntimeError: close failed"
    )
    if planner_failure is not None:
        assert transcript["cleanup_errors"][0] == (
            "episode evaluation finalization failed: RuntimeError: evaluator unavailable"
        )
    assert cleanup == ["toolkit", "daemon"]
    assert len(finalized) == 1
    assert finalized[0].termination_class == expected_termination
    assert finalized[0].agent_error == expected_error
    assert finalized[0].environment_success is None


@pytest.mark.parametrize(
    ("termination_class", "planner_error", "expected_sessions"),
    [
        ("action_outcome_unknown", "physical RPC timed out after submission", 1),
        ("api_error", "model service timed out", 1),
        ("planner_timeout", "planner deadline reached", 2),
    ],
)
@pytest.mark.parametrize("reasoning_effort", ["low", "high"])
def test_cli_exploration_continues_only_after_structured_planner_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_class: str,
    planner_error: str,
    expected_sessions: int,
    reasoning_effort: str,
) -> None:
    from reweave.planner.base import PlannerResult
    from reweave.robots.prompt_bundle import PromptBundle
    from reweave.robots.robot_spec import RobotSpec, RunConfig

    cli = _cli_module()
    lifecycle: list[str] = []
    solve_calls = 0
    planner_settings: list[str | None] = []

    class FakeToolkit:
        def __init__(self) -> None:
            lifecycle.append("reset")

        def export_harness_context(self, output_dir: Path) -> None:
            del output_dir

        def finalize_episode_evaluation(self) -> None:
            lifecycle.append("evaluate")

        def solved(self) -> bool:
            return False

        def close(self) -> None:
            lifecycle.append("close")

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
        return [SimpleNamespace(stop=lambda: lifecycle.append("stop"))], {}

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        parser.add_argument("--explore-sessions", type=int, default=1)

    def build_planner(*args: Any, **kwargs: Any) -> SimpleNamespace:
        planner_settings.append(kwargs.get("reasoning_effort"))
        return SimpleNamespace(solve=solve)

    robot_spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=add_cli_args,
        parse_config=lambda args: RunConfig(
            recipe_tag="sessions", output_dir=tmp_path, prompt_vars={}, task_desc={}
        ),
        init_runtime=init_runtime,
    )
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: FakeToolkit())
    monkeypatch.setattr(cli, "build_planner", build_planner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--robot",
            "libero",
            "--planner",
            "native",
            "--explore",
            "--explore-sessions",
            "2",
            "--memory-profile",
            "none",
            "--reasoning-effort",
            reasoning_effort,
        ],
    )

    assert cli.main() == (1 if expected_sessions == 1 else 0)

    assert solve_calls == expected_sessions
    assert planner_settings == [reasoning_effort] * expected_sessions
    assert lifecycle == [
        "runtime",
        *["reset", "evaluate", "close"] * expected_sessions,
        "stop",
    ]
    transcript = json.loads((tmp_path / "transcript_sessions.json").read_text())
    assert transcript["termination_class"] == (
        termination_class if expected_sessions == 1 else "physical_failure"
    )
    assert transcript["error"] == (planner_error if expected_sessions == 1 else None)


@pytest.mark.parametrize("cleanup_failure", [None, "toolkit", "daemon"])
def test_baseline_capture_waits_for_cleanup_before_reporting_success(
    tmp_path, monkeypatch, cleanup_failure
):
    from reweave.evolution import runtime_manifest
    from reweave.robots.prompt_bundle import PromptBundle
    from reweave.robots.robot_spec import RobotSpec, RunConfig

    cli = _cli_module()
    cleanup = []
    captures = []

    def stop(name):
        cleanup.append(name)
        if name == cleanup_failure:
            raise RuntimeError(f"{name} cleanup failed")

    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: None,
        parse_config=lambda args: RunConfig(
            recipe_tag="capture", output_dir=tmp_path, prompt_vars={}, task_desc={}
        ),
        init_runtime=lambda *args: (
            [SimpleNamespace(stop=lambda: stop("daemon"))],
            {},
        ),
    )
    limits = tmp_path / "limits.json"
    limits.write_text(
        json.dumps(
            {
                "episode": {
                    "wall_time_s": 10.0,
                    "program_calls": None,
                    "gateway_calls": None,
                },
                "conditions": {
                    "controller_request_unit": "api_attempt",
                    "adaptation_mode": "static",
                    "physical_steps": 3,
                    "wait_steps": 0,
                    "model_requests": 5,
                    "wall_time_s": 10.0,
                    "program_calls": None,
                    "gateway_calls": None,
                    "repetition": 0,
                    "model_sampling": {},
                },
            }
        )
    )
    resources = tmp_path / "resources.json"
    resources.write_text(
        json.dumps(
            {
                "models": {},
                "environment": {"/explicit/task": "0" * 64},
                "routes": {"planner": "scripted"},
            }
        )
    )
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(
        cli,
        "get_toolkit",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: stop("toolkit")),
    )
    monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runtime_manifest,
        "capture_baseline",
        lambda *args, **kwargs: captures.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--robot",
            "testrobot",
            "--planner",
            "native",
            "--capture-bundle",
            str(tmp_path / "baseline"),
            "--kernel-limits",
            str(limits),
            "--runtime-resources",
            str(resources),
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
        ],
    )

    assert cli.main() == (1 if cleanup_failure else 0)
    assert cleanup == ["toolkit", "daemon"]
    assert len(captures) == 1
    assert not (tmp_path / "transcript_capture.json").exists()


@pytest.mark.parametrize("backend", ["api", "claude_code", "codex"])
def test_removed_planner_backends_are_rejected_before_runtime_launch(backend):
    result = _run_cli_help(
        "--robot",
        "libero",
        "--suite",
        "libero_goal",
        "--task",
        "0",
        "--planner",
        backend,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_native_model_override_is_preserved(monkeypatch):
    _, args = _capture_validated_args(
        monkeypatch, ["--robot", "libero", "--model", "explicit-model"]
    )
    assert args.model == "explicit-model"


@pytest.mark.parametrize("mode", ["auth", "api"])
def test_native_codex_connection_mode_is_explicit(monkeypatch, mode):
    _, args = _capture_validated_args(
        monkeypatch, ["--robot", "libero", "--codex-auth-mode", mode]
    )
    assert args.planner == "native"
    assert args.native_provider == "codex"
    assert args.codex_auth_mode == mode


@pytest.mark.parametrize("option", ["--dashboard", "--interactive"])
def test_native_unsupported_interaction_rejected_before_runtime(monkeypatch, option):
    cli = _cli_module()
    monkeypatch.setattr(
        cli, "build_planner", lambda *args, **kwargs: pytest.fail("planner started")
    )
    monkeypatch.setattr(
        cli, "get_toolkit", lambda *args, **kwargs: pytest.fail("toolkit started")
    )
    with pytest.raises(SystemExit) as error:
        cli.parse_cli_arguments(
            ["--robot", "libero", "--suite", "libero_goal", "--task", "0", option]
        )
    assert error.value.code == 2
