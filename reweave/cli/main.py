# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Physical agent main CLI entrypoint."""

# `reweave/cli/`
#
# CLI entrypoints for Harness (currently just `main.py`).
#
# ## Run
#
# Invoke the project CLI as a Python module:
#
# ```bash
# python -m reweave.cli --robot libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# The rollout runner reuses the pure argument parser below. Runtime modules
# must not import this entrypoint, which imports their implementations.
from __future__ import annotations

import argparse
import json
import queue
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path

from reweave.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from reweave.dashboard.events import (
    NullDashboardEventSink,
    RunStartedEvent,
)
from reweave.evaluation.result import RunFinalizationContext
from reweave.evaluation.settlement import settled_episode_termination
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    API_ERROR,
    PLANNER_TIMEOUT,
    REASONING_EFFORTS,
    SUCCESS,
    build_planner,
    task_evidence_is_complete,
)
from reweave.robots.robot_spec import RobotSpec
from reweave.robots.base import enumerate_robots, get_robot_spec, get_toolkit
from reweave.runtime.capability import (
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.utils.logging import get_logger, init_output_dir

logger = get_logger("agent")




def _build_argparser() -> argparse.ArgumentParser:
    known_robots = enumerate_robots()
    known_robots_text = ", ".join(known_robots) if known_robots else "none"
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Harness: Agentic Infrastructure for the Physical World",
        add_help=False,
    )

    ap.add_argument(
        "--robot",
        dest="robot_name",
        required=False,
        choices=known_robots,
        help=f"Robot backend. Known robots: {known_robots_text}.",
    )

    ap.add_argument(
        "--agent-bundle", type=Path, help="Initial W/C/L bundle for the episode."
    )
    ap.add_argument(
        "--kernel-limits",
        type=Path,
        help="Fixed policy and total episode budgets for bundle execution.",
    )
    ap.add_argument(
        "--runtime-manifest",
        type=Path,
        help="Recorded runtime configuration, required for bundle execution.",
    )
    ap.add_argument(
        "--capture-bundle",
        type=Path,
        help="Capture a rebuildable baseline after Toolkit initialization, without a controller query.",
    )
    ap.add_argument(
        "--runtime-resources",
        type=Path,
        help="Model/environment resource configuration JSON for baseline capture.",
    )
    # models
    ap.add_argument(
        "--planner",
        default="native",
        choices=["native"],
        help="Decision backend; native executes the W/C/L harness through a persistent CLI session.",
    )
    ap.add_argument(
        "--native-provider",
        choices=["codex", "claude", "cursor"],
        default="codex",
        help="Native model process. codex uses app-server; claude uses Claude Code through the Anthropic SDK; cursor uses the logged-in cursor-cli subscription.",
    )
    from reweave.planner.codex_connection import default_codex_auth_mode

    ap.add_argument(
        "--codex-auth-mode",
        choices=("auth", "api"),
        default=default_codex_auth_mode(),
        help="Codex authentication via local login or OPENAI_API_KEY.",
    )
    ap.add_argument(
        "--adaptation-mode", choices=["static", "adaptive"], default="static"
    )
    ap.add_argument(
        "--online-workflows",
        action="store_true",
        default=False,
        help="Experimental online W authoring and immediate execution; disabled by default. Does not grant C/L editing.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model identifier used by the selected native provider.",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Shared root/local controller request limit, including retries; VLA/SAM inference is recorded separately.",
    )
    ap.add_argument(
        "--native-service-tier",
        choices=("default", "fast"),
        default="default",
        help="Native Codex speed tier; independent of reasoning effort. Fast uses the account's Fast mode.",
    )
    ap.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="medium",
        help="Native reasoning effort; RoboDojo defaults to low.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Optional native episode wall-clock cap in seconds; 0 or omitted disables it.",
    )

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--memory-profile",
        choices=["none", "local"],
        default=None,
        help=(
            "Memory profile. 'none' disables published corpus access; 'local' reads "
            "local published files. Both retain the fixed prompt and private "
            "episode notes; nothing is downloaded or published during execution."
        ),
    )
    ap.add_argument(
        "--memory-dir",
        default=None,
        help="Local memory root (environment default when omitted).",
    )
    ap.add_argument(
        "--explore",
        action="store_true",
        help="Enable multiple exploration attempts with run-local evidence archives.",
    )
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start a long-lived local Dashboard Session. Session settings "
        "are read from the CLI.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "translation; defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )
    return ap


def _handoff_message(output_dir, session_number: int, session_max: int) -> str:
    """Build the opening message for a continuation session."""
    attempts_dir = Path(output_dir) / "attempts"
    prior = (
        sorted(p.name for p in attempts_dir.glob("attempt_*_failed.json"))
        if attempts_dir.is_dir()
        else []
    )
    return (
        f"You are agent {session_number} of up to {session_max} on this cell. "
        f"{len(prior)} attempt(s) by earlier agents are archived in "
        f"{attempts_dir}/ ({', '.join(prior) if prior else 'none yet'}), and their "
        "working notes remain in these run-local attempt archives.\n\n"
        "Read every archive and the working notes before acting. Do not repeat "
        "failed approaches. A fresh toolkit has already restored a clean scene; "
        "inspect it before acting."
    )


def _start_continuation_session(
    args,
    *,
    output_dir,
    recipe_tag,
    dashboard_events,
    prompt_bundle,
    prompt_vars,
    session_number: int,
    session_max: int,
):
    """Build a fresh planner and prompts for an exploration handoff."""
    logger.info("=== handing off to agent %d/%d ===", session_number, session_max)
    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        model=args.model,
        planner_timeout_s=args.planner_timeout_s,
        reasoning_effort=args.reasoning_effort,
        native_provider=args.native_provider,
        codex_auth_mode=args.codex_auth_mode,
        native_service_tier=args.native_service_tier,
        adaptation_mode=args.adaptation_mode,
        online_workflows=getattr(args, "online_workflows", False),
    )
    system_prompt = prompt_bundle.render(
        "system",
        variables={
            **prompt_vars,
            "session_number": session_number,
            "session_max": session_max,
        },
    )
    session_message = _handoff_message(output_dir, session_number, session_max)
    return planner, system_prompt, session_message


def parse_cli_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace, RobotSpec]:
    """Parse the real CLI and mode constraints without creating runtime services.

    Return the parser as well so fixed execution recipes can enforce option
    ownership using its actual Action destinations, including aliases.
    Robot-specific configuration derivation remains a separate pure step.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_argparser()
    # Two-phase argparse: first grab --robot / --dashboard so we know
    # which robot's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args(argv)

    if early.robot_name is None:
        if "-h" not in argv and "--help" not in argv:
            parser.error("--robot is required")
    else:
        robot_spec = get_robot_spec(early.robot_name)
        robot_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    args = parser.parse_args(argv)
    args.robot_name = early.robot_name
    if args.planner == "native":
        if args.model is None:
            if args.native_provider == "claude":
                from reweave.planner.providers.claude import ClaudeDecisionSession

                args.model = ClaudeDecisionSession.DEFAULT_MODEL
            elif args.native_provider == "cursor":
                from reweave.planner.cursor_connection import DEFAULT_CURSOR_MODEL

                args.model = DEFAULT_CURSOR_MODEL
            else:
                from reweave.planner.providers.codex import CodexDecisionSession

                args.model = CodexDecisionSession.DEFAULT_MODEL
        if "--reasoning-effort" not in argv and not any(
            value.startswith("--reasoning-effort=") for value in argv
        ):
            args.reasoning_effort = "low" if args.robot_name == "robodojo" else "medium"
    if args.agent_bundle is not None or args.capture_bundle is not None:
        if args.robot_name == "libero" and args.cuda_device is None:
            parser.error("frozen LIBERO episodes require explicit --cuda-device")
        if args.kernel_limits is None or args.planner != "native":
            parser.error("--agent-bundle requires --kernel-limits and --planner native")
        if args.explore or args.dashboard or args.interactive:
            parser.error(
                "frozen bundles require isolated episodes without interactive or exploration mode"
            )
        if args.agent_bundle is not None and (
            args.capture_bundle is not None or args.runtime_manifest is None
        ):
            parser.error(
                "bundle execution requires --runtime-manifest and cannot also capture"
            )
        if args.capture_bundle is not None and args.runtime_resources is None:
            parser.error("baseline capture requires --runtime-resources")
        if args.memory_profile != "local" or args.memory_dir is None:
            parser.error(
                "frozen bundles require an explicit private --memory-profile local --memory-dir"
            )
    elif args.kernel_limits is not None:
        parser.error("--kernel-limits requires --agent-bundle")
    if args.dashboard and args.interactive:
        parser.error("--dashboard and --interactive cannot be used together")
    if args.explore and args.robot_name != "libero":
        parser.error("--explore is currently supported only for LIBERO")
    if args.explore and getattr(args, "explore_sessions", 1) <= 0:
        parser.error("--explore-sessions must be greater than 0")
    if args.dashboard or args.interactive:
        parser.error(
            "native episodes do not support Dashboard task replacement or interactive steering; use isolated CLI episodes"
        )
    args.memory_profile = args.memory_profile or "local"
    return parser, args, robot_spec


def main() -> int:
    parser, args, robot_spec = parse_cli_arguments()
    if args.dashboard:
        from reweave.cli.dashboard import run_dashboard_session

        return run_dashboard_session(args, robot_spec, parser=parser)

    run_config = robot_spec.parse_config(args)
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    task_desc = run_config.task_desc

    robot_name = args.robot_name

    # mkdir + logging wiring (robot-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    dashboard_events = NullDashboardEventSink()

    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        model=args.model,
        planner_timeout_s=args.planner_timeout_s,
        reasoning_effort=args.reasoning_effort,
        native_provider=args.native_provider,
        codex_auth_mode=args.codex_auth_mode,
        native_service_tier=args.native_service_tier,
        adaptation_mode=args.adaptation_mode,
        online_workflows=getattr(args, "online_workflows", False),
    )
    prompt_bundle = robot_spec.prompts
    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    daemons, primitives_kwargs = robot_spec.init_runtime(
        args,
        output_dir,
        dashboard_events,
        None,
    )

    if args.agent_bundle is not None or args.capture_bundle is not None:
        primitives_kwargs["frozen_bundle"] = True

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    planner_termination_class = None
    api_error_retryable = None
    stats: dict = {}
    first_user_msg: str | None = user_msg
    # Exploration may hand off between independent planner contexts.
    sessions = max(1, int(getattr(args, "explore_sessions", 1) or 1))
    if not getattr(args, "explore", False):
        sessions = 1
    bundle = None
    captured_bundle = False
    recipe_path = ""
    solved = False
    cleanup_errors: list[str] = []
    environment_success: bool | None = None
    finalization_error: dict | None = None

    def close_toolkit(toolkit) -> None:
        try:
            toolkit.close()
        except Exception as exc:
            error = f"toolkit cleanup failed: {type(exc).__name__}: {exc}"
            cleanup_errors.append(error)
            logger.error("%s", error)

    try:
        if await_first_prompt is not None:
            first_user_msg = await_first_prompt()
            if first_user_msg is None:
                logger.info("no task entered; ending session before start.")
        if first_user_msg is not None:
            dashboard_events.emit(RunStartedEvent())
        session_msg = first_user_msg
        for session_number in range(1, sessions + 1):
            if session_msg is None:
                break
            if session_number > 1:
                planner, system_prompt, session_msg = _start_continuation_session(
                    args,
                    output_dir=output_dir,
                    recipe_tag=recipe_tag,
                    dashboard_events=dashboard_events,
                    prompt_bundle=prompt_bundle,
                    prompt_vars=prompt_vars,
                    session_number=session_number,
                    session_max=sessions,
                )
            state_output_dir = output_dir
            if getattr(args, "explore", False):
                state_output_dir = (
                    output_dir / "sessions" / f"session_{session_number:03d}"
                )
            with ExitStack() as session_cleanup:
                if robot_name == "libero":
                    toolkit = get_toolkit(
                        robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=dashboard_events,
                        config=run_config,
                        mode=("exploration" if args.explore else "evaluation"),
                        attempts_per_session=getattr(
                            args,
                            "explore_attempts_per_session",
                            0,
                        ),
                        state_output_dir=state_output_dir,
                    )
                else:
                    toolkit = get_toolkit(
                        robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=dashboard_events,
                        config=run_config,
                    )
                session_cleanup.callback(close_toolkit, toolkit)
                if args.agent_bundle is not None or args.capture_bundle is not None:
                    from reweave.evolution.bundle import AgentBundle
                    from reweave.evolution.contracts import EvaluationConditions
                    from reweave.evolution.runtime_manifest import (
                        RuntimeManifest,
                        RuntimeResources,
                        capture_baseline,
                    )
                    from reweave.runtime.kernel import EpisodeLimits
                    from reweave.utils.persistence import atomic_json

                    limits = json.loads(args.kernel_limits.read_text())
                    if set(limits) != {"episode", "conditions"}:
                        raise ValueError(
                            "kernel limits require exactly episode and conditions"
                        )
                    conditions = EvaluationConditions.model_validate(
                        limits["conditions"]
                    )
                    episode_limits = EpisodeLimits.model_validate(limits["episode"])
                    execution_limits = {
                        "episode": episode_limits.model_dump(mode="json"),
                    }
                    if args.capture_bundle is not None:
                        capture_baseline(
                            args.capture_bundle.resolve(),
                            arguments=vars(args),
                            limits=execution_limits,
                            conditions=conditions,
                            resources=RuntimeResources.model_validate_json(
                                args.runtime_resources.read_text()
                            ),
                            memory=Path(args.memory_dir).expanduser().resolve(),
                            toolkit=toolkit,
                            planner=planner,
                        )
                        captured_bundle = True
                        break
                    bundle = AgentBundle.model_validate_json(
                        args.agent_bundle.read_text()
                    )
                    manifest = RuntimeManifest.model_validate_json(
                        args.runtime_manifest.read_text()
                    )
                    atomic_json(
                        output_dir / "harness_context/runtime_manifest.json",
                        manifest.model_dump(mode="json"),
                    )
                    if manifest.conditions != conditions:
                        raise ValueError(
                            "recorded execution conditions differ from requested conditions"
                        )
                    manifest.validate_conditions(
                        arguments=vars(args),
                        limits=execution_limits,
                        planner=planner,
                    )
                    planner.bind_bundle(
                        bundle,
                        episode_limits=episode_limits,
                    )
                solved = False
                try:
                    toolkit.export_harness_context(state_output_dir)
                    planner_message = session_msg
                    planner_system_prompt = system_prompt
                    planner_toolkit = toolkit
                    planner_max_turns = args.max_turns
                    result = planner.solve(
                        system_prompt=planner_system_prompt,
                        user_message=planner_message,
                        toolkit=planner_toolkit,
                        max_turns=planner_max_turns,
                        input_queue=input_queue,
                    )
                    finish_result = result.finish_result
                    messages += result.messages
                    stats = result.stats
                    agent_error = result.error
                    planner_termination_class = result.termination_class
                    api_error_retryable = result.api_error_retryable
                    solved = toolkit.solved()
                    if robot_name == "libero" and solved:
                        recipe_path = toolkit.write_recipe(recipe_tag)
                finally:
                    execution_error = sys.exc_info()[1]
                    try:
                        toolkit.finalize_episode_evaluation()
                        solved = toolkit.solved()
                        environment_success = solved
                    except Exception as exc:
                        finalization_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        environment_success = None
                        solved = False
                        error = (
                            "episode evaluation finalization failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        cleanup_errors.append(error)
                        if execution_error is None and agent_error is None:
                            agent_error = f"{type(exc).__name__}: {exc}"
                        logger.error("%s", error)
            if solved:
                break
            if agent_error:
                if (
                    getattr(args, "explore", False)
                    and session_number < sessions
                    and planner_termination_class == PLANNER_TIMEOUT
                ):
                    logger.warning(
                        "session %d/%d timed out; continuing with a fresh handoff",
                        session_number,
                        sessions,
                    )
                    continue
                break
    except Exception as exc:
        subsequent_error = f"{type(exc).__name__}: {exc}"
        if agent_error is None:
            agent_error = subsequent_error
        else:
            cleanup_errors.append(f"subsequent episode error: {subsequent_error}")
        logger.error("EXCEPTION in agent loop: %s", subsequent_error)
        if isinstance(exc, PhysicalOutcomeUnknown):
            planner_termination_class = ACTION_OUTCOME_UNKNOWN
        elif planner_termination_class == ACTION_OUTCOME_UNKNOWN:
            pass
        elif isinstance(exc, (EnvironmentInvalid, EnvironmentServiceFailure)):
            planner_termination_class = API_ERROR
            api_error_retryable = False
        elif planner_termination_class is None:
            planner_termination_class = API_ERROR
            api_error_retryable = bool(getattr(exc, "retryable", False))
    finally:
        if recipe_path:
            logger.info("recipe: %s", recipe_path)
        else:
            logger.info("recipe: not written (cell unsolved)")
        for daemon in reversed(daemons):
            try:
                daemon.stop()
            except Exception as exc:
                error = f"robot cleanup failed: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                logger.error("%s", error)

    if captured_bundle:
        return 1 if cleanup_errors else 0

    elapsed = time.time() - t0
    termination_class = settled_episode_termination(
        planner_termination_class=planner_termination_class,
        environment_success=environment_success,
        finalization_error=finalization_error,
    )

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "run_configuration": vars(args),
        "agent_run_contract": {
            "planner_backend": args.planner,
            "planner_model": args.model,
            "planner_max_turns": args.max_turns,
            "reasoning_effort": args.reasoning_effort,
            "prompt_profile": (
                getattr(args, "memory_profile", "none")
                if robot_name == "libero"
                else "none"
            ),
            "physical_step_budget": getattr(args, "max_episode_steps", None),
            "vla_execution_horizon": getattr(args, "evaluation_protocol", {}).get(
                "actions_per_chunk"
            ),
        },
        "bundle_id": bundle.record_id if bundle is not None else None,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "environment_success": environment_success,
        "finalization_error": finalization_error,
        "official_success": termination_class == SUCCESS,
        "termination_class": termination_class,
        "planner_termination_class": planner_termination_class,
        "planner_api_error_retryable": api_error_retryable,
        "api_error_retryable": (
            False if finalization_error is not None else api_error_retryable
        )
        if termination_class == API_ERROR
        else None,
        "stats": stats,
        "messages": messages,
        "system_prompt": system_prompt,
        "task_message": user_msg,
        "error": agent_error,
        "cleanup_errors": cleanup_errors,
    }
    # One process = one episode. Overwrite so a resume in the same folder
    # does not concatenate two objects (run_comparison then cannot parse).
    from reweave.evolution.evidence.records import (
        archive_record,
        deduplicate_episode_media,
    )

    with open(transcript_path, "w") as f:
        json.dump(
            archive_record(
                record,
                Path(output_dir),
                media_directory=Path(output_dir) / "harness_context/observations",
            ),
            f,
            indent=2,
            default=str,
        )

    if cleanup_errors:
        logger.warning(
            "media deduplication skipped: episode writers did not close cleanly"
        )
    else:
        logger.info(
            "settled episode media storage: %s",
            deduplicate_episode_media(Path(output_dir)),
        )
    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "termination: %s task_evidence_complete=%s",
        termination_class,
        task_evidence_is_complete(termination_class),
    )
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)

    if robot_spec.finalize_run is not None:
        try:
            result_path = robot_spec.finalize_run(
                RunFinalizationContext(
                    output_dir=Path(output_dir),
                    robot_name=robot_name,
                    task_desc=dict(task_desc),
                    environment_success=environment_success,
                    termination_class=termination_class,
                    agent_error=agent_error,
                    elapsed_s=elapsed,
                    planner=args.planner,
                    model=args.model,
                    reasoning_effort=(args.reasoning_effort),
                    max_turns=args.max_turns,
                    planner_timeout_s=args.planner_timeout_s,
                    finish_result=(
                        dict(finish_result) if finish_result is not None else None
                    ),
                    stats=dict(stats),
                )
            )
            if result_path:
                logger.info("run result: %s", result_path)
        except Exception as exc:
            agent_error = f"result finalization failed: {type(exc).__name__}: {exc}"
            logger.error("%s", agent_error)

    from reweave.memory.episode import record_episode

    record_episode(transcript_path)
    return 1 if agent_error or cleanup_errors else 0


if __name__ == "__main__":
    sys.exit(main())
