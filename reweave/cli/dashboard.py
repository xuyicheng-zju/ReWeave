# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""CLI orchestration for one long-lived Dashboard Session."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reweave.cli.main import _handoff_message
from reweave.dashboard.events import RunStartedEvent
from reweave.evaluation.result import RunFinalizationContext
from reweave.evaluation.settlement import settled_episode_termination
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    API_ERROR,
    PLANNER_TIMEOUT,
    SUCCESS,
    build_planner,
)
from reweave.robots.base import get_toolkit
from reweave.runtime.capability import (
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.utils.logging import get_logger, init_output_dir

if TYPE_CHECKING:
    from reweave.dashboard.state import ClaimedTask, DashboardState
    from reweave.robots.robot_spec import RobotSpec
    from reweave.utils.daemon import ProcessDaemon

logger = get_logger("agent")


def run_dashboard_session(
    args: argparse.Namespace,
    robot_spec: RobotSpec,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    """Run one long-lived Dashboard Session with sequential fresh TaskRuns."""
    from reweave.dashboard.server import DashboardServer
    from reweave.dashboard.session import DashboardSessionController
    from reweave.dashboard.state import DashboardState
    from reweave.utils.config import get_repo_root

    dashboard_spec = robot_spec.dashboard
    if dashboard_spec is None:
        parser.error(f"robot {robot_spec.name!r} does not support Dashboard control")
    runtime_components = dashboard_spec["runtime_components"]
    shared_components = {
        component["name"]
        for component in runtime_components
        if component["scope"] == "shared"
    }
    unique_components = {
        component["name"]
        for component in runtime_components
        if component["scope"] == "unique"
    }

    if getattr(args, "env_endpoint", None) is not None:
        parser.error(
            "Dashboard task control cannot use --env-endpoint because each "
            "TaskRun requires a fresh owned env_server"
        )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        session_root = get_repo_root() / "logs" / f"{timestamp}_dashboard_session"
    else:
        session_root = Path(args.output_dir)
    session_root = init_output_dir(session_root, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    state = DashboardState(
        output_dir=session_root,
        dashboard_spec=dashboard_spec,
    )
    dashboard_server = DashboardServer(
        host=args.dashboard_host,
        port=args.dashboard_port,
        language=args.dashboard_language,
        state=state,
        planner=args.planner,
        model=args.model,
    )
    dashboard_url = dashboard_server.start()
    print(f"Dashboard: {dashboard_url}", flush=True)
    logger.info("Dashboard: %s", dashboard_url)

    controller = DashboardSessionController(
        state=state,
        start_shared=lambda: robot_spec.init_runtime(
            args,
            session_root,
            state,
            shared_components,
        ),
        run_task=lambda claimed, shared: _run_dashboard_task(
            args=args,
            robot_spec=robot_spec,
            state=state,
            claimed=claimed,
            shared_primitives_kwargs=shared,
            unique_components=unique_components,
            session_root=session_root,
        ),
    )
    try:
        controller.run()
        if state.session_state == "fatal":
            logger.error(
                "Dashboard Session is fatal. Still serving at %s; "
                "press Ctrl+C to stop.",
                dashboard_url,
            )
            threading.Event().wait()
    except KeyboardInterrupt:
        state.request_shutdown()
    return 0


def _run_dashboard_task(
    *,
    args: argparse.Namespace,
    robot_spec: RobotSpec,
    state: DashboardState,
    claimed: ClaimedTask,
    shared_primitives_kwargs: dict[str, Any],
    unique_components: set[str],
    session_root: Path,
) -> str | None:
    """Execute one fresh Dashboard TaskRun against Session-owned services."""
    task_args = copy.copy(args)
    for name, value in claimed.request.items():
        setattr(task_args, name, value)
    task_args.output_dir = str(claimed.output_dir)
    run_config = robot_spec.parse_config(task_args)
    output_dir = init_output_dir(run_config.output_dir, verbose=args.verbose)

    recipe_tag = run_config.recipe_tag
    finish_result = None
    messages: list[dict] = []
    stats: dict = {}
    agent_error: str | None = None
    planner_termination_class: str | None = None
    api_error_retryable: bool | None = None
    environment_success: bool | None = None
    finalization_error: dict | None = None
    cleanup_errors: list[str] = []
    task_daemons: list[ProcessDaemon] = []
    recipe_path = ""
    solved = False
    started = time.time()
    try:
        task_daemons, task_primitives_kwargs = robot_spec.init_runtime(
            task_args,
            output_dir,
            state,
            unique_components,
        )
        if not state.task_replacement_requested:
            primitives_kwargs = {
                **shared_primitives_kwargs,
                **task_primitives_kwargs,
            }
            prompt_vars = {**run_config.prompt_vars, "output_dir": output_dir}
            session_message = robot_spec.prompts.render("user", variables=prompt_vars)
            sessions = max(
                1,
                int(getattr(task_args, "explore_sessions", 1) or 1),
            )
            if not getattr(task_args, "explore", False):
                sessions = 1
            if not state.task_replacement_requested:
                state.emit(RunStartedEvent())
            for session_number in range(1, sessions + 1):
                if state.task_replacement_requested:
                    break
                if session_number > 1:
                    logger.info(
                        "=== handing off to agent %d/%d ===",
                        session_number,
                        sessions,
                    )
                    session_message = _handoff_message(
                        output_dir,
                        session_number,
                        sessions,
                    )
                system_prompt = robot_spec.prompts.render(
                    "system",
                    variables={
                        **prompt_vars,
                        "session_number": session_number,
                        "session_max": sessions,
                    },
                )
                state_output_dir = output_dir
                if getattr(task_args, "explore", False):
                    state_output_dir = (
                        output_dir / "sessions" / f"session_{session_number:03d}"
                    )
                    state.begin_planner_session(
                        video_path=state_output_dir / "episode.mp4",
                    )
                if args.robot_name == "libero":
                    toolkit = get_toolkit(
                        args.robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=state,
                        config=run_config,
                        mode=("exploration" if task_args.explore else "evaluation"),
                        attempts_per_session=getattr(
                            task_args,
                            "explore_attempts_per_session",
                            0,
                        ),
                        state_output_dir=state_output_dir,
                    )
                else:
                    toolkit = get_toolkit(
                        args.robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=state,
                        config=run_config,
                    )
                state.bind_toolkit(toolkit)
                solved = False
                try:
                    planner = build_planner(
                        args.planner,
                        output_dir=output_dir,
                        model=args.model,
                        native_provider=args.native_provider,
                        codex_auth_mode=args.codex_auth_mode,
                        native_service_tier=args.native_service_tier,
                        adaptation_mode=args.adaptation_mode,
                        online_workflows=getattr(args, "online_workflows", False),
                        planner_timeout_s=args.planner_timeout_s,
                        reasoning_effort=args.reasoning_effort,
                    )
                    result = planner.solve(
                        system_prompt=system_prompt,
                        user_message=session_message,
                        toolkit=toolkit,
                        max_turns=args.max_turns,
                        dashboard_interaction=state,
                    )
                    finish_result = result.finish_result
                    messages += result.messages
                    stats = result.stats
                    agent_error = result.error
                    planner_termination_class = result.termination_class
                    api_error_retryable = result.api_error_retryable
                    solved = toolkit.solved()
                    if args.robot_name == "libero" and solved:
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
                    finally:
                        try:
                            state.unbind_toolkit(toolkit)
                        except Exception as exc:
                            error = (
                                f"toolkit unbind failed: {type(exc).__name__}: {exc}"
                            )
                            cleanup_errors.append(error)
                            logger.error("%s", error)
                        finally:
                            try:
                                toolkit.close()
                            except Exception as exc:
                                error = (
                                    "toolkit cleanup failed: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                cleanup_errors.append(error)
                                logger.error("%s", error)
                if solved or state.task_replacement_requested:
                    break
                if agent_error:
                    if (
                        session_number < sessions
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
        logger.error("EXCEPTION in Dashboard TaskRun %04d: %s", claimed.number, exc)
        subsequent_error = f"{type(exc).__name__}: {exc}"
        if agent_error is None:
            agent_error = subsequent_error
        else:
            cleanup_errors.append(f"subsequent episode error: {subsequent_error}")
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
        for daemon in reversed(task_daemons):
            try:
                daemon.stop()
            except Exception as exc:
                error = f"robot cleanup failed: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                logger.error("%s", error)

        elapsed_s = time.time() - started
        termination_class = settled_episode_termination(
            planner_termination_class=planner_termination_class,
            environment_success=environment_success,
            finalization_error=finalization_error,
        )
        transcript_path = output_dir / f"transcript_{run_config.recipe_tag}.json"
        record = {
            **run_config.task_desc,
            "model": args.model,
            "elapsed_s": round(elapsed_s, 1),
            "finish": finish_result,
            "stats": stats,
            "messages": messages,
            "run_configuration": vars(task_args),
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
            "error": agent_error,
            "cleanup_errors": cleanup_errors,
        }
        try:
            with open(transcript_path, "w") as transcript_file:
                json.dump(record, transcript_file, indent=2, default=str)
        except Exception as exc:
            logger.warning(
                "failed to write TaskRun transcript %s: %s", transcript_path, exc
            )
        init_output_dir(session_root, verbose=args.verbose)

    finalization_error: str | None = None
    if robot_spec.finalize_run is not None:
        try:
            robot_spec.finalize_run(
                RunFinalizationContext(
                    output_dir=output_dir,
                    robot_name=args.robot_name,
                    task_desc=dict(run_config.task_desc),
                    environment_success=environment_success,
                    termination_class=termination_class,
                    agent_error=agent_error,
                    elapsed_s=elapsed_s,
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
        except Exception as exc:
            finalization_error = (
                f"result finalization failed: {type(exc).__name__}: {exc}"
            )
            logger.error("%s", finalization_error)

    from reweave.memory.episode import record_episode

    record_episode(transcript_path)
    return (
        agent_error
        or finalization_error
        or ("; ".join(cleanup_errors) if cleanup_errors else None)
    )
