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

"""LIBERO robot extension — RobotSpec factory, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reweave.benchmarks.libero.evaluation import (
    LIBERO_WAIT_STEPS_DEFAULT,
    libero_horizon_contract,
)
from reweave.benchmarks.libero.prompt_bundle import system_prompt, user_prompt
from reweave.dashboard.events import DashboardEventSink
from reweave.dashboard.spec import DashboardSpec
from reweave.memory import ReadOnlyCorpus
from reweave.robots.prompt_bundle import PromptBundle
from reweave.robots.robot_spec import RobotSpec, RunConfig
from reweave.robots.runtime import try_spawn_server, try_wait_server
from reweave.utils.config import get_memory_dir, get_repo_root
from reweave.utils.daemon import ProcessDaemon, reserve_free_port
from reweave.utils.rpc import make_rpc_client
from reweave.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from reweave.utils.rpc import RpcClient


LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_10",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
)

LIBERO_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/harness-task",
        "usage": "/harness-task <suite> <task> <seed>",
        "fields": (
            {"name": "suite", "suggestions": LIBERO_SUITE_NAMES},
            {"name": "task", "kind": "integer", "minimum": 0},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{suite} / task {task} / seed {seed}",
        "output_slug": "{suite}_t{task}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
        {"name": "sam3", "label": "SAM3", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "camera",
            "label": "fixed camera",
            "artifact": "agentview.png",
        },
        {
            "name": "wrist",
            "label": "wrist camera",
            "artifact": "wrist.png",
        },
    ),
    "primitives": (
        "move_to",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
    ),
}


def get_robot_spec() -> RobotSpec:
    """Return the LIBERO robot identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=LIBERO_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    """Return the LIBERO toolkit for the current session."""
    from reweave.benchmarks.libero.toolkit import LiberoToolkit

    memory = ReadOnlyCorpus(
        enabled=config.prompt_vars.get("memory_profile", "local") != "none",
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("libero"),
    )
    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=state_output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register LIBERO CLI flags on the shared ``parser``.

    When ``use_dashboard`` is True, ``--suite`` / ``--task`` are made optional
    because each TaskRun supplies them through the Dashboard task command.
    Under CLI-only, they are required — argparse errors out early with the
    usual usage message.
    """
    required = not use_dashboard
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help=(
            "Total control-step budget including post-reset wait actions. "
            "Default: 10000."
        ),
    )
    parser.add_argument(
        "--wait-steps",
        type=int,
        default=LIBERO_WAIT_STEPS_DEFAULT,
        help="Counted post-reset no-op actions (Zetta/OpenPI default: 10).",
    )
    parser.add_argument(
        "--reset-state-id",
        type=int,
        default=None,
        help=(
            "Task-local zero-based LIBERO reset-state id. Unset preserves the "
            "Harness/Zetta seed modulo mapping; set it for cross-framework cells."
        ),
    )
    parser.add_argument(
        "--libero-type",
        default=None,
        choices=["standard", "pro", "plus"],
        help="LIBERO variant (auto-routed from suite suffix if not set).",
    )
    parser.add_argument(
        "--suite",
        default=None,
        required=required,
        help="e.g. libero_object_task, libero_spatial_swap",
    )
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--explore-attempts-per-session",
        type=int,
        default=5,
        help="Attempts per exploration session (default: 5; 0 disables limit).",
    )
    parser.add_argument(
        "--explore-sessions",
        type=int,
        default=3,
        help="Independent planner sessions per exploration run (default: 3).",
    )
    parser.add_argument(
        "--env-launch-config",
        type=Path,
        help="Operator-owned SSH environment configuration; native sessions remain local.",
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server "
        "(protocol=http|socket, defaults to http). "
        "If unset, a local env_server is spawned.",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of an existing vla_server "
        "(protocol=http|socket, defaults to http). "
        "If unset, a local vla_server is spawned.",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server "
        "(protocol=http|socket, defaults to http). "
        "If unset, a local SAM3 server is spawned.",
    )
    parser.add_argument(
        "--graspgen-config",
        default=None,
        help=(
            "YAML/JSON GraspGen-X client config. Omit to disable generate_grasps; "
            "the file must explicitly bind the Panda 'right' arm gripper and "
            "grasp-frame-from-TCP transform."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to expose via CUDA_VISIBLE_DEVICES.",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers.

    Under ``--dashboard``, ``_add_cli_args`` left ``--suite`` / ``--task``
    optional so the dashboard could fill them; this is where we enforce
    they're set now that any overrides have been applied.
    """
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")
    if args.reset_state_id is not None and args.reset_state_id < 0:
        raise ValueError("--reset-state-id must be non-negative")
    if getattr(args, "graspgen_config", None) is not None:
        from reweave.robots.components.graspgen_client import load_graspgen_config

        config = load_graspgen_config(args.graspgen_config)
        if "right" not in config.grippers:
            raise ValueError("LIBERO GraspGen-X config requires a 'right' arm binding")
    requested_total_horizon = args.max_episode_steps
    if requested_total_horizon is None:
        horizon_contract = libero_horizon_contract(
            args.suite,
            wait_steps=args.wait_steps,
        )
    else:
        policy_horizon = int(requested_total_horizon) - int(args.wait_steps)
        horizon_contract = libero_horizon_contract(
            args.suite,
            policy_action_horizon=policy_horizon,
            wait_steps=args.wait_steps,
        )
    args.max_episode_steps = int(horizon_contract["max_episode_steps"])
    args.wait_steps = int(horizon_contract["wait_steps"])
    args.evaluation_protocol = horizon_contract

    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    explore = bool(getattr(args, "explore", False))
    requested_profile = getattr(args, "memory_profile", None)
    if explore and requested_profile == "none":
        raise ValueError(
            "exploration requires file tools for run-local attempt archives"
        )
    if explore and args.explore_sessions <= 0:
        raise ValueError("--explore-sessions must be greater than 0")
    memory_profile = requested_profile or "local"
    args.memory_profile = memory_profile
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("libero")
    )
    frozen_bundle = (
        getattr(args, "agent_bundle", None) is not None
        or getattr(args, "capture_bundle", None) is not None
    )
    if frozen_bundle and not memory_dir.is_dir():
        raise ValueError(
            "frozen memory requires an explicit directory, including an empty one"
        )
    prompt_vars = {
        "frozen_bundle": frozen_bundle,
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
        "mode": "explore" if explore else "eval",
        "memory_profile": memory_profile,
        "memory_dir": str(memory_dir),
        "reference_tag": f"{args.suite.replace('libero_', '')}_t{args.task}_s0",
        "session_number": 1,
        "session_max": max(1, args.explore_sessions) if explore else 1,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
        )
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "physical_step_budget": args.max_episode_steps,
            "vla_execution_horizon": horizon_contract["actions_per_chunk"],
        },
    )


def _cuda_args(args: argparse.Namespace) -> list[str]:
    return (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )


def _resolved_libero_type(args: argparse.Namespace) -> str:
    from reweave.utils.config import get_libero_type

    return str(args.libero_type or get_libero_type()).lower()


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    remote = getattr(args, "env_launch_config", None)
    if args.env_endpoint is not None:
        if remote is not None:
            raise ValueError(
                "Choose either an environment endpoint or an owned remote launch"
            )
        return None, make_rpc_client(args.env_endpoint)
    if remote is not None and (
        args.cuda_device is None
        or args.reset_state_id is None
        or _resolved_libero_type(args) != "pro"
    ):
        raise ValueError(
            "Remote LIBERO-Pro requires an explicit device and reset-state id"
        )

    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    cmd = [
        sys.executable,
        "-m",
        "reweave.benchmarks.libero.env_server",
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--seed",
        str(args.seed),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--parent-watch",
        *_cuda_args(args),
    ]
    if args.reset_state_id is not None:
        cmd.extend(["--reset-state-id", str(args.reset_state_id)])
    if remote is not None:
        selected = []
        for option in (
            "--suite",
            "--task",
            "--seed",
            "--reset-state-id",
            "--max-episode-steps",
            "--cuda-device",
        ):
            index = cmd.index(option)
            selected.extend(cmd[index : index + 2])
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "reweave.robots.remote_environment",
            "bridge",
            "--config",
            str(remote.resolve()),
            "--",
            *selected,
        ]
    daemon = ProcessDaemon(
        name="env_server",
        cmd=cmd,
        env_overrides={
            "LIBERO_TYPE": _resolved_libero_type(args),
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
        },
        log_path=str(output_dir / "env_server.log"),
        cwd=str(output_dir.resolve()),
        port_reservation=reservation,
        stop_timeout_s=120.0 if remote is not None else 15.0,
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)

    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    daemon = ProcessDaemon(
        name="vla_server",
        cmd=[
            sys.executable,
            "-m",
            "reweave.robots.components.pi05_vla_server",
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            *_cuda_args(args),
        ],
        log_path=str(output_dir / "vla_server.log"),
        cwd=str(output_dir.resolve()),
        port_reservation=reservation,
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_sam3_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)

    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    daemon = ProcessDaemon(
        name="sam3_server",
        cmd=[
            sys.executable,
            "-m",
            "reweave.robots.components.sam3_server",
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            *_cuda_args(args),
        ],
        log_path=str(output_dir / "sam3_server.log"),
        cwd=str(output_dir.resolve()),
        port_reservation=reservation,
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every LIBERO component, or only ``components`` when given."""
    from reweave.benchmarks.libero.env_client import LiberoEnvClient
    from reweave.robots.components.pi05_vla_client import Pi05VLAClient
    from reweave.robots.components.sam3_client import Sam3Client

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
        "sam3": lambda: _spawn_sam3_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": LiberoEnvClient(
                rpc,
                expected_meta={
                    "suite": args.suite,
                    "task": args.task,
                    "seed": args.seed,
                    "max_episode_steps": args.max_episode_steps,
                    "reset_state_id": args.reset_state_id,
                    "libero_type": _resolved_libero_type(args),
                },
            )
        },
        "vla": lambda rpc: {"model": Pi05VLAClient(rpc, embodiment="libero")},
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    selected = set(starters) if components is None else components
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown LIBERO runtime components: {sorted(unknown)}")

    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    for component, starter in starters.items():
        if component in selected:
            pending[component] = try_spawn_server(
                owned_daemons,
                dashboard_events,
                component,
                starter,
            )

    primitives_kwargs: dict[str, Any] = {
        "wait_steps": args.wait_steps,
        "filesystem_tools_enabled": args.memory_profile != "none",
    }
    if getattr(args, "graspgen_config", None) is not None:
        from reweave.robots.components.graspgen_client import (
            GraspGenXClient,
            load_graspgen_config,
        )

        primitives_kwargs["graspgen_client"] = GraspGenXClient(
            load_graspgen_config(args.graspgen_config)
        )
    wait_order = ("env", "sam3", "vla")
    for component in (name for name in wait_order if name in pending):
        daemon, rpc = pending[component]
        component_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            300.0,
            post_fn=partial(connectors[component], rpc),
        )
        primitives_kwargs.update(component_kwargs)
    return list(owned_daemons.values()), primitives_kwargs
