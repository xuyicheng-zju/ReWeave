# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""RoboDojo robot extension — RobotSpec factory, toolkit factory, runtime hooks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reweave.benchmarks.robodojo.prompt_bundle import system_prompt, user_prompt
from reweave.dashboard.events import DashboardEventSink
from reweave.dashboard.spec import DashboardSpec
from reweave.evaluation.result import RunFinalizationContext
from reweave.memory.corpus import ReadOnlyCorpus
from reweave.planner.base import ACTION_OUTCOME_UNKNOWN
from reweave.robots.prompt_bundle import PromptBundle
from reweave.robots.robot_spec import RobotSpec, RunConfig
from reweave.robots.runtime import try_spawn_server, try_wait_server
from reweave.utils.config import get_memory_dir, get_repo_root
from reweave.utils.daemon import ProcessDaemon, reserve_free_port
from reweave.utils.persistence import atomic_text
from reweave.utils.rpc.client_utils import make_rpc_client
from reweave.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from reweave.utils.rpc.rpc_client import RpcClient


DEFAULT_ENV_CFG = "arx_x5"

ROBODOJO_TASK_SUGGESTIONS = (
    "put_bottles_into_dustbin",
    "fill_pen_holder",
    "stack_bowls_random",
)

ROBODOJO_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/harness-task",
        "usage": "/harness-task <task> <layout>",
        "fields": (
            {"name": "task", "suggestions": ROBODOJO_TASK_SUGGESTIONS},
            {"name": "layout", "kind": "integer", "minimum": 0},
        ),
        "display": "{task} / layout {layout}",
        "output_slug": "{task}_l{layout}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
        {"name": "sam3", "label": "SAM3", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "head",
            "label": "head camera",
            "artifact": "cam_head.png",
        },
        {
            "name": "wrist_left",
            "label": "left wrist camera",
            "artifact": "cam_left_wrist.png",
        },
        {
            "name": "wrist_right",
            "label": "right wrist camera",
            "artifact": "cam_right_wrist.png",
        },
    ),
    "primitives": (
        "view_env_state",
        "back_project",
        "segment",
        "locate",
        "move_to",
        "move_pose",
        "move_joints",
        "set_gripper",
        "pi0_pick",
    ),
}


def _runtime_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Resolve packaged code in the simulator interpreter, preserving overrides."""
    if args.source_root:
        source = Path(args.source_root).expanduser().resolve()
    else:
        source = Path(
            subprocess.check_output(
                [
                    str(Path(args.sim_python).expanduser()),
                    "-c",
                    "from robodojo_runtime import source_root; print(source_root())",
                ],
                text=True,
                timeout=30,
            ).strip()
        )
    xpolicy = (
        Path(args.xpolicylab_root).expanduser().resolve()
        if args.xpolicylab_root
        else source / "XPolicyLab"
    )
    overrides = {
        "ROBODOJO_SOURCE_ROOT": str(source),
        "ROBODOJO_ROOT": str(source),
        "PYTHONPATH": os.pathsep.join(
            [str(get_repo_root()), str(source), str(xpolicy)]
            + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ),
    }
    if os.environ.get("ROBODOJO_ASSETS_ROOT"):
        overrides["ROBODOJO_ASSETS_ROOT"] = os.environ["ROBODOJO_ASSETS_ROOT"]
    if args.cuda_device is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    return overrides


def get_robot_spec() -> RobotSpec:
    """Return the RoboDojo robot identity, prompt bundle, and runner hooks."""
    return RobotSpec(
        name="robodojo",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBODOJO_DASHBOARD_SPEC,
        finalize_run=finalize_run,
    )


def finalize_run(context: RunFinalizationContext) -> Path:
    """Write RoboDojo task identity and environment-authoritative run results."""
    valid = (
        context.environment_success is not None
        and context.termination_class != ACTION_OUTCOME_UNKNOWN
    )
    record = {
        "task": context.task_desc["task"],
        "layout_set": context.task_desc["layout_set"],
        "layout": context.task_desc["layout"],
        "success_source": "environment.is_success",
        "valid": valid,
        "success": context.environment_success if valid else None,
        "environment_success": context.environment_success,
        "environment_result_available": context.environment_success is not None,
        "agent_error_present": context.agent_error is not None,
        "elapsed_s": round(context.elapsed_s, 1),
        "planner": {
            "backend": context.planner,
            "model": context.model,
            "reasoning_effort": context.reasoning_effort,
            "max_turns": context.max_turns,
        },
        "planner_timeout_s": context.planner_timeout_s,
    }
    destination = context.output_dir / "result.json"
    atomic_text(destination, json.dumps(record, indent=2) + "\n")
    return destination


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the RoboDojo toolkit (common tools + RoboDojo primitives)."""
    from reweave.benchmarks.robodojo.toolkit import RoboDojoToolkit

    return RoboDojoToolkit(
        task_name=config.task_desc.get("task"),
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=ReadOnlyCorpus(
            root=config.prompt_vars.get("memory_dir") or get_memory_dir("robodojo"),
            enabled=config.prompt_vars.get("memory_profile", "local") != "none",
        ),
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument(
        "--source-root",
        default=None,
        help="Override the source tree shipped by rlinf-robodojo-runtime",
    )
    parser.add_argument(
        "--xpolicylab-root",
        default=None,
        help="XPolicyLab checkout (default: SOURCE_ROOT/XPolicyLab)",
    )
    parser.add_argument(
        "--sim-python",
        default=sys.executable,
        help="Python executable with Isaac Sim and RoboDojo installed",
    )
    parser.add_argument(
        "--pi05-python",
        default=sys.executable,
        help="Python executable with Pi_05 dependencies installed",
    )
    parser.add_argument(
        "--task",
        required=required,
        help="RoboDojo task name, e.g. put_bottles_into_dustbin",
    )
    parser.add_argument(
        "--layout-set",
        type=int,
        required=required,
        help="Official evaluation seed selecting an Eval_Layout directory",
    )
    parser.add_argument(
        "--layout",
        type=int,
        required=required,
        help="Scene index within the selected evaluation seed directory",
    )
    parser.add_argument(
        "--env-cfg-type", default=DEFAULT_ENV_CFG, help="RoboDojo env config (arx_x5)"
    )
    parser.add_argument("--max-episode-steps", type=int, required=required)
    parser.add_argument(
        "--wait-steps",
        type=int,
        choices=[0],
        default=0,
        help="RoboDojo resets own settling; no extra wait actions",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device for local services (default: inherit CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of an existing Pi_05 policy server",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server",
    )
    parser.add_argument(
        "--graspgen-config",
        default=None,
        help=(
            "YAML/JSON GraspGen-X client config. Omit to disable generate_grasps; "
            "the file must explicitly bind both X5 arms and their "
            "grasp-frame-from-TCP transforms."
        ),
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    from reweave.benchmarks.robodojo import tasks as robodojo_tasks

    source_root = getattr(args, "source_root", None)
    if source_root is not None:
        robodojo_tasks.validate_task_rule_coverage(source_root)
    err = robodojo_tasks.validate_task(args.task, source_root)
    if err:
        raise ValueError(err)
    if args.layout_set < 0 or args.layout < 0 or args.max_episode_steps <= 0:
        raise ValueError(
            "RoboDojo layout must be nonnegative and max_episode_steps positive"
        )
    if getattr(args, "graspgen_config", None) is not None:
        from reweave.robots.components.graspgen_client import load_graspgen_config

        graspgen = load_graspgen_config(args.graspgen_config)
        missing = {"left", "right"}.difference(graspgen.grippers)
        if missing:
            raise ValueError(
                "RoboDojo GraspGen-X config requires arm bindings: "
                + ", ".join(sorted(missing))
            )
    recipe_tag = f"{args.task}_s{args.layout_set}_l{args.layout}"
    memory_arg = getattr(args, "memory_dir", None)
    memory_dir = (
        Path(memory_arg).expanduser().resolve()
        if memory_arg
        else get_memory_dir("robodojo")
    )
    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{ts}_robodojo_{recipe_tag}"
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "task": args.task,
            "layout": args.layout,
            "env_cfg_type": args.env_cfg_type,
            "recipe_tag": recipe_tag,
            "memory_dir": str(memory_dir),
            "memory_profile": getattr(args, "memory_profile", "local") or "local",
            "graspgen_enabled": getattr(args, "graspgen_config", None) is not None,
        },
        task_desc={
            "task": args.task,
            "layout_set": args.layout_set,
            "layout": args.layout,
        },
    )


def _cuda_args(args: argparse.Namespace) -> list[str]:
    return (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )


def _inference_args(args: argparse.Namespace) -> list[str]:
    """Bind the environment's same-host perception clients before it starts."""
    result = []
    for field in ("vla_endpoint", "sam3_endpoint", "graspgen_config"):
        value = getattr(args, field, None)
        if value is not None:
            result.extend(["--" + field.replace("_", "-"), str(value)])
    return result


def _spawn_env_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the RoboDojo Isaac Sim env_server."""
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)

    sim_python = str(Path(args.sim_python).expanduser())
    if not Path(sim_python).exists():
        raise RuntimeError(
            f"RoboDojo sim env python not found: {sim_python}; configure --sim-python"
        )

    overrides = _runtime_overrides(args)
    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    daemon = ProcessDaemon(
        name="robodojo_env_server",
        stop_timeout_s=300.0,
        cmd=[
            sim_python,
            "-u",
            str(
                get_repo_root()
                / "reweave"
                / "benchmarks"
                / "robodojo"
                / "env_server.py"
            ),
            "--task",
            args.task,
            "--layout-set",
            str(args.layout_set),
            "--layout",
            str(args.layout),
            "--env-cfg-type",
            args.env_cfg_type,
            "--num-envs",
            "1",
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--host",
            host,
            "--port",
            str(port),
            "--video-dir",
            str(Path(output_dir) / "videos"),
            "--save-dir",
            str(output_dir),
            "--headless",
            "--enable_cameras",
            "--transport",
            "http",
            "--parent-watch",
            "--kit_args",
            "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ]
        + _cuda_args(args)
        + _inference_args(args),
        env_overrides=overrides,
        log_path=str(Path(output_dir) / "robodojo_env_server.log"),
        port_reservation=reservation,
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_sam3_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the env-agnostic SAM3 server."""
    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)

    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    daemon = ProcessDaemon(
        name="sam3_server",
        cmd=[
            sys.executable,
            str(
                get_repo_root() / "reweave" / "robots" / "components" / "sam3_server.py"
            ),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ]
        + _cuda_args(args),
        log_path=str(Path(output_dir) / "sam3_server.log"),
        port_reservation=reservation,
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the RoboDojo Pi_05 VLA server."""
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)

    pi05_python = str(Path(args.pi05_python).expanduser())
    if not Path(pi05_python).exists():
        raise RuntimeError(f"RoboDojo Pi_05 env python not found: {pi05_python}")
    overrides = _runtime_overrides(args)
    reservation = reserve_free_port()
    host, port = reservation.host, reservation.port
    policy_args = [
        "reweave.robots.components.pi05_vla_server",
        "--embodiment",
        "robodojo",
    ]
    daemon = ProcessDaemon(
        name="robodojo_vla_server",
        cmd=[
            pi05_python,
            "-u",
            "-m",
            *policy_args,
            "--host",
            host,
            "--port",
            str(port),
            "--transport",
            "http",
            "--parent-watch",
        ],
        env_overrides=overrides,
        log_path=str(Path(output_dir) / "robodojo_vla_server.log"),
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
    """Initialize every RoboDojo component, or only ``components`` when given."""
    from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient
    from reweave.robots.components.pi05_vla_client import Pi05VLAClient
    from reweave.robots.components.sam3_client import Sam3Client

    env_args = copy.copy(args)
    starters = {
        "vla": lambda: _spawn_vla_server(args, output_dir),
        "sam3": lambda: _spawn_sam3_server(args, output_dir),
        "env": lambda: _spawn_env_server(env_args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": RoboDojoEnvClient(
                rpc,
                expected_meta={
                    "task": args.task,
                    "layout_set": args.layout_set,
                    "layout": args.layout,
                    "env_cfg_type": args.env_cfg_type,
                    "device_id": args.cuda_device
                    if args.cuda_device is not None
                    else 0,
                    "num_envs": 1,
                    "max_episode_steps": args.max_episode_steps,
                },
            )
        },
        "vla": lambda rpc: {
            "vla_client": Pi05VLAClient(rpc, embodiment="robodojo"),
        },
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    timeouts = {"env": 900.0, "sam3": 300.0, "vla": 1800.0}
    wait_order = ("env", "sam3", "vla")
    selected = set(starters) if components is None else set(components)
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")

    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    for component, starter in starters.items():
        if component in selected:
            if component == "env":
                for model in ("vla", "sam3"):
                    if model in pending:
                        # Models start first so even auto-assigned local ports
                        # are explicit in the server's immutable launch argv.
                        setattr(
                            env_args, model + "_endpoint", pending[model][1].endpoint
                        )
            pending[component] = try_spawn_server(
                owned_daemons,
                dashboard_events,
                component,
                starter,
            )

    primitives_kwargs: dict[str, Any] = {}
    if getattr(args, "graspgen_config", None) is not None:
        from reweave.robots.components.graspgen_client import (
            GraspGenXClient,
            load_graspgen_config,
        )

        primitives_kwargs["graspgen_client"] = GraspGenXClient(
            load_graspgen_config(args.graspgen_config)
        )
    for component in (name for name in wait_order if name in pending):
        daemon, rpc = pending[component]
        component_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            timeouts[component],
            post_fn=partial(connectors[component], rpc),
        )
        primitives_kwargs.update(component_kwargs)

    return list(owned_daemons.values()), primitives_kwargs
