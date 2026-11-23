# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Construct the RoboDojo agent environment and serve RPC on the main thread."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reweave.benchmarks.robodojo.env_facade import RoboDojoEnvFacade  # noqa: E402


def prepare_source_imports(source_root: str | None) -> str:
    """Bind RoboDojo's utils before Isaac exposes OpenCV's homonymous package."""
    if source_root is None:
        from robodojo_runtime import source_root as packaged_source_root

        root = packaged_source_root()
    else:
        root = Path(source_root).expanduser().resolve()
    expected = root / "utils" / "load_file.py"
    if not expected.is_file():
        raise FileNotFoundError(f"RoboDojo source is missing {expected}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("utils.load_file")
    if Path(module.__file__).resolve() != expected.resolve():
        raise ImportError(
            f"RoboDojo utils resolved outside its source: {module.__file__}"
        )
    return str(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboDojo Isaac Sim env RPC server")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--layout-set",
        type=int,
        required=True,
        help="Official evaluation seed selecting an Eval_Layout directory",
    )
    parser.add_argument(
        "--layout",
        type=int,
        required=True,
        help="Scene index within the selected evaluation seed directory",
    )
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--save-dir", default=os.getcwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory for head and both wrist episode H.264 mp4 evidence",
    )
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--parent-watch", action="store_true")

    parser.add_argument("--source-root", default=None)
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="VLA endpoint reachable from this environment process; never a remote Host's loopback route",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="SAM endpoint reachable from this environment process",
    )
    parser.add_argument(
        "--graspgen-config",
        default=None,
        help="Server-local GraspGen-X configuration retaining calibrated gripper geometry",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--kit_args", default="")
    return parser


def build_config(args: argparse.Namespace):
    """Translate CLI settings into the RLinf wrapper and runtime task config."""
    from omegaconf import OmegaConf

    if args.layout_set < 0 or args.layout < 0 or args.max_episode_steps <= 0:
        raise ValueError(
            "RoboDojo requires a nonnegative layout and positive episode limit"
        )
    if args.num_envs != 1:
        raise ValueError("RoboDojo RPC serves exactly one environment")
    return OmegaConf.create(
        {
            "seed": args.layout_set,
            "group_size": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "use_custom_reward": False,
            "use_fixed_reset_state_ids": True,
            "max_episode_steps": args.max_episode_steps,
            "task_config": {
                "task_name": args.task,
                "env_cfg_type": args.env_cfg_type,
                "source_root": args.source_root,
                "cuda_device": args.cuda_device,
                "save_dir": args.save_dir,
                "headless": args.headless,
                "kit_args": args.kit_args,
                # Scenes are always the selected official layout, never sampled.
                "random": False,
                "max_episode_steps": args.max_episode_steps,
            },
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    args.source_root = prepare_source_imports(args.source_root)
    cfg = build_config(args)
    from reweave.benchmarks.robodojo.rlinf_env import RoboDojoAgentEnv

    env = RoboDojoAgentEnv(
        cfg,
        video_dir=args.video_dir or "",
        meta={
            "task": args.task,
            "layout_set": args.layout_set,
            "layout": args.layout,
            "env_cfg_type": args.env_cfg_type,
            "device_id": args.cuda_device,
            "num_envs": args.num_envs,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    try:
        RoboDojoEnvFacade(
            env,
            inference_config={
                "vla_endpoint": args.vla_endpoint,
                "sam3_endpoint": args.sam3_endpoint,
                "graspgen_config": args.graspgen_config,
            },
        ).serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
