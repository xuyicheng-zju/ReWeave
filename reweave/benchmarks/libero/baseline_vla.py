# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Pure-VLA baseline for the HarnessVLA comparison.

Runs Pi0.5 closed-loop on a LIBERO task using **only** the task's native
language instruction — no planner, no Action Primitives, no memory. This is
the standard pi0.5 evaluation protocol and the control condition against
which the Harness agent harness is measured: same env_server, same
vla_server, same task/seed, same step budget; the only difference is
whether an agent sits in the loop.

Usage (attach to already-running services so the comparison shares them):

    python -m reweave.benchmarks.libero.baseline_vla \
        --suite libero_10 --task 0 --seed 0 \
        --env-endpoint 127.0.0.1:19911 --vla-endpoint 127.0.0.1:19922 \
        --output-dir logs/baseline_10_t0_s0

Writes ``result.json`` and ``rollout.mp4`` into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from reweave.benchmarks.libero.env_client import LiberoEnvClient
from reweave.benchmarks.libero.evaluation import (
    LIBERO_WAIT_STEPS_DEFAULT,
    libero_horizon_contract,
)
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.components.pi05_vla_client import Pi05VLAClient
from reweave.robots.execution_trace import ExecutionTrace
from reweave.utils.logging import get_logger
from reweave.utils.rpc.rpc_client import RpcClient
from reweave.utils.rpc.client_utils import make_rpc_client, wait_for_ready

logger = get_logger("baseline")


def _connect(endpoint: str) -> RpcClient:
    """Build an RpcClient for ``[protocol://]host:port`` and wait for ready."""
    client = make_rpc_client(endpoint)
    wait_for_ready(client)
    return client


def _gripper_opening(states) -> float:
    """Finger separation proxy, matching LiberoPrimitives.set_obs."""
    gp = np.asarray(np.asarray(states)[6:8], dtype=np.float32)
    return float(abs(gp[0]) + abs(gp[1]))


def run_baseline(
    *,
    env: LiberoEnvClient,
    model: Pi05VLAClient,
    execution_trace: ExecutionTrace,
    active_execution: ActiveExecution,
    max_chunks: int,
    instruction: str | None = None,
    wait_steps: int = LIBERO_WAIT_STEPS_DEFAULT,
) -> dict:
    """Drive Pi0.5 with the task's own instruction until success or budget.

    Returns a result dict; ``frames`` holds every env-step agentview frame
    so the caller can write a video.
    """
    # LiberoEnvClient.__init__ already reset the env; reset() again here
    # would burn one episode and re-randomize nothing (the server is
    # pinned to a fixed reset id), so just read the current obs.
    obs, _info = env.reset()
    task_language = instruction or env.get_task_language()
    logger.info("baseline instruction: %r", task_language)

    frames: list[np.ndarray] = [np.ascontiguousarray(np.asarray(obs["main_images"]))]
    chunks_used = 0
    started = time.time()

    for _ in range(int(wait_steps)):
        if env.terminated or env.truncated:
            break
        obs, _rew, _term, _trunc, _info = env.step(
            np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        )
        frames.append(np.ascontiguousarray(np.asarray(obs["main_images"])))

    for chunk in range(max_chunks):
        obs["task_descriptions"] = task_language
        obs.setdefault("extra_view_images", None)
        prediction = model.predict_action_batch(obs, mode="eval")
        active_execution.record_forward(
            prediction,
            instruction=task_language,
        )
        for action in prediction.iter_actions():
            if env.terminated or env.truncated:
                break
            step_obs, _rew, term, trunc, _info = env.step(action.action)
            frames.append(np.ascontiguousarray(np.asarray(step_obs["main_images"])))
            execution_trace.record_control_step(
                action,
                terminated=term,
                truncated=trunc,
            )
            obs = step_obs
            if env.terminated or env.truncated:
                execution_trace.record_action_suffix_discarded(
                    action,
                    reason="episode_end",
                )
                break
        chunks_used = chunk + 1

        # env.terminated / env.truncated are the client's accumulated
        # episode flags (chunk_step folds the per-step signals in).
        if env.terminated or env.truncated:
            break

    return {
        "mode": "pure_vla_baseline",
        "instruction": task_language,
        "success": env.terminated,
        "terminated": env.terminated,
        "truncated": env.truncated,
        "chunks_used": chunks_used,
        "max_chunks": max_chunks,
        "wait_steps": int(wait_steps),
        "env_steps": len(frames) - 1,
        "elapsed_s": round(time.time() - started, 1),
        "final_gripper_opening": round(_gripper_opening(obs["states"]), 4),
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=LIBERO_WAIT_STEPS_DEFAULT)
    parser.add_argument("--reset-state-id", type=int, default=None)
    parser.add_argument(
        "--libero-type",
        choices=("standard", "pro", "plus"),
        default="pro",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Pi0.5 forward budget. Unset derives it from the suite horizon.",
    )
    parser.add_argument(
        "--env-endpoint",
        required=True,
        help="[protocol://]host:port of a running env_server",
    )
    parser.add_argument(
        "--vla-endpoint",
        required=True,
        help="[protocol://]host:port of a running vla_server",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the task's native instruction (debug).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if args.max_episode_steps is None:
        horizon_contract = libero_horizon_contract(
            args.suite,
            wait_steps=args.wait_steps,
        )
    else:
        horizon_contract = libero_horizon_contract(
            args.suite,
            policy_action_horizon=args.max_episode_steps - args.wait_steps,
            wait_steps=args.wait_steps,
        )
    args.max_episode_steps = int(horizon_contract["max_episode_steps"])
    args.max_chunks = (
        args.max_chunks or (int(horizon_contract["policy_action_horizon"]) + 4) // 5
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_trace = ExecutionTrace(output_dir / "execution_trace.jsonl")
    active_execution = ActiveExecution(execution_trace)

    env = LiberoEnvClient(
        _connect(args.env_endpoint),
        expected_meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "reset_state_id": args.reset_state_id,
            "libero_type": args.libero_type,
        },
    )
    model = Pi05VLAClient(_connect(args.vla_endpoint), embodiment="libero")

    execution_ref = active_execution.begin("pure_vla_baseline")
    try:
        result = run_baseline(
            env=env,
            model=model,
            execution_trace=execution_trace,
            active_execution=active_execution,
            max_chunks=args.max_chunks,
            instruction=args.instruction,
            wait_steps=args.wait_steps,
        )
    except Exception:
        active_execution.discard_open_suffix(reason="tool_failed")
        active_execution.finish("failed")
        raise
    outcome = "succeeded" if result["success"] else "completed"
    active_execution.finish(outcome)
    result["active_execution_ref"] = execution_ref
    result["active_execution_outcome"] = outcome
    frames = result.pop("frames")

    video_path = output_dir / "rollout.mp4"
    imageio.mimwrite(video_path, frames, fps=args.fps, codec="libx264")
    result["video"] = str(video_path)
    result["task_desc"] = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "reset_state_id": args.reset_state_id,
        "evaluation_protocol": horizon_contract,
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "baseline %s t%d s%d: success=%s chunks=%d steps=%d -> %s",
        args.suite,
        args.task,
        args.seed,
        result["success"],
        result["chunks_used"],
        result["env_steps"],
        video_path,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
