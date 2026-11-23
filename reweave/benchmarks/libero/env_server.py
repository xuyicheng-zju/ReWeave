# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""RPC server wrapping a single-env LIBERO environment."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from reweave.benchmarks.libero.evaluation import libero_horizon_contract
from reweave.robots.components.env_facade_base import BaseEnvFacade
from reweave.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from reweave.utils.logging import get_logger

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, (
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
)

logger = get_logger("env_server")

REWEAVE_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (REWEAVE_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (referenced at runtime in _to_numpy_tree)
    from rlinf.envs.libero.libero_env import LiberoEnv




def assert_libero_backend(requested_type: str, selected_module: str) -> None:
    """Fail closed when RLinf silently selected another LIBERO package."""

    requested = str(requested_type).lower()
    expected_prefix = {
        "pro": "liberopro.",
        "plus": "liberoplus.",
        "standard": "libero.",
    }.get(requested)
    if expected_prefix is None:
        raise ValueError(f"unsupported LIBERO_TYPE={requested!r}")
    if not str(selected_module).startswith(expected_prefix):
        raise RuntimeError(
            "LIBERO backend mismatch: "
            f"LIBERO_TYPE={requested!r} selected {selected_module!r}. "
            "Refusing the RLinf fallback because it would change the benchmark."
        )


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                # Toolkit owns the episode evidence. Do not run RLinF's second
                # video writer in a shared /tmp directory across environments.
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "unused",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so we can back-project pixels to world
                # from depth + camera calibration
                "camera_depths": True,
                # RLinF owns the authoritative evaluation truncation.  Keep
                # robosuite's internal horizon beyond it so the final outer
                # action returns truncation instead of raising on the next
                # call. This preserves the benchmark's outer truncation contract.
                "horizon": max_episode_steps + 1000,
                "ignore_done": True,
                **(
                    {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                    if os.environ.get("LIBERO_ROBOT_BASE")
                    else {}
                ),
            },
        }
    )
    return cfg


def make_env(
    task_id: int,
    seed: int,
    suite_name: str = "libero_spatial",
    max_episode_steps: int = 10000,
    reset_state_id: int | None = None,
) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    from reweave.benchmarks.libero.worker import install_libero_worker

    install_libero_worker(LiberoEnv)

    requested_type = os.environ.get("LIBERO_TYPE", "standard").lower()
    benchmark_module = getattr(_bench_mod, "benchmark", None)
    selected_module = getattr(benchmark_module, "__name__", "")
    if not selected_module:
        selected_module = str(
            getattr(getattr(_bench_mod, "get_benchmark", None), "__module__", "")
        )
    assert_libero_backend(requested_type, selected_module)

    suite = _bench_mod.get_benchmark(suite_name)()
    task_count = int(suite.get_num_tasks())
    if not 0 <= int(task_id) < task_count:
        raise ValueError(
            f"LIBERO task id {task_id} is outside suite {suite_name!r} "
            f"with {task_count} tasks"
        )
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    if trials <= 0:
        raise RuntimeError(
            f"LIBERO suite {suite_name!r} task {task_id} has no reset states"
        )
    local_reset_id = seed % trials if reset_state_id is None else int(reset_state_id)
    if not 0 <= local_reset_id < trials:
        raise ValueError(
            f"task-local reset-state id {local_reset_id} is outside [0, {trials})"
        )
    rid = first_id + local_reset_id
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )




def _to_numpy_tree(x):
    """Recursively convert torch tensors to CPU numpy arrays so the result
    pickles cleanly across the agent/env_server wire."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    return x


class LiberoEnvFacade(BaseEnvFacade):
    """Implements :class:`reweave.benchmarks.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        self._env = env
        self._env_idx = 0
        self._closed = False
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.raw_obs": self.raw_obs,
                "env.get_cartesian_control_spec": self.get_cartesian_control_spec,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
            }
        )
        self._readonly_methods.add("env.get_task_language")


    def _strip(self, v):
        """Drop the leading env dim. ``v`` is either a batched numpy array
        (shape ``[B, ...]``), a length-B list (e.g. ``task_descriptions``),
        or ``None`` (optional images). LiberoEnv runs ``num_envs=1`` so
        index ``self._env_idx`` is always present."""
        if v is None:
            return None
        return v[self._env_idx]

    def get_cartesian_control_spec(self) -> dict:
        """Expose the actual controller mapping without privileged task state."""
        return self._env.env.workers[self._env_idx].env_call(
            "harness_cartesian_control_spec", target="self"
        )

    def _strip_obs(self, obs: dict) -> dict:
        """Strip the leading env dim from every value of a LIBERO obs dict."""
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """Inject the env dim onto a single-env action shaped ``[action_dim]``."""
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """Inject the env dim onto a single-env chunk shaped
        ``[chunk_size, action_dim]``."""
        return np.asarray(actions)[None]

    def _privileged_critic_state(
        self,
        *,
        reset_tracker: bool = False,
    ) -> dict[str, Any]:
        """Read the worker-owned semantic sidecar without advancing physics."""
        workers = getattr(getattr(self._env, "env", None), "workers", None)
        if workers is None or len(workers) <= self._env_idx:
            raise RuntimeError(
                "LIBERO privileged critic requires an RLinF worker env_call channel"
            )
        state = workers[self._env_idx].env_call(
            "harness_privileged_critic_state",
            kwargs={"reset_tracker": bool(reset_tracker)},
            target="self",
        )
        converted = _to_numpy_tree(state)
        if not isinstance(converted, dict) or not converted:
            raise RuntimeError("LIBERO privileged critic returned no evidence")
        if converted.get("privileged.available") is not True:
            raise RuntimeError("LIBERO privileged critic is unavailable")
        return converted

    def _attach_critic_evidence(
        self,
        info: Any,
        *,
        observation: dict[str, Any],
        reset_tracker: bool = False,
    ) -> dict[str, Any]:
        converted = _to_numpy_tree(info)
        if not isinstance(converted, dict):
            raise TypeError("LIBERO step info must be a dictionary")
        try:
            evidence = self._privileged_critic_state(reset_tracker=reset_tracker)
        except Exception as error:
            converted["physical_evidence_error"] = f"{type(error).__name__}: {error}"
            return converted
        states = np.asarray(observation.get("states", ()), dtype=np.float64).reshape(-1)
        if states.size < 3 or not np.isfinite(states[:3]).all():
            converted["physical_evidence_error"] = (
                "LIBERO critic requires a finite EEF position"
            )
            return converted
        eef = states[:3]
        pose_prefixes = {
            feature.removesuffix(".position.x")
            for feature in evidence
            if feature.startswith("privileged.entity.")
            and feature.endswith(".position.x")
        }
        pose_prefixes.update(
            {
                "privileged.task.manipulated_object",
                "privileged.task.target",
            }
        )
        for prefix in pose_prefixes:
            position_features = [
                f"{prefix}.position.x",
                f"{prefix}.position.y",
                f"{prefix}.position.z",
            ]
            if not all(feature in evidence for feature in position_features):
                continue
            position = np.asarray(
                [evidence[feature] for feature in position_features],
                dtype=np.float64,
            )
            evidence[f"{prefix}.distance_to_eef_m"] = float(
                np.linalg.norm(position - eef)
            )
        converted["critic_evidence"] = evidence
        return converted


    def reset(self):
        obs, info = self._env.reset()
        obs = self._strip_obs(_to_numpy_tree(obs))
        return obs, self._attach_critic_evidence(
            info,
            observation=obs,
            reset_tracker=True,
        )

    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(self._expand_action(action))
        obs = self._strip_obs(_to_numpy_tree(obs))
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        return (
            obs,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            self._attach_critic_evidence(info, observation=obs),
        )

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """Run a full action chunk in one RPC. ``actions`` shape
        ``[chunk_size, action_dim]`` (single env).

        Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``. ``obs`` is
        ``list[Obs]`` when ``return_all_frames=True`` (full per-step
        trajectory), or just the final ``Obs`` dict when False (default).
        ``terminated`` / ``truncated`` carry shape ``[chunk_size]`` after
        the leading env dim is stripped — the agent reduces across the
        chunk itself.
        """
        obs_list, rew, term, trunc, info = self._env.chunk_step(
            self._expand_chunk(actions)
        )
        if return_all_frames:
            obs_field = [self._strip_obs(_to_numpy_tree(o)) for o in obs_list]
            final_obs = obs_field[-1]
        else:
            # The native chunk has already executed. Only materialize the
            # requested final image on CPU; full-video callers retain all frames.
            final_obs = self._strip_obs(_to_numpy_tree(obs_list[-1]))
            obs_field = final_obs
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        converted_info = _to_numpy_tree(info)
        if not isinstance(converted_info, (list, tuple)) or not converted_info:
            raise TypeError("LIBERO chunk info must be a non-empty sequence")
        enriched_info = list(converted_info)
        enriched_info[-1] = self._attach_critic_evidence(
            enriched_info[-1],
            observation=final_obs,
        )
        return (
            obs_field,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            tuple(enriched_info)
            if isinstance(converted_info, tuple)
            else enriched_info,
        )

    def raw_obs(self) -> dict:
        return _to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with."""
        return dict(self._meta)

    def close(self) -> None:
        """Release the server-owned LIBERO environment once."""
        if self._closed:
            return
        self._env.env.close()
        self._closed = True

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return _to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return _to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--reset-state-id", type=int, default=None)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from reweave.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = int(
            libero_horizon_contract(args.suite)["max_episode_steps"]
        )
    raw_env = make_env(
        args.task,
        args.seed,
        suite_name=args.suite,
        max_episode_steps=max_episode_steps,
        reset_state_id=args.reset_state_id,
    )
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": max_episode_steps,
            "reset_state_id": args.reset_state_id,
            "libero_type": os.environ.get("LIBERO_TYPE", "standard").lower(),
        },
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()


if __name__ == "__main__":
    main()
