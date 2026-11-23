# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Harness agent operations over the RLinf RoboDojo training adapter."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv

from reweave.benchmarks.robodojo.access import native_action, public_observation
from reweave.benchmarks.robodojo.agent_support import (
    _apply_trajectory_window,
    _obs_dict,
    _plan_joint_path,
    _plan_pose_path,
    _plan_position_path,
    _refresh_calibrated_camera_frames,
    _select_native_eval_layout,
    _solve_ik_pose,
    _synchronize_tiled_camera_apertures,
    _VideoRecorder,
)
from reweave.benchmarks.robodojo.evaluation import terminal_metrics
from reweave.benchmarks.robodojo.observation_service import ObservationService
from reweave.runtime.capability import EnvironmentInvalid
from reweave.utils.logging import get_logger

logger = get_logger(__name__)


def _write_constructor_failure(cfg, phase: str, error: BaseException) -> None:
    """Persist a constructor failure before closing Isaac Kit."""
    path = Path(str(cfg.task_config.save_dir)) / "server_init.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "phase": phase,
        "time": time.time(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_launch_receipt(cfg, environment) -> None:
    """Record which native scene file and official budget this server actually loaded."""
    path = Path(str(cfg.task_config.save_dir)) / "server_launch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": environment.meta["task"],
        "layout_selection": environment.layout_selection_receipt,
        "native_step_limit": int(environment.env.step_lim),
        "camera_calibration": environment.camera_calibration_receipt,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def _infer_action_type(action: dict) -> str:
    return "ee" if any(key.endswith("_ee_pose") for key in action) else "joint"


class RoboDojoAgentEnv(RoboDojoEnv):
    """Single-scene agent adapter with native RPC observations and diagnostics."""

    def __init__(self, cfg, *, meta: dict, video_dir: str = "") -> None:
        self.meta = {**meta, "observation_transport": "server_snapshot"}
        self._metrics_path = (
            Path(str(cfg.task_config.save_dir)) / "official_metrics.json"
        )
        self._evaluation_ready = False
        self.observations = ObservationService(str(cfg.task_config.save_dir))
        self.recorder = _VideoRecorder(video_dir)
        phase = "constructing_native_environment"
        try:
            super().__init__(cfg, 1, 0, 1, None, record_metrics=False)
            phase = "selecting_native_layout_directory"
            self.layout_selection_receipt = _select_native_eval_layout(
                self.env,
                meta["layout_set"],
                meta["layout"],
            )
            # Native cameras do not exist until the first reset launches the
            # simulator. This initialized scene becomes the owned episode.
            phase = "initializing_native_cameras"
            self.venv.reset(env_seeds=[meta["layout"]])
            phase = "registering_native_score_checks"
            # CollectEnv registers success checks but omits the score setup
            # performed by official EvalEnv. Register before any action so
            # process/transition scores observe the whole episode.
            if hasattr(self.env, "get_score"):
                self.env.get_score()
            phase = "verifying_native_step_limit"
            # The official task config owns the physical budget. A larger launch
            # cap would advertise phantom steps the native env never grants.
            if int(self.env.step_lim) != meta["max_episode_steps"]:
                raise ValueError(
                    f"max_episode_steps={meta['max_episode_steps']} differs from "
                    f"the official {meta['task']} step_lim={int(self.env.step_lim)}"
                )
            phase = "synchronizing_camera_apertures"
            self.camera_calibration_receipt = _synchronize_tiled_camera_apertures(
                self.env
            )
            phase = "refreshing_calibrated_camera_frames"
            self.camera_refresh_receipt = _refresh_calibrated_camera_frames(self.env)
            phase = "writing_launch_receipt"
            _write_launch_receipt(cfg, self)
            self._evaluation_ready = True
        except BaseException as error:
            _write_constructor_failure(cfg, phase, error)
            if hasattr(self, "venv"):
                self.close()
            raise

    def update_reset_state_ids(self, env_idx=None) -> None:
        """Agent episodes use the requested official layout ID."""
        self.reset_state_ids = torch.tensor([self.meta["layout"]], dtype=torch.long)

    def _sub_env(self, env_id: int = 0) -> Any:
        """Return the local runtime slot; RPC owns exactly one environment."""
        if env_id != 0:
            raise IndexError(f"invalid RoboDojo env_id: {env_id}")
        return self.venv.envs[env_id]

    @property
    def env(self) -> Any:
        return self._sub_env().env

    def get_native_obs(self) -> dict:
        """Record native RGB frames while preserving calibration and state."""
        return _obs_dict(self.env, self.recorder)

    def _capture_observation(self) -> dict:
        return self.observations.publish(
            public_observation(self.get_native_obs()), self.get_status()["step"]
        )

    def _observation(self) -> dict:
        if not self.observations.settled:
            if self.observations.initialized:
                raise RuntimeError(
                    "previous submitted action has no settled observation"
                )
            return self._capture_observation()
        return self.observations.compact()

    def get_env_meta(self) -> dict[str, Any]:
        """Return the meta info this server was launched with."""
        return dict(self.meta)

    def get_task_language(self) -> str:
        from reweave.benchmarks.robodojo.tasks import resolve_instruction

        return resolve_instruction(self.env)

    def get_camera_meta(
        self,
        camera_name: str,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        metadata = self._observation()["vision"]
        if camera_name not in metadata:
            raise ValueError(f"unknown camera: {camera_name!r}")
        cam = metadata[camera_name]
        native_height, native_width = cam["shape"][:2]
        if (height is not None and height != native_height) or (
            width is not None and width != native_width
        ):
            raise ValueError("RoboDojo camera metadata requires native resolution")
        return {
            "camera_name": camera_name,
            "height": native_height,
            "width": native_width,
            "intrinsic_matrix": cam["intrinsic_matrix"],
            "extrinsic_matrix": cam["extrinsic_matrix"],
        }

    def reset_episode(self) -> dict[str, Any]:
        """Attach to the scene initialized by this single-episode server."""
        if int(self.env.take_action_cnt[0]) != 0:
            raise ValueError("RoboDojo cannot reset after a physical action")
        error = self._environment_error()
        if error is not None:
            raise EnvironmentInvalid(error)
        return self._observation()

    def step_native(self, flat_action):
        """Apply one action and return ``(compact_obs, reward, done, info)``.

        Full RGB-D stays in the episode-owned snapshot service. Every native
        observation still feeds the three-camera recorder; only explicit
        retained boundaries create lossless sensor archives.

        ``flat_action`` is a 14-value policy vector or a native action dict
        (joint or ee keys). The action type is inferred from the keys; the client validates the
        four-field acknowledgement before settling its physical action.
        ``info.environment_error`` is null for a valid scene or the native
        invalidity reason after this acknowledged action. Invalidity ends the
        episode without assigning an official success or agent failure.
        ``info.server_timing_s`` records non-overlapping server wall-clock
        seconds: ``simulation_apply`` covers native action application,
        ``observation`` covers RGB-D/state acquisition, video writing and
        public projection, and ``settlement`` covers post-action terminal and
        status checks. RPC serialization and network transfer occur afterward
        and are excluded. These measurements are audit data, not observations.
        """
        flat_action = native_action(flat_action)
        action_type = _infer_action_type(flat_action)
        return self._apply_and_settle(
            lambda: self._sub_env(0).apply_action(
                flat_action, action_type, eval_fair=False
            )
        )

    def step_trajectory(self, arm, positions, velocities, gripper=0):
        def apply():
            substeps = _apply_trajectory_window(
                self.env, arm, positions, velocities, gripper
            )
            self._sub_env(0)._last_obs = None
            return {
                "trajectory_substeps": substeps,
                "trajectory_dt_s": float(self.env.robot_manager.dt),
            }

        return self._apply_and_settle(apply)

    def _apply_and_settle(self, apply):
        if self.observations.initialized and not self.observations.settled:
            raise RuntimeError("previous submitted action has no settled observation")
        if int(self.env.take_action_cnt[0]) >= self.meta["max_episode_steps"]:
            raise RuntimeError("Episode step budget exhausted")
        if self._episode_ended():
            raise RuntimeError("RoboDojo episode is already terminal")
        self.observations.invalidate()
        started = time.perf_counter()
        extra = apply() or {}
        applied = time.perf_counter()
        observation = self._capture_observation()
        observed = time.perf_counter()
        done = self._episode_ended()
        info = {
            "status": self.get_status(),
            "environment_error": self._environment_error(),
            **extra,
        }
        settled = time.perf_counter()
        info["server_timing_s"] = {
            "simulation_apply": applied - started,
            "observation": observed - applied,
            "settlement": settled - observed,
        }
        return (
            observation,
            0.0,
            done,
            info,
        )

    def get_obs(self) -> dict[str, Any]:
        return self._observation()

    def get_status(self) -> dict[str, Any]:
        """Acknowledge actions against the configured episode cap.

        The native task may terminate earlier. Its terminal acknowledgement
        stops the client independently; it is not an official success verdict.
        """
        return {
            "step": int(self.env.take_action_cnt[0]),
            "step_limit": self.meta["max_episode_steps"],
        }

    def _episode_ended(self) -> bool:
        return bool(
            self._environment_error() is not None
            or not self.env.success[0]
            or self.env.take_action_cnt[0] >= self.env.step_lim
            or self.env.end_flag[0]
            or self.is_success()
        )

    def plan_position_path(self, arm: str, xyz: list) -> dict[str, Any]:
        return _plan_position_path(self.env, arm, xyz)

    def plan_joint_path(self, arm: str, joints: list) -> dict[str, Any]:
        return _plan_joint_path(self.env, arm, joints)

    def solve_ik_pose(self, arm: str, pose: list) -> dict[str, Any]:
        return _solve_ik_pose(self.env, arm, pose)

    def plan_pose_path(
        self, arm: str, poses: list, linear_axis: str | None = None
    ) -> dict[str, Any]:
        return _plan_pose_path(self.env, arm, poses, linear_axis)

    def is_success(self) -> bool:
        error = self._environment_error()
        if error is not None:
            raise EnvironmentInvalid(error)
        return bool(self.env.is_success(env_idx=0))

    def _environment_error(self) -> str | None:
        if 0 in self.env.unstable_envs:
            return "RoboDojo native support-arm action left the environment unstable"
        return None

    def _finish_evaluation(self) -> None:
        """Persist host-only metrics once; never expose scores in observations."""
        path = getattr(self, "_metrics_path", None)
        if (
            getattr(self, "_evaluation_ready", False)
            and path is not None
            and not path.exists()
        ):
            try:
                metrics = terminal_metrics(self.env)
            except Exception as error:
                metrics = {
                    "valid": False,
                    "success": None,
                    "score": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            from reweave.utils.persistence import atomic_json

            atomic_json(
                path,
                {
                    **metrics,
                    "source": "RoboDojo.native.eval_env.run_eval",
                    "score_unit": "fraction",
                    "task": self.meta["task"],
                    "layout_set": self.meta["layout_set"],
                    "layout": self.meta["layout"],
                    "physical_steps": int(self.env.take_action_cnt[0]),
                    "service_instance_id": os.environ.get(
                        "REWEAVE_SERVICE_INSTANCE_ID"
                    ),
                },
            )

    def finish_recording(self) -> None:
        """Finalize evidence before acknowledging the close RPC."""
        try:
            self._finish_evaluation()
        finally:
            paths = self.recorder.close()
            if paths:
                logger.info("[robodojo-env] videos written: %s", paths)

    def close(self, clear_cache: bool = True) -> None:
        """Flush video before the bridge releases cameras and the Isaac app."""
        if self._closed:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Isaac teardown must run on the main thread")
        logger.info("[robodojo-env] shutdown begin")
        try:
            self.finish_recording()
        finally:
            super().close(clear_cache)
            logger.info("[robodojo-env] shutdown complete")
