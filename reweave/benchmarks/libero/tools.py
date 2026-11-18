# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""LIBERO + OpenPI tool implementation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from reweave.benchmarks.libero.cartesian_control import CartesianControl
from reweave.benchmarks.libero.env_client import LiberoEnvClient
from reweave.benchmarks.libero.evaluation import LIBERO_ACTIONS_PER_CHUNK
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.components.gripper_feedback import GripperSettling
from reweave.robots.components.object_geometry import measure_geometry
from reweave.robots.components.pi05_vla_client import Pi05VLAClient
from reweave.robots.components.sam3_client import Sam3Client
from reweave.robots.execution_trace import ExecutionTrace
from reweave.robots.physical_evidence import libero_features
from reweave.robots.vla_execution import (
    EvidenceScalar,
    VLAActionBuffer,
    VLAExecution,
)
from reweave.runtime.capability import EnvironmentServiceFailure, PhysicalOutcomeUnknown
from reweave.session import EnvState, StepRecord
from reweave.tools.toolkit import readonly
from reweave.utils.logging import get_logger
from reweave.utils.rpc.rpc_client import RpcTransportError

logger = get_logger("libero")

_LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def _normalize_xyz(xyz):
    """Coerce an LLM-supplied xyz into a length-3 list[float]."""
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(
            'xyz must be a JSON array of three numbers, e.g. "xyz":[-0.05,0,0.3]'
        )
    return [float(v) for v in xyz]


class LiberoPrimitives:
    """Wraps a single-env LIBERO-shaped env + VLA policy with primitive-
    level methods.

    ``pi0_pick`` and ``pi0_doubled`` override ``obs['task_descriptions']``
    with a sub-instruction. ``move_to`` and friends are scripted (no VLM
    call) and drive the underlying OSC controller directly.
    """

    def __init__(
        self,
        env: LiberoEnvClient,
        model: Pi05VLAClient,
        sam3_client: Sam3Client,
        check_cancelled: Callable[[], None],
        execution_trace: ExecutionTrace,
        active_execution: ActiveExecution,
        wait_steps: int = 0,
        graspgen_client: Any | None = None,
    ):
        self.env = env
        self.model = model
        self.model_budget = None
        self._sam3_client = sam3_client
        self._graspgen_client = graspgen_client
        self._check_cancelled = check_cancelled
        self._execution_trace = execution_trace
        self._active_execution = active_execution
        self._wait_steps = int(wait_steps)
        if self._wait_steps < 0:
            raise ValueError("LIBERO wait_steps must be non-negative")
        self._last_obs = None
        self._last_obs_states = np.asarray([], dtype=np.float32)
        self._last_obs_eef_pos = None
        self._last_obs_eef_z = None
        self._last_obs_gripper = None
        self._last_reward = 0.0
        self._critic_evidence: dict[str, EvidenceScalar] = {}
        # Per-env-step frame buffer for diagnostic video rendering.
        # Toggled via start_recording() / stop_recording().
        self._recording = False
        self._frames = []
        self._active_vla_buffer: VLAActionBuffer | None = None
        self._cartesian_control: CartesianControl | None = None

    def control_step_budget(self) -> dict[str, Any]:
        """Expose the exact LIBERO horizon and VLA-to-control-step cost."""
        return {
            "unit": "environment_control_step",
            "limit": self.env.max_episode_steps,
            "used": self.env.elapsed_steps(),
            "remaining": self.env.steps_remaining(),
            "operation_costs": {
                "pi0_pick": {
                    "upper_bound_formula": (
                        "max_chunks * (actions_per_chunk or "
                        f"{LIBERO_ACTIONS_PER_CHUNK})"
                    ),
                    "default_actions_per_chunk": LIBERO_ACTIONS_PER_CHUNK,
                },
                "pi0_doubled": {
                    "upper_bound_formula": (
                        "max_chunks * (actions_per_chunk or "
                        f"{LIBERO_ACTIONS_PER_CHUNK})"
                    ),
                    "default_actions_per_chunk": LIBERO_ACTIONS_PER_CHUNK,
                },
                "scripted_motion": {
                    "upper_bound_formula": "the capability's max_steps argument",
                },
            },
        }

    def start_recording(self):
        self._recording = True
        self._frames = []

    def record_frame(self, obs):
        """Append one agentview frame extracted from ``obs`` to the buffer."""
        self._frames.append(np.ascontiguousarray(np.asarray(obs["main_images"])))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    def set_obs(self, obs):
        self._last_obs = obs
        states_arr = np.asarray(obs["states"], dtype=np.float32).reshape(-1)
        if states_arr.size < 8 or not np.isfinite(states_arr).all():
            raise ValueError("LIBERO telemetry requires at least eight finite states")
        self._last_obs_states = states_arr.copy()
        self._last_obs_eef_pos = np.asarray(states_arr[:3], dtype=np.float32)
        self._last_obs_eef_z = float(self._last_obs_eef_pos[2])
        # robosuite 2f85: qpos[6] in [~0, ~0.04], qpos[7] in [~-0.04, ~0].
        # Use |qpos[6]| + |qpos[7]| ≈ finger separation proxy.
        # When open ≈ 0.08; when closed ≈ 0.
        gp = np.asarray(states_arr[6:8], dtype=np.float32)
        self._last_obs_gripper = float(abs(gp[0]) + abs(gp[1]))

    def reset(self):
        owns_execution = self._active_execution.snapshot.phase == "idle"
        if owns_execution:
            self._active_execution.begin("reset", {"wait_steps": self._wait_steps})
        try:
            self._cartesian_control = None
            obs, info = self.env.reset()
            self.set_obs(obs)
            try:
                self._set_critic_evidence(info)
                self._execution_trace.record_reset_state(
                    physical_features=libero_features(
                        {
                            **self.critic_evidence(),
                            **{
                                f"robot.eef.{axis}": float(value)
                                for axis, value in zip(
                                    "xyz", self._last_obs_eef_pos, strict=True
                                )
                            },
                            "robot.gripper.opening": float(self._last_obs_gripper),
                        }
                    )
                )
            except (ValueError, RuntimeError) as error:
                self._execution_trace.record_reset_state(
                    physical_features={}, error=str(error)
                )
            final_info = info
            for index in range(self._wait_steps):
                if self.env.terminated or self.env.truncated:
                    break
                final_info = self._scripted_control_step(
                    _LIBERO_DUMMY_ACTION, action_index=index
                )
        except BaseException:
            if owns_execution:
                self._active_execution.finish("failed")
            raise
        else:
            if owns_execution:
                self._active_execution.finish("completed")
        return self._last_obs, final_info

    def _set_critic_evidence(self, info: Any) -> None:
        evidence = info.get("critic_evidence") if isinstance(info, dict) else None
        if not isinstance(evidence, dict):
            raise RuntimeError("LIBERO env omitted required critic evidence")
        required = {
            "privileged.available",
            "privileged.task.semantic_available",
            "privileged.task.goal.predicate_count",
            "privileged.task.goal.evaluable_count",
            "privileged.task.goal.satisfied_count",
            "privileged.task.goal.predicate_fraction_available",
            "privileged.contact.robot.count",
            "privileged.contact.gripper.count",
            "privileged.contact.force_available",
            "privileged.contact.max_normal_force_n",
            "privileged.joint.count",
        }
        missing = required - set(evidence)
        if missing:
            raise RuntimeError(
                f"LIBERO critic evidence is missing features: {sorted(missing)}"
            )
        parsed: dict[str, EvidenceScalar] = {}
        for raw_feature, raw_value in evidence.items():
            feature = str(raw_feature).strip()
            if not feature.startswith("privileged."):
                raise ValueError(
                    f"LIBERO critic feature is outside its private namespace: {feature!r}"
                )
            if isinstance(raw_value, (np.bool_, bool)):
                value: EvidenceScalar = bool(raw_value)
            elif isinstance(raw_value, (np.integer, int)):
                value = int(raw_value)
            elif isinstance(raw_value, (np.floating, float)):
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"LIBERO critic feature {feature!r} must be finite"
                    )
            elif isinstance(raw_value, str):
                value = raw_value
            else:
                raise TypeError(
                    f"LIBERO critic feature {feature!r} has unsupported value "
                    f"type {type(raw_value).__name__}"
                )
            parsed[feature] = value
        self._critic_evidence = parsed

    def critic_evidence(self) -> dict[str, EvidenceScalar]:
        """Return simulator semantics reserved for the critic-only plane."""
        return dict(self._critic_evidence)

    def gripper_opening(self) -> float:
        """Return the public finger-separation proprioceptive signal in metres."""
        return float(self._last_obs_gripper)

    def _step_env(
        self,
        action,
        *,
        on_applied: Callable[[], int],
        on_unknown: Callable[[Exception], None],
    ) -> dict[str, Any]:
        """Commit acknowledged physical work before fallible local processing."""
        self._check_cancelled()
        previous_eef = self._last_obs_eef_pos.copy()
        physical_started = time.monotonic()
        try:
            obs, reward, _t, _tr, info = self.env.step(action)
        except Exception as error:
            if (
                isinstance(error, RpcTransportError)
                and error.recovery_state == "not_submitted"
            ):
                # The selected action still belongs to the unexecuted suffix.
                # Toolkit closes that suffix before ending the episode; no ACK
                # or unknown-action event may consume this action's identity.
                raise EnvironmentServiceFailure(
                    f"physical action was not submitted: {error}", retryable=True
                ) from error
            on_unknown(error)
            raise PhysicalOutcomeUnknown(
                f"physical action outcome is unknown: {error}"
            ) from error
        action_wall_time_s = time.monotonic() - physical_started
        step = on_applied()
        try:
            self._last_reward = float(np.asarray(reward).max())
            self.set_obs(obs)
            self._set_critic_evidence(info)
            evidence = self._action_evidence(action, previous_eef=previous_eef)
        except Exception as error:
            self._execution_trace.record_telemetry(
                step,
                error=f"{type(error).__name__}: {error}",
                action_wall_time_s=action_wall_time_s,
            )
            raise
        self._execution_trace.record_telemetry(
            step,
            physical_features=libero_features(evidence),
            action_wall_time_s=action_wall_time_s,
            public_evidence={
                key: value
                for key, value in evidence.items()
                if not key.startswith("privileged.")
            },
            critic_private_evidence={
                key: value
                for key, value in evidence.items()
                if key.startswith("privileged.")
            },
        )
        if self._recording:
            self.record_frame(obs)
        return info

    def _action_evidence(
        self,
        action: np.ndarray,
        *,
        previous_eef: np.ndarray,
    ) -> dict[str, EvidenceScalar]:
        """Build the complete public and critic-private action state."""
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size != 7 or not np.isfinite(command).all():
            raise ValueError("LIBERO action must contain seven finite values")
        current_eef = np.asarray(self._last_obs_eef_pos, dtype=np.float64)
        previous = np.asarray(previous_eef, dtype=np.float64).reshape(3)
        realized = current_eef - previous
        command_translation = command[:3]
        command_norm = float(np.linalg.norm(command_translation))
        realized_norm = float(np.linalg.norm(realized))
        directional = command_norm > 1e-9 and realized_norm > 1e-9
        evidence: dict[str, EvidenceScalar] = {
            "episode.step_index": self.env.elapsed_steps(),
            "episode.reward": self._last_reward,
            "episode.terminated": bool(self.env.terminated),
            "episode.truncated": bool(self.env.truncated),
            "robot.eef.x": float(current_eef[0]),
            "robot.eef.y": float(current_eef[1]),
            "robot.eef.z": float(current_eef[2]),
            "robot.gripper.opening": float(self._last_obs_gripper),
            "robot.eef.delta_available": True,
            "robot.eef.delta.x": float(realized[0]),
            "robot.eef.delta.y": float(realized[1]),
            "robot.eef.delta.z": float(realized[2]),
            "robot.eef.motion_m": realized_norm,
            "command.available": True,
            "command.translation.x": float(command[0]),
            "command.translation.y": float(command[1]),
            "command.translation.z": float(command[2]),
            "command.translation.norm": command_norm,
            "command.rotation.x": float(command[3]),
            "command.rotation.y": float(command[4]),
            "command.rotation.z": float(command[5]),
            "command.rotation.norm": float(np.linalg.norm(command[3:6])),
            "command.gripper": float(command[6]),
            "command.realization.direction_available": directional,
            "command.realization.direction_cosine": (
                float(
                    np.dot(command_translation, realized)
                    / (command_norm * realized_norm)
                )
                if directional
                else 0.0
            ),
            "command.realization.eef_motion_m": realized_norm,
            **self.critic_evidence(),
        }
        evidence.update(
            {
                f"robot.state.{index}": float(value)
                for index, value in enumerate(self._last_obs_states)
            }
        )
        return evidence

    def _scripted_control_step(
        self, action: np.ndarray, *, action_index: int
    ) -> dict[str, Any]:
        """Execute and record one action of a scripted physical tool."""
        snapshot = self._active_execution.snapshot
        if snapshot.phase != "running" or snapshot.execution_id is None:
            raise RuntimeError(
                "scripted action recording requires a running active execution"
            )
        return self._step_env(
            action,
            on_unknown=lambda error: (
                self._execution_trace.record_action_outcome_unknown(
                    execution_ref=snapshot.execution_id,
                    error=error,
                    requested_action=action,
                    action_index=action_index,
                )
            ),
            on_applied=lambda: self._execution_trace.record_scripted_control_step(
                execution_ref=snapshot.execution_id,
                action_index=action_index,
                action=action,
                terminated=self.env.terminated,
                truncated=self.env.truncated,
            ),
        )

    def reset_episode(self, reason: str = "") -> dict:
        """Restart an episode and return an explore-tool-compatible result."""
        self.reset()
        return {
            "action": "reset",
            "reason": reason,
            "libero_terminated": self.env.terminated or self.env.truncated,
        }

    def _vlm_chunk(
        self,
        instruction: str,
    ) -> VLAExecution:
        """Execute one planner-controlled VLA receding-horizon prefix."""
        self._check_cancelled()
        if self.env.terminated or self.env.truncated:
            return VLAExecution(executed_action_steps=0)
        original_task = self._last_obs.get("task_descriptions")
        try:
            self._last_obs["task_descriptions"] = instruction
            self._last_obs.setdefault("extra_view_images", None)

            # Execute the fixed policy horizon with a record per physical action.

            with (
                self.model_budget.request("vla")
                if self.model_budget is not None
                else nullcontext()
            ):
                prediction = self.model.predict_action_batch(
                    self._last_obs, mode="eval"
                )
            self._active_execution.record_forward(
                prediction,
                instruction=instruction,
            )
            buffer = VLAActionBuffer(prediction)
            if self._active_vla_buffer is not None:
                raise RuntimeError("LIBERO already owns an active VLA buffer")
            self._active_vla_buffer = buffer
            executed_action_steps = 0
            while not (self.env.terminated or self.env.truncated):
                action = buffer.pop_next()
                if action is None:
                    break

                def commit_action():
                    buffer.release_selected(action)
                    return self._execution_trace.record_control_step(
                        action,
                        terminated=self.env.terminated,
                        truncated=self.env.truncated,
                    )

                def uncertain_action(error):
                    buffer.release_selected(action)
                    self._execution_trace.record_action_outcome_unknown(
                        execution_ref=self._active_execution.snapshot.execution_id,
                        action=action,
                        error=error,
                    )

                self._step_env(
                    action.action, on_applied=commit_action, on_unknown=uncertain_action
                )
                executed_action_steps += 1

                if self.env.terminated or self.env.truncated:
                    traced_discarded = (
                        self._execution_trace.record_action_suffix_discarded(
                            action,
                            reason="episode_end",
                        )
                    )
                    buffered_discarded = buffer.discard_remaining()
                    if buffered_discarded != traced_discarded:
                        raise RuntimeError(
                            "LIBERO action buffer and execution trace disagree on "
                            f"episode-end suffix: {buffered_discarded} != "
                            f"{traced_discarded}"
                        )
                    self._active_vla_buffer = None
                    break

            self._active_vla_buffer = None
            return VLAExecution(
                executed_action_steps=executed_action_steps,
            )
        finally:
            if original_task is not None:
                self._last_obs["task_descriptions"] = original_task

    def discard_active_vla_suffix(self) -> int:
        """Discard a locally owned suffix after tool failure or cancellation."""
        buffer = self._active_vla_buffer
        if buffer is None:
            return 0
        discarded = buffer.discard_remaining()
        self._active_vla_buffer = None
        return discarded

    def leftover_vla_suffix_steps(self) -> int:
        """Return leftover actions the host failed to dump on interrupt."""
        buffer = self._active_vla_buffer
        return 0 if buffer is None else buffer.remaining_action_steps

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 24,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
        gripper_open_thresh: float = 0.0,
        descent_thresh: float = 0.10,
    ) -> dict:
        """Closed-loop Pi0.5 pick driven by ``prompt`` as the VLA instruction.

        The local heuristic requires descent by ``descent_thresh``, then
        ascent by ``lift_thresh``, with finger separation in
        [``gripper_open_thresh``, ``gripper_closed_thresh``). It does not
        establish object possession. Environment termination takes priority.
        """
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or max_chunks < 1
        ):
            raise ValueError("max_chunks must be a positive integer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty instruction")
        if (
            not np.isfinite(
                [
                    lift_thresh,
                    gripper_closed_thresh,
                    gripper_open_thresh,
                    descent_thresh,
                ]
            ).all()
            or lift_thresh <= 0
            or descent_thresh < 0
            or not 0 <= gripper_open_thresh < gripper_closed_thresh
        ):
            raise ValueError(
                "pick thresholds must be finite, with lift_thresh > 0, "
                "descent_thresh >= 0 and 0 <= gripper_open_thresh < gripper_closed_thresh"
            )
        instr = prompt
        start_z = self._last_obs_eef_z
        peak_z = start_z
        min_z = start_z
        # Track ascent AFTER min_z has been observed — descent then re-ascent
        # is the actual "lift" signal, distinct from raw |peak - min| which
        # also fires at the BOTTOM of the descent.
        post_min_peak_z = start_z
        min_grip = self._last_obs_gripper
        last_grip = min_grip
        descent_done = False
        success = bool(self.env.terminated)
        chunks_used = 0

        for c in range(max_chunks):
            if self.env.terminated or self.env.truncated:
                break
            self._vlm_chunk(
                instr,
            )
            chunks_used = c + 1
            z = self._last_obs_eef_z
            grip = self._last_obs_gripper
            peak_z = max(peak_z, z)
            if z < min_z:
                min_z = z
                post_min_peak_z = z  # reset after a new deeper min
            else:
                post_min_peak_z = max(post_min_peak_z, z)
            if (start_z - min_z) >= descent_thresh:
                descent_done = True
            min_grip = min(min_grip, grip)
            last_grip = grip
            ascended = (post_min_peak_z - min_z) >= lift_thresh
            closed = gripper_open_thresh <= grip < gripper_closed_thresh
            if self.env.terminated or self.env.truncated:
                success = self.env.terminated
                break
            if descent_done and ascended and closed:
                success = True
                break

        result = {
            "name": "pick",
            "instruction": instr,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": post_min_peak_z - min_z,  # actual post-descent ascent
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "diagnostics": {
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
                "gripper_open_thresh": gripper_open_thresh,
                "descent_thresh": descent_thresh,
            },
        }
        return result

    @readonly
    def generate_grasps(
        self,
        prompt: str,
        camera: str = "agentview",
        step: int = -1,
        min_score: float = 0.2,
        max_candidates: int = 5,
        *,
        state: EnvState,
    ) -> dict:
        """Generate Panda TCP candidates from one recorded RGB-D observation."""
        if self._graspgen_client is None:
            raise RuntimeError("GraspGen-X is not configured")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        prompt = prompt.strip()
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise ValueError("max_candidates must be an integer between 1 and 32")
        if not 1 <= max_candidates <= 32:
            raise ValueError("max_candidates must be between 1 and 32")
        record = state.get(step)
        if camera not in ("agentview", "wrist"):
            raise ValueError(f"unknown grasp camera: {camera}")
        image_name = f"{camera}.png"
        depth_name = f"{camera}_depth.npz"
        metadata_name = f"{camera}_metadata.json"
        missing = [
            name
            for name in (image_name, depth_name, metadata_name)
            if name not in record.artifacts
            or not state.exists(name, step=record.step_idx)
        ]
        if missing:
            raise RuntimeError(f"recorded RGB-D grasp artifacts missing: {missing}")
        depth = np.asarray(state.load(depth_name, step=record.step_idx))
        metadata = state.load(metadata_name, step=record.step_idx)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"recorded depth must be 2-D, got {depth.shape}")
        _validate_grasp_camera_metadata(metadata)
        world = _world_from_depth(depth, metadata)
        with (
            self.model_budget.request("sam3")
            if self.model_budget is not None
            else nullcontext()
        ):
            segment = self._sam3_client.segment(
                state.load_bytes(image_name, step=record.step_idx),
                text_prompt=prompt,
                min_score=min_score,
            )
        if not segment.found or segment.mask is None:
            raise RuntimeError(segment.reason or "SAM3 found no target mask")
        mask = np.asarray(segment.mask, dtype=bool)
        if world.ndim != 3 or world.shape[:2] != mask.shape or world.shape[2] != 3:
            raise ValueError(
                f"mask/world shape mismatch: mask={mask.shape}, world={world.shape}"
            )
        valid_depth = np.isfinite(depth) & (depth > 0)
        points = np.asarray(world[mask & valid_depth], dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if not points.size:
            raise ValueError("target mask contains no valid measured depth points")
        grasp_index = _next_grasp_index(record)
        mask_name = f"grasp_mask_{grasp_index:02d}.npz"
        overlay_name = f"grasp_mask_overlay_{grasp_index:02d}.png"
        metadata_name_out = f"grasp_mask_{grasp_index:02d}.json"
        saved_mask = state.save(mask_name, mask, step=record.step_idx)
        overlay = _make_segment_overlay(
            state.load(image_name, step=record.step_idx), mask
        )
        saved_overlay = (
            state.save(overlay_name, overlay, step=record.step_idx)
            if overlay is not None
            else None
        )
        saved_metadata = state.save(
            metadata_name_out,
            {
                "source_step": record.step_idx,
                "camera": camera,
                "image_artifact": image_name,
                "depth_artifact": depth_name,
                "camera_metadata_artifact": metadata_name,
                "mask_source": "sam3_text_prompt",
                "mask_prompt": prompt,
                "mask_score": segment.score,
                "mask_shape": list(mask.shape),
                "mask_pixels": int(mask.sum()),
            },
            step=record.step_idx,
        )
        if not all((saved_mask, saved_overlay, saved_metadata)):
            raise RuntimeError("failed to persist GraspGen-X source mask evidence")
        with (
            self.model_budget.request("graspgen")
            if self.model_budget is not None
            else nullcontext()
        ):
            result = self._graspgen_client.generate_grasps(
                points,
                arm="right",
                frame="libero_world",
                max_candidates=max_candidates,
            )
        return {
            **result,
            "provenance": {
                "observation_step": record.step_idx,
                "camera": camera,
                "image_artifact": image_name,
                "depth_artifact": depth_name,
                "camera_metadata_artifact": metadata_name,
                "mask_source": "sam3_text_prompt",
                "mask_prompt": prompt,
                "mask_score": segment.score,
                "mask_artifact": saved_mask,
                "mask_overlay_artifact": saved_overlay,
                "mask_metadata_artifact": saved_metadata,
            },
        }

    def pi0_doubled(
        self,
        prompt: str,
        *,
        max_chunks: int = 20,
    ) -> dict:
        """Closed-loop Pi0.5 contact skill.

        Intended for non-pick contact interactions such as turning knobs,
        toggling stoves, or short pushes. Success is the official LIBERO
        termination predicate, not a private object-pose oracle.
        """
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or max_chunks < 1
        ):
            raise ValueError("max_chunks must be a positive integer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty instruction")
        instr = prompt
        task_success = bool(self.env.terminated)
        chunks_used = 0

        for c in range(max_chunks):
            if self.env.terminated or self.env.truncated:
                break
            self._vlm_chunk(
                instr,
            )
            chunks_used = c + 1
            if self.env.terminated or self.env.truncated:
                task_success = self.env.terminated
                break

        result = {
            "name": "pi0_doubled",
            "instruction": instr,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "diagnostics": {
                "mode": "contact_skill_success_by_termination",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
        }
        return result

    @staticmethod
    def _orientation_angles(quaternion) -> tuple[float, float]:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_quat(quaternion).as_matrix()
        return (
            float(np.arctan2(rotation[1, 2], -rotation[2, 2])),
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        )

    def _servo_cartesian(
        self,
        name,
        xyz,
        *,
        target_pitch=None,
        target_yaw=None,
        target_quaternion_xyzw=None,
        gripper=-1.0,
        step_clip=0.025,
        pitch_step=0.10,
        yaw_step=0.10,
        quaternion_step=0.08,
        tol=0.012,
        ori_tol=0.02,
        max_steps=80,
    ) -> dict:
        target = np.asarray(_normalize_xyz(xyz), dtype=np.float64)
        numeric = [
            *target,
            gripper,
            step_clip,
            pitch_step,
            yaw_step,
            quaternion_step,
            tol,
            ori_tol,
        ]
        numeric.extend(
            value for value in (target_pitch, target_yaw) if value is not None
        )
        if target_quaternion_xyzw is not None and (
            target_pitch is not None or target_yaw is not None
        ):
            raise ValueError(
                "target_quaternion_xyzw is mutually exclusive with pitch/yaw targets"
            )
        target_rotation = None
        if target_quaternion_xyzw is not None:
            from scipy.spatial.transform import Rotation

            quaternion = np.asarray(target_quaternion_xyzw, dtype=np.float64)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
                raise ValueError(
                    "target_quaternion_xyzw must contain four finite values"
                )
            norm = float(np.linalg.norm(quaternion))
            if not np.isclose(norm, 1.0, atol=1e-4):
                raise ValueError("target_quaternion_xyzw must be unit length")
            target_rotation = Rotation.from_quat(quaternion).as_matrix()
        if (
            not np.isfinite(numeric).all()
            or min(
                step_clip,
                pitch_step,
                yaw_step,
                quaternion_step,
                tol,
                ori_tol,
            )
            <= 0
        ):
            raise ValueError(
                "servo targets must be finite and tolerances/clips positive"
            )
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer")
        if not -1 <= gripper <= 1:
            raise ValueError("gripper must be within [-1, 1]")
        if self._cartesian_control is None:
            self._cartesian_control = CartesianControl.from_spec(
                self.env.get_cartesian_control_spec()
            )
        control = self._cartesian_control
        steps_used = 0
        positions = [np.asarray(self._last_obs_eef_pos, dtype=np.float64).copy()]

        def feedback():
            from scipy.spatial.transform import Rotation

            current_quaternion = np.asarray(
                self.env.raw_obs()["robot0_eef_quat"], dtype=np.float64
            )
            pitch, yaw = self._orientation_angles(current_quaternion)
            diff = target - self._last_obs_eef_pos
            p_err = (
                0.0
                if target_pitch is None
                else float((target_pitch - pitch + np.pi) % (2 * np.pi) - np.pi)
            )
            y_err = (
                0.0
                if target_yaw is None
                else float((target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            )
            rotation_error = (
                np.zeros(3, dtype=np.float64)
                if target_rotation is None
                else Rotation.from_matrix(
                    target_rotation
                    @ Rotation.from_quat(current_quaternion).as_matrix().T
                ).as_rotvec()
            )
            orientation_error = float(np.linalg.norm(rotation_error))
            reached = (
                np.linalg.norm(diff) < tol
                and abs(p_err) < ori_tol
                and abs(y_err) < ori_tol
                and orientation_error < ori_tol
            )
            return (
                diff,
                pitch,
                yaw,
                p_err,
                y_err,
                rotation_error,
                orientation_error,
                bool(reached),
            )

        while steps_used < max_steps:
            diff, _, _, p_err, y_err, rotation_error, _, reached = feedback()
            if reached or self.env.terminated or self.env.truncated:
                break
            if target_rotation is None:
                rotation_command = np.asarray(
                    [
                        np.clip(p_err, -pitch_step, pitch_step),
                        0.0,
                        np.clip(y_err, -yaw_step, yaw_step),
                    ]
                )
            else:
                rotation_norm = float(np.linalg.norm(rotation_error))
                rotation_command = (
                    rotation_error
                    if rotation_norm <= quaternion_step
                    else rotation_error * (quaternion_step / rotation_norm)
                )
            action = control.action(
                np.clip(diff, -step_clip, step_clip),
                rotation_command,
                gripper,
            )
            self._scripted_control_step(action, action_index=steps_used)
            steps_used += 1
            positions.append(
                np.asarray(self._last_obs_eef_pos, dtype=np.float64).copy()
            )
        (
            diff,
            pitch,
            yaw,
            p_err,
            y_err,
            rotation_error,
            orientation_error,
            reached,
        ) = feedback()
        reason = (
            "terminated"
            if self.env.terminated
            else "truncated"
            if self.env.truncated
            else "target_reached"
            if reached
            else "step_budget_exhausted"
        )
        motion_window = positions[-21:]
        result = {
            "name": name,
            "target_xyz": target.tolist(),
            "final_eef_pos": [round(float(x), 4) for x in self._last_obs_eef_pos],
            "final_dist_m": round(float(np.linalg.norm(diff)), 4),
            "final_pitch": round(pitch, 4),
            "final_yaw": round(yaw, 4),
            "control_spec": control.to_spec(),
            "target_reached": reached,
            "stop_reason": reason,
            "steps_used": steps_used,
            "max_steps": max_steps,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            # Settled public EEF measurements over the last <=20 acknowledged
            # steps. These describe response, not a cause or a recovery policy.
            "motion_response": {
                "steps": min(20, steps_used),
                "start_eef_pos": motion_window[0].tolist(),
                "end_eef_pos": positions[-1].tolist(),
                "path_length_m": float(
                    sum(
                        np.linalg.norm(b - a)
                        for a, b in zip(motion_window[:-1], motion_window[1:])
                    )
                ),
                "target_distance_reduction_m": float(
                    np.linalg.norm(target - motion_window[0]) - np.linalg.norm(diff)
                ),
            },
        }
        if target_pitch is not None:
            result.update(
                target_pitch=float(target_pitch), final_pitch_error=round(p_err, 4)
            )
        if target_yaw is not None:
            result.update(target_yaw=float(target_yaw), final_yaw_error=round(y_err, 4))
        if target_quaternion_xyzw is not None:
            result.update(
                target_quaternion_xyzw=np.asarray(
                    target_quaternion_xyzw, dtype=np.float64
                ).tolist(),
                final_orientation_error_rad=round(orientation_error, 4),
                final_orientation_error_rotvec_world=np.asarray(
                    rotation_error, dtype=np.float64
                ).tolist(),
            )
        return result

    def move_to(
        self,
        xyz,
        *,
        max_steps: int = 80,
        gripper: float = -1.0,
        step_clip: float = 0.025,
        tol: float = 0.012,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
        ori_tol: float = 0.02,
    ) -> dict:
        """Servo world XYZ and optional yaw using the live OSC mapping.

        Gripper is driven only during executed motion steps. To actuate it at
        an already reached pose, use set_gripper explicitly.
        """
        return self._servo_cartesian(
            "move_to",
            xyz,
            max_steps=max_steps,
            gripper=gripper,
            step_clip=step_clip,
            tol=tol,
            target_yaw=target_yaw,
            yaw_step=yaw_step_clip,
            ori_tol=ori_tol,
        )

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Servo world yaw while maintaining the initial EEF position."""
        if (target_yaw is None) == (delta_yaw is None):
            raise ValueError("provide exactly one of target_yaw or delta_yaw")
        _, start = self._orientation_angles(self.env.raw_obs()["robot0_eef_quat"])
        target = start + float(delta_yaw) if target_yaw is None else target_yaw
        result = self._servo_cartesian(
            "rotate_wrist",
            self._last_obs_eef_pos.tolist(),
            target_yaw=target,
            gripper=gripper,
            max_steps=max_steps,
            ori_tol=tol,
            yaw_step=step_clip,
        )
        result.update(start_yaw=round(start, 4), final_err=result["final_yaw_error"])
        return result

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """Servo pitch=atan2(EEF-z.y, -EEF-z.z), using world-X rotation."""
        if (target_pitch is None) == (delta_pitch is None):
            raise ValueError("provide exactly one of target_pitch or delta_pitch")
        start, _ = self._orientation_angles(self.env.raw_obs()["robot0_eef_quat"])
        target = start + float(delta_pitch) if target_pitch is None else target_pitch
        result = self._servo_cartesian(
            "rotate_pitch",
            self._last_obs_eef_pos.tolist(),
            target_pitch=target,
            gripper=gripper,
            max_steps=max_steps,
            ori_tol=tol,
            pitch_step=step_clip,
        )
        result.update(
            start_pitch=round(start, 4), final_err=result["final_pitch_error"]
        )
        return result

    def move_pose(
        self,
        xyz,
        *,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        target_quaternion_xyzw: list[float] | None = None,
        gripper: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        quaternion_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """Servo world position and both orientation goals in one physical loop."""
        return self._servo_cartesian(
            "move_pose",
            xyz,
            target_pitch=target_pitch,
            target_yaw=target_yaw,
            target_quaternion_xyzw=target_quaternion_xyzw,
            gripper=gripper,
            step_clip=step_clip,
            pitch_step=pitch_step,
            yaw_step=yaw_step,
            quaternion_step=quaternion_step,
            tol=tol,
            ori_tol=ori_tol,
            max_steps=max_steps,
        )

    def release(
        self,
        *,
        max_steps: int = 20,
        state: EnvState,
    ) -> dict:
        """Open gripper for ``max_steps`` env steps while keeping eef in place."""
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer")
        start_grip = self._last_obs_gripper
        peak_grip = start_grip
        steps_used = 0
        for step in range(max_steps):
            if self.env.terminated or self.env.truncated:
                break
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # open
            self._scripted_control_step(action, action_index=step)
            steps_used += 1
            peak_grip = max(peak_grip, self._last_obs_gripper)
            if self.env.terminated or self.env.truncated:
                break
        result = {
            "name": "release",
            "steps_used": steps_used,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._last_obs_gripper, 4),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }
        return result

    def set_gripper(
        self,
        *,
        gripper: float = -1.0,
        steps: int = 5,
        settle: bool = False,
    ) -> dict:
        """Hold the current EEF pose and drive ``gripper`` for ``steps`` env steps."""
        g = float(gripper)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not np.isfinite(g) or not -1 <= g <= 1:
            raise ValueError("gripper must be within [-1, 1]")
        if type(settle) is not bool:
            raise ValueError("settle must be boolean")
        monitor = GripperSettling(
            getattr(self, "_last_obs_gripper", None), tolerance=0.0004
        )
        n = steps
        steps_used = 0
        for step in range(n):
            if self.env.terminated or self.env.truncated:
                break
            action = np.zeros(7, dtype=np.float32)
            action[6] = g
            self._scripted_control_step(action, action_index=step)
            steps_used += 1
            stable = monitor.observe(getattr(self, "_last_obs_gripper", None))
            if self.env.terminated or self.env.truncated:
                break
            if settle and stable:
                break
        result = {
            "name": "set_gripper",
            "gripper": g,
            "steps": n,
            "steps_used": steps_used,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "stop_reason": "environment_terminal"
            if self.env.terminated
            else "episode_budget_exhausted"
            if self.env.truncated
            else "opening_stable"
            if settle and monitor.stable
            else "max_steps",
            **monitor.feedback(unit="metres"),
        }
        return result

    # ---- introspection helpers (for LLM-in-the-loop) ----

    @readonly
    def segment(
        self,
        prompt: str = "",
        camera: str = "agentview",
        step: int = -1,
        point: list[int] | None = None,
        min_score: float = 0.2,
        support_z: float | None = None,
        *,
        state: EnvState,
    ) -> dict:
        """Call SAM3 on an existing image artifact without advancing the env.

        This tool deliberately does not render camera views or create wrist/high-res
        artifacts. Errors are structured so the agent can continue with image
        inspection and ``back_project``.
        """
        try:
            record = state.get(step)
        except Exception as exc:
            return {"error": f"state step not available: {exc}"}
        nn = record.step_idx

        camera = camera or "agentview"
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        if has_prompt == has_point:
            return {"error": "segment needs exactly one of prompt or point"}
        try:
            image_name, world_name, artifact_pairs = _select_segment_artifacts(
                state, record, camera
            )
        except ValueError as e:
            return {"error": str(e)}
        if image_name is None or world_name is None:
            return {
                "error": "complete segment artifacts not found",
                "step": nn,
                "camera": camera,
                "checked_artifacts": [
                    name
                    for image, world in artifact_pairs
                    for name in (image, world)
                    if name
                ],
            }

        try:
            with (
                self.model_budget.request("sam3")
                if self.model_budget is not None
                else nullcontext()
            ):
                data = self._sam3_client.segment(
                    state.load_bytes(image_name, step=nn),
                    text_prompt=prompt if has_prompt else None,
                    point=point,
                    min_score=min_score,
                )
        except ValueError as e:
            return {
                "error": str(e),
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
            }
        except Exception as e:
            return {
                "error": f"segmentation service call failed: {e}",
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
                "fallback": "Use manual visual localization and back_project.",
            }

        segment_index = _next_segment_index(record)
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            try:
                world_map = state.load(world_name, step=nn)
            except Exception as exc:
                world_result = {
                    "world_xyz": None,
                    "world_error": f"world map artifact not available: {exc}",
                    "expected_world_artifact": world_name,
                }
            else:
                world_result = _mask_to_world(mask, world_map)
                world_result["world_artifact"] = world_name
                if world_result.get("world_xyz") is not None:
                    points = np.asarray(world_map)[np.asarray(mask, dtype=bool)]
                    points = points[
                        np.isfinite(points).all(axis=1)
                        & (np.abs(points).sum(axis=1) > 1e-6)
                    ]
                    world_result.update(measure_geometry(points, support_z=support_z))
            overlay = _make_segment_overlay(state.load(image_name, step=nn), mask)
            if overlay is not None and state.save(
                overlay_name,
                overlay,
                step=nn,
            ):
                saved_overlay = overlay_name
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_artifact": image_name,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(world_result)
        saved_segment = state.save(
            segment_name,
            segment_blob,
            step=nn,
        )

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_artifact": image_name,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
            **{
                key: segment_blob[key]
                for key in (
                    "geometry",
                    "geometry_error",
                    "support_z",
                    "measurement_kind",
                    "input_points",
                    "valid_points",
                )
                if key in segment_blob
            },
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        else:
            result["segment_artifact"] = saved_segment
        if saved_segment is not None and "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            result["overlay_note"] = (
                "Red pixels mark the predicted mask, not the original object color. Confirm identity against image_artifact."
            )
            result["_image_labels"] = {
                "_image_bytes": f"SAM mask overlay; synthetic red annotation; camera={camera}, step={nn}; original={image_name}"
            }
            result["_image_bytes"] = state.load_bytes(saved_overlay, step=nn)
        return result


def _is_primitive_action(name: object) -> bool:
    """Whether ``name`` is a state-advancing LIBERO primitive.

    A primitive is any non-read-only method on :class:`LiberoPrimitives`;
    read-only tools and non-strings read as ``False``.
    """
    if not isinstance(name, str):
        return False
    method = getattr(LiberoPrimitives, name, None)
    return method is not None and not bool(getattr(method, "_readonly", False))


def write_recipe_from_states(
    state: EnvState, recipe_tag: str, *, output_dir: Path | str
) -> str:
    """Find a command sequence that gets ``terminated=True``.

    Export non-error LIBERO primitive commands and successful segment calls.
    """
    records = state.records()
    last_reset = max(
        (
            record.step_idx
            for record in records
            if (
                (record.command or {}).get("action") == "reset"
                and not (isinstance(record.result, dict) and record.result.get("error"))
            )
        ),
        default=-1,
    )
    command_events = []
    for record in records:
        if record.step_idx <= last_reset:
            continue
        command = record.command
        result = record.result
        if (
            command is not None
            and _is_primitive_action(command.get("action"))
            and not (isinstance(result, dict) and result.get("error"))
        ):
            command_events.append(((record.step_idx, -1), command))

        for name in sorted(record.artifacts):
            if not (name.startswith("segment_") and name.endswith(".json")):
                continue
            segment = state.load(name, step=record.step_idx)
            if segment.get("error"):
                continue
            if segment["mode"] == "text":
                segment_command = {
                    "action": "segment",
                    "prompt": segment["prompt"],
                    "camera": segment["camera"],
                }
            else:
                segment_command = {
                    "action": "segment",
                    "point": segment["point"],
                    "camera": segment["camera"],
                }
            event_order = (record.step_idx, int(segment["segment_index"]))
            command_events.append((event_order, segment_command))

    # Never publish a failed trajectory as a recipe. The environment trace is
    # authoritative; an agent's self-reported finish status is not.
    solved = any(
        record.terminated for record in records if record.step_idx > last_reset
    )
    if not solved:
        return ""
    command_events.sort(key=lambda event: event[0])
    recipe_name = f"{recipe_tag}_recipe.jsonl"
    recipe_path = Path(output_dir) / recipe_name
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(
        "".join(json.dumps(command) + "\n" for _, command in command_events)
    )
    return recipe_name


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def _validate_grasp_camera_metadata(camera_meta: dict) -> None:
    """Require a finite pinhole model and rigid camera-to-world transform."""
    intrinsic = np.asarray(camera_meta.get("intrinsic_K"), dtype=np.float64)
    camera_to_world = np.asarray(
        camera_meta.get("extrinsic_cam2world"), dtype=np.float64
    )
    if (
        intrinsic.shape != (3, 3)
        or camera_to_world.shape != (4, 4)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(camera_to_world).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
    ):
        raise ValueError("recorded camera calibration is missing or invalid")
    rotation = camera_to_world[:3, :3]
    if (
        not np.allclose(camera_to_world[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("recorded camera extrinsic must be rigid")


def dump_state(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Save one Libero observation through its owned state record."""
    raw = primitives.env.raw_obs()
    state = {
        "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
        "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
        "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
        "object_names": sorted(
            k[:-4]
            for k in raw
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k
        ),
    }
    log = log or {}
    with env_state.record_step(
        state=state,
        terminated=primitives.env.terminated,
        truncated=primitives.env.truncated,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": primitives.env.get_task_language(),
        },
    ) as step_idx:
        errors = _save_observation_artifacts(primitives, env_state, step_idx, raw)
        if errors:
            env_state.save("observation_errors.json", errors, step=step_idx)
    return env_state.get(step_idx)


def _save_observation_artifacts(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    step_idx: int,
    raw: dict[str, Any],
) -> list[dict[str, str]]:
    errors = []
    env_state.save(
        "agentview_policy.png",
        primitives._last_obs["main_images"],
        step=step_idx,
    )

    # --- camera calibration (static for agentview): fetch metadata as needed ---
    agentview_meta = (
        primitives.env.get_camera_meta(
            camera_name="agentview",
            height=256,
            width=256,
        )
        or {}
    )
    if agentview_meta:
        cam_meta_out = dict(agentview_meta)
        cam_meta_out["projection"] = (
            "Prefer the back_project(row, col, step=NN) MCP tool; it "
            "uses the 1024x1024 high-resolution world map by default. "
            "Pass resolution='low' only when row/col came from the "
            "256x256 calibration-frame image."
        )
        cam_meta_out["note"] = (
            "The agentview_depth.npz observation is aligned with agentview.png. "
            "agentview_policy.png uses the Pi0 orientation and must not supply "
            "pixels for back-projection."
        )
        env_state.save(
            "agentview_metadata.json",
            cam_meta_out,
            step=step_idx,
        )

    # --- per-step RGB in the depth/K frame (vertical-flip of the raw buffer) ---
    # Agentview pixels align with the matching depth and calibration. The policy
    # image uses Pi0 orientation and must not supply back-projection pixels.
    try:
        ci = raw.get("agentview_image")
        if ci is None:
            errors.append({"stage": "agentview image", "error": "missing from raw_obs"})
        else:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            env_state.save(
                "agentview.png",
                ci[::-1],
                step=step_idx,
            )
    except Exception as e:
        logger.warning("image_cam dump failed: %s", e)
        errors.append({"stage": "image_cam dump failed", "error": str(e)})

    # --- per-step metric depth (agentview), native orientation, in meters ---
    try:
        d = raw.get("agentview_depth")
        if d is None:
            errors.append({"stage": "agentview depth", "error": "missing from raw_obs"})
        else:
            # Vertical flip to align with the camera matrices: robosuite's
            # camera_utils projection M = K_exp @ inv(extrinsic) expects the
            # depth map in this frame. VERIFIED 5/5: projecting each GT object
            # world pos via M lands on a pixel whose depth_flip[row,col] matches
            # the object's surface depth (plate Δ6mm, cookies Δ14mm). So
            # Pixels in agentview.png align with agentview_depth.npz and the
            # per-step agentview metadata saved with the record artifacts.
            d = _metric_depth(d, agentview_meta)[::-1]
            env_state.save(
                "agentview_depth.npz",
                d.astype(np.float32),
                step=step_idx,
            )
            world = _world_from_depth(d, agentview_meta).astype(np.float32)
            env_state.save(
                "agentview_world.npz",
                world,
                step=step_idx,
            )
    except Exception as e:
        logger.warning("depth dump failed: %s", e)
        errors.append({"stage": "depth dump failed", "error": str(e)})

    # --- per-step wrist camera (robot0_eye_in_hand), calibration frame ---
    try:
        wimg = raw.get("robot0_eye_in_hand_image")
        if wimg is None:
            errors.append({"stage": "wrist image", "error": "missing from raw_obs"})
        else:
            wimg = np.asarray(wimg)
            if wimg.dtype != np.uint8:
                wimg = wimg.astype(np.uint8)
            env_state.save(
                "wrist.png",
                wimg[::-1],
                step=step_idx,
            )
    except Exception as e:
        logger.warning("wrist image dump failed: %s", e)
        errors.append({"stage": "wrist image dump failed", "error": str(e)})

    try:
        wdpt = raw.get("robot0_eye_in_hand_depth")
        if wdpt is None:
            errors.append({"stage": "wrist depth", "error": "missing from raw_obs"})
        else:
            wdpt_arr = np.asarray(wdpt, dtype=np.float32)
            height, width = wdpt_arr.shape[:2]
            wmeta = primitives.env.get_camera_meta(
                camera_name="robot0_eye_in_hand",
                height=int(height),
                width=int(width),
            )
            if wmeta is None:
                errors.append(
                    {"stage": "wrist calibration", "error": "camera metadata missing"}
                )
            else:
                wdpt_metric = _metric_depth(wdpt_arr, wmeta)[::-1]
                env_state.save(
                    "wrist_depth.npz",
                    wdpt_metric.astype(np.float32),
                    step=step_idx,
                )
                world_w = _world_from_depth(wdpt_metric, wmeta).astype(np.float32)
                env_state.save(
                    "wrist_world.npz",
                    world_w,
                    step=step_idx,
                )

                wmeta_out = dict(wmeta)
                wmeta_out["note"] = (
                    "MOVING camera: extrinsic_cam2world is for THIS step "
                    "only. The matching wrist world-map observation gives world "
                    "(x,y,z) for that pixel in the same world frame as the "
                    "agentview world-map artifact."
                )
                env_state.save(
                    "wrist_metadata.json",
                    wmeta_out,
                    step=step_idx,
                )
    except Exception as e:
        logger.warning("wrist depth/world dump failed: %s", e)
        errors.append({"stage": "wrist depth/world dump failed", "error": str(e)})

    try:
        rgb_hi, depth_hi = primitives.env.render_camera(
            camera_name="agentview",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_hi = primitives.env.get_camera_meta("agentview", 1024, 1024)
        if meta_hi is None:
            raise RuntimeError("agentview camera metadata missing")
        env_state.save(
            "agentview_high.png",
            np.asarray(rgb_hi)[::-1],
            step=step_idx,
        )
        world_hi = _world_from_depth(
            _metric_depth(depth_hi, meta_hi)[::-1],
            meta_hi,
        ).astype(np.float32)
        env_state.save(
            "agentview_world_high.npz",
            world_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("agentview high-res dump failed: %s", e)
        errors.append({"stage": "agentview high-res dump failed", "error": str(e)})

    try:
        rgb_wrist_hi, depth_wrist_hi = primitives.env.render_camera(
            camera_name="robot0_eye_in_hand",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_wrist_hi = primitives.env.get_camera_meta("robot0_eye_in_hand", 1024, 1024)
        if meta_wrist_hi is None:
            raise RuntimeError("robot0_eye_in_hand camera metadata missing")
        env_state.save(
            "wrist_high.png",
            np.asarray(rgb_wrist_hi)[::-1],
            step=step_idx,
        )
        world_wrist_hi = _world_from_depth(
            _metric_depth(depth_wrist_hi, meta_wrist_hi)[::-1],
            meta_wrist_hi,
        ).astype(np.float32)
        env_state.save(
            "wrist_world_high.npz",
            world_wrist_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("wrist high-res dump failed: %s", e)
        errors.append({"stage": "wrist high-res dump failed", "error": str(e)})

    return errors


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped canonical schema)
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "reset",
        "description": "EXPLORE MODE ONLY. Abandon the current episode and restore the same "
        "initial scene. Archive the failed attempt first and state which "
        "strategy lever will change in the next attempt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this episode is "
                    "unrecoverable and what "
                    "will change.",
                }
            },
            "required": ["reason"],
        },
    },
    {
        "name": "view_env_state",
        "description": "object_names lists movable task objects only; absence from object_names is not evidence that a fixture or articulated target is absent. "
        "Read one recorded state and its observation artifacts. Step -1 "
        "selects the latest entry. Embeds policy, agentview, and wrist images "
        "when available. Use the calibration-frame images for pixel "
        "back-projection; JSON state alone is not enough. Use agentview for "
        "global tabletop layout and object locations; use wrist for "
        "close-range details near the gripper, occlusions, and "
        "container/cabinet interiors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
                }
            },
        },
    },
    {
        "name": "generate_grasps",
        "description": (
            "Generate scored GraspGen-X Panda TCP pose candidates from a SAM3 "
            "text mask and measured RGB-D at one recorded observation. Candidates "
            "are in libero_world. Reachability, scene collision, and path collision "
            "are explicitly unchecked; this read-only tool does not execute a grasp. "
            "LIBERO move_pose accepts the complete candidate quaternion and reports "
            "the measured shortest SO(3) residual, including roll."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Text identifying the visible target for SAM3.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Recorded RGB-D camera (default agentview).",
                },
                "step": {
                    "type": "integer",
                    "minimum": -1,
                    "description": "Recorded observation step (-1 = latest).",
                },
                "min_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum SAM3 mask score (default 0.2).",
                },
                "max_candidates": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32,
                    "description": "Maximum returned grasp candidates (default 5).",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "move_to",
        "description": "Scripted EEF servo to a world-frame XYZ target via the OSC "
        "controller. Controls optional target_yaw; otherwise holds "
        "orientation (use rotate_wrist / rotate_pitch / move_pose to "
        "reorient). gripper: -1 = open, +1 = close, applied only during "
        "executed motion steps. At an already reached pose, use set_gripper. "
        "target_reached reports servo convergence; completed execution is not "
        "task success. motion_response reports measured EEF path length and target "
        "distance reduction over the last <=20 executed steps, not a stall cause. "
        "NEVER command a single move_to with |Δxy| > 0.30 — OSC "
        "flips IK and the run corrupts; split long traversal into 2-3 mid "
        "waypoints at carry z.",
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "tol": {
                    "type": "number",
                    "description": "Position tolerance, m (default 0.012)",
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step world XYZ "
                    "delta cap in meters "
                    "(default 0.025)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 80)",
                },
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Optional world-frame yaw target in radians",
                },
                "ori_tol": {
                    "type": "number",
                    "description": "Yaw tolerance in radians "
                    "when target_yaw is "
                    "provided (default "
                    "0.02).",
                },
                "yaw_step_clip": {
                    "type": "number",
                    "description": "Per-step yaw clip, rad (default 0.10)",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "pi0_pick",
        "description": "Pi0.5 closed-loop pick. Use it for the grasp; YOU then do every "
        "move_to and release. Use modest max_chunks and verify the grasp from "
        "EEF lift, gripper closure, and available images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Pi0 prompt (e.g. 'pick up the akita black bowl').",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Action-chunk budget (default 24)",
                },
                "lift_thresh": {
                    "type": "number",
                    "description": "EEF post-descent "
                    "ascent threshold for "
                    "success, m (default "
                    "0.05)",
                },
                "gripper_closed_thresh": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Exclusive upper finger-separation bound in m for "
                    "the local grasp heuristic; must exceed gripper_open_thresh "
                    "(default 0.06).",
                },
                "gripper_open_thresh": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Inclusive lower finger-separation bound in m for "
                    "the local grasp heuristic; must be below gripper_closed_thresh "
                    "(default 0.0). A positive value can reject an empty closed gripper.",
                },
                "descent_thresh": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Minimum EEF descent in m from this call's start "
                    "before the ascent heuristic can pass (default 0.10). "
                    "Zero removes the minimum descent requirement.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "pi0_doubled",
        "description": "Pi0.5 closed-loop contact skill for non-pick interactions (e.g. "
        "stove/knob/button/short push). Returned success/task_success only "
        "mirrors official termination; for intermediate contact skills, "
        "success=false does not necessarily mean the contact interaction "
        "failed. Inspect image/state evidence. Do not use it as a general "
        "pick/place shortcut.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Contact-skill prompt, e.g. 'turn on the stove'.",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Action-chunk budget (default 20)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "release",
        "description": "Open the gripper for up to max_steps env steps while holding EEF in "
        "place. Triggers libero termination if the matching On/In predicate "
        "is met.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 20)",
                }
            },
        },
    },
    {
        "name": "set_gripper",
        "description": "Hold the current EEF pose and drive the gripper command for `steps` "
        "env steps. settle=true treats steps as an upper bound and returns after PyRUA-Lean measured convergence (at least two actions, change below 0.4 mm). Returns measured opening and stop reason; stability is not possession.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of env steps (default 5)",
                },
                "settle": {
                    "type": "boolean",
                    "description": "Stop on measured opening convergence; default false preserves exact requested hold duration.",
                },
            },
        },
    },
    {
        "name": "rotate_wrist",
        "description": "Rotate the wrist around the world Z-axis. Provide either target_yaw "
        "(absolute) or delta_yaw (relative). Holds xyz fixed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Absolute world-frame yaw target, rad",
                },
                "delta_yaw": {
                    "type": ["number", "null"],
                    "description": "Relative yaw delta, rad",
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command held during rotation (default +1)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 40)",
                },
                "tol": {
                    "type": "number",
                    "description": "Yaw tolerance, rad (default 0.02)",
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step yaw clip, rad (default 0.10)",
                },
            },
        },
    },
    {
        "name": "rotate_pitch",
        "description": "Tilt the gripper around the world X-axis. Provide either "
        "target_pitch (absolute) or delta_pitch (relative). Holds xyz fixed. "
        "Pitch means atan2(EEF-z.y, -EEF-z.z), not Euler world-Y pitch. Use "
        "before threading the gripper into a narrow opening whose front face "
        "normal is along world ±y (e.g. microwave cavity).",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_pitch": {
                    "type": ["number", "null"],
                    "description": "Projected EEF-z tilt atan2(R[1,2], -R[2,2]), rad",
                },
                "delta_pitch": {
                    "type": ["number", "null"],
                    "description": "Relative change in projected EEF-z tilt, rad",
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command held during rotation (default +1)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 40)",
                },
                "tol": {
                    "type": "number",
                    "description": "Pitch tolerance, rad (default 0.02)",
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step pitch clip, rad (default 0.10)",
                },
            },
        },
    },
    {
        "name": "move_pose",
        "description": "Servo position AND orientation SIMULTANEOUSLY. Supply either "
        "the full target_quaternion_xyzw used by GraspGen-X or the existing "
        "projected pitch/yaw targets; these modes are mutually exclusive. Unlike "
        "move_to (holds orientation) + rotate_pitch (holds xyz), this "
        "co-varies xyz and orientation every env.step. The quaternion mode controls "
        "the shortest full SO(3) residual, including roll. Changing orientation may "
        "change the feasible path but does not "
        "guarantee recovery from contact or reach constraints. motion_response reports "
        "measured EEF path length and target distance reduction over the last <=20 "
        "executed steps; it does not identify a stall cause or guarantee reachability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "target_pitch": {
                    "type": ["number", "null"],
                    "description": "Absolute pitch target, rad",
                },
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Absolute yaw target, rad",
                },
                "target_quaternion_xyzw": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Unit world-frame EEF quaternion [x,y,z,w]. Mutually "
                        "exclusive with target_pitch and target_yaw."
                    ),
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command held during the move (default -1)",
                },
                "step_clip": {
                    "type": "number",
                    "description": "Per-step Δxyz cap, m (default 0.02)",
                },
                "pitch_step": {
                    "type": "number",
                    "description": "Per-step pitch clip, rad (default 0.08)",
                },
                "yaw_step": {
                    "type": "number",
                    "description": "Per-step yaw clip, rad (default 0.08)",
                },
                "quaternion_step": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Maximum shortest SO(3) rotation per step, rad (default 0.08)",
                },
                "tol": {
                    "type": "number",
                    "description": "Position tolerance, m (default 0.012)",
                },
                "ori_tol": {
                    "type": "number",
                    "description": "Orientation tolerance, rad (default 0.05)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget (default 150)",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": "Read per-step camera calibration metadata from recorded artifacts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera metadata to read (default agentview).",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Metadata step to use; -1 = latest.",
                },
            },
        },
    },
    {
        "name": "segment",
        "description": "SAM3 visual segmentation over an existing run artifact. It never "
        "renders a new camera view. Provide exactly one text prompt or single "
        "positive point. A successful top-ranked mask is projected through "
        "the matching world map to produce world_xyz. found means a mask was "
        "detected, not that task identity is confirmed. The red overlay is a "
        "synthetic mask annotation, never the object color; use the original "
        "image for color-based queries. Also returns PyRUA-Lean footprint geometry and world closing axis. support_z enables upright-on-plane height/completion; omit for visible surface geometry only. A world axis is not a robot wrist command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Object/text prompt to segment.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Artifact camera to use (default agentview).",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step to segment; -1 = latest.",
                },
                "point": {
                    "type": ["array", "null"],
                    "description": "Optional single positive "
                    "point as [row, col]. "
                    "Mutually exclusive with "
                    "prompt.",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum accepted mask score (default 0.2).",
                },
                "support_z": {
                    "type": "number",
                    "description": "Known world support-plane height in metres; do not assume it for stacked/held objects.",
                },
            },
        },
    },
    {
        "name": "back_project",
        "description": "Back-project a pixel (row, col) to a world XYZ point using the "
        "selected camera's precomputed world map. Row 0 = top of image, col 0 "
        "= left. Returns world_xyz in meters.\n"
        "\n"
        "Use when the chosen action needs a world-coordinate surface estimate "
        "not already available. Look at the "
        "embedded high-resolution image returned by view_env_state to pick a "
        "pixel on the target object, then call back_project. The default "
        "resolution is high (1024x1024). Pass resolution='low' only for "
        "pixels from the embedded/standard 256 image. The pixel coordinates "
        "must come from the same camera and resolution requested here. Use "
        "camera='agentview' for global tabletop layout and object locations; "
        "use camera='wrist' for close-range details near the gripper, "
        "occlusions, and container/cabinet interiors. Additional pixels are "
        "optional when a specific depth/edge ambiguity warrants them; "
        "reuse an adequate estimate on an unchanged scene.\n"
        "\n"
        "REGION MODE: pass row_range=[r0,r1] and col_range=[c0,c1] instead of "
        "row/col to get the midpoint of world xy over that pixel window, with "
        "an optional world-z band (z_min, z_max). Use it for the center of a "
        "container cavity or flat region, where a single-pixel or mask-median "
        "estimate is biased toward an edge/rim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": ["integer", "null"],
                    "description": "Pixel row (0=top) in the "
                    "selected resolution image.",
                },
                "col": {
                    "type": ["integer", "null"],
                    "description": "Pixel column (0=left) in the "
                    "selected resolution image.",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Depth/world-map step; 0 = initial, -1 = latest.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": "Coordinate system for "
                    "row/col (default "
                    "high). Use low only "
                    "when row/col came "
                    "from the "
                    "embedded/standard 256 "
                    "image.",
                },
                "row_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [r0, r1] "
                    "pixel row window. "
                    "Requires col_range.",
                },
                "col_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [c0, c1] "
                    "pixel col window. "
                    "Requires row_range.",
                },
                "z_min": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only "
                    "pixels with world z >= "
                    "z_min.",
                },
                "z_max": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only "
                    "pixels with world z <= "
                    "z_max.",
                },
            },
        },
    },
]


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict:
    try:
        record = state.get(step)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}

    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
    }
    out["observation_errors"] = (
        state.load("observation_errors.json", step=nn)
        if "observation_errors.json" in record.artifacts
        else []
    )
    if extras.get("recording_error"):
        out["observation_errors"].append(extras["recording_error"])
    out["task_language"] = extras.get("task_language")
    out["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for slot, names in (
        ("_image_bytes", ("agentview_policy.png",)),
        ("_image_cam_bytes", ("agentview_high.png", "agentview.png")),
        ("_image_wrist_bytes", ("wrist_high.png", "wrist.png")),
    ):
        name = next((name for name in names if name in record.artifacts), None)
        if name:
            try:
                out[slot] = state.load_bytes(name, step=nn)
                purpose = (
                    "policy orientation; NOT for pixel back-projection"
                    if slot == "_image_bytes"
                    else "calibration frame; use matching camera/resolution for back_project"
                )
                out.setdefault("_image_labels", {})[slot] = (
                    f"{name}, step={nn}; {purpose}"
                )
            except FileNotFoundError as exc:
                out["observation_errors"].append({"artifact": name, "error": str(exc)})
    return out


def _select_segment_artifacts(
    state: EnvState,
    record: StepRecord,
    camera: str,
) -> tuple[str | None, str | None, list[tuple[str | None, str | None]]]:
    if camera not in ("agentview", "wrist"):
        raise ValueError(f"unknown segment camera: {camera}")
    pairs = [
        (f"{camera}_high.png", f"{camera}_world_high.npz"),
        (f"{camera}.png", f"{camera}_world.npz"),
    ]
    for image_name, world_name in pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=record.step_idx)
            and state.exists(world_name, step=record.step_idx)
        ):
            return image_name, world_name, pairs
    return None, None, pairs


def _next_segment_index(record: StepRecord) -> int:
    idx = 0
    while f"segment_{idx:02d}.json" in record.artifacts:
        idx += 1
    return idx


def _next_grasp_index(record: StepRecord) -> int:
    idx = 0
    while f"grasp_mask_{idx:02d}.json" in record.artifacts:
        idx += 1
    return idx


def _mask_to_world(
    mask: np.ndarray, world_map: np.ndarray, min_valid: int = 10
) -> dict:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel_order": "row_col",
        "centroid_pixel": [
            int(round(float(np.median(ys)))),
            int(round(float(np.median(xs)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update(
            {
                "world_xyz": None,
                "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
            }
        )
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay


@readonly
def view_camera_meta(
    camera: str = "agentview",
    step: int = -1,
    *,
    state: EnvState,
) -> dict:
    """Read camera calibration metadata for localization."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}

    try:
        record = state.get(step)
        metadata_name = f"{camera}_metadata.json"
        if metadata_name not in record.artifacts:
            raise FileNotFoundError(metadata_name)
        meta = state.load(metadata_name, step=record.step_idx)
    except Exception as e:
        return {"error": f"{camera} camera metadata not found: {e}"}

    if camera == "agentview":
        return {"camera": "agentview", "camera_meta": meta}
    return {"camera": "wrist", "step": record.step_idx, "camera_meta": meta}


@readonly
def back_project(
    row: int | None = None,
    col: int | None = None,
    step: int = -1,
    camera: str = "agentview",
    resolution: str = "high",
    row_range: list | None = None,
    col_range: list | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    *,
    state: EnvState,
) -> dict:
    """Look up a pixel's world XYZ in the precomputed world map."""
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
    if resolution not in ("high", "low"):
        return {"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}

    region_mode = row_range is not None or col_range is not None
    if region_mode and (row is not None or col is not None):
        return {"error": "provide a pixel or region, not both"}
    if not region_mode and (row is None or col is None):
        return {
            "error": (
                "provide either (row, col) for a single pixel, or "
                "row_range=[r0,r1] and col_range=[c0,c1] for a region center"
            )
        }

    try:
        record = state.get(step)
    except Exception as e:
        return {"error": f"state step not available: {e}"}
    nn = record.step_idx

    hi_artifact = f"{camera}_world_high.npz"
    low_artifact = f"{camera}_world.npz"
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if source_artifact not in record.artifacts:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not recorded for step {nn}"
            )
        }

    try:
        world_map = state.load(str(source_artifact), step=nn)
    except Exception as e:
        return {
            "error": (
                f"{camera} {resolution}-resolution artifact not found "
                f"for step {nn}: {e}"
            )
        }

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return {
                "error": "region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
            }
        try:
            r0, r1 = int(row_range[0]), int(row_range[1])
            c0, c1 = int(col_range[0]), int(col_range[1])
        except Exception:
            return {"error": "row_range/col_range must each be [min, max] integers"}
        if r0 >= r1 or c0 >= c1:
            return {"error": "region ranges must be increasing [start, stop) intervals"}
        r0, r1 = max(0, r0), min(height, r1)
        c0, c1 = max(0, c0), min(width, c1)
        if r1 <= r0 or c1 <= c0:
            return {
                "error": (
                    f"empty region after clamping to image {height}x{width}: "
                    f"rows [{r0},{r1}] cols [{c0},{c1}]"
                )
            }
        window = (
            world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(np.float64)
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return {
                "error": (
                    f"too few valid pixels in region after z-filter "
                    f"({int(pts.shape[0])}); widen the window or the z band"
                ),
                "n_valid_before_zfilter": n_total,
            }
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return {
            "camera": camera,
            "resolution": resolution,
            "mode": "region",
            "row_range": [r0, r1],
            "col_range": [c0, c1],
            "z_band": [z_min, z_max],
            "center_xyz": center,
            "median_xyz": [
                round(float(np.median(xs)), 4),
                round(float(np.median(ys)), 4),
                round(float(np.median(zs)), 4),
            ],
            "n_valid": int(pts.shape[0]),
            "step": nn,
            "image_size": [height, width],
            "source_artifact": source_artifact,
        }

    if row < 0 or row >= height or col < 0 or col >= width:
        return {
            "error": (
                f"pixel ({row},{col}) out of bounds; {camera} image is {height}x{width}"
            )
        }

    depth_m = None
    if source_artifact == low_artifact:
        try:
            depth_artifact = f"{camera}_depth.npz"
            if depth_artifact not in record.artifacts:
                raise FileNotFoundError(depth_artifact)
            depth = state.load(depth_artifact, step=nn)
            if depth.ndim == 3:
                depth = depth[..., 0]
        except Exception as e:
            return {"error": f"{camera} depth not found for step {nn}: {e}"}
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return {
                "error": (
                    f"invalid {camera} depth {depth_m:.3f}m at pixel "
                    f"({row},{col}); pick a different pixel"
                )
            }
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return {"error": f"invalid {camera} world xyz at pixel ({row},{col})"}
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return out
