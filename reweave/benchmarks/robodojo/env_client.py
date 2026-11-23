# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""RoboDojo RPC client with acknowledged action and frozen execution ownership."""

from __future__ import annotations

import copy
import io
import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from reweave.benchmarks.robodojo.access import (
    native_action,
    public_observation,
    robot_state_features,
)
from reweave.robots.components.env_client_base import BaseEnvClient
from reweave.runtime.capability import (
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.utils.persistence import atomic_text
from reweave.utils.rpc.http_rpc import _NumpyEncoder
from reweave.utils.rpc.rpc_client import RpcError, RpcTransportError

_TASK_LANGUAGE_UNSET = object()


def decode_sensor_snapshot(payload):
    """Restore an explicitly requested immutable RGB-D archive, never a live read."""
    if not isinstance(payload, dict) or set(payload) != {
        "observation_ref",
        "archive",
        "metadata",
    }:
        raise ValueError("RoboDojo sensor archive response has unexpected fields")
    metadata = copy.deepcopy(payload["metadata"])
    if (
        not isinstance(metadata, dict)
        or metadata.get("observation_ref") != payload["observation_ref"]
    ):
        raise ValueError("RoboDojo sensor archive metadata refers to another snapshot")
    if not isinstance(payload["archive"], bytes):
        raise ValueError("RoboDojo sensor archive must be original NPZ bytes")
    expected = {}
    for camera, value in metadata["vision"].items():
        arrays = value.pop("arrays")
        for field, descriptor in arrays.items():
            key = descriptor["key"]
            if key in expected:
                raise ValueError("RoboDojo sensor archive has repeated array keys")
            expected[key] = (camera, field, descriptor)
    with np.load(io.BytesIO(payload["archive"]), allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError(
                "RoboDojo sensor archive differs from its original metadata"
            )
        for key, (camera, field, descriptor) in expected.items():
            value = arrays[key]
            if (
                list(value.shape) != descriptor["shape"]
                or str(value.dtype) != descriptor["dtype"]
            ):
                raise ValueError(
                    "RoboDojo sensor array shape or dtype differs from its metadata"
                )
            metadata["vision"][camera][field] = value.copy()
    return public_observation(metadata)


class RoboDojoEnvClient(BaseEnvClient):
    """Own one scene's terminal state, physical counters and exact action receipts."""

    def __init__(self, client, *, expected_meta: dict):
        self.max_episode_steps = expected_meta["max_episode_steps"]
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError("RoboDojo max_episode_steps must be positive")
        self._trace = None
        self._execution = None
        self._outcome_unknown = False
        self._environment_error = None
        self._service_failure = None
        self._elapsed_steps = 0
        # Step count at whose non-terminal ACK the official check was already false.
        self._unsolved_at_step: int | None = None
        self._task_language = _TASK_LANGUAGE_UNSET
        self._observation_instance = None
        self._trajectory_window_steps = None
        self._trajectory_dt_s = None
        self._native_plans_dir = None
        self.terminated = False
        self.truncated = False
        super().__init__(
            client,
            expected_meta={**expected_meta, "observation_transport": "server_snapshot"},
        )

    def reset(self):
        if self._elapsed_steps or self._outcome_unknown:
            raise ValueError("RoboDojo cannot reset after a physical action")
        self.last_obs = None
        self.last_obs = self._accept_observation(super().reset(), expected_step=0)
        self.terminated = self.is_success()
        return self.last_obs

    def bind_frozen_execution(
        self, trace, execution, *, max_episode_steps: int
    ) -> None:
        """Bind the existing scene to the shared action trace without resetting it."""
        if self._trace is not None or self._elapsed_steps:
            raise ValueError("RoboDojo execution must be bound once before acting")
        if max_episode_steps != self.max_episode_steps:
            raise ValueError("RoboDojo physical limit differs from the environment")
        self._trace, self._execution = trace, execution
        # ExecutionTrace owns the episode's audit location. In-memory traces
        # deliberately have no filesystem artifact destination.
        trace_path = getattr(trace, "_path", None)
        if trace_path is not None:
            self._native_plans_dir = Path(trace_path).parent / "native_plans"
        trace.record_reset_state(physical_features=robot_state_features(self.last_obs))

    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    def _read_call(self, method, **kwargs):
        """Preserve exhausted transport recovery as infrastructure, not author code."""
        try:
            return self._client.call(method, **kwargs)
        except RpcTransportError as error:
            failure = EnvironmentServiceFailure(str(error))
            if self._service_failure is None:
                self._service_failure = failure
            raise self._service_failure from error
        except RpcError as error:
            self._raise_resource_failure(str(error))
            raise

    def _raise_resource_failure(self, message):
        """A device allocation failure is not a geometric planning rejection."""
        if isinstance(message, str) and any(
            marker in message.lower()
            for marker in (
                "cuda out of memory",
                "cuda error: out of memory",
                "cuda_error_out_of_memory",
                "vk_error_out_of_device_memory",
            )
        ):
            if self._service_failure is None:
                self._service_failure = EnvironmentServiceFailure(message)
            # No automatic action replay or retry into the same memory pressure.
            raise self._service_failure

    def steps_remaining(self) -> int:
        return max(0, self.max_episode_steps - self._elapsed_steps)

    def _assert_action_available(self):
        if self._outcome_unknown:
            raise PhysicalOutcomeUnknown("RoboDojo physical action outcome is unknown")
        self.raise_if_invalid()
        if self.terminated or self.truncated or not self.steps_remaining():
            raise ValueError("RoboDojo episode is terminal")

    def step(self, action, *, cursor=None):
        """Submit one official policy target with its existing ACK semantics."""
        self._assert_action_available()
        command = native_action(action)
        if cursor is not None and not np.array_equal(np.asarray(action), cursor.action):
            raise ValueError("RoboDojo action differs from its VLA cursor")
        vector = np.concatenate([np.asarray(value) for value in command.values()])
        return self._submit_step(
            "env.step",
            (command,),
            {},
            vector,
            cursor=cursor,
            evidence={"action_fields": ",".join(command)},
        )

    def step_trajectory(self, arm, positions, velocities, *, gripper=0):
        """Submit exactly one original native trajectory window; never replay unknown."""
        self._assert_action_available()
        points = np.asarray(positions, dtype=np.float64)
        speeds = np.asarray(velocities, dtype=np.float64)
        if (
            arm not in {"left", "right"}
            or points.ndim != 2
            or points.shape[1:] != (6,)
            or speeds.shape != points.shape
            or not np.isfinite(points).all()
            or not np.isfinite(speeds).all()
            or not np.isfinite(gripper)
            or self._trajectory_window_steps is None
            or not 1 <= len(points) <= self._trajectory_window_steps
        ):
            raise ValueError(
                "trajectory execution requires one finite planned native control window"
            )
        command = {
            "arm": arm,
            "positions": points.tolist(),
            "velocities": speeds.tolist(),
            "gripper": float(gripper),
        }
        vector = np.concatenate((points.reshape(-1), speeds.reshape(-1), [gripper]))
        return self._submit_step(
            "env.step_trajectory",
            (),
            command,
            vector,
            evidence={
                "action_fields": "native_trajectory_positions,velocities,gripper",
                "trajectory_arm": arm,
                "trajectory_substeps": len(points),
            },
        )

    def _submit_step(self, method, args, kwargs, vector, *, cursor=None, evidence):
        """Shared physical counter/trace/unknown boundary for both action forms."""
        snapshot = self._execution.snapshot if self._execution is not None else None
        if snapshot is not None and (
            snapshot.phase != "running" or snapshot.execution_id is None
        ):
            raise RuntimeError("frozen RoboDojo action requires an active tool")
        if cursor is not None:
            if snapshot is None:
                raise RuntimeError(
                    "RoboDojo VLA action requires a bound execution trace"
                )
            self._trace.validate_control_step(
                cursor, execution_ref=snapshot.execution_id
            )
        started = time.monotonic()
        previous_obs = self.last_obs
        # Until an ACK provides the replacement, the previous scene snapshot is
        # no longer authoritative. Unknown/invalid responses must not revive it.
        self.last_obs = None
        try:
            response = self._client.call(
                method, args=args, kwargs=kwargs, timeout_s=self._TIMEOUT_S["env.step"]
            )
            if not isinstance(response, (tuple, list)) or len(response) != 4:
                raise ValueError("invalid RoboDojo env.step acknowledgement")
            obs, reward, done, info = response
            if method == "env.step_trajectory" and (
                not isinstance(info, dict)
                or type(info.get("trajectory_substeps")) is not int
                or info["trajectory_substeps"] != len(kwargs["positions"])
                or info.get("trajectory_dt_s") != self._trajectory_dt_s
            ):
                raise ValueError(
                    "trajectory acknowledgement differs from its submitted native rows"
                )
            if type(done) is not bool or not isinstance(info, dict):
                raise ValueError("invalid RoboDojo terminal acknowledgement")
            environment_error = info["environment_error"]
            if environment_error is not None and (
                type(environment_error) is not str or not environment_error.strip()
            ):
                raise ValueError(
                    "invalid RoboDojo environment validity acknowledgement"
                )
            status = info["status"]
            if (
                type(status["step"]) is not int
                or type(status["step_limit"]) is not int
                or status["step"] != self._elapsed_steps + 1
                or status["step_limit"] != self.max_episode_steps
            ):
                raise ValueError(
                    "RoboDojo action counter acknowledgement differs from requested episode"
                )
        except Exception as error:
            if (
                isinstance(error, RpcTransportError)
                and error.recovery_state == "not_submitted"
            ):
                # Only the transport can prove no physical bytes were sent.
                # An unknown/expired receipt must NEVER enter this branch.
                self.last_obs = previous_obs
                failure = EnvironmentServiceFailure(str(error))
                if self._service_failure is None:
                    self._service_failure = failure
                raise self._service_failure from error
            self._outcome_unknown = True
            unknown = PhysicalOutcomeUnknown(
                f"RoboDojo physical action outcome is unknown: {error}"
            )
            if snapshot is not None:
                try:
                    self._trace.record_action_outcome_unknown(
                        execution_ref=snapshot.execution_id,
                        error=error,
                        action=cursor,
                        requested_action=vector,
                        action_index=self._elapsed_steps,
                    )
                except BaseException as audit_error:
                    # An audit failure or cancellation cannot settle a submitted action.
                    unknown.add_note(
                        "failed to record unknown action: "
                        f"{type(audit_error).__name__}: {audit_error}"
                    )
            raise unknown from error
        self._elapsed_steps += 1
        self._environment_error = environment_error
        self.terminated = done
        # The server's `done` already includes the official success check, so a
        # non-terminal ACK proves it was false at this step without another RPC.
        self._unsolved_at_step = None if done else self._elapsed_steps
        self.truncated = self._elapsed_steps >= self.max_episode_steps
        if snapshot is not None:
            if cursor is None:
                index = self._trace.record_scripted_control_step(
                    execution_ref=snapshot.execution_id,
                    action_index=self._elapsed_steps - 1,
                    action=vector,
                    terminated=done,
                    truncated=self.truncated,
                    public_evidence=evidence,
                )
            else:
                index = self._trace.record_control_step(
                    cursor, terminated=done, truncated=self.truncated
                )
        # The action ACK is committed before observation or telemetry processing;
        # a malformed image/state must not erase an already executed control step.
        self.last_obs = self._accept_observation(obs, expected_step=self._elapsed_steps)
        if snapshot is not None:
            diagnostics = {
                f"simulator.{name}_s": value
                for name, value in info.get("server_timing_s", {}).items()
            }
            rpc_metrics = getattr(self._client, "last_call_metrics", None)
            if is_dataclass(rpc_metrics):
                diagnostics.update(
                    {
                        f"rpc.{name}": value
                        for name, value in asdict(rpc_metrics).items()
                        if value is not None
                    }
                )
            self._trace.record_telemetry(
                index,
                public_evidence=diagnostics,
                physical_features=robot_state_features(obs),
                action_wall_time_s=time.monotonic() - started,
            )
        return (
            self.last_obs,
            reward,
            done,
            {
                "status": self.get_status(),
                "environment_error": self._environment_error,
                **(
                    {
                        "trajectory_substeps": info["trajectory_substeps"],
                        "trajectory_dt_s": info["trajectory_dt_s"],
                    }
                    if method == "env.step_trajectory"
                    else {}
                ),
            },
        )

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        raise RuntimeError(
            "RoboDojo actions must use individually acknowledged env.step"
        )

    def get_obs(self) -> dict:
        """Explicitly acquire a snapshot and replace the scene's observation cache."""
        if self._outcome_unknown:
            raise PhysicalOutcomeUnknown("RoboDojo physical action outcome is unknown")
        self.raise_if_invalid()
        self.last_obs = None
        self.last_obs = self._accept_observation(
            self._read_call("env.get_obs", timeout_s=self._TIMEOUT_S["default"]),
            expected_step=self._elapsed_steps,
        )
        return self.last_obs

    def _accept_observation(self, observation, *, expected_step):
        """Accept only a compact snapshot belonging to this settled scene step."""
        from reweave.benchmarks.robodojo.access import CAMERAS

        if not isinstance(observation, dict):
            raise ValueError("RoboDojo snapshot acknowledgement must be an object")
        instance = observation.get("service_instance_id")
        step = observation.get("physical_step")
        reference = observation.get("observation_ref")
        if not isinstance(instance, str) or not instance or ":" in instance:
            raise ValueError("RoboDojo snapshot requires its service instance")
        if type(step) is not int or step != expected_step:
            raise ValueError(
                "RoboDojo snapshot differs from its acknowledged physical step"
            )
        if (
            not isinstance(reference, str)
            or not reference.startswith(f"{instance}:{step}:")
            or len(reference.rsplit(":", 1)[-1]) != 32
        ):
            raise ValueError(
                "RoboDojo observation reference is not bound to its scene and step"
            )
        if (
            self._observation_instance is not None
            and instance != self._observation_instance
        ):
            raise ValueError("RoboDojo snapshot belongs to another service instance")
        vision = observation.get("vision")
        if not isinstance(vision, dict) or set(vision) != set(CAMERAS):
            raise ValueError(
                "RoboDojo snapshot requires all three camera metadata records"
            )
        for camera in CAMERAS:
            metadata = vision[camera]
            if not isinstance(metadata, dict) or set(metadata) & {
                "color",
                "depth",
                "distance_to_image_plane",
                "mask",
                "point_cloud",
            }:
                raise ValueError(
                    "RoboDojo action ACK cannot contain RGBD or perception arrays"
                )
            shape = metadata.get("shape")
            if (
                not isinstance(shape, (tuple, list))
                or len(shape) not in (2, 3)
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValueError(
                    "RoboDojo camera metadata requires its original image shape"
                )
            for field, dimensions in [
                ("intrinsic_matrix", (3, 3)),
                ("extrinsic_matrix", (4, 4)),
            ]:
                matrix = np.asarray(metadata.get(field), dtype=np.float64)
                if matrix.shape != dimensions or not np.isfinite(matrix).all():
                    raise ValueError(
                        f"RoboDojo camera metadata requires finite {field}"
                    )
            if type(metadata.get("has_depth")) is not bool:
                raise ValueError(
                    "RoboDojo camera metadata requires an explicit depth availability flag"
                )
        robot_state_features(observation)
        accepted = public_observation(observation)
        self._observation_instance = instance
        return accepted

    @property
    def current_observation_ref(self):
        """Reference for the latest accepted state; unavailable after unknown I/O."""
        if self._outcome_unknown:
            raise PhysicalOutcomeUnknown("RoboDojo physical action outcome is unknown")
        self.raise_if_invalid()
        if self.last_obs is None:
            raise RuntimeError("RoboDojo has no accepted current snapshot")
        return self.last_obs["observation_ref"]

    def _snapshot_call(self, method, observation_ref, *, timeout_s=30, **kwargs):
        """Read a named snapshot without substituting latest or retrying a call."""
        if self._outcome_unknown:
            raise PhysicalOutcomeUnknown("RoboDojo physical action outcome is unknown")
        self.raise_if_invalid()
        if not isinstance(observation_ref, str) or not observation_ref.startswith(
            f"{self._observation_instance}:"
        ):
            raise ValueError("RoboDojo snapshot reference belongs to another scene")
        return self._read_call(
            method,
            kwargs={"observation_ref": observation_ref, **kwargs},
            timeout_s=timeout_s,
        )

    def retain_observation(self, observation_ref):
        """Retain exact sensor evidence on the server, returning only references."""
        result = self._snapshot_call("env.retain_observation", observation_ref)
        if (
            not isinstance(result, dict)
            or result.get("observation_ref") != observation_ref
        ):
            raise ValueError("RoboDojo retained evidence refers to another snapshot")
        return result

    def read_sensor_snapshot(self, observation_ref):
        """Explicit diagnostic transfer of one retained raw archive, outside control loops."""
        result = self._snapshot_call(
            "env.read_sensor_snapshot", observation_ref, timeout_s=120
        )
        if (
            not isinstance(result, dict)
            or result.get("observation_ref") != observation_ref
        ):
            raise ValueError("RoboDojo sensor archive refers to another snapshot")
        return decode_sensor_snapshot(result)

    def read_rgb(self, observation_ref, cameras):
        """Read only explicitly selected, lossless RGB PNGs from one exact frame."""
        from reweave.benchmarks.robodojo.access import CAMERAS

        cameras = list(cameras)
        if (
            not cameras
            or len(set(cameras)) != len(cameras)
            or set(cameras) - set(CAMERAS)
        ):
            raise ValueError("RoboDojo RGB read requires distinct registered cameras")
        result = self._snapshot_call("env.read_rgb", observation_ref, cameras=cameras)
        if not isinstance(result, dict) or set(result) != set(cameras):
            raise ValueError("RoboDojo RGB response differs from requested cameras")
        if any(
            not isinstance(value, bytes) or not value.startswith(b"\x89PNG\r\n\x1a\n")
            for value in result.values()
        ):
            raise ValueError("RoboDojo RGB response must contain lossless PNG bytes")
        return result

    def back_project_ref(self, observation_ref, row, col, camera="cam_head"):
        return self._snapshot_call(
            "env.back_project_ref", observation_ref, row=row, col=col, camera=camera
        )

    def segment_ref(
        self, observation_ref, text_prompt, camera="cam_head", min_score=0.2
    ):
        return self._snapshot_call(
            "env.segment_ref",
            observation_ref,
            text_prompt=text_prompt,
            camera=camera,
            min_score=min_score,
            timeout_s=120,
        )

    def locate_ref(
        self,
        observation_ref,
        text_prompt,
        camera="cam_head",
        min_score=0.2,
        support_z=None,
    ):
        return self._snapshot_call(
            "env.locate_ref",
            observation_ref,
            text_prompt=text_prompt,
            camera=camera,
            min_score=min_score,
            support_z=support_z,
            timeout_s=120,
        )

    def grasps_ref(self, observation_ref, mask_ref, arm, max_candidates=5):
        return self._snapshot_call(
            "env.grasps_ref",
            observation_ref,
            mask_ref=mask_ref,
            arm=arm,
            max_candidates=max_candidates,
            timeout_s=120,
        )

    def predict_ref(self, observation_ref, prompt, mode="eval"):
        from reweave.robots.vla_execution import VLAChunkPrediction

        result = self._snapshot_call(
            "env.predict_ref", observation_ref, prompt=prompt, mode=mode, timeout_s=120
        )
        prediction = VLAChunkPrediction.from_rpc(result)
        if prediction.actions.shape[1] != 14:
            raise ValueError("RoboDojo Pi0.5 must return 14 joint action values")
        return prediction

    def render_camera(self, camera_name, **kwargs):
        raise RuntimeError(
            "RoboDojo uses read_rgb(snapshot, cameras); raw RGBD stays on the server"
        )

    def get_task_language(self) -> str | None:
        """The task is fixed for this scene; fetch its instruction once per client."""
        if self._task_language is _TASK_LANGUAGE_UNSET:
            self._task_language = self._read_call(
                "env.get_task_language", timeout_s=self._TIMEOUT_S["default"]
            )
        return self._task_language

    def get_status(self) -> dict:
        return {"step": self._elapsed_steps, "step_limit": self.max_episode_steps}

    def _plan_call(self, method, **kwargs):
        result = self._read_call(method, **kwargs)
        if self._native_plans_dir is not None:
            plan_id = uuid.uuid4().hex
            receipt = {
                "version": 1,
                "plan_id": plan_id,
                "method": method,
                "arguments": kwargs.get("kwargs", {}),
                "physical_step": self._elapsed_steps,
                "observation_ref": (self.last_obs or {}).get("observation_ref"),
                "received_at": time.time(),
                "response": result,
            }
            # Preserve the full decoded RPC value, including tagged arrays and
            # bytes if present. This audit is never added to model feedback.
            atomic_text(
                self._native_plans_dir / f"{plan_id}.json",
                json.dumps(receipt, cls=_NumpyEncoder, allow_nan=False) + "\n",
            )
        if result.get("status") != "Success":
            self._raise_resource_failure(result.get("error"))
        if result.get("status") == "Success":
            dt = float(result["planner_dt_s"])
            period = float(result["control_period_s"])
            ratio = period / dt if dt > 0 else float("nan")
            if (
                not np.isfinite(ratio)
                or ratio < 1
                or not np.isclose(ratio, round(ratio), atol=1e-6)
            ):
                raise ValueError(
                    "native planner and environment control periods do not form whole physics windows"
                )
            self._trajectory_window_steps = round(ratio)
            self._trajectory_dt_s = dt
        return result

    def plan_position_path(self, arm: str, xyz: list) -> dict:
        return self._plan_call(
            "env.plan_position_path",
            kwargs={"arm": arm, "xyz": xyz},
            timeout_s=120.0,
        )

    def plan_joint_path(self, arm: str, joints: list) -> dict:
        return self._plan_call(
            "env.plan_joint_path",
            kwargs={"arm": arm, "joints": joints},
            timeout_s=120.0,
        )

    def solve_ik_pose(self, arm: str, pose: list) -> dict:
        return self._read_call(
            "env.solve_ik_pose",
            kwargs={"arm": arm, "pose": pose},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def plan_pose_path(
        self, arm: str, poses: list, linear_axis: str | None = None
    ) -> dict:
        return self._plan_call(
            "env.plan_pose_path",
            kwargs={"arm": arm, "poses": poses, "linear_axis": linear_axis},
            timeout_s=120.0,
        )

    def is_success(self) -> bool:
        self.raise_if_invalid()
        if self._unsolved_at_step == self._elapsed_steps:
            return False
        result = self._read_call("env.is_success", timeout_s=self._TIMEOUT_S["default"])
        if type(result) is not bool:
            raise ValueError("RoboDojo official success must be a boolean")
        return result

    def raise_if_invalid(self) -> None:
        """Reject a simulator-invalid scene after preserving its acknowledged actions."""
        if self._service_failure is not None:
            raise self._service_failure
        if self._environment_error is not None:
            raise EnvironmentInvalid(self._environment_error)

    def close(self) -> None:
        self._read_call("env.close", timeout_s=10)
