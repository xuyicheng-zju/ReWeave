# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Agent recording, diagnostics and official native motion planning."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reweave.utils.logging import get_logger

logger = get_logger(__name__)


def _ffmpeg_executable() -> str:
    """Resolve an ffmpeg that can encode H.264 for browser playback.

    OpenCV's ``mp4v`` fourcc writes MPEG-4 Part 2. VS Code's player is
    Chromium and does not decode that codec. ``imageio-ffmpeg`` ships a
    libx264 build; a PATH ffmpeg is the fallback used inside the sim env.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        bundled = get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception as error:
        logger.debug("imageio-ffmpeg unavailable: %s", error)
    found = shutil.which("ffmpeg")
    if found:
        return found
    sibling = os.path.join(os.path.dirname(os.path.realpath(sys.executable)), "ffmpeg")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    raise RuntimeError(
        "H.264 recording needs imageio-ffmpeg or an ffmpeg binary on PATH"
    )


def _even_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Pad to even width and height so yuv420p H.264 is legal."""
    array = np.ascontiguousarray(frame)
    height, width = array.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return array


class _H264Writer:
    """Stream RGB frames to an H.264 mp4 Chromium and VS Code can play."""

    def __init__(self, path: str, fps: int, frame: np.ndarray):
        array = _even_rgb_frame(frame)
        height, width = array.shape[:2]
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 RGB frame, got {array.shape} {array.dtype}"
            )
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self._frame_bytes = self.width * self.height * 3
        self._err_path = path + ".ffmpeg.err"
        self._err = open(self._err_path, "w")
        try:
            self._proc = subprocess.Popen(
                [
                    _ffmpeg_executable(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    str(int(fps)),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "high",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._err,
            )
        except Exception:
            self._err.close()
            os.remove(self._err_path)
            raise
        if self._proc.stdin is None:
            self._err.close()
            os.remove(self._err_path)
            raise RuntimeError(f"cannot open H.264 writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        array = _even_rgb_frame(frame)
        if array.shape[0] != self.height or array.shape[1] != self.width:
            raise ValueError(
                f"frame size changed for {self.path}: "
                f"{array.shape[1]}x{array.shape[0]} != {self.width}x{self.height}"
            )
        payload = array.tobytes()
        if len(payload) != self._frame_bytes:
            raise ValueError(f"unexpected RGB payload for {self.path}")
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited while writing {self.path}")
        self._proc.stdin.write(payload)

    def release(self) -> None:
        code = 0
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
            code = self._proc.wait(timeout=120)
        finally:
            self._err.close()
            error = ""
            if os.path.isfile(self._err_path):
                with open(self._err_path, encoding="utf-8", errors="replace") as handle:
                    error = handle.read().strip()
                os.remove(self._err_path)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}) for {self.path}: {error}")


class _VideoRecorder:
    """Write head and both wrist H.264 videos without changing observations."""

    def __init__(self, video_dir: str, fps: int = 25):
        self.video_dir = video_dir
        self.fps = fps
        self.writers: dict[str, Any] = {}
        self.paths: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.video_dir)

    def _ensure_writer(self, cam: str, frame) -> None:
        if cam in self.writers:
            return
        os.makedirs(self.video_dir, exist_ok=True)
        run_id = os.environ.get("ROBODOJO_RUN_ID", "run")
        path = os.path.join(self.video_dir, f"episode_{run_id}_{cam}.mp4")
        self.writers[cam] = _H264Writer(path, self.fps, frame)
        self.paths[cam] = path

    def record(self, obs: dict) -> None:
        if not self.enabled:
            return
        try:
            vision = obs.get("vision", {})
            for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                color = vision.get(cam, {}).get("color")
                if color is None:
                    continue
                frame = np.asarray(color)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                self._ensure_writer(cam, frame)
                self.writers[cam].write(frame)
        except Exception as exc:  # noqa: BLE001 - recording must never break env
            logger.warning("video record warning: %s", exc)
            self.video_dir = None  # disable after first failure

    def close(self) -> list[str]:
        paths = list(self.paths.values())
        for writer in self.writers.values():
            try:
                writer.release()
            except Exception as error:
                logger.warning("video writer close failed: %s", error)
        self.writers.clear()
        self.paths.clear()
        return paths


def _record_obs_frame(recorder, obs: dict) -> None:
    recorder.record(obs)


def _synchronize_tiled_camera_apertures(env) -> list[dict[str, Any]]:
    """Make authored camera apertures match Isaac's square-pixel renderer."""
    try:
        camera_manager = env.camera_manager
        camera_names = camera_manager.camera_names
        cameras = camera_manager.cameras
    except AttributeError as error:
        raise RuntimeError("RoboDojo camera producer is unavailable") from error
    if len(cameras) != len(camera_names) or not cameras:
        raise RuntimeError("RoboDojo camera producer inventory is invalid")
    receipts = []
    for env_index, (environment_cameras, environment_names) in enumerate(
        zip(cameras, camera_names)
    ):
        if len(environment_cameras) != len(environment_names):
            raise RuntimeError("RoboDojo camera producer names do not match cameras")
        for camera_index, (name, camera) in enumerate(
            zip(environment_names, environment_cameras)
        ):
            resolution = np.asarray(camera.get_resolution(), dtype=np.float64)
            horizontal_aperture = float(camera.get_horizontal_aperture())
            previous_vertical_aperture = float(camera.get_vertical_aperture())
            previous_intrinsics = np.asarray(
                camera_manager.get_camera_intrinsics(camera_index, env_index),
                dtype=np.float64,
            )
            if (
                resolution.shape != (2,)
                or previous_intrinsics.shape != (3, 3)
                or not np.isfinite(resolution).all()
                or not np.isfinite(previous_intrinsics).all()
                or np.any(resolution <= 0)
                or not np.isfinite(horizontal_aperture)
                or horizontal_aperture <= 0
                or not np.isfinite(previous_vertical_aperture)
                or previous_vertical_aperture <= 0
            ):
                raise RuntimeError(f"RoboDojo camera aperture is invalid: {name}")
            width, height = resolution
            expected_vertical_aperture = horizontal_aperture * height / width
            camera.set_horizontal_aperture(horizontal_aperture, True)
            actual_vertical_aperture = float(camera.get_vertical_aperture())
            actual_intrinsics = np.asarray(
                camera_manager.get_camera_intrinsics(camera_index, env_index),
                dtype=np.float64,
            )
            if not np.isclose(
                actual_vertical_aperture,
                expected_vertical_aperture,
                rtol=1e-7,
                atol=1e-7,
            ) or (
                actual_intrinsics.shape != (3, 3)
                or not np.isfinite(actual_intrinsics).all()
                or not np.isclose(
                    actual_intrinsics[0, 0],
                    actual_intrinsics[1, 1],
                    rtol=1e-7,
                    atol=1e-7,
                )
            ):
                raise RuntimeError(
                    f"RoboDojo camera rejected square-pixel aperture: {name}"
                )
            receipts.append(
                {
                    "env_index": env_index,
                    "camera": str(name),
                    "resolution": [int(width), int(height)],
                    "horizontal_aperture": horizontal_aperture,
                    "previous_vertical_aperture": previous_vertical_aperture,
                    "vertical_aperture": actual_vertical_aperture,
                    "previous_intrinsic_matrix": previous_intrinsics.tolist(),
                    "intrinsic_matrix": actual_intrinsics.tolist(),
                    "source": "Replicator tiled render product square-pixel contract",
                }
            )
    return receipts


def _select_native_eval_layout(env, layout_set: int, layout: int) -> dict[str, Any]:
    """Select the native layout directory before resetting its indexed scene."""
    if (
        type(layout_set) is not int
        or layout_set < 0
        or type(layout) is not int
        or layout < 0
    ):
        raise ValueError("native layout_set and layout must be non-negative integers")
    try:
        seed_manager = env.seed_manager
        seed_manager.config["seed"] = layout_set
        seed_manager.init_eval()
        seed_record = seed_manager.seed_info[layout]
        scene_layout = seed_record["scene_layout"]
        loaded_layout = seed_manager.get_seed_scene_info(layout)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"native layout selection is unavailable: set={layout_set} layout={layout}"
        ) from error
    if int(seed_manager.eval_seed) != layout_set:
        raise RuntimeError("native SeedManager did not apply the requested layout_set")
    if not isinstance(loaded_layout, dict):
        raise RuntimeError("native SeedManager returned an invalid layout document")
    # Official task callbacks (for example support-arm trajectories) read the
    # EvalEnv field directly. The CollectEnv bridge must expose the same
    # already-validated layout-set identity before its first reset callback.
    env.eval_seed = layout_set
    return {
        "layout_set": layout_set,
        "layout": layout,
        "available_layouts": len(seed_manager.seed_info),
        "scene_layout_path": str(Path(str(scene_layout)).resolve()),
        "source": "native SeedManager eval layout directory",
    }


def _refresh_calibrated_camera_frames(env) -> dict[str, Any]:
    """Capture and validate one discarded RGB-D frame after SDK optics change."""
    try:
        # CollectEnv.get_obs is the native camera-only update path: it calls
        # ObsManager.render_for_capture() before reading all annotators and
        # does not advance physics, reset the layout, or submit an action.
        observation = env.get_obs(env_idx=0)
        vision = observation["vision"]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("RoboDojo calibrated camera refresh failed") from error
    cameras = {}
    for name, camera in vision.items():
        color = np.asarray(camera.get("color"))
        depth = camera.get("distance_to_image_plane")
        if depth is None:
            depth = camera.get("depth")
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        intrinsic = np.asarray(camera.get("intrinsic_matrix"), dtype=np.float64)
        if (
            color.ndim != 3
            or color.shape[2] not in {3, 4}
            or depth.shape != color.shape[:2]
            or intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
            or not np.isclose(intrinsic[0, 0], intrinsic[1, 1], rtol=1e-7, atol=1e-7)
        ):
            raise RuntimeError(f"RoboDojo calibrated camera frame is invalid: {name}")
        cameras[str(name)] = {
            "rgb_shape": list(color.shape),
            "depth_shape": list(depth.shape),
            "intrinsic_matrix": intrinsic.tolist(),
        }
    if not cameras:
        raise RuntimeError("RoboDojo calibrated camera refresh returned no cameras")
    return {
        "source": "native CollectEnv.get_obs render_for_capture path",
        "physical_actions": 0,
        "cameras": cameras,
    }


def _obs_dict(env, recorder) -> dict[str, Any]:
    from reweave.benchmarks.robodojo.access import MEASURED_GRIPPER_KEYS
    from reweave.benchmarks.robodojo.tasks import official_instruction

    # Keep native command/action and dataset semantics intact. The public state
    # adds a separate measured channel rather than replacing gripper targets.
    obs = dict(env.get_obs(env_idx=0))
    obs["state"] = dict(obs["state"])
    measured = {}
    for robot in env.robot_manager.robot_list:
        if robot.type != "target" or robot.ee_type != "gripper":
            continue
        arm = str(robot.arm_name).split("_")[0]
        if arm not in {"left", "right"}:
            continue
        key = f"{arm}_gripper_measured_opening"
        if key in measured:
            raise ValueError(f"duplicate RoboDojo measured gripper: {arm}")
        scale = np.asarray(robot.gripper_scale, dtype=np.float64)
        sign = robot.gripper_move["sign"]
        if scale.shape != (2,) or not np.isfinite(scale).all() or sign not in {-1, 1}:
            raise ValueError(f"invalid RoboDojo gripper calibration: {arm}")
        width = float(scale[1] - scale[0])
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"invalid RoboDojo gripper calibration: {arm}")
        actual = np.asarray(
            env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0],
            dtype=np.float64,
        )
        if actual.ndim != 1 or not actual.size or not np.isfinite(actual).all():
            raise ValueError(f"invalid RoboDojo measured gripper joint: {arm}")
        # Native control uses the first configured gripper joint and this same
        # scale/sign. Preserve finite overshoot instead of clipping evidence.
        opening = (
            (actual[0] - scale[0]) if sign == 1 else (scale[1] - actual[0])
        ) / width
        if not np.isfinite(opening):
            raise ValueError(f"invalid RoboDojo measured gripper opening: {arm}")
        measured[key] = [float(opening)]
    if set(measured) != set(MEASURED_GRIPPER_KEYS):
        raise ValueError("RoboDojo public observation requires both measured grippers")
    obs["state"].update(measured)
    obs["instruction"] = official_instruction(env)
    _record_obs_frame(recorder, obs)
    return obs


def _find_robot(env, arm: str):
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        if str(robot.arm_name).split("_")[0] == arm:
            return robot
    return None


def _native_arm_joint_limits(env, robot, env_idx: int = 0):
    """Return the simulator articulation limits for the selected arm."""
    manager = env.robot_manager
    keys = getattr(manager, "robot_key", None)
    if keys is None:
        raise ValueError("native robot manager does not expose articulation limits")
    try:
        articulation = keys[manager.robot_list.index(robot)]
        limits = articulation.data.soft_joint_pos_limits[env_idx][
            robot.arm_joint_indices
        ]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("native arm joint limits are unavailable") from error
    detach = getattr(limits, "detach", None)
    if callable(detach):
        limits = detach().cpu().numpy()
    limits = np.asarray(limits, dtype=np.float64)
    if (
        limits.shape != (len(robot.arm_joint_indices), 2)
        or not np.isfinite(limits).all()
    ):
        raise ValueError("native arm joint limits are invalid")
    return limits


def _solve_ik_pose(env, arm: str, pose: list) -> dict:
    """Solve a world XYZ + wxyz quaternion without relaxing orientation."""
    target = np.array(pose, dtype=np.float64, copy=True)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("pose must contain finite world XYZ and a wxyz quaternion")
    norm = float(np.linalg.norm(target[3:]))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("pose quaternion must have a finite nonzero norm")
    target[3:] /= norm
    robot = _find_robot(env, arm)
    if robot is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    result = env.robot_manager.solve_ik(
        target_pose=target.tolist(), env_idx=0, robot=robot
    )
    if result.get("status") != "Success":
        return {"status": "Fail", "error": result.get("error", "full-pose IK failed")}
    joints = np.asarray(result["joint_value"], dtype=np.float64)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("full-pose IK returned invalid arm joint positions")
    limits = _native_arm_joint_limits(env, robot)
    outside = np.flatnonzero(
        (joints < limits[:, 0] - 1e-5) | (joints > limits[:, 1] + 1e-5)
    )
    if outside.size:
        joint_names = [robot.arm_joints_name[index] for index in outside]
        return {
            "status": "Fail",
            "error": "IK solution exceeds native articulation limits",
            "joints": joint_names,
        }
    return {"status": "Success", "arm": arm, "joint_value": joints.tolist()}


def _native_planner_timing(manager, planner):
    planner_dt, physics_dt = float(planner.dt), float(manager.dt)
    if not np.isfinite(planner_dt) or planner_dt <= 0 or planner_dt != physics_dt:
        raise ValueError(
            "native planner timestep differs from the physics control timestep"
        )


def _plan_pose_path(
    env,
    arm: str,
    poses: list,
    linear_axis: str | None = None,
    *,
    position_only: bool = False,
) -> dict:
    """Plan collision-aware native joint trajectories through world-frame poses."""
    robot = _find_robot(env, arm)
    if robot is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    targets = np.array(poses, dtype=np.float64, copy=True)
    if targets.ndim != 2 or targets.shape[1:] != (7,) or not targets.shape[0]:
        raise ValueError("poses must be a nonempty Nx7 array")
    if not np.isfinite(targets).all():
        raise ValueError("poses must contain only finite values")
    norms = np.linalg.norm(targets[:, 3:], axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("pose quaternions must have finite nonzero norms")
    targets[:, 3:] /= norms[:, None]
    if linear_axis not in {None, "x"}:
        raise ValueError("linear_axis must be null or native X5 axis 'x'")
    if linear_axis is not None and len(targets) != 1:
        raise ValueError("linear native pose planning accepts exactly one target")

    manager = env.robot_manager
    planner = manager.planner[robot.robot_name]
    _native_planner_timing(manager, planner)
    planner_world = {
        "source": "RoboDojo configured CuRobo planner",
        "robot_root_pose_world": list(robot.entity_origin_pose),
    }
    current = np.asarray(
        manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float64
    )
    limits = _native_arm_joint_limits(env, robot)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    raw_planner_results: list[dict[str, Any]] = []

    def plan_with_diagnostics(current_joints, target_pose):
        original_run = planner._run_with_constraint
        captured = {}

        def capture_raw(*args, **kwargs):
            raw = original_run(*args, **kwargs)
            if raw is None:
                captured.update({"result_type": None, "status": None})
                return raw
            captured["result_type"] = type(raw).__name__
            captured["status"] = (
                None if getattr(raw, "status", None) is None else str(raw.status)
            )
            captured["planning_time_s"] = float(getattr(raw, "planning_time", 0.0))
            success = getattr(raw, "success", None)
            if hasattr(success, "detach"):
                success = success.detach().cpu().numpy()
            captured["success"] = (
                None if success is None else np.asarray(success).astype(bool).tolist()
            )
            return raw

        planner._run_with_constraint = capture_raw
        try:
            result = planner.plan_path(
                current_joints,
                target_pose,
                real_robot_pose=list(robot.entity_origin_pose),
                **(
                    {"constraint_pose": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}
                    if position_only
                    else {}
                ),
            )
        finally:
            planner._run_with_constraint = original_run
        raw_planner_results.append(captured)
        return result

    for index, target in enumerate(targets):
        if linear_axis is None:
            result = plan_with_diagnostics(current, target.tolist())
        else:
            from curobo.types import ToolPoseCriteria

            criteria = ToolPoseCriteria.linear_motion(
                axis=linear_axis,
                non_terminal_scale=1.0,
                project_distance_to_goal=True,
            )
            planner.motion_planner.update_tool_pose_criteria(
                dict.fromkeys(planner.tool_frames, criteria)
            )
            try:
                result = plan_with_diagnostics(current, target.tolist())
            finally:
                planner.motion_planner.update_tool_pose_criteria(
                    planner._default_tool_pose_criteria
                )
        if result is None or result.get("status") != "Success":
            return {
                "status": "Fail",
                "error": "native collision-aware pose planning failed",
                "planner_status": raw_planner_results[-1].get("status"),
                "reason_known": raw_planner_results[-1].get("status") is not None,
                "pose_index": index,
                "raw_planner_result": raw_planner_results[-1],
            }
        segment_positions = np.asarray(result.get("position"), dtype=np.float64)
        segment_velocities = np.asarray(result.get("velocity"), dtype=np.float64)
        if (
            segment_positions.ndim != 2
            or segment_positions.shape[1:] != (6,)
            or segment_velocities.shape != segment_positions.shape
            or not segment_positions.shape[0]
            or not np.isfinite(segment_positions).all()
            or not np.isfinite(segment_velocities).all()
        ):
            raise ValueError("native pose planner returned an invalid trajectory")
        if np.any(segment_positions < limits[:, 0] - 1e-5) or np.any(
            segment_positions > limits[:, 1] + 1e-5
        ):
            raise ValueError("native pose planner exceeded articulation limits")
        if positions and np.allclose(positions[-1], segment_positions[0], atol=1e-7):
            segment_positions = segment_positions[1:]
            segment_velocities = segment_velocities[1:]
        positions.extend(segment_positions)
        velocities.extend(segment_velocities)
        current = np.asarray(result["position"][-1], dtype=np.float64)
    return {
        "status": "Success",
        "arm": arm,
        "planner_dt_s": float(planner.dt),
        "control_period_s": float(manager.dt * env.obs_manager.collect_interval),
        "linear_axis": linear_axis,
        "planner_world": planner_world,
        "raw_planner_results": raw_planner_results,
        "joint_positions": np.asarray(positions, dtype=np.float64).tolist(),
        "joint_velocities": np.asarray(velocities, dtype=np.float64).tolist(),
    }


def _plan_position_path(env, arm: str, xyz: list) -> dict:
    """Plan one official collision-aware trajectory with XYZ-only pose criteria."""
    target = np.asarray(xyz, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("xyz must contain three finite world coordinates")
    if _find_robot(env, arm) is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    observed = np.asarray(
        env.get_obs(env_idx=0)["state"][f"{arm}_ee_pose"], dtype=np.float64
    )
    if observed.shape != (7,) or not np.isfinite(observed).all():
        raise ValueError("native measured EE pose must be finite XYZ + wxyz")
    pose = np.concatenate((target, observed[3:])).tolist()
    return _plan_pose_path(env, arm, [pose], position_only=True)


def _plan_joint_path(env, arm: str, joints: list) -> dict:
    """Use the official configured plan_cspace trajectory, never a bare IK target."""
    target = np.asarray(joints, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("joint goal must contain six finite positions")
    robot = _find_robot(env, arm)
    if robot is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    manager = env.robot_manager
    limits = _native_arm_joint_limits(env, robot)
    if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
        return {
            "status": "Fail",
            "error": "joint goal exceeds native articulation limits",
        }
    current = np.asarray(
        manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float64
    )
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("native measured joints must contain six finite positions")
    planner = manager.planner[robot.robot_name]
    _native_planner_timing(manager, planner)
    original_plan = planner.motion_planner.plan_cspace
    captured = {}

    def capture_raw(*args, **kwargs):
        raw = original_plan(*args, **kwargs)
        captured["result_type"] = None if raw is None else type(raw).__name__
        captured["status"] = (
            None
            if raw is None or getattr(raw, "status", None) is None
            else str(raw.status)
        )
        success = getattr(raw, "success", None)
        if hasattr(success, "detach"):
            success = success.detach().cpu().numpy()
        captured["success"] = (
            None if success is None else np.asarray(success).astype(bool).tolist()
        )
        captured["planning_time_s"] = float(getattr(raw, "planning_time", 0.0))
        return raw

    planner.motion_planner.plan_cspace = capture_raw
    try:
        result = planner.plan_joint(current, target.tolist())
    finally:
        planner.motion_planner.plan_cspace = original_plan
    if result is None or result.get("status") != "Success":
        return {
            "status": "Fail",
            "error": "native collision-aware joint planning failed",
            "planner_status": captured.get("status"),
            "reason_known": captured.get("status") is not None,
            "raw_planner_result": captured,
        }
    positions = np.asarray(result.get("position"), dtype=np.float64)
    velocities = np.asarray(result.get("velocity"), dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (6,)
        or not len(positions)
        or velocities.shape != positions.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
    ):
        raise ValueError("native joint planner returned an invalid trajectory")
    if np.any(positions < limits[:, 0] - 1e-5) or np.any(
        positions > limits[:, 1] + 1e-5
    ):
        raise ValueError("native joint planner exceeded articulation limits")
    return {
        "status": "Success",
        "arm": arm,
        "planner_dt_s": float(planner.dt),
        "control_period_s": float(manager.dt * env.obs_manager.collect_interval),
        "linear_axis": None,
        "planner_world": {
            "source": "RoboDojo configured CuRobo planner",
            "robot_root_pose_world": list(robot.entity_origin_pose),
        },
        "raw_planner_results": [captured],
        "joint_positions": positions.tolist(),
        "joint_velocities": velocities.tolist(),
    }


def _apply_trajectory_window(env, arm: str, positions, velocities, gripper=0) -> int:
    """Execute original native waypoints through the official physics control queue.

    This deliberately avoids CollectEnv.apply_target's policy-target ramp.
    One bounded window consumes one existing environment control-step budget;
    a final short window consumes the same unit without inventing padded points.
    """
    from copy import deepcopy

    robot = _find_robot(env, arm)
    if robot is None:
        raise ValueError(f"robot for arm {arm!r} not found")
    points = np.asarray(positions, dtype=np.float64)
    speeds = np.asarray(velocities, dtype=np.float64)
    interval = int(env.obs_manager.collect_interval)
    if (
        points.ndim != 2
        or points.shape[1:] != (6,)
        or not 1 <= len(points) <= interval
        or speeds.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(speeds).all()
        or not np.isfinite(gripper)
    ):
        raise ValueError(
            "trajectory window requires finite native position/velocity rows within one control interval"
        )
    manager = env.robot_manager
    _native_planner_timing(manager, manager.planner[robot.robot_name])
    limits = _native_arm_joint_limits(env, robot)
    if np.any(points < limits[:, 0] - 1e-5) or np.any(points > limits[:, 1] + 1e-5):
        raise ValueError("trajectory window exceeds native articulation limits")
    queue = manager.control_manager
    if queue.get_empty([0]) != [0]:
        raise RuntimeError(
            "native control queue is not settled before trajectory submission"
        )
    plan = manager.plan_ee(
        env_idx=0,
        arm_tag=robot.arm_name,
        result={"status": "Success", "position": points, "velocity": speeds},
    )
    if plan is None or len(plan) != len(points):
        raise ValueError(
            "official native plan_ee did not preserve the trajectory window"
        )
    # Hold the last commanded target, rather than changing the other arm's
    # target to a lagging measurement. Only the first command needs measured
    # state to seed an otherwise empty official control history.
    holds = {}
    previous = queue.prev_control[0]
    for target_robot in manager.robot_list:
        if target_robot.type != "target":
            continue
        arm_key = manager.process_name(target_robot.arm_name)
        grip_key = manager.process_name(target_robot.gripper_name)
        arm_position = previous.get(arm_key, {}).get("position")
        if arm_position is None:
            arm_position = manager.get_joint(target_robot, env_idx_list=[0])[0]
        grip_position = previous.get(grip_key, {}).get("position")
        if grip_position is None:
            grip_position = manager.get_end_effector_real_val(
                target_robot, env_idx_list=[0]
            )[0]
        holds[arm_key] = {
            "position": list(arm_position),
            "velocity": [0.0] * len(arm_position),
        }
        holds[grip_key] = {
            "position": list(grip_position),
            "velocity": [0.0] * len(grip_position),
        }
    grip_plan = None
    if gripper != 0:
        grip_plan = manager.plan_endeffector_joint(
            env_idx=0,
            arm_tag=robot.arm_name,
            target_val=[0.0 if gripper > 0 else 1.0],
            control_step_num=len(points),
        )
        if grip_plan is None or len(grip_plan) != len(points):
            raise ValueError(
                "official gripper planner did not preserve the trajectory window"
            )
    sequence = []
    support = env.support_arm_action[0] if hasattr(env, "support_arm_action") else None
    for index, point in enumerate(plan):
        control = deepcopy(holds)
        if support:
            control.update(support.pop(0))
        control.update(point)
        if grip_plan is not None:
            control.update(grip_plan[index])
        sequence.append(control)
    # Same control-call counter, reward and support-arm boundary as apply_action
    # -> CollectEnv.apply_target. Every physics row uses the SDK's queue/step.
    env.take_action_cnt[0] += 1
    queue.push([0], [sequence])
    while queue.get_empty([0]) != [0]:
        env.step([0])
    env.reward_manager.step([0])
    if getattr(env, "interact", False):
        if hasattr(env, "query_support_arm_traj"):
            env.query_support_arm_traj(env_idx=0)
        if hasattr(env, "check_support_arm_stable"):
            env.check_support_arm_stable(env_idx=0)
    return len(points)
