# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""RoboDojo tool specs and backend-specific control adapters."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from reweave.benchmarks.robodojo.perception import _jsonable
from reweave.robots.components.gripper_feedback import GripperSettling
from reweave.tools.toolkit import readonly

# (camera key in the obs dict, artifact base name). Mirrors the ``frame_channels``
# declared in ``robot_spec.ROBODOJO_DASHBOARD_SPEC``.
CAMERA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("cam_head", "cam_head.png"),
    ("cam_left_wrist", "cam_left_wrist.png"),
    ("cam_right_wrist", "cam_right_wrist.png"),
)

TOOLS_SPEC: list[dict] = [
    {
        "name": "locate",
        "description": "Segment once and measure PyRUA-Lean object geometry from the same server RGB-D snapshot. Returns footprint center, dimensions, long axis, world closing axis, top and depth quality. support_z enables upright-on-plane body height and round-side completion; omit for visible surface only. These are measurements/model fits, not verified grasp poses or object identity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string", "minLength": 1},
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                },
                "min_score": {"type": "number", "minimum": 0, "maximum": 1},
                "support_z": {
                    "type": "number",
                    "description": "Known world support-plane height in metres; never guess for stacked, held or suspended objects.",
                },
            },
            "required": ["text_prompt"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_env_state",
        "description": (
            "Read one recorded RoboDojo state: measured arm joints and world poses "
            "of each native configured EE frame (*_ee_pose, the same frame targeted "
            "by move_to/move_pose; not guaranteed to be a fingertip/pinch contact "
            "point or calibrated TCP), "
            "native command-derived *_ee_joint_state opening targets, independent "
            "*_gripper_measured_opening (dimensionless, 1=open/0=closed from actual "
            "joints and runtime calibration; not grip force or possession), camera "
            "shapes and calibration, the task instruction, and the three "
            "camera RGB images (cam_head, cam_left_wrist, cam_right_wrist). "
            "Step -1 selects the latest record. The episode's fixed task-rule "
            "supplement is delivered beside the verbatim official instruction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "description": "Recorded step to read (-1 = latest)",
                }
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Use depth and calibration from the current server-retained snapshot; original RGB-D stays on the simulator server. "
            "Project one pixel from a camera image to the visible surface world xyz "
            "using depth + intrinsics/extrinsics. This surface point is not converted "
            "into an EE target or fingertip/pinch contact point."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Pixel row (0=top)"},
                "col": {"type": "integer", "description": "Pixel col (0=left)"},
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "description": "Camera to project from",
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "segment",
        "description": (
            "Run SAM on original RGB on the simulator server without transferring RGB-D to the Host. "
            "Segment an object in a camera image by text prompt (SAM 3.0). "
            "Returns bounding box, score, true centroid_rc and mask_point_rc: "
            "the integer [row, col] foreground pixel nearest the centroid. "
            "Use mask_point_rc for depth lookup; the centroid may be in a hole. "
            "The mask sample does not establish a grasp point or object identity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string", "description": "e.g. 'the bottle'"},
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "description": "Camera to segment (default cam_head)",
                },
                "min_score": {
                    "type": "number",
                    "description": "Min score (default 0.2)",
                },
            },
            "required": ["text_prompt"],
        },
    },
    {
        "name": "generate_grasps",
        "description": (
            "Run separately budgeted SAM and GraspGen-X on the same server-retained RGB-D snapshot; only proposals and geometry return. "
            "Generate scored GraspGen-X control-frame pose candidates for one X5 "
            "arm, performing SAM3 internally on the same live measured RGB-D snapshot. "
            "Candidates are in isaac_world and already include the configured "
            "grasp-frame-to-control-frame transform; do not add another TCP offset. "
            "Convert quaternion_xyzw [x,y,z,w] to move_pose quaternion_wxyz [w,x,y,z]. "
            "approach_direction is a world vector. Candidates follow the official "
            "GraspGen-X score ordering; reachability, contact and scene/path collision "
            "remain unchecked until planning and execution. Inspect target identity "
            "and provenance; scores are not calibrated success probabilities. "
            "This read-only tool executes no action."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text_prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Text identifying the visible target for SAM3.",
                },
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Configured X5 arm/gripper binding to score.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "description": "RGB-D camera (default cam_head).",
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
            "required": ["text_prompt", "arm"],
        },
    },
    {
        "name": "move_to",
        "description": (
            "Move an arm native configured EE frame (the frame reported by "
            "*_ee_pose) to a world xyz target using one native collision-aware CuRobo position trajectory. Its origin is not "
            "guaranteed to be a fingertip/pinch contact point or calibrated TCP. "
            "gripper: 1=close, -1=open, 0=keep. An explicit gripper "
            "command still executes when the position is already reached. Executes every native position/velocity sample once; the complete path must fit max_steps and the remaining episode budget. Reports residual error without replaying terminal targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "World xyz target for the native configured EE frame reported by *_ee_pose",
                },
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Arm to move",
                },
                "gripper": {
                    "type": "number",
                    "description": "1=close, -1=open, 0=reuse native command-derived opening target (default 0); native control may limit and update it relative to the actual joint during motion, so verify opening and object retention. A nonzero command is sent with every control window of the path, i.e. from its first action, not after arrival.",
                },
                "tol": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Position tolerance in metres (default 0.01)",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on acknowledged actions for the complete path. Omit to allow up to the remaining episode budget; an explicit value is preserved and a longer path is rejected before any action.",
                },
            },
            "required": ["arm", "xyz"],
        },
    },
    {
        "name": "move_pose",
        "description": (
            "Move one native configured EE frame (the frame reported by *_ee_pose) "
            "to world XYZ and quaternion_wxyz using a native collision-aware "
            "CuRobo arm trajectory, without terminal-IK or position-only fallback. "
            "Normalizes the nonzero quaternion. Holds the other arm/gripper; "
            "gripper: 1=close, -1=open, 0=keep. Returns target/start/final_pose "
            "(XYZ + wxyz), position_error_m, rotation_error_rad (shortest rotation), "
            "steps_used, planned_actions, reached and ik_error. reached requires both measured "
            "errors within tolerance. Remaining error or IK failure is feedback "
            "for replanning; an explicit gripper command executes even at the target. "
            "linear=true uses the native linear-motion criterion along link6 +X. "
            "The complete collision-checked native waypoint path must fit max_steps "
            "and the shared remaining episode budget before the first physical action. "
            "All pose/error fields refer to this EE frame; its origin is not guaranteed "
            "to be a fingertip/pinch contact point or calibrated TCP. No surface-to-EE "
            "offset is applied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Native configured EE frame position in world metres, as in *_ee_pose.",
                },
                "quaternion_wxyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "World quaternion of the native configured EE frame [w,x,y,z]; nonzero and normalized internally.",
                },
                "tol_m": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Allowed measured position error in metres.",
                },
                "tol_rad": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Allowed measured shortest rotation error in radians.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on acknowledged actions for the complete path. Omit to allow up to the remaining episode budget; an explicit value is preserved and a longer path is rejected before any action.",
                },
                "gripper": {
                    "type": "number",
                    "description": "1=close, -1=open, 0=reuse native command-derived opening target (default 0); native control may limit and update it relative to the actual joint during motion, so verify opening and object retention. A nonzero command is sent with every control window of the path, i.e. from its first action, not after arrival.",
                },
                "linear": {
                    "type": "boolean",
                    "description": "Use the official linear-motion criterion along native link6 +X (default false).",
                },
            },
            "required": [
                "arm",
                "xyz",
                "quaternion_wxyz",
                "tol_m",
                "tol_rad",
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "move_joints",
        "description": (
            "Move one arm to six explicit joint positions in radians using "
            "acknowledged actions and current measured joint feedback. Holds "
            "the other arm and its gripper at their observed targets. "
            "gripper: 1=close, -1=open, 0=keep. To return to a previously "
            "observed configuration, supply its public arm_joint_state. "
            "Uses one official native joint-space trajectory and executes all position/velocity samples once, without bare-goal replay. The complete path must fit max_steps and the remaining episode budget. Returns target/start/final_joint_positions, signed joint_errors_rad "
            "(target minus actual), max_joint_error_rad, steps_used and reached. "
            "reached compares the largest absolute joint error with tol_rad; "
            "remaining error after the path is feedback for further planning; a tiny tolerance does not turn this into a waiting loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Arm whose six public joints receive the target.",
                },
                "joint_positions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                    "description": "Six target arm angles in radians, in observed arm_joint_state order.",
                },
                "tol_rad": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Largest allowed absolute joint error in radians for reached=true.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on acknowledged actions for the complete path. Omit to allow up to the remaining episode budget; an explicit value is preserved and a longer path is rejected before any action.",
                },
                "gripper": {
                    "type": "number",
                    "description": "1=close, -1=open, 0=reuse native command-derived opening target (default 0); native control may limit and update it relative to the actual joint during motion, so verify opening and object retention. A nonzero command is sent with every control window of the path, i.e. from its first action, not after arrival.",
                },
            },
            "required": ["arm", "joint_positions", "tol_rad"],
            "additionalProperties": False,
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Command one gripper while holding both observed arm joint targets and the other gripper. 1=close, -1=open. max_steps defaults to one ACK; request up to 10 for a PyRUA-Lean-style measured stroke, stopping when actual opening changes less than 0.004 after at least two actions. Returns measured opening and whether it stabilized; no aperture reading means stop after the first ACK. Stability does not establish contact or possession."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                },
                "gripper": {
                    "type": "number",
                    "enum": [-1, 1],
                    "description": "1=close, -1=open",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Upper bound on acknowledged actions (default 1). Use 10 to finish a measured stroke in one call; every action consumes the episode budget.",
                },
            },
            "required": ["arm", "gripper"],
        },
    },
    {
        "name": "pi0_pick",
        "description": (
            "Each policy forward reads the current server-local RGB-D snapshot; physical ACKs contain state and observation references only. "
            "Run Pi_05 policy chunks on current server-local observations until the explicit chunk limit, official terminal state, episode budget or cancellation. No height/gripper heuristic establishes possession or task success."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional specific target instruction; omit to use the official task instruction",
                },
                "arm": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": (
                        "Reference arm reported in feedback. The policy controls "
                        "both arms and the motion heuristic monitors either arm."
                    ),
                },
                "max_chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum policy forwards; each action consumes the shared physical budget (default 8)",
                },
            },
            "required": [],
        },
    },
]


@readonly
def view_env_state(step: int = -1, *, state, include_images=True) -> dict:
    """Read one recorded RoboDojo state plus its camera artifacts.

    ``step`` selects the record (``-1`` = latest). The three camera RGB PNGs
    saved with that record are embedded as ``_image_bytes`` / ``_image_cam_bytes``
    / ``_image_wrist_bytes`` so the planner receives them as image blocks.
    """
    try:
        record = state.get(step)
    except Exception as error:
        return {"error": f"state step not available: {error}"}
    result: dict[str, Any] = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "observation_ref": record.extras.get("observation_ref"),
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
        "official_task_instruction": record.extras.get("official_task_instruction"),
        "task_rule_supplement": record.extras.get("task_rule_supplement"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    if not include_images:
        return result
    for slot, (camera, artifact) in zip(
        ("_image_bytes", "_image_cam_bytes", "_image_wrist_bytes"),
        CAMERA_ARTIFACTS,
    ):
        if artifact in record.artifacts:
            result[slot] = state.load_bytes(artifact, step=record.step_idx)
    return result


def _summarize_obs(obs: dict) -> dict:
    from reweave.benchmarks.robodojo.access import public_observation

    obs = public_observation(obs)
    vision = obs.get("vision", {})
    state_data = obs.get("state", {})
    return {
        "instruction": obs.get("instruction"),
        "observation_ref": obs.get("observation_ref"),
        "service_instance_id": obs.get("service_instance_id"),
        "physical_step": obs.get("physical_step"),
        "cameras": {
            name: {
                "shape": cam.get("shape"),
                "has_depth": cam.get("has_depth", False),
                "intrinsic_matrix": _jsonable(cam.get("intrinsic_matrix")),
                "extrinsic_matrix": _jsonable(cam.get("extrinsic_matrix")),
                "color_dtype": cam.get("color_dtype"),
            }
            for name, cam in vision.items()
        },
        "state": _jsonable(state_data),
        "eef": _jsonable(
            {
                "left": state_data.get("left_ee_pose"),
                "right": state_data.get("right_ee_pose"),
            }
        ),
    }


@readonly
def back_project(primitives, state, row, col, camera="cam_head") -> dict:
    """Project a pixel using the current server-retained RGB-D snapshot."""
    del state
    ref = _observation_ref(primitives)
    return primitives.env.back_project_ref(ref, row=row, col=col, camera=camera)


@readonly
def segment(primitives, state, text_prompt, camera="cam_head", min_score=0.2) -> dict:
    """Run SAM on original server-local RGB; return geometry and a mask reference."""
    del state
    with primitives.model_request("sam3"):
        return primitives.env.segment_ref(
            _observation_ref(primitives),
            text_prompt=text_prompt,
            camera=camera,
            min_score=min_score,
        )


@readonly
def locate(
    primitives, state, text_prompt, camera="cam_head", min_score=0.2, support_z=None
) -> dict:
    """Measure PyRUA-Lean geometry without transferring raw RGB-D to the Host."""
    del state
    with primitives.model_request("sam3"):
        return primitives.env.locate_ref(
            _observation_ref(primitives),
            text_prompt=text_prompt,
            camera=camera,
            min_score=min_score,
            support_z=support_z,
        )


@readonly
def generate_grasps(
    primitives,
    state,
    text_prompt,
    arm,
    camera="cam_head",
    min_score=0.2,
    max_candidates=5,
) -> dict:
    """Run separately budgeted SAM and X5 inference on one server snapshot."""
    del state
    if primitives.graspgen_client is None:
        raise RuntimeError("GraspGen-X is not configured")
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        raise ValueError("text_prompt must be non-empty")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
        raise ValueError("max_candidates must be an integer between 1 and 32")
    ref = _observation_ref(primitives)
    with primitives.model_request("sam3"):
        segmentation = primitives.env.segment_ref(
            ref, text_prompt=text_prompt.strip(), camera=camera, min_score=min_score
        )
    if not segmentation.get("found") or not segmentation.get("mask_ref"):
        raise RuntimeError(segmentation.get("reason") or "SAM3 found no target mask")
    with primitives.model_request("graspgen"):
        return primitives.env.grasps_ref(
            ref, segmentation["mask_ref"], arm=arm, max_candidates=max_candidates
        )


def _observation_ref(primitives) -> str:
    ref = _current_obs(primitives).get("observation_ref")
    if not isinstance(ref, str) or not ref:
        raise RuntimeError("RoboDojo requires a settled server observation reference")
    return ref


def _arm_ee_pose_key(arm: str) -> str:
    return f"{arm}_ee_pose"


def _arm_ee_joint_key(arm: str) -> str:
    return f"{arm}_ee_joint_state"


def _current_obs(primitives) -> dict:
    """Use the last settled ACK: this single-owner simulator advances only on step.

    Every step replaces env.last_obs; no read RPC is needed between actions.
    An unavailable observation (including an unknown action outcome) must never
    fall back to primitives' older image or proprioception snapshot.
    """
    obs = primitives.env.last_obs
    if not isinstance(obs, dict):
        raise RuntimeError("RoboDojo has no settled current observation")
    primitives._last_obs = obs
    return obs


def _joint_hold_action(obs: dict) -> dict:
    """Hold the observed native joint targets without invoking pose IK."""
    from reweave.benchmarks.robodojo.access import JOINT_ACTION_FIELDS, native_action

    return native_action({name: obs["state"][name] for name, _ in JOINT_ACTION_FIELDS})


def _validate_motion(arm, max_steps, gripper):
    if arm not in {"left", "right"} or (
        max_steps is not None and (type(max_steps) is not int or max_steps <= 0)
    ):
        raise ValueError("motion requires a valid arm and a positive max_steps or none")
    if not np.isfinite(gripper):
        raise ValueError("motion requires a finite gripper command")


def _measured_vector(obs, key, length):
    vector = np.asarray(obs["state"][key], dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"motion requires a finite measured {key}")
    return vector


def _gripper_opening(obs, arm):
    # Only the independently measured joint channel may drive convergence.
    return obs["state"].get(f"{arm}_gripper_measured_opening")


def _motion_feedback(primitives, arm, result, reached):
    env = primitives.env
    reason = result.get("stop_reason")
    if reason is None:
        reason = (
            "target_reached"
            if reached
            else "environment_terminal"
            if env.terminated
            else "episode_budget_exhausted"
            if env.truncated or env.steps_remaining() <= 0
            else "trajectory_completed_with_residual"
        )
    opening = GripperSettling(
        _gripper_opening(_current_obs(primitives), arm), tolerance=0.004
    )
    return {
        "stop_reason": reason,
        "remaining_steps": env.steps_remaining(),
        "gripper_feedback": opening.feedback(unit="normalized_opening"),
    }


def _execute_native_path(primitives, arm, plan, *, max_steps, gripper):
    """Execute every native position/velocity sample once, with one ACK per window.

    No endpoint-only ramp, terminal-goal replay or residual threshold changes the
    native trajectory. Caller tolerances classify the final measured feedback.
    """
    result = {
        "steps_used": 0,
        "planned_actions": 0,
        "ik_error": None,
        "planner_status": plan.get("planner_status"),
        "reason_known": plan.get("reason_known", False),
        "planner_dt_s": None,
        "control_period_s": None,
        "terminal_target_joints": None,
        "last_commanded_joints": None,
    }
    if result["planner_status"] is not None and not isinstance(
        result["planner_status"], str
    ):
        raise ValueError("native planner status must be a string or null")
    if type(result["reason_known"]) is not bool:
        raise ValueError("native planner reason_known must be boolean")
    if plan.get("status") != "Success":
        result["ik_error"] = plan.get("error", "native trajectory planning failed")
        result["stop_reason"] = "planning_failed"
        return result
    positions = np.asarray(plan.get("joint_positions"), dtype=np.float64)
    velocities = np.asarray(plan.get("joint_velocities"), dtype=np.float64)
    planner_dt = float(plan.get("planner_dt_s", 0.0))
    control_period = float(plan.get("control_period_s", 0.0))
    if (
        positions.ndim != 2
        or positions.shape[1:] != (6,)
        or not len(positions)
        or velocities.shape != positions.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
        or not np.isfinite(planner_dt)
        or planner_dt <= 0
        or not np.isfinite(control_period)
        or control_period <= 0
    ):
        raise ValueError("native planner returned an invalid path")
    ratio = control_period / planner_dt
    window_size = int(round(ratio))
    if window_size < 1 or not np.isclose(ratio, window_size, atol=1e-6):
        raise ValueError(
            "native control period is not an integer planner timestep multiple"
        )
    planned_actions = (len(positions) + window_size - 1) // window_size
    result.update(
        planned_actions=planned_actions,
        planner_dt_s=planner_dt,
        control_period_s=control_period,
        terminal_target_joints=positions[-1].tolist(),
    )
    remaining = primitives.env.steps_remaining()
    if max_steps is not None and planned_actions > max_steps:
        result["ik_error"] = (
            f"native collision-aware path requires {planned_actions} "
            f"acknowledged actions; max_steps={max_steps}"
        )
        result["stop_reason"] = "local_budget_insufficient"
        return result
    if planned_actions > remaining:
        result["ik_error"] = (
            f"native collision-aware path requires {planned_actions} "
            f"acknowledged actions; remaining episode budget={remaining}"
        )
        result["stop_reason"] = "episode_budget_insufficient"
        return result
    for offset in range(0, len(positions), window_size):
        primitives._check_cancelled()
        if primitives.env.terminated or primitives.env.truncated:
            break
        _current_obs(primitives)
        primitives._check_cancelled()
        window_positions = positions[offset : offset + window_size]
        window_velocities = velocities[offset : offset + window_size]
        _obs, _reward, done, _info = primitives.env.step_trajectory(
            arm, window_positions.tolist(), window_velocities.tolist(), gripper=gripper
        )
        result["steps_used"] += 1
        result["last_commanded_joints"] = window_positions[-1].tolist()
        if done:
            break
    return result


def _execute_reached_gripper(primitives, arm, gripper):
    """An explicit gripper command at an already-reached target is one hold ACK."""
    if gripper == 0:
        return 0
    primitives._check_cancelled()
    action = _joint_hold_action(_current_obs(primitives))
    action[_arm_ee_joint_key(arm)] = [0.0 if gripper > 0 else 1.0]
    primitives._check_cancelled()
    primitives.env.step(action)
    return 1


def _idle_path_result():
    return {
        "steps_used": 0,
        "planned_actions": 0,
        "ik_error": None,
        "planner_status": None,
        "reason_known": False,
        "planner_dt_s": None,
        "control_period_s": None,
        "terminal_target_joints": None,
        "last_commanded_joints": None,
    }


def move_to(
    primitives,
    state,
    xyz,
    arm,
    gripper=0,
    tol=0.01,
    max_steps=None,
):
    """Plan once with native XYZ criteria; execute the full position trajectory."""
    _validate_motion(arm, max_steps, gripper)
    target = np.asarray(xyz, dtype=np.float64)
    if (
        target.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("move_to requires finite XYZ and a positive finite tolerance")
    primitives._check_cancelled()
    start = _measured_vector(_current_obs(primitives), _arm_ee_pose_key(arm), 7)[
        :3
    ].copy()
    result = _idle_path_result()
    if np.linalg.norm(target - start) > tol:
        primitives._check_cancelled()
        plan = primitives.env.plan_position_path(arm, target.tolist())
        primitives._check_cancelled()
        result = _execute_native_path(
            primitives, arm, plan, max_steps=max_steps, gripper=gripper
        )
    else:
        result["steps_used"] = _execute_reached_gripper(primitives, arm, gripper)
    final = _measured_vector(_current_obs(primitives), _arm_ee_pose_key(arm), 7)[:3]
    distance = float(np.linalg.norm(target - final))
    return {
        **result,
        "arm": arm,
        "target_xyz": target.tolist(),
        "start_xyz": start.tolist(),
        "final_xyz": final.tolist(),
        "dist_to_target_m": distance,
        "reached": distance <= tol,
        **_motion_feedback(primitives, arm, result, distance <= tol),
    }


def move_pose(
    primitives,
    state,
    arm,
    xyz,
    quaternion_wxyz,
    tol_m,
    tol_rad,
    max_steps=None,
    gripper=0,
    linear=False,
):
    """Execute the official native pose trajectory; report measured final errors."""
    _validate_motion(arm, max_steps, gripper)
    if type(linear) is not bool:
        raise ValueError("move_pose linear must be boolean")
    target_xyz = np.asarray(xyz, dtype=np.float64)
    target_q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if (
        target_xyz.shape != (3,)
        or target_q.shape != (4,)
        or not np.isfinite(target_xyz).all()
        or not np.isfinite(target_q).all()
    ):
        raise ValueError("move_pose requires finite XYZ and a wxyz quaternion")
    norm = float(np.linalg.norm(target_q))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("move_pose quaternion must have a finite nonzero norm")
    if not np.isfinite(tol_m) or tol_m <= 0 or not np.isfinite(tol_rad) or tol_rad <= 0:
        raise ValueError("move_pose requires positive finite tolerances")
    target = np.concatenate((target_xyz, target_q / norm))

    def measured_pose(obs):
        pose = _measured_vector(obs, _arm_ee_pose_key(arm), 7).copy()
        q_norm = float(np.linalg.norm(pose[3:]))
        if q_norm == 0 or not np.isfinite(q_norm):
            raise ValueError("move_pose requires a nonzero measured quaternion")
        pose[3:] /= q_norm
        return pose

    def errors(pose):
        position = float(np.linalg.norm(target[:3] - pose[:3]))
        dot = float(np.clip(abs(np.dot(target[3:], pose[3:])), 0.0, 1.0))
        return position, float(2 * np.arccos(dot))

    primitives._check_cancelled()
    start = measured_pose(_current_obs(primitives))
    position_error, rotation_error = errors(start)
    result = _idle_path_result()
    if position_error > tol_m or rotation_error > tol_rad:
        primitives._check_cancelled()
        plan = primitives.env.plan_pose_path(
            arm, [target.tolist()], linear_axis="x" if linear else None
        )
        primitives._check_cancelled()
        result = _execute_native_path(
            primitives, arm, plan, max_steps=max_steps, gripper=gripper
        )
    else:
        result["steps_used"] = _execute_reached_gripper(primitives, arm, gripper)
    final_obs = _current_obs(primitives)
    final = measured_pose(final_obs)
    position_error, rotation_error = errors(final)
    joints = _measured_vector(final_obs, f"{arm}_arm_joint_state", 6)
    commanded = result["last_commanded_joints"]
    return {
        **result,
        "arm": arm,
        "target_pose": target.tolist(),
        "start_pose": start.tolist(),
        "final_pose": final.tolist(),
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "reached": position_error <= tol_m and rotation_error <= tol_rad,
        "final_measured_joints": joints.tolist(),
        "max_joint_error_rad": None
        if commanded is None
        else float(np.max(np.abs(np.asarray(commanded) - joints))),
        "linear": linear,
        **_motion_feedback(
            primitives,
            arm,
            result,
            position_error <= tol_m and rotation_error <= tol_rad,
        ),
    }


def move_joints(
    primitives,
    state,
    arm,
    joint_positions,
    tol_rad,
    max_steps=None,
    gripper=0,
):
    """Execute one native joint trajectory without bare-goal replay to settle."""
    _validate_motion(arm, max_steps, gripper)
    target = np.asarray(joint_positions, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("move_joints requires six finite joint positions in radians")
    if not np.isfinite(tol_rad) or tol_rad <= 0:
        raise ValueError("move_joints requires positive finite tol_rad")
    primitives._check_cancelled()
    key = f"{arm}_arm_joint_state"
    start = _measured_vector(_current_obs(primitives), key, 6).copy()
    result = _idle_path_result()
    if np.max(np.abs(target - start)) > tol_rad:
        primitives._check_cancelled()
        plan = primitives.env.plan_joint_path(arm, target.tolist())
        primitives._check_cancelled()
        result = _execute_native_path(
            primitives, arm, plan, max_steps=max_steps, gripper=gripper
        )
    else:
        result["steps_used"] = _execute_reached_gripper(primitives, arm, gripper)
    final = _measured_vector(_current_obs(primitives), key, 6)
    errors = target - final
    max_error = float(np.max(np.abs(errors)))
    return {
        **result,
        "arm": arm,
        "target_joint_positions": target.tolist(),
        "start_joint_positions": start.tolist(),
        "final_joint_positions": final.tolist(),
        "joint_errors_rad": errors.tolist(),
        "max_joint_error_rad": max_error,
        "reached": max_error <= tol_rad,
        **_motion_feedback(primitives, arm, result, max_error <= tol_rad),
    }


def set_gripper(primitives, state, arm, gripper, max_steps=1) -> dict:
    """PyRUA-Lean measured stroke on the existing, individually ACKed action path."""

    if (
        arm not in {"left", "right"}
        or gripper not in (-1, 1)
        or type(max_steps) is not int
        or max_steps < 1
    ):
        raise ValueError(
            "set_gripper requires a valid arm and gripper 1 (close) or -1 (open), with positive max_steps"
        )
    primitives._check_cancelled()
    monitor = GripperSettling(
        _gripper_opening(_current_obs(primitives), arm), tolerance=0.004
    )
    steps_used = 0
    info = {"status": {}}
    reason = "max_steps"
    for _ in range(max_steps):
        primitives._check_cancelled()
        if (
            primitives.env.terminated
            or primitives.env.truncated
            or primitives.env.steps_remaining() <= 0
        ):
            reason = (
                "environment_terminal"
                if primitives.env.terminated
                else "episode_budget_exhausted"
            )
            break
        action = _joint_hold_action(_current_obs(primitives))
        action[_arm_ee_joint_key(arm)] = [0.0 if gripper > 0 else 1.0]
        primitives._check_cancelled()
        _obs_step, _reward, _done, info = primitives.env.step(action)
        steps_used += 1
        stable = monitor.observe(_gripper_opening(_current_obs(primitives), arm))
        if primitives.env.terminated or primitives.env.truncated:
            reason = (
                "environment_terminal"
                if primitives.env.terminated
                else "episode_budget_exhausted"
            )
            break
        if monitor.last is None:
            reason = "measurement_unavailable"
            break
        if stable:
            reason = "opening_stable"
            break
    return {
        "arm": arm,
        "commanded_gripper": "closed" if gripper > 0 else "open",
        "steps_used": steps_used,
        "stop_reason": reason,
        "remaining_steps": primitives.env.steps_remaining(),
        **monitor.feedback(unit="normalized_opening"),
        "status": {
            key: info["status"][key]
            for key in ("step", "step_limit")
            if key in info["status"]
        },
    }


def pi0_pick(
    primitives,
    state,
    prompt=None,
    arm="right",
    max_chunks=8,
) -> dict:
    """Execute policy chunks until the explicit cap or native terminal boundary."""

    from reweave.benchmarks.robodojo.tasks import validate_instruction

    # The policy is trained on the official sentence only, not the project supplement.
    prompt = validate_instruction(
        _current_obs(primitives)["instruction"] if prompt is None else prompt
    )

    vla = getattr(primitives, "vla_client", None)
    if vla is None:
        return {"error": "vla_client not configured"}
    if arm not in {"left", "right"} or type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("pi0_pick requires a valid arm and positive max_chunks")
    chunks_used = 0
    actions_used = 0
    for c in range(max_chunks):
        primitives._check_cancelled()
        obs = _current_obs(primitives)
        if primitives.env.terminated or primitives.env.truncated:
            break
        observation_ref = _observation_ref(primitives)
        physical_step = obs["physical_step"]
        with primitives.model_request("vla"):
            prediction = primitives.env.predict_ref(
                observation_ref, prompt=prompt, mode="eval"
            )
        primitives.execution.record_forward(prediction, instruction=prompt)
        # A fixed Host-owned journal binds the received forward to its retained
        # input. No arrays, new RPC or filename derived from provider IDs.
        with (state.output_dir / "vla_inputs.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "forward_id": prediction.forward_id,
                        "observation_ref": observation_ref,
                        "physical_step": physical_step,
                        "prompt": prompt,
                        "mode": "eval",
                        "prediction_horizon_steps": prediction.prediction_horizon_steps,
                        "execution_horizon_steps": prediction.execution_horizon_steps,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        chunks_used = c + 1
        try:
            for cursor in prediction.iter_actions():
                primitives._check_cancelled()
                if primitives.env.terminated or primitives.env.truncated:
                    break
                _obs_step, _reward, _done, _info = primitives.env.step(
                    cursor.action, cursor=cursor
                )
                actions_used += 1
                if _done or primitives.env.terminated or primitives.env.truncated:
                    break
        finally:
            # Predictions are never retained across tool returns, cancellation or unknown I/O.
            primitives.execution.discard_open_suffix(reason="tool_boundary")
        if primitives.env.terminated or primitives.env.truncated:
            break
    return {
        "reference_arm": arm,
        "instruction": prompt,
        "chunks_used": chunks_used,
        "actions_used": actions_used,
        "stop_reason": (
            "physical_budget"
            if primitives.env.truncated
            else "environment_terminal"
            if primitives.env.terminated
            else "max_chunks"
        ),
        "terminated": primitives.env.terminated,
        "truncated": primitives.env.truncated,
    }


def dump_state(primitives, state, *, log: dict | None = None):
    """Record compact state and retain its exact server snapshot without reading RGB."""
    obs = _current_obs(primitives)
    status = primitives.env.get_status()
    log = log or {}
    from reweave.benchmarks.robodojo.tasks import validate_instruction

    official_instruction = validate_instruction(obs["instruction"])
    task_language = validate_instruction(primitives.env.get_task_language())
    if task_language == official_instruction:
        task_rule_supplement = ""
    else:
        prefix = official_instruction + "\n\n"
        if not task_language.startswith(prefix):
            raise ValueError(
                "RoboDojo resolved task language changed the official instruction"
            )
        task_rule_supplement = validate_instruction(task_language.removeprefix(prefix))
    with state.record_step(
        state=_summarize_obs(obs),
        terminated=primitives.env.terminated,
        truncated=primitives.env.truncated,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": task_language,
            "official_task_instruction": official_instruction,
            "task_rule_supplement": task_rule_supplement,
            "observation_ref": _observation_ref(primitives),
            "control_steps": status["step"],
            "max_episode_steps": status["step_limit"],
        },
    ) as step_idx:
        retained = primitives.env.retain_observation(_observation_ref(primitives))
        state.register_observation(step_idx, retained)
    return state.get(step_idx)
