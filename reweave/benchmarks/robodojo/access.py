# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Public observation and native dual-arm action contracts."""

import numpy as np

from reweave.robots.physical_evidence import feature

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = tuple(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in ("ee_pose", "ee_joint_state", "arm_joint_state")
)
MEASURED_GRIPPER_KEYS = tuple(
    f"{arm}_gripper_measured_opening" for arm in ("left", "right")
)
CAMERA_KEYS = (
    "color",
    "depth",
    "distance_to_image_plane",
    "intrinsic_matrix",
    "extrinsic_matrix",
    "shape",
    "color_dtype",
    "has_depth",
)

JOINT_ACTION_FIELDS = (
    ("left_arm_joint_state", 6),
    ("right_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_ee_joint_state", 1),
)
POSE_ACTION_FIELDS = (
    ("left_ee_pose", 7),
    ("right_ee_pose", 7),
    ("left_ee_joint_state", 1),
    ("right_ee_joint_state", 1),
)


def native_action(value) -> dict[str, list[float]]:
    """Validate one joint vector or full native joint/pose command before I/O."""
    if not isinstance(value, dict):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError(
                "RoboDojo policy action must have shape (14,) and finite values"
            )
        value = dict(
            zip(
                (name for name, _ in JOINT_ACTION_FIELDS), np.split(vector, [6, 12, 13])
            )
        )
    fields = (
        POSE_ACTION_FIELDS
        if any(name.endswith("_ee_pose") for name in value)
        else JOINT_ACTION_FIELDS
    )
    if set(value) != {name for name, _ in fields}:
        raise ValueError(
            "RoboDojo action must command both arms and both grippers with one representation"
        )
    result = {}
    for name, size in fields:
        array = np.asarray(value[name], dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(
                f"RoboDojo {name} must have shape ({size},) and finite values"
            )
        result[name] = array.tolist()
    return result


def public_observation(obs: dict) -> dict:
    """Filter native sensing or compact snapshot metadata to public task fields."""
    return {
        **{
            key: obs[key]
            for key in (
                "observation_ref",
                "service_instance_id",
                "physical_step",
            )
            if key in obs
        },
        "instruction": obs.get("instruction", ""),
        "state": {
            key: obs["state"][key]
            for key in (*STATE_KEYS, *MEASURED_GRIPPER_KEYS)
            if key in obs.get("state", {})
        },
        "vision": {
            name: {key: cam[key] for key in CAMERA_KEYS if key in cam}
            for name, cam in obs.get("vision", {}).items()
            if name in CAMERAS
        },
    }


def robot_state_features(obs: dict) -> dict:
    """Native pose/joints/command targets and independent measured gripper opening.

    Historical receipts have only the original six vectors. New observations
    carry both measured grippers, never a command-derived stand-in for either.
    """
    measured = set(MEASURED_GRIPPER_KEYS) & set(obs["state"])
    if measured and measured != set(MEASURED_GRIPPER_KEYS):
        raise ValueError("RoboDojo telemetry requires both measured gripper channels")
    result = {}
    for key in (
        *STATE_KEYS,
        *(key for key in MEASURED_GRIPPER_KEYS if key in measured),
    ):
        value = np.asarray(obs["state"][key], dtype=np.float64)
        size = (
            7
            if key.endswith("ee_pose")
            else 6
            if key.endswith("arm_joint_state")
            else 1
        )
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(
                f"RoboDojo public state {key} requires {size} finite values"
            )
        result[key] = value.tolist()
    return result


def graph_features(values: dict) -> dict:
    """Project native proprioception to unit-bearing scalars for offline graphs.

    Pose coordinates 0:3 are XYZ in metres; the remaining quaternion coordinates
    and normalized gripper channels are dimensionless. *_ee_joint_state is a
    native command-derived target; *_gripper_measured_opening comes from actual
    gripper joints and runtime calibration. Arm joint angles are radians.
    This read-only projection neither adds object poses nor calls the evaluator.
    """
    if set(values) not in (set(STATE_KEYS), set(STATE_KEYS + MEASURED_GRIPPER_KEYS)):
        raise ValueError(
            "RoboDojo telemetry requires the six native vectors, optionally with both measured grippers"
        )
    vectors = robot_state_features({"state": values})
    return {
        f"robot.{key}.{index}": feature(
            value,
            "m"
            if key.endswith("ee_pose") and index < 3
            else "rad"
            if key.endswith("arm_joint_state")
            else "dimensionless",
            f"RoboDojo:state.{key}[{index}]",
            "normalized actual gripper joint using native scale/sign; not force or possession"
            if key in MEASURED_GRIPPER_KEYS
            else "native command-derived normalized gripper opening target"
            if key.endswith("ee_joint_state")
            else "scalar coordinate of native public proprioception",
        )
        for key, vector in vectors.items()
        for index, value in enumerate(vector)
    }
