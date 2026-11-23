# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Server-local RGB-D perception and canonical X5 grasp geometry."""

import numpy as np

from reweave.robots.components.object_geometry import measure_geometry


def locate_observation(
    obs, *, sam3_client, text_prompt, camera="cam_head", min_score=0.2, support_z=None
):
    """PyRUA-Lean locate pipeline on one exact server RGB-D snapshot."""
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        raise ValueError("text_prompt must be nonempty")
    if not np.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be between zero and one")
    if support_z is not None and not np.isfinite(support_z):
        raise ValueError("support_z must be finite")
    result, mask = segment_observation(
        obs,
        sam3_client=sam3_client,
        text_prompt=text_prompt,
        camera=camera,
        min_score=min_score,
    )
    if mask is None:
        return {**result, "geometry": None, "geometry_error": "no object mask"}
    cam = obs["vision"][camera]
    depth = _camera_depth(cam)
    if depth is None:
        return {**result, "geometry": None, "geometry_error": "depth unavailable"}
    try:
        points = _masked_world_points(cam, depth, mask)
    except ValueError as error:
        return {**result, "geometry": None, "geometry_error": str(error)}
    return {
        **result,
        **measure_geometry(points, support_z=support_z),
        "area_px": int(mask.sum()),
        "valid_depth_fraction": round(len(points) / int(mask.sum()), 4),
    }


def back_project_observation(obs, row, col, camera="cam_head") -> dict:
    """Pixel -> world xyz via depth + camera calibration."""
    import numpy as np

    cam = obs.get("vision", {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    depth = _camera_depth(cam)
    if depth is None:
        return {"error": "depth not available in observation"}
    K = np.asarray(cam.get("intrinsic_matrix"), dtype=np.float64)
    T = np.asarray(cam.get("extrinsic_matrix"), dtype=np.float64)
    if (
        K.shape != (3, 3)
        or T.shape != (4, 4)
        or not np.isfinite(K).all()
        or not np.isfinite(T).all()
        or K[0, 0] == 0
        or K[1, 1] == 0
    ):
        return {"error": f"calibration missing/invalid: K={K.shape} T={T.shape}"}
    h, w = int(depth.shape[0]), int(depth.shape[1])
    if not (0 <= int(row) < h and 0 <= int(col) < w):
        return {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
    d = float(depth[int(row), int(col)])
    if not np.isfinite(d) or d <= 0:
        return {"error": f"invalid depth at pixel: {d}"}
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Isaac/Omniverse cameras look along -Z; the distance_to_image_plane
    # annotator returns distance along the optical axis, so the camera-frame
    # z coordinate is NEGATIVE d. Image rows grow downward while the USD
    # camera frame has +Y up, so the row offset enters negated.
    p_cam = np.array(
        [(int(col) - cx) / fx * d, -(int(row) - cy) / fy * d, -d, 1.0],
        dtype=np.float64,
    )
    p_world = T @ p_cam
    return {
        "camera": camera,
        "pixel": [int(row), int(col)],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in p_world[:3]],
    }


def segment_observation(
    obs, *, sam3_client, text_prompt, camera="cam_head", min_score=0.2
) -> tuple[dict, np.ndarray | None]:
    """Return SAM3 geometry and a foreground pixel nearest the true centroid.

    ``mask_point_rc`` is an integer image sample, not a verified grasp point.
    It shares the RGB/depth pixel grid; ``centroid_rc`` retains the true mean,
    which may lie outside an annular or concave mask.
    """
    cam = obs.get("vision", {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}, None
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}, None
    _camera_depth(cam)
    sam3 = sam3_client
    if sam3 is None:
        return {"error": "sam3_client not configured"}, None
    result = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
    if not result.found:
        return {
            "camera": camera,
            "found": False,
            "text_prompt": text_prompt,
            "reason": getattr(result, "reason", None),
        }, None
    import numpy as np

    mask = np.asarray(result.mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("SAM3 found result must contain a nonempty 2-D mask")
    if mask.shape != np.asarray(color).shape[:2]:
        raise ValueError("SAM3 mask shape differs from the camera RGB pixel grid")
    rows, cols = np.nonzero(mask)
    centroid = np.array([rows.mean(), cols.mean()])
    closest = int(np.argmin((rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2))

    return {
        "camera": camera,
        "found": True,
        "score": result.score,
        "box_px": _jsonable(result.box),
        "centroid_rc": centroid.tolist(),
        "mask_point_rc": [int(rows[closest]), int(cols[closest])],
    }, mask


def grasps_from_segment(
    obs,
    *,
    graspgen_client,
    mask,
    mask_score,
    text_prompt,
    arm,
    camera="cam_head",
    max_candidates=5,
    observation_step,
):
    """Return official score-ranked candidates from an already segmented snapshot.

    SAM is deliberately absent: the caller owns its separate inference receipt.
    """
    if graspgen_client is None:
        raise RuntimeError("GraspGen-X is not configured")
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        raise ValueError("text_prompt must be non-empty")
    prompt = text_prompt.strip()
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise ValueError("max_candidates must be an integer between 1 and 32")
    if not 1 <= max_candidates <= 32:
        raise ValueError("max_candidates must be between 1 and 32")
    cam = obs.get("vision", {}).get(camera)
    if cam is None:
        raise ValueError(f"camera {camera!r} not in observation")
    color = cam.get("color")
    if color is None:
        raise ValueError("camera color image missing")
    depth = _camera_depth(cam)
    if depth is None:
        raise ValueError("camera depth missing")
    _validate_isaac_camera(cam)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape or not mask.any():
        raise ValueError(
            f"SAM3 mask/depth shape mismatch or empty mask: mask={mask.shape}, depth={depth.shape}"
        )
    points = _masked_world_points(cam, depth, mask)
    result = graspgen_client.generate_grasps(
        points, arm=arm, frame="isaac_world", max_candidates=max_candidates
    )
    return {
        **result,
        "provenance": {
            "observation_step": observation_step,
            "camera": camera,
            "depth_source": (
                "distance_to_image_plane"
                if cam.get("distance_to_image_plane") is not None
                else "depth"
            ),
            "mask_source": "sam3_text_prompt",
            "mask_prompt": prompt,
            "mask_score": mask_score,
        },
    }


def _masked_world_points(
    camera: dict, depth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Back-project valid masked Isaac optical-axis depth into the world frame."""
    _validate_isaac_camera(camera)
    K = np.asarray(camera.get("intrinsic_matrix"), dtype=np.float64)
    camera_to_world = np.asarray(camera.get("extrinsic_matrix"), dtype=np.float64)

    rows, cols = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if not rows.size:
        raise ValueError("target mask contains no valid measured depth points")
    distances = np.asarray(depth[rows, cols], dtype=np.float64)
    camera_points = np.column_stack(
        (
            (cols - K[0, 2]) / K[0, 0] * distances,
            -(rows - K[1, 2]) / K[1, 1] * distances,
            -distances,
            np.ones_like(distances),
        )
    )
    world = (camera_to_world @ camera_points.T).T[:, :3]
    valid = np.isfinite(world).all(axis=1)
    if not valid.any():
        raise ValueError("back-projected mask contains no finite world points")
    return world[valid]


def _validate_isaac_camera(camera: dict) -> None:
    """Require positive pinhole focal lengths and a rigid camera transform."""
    K = np.asarray(camera.get("intrinsic_matrix"), dtype=np.float64)
    camera_to_world = np.asarray(camera.get("extrinsic_matrix"), dtype=np.float64)
    if (
        K.shape != (3, 3)
        or camera_to_world.shape != (4, 4)
        or not np.isfinite(K).all()
        or not np.isfinite(camera_to_world).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
    ):
        raise ValueError("camera calibration is missing or invalid")
    rotation = camera_to_world[:3, :3]
    if (
        not np.allclose(camera_to_world[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("camera extrinsic must be a rigid camera-to-world transform")


def _camera_depth(camera):
    """Read a single-channel depth grid with the camera's RGB pixel dimensions."""
    import numpy as np

    depth = camera.get("distance_to_image_plane")
    if depth is None:
        depth = camera.get("depth")
    if depth is None:
        return None
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("RoboDojo depth requires a single 2-D pixel grid")
    color = camera.get("color")
    if color is not None and depth.shape != np.asarray(color).shape[:2]:
        raise ValueError("RoboDojo depth and RGB shapes use different pixel grids")
    return depth


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
