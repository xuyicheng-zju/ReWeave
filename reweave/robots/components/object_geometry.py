# Copyright 2026 PyRUA-Lean Contributors.
# Licensed under the Apache License, Version 2.0.
# Adapted from DAGroup-PKU/PyRUA-Lean, commit
# fdf6e870654254e2c0c891ee407313fa48bdd0f9, hosts/robodojo_tools.py.
# Changes: explicit support plane, robot-neutral angle name, validated public input.
"""PyRUA-Lean footprint and upright round-body geometry, independent of robot frames.

Inputs are measured WORLD points in metres. No camera convention, table height,
robot kinematics or object identity is inferred here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def measure_region(world_map, box_rc, *, support_z=None):
    """Apply the same PyRUA-Lean geometry to a selected measured pixel region."""
    world = np.asarray(world_map)
    if world.ndim != 3 or world.shape[2] != 3:
        raise ValueError("world map must have shape H x W x 3")
    if (
        not isinstance(box_rc, (list, tuple))
        or len(box_rc) != 4
        or any(type(v) is not int for v in box_rc)
    ):
        raise ValueError(
            "box_rc must be [row_start, col_start, row_end, col_end] integers"
        )
    r0, c0, r1, c1 = box_rc
    if not (0 <= r0 < r1 <= world.shape[0] and 0 <= c0 < c1 <= world.shape[1]):
        raise ValueError(
            "box_rc must be a nonempty half-open region inside the recorded map"
        )
    points = world[r0:r1, c0:c1].reshape(-1, 3)
    points = points[
        np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
    ]
    return {
        **measure_geometry(points, support_z=support_z),
        "box_rc": list(box_rc),
        "selection": "pixel_region_not_object_identity",
    }


def region_tool_spec(*, camera_key="camera", cameras, resolutions=None):
    spec = {
        "name": "measure_region",
        "description": "Measure PyRUA-Lean footprint, top and world closing axis from a selected region of an existing world map. The region may contain background/multiple objects: inspect identity and isolate the target. No render or physical action. support_z enables upright-on-plane body height; omit for visible geometry only.",
        "input_schema": {
            "type": "object",
            "properties": {
                camera_key: {"type": "string", "enum": list(cameras)},
                "box_rc": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Half-open [row_start, col_start, row_end, col_end] in the SAME recorded camera/resolution.",
                },
                "support_z": {
                    "type": "number",
                    "description": "Known world support-plane height in metres; omit if unknown.",
                },
                "step": {
                    "type": "integer",
                    "description": "Recorded state, default latest.",
                },
            },
            "required": ["box_rc", camera_key],
            "additionalProperties": False,
        },
    }
    if resolutions is not None:
        spec["input_schema"]["properties"]["resolution"] = {
            "type": "string",
            "enum": list(resolutions),
            "description": "World-map resolution matching the image pixels; default low.",
        }
    return spec


def measure_geometry(points, *, support_z: float | None = None) -> dict[str, Any]:
    """Measure PyRUA-Lean geometry; require an explicit plane for body height.

    With no known support plane, return visible geometry only: hidden height and
    round-body completion remain unavailable. ``closing_axis_deg`` is a WORLD
    XY axis, not a robot's wrist command. Contact/grasp poses need its adapter.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("geometry requires an N x 3 world point array")
    if support_z is not None and not np.isfinite(support_z):
        raise ValueError("support_z must be finite metres or null")
    valid = points[np.isfinite(points).all(axis=1)]
    if support_z is not None:
        valid = valid[valid[:, 2] > support_z + 0.004]
    quality = {"input_points": len(points), "valid_points": len(valid)}
    if len(valid) < 12 or len(np.unique(np.floor(valid[:, :2] / 0.002), axis=0)) < 3:
        return {
            "geometry": None,
            "geometry_error": "too few distinct measured points",
            **quality,
        }
    if support_z is not None:
        geometry = object_geometry(valid, support_z=float(support_z))
    else:
        heading, length, width, mid = _footprint(valid[:, :2])
        geometry = {
            "center_xyz": [
                *np.round(mid, 4).tolist(),
                round(float(np.median(valid[:, 2])), 4),
            ],
            "top_z": round(float(np.percentile(valid[:, 2], 97)), 4),
            "height_m": None,
            "length_m": round(length, 4),
            "width_m": round(width, 4),
            "long_axis_deg": round(heading, 1),
            "closing_axis_deg": round((heading + 180.0) % 180.0 - 90.0, 1),
            "points": len(valid),
        }
    return {
        "geometry": geometry,
        "geometry_error": None
        if geometry
        else "too little surface above support plane",
        "support_z": support_z,
        "measurement_kind": "upright_on_support_plane"
        if support_z is not None
        else "visible_surface",
        **quality,
    }


FOOTPRINT_CELL_M = 0.002

LONG_FOOTPRINT = 1.4

FOOTPRINT_TRIM = 0.5

FOOT_M = 0.003

FOOT_MIN_HEIGHT_M = 0.012

SLICE_M = 0.01

ROUND_MIN_RADIUS_M = 0.008

ROUND_MAX_RADIUS_M = 0.12

ROUND_FIT = 0.04

ROUND_MIN_ARC_DEG = 100.0

ROUND_AXIS_M = 0.006


def _cells(xy: np.ndarray, cell: float = FOOTPRINT_CELL_M) -> np.ndarray:
    """One point per occupied grid cell (the mean of the points in it): the footprint with
    the same weight everywhere."""

    _, cell_of, count = np.unique(
        np.floor(xy / cell).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    cell_of = cell_of.reshape(-1)
    sums = np.stack([np.bincount(cell_of, weights=xy[:, i]) for i in (0, 1)], axis=1)
    return sums / count[:, None]


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """A footprint's rectangle: the heading of its long side (degrees from +x, -90 .. 90), its
    length and width, and its centre.

    Every occupied cell counts once, so the answer does not depend on how densely each part
    of the object was seen.  A footprint clearly longer one way lies along its principal
    axes (a T, a hammer, a shoe: their smallest rectangle can sit askew).  A square or a
    circle has no principal axis: there the rectangle of least area over all headings is
    taken, which finds the sides of a square.
    """

    cells = _cells(xy)
    trim = [FOOTPRINT_TRIM, 100.0 - FOOTPRINT_TRIM]

    def extents(
        deg: float,
    ) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
        a = math.radians(deg)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        (ul, uh), (vl, vh) = (
            np.percentile(cells @ u, trim),
            np.percentile(cells @ v, trim),
        )
        return float((uh - ul) * (vh - vl)), ul, uh, vl, vh, u, v

    spread, axes = np.linalg.eigh(np.cov((cells - cells.mean(axis=0)).T))
    if spread[1] >= LONG_FOOTPRINT * max(float(spread[0]), 1e-12):
        heading = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
    else:
        coarse = min(np.arange(0.0, 90.0, 2.0), key=lambda d: extents(d)[0])
        heading = min(
            np.arange(coarse - 2.0, coarse + 2.01, 0.25), key=lambda d: extents(d)[0]
        )
    _, ul, uh, vl, vh, u, v = extents(heading)
    centre = u * (ul + uh) / 2 + v * (vl + vh) / 2
    length, width = uh - ul, vh - vl
    if width > length:
        length, width, heading = width, length, heading + 90.0
    return (heading + 90.0) % 180.0 - 90.0, float(length), float(width), centre


def _far_side(pts: np.ndarray, *, support_z: float) -> np.ndarray:
    """Footprint points (N x 2) of what the camera cannot see of a round object.

    A camera sees the near half of an upright bottle, cup or doll, so its visible points end
    at the middle and their footprint sits towards the camera: against the scenes' true poses
    the centre of an upright bottle was 6 to 14 mm off and its far edge up to 2 cm short.
    Each horizontal slice whose visible points lie on a circular arc is completed to the full
    circle, provided the circles of the object share one upright axis.  Flat-sided objects are
    left alone: their slices are not arcs, and their tops, which the camera does see, cover
    them.  So is a rounded body that is not round about an upright axis.
    """

    circles = []
    for low in np.arange(support_z + 0.004, float(pts[:, 2].max()), SLICE_M):
        xy = pts[(pts[:, 2] >= low) & (pts[:, 2] < low + SLICE_M), :2]
        if len(xy) < 15:
            continue
        # algebraic circle fit: |p|^2 = 2 c.p + (r^2 - |c|^2)
        sol, *_ = np.linalg.lstsq(
            np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(axis=1), rcond=None
        )
        centre = sol[:2]
        radius = math.sqrt(max(float(sol[2] + centre @ centre), 0.0))
        if not ROUND_MIN_RADIUS_M <= radius <= ROUND_MAX_RADIUS_M:
            continue
        rms = float(
            np.sqrt(((np.linalg.norm(xy - centre, axis=1) - radius) ** 2).mean())
        )
        angles = np.sort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        arc = 360.0 - math.degrees(
            float(gaps.max())
        )  # what the points cover of the circle
        if rms <= ROUND_FIT * radius and arc >= ROUND_MIN_ARC_DEG:
            circles.append((centre, radius))
    if len(circles) >= 3:
        # The slices of a round object share an axis: a circle off it is a misfit.  And when no
        # more than half of them share one, the object is not round about an upright axis: the
        # body of a piggy bank lying on its feet gave four circles, two of them alike, and the
        # two put 2 cm on its width.
        axis = np.median(np.array([c for c, _ in circles]), axis=0)
        alike = [(c, r) for c, r in circles if np.linalg.norm(c - axis) <= ROUND_AXIS_M]
        circles = alike if 2 * len(alike) > len(circles) else []
    turn = np.linspace(0.0, 2 * math.pi, 90, endpoint=False)
    ring = np.c_[np.cos(turn), np.sin(turn)]
    return (
        np.vstack([c + r * ring for c, r in circles]) if circles else np.zeros((0, 2))
    )


def object_geometry(points: np.ndarray, *, support_z: float) -> dict[str, Any] | None:
    """Extent, heading and heights of an object from its visible surface points (N x 3)."""

    pts = points[
        points[:, 2] > support_z + 0.004
    ]  # the table seen through gaps in the mask
    if len(pts) < 12:
        return None
    med = np.median(pts[:, :2], axis=0)
    spread = np.median(np.linalg.norm(pts[:, :2] - med, axis=1))
    keep = np.linalg.norm(pts[:, :2] - med, axis=1) <= max(4.0 * spread, 0.03)
    pts = pts[keep]  # pixels of other things at the mask's edge
    top = float(np.percentile(pts[:, 2], 97))
    low = float(pts[:, 2].min())
    standing = pts[pts[:, 2] > low + FOOT_M] if top - low > FOOT_MIN_HEIGHT_M else pts
    heading, length, width, mid = _footprint(
        np.vstack([standing[:, :2], _far_side(pts, support_z=support_z)])
    )
    across_heading = (heading + 180.0) % 180.0 - 90.0
    return {
        "center_xyz": [
            round(float(mid[0]), 4),
            round(float(mid[1]), 4),
            round((top + support_z) / 2, 4),
        ],
        "top_z": round(top, 4),
        "height_m": round(top - support_z, 4),
        "length_m": round(length, 4),
        "width_m": round(width, 4),
        "long_axis_deg": round(heading, 1),
        # pointing straight down, this yaw closes the fingers across the narrow side
        "closing_axis_deg": round(across_heading, 1),
        "points": int(len(pts)),
    }
