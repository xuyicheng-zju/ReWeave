# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Shared observation encoding for RLinf real-world camera workers."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def to_numpy_tree(value: Any) -> Any:
    """Convert tensors and nested values into pickle-safe CPU/numpy data."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if dataclasses.is_dataclass(value):
        return to_numpy_tree(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def strip_batch(value: Any) -> Any:
    """Remove a single environment axis from an RPC observation value."""
    array = to_numpy_tree(value)
    if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    if isinstance(array, list) and len(array) == 1:
        return array[0]
    return array


def camera_observation(frames, depths, main_key) -> dict[str, Any]:
    """Keep original pixels and the policy's sorted main/extra camera layout."""
    output: dict[str, Any] = {"main_images": np.asarray(frames[main_key])}
    extras = [np.asarray(frames[name]) for name in sorted(frames) if name != main_key]
    if extras:
        output["extra_view_images"] = np.stack(extras, axis=0)
    if depths:
        if main_key in depths:
            output["main_depths"] = np.asarray(depths[main_key])
        extra_depths = [
            np.asarray(depths[name]) for name in sorted(depths) if name != main_key
        ]
        if extra_depths:
            output["extra_view_depths"] = np.stack(extra_depths, axis=0)
    return output
