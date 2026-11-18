# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Thin client wrapping the Pi0.5 VLA RPC server."""

from __future__ import annotations

from typing import Any

import numpy as np

from reweave.robots.components.vla_client_base import BaseVLAClient
from reweave.robots.vla_execution import VLAChunkPrediction


def _encode_obs_libero(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Encode one single-environment LIBERO observation for OpenPI."""

    def _batch_view(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"expected [H,W,3] image, got shape {array.shape}")
        return array.astype(np.uint8, copy=False)[None]

    main = np.asarray(env_obs["main_images"])
    if main.ndim != 3:
        raise ValueError(f"expected [H,W,3] image, got shape {main.shape}")
    states = np.asarray(env_obs["states"], dtype=np.float32)
    if states.ndim != 1:
        raise ValueError(f"states must be [state_dim], got shape {states.shape}")
    return {
        "main_images": main.astype(np.uint8, copy=False)[None],
        "wrist_images": _batch_view(env_obs.get("wrist_images")),
        "extra_view_images": _batch_view(env_obs.get("extra_view_images")),
        "states": states[None],
        "task_descriptions": [str(env_obs.get("task_descriptions") or "")],
    }


def _encode_obs_robodojo(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Three RGB cameras and native left/right 6+6+1+1 joint state."""
    from reweave.benchmarks.robodojo.tasks import validate_instruction

    images = {}
    for key, camera in (
        ("main_images", "cam_head"),
        ("wrist_images", "cam_left_wrist"),
        ("extra_view_images", "cam_right_wrist"),
    ):
        image = np.asarray(env_obs["vision"][camera]["color"])
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape) <= 0:
            raise ValueError(f"RoboDojo {camera} must contain [H,W,3] RGB")
        images[key] = image.astype(np.uint8, copy=False)[None]
    state = env_obs["state"]
    vectors = []
    for key, size in (
        ("left_arm_joint_state", 6),
        ("right_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_ee_joint_state", 1),
    ):
        vector = np.asarray(state[key], dtype=np.float32)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"RoboDojo {key} requires {size} finite values")
        vectors.append(vector)
    return {
        **images,
        "states": np.concatenate(vectors)[None],
        "task_descriptions": [validate_instruction(env_obs["instruction"])],
    }


_ENCODE_OBS: dict[str, Any] = {
    "libero": _encode_obs_libero,
    "robodojo": _encode_obs_robodojo,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Pi05VLAClient(BaseVLAClient):
    """Official Harness client surface plus typed forwards for runtime monitoring."""

    def __init__(self, client, *, embodiment: str = "libero"):
        super().__init__(client)
        if embodiment not in _ENCODE_OBS:
            raise ValueError(
                f"unknown pi05 client embodiment: {embodiment!r}; "
                f"registered={list(_ENCODE_OBS)}"
            )
        self._embodiment = embodiment

    def encode_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        return _ENCODE_OBS[self._embodiment](env_obs)

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> VLAChunkPrediction:
        """Encode one LIBERO or RoboDojo observation and return one typed VLA forward."""
        response = super().predict(self.encode_obs(env_obs), options={"mode": mode})
        prediction = VLAChunkPrediction.from_rpc(response)
        if self._embodiment == "robodojo" and prediction.actions.shape[1] != 14:
            raise ValueError("RoboDojo Pi0.5 must return 14 joint action values")
        return prediction

    def predict(
        self,
        env_obs: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Official Harness prediction API, returning ``[chunk, action_dim]``."""
        mode = str((options or {}).get("mode", "eval"))
        return self.predict_action_batch(env_obs, mode=mode).actions
