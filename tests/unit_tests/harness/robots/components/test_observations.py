"""Shared real-world observation layout preserves original camera data."""

from dataclasses import dataclass

import numpy as np

from reweave.robots.components.observations import (
    camera_observation,
    strip_batch,
    to_numpy_tree,
)


def test_camera_layout_orders_extras_without_changing_pixels_or_depth_units():
    frames = {
        name: np.full((2, 3, 3), value, dtype=np.uint8)
        for name, value in (("z_wrist", 30), ("main", 10), ("a_wrist", 20))
    }
    depths = {
        "main": np.full((2, 3), 0.75, dtype=np.float32),
        "z_wrist": np.full((2, 3), 1.25, dtype=np.float32),
    }
    result = camera_observation(frames, depths, "main")
    assert result["main_images"] is frames["main"]
    assert result["main_depths"] is depths["main"]
    np.testing.assert_array_equal(
        result["extra_view_images"], np.stack([frames["a_wrist"], frames["z_wrist"]])
    )
    np.testing.assert_array_equal(result["extra_view_depths"], depths["z_wrist"][None])
    assert result["extra_view_images"].dtype == np.uint8
    assert result["extra_view_depths"].dtype == np.float32


def test_single_camera_without_depth_has_no_invented_optional_views():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    result = camera_observation({"main": frame}, None, "main")
    assert list(result) == ["main_images"] and result["main_images"] is frame


def test_batch_removal_preserves_inner_image_axes_and_serializes_nested_state():
    @dataclass
    class State:
        value: object

    image = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    assert strip_batch(image).shape == (2, 3, 3)
    assert strip_batch(["frame"]) == "frame"
    assert strip_batch(np.zeros((2, 3))).shape == (2, 3)
    state = to_numpy_tree(
        {"state": State(np.int64(4)), "pair": (np.float32(1.5), image)}
    )
    assert state["state"] == {"value": 4}
    assert state["pair"][0] == 1.5 and state["pair"][1] is image
