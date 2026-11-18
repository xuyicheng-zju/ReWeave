"""Measured geometry is portable; support planes and wrist conventions are explicit."""

import json
from pathlib import Path

import numpy as np
import pytest

from reweave.robots.components.gripper_feedback import GripperSettling
from reweave.robots.components.object_geometry import measure_geometry, measure_region


def block_points():
    x, y = np.meshgrid(np.linspace(-0.04, 0.04, 31), np.linspace(-0.02, 0.02, 19))
    turn = np.deg2rad(31)
    return np.c_[
        0.12 + x.ravel() * np.cos(turn) - y.ravel() * np.sin(turn),
        -0.1 + x.ravel() * np.sin(turn) + y.ravel() * np.cos(turn),
        np.full(x.size, 0.84),
    ]


def test_port_matches_pinned_upstream_block_fixture():
    expected = json.loads(
        (Path(__file__).parents[4] / "fixtures/pyrualean_geometry.json").read_text()
    )
    actual = measure_geometry(block_points(), support_z=0.8)["geometry"]
    assert actual == expected["rotated_block"]


def test_world_translation_preserves_metric_dimensions_and_orientation():
    points = block_points()
    original = measure_geometry(points, support_z=0.8)["geometry"]
    translated = measure_geometry(points + [4, -3, 1.7], support_z=2.5)["geometry"]
    for key in ("height_m", "length_m", "width_m", "long_axis_deg", "closing_axis_deg"):
        assert translated[key] == pytest.approx(original[key])
    assert np.array(translated["center_xyz"]) - original["center_xyz"] == pytest.approx(
        [4, -3, 1.7], abs=1e-4
    )


def test_unknown_support_does_not_invent_hidden_body_height():
    result = measure_geometry(block_points())
    assert result["geometry"]["height_m"] is None
    assert result["measurement_kind"] == "visible_surface"
    assert "grasp_yaw" not in result["geometry"]


def test_upstream_round_completion_recovers_hidden_half_without_changing_plane():
    angle, height = np.meshgrid(
        np.linspace(0, np.pi, 60), np.linspace(0.806, 0.866, 12)
    )
    points = np.c_[
        0.12 + 0.03 * np.cos(angle.ravel()),
        -0.1 + 0.03 * np.sin(angle.ravel()),
        height.ravel(),
    ]
    geometry = measure_geometry(points, support_z=0.8)["geometry"]
    assert geometry["center_xyz"][:2] == pytest.approx([0.12, -0.1], abs=0.002)
    assert geometry["length_m"] == pytest.approx(0.06, abs=0.002)
    assert geometry["width_m"] == pytest.approx(0.06, abs=0.002)


@pytest.mark.parametrize(
    "points", [np.zeros((20, 3)), np.full((20, 3), np.nan), np.ones((2, 3))]
)
def test_degenerate_measurements_are_explicitly_unavailable(points):
    result = measure_geometry(points)
    assert result["geometry"] is None and result["geometry_error"]


def test_selected_region_keeps_pixel_frame_and_cannot_silently_clip():
    world = block_points().reshape(19, 31, 3)
    result = measure_region(world, [0, 0, 19, 31], support_z=0.8)
    assert (
        result["geometry"]
        == measure_geometry(world.reshape(-1, 3), support_z=0.8)["geometry"]
    )
    with pytest.raises(ValueError, match="inside"):
        measure_region(world, [-1, 0, 19, 31])


def test_settling_uses_measured_units_and_requires_two_acknowledged_samples():
    normalized = GripperSettling([0.8], tolerance=0.004)
    assert not normalized.observe([0.4])
    assert normalized.observe([0.401])
    metric = GripperSettling([0.08], tolerance=0.0004)
    assert not metric.observe([0.04])
    assert metric.observe([0.0401])
    assert not GripperSettling(None, tolerance=0.004).observe(None)


def test_missing_measurement_cannot_become_a_closed_or_stable_gripper():
    monitor = GripperSettling(None, tolerance=0.004)
    monitor.observe(None)
    result = monitor.feedback(unit="normalized_opening")
    assert result["measured_opening"] is None and result["opening_stable"] is None
    assert not result["measurement_available"]
