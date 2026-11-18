from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from reweave.benchmarks.robodojo import perception


class _GraspGen:
    def generate_grasps(self, points, **kwargs):
        np.testing.assert_allclose(points, [[1.0, 3.0, 1.0]])
        assert kwargs == {
            "arm": "left",
            "frame": "isaac_world",
            "max_candidates": 2,
        }
        return {
            "planner": "graspmoe",
            "conditioning": {"asset_label": "arx_x5"},
            "candidates": [
                {"candidate_id": index, "branch_tag": "diffusion"}
                for index in range(kwargs["max_candidates"])
            ],
            "checks": {},
        }


class _Primitives:
    def __init__(self):
        camera = {
            "color": np.zeros((2, 2, 3), dtype=np.uint8),
            "distance_to_image_plane": np.array(
                [[0.0, 2.0], [np.nan, -1.0]], dtype=np.float32
            ),
            "intrinsic_matrix": np.array([[2.0, 0, 0], [0, 4.0, 0], [0, 0, 1]]),
            "extrinsic_matrix": np.array(
                [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
                dtype=np.float64,
            ),
        }
        self.env = SimpleNamespace(
            last_obs={"vision": {"cam_head": camera}},
            get_status=lambda: {"step": 12, "step_limit": 700},
        )
        self.sam3_client = SimpleNamespace(
            segment=lambda *args, **kwargs: SimpleNamespace(
                found=True,
                mask=np.array([[False, True], [True, True]]),
                score=0.88,
                reason=None,
            )
        )
        self.graspgen_client = _GraspGen()
        self._last_obs = None

    @contextmanager
    def model_request(self, kind):
        assert kind in {"sam3", "graspgen"}
        yield


def generate_grasps(
    primitives,
    state,
    text_prompt,
    arm,
    camera="cam_head",
    min_score=0.2,
    max_candidates=5,
):
    """Exercise the server geometry directly, without a Host RGB-D transport."""
    del state, min_score
    segment_result = primitives.sam3_client.segment(None)
    return perception.grasps_from_segment(
        primitives.env.last_obs,
        graspgen_client=primitives.graspgen_client,
        mask=segment_result.mask,
        mask_score=segment_result.score,
        text_prompt=text_prompt,
        arm=arm,
        camera=camera,
        max_candidates=max_candidates,
        observation_step=12,
    )


def test_generate_grasps_back_projects_same_snapshot_in_isaac_world():
    result = generate_grasps(
        _Primitives(), object(), "blue bottle", "left", max_candidates=2
    )
    assert result["candidates"] == [
        {"candidate_id": 0, "branch_tag": "diffusion"},
        {"candidate_id": 1, "branch_tag": "diffusion"},
    ]
    assert result["provenance"] == {
        "observation_step": 12,
        "camera": "cam_head",
        "depth_source": "distance_to_image_plane",
        "mask_source": "sam3_text_prompt",
        "mask_prompt": "blue bottle",
        "mask_score": 0.88,
    }
    assert result["checks"] == {}
    assert "feasibility" not in result


def test_small_segment_does_not_require_a_custom_support_plane_fit():
    # One valid measured target point reaches the official client unchanged.
    # A local support-plane fit used to reject it before official inference.
    result = generate_grasps(
        _Primitives(), object(), "thin ring", "left", max_candidates=2
    )
    assert len(result["candidates"]) == 2
    assert not any(key.startswith("support_") for key in result["provenance"])


def test_generate_grasps_rejects_uncalibrated_camera():
    primitives = _Primitives()
    primitives.env.last_obs = {
        "vision": {
            "cam_head": {
                "color": np.zeros((1, 1, 3), dtype=np.uint8),
                "distance_to_image_plane": np.ones((1, 1), dtype=np.float32),
            }
        }
    }

    try:
        generate_grasps(primitives, object(), "bottle", "right")
    except ValueError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("uncalibrated RGB-D must fail before inference")


def test_generate_grasps_rejects_non_rigid_camera_extrinsic():
    primitives = _Primitives()
    camera = primitives.env.last_obs["vision"]["cam_head"]
    camera["extrinsic_matrix"] = np.diag([2.0, 1.0, 1.0, 1.0])
    primitives.env.last_obs = {"vision": {"cam_head": camera}}

    try:
        generate_grasps(primitives, object(), "bottle", "right")
    except ValueError as error:
        assert "rigid" in str(error)
    else:
        raise AssertionError("non-rigid camera extrinsic must fail before inference")
