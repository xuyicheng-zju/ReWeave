from types import SimpleNamespace

import numpy as np

from reweave.benchmarks.libero.tools import LiberoPrimitives


class _State:
    def __init__(self):
        self.record = SimpleNamespace(
            step_idx=7,
            artifacts={
                "agentview.png",
                "agentview_depth.npz",
                "agentview_metadata.json",
            },
        )
        self.depth = np.zeros((2, 2), dtype=np.float32)
        self.depth[0, 1] = 0.8
        self.metadata = {
            "intrinsic_K": [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]],
            "extrinsic_cam2world": [
                [1, 0, 0, 0],
                [0, 1, 0, -0.2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        }
        self.saved = {}

    def get(self, step):
        assert step == -1
        return self.record

    def exists(self, name, *, step):
        return name in self.record.artifacts and step == 7

    def load_bytes(self, name, *, step):
        assert (name, step) == ("agentview.png", 7)
        return b"png"

    def load(self, name, *, step):
        assert step == 7
        if name == "agentview_depth.npz":
            return self.depth
        if name == "agentview_metadata.json":
            return self.metadata
        if name == "agentview.png":
            return np.zeros((2, 2, 3), dtype=np.uint8)
        raise AssertionError(name)

    def save(self, name, value, *, step):
        assert step == 7
        self.saved[name] = value
        self.record.artifacts.add(name)
        return name


class _Sam:
    def segment(self, image, **kwargs):
        assert image == b"png"
        assert kwargs["text_prompt"] == "red mug"
        return SimpleNamespace(
            found=True,
            mask=np.array([[False, True], [False, False]]),
            score=0.91,
            reason=None,
        )


class _GraspGen:
    def generate_grasps(self, points, **kwargs):
        np.testing.assert_allclose(points, [[0.4, -0.2, 0.8]])
        assert kwargs == {
            "arm": "right",
            "frame": "libero_world",
            "max_candidates": 3,
        }
        return {"candidates": [{"candidate_id": 0}], "checks": {}}


def test_generate_grasps_uses_one_recorded_rgbd_step_and_world_frame():
    primitive = LiberoPrimitives.__new__(LiberoPrimitives)
    primitive._sam3_client = _Sam()
    primitive._graspgen_client = _GraspGen()
    primitive.model_budget = None

    result = primitive.generate_grasps(
        "red mug", step=-1, max_candidates=3, state=_State()
    )

    assert result["candidates"] == [{"candidate_id": 0}]
    assert result["provenance"] == {
        "observation_step": 7,
        "camera": "agentview",
        "image_artifact": "agentview.png",
        "depth_artifact": "agentview_depth.npz",
        "camera_metadata_artifact": "agentview_metadata.json",
        "mask_source": "sam3_text_prompt",
        "mask_prompt": "red mug",
        "mask_score": 0.91,
        "mask_artifact": "grasp_mask_00.npz",
        "mask_overlay_artifact": "grasp_mask_overlay_00.png",
        "mask_metadata_artifact": "grasp_mask_00.json",
    }


def test_generate_grasps_rejects_mask_without_measured_depth():
    state = _State()
    state.depth[:] = 0
    primitive = LiberoPrimitives.__new__(LiberoPrimitives)
    primitive._sam3_client = _Sam()
    primitive._graspgen_client = _GraspGen()
    primitive.model_budget = None

    try:
        primitive.generate_grasps("red mug", state=state)
    except ValueError as error:
        assert "no valid measured depth points" in str(error)
    else:
        raise AssertionError("missing depth must fail before GraspGen-X inference")
