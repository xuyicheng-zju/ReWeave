"""Host observations fetch pixels only for deliberate, exactly bound image reads."""

import io
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from reweave.benchmarks.robodojo import tools
from reweave.benchmarks.robodojo.toolkit import RoboDojoState
from reweave.session.base import resolve_image_artifact


def png(value):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (value, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


class RefEnv:
    def __init__(self):
        self.reads = []
        self.retains = []
        self.last_obs = {
            "observation_ref": "sid:0:old",
            "service_instance_id": "sid",
            "physical_step": 0,
            "instruction": "Place a piece",
            "state": {},
            "vision": {
                camera: {"shape": [3, 4, 3], "color_dtype": "uint8", "has_depth": True}
                for camera, _ in tools.CAMERA_ARTIFACTS
            },
        }
        self.terminated = self.truncated = False

    def get_task_language(self):
        return "Place a piece"

    def get_status(self):
        return {"step": self.last_obs["physical_step"], "step_limit": 1100}

    def retain_observation(self, ref):
        self.retains.append(ref)
        return {
            "observation_ref": ref,
            "archive_path": f"/server/{ref}/rgbd.npz",
            "rgb_paths": {},
        }

    def read_rgb(self, ref, cameras):
        self.reads.append((ref, tuple(cameras)))
        return {camera: png(1 if ref.endswith("old") else 2) for camera in cameras}


def record(state, env):
    return tools.dump_state(SimpleNamespace(env=env), state)


def test_state_feedback_is_refs_only_and_explicit_history_read_uses_exact_ref(tmp_path):
    env = RefEnv()
    state = RoboDojoState(tmp_path, env=env)
    old = record(state, env)
    output = tools.view_env_state(old.step_idx, state=state, include_images=False)
    assert env.reads == []
    assert output["observation_ref"] == "sid:0:old"
    assert not any(key.startswith("_image") for key in output)
    assert old.state["cameras"]["cam_head"]["has_depth"] is True
    assert old.state["cameras"]["cam_head"]["color_dtype"] == "uint8"
    env.last_obs = {**env.last_obs, "observation_ref": "sid:1:new", "physical_step": 1}
    record(state, env)
    resolved = resolve_image_artifact(state, "cam_left_wrist.png", step=0)
    assert resolved[1].read_bytes() == png(1)
    assert env.reads == [("sid:0:old", ("cam_left_wrist",))]
    assert state.load_bytes("cam_left_wrist.png", step=0) == png(1)
    assert len(env.reads) == 1
    assert not (tmp_path / "cam_head.png" / "00.png").exists()


def test_current_three_images_use_one_batch_and_preserve_native_png_bytes(tmp_path):
    env = RefEnv()
    state = RoboDojoState(tmp_path, env=env)
    current = record(state, env)
    cameras = [camera for camera, _ in tools.CAMERA_ARTIFACTS]
    state.materialize_rgb(current.step_idx, cameras)
    assert env.reads == [("sid:0:old", tuple(cameras))]
    assert all(
        state.load_bytes(artifact, step=0) == png(1)
        for _, artifact in tools.CAMERA_ARTIFACTS
    )
    assert len(env.reads) == 1


def test_missing_or_mismatched_reference_cannot_substitute_latest(tmp_path):
    env = RefEnv()
    state = RoboDojoState(tmp_path, env=env)
    env.retain_observation = lambda ref: {"observation_ref": "another"}
    with pytest.raises(RuntimeError, match="does not match"):
        record(state, env)
    env.last_obs.pop("observation_ref")
    with pytest.raises(RuntimeError, match="reference"):
        record(state, env)
    assert env.reads == []


@pytest.mark.parametrize("found", [False, True])
def test_grasp_inference_preserves_two_separate_budgets_without_host_pixels(found):
    calls = []
    budgets = []
    env = RefEnv()

    def segment_ref(ref, **kwargs):
        calls.append(("sam3", ref, kwargs))
        return {"found": found, "mask_ref": "mask-1" if found else None}

    def grasps_ref(ref, mask_ref, **kwargs):
        calls.append(("graspgen", ref, mask_ref, kwargs))
        return {"candidates": []}

    env.segment_ref, env.grasps_ref = segment_ref, grasps_ref

    @contextmanager
    def model_request(kind):
        budgets.append(kind)
        yield

    primitives = SimpleNamespace(
        env=env, graspgen_client=object(), model_request=model_request
    )
    if found:
        assert tools.generate_grasps(primitives, None, "piece", "right") == {
            "candidates": []
        }
    else:
        with pytest.raises(RuntimeError, match="no target mask"):
            tools.generate_grasps(primitives, None, "piece", "right")
    assert budgets == (["sam3", "graspgen"] if found else ["sam3"])
    assert all(call[1] == "sid:0:old" for call in calls)
    assert env.reads == []


def test_host_perception_rejects_removed_full_rgbd_only_path():
    primitives = SimpleNamespace(
        env=SimpleNamespace(last_obs={"vision": {"cam_head": {"color": object()}}})
    )
    with pytest.raises(RuntimeError, match="reference"):
        tools.back_project(primitives, None, 0, 0)


def test_policy_forward_uses_current_reference_and_does_not_mutate_ack_instruction(
    tmp_path,
):
    import numpy as np

    from reweave.robots.vla_execution import VLAChunkPrediction

    env = RefEnv()
    env.last_obs["state"] = {
        f"{arm}_{field}": value
        for arm in ("left", "right")
        for field, value in (
            ("ee_pose", [0, 0, 0.3, 1, 0, 0, 0]),
            ("ee_joint_state", [1.0]),
        )
    }
    predictions = []
    submitted = []
    settled = []
    budgets = []

    def predict(ref, *, prompt, mode):
        predictions.append((ref, prompt, mode))
        return VLAChunkPrediction(
            actions=np.zeros((1, 14)),
            forward_id=f"f{len(predictions)}",
            prediction_horizon_steps=1,
            execution_horizon_steps=1,
            inference_elapsed_s=0.1,
        )

    def step(action, *, cursor):
        submitted.append(cursor)
        env.last_obs = {
            **env.last_obs,
            "observation_ref": "sid:1:new",
            "physical_step": 1,
        }
        return env.last_obs, 0, False, {}

    env.predict_ref, env.step = predict, step

    @contextmanager
    def model_request(kind):
        budgets.append(kind)
        yield

    primitives = SimpleNamespace(
        env=env,
        vla_client=object(),
        model_request=model_request,
        _check_cancelled=lambda: None,
        execution=SimpleNamespace(
            record_forward=lambda prediction, **kwargs: None,
            discard_open_suffix=lambda **kwargs: settled.append(kwargs),
        ),
    )
    result = tools.pi0_pick(
        primitives,
        SimpleNamespace(output_dir=tmp_path),
        prompt="Pick up the piece",
        max_chunks=2,
    )
    assert predictions == [
        ("sid:0:old", "Pick up the piece", "eval"),
        ("sid:1:new", "Pick up the piece", "eval"),
    ]
    assert budgets == ["vla", "vla"]
    assert len(submitted) == len(settled) == result["chunks_used"] == 2
    assert env.last_obs["instruction"] == "Place a piece"
    import json

    rows = [
        json.loads(line)
        for line in (tmp_path / "vla_inputs.jsonl").read_text().splitlines()
    ]
    assert [
        (row["forward_id"], row["observation_ref"], row["physical_step"])
        for row in rows
    ] == [
        ("f1", "sid:0:old", 0),
        ("f2", "sid:1:new", 1),
    ]
    assert all("actions" not in row for row in rows)
    assert env.reads == []


def test_real_toolkit_fetches_rgb_only_at_decision_boundary(tmp_path):
    from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit

    rpc = SimulatedEnvironmentRPC(limit=10, terminal_after=99)
    original = rpc.call
    reads = []

    def call(method, *args, **kwargs):
        if method == "env.read_rgb":
            reads.append(kwargs["kwargs"])
        return original(method, *args, **kwargs)

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    try:
        assert reads == []
        first_ref = toolkit._primitives.env.current_observation_ref
        feedback = toolkit.get_env_state(None, None, 0)
        assert feedback["observation_ref"] == first_ref
        toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert reads == []
        current_ref = toolkit._primitives.env.current_observation_ref
        projected = toolkit.prepare_model_context(
            {"base_instructions": "Task", "messages": []}
        )
        assert len(reads) == 1
        assert reads[0]["observation_ref"] == current_ref
        assert reads[0]["cameras"] == [camera for camera, _ in tools.CAMERA_ARTIFACTS]
        assert len(projected["messages"][-1]["parts"][0]["content"]) == 4
        assert current_ref != first_ref
    finally:
        toolkit.close()
