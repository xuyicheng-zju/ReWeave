"""Server snapshots never put images in physical ACKs or prediction requests."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from examples.robodojo_demo import simulated_observation
from reweave.benchmarks.robodojo.access import CAMERAS
from reweave.benchmarks.robodojo.env_facade import RoboDojoEnvFacade
from reweave.benchmarks.robodojo.observation_service import ObservationService
from reweave.robots.vla_execution import VLAChunkPrediction
from tests.unit_tests.robots.robodojo import test_runtime_limits

agent_environment = test_runtime_limits.agent_environment


def store(tmp_path):
    service = ObservationService(tmp_path, service_instance_id="owned-service")
    observation = simulated_observation()
    return service, observation, service.publish(observation, 0)


def test_ack_is_small_image_free_and_metadata_is_complete(tmp_path):
    service, observation, compact = store(tmp_path)
    assert len(json.dumps(compact)) < 5000
    assert set(compact["vision"]) == set(CAMERAS)
    for camera in CAMERAS:
        assert set(compact["vision"][camera]) == {
            "shape",
            "color_dtype",
            "has_depth",
            "intrinsic_matrix",
            "extrinsic_matrix",
        }
    assert compact["observation_ref"].startswith("owned-service:0:")
    assert compact == service.compact()
    observation["vision"]["cam_head"]["color"][:] = 255
    decoded = np.asarray(
        Image.open(
            io.BytesIO(
                service.read_rgb(compact["observation_ref"], ["cam_head"])["cam_head"]
            )
        )
    )
    assert np.all(decoded == 0), (
        "server owns its snapshot independently of native buffers"
    )
    assert not service.directory.exists(), "per-step publishing must not archive RGBD"


def test_invalidation_rejects_stale_and_raw_inputs(tmp_path):
    service, observation, old = store(tmp_path)
    service.invalidate()
    for reference in (old["observation_ref"], np.zeros((4, 5, 3)), {"color": []}):
        with pytest.raises(ValueError, match="observation_ref"):
            service.predict_ref(reference, "Put the bottle into the dustbin.")
    current = service.publish(observation, 1)
    assert current["observation_ref"] != old["observation_ref"]
    with pytest.raises(ValueError, match="stale"):
        service.read_rgb(old["observation_ref"], list(CAMERAS))
    with pytest.raises(ValueError, match="cameras"):
        service.read_rgb(current["observation_ref"], ["cam_head", "cam_head"])


def test_retention_is_explicit_lossless_and_history_never_uses_latest(tmp_path):
    service, observation, old = store(tmp_path)
    ref = old["observation_ref"]
    retained = service.retain_observation(ref)
    assert service.retain_observation(ref) == retained
    assert retained["rgb_paths"] == {}
    with np.load(retained["archive_path"], allow_pickle=False) as archive:
        np.testing.assert_equal(
            archive["cam_head__depth"], observation["vision"]["cam_head"]["depth"]
        )
    metadata = json.loads(open(retained["metadata_path"]).read())
    assert metadata["vision"]["cam_head"]["arrays"]["color"]["dtype"] == "uint8"
    assert len((service.directory / "manifest.jsonl").read_text().splitlines()) == 1
    observation["vision"]["cam_head"]["color"][:] = 255
    latest = service.publish(observation, 1)
    old_png = service.read_rgb(ref, ["cam_head"])["cam_head"]
    new_png = service.read_rgb(latest["observation_ref"], ["cam_head"])["cam_head"]
    assert np.all(np.asarray(Image.open(io.BytesIO(old_png))) == 0)
    assert np.all(np.asarray(Image.open(io.BytesIO(new_png))) == 255)
    with pytest.raises(ValueError, match="stale"):
        service.retain_observation(ref)


def test_prediction_uses_server_snapshot_and_returns_only_existing_typed_forward(
    tmp_path,
):
    service, observation, compact = store(tmp_path)
    calls = []

    def predict(obs, mode):
        calls.append((obs, mode))
        assert np.array_equal(
            obs["vision"]["cam_head"]["color"],
            observation["vision"]["cam_head"]["color"],
        )
        return VLAChunkPrediction(
            np.zeros((3, 14), dtype=np.float32), "forward-owned", 3, 3, 0.1
        )

    service.vla = SimpleNamespace(predict_action_batch=predict)
    wire = service.predict_ref(
        compact["observation_ref"], "Put the bottle into the dustbin.", "eval"
    )
    assert len(calls) == 1
    prediction = VLAChunkPrediction.from_rpc(wire)
    assert prediction.actions.shape == (3, 14)
    assert set(wire) == VLAChunkPrediction._RPC_FIELDS
    assert calls[0][0]["instruction"] == "Put the bottle into the dustbin."


def test_facade_has_one_image_transport_and_separate_model_phases(tmp_path):
    service, _, _ = store(tmp_path)
    facade = RoboDojoEnvFacade(SimpleNamespace(observations=service))
    assert "env.render_camera" not in facade._rpc
    assert all(
        f"env.{name}" in facade._rpc
        for name in (
            "retain_observation",
            "read_rgb",
            "back_project_ref",
            "segment_ref",
            "grasps_ref",
            "predict_ref",
        )
    )


def test_native_step_invalidates_before_apply_and_preserves_1100_budget(
    tmp_path, agent_environment
):
    service, observation, compact = store(tmp_path)
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=1100,
        success=[True],
        end_flag=[False],
        unstable_envs=set(),
        is_success=lambda **kw: False,
    )
    calls = []

    def apply(action, action_type, *, eval_fair):
        assert not service.settled
        calls.append(action)
        native.take_action_cnt[0] += 1

    server = agent_environment.__new__(agent_environment)
    server.meta = {"max_episode_steps": 1100}
    server.observations = service
    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native, apply_action=apply)]
    )
    server.get_native_obs = lambda: observation
    result, _, done, info = server.step_native(np.zeros(14))
    assert len(calls) == 1 and not done
    assert info["status"] == {"step": 1, "step_limit": 1100}
    assert result["physical_step"] == 1
    assert result["observation_ref"] != compact["observation_ref"]
    assert len(json.dumps(result)) < 5000
    assert server.get_obs() == result, (
        "readonly observation must reuse the settled snapshot"
    )
    native.take_action_cnt[0] = 1100
    with pytest.raises(RuntimeError, match="budget exhausted"):
        server.step_native(np.zeros(14))
    assert len(calls) == 1


def test_segmentation_and_grasp_rpc_keep_model_phases_separate_and_same_frame(
    tmp_path, monkeypatch
):
    from reweave.benchmarks.robodojo import perception

    service, _, compact = store(tmp_path)
    calls = []
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    def segment(color, **kwargs):
        calls.append("sam3")
        assert color.shape == (4, 5, 3)
        return SimpleNamespace(found=True, score=0.9, box=[2, 1, 4, 3], mask=mask)

    service.sam3 = SimpleNamespace(segment=segment)
    service.graspgen = object()

    def grasps(obs, **kwargs):
        calls.append("graspgen")
        np.testing.assert_equal(kwargs["mask"], mask)
        assert kwargs["observation_step"] == 0
        assert kwargs["camera"] == "cam_head" and kwargs["mask_score"] == 0.9
        return {"candidates": [{"score": 0.5}], "provenance": {"observation_step": 0}}

    monkeypatch.setattr(perception, "grasps_from_segment", grasps)
    ref = compact["observation_ref"]
    result = service.segment_ref(ref, "bottle")
    assert calls == ["sam3"] and result["found"]
    assert "mask" not in result and isinstance(result["mask_ref"], str)
    generated = service.grasps_ref(ref, result["mask_ref"], "right")
    assert calls == ["sam3", "graspgen"] and generated["candidates"]
    service.publish(simulated_observation(), 1)
    with pytest.raises(ValueError, match="mask_ref"):
        service.grasps_ref(
            service.compact()["observation_ref"], result["mask_ref"], "right"
        )
    assert calls == ["sam3", "graspgen"], (
        "stale masks must fail before dispatching inference"
    )


def test_back_projection_stays_server_local_and_uses_same_depth(tmp_path):
    service, _, compact = store(tmp_path)
    result = service.back_project_ref(compact["observation_ref"], 1, 2)
    assert result == {
        "camera": "cam_head",
        "pixel": [1, 2],
        "depth_m": 1.0,
        "world_xyz": [2.0, -1.0, -1.0],
    }


def test_failed_apply_never_reuses_old_snapshot_or_replays_action(
    tmp_path, agent_environment
):
    service, observation, compact = store(tmp_path)
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=1100,
        success=[True],
        end_flag=[False],
        unstable_envs=set(),
        is_success=lambda **kw: False,
    )

    def apply(*args, **kwargs):
        native.take_action_cnt[0] += 1
        raise RuntimeError("action failed after movement")

    server = agent_environment.__new__(agent_environment)
    server.meta = {"max_episode_steps": 1100}
    server.observations = service
    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native, apply_action=apply)]
    )
    server.get_native_obs = lambda: observation
    with pytest.raises(RuntimeError, match="after movement"):
        server.step_native(np.zeros(14))
    assert native.take_action_cnt == [1] and not service.settled
    with pytest.raises(ValueError, match="stale"):
        service.read_rgb(compact["observation_ref"], ["cam_head"])
    with pytest.raises(RuntimeError, match="no settled observation"):
        server.get_obs()
    with pytest.raises(RuntimeError, match="no settled observation"):
        server.step_native(np.zeros(14))
    assert native.take_action_cnt == [1]


def test_sensor_snapshot_is_retained_only_lossless_and_never_captures(tmp_path):
    service, observation, compact = store(tmp_path)
    ref = compact["observation_ref"]
    with pytest.raises(ValueError, match="explicitly retained"):
        service.read_sensor_snapshot(ref)
    service.retain_observation(ref)
    service.publish(simulated_observation(), 1)
    response = service.read_sensor_snapshot(ref)
    assert response["observation_ref"] == response["metadata"]["observation_ref"] == ref
    assert response["metadata"]["physical_step"] == 0
    with np.load(io.BytesIO(response["archive"]), allow_pickle=False) as archive:
        for camera in CAMERAS:
            np.testing.assert_equal(
                archive[f"{camera}__color"], observation["vision"][camera]["color"]
            )
            np.testing.assert_equal(
                archive[f"{camera}__depth"], observation["vision"][camera]["depth"]
            )
    with pytest.raises(ValueError, match="explicitly retained"):
        service.read_sensor_snapshot(service.compact()["observation_ref"])


def test_failed_vla_retains_selected_forward_input_without_archiving_every_step(
    tmp_path,
):
    service, _, first = store(tmp_path)
    service.publish(simulated_observation(), 1)
    selected = service.compact()["observation_ref"]

    def fail(*args, **kwargs):
        raise RuntimeError("model inference failed")

    service.vla = SimpleNamespace(predict_action_batch=fail)
    with pytest.raises(RuntimeError, match="model inference failed"):
        service.predict_ref(selected, "Put the bottle into the dustbin.")
    saved = service.read_sensor_snapshot(selected)
    assert saved["metadata"]["physical_step"] == 1
    with pytest.raises(ValueError, match="explicitly retained"):
        service.read_sensor_snapshot(first["observation_ref"])
