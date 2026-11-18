"""Selective offline retention keeps actual original pixels and perception inputs."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from examples.robodojo_demo import simulated_observation
from reweave.benchmarks.robodojo.observation_service import ObservationService
from reweave.evolution.evidence import sensor_retention as retention
from reweave.evolution.evidence.server import _validate_snapshot, read_sensor_snapshot


@pytest.fixture
def episode_case(tmp_path):
    episode = tmp_path / "attempt/episode"
    episode.mkdir(parents=True)
    (episode.parent / "exit.json").write_text('{"returncode": 0}')
    (episode / "transcript_test.json").write_text(
        '{"cleanup_errors": [], "stats": {"visible_turns": 2}}'
    )
    closed = {
        "returncode": 0,
        "parent_closed": True,
        "recording_close_receipt_verified": True,
        "service_instance_id": "test-scene",
        "automatic_replay": False,
    }
    (episode / "environment_owner_exit.json").write_text(json.dumps(closed))
    service = ObservationService(tmp_path / "remote", service_instance_id="test-scene")
    references, snapshots, states, contexts = [], {}, [], []
    for step in range(3):
        obs = simulated_observation()
        for camera in obs["vision"].values():
            camera["color"] = np.full_like(camera["color"], 10 + step)
        ref = service.publish(obs, step)["observation_ref"]
        record = service.retain_observation(ref)
        references.append(ref)
        snapshots[ref] = {key: record[key] for key in ("archive_path", "metadata_path")}
        snapshots[ref].update(
            archive_bytes=Path(record["archive_path"]).stat().st_size,
            metadata_bytes=Path(record["metadata_path"]).stat().st_size,
        )
        states.append({"step_idx": step, "extras": {"observation_ref": ref}})
        paths = {}
        for index, camera in enumerate(obs["vision"]):
            path = episode / f"harness_context/observations/{step}-{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(obs["vision"][camera]["color"]).save(path)
            paths[f"/input/messages/0/parts/0/content/{index}/data"] = {
                "path": str(path.relative_to(episode / "harness_context")),
                "bytes": path.stat().st_size,
            }
        contexts.append(
            {
                "kind": "context_built",
                "input": {
                    "observation": {"observation_ref": ref},
                    "current_camera_observation": {
                        "cameras": [{"name": c} for c in obs["vision"]]
                    },
                },
                "image_payloads": paths,
            }
        )
    manifest = {
        "schema_version": 1,
        "transport_config_path": str(tmp_path / "connection.json"),
        "transport_python": "/unused/python",
        "remote_observation_root": str(service.directory),
        "snapshots": snapshots,
    }
    (episode / "server_evidence.json").write_text(json.dumps(manifest))
    (episode / "states.json").write_text(json.dumps({"steps": states}))
    events = [
        contexts[0],
        {
            "kind": "action_submitted",
            "action_id": "perception",
            "operation": "back_project",
            "arguments": {"row": 1, "col": 1},
        },
        {
            "kind": "action_completed",
            "action_id": "perception",
            "operation": "back_project",
            "result": {},
        },
    ]
    for step in (1, 2):
        events.extend(
            [
                {
                    "kind": "action_submitted",
                    "action_id": str(step),
                    "operation": "set_gripper",
                    "arguments": {},
                },
                {
                    "kind": "action_completed",
                    "action_id": str(step),
                    "operation": "set_gripper",
                    "result": {"observation_ref": references[step]},
                },
            ]
        )
        if step == 1:
            events.append(contexts[step])
    (episode / "harness_context/bundle_events.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in events)
    )
    (episode / "execution_trace.jsonl").write_text("")
    native_events = episode / "native/provider/events.jsonl"
    native_events.parent.mkdir(parents=True)
    native_events.write_text(
        "".join(
            json.dumps(
                {
                    "direction": "outgoing",
                    "message": {
                        "method": "turn/start",
                        "params": {
                            "input": [
                                {
                                    "type": "localImage",
                                    "path": str(
                                        episode / "harness_context" / image["path"]
                                    ),
                                }
                                for image in contexts[step]["image_payloads"].values()
                            ]
                        },
                    },
                }
            )
            + "\n"
            for step in (0, 1)
        )
    )
    return episode, service.directory, references


def test_retirement_preserves_original_decisions_and_exact_perception(
    episode_case, tmp_path
):
    episode, sensors, refs = episode_case
    images = {p: p.read_bytes() for p in episode.rglob("*.png")}
    raw = (sensors / refs[0].split(":")[-1] / "rgbd.npz").read_bytes()
    plan = retention.plan_retention(episode)
    assert set(plan["model_observations"]) == set(refs[:2])
    assert plan["raw_rgbd_observations"] == refs[:1]
    assert plan["state_video_observations"] == refs[2:]
    retention.verify_model_pixels(episode, sensors, plan)
    plan = retention.publish_retention_index(episode, plan)
    result = retention.retire_unneeded_sensors(sensors, plan)
    assert result["raw_snapshots_kept"] == 1
    assert (sensors / refs[0].split(":")[-1] / "rgbd.npz").read_bytes() == raw
    assert all(p.read_bytes() == data for p, data in images.items())
    assert not (sensors / refs[1].split(":")[-1] / "rgbd.npz").exists()
    assert not (sensors / refs[2].split(":")[-1] / "rgbd.npz").exists()
    assert (
        retention.retire_unneeded_sensors(sensors, plan)["removed_file_bytes_this_pass"]
        == 0
    )
    manifest = json.loads((episode / "server_evidence.json").read_text())
    _validate_snapshot(
        sensors / refs[0].split(":")[-1], refs[0], manifest["snapshots"][refs[0]]
    )
    with pytest.raises(ValueError, match="not retained"):
        read_sensor_snapshot(episode, "episode-test", refs[2], tmp_path / "cache")


def test_wrong_pixels_cannot_publish_or_retire(episode_case):
    episode, sensors, _ = episode_case
    plan = retention.plan_retention(episode)
    image = next(episode.glob("harness_context/observations/0-*.png"))
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="length"):
        retention.verify_model_pixels(episode, sensors, plan)
    with pytest.raises(ValueError, match="verified first"):
        retention.publish_retention_index(episode, plan)
    assert len(list(sensors.glob("*/rgbd.npz"))) == 3


def test_unknown_operation_is_conservatively_unproven(episode_case):
    episode, sensors, _ = episode_case
    path = episode / "harness_context/bundle_events.jsonl"
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "action_submitted",
                    "operation": "custom_perception",
                    "action_id": "unknown",
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="cannot prove"):
        retention.plan_retention(episode)
    assert len(list(sensors.glob("*/rgbd.npz"))) == 3


def test_interrupted_deletion_resumes_without_touching_required_evidence(
    episode_case, monkeypatch
):
    episode, sensors, refs = episode_case
    plan = retention.verify_model_pixels(
        episode, sensors, retention.plan_retention(episode)
    )
    plan = retention.publish_retention_index(episode, plan)
    unlink = Path.unlink
    count = 0

    def interrupt(path, *args, **kwargs):
        nonlocal count
        if path.name == "rgbd.npz":
            count += 1
            if count == 2:
                raise InterruptedError("simulated storage interruption")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(InterruptedError):
        retention.retire_unneeded_sensors(sensors, plan)
    assert (sensors / refs[0].split(":")[-1] / "rgbd.npz").exists()
    monkeypatch.setattr(Path, "unlink", unlink)
    retention.retire_unneeded_sensors(sensors, plan)
    assert len(list(sensors.glob("*/rgbd.npz"))) == 1
    assert retention.read_retention(episode)["complete"]


def test_pending_index_is_not_readable_and_recovers(episode_case, monkeypatch):
    episode, sensors, _ = episode_case
    plan = retention.verify_model_pixels(
        episode, sensors, retention.plan_retention(episode)
    )
    from reweave.utils import persistence

    original = persistence.atomic_json

    def interrupt(path, value):
        if path.name == "sensor_retention.json":
            raise InterruptedError("publication interrupted")
        return original(path, value)

    monkeypatch.setattr(persistence, "atomic_json", interrupt)
    with pytest.raises(InterruptedError):
        retention.publish_retention_index(episode, plan)
    with pytest.raises(ValueError, match="incomplete"):
        retention.read_retention(episode)
    monkeypatch.setattr(persistence, "atomic_json", original)
    retention.publish_retention_index(episode, plan)
    assert retention.read_retention(episode)["complete"]
    assert len(list(sensors.glob("*/rgbd.npz"))) == 3


@pytest.mark.parametrize("temporary_alias", [False, True])
def test_postrun_pipeline_checks_offline_reader_before_retirement(
    episode_case, tmp_path, monkeypatch, temporary_alias
):
    import shutil
    import tempfile

    from reweave.evolution.evidence import server as server_evidence
    from reweave.evolution.evidence import trace as trace_context

    episode, sensors, refs = episode_case
    order = []
    if temporary_alias:
        actual = tmp_path / "actual-tmp"
        actual.mkdir()
        alias = tmp_path / "tmp-alias"
        alias.symlink_to(actual, target_is_directory=True)
        monkeypatch.setattr(tempfile, "tempdir", str(alias))

    def fetch(request):
        staging = Path(request["staging_directory"])
        assert staging.resolve() == staging
        row = request["manifest"]["snapshots"][request["observation_ref"]]
        for key, filename in [
            ("archive_path", "rgbd.npz"),
            ("metadata_path", "metadata.json"),
        ]:
            shutil.copy2(row[key], Path(request["staging_directory"]) / filename)
        order.append("fetch")

    def reader(*args, **kwargs):
        retention.read_retention(episode)
        result = read_sensor_snapshot(
            episode, "local-test", refs[0], episode.parent / "cache"
        )
        assert result["downloaded_bytes"] == 0
        assert Path(result["archive_path"]).is_relative_to(episode / "sensor_originals")
        order.append("reader")
        return {}

    def retire(manifest, plan):
        assert order[-1] == "reader"
        order.append("retire")
        return retention.retire_unneeded_sensors(sensors, plan)

    monkeypatch.setattr(server_evidence, "_run_fetch_worker", fetch)
    monkeypatch.setattr(server_evidence, "retire_server_sensors", retire)
    monkeypatch.setattr(trace_context, "build_episode_context", reader)
    retention.finalize_episode_sensors(episode.parent)
    assert order == ["fetch", "fetch", "reader", "retire"]
    assert (episode / "cam_head.png/00.png").read_bytes() == (
        episode / "harness_context/observations/0-cam_head.png"
    ).read_bytes()
    before = list(order)
    retention.finalize_episode_sensors(episode.parent)
    assert order == before
    assert set(retention.read_retention(episode)["raw_rgbd_observations"]) == {refs[0]}


def test_local_originals_work_without_remote_and_missing_local_never_downloads(
    episode_case, tmp_path, monkeypatch
):
    import shutil

    from reweave.evolution.evidence import server as server_evidence

    episode, sensors, refs = episode_case
    plan = retention.verify_model_pixels(
        episode, sensors, retention.plan_retention(episode)
    )
    raw = (sensors / refs[0].split(":")[-1] / "rgbd.npz").read_bytes()
    retention.retain_local_sensors(episode, sensors, plan)
    retention.publish_retention_index(episode, plan)
    shutil.rmtree(sensors)

    def forbidden(*args, **kwargs):
        pytest.fail("local evidence must never contact the server")

    monkeypatch.setattr(server_evidence, "_run_fetch_worker", forbidden)
    result = read_sensor_snapshot(episode, "local-test", refs[0], tmp_path / "cache")
    assert Path(result["archive_path"]).read_bytes() == raw
    assert result["downloaded_bytes"] == 0
    assert not (tmp_path / "cache").exists()
    Path(result["archive_path"]).unlink()
    with pytest.raises(ValueError, match="regular local file"):
        read_sensor_snapshot(episode, "local-test", refs[0], tmp_path / "cache")


def test_pipeline_retains_every_sensor_when_binding_is_unproven(
    episode_case, monkeypatch
):
    from reweave.evolution.evidence import server as server_evidence

    episode, sensors, _ = episode_case
    path = episode / "harness_context/bundle_events.jsonl"
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "action_submitted",
                    "operation": "unknown_perception",
                    "action_id": "unknown",
                }
            )
            + "\n"
        )

    def forbidden(*args, **kwargs):
        pytest.fail("an unproven retention plan must never fetch or retire sensors")

    monkeypatch.setattr(server_evidence, "_run_fetch_worker", forbidden)
    monkeypatch.setattr(server_evidence, "retire_server_sensors", forbidden)
    retention.finalize_episode_sensors(episode.parent)
    assert len(list(sensors.glob("*/rgbd.npz"))) == 3
    status = json.loads((episode / "sensor_storage.json").read_text())
    assert status["sensor_retirement"] == "not_proven_all_originals_retained"


def test_checkpointed_download_survives_later_transport_failure(
    episode_case, monkeypatch
):
    import shutil

    from reweave.evolution.evidence import server as server_evidence
    from reweave.evolution.evidence import trace as trace_context

    episode, sensors, refs = episode_case
    calls = []

    def fetch(request):
        reference = request["observation_ref"]
        calls.append(reference)
        if len(calls) == 2:
            raise RuntimeError("transient SSH banner failure")
        row = request["manifest"]["snapshots"][reference]
        for key, filename in [
            ("archive_path", "rgbd.npz"),
            ("metadata_path", "metadata.json"),
        ]:
            shutil.copy2(row[key], Path(request["staging_directory"]) / filename)

    monkeypatch.setattr(server_evidence, "_run_fetch_worker", fetch)
    monkeypatch.setattr(trace_context, "build_episode_context", lambda *a, **k: {})
    monkeypatch.setattr(
        server_evidence,
        "retire_server_sensors",
        lambda manifest, plan: retention.retire_unneeded_sensors(sensors, plan),
    )
    with pytest.raises(RuntimeError, match="transient SSH"):
        retention.finalize_episode_sensors(episode.parent)
    cache = episode.parent / ".scratch/sensor-retention"
    assert len(list(cache.iterdir())) == 1
    assert not (episode / "sensor_retention.json").exists()
    retention.finalize_episode_sensors(episode.parent)
    assert len(calls) == 3 and calls[1] == calls[2] and calls[0] != calls[1]
    assert json.loads((episode / "sensor_storage.json").read_text())["complete"]
    assert not cache.exists()


def test_snapshot_transport_retry_uses_fresh_staging(episode_case, monkeypatch):
    import shutil
    import time

    from reweave.evolution.evidence import server as server_evidence

    episode, _, refs = episode_case
    manifest = json.loads((episode / "server_evidence.json").read_text())
    reference = next(iter(manifest["snapshots"]))
    root = episode.parent / "retry-cache"
    root.mkdir()
    pending_paths = []

    def fetch(request):
        pending = Path(request["staging_directory"])
        pending_paths.append(pending)
        if len(pending_paths) == 1:
            (pending / "rgbd.npz").write_bytes(b"partial")
            raise RuntimeError("server sensor fetch failed (SSHException)")
        row = manifest["snapshots"][reference]
        for key, filename in [
            ("archive_path", "rgbd.npz"),
            ("metadata_path", "metadata.json"),
        ]:
            shutil.copy2(row[key], pending / filename)

    monkeypatch.setattr(server_evidence, "_run_fetch_worker", fetch)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    retention._cached_retention_snapshot(manifest, reference, root)
    assert len(pending_paths) == 2
    assert pending_paths[0] != pending_paths[1]
    assert not pending_paths[0].exists()
    retention._cached_retention_snapshot(manifest, reference, root)
    assert len(pending_paths) == 2
