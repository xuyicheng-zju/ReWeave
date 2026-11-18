"""An admitted failed transfer is an honest VLA sensor gap, never a snapshot."""

import copy
import json

import pytest

from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode
from reweave.evolution.evidence.trace import _Episode, _forward_inputs


def episode_with_transfer(tmp_path, *, defect=None):
    root = tmp_path / "episode"
    root.mkdir()
    row = {
        "forward_id": "forward",
        "observation_ref": "scene:0:" + "a" * 32,
        "physical_step": 0,
        "prompt": "Pick the ring",
        "mode": "eval",
        "prediction_horizon_steps": 50,
        "execution_horizon_steps": 30,
    }
    event = {
        "event_type": "vla_forward",
        "forward_id": "forward",
        "execution_ref": "pick",
        "event_index": 0,
        "instruction": "Pick the ring",
        "prediction_horizon_steps": 50,
        "execution_horizon_steps": 30,
    }
    failure = {
        "error_type": "SSHException",
        "error": "SSH session not active",
        "remote_output": "/remote/owned/episode",
        "complete": False,
        "automatic_replay": False,
        "time": 123,
    }
    if defect == "mismatched_forward":
        row["forward_id"] = "foreign"
    if defect == "wrong_step":
        row["physical_step"] = 1
    if defect == "wrong_horizon":
        event["execution_horizon_steps"] = 29
    if defect == "invalid_failure":
        failure["complete"] = True
    states = {"steps": [{"extras": {"observation_ref": "scene:0:" + "b" * 32}}]}
    if defect == "foreign_scene":
        row["observation_ref"] = "foreign:0:" + "a" * 32
    if defect == "inconsistent_scene":
        states["steps"].append({"extras": {"observation_ref": "foreign:1:" + "c" * 32}})
    if defect == "missing_scene":
        states = {"steps": []}
    if defect == "wrong_instruction":
        event["instruction"] = "Pick the cross"
    if defect != "missing_states":
        (root / "states.json").write_text(json.dumps(states))
    (root / "vla_inputs.jsonl").write_text(json.dumps(row) + "\n")
    (root / "execution_trace.jsonl").write_text(json.dumps(event) + "\n")
    if defect != "no_failure":
        (root / "environment_transfer_failure.json").write_text(json.dumps(failure))
    if defect == "bad_manifest":
        (root / "server_evidence.json").write_text(json.dumps({"invalid": True}))
    if defect == "unreadable_failure":
        (root / "environment_transfer_failure.json").write_text("invalid-json")
    files = inventory_episode(root)
    if defect == "unadmitted_failure":
        files.pop("environment_transfer_failure.json")
    return (
        _Episode(tmp_path, EpisodeEvidence(evidence_id="episode", files=files), root),
        [event],
        row,
    )


def test_transfer_failure_preserves_input_text_and_reference_without_sensor_claims(
    tmp_path,
):
    episode, events, row = episode_with_transfer(tmp_path)
    original_files = copy.deepcopy(dict(episode.evidence.files))
    inputs, gaps = _forward_inputs(episode, events)
    assert len(inputs) == len(gaps) == 1
    assert inputs[0]["prompt"] == row["prompt"]
    assert inputs[0]["observation_ref"] == row["observation_ref"]
    assert (
        inputs[0]["physical_step"] == 0 and inputs[0]["last_control_step_index"] is None
    )
    assert inputs[0]["sensor_snapshot"] is None and inputs[0]["images"] == []
    assert gaps[0]["kind"] == "vla_sensor_evidence_transfer_failed"
    assert gaps[0]["transfer_failure"]["path"].endswith(
        "/environment_transfer_failure.json"
    )
    assert gaps[0]["input_record"] == inputs[0]["input_record"]
    assert "read_sensor_snapshot" not in json.dumps([inputs, gaps])
    assert inventory_episode(episode.root) == original_files


@pytest.mark.parametrize(
    "defect",
    [
        "no_failure",
        "unadmitted_failure",
        "invalid_failure",
        "unreadable_failure",
        "bad_manifest",
        "mismatched_forward",
        "wrong_step",
        "wrong_horizon",
        "foreign_scene",
        "inconsistent_scene",
        "missing_scene",
        "missing_states",
        "wrong_instruction",
    ],
)
def test_missing_sensor_admission_does_not_bypass_other_evidence_checks(
    tmp_path, defect
):
    episode, events, _ = episode_with_transfer(tmp_path, defect=defect)
    with pytest.raises(ValueError):
        _forward_inputs(episode, events)


@pytest.mark.parametrize("valid", [False, True])
def test_interrupted_collection_requires_admitted_invalid_result(tmp_path, valid):
    episode, events, row = episode_with_transfer(tmp_path, defect="no_failure")
    root = episode.root
    (root / "environment_stop_requested.json").write_text(
        json.dumps(
            {
                "service_instance_id": "original-owner",
                "remote_output": "/remote/owned/episode",
                "automatic_replay": False,
            }
        )
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "valid": valid,
                "environment_result_available": False,
                "success": None,
            }
        )
    )
    files = inventory_episode(root)
    episode = _Episode(
        tmp_path, EpisodeEvidence(evidence_id="episode", files=files), root
    )
    if valid:
        with pytest.raises(ValueError, match="interrupted-transfer"):
            _forward_inputs(episode, events)
        return
    inputs, gaps = _forward_inputs(episode, events)
    assert inputs[0]["prompt"] == row["prompt"]
    assert inputs[0]["sensor_snapshot"] is None and inputs[0]["images"] == []
    assert gaps[0]["kind"] == "vla_sensor_evidence_not_admitted_after_stop"
    assert inventory_episode(root) == files
    assert not (root / "environment_transfer_failure.json").exists()


@pytest.mark.parametrize("success", [False, True, None])
def test_stopped_official_failure_preserves_result_with_sensor_gap(tmp_path, success):
    episode, events, row = episode_with_transfer(tmp_path, defect="no_failure")
    root = episode.root
    (root / "environment_stop_requested.json").write_text(
        json.dumps(
            {
                "service_instance_id": "original-owner",
                "remote_output": "/remote/owned/episode",
                "automatic_replay": False,
            }
        )
    )
    result = {
        "valid": True,
        "environment_result_available": True,
        "success": success,
        "environment_success": success,
        "success_source": "environment.is_success",
    }
    path = root / "result.json"
    path.write_text(json.dumps(result))
    original = path.read_bytes()
    files = inventory_episode(root)
    episode = _Episode(
        tmp_path, EpisodeEvidence(evidence_id="episode", files=files), root
    )
    if success is not False:
        with pytest.raises(ValueError, match="interrupted-transfer"):
            _forward_inputs(episode, events)
    else:
        inputs, gaps = _forward_inputs(episode, events)
        assert inputs[0]["prompt"] == row["prompt"]
        assert inputs[0]["images"] == []
        assert inputs[0]["sensor_snapshot"] is None
        assert gaps[0]["kind"] == "vla_sensor_evidence_not_admitted_after_stop"
    assert path.read_bytes() == original
    assert inventory_episode(root) == files
