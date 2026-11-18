"""A paired client rejects old arrays and preserves already settled actions."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient
from reweave.runtime.capability import PhysicalOutcomeUnknown


def test_legacy_server_is_rejected_before_reset_or_any_action():
    rpc = SimulatedEnvironmentRPC()
    expected = dict(rpc.meta)
    rpc.meta.pop("observation_transport")
    calls = []
    original = rpc.call

    def spy(method, *args, **kwargs):
        calls.append(method)
        return original(method, *args, **kwargs)

    rpc.call = spy
    with pytest.raises(ValueError, match="env_meta mismatch"):
        RoboDojoEnvClient(rpc, expected_meta=expected)
    assert calls == ["env.get_env_meta"] and rpc.actions == 0


@pytest.mark.parametrize("violation", ["foreign_instance", "wrong_step", "raw_rgbd"])
def test_bad_snapshot_keeps_the_physical_ack_and_trace(tmp_path, violation):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call

    def corrupt(method, *args, **kwargs):
        result = original(method, *args, **kwargs)
        if method == "env.step":
            obs, reward, done, info = result
            obs = deepcopy(obs)
            if violation == "foreign_instance":
                obs["service_instance_id"] = "another-scene"
            elif violation == "wrong_step":
                obs["physical_step"] += 1
            else:
                obs["vision"]["cam_head"]["color"] = np.zeros((2, 2, 3), np.uint8)
            return obs, reward, done, info
        return result

    rpc.call = corrupt
    active = toolkit._execution_controller
    active.begin("set_gripper", {"arm": "right", "gripper": 1})
    try:
        with pytest.raises(ValueError):
            toolkit._primitives.env.step(np.zeros(14))
        assert toolkit._primitives.env.last_obs is None
        assert toolkit._primitives.env.elapsed_steps() == rpc.actions == 1
        events = toolkit._execution_trace.events
        assert sum(row["event_type"] == "control_step" for row in events) == 1
        assert not any(row["event_type"] == "action_outcome_unknown" for row in events)
    finally:
        active.finish("failed")
        toolkit.close()


def test_only_explicit_sensor_diagnostics_download_original_rgbd():
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    env = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    counts = Counter()
    original = rpc.call

    def spy(method, *args, **kwargs):
        counts[method] += 1
        return original(method, *args, **kwargs)

    rpc.call = spy
    for _ in range(5):
        obs, *_ = env.step(np.zeros(14))
        assert all(
            "color" not in value and "depth" not in value
            for value in obs["vision"].values()
        )
    assert counts == {"env.step": 5}
    reference = env.current_observation_ref
    env.retain_observation(reference)
    full = env.read_sensor_snapshot(reference)
    np.testing.assert_array_equal(
        full["vision"]["cam_head"]["color"],
        rpc.observation["vision"]["cam_head"]["color"],
    )
    np.testing.assert_array_equal(
        full["vision"]["cam_head"]["depth"],
        rpc.observation["vision"]["cam_head"]["depth"],
    )
    assert counts["env.read_sensor_snapshot"] == 1 and counts["env.read_rgb"] == 0


def test_unknown_action_cannot_request_server_models_or_sensor_reads():
    rpc = SimulatedEnvironmentRPC(acknowledgement="missing")
    env = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    reference = env.current_observation_ref
    with pytest.raises(PhysicalOutcomeUnknown):
        env.step(np.zeros(14))
    with pytest.raises(PhysicalOutcomeUnknown):
        env.predict_ref(reference, "Do not execute")
    with pytest.raises(PhysicalOutcomeUnknown):
        env.read_rgb(reference, ["cam_head"])
    assert rpc.actions == 1
