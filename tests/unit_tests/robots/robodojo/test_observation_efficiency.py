"""Settled RoboDojo ACKs supply tool observations without redundant RGB-D RPC."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo import tools
from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.utils.rpc.http_rpc import RpcCallMetrics


def test_repeated_gripper_steps_record_each_ack_without_observation_or_task_rpc(
    tmp_path,
):
    rpc = SimulatedEnvironmentRPC(limit=10, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    calls = Counter()
    original = rpc.call

    def call(method, *args, **kwargs):
        calls[method] += 1
        if method == "env.step":
            # A replacement snapshot proves the tools follow each ACK rather
            # than succeeding because a fake mutates their old cached object.
            rpc.observation = deepcopy(rpc.observation)
            rpc.observation["vision"]["cam_head"]["depth"][:] = rpc.actions + 2
            rpc.observation["state"]["left_arm_joint_state"][:] = rpc.actions + 1
        return original(method, *args, **kwargs)

    rpc.call = call
    try:
        for _ in range(3):
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
            point = tools.back_project(toolkit._primitives, None, row=0, col=0)
            assert point["depth_m"] == rpc.actions + 1
            np.testing.assert_array_equal(
                toolkit._primitives._last_obs["state"]["left_arm_joint_state"],
                np.full(6, rpc.actions),
            )
        assert calls == {
            "env.step": 3,
            "env.retain_observation": 3,
            "env.back_project_ref": 3,
        }
        assert calls["env.read_rgb"] == 0
        assert toolkit._state.latest_record().extras["control_steps"] == 3
        trace = toolkit._execution_trace.events
        assert sum(row["event_type"] == "control_step" for row in trace) == 3
        assert sum(row["event_type"] == "control_telemetry" for row in trace) == 3
        for step in range(4):
            for _, artifact in tools.CAMERA_ARTIFACTS:
                assert toolkit._state.load_bytes(artifact, step=step)
    finally:
        toolkit.close()


def test_explicit_observation_read_replaces_cache_and_failed_read_invalidates_it():
    rpc = SimulatedEnvironmentRPC()
    env = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    before = env.last_obs
    rpc.observation = deepcopy(rpc.observation)
    rpc.observation["vision"]["cam_head"]["depth"][:] = 7
    # A sensor mutation in this fake is not a new public frame until the
    # simulated server explicitly publishes it; get_obs never recaptures.
    rpc.observations.publish(rpc.observation, rpc.actions)
    assert env.get_obs() is env.last_obs
    assert env.last_obs is not before
    assert tools.back_project(SimpleNamespace(env=env), None, 0, 0)["depth_m"] == 7
    original = rpc.call

    def fail_read(method, *args, **kwargs):
        if method == "env.get_obs":
            raise TimeoutError("snapshot unavailable")
        return original(method, *args, **kwargs)

    rpc.call = fail_read
    with pytest.raises(TimeoutError):
        env.get_obs()
    assert env.last_obs is None
    with pytest.raises(RuntimeError, match="no settled current observation"):
        tools.back_project(SimpleNamespace(env=env, _last_obs=before), None, 0, 0)


@pytest.mark.parametrize("acknowledgement", ["missing", "malformed"])
def test_unknown_action_clears_observation_and_never_refreshes_or_replays(
    acknowledgement,
):
    rpc = SimulatedEnvironmentRPC(acknowledgement=acknowledgement)
    env = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    previous = env.last_obs
    with pytest.raises(PhysicalOutcomeUnknown):
        env.step(np.zeros(14))
    assert env.last_obs is None and rpc.actions == 1
    with pytest.raises(PhysicalOutcomeUnknown):
        env.get_obs()
    with pytest.raises(RuntimeError, match="no settled current observation"):
        tools.back_project(SimpleNamespace(env=env, _last_obs=previous), None, 0, 0)
    with pytest.raises(PhysicalOutcomeUnknown):
        env.step(np.zeros(14))
    assert rpc.actions == 1


def test_ack_diagnostics_stay_in_trace_and_do_not_change_tool_result(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call
    metrics = RpcCallMetrics(
        request_encode_s=0.01,
        request_compress_s=0.0,
        response_wait_read_s=0.7,
        response_decompress_s=0.01,
        response_decode_s=0.02,
        total_s=0.74,
        request_body_bytes=100,
        request_wire_bytes=100,
        request_content_encoding="identity",
        response_wire_bytes=1000,
        response_body_bytes=3000,
        content_encoding="gzip",
        server_dispatch_s=0.5,
        server_encode_s=None,
        server_compress_s=None,
    )

    def call(method, *args, **kwargs):
        result = original(method, *args, **kwargs)
        if method == "env.step":
            result[3]["server_timing_s"] = {"simulation_apply": 0.2, "observation": 0.3}
            rpc.last_call_metrics = metrics
        return result

    rpc.call = call
    try:
        result = toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert "server_timing_s" not in str(result.result)
        evidence = next(
            row["public_evidence"]
            for row in toolkit._execution_trace.events
            if row["event_type"] == "control_telemetry"
        )
        assert evidence["simulator.simulation_apply_s"] == 0.2
        assert evidence["rpc.response_wire_bytes"] == 1000
        assert "rpc.server_encode_s" not in evidence
    finally:
        toolkit.close()


def test_unreadable_observation_does_not_erase_an_acknowledged_control_step(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    original = rpc.call

    def call(method, *args, **kwargs):
        response = original(method, *args, **kwargs)
        if method == "env.step":
            _, reward, done, info = response
            return None, reward, done, info
        return response

    rpc.call = call
    active = toolkit._execution_controller
    active.begin("set_gripper", {"arm": "right", "gripper": 1})
    try:
        with pytest.raises(ValueError, match="snapshot acknowledgement"):
            toolkit._primitives.env.step(np.zeros(14))
        assert toolkit._primitives.env.last_obs is None
        assert toolkit._primitives.env.elapsed_steps() == rpc.actions == 1
        trace = toolkit._execution_trace.events
        assert sum(row["event_type"] == "control_step" for row in trace) == 1
        assert not any(row["event_type"] == "action_outcome_unknown" for row in trace)
    finally:
        active.finish("failed")
        toolkit.close()
