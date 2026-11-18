# Modified for the ReWeave namespace.
"""Server stage timing does not change action or acknowledgement semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit_tests.robots.robodojo import test_runtime_limits

agent_environment = test_runtime_limits.agent_environment


def _server(agent_environment, events):
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=10,
        end_flag=[False],
        success=[True],
        unstable_envs=set(),
        is_success=lambda **kwargs: False,
    )

    def apply_action(action, action_type, *, eval_fair):
        events.append("apply")
        assert action_type == "joint" and eval_fair is False
        native.take_action_cnt[0] += 1

    server = agent_environment.__new__(agent_environment)
    server.meta = {"max_episode_steps": 10}
    server.observations = SimpleNamespace(invalidate=lambda: None, initialized=False)
    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native, apply_action=apply_action)]
    )
    return server, native


def test_step_timing_separates_execution_observation_and_settlement(
    agent_environment, monkeypatch
):
    events = []
    server, native = _server(agent_environment, events)
    observation = {"state": {"acknowledged_step": 1}}

    def observe():
        events.append("observation")
        assert native.take_action_cnt == [1]
        return observation

    server._capture_observation = observe
    clock = iter([10.0, 12.0, 15.0, 16.0])
    # The fixture loads the adapter directly without registering its module.
    monkeypatch.setattr(
        agent_environment.step_native.__globals__["time"],
        "perf_counter",
        lambda: next(clock),
    )
    obs, reward, done, info = server.step_native(np.zeros(14))
    assert obs is observation and reward == 0.0 and done is False
    assert info == {
        "status": {"step": 1, "step_limit": 10},
        "environment_error": None,
        "server_timing_s": {
            "simulation_apply": 2.0,
            "observation": 3.0,
            "settlement": 1.0,
        },
    }
    assert events == ["apply", "observation"]
    assert native.take_action_cnt == [1]


@pytest.mark.parametrize("failure_stage", ["apply", "observation"])
def test_stage_failure_does_not_return_timing_ack_or_repeat_action(
    agent_environment, failure_stage
):
    events = []
    server, native = _server(agent_environment, events)

    def observe():
        events.append("observation")
        raise RuntimeError("observation failed")

    server._capture_observation = observe
    if failure_stage == "apply":

        def apply_failure(*args, **kwargs):
            events.append("apply")
            # A native failure can occur after changing physical state.
            native.take_action_cnt[0] += 1
            raise RuntimeError("apply failed after movement")

        server.venv.envs[0].apply_action = apply_failure
    with pytest.raises(RuntimeError, match=failure_stage):
        server.step_native(np.zeros(14))
    assert native.take_action_cnt == [1]
    assert events == (
        ["apply"] if failure_stage == "apply" else ["apply", "observation"]
    )
