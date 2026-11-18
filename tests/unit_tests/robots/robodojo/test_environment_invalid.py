# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""An invalid native scene preserves acknowledged work and has no ability verdict."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.robot_spec import finalize_run
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evaluation import RunFinalizationContext
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.search.rollout import RolloutEvaluator
from reweave.planner.base import API_ERROR
from reweave.runtime.capability import EnvironmentInvalid, PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo, test_runtime_limits

REPOSITORY = test_demo.REPOSITORY
completed_demo = test_demo.completed_demo
agent_environment = test_runtime_limits.agent_environment


@pytest.fixture
def invalid_scene(tmp_path, agent_environment, request):
    rpc = SimulatedEnvironmentRPC(limit=getattr(request, "param", 5), terminal_after=99)
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=5,
        end_flag=[False],
        success=[True],
        unstable_envs=set(),
    )

    def is_success(**kwargs):
        assert not native.unstable_envs, "invalidity must precede official evaluation"
        return False

    native.is_success = is_success

    def apply_action(action, action_type, *, eval_fair):
        native.take_action_cnt[0] += 1
        rpc.actions += 1
        native.unstable_envs.add(0)

    server = agent_environment.__new__(agent_environment)
    server.meta = rpc.meta
    server.observations = rpc.observations
    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native, apply_action=apply_action)]
    )
    server.get_native_obs = lambda: rpc.observation
    original_call = rpc.call

    def call(method, args=(), kwargs=None, timeout_s=None):
        if method == "env.step":
            return server.step_native(*args)
        if method == "env.is_success":
            return server.is_success()
        return original_call(method, args, kwargs, timeout_s)

    rpc.call = call
    directory = tmp_path / "attempt"
    toolkit, _, policy = create_simulated_toolkit(directory / "episode", rpc=rpc)
    yield directory, toolkit, rpc, policy, server
    toolkit.close()


@pytest.mark.parametrize("invalid_scene", [1, 5], indirect=True)
@pytest.mark.parametrize("tool", ["set_gripper", "pi0_pick"])
def test_native_invalidity_settles_action_and_public_state_before_raising(
    invalid_scene, tool
):
    _, toolkit, rpc, policy, server = invalid_scene
    arguments = (
        {"arm": "right", "gripper": 1} if tool == "set_gripper" else {"max_chunks": 1}
    )
    with pytest.raises(EnvironmentInvalid, match="support-arm"):
        toolkit.execute_tool(tool, arguments)
    client = toolkit._primitives.env
    assert client.elapsed_steps() == rpc.actions == server.get_status()["step"] == 1
    assert client.terminated
    assert client._outcome_unknown is False
    assert toolkit.physical_outcome_error is None
    assert toolkit._execution_controller.snapshot.phase == "idle"
    record = toolkit.state.latest_record()
    assert record.terminated and record.step_idx == 1
    events = toolkit._execution_trace.events
    assert sum(event["event_type"] == "control_step" for event in events) == 1
    assert any(event["event_type"] == "control_telemetry" for event in events)
    assert not any(event["event_type"] == "action_outcome_unknown" for event in events)
    if tool == "pi0_pick":
        assert policy.calls == 1
        assert (
            sum(
                event["discarded_action_steps"]
                for event in events
                if event["event_type"] == "action_suffix_discarded"
            )
            == 3
        )
    for operation in (
        lambda: client.step(np.zeros(14)),
        toolkit.solved,
        lambda: toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1}),
    ):
        with pytest.raises(EnvironmentInvalid):
            operation()
    assert rpc.actions == 1


def test_invalid_scene_is_infrastructure_with_completed_work_and_null_success(
    invalid_scene, completed_demo, tmp_path
):
    directory, toolkit, rpc, _, _ = invalid_scene
    captured, _, _, runner, request = completed_demo
    toolkit.export_harness_context(str(directory / "episode"))
    requests = []

    def model(messages, info):
        requests.append(messages)
        return ModelResponse(
            parts=[ToolCallPart("pi0_pick", {"max_chunks": 1}, "invalid-scene")]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate_json(
            (captured / "baseline/bundle.json").read_text()
        ),
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="SIMULATED",
        user_message="Inspect the fixture",
        toolkit=toolkit,
        max_turns=5,
    )
    assert result.termination_class == API_ERROR
    assert result.api_error_retryable is False
    assert "support-arm" in result.error
    assert len(requests) == 1 and rpc.actions == 1
    finalize_run(
        RunFinalizationContext(
            output_dir=directory / "episode",
            robot_name="robodojo",
            task_desc=request.instance.initialization,
            environment_success=None,
            termination_class=result.termination_class,
            agent_error=result.error,
            elapsed_s=1.0,
            planner="SIMULATED",
            model="scripted",
            reasoning_effort="none",
            max_turns=5,
            planner_timeout_s=30,
            finish_result=None,
            stats=result.stats,
        )
    )
    atomic_json(
        directory / "episode/transcript_simulated.json",
        {
            "mode": "SIMULATED",
            "run_configuration": {
                "robot_name": "robodojo",
                **request.instance.initialization,
            },
            "bundle_id": request.bundle_id,
            "official_success": False,
            "termination_class": result.termination_class,
            "api_error_retryable": result.api_error_retryable,
            "stats": result.stats,
            "error": result.error,
        },
    )
    atomic_json(directory / "exit.json", {"returncode": 0, "elapsed_s": 1.0})
    path = directory / "episode/result.json"
    original = path.read_bytes()
    verdict = json.loads(original)
    assert verdict["valid"] is False and verdict["success"] is None
    record = RolloutEvaluator(
        runner, repository=REPOSITORY, workspace=tmp_path / "search"
    )._close(request, directory)
    assert path.read_bytes() == original
    assert record.failure_origin == "infrastructure"
    assert record.infrastructure_retryable is False
    assert (
        record.usage.physical_steps == 1
        and record.usage.physical_steps_complete is True
    )
    assert record.official_success is False


@pytest.mark.parametrize("value", ["missing", "", True])
def test_missing_or_malformed_validity_acknowledgement_remains_unknown(tmp_path, value):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    original_call = rpc.call

    def call(method, *args, **kwargs):
        response = original_call(method, *args, **kwargs)
        if method == "env.step":
            if value == "missing":
                response[3].pop("environment_error")
            else:
                response[3]["environment_error"] = value
        return response

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert toolkit._primitives.env.elapsed_steps() == 0
        assert rpc.actions == 1
    finally:
        toolkit.close()
