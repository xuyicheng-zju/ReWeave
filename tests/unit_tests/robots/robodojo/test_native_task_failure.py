# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""A native task failure settles its action and remains a valid unsuccessful run."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.robot_spec import finalize_run
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evaluation import RunFinalizationContext
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.search.rollout import RolloutEvaluator
from reweave.planner.base import PHYSICAL_FAILURE, classify_episode_termination
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo, test_runtime_limits

completed_demo = test_demo.completed_demo
agent_environment = test_runtime_limits.agent_environment


@pytest.mark.parametrize("mode", ["ordinary", "frozen"])
def test_native_failure_closes_as_agent_failure_after_one_ack(
    tmp_path, completed_demo, agent_environment, mode
):
    captured, _, _, runner, request = completed_demo
    rpc = SimulatedEnvironmentRPC(limit=5, terminal_after=99)
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=5,
        end_flag=[False],
        success=[True],
        unstable_envs=set(),
        is_success=lambda **kwargs: False,
    )

    def apply_action(action, action_type, *, eval_fair):
        assert native.success == [True]
        native.take_action_cnt[0] += 1
        rpc.actions += 1
        native.success[0] = False

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
    toolkit.export_harness_context(str(directory / "episode"))
    requests = []

    def model(messages, info):
        requests.append(messages)
        assert len(requests) == 1, "native task failure must stop further planning"
        return ModelResponse(
            parts=[
                ToolCallPart("pi0_pick", {"max_chunks": 3}, "native-failure"),
                ToolCallPart("set_gripper", {"arm": "right", "gripper": 1}, "tail"),
            ]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    if mode == "frozen":
        planner.bind_bundle(
            AgentBundle.model_validate_json(
                (captured / "baseline/bundle.json").read_text()
            ),
            episode_limits=EpisodeLimits(
                program_calls=20, gateway_calls=20, wall_time_s=30
            ),
        )
    try:
        result = planner.solve(
            system_prompt="SIMULATED",
            user_message="Inspect the fixture",
            toolkit=toolkit,
            max_turns=5,
        )
        assert result.error is None and result.termination_class is None
        assert result.api_error_retryable is None
        assert len(requests) == policy.calls == rpc.actions == 1
        assert toolkit._primitives.env.elapsed_steps() == 1
        assert (
            toolkit._primitives.env.terminated and not toolkit._primitives.env.truncated
        )
        assert toolkit.episode_ended()
        assert toolkit.solved() is False
        if mode in {"ordinary", "frozen"}:
            events = [
                json.loads(line)
                for line in (directory / "episode/harness_context/bundle_events.jsonl")
                .read_text()
                .splitlines()
            ]
            assert any(event["kind"] == "model_call_not_executed" for event in events)
        subsequent = toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert subsequent.result["status"] == "not_executed"
        assert "error" not in subsequent.result and rpc.actions == 1
        assert toolkit.execute_tool("view_env_state", {"step": -1}).result["step"] == 1
        termination = classify_episode_termination(
            planner_termination_class=result.termination_class,
            official_success=toolkit.solved(),
        )
        assert termination == PHYSICAL_FAILURE
        finalize_run(
            RunFinalizationContext(
                output_dir=directory / "episode",
                robot_name="robodojo",
                task_desc=request.instance.initialization,
                environment_success=False,
                termination_class=termination,
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
                "termination_class": termination,
                "api_error_retryable": None,
                "stats": result.stats,
                "error": None,
            },
        )
        atomic_json(directory / "exit.json", {"returncode": 0, "elapsed_s": 1.0})
        verdict_path = directory / "episode/result.json"
        original = verdict_path.read_bytes()
        verdict = json.loads(original)
        assert verdict["valid"] is True and verdict["success"] is False
        assert verdict["environment_success"] is False
        record = RolloutEvaluator(
            runner, repository=test_demo.REPOSITORY, workspace=tmp_path / "search"
        )._close(request, directory)
        record.verify()
        assert verdict_path.read_bytes() == original
        assert record.failure_origin == "agent"
        assert record.infrastructure_retryable is None
        assert record.official_success is False
        assert record.usage.physical_steps == 1 and record.usage.physical_steps_complete
    finally:
        toolkit.close()
