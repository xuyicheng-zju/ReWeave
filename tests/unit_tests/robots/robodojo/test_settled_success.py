# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Final ACK loss and post-ACK errors retain both physical evidence and receipts."""

import asyncio
import json
import sys
import threading
from dataclasses import replace

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.robot_spec import get_robot_spec
from reweave.cli import main as cli
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle, LoopPolicy, PythonFile, PythonProgram
from reweave.evolution.search.rollout import RolloutEvaluator
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    AGENT_EXECUTION_FAILURE,
    API_ERROR,
    SUCCESS,
)
from reweave.planner.providers.base import NativeTurnFailed
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.utils.persistence import atomic_json
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo

completed_demo = test_demo.completed_demo


@pytest.mark.parametrize("failure", ["lost_ack", "loop", "provider"])
def test_cli_settles_real_frozen_episode_after_final_action_error(
    tmp_path, monkeypatch, completed_demo, failure
):
    captured, _, manifest, runner, request = completed_demo
    directory = tmp_path / "attempt"
    output = directory / "episode"
    limit = manifest.conditions.physical_steps  # the task's official step_lim
    rpc = SimulatedEnvironmentRPC(
        limit=limit,
        terminal_after=1,
        acknowledgement="missing" if failure == "lost_ack" else "valid",
    )
    toolkit, _, policy = create_simulated_toolkit(output, rpc=rpc)
    bundle = AgentBundle.model_validate_json(
        (captured / "baseline/bundle.json").read_text()
    )
    if failure != "lost_ack":
        loop_source = (
            "def run(runtime, context):\n"
            "    response = runtime.plan()\n"
            '    runtime.execute(response["tool_calls"][0])\n'
            '    raise RuntimeError("SIMULATED post-ACK Loop failure")\n'
            if failure == "loop"
            else "async def provider_request(runtime, context):\n"
            '    return await runtime.plan("Started before the final action")\n'
            "\n"
            "async def run(runtime, context):\n"
            "    response = await runtime.plan()\n"
            '    pending = await runtime.start("loop:provider_request", {}, ["model:plan"])\n'
            '    await runtime.execute(response["tool_calls"][0])\n'
            "    return await runtime.join(pending)\n"
        )
        bundle = bundle.model_copy(
            update={
                "loop": LoopPolicy(
                    program=PythonProgram(
                        entrypoint="loop:run",
                        files=(
                            PythonFile(
                                path="loop.py",
                                source=loop_source,
                            ),
                        ),
                    )
                )
            }
        )
    model_calls = []
    provider_started = threading.Event()
    final_action = threading.Event()
    if failure == "provider":
        original_call = rpc.call

        def call(method, args=(), kwargs=None, timeout_s=None):
            if method == "env.step":
                assert provider_started.wait(5), (
                    "the request must start before termination"
                )
            return original_call(method, args, kwargs, timeout_s)

        rpc.call = call
        rpc.after_action = final_action.set

    async def model(messages, info):
        model_calls.append(messages)
        if len(model_calls) > 1:
            provider_started.set()
            assert await asyncio.to_thread(final_action.wait, 5)
            raise NativeTurnFailed("failed", {"error": "provider unavailable"})
        return ModelResponse(
            parts=[ToolCallPart("set_gripper", {"arm": "left", "gripper": 1}, "final")]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model),
        max_tokens=123,
        dashboard_events=NullDashboardEventSink(),
    )
    planner.model = manifest.models["controller"]["model"]
    planner.reasoning_effort = "low"
    spec = replace(get_robot_spec(), init_runtime=lambda *args: ([], {}))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: toolkit)
    monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: planner)
    bundle_path = directory / "bundle.json"
    kernel_path = directory / "kernel.json"
    atomic_json(bundle_path, bundle.model_dump(mode="json"))
    atomic_json(
        kernel_path,
        {
            "conditions": manifest.conditions.model_dump(mode="json"),
            "episode": runner.episode_limits.model_dump(mode="json"),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--robot",
            "robodojo",
            "--planner",
            "native",
            "--task",
            rpc.meta["task"],
            "--layout-set",
            "0",
            "--layout",
            "0",
            "--reasoning-effort",
            "low",
            "--max-episode-steps",
            str(limit),
            "--wait-steps",
            "0",
            "--max-turns",
            "5",
            "--planner-timeout-s",
            "30",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(captured / "memory"),
            "--output-dir",
            str(output),
            "--agent-bundle",
            str(bundle_path),
            "--kernel-limits",
            str(kernel_path),
            "--runtime-manifest",
            str(captured / "baseline/runtime.json"),
        ],
    )
    assert cli.main() == 1  # Retain the Planner error for command-line diagnostics.
    transcript = json.loads(next(output.glob("transcript_*.json")).read_text())
    unknown = failure == "lost_ack"
    planner_termination = {
        "lost_ack": ACTION_OUTCOME_UNKNOWN,
        "loop": AGENT_EXECUTION_FAILURE,
        "provider": API_ERROR,
    }[failure]
    assert transcript["planner_termination_class"] == planner_termination
    assert transcript["termination_class"] == (
        ACTION_OUTCOME_UNKNOWN if unknown else SUCCESS
    )
    assert transcript["official_success"] is not unknown
    assert transcript["environment_success"] is True
    assert transcript["error"]
    assert transcript["api_error_retryable"] is None
    assert transcript["planner_api_error_retryable"] is (
        True if failure == "provider" else None
    )
    if failure == "provider":
        assert not (output / "native/recovery/blocked.json").exists()
    verdict = json.loads((output / "result.json").read_text())
    assert verdict["environment_success"] is True
    assert verdict["environment_result_available"] is True
    assert verdict["valid"] is not unknown
    assert verdict["success"] is (None if unknown else True)
    assert verdict["agent_error_present"] is True
    assert rpc.actions == 1 and policy.calls == 0
    assert len(model_calls) == (2 if failure == "provider" else 1)
    assert toolkit._primitives.env.elapsed_steps() == (0 if unknown else 1)
    if unknown:
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit._primitives.env.step(np.zeros(14))
        with pytest.raises(ValueError, match="reset"):
            toolkit._primitives.env.reset()
        assert rpc.actions == 1
    atomic_json(directory / "exit.json", {"returncode": 1, "elapsed_s": 1.0})
    request = request.model_copy(update={"bundle_id": bundle.record_id})
    record = RolloutEvaluator(
        runner, repository=test_demo.REPOSITORY, workspace=tmp_path / "search"
    )._close(request, directory)
    record.verify()
    assert record.official_success is not unknown
    assert record.failure_origin == ("unknown" if unknown else "none")
    assert record.infrastructure_retryable is None
    assert record.usage.physical_steps == (0 if unknown else 1)
    assert record.usage.physical_steps_complete is not unknown
    assert record.usage.model_calls is None
    turns = [
        json.loads(line)
        for line in (output / "native/native_turns.jsonl").read_text().splitlines()
    ]
    controller_turns = [turn for turn in turns if turn["kind"] == "controller"]
    assert len(controller_turns) == len(model_calls)
    assert len({turn["request_attempt"] for turn in controller_turns}) == len(
        model_calls
    )
    assert record.episode is not None
    assert (directory / "result.json").is_file()
