# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Host evidence faults stop I/O and cannot relabel unknown physical outcomes."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.planner.base import ACTION_OUTCOME_UNKNOWN, API_ERROR
from reweave.planner.bundle_runtime import BundleRuntime, CandidateExecutionFailure
from reweave.runtime.capability import (
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.runtime.execution import ProgramExecution
from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits
from reweave.tools.toolkit import readonly
from tests.native_fixture import ScriptedNativePlanner
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_author_interaction import (
    candidate,
    contract,
)


async def cancel():
    pass


@pytest.mark.parametrize("error_type", [EnvironmentInvalid, EnvironmentServiceFailure])
def test_invalid_scene_cannot_be_caught_to_dispatch_more_work(error_type):
    invalid = error_type("native support-arm scene is unstable")
    dispatched = []

    async def invoke(method, name, arguments):
        dispatched.append((method, name))
        raise invalid

    class CatchingSandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            with pytest.raises(error_type) as first:
                await rpc("act", "move", {})
            assert first.value is invalid
            with pytest.raises(error_type) as second:
                await rpc("observe", "view_env_state", {})
            assert second.value is invalid
            return {"handled": True}

    kernel = EpisodeKernel(
        baseline(),
        operations={},
        gateway=invoke,
        cancel=cancel,
        emit=lambda event: None,
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
        sandbox=CatchingSandbox(),
    )
    with pytest.raises(error_type) as failed:
        asyncio.run(kernel.run_plugin(baseline().loop.program, {}, invoke))
    assert failed.value is invalid
    assert dispatched == [("act", "move")]


def test_failed_rpc_audit_blocks_the_next_operation_before_dispatch():
    original = PermissionError("unavailable operation")
    audit_error = OSError("audit disk unavailable")
    dispatched = []

    async def invoke(method, name, arguments):
        if name == "undeclared":
            raise original
        dispatched.append((method, name, arguments))
        return {}

    class CatchingSandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            with pytest.raises(PermissionError) as rejected:
                await rpc("observe", "undeclared", {})
            assert rejected.value is original
            with pytest.raises(OSError) as blocked:
                await rpc("act", "move", {"distance": 0.1})
            assert blocked.value is audit_error
            return {"handled": True}

    def emit(event):
        raise audit_error

    kernel = EpisodeKernel(
        baseline(),
        operations={},
        gateway=invoke,
        cancel=cancel,
        emit=emit,
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
        sandbox=CatchingSandbox(),
    )
    with pytest.raises(OSError) as failed:
        asyncio.run(kernel.run_plugin(baseline().loop.program, {}, invoke))
    assert failed.value is audit_error
    assert dispatched == []


def test_workflow_failure_audit_preserves_unknown_physical_outcome():
    unknown = PhysicalOutcomeUnknown("submitted but unacknowledged")

    async def invoke(*args):
        raise unknown

    class DirectSandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            return await rpc("act", "move", {"distance": 0.1})

    attempted = []

    def emit(event):
        attempted.append(event["kind"])
        if event["kind"].endswith("failed"):
            raise OSError("failure evidence unavailable")

    kernel = EpisodeKernel(
        candidate("def run(runtime, context):\n    return {}\n"),
        operations=contract()["operations"],
        gateway=invoke,
        cancel=cancel,
        emit=emit,
        episode_limits=EpisodeLimits.model_validate(contract()["episode_limits"]),
        sandbox=DirectSandbox(),
    )
    with pytest.raises(PhysicalOutcomeUnknown) as failed:
        asyncio.run(
            kernel.workflow("probe", {}, observation={}, task_language="Open", epoch=0)
        )
    assert failed.value is unknown
    assert {"action_failed", "program_rpc_failed", "workflow_failed"} <= set(attempted)
    assert any("failure evidence unavailable" in note for note in unknown.__notes__)
    with pytest.raises(PhysicalOutcomeUnknown) as blocked:
        kernel.raise_if_failed()
    assert blocked.value is unknown


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_failure_reporting_preserves_ordinary_exception_identity(error_type):
    original = error_type("original failure")
    audit_error = OSError("audit failure")

    def emit(event):
        raise audit_error

    kernel = EpisodeKernel(
        baseline(),
        operations={},
        gateway=cancel,
        cancel=cancel,
        emit=emit,
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    with pytest.raises(error_type) as failed:
        kernel.raise_failure("program_failed", original)
    assert failed.value is original
    assert original.__cause__ is audit_error


@pytest.mark.parametrize("cleanup_error", [None, "cancelled", "infrastructure"])
def test_real_workflow_unknown_remains_scored_unknown_after_audit_and_cleanup_failure(
    tmp_path, monkeypatch, cleanup_error
):
    unknown = PhysicalOutcomeUnknown("submitted action lost acknowledgement")
    robot = Robot(tmp_path)
    calls = []
    robot.add_tool("view_env_state", contract()["tools"][0], readonly(lambda step: {}))

    def move(distance):
        calls.append(distance)
        raise unknown

    robot.add_tool("move", contract()["tools"][1], move)
    source = """def run(runtime, context):
    return runtime.act("move", {"distance": 0.1})
"""
    original_emit = BundleRuntime._emit
    failed_events = []

    def emit(self, event):
        if event["kind"].endswith("failed"):
            failed_events.append(event["kind"])
            raise OSError("audit sink unavailable")
        original_emit(self, event)

    monkeypatch.setattr(BundleRuntime, "_emit", emit)
    if cleanup_error is not None:

        async def drain(self, session):
            if cleanup_error == "cancelled":
                raise asyncio.CancelledError("cleanup cancelled")
            raise OSError("cleanup failed")

        monkeypatch.setattr(ProgramExecution, "_drain", drain)

    planner = ScriptedNativePlanner(
        FunctionModel(
            lambda messages, info: ModelResponse(
                parts=[
                    ToolCallPart(
                        "workflow_probe", {"arguments": {}}, tool_call_id="workflow"
                    )
                ]
            )
        ),
        dashboard_events=NullDashboardEventSink(),
    )
    planner.bind_bundle(
        candidate(source),
        episode_limits=EpisodeLimits.model_validate(contract()["episode_limits"]),
    )
    result = planner.solve(
        system_prompt="Control.",
        user_message="Open drawer",
        toolkit=robot,
        max_turns=3,
    )
    assert result.termination_class == ACTION_OUTCOME_UNKNOWN
    assert result.error == str(unknown)
    assert calls == [0.1]
    assert {"program_rpc_failed", "workflow_failed", "program_failed"} <= set(
        failed_events
    )


@pytest.mark.parametrize("audit_error_type", [OSError, RuntimeError])
def test_real_rpc_audit_failure_remains_infrastructure_without_later_motion(
    tmp_path, monkeypatch, audit_error_type
):
    robot = Robot(tmp_path)
    calls = []
    robot.add_tool("view_env_state", contract()["tools"][0], readonly(lambda step: {}))
    robot.add_tool(
        "move", contract()["tools"][1], lambda distance: calls.append(distance) or {}
    )
    source = """def run(runtime, context):
    try:
        runtime.observe("undeclared", {})
    except Exception:
        pass
    runtime.act("move", {"distance": 0.1})
    return {"summary": "done"}
"""
    original_emit = BundleRuntime._emit

    def emit(self, event):
        if event["kind"] == "program_rpc_failed":
            raise audit_error_type("audit unavailable")
        original_emit(self, event)

    monkeypatch.setattr(BundleRuntime, "_emit", emit)
    planner = ScriptedNativePlanner(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("unused")])),
        dashboard_events=NullDashboardEventSink(),
    )
    planner.bind_bundle(
        candidate("def run(runtime, context):\n    return {}\n", loop=source),
        episode_limits=EpisodeLimits.model_validate(contract()["episode_limits"]),
    )
    result = planner.solve(
        system_prompt="Control.",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.termination_class == API_ERROR
    assert "audit unavailable" in result.error
    assert calls == []


def test_recorded_rpc_rejection_cannot_be_caught_to_continue_execution(tmp_path):
    robot = Robot(tmp_path)
    calls = []
    robot.add_tool("view_env_state", contract()["tools"][0], readonly(lambda step: {}))
    robot.add_tool(
        "move", contract()["tools"][1], lambda distance: calls.append(distance) or {}
    )
    source = """def run(runtime, context):
    try:
        runtime.observe("undeclared", {})
    except Exception:
        pass
    runtime.act("move", {"distance": 0.1})
    return {"summary": "done"}
"""
    runtime = BundleRuntime(
        candidate("def run(runtime, context):\n    return {}\n", loop=source),
        robot,
        "Inspect",
        episode_limits=EpisodeLimits.model_validate(contract()["episode_limits"]),
    )

    async def model(*args):
        pytest.fail("this loop requires no model request")

    with pytest.raises(CandidateExecutionFailure, match=r"runtime\.plan"):
        asyncio.run(runtime.run(model, "Control."))
    assert calls == []


@pytest.mark.parametrize(
    "cleanup_error_type", [OSError, RuntimeError, asyncio.CancelledError]
)
def test_outer_runtime_restores_recorded_unknown_after_later_exception(
    tmp_path, monkeypatch, cleanup_error_type
):
    unknown = PhysicalOutcomeUnknown("recorded unknown physical outcome")
    cleanup = cleanup_error_type("later cleanup failed")

    async def run(self):
        self.kernel._fatal = unknown
        raise cleanup

    monkeypatch.setattr(ProgramExecution, "run", run)
    runtime = BundleRuntime(
        baseline(),
        Robot(tmp_path),
        "Inspect",
        episode_limits=EpisodeLimits.model_validate(contract()["episode_limits"]),
    )
    with pytest.raises(PhysicalOutcomeUnknown) as failed:
        asyncio.run(runtime.run(None, "Control."))
    assert failed.value is unknown
    assert unknown.__cause__ is cleanup
    assert any("later cleanup failed" in note for note in unknown.__notes__)
