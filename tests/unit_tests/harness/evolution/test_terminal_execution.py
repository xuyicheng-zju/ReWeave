# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Confirmed environment endings stop new work without changing the last ACK."""

import asyncio

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle, ProgramLimits
from reweave.evolution.validation.interaction import InteractionEnvironmentState
from reweave.planner.base import AGENT_EXECUTION_FAILURE
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.execution import ExecutionContext, ProgramExecution
from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits
from reweave.session import EnvState
from reweave.tools.toolkit import Toolkit, readonly
from reweave.utils.logging import init_output_dir
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import baseline, workflow
from tests.unit_tests.harness.evolution.test_plugin_execution import program


class TerminalRobot(Toolkit):
    def __init__(self, path):
        init_output_dir(path)
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(path),
            filesystem_tools_enabled=False,
        )
        self.moves = 0
        for name, handler in (("move", self.move), ("inspect", self.inspect)):
            self.add_tool(
                name,
                {
                    "name": name,
                    "description": "Synthetic public response",
                    "input_schema": {"type": "object", "properties": {}},
                },
                handler,
            )

    def move(self):
        self.moves += 1
        return {"acknowledged": self.moves}

    @readonly
    def inspect(self):
        return {"moves": self.moves}

    def get_env_state(self, *, command, result, elapsed_s):
        with self.state.record_step(
            state={"moves": self.moves},
            terminated=True,
            truncated=False,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return {"terminated": True, "truncated": False, "moves": self.moves}

    def solved(self):
        return False


@pytest.mark.parametrize("flags", [None, (False, False), (True, False), (False, True)])
def test_toolkit_uses_both_record_flags_and_keeps_read_only_evidence(tmp_path, flags):
    robot = TerminalRobot(tmp_path)
    if flags is not None:
        with robot.state.record_step(state={}, terminated=flags[0], truncated=flags[1]):
            pass
    ended = flags is not None and any(flags)
    assert robot.episode_ended() is ended
    before = robot.state.latest_record()
    result = robot.execute_tool("move", {}).result
    assert robot.moves == (0 if ended else 1)
    if ended:
        assert result == {
            "status": "not_executed",
            "execution": "not_executed",
            "reason": "environment episode ended",
        }
        assert robot.state.latest_record() is before
    assert robot.execute_tool("inspect", {}).result == {"moves": robot.moves}
    unknown = PhysicalOutcomeUnknown("unacknowledged action")
    robot._physical_outcome_error = unknown
    with pytest.raises(PhysicalOutcomeUnknown) as failure:
        robot.execute_tool("move", {})
    assert failure.value is unknown


def test_queued_kernel_action_rechecks_terminal_before_charge_or_submission():
    async def run():
        entered = asyncio.Event()
        settle = asyncio.Event()
        ended = False
        calls, events = [], []

        async def gateway(name, arguments):
            nonlocal ended
            calls.append(name)
            entered.set()
            await settle.wait()
            ended = True
            return {"acknowledged": True, "terminated": True, "truncated": False}

        async def cancel():
            pass

        def unavailable_program_snapshot():
            raise AssertionError("physical dispatch cannot require a program snapshot")

        kernel = EpisodeKernel(
            baseline(),
            operations={
                "move": {"effect": "physical", "input_schema": {"type": "object"}}
            },
            gateway=gateway,
            cancel=cancel,
            emit=events.append,
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=10
            ),
            snapshot=unavailable_program_snapshot,
            episode_ended=lambda: ended,
        )
        counts = {"runtime": 0, "physical": 0, "model": 0}
        token = kernel.budget_frames.set(
            (
                (
                    ProgramLimits(
                        runtime_calls=10,
                        physical_calls=1,
                        model_calls=0,
                        wall_time_s=10,
                    ),
                    counts,
                ),
            )
        )
        try:
            first = asyncio.create_task(kernel.invoke("move", {}, epoch=0))
            await entered.wait()
            second = asyncio.create_task(kernel.invoke("move", {}, epoch=0))
            await asyncio.sleep(0)
            assert not second.done()
            settle.set()
            ack, skipped = await asyncio.gather(first, second)
        finally:
            kernel.budget_frames.reset(token)
        assert ack == {"acknowledged": True, "terminated": True, "truncated": False}
        assert skipped["status"] == "not_executed" and "error" not in skipped
        assert calls == ["move"] and counts == {"runtime": 2, "physical": 1, "model": 0}
        assert kernel._calls == 1
        assert sum(event["kind"] == "action_submitted" for event in events) == 1
        assert sum(event["kind"] == "action_completed" for event in events) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "terminal_source", ["active", "terminated", "truncated", "official_success"]
)
def test_direct_kernel_dispatch_uses_environment_without_program_context(
    tmp_path, monkeypatch, terminal_source
):
    robot = TerminalRobot(tmp_path)
    if terminal_source in {"terminated", "truncated"}:
        with robot.state.record_step(
            state={},
            terminated=terminal_source == "terminated",
            truncated=terminal_source == "truncated",
        ):
            pass
    monkeypatch.setattr(robot, "solved", lambda: terminal_source == "official_success")
    bridge = BundleRuntime(
        baseline(),
        robot,
        "Synthetic",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    # Direct gateways own no ProgramExecution yet and require no fabricated context.
    assert not hasattr(bridge, "execution")
    result = asyncio.run(bridge.kernel.invoke("move", {}, epoch=0))
    if terminal_source == "active":
        assert result["terminated"]
        assert result["capability_result"] == {"acknowledged": 1}
        assert robot.moves == bridge.kernel._calls == 1
    else:
        assert result["status"] == result["execution"] == "not_executed"
        assert "error" not in result and robot.moves == bridge.kernel._calls == 0


def test_unknown_precedes_context_free_environment_terminal_check(
    tmp_path, monkeypatch
):
    robot = TerminalRobot(tmp_path)

    def forbidden_terminal_query():
        raise AssertionError("known unknown must stop before an environment query")

    monkeypatch.setattr(robot, "solved", forbidden_terminal_query)
    bridge = BundleRuntime(
        baseline(),
        robot,
        "Synthetic",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    unknown = PhysicalOutcomeUnknown("previous action has no acknowledgement")
    bridge.kernel._fatal = unknown
    with pytest.raises(PhysicalOutcomeUnknown) as failed:
        asyncio.run(bridge.kernel.invoke("move", {}, epoch=0))
    assert failed.value is unknown
    assert robot.moves == bridge.kernel._calls == 0


def test_accepted_finish_remains_terminal_before_environment_ends(tmp_path):
    async def run():
        robot = TerminalRobot(tmp_path)
        bridge = BundleRuntime(
            baseline(),
            robot,
            "Synthetic",
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=10
            ),
        )

        async def forbidden_model(*args):
            raise AssertionError("accepted finish must not dispatch another model")

        execution = ProgramExecution(bridge, forbidden_model, "Synthetic")
        bridge.execution = execution
        session = ExecutionContext(
            "root",
            "main",
            "Synthetic",
            frozenset({"model:plan"}),
            finish_result={"_finish": True, "status": "failure"},
        )
        token = execution.current.set(session)
        try:
            assert not robot.solved() and not robot.episode_ended()
            assert execution.snapshot()["terminal"]
            response = await execution.plan()
            assert response["terminal"] and response["tool_calls"] == []
        finally:
            execution.current.reset(token)
        assert robot.moves == bridge.kernel._calls == 0

    asyncio.run(run())


@pytest.mark.parametrize("path", ["root", "workflow", "invalid_root"])
def test_frozen_programs_do_not_dispatch_after_terminal_ack(tmp_path, path):
    payload = baseline().model_dump(mode="json")
    if path == "workflow":
        spec = workflow().model_dump(mode="json")
        spec["permissions"] = ["inspect", "model:plan", "move"]
        spec["limits"] = ProgramLimits(
            runtime_calls=10, physical_calls=1, model_calls=1, wall_time_s=10
        ).model_dump(mode="json")
        spec["program"] = program(
            """def run(runtime, context):
    first = runtime.act("move", {})
    second = runtime.act("move", {})
    decision = runtime.plan("No decision after terminal")
    assert decision["terminal"] and decision["tool_calls"] == []
    evidence = runtime.observe("inspect", {})
    return {"status": "completed", "summary": "environment ended", "output": {"first": first, "second": second, "evidence": evidence}}
""",
            "probe:run",
        ).model_dump(mode="json")
        payload["workflows"] = [spec]
    else:
        payload["loop"]["program"] = program(
            """def run(runtime, context):
    calls = runtime.plan()["tool_calls"]
    runtime.execute(calls[0])
    skipped = runtime.execute(calls[1])
    assert skipped["feedback"]["status"] == "not_executed"
    decision = runtime.plan("No decision after terminal")
    assert decision["terminal"] and decision["tool_calls"] == []
    return decision
"""
            if path == "root"
            else """def run(runtime, context):
    runtime.execute(runtime.plan()["tool_calls"][0])
    return runtime.plan("No decision after terminal")["tool_calls"][0]
"""
        ).model_dump(mode="json")
    requests = []

    def model(messages, info):
        requests.append(messages)
        assert len(requests) == 1, (
            "terminal feedback must not dispatch a second model request"
        )
        first = (
            ToolCallPart("workflow_probe", {"arguments": {}}, "first")
            if path == "workflow"
            else ToolCallPart("move", {}, "first")
        )
        return ModelResponse(parts=[first, ToolCallPart("move", {}, "tail")])

    robot = TerminalRobot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=15
        ),
    )
    result = planner.solve(
        system_prompt="Synthetic", user_message="Synthetic", toolkit=robot, max_turns=5
    )
    assert robot.moves == len(requests) == result.stats["visible_turns"] == 1
    assert robot.episode_ended() and not robot.solved()
    assert result.termination_class == (
        AGENT_EXECUTION_FAILURE if path == "invalid_root" else None
    )
    if path == "invalid_root":
        assert "IndexError" in result.error
    else:
        assert result.error is None


def test_interaction_environment_requires_terminated_independent_of_success():
    with pytest.raises(ValidationError, match="terminated"):
        InteractionEnvironmentState(official_success=False, truncated=False)
    state = InteractionEnvironmentState(
        official_success=False, terminated=True, truncated=False
    )
    assert state.terminated and not state.official_success and not state.truncated


def test_context_preparation_cannot_dispatch_a_model_after_another_action_ends_episode(
    tmp_path, monkeypatch
):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def prepare(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"state": {}}

        async def model(*args):
            calls.append(args)
            raise AssertionError("ended episode must not dispatch a model")

        monkeypatch.setattr("reweave.runtime.execution.build_context", prepare)
        robot = TerminalRobot(tmp_path)
        bridge = BundleRuntime(
            baseline(),
            robot,
            "Synthetic",
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=10
            ),
        )
        execution = ProgramExecution(bridge, model, "Synthetic")
        bridge.execution = execution
        session = ExecutionContext(
            "root", "main", "Synthetic", frozenset({"model:plan"})
        )
        token = execution.current.set(session)
        try:
            pending = asyncio.create_task(execution.plan())
            await entered.wait()
            ack = robot.execute_tool("move", {}).result
            release.set()
            response = await pending
        finally:
            execution.current.reset(token)
        assert ack["terminated"] and ack["capability_result"] == {"acknowledged": 1}
        assert response["terminal"] and response["tool_calls"] == []
        assert calls == []

    asyncio.run(run())
