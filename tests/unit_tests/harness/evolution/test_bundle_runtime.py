# Modified for the ReWeave namespace.
"""W/C/L are consumed by the real API loop, with protected program execution."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import (
    AgentBundle,
    ContextPackage,
    ProgramLimits,
    PythonFile,
    PythonProgram,
    TextSkill,
)
from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline, workflow

LIMITS = ProgramLimits(
    runtime_calls=20, physical_calls=0, model_calls=0, wall_time_s=10
)


def terminal_response():
    return ModelResponse(
        parts=[
            ToolCallPart(
                "read_terminal_state",
                {},
                tool_call_id="terminal-state",
            )
        ]
    )


def test_w_only_discovered_and_skill_consumed_in_actual_model_request(tmp_path):
    bundle = baseline()
    program = workflow().model_dump(mode="json")
    program["permissions"] = ["inspect_scene"]
    program["observations"] = ["inspect_scene"]
    program["program"]["files"][0]["source"] = """def run(runtime, context):
    observed = runtime.observe("inspect_scene", {})
    return {"status": "completed", "summary": "checked pose", "output": observed}
"""
    payload = bundle.model_dump(mode="json")
    payload["workflows"] = [program]
    payload["context"] = ContextPackage(
        skills=(
            TextSkill(
                scope="global",
                skill_id="feedback",
                text="Check actual progress before repeating a motion.",
            ),
        )
    ).model_dump(mode="json")
    bundle = AgentBundle.model_validate(payload)
    seen = []

    def model(messages, info):
        seen.append(info)
        if len(seen) == 1:
            names = {tool.name for tool in info.function_tools}
            assert "workflow_probe" in names
            assert not names.intersection(
                {
                    "modify_harness",
                    "run_harness_program",
                    "write_text_file",
                }
            )
            assert "Check actual progress" in info.instructions
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "workflow_probe", {"arguments": {}}, tool_call_id="invoke-w"
                    )
                ]
            )
        return terminal_response()

    robot = Robot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        bundle,
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="Control the robot.",
        user_message="Inspect the drawer",
        toolkit=robot,
        max_turns=4,
    )
    assert result.error is None, result.error
    assert robot.calls == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    completed = next(event for event in events if event["kind"] == "workflow_completed")
    response = next(
        event["response"] for event in events if event["kind"] == "model_response"
    )
    identity = next(iter(response["metadata"]["tool_call_ids"].values()))
    assert completed["planner_call_id"] == identity
    action = next(
        event
        for event in events
        if event["kind"] == "action_completed" and event["operation"] == "inspect_scene"
    )
    assert action["planner_call_id"] == identity
    assert action["workflow_invocation_id"] == completed["workflow_invocation_id"]
    submitted = next(
        event
        for event in events
        if event["kind"] == "action_submitted" and event["operation"] == "inspect_scene"
    )
    assert submitted["action_id"] == action["action_id"]
    assert all(event["bundle_id"] == bundle.record_id for event in events)


def test_l_changes_actual_schedule_without_editing_c_or_w(tmp_path):
    payload = baseline().model_dump(mode="json")
    payload["loop"]["program"]["files"][0]["source"] = """def run(runtime, context):
    observed = runtime.observe("inspect_scene", {})
    call = runtime.plan("Use this fresh observation: " + str(observed))["tool_calls"][0]
    runtime.execute(call)
    return {"summary": "finished"}
"""
    bundle = AgentBundle.model_validate(payload)
    robot = Robot(tmp_path)

    def model(messages, info):
        assert robot.calls == 1
        assert "position" in str(messages)
        return terminal_response()

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        bundle,
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=2
    )
    assert result.error is None, result.error
    assert bundle.context == baseline().context and not bundle.workflows


def test_stale_action_rejected_after_cancel_ack():
    async def run():
        events, calls = [], []

        async def gateway(name, args):
            calls.append(name)
            return {}

        async def cancel():
            events.append({"kind": "adapter_ack"})

        kernel = EpisodeKernel(
            baseline(),
            operations={
                "move": {"effect": "physical", "input_schema": {"type": "object"}}
            },
            gateway=gateway,
            cancel=cancel,
            emit=events.append,
            episode_limits=EpisodeLimits(
                program_calls=20, gateway_calls=20, wall_time_s=30
            ),
        )
        epoch = kernel.epoch
        await kernel.cancel()
        with pytest.raises(RuntimeError, match="stale"):
            await kernel.invoke("move", {}, epoch=epoch)
        assert not calls
        assert [event["kind"] for event in events] == [
            "cancel_requested",
            "adapter_ack",
            "cancel_acknowledged",
        ]

    asyncio.run(run())


def test_generated_code_cannot_read_host_file_even_via_numpy(tmp_path):
    from reweave.runtime.sandbox import ProgramSandbox

    secret = tmp_path / "private_fixture.txt"
    secret.write_text("PRIVATE-TEST-SENTINEL")
    program = PythonProgram(
        entrypoint="probe:run",
        files=(
            PythonFile(
                path="probe.py",
                source=f"""
import numpy as np

def run(runtime, context):
    try:
        np.fromfile({str(secret)!r}, dtype=np.uint8)
    except PermissionError:
        return {{"denied": True, "sum": int(np.array([1, 2]).sum())}}
    raise RuntimeError("private file read escaped isolation")
""",
            ),
        ),
    )

    async def reject(*args):
        raise AssertionError("no runtime call allowed")

    result = asyncio.run(
        ProgramSandbox().run_program(
            program, LIMITS, {}, reject, input_schema={"type": "object"}
        )
    )
    assert result == {"denied": True, "sum": 3}


def test_loop_budget_failure_is_scored_not_infrastructure(tmp_path):
    from reweave.planner.base import AGENT_EXECUTION_FAILURE, episode_is_scored

    payload = baseline().model_dump(mode="json")
    payload["loop"]["program"]["files"][0]["source"] = """def run(runtime, context):
    while True:
        runtime.observe("inspect_scene", {})
"""
    robot = Robot(tmp_path)

    def model(*args):
        raise AssertionError("L never reaches the controller")

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=EpisodeLimits(program_calls=2, gateway_calls=2, wall_time_s=20),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.termination_class == AGENT_EXECUTION_FAILURE
    assert episode_is_scored(result.termination_class)
    assert robot.calls == 2


def test_fresh_workflow_does_not_inherit_another_objects_state():
    async def run():
        item = workflow().model_dump(mode="json")
        item["input_schema"]["required"] = ["value"]
        item["program"]["files"][0]["source"] = """def run(runtime, context):
    value = context["state"].get("count", 0) + 1
    return {"status": "yielded", "summary": "next step", "output": {"count": value}, "state": {"count": value}}
"""
        payload = baseline().model_dump(mode="json")
        payload["workflows"] = [item]

        async def gateway(*args):
            raise AssertionError("no effects")

        async def cancel():
            pass

        kernel = EpisodeKernel(
            AgentBundle.model_validate(payload),
            operations={},
            gateway=gateway,
            cancel=cancel,
            emit=lambda event: None,
            episode_limits=EpisodeLimits(
                program_calls=20, gateway_calls=20, wall_time_s=30
            ),
        )

        async def call(arguments, token=None):
            return await kernel.workflow(
                "probe",
                arguments,
                continuation_id=token,
                observation={},
                task_language="inspect",
                epoch=kernel.epoch,
            )

        a = await call({"value": 1})
        b = await call({"value": 2})
        assert a.output["count"] == b.output["count"] == 1
        assert a.resume_arguments == {"value": 1}
        rejected = await call({"value": 2}, a.continuation_id)
        assert rejected.status == "failed"
        assert rejected.output == {
            "error": "invalid_continuation",
            "execution_performed": False,
        }
        assert "unchanged" in rejected.summary
        assert rejected.continuation_id is None
        for malformed in ({}, {"value": "invalid-type"}):
            rejected = await call(malformed, a.continuation_id)
            assert rejected.status == "failed"
            assert rejected.output == {
                "error": "invalid_continuation",
                "execution_performed": False,
            }
        unknown = await call({"value": 1}, "missing-token")
        assert unknown.status == "failed"
        resumed = await call({"value": 1}, a.continuation_id)
        assert resumed.output["count"] == 2

    asyncio.run(run())


def test_image_labels_and_actual_tool_definitions_survive(tmp_path):
    from reweave.planner.bundle_runtime import BundleRuntime
    from reweave.tools.toolkit import _runtime_json_object

    runtime = BundleRuntime(
        baseline(),
        Robot(tmp_path),
        "inspect",
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    import io

    from PIL import Image

    images = []
    for color in ("black", "white"):
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(stream, format="PNG")
        images.append(stream.getvalue())
    converted = runtime._tool_result(
        "inspect_scene",
        _runtime_json_object(
            {
                "_image_bytes": images[0],
                "_image_wrist_bytes": images[1],
                "_image_labels": {
                    "_image_bytes": "agent-view camera, screen coordinates",
                    "_image_wrist_bytes": "wrist camera",
                },
            }
        ),
    )
    text = json.loads(converted.content_blocks[0]["text"])
    assert text["image_inputs"] == [
        {
            "image_number": 1,
            "source_pointer": "/_runtime_images/0",
            "description": "agent-view camera, screen coordinates",
        },
        {
            "image_number": 2,
            "source_pointer": "/_runtime_images/1",
            "description": "wrist camera",
        },
    ]
    assert [block["type"] for block in converted.content_blocks] == [
        "text",
        "image",
        "image",
    ]


def test_episode_timeout_includes_controller_wait(tmp_path):
    async def model(messages, info):
        await asyncio.sleep(4)
        return ModelResponse(parts=[TextPart("Done")])

    robot = Robot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=2.0
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=2
    )
    assert result.termination_class == "planner_timeout"


def test_failed_model_attempts_consume_request_budget(tmp_path, monkeypatch):
    from reweave.planner.providers.base import NativeTurnFailed

    calls = []

    def model(messages, info):
        calls.append(1)
        raise NativeTurnFailed("failed", {"error": "unavailable"})

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    result = planner.solve(
        system_prompt="Control",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=1,
    )
    assert len(calls) == 1 and result.stats["visible_turns"] == 1
    assert result.termination_class == "api_error"
    event = json.loads((tmp_path / "native/native_turns.jsonl").read_text())
    assert event["status"] == "failed"


def test_l_arbitrates_image_reader_too(tmp_path):
    payload = baseline().model_dump(mode="json")
    payload["loop"]["program"]["files"][0]["source"] = """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    runtime.feedback(call["id"], {"reason": "deny pending request"})
    finish = runtime.plan("Choose another action or finish")["tool_calls"][0]
    runtime.execute(finish)
    return {"summary": "finished"}
"""
    calls = []

    def model(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_image", {"name": "camera.png"}, tool_call_id="image"
                    )
                ]
            )
        return terminal_response()

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    result = planner.solve(
        system_prompt="Control",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert len(calls) == 2 and result.error is None
    events = (tmp_path / "harness_context/bundle_events.jsonl").read_text()
    assert "deny pending request" in events
    assert '"operation": "read_image"' not in events


def test_concurrent_cancel_does_not_reopen_gateway():
    async def run():
        releases = [asyncio.Event(), asyncio.Event()]
        started = asyncio.Event()
        count = 0

        async def cancel():
            nonlocal count
            index = count
            count += 1
            if count == 2:
                started.set()
            await releases[index].wait()

        async def gateway(name, args):
            return {}

        kernel = EpisodeKernel(
            baseline(),
            operations={
                "move": {"effect": "physical", "input_schema": {"type": "object"}}
            },
            gateway=gateway,
            cancel=cancel,
            emit=lambda _: None,
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=10
            ),
        )
        first = asyncio.create_task(kernel.cancel())
        second = asyncio.create_task(kernel.cancel())
        await started.wait()
        releases[0].set()
        await first
        with pytest.raises(RuntimeError, match="stale action"):
            await kernel.invoke("move", {}, epoch=kernel.epoch)
        releases[1].set()
        await second
        assert await kernel.invoke("move", {}, epoch=kernel.epoch) == {}

    asyncio.run(run())


def test_continuation_cannot_reset_local_gateway_budget():
    async def run():
        payload = workflow().model_dump(mode="json")
        payload["permissions"] = ["probe"]
        payload["observations"] = ["probe"]
        payload["limits"]["runtime_calls"] = 1
        payload["limits"]["physical_calls"] = 0
        payload["program"]["files"][0]["source"] = """def run(runtime, context):
    runtime.observe('probe', {})
    return {'status': 'yielded', 'summary': 'progress', 'output': {}, 'state': {}}
"""
        root = baseline().model_dump(mode="json")
        root["workflows"] = [payload]

        async def gateway(name, args):
            return {}

        async def cancel():
            pass

        kernel = EpisodeKernel(
            AgentBundle.model_validate(root),
            operations={
                "probe": {"effect": "read", "input_schema": {"type": "object"}}
            },
            gateway=gateway,
            cancel=cancel,
            emit=lambda _: None,
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=10
            ),
        )
        first = await kernel.workflow(
            "probe", {}, observation={}, task_language="inspect", epoch=0
        )
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await kernel.workflow(
                "probe",
                {},
                observation={},
                task_language="inspect",
                epoch=0,
                continuation_id=first.continuation_id,
            )

    asyncio.run(run())


def test_invalid_workflow_image_is_scored_before_provider_request(tmp_path):
    from reweave.planner.base import AGENT_EXECUTION_FAILURE

    payload = baseline().model_dump(mode="json")
    program = workflow().model_dump(mode="json")
    program["program"]["files"][0]["source"] = """def run(runtime, context):
    return {"status": "completed", "summary": "invalid fixture", "output": {"_runtime_images": [{"name": "fake", "media_type": "image/png", "data_base64": "bm90IGFuIGltYWdl"}]}}
"""
    payload["workflows"] = [program]
    calls = []

    def model(messages, info):
        calls.append(1)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "workflow_probe", {"arguments": {}}, tool_call_id="invalid-image"
                )
            ]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=EpisodeLimits(program_calls=5, gateway_calls=5, wall_time_s=30),
    )
    result = planner.solve(
        system_prompt="Fixture",
        user_message="Fixture",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.termination_class == AGENT_EXECUTION_FAILURE
    assert "invalid image" in result.error
    assert calls == [1]


def test_finish_is_planner_control_not_workflow_or_observation(tmp_path):
    from reweave.planner.bundle_runtime import BundleRuntime

    robot = Robot(tmp_path)
    called = []
    robot.add_tool(
        "finish",
        {
            "name": "finish",
            "description": "End task",
            "input_schema": {"type": "object", "properties": {}},
        },
        lambda: called.append(True) or {"_finish": True},
    )
    runtime = BundleRuntime(
        baseline(),
        robot,
        "inspect",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=20
        ),
    )
    assert runtime.kernel._operations["finish"]["effect"] == "control"
    assert not called
    payload = baseline().model_dump(mode="json")
    item = workflow().model_dump(mode="json")
    item["permissions"] = ["finish"]
    payload["workflows"] = [item]
    with pytest.raises(ValueError, match="non-public operation"):
        BundleRuntime(
            AgentBundle.model_validate(payload),
            robot,
            "inspect",
            episode_limits=EpisodeLimits(
                program_calls=10, gateway_calls=10, wall_time_s=20
            ),
        )
    result = asyncio.run(runtime.execute("finish", {}))
    assert result.result["_finish"] and called == [True]


def test_unknown_physical_outcome_is_scored_without_becoming_api_error(tmp_path):
    from reweave.planner.base import (
        ACTION_OUTCOME_UNKNOWN,
        episode_is_scored,
        task_evidence_is_complete,
    )
    from reweave.runtime.capability import PhysicalOutcomeUnknown

    robot = Robot(tmp_path)

    def unknown():
        raise PhysicalOutcomeUnknown("submitted action lost acknowledgement")

    robot.add_tool(
        "inspect_scene",
        {
            "name": "inspect_scene",
            "description": "Inspect",
            "input_schema": {"type": "object", "properties": {}},
        },
        unknown,
    )
    planner = ScriptedNativePlanner(
        FunctionModel(
            lambda messages, info: ModelResponse(
                parts=[ToolCallPart("inspect_scene", {}, tool_call_id="lost-ack")]
            )
        ),
        dashboard_events=NullDashboardEventSink(),
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=20
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.termination_class == ACTION_OUTCOME_UNKNOWN
    assert episode_is_scored(result.termination_class)
    assert not task_evidence_is_complete(result.termination_class)


def test_memory_access_denial_returns_feedback_without_reading_outside_root(tmp_path):
    from reweave.planner.bundle_runtime import BundleRuntime

    robot = Robot(tmp_path)
    root = tmp_path / "memory"
    root.mkdir()
    reads = []
    robot._memory = SimpleNamespace(
        root=root, get_common_tool_bindings=lambda **kwargs: {}
    )
    runtime = BundleRuntime(
        baseline(),
        robot,
        "inspect",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    runtime.kernel._operations["list_dir"] = {"effect": "read"}
    runtime._memory_bindings["list_dir"] = (
        {},
        lambda **args: reads.append(args) or {"entries": []},
    )
    denied = asyncio.run(runtime._dispatch_tool("list_dir", {"path": str(tmp_path)}))
    assert (
        denied["error_type"] == "PermissionError"
        and denied["execution_performed"] is False
    )
    assert reads == []
    allowed = asyncio.run(runtime._dispatch_tool("list_dir", {"path": str(root)}))
    assert allowed == {"entries": []} and len(reads) == 1


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("view__state", ""),
        ("inspect_scene", "{broken"),
        ("inspect_scene", "[]"),
        ("read_image", '{"name":123}'),
        ("read_image", ""),
    ],
)
def test_invalid_model_proposal_is_explicit_feedback_then_corrected(
    tmp_path, name, arguments
):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    seen = []

    def model(messages, info):
        seen.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        if len(seen) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(name, arguments, "bad"),
                    ToolCallPart("inspect_scene", {}, "good"),
                ]
            )
        assert "program_feedback" in json.dumps(seen[-1])
        assert "Unknown or unavailable tool" in str(seen[-1]) or "arguments" in str(
            seen[-1]
        )
        return terminal_response()

    robot = Robot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=30, gateway_calls=30, wall_time_s=10
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=4
    )
    assert result.error is None
    # Invalid arguments stop the remaining batch before its valid sibling runs.
    assert robot.calls == 0 and result.stats["visible_turns"] == 2
    directory = tmp_path / "harness_context"
    events = [
        json.loads(x)
        for x in (directory / "bundle_events.jsonl").read_text().splitlines()
    ]
    rejected = [e for e in events if e["kind"] == "model_call_rejected"]
    assert len(rejected) == 1 and rejected[0]["execution_performed"] is False
    assert rejected[0]["call"]["arguments_json"] == arguments
    submitted = [e for e in events if e["kind"] == "action_submitted"]
    assert [e["operation"] for e in submitted] == [
        "model:plan",
        "model:plan",
        "read_terminal_state",
    ]


def test_repeated_invalid_proposals_exhaust_agent_budget_not_infrastructure(tmp_path):
    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("view__state", "", "bad")])

    robot = Robot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=30, gateway_calls=30, wall_time_s=10
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.termination_class == "agent_execution_failure"
    assert result.stats["visible_turns"] == 3 and robot.calls == 0


def test_native_empty_optional_arguments_are_rejected_without_repair(tmp_path):
    calls = []

    def model(messages, info):
        calls.append(1)
        return (
            ModelResponse(parts=[ToolCallPart("inspect_scene", "", "empty")])
            if len(calls) == 1
            else terminal_response()
        )

    robot = Robot(tmp_path)
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=30, gateway_calls=30, wall_time_s=10
        ),
    )
    result = planner.solve(
        system_prompt="Control", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.error is None and robot.calls == 0
    events = [
        json.loads(x)
        for x in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    rejected = next(e for e in events if e["kind"] == "model_call_rejected")
    assert rejected["call"]["arguments_json"] == ""
    assert rejected["execution_performed"] is False
    assert not any(e["kind"] == "model_arguments_normalized" for e in events)


def test_uncapped_kernel_counts_still_enforce_deadline_and_action_catalog():
    import time

    operations = {"inspect": {"effect": "read", "input_schema": {"type": "object"}}}
    received = []

    async def gateway(name, arguments):
        received.append(name)
        return {}

    async def cancel():
        pass

    class Sandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            assert limits.runtime_calls is None
            assert limits.physical_calls is None
            assert limits.model_calls is None
            assert limits.wall_time_s > 0
            return {"ok": True}

    kernel = EpisodeKernel(
        baseline(),
        operations=operations,
        gateway=gateway,
        cancel=cancel,
        emit=lambda event: None,
        episode_limits=EpisodeLimits(
            program_calls=None, gateway_calls=None, wall_time_s=30
        ),
        sandbox=Sandbox(),
    )

    async def run():
        for _ in range(501):
            await kernel.run_plugin(baseline().loop.program, {}, gateway)
            await kernel.invoke("inspect", {}, epoch=0)
        with pytest.raises(PermissionError, match="frozen catalog"):
            await kernel.invoke("undeclared", {}, epoch=0)
        kernel._deadline = time.monotonic() - 1
        with pytest.raises(RuntimeError, match="deadline exhausted"):
            await kernel.run_plugin(baseline().loop.program, {}, gateway)
        with pytest.raises(RuntimeError, match="gateway budget exhausted"):
            await kernel.invoke("inspect", {}, epoch=0)

    asyncio.run(run())
    assert len(received) == 501


def test_real_sandbox_supports_explicit_null_call_limit():
    from reweave.runtime.sandbox import ProgramSandbox

    program = PythonProgram(
        entrypoint="probe:run",
        files=(
            PythonFile(
                path="probe.py",
                source='def run(runtime, context):\n    for i in range(3):\n        runtime.observe("inspect", {})\n    return {"done": True}\n',
            ),
        ),
    )
    seen = []

    async def invoke(*arguments):
        seen.append(arguments)
        return {}

    result = asyncio.run(
        ProgramSandbox().run_program(
            program,
            ProgramLimits(
                runtime_calls=None, physical_calls=0, model_calls=0, wall_time_s=10
            ),
            {},
            invoke,
            input_schema={"type": "object"},
        )
    )
    assert result == {"done": True}
    assert len(seen) == 3


def test_uncapped_worker_excludes_host_wait_but_bounds_local_spin():
    from reweave.evolution.bundle import ProgramLimits, PythonFile, PythonProgram
    from reweave.runtime.capability import ProgramDeadlineExceeded
    from reweave.runtime.sandbox import ProgramSandbox

    limits = ProgramLimits(
        runtime_calls=2, physical_calls=0, model_calls=0, wall_time_s=1.5
    )
    program = PythonProgram(
        entrypoint="main:run",
        files=(
            PythonFile(
                path="main.py",
                source='def run(runtime, context):\n    value = runtime.observe("inspect", {})\n    return {"value": value}\n',
            ),
        ),
    )

    async def slow_host(*args):
        await asyncio.sleep(2)
        return {"settled": True}

    async def run():
        result = await ProgramSandbox().run_program(
            program,
            limits,
            {},
            slow_host,
            input_schema={"type": "object"},
            exclude_runtime_time=True,
        )
        assert result == {"value": {"settled": True}}
        spin = PythonProgram(
            entrypoint="main:run",
            files=(
                PythonFile(
                    path="main.py",
                    source="def run(runtime, context):\n    while True:\n        pass\n",
                ),
            ),
        )
        with pytest.raises(ProgramDeadlineExceeded, match="local execution allowance"):
            await ProgramSandbox().run_program(
                spin,
                limits,
                {},
                slow_host,
                input_schema={"type": "object"},
                exclude_runtime_time=True,
            )

    asyncio.run(run())
