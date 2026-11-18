# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Whole-program customization through real sandbox workers and model adapters."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import (
    AgentBundle,
    ContextPackage,
    PythonFile,
    PythonProgram,
    TextSkill,
)
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner, rows
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline


def program(source, entrypoint="loop:run"):
    module = entrypoint.split(":")[0]
    return PythonProgram(
        entrypoint=entrypoint, files=(PythonFile(path=module + ".py", source=source),)
    )


def run(
    tmp_path,
    source,
    model,
    *,
    context=None,
    maximum=10,
    robot=None,
    workflows=None,
    terminal=True,
    episode_limits=None,
):
    if terminal:
        # Keep the component's execution nonterminal, then explicitly end its
        # synthetic environment before the root returns.
        asynchronous = "async def run(" in source
        source += (
            "\n_component_run = run\n"
            + (
                "async def run(runtime, context):\n"
                "    result = await _component_run(runtime, context)\n"
                if asynchronous
                else "def run(runtime, context):\n"
                "    result = _component_run(runtime, context)\n"
            )
            + (
                "    await runtime.observe('read_terminal_state', {})\n"
                if asynchronous
                else "    runtime.observe('read_terminal_state', {})\n"
            )
            + "    return result\n"
        )
    payload = baseline().model_dump(mode="json")
    payload["loop"]["program"] = program(source).model_dump(mode="json")
    if workflows is not None:
        payload["workflows"] = [item.model_dump(mode="json") for item in workflows]
    if context is not None:
        payload["context"] = context.model_dump(mode="json")
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=episode_limits
        or EpisodeLimits(program_calls=100, gateway_calls=100, wall_time_s=15),
    )
    toolkit = robot or Robot(tmp_path)
    return planner.solve(
        system_prompt="BASE GUIDANCE",
        user_message="Task",
        toolkit=toolkit,
        max_turns=maximum,
    )


def test_context_replaces_complete_prompt_and_selects_skills(tmp_path):
    context = ContextPackage(
        instructions="Editable instruction",
        skills=(
            TextSkill(skill_id="irrelevant", scope="task-family", text="DO NOT LOAD"),
            TextSkill(skill_id="useful", scope="global", text="SELECTED SKILL"),
        ),
        program=program(
            """def build(runtime, context):
    selected = next(skill["text"] for skill in context["assets"]["skills"] if skill["skill_id"] == "useful")
    return {"instructions": "CUSTOM " + selected, "messages": context["messages"], "tools": []}
""",
            "builder:build",
        ),
    )

    def model(messages, info):
        assert info.context_guidance == "CUSTOM SELECTED SKILL"
        assert not info.function_tools
        return ModelResponse(parts=[TextPart("analysis")])

    result = run(
        tmp_path,
        "def run(runtime, context):\n    return runtime.plan()\n",
        model,
        context=context,
    )
    assert result.error is None, result.error
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    build = next(event for event in events if event["kind"] == "context_built")
    assert build["input"]["assets"]["skills"][0]["text"] == "DO NOT LOAD"
    assert "DO NOT LOAD" not in build["output"]["instructions"]


def test_native_incremental_delivery_keeps_complete_audit_without_history_replay(
    tmp_path,
):
    lengths = []

    def model(messages, info):
        lengths.append(len(messages))
        assert "Task" in info.instructions
        assert "Task" not in str(messages[0])
        return ModelResponse(
            parts=[
                ToolCallPart("inspect_scene", {}, tool_call_id=f"call-{len(lengths)}")
            ]
        )

    result = run(
        tmp_path,
        """def run(runtime, context):
    for _ in range(8):
        response = runtime.plan()
        runtime.execute(response["tool_calls"][0])
    return {"completed": True}
""",
        model,
        maximum=8,
    )
    assert result.error is None, result.error
    assert len(lengths) == 8
    assert max(lengths) <= 3
    assert len([m for m in result.messages if m["role"] == "assistant"]) == 8
    assert len([m for m in result.messages if m.get("name") == "inspect_scene"]) == 8
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    builds = [event for event in events if event["kind"] == "context_built"]
    assert len(builds[-1]["input"]["messages"]) <= 3
    assert len(builds[-1]["output"]["messages"]) == lengths[-1]


@pytest.mark.parametrize("inherited_context", [False, True])
def test_live_physical_budget_reaches_root_and_workflow_through_same_c(
    tmp_path, inherited_context
):
    from reweave.benchmarks.libero.toolkit import LiberoToolkit
    from reweave.benchmarks.libero.tools import LiberoPrimitives
    from reweave.evolution.bundle import WorkflowSpec
    from tests.unit_tests.harness.evolution.fixtures import workflow

    class BudgetRobot(Robot):
        def __init__(self, path):
            super().__init__(path)
            self.used = 10
            self.moves = 0
            # Real LIBERO budget producer/delegation; only environment counters
            # and physical responses are synthetic, so no robot is contacted.
            self._primitives = LiberoPrimitives.__new__(LiberoPrimitives)
            self._primitives.env = SimpleNamespace(
                max_episode_steps=310,
                elapsed_steps=lambda: self.used,
                steps_remaining=lambda: 310 - self.used,
            )
            self.state.latest_record = lambda: None
            self.state.get = lambda step: {}[step]
            self.add_tool(
                "move",
                {
                    "name": "move",
                    "description": "move",
                    "input_schema": {"type": "object"},
                },
                self.move,
            )

        def move(self):
            self.moves += 1
            self.used += 37
            return {"executed_steps": 37}

        def get_env_state(self, **kwargs):
            return {"step": self.moves, "robot_position": [1, 2, 3]}

        control_step_budget = LiberoToolkit.control_step_budget

    payload = workflow().model_dump(mode="json")
    payload["permissions"] = ["model:plan", "move"]
    payload["limits"].update(runtime_calls=12, model_calls=2)
    payload["program"] = program(
        """def run(runtime, context):
    assert context["physical_budget"]["remaining"] == 300
    runtime.plan("Before motion")
    runtime.act("move", {})
    snapshot = runtime.snapshot()
    assert snapshot["physical_budget"]["remaining"] == 263
    assert {tool["name"] for tool in snapshot["tools"]} == {"move"}
    runtime.plan("After motion")
    return {"status": "completed", "summary": "observed", "output": {}}
""",
        "probe:run",
    ).model_dump(mode="json")
    context = None
    if inherited_context:
        # Existing authored C consuming only messages, with no budget-specific code.
        context = ContextPackage(
            program=program(
                """def build(runtime, context):
    return {"instructions": "INHERITED", "messages": context["messages"], "tools": context["tools"]}
""",
                "context:build",
            )
        )
    seen = []

    def model(messages, info):
        facts = [
            part.content
            for message in messages
            for part in message.parts
            if getattr(part, "part_kind", None) == "user-prompt"
            and isinstance(part.content, str)
            and part.content.startswith("Current runtime budget.")
        ]
        assert len(facts) == 1  # No accumulation of stale per-request budgets.
        fact = json.loads(facts[0][facts[0].index("{") :])
        assert 0 < fact["remaining_wall_time_s"] <= 15
        seen.append(fact["physical_budget"])
        if len(seen) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "workflow_probe", {"arguments": {}}, tool_call_id="work"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("checked")])

    robot = BudgetRobot(tmp_path)
    result = run(
        tmp_path,
        """def run(runtime, context):
    runtime.execute(runtime.plan()["tool_calls"][0])
    return runtime.plan("Continue from the actual exit")
""",
        model,
        context=context,
        robot=robot,
        workflows=[WorkflowSpec.model_validate(payload)],
    )
    assert result.error is None, result.error
    assert [item["remaining"] for item in seen] == [300, 300, 263, 263]
    assert all(item["unit"] == "environment_control_step" for item in seen)
    assert seen[-1]["operation_costs"]["pi0_pick"]["default_actions_per_chunk"] == 5
    assert robot.moves == 1 and robot.used == 47
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    builds = [item for item in events if item["kind"] == "context_built"]
    assert [item["input"]["physical_budget"]["remaining"] for item in builds] == [
        300,
        300,
        263,
        263,
    ]
    assert builds[0]["program_call_id"] == builds[-1]["program_call_id"]
    assert (
        builds[1]["program_call_id"]
        == builds[2]["program_call_id"]
        != builds[0]["program_call_id"]
    )


def test_missing_physical_counter_is_unknown_in_actual_request(tmp_path):
    def model(messages, info):
        assert '"physical_budget": null' in str(messages)
        return ModelResponse(parts=[TextPart("checked")])

    result = run(
        tmp_path,
        """def run(runtime, context):
    assert runtime.snapshot()["physical_budget"] is None
    return runtime.plan()
""",
        model,
    )
    assert result.error is None, result.error


def test_loop_observes_then_executes_pending_call_without_host_arbitration(tmp_path):
    robot = Robot(tmp_path)
    calls = []

    def model(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("inspect_scene", {}, tool_call_id="original")]
            )
        assert robot.calls == 2
        assert "position" in str(messages)
        return ModelResponse(parts=[TextPart("Done")])

    result = run(
        tmp_path,
        """def run(runtime, context):
    response = runtime.plan()
    runtime.observe("inspect_scene", {})
    runtime.execute(response["tool_calls"][0])
    return runtime.plan()
""",
        model,
        robot=robot,
    )
    assert result.error is None and len(calls) == 2, result.error
    assert any(
        bool(message.get("provider_tool_call_id")) for message in result.messages
    )


def test_author_written_parallel_review_functions_need_no_agent_registry(tmp_path):
    active = 0
    peak = 0

    async def model(messages, info):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        assert not info.function_tools
        return ModelResponse(parts=[TextPart("reviewed")])

    result = run(
        tmp_path,
        """def review(runtime, task):
    answer = runtime.plan("Reviewer" + ": " + task["question"])
    return {"answer": answer["text"]}

def run(runtime, context):
    first = runtime.start("loop:review", {"question": "Check hypothesis A"}, ["model:plan"])
    second = runtime.start("loop:review", {"question": "Check hypothesis B"}, ["model:plan"])
    return {"first": runtime.join(first), "second": runtime.join(second)}
""",
        model,
    )
    assert result.error is None, result.error
    assert peak == 1 and result.stats["visible_turns"] == 2
    requests = [
        e
        for e in rows(tmp_path / "harness_context/bundle_events.jsonl")
        if e["kind"] == "model_response"
    ]
    assert len({e["program_call_id"] for e in requests}) == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    tasks = [event for event in events if event["kind"] == "task_started"]
    assert (
        len(tasks) == 2 and len({task["parent_program_call_id"] for task in tasks}) == 1
    )


def test_parallel_requests_share_one_hard_model_budget(tmp_path):
    calls = []

    def model(messages, info):
        calls.append(1)
        return ModelResponse(parts=[TextPart("one request")])

    result = run(
        tmp_path,
        """def work(runtime, task):
    return runtime.plan("Review" + ": " + "Inspect")

def run(runtime, context):
    a = runtime.start("loop:work", {})
    b = runtime.start("loop:work", {})
    runtime.join(a)
    return runtime.join(b)
""",
        model,
        maximum=1,
    )
    assert len(calls) == result.stats["visible_turns"] == 1
    assert result.termination_class == "agent_execution_failure"


def test_parent_exit_waits_for_cancelled_model_task(tmp_path):
    child_started = threading.Event()
    child_cancelled = threading.Event()
    robot = Robot(tmp_path)
    from reweave.tools.toolkit import readonly

    robot.add_tool(
        "await_child_started",
        {
            "name": "await_child_started",
            "description": "Read a synthetic child-start barrier",
            "input_schema": {"type": "object", "properties": {}},
        },
        readonly(lambda: {"started": child_started.wait(5)}),
    )

    async def model(messages, info):
        assert info.objective.startswith("Child:")
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_cancelled.set()

    result = run(
        tmp_path,
        """def child(runtime, task):
    return runtime.plan("Child" + ": " + "wait")

def run(runtime, context):
    runtime.start("loop:child", {})
    assert runtime.observe("await_child_started", {})["started"]
    return {"summary": "done"}
""",
        model,
        robot=robot,
    )
    assert result.error is None, result.error
    assert child_cancelled.is_set()
    events = rows(tmp_path / "native/native_turns.jsonl")
    assert len(events) == 1 and events[0]["status"] == "failed"
    assert events[0]["error"] == "CancelledError"


def test_child_cannot_regain_removed_model_permission(tmp_path):
    def model(messages, info):
        raise AssertionError("restricted task must not reach provider")

    result = run(
        tmp_path,
        """def child(runtime, task):
    return runtime.plan()

def run(runtime, context):
    job = runtime.start("loop:child", {}, [])
    return runtime.join(job)
""",
        model,
    )
    assert result.termination_class == "agent_execution_failure"
    assert result.stats["visible_turns"] == 0


def test_root_loop_task_cannot_inherit_workflow_execution_authority(tmp_path):
    from reweave.evolution.bundle import WorkflowSpec
    from tests.unit_tests.harness.evolution.fixtures import workflow

    result = run(
        tmp_path,
        """def child(runtime, task):
    return runtime.control("workflow:probe", {"arguments": {}})

def run(runtime, context):
    return runtime.join(runtime.start("loop:child", {}))
""",
        lambda *_: None,
        workflows=[WorkflowSpec.model_validate(workflow().model_dump(mode="json"))],
    )
    assert result.termination_class == "agent_execution_failure"
    assert "tool exceeds execution authority" in result.error


def test_concurrent_provider_ids_link_to_exact_request_without_history_guessing(
    tmp_path,
):
    from types import SimpleNamespace

    from reweave.evolution.evidence.trace import _decision_contexts

    directory = tmp_path / "harness_context"
    directory.mkdir()
    requests = [
        {"request_id": f"r{i}", "program_call_id": f"e{i}", "full_history": []}
        for i in range(2)
    ]
    responses = [
        [
            {
                "kind": "response",
                "metadata": {
                    "request_id": f"r{i}",
                    "program_call_id": f"e{i}",
                    "tool_call_ids": {"same-provider-id": f"global-{i}"},
                },
            }
        ]
        for i in range(2)
    ]
    (directory / "planner_requests.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in requests)
    )
    (directory / "planner_responses.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in responses)
    )
    episode = SimpleNamespace(
        root=tmp_path,
        ref=lambda name, pointer="": (
            {"path": name, "pointer": pointer} if (tmp_path / name).exists() else None
        ),
    )
    linked = _decision_contexts(episode)
    assert linked["global-0"][0]["pointer"] == "/0"
    assert linked["global-1"][0]["pointer"] == "/1"
    responses[1][0]["metadata"]["program_call_id"] = "e0"
    (directory / "planner_responses.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in responses)
    )
    import pytest

    with pytest.raises(ValueError, match="execution identity"):
        _decision_contexts(episode)


def test_context_selection_keeps_source_history_and_private_builder_state(tmp_path):
    context = ContextPackage(
        program=program(
            """def build(runtime, context):
    count = context["state"].get("count", 0)
    messages = context["messages"]
    if count == 1:
        messages = messages[-1:]
    if count == 2:
        assert "first" not in str(context["messages"])
    return {"instructions": "custom", "messages": messages, "tools": [], "state": {"count": count + 1}}
""",
            "builder:build",
        )
    )
    seen = []

    def model(messages, info):
        seen.append(str(messages))
        return ModelResponse(parts=[TextPart("answer")])

    result = run(
        tmp_path,
        """def run(runtime, context):
    runtime.plan("first")
    runtime.plan("second")
    return runtime.plan("third")
""",
        model,
        context=context,
    )
    assert result.error is None, result.error
    assert "first" not in seen[1] and "first" not in seen[2]


def test_loop_can_decline_a_call_and_replan_without_executing_it(tmp_path):
    calls = []

    def model(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("inspect_scene", {}, tool_call_id="declined")]
            )
        assert "program_feedback" in str(messages)
        return ModelResponse(parts=[TextPart("alternative")])

    robot = Robot(tmp_path)
    result = run(
        tmp_path,
        """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    runtime.feedback(call["id"], {"reason": "existing observation is sufficient"})
    return runtime.plan("Choose another approach")
""",
        model,
        robot=robot,
    )
    assert result.error is None and robot.calls == 0, result.error
    feedback = next(
        message for message in result.messages if message.get("provider_tool_call_id")
    )
    assert feedback["executed"] is None


def test_root_loop_cannot_replace_a_planner_tool_call(tmp_path):
    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "read_image", {"name": "missing.png"}, tool_call_id="replace"
                )
            ]
        )

    robot = Robot(tmp_path)
    result = run(
        tmp_path,
        """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    call["name"] = "inspect_scene"
    call["arguments"] = {}
    return runtime.execute(call)
""",
        model,
        robot=robot,
    )
    assert result.termination_class == "agent_execution_failure"
    assert "unchanged tool call returned by runtime.plan()" in result.error
    assert robot.calls == 0


def test_workflow_cannot_reset_its_model_allowance_by_starting_a_function(tmp_path):
    from reweave.evolution.bundle import WorkflowSpec
    from tests.unit_tests.harness.evolution.fixtures import workflow

    payload = workflow().model_dump(mode="json")
    payload["permissions"] = ["model:plan", "task:join", "task:start"]
    payload["program"] = program(
        """def child(runtime, task):
    return runtime.plan("Reviewer" + ": " + "Check")

def run(runtime, context):
    child = runtime.start("probe:child", {})
    result = runtime.join(child)
    return {"status": "completed", "summary": "checked", "output": result}
""",
        "probe:run",
    ).model_dump(mode="json")
    payload["limits"]["model_calls"] = 0

    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "workflow_probe", {"arguments": {}}, tool_call_id="workflow"
                )
            ]
        )

    result = run(
        tmp_path,
        """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    return runtime.execute(call)
""",
        model,
        workflows=[WorkflowSpec.model_validate(payload)],
    )
    assert result.termination_class == "agent_execution_failure"
    assert "model budget exhausted" in result.error
    assert result.stats["visible_turns"] == 1


def test_root_uses_the_same_plan_decision_for_a_local_question(tmp_path):
    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart("inspect_scene", {}, tool_call_id=f"plan-{len(messages)}")
            ]
        )

    robot = Robot(tmp_path)
    result = run(
        tmp_path,
        """def run(runtime, context):
    call = runtime.plan("Reviewer" + ": " + "Inspect")["tool_calls"][0]
    runtime.execute(call)
    return runtime.plan("Use the observation")
""",
        model,
        robot=robot,
    )
    assert result.error is None, result.error
    assert robot.calls == 1 and result.stats["visible_turns"] == 2


def test_context_cannot_reenter_itself_through_model_reason(tmp_path):
    context = ContextPackage(
        program=program(
            """def build(runtime, context):
    result = runtime.plan("Context analysis" + ": " + "Select guidance")
    return {"instructions": result["text"], "messages": context["messages"], "tools": []}
""",
            "context:build",
        )
    )
    calls = []

    def model(messages, info):
        calls.append(info.instructions)
        return ModelResponse(parts=[TextPart("Selected guidance")])

    result = run(
        tmp_path,
        "def run(runtime, context):\n    return runtime.plan()\n",
        model,
        context=context,
    )
    assert result.termination_class == "agent_execution_failure"
    assert "call models recursively" in result.error
    assert calls == [] and result.stats["visible_turns"] == 0


def test_one_plan_context_supplies_local_feedback_and_preserves_builder_state(tmp_path):
    context = ContextPackage(
        program=program(
            """def build(runtime, context):
    count = context["state"].get("count", 0) + 1
    return {"instructions": "COMMON C " + str(count),
            "messages": context["messages"], "tools": context["tools"],
            "state": {"count": count}}
""",
            "context:build",
        )
    )
    seen = []

    def model(messages, info):
        seen.append((info.context_guidance, messages))
        return ModelResponse(parts=[TextPart("checked")])

    result = run(
        tmp_path,
        """def run(runtime, context):
    runtime.plan("First question")
    runtime.observe("inspect_scene", {})
    runtime.plan("Check current evidence")
    runtime.plan("Next question")
    return {}
""",
        model,
        context=context,
    )
    assert result.error is None, result.error
    assert [item[0] for item in seen] == ["COMMON C 1", "COMMON C 2", "COMMON C 3"]
    assert '"history_index": 0' in str(seen[1][1])
    assert '"position": 1' in str(seen[1][1])
    assert str(seen[2][1]).count('"history_index": 0') == 0


def test_malformed_provider_tool_arguments_preserve_original_response(tmp_path):
    def model(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("inspect_scene", "{broken", tool_call_id="malformed")]
        )

    result = run(
        tmp_path, "def run(runtime, context):\n    return runtime.plan()\n", model
    )
    assert result.error is None
    events = (tmp_path / "harness_context/bundle_events.jsonl").read_text()
    assert '"kind": "model_call_rejected"' in events
    rejected = next(
        json.loads(line)
        for line in events.splitlines()
        if json.loads(line)["kind"] == "model_call_rejected"
    )
    assert rejected["call"]["arguments_json"] == "{broken"
    assert result.stats["visible_turns"] == 1


def test_workflow_exit_drains_children_before_outer_loop_continues(tmp_path):
    from reweave.evolution.bundle import WorkflowSpec
    from reweave.tools.toolkit import readonly
    from tests.unit_tests.harness.evolution.fixtures import workflow

    started = threading.Event()
    cancelled = threading.Event()
    robot = Robot(tmp_path)
    robot.add_tool(
        "await_child_started",
        {
            "name": "await_child_started",
            "description": "Read a synthetic child-start barrier",
            "input_schema": {"type": "object", "properties": {}},
        },
        readonly(lambda: {"started": started.wait(5)}),
    )

    async def model(messages, info):
        if info.objective is None:
            return ModelResponse(
                parts=[ToolCallPart("workflow_probe", {"arguments": {}})]
            )
        if info.objective.startswith("Child:"):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        assert info.objective.startswith("Outer:")
        assert cancelled.is_set(), "W must drain its children before returning to L"
        return ModelResponse(parts=[TextPart("done")])

    payload = workflow().model_dump(mode="json")
    payload["permissions"] = ["await_child_started", "model:plan", "task:start"]
    payload["observations"] = ["await_child_started"]
    payload["limits"]["model_calls"] = 3
    payload["program"] = program(
        """def child(runtime, context):
    return runtime.plan("Child" + ": " + "Wait")

def run(runtime, context):
    runtime.start("probe:child", {})
    assert runtime.observe("await_child_started", {})["started"]
    return {"status": "completed", "summary": "done", "output": {}}
""",
        "probe:run",
    ).model_dump(mode="json")
    result = run(
        tmp_path,
        """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    runtime.execute(call)
    return runtime.plan("Outer" + ": " + "Continue")
""",
        model,
        robot=robot,
        workflows=[WorkflowSpec.model_validate(payload)],
    )
    assert result.error is None, result.error
    assert cancelled.is_set() and result.stats["visible_turns"] == 3


def test_async_worker_runs_event_loop_and_serializes_runtime_calls(tmp_path):
    def model(messages, info):
        return ModelResponse(parts=[TextPart(info.objective)])

    result = run(
        tmp_path,
        """import asyncio

async def run(runtime, context):
    ready = asyncio.Event()
    async def producer():
        await asyncio.sleep(0.01)
        ready.set()
        return await runtime.plan("one" + ": " + "inspect")
    async def consumer():
        await ready.wait()
        return await runtime.plan("two" + ": " + "inspect")
    a, b = await asyncio.gather(producer(), consumer())
    assert a["text"] == "one: inspect" and b["text"] == "two: inspect"
    return {"done": True}
""",
        model,
    )
    assert result.error is None, result.error
    assert result.stats["visible_turns"] == 2


def test_async_cancellation_drains_rpc_before_next_call(tmp_path):
    async def model(messages, info):
        await asyncio.sleep(0.05)
        return ModelResponse(parts=[TextPart("first result")])

    robot = Robot(tmp_path)
    result = run(
        tmp_path,
        """import asyncio

async def run(runtime, context):
    task = asyncio.create_task(runtime.plan("one" + ": " + "inspect"))
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    result = await runtime.observe("inspect_scene", {})
    assert result["position"] == 1
    return result
""",
        model,
        robot=robot,
    )
    assert result.error is None, result.error
    assert robot.calls == 1 and result.stats["visible_turns"] == 1


def test_read_results_do_not_replace_environment_observation_and_reach_workflow(
    tmp_path,
):
    from reweave.evolution.bundle import WorkflowSpec
    from tests.unit_tests.harness.evolution.fixtures import workflow

    class PhysicalRobot(Robot):
        def __init__(self, path):
            super().__init__(path)
            self.state.latest_record = lambda: None
            self.state.get = lambda step: {}[step]
            self.add_tool(
                "move",
                {
                    "name": "move",
                    "description": "move",
                    "input_schema": {"type": "object"},
                },
                lambda: {"moved": True},
            )

        def get_env_state(self, **kwargs):
            return {"robot_position": [1, 2, 3]}

    payload = workflow().model_dump(mode="json")
    payload["program"] = program(
        """def run(runtime, context):
    assert context["observation"]["robot_position"] == [1, 2, 3]
    assert context["last_tool_result"]["operation"] == "read_image"
    assert "error" in context["last_tool_result"]["result"]
    assert [item["operation"] for item in context["history"]] == ["move", "inspect_scene", "read_image"]
    return {"status": "completed", "summary": "verified", "output": {}}
""",
        "probe:run",
    ).model_dump(mode="json")

    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart("move", {}, tool_call_id="move"),
                ToolCallPart("inspect_scene", {}, tool_call_id="inspect"),
                ToolCallPart(
                    "read_image", {"name": "missing.png"}, tool_call_id="image"
                ),
                ToolCallPart(
                    "workflow_probe", {"arguments": {}}, tool_call_id="workflow"
                ),
            ]
        )

    result = run(
        tmp_path,
        """def run(runtime, context):
    calls = runtime.plan()["tool_calls"]
    result = None
    for call in calls:
        result = runtime.execute(call)
    return result
""",
        model,
        robot=PhysicalRobot(tmp_path),
        workflows=[WorkflowSpec.model_validate(payload)],
    )
    assert result.error is None, result.error
