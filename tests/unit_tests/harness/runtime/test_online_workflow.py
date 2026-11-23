"""Online W traverses the real native/C/L/Kernel path without model or robot calls."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from reweave.evolution.bundle import AgentBundle, WorkflowSpec
from reweave.evolution.runtime_manifest import (
    execution_configuration,
    validate_budget_arguments,
    validate_configuration,
)
from reweave.planner.decisions import decision_error
from reweave.runtime.adaptation import read_adaptation_receipt
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol, workflow


class MovingRobot(Robot):
    def __init__(self, path, *, unknown=False):
        super().__init__(path)
        self.position = 0
        self.unknown = unknown
        self.add_tool(
            "move",
            {
                "name": "move",
                "description": "One measured synthetic motion.",
                "input_schema": {
                    "type": "object",
                    "properties": {"delta": {"type": "integer"}},
                    "required": ["delta"],
                    "additionalProperties": False,
                },
            },
            self.move,
        )

    def move(self, delta):
        self.position += delta
        if self.unknown:
            raise PhysicalOutcomeUnknown("submitted synthetic action has no ACK")
        return {"position": self.position, "reached": True}


def online_w(
    source, *, name="online_probe", permissions=("inspect_scene", "move"), limits=None
):
    value = workflow().model_dump(mode="json")
    value.update(
        workflow_id=name,
        description="Feedback-dependent synthetic W",
        applicability="Synthetic scene with public motion/observation tools",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        observations=[],
        permissions=list(permissions),
        dependencies=[],
        program={
            "entrypoint": "online:run",
            "files": [{"path": "online.py", "source": source}],
        },
        limits=limits
        or {
            "runtime_calls": 30,
            "physical_calls": 10,
            "model_calls": 0,
            "wall_time_s": 10,
        },
    )
    return WorkflowSpec.model_validate(value).model_dump(mode="json")


def response(name, arguments):
    return ModelResponse(parts=[ToolCallPart(name, arguments)])


def finish():
    return response("read_terminal_state", {})


def solve(robot, callback, *, enabled=True, bundle=None):
    planner = ScriptedNativePlanner(callback, online_workflows=enabled)
    planner.bind_bundle(
        bundle or baseline(),
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="Control",
        user_message="Synthetic task",
        toolkit=robot,
        max_turns=10,
    )
    return planner, result


def events(path):
    return [
        json.loads(line)
        for line in (path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]


def test_disabled_preserves_main_tools_prompt_and_receipt(tmp_path):
    def model(messages, info):
        assert "write_workflow" not in {tool.name for tool in info.function_tools}
        assert "Online W / Code-as-Policy" not in info.instructions
        return finish()

    planner, result = solve(MovingRobot(tmp_path), model, enabled=False)
    assert result.error is None
    assert "online_workflows" not in planner.runtime_configuration()
    receipt = read_adaptation_receipt(tmp_path / "native/adaptation/receipt.json")
    assert "online_workflows" not in receipt and len(receipt["revisions"]) == 1


def test_one_decision_writes_and_runs_feedback_dependent_w(tmp_path):
    robot = MovingRobot(tmp_path)
    program = online_w("""def run(runtime, context):
    total = 0
    for _ in range(3):
        seen = runtime.observe("inspect_scene", {})
        moved = runtime.act("move", {"delta": seen["position"]})
        if not moved["reached"]:
            return {"status": "failed", "summary": "Missed", "output": moved}
        total += seen["position"]
    return {"status": "completed", "summary": "Computed from feedback", "output": {"sum": total}}
""")
    turns = []

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            assert "write_workflow" in {tool.name for tool in info.function_tools}
            assert "Prefer programs that carry out several steps" in info.instructions
            return response("write_workflow", {"workflow": program, "arguments": {}})
        assert robot.position == 6 and robot.calls == 3
        assert "workflow_online_probe" in {tool.name for tool in info.function_tools}
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert len(turns) == 2
    receipt = read_adaptation_receipt(tmp_path / "native/adaptation/receipt.json")
    assert receipt["mode"] == "static" and receipt["online_workflows"]
    assert len(receipt["revisions"]) == 2
    assert receipt["revisions"][-1]["changed_scopes"] == ["W"]
    retained = AgentBundle.model_validate_json(
        Path(receipt["revisions"][-1]["bundle_path"]).read_text()
    )
    assert retained.context == baseline().context and retained.loop == baseline().loop
    recorded = events(tmp_path)
    assert sum(row["kind"] == "workflow_written" for row in recorded) == 1
    assert sum(row["kind"] == "workflow_started" for row in recorded) == 1
    assert (
        next(row for row in recorded if row["kind"] == "workflow_timing")["elapsed_s"]
        >= 0
    )


def test_published_online_w_uses_normal_callable_without_rewriting(tmp_path):
    robot = MovingRobot(tmp_path)
    program = online_w("""def run(runtime, context):
    result = runtime.act("move", {"delta": context["arguments"]["delta"]})
    return {"status": "completed", "summary": "Moved", "output": result}
""")
    turns = []

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return response("write_workflow", {"workflow": program})
        if len(turns) == 2:
            assert robot.position == 0
            return response("workflow_online_probe", {"arguments": {"delta": 4}})
        if len(turns) == 3:
            assert robot.position == 4
            return response("workflow_online_probe", {"arguments": {"delta": 3}})
        assert robot.position == 7
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert (
        len(
            read_adaptation_receipt(tmp_path / "native/adaptation/receipt.json")[
                "revisions"
            ]
        )
        == 2
    )


def test_known_code_error_can_be_revised_without_replaying_effects(tmp_path):
    robot = MovingRobot(tmp_path)
    broken = online_w("""def run(runtime, context):
    runtime.act("move", {"delta": 2})
    raise ValueError("known source defect after acknowledged motion")
""")
    repaired = online_w("""def run(runtime, context):
    moved = runtime.act("move", {"delta": 3})
    return {"status": "completed", "summary": "Continued from current state", "output": moved}
""")
    turns = []

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return response("write_workflow", {"workflow": broken, "arguments": {}})
        if len(turns) == 2:
            assert robot.position == 2
            assert "known source defect" in str(messages)
            return response("write_workflow", {"workflow": repaired, "arguments": {}})
        assert robot.position == 5
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert robot.position == 5 and len(turns) == 3
    receipt = read_adaptation_receipt(tmp_path / "native/adaptation/receipt.json")
    assert len(receipt["revisions"]) == 3


@pytest.mark.parametrize("defect", ["global", "unknown_permission", "forbidden_import"])
def test_rejected_w_has_no_actions_and_is_repair_feedback(tmp_path, defect):
    robot = MovingRobot(tmp_path)
    program = online_w(
        'def run(runtime, context):\n    return {"status": "completed", "summary": "Empty", "output": {}}\n'
    )
    if defect == "global":
        program["layer"] = "global"
    elif defect == "unknown_permission":
        program["permissions"] = ["private_simulator"]
    else:
        program["program"]["files"][0]["source"] = (
            "import os\n" + program["program"]["files"][0]["source"]
        )
    turns = []

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return response("write_workflow", {"workflow": program, "arguments": {}})
        assert robot.position == 0
        assert "execution_performed" in str(messages)
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert len(turns) == 2


def test_unknown_physical_outcome_is_terminal_and_never_code_retried(tmp_path):
    robot = MovingRobot(tmp_path, unknown=True)
    program = online_w("""def run(runtime, context):
    runtime.act("move", {"delta": 1})
    runtime.act("move", {"delta": 1})
    return {"status": "completed", "summary": "Moved", "output": {}}
""")
    turns = []

    def model(messages, info):
        turns.append(messages)
        return response("write_workflow", {"workflow": program, "arguments": {}})

    _, result = solve(robot, model)
    assert result.error is not None and "ACK" in result.error
    assert robot.position == 1 and len(turns) == 1


def test_default_conditions_and_configuration_are_historically_compatible():
    conditions = protocol().execution
    assert not conditions.online_workflows
    assert "online_workflows" not in conditions.model_dump(mode="json")
    old = execution_configuration(
        {"planner": "native", "adaptation_mode": "static"}, {}
    )
    disabled = execution_configuration(
        {"planner": "native", "adaptation_mode": "static", "online_workflows": False},
        {},
    )
    assert old == disabled
    validate_configuration(old, disabled)
    with pytest.raises(ValueError, match="configuration differs"):
        validate_configuration(
            old,
            execution_configuration(
                {
                    "planner": "native",
                    "adaptation_mode": "static",
                    "online_workflows": True,
                },
                {},
            ),
        )
    with pytest.raises(ValueError, match="online W"):
        validate_budget_arguments(
            conditions,
            {
                "planner": "native",
                "adaptation_mode": "static",
                "online_workflows": True,
            },
            {},
        )
    assert decision_error(["write_workflow", "move"])


def test_old_continuation_keeps_its_original_online_w_after_rewrite(tmp_path):
    robot = MovingRobot(tmp_path)
    initial = online_w("""def run(runtime, context):
    if not context["state"]:
        runtime.act("move", {"delta": 1})
        return {"status": "yielded", "summary": "Check", "output": {}, "state": {"phase": 1}}
    result = runtime.act("move", {"delta": 2})
    return {"status": "completed", "summary": "Original continuation", "output": result}
""")
    revised = online_w("""def run(runtime, context):
    result = runtime.act("move", {"delta": 20})
    return {"status": "completed", "summary": "New call", "output": result}
""")
    turns, continuation = [], {}

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return response("write_workflow", {"workflow": initial, "arguments": {}})
        if len(turns) == 2:
            completed = next(
                row
                for row in reversed(events(tmp_path))
                if row["kind"] == "workflow_completed"
            )
            continuation.update(completed["result"])
            return response("write_workflow", {"workflow": revised})
        if len(turns) == 3:
            return response(
                "workflow_online_probe",
                {
                    "arguments": continuation["resume_arguments"],
                    "continuation_id": continuation["continuation_id"],
                },
            )
        assert robot.position == 3
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert robot.position == 3 and len(turns) == 4
    assert not read_adaptation_receipt(tmp_path / "native/adaptation/receipt.json")[
        "retained_continuations"
    ]


def test_online_w_local_budget_is_not_reset_inside_loop(tmp_path):
    robot = MovingRobot(tmp_path)
    program = online_w(
        """def run(runtime, context):
    for _ in range(3):
        runtime.act("move", {"delta": 1})
    return {"status": "completed", "summary": "Done", "output": {}}
""",
        limits={
            "runtime_calls": 10,
            "physical_calls": 1,
            "model_calls": 0,
            "wall_time_s": 10,
        },
    )
    turns = []

    def model(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return response("write_workflow", {"workflow": program, "arguments": {}})
        assert robot.position == 1
        assert "budget exhausted" in str(messages)
        return finish()

    _, result = solve(robot, model)
    assert result.error is None, result.error
    assert robot.position == 1


def test_static_online_w_does_not_grant_c_or_l_authority(tmp_path):
    from reweave.evolution.authoring.candidate import write_bundle_source
    from reweave.runtime.adaptation import HarnessAdaptation

    manager = HarnessAdaptation(
        baseline(),
        candidate_directory=tmp_path / "candidate",
        audit_directory=tmp_path / "audit",
        mode="static",
        online_workflows=True,
    )
    data = baseline().model_dump(mode="json")
    data["context"]["instructions"] = "Unrelated C edit"
    write_bundle_source(AgentBundle.model_validate(data), manager.candidate_directory)
    with pytest.raises(PermissionError, match="static"):
        manager.stage(reason="C is not writable under the W-only switch")
    assert manager.active.bundle.context == baseline().context


def test_online_workflow_cli_flag_defaults_off():
    from reweave.cli.main import parse_cli_arguments

    arguments = ["--robot", "libero", "--suite", "libero_goal", "--task", "0"]
    _, plain, _ = parse_cli_arguments(arguments)
    assert not plain.online_workflows


def test_offline_view_keeps_w_source_and_zero_action_events(tmp_path):
    from reweave.evolution.evidence.records import inventory_episode
    from reweave.evolution.evidence.trace import build_episode_context, query_trace
    from tests.unit_tests.harness.evolution.test_trace_context import _episode

    evidence, source = _episode(tmp_path)
    context = source / "harness_context"
    context.mkdir(exist_ok=True)
    (context / "bundle_events.jsonl").write_text(
        json.dumps(
            {
                "kind": "workflow_written",
                "workflow_id": "online_probe",
                "revision_id": "revision-one",
                "write_elapsed_s": 0.01,
            }
        )
        + "\n"
    )
    revision = source / "native/adaptation/revisions/revision-one"
    revision.mkdir(parents=True)
    (revision / "revision.json").write_text('{"revision_id":"revision-one"}')
    (revision.parent.parent / "receipt.json").write_text('{"online_workflows":true}')
    evidence = evidence.model_copy(update={"files": inventory_episode(source)})
    build_episode_context(tmp_path, evidence)
    viewed = query_trace(
        tmp_path, evidence_id=evidence.evidence_id, view="online_workflows"
    )
    assert viewed["coverage"]["selection_complete"]
    data = viewed["data"]
    assert data["adaptation_receipt"] and len(data["revision_records"]) == 1
    assert data["events"][0]["workflow_id"] == "online_probe"
    assert data["events"][0]["source"]


def test_offline_admission_propagates_online_w_condition_to_real_interaction(tmp_path):
    import asyncio

    from reweave.evolution.authoring.candidate import capture_bundle
    from reweave.evolution.validation.admission import admit_candidate
    from reweave.utils.persistence import atomic_json
    from tests.unit_tests.harness.evolution.test_bundle_author import host_input
    from tests.unit_tests.harness.evolution.test_native_interaction_admission import (
        native_workspace,
    )

    root = native_workspace(tmp_path, "static")
    proposal_path = host_input(root, "proposal.json")
    proposal = json.loads(proposal_path.read_text())
    proposal["task_protocol"]["execution"]["online_workflows"] = True
    atomic_json(proposal_path, proposal)
    item = capture_bundle(root / "candidate", baseline().runtime).bundle.workflows[0]
    suite = root / ".scratch/tests/interaction.json"
    scenarios = json.loads(suite.read_text())
    scenarios[0]["model_responses"][0].update(
        tool_calls=[
            {
                "name": "write_workflow",
                "arguments": {
                    "workflow": item.model_dump(mode="json"),
                    "arguments": {},
                },
            }
        ],
        required_tools=["write_workflow"],
    )
    atomic_json(suite, scenarios)
    cases = []
    asyncio.run(admit_candidate(root, case_results=cases))
    interactions = [case for case in cases if case["scope"] == "interaction"]
    assert all(case["online_workflows"] for case in interactions)
    assert any(row["kind"] == "workflow_written" for row in interactions[0]["events"])
