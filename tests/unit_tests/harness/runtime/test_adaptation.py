# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Actual candidate capture and immutable program execution across revisions."""

import asyncio
import json
from pathlib import Path

import pytest

from reweave.evolution.authoring.candidate import write_bundle_source
from reweave.evolution.bundle import (
    AgentBundle,
    ProgramLimits,
    PythonFile,
    PythonProgram,
)
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime import adaptation
from reweave.runtime.adaptation import HarnessAdaptation, read_adaptation_receipt
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.runtime.execution import ExecutionContext, ProgramExecution
from reweave.runtime.kernel import EpisodeLimits, validate_json
from reweave.runtime.sandbox import ProgramSandbox
from tests.unit_tests.harness.evolution.fixtures import baseline, workflow
from tests.unit_tests.harness.evolution.test_bundle_runtime import Robot


def program(module, source):
    return PythonProgram(
        entrypoint=module + ":run",
        files=(PythonFile(path=module + ".py", source=source),),
    )


def bundle(marker, *, loop_marker="original"):
    data = baseline().model_dump(mode="json")
    item = workflow().model_dump(mode="json")
    item["program"] = program(
        "probe",
        f'''def run(runtime, context):
    count = context.get("state", {{}}).get("count", 0) + 1
    return {{"status": "completed", "summary": "fixture", "output": {{"marker": "{marker}", "count": count}}}}
''',
    ).model_dump(mode="json")
    data["workflows"] = [item]
    data["context"]["program"] = program(
        "context",
        f'''def run(runtime, context):
    return {{"instructions": "{marker}", "messages": context["messages"], "tools": context["tools"]}}
''',
    ).model_dump(mode="json")
    data["loop"]["program"] = program(
        "loop",
        f'''def run(runtime, context):
    return {{"status": "checkpoint", "state": {{"marker": "{loop_marker}", "previous": context.get("state")}}}}
''',
    ).model_dump(mode="json")
    return AgentBundle.model_validate(data)


def controller(tmp_path, initial=None, *, mode="adaptive"):
    return HarnessAdaptation(
        initial or bundle("A"),
        candidate_directory=tmp_path / "candidate",
        audit_directory=tmp_path / "audit",
        mode=mode,
    )


def execute(binding, payload=None):
    async def forbidden(*args):
        raise AssertionError("fixture did not request any external action/model")

    return asyncio.run(
        ProgramSandbox().run_program(
            binding.program,
            ProgramLimits(
                runtime_calls=10, physical_calls=0, model_calls=0, wall_time_s=10
            ),
            payload or {},
            forbidden,
            input_schema={"type": "object"},
        )
    )


def test_cw_activate_after_call_settles_without_reloading_active_l(tmp_path):
    manager = controller(tmp_path)
    initial = manager.active
    with manager.bind("L") as active_loop:
        with manager.bind("W", workflow_id="probe") as old:
            write_bundle_source(bundle("B"), manager.candidate_directory)
            revision = manager.stage(
                reason="Use observed feedback", evidence=("step:4",)
            )
            assert manager.pending is revision
            assert manager.active is initial
            assert execute(old)["output"]["marker"] == "A"
        assert manager.active is revision
        assert manager.pending is None
        assert active_loop.program == initial.bundle.loop.program
        with manager.bind("C") as current:
            assert (
                execute(current, {"messages": [], "tools": []})["instructions"] == "B"
            )
        with manager.bind(
            "W", workflow_id="probe", revision_id=initial.revision_id
        ) as cached_proposal:
            assert execute(cached_proposal)["output"]["marker"] == "A"
    receipt = manager.receipt()
    assert receipt["initial_bundle_id"] == initial.bundle.record_id
    assert receipt["active_revision_id"] == revision.revision_id
    assert receipt["revisions"][1]["parent_revision_id"] == initial.revision_id
    record = receipt["revisions"][1]
    assert (
        AgentBundle.model_validate_json(Path(record["bundle_path"]).read_text())
        == revision.bundle
    )
    assert '"B"' in (revision.directory / "diff.patch").read_text()


def test_old_workflow_continuation_keeps_source_contract_and_limits(tmp_path):
    manager = controller(tmp_path)
    with manager.bind("W", workflow_id="probe") as old:
        first = execute(old)
        manager.retain_continuation("kernel-token", old)
    write_bundle_source(bundle("B"), manager.candidate_directory)
    manager.stage(reason="New future workflow")
    with manager.bind(
        "W", workflow_id="probe", continuation_id="kernel-token"
    ) as resumed:
        assert resumed.revision is old.revision
        assert resumed.workflow is old.workflow
        assert resumed.workflow.limits == old.workflow.limits
        assert execute(resumed, {"state": {"count": first["output"]["count"]}})[
            "output"
        ] == {"marker": "A", "count": 2}
    with pytest.raises(ValueError, match="retained workflow revision"):
        with manager.bind(
            "W",
            workflow_id="probe",
            continuation_id="kernel-token",
            revision_id=manager.active.revision_id,
        ):
            pass
    manager.consume_continuation("kernel-token")
    assert manager.receipt()["retained_continuations"] == {}


def test_l_requires_cooperative_settled_checkpoint_and_receives_detached_state(
    tmp_path,
):
    manager = controller(tmp_path)
    initial = manager.active
    with manager.bind("L") as old:
        write_bundle_source(
            bundle("B", loop_marker="revised"), manager.candidate_directory
        )
        revised = manager.stage(reason="Resume L with explicit state")
        assert manager.active is initial
        with pytest.raises(RuntimeError, match="all calls to settle"):
            manager.checkpoint({"counter": 3})
        returned = execute(old, {"state": {"counter": 3}})
    assert manager.pending is revised
    manager.checkpoint(returned["state"])
    assert manager.active is revised
    copied = manager.loop_state
    copied["marker"] = "caller mutation"
    with manager.bind("L") as resumed:
        assert resumed.state["marker"] == "original"
        assert execute(resumed, {"state": resumed.state})["state"] == {
            "marker": "revised",
            "previous": returned["state"],
        }
        assert manager.loop_state is None
    with pytest.raises(ValueError):
        manager.checkpoint({"invalid": float("nan")})


def test_invalid_import_is_visible_and_does_not_activate_or_replace_old_source(
    tmp_path,
):
    manager = controller(tmp_path)
    initial = manager.active
    (manager.candidate_directory / "task-family/context/src/context.py").write_text(
        'def run(runtime, context):\n    return {}\nraise RuntimeError("bad edit")\n'
    )
    with pytest.raises(ValueError, match="bad edit"):
        manager.stage(reason="Invalid edit")
    assert manager.active is initial
    assert manager.pending is None
    events = [
        json.loads(line)
        for line in (manager.audit_directory / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "stage_failed"
    assert (
        Path(events[-1]["failed_source"]["directory"])
        / "source/task-family/context/src/context.py"
    ).read_bytes() == (
        manager.candidate_directory / "task-family/context/src/context.py"
    ).read_bytes()
    write_bundle_source(bundle("B"), manager.candidate_directory)
    manager.stage(reason="Repair the same private candidate")
    assert manager.active.bundle.context.program != initial.bundle.context.program


def test_activation_publication_failure_retains_pending_and_old_active(
    tmp_path, monkeypatch
):
    manager = controller(tmp_path)
    initial = manager.active
    original = adaptation.atomic_json

    def fail_active(path, value):
        if path == manager.audit_directory / "active.json":
            raise OSError("cannot publish activation")
        return original(path, value)

    write_bundle_source(bundle("B"), manager.candidate_directory)
    monkeypatch.setattr(adaptation, "atomic_json", fail_active)
    with pytest.raises(OSError, match="cannot publish activation"):
        manager.stage(reason="Candidate remains pending on disk failure")
    assert manager.active is initial
    assert manager.pending is not None
    assert (
        json.loads((manager.audit_directory / "active.json").read_text())["revision_id"]
        == initial.revision_id
    )
    monkeypatch.setattr(adaptation, "atomic_json", original)
    manager.stage(reason="Explicit same captured bytes at a settled boundary")
    assert manager.pending is None
    assert manager.active is not initial


def test_unknown_outcome_survives_activation_failure_during_call_cleanup(
    tmp_path, monkeypatch
):
    manager = controller(tmp_path)
    initial = manager.active
    original = adaptation.atomic_json
    unknown = PhysicalOutcomeUnknown("ACK unavailable")

    def fail_active(path, value):
        if path == manager.audit_directory / "active.json":
            raise OSError("activation disk failure")
        return original(path, value)

    with pytest.raises(PhysicalOutcomeUnknown) as captured:
        with manager.bind("W", workflow_id="probe"):
            write_bundle_source(bundle("B"), manager.candidate_directory)
            manager.stage(reason="Wait for active W")
            monkeypatch.setattr(adaptation, "atomic_json", fail_active)
            raise unknown
    assert captured.value is unknown
    assert isinstance(unknown.__cause__, OSError)
    assert manager.active is initial


def test_static_mode_rejects_online_edit_with_no_activation(tmp_path):
    manager = controller(tmp_path, mode="static")
    initial = manager.active
    assert manager.stage(reason="No edit") is initial
    write_bundle_source(bundle("B"), manager.candidate_directory)
    with pytest.raises(PermissionError, match="static mode"):
        manager.stage(reason="Attempted edit")
    assert manager.active is initial
    assert manager.receipt()["mode"] == "static"


DECISION_LOOP = """def run(runtime, context):
    results = []
    for _ in range(3):
        decision = runtime.plan("Use the settled feedback")
        for call in decision["tool_calls"]:
            results.append(runtime.execute(call))
    return {"results": results}
"""


def executable_bundle(marker="A", *, loop=DECISION_LOOP, workflows=None):
    data = bundle(marker).model_dump(mode="json")
    data["loop"]["program"] = program("loop", loop).model_dump(mode="json")
    if workflows is None:
        item = data["workflows"][0]
        item["permissions"] = ["inspect_scene"]
        item["observations"] = ["inspect_scene"]
        item["program"] = program(
            "probe",
            f'''def run(runtime, context):
    observed = runtime.observe("inspect_scene", {{}})
    return {{"status": "completed", "summary": "inspected", "output": {{"marker": "{marker}", "observed": observed}}}}
''',
        ).model_dump(mode="json")
    else:
        data["workflows"] = workflows
    return AgentBundle.model_validate(data)


def runtime(tmp_path, initial, *, mode="adaptive"):
    manager = controller(tmp_path / "sources", initial, mode=mode)
    robot = Robot(tmp_path / "episode")
    bridge = BundleRuntime(
        initial,
        robot,
        "Inspect current evidence",
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=60
        ),
        adaptation=manager,
    )
    return manager, robot, bridge


def proposal(context, index, name, arguments=None):
    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter,
        ModelResponse,
        ToolCallPart,
    )

    arguments = {} if arguments is None else arguments
    spec = next(item for item in context["tools"] if item["name"] == name)
    validate_json(spec["input_schema"], arguments)
    identity = f"decision-{index}"
    return {
        "text": "Use the current public evidence",
        "message": ModelMessagesTypeAdapter.dump_python(
            [
                ModelResponse(
                    parts=[ToolCallPart(name, arguments, tool_call_id=identity)]
                )
            ],
            mode="json",
        )[0],
        "tool_calls": [
            {
                "id": identity,
                "provider_id": identity,
                "name": name,
                "arguments": arguments,
            }
        ],
    }


def test_actual_l_c_w_activates_added_workflow_and_pins_old_proposal(tmp_path):
    initial = executable_bundle()
    revised_data = executable_bundle("B").model_dump(mode="json")
    added = revised_data["workflows"][0].copy()
    added["workflow_id"] = "second"
    revised_data["workflows"].append(added)
    revised = AgentBundle.model_validate(revised_data)
    manager, robot, bridge = runtime(tmp_path, initial)
    original_kernel = bridge.kernel
    seen = []

    async def request(context, session_id):
        seen.append((context, session_id, bridge.kernel._calls))
        if len(seen) == 1:
            assert context["instructions"] == "A"
            write_bundle_source(revised, manager.candidate_directory)
            return proposal(context, 1, "workflow_probe", {"arguments": {}})
        if len(seen) == 2:
            assert context["instructions"] == "B"
            assert "workflow_second" in {item["name"] for item in context["tools"]}
            return proposal(context, 2, "workflow_second", {"arguments": {}})
        return proposal(context, 3, "read_terminal_state")

    result = asyncio.run(bridge.run(request, "Fixed instructions"))
    assert result["results"][0]["output"]["marker"] == "A"
    assert result["results"][1]["output"]["marker"] == "B"
    assert robot.calls == 2 and robot.terminal
    assert bridge.kernel is original_kernel and bridge.kernel.epoch == 0
    assert len({session_id for _, session_id, _ in seen}) == 1
    assert [calls for _, _, calls in seen] == sorted(calls for _, _, calls in seen)
    messages = bridge.execution.root.messages
    calls = list(bridge.execution.root.pending.values())
    assert not calls and len(manager.receipt()["revisions"]) == 2
    assert all(
        message["kind"] == "request"
        for context, _, _ in seen
        for message in context["messages"]
    )
    assert sum(message["kind"] == "response" for message in messages) == 3
    assert all(
        '"position": 1' not in json.dumps(message) for message in seen[2][0]["messages"]
    )
    context_events = [
        event
        for line in bridge._trace.read_text().splitlines()
        if (event := json.loads(line))["kind"] == "context_built"
    ]
    for event in context_events:
        assert set(event["preparation_timings_s"]) == {
            "feedback_and_snapshot",
            "observation",
            "context_program",
            "validation",
        }
        assert all(value >= 0 for value in event["preparation_timings_s"].values())
        assert "preparation_timings_s" not in event["input"]
        assert "preparation_timings_s" not in event["output"]
    contexts = [event["input"] for event in context_events]
    assert [len(context["history"]) for context in contexts] == [0, 1, 1]
    assert contexts[-1]["history"][0]["result"] == {"position": 2}


def test_actual_continuation_survives_deleted_workflow_and_uses_original_limits(
    tmp_path,
):
    initial_data = executable_bundle().model_dump(mode="json")
    item = initial_data["workflows"][0]
    item["input_schema"] = {
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "array"}},
        "additionalProperties": False,
    }
    item["limits"]["runtime_calls"] = 2
    item["program"] = program(
        "probe",
        """def run(runtime, context):
    count = context.get("state", {}).get("count", 0) + 1
    runtime.observe("inspect_scene", {})
    return {"status": "yielded" if count == 1 else "completed", "summary": "continued", "output": {"marker": "A", "count": count}, "state": {"count": count}}
""",
    ).model_dump(mode="json")
    initial = AgentBundle.model_validate(initial_data)
    revised = executable_bundle("B", workflows=[])
    manager, robot, bridge = runtime(tmp_path, initial)
    calls, counters = [], []

    async def request(context, session_id):
        calls.append(session_id)
        if len(calls) == 1:
            write_bundle_source(revised, manager.candidate_directory)
            assert not (
                manager.candidate_directory / "task-family/workflows/probe"
            ).exists()
            return proposal(
                context, 1, "workflow_probe", {"arguments": {"value": [1, {"x": 2}]}}
            )
        if len(calls) == 2:
            ((token, binding),) = manager.continuations
            assert binding.revision is not manager.active
            stored = bridge.kernel._continuations[token]
            counters.append(stored["counters"])
            assert stored["counters"]["runtime"] == 1
            rejected = await bridge.kernel.workflow(
                "probe",
                {"value": [True, {"x": 2}]},
                observation={},
                task_language="Inspect",
                epoch=bridge.kernel.epoch,
                continuation_id=token,
                workflows=(),
            )
            assert rejected.output == {
                "error": "invalid_continuation",
                "execution_performed": False,
            }
            assert stored["counters"]["runtime"] == 1 and robot.calls == 1
            return proposal(
                context,
                2,
                "workflow_probe",
                {
                    "arguments": {"value": [1.0, {"x": 2.0}]},
                    "continuation_id": token,
                },
            )
        assert "workflow_probe" not in {spec["name"] for spec in context["tools"]}
        return proposal(context, 3, "read_terminal_state")

    result = asyncio.run(bridge.run(request, "Fixed instructions"))
    assert result["results"][1]["output"] == {"marker": "A", "count": 2}
    assert counters[0]["runtime"] == 2
    assert manager.continuations == () and not bridge.kernel._continuations
    assert robot.calls == 2 and bridge.kernel.epoch == 0


def test_actual_l_checkpoint_resumes_new_program_same_root_and_cumulative_kernel(
    tmp_path,
):
    old = """def run(runtime, context):
    call = runtime.plan("Inspect before handoff")["tool_calls"][0]
    observed = runtime.execute(call)
    return {"status": "checkpoint", "state": {"inspected": observed["position"]}}
"""
    new = """def run(runtime, context):
    assert context["state"] == {"inspected": 1}
    call = runtime.plan("Continue the same episode")["tool_calls"][0]
    runtime.execute(call)
    return {"received_state": context["state"], "program_call_id": context["program_call_id"]}
"""
    initial = executable_bundle(loop=old)
    revised = executable_bundle(loop=new)
    manager, robot, bridge = runtime(tmp_path, initial)
    calls = []

    async def request(context, session_id):
        calls.append((session_id, bridge.kernel._calls, bridge.kernel._program_calls))
        if len(calls) == 1:
            write_bundle_source(revised, manager.candidate_directory)
            return proposal(context, 1, "inspect_scene")
        return proposal(context, 2, "read_terminal_state")

    result = asyncio.run(bridge.run(request, "Fixed instructions"))
    assert result["received_state"] == {"inspected": 1}
    assert result["program_call_id"] == calls[0][0] == calls[1][0]
    assert calls[1][1] > calls[0][1] and calls[1][2] > calls[0][2]
    assert robot.calls == 1 and bridge.kernel.epoch == 0
    assert manager.active.bundle.loop == revised.loop
    assert manager.receipt()["activations"][-1]["boundary"] == "loop_checkpoint"


@pytest.mark.parametrize("mode", ["static", "adaptive"])
def test_actual_rejected_edit_is_explicit_same_session_feedback(tmp_path, mode):
    initial = executable_bundle()
    manager, robot, bridge = runtime(tmp_path, initial, mode=mode)
    seen = []

    async def request(context, session_id):
        seen.append((context, session_id))
        if len(seen) == 1:
            write_bundle_source(executable_bundle("B"), manager.candidate_directory)
            if mode == "adaptive":
                (
                    manager.candidate_directory / "task-family/context/src/context.py"
                ).write_text(
                    'def run(runtime, context):\n    return {}\nraise RuntimeError("bad edit")\n'
                )
        elif len(seen) == 2:
            assert "harness_edit_rejected" in json.dumps(context["messages"])
            assert context["instructions"] == "A"
            write_bundle_source(executable_bundle("B"), manager.candidate_directory)
        if len(seen) < 3:
            return proposal(context, len(seen), "inspect_scene")
        assert context["instructions"] == ("B" if mode == "adaptive" else "A")
        return proposal(context, 3, "read_terminal_state")

    asyncio.run(bridge.run(request, "Fixed instructions"))
    assert len({session_id for _, session_id in seen}) == 1
    assert robot.calls == 2 and bridge.kernel.epoch == 0


def test_native_context_cursor_advances_after_delivery_without_cloning_old_history(
    tmp_path,
):
    manager, robot, bridge = runtime(tmp_path, executable_bundle())
    seen = []

    async def request(context, session_id):
        seen.append(context)
        if len(seen) == 1:
            raise RuntimeError("delivery failed")
        return {
            "message": {"kind": "response", "parts": []},
            "text": "read",
            "tool_calls": [],
        }

    async def run():
        execution = ProgramExecution(bridge, request, "Fixed instructions")
        bridge.execution = execution
        session = ExecutionContext(
            "same-root", "main", "Inspect", frozenset({"model:plan"})
        )
        token = execution.current.set(session)
        try:
            with pytest.raises(RuntimeError, match="delivery failed"):
                await execution.plan()
            assert (
                session.delivered_message_position
                == execution._native_history_position
                == 0
            )
            await execution.plan()
            stable_parts = []
            remaining = []
            for context in seen[:2]:
                parts = json.loads(json.dumps(context["messages"][0]["parts"][1:]))
                observation = json.loads(parts[-1]["content"])
                remaining.append(observation.pop("remaining_wall_time_s"))
                parts[-1]["content"] = observation
                stable_parts.append(parts)
            # Host now appends current observation itself. Retry preserves the
            # source evidence while its remaining wall budget still decreases.
            assert stable_parts[0] == stable_parts[1]
            assert 0 < remaining[1] <= remaining[0]
            delivered = session.delivered_message_position
            session.messages[0] = {"kind": "request", "parts": [object()]}
            bridge.history.extend([{"uncopied": object()}, {"result": {"position": 2}}])
            execution._native_history_position = 1
            session.history_position = 2
            await execution.plan()
            assert session.delivered_message_position > delivered
            assert seen[-1]["messages"] and all(
                message["kind"] == "request" for message in seen[-1]["messages"]
            )
        finally:
            execution.current.reset(token)

    asyncio.run(run())
    assert not robot.terminal and robot.calls == 0 and manager.active is not None


def test_receipt_reads_real_snapshots_and_rejects_broken_revision_links_without_hashes(
    tmp_path,
):
    manager = controller(tmp_path)
    write_bundle_source(bundle("B"), manager.candidate_directory)
    manager.stage(reason="Changed C and W")
    path = manager.audit_directory / "receipt.json"
    path.write_text(json.dumps(manager.receipt()))
    receipt = read_adaptation_receipt(path)
    assert receipt == manager.receipt()
    assert "sha256" not in json.dumps(receipt)
    receipt["revisions"][1]["parent_revision_id"] = "missing"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="revision chain"):
        read_adaptation_receipt(path)


def test_repeated_complete_captures_without_edits_do_not_publish_new_revisions(
    tmp_path,
):
    manager = controller(tmp_path)
    initial = manager.active
    for _ in range(3):
        assert manager.stage(reason="Next settled decision") is initial
    # JSON file formatting does not change the actual admitted W/C/L values.
    config = manager.candidate_directory / "task-family/context/context.json"
    config.write_text(json.dumps(json.loads(config.read_text()), indent=4))
    assert manager.stage(reason="Formatting only") is initial
    assert len(manager.receipt()["revisions"]) == 1
    write_bundle_source(bundle("B"), manager.candidate_directory)
    changed = manager.stage(reason="Use the new context")
    assert changed.bundle.record_id != initial.bundle.record_id
    for _ in range(3):
        assert manager.stage(reason="No further edits") is changed
    assert len(manager.receipt()["revisions"]) == 2
    receipt = manager.audit_directory / "receipt.json"
    receipt.write_text(json.dumps(manager.receipt()))
    assert read_adaptation_receipt(receipt) == read_adaptation_receipt(receipt)


def test_import_validation_reused_only_for_identical_programs(tmp_path, monkeypatch):
    import os

    original = ProgramSandbox.validate_program_import
    checked = []

    def validate(self, program, **kwargs):
        checked.append(program)
        return original(self, program, **kwargs)

    monkeypatch.setattr(ProgramSandbox, "validate_program_import", validate)
    manager = controller(tmp_path)
    manager.stage(reason="First capture must validate actual imports")
    assert len(checked) == 3
    manager.stage(reason="Unchanged source")
    assert len(checked) == 3
    source = manager.candidate_directory / "task-family/context/src/context.py"
    info = source.stat()
    source.write_text(source.read_text().replace('"A"', '"B"'))
    os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
    manager.stage(reason="Same size and mtime, different source")
    assert len(checked) == 4
    source.write_text(source.read_text() + '\nraise RuntimeError("bad import")\n')
    for _ in range(2):
        with pytest.raises(ValueError, match="candidate callable validation failed"):
            manager.stage(reason="Failed imports are never cached")
    assert len(checked) == 6


@pytest.mark.parametrize("invalid", ["symlink", "hardlink", "undeclared", "directory"])
def test_import_reuse_does_not_bypass_candidate_boundary(tmp_path, invalid):
    import os

    manager = controller(tmp_path)
    manager.stage(reason="Validate source before tampering")
    root = manager.candidate_directory
    source = root / "task-family/context/src/context.py"
    if invalid in {"symlink", "hardlink"}:
        external = tmp_path / "external.py"
        external.write_bytes(source.read_bytes())
        source.unlink()
        if invalid == "symlink":
            source.symlink_to(external)
        else:
            os.link(external, source)
    elif invalid == "undeclared":
        (root / "task-family/extra.txt").write_text("not declared")
    else:
        (root / "task-family/context/skills").rmdir()
    with pytest.raises(ValueError):
        manager.stage(reason="Reject invalid tree even with matching program")
