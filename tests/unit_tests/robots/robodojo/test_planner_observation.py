"""Default native deltas and freely authored evidence share bounded delivery."""

import base64
import copy
import io
import json

import pytest
from PIL import Image

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.planner_observation import (
    PlannerObservation,
    feedback_text,
)
from reweave.evolution.bundle import (
    AgentBundle,
    ProgramLimits,
    PythonFile,
    PythonProgram,
    RequestLimits,
    WorkflowSpec,
)
from reweave.planner.native_context import baseline_context, prepare_native
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.runtime.kernel import EpisodeLimits
from reweave.runtime.model_input import image_inventory
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.planner.test_native_loop import FakeSession


def pixels(value):
    stream = io.BytesIO()
    Image.new("RGB", (7, 5), (value, 20, 40)).save(stream, "PNG")
    return stream.getvalue()


def binary(data):
    return {
        "kind": "binary",
        "media_type": "image/png",
        "data": base64.b64encode(data).decode(),
    }


def build(context):
    namespace = {}
    exec(baseline_context().program.files[0].source, namespace)
    return namespace["build"](None, context)


@pytest.mark.parametrize("action_count", [20, 30])
def test_long_workflow_feedback_sends_only_exact_current_cameras(
    tmp_path, action_count
):
    current = [pixels(v) for v in (1, 2, 3)]
    messages = [
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "user-prompt",
                    "content": [
                        json.dumps({"history_index": i, "feedback": {"error": "none"}}),
                        *[binary(pixels(10 + i)) for _ in range(3)],
                    ],
                }
            ],
        }
        for i in range(action_count)
    ]
    source = {
        "messages": messages,
        "observation": {"_runtime_images": [{"data_base64": "archived"}]},
        "physical_budget": {"limit": 1100, "used": 109, "remaining": 991},
        "remaining_wall_time_s": None,
        "objective": "Continue from the settled Workflow",
        "assets": {"instructions": "Conditional task guidance", "skills": []},
        "base_instructions": "Fixed task",
        "tools": [],
    }
    original = copy.deepcopy(source)
    view = PlannerObservation()
    projected = view.prepare(
        source, {"step": 31, "state": {"pose": "current"}}, current
    )
    prepared = build(projected)
    text, images = prepare_native(prepared, tmp_path, RequestLimits(max_images=3))
    assert [path.read_bytes() for path in images] == current
    assert source == original
    assert "archived" not in text
    assert "data_base64" not in text
    assert "robodojo-record-31" in text
    for i in range(action_count):
        assert f'\\"history_index\\": {i}' in text
    assert projected["physical_budget"] == source["physical_budget"]


@pytest.mark.parametrize(
    "mutation", ["append", "omit", "replace", "reorder", "select", "hide_extra"]
)
def test_authored_c_freely_selects_views_and_keyframes(tmp_path, mutation):
    view = PlannerObservation()
    prepared = {
        "messages": [{"kind": "request", "parts": []}],
        "assets": {},
    }
    projected = view.prepare(prepared, {"step": 0}, [pixels(v) for v in (1, 2, 3)])
    content = projected["messages"][-1]["parts"][0]["content"]
    if mutation in {"append", "hide_extra"}:
        content.append(binary(pixels(4)))
        if mutation == "hide_extra":
            projected["image_selection"] = {
                "paths": [f"/1/parts/0/content/{i}" for i in (1, 2, 3)],
                "reason": "Hide the extra historical camera",
            }
    elif mutation == "omit":
        content.pop()
    elif mutation == "replace":
        content[-1] = binary(pixels(4))
    elif mutation == "reorder":
        content[-2], content[-1] = content[-1], content[-2]
    else:
        projected["image_selection"] = {"paths": [], "reason": "Drop current evidence"}
    projected.update(instructions="Synthetic evidence choice", tools=[])
    _, paths = prepare_native(projected, tmp_path, RequestLimits(max_images=10))
    expected = {
        "append": 4,
        "hide_extra": 3,
        "omit": 2,
        "replace": 3,
        "reorder": 3,
        "select": 0,
    }
    assert len(paths) == expected[mutation]


@pytest.mark.parametrize("internal_reads", [False, True])
def test_native_long_workflow_retains_one_session_and_three_images_per_turn(
    tmp_path, internal_reads
):
    rpc = SimulatedEnvironmentRPC(limit=80, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    value = baseline().model_dump(mode="json")
    value["context"] = baseline_context().model_dump(mode="json")
    value["runtime"]["tool_api"] = toolkit.get_tools_contract_spec()
    value["workflows"] = [
        WorkflowSpec(
            workflow_id="stationary_interval",
            description="Synthetic 30-action feedback burst",
            applicability="Only the simulated integration boundary",
            input_schema={"type": "object", "additionalProperties": False},
            output_schema={"type": "object"},
            permissions=("set_gripper", "view_env_state"),
            observations=(),
            contract_cases=(
                {
                    "arguments": {},
                    "observation": {},
                    "responses": [
                        {
                            "operation": "set_gripper",
                            "arguments": {"arm": "left", "gripper": 1},
                            "result": {"completed": True},
                        }
                        for _ in range(30)
                    ],
                    "expected_status": "completed",
                },
            ),
            limits=ProgramLimits(
                runtime_calls=65, physical_calls=30, model_calls=0, wall_time_s=30
            ),
            program=PythonProgram(
                entrypoint="stationary:run",
                files=(
                    PythonFile(
                        path="stationary.py",
                        source="""def run(runtime, context):
    for _ in range(30):
        runtime.act("set_gripper", {"arm": "left", "gripper": 1})
    return {"status": "completed", "summary": "Synthetic interval settled", "output": {"actions": 30}}
""",
                    ),
                ),
            ),
        ).model_dump(mode="json")
    ]
    if internal_reads:
        source = value["workflows"][0]["program"]["files"][0]["source"]
        value["workflows"][0]["program"]["files"][0]["source"] = source.replace(
            "        runtime.act(",
            '        runtime.observe("view_env_state", {})\n        runtime.act(',
        )
    sessions = []

    class Session(FakeSession):
        async def decide(self, text, images, output_schema, request_id):
            self.inputs.append((text, images, request_id))
            return {
                "text": "Synthetic test; no physical success claim",
                "tool_calls": [
                    {
                        "name": "workflow_stationary_interval"
                        if len(self.inputs) == 1
                        else "finish",
                        "arguments_json": '{"arguments": {}}'
                        if len(self.inputs) == 1
                        else '{"status": "stuck", "summary": "Synthetic test ended"}',
                    }
                ],
            }

    def factory(directory):
        session = Session(directory)
        sessions.append(session)
        return session

    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory)
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            wall_time_s=60,
            program_calls=100,
            gateway_calls=100,
            request_limits=RequestLimits(max_images=3),
        ),
    )
    try:
        result = planner.solve(
            system_prompt="Synthetic task",
            user_message="Exercise feedback",
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None, result.error
        assert rpc.actions == 30
        (session,) = sessions
        assert session.starts == 1 and len(session.inputs) == 2
        assert rpc.observation["instruction"] in (session.cwd / "AGENTS.md").read_text()
        assert all(len(images) == 3 for _, images, _ in session.inputs)
        assert "Synthetic interval settled" in session.inputs[-1][0]
        if internal_reads:
            events = [
                json.loads(line)
                for line in (tmp_path / "harness_context/bundle_events.jsonl")
                .read_text()
                .splitlines()
            ]
            reads = [
                event
                for event in events
                if event["kind"] == "operation_settled"
                and event["operation"] == "view_env_state"
            ]
            assert len(reads) == 30
            assert all(event["feedback_source"] == "program" for event in reads)
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "repair",
    [
        "native_edit",
        "static_feedback",
        "never_fix",
        "deployment_limit",
        "raw_trace",
        "raw_traceback",
        "expanded_gg",
    ],
)
def test_native_input_rejection_corrects_in_same_session_without_executing_proposals(
    tmp_path, repair
):
    toolkit, rpc, _ = create_simulated_toolkit(tmp_path)
    value = baseline().model_dump(mode="json")
    value["context"] = baseline_context().model_dump(mode="json")
    valid_source = value["context"]["program"]["files"][0]["source"]
    invalid_source = """def build(runtime, context):
    import json
    messages = json.loads(json.dumps(context["messages"]))
    content = messages[-1]["parts"][0]["content"]
    if not context.get("context_rejection") or REPAIR_MODE == "never_fix":
        content.extend(EXTRA_IMAGES)
    return {"instructions": context["base_instructions"], "messages": messages,
            "tools": context["tools"], "state": {}}
""".replace("REPAIR_MODE", repr(repair)).replace(
        "EXTRA_IMAGES", repr([binary(pixels(i + 50)) for i in range(8)])
    )
    audit_payloads = {
        "raw_trace": {
            "event_type": "control_step",
            "control_step_index": 1,
            "execution_ref": "x",
            "action": {"joints": [0] * 14},
            "terminated": False,
            "truncated": False,
        },
        "raw_traceback": {
            "error": "SAM unavailable",
            "traceback": "full stack must stay in audit",
        },
        "expanded_gg": {
            "provenance": {
                "support_rejections": [{"candidate_id": i} for i in range(4)]
            }
        },
    }
    if repair in audit_payloads:
        invalid_source = """def build(runtime, context):
    import json
    messages = json.loads(json.dumps(context["messages"]))
    if not context.get("context_rejection"):
        messages.append({"kind": "request", "parts": [{"part_kind": "user-prompt", "content": PAYLOAD}]})
    return {"instructions": context["base_instructions"], "messages": messages,
            "tools": context["tools"], "state": {}}
""".replace("PAYLOAD", repr(json.dumps(audit_payloads[repair])))
    value["context"]["program"]["files"][0]["source"] = (
        valid_source if repair == "deployment_limit" else invalid_source
    )
    sessions = []

    class Session(FakeSession):
        async def decide(self, text, images, output_schema, request_id):
            self.inputs.append((text, images, request_id))
            if not images:
                assert "model_input_rejected" in text
                assert "context_correction" not in text
                if repair in {"native_edit", "deployment_limit", *audit_payloads}:
                    source = valid_source
                    if repair == "deployment_limit":
                        source = source.replace(
                            '    assets = context["assets"]',
                            '    current_index = len(context["messages"]) - 1\n    assets = context["assets"]',
                        )
                        source = source.replace(
                            '"tools": context["tools"]}',
                            '"tools": context["tools"], "image_selection": {"paths": [f"/{current_index}/parts/0/content/1"], "reason": "Relevant head view only"}}',
                        )
                    (
                        self.cwd / "harness/task-family/context/src/context.py"
                    ).write_text(source)
                return {
                    "text": "Correct C before retrying; synthetic recovery only.",
                    "tool_calls": [
                        {
                            "name": "set_gripper",
                            "arguments_json": '{"arm":"left","gripper":1}',
                        }
                    ],
                }
            return {
                "text": "Synthetic test ended",
                "tool_calls": [
                    {
                        "name": "finish",
                        "arguments_json": '{"status":"stuck","summary":"Synthetic context recovery"}',
                    }
                ],
            }

    def factory(directory):
        session = Session(directory)
        sessions.append(session)
        return session

    planner = NativeBundlePlanner(
        output_dir=tmp_path,
        session_factory=factory,
        adaptation_mode="static" if repair == "static_feedback" else "adaptive",
    )
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            wall_time_s=30,
            program_calls=30,
            gateway_calls=30,
            request_limits=RequestLimits(
                max_images=2 if repair == "deployment_limit" else 50
            ),
        ),
    )
    try:
        result = planner.solve(
            system_prompt="Synthetic task",
            user_message="Inspect",
            toolkit=toolkit,
            max_turns=2,
        )
        assert rpc.actions == 0
        (session,) = sessions
        if repair == "static_feedback":
            assert session.starts == 0 and not session.inputs
            assert result.error is not None and result.stats["visible_turns"] == 0
            events = [
                json.loads(line)
                for line in (tmp_path / "harness_context/bundle_events.jsonl")
                .read_text()
                .splitlines()
            ]
            assert any(event["kind"] == "context_rejected" for event in events)
            assert not any(event["kind"] == "context_correction" for event in events)
            return
        expected_turns = 1 if repair == "never_fix" else 2
        assert session.starts == 1 and len(session.inputs) == expected_turns
        assert not session.inputs[0][1]
        assert result.stats["visible_turns"] == expected_turns
        if repair == "never_fix":
            assert result.error is not None
            assert all(not images for _, images, _ in session.inputs)
        else:
            assert result.error is None, result.error
            assert len(session.inputs[1][1]) == (
                1 if repair == "deployment_limit" else 3
            )
        events = [
            json.loads(line)
            for line in (tmp_path / "harness_context/bundle_events.jsonl")
            .read_text()
            .splitlines()
        ]
        rejected = [
            event for event in events if event["kind"] == "model_input_rejected"
        ]
        expected_error = {
            "raw_trace": "execution audit record",
            "raw_traceback": "exception traceback",
            "expanded_gg": "expanded GraspGen",
        }.get(repair, "3 > 2" if repair == "deployment_limit" else "11 > 10")
        assert rejected and expected_error in rejected[0]["error"]
        corrected = [event for event in events if event["kind"] == "context_correction"]
        assert corrected and len(corrected[0]["discarded_proposals"]) == 1
        assert any(
            event["kind"] == "context_correction_stalled" for event in events
        ) == (repair == "never_fix")
        assert len({event["program_call_id"] for event in corrected}) == 1
        if repair != "deployment_limit":
            raw = next(event for event in events if event["kind"] == "context_rejected")
            # Large payloads may be externalized in the audit, never silently clipped.
            from reweave.evolution.evidence.records import restore_record

            raw = restore_record(raw, tmp_path / "harness_context")
            assert len(image_inventory(raw["output"]["messages"])) == (
                3 if repair in audit_payloads else 11
            )
    finally:
        toolkit.close()


@pytest.mark.parametrize("outcome,known", [("completed", True), ("unknown", False)])
def test_concise_feedback_keeps_settlement_and_motion_errors(outcome, known):
    raw = {
        "step": 31,
        "state": {"cameras": {"calibration": "archived"}, "state": {"joints": [1]}},
        "artifacts": ["camera.png"],
        "log": {"duplicate_result": "archived"},
        "active_execution_outcome": outcome,
        "capability_result": {"reached": False, "position_error_m": 0.03},
        "error": "tracking incomplete",
        "interrupted": True,
    }
    original = copy.deepcopy(raw)
    projected = feedback_text(raw)
    assert projected["outcome_known"] is known
    assert projected["capability_result"] == raw["capability_result"]
    assert projected["error"] == raw["error"]
    assert projected["interrupted"]
    assert "state" not in projected and "log" not in projected
    assert "artifacts" not in projected
    assert raw == original


@pytest.mark.parametrize("operation", ["read_image", "view_env_state"])
@pytest.mark.parametrize("envelope", ["tool_return", "executed", "public_feedback"])
def test_explicit_new_image_read_survives_incremental_feedback(operation, envelope):
    old = binary(pixels(8))
    if envelope == "tool_return":
        parts = [
            {
                "part_kind": "tool-return",
                "tool_name": operation,
                "content": "Recorded step 0",
            },
            {"part_kind": "user-prompt", "content": [old]},
        ]
    else:
        text = (
            {"executed": {"name": operation}, "result": {"step": 0}}
            if envelope == "executed"
            else {
                "history_index": 0,
                "feedback": {"operation": operation, "result": {"step": 0}},
            }
        )
        parts = [{"part_kind": "user-prompt", "content": [json.dumps(text), old]}]
    source = {
        "messages": [{"kind": "request", "parts": parts}],
        "base_instructions": "Task",
    }
    original = copy.deepcopy(source)
    projected = PlannerObservation().prepare(
        source, {"step": 1}, [pixels(v) for v in (1, 2, 3)]
    )
    assert len(image_inventory(projected["messages"])) == 4
    assert projected["messages"][0]["parts"][-1]["content"][-1] == old
    assert source == original


def test_explicit_images_still_hit_complete_c_ceiling_without_silent_truncation(
    tmp_path,
):
    from reweave.runtime.capability import ModelInputRejected

    source = {
        "base_instructions": "Task",
        "assets": {"instructions": "Task guidance", "skills": []},
        "tools": [],
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            json.dumps({"executed": {"name": "read_image"}}),
                            *[binary(pixels(10 + i)) for i in range(8)],
                        ],
                    }
                ],
            }
        ],
    }
    projected = PlannerObservation().prepare(
        source, {"step": 1}, [pixels(v) for v in (1, 2, 3)]
    )
    prepared = build(projected)
    assert len(image_inventory(prepared["messages"])) == 11
    with pytest.raises(ModelInputRejected, match="11 > 10"):
        prepare_native(prepared, tmp_path, RequestLimits(max_images=10))


def test_native_explicit_historical_read_image_delivers_old_one_plus_current_three(
    tmp_path,
):
    rpc = SimulatedEnvironmentRPC(limit=10, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    # Change only the synthetic sensor before a real fake ACK; the retained old
    # observation remains immutable and differs from the current head frame.
    rpc.observation["vision"]["cam_head"]["color"][:] = 25
    toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
    old_png = toolkit.state.load_bytes("cam_head.png", step=0)
    sessions = []

    class Session(FakeSession):
        async def decide(self, text, images, output_schema, request_id):
            self.inputs.append((text, images, request_id))
            return {
                "text": "Synthetic image selection",
                "tool_calls": [
                    {
                        "name": "read_image" if len(self.inputs) == 1 else "finish",
                        "arguments_json": '{"name":"cam_head.png","step":0}'
                        if len(self.inputs) == 1
                        else '{"status":"stuck","summary":"Synthetic explicit image verified"}',
                    }
                ],
            }

    def factory(directory):
        session = Session(directory)
        sessions.append(session)
        return session

    value = baseline().model_dump(mode="json")
    value["context"] = baseline_context().model_dump(mode="json")
    value["runtime"]["tool_api"] = toolkit.get_tools_contract_spec()
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory)
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            wall_time_s=60,
            program_calls=30,
            gateway_calls=30,
            request_limits=RequestLimits(max_images=10),
        ),
    )
    try:
        result = planner.solve(
            system_prompt="Synthetic task",
            user_message="Inspect old head explicitly",
            toolkit=toolkit,
            max_turns=2,
        )
        assert result.error is None, result.error
        (session,) = sessions
        assert session.starts == 1 and len(session.inputs) == 2
        first = [path.read_bytes() for path in session.inputs[0][1]]
        second = [path.read_bytes() for path in session.inputs[1][1]]
        assert len(first) == 3 and old_png not in first
        assert len(second) == 4 and old_png in second
        assert all(current in second for current in first)
        assert rpc.actions == 1
    finally:
        toolkit.close()


def test_default_grasp_feedback_summarizes_rejections_and_keeps_full_original():
    result = {
        "candidates": [{"candidate_id": "accepted", "position_m": [1, 2, 3]}],
        "provenance": {
            "candidate_pool_checked": 417,
            "support_rejected_candidates": 416,
            "support_rejections": [
                {
                    "candidate_id": index,
                    "position_m": [index, 0, 0],
                    "rejection_reasons": ["support_collision"]
                    if index < 400
                    else ["support_collision", "target_absent"],
                }
                for index in range(416)
            ],
        },
    }
    original = copy.deepcopy(result)
    projected = feedback_text(
        json.dumps({"feedback": {"operation": "generate_grasps", "result": result}})
    )
    summary = json.loads(projected)["feedback"]["result"]
    provenance = summary["provenance"]
    assert len(provenance["support_rejections"]) == 3
    assert provenance["support_rejections_summary"] == {
        "total_records": 416,
        "displayed_records": 3,
        "omitted_records": 413,
        "reason_counts": {"support_collision": 416, "target_absent": 16},
        "truncated": True,
        "selection": "First distinct reason combinations, then first remaining records; not a statistical sample.",
        "complete_original": "Host original tool result and execution audit retain every rejection.",
    }
    assert provenance["support_rejected_candidates"] == 416
    assert provenance["candidate_pool_checked"] == 417
    assert summary["candidates"] == original["candidates"]
    assert result == original and len(result["provenance"]["support_rejections"]) == 416
    assert len(projected) < len(json.dumps(result)) / 10


def test_explicit_state_read_keeps_selected_reference_and_concise_log():
    raw = {
        "step": 4,
        "state": {"state": {"pose": [1, 2, 3]}, "cameras": "archive"},
        "observation_ref": "exact-selected-ref",
        "artifacts": ["head.png"],
        "log": {
            "command": {"action": "move_pose"},
            "result": {
                "error": "known planning failure",
                "traceback": "full private stack",
            },
            "elapsed_s": 0.5,
        },
    }
    selected = feedback_text(raw, preserve_images=True)
    assert selected["observation_ref"] == "exact-selected-ref"
    assert selected["log"]["result"] == {"error": "known planning failure"}
    assert selected["state"] == {"pose": [1, 2, 3]}
    assert "log" not in feedback_text(raw)
    assert raw["log"]["result"]["traceback"] == "full private stack"


def test_final_model_input_must_retain_the_verbatim_official_instruction(tmp_path):
    from reweave.runtime.capability import ModelInputRejected

    rpc = SimulatedEnvironmentRPC(limit=10, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        toolkit.execute_tool("view_env_state", {"step": 0})
        official = toolkit._state.latest_record().extras["official_task_instruction"]
        kept = {
            "instructions": f"Rules.\nOfficial instruction:\n{official}",
            "messages": [],
        }
        toolkit.validate_model_context(kept)
        serialized = {
            "instructions": "Rules",
            "messages": [
                {"parts": [{"content": [json.dumps({"task_language": official})]}]}
            ],
        }
        toolkit.validate_model_context(serialized)
        with pytest.raises(ModelInputRejected, match="official RoboDojo task"):
            toolkit.validate_model_context({"instructions": "Rules", "messages": []})
    finally:
        toolkit.close()
