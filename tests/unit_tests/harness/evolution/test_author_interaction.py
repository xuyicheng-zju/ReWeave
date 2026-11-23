# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Author assumptions are checked at actual production plugin entrypoints."""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import AgentBundle, ProgramLimits
from reweave.evolution.validation.interaction import (
    InteractionFailure,
    InteractionScenario,
    simulate_interaction,
)
from reweave.runtime.sandbox import ProgramSandbox
from tests.unit_tests.harness.evolution.fixtures import baseline, workflow

VIEW_SCHEMA = {
    "type": "object",
    "properties": {"step": {"type": "integer"}},
    "required": ["step"],
    "additionalProperties": False,
}
MOVE_SCHEMA = {
    "type": "object",
    "properties": {"distance": {"type": "number"}},
    "required": ["distance"],
    "additionalProperties": False,
}


def contract():
    return {
        "operations": {
            "view_env_state": {"effect": "read", "input_schema": VIEW_SCHEMA},
            "move": {"effect": "physical", "input_schema": MOVE_SCHEMA},
        },
        "tools": [
            {
                "name": "view_env_state",
                "description": "Read public task and scene",
                "input_schema": VIEW_SCHEMA,
            },
            {
                "name": "move",
                "description": "Move robot by measured distance",
                "input_schema": MOVE_SCHEMA,
            },
        ],
        "episode_limits": {"program_calls": 30, "gateway_calls": 30, "wall_time_s": 20},
    }


@pytest.mark.parametrize(
    "destination",
    [
        "draft",
        "published",
        "outside",
        "private_root",
        "unscoped",
        "not_markdown",
        "blank",
    ],
)
def test_captured_note_write_catalog_and_fixture_use_the_production_route(
    tmp_path, destination
):
    from reweave.memory.corpus import ReadOnlyCorpus

    directory = tmp_path / "interaction"
    path = {
        "draft": directory / "planner_memory/drafts/task-family/working.md",
        "published": directory / "published/forbidden.md",
        "outside": tmp_path / "outside.md",
        "private_root": directory / "planner_memory/forbidden.md",
        "unscoped": directory / "planner_memory/drafts/forbidden.md",
        "not_markdown": directory / "planner_memory/drafts/task-family/forbidden.json",
        "blank": directory / "planner_memory/drafts/task-family/blank.md",
    }[destination]
    spec, _ = ReadOnlyCorpus(tmp_path / "published").get_common_tool_bindings()[
        "write_text_file"
    ]
    gateway = contract()
    gateway["episode_limits"]["note_closeout_s"] = 1
    gateway["tools"].append(spec)
    gateway["operations"][spec["name"]] = {
        "effect": "write",
        "input_schema": spec["input_schema"],
    }
    arguments = {"path": str(path), "content": "Synthetic failed-approach note."}
    if destination == "blank":
        arguments["content"] = " "
    case = {
        "name": "private-note-before-terminal",
        "system_prompt": "Record failed approaches honestly.",
        "task_message": "Inspect the scene.",
        "model_responses": [
            {"tool_calls": [{"name": "write_text_file", "arguments": arguments}]},
            {"tool_calls": [{"name": "view_env_state", "arguments": {"step": 1}}]},
        ],
        "operations": [
            {
                "name": "write_text_file",
                "arguments": arguments,
                "result": {"written": True},
            },
            {
                "name": "view_env_state",
                "arguments": {"step": 1},
                "result": {"step": 1},
                "environment_state": {
                    "official_success": False,
                    "terminated": False,
                    "truncated": True,
                },
            },
        ],
        "assertions": [{"path": ["calls", 0, "name"], "equals": "write_text_file"}],
    }
    if destination != "draft":
        with pytest.raises(InteractionFailure) as failure:
            simulate_interaction(baseline(), gateway, case, directory)
        writes = [
            row
            for row in failure.value.report["calls"]
            if row["name"] == "write_text_file"
        ]
        assert writes[0]["result"]["error_type"] == (
            "PermissionError"
            if destination in {"published", "outside", "private_root"}
            else "ValueError"
        )
        assert not any(
            row["name"] == "write_text_file"
            for row in failure.value.report["fixture_calls"]
        )
        assert not path.exists()
        return
    report = simulate_interaction(baseline(), gateway, case, directory)
    assert report["validation_status"] == "passed"
    assert report["planner_tool_exposure_verified"]
    assert (
        report["runtime_contract"]["operations"]["write_text_file"]["effect"] == "write"
    )
    assert [row["name"] for row in report["fixture_calls"]] == [
        "write_text_file",
        "view_env_state",
    ]
    assert path.read_text() == arguments["content"]
    assert "host_note_closeout" not in report
    assert len(list((directory / "planner_memory/revisions").glob("*.json"))) == 1
    del gateway["tools"][-1]
    with pytest.raises(InteractionFailure, match="complete public gateway catalog"):
        simulate_interaction(baseline(), gateway, case, tmp_path / "incomplete")


def test_deployment_closeout_has_a_separate_host_fixture_and_keeps_authored_queue_strict(
    tmp_path,
):
    from reweave.memory.corpus import ReadOnlyCorpus

    gateway = contract()
    gateway["episode_limits"]["note_closeout_s"] = 1
    spec, _ = ReadOnlyCorpus(tmp_path / "published").get_common_tool_bindings()[
        "write_text_file"
    ]
    gateway["tools"].append(spec)
    gateway["operations"]["write_text_file"] = {
        "effect": "write",
        "input_schema": spec["input_schema"],
    }
    case = {
        "name": "terminal-with-host-closeout",
        "system_prompt": "Inspect public evidence.",
        "task_message": "Inspect the scene.",
        "model_responses": [
            {"tool_calls": [{"name": "view_env_state", "arguments": {"step": 1}}]}
        ],
        "operations": [
            {
                "name": "view_env_state",
                "arguments": {"step": 1},
                "result": {"step": 1},
                "environment_state": {
                    "official_success": False,
                    "terminated": False,
                    "truncated": True,
                },
            }
        ],
        "assertions": [{"path": ["calls", 0, "name"], "equals": "view_env_state"}],
    }
    directory = tmp_path / "with-closeout"
    report = simulate_interaction(baseline(), gateway, case, directory)
    assert len(report["model_requests"]) == 2
    assert report["native_turns"][-1]["controller_kind"] == "note_closeout"
    assert report["model_requests"][-1]["tools"] == []
    assert len(report["fixture_calls"]) == 1
    note = directory / "planner_memory/drafts/task-family/episode-closeout.md"
    assert note.read_text() == report["host_note_closeout"]["model_response"]
    assert report["host_note_closeout"]["write"]["verification"] == "unverified"
    assert len(list((directory / "planner_memory/revisions").glob("*.json"))) == 1
    extra = deepcopy(case)
    extra["model_responses"].append({"text": "Unused author response"})
    with pytest.raises(InteractionFailure, match="remaining_model_responses"):
        simulate_interaction(baseline(), gateway, extra, tmp_path / "extra")
    missing = deepcopy(case)
    missing["model_responses"] = []
    with pytest.raises(InteractionFailure, match="unexpected model request"):
        simulate_interaction(baseline(), gateway, missing, tmp_path / "missing")


def candidate(source, *, loop=None, context=None):
    value = baseline().model_dump(mode="json")
    item = workflow().model_dump(mode="json")
    item["permissions"] = ["move", "view_env_state"]
    item["observations"] = ["view_env_state"]
    item["program"]["files"][0]["source"] = source
    value["workflows"] = [item]
    if loop is not None:
        value["loop"]["program"]["files"][0]["source"] = loop
    if context is not None:
        value["context"]["program"]["files"][0]["source"] = context
    return AgentBundle.model_validate(value)


OLD_GATE = """def run(runtime, context):
    if context["observation"].get("task_language") != "Open drawer":
        return {"status": "failed", "summary": "inapplicable", "output": {"reason": "inapplicable_task"}}
    return {"status": "completed", "summary": "ready", "output": {"reason": "ready"}}
"""
FIXED_GATE = """def run(runtime, context):
    fresh = runtime.observe("view_env_state", {"step": -1})
    if fresh["task_language"] != "Open drawer":
        return {"status": "failed", "summary": "inapplicable", "output": {"reason": "inapplicable_task"}}
    moved = runtime.act("move", {"distance": 0.1})
    return {"status": "completed", "summary": "checked move", "output": {"reason": "ready", "move": moved}}
"""


def scenario(*, fixed=False):
    operations = [
        {
            "name": "view_env_state",
            "arguments": {"step": 0},
            "result": {"task_language": "Open drawer", "step": 0},
        }
    ]
    if fixed:
        operations.extend(
            [
                {
                    "name": "view_env_state",
                    "arguments": {"step": -1},
                    "result": {"task_language": "Open drawer", "step": 0},
                },
                {
                    "name": "move",
                    "arguments": {"distance": 0.1},
                    "result": {"step": 1, "target_reached": True},
                },
            ]
        )
    operations.append(
        {
            "name": "move",
            "arguments": {"distance": 0.1},
            "result": {"step": 2},
            "environment_state": {
                "official_success": False,
                "terminated": False,
                "truncated": True,
            },
        }
    )
    return {
        "name": "open-from-read-only-entry",
        "system_prompt": "Use public tools.",
        "task_message": "Open drawer",
        "model_responses": [
            {
                "tool_calls": [{"name": "view_env_state", "arguments": {"step": 0}}],
                "required_tools": ["workflow_probe"],
            },
            {
                "tool_calls": [
                    {"name": "workflow_probe", "arguments": {"arguments": {}}}
                ]
            },
            {
                "tool_calls": [
                    {
                        "name": "move",
                        "arguments": {"distance": 0.1},
                    }
                ]
            },
        ],
        "operations": operations,
        "assertions": [
            {"path": ["workflows", 0, "result", "output", "reason"], "equals": "ready"}
        ],
    }


def test_prefilled_unit_fixture_passes_but_real_entry_rejects_old_gate(tmp_path):
    bundle = candidate(OLD_GATE)

    async def forbidden(*args):
        raise AssertionError("gate fixture should not touch runtime")

    isolated = asyncio.run(
        ProgramSandbox().run_program(
            bundle.workflows[0].program,
            ProgramLimits(
                runtime_calls=4, physical_calls=2, model_calls=0, wall_time_s=10
            ),
            {"observation": {"task_language": "Open drawer"}},
            forbidden,
            input_schema={"type": "object"},
        )
    )
    assert isolated["status"] == "completed"
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(bundle, contract(), scenario(), tmp_path / "old")
    report = failure.value.report
    assert report["workflows"][0]["result"]["output"]["reason"] == "inapplicable_task"
    assert report["assertion_results"][0]["validation_status"] == "failed"
    assert len(report["calls"]) == 2
    second_context = [
        event for event in report["events"] if event["kind"] == "context_built"
    ][1]["input"]
    assert second_context["observation"] == {}
    assert (
        second_context["last_tool_result"]["result"]["task_language"] == "Open drawer"
    )
    assert second_context["task_language"] == "Open drawer"
    assert (tmp_path / "old/interaction.json").is_file()


def test_w_only_repair_executes_through_real_model_context_and_physical_gateway(
    tmp_path,
):
    report = simulate_interaction(
        candidate(FIXED_GATE), contract(), scenario(fixed=True), tmp_path / "fixed"
    )
    assert report["validation_status"] == "passed"
    assert report["physical_execution_tested"] is False
    assert report["planner_tool_exposure_verified"] is True
    assert report["coverage"] == {"C": True, "L": True, "workflows": ["probe"]}
    assert len(report["native_turns"]) == len(report["model_requests"]) == 3
    third = [event for event in report["events"] if event["kind"] == "context_built"][
        2
    ]["input"]
    assert third["observation"] == {"step": 1, "target_reached": True}
    assert [record["name"] for record in report["calls"]] == [
        "view_env_state",
        "view_env_state",
        "move",
        "move",
    ]


def test_modified_c_tools_cannot_pass_even_with_valid_standalone_return(tmp_path):
    invalid = """def build(runtime, context):
    tools = context["tools"]
    tools[0]["input_schema"] = {"type": "object"}
    return {"instructions": "tampered", "messages": context["messages"], "tools": tools}
"""
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, context=invalid), contract(), scenario(), tmp_path / "c"
        )
    report = failure.value.report
    assert "changed a tool interface" in report["result"]["error"]
    assert report["model_requests"] == []
    assert report["coverage"]["C"] is False


def test_actual_l_routes_c_and_w_without_a_host_schedule(tmp_path):
    loop = """def run(runtime, context):
    runtime.observe("view_env_state", {"step": 0})
    proposal = runtime.plan("Choose the installed workflow")
    runtime.execute(proposal["tool_calls"][0])
    recovery = runtime.plan("Review the workflow result and choose the next move")
    runtime.execute(recovery["tool_calls"][0])
    return {"summary": "finished"}
"""
    context = """def build(runtime, context):
    return {"instructions": "CUSTOM context", "messages": context["messages"], "tools": context["tools"]}
"""
    case = scenario(fixed=True)
    case["model_responses"] = [
        {
            "tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}],
            "instructions_contains": ["CUSTOM context"],
            "required_tools": ["workflow_probe"],
        },
        {
            "tool_calls": [
                {
                    "name": "move",
                    "arguments": {"distance": 0.1},
                }
            ]
        },
    ]
    report = simulate_interaction(
        candidate(FIXED_GATE, loop=loop, context=context),
        contract(),
        case,
        tmp_path / "all",
    )
    assert report["coverage"] == {"C": True, "L": True, "workflows": ["probe"]}
    assert len(report["native_turns"]) == 2
    assert "Choose the installed workflow" in str(
        report["model_requests"][0]["messages"]
    )


@pytest.mark.parametrize("extra", ["model_responses", "operations"])
def test_unconsumed_fixtures_are_explicit_failures(tmp_path, extra):
    case = scenario(fixed=True)
    case[extra].append(deepcopy(case[extra][-1]))
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(candidate(FIXED_GATE), contract(), case, tmp_path / extra)
    assert failure.value.report["remaining_" + extra]
    assert any(
        "unconsumed fixtures" in error for error in failure.value.report["errors"]
    )


def test_catching_gateway_fixture_error_cannot_turn_validation_green(tmp_path):
    loop = """def run(runtime, context):
    try:
        runtime.observe("view_env_state", {"step": 99})
    except Exception:
        pass
    return {"done": True}
"""
    case = scenario()
    case["model_responses"] = []
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "caught"
        )
    report = failure.value.report
    assert (
        "root Loop returned while the episode was nonterminal"
        in report["result"]["error"]
    )
    assert any("gateway fixture 0 mismatch" in error for error in report["errors"])
    assert report["assertion_results"][0]["validation_status"] == "failed"


def test_undeclared_extra_model_request_is_not_silently_accepted(tmp_path):
    case = scenario()
    case["model_responses"] = []
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE), contract(), case, tmp_path / "extra-request"
        )
    assert len(failure.value.report["model_requests"]) == 1
    assert any(
        "unexpected model request" in error for error in failure.value.report["errors"]
    )


def test_read_image_remains_exposed_but_cannot_read_host_files(tmp_path):
    image_spec = {
        "name": "read_image",
        "description": "Read a recorded episode image by artifact name and step; this is a read-only gateway operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "step": {"type": "integer", "default": -1},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    gateway = contract()
    gateway["tools"].append(image_spec)
    gateway["operations"]["read_image"] = {
        "effect": "read",
        "input_schema": image_spec["input_schema"],
    }
    loop = """def run(runtime, context):
    runtime.plan()
    try:
        runtime.observe("read_image", {"name": "private.png"})
    except Exception:
        pass
    return {"done": True}
"""
    case = scenario()
    case["operations"] = []
    case["model_responses"] = [{"text": "Inspect", "required_tools": ["read_image"]}]
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), gateway, case, tmp_path / "image"
        )
    report = failure.value.report
    assert report["planner_tool_exposure_verified"] is True
    assert (
        "root Loop returned while the episode was nonterminal"
        in report["result"]["error"]
    )
    assert any(
        "read_image requested undeclared artifact step" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize("field", ["observation", "history", "last_tool_result"])
def test_scenario_cannot_inject_runtime_state(field):
    case = scenario()
    case[field] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        InteractionScenario.model_validate(case)


def test_scenario_requires_behavior_assertions():
    case = scenario()
    case["assertions"] = []
    with pytest.raises(ValidationError):
        InteractionScenario.model_validate(case)


def test_missing_real_tool_metadata_is_not_a_verified_exposure(tmp_path):
    gateway = contract()
    del gateway["tools"]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE), gateway, scenario(), tmp_path / "metadata"
        )
    assert failure.value.report["planner_tool_exposure_verified"] is False
    assert any(
        "captured fixed tools metadata" in error
        for error in failure.value.report["errors"]
    )


def test_parallel_simulations_have_independent_outputs_and_fixtures(tmp_path):
    def run(index):
        case = scenario(fixed=True)
        case["system_prompt"] = f"PRIVATE PROMPT {index}"
        case["model_responses"][0]["instructions_contains"] = [case["system_prompt"]]
        return simulate_interaction(
            candidate(FIXED_GATE), contract(), case, tmp_path / str(index)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run, [0, 1]))
    assert first["validation_status"] == second["validation_status"] == "passed"
    assert len(first["calls"]) == len(second["calls"]) == 4
    assert first["scenario"] != second["scenario"]
    assert "PRIVATE PROMPT 1" not in first["model_requests"][0]["instructions"]
    assert "PRIVATE PROMPT 0" not in second["model_requests"][0]["instructions"]


def test_interruption_never_writes_a_passing_receipt(tmp_path, monkeypatch):
    import json

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("author simulation interrupted")

    monkeypatch.setattr(
        "reweave.planner.native_loop.NativeBundlePlanner.solve", interrupted
    )
    with pytest.raises(KeyboardInterrupt):
        simulate_interaction(
            candidate(OLD_GATE), contract(), scenario(), tmp_path / "interrupt"
        )
    report = json.loads((tmp_path / "interrupt/interaction.json").read_text())
    assert report["validation_status"] == "failed"
    assert any("KeyboardInterrupt" in error for error in report["errors"])


def test_long_feedback_is_preserved_in_full(tmp_path):
    case = scenario(fixed=True)
    complete = "public feedback " * 5000
    case["operations"][-2]["result"]["detail"] = complete
    case["assertions"].append(
        {
            "path": ["workflows", 0, "result", "output", "move", "detail"],
            "equals": complete,
        }
    )
    report = simulate_interaction(
        candidate(FIXED_GATE), contract(), case, tmp_path / "full"
    )
    assert report["calls"][-2]["result"]["detail"] == complete
    assert complete in str(report["model_requests"][-1]["messages"])


def test_mismatched_captured_runtime_operation_is_rejected(tmp_path):
    gateway = contract()
    gateway["operations"]["model:plan"] = {
        "effect": "model",
        "input_schema": {"type": "object"},
    }
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(FIXED_GATE), gateway, scenario(fixed=True), tmp_path / "contract"
        )
    assert any(
        "actual runtime operation differs from captured contract: model:plan" in error
        for error in failure.value.report["errors"]
    )


def image_contract():
    gateway = contract()
    spec = {
        "name": "read_image",
        "description": "Read a recorded episode image by artifact name and step; this is a read-only gateway operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "step": {"type": "integer", "default": -1},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    gateway["tools"].append(spec)
    gateway["operations"]["read_image"] = {
        "effect": "read",
        "input_schema": spec["input_schema"],
    }
    return gateway


def image_fixture(*, step=0, name="camera.png"):
    import base64
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(stream, format="PNG")
    return {
        "name": name,
        "step": step,
        "media_type": "image/png",
        "data_base64": base64.b64encode(stream.getvalue()).decode(),
    }


def image_scenario(*, step=0):
    case = scenario()
    case["operations"] = [
        {
            "name": "move",
            "arguments": {"distance": 0.1},
            "result": {"step": step + 1},
            "environment_state": {
                "official_success": True,
                "terminated": False,
                "truncated": False,
            },
        }
    ]
    case["artifacts"] = [image_fixture(step=step)]
    case["initial_artifact_step"] = step
    case["model_responses"] = [
        {"tool_calls": [{"name": "read_image", "arguments": {"name": "camera.png"}}]},
        {
            "tool_calls": [
                {
                    "name": "move",
                    "arguments": {"distance": 0.1},
                }
            ]
        },
    ]
    case["assertions"] = [{"path": ["calls", 0, "result", "step"], "equals": step}]
    return case


def test_declared_image_reaches_real_reader_and_next_model_request(tmp_path):
    import base64
    from pathlib import Path

    case = image_scenario()
    report = simulate_interaction(
        baseline(), image_contract(), case, tmp_path / "read-image"
    )
    assert report["validation_status"] == "passed"
    assert report["remaining_artifacts"] == []
    assert report["calls"][0]["arguments"] == {"name": "camera.png"}
    assert report["artifact_reads"][0]["requested_step"] == -1
    assert (
        report["calls"][0]["result"]["_runtime_images"][0]["data_base64"]
        == case["artifacts"][0]["data_base64"]
    )
    assert [
        Path(path).read_bytes() for path in report["model_requests"][1]["image_files"]
    ] == [base64.b64decode(case["artifacts"][0]["data_base64"])]
    assert len(list((tmp_path / "read-image/artifacts").glob("image-*.png"))) == 1


@pytest.mark.parametrize("keep_images", [True, False])
def test_workflow_plan_receives_actual_image_content_not_only_a_filename(
    tmp_path, keep_images
):
    from tests.unit_tests.harness.evolution.test_plugin_execution import program

    source = """def run(runtime, context):
    runtime.observe("read_image", {"name": "camera.png"})
    answer = runtime.plan("Inspect the handle")
    return {"status": "completed", "summary": answer["text"], "output": {}}
"""
    value = candidate(source).model_dump(mode="json")
    if not keep_images:
        value["context"]["program"] = program(
            """def build(runtime, context):
    return {"instructions": context["base_instructions"], "messages": context["messages"], "tools": context["tools"], "image_selection": {"paths": [], "reason": "Negative fixture discards required visual evidence."}}
""",
            "context:build",
        ).model_dump(mode="json")
    value["workflows"][0]["permissions"] = ["model:plan", "read_image"]
    value["workflows"][0]["observations"] = ["read_image"]
    value["workflows"][0]["limits"]["model_calls"] = 1
    case = image_scenario()
    case["model_responses"] = [
        {"tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}]},
        {"text": "Red image received", "required_images": ["camera.png"]},
        case["model_responses"][-1],
    ]
    if not keep_images:
        with pytest.raises(InteractionFailure, match="lacks binary image"):
            simulate_interaction(
                AgentBundle.model_validate(value),
                image_contract(),
                case,
                tmp_path / "missing-image",
            )
        return
    report = simulate_interaction(
        AgentBundle.model_validate(value),
        image_contract(),
        case,
        tmp_path / "visual-reason",
    )
    import base64
    from pathlib import Path

    images = report["model_requests"][1]["image_files"]
    assert [Path(path).read_bytes() for path in images] == [
        base64.b64decode(case["artifacts"][0]["data_base64"])
    ]


def test_read_image_latest_step_follows_actual_public_gateway_result(tmp_path):
    case = image_scenario(step=7)
    case["initial_artifact_step"] = 0
    finish = case["operations"][-1]
    case["operations"] = [
        {"name": "view_env_state", "arguments": {"step": -1}, "result": {"step": 7}},
        finish,
    ]
    case["model_responses"].insert(
        0, {"tool_calls": [{"name": "view_env_state", "arguments": {"step": -1}}]}
    )
    case["assertions"] = [{"path": ["calls", 1, "result", "step"], "equals": 7}]
    report = simulate_interaction(
        baseline(), image_contract(), case, tmp_path / "latest-image"
    )
    assert report["artifact_reads"][0]["step"] == 7
    assert report["artifact_reads"][0]["requested_step"] == -1


def test_extra_declared_image_is_an_unconsumed_fixture(tmp_path):
    case = image_scenario()
    case["artifacts"].append(image_fixture(name="unused.png"))
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            baseline(), image_contract(), case, tmp_path / "unused-image"
        )
    assert failure.value.report["remaining_artifacts"] == [
        {"step": 0, "name": "unused.png"}
    ]


def test_image_read_cannot_resolve_an_external_file(tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"PRIVATE IMAGE MUST NOT BE READ")
    case = image_scenario()
    case["model_responses"][0]["tool_calls"][0]["arguments"]["name"] = str(secret)
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            baseline(), image_contract(), case, tmp_path / "private-image"
        )
    report = failure.value.report
    assert any("no external file was read" in error for error in report["errors"])
    assert "PRIVATE IMAGE MUST NOT BE READ" not in str(report)
    assert report["artifact_reads"] == []


@pytest.mark.parametrize("name", ["../private.png", "/private.png", "a\\private.png"])
def test_artifact_declaration_cannot_name_external_paths(name):
    case = image_scenario()
    case["artifacts"][0]["name"] = name
    with pytest.raises(ValidationError, match="without directory components"):
        InteractionScenario.model_validate(case)


def test_artifact_declared_encoding_is_verified(tmp_path):
    case = image_scenario()
    case["artifacts"][0]["media_type"] = "image/jpeg"
    case["artifacts"][0]["name"] = "camera.jpg"
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            baseline(), image_contract(), case, tmp_path / "bad-encoding"
        )
    assert any(
        "media_type disagree" in error for error in failure.value.report["errors"]
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"official_success": True, "terminated": False, "truncated": False},
        {"official_success": False, "terminated": False, "truncated": True},
        {"official_success": False, "terminated": True, "truncated": False},
    ],
)
def test_explicit_synthetic_environment_state_drives_real_terminal_check(
    tmp_path, environment
):
    loop = """def run(runtime, context):
    before = runtime.snapshot()["terminal"]
    call = runtime.plan("Confirm the move")["tool_calls"][0]
    runtime.execute(call)
    return {"before": before, "after": runtime.snapshot()["terminal"]}
"""
    case = scenario()
    case["model_responses"] = [
        {"tool_calls": [{"name": "move", "arguments": {"distance": 0.1}}]}
    ]
    case["operations"] = [
        {
            "name": "move",
            "arguments": {"distance": 0.1},
            "result": {"step": 1},
            "environment_state": environment,
        }
    ]
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    report = simulate_interaction(
        candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "terminal"
    )
    completion = next(
        event for event in report["events"] if event["kind"] == "program_completed"
    )
    assert completion["result"] == {"before": False, "after": True}
    assert report["synthetic_environment_state"] == environment
    assert report["physical_execution_tested"] is False


def test_terminal_is_not_guessed_from_arbitrary_tool_feedback(tmp_path):
    loop = """def run(runtime, context):
    call = runtime.plan("Confirm the move")["tool_calls"][0]
    runtime.execute(call)
    return {"terminal": runtime.snapshot()["terminal"]}
"""
    case = scenario()
    case["model_responses"] = [
        {"tool_calls": [{"name": "move", "arguments": {"distance": 0.1}}]}
    ]
    case["operations"] = [
        {
            "name": "move",
            "arguments": {"distance": 0.1},
            "result": {"success": True, "terminated": True},
        }
    ]
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "not-guessed"
        )
    assert (
        "root Loop returned while the episode was nonterminal"
        in failure.value.report["result"]["error"]
    )


def test_failed_workflow_feedback_reaches_replanning_and_recovery_until_horizon(
    tmp_path,
):
    case = scenario()
    case["recovery"] = True
    case["model_responses"] = [
        {
            "tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}],
        },
    ] + [
        {"tool_calls": [{"name": "move", "arguments": {"distance": 0.1}}]}
        for _ in range(3)
    ]
    case["operations"] = [
        {
            "name": "move",
            "arguments": {"distance": 0.1},
            # Tool-owned success/finish claims cannot terminate the environment.
            "result": {"step": step, "_finish": True, "status": "success"},
            "environment_state": {
                "official_success": False,
                "terminated": False,
                "truncated": step == 310,
            },
        }
        for step in (110, 210, 310)
    ]
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"},
        {"path": ["calls", 2, "result", "step"], "equals": 310},
    ]
    gateway = contract()
    gateway["episode_limits"]["program_calls"] = 100
    report = simulate_interaction(
        candidate(OLD_GATE), gateway, case, tmp_path / "recovery"
    )
    assert report["validation_status"] == "passed"
    assert report["synthetic_environment_state"] == {
        "official_success": False,
        "terminated": False,
        "truncated": True,
    }
    assert [call["name"] for call in report["calls"]] == ["move"] * 3
    assert len(report["model_requests"]) == 4
    assert "inapplicable_task" in str(report["model_requests"][1]["messages"])


@pytest.mark.parametrize("queued", [False, True])
def test_recovery_fixture_rejects_finish_and_replans_prequeued_action(tmp_path, queued):
    gateway = contract()
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["status", "summary"],
        "additionalProperties": False,
    }
    gateway["operations"]["finish"] = {"effect": "control", "input_schema": schema}
    gateway["tools"].append(
        {"name": "finish", "description": "Finish the task", "input_schema": schema}
    )
    case = scenario()
    case["recovery"] = True
    fail_call = {"name": "workflow_probe", "arguments": {"arguments": {}}}
    if queued:
        case["model_responses"] = [
            {
                "tool_calls": [
                    fail_call,
                    {"name": "move", "arguments": {"distance": 0.1}},
                ]
            },
            {"tool_calls": [{"name": "move", "arguments": {"distance": 0.1}}]},
        ]
        case["operations"] = [
            {
                "name": "move",
                "arguments": {"distance": 0.1},
                "result": {"step": 2},
                "environment_state": {
                    "official_success": False,
                    "terminated": False,
                    "truncated": True,
                },
            }
        ]
    else:
        case["model_responses"] = [
            {"tool_calls": [fail_call]},
            {
                "tool_calls": [
                    {
                        "name": "finish",
                        "arguments": {"status": "failure", "summary": "local failure"},
                    }
                ]
            },
        ]
        case["operations"] = [
            {
                "name": "finish",
                "arguments": {"status": "failure", "summary": "local failure"},
                "result": {
                    "_finish": True,
                    "status": "failure",
                    "summary": "local failure",
                },
            }
        ]
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"}
    ]
    if queued:
        report = simulate_interaction(
            candidate(OLD_GATE), gateway, case, tmp_path / "replanned-recovery"
        )
        assert report["validation_status"] == "passed"
        assert [call["name"] for call in report["calls"]] == ["move"]
        assert len(report["model_requests"]) == 2
        assert "program_feedback" in str(report["model_requests"][1]["messages"])
        return
    ordinary = deepcopy(case)
    ordinary["recovery"] = False
    report = simulate_interaction(
        candidate(OLD_GATE), gateway, ordinary, tmp_path / "ordinary-finish"
    )
    assert report["result"]["finish_result"]["_finish"] is True
    with pytest.raises(InteractionFailure, match="recovery fixture must execute"):
        simulate_interaction(
            candidate(OLD_GATE), gateway, case, tmp_path / "false-recovery"
        )


@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("scope", ["C", "L"])
def test_final_admission_requires_host_checked_recovery_for_changed_combination(
    tmp_path, monkeypatch, declared, scope
):
    from reweave.evolution.validation.admission import check_interaction_suite
    from reweave.utils.persistence import atomic_json
    from tests.unit_tests.harness.evolution.test_bundle_author import workspace

    root = workspace(tmp_path)
    atomic_json(root / "context/gateway.json", contract())
    parent = candidate(OLD_GATE)
    value = parent.model_dump(mode="json")
    if scope == "C":
        value["context"]["instructions"] += "\nInspect settled failure feedback."
    else:
        value["loop"]["program"]["files"][0]["source"] += (
            "\n# Changed orchestration under test.\n"
        )
    monkeypatch.setattr(
        "reweave.evolution.validation.admission.read_references", lambda _: (parent,)
    )
    assert AgentBundle.model_validate(value).novel_scopes((parent,)) == frozenset(
        {scope}
    )
    case = scenario()
    case["recovery"] = declared
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"}
    ]
    atomic_json(root / ".scratch/tests/interaction.json", [case])

    async def run():
        return await check_interaction_suite(
            root,
            AgentBundle.model_validate(value),
            scenario_path=".scratch/tests/interaction.json",
        )

    if declared:
        from reweave.evolution.validation.interaction import check_recovery

        report = asyncio.run(run())[0]
        assert report["validation_status"] == "passed"
        for event in report["events"]:
            if event["kind"] == "context_built":
                event["output"]["messages"] = []
        with pytest.raises(ValueError, match="dropped the failed Workflow result"):
            check_recovery(report, contract())
    else:
        with pytest.raises(ValueError, match="requires a recovery=true"):
            asyncio.run(run())


def test_loop_return_after_failed_workflow_is_rejected_before_rollout(tmp_path):
    loop = """def run(runtime, context):
    call = runtime.plan()["tool_calls"][0]
    runtime.execute(call)
    return {"summary": "Workflow failed, stop"}
"""
    case = scenario()
    case["model_responses"] = [
        {"tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}]}
    ]
    case["operations"] = []
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"}
    ]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "premature"
        )
    assert (
        "root Loop returned while the episode was nonterminal"
        in failure.value.report["result"]["error"]
    )


def test_root_loop_cannot_bypass_plan_for_physical_execution(tmp_path):
    loop = """def run(runtime, context):
    return runtime.act("move", {"distance": 0.1})
"""
    case = scenario()
    case["model_responses"] = []
    case["operations"] = []
    case["assertions"] = [{"path": ["calls"], "equals": []}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "direct-act"
        )
    report = failure.value.report
    assert (
        "physical execution requires a tool call returned by runtime.plan()"
        in report["result"]["error"]
    )
    assert report["calls"] == []


def test_root_loop_cannot_bypass_plan_for_workflow_execution(tmp_path):
    loop = """def run(runtime, context):
    return runtime.control("workflow:probe", {"arguments": {}})
"""
    case = scenario()
    case["model_responses"] = []
    case["operations"] = []
    case["assertions"] = [{"path": ["workflows"], "equals": []}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop),
            contract(),
            case,
            tmp_path / "direct-workflow",
        )
    report = failure.value.report
    assert "unchanged tool call returned by runtime.plan()" in report["result"]["error"]
    assert report["workflows"] == []


def test_historical_public_read_does_not_rewind_latest_image(tmp_path):
    case = image_scenario(step=7)
    finish = case["operations"][-1]
    case["operations"] = [
        {"name": "view_env_state", "arguments": {"step": 0}, "result": {"step": 0}},
        finish,
    ]
    case["model_responses"].insert(
        0, {"tool_calls": [{"name": "view_env_state", "arguments": {"step": 0}}]}
    )
    case["assertions"] = [{"path": ["calls", 1, "result", "step"], "equals": 7}]
    report = simulate_interaction(
        baseline(), image_contract(), case, tmp_path / "historical-image"
    )
    assert report["artifact_reads"][0]["step"] == 7


def test_future_image_cannot_be_read_before_its_step_is_published(tmp_path):
    case = image_scenario(step=7)
    case["initial_artifact_step"] = 0
    case["model_responses"][0]["tool_calls"][0]["arguments"]["step"] = 7
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            baseline(), image_contract(), case, tmp_path / "future-image"
        )
    assert any("future step 7" in error for error in failure.value.report["errors"])
    assert failure.value.report["artifact_reads"] == []


def test_caught_unknown_l_operation_is_still_an_interaction_failure(tmp_path):
    loop = """def run(runtime, context):
    try:
        runtime.observe("undeclared", {})
    except Exception:
        return {"handled": True}
"""
    case = scenario()
    case["model_responses"] = []
    case["operations"] = []
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, loop=loop), contract(), case, tmp_path / "l-rpc"
        )
    report = failure.value.report
    assert (
        "root Loop returned while the episode was nonterminal"
        in report["result"]["error"]
    )
    rejected = [
        event for event in report["events"] if event["kind"] == "program_rpc_failed"
    ]
    assert len(rejected) == 1
    assert rejected[0]["method"] == "observe"
    assert rejected[0]["name"] == "undeclared"
    assert rejected[0]["arguments"] == {}
    assert rejected[0]["error"] == "PermissionError"
    assert "operation exceeds caller authority" in rejected[0]["detail"]
    assert report["calls"] == []


def test_caught_w_permission_error_is_still_an_interaction_failure(tmp_path):
    source = """def run(runtime, context):
    try:
        runtime.observe("view_env_state", {"step": 0})
    except Exception:
        return {"status": "completed", "summary": "handled", "output": {"handled": True}}
"""
    value = candidate(source).model_dump(mode="json")
    value["workflows"][0]["permissions"] = []
    value["workflows"][0]["observations"] = []
    case = scenario()
    case["operations"] = [case["operations"][-1]]
    case["model_responses"].pop(0)
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "output", "handled"], "equals": True}
    ]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            AgentBundle.model_validate(value), contract(), case, tmp_path / "w-rpc"
        )
    report = failure.value.report
    assert report["result"]["error"] is None
    rejected = next(
        event for event in report["events"] if event["kind"] == "program_rpc_failed"
    )
    assert rejected["workflow_id"] == "probe"
    assert rejected["method"] == "observe"
    assert rejected["name"] == "view_env_state"
    assert rejected["arguments"] == {"step": 0}
    assert "exceeds declared permissions" in rejected["detail"]
    assert [call["name"] for call in report["calls"]] == ["move"]


def test_c_catching_forbidden_action_does_not_hide_rpc_rejection(tmp_path):
    source = """def build(runtime, context):
    try:
        runtime.act("move", {"distance": 0.1})
    except Exception:
        pass
    return {"instructions": "handled", "messages": context["messages"], "tools": context["tools"]}
"""
    case = scenario()
    case["operations"] = [case["operations"][-1]]
    case["model_responses"] = [case["model_responses"][-1]]
    case["assertions"] = [{"path": ["result", "error"], "equals": None}]
    with pytest.raises(InteractionFailure) as failure:
        simulate_interaction(
            candidate(OLD_GATE, context=source), contract(), case, tmp_path / "c-rpc"
        )
    report = failure.value.report
    assert report["result"]["error"] is None
    assert report["coverage"]["C"] is True
    rejected = next(
        event for event in report["events"] if event["kind"] == "program_rpc_failed"
    )
    assert rejected["method"] == "act"
    assert rejected["name"] == "move"
    assert rejected["arguments"] == {"distance": 0.1}
    assert "cannot perform actions" in rejected["detail"]
    assert [call["name"] for call in report["calls"]] == ["move"]


@pytest.mark.parametrize("kind", ["physical_unknown", "cancelled"])
def test_rpc_audit_failure_preserves_original_exception_priority(kind):
    from reweave.runtime.capability import PhysicalOutcomeUnknown
    from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits

    original = (
        PhysicalOutcomeUnknown("submitted but no acknowledgement")
        if kind == "physical_unknown"
        else asyncio.CancelledError("caller cancelled")
    )

    async def invoke(*args):
        raise original

    class DirectSandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            return await rpc("act", "move", {"distance": 0.1})

    def failing_evidence(event):
        assert event["kind"] == "program_rpc_failed"
        raise OSError("audit disk unavailable")

    async def cancel():
        pass

    kernel = EpisodeKernel(
        baseline(),
        operations={},
        gateway=invoke,
        cancel=cancel,
        emit=failing_evidence,
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
        sandbox=DirectSandbox(),
    )
    with pytest.raises(type(original)) as captured:
        asyncio.run(kernel.run_plugin(baseline().loop.program, {}, invoke))
    if kind == "cancelled" and sys.version_info < (3, 11):
        # asyncio.run() re-raises a new CancelledError on Python 3.10.
        assert isinstance(captured.value, asyncio.CancelledError)
    else:
        assert captured.value is original
    assert isinstance(original.__cause__, OSError)
    assert any("audit disk unavailable" in note for note in original.__notes__)


def test_caught_rpc_cannot_pass_when_its_error_record_could_not_be_written():
    from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits

    original = PermissionError("unavailable operation")

    async def invoke(*args):
        raise original

    class CatchingSandbox:
        async def run_program(self, program, limits, payload, rpc, **kwargs):
            try:
                await rpc("observe", "undeclared", {})
            except PermissionError as caught:
                assert caught is original
            return {"handled": True}

    def failing_evidence(event):
        raise OSError("cannot write RPC evidence")

    async def cancel():
        pass

    kernel = EpisodeKernel(
        baseline(),
        operations={},
        gateway=invoke,
        cancel=cancel,
        emit=failing_evidence,
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
        sandbox=CatchingSandbox(),
    )
    with pytest.raises(OSError, match="cannot write RPC evidence"):
        asyncio.run(kernel.run_plugin(baseline().loop.program, {}, invoke))


@pytest.mark.parametrize(
    ("actual", "expected", "equal"),
    [
        ([0.2, 0.3, 1.02, 1.0, 0.0, 0.0, 0.0], [0.2, 0.3, 1.02, 1, 0, 0, 0], True),
        (
            {"pose": [1, {"offset": 0.0, "valid": True}]},
            {"pose": [1.0, {"valid": True, "offset": 0}]},
            True,
        ),
        (0, -0.0, True),
        (2**53 + 1, float(2**53 + 1), False),
        (1, 1.0000000001, False),
        (True, 1, False),
        (1.0, True, False),
        (False, 0.0, False),
        ({"result": [True]}, {"result": [1]}, False),
        ([1, 2], [2, 1], False),
        ([1], [1.0, 1.0], False),
        ({"value": None}, {}, False),
        ({"value": 1}, {"other": 1.0}, False),
        ("1", 1, False),
        (None, False, False),
        ([1], {"0": 1}, False),
    ],
)
def test_interaction_compares_json_values_without_numeric_coercion(
    actual, expected, equal
):
    from reweave.evolution.validation.interaction import _same_json

    assert _same_json(actual, expected) is equal
    assert _same_json(expected, actual) is equal


def test_numeric_pose_assertion_uses_behavior_equality_and_retains_bundle_reference(
    tmp_path,
):

    source = """def run(runtime, context):
    return {"status": "completed", "summary": "Observed target pose",
            "output": {"target_pose": [0.2, 0.3, 1.02, 1.0, 0.0, 0.0, 0.0]}}
"""
    bundle = candidate(source)
    case = {
        "name": "float-pose-agrees-with-json-number-expectation",
        "system_prompt": "Inspect public feedback.",
        "task_message": "Check the observed pose.",
        "model_responses": [
            {"tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}]}
            for _ in range(4)
        ]
        + [{"tool_calls": [{"name": "move", "arguments": {"distance": 0.1}}]}],
        "operations": [
            {
                "name": "move",
                "arguments": {"distance": 0.1},
                "result": {"step": 1},
                "environment_state": {
                    "official_success": False,
                    "terminated": False,
                    "truncated": True,
                },
            }
        ],
        "assertions": [
            {
                "path": ["workflows", 3, "result", "output", "target_pose"],
                "equals": [0.2, 0.3, 1.02, 1, 0, 0, 0],
            }
        ],
    }
    identity = bundle.record_id
    report = simulate_interaction(bundle, contract(), case, tmp_path / "numeric-pose")
    assert report["validation_status"] == "passed"
    assert report["assertion_results"][0]["actual"] == [
        0.2,
        0.3,
        1.02,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert report["bundle_id"] == bundle.record_id == identity
    assert report["scenario"]["assertions"][0]["equals"] == [0.2, 0.3, 1.02, 1, 0, 0, 0]


def test_numeric_gateway_arguments_and_schema_bounds_compare_by_value(tmp_path):
    gateway = contract()
    gateway["tools"][1]["input_schema"] = deepcopy(MOVE_SCHEMA)
    gateway["tools"][1]["input_schema"]["properties"]["distance"]["minimum"] = 0
    gateway["operations"]["move"]["input_schema"] = deepcopy(MOVE_SCHEMA)
    gateway["operations"]["move"]["input_schema"]["properties"]["distance"][
        "minimum"
    ] = 0.0
    case = {
        "name": "numeric-gateway-values",
        "system_prompt": "Use public tools.",
        "task_message": "Move one unit.",
        "model_responses": [
            {"tool_calls": [{"name": "move", "arguments": {"distance": 1.0}}]}
        ],
        "operations": [
            {
                "name": "move",
                "arguments": {"distance": 1},
                "result": {"step": 1},
                "environment_state": {
                    "official_success": False,
                    "terminated": False,
                    "truncated": True,
                },
            }
        ],
        "assertions": [
            {"path": ["fixture_calls", 0, "arguments", "distance"], "equals": 1}
        ],
    }
    report = simulate_interaction(
        baseline(), gateway, case, tmp_path / "numeric-gateway"
    )
    assert report["validation_status"] == "passed"
    assert report["planner_tool_exposure_verified"] is True
    assert report["fixture_calls"][0]["arguments"] == {"distance": 1.0}
