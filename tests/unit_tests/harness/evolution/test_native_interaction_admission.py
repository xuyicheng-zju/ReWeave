"""Final author admission uses the real native W/C/L adapter with synthetic turns."""

import asyncio
import json

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.validation.admission import admit_candidate
from reweave.evolution.validation.interaction import InteractionFailure
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    author_candidate,
    baseline,
    workflow,
)
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def native_workspace(tmp_path, mode):
    root = workspace(tmp_path)
    gateway_path = root / "context/gateway.json"
    gateway = json.loads(gateway_path.read_text())
    gateway["episode_limits"].update(program_calls=20, gateway_calls=20)
    atomic_json(gateway_path, gateway)
    value = baseline().model_dump(mode="json")
    value["context"] = NativeBundlePlanner.baseline_context().model_dump(mode="json")
    value["context"]["instructions"] = "Inspect current native workflow evidence."
    item = workflow().model_dump(mode="json")
    item.update(
        permissions=["read_terminal_state"],
        observations=["read_terminal_state"],
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        contract_cases=[
            {
                "arguments": {},
                "observation": {},
                "responses": [
                    {
                        "operation": "read_terminal_state",
                        "arguments": {},
                        "result": {"observed": True},
                    }
                ],
                "expected_status": "completed",
            }
        ],
    )
    item["program"]["files"][0]["source"] = (
        "def run(runtime, context):\n"
        "    value = runtime.observe('read_terminal_state', {})\n"
        "    if value.get('observed') is False:\n"
        "        return {'status': 'failed', 'summary': 'Observation unavailable', 'output': {}}\n"
        "    return {'status': 'completed', 'summary': 'State inspected', 'output': value}\n"
    )
    value["workflows"] = [item]
    author_candidate(AgentBundle.model_validate(value), root / "candidate")
    proposal_path = host_input(root, "proposal.json")
    proposal = json.loads(proposal_path.read_text())
    proposal["task_protocol"]["execution"].update(
        controller_request_unit="native_turn",
        adaptation_mode=mode,
        model_sampling={"reasoning_effort": "high"},
    )
    atomic_json(proposal_path, proposal)
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "Current observations must reach the native session and reusable W.",
            "summary": "Use a native C view and a reusable public-state inspection.",
            "validation": "Synthetic native interaction, no model or physical task result.",
            "scopes": ["W", "C"],
            "cross_scope_reason": "Native current-observation guidance exposes the reusable W.",
        },
    )
    atomic_json(
        root / ".scratch/tests/interaction.json",
        [
            {
                "name": "native-context-and-workflow",
                "system_prompt": "Use public state and workflow feedback.",
                "task_message": "Inspect the synthetic scene.",
                "model_responses": [
                    {
                        "tool_calls": [
                            {"name": "workflow_probe", "arguments": {"arguments": {}}}
                        ],
                        "required_tools": ["workflow_probe"],
                    },
                    {
                        "text": "The returned workflow evidence was consumed.",
                        "tool_calls": [
                            {"name": "read_terminal_state", "arguments": {}}
                        ],
                    },
                ],
                "operations": [
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"observed": True},
                    },
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"observed": True, "terminated": True},
                        "environment_state": {
                            "official_success": False,
                            "terminated": True,
                            "truncated": False,
                        },
                    },
                ],
                "assertions": [
                    {"path": ["coverage", "C"], "equals": True},
                    {"path": ["coverage", "L"], "equals": True},
                    {"path": ["coverage", "workflows"], "equals": ["probe"]},
                ],
            },
            {
                "name": "native-context-preserves-workflow-failure",
                "system_prompt": "Inspect settled failure feedback before recovery.",
                "task_message": "Recover from an unavailable observation.",
                "recovery": True,
                "model_responses": [
                    {
                        "tool_calls": [
                            {"name": "workflow_probe", "arguments": {"arguments": {}}}
                        ]
                    },
                    {
                        "tool_calls": [
                            {"name": "workflow_probe", "arguments": {"arguments": {}}}
                        ]
                    },
                ],
                "operations": [
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"observed": False},
                    },
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"observed": True},
                        "environment_state": {
                            "official_success": False,
                            "terminated": True,
                            "truncated": False,
                        },
                    },
                ],
                "assertions": [
                    {"path": ["workflows", 0, "result", "status"], "equals": "failed"},
                    {
                        "path": ["workflows", 1, "result", "status"],
                        "equals": "completed",
                    },
                ],
            },
        ],
    )
    return root


@pytest.mark.parametrize("mode", ["static", "adaptive"])
def test_native_admission_runs_one_session_and_real_c_l_w_without_api(
    tmp_path, mode, monkeypatch
):
    root = native_workspace(tmp_path, mode)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "native admission must not construct an API planner or real native provider"
        )

    from reweave.evolution.validation import interaction

    assert not hasattr(interaction, "ApiAgentLoop")
    monkeypatch.setattr(
        "reweave.planner.providers.CodexDecisionSession.start", forbidden
    )
    cases = []
    bundle = asyncio.run(admit_candidate(root, case_results=cases))
    report = next(value for value in cases if value["scope"] == "interaction")
    assert report["bundle_id"] == bundle.record_id
    assert report["controller_request_unit"] == "native_turn"
    assert report["adaptation_mode"] == mode
    assert report["native_session"]["starts"] == report["native_session"]["closes"] == 1
    assert report["native_session"]["turns"] == 2
    assert report["result"]["stats"]["requests"] is None
    assert report["result"]["stats"]["visible_turns"] == 2
    assert len({value["session_id"] for value in report["model_requests"]}) == 1
    assert all("native_input" in value for value in report["model_requests"])
    first, second = [
        json.loads(value["native_input"].rsplit("\n\n", 1)[-1])
        for value in report["model_requests"]
    ]
    assert "tools" not in first
    assert first["available_tools"] == [
        tool["name"] for tool in report["model_requests"][0]["tools"]
    ]
    assert "tools" not in second
    assert second["tool_catalog"] == "unchanged"
    assert second["available_tools"] == [
        tool["name"] for tool in report["model_requests"][1]["tools"]
    ]
    assert not report["physical_execution_tested"]
    assert report["remaining_model_responses"] == report["remaining_operations"] == []


def test_native_admission_rejects_missing_exposed_tool_without_falling_back(tmp_path):
    root = native_workspace(tmp_path, "static")
    path = root / ".scratch/tests/interaction.json"
    scenarios = json.loads(path.read_text())
    scenarios[0]["model_responses"][0]["required_tools"] = ["nonexistent_tool"]
    atomic_json(path, scenarios)
    with pytest.raises(InteractionFailure, match="lacks required tools"):
        asyncio.run(admit_candidate(root))


def test_native_admission_checks_forbidden_tools_on_catalog_reference(tmp_path):
    root = native_workspace(tmp_path, "static")
    path = root / ".scratch/tests/interaction.json"
    scenarios = json.loads(path.read_text())
    scenarios[0]["model_responses"][1]["forbidden_tools"] = ["workflow_probe"]
    atomic_json(path, scenarios)
    with pytest.raises(InteractionFailure, match="exposes forbidden tools"):
        asyncio.run(admit_candidate(root))


@pytest.mark.parametrize(
    "invalid", ["missing_definitions", "changed_names", "changed_marker", "mixed"]
)
def test_native_admission_rejects_invalid_catalog_reference(
    tmp_path, monkeypatch, invalid
):
    from reweave.planner import native_loop

    root = native_workspace(tmp_path, "static")
    original_prepare = native_loop.prepare_native
    calls = 0

    def prepare(context, directory, limits, *, include_tools=True):
        nonlocal calls
        calls += 1
        text, images = original_prepare(
            context,
            directory,
            limits,
            include_tools=False if invalid == "missing_definitions" else include_tools,
        )
        payload = json.loads(text)
        if calls == 1 and invalid == "missing_definitions":
            agents = directory.parent.parent / "workspace" / "AGENTS.md"
            agents.unlink()
            agents.write_text("# Episode instructions\nMissing fixed tool catalog.\n")
        if calls == 2:
            if invalid == "changed_names":
                payload["available_tools"].append("never_delivered_tool")
            elif invalid == "changed_marker":
                payload["tool_catalog"] = "use_an_unknown_catalog"
            elif invalid == "mixed":
                payload["tools"] = context["tools"]
        return json.dumps(payload), images

    monkeypatch.setattr(native_loop, "prepare_native", prepare)
    with pytest.raises(InteractionFailure, match="native tool catalog"):
        asyncio.run(admit_candidate(root))


def test_native_recovery_uses_host_call_identity_and_requires_failed_feedback(tmp_path):
    from reweave.evolution.validation.interaction import (
        check_recovery,
        simulate_interaction,
    )
    from tests.unit_tests.harness.evolution.test_author_interaction import (
        OLD_GATE,
        candidate,
        contract,
        scenario,
    )

    bundle = candidate(OLD_GATE).model_copy(
        update={"context": NativeBundlePlanner.baseline_context()}
    )
    case = scenario()
    case["recovery"] = True
    case["assertions"] = [
        {"path": ["workflows", 0, "result", "status"], "equals": "failed"}
    ]
    report = simulate_interaction(
        bundle,
        contract(),
        case,
        tmp_path / "native-recovery",
        controller_request_unit="native_turn",
    )
    assert report["validation_status"] == "passed"
    assert len(report["model_requests"]) == 3
    assert report["calls"][-1]["name"] == "move"
    for event in report["events"]:
        if event["kind"] == "context_built":
            event["output"]["messages"] = []
    with pytest.raises(ValueError, match="dropped the failed Workflow result"):
        check_recovery(report, contract())


def test_visual_admission_rejects_text_only_contract_and_interaction_cases(tmp_path):
    root = native_workspace(tmp_path, "static")
    path = root / "context/gateway.json"
    contract = json.loads(path.read_text())
    contract["native_image_contract"] = {
        "observation": 3,
        "reads": {"read_terminal_state": 0},
    }
    atomic_json(path, contract)
    cases = []
    with pytest.raises(ValueError, match="visual input coverage missing"):
        asyncio.run(admit_candidate(root, case_results=cases))
    requests = [
        request
        for case in cases
        if case["scope"] == "interaction"
        for request in case["model_requests"]
    ]
    assert requests and all(request["distinct_images"] == 0 for request in requests)


@pytest.mark.parametrize("text_only_output", [False, True])
def test_visual_admission_counts_real_bytes_selected_from_workflow_result(
    tmp_path, text_only_output
):
    import base64

    from tests.unit_tests.harness.planner.test_native_context import pixels

    root = native_workspace(tmp_path, "static")
    path = root / "context/gateway.json"
    contract = json.loads(path.read_text())
    contract["native_image_contract"] = {
        "observation": 3,
        "reads": {"read_terminal_state": 0},
    }
    atomic_json(path, contract)
    path = root / ".scratch/tests/interaction.json"
    suite = json.loads(path.read_text())
    suite[0]["operations"][0]["result"]["_runtime_images"] = [
        {
            "name": "synthetic-view",
            "media_type": "image/png",
            "data_base64": base64.b64encode(pixels("red")).decode(),
        }
    ]
    atomic_json(path, suite)
    if text_only_output:
        context_file = root / "candidate/task-family/context/src/context.py"
        source = context_file.read_text()
        source = source.replace(
            '"messages": context["messages"]',
            '"messages": strip_pixels(context["messages"])',
        )
        source += """

def strip_pixels(value):
    if isinstance(value, dict):
        if value.get("kind") in ("binary", "image-url"):
            return "C selected text evidence for this request"
        return {key: strip_pixels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_pixels(item) for item in value]
    return value
"""
        context_file.write_text(source)
    cases = []
    asyncio.run(admit_candidate(root, case_results=cases))
    requests = [
        request
        for case in cases
        if case["scope"] == "interaction"
        for request in case["model_requests"]
    ]
    assert max(request["distinct_images"] for request in requests) == (
        0 if text_only_output else 1
    )
