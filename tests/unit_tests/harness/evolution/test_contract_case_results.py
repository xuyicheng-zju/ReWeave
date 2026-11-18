# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Production case evidence supports semantic assertions without a second runner."""

import asyncio
import json
from copy import deepcopy

import pytest

from reweave.evolution.authoring.examples import author_examples
from reweave.evolution.authoring.tools import (
    AuthorDevelopmentTools,
    DevelopmentToolError,
    call_development_tool,
)
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.validation.admission import check_runtime_contracts
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    author_candidate,
    baseline,
    workflow,
)
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def case_candidate(marker="complete", *, failing_workflow=False):
    payload = baseline().model_dump(mode="json")
    for scope in ("context", "loop"):
        payload[scope]["program"] = {
            "entrypoint": f"{scope}:run",
            "files": [
                {
                    "path": f"{scope}.py",
                    "source": 'def run(runtime, context):\n    return runtime.observe("inspect_scene", context)\n',
                }
            ],
        }
        payload[scope]["contract_cases"] = [
            {
                "input": {"scope": scope},
                "responses": [
                    {
                        "method": "observe",
                        "name": "inspect_scene",
                        "arguments": {"scope": scope},
                        "result": {"marker": marker},
                    }
                ],
                "expected": {"marker": marker},
            }
        ]
        if scope == "context":
            payload[scope]["program"]["files"][0]["source"] = (
                "def run(runtime, context):\n"
                '    value = runtime.observe("inspect_scene", context)\n'
                '    return {"instructions": value["marker"], "messages": [{"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Synthetic fixture"}]}], "tools": [], "state": {}}\n'
            )
            payload[scope]["contract_cases"][0]["expected"] = {
                "instructions": marker,
                "messages": [
                    {
                        "kind": "request",
                        "parts": [
                            {"part_kind": "user-prompt", "content": "Synthetic fixture"}
                        ],
                    }
                ],
                "tools": [],
                "state": {},
            }
    item = workflow().model_dump(mode="json")
    item["permissions"] = item["observations"] = ["inspect_scene"]
    item["program"]["files"][0]["source"] = """def run(runtime, context):
    try:
        value = runtime.observe("inspect_scene", context["arguments"])
    except Exception:
        return {"status": "failed", "summary": "caught fixture failure", "output": {"caught": True}}
    return {"status": "completed", "summary": "observed", "output": value}
"""
    item["contract_cases"] = [
        {
            "arguments": {"value": index},
            "observation": {},
            "responses": [
                {
                    "operation": "inspect_scene",
                    "arguments": {"value": index},
                    "result": {"marker": marker, "value": index},
                }
            ],
            "expected_status": "completed",
        }
        for index in range(2)
    ]
    if failing_workflow:
        broken = item["contract_cases"][1]
        broken["responses"][0]["operation"] = "wrong_operation"
        broken["responses"].append(
            {
                "operation": "inspect_scene",
                "arguments": {},
                "result": {"unused": marker},
            }
        )
        broken["expected_status"] = "failed"
        item["contract_cases"].append(deepcopy(item["contract_cases"][0]))
    payload["workflows"] = [item]
    return AgentBundle.model_validate(payload)


def contract():
    return {
        "operations": {
            "inspect_scene": {"effect": "read", "input_schema": {"type": "object"}}
        },
        "episode_limits": {
            "program_calls": 10,
            "gateway_calls": 10,
            "wall_time_s": 10,
        },
    }


def test_real_cases_return_complete_outputs_and_ordered_calls():
    marker = "complete-evidence-" * 20000
    candidate = case_candidate(marker)
    before = candidate.model_dump_json()
    records = []
    asyncio.run(check_runtime_contracts(candidate, contract(), case_results=records))
    assert [(r["scope"], r["case_index"]) for r in records] == [
        ("C", 0),
        ("L", 0),
        ("W", 0),
        ("W", 1),
    ]
    assert all(r["validation_status"] == "passed" for r in records)
    assert all(r["unconsumed_responses"] == [] for r in records)
    for record in records[:2]:
        assert record["result"] == (
            {
                "instructions": marker,
                "messages": [
                    {
                        "kind": "request",
                        "parts": [
                            {"part_kind": "user-prompt", "content": "Synthetic fixture"}
                        ],
                    }
                ],
                "tools": [],
                "state": {},
            }
            if record["scope"] == "C"
            else {"marker": marker}
        )
        assert record["calls"][0]["result"] == {"marker": marker}
        assert record["calls"][0]["method"] == "observe"
    for index, record in enumerate(records[2:]):
        assert record["workflow_id"] == "probe"
        assert record["result"]["output"] == {"marker": marker, "value": index}
        assert record["calls"] == [
            {
                "operation": "inspect_scene",
                "arguments": {"value": index},
                "result": {"marker": marker, "value": index},
            }
        ]
    # Observing or editing a development report must not mutate the frozen asset.
    records[0]["calls"][0]["result"]["marker"] = "edited report"
    assert records[0]["result"]["instructions"] == marker
    assert candidate.model_dump_json() == before


def test_failed_case_keeps_prior_results_and_swallowed_fixture_violation():
    records = []
    with pytest.raises(ValueError, match="contract fixture"):
        asyncio.run(
            check_runtime_contracts(
                case_candidate(failing_workflow=True), contract(), case_results=records
            )
        )
    assert len(records) == 4  # The third W case was never run.
    assert [r["validation_status"] for r in records] == ["passed"] * 3 + ["failed"]
    failed = records[-1]
    assert failed["result"]["status"] == "failed"
    assert failed["result"]["output"] == {"caught": True}
    assert failed["calls"][0]["arguments"] == {"value": 1}
    assert "result" not in failed["calls"][0]
    assert failed["calls"][0]["error"]["type"] == "ValueError"
    assert failed["unconsumed_responses"] == [
        {
            "operation": "inspect_scene",
            "arguments": {},
            "result": {"unused": "complete"},
        }
    ]
    assert records[2]["result"]["output"]["value"] == 0


def test_workflow_call_order_and_expected_failure_are_recorded_separately():
    payload = baseline().model_dump(mode="json")
    payload["workflows"] = [author_examples()["W"]["workflow"]]
    gateway = contract()
    gateway["operations"] = {
        "release": {"effect": "physical", "input_schema": {"type": "object"}},
        "view_env_state": {"effect": "read", "input_schema": {"type": "object"}},
    }
    records = []
    asyncio.run(
        check_runtime_contracts(
            AgentBundle.model_validate(payload), gateway, case_results=records
        )
    )
    assert [call["operation"] for call in records[0]["calls"]] == [
        "release",
        "view_env_state",
    ]
    assert records[0]["calls"][0]["result"]["capability_result"]["steps_used"] == 2
    assert records[1]["validation_status"] == "passed"
    assert records[1]["result"]["status"] == "failed"
    assert len(records[1]["calls"]) == 1


def test_plugin_exception_retains_returned_call_but_does_not_invent_case_result():
    candidate = case_candidate().model_dump(mode="json")
    candidate["context"]["program"]["files"][0][
        "source"
    ] = """def run(runtime, context):
    runtime.observe("inspect_scene", context)
    raise ValueError("after observed evidence")
"""
    records = []
    with pytest.raises(Exception, match="after observed evidence"):
        asyncio.run(
            check_runtime_contracts(
                AgentBundle.model_validate(candidate), contract(), case_results=records
            )
        )
    assert len(records) == 1
    assert records[0]["validation_status"] == "failed"
    assert records[0]["calls"][0]["result"] == {"marker": "complete"}
    assert "result" not in records[0]


def test_cancelled_worker_keeps_call_evidence():
    candidate = case_candidate().model_dump(mode="json")
    candidate["context"]["program"]["files"][0]["source"] = """import asyncio
async def run(runtime, context):
    await runtime.observe("inspect_scene", context)
    await asyncio.sleep(30)
"""
    records = []

    async def run():
        pending = asyncio.create_task(
            check_runtime_contracts(
                AgentBundle.model_validate(candidate), contract(), case_results=records
            )
        )
        try:

            async def wait_for_call():
                while not records or not records[0]["calls"]:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_call(), timeout=5)
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

    asyncio.run(run())
    assert records[0]["validation_status"] == "interrupted"
    assert records[0]["error"]["type"] == "CancelledError"
    assert records[0]["calls"][0]["result"] == {"marker": "complete"}
    assert "result" not in records[0]


def test_concurrent_validation_collectors_have_no_shared_case_state():
    async def run():
        records = [[], []]
        await asyncio.gather(
            *(
                check_runtime_contracts(
                    case_candidate(marker), contract(), case_results=records[index]
                )
                for index, marker in enumerate(("branch-a", "branch-b"))
            )
        )
        return records

    for marker, records in zip(
        ("branch-a", "branch-b"), asyncio.run(run()), strict=True
    ):
        assert len(records) == 4
        assert {r["calls"][0]["result"]["marker"] for r in records} == {marker}


def test_development_client_and_audit_preserve_failure_results(tmp_path):
    root = workspace(tmp_path)
    marker = "complete-http-result-" * 15000
    candidate = case_candidate(marker, failing_workflow=True)
    author_candidate(candidate, root / "candidate")
    proposal = json.loads(host_input(root, "proposal.json").read_text())
    proposal["scopes"] = ["W", "C", "L"]
    atomic_json(host_input(root, "proposal.json"), proposal)
    atomic_json(root / "context/gateway.json", contract())
    service = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    service.server.start()
    try:
        with pytest.raises(DevelopmentToolError) as caught:
            call_development_tool(
                service.server.url,
                "validate_candidate",
                {"phase": "code"},
            )
        receipt = caught.value.result["structuredContent"]
        details = call_development_tool(
            service.server.url,
            "read_original",
            {"reference": receipt["complete_report"]},
        )["value"]
        assert "contract fixture" in details["error"]
        records = details["contract_case_results"]
        assert receipt["ok"] is details["ok"] is False
        assert records[2]["result"]["output"]["value"] == 0
        assert records[2]["result"]["output"]["marker"] == marker
        assert records[3]["validation_status"] == "failed"
        audit = next(
            row
            for row in map(json.loads, service.audit.read_text().splitlines())
            if row["tool"] == "validate_candidate"
        )
        assert audit["contract_case_results"] == records
        assert audit["ok"] is False
        author_candidate(case_candidate(marker), root / "candidate")
        passed_receipt = call_development_tool(
            service.server.url,
            "validate_candidate",
            {"phase": "code"},
        )
        passed_entry = call_development_tool(
            service.server.url,
            "read_original",
            {"reference": passed_receipt["complete_report"]},
        )["value"]
        passed = passed_entry["result"]
        assert passed["structurally_valid"] is True
        assert passed["physical_execution_tested"] is False
        assert passed["contract_cases"] == len(passed["contract_case_results"]) == 4
        assert passed["contract_case_results"][2]["result"]["output"] == {
            "marker": marker,
            "value": 0,
        }
        final_audit = next(
            row
            for row in map(json.loads, reversed(service.audit.read_text().splitlines()))
            if row["tool"] == "validate_candidate"
        )
        assert (
            final_audit["result"]["contract_case_results"]
            == passed["contract_case_results"]
        )
    finally:
        service.server.stop(timeout_s=None)
