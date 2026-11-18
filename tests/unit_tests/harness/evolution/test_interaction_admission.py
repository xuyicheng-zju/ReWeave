# Modified for the ReWeave namespace.
"""Final admission exercises changed assets, not just authors' unit fixtures."""

import asyncio
import json

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.validation.admission import (
    _read_interaction_file,
    admit_candidate,
)
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    author_candidate,
    baseline,
    workflow,
)
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def context_change(root):
    (root / "candidate/task-family/context/instructions.md").write_text(
        "Inspect uncertain evidence."
    )
    path = root / "candidate/plan.json"
    plan = json.loads(path.read_text())
    plan["scopes"] = ["C"]
    atomic_json(path, plan)


def test_changed_candidate_requires_real_interaction_and_rechecks_current_bytes(
    tmp_path,
):
    root = workspace(tmp_path)
    context_change(root)
    suite = root / ".scratch/tests/interaction.json"
    original = suite.read_bytes()
    suite.unlink()
    # A code check is still useful before a complete interaction scenario exists.
    asyncio.run(admit_candidate(root, require_plan=False))
    with pytest.raises(ValueError, match="require .scratch/tests/interaction.json"):
        asyncio.run(admit_candidate(root, require_plan=False, check_interactions=True))
    with pytest.raises(ValueError, match="require .scratch/tests/interaction.json"):
        asyncio.run(admit_candidate(root))
    suite.write_bytes(original)
    preview_cases = []
    asyncio.run(
        admit_candidate(
            root,
            require_plan=False,
            check_interactions=True,
            case_results=preview_cases,
        )
    )
    assert any(c["scope"] == "interaction" for c in preview_cases)
    assert not (root.parent / "result.json").exists()
    cases = []
    bundle = asyncio.run(admit_candidate(root, case_results=cases))
    report = next(r for r in cases if r["scope"] == "interaction")
    assert report["bundle_id"] == bundle.record_id
    assert report["coverage"]["C"] and report["coverage"]["L"]
    assert report["scenario"]["task_message"] == "Inspect the scene."
    assert not report["physical_execution_tested"]
    assert report["suite_bytes"] == len(original)
    # Existing successful reports cannot certify changed, invalid C code.
    (root / "candidate/task-family/context/src/context.py").write_text(
        "def build(runtime, context):\n    return {'garbage': True}\n"
    )
    with pytest.raises(ValueError, match="ModelContext"):
        asyncio.run(admit_candidate(root))


def test_text_only_simulation_does_not_cover_a_new_workflow(tmp_path):
    root = workspace(tmp_path)
    gateway_path = root / "context/gateway.json"
    gateway = json.loads(gateway_path.read_text())
    gateway["episode_limits"].update(program_calls=30, gateway_calls=30)
    atomic_json(gateway_path, gateway)
    value = baseline().model_dump(mode="json")
    item = workflow().model_dump(mode="json")
    item.update(
        permissions=[],
        observations=[],
        contract_cases=[
            {
                "arguments": {},
                "observation": {},
                "responses": [],
                "expected_status": "completed",
            }
        ],
    )
    item["program"]["files"][0]["source"] = (
        "def run(runtime, context):\n"
        "    return {'status':'completed','summary':'probe','output':{}}\n"
    )
    value["workflows"] = [item]
    author_candidate(AgentBundle.model_validate(value), root / "candidate")
    proposal_path = host_input(root, "proposal.json")
    proposal = json.loads(proposal_path.read_text())
    proposal["scopes"] = ["W", "C", "L"]
    atomic_json(proposal_path, proposal)
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "A reusable probe may test an interface.",
            "summary": "Expose a probe.",
            "validation": "Synthetic tests only.",
            "scopes": ["W"],
            "cross_scope_reason": None,
        },
    )
    with pytest.raises(
        ValueError, match="did not exercise changed components: workflow:probe"
    ):
        asyncio.run(admit_candidate(root))
    path = root / ".scratch/tests/interaction.json"
    suite = json.loads(path.read_text())
    suite[0]["model_responses"].insert(
        0,
        {
            "tool_calls": [{"name": "workflow_probe", "arguments": {"arguments": {}}}],
            "required_tools": ["workflow_probe"],
        },
    )
    suite[0]["assertions"].append(
        {
            "path": ["workflows", 0, "result", "summary"],
            "equals": "probe",
        }
    )
    atomic_json(path, suite)
    assert asyncio.run(admit_candidate(root)).workflows[0].workflow_id == "probe"


@pytest.mark.parametrize(
    "evidence,verdict",
    [
        ({"goal_satisfied": True}, "supported"),
        ({"goal_satisfied": False}, "refuted"),
        ({}, "unknown"),
    ],
)
def test_verifier_w_only_edit_returns_judgment_to_the_episode_loop(
    tmp_path, evidence, verdict
):
    root = workspace(tmp_path)
    parent = baseline()
    item = workflow().model_dump(mode="json")
    item.update(
        workflow_id="verify_goal",
        description="Check public goal evidence without changing the environment.",
        observations=["read_terminal_state"],
        permissions=["read_terminal_state"],
        output_schema={
            "type": "object",
            "properties": {
                "verdict": {"enum": ["supported", "refuted", "unknown"]},
                "evidence": {"type": "object"},
            },
            "required": ["verdict", "evidence"],
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
                        "result": evidence,
                    }
                ],
                "expected_status": "completed",
            }
        ],
    )
    item["limits"].update(physical_calls=0, model_calls=0)
    payload = parent.model_dump(mode="json")
    payload["workflows"] = [item]
    author_candidate(AgentBundle.model_validate(payload), root / "candidate")
    # Exercise the same file edit and recapture used by a real author.
    source = root / "candidate/task-family/workflows/verify_goal/src/probe.py"
    source.write_text(
        """def run(runtime, context):
    evidence = runtime.observe("read_terminal_state", {})
    satisfied = evidence.get("goal_satisfied")
    if satisfied is None:
        verdict = "unknown"
    else:
        verdict = "supported" if satisfied else "refuted"
    return {"status": "completed", "summary": "Goal evidence inspected",
            "output": {"verdict": verdict, "evidence": evidence}}
"""
    )
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "Task claims need an explicit evidence check.",
            "summary": "Add a read-only verifier through the existing W interface.",
            "validation": "Synthetic interaction only; no physical efficacy claim.",
            "scopes": ["W"],
            "cross_scope_reason": None,
        },
    )
    gateway_path = root / "context/gateway.json"
    gateway = json.loads(gateway_path.read_text())
    gateway["episode_limits"].update(program_calls=30, gateway_calls=30)
    atomic_json(gateway_path, gateway)
    suite_path = root / ".scratch/tests/interaction.json"
    suite = json.loads(suite_path.read_text())
    case = suite[0]
    case["name"] = "episode-goal-verifier-through-default-loop"
    case["model_responses"].insert(
        0,
        {
            "tool_calls": [
                {"name": "workflow_verify_goal", "arguments": {"arguments": {}}}
            ],
            "required_tools": ["workflow_verify_goal"],
        },
    )
    case["operations"].insert(
        0, {"name": "read_terminal_state", "arguments": {}, "result": evidence}
    )
    # Even a positive verifier judgment must not become the official verdict.
    case["operations"][-1]["environment_state"] = {
        "official_success": False,
        "terminated": False,
        "truncated": True,
    }
    case["assertions"].append(
        {
            "path": ["workflows", 0, "result", "output"],
            "equals": {"verdict": verdict, "evidence": evidence},
        }
    )
    atomic_json(suite_path, suite)

    records = []
    admitted = asyncio.run(admit_candidate(root, case_results=records))
    assert admitted.context == parent.context
    assert admitted.loop == parent.loop
    assert admitted.runtime == parent.runtime
    assert admitted.workflows[0].program.files[0].source == source.read_text()
    report = next(row for row in records if row["scope"] == "interaction")
    assert report["coverage"]["workflows"] == ["verify_goal"]
    assert report["workflows"][0]["result"]["status"] == "completed"
    assert len(report["model_requests"]) == 2
    assert verdict in json.dumps(report["model_requests"][1]["messages"])
    assert report["synthetic_environment_state"]["official_success"] is False
    dispatched = [
        event for event in report["events"] if event["kind"] == "action_submitted"
    ]
    assert [event["operation"] for event in dispatched] == [
        "model:plan",
        "read_terminal_state",
    ] * 2
    assert [event["effect"] for event in dispatched] == ["model", "read"] * 2


def test_scenario_paths_cannot_read_outside_the_test_workspace(tmp_path):
    root = workspace(tmp_path)
    outside = tmp_path / "private.json"
    outside.write_text("[]")
    with pytest.raises(ValueError, match="inside .scratch/tests"):
        _read_interaction_file(root, str(outside))
    with pytest.raises(ValueError, match="inside .scratch/tests"):
        _read_interaction_file(root, ".scratch/tests/../../../private.json")
    linked = root / ".scratch/tests/linked.json"
    linked.symlink_to(outside)
    with pytest.raises(OSError):
        _read_interaction_file(root, ".scratch/tests/linked.json")


def test_host_outputs_ignore_author_controlled_interactions_symlink(tmp_path):
    root = workspace(tmp_path)
    context_change(root)
    outside = tmp_path / "not_an_output"
    outside.mkdir()
    (root / ".scratch/tests/interactions").symlink_to(outside, target_is_directory=True)
    asyncio.run(admit_candidate(root))
    assert list(outside.iterdir()) == []
    assert list((tmp_path / "audit/interactions").glob("*/0/interaction.json"))


def test_scenario_parent_symlinks_are_rejected(tmp_path):
    root = workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "suite.json").write_text("[]")
    (root / ".scratch/tests/redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        _read_interaction_file(root, ".scratch/tests/redirect/suite.json")
