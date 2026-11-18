# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Routed author guidance keeps one contract copy and concise requests."""

import asyncio
import json
import sys

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.candidate import SCOPE_DIRECTORIES
from reweave.evolution.authoring.prompts import (
    AUTHOR_METHOD_REFERENCE,
    PLUGIN_CONTRACTS,
    RUNTIME_CONTRACT,
    author_instructions,
    author_method_reference,
    author_reference_documents,
    author_request,
    development_guide,
    engineering_reference,
    runtime_interfaces,
)
from reweave.evolution.contracts import Proposal
from reweave.evolution.validation.admission import admit_candidate
from reweave.memory.experience import EXPERIENCE_FORMAT
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def test_batch_exposes_all_plugin_contracts(tmp_path):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    instructions = author_instructions(proposal)
    guide = development_guide(proposal, sys.executable)
    for scope, contract in PLUGIN_CONTRACTS.items():
        assert f"candidate/{SCOPE_DIRECTORIES[scope]}/" in instructions
        assert contract not in instructions
        assert contract not in guide
        assert contract in runtime_interfaces()[scope]
    assert RUNTIME_CONTRACT not in guide
    references = "\n".join(
        author_reference_documents(proposal, sys.executable).values()
    )
    assert guide.count(EXPERIENCE_FORMAT) == 0
    assert references.count(EXPERIENCE_FORMAT) == 1
    assert EXPERIENCE_FORMAT not in instructions
    assert "runtime.schema.json" in guide
    assert "context/capability_catalog.json" in guide
    assert "Use issued handles" in guide
    assert "context/source_inventory.json" in guide
    assert '{"arguments": <W input>' in runtime_interfaces()["L"]


def test_mutation_instructions_require_one_candidate_per_session(tmp_path):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    instructions = " ".join(author_instructions(proposal).split())
    reference = " ".join(
        (
            development_guide(proposal, sys.executable)
            + "\n"
            + "\n".join(author_reference_documents(proposal, sys.executable).values())
        ).split()
    )
    assert "ONE complete candidate" in instructions
    assert "SKILL.md is optional" in instructions
    assert "only correction/recovery continues this session" in instructions
    assert "candidate/ already contains the parent" in instructions
    assert "Every historical implementation remains readable" in reference
    assert "Host derives changed components from code" in reference
    assert "additional reusable lessons" in reference
    assert "context/donors" not in instructions
    assert "candidate/mutation.json" not in instructions
    assert "context/EVOLUTION_HISTORY.md" in reference
    assert "context/history_originals.json" in reference
    assert "Each assignment is independent" in instructions
    assert "context/route.json" not in instructions
    assert "no investigation sequence is assigned" in instructions


@pytest.mark.parametrize("scopes", [("W", "C", "L"), ("W", "C"), ("C",)])
def test_entry_keeps_component_roles_and_distinguishes_loop_policy_from_contracts(
    tmp_path, scopes
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    ).model_copy(update={"scopes": scopes})
    text = " ".join(author_instructions(proposal).split())
    assert "C shapes usable evidence" in text
    assert ("W implements callable feedback-dependent work" in text) == ("W" in scopes)
    if "L" in scopes:
        assert "L controls planning, execution and feedback boundaries" in text
    else:
        assert "L is read-only" in text
    assert "Editable L may change when to plan, execute and return feedback" in text
    assert "Preserve fixed runtime, tool, observation and budget contracts" in text
    assert "cancellation, feedback boundaries" not in text
    assert "without quotas" in text


@pytest.mark.parametrize("purpose", ["mutation", "success_consolidation"])
def test_short_entry_routes_complete_guidance_without_dropping_contracts(
    tmp_path, purpose
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    ).model_copy(update={"purpose": purpose})
    instructions = author_instructions(proposal)
    normalized = " ".join(instructions.split())
    guide = development_guide(proposal, sys.executable)
    # The method and engineering layers are each delivered once, in the guide.
    method = author_method_reference(proposal.scopes, purpose=purpose)
    assert guide.count(method.replace("\n## ", "\n### ").rstrip("\n")) == 1
    assert guide.count(engineering_reference(proposal, sys.executable)) == 1
    assert AUTHOR_METHOD_REFERENCE not in instructions
    assert len(instructions) < 5000
    assert "Online provider history stays native" in normalized
    assert "C sends incremental context with at most 10 images" in normalized
    assert "consult affected runtime.schema.json interfaces as needed" in normalized
    assert "no investigation sequence is assigned" in normalized
    assert "AUTHOR_WORKFLOW.md" not in instructions
    assert "AUTHOR_WORKFLOW.md" not in author_reference_documents(
        proposal, sys.executable
    )
    assert 'validate_candidate(phase="code")' in instructions
    assert 'validate_candidate(phase="interaction")' in instructions
    assert "repair failures before returning" in normalized
    assert "physical trials, external model probes or service launches" in instructions
    assert "No task/seed dispatch" in instructions
    assert "without quotas" in instructions
    if purpose == "mutation":
        assert "write and edit only task-family/" in instructions
        assert "consolidation" not in guide.split("## Engineering reference")[0]
        assert "For success consolidation" not in instructions
    else:
        assert "factor reusable task-bound Workflow logic" in guide
        assert "For success consolidation, start from winning" in normalized
        assert "AFTER FIXED-EVALUATOR SUCCESS" in instructions
        assert "reorganized content has synthetic admission only" in normalized


def test_first_author_request_points_to_shared_graph_and_independent_diagnosis(
    tmp_path,
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    request = author_request(proposal)
    history = (root / "context/EVOLUTION_HISTORY.md").read_text()
    assert history not in request
    assert "not assigned repair methods" in request
    assert "context/route.json" not in request
    assert "context/progress_map.json" in request
    assert "context/EVOLUTION_HISTORY.md" in request


def test_author_guide_states_the_actual_parent_selection_rule(tmp_path):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    global_guide = development_guide(proposal, sys.executable)
    latest = proposal.model_copy(
        update={
            "task_protocol": proposal.task_protocol.model_copy(
                update={
                    "parent_selection": "latest_evaluated",
                    "width_mode": "fixed",
                    "max_authors_per_round": 1,
                }
            )
        }
    )
    latest_guide = development_guide(latest, sys.executable)
    global_rule = (
        "the parent is selected by confirmed official development success first"
    )
    latest_rule = "the parent is the newest historical Harness with a complete development evaluation"
    assert global_rule in global_guide and latest_rule not in global_guide
    assert latest_rule in latest_guide and global_rule not in latest_guide
    assert engineering_reference(proposal, sys.executable) == engineering_reference(
        latest, sys.executable
    )


def test_proposal_rejects_the_removed_single_history_file(tmp_path):
    root = workspace(tmp_path)
    value = json.loads(host_input(root, "proposal.json").read_text())
    value["history"] = value["history"]["summary"]
    with pytest.raises(ValidationError):
        Proposal.model_validate(value)


def test_proposal_requires_the_actual_evaluation_protocol(tmp_path):
    root = workspace(tmp_path)
    value = json.loads(host_input(root, "proposal.json").read_text())
    proposal = Proposal.model_validate(value)
    value["task_protocol"] = proposal.task_protocol.record_id
    with pytest.raises(ValidationError, match="task_protocol"):
        Proposal.model_validate(value)


@pytest.mark.parametrize("removed", ["hypothesis", "frontier", "route", "forecast"])
def test_proposal_rejects_assigned_diagnosis(tmp_path, removed):
    root = workspace(tmp_path)
    value = json.loads(host_input(root, "proposal.json").read_text())
    value[removed] = {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Proposal.model_validate(value)


def test_partial_diagnosis_plan_is_executable_but_never_final(tmp_path):
    root = workspace(tmp_path)
    path = root / "candidate/plan.json"
    plan = json.loads(path.read_text())
    plan.update(summary="", validation="")
    path.write_text(json.dumps(plan))
    assert asyncio.run(admit_candidate(root, require_plan=False))
    with pytest.raises(ValueError):
        asyncio.run(admit_candidate(root))


def test_feedback_validation_and_retrieval_are_opt_in(
    tmp_path,
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    guide = (
        development_guide(proposal, sys.executable)
        + "\n"
        + "\n".join(author_reference_documents(proposal, sys.executable).values())
    )
    assert proposal.task_protocol.validation is None
    assert "Feedback validation is measured only after" not in guide
    assert "retrieve_graphs(query=None" not in guide
    assert "do not relabel them as the current observation" in guide
    assert "retrieve_graphs(query=None" in development_guide(
        proposal, sys.executable, graph_retrieval=True
    )
    assert "Feedback validation is measured only after" in author_method_reference(
        proposal.scopes, feedback_validation=True
    )


def test_author_has_no_nested_model_experiment_instructions(tmp_path):
    from tests.unit_tests.harness.evolution.test_ablation_prompt_contract import (
        prepared_author,
    )

    _, _, directory = prepared_author(tmp_path, ("W", "C", "L"))
    context = directory / "workspace/context"
    assert not (context / "NATIVE_EXPERIMENTS.md").exists()
    assert not (context / "experiment_access.json").exists()
    guide = (context / "DEVELOPMENT.md").read_text()
    assert "do not start additional model sessions" in guide
    assert "read_sensor_snapshot" in guide
