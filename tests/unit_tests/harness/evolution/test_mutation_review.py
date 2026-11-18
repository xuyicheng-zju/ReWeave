"""Observed activation, topology tradeoffs and real-input admission regressions."""

import asyncio
import base64
import json
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest

from reweave.evolution.authoring.candidate import AuthorPlan, MutationProbe
from reweave.evolution.bundle import AgentBundle, PythonFile, PythonProgram
from reweave.evolution.contracts import Candidate
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.search.mutation_review import observe_probes, review_mutation
from reweave.evolution.search.review import compare_versions, review_search
from reweave.evolution.validation.context_replay import (
    check_recorded_contexts,
    recorded_context_inputs,
)
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline, mutated
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace
from tests.unit_tests.harness.evolution.test_semantic_graph import width_map


def with_events(root, events):
    """Attach exact saved public events to a frozen parent receipt."""
    path = root.parent / "recorded/harness_context/bundle_events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    proposal_path = host_input(root, "proposal.json")
    value = json.loads(proposal_path.read_text())
    value["evidence"][0]["evidence"].append(artifact_record(path))
    atomic_json(proposal_path, value)
    from reweave.evolution.authoring.candidate import read_proposal

    return read_proposal(root)


def test_probe_requires_ordered_activation_and_does_not_certify_causality(tmp_path):
    root = workspace(tmp_path)
    events = [
        {"kind": "operation_settled", "operation": "move", "result": {"reached": True}},
        {"kind": "workflow_started", "workflow_id": "recover"},
        {
            "kind": "operation_settled",
            "operation": "move",
            "result": {"reached": False},
        },
    ]
    proposal = with_events(root, events)
    probe = MutationProbe(
        claim="Recovery invocation should precede a reached target.",
        activation={"kind": "workflow_started", "fields": {"workflow_id": "recover"}},
        effect={
            "kind": "operation_settled",
            "fields": {"operation": "move", "result": {"reached": True}},
        },
    )
    assert (
        observe_probes(proposal.evidence[0], [probe])[0]["status"]
        == "activation_observed"
    )
    events.append(
        {
            "kind": "operation_settled",
            "operation": "move",
            "result": {"reached": True},
        }
    )
    path = tmp_path / "recorded/harness_context/bundle_events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    observation = observe_probes(proposal.evidence[0], [probe])[0]
    assert observation["status"] == "local_effect_observed"
    assert observation["activation"]["pointer"] == "/1"
    assert observation["effect"]["pointer"] == "/3"
    value = probe.model_copy(
        update={
            "activation": probe.activation.model_copy(
                update={"fields": {"workflow_id": "unused"}}
            )
        }
    )
    assert observe_probes(proposal.evidence[0], [value])[0]["status"] == "not_observed"
    candidate = Candidate(
        branch_id="trial",
        bundle=mutated(baseline(), "C"),
        summary=AuthorPlan(
            diagnosis="Uncertain",
            summary="Trial",
            validation="Synthetic only",
            scopes=("C",),
            probes=(probe,),
        ).model_dump_json(),
        experience="Unverified.",
    )
    review = review_mutation(proposal, candidate, proposal.evidence)
    assert review["conditions"][0]["official_change"] == "unknown"
    assert review["conditions"][0]["probes"][0]["status"] == "unknown"
    assert not review["causal_improvement_established"]
    child = proposal.evidence[0].model_copy(
        update={
            "record_id": "measured-child",
            "request": proposal.evidence[0].request.model_copy(
                update={"bundle_id": candidate.bundle.record_id}
            ),
            "official_success": True,
            "failure_origin": "none",
        }
    )
    review = review_mutation(proposal, candidate, (*proposal.evidence, child))
    assert review["conditions"][0]["official_change"] == "gained"
    assert review["conditions"][0]["probes"][0]["status"] == "local_effect_observed"
    assert not review["reliability_established"]
    assert review["conditions"][0]["receipt_in_author_inputs"] is False
    child = child.model_copy(
        update={
            "request": child.request.model_copy(
                update={
                    "conditions": child.request.conditions.model_copy(
                        update={"repetition": 1}
                    )
                }
            )
        }
    )
    assert (
        review_mutation(proposal, candidate, (*proposal.evidence, child))["conditions"][
            0
        ]["official_change"]
        == "unknown"
    )


def test_topology_tradeoff_and_unknown_are_not_an_improvement_or_efficiency_gain():
    value = width_map(samples=2, completed=2)
    value["graph_id"] = "one-current-graph"
    value["episodes"]["1:0"]["completed_nodes"] = ["near", "open"]
    for cell in value["episodes"].values():
        cell["usage"] = {"physical_steps": 50 if cell["bundle_id"] == "0" else 20}
    comparison = compare_versions(value, "0", "1")
    assert comparison["status"] == "tradeoff"
    cell = comparison["conditions"]["0"]
    assert cell["gained_nodes"] == ["open"] and cell["lost_nodes"] == ["held"]
    assert cell["physical_step_delta"] is None
    assert comparison["conditions"]["1"]["physical_step_delta"] == -30
    state = {
        "order": ["0", "1"],
        "versions": {"0": {"proposal_id": None}, "1": {"proposal_id": "p"}},
        "proposals": {"p": {"parent": {"version_id": "0"}, "round_index": 0}},
    }
    versions = [SimpleNamespace(version_id="0"), SimpleNamespace(version_id="1")]
    review = review_search(value, versions, state, versions[0])
    assert review["consecutive_without_observed_retained_gain"] == 1
    value["episodes"]["1:0"]["completed_nodes"] = None
    assert compare_versions(value, "0", "1")["status"] == "unknown"
    assert (
        review_search(value, versions, state, versions[0])[
            "consecutive_without_observed_retained_gain"
        ]
        == 0
    )


@pytest.mark.parametrize("siblings", tuple(permutations(("1", "2", "3"))))
def test_search_stagnation_counts_rounds_independently_of_sibling_order(siblings):
    value = width_map(samples=4, completed=1)
    value["graph_id"] = "one-current-graph"
    for condition in range(4):
        value["episodes"][f"1:{condition}"]["completed_nodes"] = ["near", "held"]
        value["episodes"][f"2:{condition}"]["completed_nodes"] = []
        value["episodes"][f"3:{condition}"]["completed_nodes"] = []
    state = {
        "order": ["0", *siblings],
        "versions": {str(i): {"proposal_id": f"p{i}" if i else None} for i in range(4)},
        "proposals": {
            f"p{i}": {"parent": {"version_id": "0"}, "round_index": 0}
            for i in range(1, 4)
        },
    }
    versions = [SimpleNamespace(version_id=str(i)) for i in range(4)]
    result = review_search(value, versions, state, versions[0])
    assert result["mutations"]["1"]["status"] == "improved"
    assert result["consecutive_without_observed_retained_gain"] == 0


@pytest.mark.parametrize("width", [1, 2, 3])
def test_search_stagnation_is_number_of_rounds_not_number_of_authors(width):
    value = width_map(samples=1 + 2 * width, completed=1)
    value["graph_id"] = "one-current-graph"
    identities = [str(i) for i in range(1 + 2 * width)]
    state = {
        "order": identities,
        "versions": {
            identity: {"proposal_id": f"p{identity}" if identity != "0" else None}
            for identity in identities
        },
        "proposals": {
            f"p{i}": {"parent": {"version_id": "0"}, "round_index": (i - 1) // width}
            for i in range(1, 1 + 2 * width)
        },
    }
    versions = [SimpleNamespace(version_id=identity) for identity in identities]
    assert (
        review_search(value, versions, state, versions[0])[
            "consecutive_without_observed_retained_gain"
        ]
        == 2
    )


def test_alternative_goal_nodes_share_completion_and_incomplete_cost_is_unknown():
    value = width_map(samples=2, completed=3)
    for cell in value["episodes"].values():
        cell["completed_nodes"] = [
            *value["milestone_nodes"],
            "goal_" + cell["bundle_id"],
        ]
        cell["official_success"] = True
        cell["usage"] = {"physical_steps": 20, "physical_steps_complete": True}
    assert compare_versions(value, "0", "1")["status"] == "unchanged"
    value["episodes"]["1:0"]["usage"]["physical_steps_complete"] = False
    assert (
        compare_versions(value, "0", "1")["conditions"]["0"]["physical_step_delta"]
        is None
    )


@pytest.mark.parametrize("failure", ["null", "images", "pointer"])
def test_recorded_parent_input_is_checked_by_complete_native_host_contract(
    tmp_path, failure
):
    root = workspace(tmp_path)
    gateway = json.loads((root / "context/gateway.json").read_text())
    messages = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "Current evidence"}],
        }
    ]
    context = {
        "base_instructions": "Control",
        "assets": baseline().context.model_dump(mode="json"),
        "tools": gateway["tools"],
        "messages": messages,
        "state": {},
        "observation": None,
    }
    if failure in {"images", "pointer"}:
        messages[0]["parts"][0]["content"] = [
            {
                "kind": "binary",
                "media_type": "image/png",
                "data": base64.b64encode(bytes([index])).decode(),
            }
            for index in range(12 if failure == "images" else 1)
        ]
    proposal = with_events(root, [{"kind": "context_rejected", "input": context}])
    original = deepcopy(context)
    bundle = baseline()
    source = bundle.context.program.files[0].source
    if failure == "null":
        source = source.replace(
            "    assets =", '    context["observation"].get("field")\n    assets ='
        )
    if failure == "pointer":
        source = source.replace(
            'return {"instructions":',
            'return {"image_selection": {"paths": ["/messages/0/parts/0/content/0"], "reason": "Wrong namespace"}, "instructions":',
        )
    program = PythonProgram(
        entrypoint="context:build",
        files=(PythonFile(path="context.py", source=source),),
    )
    candidate = AgentBundle.model_validate(
        {
            **bundle.model_dump(mode="json"),
            "context": {
                **bundle.context.model_dump(mode="json"),
                "program": program.model_dump(mode="json"),
            },
        }
    )
    reports = []
    with pytest.raises(ValueError, match="historical C input rejected"):
        asyncio.run(
            check_recorded_contexts(root, candidate, gateway, case_results=reports)
        )
    assert reports[0]["validation_status"] == "failed"
    assert not reports[0]["physical_execution_tested"]
    assert next(recorded_context_inputs(proposal.evidence[0]))["input"] == original


def test_recorded_replay_never_fabricates_a_new_read_or_certifies_full_loop(tmp_path):
    root = workspace(tmp_path)
    gateway = json.loads((root / "context/gateway.json").read_text())
    context = {
        "base_instructions": "Control",
        "assets": baseline().context.model_dump(mode="json"),
        "tools": gateway["tools"],
        "messages": [
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": "Inspect"}],
            }
        ],
    }
    with_events(root, [{"kind": "context_built", "input": context}])
    bundle = baseline()
    source = bundle.context.program.files[0].source.replace(
        "    assets =", '    runtime.observe("read_terminal_state", {})\n    assets ='
    )
    program = PythonProgram(
        entrypoint="context:build",
        files=(PythonFile(path="context.py", source=source),),
    )
    candidate = bundle.model_copy(
        update={"context": bundle.context.model_copy(update={"program": program})}
    )
    report = asyncio.run(check_recorded_contexts(root, candidate, gateway))[0]
    assert report["validation_status"] == "not_tested"
    assert report["new_reads"][0]["name"] == "read_terminal_state"
    assert not report["full_loop_tested"]


def test_native_context_receives_real_shared_turn_counters(tmp_path):
    from reweave.runtime.kernel import EpisodeLimits
    from tests.native_fixture import ScriptedNativePlanner, rows
    from tests.native_robot import Robot

    count = 0

    def decide(messages, info):
        nonlocal count
        count += 1
        return {
            "text": "Synthetic",
            "tool_calls": [
                {
                    "name": "inspect_scene" if count == 1 else "read_terminal_state",
                    "arguments_json": "{}",
                }
            ],
        }

    planner = ScriptedNativePlanner(decide)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=None
        ),
    )
    result = planner.solve(
        system_prompt="Inspect",
        user_message="Complete",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    inputs = [
        event["input"]
        for event in rows(tmp_path / "harness_context/bundle_events.jsonl")
        if event["kind"] == "context_built"
    ]
    assert [value["controller_budget"]["used"] for value in inputs] == [0, 1]
    assert [value["controller_budget"]["remaining"] for value in inputs] == [3, 2]
    assert result.stats["visible_turns"] == 2


def test_new_c_does_not_inherit_an_old_c_private_state_contract(tmp_path):
    root = workspace(tmp_path)
    gateway = json.loads((root / "context/gateway.json").read_text())
    with_events(root, [{"kind": "context_built", "input": {"state": {"old_cache": 1}}}])
    bundle = baseline()
    source = bundle.context.program.files[0].source + "\n# New implementation\n"
    program = PythonProgram(
        entrypoint="context:build",
        files=(PythonFile(path="context.py", source=source),),
    )
    candidate = bundle.model_copy(
        update={"context": bundle.context.model_copy(update={"program": program})}
    )
    report = asyncio.run(check_recorded_contexts(root, candidate, gateway))[0]
    assert report["validation_status"] == "not_tested"
    assert "private C state" in report["reason"]


def test_search_review_uses_issued_version_condition_and_episode_handles(tmp_path):
    from reweave.evolution.authoring.references import AuthorReferences

    root = workspace(tmp_path)
    catalog = AuthorReferences.load(root)
    review = {
        "graph_id": "graph-original",
        "parent_version_id": "v0",
        "mutations": {
            "v1": {
                "conditions": {
                    "c0": {
                        "before_receipt_id": "e0",
                        "after_receipt_id": "e1",
                        "gained_nodes": ["held"],
                        "lost_nodes": ["near"],
                    }
                }
            }
        },
    }
    encoded = catalog.encode(review)
    assert "v1" not in encoded["mutations"]
    candidate = next(iter(encoded["mutations"]))
    assert candidate.startswith("version:")
    condition = next(iter(encoded["mutations"][candidate]["conditions"]))
    assert condition.startswith("condition:")
    cell = encoded["mutations"][candidate]["conditions"][condition]
    assert cell["before_receipt_id"].startswith("episode:")
    assert cell["after_receipt_id"].startswith("episode:")
    assert catalog.decode(encoded) == review


def test_recorded_v1_physical_batches_use_same_v2_host_projection(tmp_path):
    from reweave.planner.native_context import NATIVE_INPUT_CONTRACT
    from tests.unit_tests.harness.planner.test_native_image_contract import (
        binary,
        frame,
        observation,
    )

    root = workspace(tmp_path)
    gateway = json.loads((root / "context/gateway.json").read_text())
    messages = [
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "tool-return",
                    "tool_name": "move_once",
                    "content": f"ack-{i}",
                    "tool_call_id": str(i),
                },
                {
                    "part_kind": "user-prompt",
                    "content": [binary(frame(3 * i + j)) for j in range(3)],
                },
            ],
        }
        for i in range(4)
    ]
    current = observation([frame(i) for i in range(9, 12)])
    messages[-1]["parts"].append(
        {
            "part_kind": "user-prompt",
            "content": [
                json.dumps(
                    {
                        "observation": current,
                        "physical_budget": None,
                        "remaining_wall_time_s": None,
                        "objective": "Inspect",
                    }
                ),
                *[binary(frame(i)) for i in range(9, 12)],
            ],
        }
    )
    context = {
        "base_instructions": "Control",
        "assets": baseline().context.model_dump(mode="json"),
        "tools": gateway["tools"],
        "messages": messages,
        "state": {},
        "observation": current,
        "physical_budget": None,
        "remaining_wall_time_s": None,
        "objective": "Inspect",
        "current_observation_in_messages": True,
    }
    gateway["operations"]["move_once"] = {
        "effect": "physical",
        "input_schema": {"type": "object"},
    }
    proposal = with_events(root, [{"kind": "context_rejected", "input": context}])
    reports = asyncio.run(check_recorded_contexts(root, baseline(), gateway))
    assert reports[0]["validation_status"] == "passed"
    assert reports[0]["native_input_contract"] == NATIVE_INPUT_CONTRACT
    assert reports[0]["request_images"] == 3
    assert next(recorded_context_inputs(proposal.evidence[0]))["input"] == context


@pytest.mark.parametrize("distinct", [False, True])
def test_declared_c_contract_uses_online_distinct_image_limit(tmp_path, distinct):
    from reweave.evolution.validation.admission import check_runtime_contracts
    from tests.unit_tests.harness.planner.test_native_image_contract import (
        binary,
        frame,
    )

    root = workspace(tmp_path)
    gateway = json.loads((root / "context/gateway.json").read_text())
    context = {
        "base_instructions": "Control",
        "assets": {"instructions": "", "skills": []},
        "tools": [],
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            binary(frame(i if distinct else 0)) for i in range(12)
                        ],
                    }
                ],
            }
        ],
        "state": {},
    }
    original = baseline()
    namespace = {}
    exec(original.context.program.files[0].source, namespace)
    expected = namespace["build"](None, context)
    value = original.model_dump(mode="json")
    value["context"]["contract_cases"] = [
        {"input": context, "expected": expected, "responses": []}
    ]
    candidate = AgentBundle.model_validate(value)
    if distinct:
        with pytest.raises(Exception, match="12 > 10"):
            asyncio.run(check_runtime_contracts(candidate, gateway))
    else:
        asyncio.run(check_runtime_contracts(candidate, gateway))
