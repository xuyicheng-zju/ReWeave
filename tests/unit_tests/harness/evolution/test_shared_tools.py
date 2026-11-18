"""Shared Codex tools must remain autonomous and strictly source scoped."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from reweave.evolution.authoring.client import call_development_tool
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.conditions import encoding_request
from reweave.evolution.distillation.tools import (
    SharedAuthorTools,
    implementation_schema,
    validate_implementation,
)
from reweave.evolution.distillation.topology import occurrence_key
from tests.unit_tests.harness.evolution.test_shared import bundle, plan
from tests.unit_tests.harness.evolution.test_shared_topology import (
    graphs_fixture,
    initial_topology,
    teachers,
)


def prepared_workspace(tmp_path):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    p = plan().model_copy(update={"teachers": teachers(graphs)})
    request = encoding_request(p)
    request["method"] = "aligned"
    request["tasks"] = [{"name": name, "family": name, "dev": []} for name in graphs]
    request["source_graphs"] = {
        name: g.model_dump(mode="json") for name, g in graphs.items()
    }
    request["shared_topology"] = topology.model_dump(mode="json")
    author = NativeSharedAuthor(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
        runtime_manifest=SimpleNamespace(
            authoring_contract=SimpleNamespace(
                operations={}, tools=[], native_image_contract=None
            ),
            configuration={"limits": {"episode": {}}},
        ),
    )
    workspace, _ = author.prepare(request, tmp_path / "author")
    return request, workspace, graphs, topology, author


def test_source_tools_support_local_and_whole_queries_and_native_catalog(tmp_path):
    _, workspace, graphs, _, _ = prepared_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit/source_tools.jsonl")
    names = {t.name for t in asyncio.run(tools.catalog())}
    assert names == {
        "retrieve_graphs",
        "read_reference_graph",
        "read_original",
        "validate_topology",
    }
    assert len(asyncio.run(tools.model_call("read_reference_graph", {}))["tasks"]) == 2
    result = asyncio.run(
        tools.model_call("retrieve_graphs", {"query_task": "alpha", "top_k": 2})
    )
    assert result["evaluated_tasks"] == 2
    assert result["matches"][0]["task"] == "alpha"
    with pytest.raises(ValueError, match="explicit query"):
        asyncio.run(tools.model_call("retrieve_graphs", {}))
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(tools.model_call("retrieve_graphs", {"query_task": "unseen"}))
    catalog = asyncio.run(tools.model_call("read_reference_graph", {"view": "catalog"}))
    assert catalog["catalog"] and catalog["embedding"] is None
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(tools.model_call("ask_session", {}))


def test_without_topology_has_same_author_but_no_graph_inputs_or_tools(tmp_path):
    from reweave.evolution.distillation.engine import SharedEvolution
    from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator

    # Even when the Host has source graphs for common evaluation, the synthesis
    # request cannot expose them or their per-witness projection explanations.
    graphs = graphs_fixture(tmp_path)
    base = plan()
    source_graph = graphs["alpha"]
    settings = base.model_copy(
        update={
            "source_graphs": {"source": source_graph},
            "teachers": (base.teachers[0].model_copy(update={"graph": source_graph}),),
        }
    )
    recorder = Author()
    from tests.unit_tests.harness.evolution.fixtures import artifact

    class SourceEvaluator(Evaluator):
        async def __call__(self, request, bundle, directory, *, before_launch):
            result = await super().__call__(
                request, bundle, directory, before_launch=before_launch
            )
            return result.model_copy(
                update={"evidence": (artifact(directory, "source.json", {}),)}
            )

    asyncio.run(
        SharedEvolution(tmp_path / "search", recorder, SourceEvaluator()).run(settings)
    )
    request = recorder.requests[0]
    assert request["source_graphs"] == {}
    assert all(t["graph"] is None for t in request["teachers"])
    assert request["teachers"][0]["bundle"] == base.teachers[0].bundle.model_dump(
        mode="json"
    )
    assert request["mechanisms"] == [] and request["fixed_alignments"] == []
    author = NativeSharedAuthor(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
        runtime_manifest=SimpleNamespace(
            authoring_contract=SimpleNamespace(
                operations={}, tools=[], native_image_contract=None
            ),
            configuration={"limits": {"episode": {}}},
        ),
    )
    workspace, _ = author.prepare(request, tmp_path / "native")
    tools = SharedAuthorTools(workspace, tmp_path / "audit/no_topology.jsonl")
    assert asyncio.run(tools.catalog()) == []
    assert not (workspace / "context/teachers/000/graph.json").exists()
    assert "retrieve_graphs" not in (workspace / "AGENTS.md").read_text()
    assert "query_student_gap" not in (workspace / "AGENTS.md").read_text()
    assert "{{" not in (workspace / "AGENTS.md").read_text()
    for changed in (
        {"source_graphs": {"source": source_graph.model_dump(mode="json")}},
        {"history": [{"round": 0, "score": {"topology_records": {}}}]},
        {"student_gaps": {"hidden": True}},
    ):
        with pytest.raises(ValueError, match="w/o topology"):
            author.prepare({**request, **changed}, tmp_path / "forbidden")


def test_http_mcp_calls_work_without_model_or_live_environment(tmp_path):
    _, workspace, _, _, _ = prepared_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit/source_tools.jsonl")
    try:
        tools.server.start()
        result = call_development_tool(tools.server.url, "read_reference_graph", {})
        assert len(result["tasks"]) == 2
        result = call_development_tool(
            tools.server.url, "retrieve_graphs", {"query_task": "alpha", "top_k": 1}
        )
        assert result["matches"][0]["task"] == "alpha"
    finally:
        tools.server.stop()
    rows = [json.loads(line) for line in tools.audit.read_text().splitlines()]
    assert all(row["ok"] for row in rows)
    assert all(row["elapsed_s"] >= 0 for row in rows)


def test_raw_read_rejects_foreign_original_and_replaced_file_link(tmp_path):
    _, workspace, graphs, _, _ = prepared_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit/source_tools.jsonl")
    ref = next(iter(graphs["alpha"].original_references()))
    value = asyncio.run(tools.model_call("read_original", ref.model_dump(mode="json")))
    assert "fixture" in value["text"]
    foreign = ref.model_copy(update={"path": str(tmp_path / "unseen.json")})
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(tools.model_call("read_original", foreign.model_dump(mode="json")))
    from pathlib import Path

    outside = tmp_path / "outside.json"
    outside.write_text("unrelated evidence")
    Path(ref.path).unlink()
    Path(ref.path).symlink_to(outside)
    with pytest.raises(ValueError):
        asyncio.run(tools.model_call("read_original", ref.model_dump(mode="json")))


def test_topology_proposals_are_checked_without_mutating_current_state(tmp_path):
    _, workspace, graphs, topology, _ = prepared_workspace(tmp_path)
    tools = SharedAuthorTools(workspace, tmp_path / "audit/source_tools.jsonl")
    result = asyncio.run(
        tools.model_call(
            "validate_topology",
            {
                "merges": [
                    {
                        "left": occurrence_key("alpha", "edge0"),
                        "right": occurrence_key("beta", "edge0"),
                        "entity_mapping": {"object": "object"},
                    }
                ],
                "reason": "Common movement; preserve each source guard in its residual.",
            },
        )
    )
    assert result["source_constraints_preserved"]
    assert tools.topology == topology
    with pytest.raises((KeyError, ValueError)):
        asyncio.run(
            tools.model_call(
                "validate_topology", {"split": ["unseen"], "reason": "invalid"}
            )
        )


def test_implementation_requires_all_sources_and_resolving_executable_locations(
    tmp_path,
):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    site = {
        "pointers": ["/context/instructions"],
        "explanation": "Shared planner interprets these explicit physical constraints.",
    }
    raw = {
        "source_constraints": dict.fromkeys(graphs, site),
        "interfaces": {
            m.key: site for group in topology.mechanisms.values() for m in group.members
        },
    }
    assert validate_implementation(raw, topology, bundle("student"))
    for field in ["source_constraints", "interfaces"]:
        broken = {**raw, field: {}}
        with pytest.raises(ValueError, match="omits"):
            validate_implementation(broken, topology, bundle("student"))
    site["pointers"] = ["/record_id"]
    with pytest.raises(ValueError, match="deployed"):
        validate_implementation(raw, topology, bundle("student"))
    site["pointers"] = ["/context/absent"]
    with pytest.raises(ValueError, match="does not resolve"):
        validate_implementation(raw, topology, bundle("student"))


def test_implementation_schema_exposes_exact_host_key_contract(tmp_path):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    schema = implementation_schema(topology)
    Draft202012Validator.check_schema(schema)
    expected = {
        "source_constraints": sorted(topology.sources),
        "interfaces": sorted(
            {m.key for group in topology.mechanisms.values() for m in group.members}
        ),
    }
    site = {"pointers": ["/context/instructions"], "explanation": "Deployed code."}
    raw = {field: dict.fromkeys(keys, site) for field, keys in expected.items()}
    validator = Draft202012Validator(schema)
    assert validator.is_valid(raw)
    for field, keys in expected.items():
        contract = schema["properties"][field]
        assert contract["propertyNames"] == {"enum": keys}
        assert contract["required"] == keys
        assert "EXACTLY" in contract["description"]
        extra = {**raw, field: {**raw[field], "condition-specific-extra": site}}
        missing = {**raw, field: dict.fromkeys(keys[1:], site)}
        assert not validator.is_valid(extra)
        assert not validator.is_valid(missing)


@pytest.mark.parametrize("field", ["source_constraints", "interfaces"])
def test_implementation_errors_distinguish_missing_and_extra_exact_keys(
    tmp_path, field
):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    site = {"pointers": ["/context/instructions"], "explanation": "Deployed code."}
    raw = {
        "source_constraints": dict.fromkeys(graphs, site),
        "interfaces": {
            m.key: site for group in topology.mechanisms.values() for m in group.members
        },
    }
    expected = sorted(raw[field])
    removed = expected[0]
    raw[field].pop(removed)
    raw[field]["condition-specific-extra"] = site
    with pytest.raises(ValueError) as caught:
        validate_implementation(raw, topology, bundle("student"))
    details = json.loads(str(caught.value).split(": ", 1)[1])
    assert details == {
        "field": field,
        "missing_count": 1,
        "missing_keys_sample": [removed],
        "extra_count": 1,
        "extra_keys_sample": ["condition-specific-extra"],
        "required_keys": expected,
    }


def test_extra_condition_keys_have_truthful_bounded_diagnostics(tmp_path):
    graphs = graphs_fixture(tmp_path)
    topology, _ = initial_topology(graphs)
    site = {"pointers": ["/context/instructions"], "explanation": "Deployed code."}
    raw = {
        "source_constraints": {
            **dict.fromkeys(graphs, site),
            **{f"condition-{i:04}": site for i in range(2512)},
        },
        "interfaces": {
            m.key: site for group in topology.mechanisms.values() for m in group.members
        },
    }
    with pytest.raises(ValueError) as caught:
        validate_implementation(raw, topology, bundle("student"))
    error = str(caught.value)
    details = json.loads(error.split(": ", 1)[1])
    assert "omits" not in error
    assert details["missing_count"] == 0 and details["missing_keys_sample"] == []
    assert details["extra_count"] == 2512
    assert details["extra_keys_sample"] == [f"condition-{i:04}" for i in range(8)]
    assert details["required_keys"] == sorted(graphs)
    assert len(error) < 1000


def test_direct_author_reuses_validation_tool_without_graph_retrieval(tmp_path):
    from reweave.evolution.authoring.tools import candidate_validation_tool

    _, workspace, _, _, _ = prepared_workspace(tmp_path)
    (workspace / "context/source_graphs.json").write_text("{}")
    (workspace / "context/shared_topology.json").unlink()
    calls = []

    async def check(**options):
        calls.append(options)
        return {
            "phase": "interaction" if options.get("interactions") else "code",
            "physical_execution_tested": False,
        }

    tools = SharedAuthorTools(
        workspace, tmp_path / "audit/validation.jsonl", validator=check
    )
    catalog = asyncio.run(tools.catalog())
    assert [t.name for t in catalog] == ["validate_candidate"]
    assert catalog[0].model_dump() == candidate_validation_tool().model_dump()
    try:
        tools.server.start()
        environment = call_development_tool(
            tools.server.url, "validate_candidate", {"phase": "environment"}
        )
        assert environment["environment_ready"]
        result = call_development_tool(
            tools.server.url, "validate_candidate", {"phase": "code"}
        )
        assert result["physical_execution_tested"] is False and len(calls) == 1
        result = call_development_tool(
            tools.server.url, "validate_candidate", {"phase": "interaction"}
        )
        assert result["phase"] == "interaction"
        assert calls == [{}, {"interactions": True}]
    finally:
        tools.server.stop()
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(tools.model_call("retrieve_graphs", {}))


def test_shared_editing_check_returns_original_image_contract_error(
    tmp_path, monkeypatch
):
    from reweave.runtime.capability import ModelInputRejected

    request, workspace, _, _, author = prepared_workspace(tmp_path)

    async def reject(*args, **kwargs):
        kwargs["case_results"].append({"scope": "C", "case_index": 4})
        raise ModelInputRejected("invalid embedded base64 bytes")

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts", reject
    )
    with pytest.raises(ValueError, match="candidate_validation.json"):
        asyncio.run(author.validate_code(request, workspace))
    report = json.loads((workspace / "context/candidate_validation.json").read_text())
    assert report["error_type"] == "ModelInputRejected"
    assert report["contract_case_results"] == [{"scope": "C", "case_index": 4}]


def test_shared_interaction_preview_exposes_host_failure_before_return(
    tmp_path, monkeypatch
):
    from reweave.evolution.authoring.tools import validate_author_candidate
    from reweave.evolution.validation.interaction import InteractionFailure

    request, workspace, _, _, author = prepared_workspace(tmp_path)
    repaired = False

    async def components(*args, **kwargs):
        kwargs["case_results"].append({"scope": "C", "validation_status": "passed"})

    async def interaction(*args, **kwargs):
        kwargs["case_results"].append({"scope": "interaction"})
        assert len(kwargs["references"]) == 1 + len(request["teachers"])
        if not repaired:
            raise InteractionFailure({"errors": ["unexpected gateway call: finish"]})

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts", components
    )
    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_interaction_suite", interaction
    )

    async def preview():
        return await validate_author_candidate(
            {"phase": "interaction"},
            lambda **options: author.validate_code(request, workspace, **options),
        )

    with pytest.raises(ValueError, match="unexpected gateway call: finish"):
        asyncio.run(preview())
    report = json.loads((workspace / "context/candidate_validation.json").read_text())
    assert report["error_type"] == "InteractionFailure"
    assert [c["scope"] for c in report["contract_case_results"]] == ["C", "interaction"]
    repaired = True
    result = asyncio.run(preview())
    assert result["interaction_tested"] and result["interaction_cases"] == 1
    assert result["physical_execution_tested"] is False
    assert not (workspace.parent / "result.json").exists()
    assert not list((workspace.parent / "audit/queries").glob("*.json"))


def test_shared_author_reuses_single_task_runtime_schema_and_restores_it_on_resume(
    tmp_path,
):
    request, workspace, _, _, author = prepared_workspace(tmp_path)
    schema = workspace / "context/runtime.schema.json"
    expected = author.runtime_schema(request["conditions"]["controller_request_unit"])
    assert json.loads(schema.read_text()) == expected
    assert expected["interaction_scenario"]["required"]
    assert "model_responses" in expected["interaction_scenario"]["properties"]
    schema.unlink()  # Old shared workspaces did not receive this contract.
    recovered, _ = author.prepare(request, workspace.parent)
    assert recovered == workspace and json.loads(schema.read_text()) == expected
