# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Native graph analysis hands off complete graphs across independent round sessions."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    BundleVersion,
    EvaluationRecord,
    EvaluationRequest,
    EvidenceFile,
    GraphAnalysisRequest,
    GraphAnalysisResult,
    VersionReference,
)
from reweave.evolution.evidence.records import artifact_record, inspect_episode
from reweave.evolution.evidence.trajectory import build_graph, episode_from_record
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.robots.physical_evidence import feature
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    evolution_history,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_trace_context import _episode, _events


def analysis_request(tmp_path, *, bundle=None):
    events = _events()
    for event in events:
        if event["event_type"] == "control_step":
            event["physical_features"] = {
                "entity.drawer.distance_to_eef_m": feature(
                    0.05, "m", "synthetic drawer"
                )
            }
    _, original = _episode(
        tmp_path, events=events, task_languages=["Open the drawer"] * 4
    )
    bundle = baseline() if bundle is None else bundle
    atomic_json(
        original / "harness_context/planner_context.json",
        {"system_prompt": "complete", "tools": []},
    )
    atomic_json(original / "harness_context/capability_catalog.json", {})
    atomic_json(
        original / "harness_context/bundle_contract.json",
        {"bundle_id": bundle.record_id, "operations": {}, "episode_limits": {}},
    )
    source = original / "harness_context/source/reweave/benchmarks/executed.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic online implementation\n")
    inventory = inspect_episode(original)
    request = EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=bundle.record_id,
        instance=protocol().dev[0],
        split="dev",
        manifest_id=protocol().manifest_id,
        conditions=protocol().execution,
        execution=protocol().execution_id,
    )
    record = EvaluationRecord(
        record_id=new_record_id(),
        request=request,
        official_success=False,
        failure_origin="agent",
        infrastructure_retryable=None,
        usage=usage(),
        evidence=tuple(
            EvidenceFile(path=str(original / name), **metadata)
            for name, metadata in inventory.files.items()
        ),
        episode=inventory,
        initialization=request.instance.initialization,
    )
    receipt = artifact(tmp_path, "evaluation.json", record.model_dump(mode="json"))
    version = VersionReference(
        version_id=BundleVersion(
            record_id=f"version:{bundle.record_id}",
            bundle_id=bundle.record_id,
            proposal_id=None,
            branch_id=None,
        ).record_id,
        bundle_id=bundle.record_id,
    )
    graph = build_graph(
        [episode_from_record(record, Path(receipt.path), (version,))], protocol().graph
    )
    graph_file = artifact(tmp_path, "graph.json", graph.model_dump(mode="json"))
    request = GraphAnalysisRequest(
        previous_analysis=None,
        record_id="round_0",
        round_index=0,
        protocol=protocol(),
        graph=graph_file,
        history=evolution_history(tmp_path),
        evidence=(record,),
        versions=(version,),
        policy=AuthorPolicy().model_dump(mode="json"),
    )
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture

    result = GraphAnalysisResult(
        forecast={
            "outcomes": [{"progress": 100.0, "probability": 1.0}],
            "confidence": 0.8,
            "rationale": "Synthetic prediction; no physical efficacy claim.",
        },
        graph=semantic_fixture(graph),
        changes="Synthetic graph refinement",
    )
    return request, result


def test_graph_prompt_retains_observed_progress_without_scoring_infrastructure(
    tmp_path,
):
    from reweave.evolution.graph.projection import project_map, selected_graph_receipts
    from reweave.evolution.graph.refinement import validate_refinement
    from reweave.evolution.search.matrix import result_matrix
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture

    request, result = analysis_request(tmp_path)
    record = request.evidence[0].model_copy(
        update={
            "failure_origin": "infrastructure",
            "infrastructure_retryable": False,
        }
    )
    receipt = artifact(tmp_path, "infrastructure.json", record.model_dump(mode="json"))
    episode = episode_from_record(record, Path(receipt.path), request.versions)
    assert episode.terminal_state_known and not episode.official_success
    raw = build_graph([episode], request.protocol.graph)
    request = request.model_copy(
        update={
            "evidence": (record,),
            "graph": artifact(
                tmp_path, "infrastructure-graph.json", raw.model_dump(mode="json")
            ),
        }
    )
    result = result.model_copy(update={"graph": semantic_fixture(raw)})
    validate_refinement(request, result)
    value = project_map(
        raw,
        result.graph,
        request.versions,
        request.protocol.dev,
        cell_receipts=selected_graph_receipts(
            request.protocol,
            (record.request.bundle_id,),
            (record,),
            raw,
            execution_ids=frozenset({record.request.execution}),
        ),
    )
    observed = value["episodes"][record.record_id]
    assert observed["terminal_state_known"]
    assert 0 < observed["progress"] < 1
    assert observed["official_success"] is None
    assert all(
        result.graph.nodes[node].role != "goal" for node in observed["completed_nodes"]
    )
    # Observed graph state does not fill a missing official capability score.
    assert result_matrix(request.protocol, (record.request.bundle_id,), (record,))[
        record.request.bundle_id
    ] == (None,)
    assert value["cells"][request.versions[0].version_id] == {
        request.protocol.dev[0].record_id: record.record_id
    }
    instructions, _ = NativeGraphAnalysis._prompts(request, tmp_path)
    instructions = " ".join(instructions.split())
    assert "it does not make an acknowledged physical state unknown" in instructions
    assert (
        "retain supported non-goal achievements that still hold" in instructions.lower()
    )
    assert (
        "Unacknowledged physical actions still leave terminal state unknown"
        in instructions
    )
    assert (
        "Historical observations do not supply current-execution scores" in instructions
    )


def test_concurrent_host_preparations_share_one_episode_navigation(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from reweave.evolution.evidence import trace as trace_context

    request, _ = analysis_request(tmp_path)
    stream_rows = trace_context._iter_jsonl
    reads = []

    def counted(path):
        reads.append(path)
        return stream_rows(path)

    monkeypatch.setattr(trace_context, "_iter_jsonl", counted)
    shared = tmp_path / "evidence"
    analyzers = [
        NativeGraphAnalysis(
            bundle_loader=lambda _: baseline(),
            policy=AuthorPolicy(),
            evidence_workspace=shared,
        )
        for _ in range(2)
    ]
    workspaces = [tmp_path / f"branch-{i}/workspace" for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(author._prepare, request, workspace)
            for author, workspace in zip(analyzers, workspaces, strict=True)
        ]
        for future in futures:
            future.result()
    assert sum(path.name == "execution_trace.jsonl" for path in reads) == 1
    relative = (
        Path("context/trace") / request.evidence[0].episode.evidence_id / "index.json"
    )
    assert (workspaces[0] / ".host_inputs/evidence" / relative).samefile(
        workspaces[1] / ".host_inputs/evidence" / relative
    )
    public = (
        Path("context/trace")
        / AuthorReferences.load(workspaces[0]).identity(
            "trace", request.evidence[0].episode.evidence_id
        )
        / "index.json"
    )
    assert (workspaces[0] / public).read_bytes() == (
        workspaces[1] / public
    ).read_bytes()
    assert (workspaces[0] / ".host_inputs/evidence" / relative).samefile(
        shared / relative
    )
    assert AuthorReferences.load(workspaces[0]).read_json(
        workspaces[0] / "context/execution_graph.json"
    ) == json.loads(Path(request.graph.path).read_text())


def test_new_round_session_preserves_graph_and_same_round_corrections(
    tmp_path, monkeypatch
):
    from claude_agent_sdk import ResultMessage, SystemMessage

    request, result = analysis_request(tmp_path)
    session = str(uuid4())
    next_session = str(uuid4())
    calls = []
    author = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
        native_runtime_root=tmp_path / "global/runtime",
    )

    async def query(*, prompt, options):
        calls.append(options.resume)
        active_session = session if len(calls) <= 2 else next_session
        directory = Path(options.cwd)
        if len(calls) == 1:
            assert not (directory / "context/current_task_rules.json").exists()
            assert (
                "Current published task contract" not in options.system_prompt["append"]
            )
            assert "current_task_rules.json" not in prompt
        catalog = AuthorReferences.load(directory)
        inputs = catalog.read_json(directory / "context/input.json")
        if len(calls) == 3:
            seeded = GraphAnalysisResult.model_validate(
                catalog.read_json(directory / "analysis/result.json")
            )
            assert seeded.graph == result.graph
            assert inputs["new_episode_ids"] == []
        elif len(calls) == 1:
            assert not (directory / "analysis/result.json").exists()
            assert set(inputs["new_episode_ids"]) == set(result.graph.episodes)
        # First result is structurally invalid; correction stays in the same session.
        atomic_json(
            directory / "analysis/result.json",
            {} if len(calls) == 1 else catalog.encode(result.model_dump(mode="json")),
        )
        await options.session_store.append(
            {"session_id": active_session, "project_key": "graph"},
            [
                {
                    "uuid": str(uuid4()),
                    "type": "assistant",
                    "message": {"content": f"graph analysis {len(calls)}"},
                }
            ],
        )
        yield SystemMessage(subtype="init", data={"session_id": active_session})
        yield ResultMessage(
            subtype="success",
            duration_ms=5,
            duration_api_ms=2,
            is_error=False,
            num_turns=1,
            session_id=active_session,
            total_cost_usd=1,
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    runtime_roots = []

    def options(*args, **kwargs):
        runtime_roots.append(kwargs["native_runtime_root"])
        return {}

    monkeypatch.setattr(
        "reweave.evolution.authoring.author.claude_workspace_options", options
    )

    first = asyncio.run(
        author(request, tmp_path / "round0", before_launch=lambda: None)
    )
    with monkeypatch.context() as cached:
        cached.setattr(
            "reweave.evolution.graph.analysis.validate_analysis_output",
            lambda *args: pytest.fail("settled publication must use its saved record"),
        )
        assert asyncio.run(author._publish(tmp_path / "round0")) == first
        assert len(calls) == 2
    # An unrelated historical audit file is not needed to continue the native
    # session. Its provenance stays in the receipt; required session/graph files
    # remain available and are read through their actual consumers.
    audit_log = tmp_path / "round0/audit/events.jsonl"
    assert any(Path(item.path) == audit_log for item in first.artifacts)
    audit_log.unlink()
    frozen_bytes = (tmp_path / "round0/session/transcript.jsonl").read_bytes()
    first_file = EvidenceFile(
        **artifact_record(tmp_path / "round0/result.json"),
    )
    second_request = request.model_copy(
        update={
            "round_index": 1,
            "record_id": "round_1",
            "previous_analysis": first_file,
        }
    )
    second = asyncio.run(
        author(second_request, tmp_path / "round1", before_launch=lambda: None)
    )
    assert calls == [None, session, None]
    assert first.session_id == session
    assert second.session_id == next_session
    assert first.session_id != second.session_id
    assert runtime_roots == [tmp_path / "global/runtime"] * 3
    assert first.usage.cost_usd == 2 and second.usage.cost_usd == 1
    assert (tmp_path / "round0/session/transcript.jsonl").read_bytes() == frozen_bytes
    assert author.verify_result(request, tmp_path / "round0") == first
    assert (
        asyncio.run(
            author(
                second_request,
                tmp_path / "round1",
                before_launch=lambda: pytest.fail("cached result must not query"),
            )
        )
        == second
    )
    assert not (tmp_path / "round1/workspace/candidate").exists()
    assert (
        len(list((tmp_path / "round0/audit/corrections").glob("*/feedback.json"))) == 1
    )


def test_analysis_tools_cannot_validate_or_mutate_a_candidate(tmp_path):
    request, _ = analysis_request(tmp_path)
    author = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
        native_runtime_root=tmp_path / "global/runtime",
    )
    directory = tmp_path / "round0"
    directory.mkdir()
    author._prepare_inputs(request, directory, resume=False)
    tools = AuthorDevelopmentTools(
        directory / "workspace", directory / "audit/development.jsonl"
    )
    with pytest.raises(ValueError, match="no candidate validation authority"):
        asyncio.run(
            tools._call("validate_candidate", {"phase": "environment"}, case_results=[])
        )
    summary = asyncio.run(
        tools._call("read_graph", {"view": "summary"}, case_results=[])
    )
    assert summary["episodes"][0]["official_success"] is False
    from reweave.evolution.contracts import GraphSnapshot
    from reweave.evolution.evidence.trajectory import read_original

    raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
    reference = raw.episodes[0].frames[0].references[0]
    handle = tools.references.reference(**reference.model_dump(mode="json"))
    original = asyncio.run(
        tools.model_call("read_original", {"reference": handle}, case_results=[])
    )
    assert original["value"] == read_original(reference)
    with pytest.raises(ValueError, match="unknown author reference"):
        asyncio.run(
            tools.model_call(
                "read_original", {"reference": "file:999999#"}, case_results=[]
            )
        )
    assert not asyncio.run(
        tools._call("read_graph", {"layer": "semantic"}, case_results=[])
    )["available"]
    assert not asyncio.run(
        tools._call("read_graph", {"layer": "map"}, case_results=[])
    )["available"]
    assert not (directory / "workspace/candidate").exists()


def test_previous_complete_draft_must_cover_new_physical_evidence(tmp_path):
    from reweave.evolution.contracts import GraphAnalysisReceipt, GraphSnapshot
    from reweave.evolution.graph.refinement import validate_analysis_output
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture

    request, result = analysis_request(tmp_path)
    previous = GraphAnalysisReceipt(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=result,
        session_id="synthetic-session",
        usage=usage(),
        artifacts=(request.graph,),
    )
    new_bundle = baseline()
    new_bundle = new_bundle.model_copy(
        update={
            "record_id": "bundle:changed-context",
            "context": new_bundle.context.model_copy(
                update={"instructions": "Synthetic changed context"}
            ),
        }
    )
    (tmp_path / "incoming").mkdir()
    incoming, _ = analysis_request(tmp_path / "incoming", bundle=new_bundle)
    new_record = incoming.evidence[0]
    raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
    incoming_raw = GraphSnapshot.model_validate_json(
        Path(incoming.graph.path).read_text()
    )
    updated_raw = build_graph(
        (*raw.episodes, *incoming_raw.episodes), request.protocol.graph
    )
    request = request.model_copy(
        update={
            "round_index": 1,
            "previous_analysis": artifact(
                tmp_path, "previous-analysis.json", previous.model_dump(mode="json")
            ),
            "graph": artifact(
                tmp_path, "updated-raw.json", updated_raw.model_dump(mode="json")
            ),
            "evidence": (*request.evidence, new_record),
            "versions": (*request.versions, *incoming.versions),
        }
    )
    bundles = {baseline().record_id: baseline(), new_bundle.record_id: new_bundle}
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=bundles.__getitem__,
        policy=AuthorPolicy(),
    )
    root = tmp_path / "next-workspace"
    analyzer._prepare(request, root)
    catalog = AuthorReferences.load(root)
    draft = GraphAnalysisResult.model_validate(
        catalog.read_json(root / "analysis/result.json")
    )
    assert draft.graph == result.graph
    assert catalog.read_json(root / "context/input.json")["new_episode_ids"] == [
        new_record.record_id
    ]
    with pytest.raises(ValueError, match="every historical episode"):
        validate_analysis_output(request, root / "analysis")
    result = GraphAnalysisResult(
        forecast={
            "outcomes": [{"progress": 100.0, "probability": 1.0}],
            "confidence": 0.8,
            "rationale": "Synthetic prediction; no physical efficacy claim.",
        },
        graph=semantic_fixture(updated_raw),
        changes="Include the new evidence",
    )
    atomic_json(
        root / "analysis/result.json", catalog.encode(result.model_dump(mode="json"))
    )
    assert validate_analysis_output(request, root / "analysis") == result


@pytest.mark.parametrize("purpose", ["search", "final_selection"])
def test_untouched_scaffold_is_not_completed_analysis_without_new_evidence(
    tmp_path, purpose
):
    from reweave.evolution.contracts import GraphAnalysisReceipt
    from reweave.evolution.graph.refinement import validate_analysis_output

    request, result = analysis_request(tmp_path)
    previous = GraphAnalysisReceipt(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=result,
        session_id="synthetic-session",
        usage=usage(),
        artifacts=(request.graph,),
    )
    request = request.model_copy(
        update={
            "purpose": purpose,
            "round_index": 1,
            "previous_analysis": artifact(
                tmp_path, "previous-analysis.json", previous.model_dump(mode="json")
            ),
        }
    )
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    root = tmp_path / "next-workspace"
    analyzer._prepare(request, root)
    catalog = AuthorReferences.load(root)
    with pytest.raises(ValueError, match="pending graph analysis"):
        validate_analysis_output(request, root / "analysis")

    reviewed = result.model_copy(
        update={
            "changes": "Reviewed unchanged evidence; retain the graph.",
            "forecast": result.forecast if request.needs_forecast else None,
        }
    )
    atomic_json(
        root / "analysis/result.json", catalog.encode(reviewed.model_dump(mode="json"))
    )
    admitted = validate_analysis_output(request, root / "analysis")
    assert admitted == reviewed
    assert admitted.graph == previous.analysis.graph
    assert admitted.forecast == (
        previous.analysis.forecast if request.needs_forecast else None
    )


def test_development_program_view_reaches_actual_code_and_compact_map(tmp_path):
    from reweave.evolution.contracts import GraphSnapshot
    from reweave.evolution.graph.projection import project_map
    from reweave.evolution.graph.refinement import validate_refinement

    request, result = analysis_request(tmp_path)
    bundle = artifact(
        tmp_path, "actual-bundle.json", baseline().model_dump(mode="json")
    )
    request = request.model_copy(
        update={
            "history": evolution_history(
                tmp_path,
                originals={
                    bundle.path: {
                        "artifact_id": bundle.artifact_id,
                        "bytes": bundle.bytes,
                    }
                },
            )
        }
    )
    payload = result.model_dump(mode="json")
    payload["graph"]["programs"]["feedback"] = {
        "bundle": {**bundle.model_dump(mode="json"), "pointer": ""},
        "pointers": ["/context", "/loop"],
        "mechanism": "Synthetic observed Planner feedback loop",
        "uncertainty": "No causal or isolation claim",
    }
    for edge in payload["graph"]["edges"].values():
        for witness in edge["witnesses"]:
            witness["program_id"] = "feedback"
    result = GraphAnalysisResult.model_validate(payload)
    validate_refinement(request, result)
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    root = tmp_path / "workspace"
    analyzer._prepare(request, root)
    raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
    value = project_map(
        raw, result.graph, request.versions, request.protocol.dev, partial=True
    )
    catalog = AuthorReferences.load(root)
    atomic_json(
        root / "context/graph.json",
        catalog.encode(result.graph.model_dump(mode="json")),
    )
    atomic_json(root / "context/progress_map.json", catalog.encode(value))
    tools = AuthorDevelopmentTools(root, tmp_path / "development.jsonl")

    def read(**kwargs):
        return asyncio.run(tools.model_call("read_graph", kwargs, case_results=[]))

    program = read(layer="semantic", view="program", identity="feedback")
    assert program["edges"]
    source = asyncio.run(
        tools.model_call(
            "read_original",
            {"reference": program["bundle"].split("#")[0] + "#/loop"},
            case_results=[],
        )
    )
    assert source["value"] == baseline().loop.model_dump(mode="json")
    summary = read(layer="map")
    assert summary["program_ids"] == ["feedback"] and "map" not in summary
    assert read(layer="map", view="program", identity="feedback")[
        "bundle_id"
    ] == catalog.identity("bundle", baseline().record_id)


@pytest.mark.parametrize("conflicting_inventory", [False, True])
def test_migrated_graph_keeps_published_originals_and_recovers_preparation(
    tmp_path, monkeypatch, conflicting_inventory
):
    from reweave.evolution.authoring import author as bundle_author
    from reweave.evolution.contracts import GraphAnalysisReceipt

    request, result = analysis_request(tmp_path)
    old_bundle = artifact(
        tmp_path / "old-workspace/bundles",
        "bundle.json",
        baseline().model_dump(mode="json"),
    )
    copied_bundle = artifact(
        tmp_path / "new-workspace/bundles",
        "bundle.json",
        baseline().model_dump(mode="json"),
    )
    payload = result.model_dump(mode="json")
    payload["graph"]["programs"]["feedback"] = {
        "bundle": {**old_bundle.model_dump(mode="json"), "pointer": ""},
        "pointers": ["/context", "/loop"],
        "mechanism": "Synthetic historical Planner feedback loop",
        "uncertainty": "Fixture provenance, no physical efficacy claim",
    }
    for edge in payload["graph"]["edges"].values():
        for witness in edge["witnesses"]:
            witness["program_id"] = "feedback"
    result = GraphAnalysisResult.model_validate(payload)
    previous_directory = tmp_path / "previous-attempt"
    session_id = str(uuid4())
    previous = GraphAnalysisReceipt(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=result,
        session_id=session_id,
        usage=usage(),
        artifacts=(
            artifact(
                previous_directory / "workspace/analysis",
                "result.json",
                result.model_dump(mode="json"),
            ),
        ),
    )
    receipt = artifact(
        previous_directory, "result.json", previous.model_dump(mode="json")
    )
    atomic_json(
        previous_directory / "session/identity.json", {"session_id": session_id}
    )
    journal = previous_directory / "session/transcript.jsonl"
    journal.write_text('{"fixture": "unchanged prior graph journal"}\n')
    originals = {
        copied_bundle.path: {
            "artifact_id": copied_bundle.artifact_id,
            "bytes": copied_bundle.bytes,
        }
    }
    if conflicting_inventory:
        originals[old_bundle.path] = "f" * 64
    request = request.model_copy(
        update={
            "round_index": 1,
            "previous_analysis": receipt,
            "history": evolution_history(
                tmp_path / "new-workspace/history", originals=originals
            ),
        }
    )
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "new-workspace/evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    directory = tmp_path / "new-workspace/attempts/round-02/graph-analysis"
    if conflicting_inventory:
        with pytest.raises(ValueError, match="published graph original differs"):
            analyzer._prepare_inputs(request, directory, resume=False)
        return

    # Reproduce the previous inventory omission, then recover this exact attempt.
    originals_inventory = bundle_author.history_originals

    def omitted_graph_originals(value):
        history = originals_inventory(value)
        history["files"].pop(old_bundle.path, None)
        return history

    with monkeypatch.context() as missing:
        missing.setattr(bundle_author, "history_originals", omitted_graph_originals)
        with pytest.raises(ValueError, match="outside this request's inventory"):
            analyzer._prepare_inputs(request, directory, resume=False)
    staging = directory / ".scratch/runtime/preparation"
    failed_catalog = (staging / ".host_inputs/author_references.json").read_bytes()
    assert not (directory / "workspace").exists()
    analyzer._prepare_inputs(request, directory, resume=True)
    preserved = list((directory / "audit/preparation_failures").glob("*"))
    assert len(preserved) == 1
    assert (
        preserved[0] / ".host_inputs/author_references.json"
    ).read_bytes() == failed_catalog
    workspace = directory / "workspace"
    catalog = AuthorReferences.load(workspace)
    assert catalog.read_json(
        workspace / "context/graph.json"
    ) == result.graph.model_dump(mode="json")
    assert catalog.originals[old_bundle.path] == {
        "artifact_id": old_bundle.artifact_id,
        "bytes": old_bundle.bytes,
    }
    assert not (directory / "session").exists()
    session_id = str(uuid4())
    atomic_json(directory / "session/identity.json", {"session_id": session_id})
    current_journal = directory / "session/transcript.jsonl"
    current_journal.write_text('{"fixture": "new round graph journal"}\n')
    tools = AuthorDevelopmentTools(workspace, directory / "audit/development.jsonl")
    original = asyncio.run(
        tools.model_call(
            "read_original",
            {
                "reference": catalog.encode(
                    {
                        **old_bundle.model_dump(mode="json"),
                        "pointer": "",
                    }
                )
            },
            case_results=[],
        )
    )
    assert original["value"] == baseline().model_dump(mode="json")
    old_receipt_bytes = Path(receipt.path).read_bytes()
    atomic_json(
        workspace / "analysis/result.json",
        catalog.encode(result.model_dump(mode="json")),
    )
    checked = asyncio.run(tools.model_call("validate_graph", {}, case_results=[]))
    assert checked["structurally_valid"] and not checked["physical_execution_tested"]
    # Synthetic completion metadata lets the CPU test exercise the real finalizer;
    # no native SDK, provider, physical episode, or authentication is used.
    no_model_usage = usage().model_copy(
        update={"physical_steps": 0, "model_calls": 0, "wall_time_s": 0, "cost_usd": 0}
    )
    atomic_json(
        directory / "audit/queries/000000.json", no_model_usage.model_dump(mode="json")
    )
    bundle_author.native_completion(
        directory, no_model_usage, session_id, {"fixture": "CPU-only completion"}
    )
    published = asyncio.run(analyzer._publish(directory))
    assert published.analysis == result and published.session_id == session_id
    assert published.usage.physical_steps == published.usage.model_calls == 0
    assert analyzer.verify_result(request, directory) == published
    assert Path(receipt.path).read_bytes() == old_receipt_bytes
    assert (
        directory / "session/transcript.jsonl"
    ).read_bytes() == current_journal.read_bytes()
    assert journal.read_text() == '{"fixture": "unchanged prior graph journal"}\n'
    unpublished = artifact(tmp_path / "unpublished", "bundle.json", {"unused": True})
    with pytest.raises(ValueError, match="outside this request's inventory"):
        catalog.encode({**unpublished.model_dump(mode="json"), "pointer": ""})
    from reweave.evolution.graph.refinement import validate_refinement

    invalid = result.model_dump(mode="json")
    invalid["graph"]["programs"]["feedback"]["bundle"] = {
        **unpublished.model_dump(mode="json"),
        "pointer": "",
    }
    with pytest.raises(ValueError, match="outside the frozen originals"):
        validate_refinement(request, GraphAnalysisResult.model_validate(invalid))


@pytest.mark.parametrize(
    "current_outcome",
    ["agent", "infrastructure", "initialization", "unknown", "unmeasured"],
)
def test_graph_tool_and_publication_select_current_execution_with_retest_history(
    tmp_path, current_outcome
):
    from reweave.evolution.authoring import author as bundle_author
    from reweave.evolution.authoring.author import NativeBundleAuthor
    from reweave.evolution.authoring.candidate import read_proposal
    from reweave.evolution.contracts import (
        BudgetRevision,
        ExecutionRevision,
        GraphRound,
        Proposal,
    )
    from reweave.evolution.graph.projection import project_map, selected_graph_receipts
    from reweave.evolution.graph.selection import forecast_context, select_parent
    from reweave.evolution.runtime_manifest import (
        RuntimeAuthoringContract,
        RuntimeManifest,
        capture_baseline,
    )
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
    from reweave.evolution.search.matrix import evaluation_cells
    from reweave.evolution.search.report import publish_report
    from reweave.planner.bundle_runtime import BundleRuntime
    from reweave.runtime.kernel import EpisodeLimits
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture
    from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture

    old_workspace = tmp_path / "old-workspace"
    old_workspace.mkdir()
    request, result = analysis_request(old_workspace)
    historical = request.evidence[0]
    uncapped = request.protocol.model_copy(
        update={
            "record_id": "protocol:uncapped",
            "execution_id": "execution:uncapped",
            "execution": request.protocol.execution.model_copy(
                update={"wall_time_s": None}
            ),
        }
    )
    actual = fixture(tmp_path / "active")
    capture_baseline(tmp_path / "active/baseline", **actual)
    manifest = RuntimeManifest.model_validate_json(
        (tmp_path / "active/baseline/runtime.json").read_text()
    )
    limits = EpisodeLimits(program_calls=20, gateway_calls=20, wall_time_s=None)
    runtime = BundleRuntime(
        baseline(), actual["toolkit"], "CPU authoring fixture", episode_limits=limits
    )
    configuration = manifest.configuration.copy()
    configuration["limits"] = {"episode": limits.model_dump(mode="json")}
    manifest = manifest.model_copy(
        update={
            "conditions": uncapped.execution,
            "configuration": configuration,
            "authoring_contract": RuntimeAuthoringContract.from_runtime(runtime),
        }
    )
    current_protocol = uncapped.model_copy(
        update={
            "record_id": "protocol:active-runtime",
            "execution_id": "execution:active-runtime",
            "execution_runtime": manifest.runtime_configuration(),
        }
    )
    manifest = manifest.model_copy(
        update={
            "execution_revision": ExecutionRevision(
                previous=uncapped,
                current=current_protocol,
                reason="Synthetic CPU environment revision, no physical claim.",
            )
        }
    )

    def measured_record(name, execution, *, unacknowledged=False):
        source = tmp_path / name
        source.mkdir()
        measured, _ = analysis_request(source)
        record = measured.evidence[0]
        context = next(
            Path(item.path)
            for item in record.evidence
            if Path(item.path).name == "planner_context.json"
        )
        # Each synthetic attempt has its own complete immutable evidence archive.
        atomic_json(
            context, {**json.loads(context.read_text()), "fixture_attempt": name}
        )
        original = context.parent.parent
        if unacknowledged:
            trace = original / "execution_trace.jsonl"
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            events.append(
                {
                    "event_type": "action_outcome_unknown",
                    "event_index": len(events),
                    "execution_ref": "release",
                }
            )
            trace.write_text("".join(json.dumps(event) + "\n" for event in events))
        inventory = inspect_episode(original)
        return record.model_copy(
            update={
                "request": record.request.model_copy(
                    update={
                        "execution": execution,
                        "conditions": uncapped.execution
                        if execution == uncapped.execution_id
                        else current_protocol.execution,
                    }
                ),
                "episode": inventory,
                "evidence": tuple(
                    EvidenceFile(path=str(original / path), **metadata)
                    for path, metadata in inventory.files.items()
                ),
            }
        )

    retest = measured_record("old-budget-retest", uncapped.execution_id)
    initialization_failure = historical.model_copy(
        update={
            "record_id": "receipt:failed-initialization",
            "failure_origin": "infrastructure",
            "infrastructure_retryable": True,
            "episode": None,
            "usage": usage().model_copy(update={"physical_steps": 0, "model_calls": 0}),
            "evidence": (
                artifact(
                    tmp_path / "failed-initialization",
                    "failure.json",
                    {"fixture": "initialization failed before physical execution"},
                ),
            ),
        }
    )
    assert initialization_failure.request == historical.request
    records = [initialization_failure, historical, retest]
    if current_outcome == "initialization":
        records.append(
            initialization_failure.model_copy(
                update={
                    "record_id": "receipt:current-failed-initialization",
                    "request": initialization_failure.request.model_copy(
                        update={
                            "record_id": "request:current-failed-initialization",
                            "execution": current_protocol.execution_id,
                            "conditions": current_protocol.execution,
                        }
                    ),
                    "infrastructure_retryable": False,
                }
            )
        )
    elif current_outcome != "unmeasured":
        records.append(
            measured_record(
                "current-execution",
                current_protocol.execution_id,
                unacknowledged=current_outcome == "unknown",
            ).model_copy(
                update={
                    "failure_origin": current_outcome,
                    "infrastructure_retryable": False
                    if current_outcome == "infrastructure"
                    else None,
                }
            )
        )
    receipts = [
        artifact(tmp_path / f"execution-{i}", "result.json", r.model_dump(mode="json"))
        for i, r in enumerate(records)
    ]
    receipt_bytes = [Path(r.path).read_bytes() for r in receipts]
    raw = build_graph(
        [
            episode_from_record(record, Path(receipt.path), request.versions)
            for record, receipt in zip(records, receipts, strict=True)
            if record.episode is not None
        ],
        current_protocol.graph,
    )
    request = request.model_copy(
        update={
            "protocol": current_protocol,
            "graph": artifact(
                tmp_path / "new-workspace", "graph.json", raw.model_dump(mode="json")
            ),
            "evidence": tuple(records),
        }
    )
    result = result.model_copy(update={"graph": semantic_fixture(raw)})
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "new-workspace/evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
        runtime_manifest=manifest,
    )
    directory = tmp_path / "new-workspace/attempts/round-01/graph-analysis"
    analyzer._prepare_inputs(request, directory, resume=False)
    workspace = directory / "workspace"
    catalog = AuthorReferences.load(workspace)
    atomic_json(
        workspace / "analysis/result.json",
        catalog.encode(result.model_dump(mode="json")),
    )
    tools = AuthorDevelopmentTools(workspace, directory / "audit/development.jsonl")
    checked = asyncio.run(tools.model_call("validate_graph", {}, case_results=[]))
    assert checked["structurally_valid"] and not checked["physical_execution_tested"]

    selected = selected_graph_receipts(
        request.protocol,
        (version.bundle_id for version in request.versions),
        request.evidence,
        raw,
        execution_ids=frozenset(r.request.execution for r in request.evidence),
    )
    projection = project_map(
        raw,
        result.graph,
        request.versions,
        request.protocol.dev,
        cell_receipts=selected,
    )
    key = (historical.request.bundle_id, historical.request.instance.record_id)
    selected_receipt = (
        records[-1].record_id
        if current_outcome in {"agent", "infrastructure", "unknown"}
        else None
    )
    assert (
        projection["cells"][request.versions[0].version_id][key[1]] == selected_receipt
    )
    assert set(projection["episodes"]) == {
        r.record_id for r in records if r.episode is not None
    }
    assert projection["episodes"][historical.record_id]["progress"] > 0
    expected_forecast = forecast_context(
        projection, select_parent(projection, request.versions)
    )
    decoded_check = AuthorReferences.load(workspace).decode(checked)
    assert all(
        decoded_check["forecast_context"][key] == value
        for key, value in expected_forecast.items()
    )
    if current_outcome in {"initialization", "unknown", "unmeasured"}:
        assert expected_forecast["current_progress_lower"] == 0
        assert expected_forecast["current_progress_upper"] == 100
    else:
        assert 0 < expected_forecast["current_progress_lower"] < 100
        assert (
            expected_forecast["current_progress_lower"]
            == expected_forecast["current_progress_upper"]
        )
    if current_outcome == "infrastructure":
        observed = projection["episodes"][selected_receipt]
        assert observed["terminal_state_known"]
        assert observed["official_success"] is None
    if current_outcome == "unknown":
        observed = projection["episodes"][selected_receipt]
        assert not observed["terminal_state_known"]
        assert observed["progress"] is None and observed["peak_progress"] > 0

    # Explicit CPU-only completion fixture exercises final publication without
    # making a native/model/physical request or claiming physical success.
    session_id = str(uuid4())
    no_model_usage = usage().model_copy(
        update={"physical_steps": 0, "model_calls": 0, "wall_time_s": 0, "cost_usd": 0}
    )
    atomic_json(
        directory / "audit/queries/000000.json", no_model_usage.model_dump(mode="json")
    )
    bundle_author.native_completion(
        directory, no_model_usage, session_id, {"fixture": "CPU-only completion"}
    )
    published = asyncio.run(analyzer._publish(directory))
    assert published.analysis == result and published.session_id == session_id
    assert published.usage.physical_steps == published.usage.model_calls == 0
    assert analyzer.verify_result(request, directory) == published
    following = request.model_copy(
        update={
            "previous_analysis": EvidenceFile(
                **artifact_record(directory / "result.json"),
            )
        }
    )
    prepared_following = tmp_path / "following-workspace"
    analyzer._prepare(following, prepared_following)
    previous_map = AuthorReferences.load(prepared_following).read_json(
        prepared_following / "context/progress_map.json"
    )
    assert previous_map["cells"][request.versions[0].version_id].get(key[1]) == (
        selected_receipt
    )
    assert [Path(r.path).read_bytes() for r in receipts] == receipt_bytes
    assert actual["toolkit"].calls == 0 and not actual["toolkit"].terminal

    # Resume a genuinely completed analysis through production proposal creation
    # and each author's input preparation; no analyzer/evaluator may launch.
    class CachedAnalysis(NativeGraphAnalysis):
        async def __call__(self, *args, **kwargs):
            raise AssertionError("completed graph analysis must be reused")

    async def forbidden_evaluator(*args, **kwargs):
        raise AssertionError("proposal preparation must not evaluate a candidate")

    continuation = EvolutionWorkspace(
        tmp_path / "continuation", baseline(), current_protocol
    )
    atomic_json(
        continuation.root / "configuration.json",
        {
            "protocol": current_protocol.model_dump(mode="json"),
            "runtime_manifest": manifest.model_dump(mode="json"),
        },
    )
    continuation.put_bundle(baseline())
    author = NativeBundleAuthor(
        evidence_workspace=continuation.root / "evidence",
        bundle_loader=continuation.bundle,
        policy=AuthorPolicy(),
        runtime_manifest=manifest,
    )
    cached = CachedAnalysis(
        evidence_workspace=continuation.root / "evidence",
        bundle_loader=continuation.bundle,
        policy=AuthorPolicy(),
        runtime_manifest=manifest,
    )
    evolution = GraphEvolution(
        continuation, author, forbidden_evaluator, analyzer=cached, recover=True
    )
    with continuation.transaction() as state:
        continuation.add_version(
            state,
            BundleVersion(
                record_id=request.versions[0].version_id,
                bundle_id=baseline().record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        state["budget_revisions"] = [
            BudgetRevision(
                previous=protocol(),
                current=uncapped,
            ).model_dump(mode="json")
        ]
        state["execution_revisions"] = [
            manifest.execution_revision.model_dump(mode="json")
        ]
        state["rounds"]["0"] = GraphRound(
            request=request, decision=None, outcome=None, construction_wall_time_s=0
        ).model_dump(mode="json")
        for index, (record, receipt) in enumerate(zip(records, receipts, strict=True)):
            key = f"evaluation/attempt-{index}"
            state["attempts"][key] = {
                "key": key,
                "kind": "evaluation",
                "status": "completed",
                "request": record.request.model_dump(mode="json"),
                "launches": 1,
                "deadline_at": None,
                "result_id": record.record_id,
                "result_directory": str(Path(receipt.path).parent),
            }
        key = "round-01/graph-analysis"
        state["attempts"][key] = {
            "key": key,
            "kind": "graph_analysis",
            "status": "completed",
            "request": request.model_dump(mode="json"),
            "launches": 1,
            "deadline_at": 1.0,
            "result_id": published.record_id,
            "result_directory": str(directory),
        }
        continuation.save(state)
        rows = asyncio.run(evolution.matrix(state))
        proposals = asyncio.run(evolution._proposals(state, rows, 0))
        assert proposals
        for proposal in proposals:
            assert proposal.evidence == tuple(records)
            with pytest.raises(ValueError, match="duplicate evaluation evidence"):
                Proposal.model_validate(
                    {
                        **proposal.model_dump(mode="json"),
                        "evidence": [
                            *(r.model_dump(mode="json") for r in records),
                            records[0].model_dump(mode="json"),
                        ],
                    }
                )
            author_directory = continuation.root / "prepared" / proposal.record_id
            author._prepare_inputs(proposal, author_directory, resume=False)
            prepared = author_directory / "workspace"
            assert read_proposal(prepared) == proposal
            assert (prepared / "candidate").is_dir()
            reference_map = AuthorReferences.load(prepared).read_json(
                prepared / "context/progress_map.json"
            )
            assert reference_map["cells"] == projection["cells"]
            assert set(reference_map["episodes"]) == set(projection["episodes"])
        publish_report(continuation, state, records, evolution.accounting(state))
        report = json.loads((continuation.root / "progress.json").read_text())
        if current_outcome in {"infrastructure", "initialization", "unmeasured"}:
            score = report["scores"][state["order"][0]]
            assert score["success_rate"] is None and not score["complete"]
            assert "baseline_delta" not in score
        decision = json.loads(Path(proposals[0].decision.path).read_text())
        if current_outcome in {"infrastructure", "initialization", "unmeasured"}:
            assert decision["statistics"]["parent_success_rate"] is None
        repeated_valid = historical.model_copy(
            update={"usage": historical.usage.model_copy(update={"wall_time_s": 2})}
        )
        with pytest.raises(ValueError, match="duplicate settled matrix cell"):
            evaluation_cells(
                request.protocol,
                (baseline().record_id,),
                (*records, repeated_valid),
                execution_ids=continuation.execution_ids(state),
            )
    assert [Path(r.path).read_bytes() for r in receipts] == receipt_bytes
    assert actual["toolkit"].calls == 0 and not actual["toolkit"].terminal


def test_semantic_names_resembling_registered_handles_remain_literal(tmp_path):
    request, _ = analysis_request(tmp_path)
    author = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    root = tmp_path / "workspace"
    author._prepare(request, root)
    catalog = AuthorReferences.load(root)
    literal = catalog.identity("partition", "f" * 64)
    graph = {
        "nodes": {
            literal: {
                "meaning": "Synthetic unobserved evaluator goal",
                "role": "goal",
                "predecessors": [],
                "references": [],
                "uncertainty": "Not observed",
            }
        },
        "edges": {},
        "programs": {},
        "episodes": {},
        "unclassified": [],
    }
    atomic_json(root / "context/graph.json", catalog.encode(graph))
    tools = AuthorDevelopmentTools(root, tmp_path / "development.jsonl")
    selected = asyncio.run(
        tools.model_call(
            "read_graph",
            {"layer": "semantic", "view": "node", "identity": literal},
            case_results=[],
        )
    )
    assert selected["node_id"] == literal
    assert selected["meaning"] == graph["nodes"][literal]["meaning"]
    assert (
        tools._model_arguments(
            "read_graph", {"layer": "map", "view": "transitions", "identity": literal}
        )["identity"]
        == literal
    )


def test_graph_draft_check_uses_publication_gate_after_workspace_move(tmp_path):
    from reweave.evolution.authoring.tools import call_development_tool
    from reweave.evolution.graph.refinement import validate_analysis_output

    request, result = analysis_request(tmp_path)
    author = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
        native_runtime_root=tmp_path / "global/runtime",
    )
    directory = tmp_path / "round0"
    directory.mkdir()
    author._prepare_inputs(request, directory, resume=False)
    root = directory / "workspace"
    output = root / "analysis"
    catalog = AuthorReferences.load(root)
    atomic_json(output / "result.json", catalog.encode(result.model_dump(mode="json")))
    tools = AuthorDevelopmentTools(root, directory / "audit/development.jsonl")
    tools.server.start()
    try:
        checked = subprocess.run(
            [sys.executable, "context/dev.py", "validate-graph"],
            cwd=root,
            env={**os.environ, "REWEAVE_AUTHOR_TOOLS_URL": tools.server.url},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert checked.returncode == 0, checked.stderr
        receipt = json.loads(checked.stdout)
        assert receipt["structurally_valid"] and not receipt["semantics_verified"]
        assert receipt["forecast_context"]["parent_version_id"] == catalog.identity(
            "version", request.versions[0].version_id
        )
        assert receipt["forecast_context"]["conditions"] == [
            catalog.identity("condition", request.protocol.dev[0].record_id)
        ]
        assert (
            receipt["forecast_context"]["author_wall_time_s"]
            == request.protocol.author_wall_time_s
        )
        full_report = call_development_tool(
            tools.server.url, "read_original", {"reference": receipt["complete_report"]}
        )["value"]
        assert full_report["result"]["structurally_valid"] is True
        assert validate_analysis_output(request, output) == result
        (output / "extra.json").write_text("{}")
        failed = tools._call("validate_graph", {}, case_results=[])
        with pytest.raises(ValueError, match="exactly result.json"):
            asyncio.run(failed)
        with pytest.raises(ValueError, match="exactly result.json"):
            validate_analysis_output(request, output)
        (output / "extra.json").unlink()
        reference = next(
            n.references[0] for n in result.graph.nodes.values() if n.references
        )
        original = Path(reference.path)
        lines = original.read_text().splitlines()
        lines[int(reference.pointer.split("/")[1])] = "{"
        original.write_text("\n".join(lines) + "\n")
        with pytest.raises(RuntimeError, match="error_type"):
            call_development_tool(tools.server.url, "validate_graph", {})
    finally:
        tools.server.stop(timeout_s=None)


def test_forecast_distribution_derives_mean_without_serializing_input_alias():
    from reweave.evolution.contracts import ProgressForecast

    forecast = ProgressForecast(
        outcomes=(
            {"progress": 20.0, "probability": 0.2},
            {"progress": 60.0, "probability": 0.6},
            {"progress": 100.0, "probability": 0.2},
        ),
        confidence=0.6,
        rationale="Regression and improvement tails remain plausible; retention dominates.",
    )
    assert forecast.expected_progress == pytest.approx(60.0)
    assert "expected_progress" not in forecast.model_dump()
    assert "expected_progress" not in ProgressForecast.model_json_schema()["properties"]
    assert (
        forecast.model_copy(update={"confidence": 0.0}).expected_progress
        == forecast.expected_progress
    )
    assert ProgressForecast.model_validate_json(forecast.model_dump_json()) == forecast


@pytest.mark.parametrize(
    "outcomes",
    [
        [],
        [{"progress": 50.0, "probability": 0.0}],
        [{"progress": 50.0, "probability": -0.1}],
        [{"progress": 50.0, "probability": 1.1}],
        [{"progress": -1.0, "probability": 1.0}],
        [{"progress": 101.0, "probability": 1.0}],
        [{"progress": float("nan"), "probability": 1.0}],
        [{"progress": float("inf"), "probability": 1.0}],
        [{"progress": 50.0, "probability": float("nan")}],
        [{"progress": 50.0, "probability": float("inf")}],
        [{"progress": 50.0, "probability": 0.5}],
        [
            {"progress": 50.0, "probability": 0.5},
            {"progress": 50.0, "probability": 0.5},
        ],
        [
            {"progress": 60.0, "probability": 0.5},
            {"progress": 20.0, "probability": 0.5},
        ],
        [
            {"progress": 20.0, "probability": 0.5},
            {"progress": 60.0, "probability": 0.499999998},
        ],
        [{"progress": 50.0, "probability": 1.0, "confidence": 0.5}],
    ],
)
def test_forecast_rejects_invalid_candidate_distribution(outcomes):
    from pydantic import ValidationError

    from reweave.evolution.contracts import ProgressForecast

    with pytest.raises(ValidationError):
        ProgressForecast(
            outcomes=outcomes, confidence=0.6, rationale="Synthetic evidence"
        )


@pytest.mark.parametrize("problem", ["legacy_mean", "missing_confidence", "extra_mean"])
def test_forecast_requires_distribution_and_audited_confidence(problem):
    from pydantic import ValidationError

    from reweave.evolution.contracts import ProgressForecast

    payload = {
        "outcomes": [{"progress": 50.0, "probability": 1.0}],
        "confidence": 0.5,
        "rationale": "Synthetic evidence",
    }
    if problem == "legacy_mean":
        del payload["outcomes"]
        payload["expected_progress"] = 50.0
    elif problem == "missing_confidence":
        del payload["confidence"]
    else:
        payload["expected_progress"] = 50.0
    with pytest.raises(ValidationError):
        ProgressForecast.model_validate(payload)


def test_forecast_accepts_rounding_tolerance_without_inventing_probability_mass():
    from reweave.evolution.contracts import ProgressForecast

    probabilities = (0.5, 0.4999999995)
    forecast = ProgressForecast(
        outcomes=(
            {"progress": 20.0, "probability": probabilities[0]},
            {"progress": 80.0, "probability": probabilities[1]},
        ),
        confidence=0.5,
        rationale="Rounded subjective probabilities",
    )
    assert tuple(point.probability for point in forecast.outcomes) == probabilities


@pytest.mark.parametrize(
    "field, value",
    [
        ("author_cost_usd", 0.0),
        ("rollout_cost_usd", -1.0),
        ("success_value", 0.0),
        ("prior_strength", 0.0),
        ("success_value", float("inf")),
        ("prior_strength", float("nan")),
    ],
)
def test_search_policy_requires_finite_positive_planning_values(field, value):
    from pydantic import ValidationError

    from reweave.evolution.contracts import SearchPolicy

    with pytest.raises(ValidationError):
        SearchPolicy.model_validate({field: value})
    assert SearchPolicy.model_validate({"rollout_cost_usd": 0.0}).rollout_cost_usd == 0
