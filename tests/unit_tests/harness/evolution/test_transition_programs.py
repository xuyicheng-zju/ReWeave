# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Executable provenance, physical continuation and compact graph consumption."""

import copy
import json
from pathlib import Path

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    GraphAnalysisResult,
    OriginalReference,
    SemanticGraph,
    VersionReference,
)
from reweave.evolution.evidence.trajectory import (
    build_graph,
    episode_from_record,
    read_original,
)
from reweave.evolution.graph.projection import map_cell, map_row, project_map
from reweave.evolution.graph.views import query_map, query_semantic
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    evolution_history,
)
from tests.unit_tests.harness.evolution.test_semantic_graph import (
    case,
    mapped,
    validate,
)


def program_case(tmp_path, paths=(("start", "held", "open"),) * 2):
    """Bind scripted semantic attempts to archived, schema-valid executed Bundles."""
    request, _, graph = case(tmp_path, paths=paths)
    payload = graph.model_dump(mode="json")
    records, versions, episodes, originals, remap = [], [], [], {}, {}
    for index, record in enumerate(request.evidence):
        bundle = baseline()
        bundle = bundle.model_copy(
            update={
                "record_id": f"bundle:program:{index}",
                "context": bundle.context.model_copy(
                    update={"instructions": f"Synthetic version {index}"}
                ),
            }
        )
        source = artifact(
            tmp_path / f"program-{index}", "bundle.json", bundle.model_dump(mode="json")
        )
        originals[source.path] = {
            "artifact_id": source.artifact_id,
            "bytes": source.bytes,
        }
        version = VersionReference(
            version_id=f"version:{index}", bundle_id=bundle.record_id
        )
        updated = record.model_copy(
            update={
                "record_id": f"evaluation:program:{index}",
                "request": record.request.model_copy(
                    update={
                        "record_id": f"request:program:{index}",
                        "bundle_id": bundle.record_id,
                    }
                ),
            }
        )
        receipt = artifact(
            tmp_path / f"program-{index}",
            "evaluation.json",
            updated.model_dump(mode="json"),
        )
        remap[record.record_id] = updated.record_id
        records.append(updated)
        versions.append(version)
        episodes.append(episode_from_record(updated, Path(receipt.path), (version,)))
        payload["programs"][f"program-{index}"] = {
            "bundle": {**source.model_dump(mode="json"), "pointer": ""},
            "pointers": ["/loop"],
            "mechanism": "Synthetic Planner-mediated feedback loop",
            "uncertainty": "Recorded provenance does not establish isolated causal sufficiency",
        }
    payload["episodes"] = {
        remap[key]: spans for key, spans in payload["episodes"].items()
    }
    for edge in payload["edges"].values():
        for witness in edge["witnesses"]:
            index = next(
                i
                for i, record in enumerate(request.evidence)
                if record.record_id == witness["episode_id"]
            )
            witness.update(
                episode_id=remap[witness["episode_id"]], program_id=f"program-{index}"
            )
    raw = build_graph(episodes, request.protocol.graph)
    request = request.model_copy(
        update={
            "graph": artifact(
                tmp_path, "bound-graph.json", raw.model_dump(mode="json")
            ),
            "history": evolution_history(tmp_path, originals=originals),
            "evidence": tuple(records),
            "versions": tuple(versions),
        }
    )
    return request, raw, SemanticGraph.model_validate(payload)


def test_many_to_many_programs_preserve_exact_execution_and_code(tmp_path):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    assert len(value["transitions"]) == 4
    assert all(len(item["transition_ids"]) == 2 for item in value["programs"].values())
    assert all(
        {w.program_id for w in edge.witnesses} == {"program-0", "program-1"}
        for edge in graph.edges.values()
    )
    # Same loop slice, different complete Harnesses: grouping does not duplicate code
    # or erase each actual Bundle and its distinct invocation parameters.
    assert all(program.pointers == ("/loop",) for program in graph.programs.values())
    assert (
        len(
            {
                json.dumps(read_original(program.bundle)["loop"], sort_keys=True)
                for program in graph.programs.values()
            }
        )
        == 1
    )
    assert len({item["bundle_id"] for item in value["programs"].values()}) == 2
    distances = []
    for episode in raw.episodes:
        selected = query_map(value, view="episode", identity=episode.receipt_id)
        first, last = selected["transitions"].values()
        assert first["successor_id"] == selected["cell"]["transition_ids"][1]
        assert last["successor_id"] is None
        assert first["actual_target"] == last["actual_source"]
        distances.append(
            read_original(
                OriginalReference.model_validate(first["executions"][0]["invocation"])
            )["arguments"]["distance"]
        )
        assert first["cost"] == {"physical_steps": 1, "action_wall_time_s": None}
    assert distances == [0.1, 0.11]
    program = query_semantic(graph, view="program", identity="program-0")
    assert set(program["edges"]) == {"start_held", "held_open"}
    assert all(len(edge["witnesses"]) == 1 for edge in program["edges"].values())
    assert program["pointers"] == ["/loop"]
    assert json.loads(Path(program["bundle"]["path"]).read_text())["loop"]


@pytest.mark.parametrize(
    "defect,match",
    [
        ("wrong_bundle", "different executed Bundle"),
        ("foreign_original", "outside the frozen originals"),
        ("missing_pointer", "does not resolve"),
        ("runtime_pointer", "W/C/L"),
        ("duplicate_pointer", "sorted unique"),
        ("overlapping_pointer", "duplicate a parent"),
        ("unknown_program", "undeclared program"),
        ("orphan_program", "actual execution witnesses"),
        ("missing_role", "exactly its edge roles"),
        ("extra_role", "exactly its edge roles"),
        ("unknown_without_reason", "needs uncertainty"),
        ("legacy_mechanism", "extra_forbidden"),
    ],
)
def test_rejects_false_or_ambiguous_program_evidence(tmp_path, defect, match):
    request, _, graph = program_case(tmp_path)
    payload = graph.model_dump(mode="json")
    program = payload["programs"]["program-0"]
    edge = payload["edges"]["start_held"]
    witness = edge["witnesses"][0]
    if defect == "wrong_bundle":
        program["bundle"] = payload["programs"]["program-1"]["bundle"]
    elif defect == "foreign_original":
        program["bundle"]["path"] = str(tmp_path / "unbound-bundle.json")
    elif defect == "missing_pointer":
        program["pointers"] = ["/loop/nonexistent"]
    elif defect == "runtime_pointer":
        program["pointers"] = ["/runtime"]
    elif defect == "duplicate_pointer":
        program["pointers"] = ["/loop", "/loop"]
    elif defect == "overlapping_pointer":
        program["pointers"] = ["/loop", "/loop/program"]
    elif defect == "unknown_program":
        witness["program_id"] = "invented"
    elif defect == "orphan_program":
        payload["programs"]["unused"] = copy.deepcopy(program)
    elif defect == "missing_role":
        witness["bindings"] = {}
    elif defect == "extra_role":
        witness["bindings"]["invented"] = "object"
    elif defect == "unknown_without_reason":
        witness.update(program_id=None, uncertainty="")
    else:
        edge["mechanism"] = "Removed edge-local program description"
    with pytest.raises(ValueError, match=match):
        validate(request, SemanticGraph.model_validate(payload))


def test_unresolved_bindings_remain_explicit_without_fabricated_plugins(tmp_path):
    request, raw, graph = program_case(tmp_path)
    payload = graph.model_dump(mode="json")
    witness = payload["edges"]["start_held"]["witnesses"][0]
    witness.update(
        program_id=None,
        bindings={"target": None},
        uncertainty="Entity and implementation attribution unresolved",
    )
    value = mapped(request, raw, SemanticGraph.model_validate(payload))
    summary = query_map(value)
    assert summary["unattributed_transition_count"] == 1
    observed = next(w for w in value["transitions"].values() if w["program_id"] is None)
    assert observed["bindings"]["target"] is None
    assert observed["executions"][0]["invocation"] is not None


def test_projected_invocations_are_read_only_on_explicit_original_request(
    tmp_path, monkeypatch
):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)

    def unexpected_read(*_):
        raise AssertionError("map navigation must not read invocation originals")

    with monkeypatch.context() as patch:
        patch.setattr(
            "reweave.evolution.evidence.trajectory.read_originals", unexpected_read
        )
        query_map(value)
        query_map(value, view="version", identity=request.versions[0].version_id)
        selected = query_map(value, view="episode", identity=raw.episodes[0].receipt_id)
        query_map(value, view="program", identity="program-0")
    execution = next(iter(selected["transitions"].values()))["executions"][0]
    assert set(execution["invocation"]) == {"artifact_id", "path", "bytes", "pointer"}
    assert "arguments" not in json.dumps(value)
    assert execution["invocation_tool"] == "pull"
    original = read_original(OriginalReference.model_validate(execution["invocation"]))
    assert (
        original
        == raw.episodes[0].model_dump(mode="json")["operation_starts"][0]["event"]
    )
    assert original["arguments"]["distance"] == 0.1


def test_continuation_never_crosses_unclassified_work_or_another_episode(tmp_path):
    request, raw, graph = program_case(
        tmp_path, paths=(("start", "held", "held", "open"),) * 2
    )
    payload = graph.model_dump(mode="json")
    middle = payload["edges"].pop("held_held")
    payload["unclassified"] = [
        {key: witness[key] for key in ("episode_id", "first_frame", "last_frame")}
        | {"reason": "Mechanism unresolved between these observed states"}
        for witness in middle["witnesses"]
    ]
    value = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert all(w["successor_id"] is None for w in value["transitions"].values())
    assert sum(w["cost"]["physical_steps"] for w in value["transitions"].values()) == 4
    assert (
        sum(
            gap["cost"]["physical_steps"]
            for cell in value["episodes"].values()
            for gap in cell["unclassified"]
        )
        == 2
    )


def test_version_aliases_do_not_duplicate_physical_samples(tmp_path):
    request, raw, graph = program_case(tmp_path)
    validate(request, graph)
    alias = VersionReference(
        version_id="version:alias", bundle_id=request.versions[0].bundle_id
    )
    value = project_map(raw, graph, (*request.versions, alias), request.protocol.dev)
    assert len(value["cells"]) == 3
    assert len(value["episodes"]) == len(raw.episodes) == 2
    assert len(value["transitions"]) == 4
    assert (
        value["cells"][alias.version_id]
        == value["cells"][request.versions[0].version_id]
    )
    assert all(stats["episodes"] == 2 for stats in value["edge_statistics"].values())
    # JSON persistence must retain sharing, not expand each version's complete cell.
    frozen = json.loads(json.dumps(value))
    condition = request.protocol.dev[0].record_id
    assert isinstance(frozen["cells"][alias.version_id][condition], str)
    assert map_cell(frozen, alias.version_id, condition) == map_cell(
        frozen, request.versions[0].version_id, condition
    )
    assert query_map(frozen, view="version", identity=alias.version_id)["cells"] == {
        condition: query_map(
            frozen, view="episode", identity=raw.episodes[0].receipt_id
        )["cell"]
    }


@pytest.mark.parametrize("defect", ["nested", "bundle", "condition"])
def test_map_rejects_removed_nested_cells_and_cross_bound_receipts(tmp_path, defect):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    version = request.versions[0].version_id
    condition = request.protocol.dev[0].record_id
    if defect == "nested":
        value["cells"][version][condition] = copy.deepcopy(
            map_cell(value, version, condition)
        )
        message = "receipt reference"
    else:
        cell = map_cell(value, version, condition)
        cell["bundle_id" if defect == "bundle" else "condition_id"] = "foreign"
        message = "version or condition"
    with pytest.raises(ValueError, match=message):
        query_map(value, view="version", identity=version)


def test_missing_dev_cell_is_derived_without_fabricated_episode_or_validation(tmp_path):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    version = request.versions[0].version_id
    condition = request.protocol.dev[0].record_id
    value["cells"][version][condition] = None
    frozen = json.loads(json.dumps(value))
    cell = query_map(frozen, view="version", identity=version)["cells"][condition]
    assert cell["receipt_id"] is None and cell["progress"] is None
    assert cell["official_success"] is None and cell["usage"] is None
    assert cell["state_spans"] == [] and cell["transition_ids"] == []
    assert cell["bundle_id"] == request.versions[0].bundle_id
    assert cell["condition_id"] == condition
    assert None not in frozen["episodes"]
    assert query_map(frozen, view="version", identity=version)["validation_cells"] == {}
    frozen["validation_cells"][version][condition] = None
    with pytest.raises(ValueError, match="must remain absent"):
        map_row(frozen, version, split="validation")


def test_compact_spans_preserve_progress_regressions_and_unknowns(tmp_path):
    path = ("start",) * 40 + ("held",) * 40 + ("open",) * 40 + ("held",) * 20
    request, raw, graph = case(tmp_path, paths=(path,))
    before = mapped(request, raw, graph)
    payload = graph.model_dump(mode="json")
    for identity, spans in payload["episodes"].items():
        compact = []
        for span in spans:
            if compact and compact[-1]["node_id"] == span["node_id"]:
                compact[-1]["last_frame"] = span["last_frame"]
            else:
                compact.append(span)
        payload["episodes"][identity] = compact
    after = mapped(request, raw, SemanticGraph.model_validate(payload))
    old_cell = next(iter(before["episodes"].values()))
    new_cell = next(iter(after["episodes"].values()))
    assert len(new_cell["state_spans"]) == 4
    assert "path" not in new_cell and "methods" not in new_cell
    assert new_cell["lost_nodes"] == ["open"]
    assert {k: v for k, v in old_cell.items() if k != "state_spans"} == {
        k: v for k, v in new_cell.items() if k != "state_spans"
    }
    assert before["transitions"] == after["transitions"]


def test_targeted_views_preserve_all_selected_evidence_without_full_map(tmp_path):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    summary = query_map(value)
    assert "cells" not in summary and "transitions" not in summary
    assert summary["transition_count"] == len(value["transitions"])
    assert len(json.dumps(summary)) < len(json.dumps(value)) / 2
    assert query_map(value, view="version", identity=request.versions[0].version_id)[
        "cells"
    ] == map_row(value, request.versions[0].version_id)
    program = query_map(value, view="program", identity="program-0")
    assert len(program["transitions"]) == 2
    assert {w["episode_id"] for w in program["transitions"].values()} == {
        raw.episodes[0].receipt_id
    }
    edge = query_map(value, view="transitions", identity="held_open")
    assert len(edge["transitions"]) == 2
    one_id = next(iter(edge["transitions"]))
    assert len(query_map(value, view="witness", identity=one_id)["transitions"]) == 1
    with pytest.raises(ValueError, match="unknown projected"):
        query_map(value, view="transitions", identity="invented")
    with pytest.raises(ValueError, match="unknown projected"):
        query_map(value, view="transitions", identity=one_id)
    with pytest.raises(ValueError, match="unknown projected"):
        query_map(value, view="witness", identity="held_open")
    with pytest.raises(ValueError, match="not projected"):
        query_map(value, view="episode", identity="pending")
    assert query_map(None)["available"] is False


def test_edge_labels_cannot_collide_with_physical_witness_selection(tmp_path):
    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    witness_id, edge_witness = list(value["transitions"])[:2]
    value["transitions"][edge_witness]["edge_id"] = witness_id
    assert set(
        query_map(value, view="witness", identity=witness_id)["transitions"]
    ) == {witness_id}
    assert set(
        query_map(value, view="transitions", identity=witness_id)["transitions"]
    ) == {edge_witness}


def test_program_sources_resolve_once_per_original_per_validation(
    tmp_path, monkeypatch
):
    from reweave.evolution.graph import refinement as graph_refinement

    request, _, graph = program_case(tmp_path)
    original_open = Path.open
    originals = {Path(ref.path) for ref in graph.original_references()}
    calls = []

    def read(path, *args, **kwargs):
        if path in originals:
            calls.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read)
    graph_refinement.validate_refinement(
        request,
        GraphAnalysisResult(
            forecast={
                "outcomes": [{"progress": 100.0, "probability": 1.0}],
                "confidence": 0.8,
                "rationale": "Synthetic prediction; no physical efficacy claim.",
            },
            graph=graph,
            changes="Synthetic",
        ),
    )
    assert len(calls) == len(set(calls)) == 4


@pytest.mark.parametrize("index", ["-1", "01", "+1", "１"])
def test_source_pointers_do_not_accept_python_index_aliases(index):
    from reweave.evolution.evidence.trajectory import resolve_original_pointer

    with pytest.raises(ValueError, match="canonical nonnegative"):
        resolve_original_pointer({"files": ["first", "last"]}, "/files/" + index)
    assert resolve_original_pointer({"a/b": ["first"]}, "/a~1b/0") == "first"


def test_report_reads_only_latest_completed_projection(tmp_path):
    from reweave.evolution.contracts import RoundDecision
    from reweave.evolution.search.report import semantic_projection

    request, raw, graph = program_case(tmp_path)
    value = mapped(request, raw, graph)
    decision = RoundDecision(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=request.graph,
        graph=artifact(tmp_path, "semantic.json", graph.model_dump(mode="json")),
        progress_map=artifact(tmp_path, "projection.json", value),
        parent=request.versions[0],
        references=(request.versions[0],),
        width=1,
        candidate_cost={"rollout_usd": None},
        alternatives=({"width": 1.0},),
        statistics={},
        reason="Synthetic report fixture",
    )
    receipt = artifact(tmp_path, "decision.json", decision.model_dump(mode="json"))
    state = {
        "rounds": {
            "0": {"decision": {"path": str(tmp_path / "irrelevant-old-decision.json")}},
            "1": {"decision": receipt.model_dump(mode="json")},
            "2": {"decision": None},
        }
    }
    result = semantic_projection(state, request.evidence)
    assert result["projected_episode_ids"] == sorted(graph.episodes)
    assert not result["pending_episode_ids"]
