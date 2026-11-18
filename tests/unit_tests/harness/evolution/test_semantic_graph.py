# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Semantic re-projection, physical witness integrity and fixed allocation rules."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvaluationRequest,
    GraphAnalysisRequest,
    GraphAnalysisResult,
    GraphPolicy,
    GraphSnapshot,
    OriginalReference,
    SemanticGraph,
    VersionReference,
)
from reweave.evolution.evidence.trajectory import (
    build_graph,
    episode_from_record,
    read_original,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.projection import map_cell, map_row, project_map
from reweave.evolution.graph.refinement import validate_refinement
from reweave.evolution.graph.selection import (
    select_parent,
)
from reweave.evolution.graph.views import query_map
from reweave.evolution.search.matrix import evaluation_cells, result_matrix
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    evolution_history,
    protocol,
    usage,
)

BACKBONE = {
    "start": ("branch", ()),
    "near": ("milestone", ()),
    "held": ("milestone", ("near",)),
    "open": ("milestone", ("held",)),
    "goal": ("goal", ("open",)),
}


def test_validation_reads_each_original_once_without_cross_check_cache(
    tmp_path, monkeypatch
):
    request, _, graph = case(tmp_path)
    path = Path(next(ref.path for ref in graph.original_references() if ref.pointer))
    original_open = Path.open
    reads = []

    def counted(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    result = GraphAnalysisResult(
        forecast={
            "outcomes": [{"progress": 100.0, "probability": 1.0}],
            "confidence": 0.8,
            "rationale": "Synthetic prediction; no physical efficacy claim.",
        },
        graph=graph,
        changes="Synthetic fixture",
    )
    validate_refinement(request, result)
    assert reads == [path]
    lines = path.read_text().splitlines()
    lines[0] = "{"
    path.write_text("\n".join(lines) + "\n")
    reads.clear()
    with pytest.raises(json.JSONDecodeError):
        validate_refinement(request, result)
    assert reads == [path]


def case(tmp_path, paths=(("start", "held", "open"),), *, success=False):
    """Real trace/receipt plumbing with explicitly scripted physical interpretations."""
    records, versions, episodes = [], [], []
    config = protocol()
    for i, path in enumerate(paths):
        verdict = success[i] if isinstance(success, tuple) else success
        bundle_id = f"bundle:synthetic:{i}"
        version = VersionReference(version_id=f"version:{i}", bundle_id=bundle_id)
        events = [
            {
                "event_type": "initial_state",
                "event_index": 0,
                "physical_features": {},
                "error": None,
            }
        ]
        for frame in range(1, len(path)):
            execution_id = f"execution:{i}:{frame}"
            events.append(
                {
                    "event_type": "active_execution_started",
                    "event_index": len(events),
                    "execution_id": execution_id,
                    "tool_name": "pull",
                    "arguments": {"direction": [1, 0, 0], "distance": 0.1 + i / 100},
                }
            )
            events.append(
                {
                    "event_type": "control_step",
                    "event_index": len(events),
                    "control_step_index": frame - 1,
                    "execution_ref": execution_id,
                    "tool_name": "pull",
                    "action": [0.1],
                    "physical_features": {},
                }
            )
        trace_path = tmp_path / str(i) / "execution_trace.jsonl"
        trace_path.parent.mkdir(parents=True)
        trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        from reweave.evolution.contracts import EvidenceFile
        from reweave.evolution.evidence.records import artifact_record

        trace = EvidenceFile(**artifact_record(trace_path))
        record = EvaluationRecord(
            record_id=new_record_id(),
            request=EvaluationRequest(
                record_id=new_record_id(),
                bundle_id=bundle_id,
                instance=config.dev[0],
                split="dev",
                manifest_id=config.manifest_id,
                conditions=config.execution,
                execution=config.execution_id,
            ),
            official_success=verdict,
            failure_origin="none" if verdict else "agent",
            infrastructure_retryable=None,
            usage=usage(),
            evidence=(trace,),
            episode=None,
            initialization=config.dev[0].initialization,
        )
        receipt = artifact(
            tmp_path / str(i), "evaluation.json", record.model_dump(mode="json")
        )
        records.append(record)
        versions.append(version)
        episodes.append(episode_from_record(record, Path(receipt.path), (version,)))
    raw = build_graph(episodes, config.graph)
    nodes = {
        key: {
            "meaning": "Synthetic task state: " + key,
            "role": role,
            "predecessors": predecessors,
            "references": [],
            "uncertainty": "Synthetic unobserved backbone",
        }
        for key, (role, predecessors) in BACKBONE.items()
        if key != "start"
    }
    edges, assignments = {}, {}
    for episode, path in zip(episodes, paths):
        assignments[episode.receipt_id] = []
        for i, node in enumerate(path):
            refs = [r.model_dump(mode="json") for r in episode.frames[i].references]
            nodes[node] = {
                "meaning": f"Synthetic physical state: {node}",
                "role": BACKBONE[node][0],
                "predecessors": BACKBONE[node][1],
                "references": refs,
                "uncertainty": "Scripted fixture",
            }
            assignments[episode.receipt_id].append(
                {
                    "first_frame": i,
                    "last_frame": i,
                    "status": "node",
                    "node_id": node,
                    "uncertainty": "",
                }
            )
            if i:
                key = path[i - 1] + "_" + node
                edge = edges.setdefault(
                    key,
                    {
                        "source": path[i - 1],
                        "target": node,
                        "roles": {"target": "object whose task progress is changed"},
                        "entry_conditions": "Synthetic supported entry condition",
                        "exit_conditions": "Synthetic useful condition and continuation",
                        "explanation": "Synthetic interpretation; no empirical causal claim",
                        "references": refs,
                        "witnesses": [],
                    },
                )
                edge["witnesses"].append(
                    {
                        "episode_id": episode.receipt_id,
                        "first_frame": i - 1,
                        "last_frame": i,
                        "outcome": "achieved",
                        "program_id": None,
                        "bindings": {"target": "synthetic object"},
                        "uncertainty": "Synthetic trace has no isolated code attribution",
                    }
                )
    graph = SemanticGraph(
        nodes=nodes,
        edges=edges,
        programs={},
        episodes=assignments,
        unclassified=(),
    )
    request = GraphAnalysisRequest(
        previous_analysis=None,
        record_id="synthetic",
        round_index=0,
        protocol=config,
        graph=artifact(tmp_path, "raw.json", raw.model_dump(mode="json")),
        history=evolution_history(tmp_path),
        evidence=tuple(records),
        versions=tuple(versions),
        policy=AuthorPolicy().model_dump(mode="json"),
    )
    return request, raw, graph


def validate(request, graph):
    return validate_refinement(
        request,
        GraphAnalysisResult(
            forecast={
                "outcomes": [{"progress": 100.0, "probability": 1.0}],
                "confidence": 0.8,
                "rationale": "Synthetic prediction; no physical efficacy claim.",
            },
            graph=graph,
            changes="Evidence-based synthetic refinement",
        ),
    )


def mapped(request, raw, graph):
    validate(request, graph)
    return project_map(raw, graph, request.versions, request.protocol.dev)


def first_cell(value):
    return next(iter(map_row(value, next(iter(value["cells"]))).values()))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("selected_has_episode", [False, True])
def test_infrastructure_graph_references_follow_the_latest_archive_attempt(
    tmp_path, reverse, selected_has_episode
):
    from reweave.evolution.graph.projection import selected_graph_receipts
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture

    request, _, _ = case(tmp_path)
    original = request.evidence[0]
    records = tuple(
        original.model_copy(
            update={
                "failure_origin": "infrastructure",
                "infrastructure_retryable": False,
                "record_id": f"infra:{i}",
                "usage": original.usage.model_copy(update={"wall_time_s": i + 1}),
            }
        )
        for i in range(2)
    )
    records = tuple(reversed(records)) if reverse else records
    selected, other = records[-1], records[0]
    physical = (selected, other) if selected_has_episode else (other,)
    raw = build_graph(
        [
            episode_from_record(
                record,
                Path(
                    artifact(
                        tmp_path / str(i),
                        "infra.json",
                        record.model_dump(mode="json"),
                    ).path
                ),
                request.versions,
            )
            for i, record in enumerate(physical)
        ],
        request.protocol.graph,
    )
    executions = frozenset({request.protocol.execution_id})
    key = (original.request.bundle_id, original.request.instance.record_id)
    assert (
        evaluation_cells(
            request.protocol, (key[0],), records, execution_ids=executions
        )[key]
        == selected
    )
    references = selected_graph_receipts(
        request.protocol, (key[0],), records, raw, execution_ids=executions
    )
    assert references == {key: selected.record_id if selected_has_episode else None}
    value = project_map(
        raw,
        semantic_fixture(raw),
        request.versions,
        request.protocol.dev,
        cell_receipts=references,
    )
    # A different retry's positive physical evidence cannot fill the selected gap.
    assert value["episodes"][other.record_id]["progress"] > 0
    selected_cell = first_cell(value)
    assert selected_cell["receipt_id"] == references[key]
    assert selected_cell["official_success"] is None
    if selected_has_episode:
        assert selected_cell["progress"] > 0
        assert selected_cell["usage"] == selected.usage.model_dump(mode="json")
    else:
        assert selected_cell["progress"] is None


def test_graph_references_reject_a_missing_settled_physical_episode(tmp_path):
    from reweave.evolution.graph.projection import selected_graph_receipts

    request, _, _ = case(tmp_path)
    with pytest.raises(ValueError, match="selected evaluation receipt has no physical"):
        selected_graph_receipts(
            request.protocol,
            (request.versions[0].bundle_id,),
            request.evidence,
            build_graph([], request.protocol.graph),
            execution_ids=frozenset({request.protocol.execution_id}),
        )


def test_observed_infrastructure_progress_cannot_create_official_gains(tmp_path):
    from reweave.evolution.graph.projection import selected_graph_receipts
    from reweave.evolution.graph.selection import (
        forecast_context,
        parent_statistics,
    )
    from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture

    request, _, _ = case(tmp_path, paths=(("start", "held"), ("start", "held")))
    infrastructure = request.evidence[0].model_copy(
        update={"failure_origin": "infrastructure", "infrastructure_retryable": False}
    )
    records = (infrastructure, request.evidence[1])
    raw = build_graph(
        [
            episode_from_record(
                record,
                Path(
                    artifact(
                        tmp_path / str(i),
                        "observed.json",
                        record.model_dump(mode="json"),
                    ).path
                ),
                (version,),
            )
            for i, (record, version) in enumerate(
                zip(records, request.versions, strict=True)
            )
        ],
        request.protocol.graph,
    )
    request = request.model_copy(
        update={
            "evidence": records,
            "graph": artifact(
                tmp_path, "observed-raw.json", raw.model_dump(mode="json")
            ),
        }
    )
    graph = semantic_fixture(raw)
    validate(request, graph)
    value = project_map(
        raw,
        graph,
        request.versions,
        request.protocol.dev,
        cell_receipts=selected_graph_receipts(
            request.protocol,
            (version.bundle_id for version in request.versions),
            records,
            raw,
            execution_ids=frozenset({request.protocol.execution_id}),
        ),
    )
    parent = request.versions[0]
    row = map_row(value, parent.version_id)
    cell = next(iter(row.values()))
    assert 0 < cell["progress"] < 1 and cell["official_success"] is None
    assert parent_statistics(row)["parent_success_rate"] is None
    assert (
        forecast_context(value, parent)["current_progress_lower"]
        == 100 * cell["progress"]
    )
    summary = query_map(value)
    assert summary["versions"][parent.version_id][cell["condition_id"]][
        "exit_type"
    ] == ("infrastructure")
    comparison = query_map(value, view="comparison", identity=parent.version_id)
    contrast = comparison["dev"][cell["condition_id"]]["contrasts"][0]
    assert contrast["observed_relation"] == "equal_progress"
    assert contrast["official_result_differs"] is None
    witness_id = cell["transition_ids"][0]
    witness = query_map(value, view="witness", identity=witness_id)["transitions"][
        witness_id
    ]
    assert (
        witness["official_success"] is None and witness["exit_type"] == "infrastructure"
    )
    assert result_matrix(request.protocol, (parent.bundle_id,), (infrastructure,))[
        parent.bundle_id
    ] == (None,)


@pytest.mark.parametrize("origin", ["agent", "infrastructure", "none"])
@pytest.mark.parametrize("reverse", [False, True])
def test_budget_retest_selects_current_observation_independent_of_order(
    tmp_path, origin, reverse
):
    request, _, _ = case(tmp_path, success=True)
    old = request.evidence[0]
    current_protocol = request.protocol.model_copy(
        update={
            "record_id": "protocol:retest",
            "execution_id": "execution:retest",
            "execution": request.protocol.execution.model_copy(
                update={"wall_time_s": None}
            ),
        }
    )
    current = old.model_copy(
        update={
            "record_id": "receipt:retest",
            "request": old.request.model_copy(
                update={
                    "record_id": "request:retest",
                    "execution": current_protocol.execution_id,
                    "conditions": current_protocol.execution,
                }
            ),
            "official_success": origin == "none",
            "failure_origin": origin,
            "infrastructure_retryable": False if origin == "infrastructure" else None,
        }
    )
    records = (current, old) if reverse else (old, current)
    executions = frozenset({old.request.execution, current.request.execution})
    selected = evaluation_cells(
        current_protocol, (old.request.bundle_id,), records, execution_ids=executions
    )
    assert tuple(selected.values()) == (current,)
    row = result_matrix(
        current_protocol, (old.request.bundle_id,), records, execution_ids=executions
    )[old.request.bundle_id]
    assert row == (None if origin == "infrastructure" else origin == "none",)
    with pytest.raises(ValueError, match="different execution conditions"):
        result_matrix(current_protocol, (old.request.bundle_id,), records)
    if origin != "infrastructure":
        with pytest.raises(ValueError, match="duplicate settled matrix cell"):
            result_matrix(
                current_protocol,
                (old.request.bundle_id,),
                (*records, current),
                execution_ids=executions,
            )


def test_map_retest_keeps_historical_witnesses_and_requires_explicit_selection(
    tmp_path,
):
    request, raw, graph = case(tmp_path, paths=(("held", "open"), ("held", "goal")))
    first, second = raw.episodes
    # Two synthetic independent episodes now represent the same exact Bundle.
    raw = raw.model_copy(
        update={
            "episodes": (
                first,
                second.model_copy(update={"bundle_id": first.bundle_id}),
            )
        }
    )
    key = (first.bundle_id, first.condition_id)
    with pytest.raises(ValueError, match="multiple physical receipts"):
        project_map(raw, graph, request.versions, request.protocol.dev)
    selected = project_map(
        raw,
        graph,
        request.versions,
        request.protocol.dev,
        cell_receipts={key: second.receipt_id},
    )
    assert set(selected["episodes"]) == {first.receipt_id, second.receipt_id}
    assert first_cell(selected)["receipt_id"] == second.receipt_id
    missing = project_map(
        raw, graph, request.versions, request.protocol.dev, cell_receipts={key: None}
    )
    assert set(missing["episodes"]) == {first.receipt_id, second.receipt_id}
    assert next(iter(missing["cells"].values()))[first.condition_id] is None
    with pytest.raises(ValueError, match="different map cell"):
        project_map(
            raw,
            graph,
            request.versions,
            request.protocol.dev,
            cell_receipts={key: "f" * 64},
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("current_infrastructure", [False, True])
def test_current_missing_never_falls_back_to_multiple_valid_historical_executions(
    tmp_path, reverse, current_infrastructure
):
    request, _, _ = case(tmp_path, success=True)
    old = request.evidence[0]
    intermediate = request.protocol.model_copy(
        update={
            "record_id": "protocol:retest",
            "execution_id": "execution:retest",
            "execution": request.protocol.execution.model_copy(
                update={"wall_time_s": None}
            ),
        }
    )
    current_protocol = intermediate.model_copy(
        update={
            "record_id": "protocol:runtime-revision",
            "execution_id": "execution:runtime-revision",
            "execution_runtime": baseline().runtime.model_copy(
                update={
                    "revision_id": "runtime:revised",
                    "configuration": {"diagnostics": True},
                }
            ),
        }
    )
    second = old.model_copy(
        update={
            "record_id": "receipt:intermediate",
            "request": old.request.model_copy(
                update={
                    "record_id": "request:intermediate",
                    "execution": intermediate.execution_id,
                    "conditions": intermediate.execution,
                }
            ),
        }
    )
    historical = (second, old) if reverse else (old, second)
    records = historical
    if current_infrastructure:
        current = old.model_copy(
            update={
                "record_id": "receipt:current",
                "request": old.request.model_copy(
                    update={
                        "record_id": "request:current",
                        "execution": current_protocol.execution_id,
                        "conditions": current_protocol.execution,
                    }
                ),
                "official_success": False,
                "failure_origin": "infrastructure",
                "infrastructure_retryable": False,
            }
        )
        records = (*records, current)
    executions = frozenset(
        (
            request.protocol.execution_id,
            intermediate.execution_id,
            current_protocol.execution_id,
        )
    )
    cells = evaluation_cells(
        current_protocol, (old.request.bundle_id,), records, execution_ids=executions
    )
    assert tuple(cells.values()) == ((current,) if current_infrastructure else ())
    assert result_matrix(
        current_protocol, (old.request.bundle_id,), records, execution_ids=executions
    )[old.request.bundle_id] == (None,)


def test_unmeasured_current_map_keeps_all_historical_physical_witnesses(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("held", "open"), ("held", "goal")))
    historical = project_map(raw, graph, request.versions, request.protocol.dev)
    current = project_map(
        raw, graph, request.versions, request.protocol.dev, cell_receipts={}
    )
    assert current["episodes"] == historical["episodes"]
    assert current["transitions"] == historical["transitions"]
    assert all(
        cell["receipt_id"] is None and cell["progress"] is None
        for version in current["cells"]
        for cell in map_row(current, version).values()
    )


def test_refinement_reselects_parent_from_reprojected_history_without_new_rollouts(
    tmp_path,
):
    request, raw, graph = case(
        tmp_path, paths=(("start", "held", "open"), ("start", "held"))
    )
    original = mapped(request, raw, graph)
    assert select_parent(original, request.versions) == request.versions[0]
    payload = graph.model_dump(mode="json")
    # Same traces, but revised evidence interpretation says the open state has
    # lost proximity and contact. Its earlier peak cannot win parent selection.
    payload["nodes"]["open"]["predecessors"] = []
    payload["nodes"]["goal"]["predecessors"] = ["held", "open"]
    refreshed = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert first_cell(refreshed)["peak_progress"] == 0.5
    assert first_cell(refreshed)["progress"] == 0.25
    assert select_parent(refreshed, request.versions) == request.versions[1]
    assert select_parent(original, request.versions) == request.versions[0]
    assert not refreshed["pending_episode_ids"]
    assert refreshed["execution_graph_id"] == original["execution_graph_id"]


def test_split_merge_and_changed_physical_facts_reproject_all_history(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "held", "open"),) * 2)
    before = mapped(request, raw, graph)
    payload = graph.model_dump(mode="json")
    second = raw.episodes[1].receipt_id
    payload["nodes"]["held_variant"] = copy.deepcopy(payload["nodes"]["held"])
    payload["nodes"]["held_variant"].update(role="branch", predecessors=["held"])
    payload["nodes"]["held_variant"]["meaning"] = (
        "A different physical contact configuration with the same achieved facts"
    )
    payload["episodes"][second][1]["node_id"] = "held_variant"
    for edge_id, edge in list(payload["edges"].items()):
        variant = copy.deepcopy(edge)
        edge["witnesses"] = [w for w in edge["witnesses"] if w["episode_id"] != second]
        variant["witnesses"] = [
            w for w in variant["witnesses"] if w["episode_id"] == second
        ]
        variant["source"] = (
            "held_variant" if variant["source"] == "held" else variant["source"]
        )
        variant["target"] = (
            "held_variant" if variant["target"] == "held" else variant["target"]
        )
        payload["edges"][edge_id + "_variant"] = variant
    split = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert [
        next(iter(map_row(split, version).values()))["progress"]
        for version in split["cells"]
    ] == [0.75, 0.75]
    assert split["graph_id"] is before["graph_id"] is None
    assert split["node_evidence"] != before["node_evidence"]
    # Splitting mechanisms without changing retained facts cannot manufacture
    # an advantage or change the globally selected parent.
    parent = select_parent(before, request.versions)
    assert select_parent(split, request.versions) == parent
    assert mapped(request, raw, graph)["cells"] == before["cells"]
    payload = graph.model_dump(mode="json")
    payload["nodes"]["open"]["predecessors"] = []
    payload["nodes"]["goal"]["predecessors"] = ["held", "open"]
    updated = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert all(
        next(iter(map_row(updated, version).values()))["progress"] == 0.25
        for version in updated["cells"]
    )
    assert all(
        next(iter(map_row(updated, version).values()))["peak_progress"] == 0.5
        for version in updated["cells"]
    )


def test_regression_and_repeated_loops_do_not_inflate_progress(tmp_path):
    request, raw, graph = case(
        tmp_path, paths=(("start", "held", "held", "open", "held", "start"),)
    )
    value = mapped(request, raw, graph)
    cell = first_cell(value)
    assert cell["progress"] == 0 and cell["peak_progress"] == 0.75
    assert cell["attained_nodes"] == ["held", "near", "open"]
    assert len(cell["state_spans"]) == 6 and len(cell["path_changes"]) == 5
    assert not cell["official_success"]
    assert all(
        w["outcome"] == "achieved"
        for w in (value["transitions"][key] for key in cell["transition_ids"])
    )


def test_comparison_pairs_same_conditions_without_splicing_wins_or_scoring_unknown(
    tmp_path,
):
    import copy

    request, raw, graph = case(
        tmp_path, paths=(("start", "held"), ("start",)), success=(False, False)
    )
    value = mapped(request, raw, graph)
    parent, other = [v.version_id for v in request.versions]
    condition = next(iter(value["cells"][parent]))
    map_cell(value, other, condition).update(progress=None, official_success=None)
    value["cells"]["cached"] = copy.deepcopy(value["cells"][other])
    value["versions"]["cached"] = value["versions"][other]
    value["validation_cells"]["cached"] = {}
    # A different condition is not a contrast; duplicate receipt reuse is one observation.
    value["cells"]["foreign"] = {
        "different-condition": copy.deepcopy(value["cells"][other][condition])
    }
    result = query_map(value, view="comparison", identity=parent)
    contrasts = result["dev"][condition]["contrasts"]
    assert len(contrasts) == 1 and contrasts[0]["version_id"] == other
    assert contrasts[0]["observed_relation"] == "unscored"
    assert contrasts[0]["progress"] is None and contrasts[0]["official_success"] is None
    assert "usage" in contrasts[0] and "transition_ids" in contrasts[0]
    assert (
        result["dev"][condition]["selected"]["progress"]
        == map_cell(value, parent, condition)["progress"]
    )
    assert result["validation"] == {}
    # An actually observed higher-progress result remains a distinct candidate.
    map_cell(value, other, condition).update(progress=1, official_success=True)
    contrasts = query_map(value, view="comparison", identity=parent)["dev"][condition][
        "contrasts"
    ]
    assert contrasts[0]["observed_relation"] == "higher_progress"
    assert contrasts[0]["official_result_differs"]


@pytest.mark.parametrize("selected_missing", [False, True])
def test_comparison_preserves_distinct_missing_bundles_without_duplicate_aliases(
    tmp_path, selected_missing
):
    request, raw, graph = case(
        tmp_path, paths=(("start", "held"), ("start",)), success=(False, False)
    )
    value = mapped(request, raw, graph)
    parent, other = [v.version_id for v in request.versions]
    condition = next(iter(value["cells"][parent]))
    value["cells"][other][condition] = None
    if selected_missing:
        value["cells"][parent][condition] = None
    for identity, bundle in (
        ("missing", "e" * 64),
        ("alias", value["versions"][other]),
    ):
        value["versions"][identity] = bundle
        value["cells"][identity] = {condition: None}
        value["validation_cells"][identity] = {}
    contrasts = query_map(value, view="comparison", identity=parent)["dev"][condition][
        "contrasts"
    ]
    assert {entry["version_id"] for entry in contrasts} == {other, "missing"}
    assert all(entry["observed_relation"] == "unscored" for entry in contrasts)
    assert all(entry["official_result_differs"] is None for entry in contrasts)


def test_merged_mechanism_retains_distinct_real_arguments_and_versions(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "held"),) * 2)
    value = mapped(request, raw, graph)
    assert value["edge_statistics"]["start_held"] == {
        "achieved": 2,
        "not_achieved": 0,
        "unknown": 0,
        "bundles": 2,
        "episodes": 2,
        "conditions": 1,
    }
    methods = []
    for episode in raw.episodes:
        selected = query_map(value, view="episode", identity=episode.receipt_id)
        first = next(iter(selected["transitions"].values()))
        methods.append({**first, **first["executions"][0]})
    assert [
        read_original(OriginalReference.model_validate(m["invocation"]))["arguments"][
            "distance"
        ]
        for m in methods
    ] == [
        0.1,
        0.11,
    ]
    assert all(m["invocation"] is not None for m in methods)
    assert len({m["bundle_id"] for m in methods}) == 2
    assert [m["bundle_id"] for m in methods] == [v.bundle_id for v in request.versions]
    payload = graph.model_dump(mode="json")
    edge = payload["edges"]["start_held"]
    variant = copy.deepcopy(edge)
    variant["entry_conditions"] = (
        "Same endpoints under a distinct physical entry condition"
    )
    variant["witnesses"] = [edge["witnesses"].pop()]
    payload["edges"]["alternate_pull"] = variant
    split = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert len(split["edge_statistics"]) == 2
    assert all(stats["episodes"] == 1 for stats in split["edge_statistics"].values())


@pytest.mark.parametrize(
    "defect,match",
    [
        ("duplicate", "double-count"),
        ("missing_interval", "every original frame interval"),
        ("missing_episode", "every historical episode"),
        ("outside_episode", "outside its episode"),
        ("wrong_source", "source differs"),
        ("false_success", "goal state"),
    ],
)
def test_rejects_invented_or_lost_physical_evidence(tmp_path, defect, match):
    request, _, graph = case(tmp_path)
    payload = graph.model_dump(mode="json")
    edge = payload["edges"]["start_held"]
    if defect == "duplicate":
        payload["edges"]["duplicate"] = copy.deepcopy(edge)
    elif defect == "missing_interval":
        del payload["edges"]["start_held"]
    elif defect == "missing_episode":
        payload["episodes"] = {}
    elif defect == "outside_episode":
        edge["witnesses"][0]["last_frame"] = 8
    elif defect == "wrong_source":
        edge["source"] = "open"
    else:
        payload["nodes"]["open"].update(role="goal", predecessors=["held"])
        payload["nodes"]["goal"]["predecessors"] = ["held"]
    with pytest.raises(ValueError, match=match):
        validate(request, SemanticGraph.model_validate(payload))


@pytest.mark.parametrize(
    "defect",
    ["foreign", "tampered", "bad_pointer", "missing_span", "unknown_without_reason"],
)
def test_invalid_references_and_assignments_fail(tmp_path, defect):
    request, _, graph = case(tmp_path)
    payload = graph.model_dump(mode="json")
    ref = payload["nodes"]["held"]["references"][0]
    if defect == "foreign":
        ref["path"] = str(tmp_path / "foreign.json")
    elif defect == "tampered":
        Path(ref["path"]).write_text("{}")
    elif defect == "bad_pointer":
        ref["pointer"] = "/99999"
    elif defect == "missing_span":
        next(iter(payload["episodes"].values())).pop()
    else:
        next(iter(payload["episodes"].values()))[0].update(
            status="unknown", node_id=None
        )
    with pytest.raises(ValueError):
        validate(request, SemanticGraph.model_validate(payload))


@pytest.mark.parametrize(
    "gap", ["reset", "missing_action", "unknown_action", "unknown_semantics"]
)
def test_gaps_require_explicit_unclassified_intervals(tmp_path, gap):
    request, raw, graph = case(tmp_path, paths=(("start", "held"),))
    raw_payload = raw.model_dump(mode="json")
    frame = raw_payload["episodes"][0]["frames"][1]
    if gap == "reset":
        frame["control_step"] = None
    elif gap == "missing_action":
        raw_payload["episodes"][0]["frames"][0]["control_step"] = 0
        frame["control_step"] = 2
    elif gap == "unknown_action":
        frame["error"] = "sensor unavailable; unacknowledged physical interval"
    payload = graph.model_dump(mode="json")
    if gap == "unknown_semantics":
        next(iter(payload["episodes"].values()))[-1].update(
            status="unknown",
            node_id=None,
            uncertainty="No reliable semantic assignment",
        )
        # The required backbone remains as an explicit unobserved hypothesis.
    else:
        modified = GraphSnapshot.model_validate(raw_payload)
        request = request.model_copy(
            update={
                "graph": artifact(
                    tmp_path, "gap.json", modified.model_dump(mode="json")
                )
            }
        )
        with pytest.raises(ValueError, match="cannot bridge"):
            validate(request, graph)
    witness = payload["edges"].pop("start_held")["witnesses"][0]
    for field in ("outcome", "program_id", "bindings", "uncertainty"):
        witness.pop(field)
    payload["unclassified"] = [{**witness, "reason": "Preserve the original gap"}]
    result = mapped(
        request,
        GraphSnapshot.model_validate(raw_payload),
        SemanticGraph.model_validate(payload),
    )
    cell = first_cell(result)
    assert not cell["transition_ids"] and len(cell["unclassified"]) == 1
    if gap == "unknown_semantics":
        assert cell["progress"] is None and not cell["terminal_state_known"]


def test_unknown_trailing_action_keeps_history_but_has_no_current_progress(tmp_path):
    request, raw, graph = case(tmp_path)
    raw = raw.model_copy(
        update={
            "episodes": (
                raw.episodes[0].model_copy(
                    update={"terminal_state_known": False, "exit_type": "unknown"}
                ),
            )
        }
    )
    request = request.model_copy(
        update={
            "graph": artifact(tmp_path, "unknown.json", raw.model_dump(mode="json"))
        }
    )
    value = mapped(request, raw, graph)
    cell = first_cell(value)
    assert cell["progress"] is None and cell["frontier"] is None
    assert cell["official_success"] is None
    assert cell["peak_progress"] == 0.75 and cell["path_changes"][-1] == "open"


def test_success_requires_official_terminal_witness(tmp_path):
    request, raw, graph = case(
        tmp_path, paths=(("start", "held", "goal"),), success=True
    )
    value = mapped(request, raw, graph)
    cell = first_cell(value)
    assert cell["official_success"] and cell["progress"] == 1
    payload = graph.model_dump(mode="json")
    payload["nodes"]["unobserved_goal"] = copy.deepcopy(payload["nodes"]["goal"])
    payload["nodes"]["unobserved_goal"]["references"] = []
    payload["nodes"]["goal"]["role"] = "branch"
    with pytest.raises(ValueError, match="terminal frame must be a goal"):
        validate(request, SemanticGraph.model_validate(payload))


def width_map(*, samples=1, shared=False, completed=3, aliases=False):
    cells, episodes, versions, transitions, interfaces = {}, {}, {}, {}, {}
    for i in range(samples):
        row = {}
        for c in range(4):
            edge = "shared" if shared else str(c)
            key = f"{0 if aliases else i}:{c}"
            transitions[key] = {"edge_id": edge}
            interfaces[edge] = edge
            row[str(c)] = key
            episodes[key] = {
                "receipt_id": key,
                "bundle_id": str(0 if aliases else i),
                "condition_id": str(c),
                "progress": None if completed is None else completed / 4,
                "peak_progress": (completed or 0) / 4,
                "frontier": edge if completed is not None else None,
                "official_success": False,
                "completed_nodes": None
                if completed is None
                else ["near", "held", "open", "goal"][:completed],
                "path_changes": ["start", edge],
                "transition_ids": [key],
                "unresolved_transition_ids": None if completed is None else [key],
                "remaining_frontier": None if completed is None else ["goal"],
            }
        cells[str(i)] = row
        versions[str(i)] = str(0 if aliases else i)
    return {
        "versions": versions,
        "episodes": episodes,
        "cells": cells,
        "validation_cells": {key: {} for key in cells},
        "transitions": transitions,
        "edge_interfaces": interfaces,
        "milestone_nodes": ["near", "held", "open"],
    }


@pytest.mark.parametrize(
    "field,value",
    [("routes", []), ("p_i", [0.5]), ("Q_k", [0.5]), ("split_candidate_ids", [])],
)
def test_graph_analysis_rejects_removed_evolution_outputs(tmp_path, field, value):
    _, _, graph = case(tmp_path)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphAnalysisResult.model_validate(
            {
                "graph": graph.model_dump(mode="json"),
                "changes": "refine",
                field: value,
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "cost_penalty",
        "split_proposals_per_round",
        "split_child_episodes",
        "split_brier_reduction",
    ],
)
def test_graph_policy_rejects_removed_probability_and_finite_split_fields(field):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphPolicy.model_validate({field: 0.1})


def test_graph_refiner_public_inputs_do_not_contain_scheduling_policy(tmp_path):
    from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request

    request, _ = analysis_request(tmp_path)
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    root = tmp_path / "analyst"
    analyzer._prepare(request, root)
    public = json.loads((root / "context/analysis_request.json").read_text())
    assert set(public) == {
        "record_id",
        "round_index",
        "conditions",
        "versions",
        "execution_graph_id",
        "validation_conditions",
        "author_wall_time_s",
        "execution",
    }
    assert public["author_wall_time_s"] == request.protocol.author_wall_time_s
    assert public["execution"] == request.protocol.execution.model_dump(mode="json")
    assert json.loads((root / "context/graph.json").read_text()) is None
    assert json.loads((root / "context/progress_map.json").read_text()) is None
    assert "protocol" in json.loads(
        (root / ".host_inputs/analysis_request.json").read_text()
    )


def test_pending_history_requires_refinement_and_cannot_silently_enter_map(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "held"),) * 2)
    payload = graph.model_dump(mode="json")
    pending = raw.episodes[-1].receipt_id
    payload["episodes"].pop(pending)
    for edge in payload["edges"].values():
        edge["witnesses"] = [w for w in edge["witnesses"] if w["episode_id"] != pending]
    earlier = SemanticGraph.model_validate(payload)
    with pytest.raises(ValueError, match="different set"):
        project_map(raw, earlier, request.versions, request.protocol.dev)
    partial = project_map(
        raw, earlier, request.versions, request.protocol.dev, partial=True
    )
    assert partial["pending_episode_ids"] == [pending]
    assert partial["cells"][request.versions[-1].version_id] == {}
    assert not mapped(request, raw, graph)["pending_episode_ids"]


def test_failed_mechanism_retains_actual_endpoint(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "held", "start"),))
    payload = graph.model_dump(mode="json")
    payload["nodes"]["goal"] = {
        "meaning": "Unobserved official goal",
        "role": "goal",
        "predecessors": ["open"],
        "references": [],
        "uncertainty": "No successful execution",
    }
    edge = payload["edges"]["held_start"]
    edge["target"] = "goal"
    edge["witnesses"][0]["outcome"] = "not_achieved"
    value = mapped(request, raw, SemanticGraph.model_validate(payload))
    cell = first_cell(value)
    assert cell["path_changes"][-1] == "start" and cell["progress"] == 0
    assert value["transitions"][cell["transition_ids"][-1]]["outcome"] == "not_achieved"


def test_missing_invocation_stays_explicitly_unknown(tmp_path):
    request, raw, graph = case(tmp_path)
    episode = raw.episodes[0].model_copy(update={"operation_starts": ()})
    raw = raw.model_copy(update={"episodes": (episode,)})
    request = request.model_copy(
        update={
            "graph": artifact(
                tmp_path, "missing-invocations.json", raw.model_dump(mode="json")
            )
        }
    )
    value = mapped(request, raw, graph)
    cell = first_cell(value)
    assert all(
        method["invocation_tool"] is None and method["invocation"] is None
        for witness in (value["transitions"][key] for key in cell["transition_ids"])
        for method in witness["executions"]
    )


def test_text_code_is_a_valid_complete_graph_reference(tmp_path):
    from reweave.evolution.contracts import OriginalReference
    from reweave.evolution.evidence.trajectory import read_original

    code = artifact(tmp_path, "source.py", {})
    reference = OriginalReference(**code.model_dump(mode="json"), pointer="")
    assert read_original(reference) == Path(code.path).read_text()


@pytest.mark.parametrize("following", ["none", "telemetry_error", "reset"])
def test_trace_unknown_interval_survives_sensor_error_until_explicit_reset(
    tmp_path, following
):
    from reweave.evolution.contracts import EvidenceFile
    from reweave.evolution.evidence.records import artifact_record

    request, _, _ = case(tmp_path, paths=(("start", "held"),))
    record = request.evidence[0]
    trace_path = Path(record.evidence[0].path)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    events.append(
        {
            "event_type": "action_outcome_unknown",
            "event_index": len(events),
            "execution_ref": "execution:0:1",
        }
    )
    if following == "telemetry_error":
        events.append(
            {
                "event_type": "control_step",
                "event_index": len(events),
                "control_step_index": 1,
                "execution_ref": "execution:0:1",
                "tool_name": "pull",
                "action": [0.1],
                "telemetry_error": "sensor offline",
                "physical_features": {},
            }
        )
    elif following == "reset":
        events.append(
            {
                "event_type": "reset_state",
                "event_index": len(events),
                "physical_features": {},
            }
        )
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    trace = EvidenceFile(**artifact_record(trace_path))
    record = record.model_copy(update={"evidence": (trace,)})
    receipt = artifact(
        tmp_path, "unknown-evaluation.json", record.model_dump(mode="json")
    )
    episode = episode_from_record(record, Path(receipt.path), request.versions)
    assert episode.terminal_state_known == (following == "reset")
    if following == "telemetry_error":
        assert "sensor offline" in episode.frames[-1].error
        assert "unacknowledged physical interval" in episode.frames[-1].error
    elif following == "reset":
        assert "unacknowledged physical interval" not in episode.frames[-1].error
