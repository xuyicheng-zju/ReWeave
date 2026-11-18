"""Topology evolves while historical interpretations remain immutable everywhere."""

import json
from pathlib import Path

import pytest

from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    EvolutionProtocol,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
    SemanticGraph,
    SemanticNode,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.refinement import validate_refinement
from reweave.evolution.graph.reprojection import projection_episode_ids
from reweave.evolution.graph.selection import select_parent
from reweave.evolution.graph.views import query_map
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.evolution.search.review import compare_versions
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.graph_fixtures import (
    FixtureGraphAnalysis,
    semantic_fixture,
)
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FirstObservationAnalysis(FixtureGraphAnalysis):
    def __init__(self):
        self.requests = []

    async def __call__(self, request, directory, *, before_launch):
        before_launch()
        self.requests.append(request)
        raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
        fresh = projection_episode_ids(request, raw)
        graph = semantic_fixture(
            raw.model_copy(
                update={
                    "episodes": tuple(e for e in raw.episodes if e.receipt_id in fresh)
                }
            )
        )
        if request.round_index:
            # Add a required milestone and change the meaning of an old node.
            # Old raw node names deliberately collide across interpretations.
            nodes = dict(graph.nodes)
            nodes["extra"] = SemanticNode(
                meaning="Newly distinguished requirement",
                role="milestone",
                predecessors=(),
                references=(),
                uncertainty="Unobserved synthetic requirement",
            )
            nodes["complete"] = nodes["complete"].model_copy(
                update={"predecessors": ("contact", "extra")}
            )
            nodes["contact"] = nodes["contact"].model_copy(
                update={"meaning": "Revised contact definition"}
            )
            graph = SemanticGraph.model_validate(
                {**graph.model_dump(mode="json"), "nodes": nodes}
            )
        result = GraphAnalysisResult(
            graph=graph, changes="Synthetic first-observation analysis"
        )
        validate_refinement(request, result)
        if request.round_index:
            with pytest.raises(ValueError, match="only all new receipts"):
                validate_refinement(
                    request, result.model_copy(update={"graph": semantic_fixture(raw)})
                )
        return GraphAnalysisReceipt(
            record_id=request.record_id + "/fixture-receipt",
            request_id=request.record_id,
            analysis=result,
            session_id="fixture",
            usage=usage(),
            artifacts=(
                artifact(directory, "analysis.json", result.model_dump(mode="json")),
            ),
        )


@pytest.mark.anyio
async def test_first_interpretations_survive_topology_change_and_final_selection(
    tmp_path,
):
    config = EvolutionProtocol.model_validate(
        {
            **protocol().model_dump(mode="json"),
            "max_authors_per_round": 1,
            "max_rounds": 2,
            "success_consolidation": False,
            "history_projection": "first_observation",
        }
    )
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    proposals = []

    async def author(p, directory, *, before_launch):
        before_launch()
        proposals.append(p)
        return AuthoredMutation(
            record_id=f"author-{p.round_index}",
            proposal_id=p.record_id,
            candidates=(
                Candidate(
                    branch_id="edit",
                    bundle=child(str(p.round_index)),
                    summary="C only",
                    experience="Synthetic",
                ),
            ),
            summary="Synthetic",
            session_id=f"session-{p.round_index}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    analyzer = FirstObservationAnalysis()
    engine = GraphEvolution(
        workspace, verified_author(author), evaluator, analyzer=analyzer
    )
    state = await engine.run()
    assert state["completed"] and state["stop_reason"] == "round_limit"
    assert len(proposals) == 2
    assert [r.purpose for r in analyzer.requests] == [
        "search",
        "search",
        "final_selection",
    ]
    maps = [
        json.loads((tmp_path / "maps" / name).read_text())
        for name in ("round-000000.json", "round-000001.json", "final-000000.json")
    ]
    first, second, final = maps
    old = next(iter(first["episodes"]))
    for later in (second, final):
        assert later["episodes"][old] == first["episodes"][old]
        for identity, transition in first["transitions"].items():
            assert later["transitions"][identity] == transition
        view = query_map(later, view="episode", identity=old)
        assert view["cell"]["progress"] == pytest.approx(2 / 3)
        assert all(
            w["interface"]["target"].startswith(first["graph_id"])
            for w in view["transitions"].values()
        )
    assert len(final["episodes"]) == 3 and final["pending_episode_ids"] == []
    assert len(final["interpretations"]) == 3
    versions = analyzer.requests[-1].versions
    # Cached old 2/3 beats new 2/4; exact node counts would incorrectly tie.
    assert select_parent(final, versions).version_id == versions[0].version_id
    assert (
        compare_versions(final, versions[0].version_id, versions[-1].version_id)[
            "status"
        ]
        == "unknown"
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["best_version"] == versions[0].version_id
    assert len(report["semantic_projection"]["projected_episode_ids"]) == 3
    assert report["semantic_projection"]["pending_episode_ids"] == []
    prompt, final_prompt = NativeGraphAnalysis._prompts(analyzer.requests[-1], tmp_path)
    assert "project ONLY" in prompt and "only new_episode_ids" in final_prompt
    assert "all history reprojected" not in prompt
    # Resume must not reanalyze or reinterpret any settled receipt.
    await GraphEvolution(
        workspace, engine.author, evaluator, analyzer=analyzer, recover=True
    ).run()
    assert len(analyzer.requests) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"topology_mode": "fixed_initial"},
        {"evolution_mode": "no_topology"},
        {"max_authors_per_round": 3},
    ],
)
def test_reprojection_ablation_rejects_other_mechanism_combinations(change):
    with pytest.raises(ValueError, match="first-observation"):
        EvolutionProtocol.model_validate(
            {
                **protocol().model_dump(mode="json"),
                "max_authors_per_round": 1,
                "history_projection": "first_observation",
                **change,
            }
        )


def test_native_preparation_and_admission_use_only_new_receipts(tmp_path):
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.authoring.references import AuthorReferences
    from reweave.evolution.contracts import GraphAnalysisRequest
    from reweave.evolution.evidence.trajectory import build_graph
    from reweave.evolution.graph.projection import analysis_graph_receipts
    from reweave.evolution.graph.refinement import validate_analysis_output
    from reweave.evolution.graph.reprojection import project_analysis
    from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request

    first_dir, incoming_dir = tmp_path / "first", tmp_path / "incoming"
    first_dir.mkdir()
    incoming_dir.mkdir()
    first, result = analysis_request(first_dir)
    config = EvolutionProtocol.model_validate(
        {
            **first.protocol.model_dump(mode="json"),
            "max_authors_per_round": 1,
            "history_projection": "first_observation",
            "success_consolidation": False,
        }
    )
    first = first.model_copy(update={"protocol": config})
    raw = GraphSnapshot.model_validate_json(Path(first.graph.path).read_text())
    cached = project_analysis(
        first,
        raw,
        result.graph,
        graph_id="first/graph",
        cell_receipts=analysis_graph_receipts(first, raw),
    )
    previous = GraphAnalysisReceipt(
        record_id="first",
        request_id=first.record_id,
        analysis=result,
        session_id="fixture",
        usage=usage(),
        artifacts=(first.graph,),
    )
    candidate = child("next")
    incoming, result = analysis_request(incoming_dir, bundle=candidate)
    incoming_raw = GraphSnapshot.model_validate_json(
        Path(incoming.graph.path).read_text()
    )
    combined = build_graph((*raw.episodes, *incoming_raw.episodes), config.graph)
    request = GraphAnalysisRequest.model_validate(
        {
            **first.model_dump(mode="json"),
            "record_id": "final",
            "purpose": "final_selection",
            "round_index": 1,
            "previous_analysis": artifact(
                tmp_path, "previous.json", previous.model_dump(mode="json")
            ),
            "previous_projection": artifact(tmp_path, "cached.json", cached),
            "graph": artifact(
                tmp_path, "combined.json", combined.model_dump(mode="json")
            ),
            "evidence": (*first.evidence, *incoming.evidence),
            "versions": (*first.versions, *incoming.versions),
        }
    )
    bundles = {b.record_id: b for b in (baseline(), candidate)}
    analyzer = NativeGraphAnalysis(
        bundle_loader=bundles.__getitem__,
        policy=AuthorPolicy(),
        evidence_workspace=tmp_path / "archive",
    )
    workspace = tmp_path / "native/workspace"
    analyzer._prepare(request, workspace)
    instructions, _ = analyzer._prompts(request, workspace)
    assert "Omit graph.matching, including final analysis" in instructions
    assert "otherwise encode it in final analysis" not in instructions
    references = AuthorReferences.load(workspace)
    assert references.read_json(workspace / "context/progress_map.json") == cached
    inputs = references.read_json(workspace / "context/input.json")
    assert inputs["new_episode_ids"] == [incoming.evidence[0].record_id]
    draft = references.read_json(workspace / "analysis/result.json")["graph"]
    assert draft["episodes"] == {} and draft["programs"] == {}
    assert all(edge["witnesses"] == [] for edge in draft["edges"].values())
    output = workspace / "analysis/result.json"
    result = result.model_copy(update={"forecast": None})
    output.write_text(json.dumps(references.encode(result.model_dump(mode="json"))))
    admitted = validate_analysis_output(request, workspace / "analysis")
    assert set(admitted.graph.episodes) == {incoming.evidence[0].record_id}
