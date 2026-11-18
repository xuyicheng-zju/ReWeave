# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Graph allocation invariants, with synthetic semantics rather than physics claims."""

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
    RoundDecision,
    SemanticGraph,
)
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    usage,
)
from tests.unit_tests.harness.evolution.graph_fixtures import (
    FixtureGraphAnalysis,
    semantic_fixture,
)
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    configured,
    record,
    verified_author,
)
from tests.unit_tests.harness.evolution.test_semantic_graph import (
    case,
    first_cell,
    mapped,
)


def test_failed_transition_recovered_by_later_root_continuation_is_not_a_bottleneck(
    tmp_path,
):
    request, raw, graph = case(tmp_path, paths=(("start", "start", "held", "open"),))
    payload = graph.model_dump(mode="json")
    edge = payload["edges"]["start_start"]
    edge["target"] = "held"
    for witness in edge["witnesses"]:
        witness["outcome"] = "not_achieved"
    result = mapped(request, raw, SemanticGraph.model_validate(payload))
    cell = first_cell(result)
    assert any(t["outcome"] == "not_achieved" for t in result["transitions"].values())
    assert cell["unresolved_transition_ids"] == []
    assert cell["remaining_frontier"] == ["goal"]


def test_loop_and_duplicate_edge_labels_preserve_one_unresolved_interface(tmp_path):
    request, raw, graph = case(tmp_path, paths=(("start", "start", "start"),))
    payload = graph.model_dump(mode="json")
    edge = payload["edges"]["start_start"]
    edge["target"] = "held"
    for witness in edge["witnesses"]:
        witness["outcome"] = "not_achieved"
    repeated = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert len(first_cell(repeated)["unresolved_transition_ids"]) == 1
    duplicate = copy.deepcopy(edge)
    duplicate["witnesses"] = [edge["witnesses"].pop()]
    duplicate["explanation"] = "Another label for the same physical interface"
    payload["edges"]["renamed_same_interface"] = duplicate
    split = mapped(request, raw, SemanticGraph.model_validate(payload))
    assert len(first_cell(split)["unresolved_transition_ids"]) == 1
    assert len(set(split["edge_interfaces"].values())) == 1


def test_graph_author_cannot_supply_width_or_numeric_search_assessment(tmp_path):
    _, _, graph = case(tmp_path)
    for field in (
        "width",
        "search_assessment",
        "remaining_gap",
        "effective_frontiers",
    ):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            GraphAnalysisResult.model_validate(
                {
                    "graph": graph.model_dump(mode="json"),
                    "changes": "Synthetic interpretation",
                    field: 2,
                }
            )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "distinct_interfaces, forecast, expected",
    [
        (False, ((0.0, 0.5), (100.0, 0.5)), 4),
        (True, ((0.0, 0.5), (100.0, 0.5)), 4),
        (True, ((100.0, 1.0),), 1),
        (True, ((10.0, 0.4), (60.0, 0.6)), 1),
    ],
)
async def test_forecast_completion_drives_actual_independent_proposal_count(
    tmp_path, distinct_interfaces, forecast, expected
):
    """Real Host projection/allocation/lifecycle, with explicitly synthetic physics.

    In the last round an even completion chance is worth every sibling; certain or
    impossible completion needs one author. Topology enters only through the forecast.
    """

    class Analysis(FixtureGraphAnalysis):
        async def __call__(self, request, directory, *, before_launch):
            before_launch()
            raw = GraphSnapshot.model_validate_json(
                Path(request.graph.path).read_text()
            )
            graph = semantic_fixture(raw).model_dump(mode="json")
            graph["nodes"] = {
                k: n for k, n in graph["nodes"].items() if n["role"] != "branch"
            }
            for spans in graph["episodes"].values():
                for span in spans:
                    span.update(status="background", node_id=None)
            for identity, edge in graph["edges"].items():
                edge["target"] = "approach"
                edge["entry_conditions"] = (
                    identity if distinct_interfaces else "shared synthetic entry"
                )
                for witness in edge["witnesses"]:
                    witness["outcome"] = "not_achieved"
            result = GraphAnalysisResult(
                forecast={
                    "outcomes": [
                        {"progress": p, "probability": w} for p, w in forecast
                    ],
                    "confidence": 0.8,
                    "rationale": "Synthetic prediction; no physical efficacy claim.",
                },
                graph=SemanticGraph.model_validate(graph),
                changes="Synthetic common or distinct witnessed interfaces",
            )
            return GraphAnalysisReceipt(
                record_id=new_record_id(),
                request_id=request.record_id,
                analysis=result,
                session_id="graph-session",
                usage=usage().model_copy(update={"physical_steps": 0}),
                artifacts=(
                    artifact(
                        directory, "analysis.json", result.model_dump(mode="json")
                    ),
                ),
            )

    config = configured(1).model_copy(update={"max_authors_per_round": 4})
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    proposals = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic integration; physical efficacy is untested.",
                    branch_id="repair",
                    bundle=child(proposal.record_id),
                    summary="Independent synthetic candidate",
                ),
            ),
            summary="One candidate per independent proposal",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    state = await GraphEvolution(
        workspace, verified_author(author), evaluator, analyzer=Analysis()
    ).run()
    decision = RoundDecision.model_validate_json(
        (tmp_path / "decisions/round-000000.json").read_text()
    )
    assert decision.width == len(proposals) == expected
    assert decision.statistics["completion_probability"] == sum(
        w for p, w in forecast if p == 100.0
    )
    assert decision.statistics["rounds_remaining"] == 1
    assert [a["width"] for a in decision.alternatives] == [1.0, 2.0, 3.0, 4.0]
    import json

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["search_allocation"]["0"]["statistics"] == decision.statistics
    assert len({p.record_id for p in proposals}) == expected
    assert len({p.graph for p in proposals}) == 1
    assert len({p.progress_map for p in proposals}) == 1
    assert len({p.references for p in proposals}) == 1
    assert len(state["versions"]) == expected + 1


@pytest.mark.parametrize("invalid", [True, -1, 101, float("nan"), float("inf"), "50"])
def test_forecast_rejects_non_numeric_or_unbounded_predictions(invalid):
    from reweave.evolution.contracts import ProgressForecast

    with pytest.raises(ValidationError):
        ProgressForecast(
            outcomes=({"progress": invalid, "probability": 1.0},),
            confidence=0.8,
            rationale="Synthetic",
        )


def test_graph_only_output_is_rejected_instead_of_defaulting_to_optimism(tmp_path):
    from reweave.evolution.graph.refinement import validate_refinement
    from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request

    request, result = analysis_request(tmp_path)
    payload = result.model_dump(mode="json")
    del payload["forecast"]
    result = GraphAnalysisResult.model_validate(payload)
    with pytest.raises(ValueError, match="forecast"):
        validate_refinement(request, result)
    payload["forecast"] = {
        "outcomes": [{"progress": 50.0, "probability": 1.0}],
        "confidence": 0.8,
        "rationale": "Evidence missing",
        "success_probability": 0.5,
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphAnalysisResult.model_validate(payload)
