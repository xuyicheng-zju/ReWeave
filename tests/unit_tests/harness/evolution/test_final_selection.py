"""Final selection compares every settled candidate on one topology."""

import json

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    BudgetRevision,
    Candidate,
    SemanticGraph,
)
from reweave.evolution.graph.refinement import validate_refinement
from reweave.evolution.search.engine import EvolutionWorkspace
from tests.unit_tests.harness.evolution.fixtures import artifact, baseline, usage
from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    configured,
    graph_search,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FinalArchiveAnalysis(FixtureGraphAnalysis):
    """Synthetic retained achievement differentiates child from partial parent."""

    def __init__(self, interrupt):
        self.interrupt = interrupt
        self.requests = []

    async def __call__(self, request, directory, *, before_launch):
        self.requests.append(request)
        if request.purpose == "final_selection" and self.interrupt:
            self.interrupt = False
            before_launch()
            raise RuntimeError("interrupted final analysis")
        result = await super().__call__(request, directory, before_launch=before_launch)
        if request.purpose != "final_selection":
            return result
        graph = result.analysis.graph.model_dump(mode="json")
        graph["nodes"]["retained"] = {
            **graph["nodes"]["contact"],
            "meaning": "Synthetic retained achievement",
            "predecessors": ["contact"],
        }
        graph["nodes"]["complete"]["predecessors"] = ["retained"]
        children = {
            r.record_id
            for r in request.evidence
            if r.request.bundle_id != baseline().record_id
        }
        for identity in children:
            span = graph["episodes"][identity][-1]
            node = span["node_id"]
            replacement = "retained_" + node
            graph["nodes"][replacement] = {
                **graph["nodes"][node],
                "predecessors": ["retained"],
            }
            span["node_id"] = replacement
        edges = {}
        for edge in graph["edges"].values():
            for witness in edge["witnesses"]:
                identity = witness["episode_id"]
                edges[f"transition_{len(edges)}"] = {
                    **edge,
                    "target": graph["episodes"][identity][-1]["node_id"],
                    "witnesses": [witness],
                }
        graph["edges"] = edges
        observed = {
            span["node_id"] for spans in graph["episodes"].values() for span in spans
        }
        graph["nodes"] = {
            key: node
            for key, node in graph["nodes"].items()
            if node["role"] != "branch" or key in observed
        }
        analysis = result.analysis.model_copy(
            update={"graph": SemanticGraph.model_validate(graph)}
        )
        validate_refinement(request, analysis)
        return result.model_copy(update={"analysis": analysis})


@pytest.mark.anyio
@pytest.mark.parametrize(
    "interrupt,revision", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("parent_success", [False, True])
async def test_final_selection_prioritizes_success_then_retained_progress(
    tmp_path, interrupt, revision, parent_success
):
    protocol = configured(1).model_copy(update={"max_authors_per_round": 1})
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol)
    (tmp_path / "configuration.json").write_text("{}")
    mutations, evaluations = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        mutations.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="retention",
                    bundle=child("retained"),
                    summary="Synthetic retained achievement",
                    experience="Lifecycle fixture only.",
                ),
            ),
            summary="Synthetic candidate",
            session_id="fixture-retention",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        return record(
            request,
            directory,
            parent_success
            and bundle.record_id == baseline().record_id
            and request.instance.initialization["seed"] == 1,
        )

    analyzer = FinalArchiveAnalysis(interrupt)
    search = graph_search(
        workspace, verified_author(author), evaluator, analyzer=analyzer
    )
    if interrupt:
        with pytest.raises(RuntimeError, match="interrupted final analysis"):
            await search.run()
        pending = json.loads(workspace.state_path.read_text())
        assert not pending["completed"]
        frozen = pending["final_projections"]["0"]["request"]
        if revision:
            revised = protocol.model_copy(
                update={
                    "record_id": "protocol:final-analysis-revision",
                    "graph": protocol.graph.model_copy(
                        update={
                            "analysis_wall_time_s": protocol.graph.analysis_wall_time_s
                            * 2
                        }
                    ),
                }
            )
            pending["protocol"] = revised.record_id
            pending["budget_revisions"] = [
                BudgetRevision(previous=protocol, current=revised).model_dump(
                    mode="json"
                )
            ]
            workspace = EvolutionWorkspace(tmp_path, baseline(), revised)
            workspace.save(pending)
        search = graph_search(
            workspace,
            verified_author(author),
            evaluator,
            analyzer=analyzer,
            recover=True,
        )
    state = await search.run()
    assert len(state["rounds"]) == 1
    assert len(mutations) == 1 and len(evaluations) == 6
    stage = state["final_projections"]["1" if revision else "0"]
    if interrupt and not revision:
        assert stage["request"] == frozen
        assert analyzer.requests[-1] == analyzer.requests[-2]
    if revision:
        assert state["final_projections"]["0"]["request"] == frozen
        assert state["final_projections"]["0"]["progress_map"] is not None
    assert len(stage["request"]["versions"]) == 2
    report_path = tmp_path / "report.json"
    original = report_path.read_bytes()
    report = json.loads(original)
    expected = baseline().record_id if parent_success else child("retained").record_id
    assert report["scores"][report["best_version"]]["bundle_id"] == expected
    assert report["scores"][report["official_best_version"]]["bundle_id"] == expected
    assert not report["target_reached"]
    assert not report["semantic_projection"]["pending_episode_ids"]
    assert report["selection"]["statistics"]["parent_mean_progress"] == pytest.approx(
        2 / 3 if parent_success else 0.75
    )
    assert report["selection"]["criterion"] == (
        "official_success_then_topology_progress_then_successful_harness_size"
    )
    assert search.accounting(state)["by_kind"]["graph_construction"]["model_calls"] == 0
    assert await search.run() == state
    assert report_path.read_bytes() == original
    assert len(mutations) == 1 and len(evaluations) == 6
