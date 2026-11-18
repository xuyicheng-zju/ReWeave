# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Deterministic LIBERO lifecycle, exact analysis recovery and read-only replay."""

import asyncio
import json

import pytest

from reweave.benchmarks.libero.privileged_state import collect_privileged_critic_state
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    EvaluationRecord,
    EvidenceFile,
    RoundDecision,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.graph.replay import replay_graph
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace
from reweave.robots.physical_evidence import libero_features
from tests.test_libero_privileged_state import _task_env
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    mutated,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis
from tests.unit_tests.harness.evolution.test_graph_evolution import verified_author


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize(
    "interruption", [None, "analysis_completed", "decision_committed"]
)
def test_graph_round_through_physical_collectors_and_exact_recovery(
    tmp_path, width, interruption
):
    # Collectors and recovery, not allocation: fixed width launches exactly K.
    declared = protocol().model_copy(
        update={"max_authors_per_round": width, "width_mode": "fixed"}
    )
    workspace = EvolutionWorkspace(tmp_path / "run", baseline(), declared)
    workspace.root.mkdir()
    (workspace.root / "configuration.json").write_text("{}")
    analyzed, evaluated, authored = [], [], []

    class Analyst(FixtureGraphAnalysis):
        async def __call__(self, request, directory, *, before_launch):
            analyzed.append(request.record_id)
            return await super().__call__(
                request, directory, before_launch=before_launch
            )

    analyst = Analyst()

    async def evaluate(request, bundle, directory, *, before_launch):
        before_launch()
        evaluated.append(request.record_id)
        env = _task_env()

        def measure():
            return libero_features(collect_privileged_critic_state(env))

        trace = ExecutionTrace(directory / "execution_trace.jsonl")
        trace.record_reset_state(physical_features=measure())
        active = ActiveExecution(trace)
        active.begin("move_to", {})
        step = trace.record_scripted_control_step(
            execution_ref=active.snapshot.execution_id,
            action_index=0,
            action=[0.1],
            terminated=False,
            truncated=False,
        )
        env.sim.data.body_xpos[2, 0] = 0.03
        trace.record_telemetry(step, physical_features=measure())
        active.finish("completed")
        path = directory / "execution_trace.jsonl"
        success = bundle.record_id != baseline().record_id
        return EvaluationRecord(
            record_id=new_record_id(),
            request=request,
            official_success=success,
            failure_origin="none" if success else "agent",
            infrastructure_retryable=None,
            usage=usage(),
            evidence=(EvidenceFile(**artifact_record(path)),),
            episode=None,
            initialization=request.instance.initialization,
        )

    async def author(proposal, directory, *, before_launch):
        before_launch()
        authored.append(proposal.record_id)
        assert proposal.references == (proposal.parent,)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=mutated(baseline(), "C").model_copy(
                        update={
                            "record_id": f"equivalent-child-{proposal.author_index}"
                        }
                    ),
                    summary="fixture repair",
                ),
            ),
            summary="Full task candidate",
            session_id="author-fixture",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    from tests.unit_tests.harness.evolution.fixtures import ConsolidatingAuthor

    author = ConsolidatingAuthor(workspace, verified_author(author))
    search = GraphEvolution(workspace, author, evaluate, analyzer=analyst)
    if interruption:
        if interruption == "analysis_completed":
            original = search._artifact

            def interrupt(relative, value):
                if relative.startswith("decisions/"):
                    raise RuntimeError("simulated host interruption")
                return original(relative, value)

            search._artifact = interrupt
        else:

            async def interrupt(*args):
                raise RuntimeError("simulated host interruption")

            search._finish_round = interrupt
        with pytest.raises(RuntimeError, match="simulated"):
            asyncio.run(search.run())
        assert len(analyzed) == 1 and not authored and len(evaluated) == 1
        search = GraphEvolution(
            workspace, author, evaluate, analyzer=analyst, recover=True
        )
    state = asyncio.run(search.run())
    assert state["stop_reason"] == "target_reached"
    assert len(analyzed) == 2 and len(authored) == width and len(evaluated) == 2
    assert state["rounds"]["0"]["outcome"] is not None
    decision = RoundDecision.model_validate_json(
        (workspace.root / "decisions/round-000000.json").read_text()
    )
    proposal = next(iter(state["proposals"].values()))
    assert proposal["parent"] == decision.parent.model_dump(mode="json")
    assert proposal["parent"]["bundle_id"] == baseline().record_id
    report = json.loads((workspace.root / "report.json").read_text())
    score = report["scores"][report["best_version"]]
    assert score["parent"] == proposal["parent"]
    assert score["changed_scopes"] == ["C"]
    assert (
        score["parent_changes"]
        == score["reference_changes"][proposal["parent"]["version_id"]]
    )

    assert asyncio.run(search.run()) == state
    assert len(analyzed) == 2 and len(authored) == width and len(evaluated) == 2
    policy = tmp_path / "policy.json"
    policy.write_text(declared.graph.model_dump_json())
    state_bytes = workspace.state_path.read_bytes()
    graph = replay_graph(workspace.root, tmp_path / "replayed.json", policy)
    assert len(graph.episodes) == 2 and len(graph.transitions) == 2
    assert sum(e.official_success for e in graph.episodes) == 1
    assert workspace.state_path.read_bytes() == state_bytes
    with pytest.raises(ValueError, match="fresh output"):
        replay_graph(workspace.root, tmp_path / "replayed.json", policy)
    alias = tmp_path / "archive-alias"
    alias.symlink_to(workspace.root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the original archive"):
        replay_graph(workspace.root, alias / "replayed.json", policy)
    assert not (workspace.root / "replayed.json").exists()
    outcome = json.loads((workspace.root / "outcomes/round-000000.json").read_text())
    assert len(outcome["observations"]) == width
    assert not outcome["observations"][0]["mutation_review"]["probes_declared"]
    assert all(
        item["mutation_review"]["conditions"][0]["official_change"] == "gained"
        for item in outcome["observations"]
    )
    assert report["round_outcomes"]["0"] == outcome

    # Older decisions never observed the new diagnostics. Reconstructing their
    # terminal report must preserve the original shape and bytes, without a run.
    decision_path = workspace.root / "decisions/round-000000.json"
    legacy_decision = json.loads(decision_path.read_text())
    legacy_decision.pop("search_review")
    decision_path.write_text(json.dumps(legacy_decision))
    report.pop("round_outcomes")
    report["search_allocation"]["0"].pop("search_review")
    report_path = workspace.root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    original_report = report_path.read_bytes()
    from reweave.evolution.search.report import publish_report

    publish_report(workspace, state, search._records(state), search.accounting(state))
    assert report_path.read_bytes() == original_report
    assert len(analyzed) == 2 and len(authored) == width and len(evaluated) == 2
