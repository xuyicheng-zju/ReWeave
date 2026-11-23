# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""A settled graph keeps its decision while new authors adopt a runtime repair."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    BudgetRevision,
    BundleVersion,
    EvaluationRecord,
    EvolutionState,
    ExecutionRevision,
    GraphAnalysisReceipt,
    GraphRound,
    RoundDecision,
    execution_handoff_allowed,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.runtime_manifest import (
    RuntimeAuthoringContract,
    RuntimeManifest,
    capture_baseline,
)
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    mutated,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_execution_revision import revised_protocol
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture


def test_execution_handoff_requires_the_exact_published_runtime_chain():
    original = protocol()
    middle = revised_protocol(original)
    current = middle.model_copy(
        update={
            "record_id": "second-repaired-protocol",
            "execution_id": "second-repaired-execution",
            "execution_runtime": middle.execution_runtime.model_copy(
                update={
                    "revision_id": "second-repaired-runtime",
                    "environment": {"adapter": "second repair"},
                }
            ),
        }
    )
    first = ExecutionRevision(previous=original, current=middle, reason="First repair.")
    second = ExecutionRevision(
        previous=middle, current=current, reason="Second repair."
    )
    assert execution_handoff_allowed(original, original, ())
    assert execution_handoff_allowed(original, current, (second, first))
    assert not execution_handoff_allowed(original, current, ())
    assert not execution_handoff_allowed(original, current, (second,))
    assert not execution_handoff_allowed(current, original, (first, second))
    extended = current.model_copy(
        update={"record_id": "extended-protocol", "max_rounds": current.max_rounds + 1}
    )
    assert not execution_handoff_allowed(original, extended, (first, second))


class DecisionReady(Exception):
    pass


def settled_graph(tmp_path):
    """Complete typed CPU evidence and an immutable decision, with no authors."""
    source = tmp_path / "original"
    source.mkdir()
    request, result = analysis_request(source)
    policy = AuthorPolicy()

    class DecisionWorkspace(EvolutionWorkspace):
        armed = False

        def save(self, state):
            super().save(state)
            if self.armed and state["rounds"]["0"]["decision"] is not None:
                raise DecisionReady

    workspace = DecisionWorkspace(source, baseline(), request.protocol)
    atomic_json(source / "configuration.json", {})
    workspace.put_bundle(baseline())
    directory = source / "settled-analysis"
    receipt = GraphAnalysisReceipt(
        record_id="settled-graph-receipt",
        request_id=request.record_id,
        analysis=result,
        session_id="01a12172-f3e3-7503-bac4-8429ce3660f0",
        usage=usage(),
        artifacts=(artifact(directory, "synthetic-native.json", {"synthetic": True}),),
    )
    atomic_json(directory / "result.json", receipt.model_dump(mode="json"))
    atomic_json(
        directory / "audit/inputs/analysis_request.json",
        request.model_dump(mode="json"),
    )
    evaluation = source / "settled-evaluation"
    atomic_json(evaluation / "result.json", request.evidence[0].model_dump(mode="json"))
    analyzer = NativeGraphAnalysis(
        bundle_loader=workspace.bundle,
        policy=policy,
        evidence_workspace=source / "evidence",
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("cached decision preparation cannot dispatch work")

    evolution = GraphEvolution(
        workspace,
        SimpleNamespace(policy=policy),
        forbidden,
        analyzer=analyzer,
        recover=True,
    )
    with workspace.transaction() as state:
        workspace.add_version(
            state,
            BundleVersion(
                record_id=request.versions[0].version_id,
                bundle_id=baseline().record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        state["rounds"]["0"] = GraphRound(
            request=request, decision=None, outcome=None, construction_wall_time_s=0
        ).model_dump(mode="json")
        for kind, key, value, path in [
            ("graph_analysis", "round-01/graph-analysis", request, directory),
            (
                "evaluation",
                "evaluation/V000/condition-001/attempt-01",
                request.evidence[0].request,
                evaluation,
            ),
        ]:
            state["attempts"][key] = {
                "key": key,
                "kind": kind,
                "request": value.model_dump(mode="json"),
                "status": "completed",
                "launches": 1,
                "deadline_at": 1 if kind == "graph_analysis" else None,
                "result_id": receipt.record_id
                if kind == "graph_analysis"
                else request.evidence[0].record_id,
                "result_directory": str(path),
            }
        workspace.save(state)
        workspace.armed = True
        with pytest.raises(DecisionReady):
            asyncio.run(
                evolution._proposals(state, asyncio.run(evolution.matrix(state)), 0)
            )
        assert not state["proposals"]
    frozen = (source / "evolution.json").read_bytes()
    return workspace, request, receipt, frozen


def active_runtime(tmp_path, previous):
    actual = fixture(tmp_path)
    capture_baseline(tmp_path / "baseline", **actual)
    manifest = RuntimeManifest.model_validate_json(
        (tmp_path / "baseline/runtime.json").read_text()
    )
    limits = EpisodeLimits(
        program_calls=20, gateway_calls=20, wall_time_s=previous.execution.wall_time_s
    )
    runtime = BundleRuntime(
        baseline(),
        actual["toolkit"],
        "Synthetic current contract",
        episode_limits=limits,
    )
    manifest = manifest.model_copy(
        update={
            "conditions": previous.execution,
            "configuration": {
                **manifest.configuration,
                "limits": {"episode": limits.model_dump(mode="json")},
            },
            "authoring_contract": RuntimeAuthoringContract.from_runtime(runtime),
        }
    )
    current = previous.model_copy(
        update={
            "record_id": "active-protocol",
            "execution_id": manifest.execution_id,
            "execution_runtime": manifest.runtime_configuration(),
        }
    )
    revision = ExecutionRevision(
        previous=previous, current=current, reason="CPU runtime repair."
    )
    return manifest.model_copy(update={"execution_revision": revision}), current, actual


def test_cached_graph_decision_creates_new_execution_authors_without_reselection(
    tmp_path, monkeypatch
):
    original, request, receipt, original_bytes = settled_graph(tmp_path)
    manifest, current, actual = active_runtime(tmp_path / "active", request.protocol)
    continuation = EvolutionWorkspace(tmp_path / "continuation", baseline(), current)
    continuation.put_bundle(baseline())
    atomic_json(continuation.root / "configuration.json", {})
    state = json.loads(original_bytes)
    state["protocol"] = current.record_id
    state["execution_revisions"] = [manifest.execution_revision.model_dump(mode="json")]
    continuation.save(state)
    old_round = copy.deepcopy(state["rounds"]["0"])
    decision_file = Path(old_round["decision"]["path"])
    decision_bytes = decision_file.read_bytes()
    decision = RoundDecision.model_validate_json(decision_bytes)
    calls = {"graph": 0, "evaluation": 0}

    class CachedAnalysis(NativeGraphAnalysis):
        async def __call__(self, *args, **kwargs):
            calls["graph"] += 1
            raise AssertionError("frozen graph decision cannot start another query")

        def verify_result(self, *args, **kwargs):
            raise AssertionError("a published decision does not reconsume its receipt")

    async def forbidden_evaluator(*args, **kwargs):
        calls["evaluation"] += 1
        raise AssertionError("author preparation cannot evaluate")

    def no_reselection(*args, **kwargs):
        raise AssertionError(
            "published parent, starting reference and width are immutable"
        )

    monkeypatch.setattr(
        "reweave.evolution.graph.selection.decide_round", no_reselection
    )
    monkeypatch.setattr(
        "reweave.evolution.graph.projection.project_map", no_reselection
    )
    author = NativeBundleAuthor(
        bundle_loader=continuation.bundle,
        policy=AuthorPolicy(),
        evidence_workspace=continuation.root / "evidence",
        runtime_manifest=manifest,
    )
    analyzer = CachedAnalysis(
        bundle_loader=continuation.bundle,
        policy=author.policy,
        evidence_workspace=continuation.root / "evidence",
        runtime_manifest=manifest,
    )
    evolution = GraphEvolution(
        continuation, author, forbidden_evaluator, analyzer=analyzer, recover=True
    )
    rows = asyncio.run(evolution.matrix(state))
    assert rows[baseline().record_id] == (None,)
    assert (
        asyncio.run(evolution._evaluate(state, state["order"][0])) == request.evidence
    )
    proposals = asyncio.run(evolution._proposals(state, rows, 0))
    assert len(proposals) == decision.width
    for proposal in proposals:
        assert proposal.task_protocol == current
        assert (
            proposal.parent == decision.parent
            and proposal.references == decision.references
        )
        assert proposal.decision == decision_file_ref(old_round)
        assert proposal.progress_map == decision.progress_map
        assert proposal.evidence == request.evidence
        directory = continuation.root / "prepared" / proposal.record_id
        author._prepare_inputs(proposal, directory, resume=False)
        workspace = directory / "workspace"
        gateway = AuthorReferences.load(workspace).read_json(
            workspace / "context/gateway.json"
        )
        contract = manifest.authoring_contract.model_dump(mode="json")
        assert gateway["tools"] == contract["tools"]
        assert gateway["operations"] == contract["operations"]
    assert asyncio.run(evolution._proposals(state, rows, 0)) == proposals
    assert state["rounds"]["0"] == old_round
    assert decision_file.read_bytes() == decision_bytes
    assert (original.root / "evolution.json").read_bytes() == original_bytes
    assert receipt.request_id == request.record_id
    assert calls == {"graph": 0, "evaluation": 0}
    assert actual["toolkit"].calls == 0 and not actual["toolkit"].terminal
    # A later host cannot retroactively turn the old map into a new official score.
    assert asyncio.run(evolution.matrix(state))[baseline().record_id] == (None,)
    submitted = []

    async def fixture_evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        submitted.append(request)
        return EvaluationRecord(
            record_id=new_record_id(),
            request=request,
            official_success=False,
            failure_origin="infrastructure",
            infrastructure_retryable=False,
            usage=usage().model_copy(update={"physical_steps": 0, "model_calls": 0}),
            evidence=(
                artifact(
                    directory, "synthetic-initialization.json", {"synthetic": True}
                ),
            ),
            episode=None,
            initialization=request.instance.initialization,
        )

    evolution.evaluator = fixture_evaluator
    candidate = mutated(baseline(), "W")
    continuation.put_bundle(candidate)
    asyncio.run(evolution._evaluate_candidate(state, candidate, "round-01-author-01"))
    assert len(submitted) == 1 and submitted[0].execution == current.execution_id
    assert submitted[0].bundle_id == candidate.record_id != baseline().record_id


def decision_file_ref(round_record):
    from reweave.evolution.contracts import EvidenceFile

    return EvidenceFile.model_validate(round_record["decision"])


def test_author_cap_increase_does_not_expand_a_frozen_single_author_round(tmp_path):
    from reweave.evolution.evidence.records import artifact_record

    original, request, _, original_bytes = settled_graph(tmp_path)
    state = json.loads(original_bytes)
    reduced = request.protocol.model_copy(
        update={"record_id": "single-author", "max_authors_per_round": 1}
    )
    current = reduced.model_copy(
        update={"record_id": "two-authors", "max_authors_per_round": 2}
    )
    revisions = (
        BudgetRevision(previous=request.protocol, current=reduced),
        BudgetRevision(previous=reduced, current=current),
    )
    workspace = EvolutionWorkspace(tmp_path / "increased", baseline(), current)
    workspace.put_bundle(baseline())
    state["protocol"] = current.record_id
    state["budget_revisions"] = [r.model_dump(mode="json") for r in revisions]
    decision = RoundDecision.model_validate_json(
        Path(state["rounds"]["0"]["decision"]["path"]).read_text()
    ).model_copy(update={"width": 1})
    path = tmp_path / "single-author-decision.json"
    atomic_json(path, decision.model_dump(mode="json"))
    state["rounds"]["0"]["decision"] = artifact_record(path)
    frozen_bytes = path.read_bytes()
    workspace.save(state)

    async def forbidden(*args, **kwargs):
        raise AssertionError("author/evaluation launch is forbidden")

    evolution = GraphEvolution(
        workspace, SimpleNamespace(policy=None), forbidden, analyzer=forbidden
    )
    proposals = asyncio.run(evolution._proposals(state, {}, 0))
    assert len(proposals) == 1 and proposals[0].author_index == 0
    assert proposals[0].task_protocol == current
    assert asyncio.run(evolution._proposals(state, {}, 0)) == proposals
    assert path.read_bytes() == frozen_bytes
    assert (original.root / "evolution.json").read_bytes() == original_bytes
    assert state["attempts"] == json.loads(original_bytes)["attempts"]


@pytest.mark.parametrize("change", ["unpublished", "budget", "reverse"])
def test_archive_rejects_author_protocol_changes_outside_execution_chain(
    tmp_path, change
):
    original, request, _, original_bytes = settled_graph(tmp_path)
    manifest, current, _ = active_runtime(tmp_path / "active", request.protocol)
    continuation = EvolutionWorkspace(tmp_path / "continuation", baseline(), current)
    continuation.put_bundle(baseline())
    state = json.loads(original_bytes)
    state["protocol"] = current.record_id
    state["execution_revisions"] = [manifest.execution_revision.model_dump(mode="json")]
    decision = RoundDecision.model_validate_json(
        Path(state["rounds"]["0"]["decision"]["path"]).read_text()
    )
    from reweave.evolution.contracts import Proposal

    value = current
    if change == "unpublished":
        value = revised_protocol(current)
    elif change == "budget":
        value = current.model_copy(
            update={
                "record_id": "extended-protocol",
                "max_rounds": current.max_rounds + 1,
            }
        )
        state["budget_revisions"] = [
            BudgetRevision(previous=current, current=value).model_dump(mode="json")
        ]
        state["protocol"] = value.record_id
    else:
        state["rounds"]["0"]["request"]["protocol"] = current.model_dump(mode="json")
        value = request.protocol
    proposal = Proposal(
        record_id="round_000000_author_000",
        round_index=0,
        author_index=0,
        parent=decision.parent,
        references=decision.references,
        scopes=("W", "C", "L"),
        evidence=request.evidence,
        history=request.history,
        graph=decision.graph,
        execution_graph=request.graph,
        progress_map=decision.progress_map,
        decision=decision_file_ref(state["rounds"]["0"]),
        task_protocol=value,
    )
    state["proposals"][proposal.record_id] = proposal.model_dump(mode="json")
    with pytest.raises(
        ValidationError, match="author differs from its frozen graph round"
    ):
        EvolutionState.model_validate(state)


@pytest.mark.parametrize("frozen_width", [1, 2])
def test_author_cap_reduction_preserves_frozen_width_and_blocks_extra_pending_authors(
    tmp_path, frozen_width
):
    from reweave.evolution.contracts import EvolutionIncomplete, Proposal
    from reweave.evolution.evidence.records import artifact_record

    original, request, _, original_bytes = settled_graph(tmp_path)
    state = json.loads(original_bytes)
    old = request.protocol
    current = old.model_copy(
        update={"record_id": "single-author-protocol", "max_authors_per_round": 1}
    )
    revision = BudgetRevision(previous=old, current=current)
    workspace = EvolutionWorkspace(tmp_path / "reduced", baseline(), current)
    workspace.put_bundle(baseline())
    state["protocol"] = current.record_id
    state["budget_revisions"] = [revision.model_dump(mode="json")]
    old_decision = RoundDecision.model_validate_json(
        Path(state["rounds"]["0"]["decision"]["path"]).read_text()
    )
    decision = old_decision.model_copy(update={"width": frozen_width})
    path = tmp_path / "frozen-decision.json"
    atomic_json(path, decision.model_dump(mode="json"))
    state["rounds"]["0"]["decision"] = artifact_record(path)
    frozen_bytes = path.read_bytes()
    workspace.save(state)

    async def forbidden(*args, **kwargs):
        raise AssertionError("author/evaluation launch is forbidden")

    evolution = GraphEvolution(
        workspace, SimpleNamespace(policy=None), forbidden, analyzer=forbidden
    )
    if frozen_width == 1:
        proposals = asyncio.run(evolution._proposals(state, {}, 0))
        assert len(proposals) == 1 and proposals[0].author_index == 0
        assert proposals[0].task_protocol == current
        assert proposals[0].decision.path == str(path)
        assert asyncio.run(evolution._proposals(state, {}, 0)) == proposals
    else:
        unchanged = copy.deepcopy(state)
        with pytest.raises(EvolutionIncomplete, match="frozen round width exceeds"):
            asyncio.run(evolution._proposals(state, {}, 0))
        proposal = Proposal(
            record_id="pending-old-author",
            round_index=0,
            author_index=0,
            parent=decision.parent,
            references=decision.references,
            scopes=("W", "C", "L"),
            evidence=request.evidence,
            history=request.history,
            graph=decision.graph,
            execution_graph=request.graph,
            progress_map=decision.progress_map,
            decision=state["rounds"]["0"]["decision"],
            task_protocol=old,
        )
        with pytest.raises(EvolutionIncomplete, match="pending authors cannot launch"):
            asyncio.run(evolution._author(state, proposal))
        assert state == unchanged
    assert path.read_bytes() == frozen_bytes
    assert (original.root / "evolution.json").read_bytes() == original_bytes
    assert state["attempts"] == json.loads(original_bytes)["attempts"]
