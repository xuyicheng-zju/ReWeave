# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Bounded physical-attempt retries and receipt recovery without real rollouts."""

import json
from pathlib import Path

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    BundleVersion,
    EvaluationRecord,
    EvolutionIncomplete,
    EvolutionProtocol,
    Usage,
)
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.evolution.search.matrix import passes, result_matrix
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def graph(tmp_path):
    values = protocol().model_dump(mode="json")
    values.update(
        infrastructure_retries=2,
        max_rounds=0,
    )
    return EvolutionWorkspace(
        tmp_path, baseline(), EvolutionProtocol.model_validate(values)
    )


class InterruptedAfterReceipt(RuntimeError):
    """Simulate driver loss after durable result, before graph completion."""


class Evaluator:
    def __init__(self, outcomes, *, interrupt_after=None):
        self.outcomes = outcomes
        self.interrupt_after = interrupt_after
        self.calls = []

    async def __call__(self, request, bundle, directory, *, before_launch):
        ordinal = len(self.calls)
        assert ordinal < len(self.outcomes), "unexpected extra rollout"
        before_launch()
        self.calls.append((request, directory))
        outcome = self.outcomes[ordinal]
        failure_origin = "infrastructure" if outcome == "permanent" else outcome
        result = EvaluationRecord(
            record_id=new_record_id(),
            request=request,
            official_success=outcome == "none",
            failure_origin=failure_origin,
            infrastructure_retryable=(
                outcome == "infrastructure"
                if failure_origin == "infrastructure"
                else None
            ),
            usage=Usage(
                physical_steps=11 * (ordinal + 1),
                model_calls=ordinal + 1,
                wall_time_s=1.5 * (ordinal + 1),
                cost_usd=0.1 * (ordinal + 1),
            ),
            evidence=(artifact(directory, "trace.json", {"attempt": ordinal}),),
            episode=None,
            initialization=request.instance.initialization,
        )
        atomic_json(directory / "result.json", result.model_dump(mode="json"))
        if ordinal == self.interrupt_after:
            raise InterruptedAfterReceipt
        return result


async def forbidden_adapter(*args, **kwargs):
    pytest.fail("a completed attempt must not invoke an adapter")


async def evaluate(search):
    with search.workspace.transaction() as state:
        search.workspace.put_bundle(search.workspace.baseline)
        node_id = search.workspace.add_version(
            state,
            BundleVersion(
                record_id="baseline",
                bundle_id=search.workspace.baseline.record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        return await search._evaluate(state, node_id)


def receipts(graph):
    return {
        path: path.read_bytes()
        for path in sorted((graph.root / "attempts").rglob("result.json"))
    }


def assert_accounting(search, evaluator, launches):
    state = json.loads(search.workspace.state_path.read_text())
    attempts = list(state["attempts"].values())
    assert len(attempts) == launches
    assert all(item["launches"] == 1 for item in attempts)
    assert all(item["status"] == "completed" for item in attempts)
    assert len({path for _, path in evaluator.calls}) == launches
    assert len({request.record_id for request, _ in evaluator.calls}) == 1
    assert {request.execution for request, _ in evaluator.calls} == {
        search.workspace.protocol.execution_id
    }
    assert {Path(item["key"]).name for item in attempts} == {
        f"attempt-{index + 1:02d}" for index in range(launches)
    }
    usage = search.accounting(state)
    cumulative = launches * (launches + 1) // 2
    assert usage["episode_launches"] == launches
    assert usage["author_launches"] == 0
    assert usage["physical_steps"] == 11 * cumulative
    assert usage["model_calls"] == cumulative
    assert usage["wall_time_s"] == 1.5 * cumulative
    assert usage["cost_usd"] == pytest.approx(0.1 * cumulative)
    assert usage["reserved_physical_steps"] == 0
    assert usage["physical_steps_complete"]
    return usage


@pytest.mark.anyio
@pytest.mark.parametrize("infrastructure_failures", [0, 1, 2])
@pytest.mark.parametrize("outcome", ["agent", "none"])
async def test_infrastructure_retries_stop_at_first_scored_result_and_resume(
    graph, outcome, infrastructure_failures
):
    evaluator = Evaluator(("infrastructure",) * infrastructure_failures + (outcome,))
    launches = infrastructure_failures + 1
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    records = await evaluate(search)
    assert len(records) == 1
    assert records[0].failure_origin == outcome
    assert records[0].usage.physical_steps == 11 * launches
    assert records[0].official_success is (outcome == "none")
    measured = assert_accounting(search, evaluator, launches)
    original = receipts(graph)
    assert len(original) == launches
    state = json.loads(graph.state_path.read_text())
    history = await search._freeze_history(state, await search.matrix(state), 0)
    navigation = json.loads(Path(history.originals.path).read_text())["navigation"]
    assert len(navigation["bundles"]) == 1
    labels = navigation["bundles"][0]["evaluation_attempts"]
    assert len(labels) == launches
    from reweave.evolution.contracts import OriginalReference
    from reweave.evolution.evidence.trajectory import read_original

    indexed = [navigation["attempts"][label] for label in labels]
    assert all(entry["evidence_id"] is None for entry in indexed)
    assert [entry["failure_origin"] for entry in indexed] == [
        "infrastructure"
    ] * infrastructure_failures + [outcome]
    assert [
        read_original(OriginalReference(**entry["receipt"]))["usage"]["cost_usd"]
        for entry in indexed
    ] == pytest.approx([0.1 * (index + 1) for index in range(launches)])
    assert receipts(graph) == original

    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    assert await evaluate(resumed) == records
    assert receipts(graph) == original
    assert assert_accounting(resumed, evaluator, launches) == measured


@pytest.mark.anyio
async def test_three_infrastructure_failures_exhaust_without_replay_on_resume(graph):
    evaluator = Evaluator(("infrastructure",) * 3)
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    selected = await evaluate(search)
    records = search._records(json.loads(graph.state_path.read_text()))
    assert records[0].failure_origin == "infrastructure"
    assert result_matrix(graph.protocol, (baseline().record_id,), records) == {
        baseline().record_id: (None,)
    }
    measured = assert_accounting(search, evaluator, 3)
    original = receipts(graph)
    assert len(original) == 3

    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    assert await evaluate(resumed) == selected
    assert receipts(graph) == original
    assert assert_accounting(resumed, evaluator, 3) == measured


@pytest.mark.anyio
async def test_nonretryable_infrastructure_receipt_stays_null_without_replay(graph):
    evaluator = Evaluator(("permanent",))
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    records = await evaluate(search)
    assert result_matrix(graph.protocol, (baseline().record_id,), records) == {
        baseline().record_id: (None,)
    }
    original = receipts(graph)
    assert len(evaluator.calls) == 1
    assert len(original) == 1

    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    assert await evaluate(resumed) == records
    assert receipts(graph) == original
    assert len(evaluator.calls) == 1


@pytest.mark.anyio
async def test_cached_receipt_does_not_require_unread_attachment(graph):
    evaluator = Evaluator(("agent",))
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    records = await evaluate(search)
    Path(records[0].evidence[0].path).unlink()
    originals = receipts(graph)
    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    assert await evaluate(resumed) == records
    assert receipts(graph) == originals
    assert len(evaluator.calls) == 1
    state = json.loads(graph.state_path.read_text())
    assert resumed._records(state) == records


@pytest.mark.anyio
async def test_unknown_physical_outcome_stops_without_replay(graph):
    evaluator = Evaluator(("unknown",))
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    with pytest.raises(EvolutionIncomplete, match="unacknowledged physical outcome"):
        await evaluate(search)
    original = receipts(graph)
    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    with pytest.raises(EvolutionIncomplete, match="unacknowledged physical outcome"):
        await evaluate(resumed)
    assert receipts(graph) == original
    assert len(evaluator.calls) == 1


@pytest.mark.parametrize(
    "row,threshold",
    [
        ((False, False, False, None, False), 1.0),
        ((True, True, True, None, True), 1.0),
        ((None,) * 5, 1.0),
        ((True, True, True, False, None), 0.8),
        ((True, True, False, False, None), 0.8),
    ],
)
def test_missing_outcomes_never_qualify(row, threshold):
    assert not passes(row, threshold)


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["permanent", "infrastructure"])
async def test_disqualified_partial_pool_preserves_null_costs_and_no_replay(
    tmp_path, outcome
):
    values = protocol().model_dump(mode="json")
    values.update(
        dev=[
            {
                "record_id": f"drawer-{i}",
                "task": "drawer",
                "initialization": {"seed": i},
            }
            for i in range(1, 6)
        ],
        infrastructure_retries=0,
    )
    workspace = EvolutionWorkspace(
        tmp_path, baseline(), EvolutionProtocol.model_validate(values)
    )
    evaluator = Evaluator(("agent", "agent", "agent", outcome, "agent"))
    search = GraphEvolution(workspace, forbidden_adapter, evaluator)
    records = await evaluate(search)
    state = json.loads(workspace.state_path.read_text())
    assert result_matrix(workspace.protocol, (baseline().record_id,), records)[
        baseline().record_id
    ] == (False, False, False, None, False)
    assert search.accounting(state)["episode_launches"] == 5
    assert search.accounting(state)["physical_steps"] == 165
    assert search.accounting(state)["cost_usd"] == pytest.approx(1.5)
    original = receipts(workspace)
    resumed = GraphEvolution(
        workspace, forbidden_adapter, forbidden_adapter, recover=True
    )
    assert await evaluate(resumed) == records
    assert receipts(workspace) == original
    assert len(evaluator.calls) == 5


@pytest.mark.anyio
async def test_receipt_before_graph_completion_is_reused_before_remaining_retry(graph):
    evaluator = Evaluator(
        ("infrastructure", "infrastructure", "agent"), interrupt_after=1
    )
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    with pytest.raises(InterruptedAfterReceipt):
        await evaluate(search)
    original = receipts(graph)
    assert len(original) == 2
    state = json.loads(graph.state_path.read_text())
    pending = [
        item for item in state["attempts"].values() if item["status"] == "failed"
    ]
    assert len(pending) == 1 and pending[0]["result_id"] is None
    assert search.accounting(state)["episode_launches"] == 2
    assert search.accounting(state)["physical_steps"] == 33

    resumed = GraphEvolution(graph, forbidden_adapter, evaluator, recover=True)
    records = await evaluate(resumed)
    assert len(records) == 1 and records[0].failure_origin == "agent"
    assert len(evaluator.calls) == 3
    assert all(path.read_bytes() == value for path, value in original.items())
    assert_accounting(resumed, evaluator, 3)
