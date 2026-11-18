# Modified for the ReWeave namespace.
"""Failed and cancelled adapter attempts remain durably visible."""

import asyncio
import json

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import AuthoredMutation, Candidate
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_author_concurrency import setup_author
from tests.unit_tests.harness.evolution.test_bundle_author import host_input


def authored(directory):
    return AuthoredMutation(
        record_id=new_record_id(),
        proposal_id="a" * 64,
        candidates=(
            Candidate(
                experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                branch_id="repair",
                bundle=baseline(),
                summary="Synthetic branch.",
            ),
        ),
        summary="Synthetic controller completion; no model or environment executed.",
        session_id="test-session",
        usage=usage().model_copy(update={"physical_steps": 0, "model_calls": 1}),
        artifacts=(artifact(directory, "output.json", {"accepted": True}),),
    )


@pytest.mark.anyio
async def test_cancelled_adapter_records_interruption_without_execution_receipt(
    tmp_path, monkeypatch
):
    _, proposal, _ = setup_author(tmp_path, monkeypatch)
    graph = EvolutionWorkspace(tmp_path / "search", baseline(), protocol())
    search = GraphEvolution(graph, None, None)
    started = asyncio.Event()

    async def invoke(directory, before_launch):
        before_launch()
        started.set()
        await asyncio.Event().wait()

    with graph.transaction() as state:
        task = asyncio.create_task(
            search._attempt(
                state, "author", "cancel", proposal, invoke, AuthoredMutation
            )
        )
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        aid, item = next(iter(state["attempts"].items()))
        assert item["status"] == "cancelled" and item["launches"] == 1
        directory = graph.root / "attempts" / aid
        failure = json.loads((graph.root / item["last_failure"]["path"]).read_text())
        assert failure["error_type"] == "CancelledError"
        assert failure["launch_registered_during_invocation"]
        assert "CancelledError" in failure["traceback"]
        assert not (directory / "result.json").exists()
        assert not search._active_attempts


@pytest.mark.anyio
async def test_uncapping_recovers_timed_out_exact_request_without_replaying_invoke(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from reweave.evolution.contracts import BudgetRevision
    from reweave.evolution.search.engine import AuthorWallTimeExceeded

    _, proposal, _ = setup_author(tmp_path, monkeypatch)
    capped = protocol().model_copy(update={"author_wall_time_s": 0.01})
    graph = EvolutionWorkspace(tmp_path / "search", baseline(), capped)
    search = GraphEvolution(graph, None, None)

    async def invoke(directory, before_launch):
        before_launch()
        await asyncio.Event().wait()

    with graph.transaction() as state:
        with pytest.raises(AuthorWallTimeExceeded):
            await search._attempt(
                state, "author", "resume", proposal, invoke, AuthoredMutation
            )
        original = json.loads(json.dumps(state["attempts"]["resume"]))
        failure_path = graph.root / original["last_failure"]["path"]
        failure_bytes = failure_path.read_bytes()
        uncapped = capped.model_copy(
            update={"record_id": "uncapped-author", "author_wall_time_s": None}
        )
        revision = BudgetRevision(previous=capped, current=uncapped)
        state["budget_revisions"].append(revision.model_dump(mode="json"))
        state["protocol"] = uncapped.record_id
        graph.protocol = uncapped
        graph.save(state)

        async def recover(request, directory, *, before_launch):
            assert request == proposal
            before_launch()
            accounting = resumed.accounting(state)
            assert accounting["reserved_wall_time_s"] is None
            await asyncio.sleep(0.02)
            return authored(directory)

        resumed = GraphEvolution(
            graph, SimpleNamespace(recover=recover), None, recover=True
        )
        result = await resumed._attempt(
            state,
            "author",
            "resume",
            proposal,
            lambda *_: pytest.fail("must use adapter recovery, never replay invoke"),
            AuthoredMutation,
        )
        item = state["attempts"]["resume"]
        assert item["request"] == original["request"]
        assert item["launches"] == 2 and item["status"] == "completed"
        assert item["deadline_at"] is None and result.session_id == "test-session"
        assert failure_path.read_bytes() == failure_bytes
        recovery = list((graph.root / "audit/attempt_recoveries/resume").glob("*.json"))
        assert len(recovery) == 1
        assert json.loads(recovery[0].read_text())["previous_status"] == "timed_out"


@pytest.mark.anyio
async def test_recovery_preserves_each_failed_preparation_tree(tmp_path, monkeypatch):
    author, proposal, prepare = setup_author(tmp_path, monkeypatch)
    directory = tmp_path / "branch"
    attempts = 0
    originals = [
        b"first complete partial artifact\x00\xff",
        b"second partial artifact" * 1000,
    ]

    def fail_prepare(request, destination):
        nonlocal attempts
        destination.mkdir(parents=True)
        (destination / "partial.bin").write_bytes(originals[attempts])
        attempts += 1
        raise ValueError("preparation interrupted")

    monkeypatch.setattr(author, "_prepare", fail_prepare)
    for resume in [False, True]:
        with pytest.raises(ValueError, match="preparation interrupted"):
            await author(
                proposal,
                directory,
                resume=resume,
                before_launch=lambda: pytest.fail("no native query is allowed"),
            )
    monkeypatch.setattr(author, "_prepare", prepare)
    author._prepare_inputs(proposal, directory, resume=True)
    archived = sorted((directory / "audit/preparation_failures").glob("*/partial.bin"))
    assert len(archived) == 2
    assert {p.read_bytes() for p in archived} == set(originals)
    assert host_input(directory / "workspace", "proposal.json").exists()
    assert not (directory / "workspace/.host_inputs").exists()
    assert not (directory / ".scratch/runtime/preparation").exists()
    assert not (directory / "native_result.json").exists()
    assert not (directory / "result.json").exists()
    assert not (directory / "audit/queries").exists()


@pytest.mark.anyio
async def test_offline_queue_does_not_spend_execution_deadline(tmp_path, monkeypatch):
    _, proposal, _ = setup_author(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_state.json").write_text("{}")
    (campaign / "scheduler_policy.json").write_text('{"mode":"episode_pool"}')
    drain = campaign / "scheduler_drain.json"
    drain.write_text('{"owner":"operator"}')
    graph = EvolutionWorkspace(campaign / "tasks/a/evolution", baseline(), protocol())
    search = GraphEvolution(graph, None, None)

    async def invoke(directory, before_launch):
        before_launch()
        return authored(directory)

    with graph.transaction() as state:
        task = asyncio.create_task(
            search._attempt(
                state,
                "author",
                "queued",
                proposal,
                invoke,
                AuthoredMutation,
                wall_time_s=0.03,
            )
        )
        await asyncio.sleep(0.08)
        pending = state["attempts"]["queued"]
        assert pending["deadline_at"] is None and pending["launches"] == 0
        assert not task.done()
        drain.unlink()
        result = await asyncio.wait_for(task, 3)
        assert result.session_id == "test-session"
        assert state["attempts"]["queued"]["status"] == "completed"
    waiting = list((campaign / "reports/offline_admissions").glob("*.json"))
    assert len(waiting) == 1
    assert json.loads(waiting[0].read_text())["waiting_s"] >= 0.08


@pytest.mark.anyio
async def test_actual_adapter_preparation_still_spends_execution_deadline(
    tmp_path, monkeypatch
):
    from reweave.evolution.search.engine import AuthorWallTimeExceeded

    _, proposal, _ = setup_author(tmp_path, monkeypatch)
    graph = EvolutionWorkspace(tmp_path / "search", baseline(), protocol())
    search = GraphEvolution(graph, None, None)

    async def prepare_forever(directory, before_launch):
        await asyncio.Event().wait()

    with graph.transaction() as state:
        with pytest.raises(AuthorWallTimeExceeded):
            await search._attempt(
                state,
                "author",
                "preparation",
                proposal,
                prepare_forever,
                AuthoredMutation,
                wall_time_s=0.01,
            )
        assert state["attempts"]["preparation"]["status"] == "timed_out"
        assert state["attempts"]["preparation"]["launches"] == 0


@pytest.mark.anyio
async def test_reconciled_old_queue_time_is_not_subtracted_on_later_recovery(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from reweave.evolution.search.engine import AuthorWallTimeExceeded

    _, proposal, _ = setup_author(tmp_path, monkeypatch)
    graph = EvolutionWorkspace(
        tmp_path / "search",
        baseline(),
        protocol().model_copy(update={"author_wall_time_s": 0.15}),
    )
    search = GraphEvolution(graph, None, None)

    async def old_wrapper_waits_without_entering_delegate(directory, before_launch):
        await asyncio.Event().wait()

    with graph.transaction() as state:
        with pytest.raises(AuthorWallTimeExceeded):
            await search._attempt(
                state,
                "author",
                "queued",
                proposal,
                old_wrapper_waits_without_entering_delegate,
                AuthoredMutation,
            )
        original = json.loads(json.dumps(state["attempts"]["queued"]))
        failure = graph.root / original["last_failure"]["path"]
        original_bytes = failure.read_bytes()
        correction = graph.root / "audit/offline_queue_reconciliation/queue.json"
        correction.parent.mkdir(parents=True)
        correction.write_text(json.dumps({"attempt": "queued", "original": original}))
        state["attempts"]["queued"].update(status="prepared", deadline_at=None)
        graph.save(state)
        calls = []

        async def recover(request, directory, *, before_launch):
            before_launch()
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient execution failure")
            return authored(directory)

        resumed = GraphEvolution(
            graph, SimpleNamespace(recover=recover), None, recover=True
        )
        with pytest.raises(RuntimeError, match="transient"):
            await resumed._attempt(
                state, "author", "queued", proposal, None, AuthoredMutation
            )
        result = await resumed._attempt(
            state, "author", "queued", proposal, None, AuthoredMutation
        )
        assert result.session_id == "test-session"
        assert len(calls) == 2
        assert failure.read_bytes() == original_bytes
