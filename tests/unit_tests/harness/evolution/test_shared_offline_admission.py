"""Shared native queries honor ordinary campaign admission before spending work."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from reweave.evolution.infrastructure.campaign_slots import offline_admission
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_shared_native_retry import SharedAuthor


def campaign(root, *, paused=False):
    atomic_json(root / "campaign_state.json", {})
    atomic_json(
        root / "scheduler_policy.json",
        {
            "mode": "episode_pool",
            "offline_admission": "operator_pause"
            if paused
            else "shared_request_gate_only",
        },
    )
    if paused:
        atomic_json(root / "scheduler_drain.json", {"owner": "test-operator-pause"})


def queries(directory):
    return sorted((directory / "audit/queries").glob("*.json"))


def test_paused_campaign_blocks_query_and_usage_start(tmp_path, monkeypatch):
    campaign(tmp_path, paused=True)
    directory = tmp_path / "rounds/001/author"
    author = SharedAuthor(tmp_path, [])

    async def run():
        waiting = asyncio.Event()

        @asynccontextmanager
        async def ordinary_gate(owner, *, required):
            assert owner == directory and required is True
            waiting.set()
            async with offline_admission(owner, required=required):
                yield

        monkeypatch.setattr(
            "reweave.evolution.distillation.author.offline_admission", ordinary_gate
        )
        task = asyncio.create_task(author({}, directory))
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            assert not task.done()
            assert author.calls == 0
            assert not queries(directory)
            assert not (directory / "native_result.json").exists()
            assert not (directory / "result.json").exists()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert author.calls == 0 and not queries(directory)
    assert not list((tmp_path / "scheduler/offline").glob("*.ticket"))
    receipts = list((tmp_path / "reports/offline_admissions").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["admitted_at"] is None


@pytest.mark.parametrize("transport_failure", [False, True])
def test_each_native_query_reacquires_and_releases_ordinary_admission(
    tmp_path, monkeypatch, transport_failure
):
    campaign(tmp_path)
    directory = tmp_path / "rounds/001/author"
    events, owners = [], []
    admitted = False

    @asynccontextmanager
    async def observed_gate(owner, *, required):
        nonlocal admitted
        assert required is True and not admitted
        owners.append(owner)
        async with offline_admission(owner, required=required):
            events.append("admitted")
            admitted = True
            try:
                yield
            finally:
                admitted = False
                events.append("released")

    class ObservedAuthor(SharedAuthor):
        async def _codex(self, *args, **kwargs):
            assert admitted
            events.append("query")
            return await super()._codex(*args, **kwargs)

        async def admit(self, *args, **kwargs):
            assert not admitted
            return await super().admit(*args, **kwargs)

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.offline_admission", observed_gate
    )
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _: 0
    )
    failures = (
        [RuntimeError("Connection failed: error sending request")]
        if transport_failure
        else []
    )
    author = ObservedAuthor(tmp_path, failures)
    result = asyncio.run(author({}, directory))
    expected = 2 if transport_failure else 1
    assert result["admitted"] and author.calls == expected
    assert owners == [directory] * expected
    assert events == ["admitted", "query", "released"] * expected
    assert len(queries(directory)) == expected
    assert all(
        json.loads(path.read_text())["wall_time_complete"]
        for path in queries(directory)
    )
    assert not (tmp_path / "scheduler/offline").exists()
    assert not (tmp_path / "reports/offline_admissions").exists()


def test_pause_after_host_rejection_blocks_same_authors_next_correction_query(
    tmp_path, monkeypatch
):
    campaign(tmp_path)
    directory = tmp_path / "rounds/001/author"

    class CorrectingAuthor(SharedAuthor):
        def prepare(self, request, directory):
            workspace, readable = super().prepare(request, directory)
            (workspace / "context").mkdir(exist_ok=True)
            (workspace / "candidate").mkdir(exist_ok=True)
            atomic_json(
                workspace / "context/input.json", {"kind": "shared_distillation"}
            )
            return workspace, readable

        async def admit(self, request, workspace, audit):
            campaign(tmp_path, paused=True)
            raise ValueError("Host rejected the source implementation manifest")

    author = CorrectingAuthor(tmp_path, [])

    async def run():
        correction_waiting = asyncio.Event()
        entries = 0

        @asynccontextmanager
        async def ordinary_gate(owner, *, required):
            nonlocal entries
            assert owner == directory and required is True
            entries += 1
            if entries == 2:
                correction_waiting.set()
            async with offline_admission(owner, required=required):
                yield

        monkeypatch.setattr(
            "reweave.evolution.distillation.author.offline_admission", ordinary_gate
        )
        task = asyncio.create_task(author({}, directory))
        try:
            await asyncio.wait_for(correction_waiting.wait(), 1)
            assert not task.done()
            assert entries == 2 and author.calls == 1
            assert len(queries(directory)) == 1
            assert not (directory / "result.json").exists()
            archived = list((directory / "audit/corrections").glob("*/feedback.json"))
            assert len(archived) == 1
            assert (
                "implementation manifest"
                in json.loads(archived[0].read_text())["error"]
            )
            assert (
                json.loads((directory / "session/identity.json").read_text())[
                    "session_id"
                ]
                == "original-thread"
            )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert author.calls == 1 and len(queries(directory)) == 1
    assert not list((tmp_path / "scheduler/offline").glob("*.ticket"))
