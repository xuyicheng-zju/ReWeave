"""Native text streaming retains evidence without rewriting derived usage per token."""

import asyncio
import json
from pathlib import Path

import pytest

import reweave.evolution.authoring.author as author_module
import reweave.evolution.infrastructure.native_accounting as native_accounting
from reweave.evolution.authoring.author import NativeBundleAuthor, cumulative_usage
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.contracts import Proposal
from reweave.planner.utils.http_mcp_server import HttpMcpServer
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_author_timing import Clock
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


def delta(text="part"):
    return {
        "method": "item/agentMessage/delta",
        "payload": {
            "delta": text,
            "item_id": "message",
            "thread_id": "session",
            "turn_id": "turn",
        },
    }


def stream_author(tmp_path, monkeypatch, events, failures=()):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    clock = Clock()
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    monkeypatch.setattr(HttpMcpServer, "start", lambda self: None)
    monkeypatch.setattr(
        HttpMcpServer,
        "stop",
        lambda self, **kw: setattr(clock, "elapsed", clock.elapsed + 3),
    )
    monkeypatch.setattr(AuthorDevelopmentTools, "preflight", lambda *a, **kw: None)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(author="codex"),
    )
    writes, syncs, changes, snapshots = [], [], [], []
    original_atomic = author_module.atomic_json
    original_sync = author_module.os.fsync

    def write(path, value):
        writes.append(Path(path))
        return original_atomic(path, value)

    def sync(fd):
        syncs.append(fd)
        return original_sync(fd)

    monkeypatch.setattr(author_module, "atomic_json", write)
    monkeypatch.setattr(author_module.os, "fsync", sync)
    invocation = 0

    async def codex(*args):
        nonlocal invocation
        record, query_path, begin = args[4], args[6], args[7]
        begin()
        for event in events:
            clock.elapsed += 1
            before = len(writes), len(syncs)
            record(event)
            changes.append((len(writes) - before[0], len(syncs) - before[1]))
            snapshots.append(json.loads(query_path.read_text()))
        failure = failures[invocation] if invocation < len(failures) else None
        invocation += 1
        if failure:
            raise failure("stream interrupted")
        atomic_json(tmp_path / "native_result.json", {"fixture_completion": True})
        return cumulative_usage(tmp_path / "audit"), "session"

    async def publish(*args, **kwargs):
        return cumulative_usage(tmp_path / "audit")

    monkeypatch.setattr(author, "_codex", codex)
    monkeypatch.setattr(author, "_publish", publish)
    return author, proposal, clock, changes, snapshots


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_text_deltas_keep_complete_journal_and_final_usage_after_recovery(
    tmp_path, monkeypatch, failure
):
    events = [delta("one"), delta("two"), {"method": "item/completed", "payload": {}}]
    author, proposal, _, changes, snapshots = stream_author(
        tmp_path, monkeypatch, events, failures=[failure]
    )

    def run():
        return asyncio.run(author._execute_once(proposal, tmp_path, resume=True))

    if failure:
        with pytest.raises(failure, match="stream interrupted"):
            run()
        first = cumulative_usage(tmp_path / "audit")
        assert first.wall_time_s == 6 and first.wall_time_complete
        result = run()
        assert result.wall_time_s == 12
    else:
        result = run()
        assert result.wall_time_s == 6
    assert result.wall_time_complete
    assert result.physical_steps == 0
    assert result.model_calls is None and result.cost_usd is None
    assert changes[:3] == [(0, 0), (0, 0), (2, 4)]
    assert [item["wall_time_s"] for item in snapshots[:3]] == [0, 0, 3]
    rows = [
        json.loads(row)
        for row in (tmp_path / "audit/events.jsonl").read_text().splitlines()
    ]
    retained = [row for row in rows if "method" in row]
    assert retained == events * (2 if failure else 1)


@pytest.mark.parametrize(
    "event",
    [
        {"method": "thread/tokenUsage/updated", "payload": {"total_tokens": 20}},
        {"method": "item/completed", "payload": {}},
        {"method": "turn/completed", "payload": {}},
        {"method": "thread/started", "payload": {"thread_id": "session"}},
        {"method": "turn/interrupted", "payload": {}},
        {"method": "error", "payload": {"message": "network failed"}},
        {**delta(), "payload": {**delta()["payload"], "usage": 20}},
        {**delta(), "payload": {**delta()["payload"], "delta": None}},
    ],
)
def test_usage_identity_terminal_and_unknown_delta_payloads_still_commit(
    tmp_path, monkeypatch, event
):
    author, proposal, _, changes, _ = stream_author(tmp_path, monkeypatch, [event])
    measured = asyncio.run(author._execute_once(proposal, tmp_path, resume=True))
    assert changes == [(2, 4)]
    assert measured.wall_time_s == 4 and measured.wall_time_complete


def test_crash_reconciliation_uses_query_start_even_after_unsnapshotted_delta(
    tmp_path, monkeypatch
):
    author, proposal, clock, _, snapshots = stream_author(
        tmp_path, monkeypatch, [delta()]
    )
    asyncio.run(author._execute_once(proposal, tmp_path, resume=True))
    # The live lower bound before settlement is what an abrupt process exit leaves.
    crash = tmp_path / "crash"
    atomic_json(crash / "audit/queries/000000.json", snapshots[0])
    (crash / "audit/events.jsonl").write_text(
        json.dumps(
            {
                "event": "native_query_started",
                "query": "000000.json",
                "started_at": 1000,
            }
        )
        + "\n"
    )
    clock.calendar += 100
    author_module.reconcile_interrupted_timing(crash)
    measured = cumulative_usage(crash / "audit")
    assert measured.wall_time_s == 0
    assert measured.wall_time_charge_s == 104
    assert not measured.wall_time_complete
