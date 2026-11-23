# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Author accounting covers native cleanup and final immutable evidence binding."""

import asyncio
import json
import sys
import uuid
from types import SimpleNamespace

import pytest

import reweave.evolution.authoring.author as author_module
import reweave.evolution.infrastructure.native_accounting as native_accounting
from reweave.evolution.authoring.author import NativeBundleAuthor, cumulative_usage
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import Proposal, Usage
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.evolution.validation import admission
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol, usage
from tests.unit_tests.harness.evolution.test_bundle_author import (
    Journal,
    host_input,
    native_completion,
    workspace,
)


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.calendar = 1000.0

    def monotonic(self):
        return self.elapsed

    def time(self):
        return self.calendar + self.elapsed


def test_failed_preparation_is_durable_complete_search_usage(tmp_path, monkeypatch):
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture = workspace(fixture_root)
    proposal = Proposal.model_validate_json(
        host_input(fixture, "proposal.json").read_text()
    )
    directory = tmp_path / "search/attempts/preparation-failed"
    clock = Clock()
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )

    def fail_preparation(*args, **kwargs):
        del args, kwargs
        clock.elapsed += 7
        raise OSError("preparation failed")

    monkeypatch.setattr(author, "_prepare_inputs", fail_preparation)
    with pytest.raises(OSError, match="preparation failed"):
        asyncio.run(author.recover(proposal, directory, before_launch=lambda: None))

    measured = cumulative_usage(directory / "audit")
    assert measured.wall_time_s == 7
    assert measured.wall_time_complete
    assert measured.model_calls == 0 and measured.cost_usd == 0
    phase = json.loads((directory / "audit/phases.jsonl").read_text().splitlines()[-1])
    assert phase["event"] == "failed"
    assert phase["usage"] == "audit/preparation_usage/000000.json"

    graph = EvolutionWorkspace(tmp_path / "search", baseline(), protocol())
    search = GraphEvolution(graph, None, None)
    totals = search.accounting(
        {
            "rounds": {},
            "attempts": {
                "preparation-failed": {
                    "kind": "author",
                    "launches": 0,
                    "status": "failed",
                }
            },
        }
    )
    assert totals["author_launches"] == 0
    assert totals["wall_time_s"] == 7
    assert totals["wall_time_complete"]
    assert totals["model_calls"] == 0 and totals["cost_usd"] == 0


def test_prequery_failure_accounts_cleanup_and_each_recovery(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    clock = Clock()
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )

    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from reweave.planner.utils.http_mcp_server import HttpMcpServer

    def start(server):
        del server
        clock.elapsed += 2

    def preflight(*args, **kwargs):
        del args, kwargs
        clock.elapsed += 5
        raise PermissionError("development preflight rejected")

    def stop(server, **kwargs):
        del server, kwargs
        clock.elapsed += 3

    monkeypatch.setattr(HttpMcpServer, "start", start)
    monkeypatch.setattr(HttpMcpServer, "stop", stop)
    monkeypatch.setattr(AuthorDevelopmentTools, "preflight", preflight)
    launches = []
    for _ in range(2):
        with pytest.raises(PermissionError, match="preflight rejected"):
            asyncio.run(
                author.recover(
                    proposal,
                    tmp_path,
                    before_launch=lambda: launches.append(1),
                )
            )

    records = [
        Usage.model_validate_json(path.read_text())
        for path in sorted((tmp_path / "audit/preparation_usage").glob("*.json"))
    ]
    assert [record.wall_time_s for record in records] == [10, 10]
    assert all(record.wall_time_complete for record in records)
    assert all(record.model_calls == 0 and record.cost_usd == 0 for record in records)
    assert launches == []
    assert not list((tmp_path / "audit/queries").glob("*.json"))
    measured = cumulative_usage(tmp_path / "audit")
    assert measured.wall_time_s == 20 and measured.wall_time_complete
    assert measured.model_calls == 0 and measured.cost_usd == 0


def test_query_failure_is_not_double_counted_as_prequery_work(tmp_path, monkeypatch):
    from claude_agent_sdk import SystemMessage

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )

    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from reweave.planner.utils.http_mcp_server import HttpMcpServer

    monkeypatch.setattr(HttpMcpServer, "start", lambda self: None)
    monkeypatch.setattr(HttpMcpServer, "stop", lambda self, **kwargs: None)
    monkeypatch.setattr(AuthorDevelopmentTools, "preflight", lambda *a, **kw: None)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(author_module, "claude_workspace_options", lambda *a, **kw: {})

    async def query(*, prompt, options):
        del prompt
        yield SystemMessage(subtype="init", data={"session_id": str(uuid.uuid4())})
        del options
        raise RuntimeError("provider stream failed after query start")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    with pytest.raises(RuntimeError, match="provider stream failed"):
        asyncio.run(author.recover(proposal, tmp_path, before_launch=lambda: None))

    assert not list((tmp_path / "audit/preparation_usage").glob("*.json"))
    queries = list((tmp_path / "audit/queries").glob("*.json"))
    assert len(queries) == 1
    assert Usage.model_validate_json(queries[0].read_text()).wall_time_complete


def test_complete_wall_time_counts_each_stage_once_and_keeps_artifacts(
    tmp_path, monkeypatch
):
    from claude_agent_sdk import ResultMessage, SystemMessage

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    clock = Clock()
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    identity = str(uuid.uuid4())

    def timed(owner, name, seconds):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.elapsed += seconds
            return result

        monkeypatch.setattr(owner, name, wrapped)

    timed(author, "_prepare_inputs", 2)
    timed(author_module, "complete_query", 7)
    timed(author, "_load_completion", 11)
    timed(author, "_collect_result", 13)
    original_admit = admission.admit_candidate

    async def admit(*args, **kwargs):
        result = await original_admit(*args, **kwargs)
        clock.elapsed += 17
        return result

    monkeypatch.setattr(admission, "admit_candidate", admit)

    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from reweave.planner.utils.http_mcp_server import HttpMcpServer

    monkeypatch.setattr(HttpMcpServer, "start", lambda self: None)
    monkeypatch.setattr(
        HttpMcpServer,
        "stop",
        lambda self, **kw: setattr(clock, "elapsed", clock.elapsed + 9),
    )
    monkeypatch.setattr(
        AuthorDevelopmentTools,
        "preflight",
        lambda *a, **kw: setattr(clock, "elapsed", clock.elapsed + 3),
    )

    async def query(*, prompt, options):
        clock.elapsed += 5
        await options.session_store.append(
            {"session_id": identity, "project_key": "timing"},
            [
                {
                    "uuid": str(uuid.uuid4()),
                    "type": "assistant",
                    "message": {"content": "done"},
                }
            ],
        )
        yield SystemMessage(subtype="init", data={"session_id": identity})
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=identity,
            total_cost_usd=0.25,
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(author_module, "claude_workspace_options", lambda *a, **kw: {})
    result = asyncio.run(author.recover(proposal, tmp_path, before_launch=lambda: None))
    assert result.proposal_id == proposal.record_id
    assert result.usage.wall_time_s == 67
    assert result.usage.wall_time_complete
    assert result.usage.model_calls == 1 and result.usage.cost_usd == 0.25
    assert (
        json.loads((tmp_path / "audit/queries/000000.json").read_text())["wall_time_s"]
        == 26
    )
    check = json.loads((tmp_path / "audit/checks/000000.json").read_text())
    assert check["wall_time_s"] == 41
    assert check["validation_status"] == "passed"
    # Receipt is the earlier immutable commit-time snapshot, not accounting authority.
    receipt_bytes = (tmp_path / "native_result.json").read_bytes()
    assert json.loads(receipt_bytes)["usage"]["wall_time_s"] == 10
    for artifact in result.artifacts:
        artifact.verify()
    clock.calendar += 100000
    restored = asyncio.run(
        author.recover(
            proposal, tmp_path, before_launch=lambda: pytest.fail("already complete")
        )
    )
    assert restored == result
    assert (tmp_path / "native_result.json").read_bytes() == receipt_bytes
    assert len(list((tmp_path / "audit/checks").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="another proposal"):
        author.verify_result(
            proposal.model_copy(update={"record_id": "different_batch"}), tmp_path
        )
    parent = host_input(tmp_path / "workspace", "references.json")
    parent.write_text(parent.read_text() + "\n")
    assert author.verify_result(proposal, tmp_path) == result
    assert (tmp_path / "native_result.json").read_bytes() == receipt_bytes
    assert len(list((tmp_path / "audit/checks").glob("*.json"))) == 1


@pytest.mark.parametrize("error", [asyncio.CancelledError, RuntimeError])
def test_final_check_preserves_partial_cases_and_propagates_cancellation(
    tmp_path, monkeypatch, error
):
    workspace(tmp_path)
    monkeypatch.setattr(author_module, "ClaudeSessionStore", Journal)
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    marker = "complete response " * 20000
    partial = {
        "scope": "C",
        "case_index": 0,
        "calls": [{"result": {"marker": marker}}],
        "validation_status": "interrupted"
        if error is asyncio.CancelledError
        else "failed",
    }

    async def admit(*args, case_results, **kwargs):
        case_results.append(partial)
        raise error("test interruption")

    monkeypatch.setattr(admission, "admit_candidate", admit)
    expected = (
        asyncio.CancelledError
        if error is asyncio.CancelledError
        else author_module.AuthorCorrectionRequired
    )
    with pytest.raises(expected):
        asyncio.run(author._publish(tmp_path))
    check = json.loads((tmp_path / "audit/checks/000000.json").read_text())
    evidence = json.loads((tmp_path / "audit" / check["evidence"]["path"]).read_text())
    assert evidence["contract_case_results"] == [partial]
    assert check["validation_status"] == (
        "interrupted" if error is asyncio.CancelledError else "failed"
    )
    assert check["wall_time_complete"]
    assert not (tmp_path / "result.json").exists()
    assert bool(list((tmp_path / "audit/corrections").glob("*/feedback.json"))) is (
        error is RuntimeError
    )


def test_resume_uses_query_journal_not_receipt_snapshot_or_downtime(
    tmp_path, monkeypatch
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    monkeypatch.setattr(author_module, "ClaudeSessionStore", Journal)
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    receipt_bytes = (tmp_path / "native_result.json").read_bytes()
    measured = usage().model_copy(
        update={"wall_time_s": 70, "wall_time_complete": False}
    )
    atomic_json(
        tmp_path / "audit/queries/000000.json", measured.model_dump(mode="json")
    )
    (tmp_path / "audit/events.jsonl").write_text(
        json.dumps(
            {
                "event": "native_query_started",
                "query": "000000.json",
                "started_at": 100,
            }
        )
        + "\n"
    )
    clock = Clock()
    clock.calendar = 100000
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    original = author._collect_result

    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.elapsed += 3
        return result

    monkeypatch.setattr(author, "_collect_result", collect)
    result = asyncio.run(
        author.recover(
            proposal, tmp_path, before_launch=lambda: pytest.fail("no new query")
        )
    )
    assert result.usage.wall_time_s == 73
    assert not result.usage.wall_time_complete
    assert result.usage.wall_time_charge_s == 99903
    assert (tmp_path / "native_result.json").read_bytes() == receipt_bytes
    assert len(list((tmp_path / "audit/queries").glob("*.json"))) == 1
    for artifact in result.artifacts:
        artifact.verify()


def test_collection_failure_closes_check_without_publishing(tmp_path, monkeypatch):
    workspace(tmp_path)
    monkeypatch.setattr(author_module, "ClaudeSessionStore", Journal)
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    clock = Clock()
    monkeypatch.setattr(author_module, "time", clock)
    monkeypatch.setattr(native_accounting, "time", clock)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )

    def collect(*args, **kwargs):
        clock.elapsed += 19
        raise OSError("artifact integrity failure")

    monkeypatch.setattr(author, "_collect_result", collect)
    with pytest.raises(OSError, match="artifact integrity failure"):
        asyncio.run(author._publish(tmp_path))
    check = json.loads((tmp_path / "audit/checks/000000.json").read_text())
    assert check["wall_time_s"] == 19
    assert check["wall_time_complete"] and check["validation_status"] == "failed"
    evidence = json.loads((tmp_path / "audit" / check["evidence"]["path"]).read_text())
    assert evidence["error"]["type"] == "OSError"
    assert not (tmp_path / "result.json").exists()


def test_cancelled_candidate_worker_error_never_becomes_an_author_retry(
    tmp_path, monkeypatch
):
    import threading

    from reweave.evolution.evidence.records import inventory_episode

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    atomic_json(host_input(root, "proposal.json"), proposal.model_dump(mode="json"))
    atomic_json(
        tmp_path / "input_inventory.json",
        {
            "context": inventory_episode(root / "context"),
            "archive": {},
        },
    )
    monkeypatch.setattr(author_module, "ClaudeSessionStore", Journal)
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    launches = []

    def broken_candidate(workspace, **kwargs):
        entered.set()
        assert release.wait(5), "test worker was never released"
        exited.set()
        raise ValueError("complete candidate worker error after cancellation")

    monkeypatch.setattr(admission, "read_candidate", broken_candidate)

    async def scenario():
        task = asyncio.create_task(
            author.recover(proposal, tmp_path, before_launch=lambda: launches.append(1))
        )
        try:
            await asyncio.to_thread(entered.wait, 5)
            assert entered.is_set()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not exited.is_set()
            with pytest.raises(RuntimeError, match="already owned"):
                await author.recover(
                    proposal, tmp_path, before_launch=lambda: launches.append(1)
                )
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            if sys.version_info >= (3, 11):
                assert isinstance(caught.value.__cause__, ValueError)
            assert exited.is_set() and not launches
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    check = json.loads((tmp_path / "audit/checks/000000.json").read_text())
    evidence = json.loads((tmp_path / "audit" / check["evidence"]["path"]).read_text())
    assert check["validation_status"] == "interrupted"
    assert evidence["error"]["type"] == "CancelledError"
    assert (
        "complete candidate worker error after cancellation"
        in evidence["error"]["traceback"]
    )
    assert not list((tmp_path / "audit/corrections").glob("*/feedback.json"))
    assert not (tmp_path / "result.json").exists()
    # The branch lock is released only after the native worker is drained.
    import fcntl

    with (tmp_path / ".native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
