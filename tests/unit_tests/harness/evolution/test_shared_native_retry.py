"""Shared author transport recovery preserves sessions and bounded accounting."""

import asyncio
import json

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.infrastructure.native_retry import NATIVE_RETRY_LIMIT
from reweave.utils.persistence import atomic_json


class SharedAuthor(NativeSharedAuthor):
    source_queries = False

    def _instructions(self, request):
        return "Synthetic transport recovery fixture."

    def __init__(self, root, failures):
        super().__init__(
            bundle_loader=None,
            policy=AuthorPolicy(author="codex"),
            evidence_workspace=root,
        )
        self.failures = failures
        self.calls = 0

    def prepare(self, request, directory):
        workspace = directory / "workspace"
        workspace.mkdir(exist_ok=True)
        atomic_json(
            directory / "session/identity.json", {"session_id": "original-thread"}
        )
        return workspace, ()

    async def _codex(
        self,
        workspace,
        directory,
        readable,
        writable,
        record,
        started,
        query_path,
        begin,
        *args,
    ):
        begin()
        self.calls += 1
        assert (
            json.loads((directory / "session/identity.json").read_text())["session_id"]
            == "original-thread"
        )
        if self.failures:
            raise self.failures.pop(0)
        atomic_json(directory / "native_result.json", {"session_id": "original-thread"})

    async def admit(self, request, workspace, audit):
        return {"admitted": True}


def test_shared_transport_recovers_original_thread_and_retains_queries(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _: 0
    )
    author = SharedAuthor(
        tmp_path, [RuntimeError("Connection failed: error sending request")]
    )
    result = asyncio.run(author({}, tmp_path))
    assert result["admitted"] and author.calls == 2
    queries = sorted((tmp_path / "audit/queries").glob("*.json"))
    assert len(queries) == 2
    assert all(json.loads(p.read_text())["wall_time_complete"] for p in queries)
    event = json.loads((tmp_path / "audit/native_transport_retries.jsonl").read_text())
    assert event["session_id"] == "original-thread"
    assert event["maximum"] == NATIVE_RETRY_LIMIT


def test_shared_transport_budget_survives_controller_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _: 0
    )
    for _ in range(2):
        author = SharedAuthor(
            tmp_path, [RuntimeError("Connection failed: error sending request")] * 8
        )
        with pytest.raises(RuntimeError, match="Connection failed"):
            asyncio.run(author({}, tmp_path))
    ledger = (
        (tmp_path / "audit/native_transport_retries.jsonl").read_text().splitlines()
    )
    assert len(ledger) == NATIVE_RETRY_LIMIT
    assert author.calls == 1
    assert not (tmp_path / "result.json").exists()


def test_shared_authentication_failure_is_not_retried(tmp_path):
    author = SharedAuthor(tmp_path, [RuntimeError("401 Unauthorized")])
    with pytest.raises(RuntimeError, match="Unauthorized"):
        asyncio.run(author({}, tmp_path))
    assert author.calls == 1
    assert not (tmp_path / "audit/native_transport_retries.jsonl").exists()
