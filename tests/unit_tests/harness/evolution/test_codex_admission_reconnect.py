"""A gate handoff can reconnect queued jobs without another native query."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from reweave.evolution.infrastructure.native_codex import CodexSession
from reweave.planner.providers.base import NativeTransportError
from reweave.utils.persistence import atomic_json
from reweave.utils.provider_admission import NativeQuotaExhausted


class Session(CodexSession):
    def __init__(self, query):
        self.policy = SimpleNamespace(codex_auth_mode="auth")
        self._codex_query = query


def test_closed_waiting_socket_reconnects_without_spending_native_retry_budget(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "reweave.evolution.infrastructure.native_codex.native_retry_wait_s", lambda _: 0
    )
    directory = tmp_path / "original-attempt"
    identity = directory / "session/identity.json"
    atomic_json(identity, {"session_id": "original-session"})
    ledger = directory / "audit/native_transport_retries.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"attempt":1}\n{"attempt":2}\n{"attempt":3}\n')
    original_identity, original_ledger = identity.read_bytes(), ledger.read_bytes()
    token = tmp_path / "token"
    token.write_text("local-test-token")
    monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_TOKEN_FILE", str(token))
    connections, queries = [], []

    async def run():
        async def gate(reader, writer):
            connections.append(json.loads(await reader.readline()))
            if len(connections) == 1:
                writer.close()  # The old gate exits while this request waits.
                await writer.wait_closed()
                return
            writer.write(b'{"status":"admitted","capacity":48}\n')
            await writer.drain()
            assert json.loads(await reader.readline()) == {"succeeded": True}
            writer.write(b'{"status":"released"}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def query(workspace, task_directory, *args, admission, **kwargs):
            queries.append((workspace, task_directory, args, kwargs))
            assert len(connections) == 2
            assert identity.read_bytes() == original_identity
            assert ledger.read_bytes() == original_ledger
            assert not (directory / "audit/queries").exists()
            return "completed-original-query"

        async with await asyncio.start_server(gate, "127.0.0.1", 0) as server:
            host, port = server.sockets[0].getsockname()
            monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_ENDPOINT", f"{host}:{port}")
            assert (
                await Session(query)._codex(
                    tmp_path / "original-workspace", directory, "original-input"
                )
                == "completed-original-query"
            )

    asyncio.run(run())
    assert len(queries) == 1
    assert queries[0][:3] == (
        tmp_path / "original-workspace",
        directory,
        ("original-input",),
    )
    assert identity.read_bytes() == original_identity
    assert ledger.read_bytes() == original_ledger
    reconnects = list(
        (directory / "audit/provider_admission_reconnects").glob("*.json")
    )
    assert len(reconnects) == 1
    assert json.loads(reconnects[0].read_text())["request_sent"] is False


@pytest.mark.parametrize("phase", ["query", "release"])
def test_error_after_lease_entry_is_never_taken_by_admission_retry(
    tmp_path, monkeypatch, phase
):
    entered, queried = [], []

    class Admission:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def lease(self, *args):
            entered.append(True)
            yield self
            raise NativeTransportError("release failed", request_sent=False)

    async def query(*args, **kwargs):
        queried.append(True)
        if phase == "query":
            raise NativeTransportError("query failed", request_sent=False)
        return "completed"

    monkeypatch.setattr("reweave.utils.provider_admission.ProviderAdmission", Admission)
    with pytest.raises(NativeTransportError, match=phase):
        asyncio.run(Session(query)._codex(tmp_path, tmp_path))
    assert len(entered) == len(queried) == 1
    assert not (tmp_path / "audit/provider_admission_reconnects").exists()


@pytest.mark.parametrize(
    "error",
    [
        NativeTransportError("submission unknown", request_sent=True),
        NativeTransportError("submission unspecified", request_sent=None),
        NativeQuotaExhausted("usage limit reached"),
        RuntimeError("Shared admission refused a native request"),
    ],
)
def test_only_proven_unsubmitted_transport_errors_reconnect(
    tmp_path, monkeypatch, error
):
    entered = []

    class Admission:
        def __init__(self, *args, **kwargs):
            pass

        @asynccontextmanager
        async def lease(self, *args):
            entered.append(True)
            raise error
            yield self

    async def forbidden_query(*args, **kwargs):
        pytest.fail("No model query was admitted")

    monkeypatch.setattr("reweave.utils.provider_admission.ProviderAdmission", Admission)
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(Session(forbidden_query)._codex(tmp_path, tmp_path))
    assert len(entered) == 1
    assert not (tmp_path / "audit/provider_admission_reconnects").exists()


def test_waiting_admission_reconnect_remains_cancellable(tmp_path, monkeypatch):
    async def run():
        waiting = asyncio.Event()

        class Admission:
            def __init__(self, *args, **kwargs):
                pass

            @asynccontextmanager
            async def lease(self, *args):
                raise NativeTransportError("gate stopped", request_sent=False)
                yield self

        async def sleep(_):
            waiting.set()
            await asyncio.Event().wait()

        async def forbidden_query(*args, **kwargs):
            pytest.fail("No model query was admitted")

        monkeypatch.setattr(
            "reweave.utils.provider_admission.ProviderAdmission", Admission
        )
        monkeypatch.setattr(
            "reweave.evolution.infrastructure.native_codex.asyncio.sleep", sleep
        )
        task = asyncio.create_task(Session(forbidden_query)._codex(tmp_path, tmp_path))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert not (tmp_path / "audit/queries").exists()
