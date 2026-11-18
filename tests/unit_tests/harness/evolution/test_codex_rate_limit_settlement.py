"""A terminal native turn can race the Host's 429 interruption request."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from openai_codex.errors import InvalidRequestError

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.infrastructure.native_codex import _interrupt_codex_turn
from reweave.utils.persistence import atomic_json
from reweave.utils.provider_admission import ProviderAdmission
from tests.unit_tests.harness.evolution.fixtures import usage


def test_completed_turn_does_not_replace_rate_limit_with_rpc_error():
    async def interrupt():
        raise InvalidRequestError(-32600, "no active turn to interrupt")

    assert (
        asyncio.run(_interrupt_codex_turn(SimpleNamespace(interrupt=interrupt)))
        is False
    )


@pytest.mark.parametrize(
    "code,message",
    [(-32600, "wrong turn identity"), (-32601, "no active turn to interrupt")],
)
def test_other_rpc_errors_remain_visible(code, message):
    async def interrupt():
        raise InvalidRequestError(code, message)

    with pytest.raises(InvalidRequestError):
        asyncio.run(_interrupt_codex_turn(SimpleNamespace(interrupt=interrupt)))


def test_active_turn_interruption_is_preserved():
    called = []

    async def interrupt():
        called.append(True)

    assert (
        asyncio.run(_interrupt_codex_turn(SimpleNamespace(interrupt=interrupt))) is True
    )
    assert called == [True]


@pytest.mark.parametrize("error_event", [True, False])
def test_failed_native_stream_preserves_journal_and_reports_one_transport_incident(
    tmp_path, monkeypatch, error_event
):
    workspace = tmp_path / "workspace"
    (workspace / "context").mkdir(parents=True)
    atomic_json(workspace / "context/input.json", {"kind": "mutation"})
    path = tmp_path / "audit/queries/000000.json"
    native = tmp_path / "native.jsonl"
    native.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "native-thread"}}) + "\n"
    )
    error = {
        "codex_error_info": "other",
        "message": "stream disconnected before completion: Transport error: network error: error decoding response body",
    }
    closed = []

    class Turn:
        async def stream(self):
            if error_event:
                yield {"method": "error", "payload": {"error": error}}
            yield {
                "method": "turn/completed",
                "payload": {"turn": {"status": "failed", "error": error}},
            }

    class Thread:
        id = "native-thread"

        async def turn(self, *args, **kwargs):
            return Turn()

        async def read(self, **kwargs):
            return SimpleNamespace(thread=SimpleNamespace(id=self.id, path=str(native)))

    class Client:
        async def thread_start(self, **kwargs):
            return Thread()

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "reweave.planner.codex_connection.create_codex_client", lambda **kw: Client()
    )
    monkeypatch.setattr(
        "reweave.planner.codex_connection.codex_connection_config",
        lambda _auth: {"model_provider": "openai"},
    )
    gate = ProviderAdmission("test", root=tmp_path / "admission", capacity=2)
    reported = []
    note_transport_failure = gate.note_transport_failure

    def note_failure(error):
        reported.append(error)
        return note_transport_failure(error)

    monkeypatch.setattr(gate, "note_transport_failure", note_failure)
    author = NativeBundleAuthor(
        bundle_loader=None,
        evidence_workspace=tmp_path / "evidence",
        policy=AuthorPolicy(author="codex"),
    )
    with pytest.raises(RuntimeError, match="error decoding response body"):
        asyncio.run(
            author._codex_query(
                workspace,
                tmp_path,
                (),
                (),
                lambda _: None,
                time.monotonic(),
                path,
                lambda: atomic_json(path, usage().model_dump(mode="json")),
                "Read only",
                "Inspect retained evidence",
                None,
                gate,
            )
        )
    state = json.loads((gate.root / "state.json").read_text())
    assert len(reported) == 1 and state["transport_strikes"] == 1
    assert not state.get("probe_required")
    assert state.get("generation", 0) == 0
    assert state.get("cooldown_until", 0) <= time.time()
    assert (tmp_path / "session/transcript.jsonl").exists()
    assert closed == [True]
