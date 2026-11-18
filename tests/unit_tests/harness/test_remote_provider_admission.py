"""Remote clients share the original account locks and its 429 recovery probe."""

import asyncio
import json
import threading

import pytest

from reweave.utils import provider_admission
from reweave.utils.provider_admission import ProviderAdmission
from reweave.utils.provider_admission_server import AdmissionServer


@pytest.fixture
def shared_gate(tmp_path, monkeypatch):
    gate = ProviderAdmission("test", root=tmp_path, capacity=2, start_interval_s=0)
    token = tmp_path / "token"
    token.write_text("test-private-token")
    server = AdmissionServer(("127.0.0.1", 0), gate, token.read_text())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "REWEAVE_PROVIDER_ADMISSION_ENDPOINT", f"127.0.0.1:{server.server_address[1]}"
    )
    monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_TOKEN_FILE", str(token))
    try:
        yield gate, ProviderAdmission("test")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_remote_and_existing_local_workers_share_one_capacity(shared_gate):
    local, remote = shared_gate

    async def run():
        async with local.lease(), remote.lease():
            entered = asyncio.Event()

            async def third():
                async with remote.lease():
                    entered.set()

            waiting = asyncio.create_task(third())
            await asyncio.sleep(0.05)
            assert not entered.is_set()
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        async with remote.lease(), local.lease():
            assert local._acquire()[0] is None

    asyncio.run(run())


def test_remote_429_shares_cooldown_and_generation_safe_recovery(
    shared_gate, monkeypatch
):
    local, remote = shared_gate
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    assert remote.note_rate_limit({"httpStatusCode": 429, "Retry-After": 90})
    state = json.loads((local.root / "state.json").read_text())
    assert state["cooldown_until"] == 1090
    assert local._acquire()[0] is None
    clock[0] = 1091

    async def run():
        async with remote.lease(), remote.lease():
            assert local._acquire()[0] is None
            remote.note_rate_limit("HTTP 429 Too Many Requests")

    asyncio.run(run())
    assert local._acquire()[0] is None
    assert json.loads((local.root / "state.json").read_text())["probe_required"]


@pytest.mark.parametrize(
    "reason",
    [
        "idle timeout waiting for SSE",
        "stream closed before response.completed",
        "Transport error: network error: error decoding response body",
    ],
)
def test_remote_transport_event_is_recorded_without_stalling_other_callers(
    shared_gate, reason
):
    local, remote = shared_gate
    assert remote.note_transport_failure(
        {
            "codexErrorInfo": "other",
            "message": "stream disconnected before completion: " + reason,
        }
    )
    state = json.loads((local.root / "state.json").read_text())
    assert state["transport_strikes"] == 1
    assert "last_rate_limit_at" not in state
    assert not state.get("probe_required") and not state.get("cooldown_until")

    async def run():
        async with remote.lease(), local.lease():
            assert local._acquire()[0] is None

    asyncio.run(run())


def test_remote_recovery_runs_four_long_requests_before_any_finishes(
    shared_gate, monkeypatch
):
    local, remote = shared_gate
    local.capacity = 48
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    remote.note_rate_limit({"http_status_code": 429})
    clock[0] += 61

    async def run():
        async with remote.lease(), remote.lease(), remote.lease(), remote.lease():
            assert local._acquire()[0] is None
        async with remote.lease():
            assert not json.loads((local.root / "state.json").read_text())[
                "probe_required"
            ]

    asyncio.run(run())


def test_unavailable_admission_is_infrastructure_before_model_submission(
    tmp_path, monkeypatch
):
    from reweave.planner.providers.base import NativeTransportError

    token = tmp_path / "token"
    token.write_text("private")
    monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_ENDPOINT", "127.0.0.1:0")
    monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_TOKEN_FILE", str(token))

    async def run():
        with pytest.raises(NativeTransportError) as failure:
            async with ProviderAdmission("test").lease():
                pytest.fail("A model request must not be admitted")
        assert failure.value.request_sent is False

    asyncio.run(run())
