"""Shared native admission covers worker races, process death and cooling down."""

import asyncio
import json
import multiprocessing

import pytest

from reweave.utils import provider_admission
from reweave.utils.provider_admission import (
    ProviderAdmission,
    is_native_transport_failure,
    is_rate_limit,
)


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            {
                "codexErrorInfo": "other",
                "message": "stream disconnected before completion: idle timeout waiting for SSE",
            },
            True,
        ),
        (
            {
                "codex_error_info": "other",
                "message": "stream disconnected before completion: stream closed before response.completed",
            },
            True,
        ),
        (
            {
                "codex_error_info": "other",
                "message": "stream disconnected before completion: Transport error: network error: error decoding response body",
            },
            True,
        ),
        (
            {
                "codex_error_info": "server_overloaded",
                "message": "Subscription Sharing usage limit reached",
            },
            False,
        ),
        ({"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": None}}}, True),
        ({"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 401}}}, False),
        ({"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 429}}}, False),
        ({"codexErrorInfo": "other", "message": "request timed out"}, True),
        ({"codex_error_info": "other", "message": "request timed out"}, True),
        ({"codexErrorInfo": "other", "message": "arbitrary task error timeout"}, False),
    ],
)
def test_only_explicit_native_transport_faults_trip_transport_circuit(error, expected):
    assert is_native_transport_failure(error) is expected


def test_transport_fault_does_not_block_unrelated_requests(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    gate = ProviderAdmission("test", root=tmp_path, capacity=2, start_interval_s=0)
    error = {
        "codexErrorInfo": "other",
        "message": "stream disconnected before completion: idle timeout waiting for SSE",
    }
    assert gate.note_transport_failure(error)
    state = json.loads((gate.root / "state.json").read_text())
    assert state["transport_strikes"] == 1
    assert "last_rate_limit_at" not in state
    assert not state.get("probe_required") and not state.get("cooldown_until")

    async def requests():
        async with gate.lease(), gate.lease():
            assert gate._acquire()[0] is None

    asyncio.run(requests())


def test_transport_fault_preserves_existing_rate_limit_and_quota_hold(tmp_path):
    gate = ProviderAdmission("test", root=tmp_path, capacity=48)
    gate.note_rate_limit({"http_status_code": 429})
    gate.note_quota_exhausted("insufficient_quota")
    before = json.loads((gate.root / "state.json").read_text())
    assert gate.note_transport_failure(
        {"codex_error_info": "response_stream_disconnected"}
    )
    after = json.loads((gate.root / "state.json").read_text())
    for key in ("cooldown_until", "probe_required", "generation", "quota_blocked"):
        assert after[key] == before[key]
    assert gate._acquire()[0] is None


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {
                "codex_error_info": {
                    "response_stream_disconnected": {"http_status_code": 429}
                }
            },
            True,
        ),
        (
            {
                "message": "Reconnecting",
                "additional_details": "last status: 429 Too Many Requests",
            },
            True,
        ),
        ({"method": "account/rateLimits/updated", "params": {"usedPercent": 8}}, False),
        ({"message": "authentication failed", "http_status_code": 401}, False),
        ({"message": "id=abcdef429abc"}, False),
    ],
)
def test_only_actual_rate_errors_close_admission(payload, expected):
    assert is_rate_limit(payload) is expected


def test_capacity_and_cancelled_wait_do_not_leak_slots(tmp_path):
    gate = ProviderAdmission("test", root=tmp_path, capacity=1, start_interval_s=0)

    async def run():
        async with gate.lease():
            entered = asyncio.Event()

            async def second():
                async with gate.lease():
                    entered.set()

            task = asyncio.create_task(second())
            await asyncio.sleep(0.02)
            assert not entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with gate.lease():
            pass

    asyncio.run(run())


def test_cooldown_has_four_probes_then_success_reopens_same_48_cap(
    tmp_path, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    gate = ProviderAdmission("test", root=tmp_path, capacity=48, start_interval_s=0)
    gate.note_rate_limit({"httpStatusCode": 429})
    assert gate._acquire()[0] is None
    clock[0] += 61

    leases = []
    try:
        for _ in range(4):
            lease, _ = gate._acquire()
            assert lease is not None and lease[1] is not None
            leases.append(lease)
        assert gate._acquire()[0] is None
        gate._release(leases.pop(), True)
        # Three original long probes remain in flight. Other workers can use
        # all remaining slots after a recovery probe really completes.
        for _ in range(45):
            lease, _ = gate._acquire()
            assert lease is not None and lease[1] is None
            leases.append(lease)
        assert gate._acquire()[0] is None
    finally:
        for lease in leases:
            gate._release(lease, False)
    state = json.loads((gate.root / "state.json").read_text())
    assert not state["probe_required"] and state["strikes"] == 0


def test_failed_recovery_probe_does_not_globalize_individual_failure(
    tmp_path, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    gate = ProviderAdmission("test", root=tmp_path, capacity=48)
    gate.note_rate_limit("HTTP 429 Too Many Requests")
    clock[0] += 61
    lease, _ = gate._acquire()
    gate._release(lease, False)
    state = json.loads((gate.root / "state.json").read_text())
    assert state["probe_required"] and state["cooldown_until"] < clock[0]
    replacement, _ = gate._acquire()
    assert replacement is not None
    gate._release(replacement, True)


def test_new_limit_during_probe_cannot_be_cleared_by_old_success(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    gate = ProviderAdmission("test", root=tmp_path, start_interval_s=0)
    gate.note_rate_limit("HTTP 429 Too Many Requests")
    clock[0] += 61

    async def run():
        async with gate.lease():
            gate.note_rate_limit("HTTP 429 Too Many Requests")

    asyncio.run(run())
    assert gate._acquire()[0] is None
    assert json.loads((gate.root / "state.json").read_text())["probe_required"]


def _held_process(root, pipe):
    gate = ProviderAdmission("test", root=root, capacity=1, start_interval_s=0)

    async def run():
        async with gate.lease():
            pipe.send("entered")
            pipe.recv()

    asyncio.run(run())


def test_process_death_releases_shared_capacity(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    parent1, child1 = ctx.Pipe()
    parent2, child2 = ctx.Pipe()
    first = ctx.Process(target=_held_process, args=(tmp_path, child1))
    second = ctx.Process(target=_held_process, args=(tmp_path, child2))
    try:
        first.start()
        assert parent1.poll(10) and parent1.recv() == "entered"
        second.start()
        assert not parent2.poll(0.5)
        first.terminate()
        first.join(10)
        assert parent2.poll(10) and parent2.recv() == "entered"
        parent2.send("release")
        second.join(10)
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(10)


def test_retry_after_exceeds_local_backoff_and_policy_is_shared(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(provider_admission, "_wall_time", lambda: clock[0])
    gate = ProviderAdmission("test", root=tmp_path, capacity=1, start_interval_s=0)
    gate.note_rate_limit({"http_status_code": 429, "headers": {"retry-after": "300"}})
    assert json.loads((gate.root / "state.json").read_text())["cooldown_until"] == 1300
    assert gate._acquire()[0] is None
    other = ProviderAdmission("test", root=tmp_path, capacity=2, start_interval_s=0)
    with pytest.raises(ValueError, match="shared account policy"):
        other._acquire()
    with pytest.raises(ValueError):
        ProviderAdmission("test", root=tmp_path, capacity=0)


def test_online_and_offline_share_every_slot_without_separate_quota(tmp_path):
    offline = ProviderAdmission(
        "test", root=tmp_path, capacity=4, start_interval_s=0, lane="offline"
    )
    online = ProviderAdmission("test", root=tmp_path, capacity=4, start_interval_s=0)

    async def run():
        async with offline.lease(), offline.lease():
            async with offline.lease(), online.lease():
                assert online._acquire()[0] is None

    asyncio.run(run())


def test_default_is32_for_all_routes_and_explicit_larger_capacity_is_supported(
    tmp_path, monkeypatch
):
    for route in ("codex-auth", "codex-api", "api"):
        gate = ProviderAdmission(route, root=tmp_path)
        assert gate.capacity == 32
    monkeypatch.setenv("REWEAVE_NATIVE_MODEL_CONCURRENCY", "64")
    assert ProviderAdmission("api", root=tmp_path).capacity == 64
