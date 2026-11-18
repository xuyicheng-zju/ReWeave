"""Actual HTTP faults recover receipts without reposting unknown mutations."""

import json
import socket
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from reweave.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from reweave.utils.rpc.request_receipts import RequestReceipts
from reweave.utils.rpc.rpc_client import RpcError, RpcTransportError


@contextmanager
def service(dispatch, **kwargs):
    server = HttpRpcServer(("127.0.0.1", 0), dispatch, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    client.call("healthz", timeout_s=2)
    try:
        yield server, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def lose_submission(client, monkeypatch, method, *, after_send=None, before_send=None):
    original = client._opener.open
    requests = []
    lost = False

    def open_request(request, **kwargs):
        nonlocal lost
        requests.append(
            (urllib.parse.urlsplit(request.full_url).path, json.loads(request.data))
        )
        if (
            request.full_url.endswith("/call")
            and json.loads(request.data)["method"] == method
            and not lost
        ):
            lost = True
            if before_send:
                before_send()
            if after_send:
                url = urllib.parse.urlsplit(request.full_url)
                with socket.create_connection(
                    (url.hostname, url.port), timeout=1
                ) as connection:
                    connection.sendall(
                        b"POST /call HTTP/1.0\r\nContent-Length: "
                        + str(len(request.data)).encode()
                        + b"\r\nContent-Type: application/json\r\n\r\n"
                        + request.data
                    )
                after_send()
            raise OSError("synthetic wifi disconnect")
        return original(request, **kwargs)

    monkeypatch.setattr(client._opener, "open", open_request)
    return requests


def test_pending_receipt_survives_disconnection_without_mutation_repost(monkeypatch):
    started, completed = threading.Event(), threading.Event()
    count = 0

    def dispatch(method, args, kwargs):
        nonlocal count
        if method == "healthz":
            return {"status": "ok"}
        started.set()
        count += 1
        time.sleep(0.32)
        completed.set()
        return {"counter": count, "array": np.array([1, 2], dtype=np.int16)}

    with service(dispatch) as (server, client):
        requests = lose_submission(
            client, monkeypatch, "env.step", after_send=lambda: started.wait(1)
        )
        result = client.call("env.step", timeout_s=2)
        assert completed.is_set() and result["counter"] == count == 1
        np.testing.assert_array_equal(result["array"], [1, 2])
        assert sum(path == "/call" for path, _ in requests) == 1
        assert len({body["request_id"] for _, body in requests}) == 1
        assert all(
            body["expected_instance_id"] == server.instance_id for _, body in requests
        )
        assert client.last_call_metrics.receipt_recovered
        assert client.last_call_metrics.recovery_queries >= 2
        assert client.last_call_metrics.server_dispatch_s >= 0.3


def test_missing_mutation_receipt_is_unknown_not_permission_to_replay(monkeypatch):
    calls = []
    with service(lambda method, *_: calls.append(method) or "ok") as (_, client):
        requests = lose_submission(client, monkeypatch, "env.step")
        with pytest.raises(RpcTransportError, match="unknown or expired") as error:
            client.call("env.step", timeout_s=0.5)
        assert error.value.recovery_state == "unknown"
        assert error.value.request_id and error.value.instance_id
        assert calls == ["healthz"]
        assert sum(path == "/call" for path, _ in requests) == 1
        assert client.last_call_metrics is None


def test_instance_change_never_dispatches_original_call(monkeypatch):
    calls = []
    with service(lambda method, *_, **__: calls.append(method) or "ok") as (
        server,
        client,
    ):
        requests = lose_submission(
            client,
            monkeypatch,
            "env.step",
            before_send=lambda: setattr(server, "instance_id", uuid.uuid4().hex),
        )
        with pytest.raises(RpcTransportError) as error:
            client.call("env.step", timeout_s=0.5)
        assert error.value.recovery_state == "instance_changed"
        assert calls == ["healthz"]
        assert sum(path == "/call" for path, _ in requests) == 1


def test_only_explicit_readonly_method_can_retry_unknown_receipt(monkeypatch):
    calls = []
    with service(
        lambda method, *_: calls.append(method) or "pixels", readonly_methods={"read"}
    ) as (_, client):
        requests = lose_submission(client, monkeypatch, "read")
        assert client.call("read", timeout_s=0.5) == "pixels"
        assert calls == ["healthz", "read"]
        assert sum(path == "/call" for path, _ in requests) == 2
        assert len({body["request_id"] for _, body in requests}) == 1
        assert client.last_call_metrics.readonly_retries == 1
    with service(lambda *_: "ok") as (_, client):
        lose_submission(client, monkeypatch, "env.get_not_declared")
        with pytest.raises(RpcTransportError, match="unknown or expired"):
            client.call("env.get_not_declared", timeout_s=0.5)


def test_deadline_is_not_reset_by_pending_receipt_polling(monkeypatch):
    started, completed = threading.Event(), threading.Event()
    count = 0
    client_thread = threading.get_ident()
    requested_waits = []
    sleep = time.sleep

    def record_wait(seconds):
        if threading.get_ident() == client_thread:
            requested_waits.append(seconds)
        sleep(seconds)

    monkeypatch.setattr(time, "sleep", record_wait)

    def dispatch(method, *_):
        nonlocal count
        if method == "healthz":
            return "ok"
        count += 1
        started.set()
        time.sleep(0.4)
        completed.set()
        return count

    with service(dispatch) as (_, client):
        requests = lose_submission(
            client, monkeypatch, "env.step", after_send=lambda: started.wait(1)
        )
        with pytest.raises(RpcTransportError, match="deadline exhausted") as error:
            client.call("env.step", timeout_s=0.08)
        # The OS may wake this thread late (notably on macOS). Check that
        # polling never requests a new timeout allowance, rather than imposing
        # a real-time scheduling guarantee on the client or its test runner.
        assert sum(requested_waits) <= 0.08
        assert error.value.recovery_state == "pending"
        assert count == 1 and sum(path == "/call" for path, _ in requests) == 1
        assert completed.wait(1)


def test_recovered_remote_application_error_is_not_transport_failure(monkeypatch):
    called = threading.Event()

    def dispatch(method, *_):
        if method == "healthz":
            return "ok"
        called.set()
        raise ValueError("invalid author argument")

    with service(dispatch) as (_, client):
        lose_submission(client, monkeypatch, "read", after_send=lambda: called.wait(1))
        with pytest.raises(RpcError, match="invalid author argument") as error:
            client.call("read", timeout_s=0.5)
        assert not isinstance(error.value, RpcTransportError)
        assert error.value.server_traceback
        assert client.last_call_metrics.receipt_recovered


def test_cached_response_is_immutable_and_session_bound():
    receipts = RequestReceipts(max_entries=2, max_bytes=1000, ttl_s=1)
    assert receipts.begin("a", "id", b"input") == (True, None)
    receipts.complete("a", "id", b'{"ok":true,"result":[1]}', 0.2, 0.01)
    assert receipts.lookup("a", "id")["response_json"] == '{"ok":true,"result":[1]}'
    assert receipts.lookup("b", "id") == {"state": "unknown"}
    assert receipts.begin("a", "id", b"input")[0] is False
    with pytest.raises(ValueError, match="different input"):
        receipts.begin("a", "id", b"changed")


def test_receipt_limits_evict_completed_never_pending_and_expiry_is_unknown():
    receipts = RequestReceipts(max_entries=2, max_bytes=1000, ttl_s=0.01)
    receipts.begin(None, "pending", b"input")
    receipts.begin(None, "done", b"input")
    receipts.complete(None, "done", b"{}", 0, 0)
    receipts.begin(None, "next", b"input")
    assert receipts.lookup(None, "done") == {"state": "unknown"}
    assert receipts.lookup(None, "pending") == {"state": "pending"}
    with pytest.raises(RuntimeError, match="pending"):
        receipts.begin(None, "extra", b"input")
    receipts.complete(None, "next", b"{}", 0, 0)
    time.sleep(0.02)
    assert receipts.lookup(None, "next") == {"state": "unknown"}
    assert receipts.lookup(None, "pending") == {"state": "pending"}


def test_slow_mutation_recovers_after_io_timeout_within_original_deadline():
    count = 0

    def dispatch(method, *_):
        nonlocal count
        if method == "healthz":
            return "ok"
        count += 1
        time.sleep(0.25)
        return count

    with service(dispatch) as (_, client):
        client._recovery_io_timeout_s = 0.05
        before = time.monotonic()
        assert client.call("env.step", timeout_s=1) == 1
        assert 0.25 <= time.monotonic() - before < 1
        assert count == 1
        assert client.last_call_metrics.receipt_recovered
        assert client.last_call_metrics.readonly_retries == 0


def test_completed_duplicate_http_payload_never_dispatches_twice():
    calls = []
    with service(lambda method, *_, **__: calls.append(method) or "ok") as (
        server,
        client,
    ):
        payload = {
            "method": "env.step",
            "args": [],
            "kwargs": {},
            "session_id": "episode",
            "request_id": uuid.uuid4().hex,
            "expected_instance_id": server.instance_id,
        }
        first = client._exchange("env.step", payload, timeout_s=1)
        second = client._exchange("env.step", payload, timeout_s=1)
        assert first == second == {"ok": True, "result": "ok"}
        assert calls == ["healthz", "env.step"]
        payload["session_id"] = "other-episode"
        assert client._exchange("env.step", payload, timeout_s=1) == first
        assert calls == ["healthz", "env.step", "env.step"]


def test_failed_fresh_route_probe_never_sends_mutation_and_can_reconnect(monkeypatch):
    """Accepting a loopback socket is not proof the far-end channel opened."""
    import http.client

    calls = []
    with service(lambda method, *_: calls.append(method) or "ok") as (_, client):
        client._opener._idle.close()
        client._opener._idle = None
        original = http.client.HTTPConnection
        paths = []
        failures = 0

        class InterruptedRoute(original):
            def request(self, method, url, *args, **kwargs):
                nonlocal failures
                paths.append(url)
                if url == "/call-receipt" and failures == 0:
                    failures += 1
                    # A real local TCP accept followed by failed remote setup:
                    # even though a socket existed, no mutation may enter it.
                    self.connect()
                    self.sock.shutdown(socket.SHUT_RDWR)
                    raise ConnectionResetError("remote SSH channel failed to open")
                return super().request(method, url, *args, **kwargs)

        monkeypatch.setattr(http.client, "HTTPConnection", InterruptedRoute)
        assert client.call("env.step", timeout_s=2) == "ok"
        assert calls == ["healthz", "env.step"]
        assert paths == ["/call-receipt", "/call-receipt", "/call"]
        assert client.last_call_metrics.pre_submission_retries == 1
        assert not client.last_call_metrics.receipt_recovered


def test_probe_deadline_is_known_unsubmitted_not_unknown_receipt(monkeypatch):
    import http.client

    calls = []
    with service(lambda method, *_: calls.append(method) or "ok") as (_, client):
        client._opener._idle.close()
        client._opener._idle = None
        original = http.client.HTTPConnection
        paths = []

        class UnavailableRoute(original):
            def request(self, method, url, *args, **kwargs):
                paths.append(url)
                raise ConnectionResetError("unavailable before remote submission")

        monkeypatch.setattr(http.client, "HTTPConnection", UnavailableRoute)
        started = time.monotonic()
        with pytest.raises(RpcTransportError) as error:
            client.call("env.step", timeout_s=0.12)
        assert time.monotonic() - started < 0.4
        assert error.value.recovery_state == "not_submitted"
        assert error.value.request_id and error.value.instance_id
        assert calls == ["healthz"]
        assert paths == ["/call-receipt"]


def test_fresh_route_probe_rejects_replacement_before_physical_dispatch():
    calls = []
    with service(lambda method, *_: calls.append(method) or "ok") as (server, client):
        client._opener._idle.close()
        client._opener._idle = None
        server.instance_id = uuid.uuid4().hex
        with pytest.raises(RpcTransportError) as error:
            client.call("env.step", timeout_s=0.5)
        assert error.value.recovery_state == "instance_changed"
        assert calls == ["healthz"]
