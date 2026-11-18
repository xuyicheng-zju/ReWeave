from __future__ import annotations

import gzip
import json
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from reweave.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer, _from_json
from reweave.utils.rpc.rpc_client import RpcError


@contextmanager
def _server(dispatch) -> Iterator[tuple[str, list[str]]]:
    calls: list[str] = []

    def recording_dispatch(method, args, kwargs):
        calls.append(method)
        return dispatch(method, args, kwargs)

    server = HttpRpcServer(("127.0.0.1", 0), recording_dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()


def _request(url: str, accept_encoding: str | None):
    headers = {"Content-Type": "application/json"}
    if accept_encoding is not None:
        headers["Accept-Encoding"] = accept_encoding
    request = urllib.request.Request(
        f"{url}/call",
        data=b'{"method":"observe"}',
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=2) as response:
        return response.headers, response.read()


@pytest.mark.parametrize(
    ("accept_encoding", "compressed"),
    [
        (None, False),
        ("identity", False),
        ("gzip", True),
        ("br, gzip;q=0.5", True),
        ("gzip;q=0", False),
        ("*;q=1, gzip;q=0", False),
        ("*;q=0.5", True),
        ("gzip;q=invalid", False),
    ],
)
def test_response_compression_is_negotiated(accept_encoding, compressed) -> None:
    value = np.zeros((200, 300), dtype=np.float32)
    with _server(lambda *_: value) as (url, calls):
        headers, wire = _request(url, accept_encoding)
    assert calls == ["observe"]
    assert int(headers["Content-Length"]) == len(wire)
    assert headers["Vary"] == "Accept-Encoding"
    assert (headers.get("Content-Encoding") == "gzip") is compressed
    raw = gzip.decompress(wire) if compressed else wire
    np.testing.assert_array_equal(_from_json(json.loads(raw))["result"], value)
    if compressed:
        assert len(wire) < len(raw) / 20


def test_small_response_stays_identity() -> None:
    with _server(lambda *_: {"status": "ok"}) as (url, _):
        client = HttpRpcClient(url)
        assert client.call("healthz") == {"status": "ok"}
    metrics = client.last_call_metrics
    assert metrics is not None
    assert metrics.content_encoding == "identity"
    assert metrics.response_wire_bytes == metrics.response_body_bytes
    assert metrics.response_decompress_s == 0
    assert metrics.request_compress_s == 0
    assert metrics.request_content_encoding == "identity"
    assert metrics.request_body_bytes == metrics.request_wire_bytes


def test_compressed_rgbd_preserves_exact_arrays_and_exposes_stage_metrics() -> None:
    rng = np.random.default_rng(22)
    values = {
        "color": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        "depth": rng.uniform(0.1, 3, (240, 320)).astype(np.float32),
        "special_depth": np.array([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32),
    }
    with _server(lambda *_: values) as (url, calls):
        client = HttpRpcClient(url)
        result = client.call("get_obs")
    assert calls == ["get_obs"]
    for name, original in values.items():
        assert result[name].dtype == original.dtype
        assert result[name].shape == original.shape
        assert result[name].tobytes() == original.tobytes()
        assert result[name].flags.writeable
    metrics = client.last_call_metrics
    assert metrics is not None
    assert metrics.content_encoding == "gzip"
    assert metrics.response_wire_bytes < metrics.response_body_bytes
    assert metrics.request_body_bytes > 0
    for duration in (
        metrics.request_encode_s,
        metrics.response_wait_read_s,
        metrics.response_decompress_s,
        metrics.response_decode_s,
        metrics.server_dispatch_s,
        metrics.server_encode_s,
        metrics.server_compress_s,
    ):
        assert duration is not None and duration >= 0
    assert metrics.total_s >= metrics.response_wait_read_s


@contextmanager
def _raw_server(body: bytes, encoding: str, status: int = 200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(self.headers.get("Accept-Encoding"))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Encoding", encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize(
    "body",
    [
        b"not gzip",
        gzip.compress(b"ok")[:-4],
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + b"\xff" * 12,
    ],
)
def test_invalid_gzip_does_not_retry_the_remote_action(body) -> None:
    with _raw_server(body, "gzip") as (url, calls):
        client = HttpRpcClient(url)
        with pytest.raises(RpcError, match="invalid gzip response"):
            client.call("step")
    assert calls == ["gzip"]
    assert client.last_call_metrics is None


def test_request_gzip_is_advertised_then_used_without_extra_calls():
    value = np.zeros((240, 320, 3), np.uint8)

    def echo(method, args, kwargs):
        assert method == "infer"
        assert args[0].dtype == value.dtype
        assert args[0].tobytes() == value.tobytes()
        return {"accepted": True}

    with _server(echo) as (url, calls):
        client = HttpRpcClient(url)
        assert client.call("infer", args=(value,)) == {"accepted": True}
        first = client.last_call_metrics
        assert first.request_content_encoding == "identity"
        assert first.request_body_bytes == first.request_wire_bytes
        assert first.request_compress_s == 0
        assert client.call("infer", args=(value,)) == {"accepted": True}
        second = client.last_call_metrics
    assert calls == ["infer", "infer"]
    assert second.request_content_encoding == "gzip"
    # The second request binds the server instance learned from the first
    # response: JSON null (4 bytes) becomes a quoted UUID hex (34 bytes).
    assert second.request_body_bytes == first.request_body_bytes + 30
    assert second.request_wire_bytes < second.request_body_bytes / 20
    assert second.request_compress_s > 0


def test_endpoint_without_request_coding_advertisement_keeps_identity():
    body = b'{"ok":true,"result":"received"}'
    with _raw_server(body, "identity") as (url, calls):
        client = HttpRpcClient(url)
        for _ in range(2):
            assert client.call("infer", args=(np.zeros((200, 300)),)) == "received"
            metrics = client.last_call_metrics
            assert metrics.request_content_encoding == "identity"
            assert metrics.request_wire_bytes == metrics.request_body_bytes
            assert metrics.request_compress_s == 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("body", "encoding", "message"),
    [
        (b"broken", "gzip", "invalid gzip request"),
        (gzip.compress(b'{"method":"step"}')[:-4], "gzip", "invalid gzip request"),
        (b'{"method":"step"}', "br", "unsupported request content encoding"),
    ],
)
def test_invalid_request_coding_never_dispatches_action(body, encoding, message):
    with _server(lambda *_: pytest.fail("must not dispatch")) as (url, calls):
        request = urllib.request.Request(
            f"{url}/call",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": encoding,
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=2) as response:
            result = json.loads(response.read())
            assert response.headers["Accept-Encoding"] == "gzip"
        assert result["ok"] is False
        assert message in str(result["error"])
    assert calls == []


def test_unknown_content_encoding_is_rejected_without_retry() -> None:
    with _raw_server(b"{}", "br") as (url, calls):
        with pytest.raises(RpcError, match="unsupported response content encoding"):
            HttpRpcClient(url).call("step")
    assert calls == ["gzip"]


def test_compressed_http_error_retains_remote_error_and_metrics() -> None:
    body = gzip.compress(
        json.dumps({"ok": False, "error": "actuator unavailable"}).encode()
    )
    with _raw_server(body, "gzip", status=503) as (url, calls):
        client = HttpRpcClient(url)
        with pytest.raises(RpcError, match="actuator unavailable"):
            client.call("step")
    assert calls == ["gzip"]
    assert client.last_call_metrics is not None
    assert client.last_call_metrics.content_encoding == "gzip"
    assert client.last_call_metrics.server_dispatch_s is None


def test_failed_io_clears_previous_measurement_without_action_retry(
    monkeypatch,
) -> None:
    with _server(lambda *_: "received") as (url, calls):
        client = HttpRpcClient(url)
        assert client.call("get_obs") == "received"
        assert client.last_call_metrics is not None
        submissions = []

        def failed_open(*args, **kwargs):
            submissions.append(args[0])
            raise OSError("connection lost")

        monkeypatch.setattr(client._opener, "open", failed_open)
        with pytest.raises(RpcError, match="connection lost"):
            client.call("step", timeout_s=0.05)
    assert calls == ["get_obs"]
    assert sum(request.full_url.endswith("/call") for request in submissions) == 1
    assert all(
        request.full_url.endswith("/call-receipt") for request in submissions[1:]
    )
    assert client.last_call_metrics is None
