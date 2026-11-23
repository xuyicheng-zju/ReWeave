# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""HTTP-transport RPC for the env + model RPC boundary.

Uses HTTP POST with JSON payloads instead of pickle-framed TCP.
Numpy arrays cross the wire tagged as
``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}`` — the raw
bytes stay compact (vs ``tolist()``, which stringifies every element)
and the decode is explicit. Large responses negotiate lossless gzip over this
same JSON contract. Client metrics separate server work, wire receipt and local
decode. Failed mutations recover their original receipt without replay;
only explicitly advertised readonly calls may be reissued.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import http.client
import json
import math
import select
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

from reweave.utils.logging import get_logger
from reweave.utils.rpc.request_receipts import RequestReceipts
from reweave.utils.rpc.rpc_client import RpcClient, RpcError, RpcTransportError
from reweave.utils.rpc.rpc_facade import make_error_response

DEFAULT_TIMEOUT_S = 30.0
_GZIP_MIN_BYTES = 64 * 1024
_DIRECT_HOSTS = frozenset({"127.0.0.1", "localhost"})

logger = get_logger("http_rpc")


@dataclass(frozen=True)
class RpcCallMetrics:
    """Client measurements for the latest received RPC response, in seconds/bytes.

    HTTP wait/read includes server execution, encoding and network transfer; it
    must not be interpreted as network latency alone. Server measurements are
    absent for servers that do not publish timing headers. No metric denotes
    whether a physical action completed; the RPC result remains authoritative.
    During recovery, byte/codec/wait fields describe the final HTTP exchange;
    total_s includes the entire call and recovery. Original server dispatch
    and encode durations are restored from the immutable completed receipt.
    """

    # Local JSON/base64 request encoding duration.
    request_encode_s: float
    # Local gzip attempt duration; zero without an advertised request coding.
    request_compress_s: float
    # From HTTP submission through receipt of the complete wire response.
    response_wait_read_s: float
    # Local gzip decompression duration; zero for identity responses.
    response_decompress_s: float
    # Local JSON parsing and ndarray reconstruction duration.
    response_decode_s: float
    # Full client call duration through response decoding, before result checks.
    total_s: float
    # Original JSON/base64 request size before compression, excluding headers.
    request_body_bytes: int
    # Submitted request size after optional content compression.
    request_wire_bytes: int
    # Submitted HTTP request content coding: gzip or identity.
    request_content_encoding: str
    # Received response body size before content decompression.
    response_wire_bytes: int
    # Response JSON size after content decompression.
    response_body_bytes: int
    # HTTP response content coding: gzip or identity.
    content_encoding: str
    # Server request decoding and dispatch, including simulator/model work.
    server_dispatch_s: float | None
    # Server JSON/base64 response encoding duration.
    server_encode_s: float | None
    # Server gzip attempt duration, even if identity was smaller.
    server_compress_s: float | None
    request_id: str = ""
    service_instance_id: str = ""
    recovery_queries: int = 0
    readonly_retries: int = 0
    receipt_recovered: bool = False
    pre_submission_retries: int = 0


def _accepts_gzip(header: str) -> bool:
    """Honor explicit gzip exclusions before a wildcard coding preference."""
    qualities: dict[str, float] = {}
    for item in header.split(","):
        coding, *parameters = item.strip().lower().split(";")
        quality = 1.0
        for parameter in parameters:
            name, _, value = parameter.strip().partition("=")
            if name == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality
    return qualities.get("gzip", qualities.get("*", 0.0)) > 0.0


def _is_direct_url(url: str) -> bool:
    """Return whether *url* uses a hostname that must bypass HTTP proxies."""
    hostname = urllib.parse.urlsplit(url).hostname
    return hostname is not None and hostname.lower() in _DIRECT_HOSTS


def _from_json(obj: Any) -> Any:
    """Rehydrate ``{"__ndarray__": <b64>, "dtype": ..., "shape": [...]}``
    back into ndarrays, and reserved singleton ``__bytes__`` tags into bytes.
    Untagged values retain their JSON meaning.
    """
    if isinstance(obj, dict):
        if "__bytes__" in obj:
            if set(obj) != {"__bytes__"} or type(obj["__bytes__"]) is not str:
                raise ValueError("invalid or ambiguous RPC bytes tag")
            try:
                return base64.b64decode(obj["__bytes__"], validate=True)
            except ValueError as exc:
                raise ValueError("invalid base64 in RPC bytes tag") from exc
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            arr = np.frombuffer(raw, dtype=obj.get("dtype"))
            # frombuffer returns a read-only view of the base64 bytes; copy
            # so callers can mutate the returned array like they would with
            # a pickle round-tripped one.
            return arr.reshape(obj.get("shape", (-1,))).copy()
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class _DirectHttpResponse:
    """Return a fully consumed connection to its owner's idle slot."""

    def __init__(self, owner, connection, response):
        self.owner, self.connection, self.response = owner, connection, response
        self.headers = response.headers
        self.consumed = False

    def __enter__(self):
        return self

    def read(self):
        data = self.response.read()
        self.consumed = True
        return data

    def __exit__(self, exc_type, exc, traceback):
        reusable = self.consumed and exc_type is None and not self.response.will_close
        self.response.close()
        self.owner.release(self.connection, reusable=reusable)


class _DirectHttpOpener:
    """Reuse one idle direct connection; never retry a submitted request.

    Concurrent calls acquire separate sockets, so receipt queries and cancellation
    never wait behind a long-running operation. Proxy routes retain urllib's
    existing handling. A failed socket is discarded before receipt recovery.
    """

    def __init__(self, base_url):
        self.route = urllib.parse.urlsplit(base_url)
        self._idle = None
        self._idle_since = 0.0
        self._lock = threading.Lock()
        self._closed = False

    def open(self, request, *, timeout):
        deadline = time.monotonic() + timeout
        route = urllib.parse.urlsplit(request.full_url)
        if (route.scheme, route.netloc) != (self.route.scheme, self.route.netloc):
            raise ValueError("direct RPC request changed its endpoint")
        with self._lock:
            connection, self._idle = self._idle, None
            idle_since = self._idle_since
        if connection is not None and (
            time.monotonic() - idle_since >= 5.0
            or connection.sock is None
            or select.select([connection.sock], [], [], 0)[0]
        ):
            # The server closes idle keep-alive connections after 30 seconds.
            # A tunnel can delay EOF beyond our readiness check, racing a new
            # mutation after model reasoning. Retire long-idle sockets before
            # submission; never retry an already submitted mutation.
            # An idle socket has no response pending. Readability means the
            # peer closed it or sent unsolicited bytes; discard before sending.
            connection.close()
            connection = None
        fresh = connection is None
        if fresh:
            factory = (
                http.client.HTTPSConnection
                if route.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = factory(route.hostname, route.port, timeout=timeout)
        try:
            expected = request.get_header("X-rpc-expected-instance-id")
            if fresh and expected is not None:
                # A loopback proxy accepting TCP proves nothing about its SSH
                # route. Establish the SAME socket end to end using a receipt
                # lookup, which never dispatches task work, before mutation bytes.
                self._probe(connection, expected, min(timeout, 5.0))
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise RpcTransportError(
                        "transport",
                        "Route probe exhausted the submission deadline",
                        recovery_state="not_submitted",
                    )
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            target = urllib.parse.urlunsplit(
                ("", "", route.path or "/", route.query, "")
            )
            connection.request(
                request.get_method(),
                target,
                body=request.data,
                headers=dict(request.header_items()),
            )
            response = connection.getresponse()
            return _DirectHttpResponse(self, connection, response)
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _probe(connection, expected, timeout):
        request_id = uuid.uuid4().hex
        connection.timeout = timeout
        try:
            connection.request(
                "POST",
                "/call-receipt",
                body=json.dumps(
                    {
                        "expected_instance_id": expected,
                        "request_id": request_id,
                        "session_id": None,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            try:
                if response.headers.get("X-RPC-Instance-ID") != expected:
                    raise RpcTransportError(
                        "transport",
                        "RPC route probe found a different service instance",
                        recovery_state="instance_changed",
                    )
                body = json.loads(response.read())
                result = body.get("result", {})
                if (
                    response.status != 200
                    or not body.get("ok")
                    or result.get("instance_id") != expected
                    or result.get("request_id") != request_id
                    or result.get("state") != "unknown"
                    or response.will_close
                ):
                    raise ValueError("invalid RPC route probe response")
            finally:
                response.close()
        except (
            OSError,
            http.client.HTTPException,
            ValueError,
            AttributeError,
        ) as error:
            raise RpcTransportError(
                "transport",
                f"Route unavailable before RPC submission: {error}",
                recovery_state="not_submitted",
            ) from error

    def release(self, connection, *, reusable):
        with self._lock:
            if reusable and not self._closed and self._idle is None:
                self._idle = connection
                self._idle_since = time.monotonic()
                return
        connection.close()

    def close(self):
        with self._lock:
            self._closed = True
            connection, self._idle = self._idle, None
        if connection is not None:
            connection.close()


class HttpRpcClient(RpcClient):
    """RPC client that talks to an RPC server via HTTP POST.

    Parameters
    ----------
    base_url : str
        Server address, e.g. ``"http://127.0.0.1:8080"``.

        ``127.0.0.1`` and ``localhost`` bypass HTTP proxies. Other hostnames
        use the process proxy configuration.
    enable_sessions : bool
        If True, the client is session-aware: it auto-generates a session id
        and carries it on every call; :func:`wait_for_ready` registers it
        with the server on connect. If False, the client sends no session_id
        and is fully session-unaware.
    """

    def __init__(
        self,
        base_url: str,
        *,
        enable_sessions: bool = False,
        recovery_io_timeout_s: float = 15.0,
    ) -> None:
        if not math.isfinite(recovery_io_timeout_s) or recovery_io_timeout_s <= 0:
            raise ValueError("RPC recovery_io_timeout_s must be finite and positive")
        super().__init__(enable_sessions=enable_sessions)
        self._recovery_io_timeout_s = recovery_io_timeout_s
        self._base_url = base_url.rstrip("/")
        self._server_instance_id = None
        self._receipt_capable = False
        self._readonly_methods = frozenset()

        # Request compression is opt-in by the endpoint's previous HTTP
        # response, never by probing or retrying a submitted method.
        self._request_gzip_accepted = False
        # Cleared before submission; populated once a body has decoded, even
        # when the remote method returned an error. Failed I/O leaves None.
        # Concurrent calls each measure locally; this convenience snapshot is
        # overwritten by the last completed call and is not per-request history.
        self.last_call_metrics: RpcCallMetrics | None = None
        self._opener = (
            _DirectHttpOpener(self._base_url)
            if _is_direct_url(self._base_url)
            else None
        )

    @property
    def endpoint(self) -> str:
        return self._base_url

    def close(self):
        try:
            super().close()
        finally:
            if self._opener is not None:
                self._opener.close()

    def _remember_route(self, headers):
        instance = headers.get("X-RPC-Instance-ID")
        if instance is not None:
            if (
                self._server_instance_id is not None
                and instance != self._server_instance_id
            ):
                raise RpcTransportError(
                    "transport",
                    "RPC service instance changed",
                    instance_id=self._server_instance_id,
                    recovery_state="instance_changed",
                )
            self._server_instance_id = instance
        if headers.get("X-RPC-Receipts") == "1" and instance:
            try:
                methods = json.loads(headers.get("X-RPC-Readonly-Methods", "[]"))
            except ValueError as error:
                raise RpcTransportError(
                    "transport",
                    "Invalid RPC readonly declaration",
                    recovery_state="invalid_response",
                ) from error
            if not isinstance(methods, list) or any(
                not isinstance(name, str) for name in methods
            ):
                raise RpcTransportError(
                    "transport",
                    "Invalid RPC readonly declaration",
                    recovery_state="invalid_response",
                )
            self._readonly_methods = frozenset(methods)
            self._receipt_capable = True

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        """Submit once; after lost I/O only recover the same instance's receipt.

        Readonly methods explicitly advertised by the server may be reissued
        after an unknown receipt. Mutations are never reposted. Recovery uses
        the original absolute timeout, including all waits and queries.
        """
        self.last_call_metrics = None
        started = time.perf_counter()
        timeout = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout
        request_id = uuid.uuid4().hex
        payload = {
            "method": method,
            "args": list(args),
            "kwargs": kwargs or {},
            "session_id": self._session_id,
            "request_id": request_id,
            "expected_instance_id": self._server_instance_id,
        }
        queries = retries = 0
        recovered = False
        original_timing = None
        pre_submission_retries = 0
        try:
            try:
                while True:
                    try:
                        response = self._exchange(
                            method,
                            payload,
                            timeout_s=min(
                                deadline - time.monotonic(), self._recovery_io_timeout_s
                            )
                            if self._receipt_capable
                            else deadline - time.monotonic(),
                        )
                        break
                    except RpcTransportError as error:
                        if error.recovery_state != "not_submitted":
                            raise
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise
                        pre_submission_retries += 1
                        time.sleep(min(0.25, remaining))
                        if time.monotonic() >= deadline:
                            raise
                if response.get("rpc_state") == "pending":
                    raise RpcTransportError(method, "Original RPC request is pending")
            except RpcTransportError as error:
                if (
                    error.recovery_state in {"instance_changed", "not_submitted"}
                    or not self._receipt_capable
                ):
                    raise
                response, queries, retries, original_timing = self._recover(
                    method, payload, deadline, error
                )
                recovered = True
        except RpcTransportError as error:
            self.last_call_metrics = None
            error.request_id = request_id
            error.instance_id = self._server_instance_id
            raise
        if self.last_call_metrics is not None:
            updates = {
                "total_s": time.perf_counter() - started,
                "request_id": request_id,
                "service_instance_id": self._server_instance_id or "",
                "recovery_queries": queries,
                "readonly_retries": retries,
                "receipt_recovered": recovered,
                "pre_submission_retries": pre_submission_retries,
            }
            if original_timing is not None:
                updates.update(
                    server_dispatch_s=original_timing[0],
                    server_encode_s=original_timing[1],
                )
            self.last_call_metrics = replace(self.last_call_metrics, **updates)
        return self._check_response(response, method)

    def _recover(self, method, payload, deadline, original_error):
        queries = retries = 0
        last_state = "unavailable"
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                queries += 1
                envelope = self._exchange(
                    method,
                    {
                        "request_id": payload["request_id"],
                        "session_id": self._session_id,
                        "expected_instance_id": self._server_instance_id,
                    },
                    path="call-receipt",
                    timeout_s=min(5.0, remaining),
                )
                receipt = self._check_response(envelope, "call-receipt")
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("request_id") != payload["request_id"]
                    or receipt.get("instance_id") != self._server_instance_id
                ):
                    raise RpcTransportError(
                        method,
                        "Invalid RPC receipt identity",
                        recovery_state="invalid_response",
                    )
                last_state = receipt.get("state")
                if last_state == "completed":
                    try:
                        response = _from_json(json.loads(receipt["response_json"]))
                    except (KeyError, ValueError, TypeError) as error:
                        raise RpcTransportError(
                            method,
                            "Invalid completed RPC receipt",
                            recovery_state="invalid_response",
                        ) from error
                    return (
                        response,
                        queries,
                        retries,
                        (receipt["dispatch_s"], receipt["encode_s"]),
                    )
                if last_state == "unknown":
                    if method not in self._readonly_methods:
                        raise RpcTransportError(
                            method,
                            "Original RPC receipt is unknown or expired; mutation was not replayed; "
                            f"request_id={payload['request_id']}; "
                            f"instance_id={self._server_instance_id}; "
                            f"original_error={original_error}",
                            recovery_state="unknown",
                        ) from original_error
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    retries += 1
                    response = self._exchange(
                        method,
                        payload,
                        timeout_s=min(remaining, self._recovery_io_timeout_s),
                    )
                    if response.get("rpc_state") != "pending":
                        return response, queries, retries, None
                elif last_state != "pending":
                    raise RpcTransportError(
                        method,
                        "Invalid RPC receipt state",
                        recovery_state="invalid_response",
                    )
            except RpcTransportError as error:
                if error.recovery_state in {
                    "instance_changed",
                    "unknown",
                    "invalid_response",
                }:
                    raise
            except RpcError as error:
                raise RpcTransportError(
                    method,
                    f"RPC receipt query failed: {error}",
                    recovery_state="invalid_response",
                ) from error
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.25, remaining))
        raise RpcTransportError(
            method,
            f"Original RPC deadline exhausted after {original_error}; receipt={last_state}; mutation was not replayed",
            recovery_state=last_state,
        ) from original_error

    def _exchange(self, method, payload, *, timeout_s, path="call"):
        """One HTTP exchange. No operation retry or response semantics here."""
        started = time.perf_counter()
        body = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
        encoded = time.perf_counter()
        wire_body = body
        request_encoding = "identity"
        request_compress_s = 0.0
        if self._request_gzip_accepted and len(body) >= _GZIP_MIN_BYTES:
            compression_started = time.perf_counter()
            compressed_body = gzip.compress(body, compresslevel=1, mtime=0)
            request_compress_s = time.perf_counter() - compression_started
            if len(compressed_body) < len(body):
                wire_body = compressed_body
                request_encoding = "gzip"
        url = f"{self._base_url}/{path}"
        request_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

        request_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        }
        if (
            path == "call"
            and self._receipt_capable
            and method not in self._readonly_methods
        ):
            request_headers["X-RPC-Expected-Instance-ID"] = self._server_instance_id
        if request_encoding == "gzip":
            request_headers["Content-Encoding"] = "gzip"
        req = urllib.request.Request(
            url,
            data=wire_body,
            headers=request_headers,
            method="POST",
        )
        submitted = time.perf_counter()
        try:
            open_request = self._opener.open if self._opener else urllib.request.urlopen
            try:
                resp = open_request(req, timeout=request_timeout)
            except urllib.error.HTTPError as error:
                resp = error
            with resp:
                headers = resp.headers
                self._remember_route(headers)
                raw = resp.read()
        except RpcTransportError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise RpcTransportError(method, f"HTTP request failed: {exc}") from exc

        received = time.perf_counter()
        wire_bytes = len(raw)
        content_encoding = headers.get("Content-Encoding", "identity").strip().lower()
        if content_encoding == "gzip":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError, zlib.error) as exc:
                raise RpcTransportError(
                    method,
                    f"invalid gzip response: {exc}",
                    recovery_state="invalid_response",
                ) from exc
        elif content_encoding != "identity":
            raise RpcTransportError(
                method,
                f"unsupported response content encoding: {content_encoding}",
                recovery_state="invalid_response",
            )
        decompressed = time.perf_counter()
        try:
            response = _from_json(json.loads(raw))
        except (ValueError, TypeError) as exc:
            raise RpcTransportError(
                method,
                f"invalid JSON response or tagged value: {exc}",
                recovery_state="invalid_response",
            ) from exc

        decoded = time.perf_counter()
        self._request_gzip_accepted = _accepts_gzip(headers.get("Accept-Encoding", ""))
        try:
            server_timings = {
                name: float(value)
                if (value := headers.get(header)) is not None
                else None
                for name, header in (
                    ("server_dispatch_s", "X-RPC-Dispatch-Seconds"),
                    ("server_encode_s", "X-RPC-Encode-Seconds"),
                    ("server_compress_s", "X-RPC-Compress-Seconds"),
                )
            }
        except ValueError as error:
            raise RpcTransportError(
                method, "Invalid RPC timing headers", recovery_state="invalid_response"
            ) from error
        self.last_call_metrics = RpcCallMetrics(
            request_encode_s=encoded - started,
            request_compress_s=request_compress_s,
            response_wait_read_s=received - submitted,
            response_decompress_s=decompressed - received
            if content_encoding == "gzip"
            else 0.0,
            response_decode_s=decoded - decompressed,
            total_s=decoded - started,
            request_body_bytes=len(body),
            request_wire_bytes=len(wire_body),
            request_content_encoding=request_encoding,
            response_wire_bytes=wire_bytes,
            response_body_bytes=len(raw),
            content_encoding=content_encoding,
            **server_timings,
        )
        if not isinstance(response, dict):
            raise RpcTransportError(
                method,
                "Invalid RPC response envelope",
                recovery_state="invalid_response",
            )
        return response




class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder for tagged bytes/arrays and normalized numpy scalars.

    ``__bytes__`` is a reserved singleton dictionary tag, not an application key.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class _HttpRpcHandler(BaseHTTPRequestHandler):
    """Handles POST /call with JSON-RPC body, dispatches to server.dispatch()."""

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 30

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        if self.path not in {"/call", "/call-receipt"}:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.close_connection = True
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        dispatch_started = time.perf_counter()
        try:
            request_encoding = (
                self.headers.get("Content-Encoding", "identity").strip().lower()
            )
            if request_encoding == "gzip":
                try:
                    body = gzip.decompress(body)
                except (OSError, EOFError, zlib.error) as exc:
                    raise ValueError(f"invalid gzip request: {exc}") from exc
            elif request_encoding != "identity":
                raise ValueError(
                    f"unsupported request content encoding: {request_encoding}"
                )
            request = json.loads(body)
            expected = request.get("expected_instance_id")
            if expected is not None and expected != self.server.instance_id:
                raise ValueError(
                    "RPC service instance does not match the bound instance"
                )
            request_id = request.get("request_id")
            if request_id is not None and (
                not isinstance(request_id, str)
                or len(request_id) != 32
                or any(char not in "0123456789abcdef" for char in request_id)
            ):
                raise ValueError("RPC request_id must be a UUID hex string")
            session_id = request.get("session_id")
            if self.path == "/call-receipt":
                if request_id is None or expected is None:
                    raise ValueError(
                        "RPC receipt query requires request and instance identities"
                    )
                receipt = self.server.receipts.lookup(session_id, request_id)
                response = {
                    "ok": True,
                    "result": {
                        **receipt,
                        "request_id": request_id,
                        "instance_id": self.server.instance_id,
                    },
                }
            else:
                method = request["method"]
                args = tuple(_from_json(v) for v in request.get("args", []))
                kwargs = {
                    k: _from_json(v) for k, v in request.get("kwargs", {}).items()
                }
                first, cached = (
                    (True, None)
                    if request_id is None
                    else self.server.receipts.begin(
                        session_id, request_id, hashlib.sha256(body).digest()
                    )
                )
                if not first:
                    if cached is None:
                        response = {
                            "ok": False,
                            "rpc_state": "pending",
                            "error": "RPC request is pending",
                        }
                    else:
                        response = json.loads(cached)
                else:
                    try:
                        if session_id is None:
                            result = self.server.dispatch(method, args, kwargs)
                        else:
                            result = self.server.dispatch(
                                method, args, kwargs, session_id=session_id
                            )
                        response = {"ok": True, "result": result}
                    except Exception as error:
                        response = make_error_response(error)
            dispatch_finished = time.perf_counter()
            try:
                response_body = json.dumps(response, cls=_NumpyEncoder).encode("utf-8")
            except (ValueError, TypeError) as error:
                response_body = json.dumps(make_error_response(error)).encode("utf-8")
            encode_finished = time.perf_counter()
            if self.path == "/call" and request_id is not None and first:
                self.server.receipts.complete(
                    session_id,
                    request_id,
                    response_body,
                    dispatch_finished - dispatch_started,
                    encode_finished - dispatch_finished,
                )
        except Exception as exc:
            response_body = json.dumps(make_error_response(exc)).encode("utf-8")
            dispatch_finished = time.perf_counter()
            encode_finished = dispatch_finished

        compressed = False
        if len(response_body) >= _GZIP_MIN_BYTES and _accepts_gzip(
            self.headers.get("Accept-Encoding", "")
        ):
            compressed_body = gzip.compress(response_body, compresslevel=1, mtime=0)
            if len(compressed_body) < len(response_body):
                response_body = compressed_body
                compressed = True
        compress_finished = time.perf_counter()
        # Always 200; failures are described inside the body via ok=False.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-RPC-Instance-ID", self.server.instance_id)
        self.send_header("X-RPC-Receipts", "1")
        self.send_header(
            "X-RPC-Readonly-Methods",
            json.dumps(sorted(self.server.readonly_methods), separators=(",", ":")),
        )
        self.send_header("Accept-Encoding", "gzip")
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Vary", "Accept-Encoding")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header(
            "X-RPC-Dispatch-Seconds", f"{dispatch_finished - dispatch_started:.9f}"
        )
        self.send_header(
            "X-RPC-Encode-Seconds", f"{encode_finished - dispatch_finished:.9f}"
        )
        self.send_header(
            "X-RPC-Compress-Seconds", f"{compress_finished - encode_finished:.9f}"
        )
        self.end_headers()
        try:
            self.wfile.write(response_body)
        except (BrokenPipeError, ConnectionResetError) as exc:
            # Client went away mid-response — log for visibility and move on.
            logger.debug("rpc http write failed: %s", exc)


class HttpRpcServer(ThreadingHTTPServer):
    """HTTP server that dispatches JSON-RPC calls.

    Same interface as ``SocketRpcServer`` — drop-in replacement at dispatch
    sites that use ``server_address``, ``serve_forever``, ``shutdown``, and
    ``server_close``.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[..., Any],
        *,
        inherited_socket: socket.socket | None = None,
        instance_id: str | None = None,
        readonly_methods: set[str] | None = None,
        receipt_max_entries: int = 4096,
        receipt_max_bytes: int = 256 * 1024 * 1024,
        receipt_ttl_s: float = 3600.0,
    ) -> None:
        self.instance_id = instance_id or uuid.uuid4().hex
        self.readonly_methods = frozenset({"healthz", *(readonly_methods or ())})
        self.receipts = RequestReceipts(
            max_entries=receipt_max_entries,
            max_bytes=receipt_max_bytes,
            ttl_s=receipt_ttl_s,
        )
        super().__init__(
            server_address,
            _HttpRpcHandler,
            bind_and_activate=inherited_socket is None,
        )
        if inherited_socket is not None:
            self.socket = inherited_socket
            self.server_address = inherited_socket.getsockname()
            self.server_name = socket.getfqdn(self.server_address[0])
            self.server_port = self.server_address[1]
            self.server_activate()
        self.dispatch = dispatch
