# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Client utility functions: endpoint parsing, server discovery, and readiness polling."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from reweave.utils.logging import get_logger

if TYPE_CHECKING:
    from reweave.utils.daemon import ProcessDaemon
    from reweave.utils.rpc.rpc_client import RpcClient

logger = get_logger("client_utils")


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def make_rpc_client(endpoint: str, *, enable_sessions: bool = False) -> "RpcClient":
    """Build an HTTP or socket client for ``endpoint``."""
    from reweave.utils.rpc.http_rpc import HttpRpcClient
    from reweave.utils.rpc.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}", enable_sessions=enable_sessions)
    if protocol == "socket":
        return SocketRpcClient(host, port, enable_sessions=enable_sessions)
    raise ValueError(f"endpoint protocol must be http or socket, got {protocol!r}")


def wait_for_ready(
    client: "RpcClient",
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    daemon: "ProcessDaemon | None" = None,
    process_owner: "ProcessDaemon | None" = None,
) -> None:
    """Poll ``client.call("healthz")`` until it succeeds or ``timeout_s`` elapses.

    If ``daemon`` is given and its subprocess exits before becoming ready,
    fail fast with ``RuntimeError`` (carrying the exit code) instead of
    blocking until ``timeout_s``.

    ``process_owner`` checks relay liveness without comparing the remote
    service's identity with its local relay. ``daemon`` also checks RPC identity.

    Owned children must echo their per-start identity. A healthy server at
    the chosen port may belong to another launch; reject it before registering
    a session or allowing the caller to connect business clients.

    Once the server is reachable, if the client is session-aware (it owns a
    private ``_session_id``), this function registers the session with the
    server once. The sid stays private to the client; business code never
    sees it.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        owner = daemon or process_owner
        if owner is not None:
            rc = owner.poll()
            if rc is not None:
                detail = last_err if last_err is not None else "no healthz attempt yet"
                raise RuntimeError(
                    f"{getattr(owner, 'name', 'Owned service forwarder')} exited with code {rc} before becoming "
                    f"ready; check its log. last healthz error: {detail}"
                )
        try:
            health = client.call("healthz", timeout_s=1.0)
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval_s)
            continue
        if daemon is not None:
            actual = health.get("instance_id") if isinstance(health, dict) else None
            expected = daemon.instance_id
            if not expected or actual != expected:
                message = (
                    f"{daemon.name} RPC service identity mismatch at {client.endpoint}: "
                    f"expected instance_id={expected!r}, actual instance_id={actual!r}; "
                    "refusing registration and business calls"
                )
                logger.error(message)
                raise RuntimeError(message)
        sid = getattr(client, "_session_id", None)
        if sid is not None:
            try:
                client.call("session.register", timeout_s=30.0)
            except Exception as exc:
                raise RuntimeError(
                    f"server is reachable but session.register failed for "
                    f"sid={sid!r}: {exc}. The server process is now in a "
                    f"half-ready state; check its log."
                ) from exc
        return
    raise TimeoutError(
        f"server did not become ready within {timeout_s:.0f}s: {last_err}"
    )
