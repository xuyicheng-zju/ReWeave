# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

from __future__ import annotations

import signal
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reweave.utils.daemon import ProcessDaemon, reserve_free_port
from reweave.utils.rpc.client_utils import make_rpc_client, wait_for_ready
from reweave.utils.rpc.http_rpc import HttpRpcClient


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _python_daemon(
    tmp_path: Path,
    script: str,
    *,
    name: str = "contract-daemon",
) -> tuple[ProcessDaemon, Path]:
    log_path = tmp_path / f"{name}.log"
    return (
        ProcessDaemon(
            name,
            [sys.executable, "-c", script],
            log_path=str(log_path),
            cwd=str(tmp_path),
        ),
        log_path,
    )


class UnavailableClient:
    def call(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ConnectionError("not ready")


def test_daemon_starts_logs_and_stops_idempotently(tmp_path: Path) -> None:
    daemon, log_path = _python_daemon(
        tmp_path,
        "import time; print('ready', flush=True); time.sleep(30)",
    )

    try:
        assert daemon.poll() is None
        daemon.start()
        _wait_until(lambda: log_path.exists() and "ready" in log_path.read_text())
        assert daemon.poll() is None
    finally:
        daemon.stop(timeout=1.0)

    assert daemon.poll() is not None
    assert log_path.read_text().strip() == "ready"
    daemon.stop(timeout=0.01)


def test_wait_for_ready_fails_fast_when_daemon_exits(tmp_path: Path) -> None:
    daemon, _ = _python_daemon(
        tmp_path,
        "import sys; sys.exit(7)",
        name="crashing-daemon",
    )
    daemon.start()

    try:
        with pytest.raises(RuntimeError, match="exited with code 7"):
            wait_for_ready(
                UnavailableClient(),
                timeout_s=3.0,
                poll_interval_s=0.01,
                daemon=daemon,
            )
    finally:
        daemon.stop(timeout=1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX SIGTERM semantics")
def test_daemon_force_kills_after_terminate_timeout(tmp_path: Path) -> None:
    daemon, log_path = _python_daemon(
        tmp_path,
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ignoring-term', flush=True); "
        "time.sleep(30)",
        name="stubborn-daemon",
    )
    daemon.start()

    try:
        _wait_until(
            lambda: log_path.exists() and "ignoring-term" in log_path.read_text()
        )
        daemon.stop(timeout=0.05)
        _wait_until(lambda: daemon.poll() is not None)
    finally:
        daemon.stop(timeout=1.0)


@pytest.mark.parametrize("bad_lease", [False, True])
def test_failed_start_closes_log_handle(tmp_path, monkeypatch, bad_lease):
    daemon = ProcessDaemon(
        "broken",
        ["/nonexistent/reweave-test-command"],
        log_path=str(tmp_path / "server.log"),
    )
    if bad_lease:
        daemon.subprocess_env["REWEAVE_ROLLOUT_LEASE_FD"] = "not-an-integer"
    with pytest.raises((OSError, ValueError)):
        daemon.start()
    assert daemon._log_f is None
    assert daemon._proc is None


def test_reserved_port_remains_owned_across_rpc_child_spawn(tmp_path: Path) -> None:
    reservation = reserve_free_port()
    endpoint = f"http://{reservation.host}:{reservation.port}"
    daemon = ProcessDaemon(
        "reserved-rpc",
        [
            sys.executable,
            "-c",
            """
from reweave.utils.rpc.rpc_facade import RpcFacade
facade = RpcFacade()
facade.serve(transport='http', host=__import__('sys').argv[1],
             port=int(__import__('sys').argv[2]), parent_watch=True)
""",
            reservation.host,
            str(reservation.port),
        ],
        cwd=str(Path(__file__).resolve().parents[4]),
        log_path=str(tmp_path / "reserved-rpc.log"),
        port_reservation=reservation,
    )
    competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            competitor.bind((reservation.host, reservation.port))
        daemon.start()
        client = HttpRpcClient(endpoint)
        wait_for_ready(client, timeout_s=5.0, poll_interval_s=0.01, daemon=daemon)
        assert client.call("healthz")["instance_id"] == daemon.instance_id
    finally:
        competitor.close()
        daemon.stop(timeout=1.0)


def test_failed_spawn_atomically_releases_reserved_port(tmp_path: Path) -> None:
    reservation = reserve_free_port()
    endpoint = (reservation.host, reservation.port)
    daemon = ProcessDaemon(
        "failed-reserved-rpc",
        ["/nonexistent/reweave-test-command"],
        log_path=str(tmp_path / "failed-reserved-rpc.log"),
        port_reservation=reservation,
    )
    with pytest.raises(OSError):
        daemon.start()
    competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        competitor.bind(endpoint)
    finally:
        competitor.close()
        daemon.stop()


def test_failed_log_open_atomically_releases_reserved_port(tmp_path: Path) -> None:
    reservation = reserve_free_port()
    endpoint = (reservation.host, reservation.port)
    daemon = ProcessDaemon(
        "failed-log-open",
        [sys.executable, "-c", "raise AssertionError('must not spawn')"],
        log_path=str(tmp_path / "missing" / "server.log"),
        port_reservation=reservation,
    )
    with pytest.raises(FileNotFoundError):
        daemon.start()
    competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        competitor.bind(endpoint)
    finally:
        competitor.close()
        daemon.stop()


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_reserved_rpc_daemon_restarts_on_same_endpoint(
    tmp_path: Path, transport: str
) -> None:
    reservation = reserve_free_port("localhost")
    endpoint = f"{transport}://{reservation.host}:{reservation.port}"
    daemon = ProcessDaemon(
        f"restart-{transport}",
        [
            sys.executable,
            "-c",
            """
from reweave.utils.rpc.rpc_facade import RpcFacade
facade = RpcFacade()
facade.serve(transport=__import__('sys').argv[1],
             host=__import__('sys').argv[2],
             port=int(__import__('sys').argv[3]), parent_watch=True)
""",
            transport,
            reservation.host,
            str(reservation.port),
        ],
        cwd=str(Path(__file__).resolve().parents[4]),
        log_path=str(tmp_path / f"restart-{transport}.log"),
        port_reservation=reservation,
    )
    competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            competitor.bind((reservation.host, reservation.port))
        first_client = make_rpc_client(endpoint)
        daemon.start()
        wait_for_ready(first_client, timeout_s=5.0, poll_interval_s=0.01, daemon=daemon)
        first_identity = first_client.call("healthz")["instance_id"]
        first_client.close()
        daemon.stop(timeout=1.0)

        daemon.start()
        second_client = make_rpc_client(endpoint)
        wait_for_ready(
            second_client, timeout_s=5.0, poll_interval_s=0.01, daemon=daemon
        )
        second_identity = second_client.call("healthz")["instance_id"]
        second_client.close()
        assert second_identity != first_identity
        with pytest.raises(OSError):
            competitor.bind((reservation.host, reservation.port))
    finally:
        competitor.close()
        daemon.stop(timeout=1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX SIGTERM semantics")
@pytest.mark.parametrize(
    "configured, explicit, completed",
    [(1.0, None, True), (0.03, 1.0, True), (1.0, 0.03, False)],
)
def test_configured_shutdown_grace_drains_evidence_and_explicit_override_wins(
    tmp_path, configured, explicit, completed
):
    ready = tmp_path / "ready"
    evidence = tmp_path / "settled-evidence.bin"
    script = """
import signal, sys, time
from pathlib import Path
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text('ready')
while not stopping:
    time.sleep(.005)
time.sleep(.15)
Path(sys.argv[2]).write_bytes(b'complete original evidence')
"""
    daemon = ProcessDaemon(
        "evidence-drain",
        [sys.executable, "-c", script, str(ready), str(evidence)],
        stop_timeout_s=configured,
    )
    daemon.start()
    try:
        _wait_until(ready.is_file)
        daemon.stop(timeout=explicit)
        assert daemon.poll() == (0 if completed else -signal.SIGKILL)
        assert evidence.exists() is completed
        if completed:
            assert evidence.read_bytes() == b"complete original evidence"
    finally:
        daemon.stop(timeout=1.0)


@pytest.mark.parametrize(
    "timeout", [None, 0, -1, True, "300", float("inf"), float("nan")]
)
def test_daemon_rejects_unbounded_or_invalid_constructor_shutdown_grace(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        ProcessDaemon("invalid-grace", [sys.executable], stop_timeout_s=timeout)
