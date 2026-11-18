"""Remote environment allocation and process ownership regressions."""

import asyncio
import errno
import json
import os
import signal
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

from reweave.robots import remote_environment as remote


def arguments():
    return [
        "--suite",
        "libero_spatial_task",
        "--task",
        "3",
        "--seed",
        "0",
        "--reset-state-id",
        "0",
        "--max-episode-steps",
        "12",
        "--cuda-device",
        "0",
    ]


@pytest.mark.parametrize("lane", [4, 7, 8, 15])
def test_extended_lane_requires_explicit_allocation_and_cannot_change_gpu(lane):
    from reweave.utils.gpu_slots import encode_slot

    configuration = {"devices": [0, 1], "suites": ["libero_spatial_task"]}
    slot = encode_slot(0, lane)
    with pytest.raises(ValueError, match="outside"):
        remote.validate_request(configuration, "a" * 32, arguments(), slot)
    configuration["slots_per_gpu"] = 16
    remote.validate_request(configuration, "a" * 32, arguments(), slot)
    with pytest.raises(ValueError, match="outside"):
        remote.validate_request(
            configuration, "a" * 32, arguments(), encode_slot(1, lane)
        )


def test_nonpooled_legacy_remote_gpu_indices_are_preserved():
    args = arguments()
    args[-1] = "7"
    remote.validate_request(
        {"devices": [7], "suites": ["libero_spatial_task"]}, "a" * 32, args, 28
    )


@pytest.mark.parametrize(
    "change", ["wrong_device", "wrong_suite", "duplicate", "negative_reset"]
)
def test_environment_rejects_misrouted_or_ambiguous_conditions(change):
    args = arguments()
    if change == "wrong_device":
        slot = 4
    else:
        slot = 0
    if change == "wrong_suite":
        args[1] = "libero_spatial"
    elif change == "duplicate":
        args[2] = "--seed"
    elif change == "negative_reset":
        args[7] = "-1"
    with pytest.raises(ValueError):
        remote.validate_request(
            {"devices": [0, 1], "suites": ["libero_spatial_task"]}, "a" * 32, args, slot
        )


def test_remote_supervisor_closes_descendants_before_publishing_exit(
    tmp_path, monkeypatch
):
    module = tmp_path / "reweave/benchmarks/libero"
    module.mkdir(parents=True)
    (module.parent / "__init__.py").touch()
    (module / "__init__.py").touch()
    (module / "env_server.py").write_text(
        "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\ntime.sleep(60)\n"
    )
    stopping = threading.Event()
    timer = threading.Timer(0.3, stopping.set)
    monkeypatch.setattr(remote, "stop_event", lambda **kwargs: stopping)
    configuration = {
        "python": sys.executable,
        "repository": str(tmp_path),
        "remote_root": str(tmp_path / "runtime"),
        "asset_config": str(tmp_path),
        "devices": [0, 1],
        "suites": ["libero_spatial_task"],
    }
    timer.start()
    try:
        asyncio.run(remote.supervise(configuration, "a" * 32, arguments(), 0))
    finally:
        timer.cancel()
    state = json.loads(
        (tmp_path / "runtime/owners" / ("a" * 32) / "owner.json").read_text()
    )
    assert state["status"] == "exited"
    assert state["exit"]["parent_closed"] is True
    with pytest.raises((ProcessLookupError, PermissionError)):
        os.killpg(state["pid"], 0)
    with pytest.raises(FileExistsError):
        asyncio.run(remote.supervise(configuration, "a" * 32, arguments(), 0))


def test_process_group_that_cannot_close_is_quarantined_instead_of_waiting_forever(
    monkeypatch,
):
    from reweave.evolution.infrastructure import processes

    calls = []
    clock = iter(range(0, 300, 10))
    monkeypatch.setattr(
        processes, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    monkeypatch.setattr(processes.os, "killpg", lambda pid, sig: calls.append(sig))

    async def done():
        return 0

    child = SimpleNamespace(pid=123, returncode=0, wait=done)
    with pytest.raises(RuntimeError, match="quarantine"):
        asyncio.run(processes.drain_process_group(child))
    assert signal.SIGTERM in calls and signal.SIGKILL in calls


def test_reconciliation_releases_only_the_original_closed_remote_owner(
    tmp_path, monkeypatch
):
    from reweave.utils.persistence import atomic_json

    path = tmp_path / "scheduler/online/slot-00.remote.json"
    original = {
        "status": "unsettled",
        "slot": 0,
        "service_instance_id": "a" * 32,
        "remote_output": "/remote/original",
    }
    atomic_json(path, original)
    state = {"status": "running", "service_instance_id": "a" * 32}
    monkeypatch.setattr(remote, "remote_state", lambda *_: state)
    remote.reconcile_slots({}, tmp_path)
    assert json.loads(path.read_text()) == original
    state.update(status="exited", exit={"parent_closed": True, "returncode": -15})
    state["service_instance_id"] = "b" * 32
    with pytest.raises(ValueError, match="owner's closed exit"):
        remote.reconcile_slots({}, tmp_path)
    assert json.loads(path.read_text()) == original
    state["service_instance_id"] = "a" * 32
    remote.reconcile_slots({}, tmp_path)
    assert json.loads(path.read_text())["status"] == "exited"


def test_failed_spawn_publishes_closed_resources_without_inventing_an_episode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(remote, "stop_event", lambda **kwargs: threading.Event())

    async def fail(*args, **kwargs):
        raise OSError("injected process creation failure")

    monkeypatch.setattr(remote.asyncio, "create_subprocess_exec", fail)
    configuration = {
        "python": sys.executable,
        "repository": str(tmp_path),
        "remote_root": str(tmp_path / "runtime"),
        "asset_config": str(tmp_path),
        "devices": [0],
        "suites": ["libero_spatial_task"],
    }
    with pytest.raises(OSError, match="creation failure"):
        asyncio.run(remote.supervise(configuration, "a" * 32, arguments(), 0))
    state = json.loads(
        (tmp_path / "runtime/owners" / ("a" * 32) / "owner.json").read_text()
    )
    assert state["status"] == "exited"
    assert state["exit"] == {
        "returncode": None,
        "parent_closed": True,
        "never_spawned": True,
    }
    assert "pid" not in state


@pytest.mark.parametrize("transport_dies", [False, True])
def test_channel_disconnect_does_not_retire_a_healthy_shared_transport(
    monkeypatch, transport_dies
):
    from reweave.utils.ssh import ReconnectingSSH

    clients = []
    completed = threading.Event()
    calls = []

    class Client:
        active = True

        def get_transport(self):
            return self

        def is_active(self):
            return self.active

        def open_channel(self, *args, **kwargs):
            return SimpleNamespace(close=lambda: None)

        def close(self):
            self.active = False

    def connect():
        client = Client()
        clients.append(client)
        return client

    pool = ReconnectingSSH(connect)

    def disconnect(*args):
        calls.append(1)
        if transport_dies:
            clients[-1].active = False
        raise OSError(errno.ENOTCONN, "HTTP peer already closed")

    release = pool.release

    def released(client):
        release(client)
        completed.set()

    monkeypatch.setattr(pool, "release", released)
    monkeypatch.setattr(remote, "relay", disconnect)
    server = remote.listener({}, 12345, connections=pool)
    try:
        for _ in range(2):
            completed.clear()
            with socket.create_connection(server.server_address):
                assert completed.wait(5)
        assert len(clients) == (2 if transport_dies else 1)
        assert len(calls) == 2  # Failed physical bytes are never replayed.
    finally:
        server.shutdown()
        server.server_close()
        pool.close()


def test_relay_keeps_draining_after_an_already_closed_write_half(monkeypatch):
    class Endpoint:
        def __init__(self, chunks):
            self.chunks = iter(chunks)
            self.received = []

        def recv(self, size):
            return next(self.chunks)

        def sendall(self, chunk):
            self.received.append(chunk)

        def shutdown_write(self):
            raise OSError(errno.ENOTCONN, "peer closed write half")

    left = Endpoint([b"request", b""])
    right = Endpoint([b"response-prefix", b"response-tail", b""])
    monkeypatch.setattr(
        remote.select, "select", lambda readers, *args: (readers[:], [], [])
    )
    remote.relay(left, right)
    assert left.received == [b"response-prefix", b"response-tail"]
    assert right.received == [b"request"]


@pytest.mark.parametrize("reusable", [False, True])
def test_forwarder_restarts_after_closed_connections_without_stealing_live_port(
    reusable,
):
    import socketserver

    class Server(remote.ForwardServer):
        allow_reuse_address = reusable

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(1)
            # The server closes first, leaving its accepted connection in TIME_WAIT.

    first = Server(("127.0.0.1", 0), Handler)
    address = first.server_address
    thread = threading.Thread(target=first.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(OSError) as busy:
            Server(address, Handler)
        assert busy.value.errno == errno.EADDRINUSE
        with socket.create_connection(address) as peer:
            peer.sendall(b"x")
            assert peer.recv(1) == b""
    finally:
        first.shutdown()
        first.server_close()
        thread.join(timeout=5)
    if reusable:
        second = Server(address, Handler)
        second.server_close()
    else:
        with pytest.raises(OSError) as time_wait:
            Server(address, Handler)
        assert time_wait.value.errno == errno.EADDRINUSE
