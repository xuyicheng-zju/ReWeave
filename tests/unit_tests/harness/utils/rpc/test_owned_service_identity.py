# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Real delayed subprocesses must not connect to another service's ready port."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reweave.robots.runtime import try_wait_server
from reweave.utils.daemon import SERVICE_INSTANCE_ENV, ProcessDaemon, pick_free_port
from reweave.utils.rpc.client_utils import make_rpc_client, wait_for_ready
from reweave.utils.rpc.client_utils import logger

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = r"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from reweave.utils.daemon import SERVICE_INSTANCE_ENV
from reweave.utils.rpc import RpcFacade
transport, port, events, gate, omit_identity = sys.argv[2:]
if omit_identity == 'yes':
    os.environ.pop(SERVICE_INSTANCE_ENV, None)
if gate != '-':
    while not Path(gate).exists():
        time.sleep(.01)
class Service(RpcFacade):
    def __init__(self):
        super().__init__(enable_sessions=True)
        self._rpc['env.reset'] = lambda *, session_id: 'reset'
    def _dispatch(self, method, args, kwargs, *, session_id=None):
        with open(events, 'a') as f:
            f.write(json.dumps({'method': method, 'instance_id': self._instance_id}) + '\n')
        return super()._dispatch(method, args, kwargs, session_id=session_id)
Service().serve(transport=transport, host='127.0.0.1', port=int(port),
                parent_watch=True, session_sweep_s=30.)
"""


def daemon_for(tmp_path, name, transport, port, *, gate="-", omit_identity=False):
    return ProcessDaemon(
        name,
        [
            sys.executable,
            "-c",
            SCRIPT,
            str(ROOT),
            transport,
            str(port),
            str(tmp_path / f"{name}.events.jsonl"),
            str(gate),
            "yes" if omit_identity else "no",
        ],
        env_overrides={SERVICE_INSTANCE_ENV: "must-be-overridden"},
        cwd=str(tmp_path),
        log_path=str(tmp_path / f"{name}.log"),
    )


def events(tmp_path, name):
    path = tmp_path / f"{name}.events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("transport", ["http", "socket"])
@pytest.mark.parametrize(
    "omit_identity, start_second", [(False, True), (True, True), (True, False)]
)
def test_forced_port_collision_rejected_before_register_or_reset(
    tmp_path, transport, omit_identity, start_second
):
    port = pick_free_port()
    endpoint = f"{transport}://127.0.0.1:{port}"
    first = daemon_for(tmp_path, "first", transport, port, omit_identity=omit_identity)
    # The second child is still loading when its client reaches the first server.
    second = daemon_for(tmp_path, "second", transport, port, gate=tmp_path / "load")
    probe = make_rpc_client(endpoint)
    wrong_client = make_rpc_client(endpoint, enable_sessions=True)
    log_path = tmp_path / "host.log"
    handler = logging.FileHandler(log_path)
    logger.addHandler(handler)
    first.start()
    try:
        wait_for_ready(probe, timeout_s=10, poll_interval_s=0.01)
        if start_second:
            second.start()
        expected = second.instance_id
        actual = None if omit_identity else first.instance_id
        assert expected != first.instance_id
        assert (expected is not None) == start_second
        emitted = []
        with pytest.raises(RuntimeError, match="service identity mismatch"):
            try_wait_server(
                {"second": second},
                SimpleNamespace(emit=emitted.append),
                "second",
                wrong_client,
                second,
                10,
                post_fn=lambda: wrong_client.call("env.reset"),
            )
        handler.flush()
        diagnostic = log_path.read_text()
        assert endpoint in diagnostic
        assert repr(expected) in diagnostic
        assert repr(actual) in diagnostic
        assert "refusing registration and business calls" in diagnostic
        assert (second.poll() is not None) == start_second
        wrong_client.close()
        assert {event["method"] for event in events(tmp_path, "first")} == {"healthz"}
        assert not (tmp_path / "second.events.jsonl").exists()

        # External endpoints deliberately have no owned-child identity contract.
        external = make_rpc_client(endpoint, enable_sessions=True)
        try:
            wait_for_ready(external, timeout_s=2, poll_interval_s=0.01)
            assert external.call("env.reset") == "reset"
        finally:
            external.close()
        methods = [event["method"] for event in events(tmp_path, "first")]
        assert methods.index("session.register") < methods.index("env.reset")
    finally:
        wrong_client.close()
        probe.close()
        second.stop(timeout=1)
        first.stop(timeout=1)
        logger.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_owned_service_restart_uses_new_identity_and_preserves_normal_connection(
    tmp_path, monkeypatch, transport
):
    monkeypatch.setenv(SERVICE_INSTANCE_ENV, "inherited-stale-identity")
    port = pick_free_port()
    endpoint = f"{transport}://127.0.0.1:{port}"
    daemon = daemon_for(tmp_path, "normal", transport, port)
    identities = []
    for _ in range(2):
        client = make_rpc_client(endpoint, enable_sessions=True)
        daemon.start()
        try:
            identities.append(daemon.instance_id)
            assert daemon.instance_id not in (
                "inherited-stale-identity",
                "must-be-overridden",
                None,
            )
            wait_for_ready(client, timeout_s=10, poll_interval_s=0.01, daemon=daemon)
            assert client.call("env.reset") == "reset"
            assert client.call("healthz")["instance_id"] == daemon.instance_id
            with pytest.raises(RuntimeError, match="already running"):
                daemon.start()
            assert daemon.instance_id == identities[-1]
        finally:
            client.close()
            daemon.stop(timeout=1)
    assert len(set(identities)) == 2
    assert {e["instance_id"] for e in events(tmp_path, "normal")} == set(identities)
    assert os.environ[SERVICE_INSTANCE_ENV] == "inherited-stale-identity"


def test_remote_service_readiness_fails_fast_when_its_local_forwarder_exits(
    monkeypatch,
):
    from reweave.utils.rpc import client_utils

    statuses = iter([None, 7])
    owner = SimpleNamespace(
        name="forwarder", instance_id="local-relay", poll=lambda: next(statuses)
    )
    calls = []

    class Client:
        def call(self, *args, **kwargs):
            calls.append(args)
            raise ConnectionRefusedError("relay unavailable")

    monkeypatch.setattr(client_utils.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="forwarder exited with code 7"):
        wait_for_ready(Client(), timeout_s=120, process_owner=owner)
    assert len(calls) == 1


def test_remote_service_identity_is_independent_of_local_forwarder():
    owner = SimpleNamespace(
        name="forwarder", instance_id="local-relay", poll=lambda: None
    )
    client = SimpleNamespace(
        call=lambda *args, **kwargs: {"instance_id": "remote-model"}
    )
    wait_for_ready(client, timeout_s=1, process_owner=owner)


def test_readiness_budget_survives_a_wall_clock_jump(monkeypatch):
    from reweave.utils.rpc import client_utils

    wall = iter((1000.0, 1001.0, 2000.0))
    elapsed = [0.0]
    monkeypatch.setattr(client_utils.time, "time", lambda: next(wall))
    monkeypatch.setattr(client_utils.time, "monotonic", lambda: elapsed[0])

    def advance(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(client_utils.time, "sleep", advance)
    calls = []

    class Client:
        def call(self, *args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise ConnectionRefusedError("still starting")
            return {"ready": True}

    wait_for_ready(Client(), timeout_s=2, poll_interval_s=1)
    assert len(calls) == 2 and elapsed[0] == 1
