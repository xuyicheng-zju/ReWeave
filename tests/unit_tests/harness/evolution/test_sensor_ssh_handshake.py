"""Bound simultaneous SSH handshakes without changing immutable sensor reads."""

import concurrent.futures
import threading
import time

import paramiko
import pytest

from reweave.evolution.evidence import server as server_evidence


def test_sensor_handshakes_are_serialized_across_independent_lock_descriptors(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server_evidence.tempfile, "gettempdir", lambda: str(tmp_path))
    start = threading.Barrier(4)
    guard = threading.Lock()
    active = maximum = 0

    def handshake():
        nonlocal active, maximum
        start.wait(timeout=2)
        with server_evidence._ssh_handshake(2) as remaining:
            assert 0 < remaining <= 2
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with guard:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(handshake) for _ in range(4)]
        for future in futures:
            future.result(timeout=3)
    assert maximum == 1


def test_handshake_wait_has_deadline_and_releases_after_owner_exits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server_evidence.tempfile, "gettempdir", lambda: str(tmp_path))
    with server_evidence._ssh_handshake(2):
        with pytest.raises(TimeoutError, match="queue timed out"):
            with server_evidence._ssh_handshake(0.05):
                raise AssertionError("another owner must not enter")
    with server_evidence._ssh_handshake(0.1):
        pass


@pytest.mark.parametrize(
    "failure", ["banner", "authentication", "host_key", "persistent"]
)
def test_only_pre_authentication_banner_drops_retry_with_bounded_backoff(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(server_evidence.tempfile, "gettempdir", lambda: str(tmp_path))
    clients, delays = [], []
    monkeypatch.setattr(server_evidence.time, "sleep", delays.append)

    class Client:
        def __init__(self):
            self.closed = False
            clients.append(self)

        def load_host_keys(self, path):
            pass

        def set_missing_host_key_policy(self, policy):
            assert isinstance(policy, paramiko.RejectPolicy)

        def connect(self, host, **kwargs):
            assert not kwargs["allow_agent"] and not kwargs["look_for_keys"]
            assert 0 < kwargs["timeout"] <= 20
            if failure == "banner" and len(clients) > 1:
                return
            if failure == "authentication":
                raise paramiko.AuthenticationException("authentication rejected")
            if failure == "host_key":
                raise paramiko.SSHException("host key verification rejected")
            raise paramiko.SSHException("Error reading SSH protocol banner")

        def close(self):
            self.closed = True

    monkeypatch.setattr(paramiko, "SSHClient", Client)
    config = {
        "known_hosts": "known_hosts",
        "hostname": "server",
        "port": 22,
        "username": "user",
        "identity_file": "key",
    }
    if failure == "banner":
        assert server_evidence._connect(config, 20) is clients[-1]
        assert [c.closed for c in clients] == [True, False]
        assert delays == [1]
    else:
        with pytest.raises(paramiko.SSHException):
            server_evidence._connect(config, 20)
        assert all(c.closed for c in clients)
        assert len(clients) == (4 if failure == "persistent" else 1)
        assert delays == ([1, 2, 4] if failure == "persistent" else [])
