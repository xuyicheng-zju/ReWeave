"""SSH authentication bursts must not become unknown physical submissions."""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from reweave.robots import remote_environment as remote
from tests.unit_tests.harness.robots.test_remote_environment import arguments


def test_failed_authentication_is_known_not_submitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REWEAVE_SERVICE_INSTANCE_ID", "a" * 32)
    monkeypatch.setattr(remote, "stop_event", lambda: threading.Event())

    def connect(_):
        raise OSError("SSH banner unavailable before authentication")

    def remote_state(*args):
        raise AssertionError("No remote command could have been submitted")

    monkeypatch.setattr(remote, "connect", connect)
    monkeypatch.setattr(remote, "remote_state", remote_state)
    configuration = {
        "devices": [0],
        "suites": ["libero_spatial_task"],
        "remote_root": "/not-launched",
    }
    with pytest.raises(OSError, match="banner"):
        remote.bridge(configuration, arguments())
    receipt = json.loads((tmp_path / "remote_environment_closed.json").read_text())
    assert receipt["submission"] == "not_submitted"
    assert receipt["exit"] == {"parent_closed": True, "never_spawned": True}


AUTHENTICATE = """
import fcntl,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import paramiko
from reweave.robots.remote_environment import connect
root=Path(sys.argv[1])
class Client:
    def load_host_keys(self, path): pass
    def set_missing_host_key_policy(self, policy): pass
    def close(self): pass
    def get_transport(self): return SimpleNamespace(set_keepalive=lambda seconds:None)
    def connect(self, *args, **kwargs):
        def change(delta):
            with (root/'counts.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                value=json.loads((root/'counts.json').read_text())
                value['active']+=delta
                value['peak']=max(value['peak'],value['active'])
                (root/'counts.json').write_text(json.dumps(value))
        change(1)
        time.sleep(.1)
        change(-1)
paramiko.SSHClient=Client
(root/f'ready-{os.getpid()}').touch()
while not (root/'release').exists():time.sleep(.01)
connect({'connection_file':str(root/'connection.json')}).close()
"""


def test_separate_bridge_processes_serialize_authentication_only(tmp_path):
    (tmp_path / "connection.json").write_text(
        json.dumps(
            {
                "known_hosts": "synthetic",
                "hostname": "synthetic",
                "port": 22,
                "username": "synthetic",
                "identity_file": "synthetic",
            }
        )
    )
    (tmp_path / "counts.json").write_text('{"active":0,"peak":0}')
    processes = []
    try:
        for _ in range(4):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", AUTHENTICATE, str(tmp_path)],
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        deadline = time.monotonic() + 15
        while len(list(tmp_path.glob("ready-*"))) < 4:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        (tmp_path / "release").touch()
        for process in processes:
            _, error = process.communicate(timeout=15)
            assert process.returncode == 0, error.decode()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
    assert json.loads((tmp_path / "counts.json").read_text()) == {
        "active": 0,
        "peak": 1,
    }
