"""SDK shutdown must not kill the trusted owner before its drain is committed."""

import asyncio
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from reweave.evolution.infrastructure.native_codex import _close_codex_client
from reweave.utils.persistence import atomic_json


class SDK:
    """Retain the SDK's actual two-second escalation when it is called too soon."""

    def __init__(self, process):
        self._client = SimpleNamespace(_sync=SimpleNamespace(_proc=process))
        self.closed = False
        self.killed = False

    async def close(self):
        def close():
            process = self._client._sync._proc
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.killed = True
                    process.kill()
                    process.wait()
            self._client._sync._proc = None
            self.closed = True

        await asyncio.to_thread(close)


def launch_owner(directory, delay):
    code = """
import json,os,signal,sys,time
from pathlib import Path
directory,delay=Path(sys.argv[1]),float(sys.argv[2])
receipt=directory/'launcher/process.json'
receipt.parent.mkdir(parents=True)
receipt.write_text(json.dumps({'pid':os.getpid(),'descendants_drained':False}))
def stop(*_):
    (directory/'stopping').write_text('settling original writers')
    time.sleep(delay)
    temporary=receipt.with_name('.process.json.tmp')
    with temporary.open('w') as stream:
        json.dump({'pid':os.getpid(),'descendants_drained':True},stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(receipt)
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
(directory/'ready').write_text('ready')
while True: time.sleep(1)
"""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(directory), str(delay)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


async def wait_for(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5)


def test_slow_supervisor_drains_before_sdk_two_second_kill(tmp_path, monkeypatch):
    process = launch_owner(tmp_path, 2.2)
    sdk = SDK(process)

    async def run():
        await wait_for(tmp_path / "ready")
        monkeypatch.setattr(
            "reweave.evolution.infrastructure.native_codex.sys.platform", "linux"
        )
        started = time.monotonic()
        await _close_codex_client(sdk, tmp_path)
        assert time.monotonic() - started >= 2.0

    try:
        asyncio.run(run())
        assert process.returncode == 0
        assert sdk.closed and not sdk.killed
        assert json.loads((tmp_path / "launcher/process.json").read_text()) == {
            "pid": process.pid,
            "descendants_drained": True,
        }
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {"pid": 42, "descendants_drained": False},
        {"pid": 42, "descendants_drained": 1},
        {"pid": 43, "descendants_drained": True},
        "incomplete terminal receipt",
    ],
)
def test_missing_or_bad_receipt_never_becomes_a_confirmed_drain(
    tmp_path, monkeypatch, receipt
):
    monkeypatch.setattr(
        "reweave.evolution.infrastructure.native_codex.sys.platform", "linux"
    )
    process = SimpleNamespace(pid=42, poll=lambda: 0, wait=lambda: 0)
    sdk = SDK(process)
    path = tmp_path / "launcher/process.json"
    if isinstance(receipt, dict):
        atomic_json(path, receipt)
    elif receipt is not None:
        path.parent.mkdir(parents=True)
        path.write_text(receipt)
    original = path.read_bytes() if path.exists() else None
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        asyncio.run(_close_codex_client(sdk, tmp_path))
    assert not sdk.closed
    assert (path.read_bytes() if path.exists() else None) == original


def test_repeated_cancellation_keeps_owner_until_receipt_is_committed(
    tmp_path, monkeypatch
):
    process = launch_owner(tmp_path, 0.3)
    sdk = SDK(process)

    async def run():
        await wait_for(tmp_path / "ready")
        monkeypatch.setattr(
            "reweave.evolution.infrastructure.native_codex.sys.platform", "linux"
        )
        closing = asyncio.create_task(_close_codex_client(sdk, tmp_path))
        await wait_for(tmp_path / "stopping")
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert sdk.closed and not sdk.killed
        assert process.returncode == 0
        assert (
            json.loads((tmp_path / "launcher/process.json").read_text())[
                "descendants_drained"
            ]
            is True
        )

    try:
        asyncio.run(run())
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_already_closed_sdk_keeps_idempotent_close(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.infrastructure.native_codex.sys.platform", "linux"
    )
    closed = []

    async def close():
        closed.append(True)

    client = SimpleNamespace(
        _client=SimpleNamespace(_sync=SimpleNamespace(_proc=None)), close=close
    )
    asyncio.run(_close_codex_client(client, tmp_path))
    assert closed == [True]
