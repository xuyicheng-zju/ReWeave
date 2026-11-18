# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""SSH pacing changes transport writes without replaying issued requests."""

import asyncio
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reweave.cli import remote_evolution
from reweave.evolution.infrastructure.ssh_channel import SshChannel


class Writer:
    def __init__(self):
        self.chunks = []
        self.drains = 0
        self.closed = False

    def write(self, raw):
        assert not self.closed
        self.chunks.append(raw)

    async def drain(self):
        self.drains += 1
        await asyncio.sleep(0)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("value", [0, -1, 0.5, True, float("inf"), "8192"])
def test_invalid_chunk_size_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        SshChannel(None, None, None, write_chunk_bytes=value)


@pytest.mark.parametrize(
    "value", [-0.1, float("inf"), float("-inf"), float("nan"), True, "0.05", 10**400]
)
def test_invalid_interval_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SshChannel(None, None, None, write_interval_s=value)


def test_chunks_drain_and_pace_between_chunks_without_frame_interleaving(monkeypatch):
    async def run():
        writer = Writer()
        original_sleep = asyncio.sleep
        waits = []

        async def sleep(seconds):
            waits.append(seconds)
            await original_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        channel = SshChannel(
            None, writer, None, write_chunk_bytes=8, write_interval_s=0.05
        )
        frames = [{"id": 1, "result": "字" * 50}, {"id": 2, "result": "y" * 70}]
        await asyncio.gather(*(channel.send(frame) for frame in frames))
        wire = b"".join(writer.chunks)
        assert [json.loads(line) for line in wire.splitlines()] == frames
        assert all(0 < len(chunk) <= 8 for chunk in writer.chunks)
        assert writer.drains == len(writer.chunks)
        assert waits.count(0.05) == len(writer.chunks) - len(frames)
        assert waits.count(0) == writer.drains
        assert not writer.closed

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["call", "reply"])
def test_waiting_lock_cancellation_does_not_issue_id_or_close_channel(kind):
    async def run():
        reader, writer = asyncio.StreamReader(), Writer()
        channel = SshChannel(reader, writer, None, write_chunk_bytes=8)
        pump = asyncio.create_task(channel.run())
        await channel.write_lock.acquire()
        queued = asyncio.create_task(
            channel.call("cancelled", {})
            if kind == "call"
            else channel.send({"id": 7, "result": "cancelled"})
        )
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert channel.sequence == 0
        assert channel.pending == {}
        assert writer.chunks == []
        assert not channel.closed
        channel.write_lock.release()
        sibling = asyncio.create_task(channel.call("sibling", {}))
        while not writer.chunks or not writer.chunks[-1].endswith(b"\n"):
            await asyncio.sleep(0)
        request = json.loads(b"".join(writer.chunks))
        assert request == {"id": 1, "method": "sibling", "params": {}}
        reader.feed_data(b'{"id":1,"result":"settled"}\n')
        assert await asyncio.wait_for(sibling, 1) == "settled"
        reader.feed_eof()
        await pump

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["drain", "interval"])
def test_partial_frame_cancellation_closes_without_appending_sibling(
    boundary, monkeypatch
):
    async def run():
        reader = asyncio.StreamReader()
        blocked = asyncio.Event()
        original_sleep = asyncio.sleep

        class BlockingWriter(Writer):
            async def drain(self):
                self.drains += 1
                if boundary == "drain":
                    blocked.set()
                    await asyncio.Event().wait()

        async def sleep(seconds):
            if seconds:
                blocked.set()
                await asyncio.Event().wait()
            else:
                await original_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        writer = BlockingWriter()
        channel = SshChannel(
            reader, writer, None, write_chunk_bytes=8, write_interval_s=0.05
        )
        pump = asyncio.create_task(channel.run())
        first = asyncio.create_task(channel.call("first", {"image": "x" * 100}))
        await blocked.wait()
        sibling = asyncio.create_task(channel.call("sibling", {}))
        await original_sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(ConnectionError, match="closed"):
            await sibling
        with pytest.raises(ConnectionError, match="cancelled before completion"):
            await pump
        assert writer.closed and channel.closed
        assert len(writer.chunks) == 1
        assert len(writer.chunks[0]) == 8
        assert b"\n" not in writer.chunks[0]
        assert channel.sequence == 1
        assert channel.pending == {}

    asyncio.run(run())


def test_json_encoding_error_does_not_consume_request_id():
    async def run():
        writer = Writer()
        channel = SshChannel(None, writer, None)
        with pytest.raises(ValueError, match="JSON compliant"):
            await channel.call("invalid", {"value": float("nan")})
        assert channel.sequence == 0
        assert not channel.pending
        assert not writer.chunks
        assert not channel.closed

    asyncio.run(run())


def cli_arguments(command):
    arguments = [
        command,
        "--workspace",
        "/runs/original",
        "--runtime-source",
        "/frozen/runtime",
    ]
    if command == "run":
        arguments += [
            "--host",
            "company48-agentic",
            "--python",
            "/env/bin/python",
            "--repository",
            "/frozen/host",
            "--local-state",
            "/local/state",
        ]
    return arguments


@pytest.mark.parametrize("command", ["run", "serve"])
@pytest.mark.parametrize("custom", [False, True])
def test_cli_pacing_parsed_on_both_endpoints(command, custom, monkeypatch):
    captured = {}

    async def endpoint(args, *unused):
        captured["args"] = args

    monkeypatch.setattr(remote_evolution, "local_host", endpoint)
    monkeypatch.setattr(remote_evolution, "serve", endpoint)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    arguments = cli_arguments(command)
    if custom:
        arguments += [
            "--ssh-write-chunk-bytes",
            "8192",
            "--ssh-write-interval-s",
            "0.005",
        ]
    monkeypatch.setattr(sys, "argv", ["remote_evolution", *arguments])
    remote_evolution.main()
    args = captured["args"]
    assert args.ssh_write_chunk_bytes == (8192 if custom else 65536)
    assert args.ssh_write_interval_s == (0.005 if custom else 0.0)
    if command == "run":
        words = shlex.split(remote_evolution.ssh_command(args)[-1])
        assert words[words.index("--ssh-write-chunk-bytes") + 1] == str(
            args.ssh_write_chunk_bytes
        )
        assert words[words.index("--ssh-write-interval-s") + 1] == str(
            args.ssh_write_interval_s
        )


@pytest.mark.parametrize("command", ["run", "serve"])
@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--ssh-write-chunk-bytes", "0"),
        ("--ssh-write-chunk-bytes", "-8"),
        ("--ssh-write-chunk-bytes", "1.5"),
        ("--ssh-write-interval-s", "-0.01"),
        ("--ssh-write-interval-s", "inf"),
        ("--ssh-write-interval-s", "nan"),
    ],
)
def test_cli_rejects_invalid_pacing_before_endpoint(command, flag, value, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["remote_evolution", *cli_arguments(command), flag, value]
    )
    with pytest.raises(SystemExit) as error:
        remote_evolution.main()
    assert error.value.code == 2


def test_remote_host_channel_receives_pacing(monkeypatch):
    captured = {}

    class Channel:
        def __init__(self, reader, writer, handler, **options):
            captured.update(options)

        async def run(self):
            pass

    monkeypatch.setattr(remote_evolution, "SshChannel", Channel)
    asyncio.run(
        remote_evolution.remote_host(
            SimpleNamespace(ssh_write_chunk_bytes=8192, ssh_write_interval_s=0.005),
            None,
            None,
        )
    )
    assert captured == {"write_chunk_bytes": 8192, "write_interval_s": 0.005}


def test_local_host_records_and_applies_pacing(tmp_path, monkeypatch):
    from reweave.evolution.authoring import local as local_author
    from reweave.evolution.infrastructure import processes
    from reweave.planner import codex_connection

    async def run():
        stopped = asyncio.Event()
        captured = {}

        class Input:
            def close(self):
                stopped.set()

        process = SimpleNamespace(stdin=Input(), stdout=object())

        async def launch(*command, **unused):
            captured["command"] = command
            return process

        class Channel:
            def __init__(self, reader, writer, handler, **options):
                captured["options"] = options

            async def run(self):
                await stopped.wait()

            async def call(self, method, params):
                assert method == "run"
                return {"completed": True}

        async def drain(owned):
            assert owned is process
            captured["drained"] = True

        monkeypatch.setattr(
            codex_connection, "codex_executable", lambda value: "/local/codex"
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        monkeypatch.setattr(remote_evolution, "SshChannel", Channel)
        monkeypatch.setattr(
            local_author,
            "LocalCodexAuthor",
            lambda *args, **kwargs: SimpleNamespace(handle=None),
        )
        monkeypatch.setattr(processes, "drain_process_group", drain)
        args = SimpleNamespace(
            host="company48-agentic",
            port=None,
            python="/env/bin/python",
            repository="/frozen/host",
            runtime_source=Path("/frozen/runtime"),
            workspace=Path("/runs/original"),
            environment=None,
            recover=False,
            local_state=tmp_path,
            codex_bin=None,
            ssh_write_chunk_bytes=8192,
            ssh_write_interval_s=0.005,
        )
        await remote_evolution.local_host(args)
        stored = json.loads((tmp_path / "connection.json").read_text())
        assert stored["connection"]["ssh_write_chunk_bytes"] == 8192
        assert stored["connection"]["ssh_write_interval_s"] == 0.005
        assert captured["options"] == {
            "write_chunk_bytes": 8192,
            "write_interval_s": 0.005,
        }
        assert captured["drained"]

    asyncio.run(run())
