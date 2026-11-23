# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""SSH deployment preserves remote evidence, scoped writes and native identities."""

import argparse
import asyncio
import base64
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from reweave.cli.remote_evolution import (
    bind_deployment,
    ssh_command,
)
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.local import LocalCodexAuthor
from reweave.evolution.authoring.remote import RemoteCodexAuthor, RemoteWorkspaceTools
from reweave.evolution.infrastructure.ssh_channel import SshChannel


class PipeWriter:
    def __init__(self, reader):
        self.reader = reader

    def write(self, raw):
        self.reader.feed_data(raw)

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        self.reader.feed_eof()


def test_independent_local_requests_overlap_and_remote_events_are_isolated(tmp_path):
    async def run():
        author = LocalCodexAuthor(None, tmp_path, codex_bin=sys.executable)
        entered = set()
        barrier = asyncio.Event()

        async def query(request):
            entered.add(request["request_id"])
            if len(entered) == 3:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 5)
            return request["request_id"]

        author.query = query
        assert await asyncio.gather(
            *(
                author.handle(
                    "native_query", {"request_id": str(i), "session_key": str(i)}
                )
                for i in range(3)
            )
        ) == ["0", "1", "2"]
        remote = RemoteCodexAuthor(
            evidence_workspace=tmp_path / "evidence",
            channel=None,
            bundle_loader=None,
            policy=AuthorPolicy(author="codex", model="gpt-5.6-sol"),
        )
        events = {str(i): [] for i in range(3)}
        remote.active = {
            key: (None, None, tmp_path, values.append) for key, values in events.items()
        }
        await asyncio.gather(
            *(
                remote.handle("event", {"request_id": key, "event": key})
                for key in events
            )
        )
        assert events == {"0": ["0"], "1": ["1"], "2": ["2"]}
        with pytest.raises(ValueError, match="No active author"):
            await remote.handle("event", {"request_id": "missing", "event": "wrong"})

    asyncio.run(run())


def test_channel_reverse_calls_large_frames_and_errors():
    async def run():
        left_reader, right_reader = asyncio.StreamReader(), asyncio.StreamReader()

        async def left_handler(method, params):
            assert method == "reverse"
            return params

        async def right_handler(method, params):
            if method == "fail":
                raise ValueError("explicit rejection")
            return await right.call("reverse", params)

        left = SshChannel(left_reader, PipeWriter(right_reader), left_handler)
        right = SshChannel(right_reader, PipeWriter(left_reader), right_handler)
        pumps = [asyncio.create_task(left.run()), asyncio.create_task(right.run())]
        try:
            payload = {"original": "字" * 800000}
            assert await left.call("echo", payload) == payload
            with pytest.raises(RuntimeError, match="explicit rejection"):
                await left.call("fail", {})
        finally:
            left_reader.feed_eof()
            right_reader.feed_eof()
            await asyncio.gather(*pumps)

    asyncio.run(run())


def test_disconnect_cancels_remote_work_without_replay():
    async def run():
        reader = asyncio.StreamReader()
        sink = asyncio.StreamReader()
        started, stopped = asyncio.Event(), asyncio.Event()
        calls = []

        async def handler(method, params):
            calls.append(method)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        channel = SshChannel(reader, PipeWriter(sink), handler)
        pump = asyncio.create_task(channel.run())
        reader.feed_data(b'{"id":1,"method":"rollout","params":{}}\n')
        await started.wait()
        reader.feed_eof()
        await pump
        assert stopped.is_set()
        assert calls == ["rollout"]
        with pytest.raises(ConnectionError, match="closed"):
            await channel.call("rollout", {})

    asyncio.run(run())


@pytest.mark.parametrize(
    "reply",
    [{"result": {}}, {"error": {"type": "TimeoutError", "message": "expired"}}],
)
@pytest.mark.parametrize("during_send", [False, True])
def test_cancelled_request_reply_preserves_sibling(reply, during_send):
    async def run():
        reader, sink = asyncio.StreamReader(), asyncio.StreamReader()

        class Writer(PipeWriter):
            async def drain(self):
                if during_send and channel.sequence == 1:
                    await asyncio.Event().wait()
                await super().drain()

        channel = SshChannel(reader, Writer(sink), None)
        pump = asyncio.create_task(channel.run())
        cancelled = asyncio.create_task(channel.call("author", {}))
        first = json.loads(await sink.readline())
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        sibling = asyncio.create_task(channel.call("rollout", {}))
        second = json.loads(await sink.readline())
        reader.feed_data(
            json.dumps({"id": first["id"], **reply}).encode()
            + b"\n"
            + json.dumps({"id": second["id"], "result": "settled"}).encode()
            + b"\n"
        )
        try:
            assert await asyncio.wait_for(sibling, 1) == "settled"
            assert not channel.closed
            assert not channel.pending
            assert sink._buffer == b""  # Neither request was replayed.
        finally:
            reader.feed_eof()
            await pump

    asyncio.run(run())


@pytest.mark.parametrize(
    "extra",
    [
        {"id": 1, "result": {}},  # Duplicate late reply.
        {"id": 2, "result": {}},  # Never issued.
        {"id": 1, "result": {}, "extra": True},
    ],
)
def test_cancelled_request_does_not_accept_invalid_replies(extra):
    async def run():
        reader, sink = asyncio.StreamReader(), asyncio.StreamReader()
        channel = SshChannel(reader, PipeWriter(sink), None)
        call = asyncio.create_task(channel.call("author", {}))
        await sink.readline()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        if "extra" not in extra:
            reader.feed_data(b'{"id":1,"result":{}}\n')
        reader.feed_data(json.dumps(extra).encode() + b"\n")
        reader.feed_eof()
        with pytest.raises(ValueError, match="SSH author response"):
            await channel.run()

    asyncio.run(run())


@pytest.mark.parametrize("raw", [b'{"id":1', b'{"id":1,"result":{}}\n', b"[]\n"])
def test_torn_or_unmatched_frames_fail(raw):
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        channel = SshChannel(reader, PipeWriter(asyncio.StreamReader()), None)
        with pytest.raises((ValueError, ConnectionError)):
            await channel.run()

    asyncio.run(run())


def test_ssh_quotes_paths_and_binds_original_client(tmp_path):
    args = argparse.Namespace(
        host="worker.example.invalid",
        port=31259,
        python="/env/bin/python",
        repository="/project/a b'c",
        runtime_source=Path("/frozen/runtime a"),
        workspace=Path("/runs/a b"),
        environment="/env/planner.sh",
        recover=True,
        ssh_write_chunk_bytes=8192,
        ssh_write_interval_s=0.005,
    )
    command = ssh_command(args)
    assert command[-2] == "worker.example.invalid"
    assert command[command.index("-p") + 1] == "31259"
    assert shlex.quote(args.repository) in command[-1]
    assert shlex.quote(str(args.runtime_source)) in command[-1]
    assert "--recover" in command[-1]
    args.host = "-oProxyCommand=bad"
    with pytest.raises(ValueError, match="host"):
        ssh_command(args)
    args.host = "worker.example.invalid"
    args.port = 0
    with pytest.raises(ValueError, match="port"):
        ssh_command(args)
    args.port = 31259
    args.runtime_source = Path("relative/runtime")
    with pytest.raises(ValueError, match="absolute"):
        ssh_command(args)
    identity = str(uuid4())
    bind_deployment(tmp_path, identity)
    bind_deployment(tmp_path, identity)
    bind_deployment(tmp_path, str(uuid4()))
    binding = tmp_path / "author_deployment.json"
    stored = json.loads(binding.read_text())
    stored["authentication"] = "local_api"
    binding.write_text(json.dumps(stored))
    bind_deployment(tmp_path, identity)
    assert json.loads(binding.read_text())["authentication"] == "local_chatgpt"


def test_duplicate_request_is_not_replayed():
    async def run():
        reader = asyncio.StreamReader()
        frame = b'{"id":1,"method":"author","params":{}}\n'
        reader.feed_data(frame + frame)
        channel = SshChannel(reader, PipeWriter(asyncio.StreamReader()), None)
        with pytest.raises(ValueError, match="Malformed SSH author request"):
            await channel.run()

    asyncio.run(run())


def test_response_write_failure_stops_channel():
    class BrokenWriter:
        def write(self, raw):
            raise BrokenPipeError("peer vanished")

        def close(self):
            pass

    async def run():
        async def handler(method, params):
            return {}

        reader = asyncio.StreamReader()
        reader.feed_data(b'{"id":1,"method":"author","params":{}}\n')
        channel = SshChannel(reader, BrokenWriter(), handler)
        with pytest.raises(BrokenPipeError, match="peer vanished"):
            await asyncio.wait_for(channel.run(), timeout=1)
        assert channel.closed

    asyncio.run(run())


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="Host OS boundary")
@pytest.mark.parametrize("python_command", ["python", "python3"])
def test_remote_shell_permissions_images_and_no_credentials(
    tmp_path, monkeypatch, python_command
):
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-remote-tools")
    foreign_bin = tmp_path / "unrelated-venv/bin"
    foreign_bin.mkdir(parents=True)
    for name in ("python", "python3"):
        executable = foreign_bin / name
        executable.write_text("#!/bin/sh\nexit 77\n")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(foreign_bin) + os.pathsep + os.environ["PATH"])
    packages = tmp_path / "host-dependencies/site-packages"
    packages.mkdir(parents=True)
    (packages / "host_runtime_dependency.py").write_text("VALUE = 'host-runtime'\n")
    monkeypatch.syspath_prepend(str(packages))
    monkeypatch.setenv("PYTHONPATH", str(packages))
    workspace = tmp_path / "job/workspace"
    workspace.mkdir(parents=True)
    (workspace / "evidence.txt").write_text("original")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    readable = tuple(
        Path(p).resolve()
        for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/System", "/dev")
        if Path(p).exists()
    )
    (workspace / "context").mkdir(exist_ok=True)
    (workspace / "context/input.json").write_text(json.dumps({"kind": "mutation"}))
    tools = RemoteWorkspaceTools(
        workspace,
        workspace.parent,
        (
            *readable,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(sys.base_prefix).parent.resolve(),
            Path.cwd(),
        ),
        (workspace / "candidate",),
        "http://127.0.0.1:1/unavailable",
    )
    script = f"""
import os
from pathlib import Path
from host_runtime_dependency import VALUE
assert VALUE == 'host-runtime'
assert 'OPENAI_API_KEY' not in os.environ
assert Path('evidence.txt').read_text() == 'original'
Path('candidate/result.txt').write_text('verified')
Path('candidate/link').symlink_to({str(outside)!r})
for p in [Path('evidence.txt'),Path('candidate/link')]:
    try: p.write_text('corrupt')
    except PermissionError: pass
    else: raise AssertionError('write escaped')
try: Path({str(outside)!r}).read_text()
except PermissionError: pass
else: raise AssertionError('read escaped')
from PIL import Image
Image.new('RGB',(2,2),(120,50,30)).save('.scratch/media/frame.png')
print('boundary_verified')
"""
    result = asyncio.run(
        tools.call(
            "remote_shell", {"command": shlex.join([python_command, "-c", script])}
        )
    )
    assert not result["isError"], result
    assert result["structuredContent"]["stdout"].strip() == "boundary_verified"
    assert outside.read_text() == "private"
    image = asyncio.run(
        tools.call("remote_image", {"path": ".scratch/media/frame.png"})
    )
    assert not image["isError"], image
    assert (
        base64.b64decode(image["content"][0]["data"])
        == (workspace / ".scratch/media/frame.png").read_bytes()
    )
    denied = asyncio.run(tools.call("remote_image", {"path": str(outside)}))
    assert denied["isError"]


def test_remote_shell_bounds_model_output_but_archives_complete_bytes(tmp_path):
    workspace = tmp_path / "job/workspace"
    workspace.mkdir(parents=True)
    readable = tuple(
        Path(p).resolve()
        for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/System", "/dev")
        if Path(p).exists()
    )
    (workspace / "context").mkdir(exist_ok=True)
    (workspace / "context/input.json").write_text(json.dumps({"kind": "mutation"}))
    tools = RemoteWorkspaceTools(
        workspace,
        workspace.parent,
        (
            *readable,
            Path(sys.executable).resolve().parent,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(sys.base_prefix).parent.resolve(),
            Path.cwd(),
        ),
        (workspace / "candidate",),
        "http://127.0.0.1:1/unavailable",
    )
    result = asyncio.run(
        tools.call(
            "remote_shell",
            {"command": shlex.join([sys.executable, "-c", "print('x' * 200000)"])},
        )
    )["structuredContent"]
    assert not result["stdout_archive"]["complete"], result
    assert result["stdout_archive"]["bytes"] == 200001
    assert "output omitted" in result["stdout"]
    archived = Path(result["stdout_archive"]["path"])
    assert archived.read_bytes() == b"x" * 200000 + b"\n"


@pytest.mark.parametrize("ambient_api_env", [False, True])
@pytest.mark.parametrize("codex_auth_mode", ["auth", "api"])
@pytest.mark.parametrize("role", ["author", "graph_analysis"])
def test_local_codex_session_resume_without_downloading_remote_evidence(
    tmp_path, monkeypatch, ambient_api_env, role, codex_auth_mode
):
    from reweave.evolution.authoring import local as local_author

    account = tmp_path / "account"
    account.mkdir()
    (account / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "test-secret"}})
    )
    monkeypatch.setenv("CODEX_HOME", str(account))
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-token")
    model_provider = "openai" if codex_auth_mode == "auth" else "responses_api"
    if ambient_api_env:
        monkeypatch.setenv("CODEX_BASE_URL", "https://provider.example.invalid/v1")
        monkeypatch.setenv("CODEX_API_KEY", "private-test-key")
    else:
        monkeypatch.delenv("CODEX_BASE_URL", raising=False)
        monkeypatch.delenv("CODEX_API_KEY", raising=False)
    native = tmp_path / "native.jsonl"
    identity = str(uuid4())
    actions = []
    provider_error = None
    returned_provider = model_provider

    def launcher(workspace, output, control, **kwargs):
        nonlocal native
        home = workspace / ".scratch/runtime/codex"
        home.mkdir(parents=True, exist_ok=True)
        native = home / "sessions" / f"rollout-{identity}.jsonl"
        native.parent.mkdir(parents=True, exist_ok=True)
        return sys.executable, {"CODEX_HOME": str(home)}

    monkeypatch.setattr(local_author, "native_author_launcher", launcher)

    class Thread:
        id = identity

        async def turn(self, prompt, **kwargs):
            with native.open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": prompt,
                            },
                        }
                    )
                    + "\n"
                )
            return self

        async def stream(self):
            yield {
                "method": "turn/completed",
                "payload": {
                    "turn": {
                        "status": "failed" if provider_error else "completed",
                        "error": provider_error,
                    }
                },
            }

        async def read(self, **kwargs):
            return SimpleNamespace(
                thread=SimpleNamespace(
                    id=identity, path=str(native), model_provider=returned_provider
                )
            )

    class Client:
        def __init__(self, config):
            native_auth = Path(config.env["CODEX_HOME"]) / "auth.json"
            if codex_auth_mode == "auth":
                assert json.loads(native_auth.read_text())["auth_mode"] == "chatgpt"
                assert 'forced_login_method="chatgpt"' in config.config_overrides
                assert "OPENAI_API_KEY" not in config.env
            else:
                assert not native_auth.exists()
                assert config.env["OPENAI_API_KEY"] == "test-api-token"
                assert 'forced_login_method="api"' in config.config_overrides
            assert f'model_provider="{model_provider}"' in config.config_overrides
            assert "CODEX_API_KEY" not in config.env
            assert "CODEX_BASE_URL" not in config.env
            assert all(
                "private-test-key" not in item for item in config.config_overrides
            )

        async def account(self):
            return {"account": {"type": "chatgpt"}}

        async def thread_start(self, **kwargs):
            assert kwargs["model_provider"] == model_provider
            actions.append("start")
            native.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": identity}}) + "\n"
            )
            return Thread()

        async def thread_resume(self, session_id, **kwargs):
            assert kwargs["model_provider"] == model_provider
            assert session_id == identity
            actions.append("resume")
            with native.open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {"type": "thread_settings_applied"},
                        }
                    )
                    + "\n"
                )
            return Thread()

        async def close(self):
            pass

    monkeypatch.setattr(local_author.openai_codex, "AsyncCodex", Client)
    packets = []

    class Channel:
        async def call(self, method, params):
            packets.append((method, params))
            if method == "catalog":
                return []
            if method == "journal":
                from reweave.evolution.infrastructure.native_history import (
                    receive_history,
                )

                assert params["request_id"] in author.history_sources

                async def request(method, arguments):
                    return await author.handle(
                        method, {"request_id": params["request_id"], **arguments}
                    )

                received = await receive_history(
                    params["checkpoint"],
                    params["session_id"],
                    tmp_path / "received",
                    request,
                )
                assert received.read_bytes() == native.read_bytes()
            return {}

    author = LocalCodexAuthor(Channel(), tmp_path / "local", codex_bin=sys.executable)
    request = {
        "request_id": "/remote/job",
        "session_key": "/remote/job"
        if role != "graph_analysis"
        else "/remote/global-graph-session",
        "role": role,
        "workspace": "/remote/job/workspace",
        "session_id": None,
        "policy": AuthorPolicy(
            author="codex",
            codex_auth_mode=codex_auth_mode,
        ).model_dump(),
        "instructions": "Use remote originals",
        "prompt": "first",
    }
    first = asyncio.run(author.query(request))
    assert not author.history_sources
    local_binding = (
        author._record_directory(author.directory, request["session_key"], create=False)
        / "remote.json"
    )
    if codex_auth_mode == "auth":
        # Existing auth records predate the explicit connection field.
        recorded = json.loads(local_binding.read_text())
        recorded["policy"].pop("codex_auth_mode")
        local_binding.write_text(json.dumps(recorded) + "\n")
    binding_original = local_binding.read_bytes()
    changed_policy = request["policy"] | {
        "codex_auth_mode": "api" if codex_auth_mode == "auth" else "auth"
    }
    with pytest.raises(ValueError, match="session model, provider or role changed"):
        asyncio.run(
            author.query(
                request | {"session_id": first["session_id"], "policy": changed_policy}
            )
        )
    assert local_binding.read_bytes() == binding_original
    assert actions == ["start"]
    second = asyncio.run(
        author.query(
            request
            | {
                "session_id": first["session_id"],
                "prompt": "correction",
                "request_id": "/remote/job" if role == "author" else "/remote/round-2",
                "workspace": "/remote/job/workspace"
                if role == "author"
                else "/remote/round-2/workspace",
            }
        )
    )
    assert first["session_id"] == second["session_id"]
    assert local_binding.read_bytes() == binding_original
    assert actions == ["start", "resume"]
    assert {method for method, _ in packets} == {"catalog", "bind", "event", "journal"}
    assert {params["request_id"] for _, params in packets} == (
        {"/remote/job"} if role == "author" else {"/remote/job", "/remote/round-2"}
    )
    assert "test-secret" not in json.dumps(packets)
    assert not list((tmp_path / "local").rglob("*.mp4"))
    returned_provider = "responses_api" if model_provider == "openai" else "openai"
    original_journal = native.read_bytes()
    with pytest.raises(ValueError, match="session provider changed"):
        asyncio.run(author.query(request | {"session_id": identity}))
    # Resume may add provider metadata, but no new decision turn is submitted.
    assert native.read_bytes().count(b'"type": "response_item"') == (
        original_journal.count(b'"type": "response_item"')
    )
    assert local_binding.read_bytes() == binding_original
    returned_provider = model_provider
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(author.query(request))
    provider_error = {
        "message": "Provider rejected invalid_encrypted_content; request id: fixture"
    }
    prior_journals = sum(method == "journal" for method, _ in packets)
    with pytest.raises(RuntimeError, match="invalid_encrypted_content.*fixture"):
        asyncio.run(
            author.query(
                request
                | {
                    "session_id": second["session_id"],
                    "prompt": "continue into provider failure",
                }
            )
        )
    assert actions == ["start", "resume", "resume", "resume"]
    assert sum(method == "journal" for method, _ in packets) == prior_journals + 1
    assert not author.history_sources


def test_remote_journal_roundtrip_preserves_fork_lineage_and_updates_own_turn(tmp_path):
    from reweave.evolution.infrastructure.native_history import (
        NativeHistorySource,
        native_history,
    )
    from reweave.evolution.infrastructure.session import CodexSessionStore
    from tests.unit_tests.harness.evolution.test_native_history import (
        fork_journal,
        journal,
    )

    parent, child = str(uuid4()), str(uuid4())
    source_dir = tmp_path / "source"
    (source_dir / "lineage").mkdir(parents=True)
    journal(source_dir / f"lineage/{parent}.jsonl", parent)
    source = fork_journal(source_dir / "transcript.jsonl", child, parent)
    directory = tmp_path / "remote"
    store = CodexSessionStore(directory / "session")
    store.bind(child)

    class Channel:
        async def call(self, method, params):
            assert params["request_id"] == "request"
            return sender.handle(
                method,
                {key: value for key, value in params.items() if key != "request_id"},
            )

    remote = RemoteCodexAuthor(
        evidence_workspace=tmp_path / "evidence",
        channel=Channel(),
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
    )
    remote.active = {"request": (None, store, directory, None)}
    for turn in range(2):
        if turn:
            with source.open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {"type": "thread_settings_applied"},
                        }
                    )
                    + "\n"
                )
        sender = NativeHistorySource(source, child)
        result = asyncio.run(
            remote.handle(
                "journal",
                {
                    "request_id": "request",
                    "session_id": child,
                    "checkpoint": sender.checkpoint,
                },
            )
        )
        assert result == store.checkpoint()
        assert store.path.read_bytes() == source.read_bytes()
    assert len(native_history(store.path, child)) == 2
    assert not (directory / "session/upload").exists()
