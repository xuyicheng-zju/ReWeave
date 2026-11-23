from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reweave.planner.providers.base import (
    NativeProtocolError,
    NativeRpcError,
    NativeTransportError,
)
from reweave.planner.providers.stdio import (
    ONLINE_DISABLED_FEATURES,
    CodexStdioTransport,
)


@pytest.fixture(autouse=True)
def isolated_account_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setattr(
        "reweave.planner.codex_connection.codex_environment",
        lambda native_home, *, resume, codex_auth_mode: {
            "CODEX_HOME": str(native_home)
        },
    )


class FakeWriter:
    def __init__(self, process):
        self.process = process
        self.messages = []
        self.sent = asyncio.Event()

    def write(self, raw):
        message = json.loads(raw)
        self.messages.append(message)
        self.sent.set()
        if message.get("method") == "echo":
            self.process.emit({"id": message["id"], "result": message["params"]})
        elif message.get("method") == "reject":
            self.process.emit(
                {
                    "id": message["id"],
                    "error": {"code": 403, "message": "refused"},
                }
            )

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        self.process.finish()


class FakeProcess:
    def __init__(self):
        self.returncode = None
        # Deliberately tiny to prove a native event is not capped by readline.
        self.stdout = asyncio.StreamReader(limit=1024)
        self.stderr = asyncio.StreamReader()
        self.stdin = FakeWriter(self)

    def emit(self, message):
        self.stdout.feed_data((json.dumps(message) + "\n").encode())

    def finish(self):
        self.returncode = 0
        if not self.stdout.at_eof():
            self.stdout.feed_eof()
        if not self.stderr.at_eof():
            self.stderr.feed_eof()

    async def wait(self):
        return self.returncode


class ConfigProcess:
    def __init__(self, stdout=b"[]", *, returncode=0, stderr=b""):
        self.returncode = None
        self.result = (stdout, stderr)
        self.exit_code = returncode

    async def communicate(self):
        self.returncode = self.exit_code
        return self.result


async def transport_with_process(tmp_path):
    transport = CodexStdioTransport(tmp_path)
    observations = []
    transport.set_observer(observations.append)
    process = FakeProcess()
    transport._process = process
    transport._reader_task = asyncio.create_task(transport._read_loop())
    transport._stderr_task = asyncio.create_task(transport._drain_stderr())
    return transport, process, observations


def test_stdio_demultiplexes_rpc_and_preserves_notifications_and_large_events(tmp_path):
    async def exercise():
        transport, process, observations = await transport_with_process(tmp_path)
        process.emit(
            {
                "method": "thread/compacted",
                "params": {"threadId": "thread-one", "detail": "x" * 200_000},
            }
        )
        result = await transport.request("echo", {"threadId": "thread-one"})
        assert result == {"threadId": "thread-one"}
        event = await transport.next_event()
        assert len(event["params"]["detail"]) == 200_000
        assert any(
            item["direction"] == "incoming" and item["message"].get("result") == result
            for item in observations
        )
        assert observations[0]["direction"] == "outgoing"
        await transport.close()
        assert transport._reader_task.done()
        assert transport._stderr_task.done()

    asyncio.run(exercise())


def test_stdio_explicit_rpc_error_remains_distinct_from_connection_failure(tmp_path):
    async def exercise():
        transport, _, observations = await transport_with_process(tmp_path)
        with pytest.raises(NativeRpcError) as caught:
            await transport.request("reject", {})
        assert caught.value.error == {"code": 403, "message": "refused"}
        assert not isinstance(caught.value, NativeTransportError)
        assert any(
            item["message"].get("error", {}).get("code") == 403 for item in observations
        )
        await transport.close()

    asyncio.run(exercise())


def test_unconfigured_server_tool_callback_is_rejected_without_execution(tmp_path):
    async def exercise():
        transport, process, _ = await transport_with_process(tmp_path)
        process.emit(
            {
                "id": "server-call",
                "method": "item/tool/call",
                "params": {"tool": "move_pose", "arguments": {"arm": "left"}},
            }
        )
        await transport.request("echo", {})
        responses = [
            message
            for message in process.stdin.messages
            if message.get("id") == "server-call"
        ]
        assert responses == [
            {
                "jsonrpc": "2.0",
                "id": "server-call",
                "error": {"code": -32601, "message": "Client method is not configured"},
            }
        ]
        await transport.close()

    asyncio.run(exercise())


def test_close_wakes_inflight_rpc_and_drains_owned_stderr(tmp_path):
    async def exercise():
        transport, process, _ = await transport_with_process(tmp_path)
        process.stderr.feed_data(b"diagnostic\n" * 20_000)
        pending = asyncio.create_task(transport.request("pending", {}))
        await process.stdin.sent.wait()
        await transport.close()
        with pytest.raises(NativeTransportError) as caught:
            await pending
        assert caught.value.request_sent is True
        assert (tmp_path / "stderr.log").read_bytes() == b"diagnostic\n" * 20_000
        assert transport._pending == {}

    asyncio.run(exercise())


def test_cancelled_rpc_response_is_still_observed_and_not_replayed(tmp_path):
    async def exercise():
        transport, process, observations = await transport_with_process(tmp_path)
        pending = asyncio.create_task(transport.request("pending", {}))
        await process.stdin.sent.wait()
        identity = process.stdin.messages[-1]["id"]
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        process.emit({"id": identity, "result": {"turn": {"id": "original-turn"}}})
        await transport.request("echo", {})
        assert any(
            item["message"].get("result", {}).get("turn", {}).get("id")
            == "original-turn"
            for item in observations
        )
        assert (
            sum(
                message.get("method") == "pending" for message in process.stdin.messages
            )
            == 1
        )
        await transport.close()

    asyncio.run(exercise())


def test_broken_json_protocol_fails_pending_request_without_new_write(tmp_path):
    async def exercise():
        transport, process, _ = await transport_with_process(tmp_path)
        pending = asyncio.create_task(transport.request("pending", {}))
        await process.stdin.sent.wait()
        process.stdout.feed_data(b"not-json\n")
        with pytest.raises(NativeProtocolError):
            await pending
        before = len(process.stdin.messages)
        with pytest.raises(NativeTransportError) as caught:
            await transport.request("echo", {})
        assert caught.value.request_sent is False
        assert len(process.stdin.messages) == before
        await transport.close()

    asyncio.run(exercise())


def test_audit_failure_prevents_native_write(tmp_path):
    async def exercise():
        transport, process, _ = await transport_with_process(tmp_path)

        def broken_evidence(event):
            raise OSError("evidence disk unavailable")

        transport.set_observer(broken_evidence)
        with pytest.raises(OSError, match="evidence disk"):
            await transport.request("echo", {})
        assert process.stdin.messages == []
        await transport.close()

    asyncio.run(exercise())


def test_connect_uses_installed_cli_stdio_and_no_sdk_or_login_request(
    tmp_path, monkeypatch
):
    async def exercise():
        commands = []
        process = FakeProcess()

        async def spawn(*command, **kwargs):
            commands.append(command)
            assert kwargs["cwd"] == tmp_path
            if command[-3:] == ("mcp", "list", "--json"):
                return ConfigProcess()
            return process

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which",
            lambda executable: "/installed/codex",
        )
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        transport = CodexStdioTransport(tmp_path)
        transport.set_observer(lambda event: None)
        await transport.connect(tmp_path)
        assert len(commands) == 3
        assert [command[-3:] for command in commands] == [
            ("mcp", "list", "--json"),
            ("mcp", "list", "--json"),
            ("app-server", "--listen", "stdio://"),
        ]
        for command in commands:
            assert 'model_provider="openai"' in command
            assert 'web_search="disabled"' in command
            assert 'developer_instructions=""' in command
            assert "project_doc_max_bytes=0" in command
            assert all(
                f"features.{feature}=false" in command
                for feature in ONLINE_DISABLED_FEATURES
            )
        assert "mcp_servers={}" in commands[-1]
        assert transport.metadata == {
            "protocol": "stdio-jsonrpc",
            "codex_auth_mode": "auth",
            "executable": "/installed/codex",
            "disabled_mcp_servers": [],
            "native_home": json.loads((tmp_path / "runtime_home.json").read_text())[
                "path"
            ],
            "disabled_features": list(ONLINE_DISABLED_FEATURES),
            "api_transport_preference": "responses_websocket",
        }
        await transport.close()

    asyncio.run(exercise())


def test_connect_resolves_native_package_manager_alias_before_all_cli_calls(
    tmp_path, monkeypatch
):
    async def exercise():
        binary = tmp_path / "installed/codex"
        binary.parent.mkdir()
        binary.write_text("native executable placeholder")
        alias = tmp_path / "bin/codex"
        alias.parent.mkdir()
        alias.symlink_to(binary)
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which", lambda _: str(alias)
        )
        commands = []

        async def spawn(*command, **kwargs):
            commands.append(command)
            if command[-3:] == ("mcp", "list", "--json"):
                return ConfigProcess()
            return FakeProcess()

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        transport = CodexStdioTransport(tmp_path)
        transport.set_observer(lambda _: None)
        await transport.connect(tmp_path)
        assert all(command[0] == str(binary) for command in commands[:2])
        config = json.loads((tmp_path / "launcher/workspace.json").read_text())
        assert config["cli"] == str(binary)
        assert transport.metadata["executable"] == str(binary)
        await transport.close()

    asyncio.run(exercise())


def test_connect_disables_every_inherited_mcp_without_recording_secrets(
    tmp_path, monkeypatch
):
    async def exercise():
        observations = []
        commands = []
        process = FakeProcess()
        rows = [
            {
                "name": name,
                "enabled": enabled,
                "transport": {"env": {"TOKEN": "private-credential"}},
            }
            for name, enabled in [("global-robot", True), ("project.robot", False)]
        ]

        async def spawn(*command, **kwargs):
            assert kwargs["cwd"] == tmp_path
            commands.append(command)
            if command[-3:] == ("mcp", "list", "--json"):
                disabled = any(
                    '"global-robot"={enabled=false,command="/installed/codex"}'
                    in argument
                    for argument in command
                )
                result = (
                    [{**row, "enabled": False} for row in rows] if disabled else rows
                )
                return ConfigProcess(json.dumps(result).encode())
            return process

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which", lambda _: "/installed/codex"
        )
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        transport = CodexStdioTransport(tmp_path)
        transport.set_observer(observations.append)
        await transport.connect(tmp_path)
        command = commands[-1]
        assert (
            'mcp_servers={"global-robot"={enabled=false,command="/installed/codex"},'
            '"project.robot"={enabled=false,command="/installed/codex"}}'
        ) in command
        assert "mcp_servers={}" not in command
        assert transport.metadata["disabled_mcp_servers"] == [
            "global-robot",
            "project.robot",
        ]
        assert "private-credential" not in json.dumps(
            {
                "commands": commands,
                "metadata": transport.metadata,
                "events": observations,
            }
        )
        assert observations == []
        await transport.close()

    asyncio.run(exercise())


def test_mcp_disable_failure_never_starts_appserver(tmp_path, monkeypatch):
    async def exercise():
        commands = []

        async def spawn(*command, **kwargs):
            commands.append(command)
            return ConfigProcess(b'[{"name":"robot","enabled":true}]')

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which", lambda _: "/installed/codex"
        )
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        transport = CodexStdioTransport(tmp_path)
        transport.set_observer(lambda _: None)
        with pytest.raises(NativeProtocolError, match="did not disable"):
            await transport.connect(tmp_path)
        assert len(commands) == 2
        assert all(command[-3:] == ("mcp", "list", "--json") for command in commands)
        assert transport._process is None
        await transport.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("stdout", "returncode", "error_type"),
    [
        (b"private-invalid-json", 0, NativeProtocolError),
        (b"private stdout", 1, NativeTransportError),
        (b'[{"name":"private-name"}]', 0, NativeProtocolError),
    ],
)
def test_mcp_configuration_errors_do_not_expose_raw_output(
    tmp_path, monkeypatch, stdout, returncode, error_type
):
    async def exercise():
        async def spawn(*command, **kwargs):
            return ConfigProcess(
                stdout, returncode=returncode, stderr=b"private-header-secret"
            )

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which", lambda _: "/installed/codex"
        )
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        transport = CodexStdioTransport(tmp_path)
        observations = []
        transport.set_observer(observations.append)
        with pytest.raises(error_type) as error:
            await transport.connect(tmp_path)
        assert "private" not in str(error.value)
        assert error.value.__cause__ is None
        assert observations == []
        assert transport._process is None

    asyncio.run(exercise())


def test_online_home_is_private_outside_episode_and_persists_on_reconnect(
    tmp_path, monkeypatch
):
    async def exercise():
        episode = tmp_path / "episode"
        episode.mkdir()
        environments = []

        async def spawn(*command, **kwargs):
            environments.append(kwargs["env"])
            if command[-3:] == ("mcp", "list", "--json"):
                return ConfigProcess()
            return FakeProcess()

        monkeypatch.setattr(
            "reweave.planner.providers.stdio.shutil.which", lambda _: "/installed/codex"
        )
        monkeypatch.setattr(
            "reweave.planner.providers.stdio.asyncio.create_subprocess_exec", spawn
        )
        homes = []
        for _ in range(2):
            transport = CodexStdioTransport(episode)
            transport.set_observer(lambda _: None)
            await transport.connect(episode)
            home = Path(transport.metadata["native_home"])
            assert not home.is_relative_to(episode)
            assert home.is_relative_to(tmp_path / "user/.codex/harness-online")
            assert not (home / "config.toml").exists()  # Global config was not copied.
            homes.append(home)
            await transport.close()
        assert homes[0] == homes[1]
        assert all(item["CODEX_HOME"] == str(homes[0]) for item in environments)
        assert not list(episode.rglob("auth.json"))
        assert json.loads((episode / "runtime_home.json").read_text()) == {
            "path": str(homes[0])
        }

    asyncio.run(exercise())


def test_native_boundary_reads_submitted_images_without_request_audit_access(tmp_path):
    from reweave.evolution.authoring.workspace import native_author_launcher
    from reweave.planner.providers.codex import CodexDecisionSession

    workspace = tmp_path / "workspace"
    provider = tmp_path / "provider"
    session = CodexDecisionSession(provider)
    request = provider / "requests/0123456789abcdef0123456789abcdef"
    archive = request / "images/0.png"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"immutable camera pixels")
    (request / "input.json").write_text('"Host-only request audit"')
    submitted = session._submission_image_path(request, {"path": "head.png"}, 0)
    launcher, environment = native_author_launcher(
        workspace,
        workspace / "harness",
        provider / "launcher",
        backend="codex",
        cli=str(Path(sys.executable).resolve()),
        readable_roots=tuple(
            path.resolve()
            for path in (
                Path(sys.prefix),
                Path(sys.base_prefix),
                *(
                    Path(p)
                    for p in (
                        "/usr",
                        "/bin",
                        "/lib",
                        "/lib64",
                        "/etc",
                        "/System",
                        "/dev",
                    )
                ),
            )
            if path.exists()
        ),
        writable_roots=(workspace / "harness/task-family",),
    )
    code = """
from pathlib import Path
import sys
image, archive, audit = map(Path, sys.argv[1:])
assert image.read_bytes() == b'immutable camera pixels'
try:
    image.write_bytes(b'modified')
except PermissionError:
    pass
else:
    raise AssertionError('submitted image is writable')
for forbidden in (archive, audit):
    try:
        forbidden.read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(f'Host audit is readable: {forbidden}')
print('native_image_boundary_verified')
"""
    result = subprocess.run(
        [
            launcher,
            "-c",
            code,
            str(submitted),
            str(archive),
            str(request / "input.json"),
        ],
        cwd=workspace,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "native_image_boundary_verified"
    assert archive.read_bytes() == b"immutable camera pixels"
