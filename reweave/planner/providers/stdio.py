"""Owned installed CLI processes speaking newline-delimited JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from .base import NativeProtocolError, NativeRpcError, NativeTransportError
from .transport import EventObserver

# Installed Codex feature keys, scoped to online task decisions. Native local
# file tools stay available; Harness supplies its own task experience explicitly.
ONLINE_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "memories",
    "multi_agent",
    "plugins",
    "recommended_plugins",
    "remote_plugin",
    "skill_search",
    "tool_suggest",
    "workspace_dependencies",
)


class StdioJsonRpcTransport:
    """One owned native process; model routing never owns physical execution.

    A subclass launches the process in ``connect``. Server-initiated requests
    are answered from ``CLIENT_RESULTS``; any other method is refused, so a
    native callback never becomes robot I/O.
    """

    PROTOCOL: str
    CLIENT_RESULTS: dict[str, dict] = {}

    def __init__(self, directory: Path, *, executable: str):
        self.directory = Path(directory)
        self.executable = executable
        self._metadata: dict = {"protocol": self.PROTOCOL, "executable": executable}
        self._observer: EventObserver | None = None
        self._process = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, tuple[str, asyncio.Future]] = {}
        self._events: asyncio.Queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._sequence = 0
        self._closing = False

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def set_observer(self, observer: EventObserver) -> None:
        self._observer = observer

    def _observe(self, direction: str, message: dict) -> None:
        if self._observer is None:
            raise NativeProtocolError("Native transport requires an evidence observer")
        self._observer({"direction": direction, "message": message})

    async def _spawn(self, *argv: str, cwd: Path, environment: dict) -> None:
        self._closing = False
        self._events = asyncio.Queue()
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _write(self, message: dict) -> None:
        raw = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        async with self._write_lock:
            if (
                self._process is None
                or self._process.returncode is not None
                or (self._reader_task is not None and self._reader_task.done())
            ):
                raise NativeTransportError(
                    "Native stdio connection is closed", request_sent=False
                )
            # Persist before write: any crash after this boundary may have sent bytes.
            self._observe("outgoing", message)
            try:
                self._process.stdin.write(raw)
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionError, OSError) as error:
                raise NativeTransportError(
                    "Native stdio write was not confirmed"
                ) from error

    async def request(self, method: str, params: dict) -> dict:
        self._sequence += 1
        identity = self._sequence
        future = asyncio.get_running_loop().create_future()
        self._pending[identity] = (method, future)
        try:
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": identity,
                    "method": method,
                    "params": params,
                }
            )
            return await future
        finally:
            self._pending.pop(identity, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def notify(self, method: str, params: dict) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def next_event(self) -> dict:
        value = await self._events.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def _read_message(self) -> dict | None:
        # StreamReader's default line limit is not an output quota. Large native
        # events are assembled in chunks rather than rejected at 64 KiB.
        chunks = []
        while True:
            try:
                chunks.append(await self._process.stdout.readuntil(b"\n"))
                break
            except asyncio.LimitOverrunError as error:
                chunks.append(await self._process.stdout.readexactly(error.consumed))
            except asyncio.IncompleteReadError as error:
                if chunks or error.partial:
                    raise NativeProtocolError(
                        "Native stdio ended inside a JSON frame"
                    ) from error
                return None
        try:
            value = json.loads(b"".join(chunks))
        except ValueError as error:
            raise NativeProtocolError("Native stdio contained invalid JSON") from error
        if not isinstance(value, dict):
            raise NativeProtocolError("Native JSON-RPC frame must be an object")
        return value

    async def _read_loop(self) -> None:
        failure: BaseException = NativeTransportError("Native stdio ended")
        try:
            while (message := await self._read_message()) is not None:
                self._observe("incoming", message)
                if "method" not in message:
                    pending = self._pending.get(message.get("id"))
                    if pending is None or pending[1].done():
                        continue
                    method, future = pending
                    if "error" in message:
                        future.set_exception(NativeRpcError(method, message["error"]))
                    elif isinstance(message.get("result"), dict):
                        future.set_result(message["result"])
                    else:
                        future.set_exception(
                            NativeProtocolError("Native RPC result must be an object")
                        )
                elif "id" in message:
                    result = self.CLIENT_RESULTS.get(message["method"])
                    await self._write(
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "error": {
                                "code": -32601,
                                "message": "Client method is not configured",
                            },
                        }
                        if result is None
                        else {"jsonrpc": "2.0", "id": message["id"], "result": result}
                    )
                else:
                    self._events.put_nowait(message)
        except asyncio.CancelledError:
            if not self._closing:
                raise
        except BaseException as error:
            failure = error
        finally:
            for _, future in self._pending.values():
                if not future.done():
                    future.set_exception(failure)
            self._events.put_nowait(failure)

    async def _drain_stderr(self) -> None:
        with (self.directory / "stderr.log").open("ab") as stream:
            while chunk := await self._process.stderr.read(64 * 1024):
                stream.write(chunk)
                stream.flush()

    async def close(self) -> None:
        self._closing = True
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        # The owned process has exited; drain both pipes before releasing it.
        try:
            for task in (self._reader_task, self._stderr_task):
                if task is not None:
                    await task
        finally:
            self._process = None


class CodexStdioTransport(StdioJsonRpcTransport):
    """Installed Codex ``app-server`` with its selected local connection."""

    PROTOCOL = "stdio-jsonrpc"

    def __init__(
        self,
        directory: Path,
        *,
        executable: str = "codex",
        codex_auth_mode: str = "auth",
    ):
        from reweave.planner.codex_connection import codex_connection_config

        codex_connection_config(codex_auth_mode)
        self.codex_auth_mode = codex_auth_mode
        super().__init__(directory, executable=executable)

    @staticmethod
    async def _configured_mcp(
        executable: str, cwd: Path, overrides: list[str], environment: dict
    ) -> dict:
        # This command only reads effective configuration. Its output can contain
        # credentials in transport fields, so neither raw stream enters evidence.
        process = await asyncio.create_subprocess_exec(
            executable,
            *overrides,
            "mcp",
            "list",
            "--json",
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            raise
        if process.returncode != 0:
            raise NativeTransportError(
                "Installed Codex could not enumerate its MCP configuration",
                request_sent=False,
            )
        try:
            rows = json.loads(stdout)
        except (ValueError, UnicodeDecodeError):
            raise NativeProtocolError(
                "Installed Codex returned invalid MCP configuration metadata"
            ) from None
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or type(row.get("enabled")) is not bool
            for row in rows
        ):
            raise NativeProtocolError(
                "Installed Codex returned invalid MCP configuration metadata"
            )
        servers = {row["name"]: row["enabled"] for row in rows}
        if len(servers) != len(rows):
            raise NativeProtocolError("Installed Codex returned duplicate MCP names")
        return servers

    async def connect(self, cwd: Path) -> None:
        if self._process is not None:
            raise NativeProtocolError("Native transport is already connected")
        executable = shutil.which(self.executable)
        if executable is None:
            raise NativeTransportError(
                "Installed Codex CLI was not found", request_sent=False
            )
        # A package-manager alias can execute successfully inside Seatbelt but
        # fail native account routing. Use the same canonical binary as offline
        # authors and the filesystem allowlist, including native installations.
        executable = str(Path(executable).resolve())
        if Path(executable).name == "codex.js":
            from reweave.planner.codex_connection import codex_executable

            executable = codex_executable(executable)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._metadata["executable"] = executable
        # Use the selected local connection; never inherit global config, plugin
        # registries or session databases into this online episode runtime.
        from reweave.planner.codex_connection import (
            codex_connection_config,
            codex_connection_overrides,
            codex_environment,
        )

        # Auth/runtime state must stay outside recursively indexed episode
        # evidence. Only its non-secret locator belongs to the Host record.
        from .runtime_storage import online_runtime_home

        native_home, resume_runtime = online_runtime_home(
            self.directory, source_root=Path(__file__).resolve().parents[3]
        )
        environment = codex_environment(
            native_home, codex_auth_mode=self.codex_auth_mode, resume=resume_runtime
        )
        environment["CODEX_HOME"] = str(native_home)
        from reweave.evolution.authoring.workspace import native_author_launcher

        agents = cwd / "AGENTS.md"
        launcher, runtime_environment = native_author_launcher(
            cwd,
            cwd / "harness",
            self.directory / "launcher",
            backend="codex",
            cli=executable,
            readable_roots=tuple(
                path.resolve()
                for path in (
                    Path("/usr"),
                    Path("/bin"),
                    Path("/lib"),
                    Path("/lib64"),
                    Path("/etc"),
                    Path("/System"),
                    Path("/dev"),
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                )
                if path.exists()
            ),
            writable_roots=(cwd / "harness/task-family",),
            native_runtime_root=native_home.parent,
        )
        if Path(runtime_environment["CODEX_HOME"]) != native_home:
            raise NativeProtocolError(
                "Online native runtime layout differs from locator"
            )
        environment.update(runtime_environment)
        overrides = [
            argument
            for override in codex_connection_overrides(self.codex_auth_mode)
            for argument in ("-c", override)
        ]
        overrides.extend(
            [
                "-c",
                f"project_doc_max_bytes={agents.stat().st_size + 1 if agents.is_file() else 0}",
                "-c",
                'project_root_markers=["AGENTS.md"]',
                "-c",
                'developer_instructions=""',
                "-c",
                'web_search="disabled"',
                "-c",
                "skills.max_context_tokens=1",
            ]
        )
        for feature in ONLINE_DISABLED_FEATURES:
            overrides.extend(["-c", f"features.{feature}=false"])
        servers = await self._configured_mcp(executable, cwd, overrides, environment)
        # Empty tables merge with inherited servers. Leaf overrides also replace
        # each server's required transport, and dotted CLI paths cannot quote a
        # server name. A TOML table with disabled inert entries handles all three
        # cases without copying original command arguments, env or headers.
        disabled = ",".join(
            f"{json.dumps(name, ensure_ascii=False)}="
            f"{{enabled=false,command={json.dumps(executable)}}}"
            for name in sorted(servers)
        )
        overrides.extend(["-c", "mcp_servers={" + disabled + "}"])
        effective = await self._configured_mcp(executable, cwd, overrides, environment)
        if any(effective.values()):
            raise NativeProtocolError(
                "Installed Codex did not disable its configured MCP servers"
            )
        self._metadata["disabled_mcp_servers"] = sorted(servers)
        self._metadata["native_home"] = str(native_home)
        self._metadata["disabled_features"] = list(ONLINE_DISABLED_FEATURES)
        self._metadata["codex_auth_mode"] = self.codex_auth_mode
        connection = codex_connection_config(self.codex_auth_mode)
        provider = connection["model_provider"]
        # A preference is configuration, not proof of an actual socket or cache
        # hit. Native fallback and reconnect behavior remain owned by Codex.
        self._metadata["api_transport_preference"] = (
            "responses_websocket"
            if connection.get(f"model_providers.{provider}.supports_websockets")
            else "responses_http"
        )
        await self._spawn(
            launcher,
            *overrides,
            "app-server",
            "--listen",
            "stdio://",
            cwd=cwd,
            environment=environment,
        )
