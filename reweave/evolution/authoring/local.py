# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Local Codex sessions with remote, on-demand workspace tools."""

from __future__ import annotations

import asyncio
import fcntl
import json
import secrets
import sys
import uuid
from pathlib import Path

import openai_codex
from mcp import types
from mcp.server.lowlevel import Server
from openai_codex.generated.v2_all import ReasoningEffort

from reweave.evolution.authoring.author import NativeBundleAuthor, system_read_roots
from reweave.evolution.authoring.config import AuthorPolicy, connection_changed
from reweave.evolution.authoring.workspace import native_author_launcher
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.native_history import (
    NativeHistorySource,
)
from reweave.evolution.infrastructure.session import (
    CodexSessionStore,
    native_event_field,
    serialize_native_event,
)
from reweave.planner.codex_connection import (
    codex_connection_config,
    codex_connection_overrides,
    codex_environment,
    codex_model_config,
)
from reweave.planner.utils.http_mcp_server import HttpMcpServer
from reweave.utils.persistence import atomic_json


class LocalCodexAuthor:
    """Keep credentials and native session registry local; no rollout assets are fetched."""

    def __init__(self, channel, directory: Path, *, codex_bin: str):
        self.channel = channel
        self.directory = directory.resolve()
        self.codex_bin = str(Path(codex_bin).resolve(strict=True))
        self.locks: dict[str, asyncio.Lock] = {}
        self.history_sources: dict[str, NativeHistorySource] = {}

    @staticmethod
    def _record_directory(root: Path, record_id: str, *, create=True) -> Path:
        """Retain an allocated directory mapping; record text never becomes a path."""
        root.mkdir(parents=True, exist_ok=True)
        index = root / "directories.json"
        with (root / "directories.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = json.loads(index.read_text()) if index.exists() else {}
            if record_id not in records:
                if not create:
                    raise ValueError("parent session directory has not been registered")
                records[record_id] = uuid.uuid4().hex
                atomic_json(index, records)
            return root / records[record_id]

    async def handle(self, method: str, params: dict):
        if method == "history_chunk":
            source = self.history_sources[params["request_id"]]
            return await run_blocking(
                source.handle,
                method,
                {key: value for key, value in params.items() if key != "request_id"},
            )
        if method != "native_query":
            raise ValueError(f"Unknown local author request: {method}")
        async with self.locks.setdefault(params["session_key"], asyncio.Lock()):
            return await self.query(params)

    async def query(self, request: dict) -> dict:
        async def call(method, params):
            return await self.channel.call(
                method, {"request_id": request["request_id"], **params}
            )

        policy = AuthorPolicy.model_validate(request["policy"])
        if policy.author != "codex":
            raise ValueError("SSH account author cannot substitute another backend")
        directory = self._record_directory(self.directory, request["session_key"])
        workspace = directory / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        binding = directory / "remote.json"
        remote = {key: request[key] for key in ("session_key", "role", "policy")}
        if request["role"] not in {"author", "graph_analysis"}:
            raise ValueError("unsupported native job role")
        if (
            request["role"] == "author"
            and request["session_key"] != request["request_id"]
        ):
            raise ValueError("author sessions cannot share identities")
        if binding.exists():
            original = json.loads(binding.read_text())
            if (
                original["session_key"] != request["session_key"]
                or original["role"] != request["role"]
                or connection_changed(original["policy"], policy)
            ):
                raise ValueError("Local native session model, provider or role changed")
        else:
            atomic_json(binding, remote)
        round_binding = (
            self._record_directory(directory / "requests", request["request_id"])
            / "request.json"
        )
        frozen = {
            key: request[key]
            for key in ("request_id", "workspace", "policy", "role", "session_key")
        }
        atomic_json(round_binding, frozen)
        store = CodexSessionStore(directory / "session")
        NativeBundleAuthor._verify_process_exit(directory)
        session_id = request["session_id"]
        if session_id is not None:
            store.bind(session_id)
        elif (
            store.path.exists()
            or (directory / "session/identity.json").exists()
            or any((workspace / ".scratch/runtime/codex").rglob("*.jsonl"))
        ):
            raise ValueError(
                "Local session already exists; refusing a duplicate author"
            )

        owner_loop = asyncio.get_running_loop()
        catalog = await call("catalog", {})
        app = Server("remote_author", version="1.0.0")

        @app.list_tools()
        async def list_tools():
            return [types.Tool.model_validate(tool) for tool in catalog]

        @app.call_tool()
        async def call_tool(name, arguments):
            # HttpMcpServer runs in another thread; only the owning loop uses SSH.
            future = asyncio.run_coroutine_threadsafe(
                call("tool", {"name": name, "arguments": arguments or {}}),
                owner_loop,
            )
            result = await asyncio.wrap_future(future)
            return types.CallToolResult.model_validate(result)

        server = HttpMcpServer(app, path="/" + secrets.token_urlsafe(32))
        await run_blocking(server.start)
        client = None
        thread = None
        try:
            launcher, runtime_env = native_author_launcher(
                workspace,
                workspace / "candidate",
                directory / "launcher",
                backend="codex",
                cli=self.codex_bin,
                readable_roots=(
                    *system_read_roots(),
                    Path(self.codex_bin).parent,
                    Path(sys.prefix).resolve(),
                    Path(sys.base_prefix).resolve(),
                ),
                writable_roots=None,
            )
            native_home = Path(runtime_env["CODEX_HOME"])
            env = codex_environment(
                native_home,
                codex_auth_mode=policy.codex_auth_mode,
                resume=session_id is not None,
            )
            env.update(
                runtime_env,
                NO_PROXY="127.0.0.1,localhost",
                no_proxy="127.0.0.1,localhost",
            )
            overrides = [
                f"mcp_servers.remote_author.url={json.dumps(server.url)}",
                "mcp_servers.remote_author.required=true",
                'mcp_servers.remote_author.default_tools_approval_mode="approve"',
                *codex_connection_overrides(policy.codex_auth_mode),
                *(
                    f"{key}={value}"
                    for key, value in codex_model_config(policy.model).items()
                ),
            ]
            client = openai_codex.AsyncCodex(
                openai_codex.CodexConfig(
                    codex_bin=launcher,
                    cwd=str(workspace),
                    env=env,
                    config_overrides=tuple(overrides),
                )
            )
            if policy.codex_auth_mode == "auth":
                account = await client.account()
                if (
                    native_event_field(native_event_field(account, "account"), "type")
                    != "chatgpt"
                ):
                    raise ValueError("Local Codex did not authenticate with ChatGPT")
            instructions = request["instructions"] + (
                "\nDEPLOYMENT: All task paths in the request refer to the remote machine, not this local cwd. "
                "Use remote_shell for ALL workspace reads, searches, writes and commands; use remote_image to inspect images. "
                "Development tools execute on that same remote Host. Never synchronize/download the evidence directories. "
                "Search/select text and extract chosen video frames remotely. Do not use local shell/file tools for task work. "
                "Remote writes are limited by the Host OS to the role's output directory and .scratch/ as specified above. "
                "Use remote python3 scripts for structured JSON edits and mechanical indexing. "
                f"Remote workspace: {request['workspace']}"
            )
            settings = {
                "cwd": str(workspace),
                "model": policy.model,
                "model_provider": codex_connection_config(policy.codex_auth_mode)[
                    "model_provider"
                ],
                "approval_mode": openai_codex.ApprovalMode.deny_all,
                "sandbox": openai_codex.Sandbox.full_access,
                "developer_instructions": instructions,
            }
            if session_id is None:
                thread = await client.thread_start(ephemeral=False, **settings)
                store.bind(thread.id)
                atomic_json(
                    directory / "session/identity.json", {"session_id": thread.id}
                )
            else:
                thread = await client.thread_resume(session_id, **settings)
                before = await thread.read(include_turns=True)
                if before.thread.id != session_id or not before.thread.path:
                    raise ValueError("Original local native session is missing")
                if before.thread.model_provider != settings["model_provider"]:
                    raise ValueError("Original local native session provider changed")
                store.capture(Path(before.thread.path))
            await call("bind", {"session_id": thread.id})
            atomic_json(directory / "request.json", request)
            turn = await thread.turn(
                request["prompt"],
                model=policy.model,
                effort=ReasoningEffort(policy.reasoning_effort),
            )
            status = None
            turn_error = None
            async for event in turn.stream():
                serialized = serialize_native_event(event)
                if native_event_field(event, "method") == "error":
                    from reweave.utils.provider_admission import ProviderAdmission

                    payload = serialize_native_event(
                        native_event_field(event, "payload")
                    )
                    if isinstance(payload, dict):
                        ProviderAdmission(
                            "codex-" + policy.codex_auth_mode, lane="offline"
                        ).note_transport_failure(payload.get("error", {}))
                with (directory / "events.jsonl").open("a") as output:
                    output.write(json.dumps(serialized, ensure_ascii=False) + "\n")
                await call("event", {"event": serialized})
                if native_event_field(event, "method") == "turn/completed":
                    completed = native_event_field(
                        native_event_field(event, "payload"), "turn"
                    )
                    status = native_event_field(completed, "status")
                    turn_error = native_event_field(completed, "error")
            if native_event_field(status, "value", status) != "completed":
                raise RuntimeError(
                    "Local Codex turn did not complete: "
                    + json.dumps(serialize_native_event(turn_error), ensure_ascii=False)
                )
        finally:
            try:
                if thread is not None:
                    response = await thread.read(include_turns=True)
                    if response.thread.id != thread.id or not response.thread.path:
                        raise ValueError("Native Codex journal was not materialized")
                    store.capture(Path(response.thread.path))
                    source = await run_blocking(
                        NativeHistorySource, store.path, thread.id
                    )
                    self.history_sources[request["request_id"]] = source
                    try:
                        await call(
                            "journal",
                            {"session_id": thread.id, "checkpoint": source.checkpoint},
                        )
                    finally:
                        del self.history_sources[request["request_id"]]
            finally:
                if client is not None:
                    await client.close()
                await run_blocking(server.stop)
        return {"session_id": thread.id}
