# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Remote matrix Host: local Codex is a transport, never a second evaluator."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

import httpx

from reweave.evolution.authoring.author import NativeBundleAuthor, NativeSession
from reweave.evolution.authoring.workspace import native_author_launcher, native_output
from reweave.evolution.contracts import Usage
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.native_accounting import (
    complete_query,
    cumulative_usage,
)
from reweave.evolution.infrastructure.native_history import receive_history
from reweave.evolution.infrastructure.processes import drain_process_group
from reweave.evolution.infrastructure.session import CodexSessionStore
from reweave.utils.persistence import atomic_json

_REMOTE_TEXT_LIMIT = 128 * 1024


def _bounded_text(raw: bytes, *, path: str) -> tuple[str, dict]:
    """Keep complete remote output archived while bounding one model tool result."""
    text = raw.decode(errors="replace")
    if len(text) <= _REMOTE_TEXT_LIMIT:
        return text, {"complete": True, "bytes": len(raw)}
    half = _REMOTE_TEXT_LIMIT // 2
    marker = f"\n... output omitted; complete bytes remain on the remote Host at {path} ...\n"
    return text[:half] + marker + text[-half:], {
        "complete": False,
        "bytes": len(raw),
        "path": path,
    }


class RemoteWorkspaceTools:
    """Remote shell inherits the production OS boundary, not the SSH user's authority."""

    def __init__(self, workspace, directory, readable, writable, development_url):
        self.workspace = workspace
        self.directory = directory
        self.readable = readable
        self.writable = writable
        self.development_url = development_url
        self.lock = asyncio.Lock()

    async def development(self, method: str, params: dict) -> dict:
        async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
            response = await client.post(
                self.development_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params,
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            response.raise_for_status()
            envelope = response.json()
        if "error" in envelope:
            raise RuntimeError(f"Remote development protocol: {envelope['error']}")
        return envelope["result"]

    async def catalog(self) -> list[dict]:
        catalog = (await self.development("tools/list", {}))["tools"]
        write_paths = ", ".join(
            path.relative_to(self.workspace).as_posix() for path in self.writable
        )
        return [
            *catalog,
            {
                "name": "remote_shell",
                "description": "Run a shell command in the remote author workspace. Searches, text selection and video frame extraction run remotely. Large stdout/stderr remain complete on the remote Host while this tool returns an explicitly marked head/tail view; narrow the command to read omitted material. Read inputs and evidence; write only "
                + write_paths
                + " and .scratch/. Other component interfaces are read-only. No physical rollout or background services. Use this instead of local shell/file tools.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "remote_image",
                "description": "View one remote PNG/JPEG/WebP/GIF without copying its containing directory. Returns original image bytes; no resizing. For videos, extract requested frames remotely with remote_shell, then view those frames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        ]

    async def shell(self, command: str, *, bounded: bool = True) -> dict:
        control = self.directory / "audit/remote_tools" / uuid.uuid4().hex / "launcher"
        launcher, runtime_env = native_author_launcher(
            self.workspace,
            native_output(self.workspace),
            control,
            backend="codex",
            cli="/bin/bash",
            readable_roots=(
                *self.readable,
                # Overlay environments can load dependencies from a second
                # site-packages directory without granting its whole venv.
                *(
                    Path(path).resolve()
                    for path in sys.path
                    if Path(path).name in {"site-packages", "dist-packages"}
                ),
            ),
            writable_roots=self.writable,
        )
        # SSH/Planner/account credentials are never inherited by remote commands.
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "HF_HUB_OFFLINE",
            }
        }
        env.update(runtime_env, REWEAVE_AUTHOR_TOOLS_URL=self.development_url)
        # Use the Host interpreter whose runtime roots are readable; inherited
        # PATH may select an unrelated, deliberately inaccessible environment.
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        atomic_json(
            control / "request.json", {"command": command, "cwd": str(self.workspace)}
        )
        process = await asyncio.create_subprocess_exec(
            launcher,
            "--noprofile",
            "--norc",
            "-c",
            command,
            cwd=self.workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, stderr = await asyncio.shield(communication)
        finally:
            await drain_process_group(process, terminate=True)
            stdout, stderr = await communication
            (control / "stdout").write_bytes(stdout)
            (control / "stderr").write_bytes(stderr)
            atomic_json(control / "exit.json", {"exit_code": process.returncode})
        if bounded:
            stdout_text, stdout_archive = _bounded_text(
                stdout, path=str(control / "stdout")
            )
            stderr_text, stderr_archive = _bounded_text(
                stderr, path=str(control / "stderr")
            )
        else:
            stdout_text, stderr_text = stdout.decode(), stderr.decode()
            stdout_archive = {"complete": True, "bytes": len(stdout)}
            stderr_archive = {"complete": True, "bytes": len(stderr)}
        return {
            "exit_code": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_archive": stdout_archive,
            "stderr_archive": stderr_archive,
        }

    async def call(self, name: str, arguments: dict) -> dict:
        async with self.lock:
            if name == "remote_shell":
                if set(arguments) != {"command"} or not isinstance(
                    arguments["command"], str
                ):
                    raise ValueError("remote_shell requires exactly one command string")
                result = await self.shell(arguments["command"])
            elif name == "remote_image":
                if set(arguments) != {"path"} or not isinstance(arguments["path"], str):
                    raise ValueError("remote_image requires exactly one path string")
                script = (
                    "import base64,json,sys; from pathlib import Path; from PIL import Image; "
                    "p=Path(sys.argv[1]); image=Image.open(p); image.verify(); "
                    "mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp','GIF':'image/gif'}[image.format]; "
                    "print(json.dumps({'type':'image','mimeType':mime,'data':base64.b64encode(p.read_bytes()).decode()}))"
                )
                result = await self.shell(
                    shlex.join([sys.executable, "-c", script, arguments["path"]]),
                    bounded=False,
                )
                if result["exit_code"] == 0:
                    return {"content": [json.loads(result["stdout"])], "isError": False}
            else:
                return await self.development(
                    "tools/call", {"name": name, "arguments": arguments}
                )
            return {
                "content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                ],
                "structuredContent": result,
                "isError": result["exit_code"] != 0,
            }


class RemoteCodexSession:
    """Reuse preparation/corrections/admission; only the native query runs locally."""

    def __init__(self, *, channel, **kwargs):
        super().__init__(**kwargs)
        if self.policy.author != "codex":
            raise ValueError(
                "Local-account SSH deployment requires an explicit Codex author"
            )
        self.channel = channel
        self.active = {}

    @staticmethod
    def _verify_process_exit(directory):
        NativeSession._verify_process_exit(directory)
        for job in (directory / "audit/remote_tools").glob("*"):
            NativeSession._verify_process_exit(job)

    async def _query_native(self, packet):
        return await self.channel.call("native_query", packet)

    async def handle(self, method: str, params: dict):
        request_id = params["request_id"]
        if request_id not in self.active:
            raise ValueError(
                "No active author request; remote workspace is unavailable"
            )
        tools, store, directory, record = self.active[request_id]
        if method == "catalog":
            return await tools.catalog()
        if method == "tool":
            return await tools.call(params["name"], params["arguments"])
        if method == "event":
            record(params["event"])
            return {}
        if method == "bind":
            store.bind(params["session_id"])
            identity = directory / "session/identity.json"
            if (
                identity.exists()
                and json.loads(identity.read_text())["session_id"] != store.session_id
            ):
                raise ValueError("Remote author session identity changed")
            atomic_json(identity, {"session_id": store.session_id})
            return {}
        if method == "journal":
            if set(params) != {"request_id", "session_id", "checkpoint"}:
                raise ValueError(
                    "Native journal requires its small checkpoint manifest"
                )
            if params["session_id"] != store.session_id:
                raise ValueError("Native journal belongs to another session")

            async def request(method, arguments):
                return await self.channel.call(
                    method, {"request_id": request_id, **arguments}
                )

            native = await receive_history(
                params["checkpoint"],
                store.session_id,
                directory / "audit/history_transfers",
                request,
                existing_journal=store.path if store.path.exists() else None,
            )
            await run_blocking(store.capture, native)
            return await run_blocking(store.checkpoint)
        raise ValueError(f"Unknown remote author method: {method}")

    async def _codex(
        self,
        workspace,
        directory,
        readable,
        writable,
        record,
        started,
        query_path,
        begin_query,
        instructions,
        native_prompt,
        development_url,
    ):
        store = CodexSessionStore(directory / "session")
        identity = directory / "session/identity.json"
        if identity.exists():
            store.bind(json.loads(identity.read_text())["session_id"])
        tools = RemoteWorkspaceTools(
            workspace, directory, readable, writable, development_url
        )
        request_id = str(directory.resolve())
        if request_id in self.active:
            raise ValueError("Remote author request already active")
        self.active[request_id] = (tools, store, directory, record)
        try:
            begin_query()
            result = await self._query_native(
                {
                    "role": self.role,
                    "session_key": request_id,
                    "request_id": request_id,
                    "workspace": str(workspace),
                    "session_id": store.session_id,
                    "policy": self.policy.model_dump(mode="json"),
                    "instructions": instructions,
                    "prompt": native_prompt,
                },
            )
            if result["session_id"] != store.session_id:
                raise ValueError("Local native completion belongs to another session")
            usage = Usage(
                physical_steps=0,
                model_calls=None,
                wall_time_complete=False,
                wall_time_s=time.monotonic() - started,
                cost_usd=None,
            )
            atomic_json(query_path, usage.model_dump(mode="json"))
            usage = cumulative_usage(directory / "audit")
            self._update_usage(directory)
            await run_blocking(
                complete_query, directory, usage, store.session_id, store
            )
            return usage, store.session_id
        finally:
            del self.active[request_id]


class RemoteCodexAuthor(RemoteCodexSession, NativeBundleAuthor):
    """Remote complete Harness job using the common Codex transport."""

    role = "author"


class RemoteGraphAnalysis(RemoteCodexSession, NativeGraphAnalysis):
    """Run-wide remote graph analysis session using the same Codex transport."""

    role = "graph_analysis"
