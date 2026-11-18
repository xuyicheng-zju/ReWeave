"""Native Cursor headless coding sessions for offline analysis and authoring."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from pwd import getpwuid

from reweave import __version__
from reweave.evolution.authoring.workspace import native_author_launcher, native_output
from reweave.evolution.contracts import Usage
from reweave.evolution.infrastructure.native_accounting import (
    cumulative_usage,
    native_completion,
)
from reweave.evolution.infrastructure.native_retry import RetryableNativeProviderError
from reweave.evolution.infrastructure.processes import drain_process_group
from reweave.planner.cursor_connection import (
    cursor_environment,
    resolve_cursor_executable,
)
from reweave.utils.persistence import atomic_json
from reweave.utils.provider_admission import ProviderAdmission

CURSOR_PERSISTENT_ROOT = Path("/tmp") / f"cursor-agent-persist-{os.getuid()}"


def cursor_workspace(workspace, directory, readable, writable):
    """Keep the existing login in place and isolate all new native chat state."""
    origin = dict(os.environ)
    origin["HOME"] = os.environ.get(
        "REWEAVE_CURSOR_ACCOUNT_HOME", getpwuid(os.getuid()).pw_dir
    )
    environment = cursor_environment(origin)
    proxy = os.environ.get("REWEAVE_CURSOR_PROXY")
    if proxy:
        environment.update(
            dict.fromkeys(
                (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "ALL_PROXY",
                    "all_proxy",
                ),
                proxy,
            )
        )
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        environment.pop(key, None)
    executable = resolve_cursor_executable(
        os.environ.get("REWEAVE_CURSOR_BIN", "agent"), environment
    )
    private = Path(environment["CURSOR_CONFIG_DIR"])
    session = directory / "session"
    session.mkdir(exist_ok=True)
    runtime_binding = session / "cursor_runtime.json"
    if runtime_binding.exists():
        runtime_root = Path(json.loads(runtime_binding.read_text())["root"])
    else:
        # Cursor falls back to shared /tmp/.cursor when its data path exceeds
        # its Unix-socket length bound. Allocate a short, private job directory.
        runtime_base = Path(
            os.environ.get(
                "REWEAVE_CURSOR_RUNTIME_ROOT",
                str(Path(origin["HOME"]) / ".local/state/reweave/cursor"),
            )
        )
        runtime_base.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime_root = Path(tempfile.mkdtemp(prefix="job-", dir=runtime_base))
        atomic_json(runtime_binding, {"root": str(runtime_root)})
    config = workspace / ".cursor"
    config.mkdir(exist_ok=True)
    atomic_json(
        config / "cli.json",
        {
            "permissions": {
                "allow": [
                    "Read(**)",
                    "Shell(*)",
                    *[f"Write({p}/**)" for p in (*writable, workspace / ".scratch")],
                ],
                "deny": [f"Read({private}/**)", f"Write({config}/**)"],
            }
        },
    )
    # The native engine needs its installed code and existing account files.
    # Neither the home root nor another account's profile is granted.
    account_files = (
        (private / "auth.json",) if (private / "auth.json").is_file() else ()
    )
    persistent = CURSOR_PERSISTENT_ROOT
    persistent.mkdir(mode=0o700, exist_ok=True)
    claims = persistent / "claim-locks"
    claims.mkdir(mode=0o700, exist_ok=True)
    (persistent / "bindings").mkdir(mode=0o700, exist_ok=True)
    launcher, runtime = native_author_launcher(
        workspace,
        native_output(workspace),
        directory / "launcher",
        backend="cursor",
        cli=executable,
        readable_roots=(*readable, Path(executable).parent, persistent, *account_files),
        writable_roots=writable,
        native_runtime_root=runtime_root,
        native_state_roots=(claims,),
    )
    environment.update(runtime)
    # Chat databases/config use CURSOR_CONFIG_DIR, while Linux login uses
    # XDG_CONFIG_HOME/cursor/auth.json. Keep those locations separate.
    environment["CURSOR_CONFIG_DIR"] = runtime["CURSOR_DATA_DIR"]
    environment["HOME"] = runtime["CURSOR_DATA_DIR"]
    environment["NODE_COMPILE_CACHE"] = str(workspace / ".scratch/runtime/node-cache")
    environment["XDG_CACHE_HOME"] = str(workspace / ".scratch/runtime/cache")
    environment["NODE_USE_ENV_PROXY"] = "1"
    bypass = dict.fromkeys(
        host.strip()
        for value in (
            environment.get("NO_PROXY", ""),
            environment.get("no_proxy", ""),
            "127.0.0.1,localhost,::1",
        )
        for host in value.split(",")
        if host.strip()
    )
    environment["NO_PROXY"] = environment["no_proxy"] = ",".join(bypass)
    # Native startup must not fetch optional plugin repositories. This job
    # edits source files directly and the checkout contract uses Dulwich.
    tools = directory / "launcher/bin"
    tools.mkdir(exist_ok=True)
    git = tools / "git"
    git.write_text("#!/bin/sh\nexit 1\n")
    git.chmod(0o700)
    environment["PATH"] = str(tools) + os.pathsep + environment.get("PATH", os.defpath)
    return launcher, environment


def cursor_event(event, identity):
    """Reject mixed-session streams; completion must be an explicit success."""
    session = event.get("session_id")
    if session is not None and session != identity:
        raise ValueError("Cursor event belongs to another native session")
    return (
        event.get("type") == "result"
        and not event.get("is_error", False)
        and event.get("subtype") == "success"
    )


async def _run_cli(command, workspace, environment, stderr, consume):
    with stderr.open("ab") as errors:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=errors,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )
        try:
            async for line in process.stdout:
                consume(line)
            return await process.wait()
        finally:
            await drain_process_group(process, terminate=process.returncode is None)


async def _select_long_context(launcher, workspace, environment, directory):
    """Bind the CLI's native chat through its typed model configuration API."""
    binding = directory / "session/model_selection.json"
    wanted = {
        "model": "grok-4.7",
        "context": "500k",
        "reasoning_effort": "xhigh",
        "fast": "false",
    }
    if binding.exists():
        saved = json.loads(binding.read_text())
        identity = json.loads((directory / "session/identity.json").read_text())
        if (
            saved["selection"] != wanted
            or saved["session_id"] != identity["session_id"]
        ):
            raise ValueError("Cursor native model selection changed")
        return saved["session_id"]
    pending = directory / "session/model_creation_pending.json"
    if pending.exists():
        raise ValueError("Cursor native chat creation requires reconciliation")
    with (directory / "audit/cursor_model_stderr.log").open("ab") as errors:
        process = await asyncio.create_subprocess_exec(
            launcher,
            "acp",
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=errors,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )
        sequence = 0

        async def rpc(method, params):
            nonlocal sequence
            sequence += 1
            identity = sequence
            process.stdin.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": identity,
                            "method": method,
                            "params": params,
                        }
                    )
                    + "\n"
                ).encode()
            )
            await process.stdin.drain()
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=60)
                if not line:
                    raise RuntimeError("Cursor model configuration process exited")
                response = json.loads(line)
                if response.get("id") == identity:
                    if "error" in response:
                        raise RuntimeError(
                            f"Cursor model configuration failed: {response['error']}"
                        )
                    return response["result"]

        try:
            await rpc(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientInfo": {"name": "reweave", "version": __version__},
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                        "terminal": False,
                        "_meta": {"parameterizedModelPicker": True},
                    },
                },
            )
            await rpc("authenticate", {"methodId": "cursor_login"})
            identity_path = directory / "session/identity.json"
            if identity_path.exists():
                identity = json.loads(identity_path.read_text())["session_id"]
                opened = await rpc(
                    "session/load",
                    {
                        "sessionId": identity,
                        "cwd": str(workspace),
                        "mcpServers": [],
                    },
                )
            else:
                atomic_json(pending, {"phase": "creating", "generative_queries": 0})
                opened = await rpc(
                    "session/new", {"cwd": str(workspace), "mcpServers": []}
                )
                identity = str(uuid.UUID(opened["sessionId"]))
                atomic_json(
                    identity_path, {"session_id": identity, "backend": "cursor"}
                )
                pending.unlink()
            options = opened["configOptions"]
            for key, value in wanted.items():
                current = {item["id"]: item for item in options}
                if current[key]["currentValue"] != value:
                    changed = await rpc(
                        "session/set_config_option",
                        {
                            "sessionId": identity,
                            "configId": key,
                            "value": value,
                        },
                    )
                    options = changed["configOptions"]
                if {item["id"]: item["currentValue"] for item in options}[key] != value:
                    raise ValueError(f"Cursor did not confirm {key}={value}")
            atomic_json(binding, {"session_id": identity, "selection": wanted})
            return identity
        finally:
            await drain_process_group(process, terminate=True)


async def cursor_query(
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
    prompt,
    tools_url,
):
    """One native CLI query, using Cursor's own coding tools and resumable chat."""
    launcher, environment = cursor_workspace(workspace, directory, readable, writable)
    environment["REWEAVE_AUTHOR_TOOLS_URL"] = tools_url
    atomic_json(
        workspace / ".cursor/mcp.json", {"mcpServers": {"author": {"url": tools_url}}}
    )
    session = directory / "session"
    session.mkdir(exist_ok=True)
    identity_path = session / "identity.json"
    stderr = directory / "audit/cursor_stderr.log"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())["session_id"]
    else:
        if any((directory / "audit/queries").glob("*.json")):
            raise ValueError("Cursor query evidence exists without a bound native chat")
        identity = None
    long_context = self.policy.model == "grok-4.7[context=500k,effort=xhigh,fast=false]"
    if long_context:
        identity = await _select_long_context(
            launcher, workspace, environment, directory
        )
    command = [
        launcher,
        "--print",
        "--output-format",
        "stream-json",
        *([] if long_context else ["--model", self.policy.model]),
        "--workspace",
        str(workspace),
        "--trust",
        "--force",
        "--sandbox",
        "disabled",
        "--disable-indexing",
        "--disable-codebase-ref",
        "--approve-mcps",
        *(["--resume", identity] if identity else []),
        instructions + "\n\n" + prompt,
    ]
    completed = False

    def consume(line):
        nonlocal completed, identity
        event = json.loads(line)
        record(event)
        if (
            event.get("type") == "system"
            and event.get("subtype") == "init"
            and identity is None
        ):
            identity = str(uuid.UUID(event["session_id"]))
            atomic_json(
                identity_path,
                {
                    "session_id": identity,
                    "backend": "cursor",
                    "model": self.policy.model,
                },
            )
        if identity is None:
            return
        completed |= cursor_event(event, identity)

    admission = ProviderAdmission("cursor-login", lane="offline")
    async with admission.lease(directory / "audit/provider_admission.json"):
        begin_query()
        stderr_offset = stderr.stat().st_size if stderr.exists() else 0
        status = await _run_cli(command, workspace, environment, stderr, consume)
    usage = Usage(
        physical_steps=0,
        model_calls=None,
        wall_time_s=time.monotonic() - started,
        wall_time_complete=False,
        cost_usd=None,
    )
    atomic_json(query_path, usage.model_dump(mode="json"))
    if status or not completed:
        if stderr.exists():
            with stderr.open("rb") as errors:
                errors.seek(stderr_offset)
                current_error = errors.read().decode("utf-8", errors="replace")
            if (
                "RetriableError: Agent turn stopped after repeated resume attempts made no progress"
                in current_error
            ):
                raise RetryableNativeProviderError(
                    "Cursor declared a retriable turn failure after repeated resume attempts"
                )
        raise RuntimeError(
            "Cursor coding query failed or ended without a successful result; inspect native events and cursor_stderr.log"
        )
    usage = cumulative_usage(directory / "audit")
    native_completion(
        directory,
        usage,
        identity,
        {
            "backend": "cursor",
            "session_id": identity,
            "data_directory": environment["CURSOR_DATA_DIR"],
            "events": str(directory / "audit/events.jsonl"),
        },
    )
    return usage, identity
