# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Run remote GraphEvolution with a local Codex author over SSH."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shlex
import sys
import uuid
from pathlib import Path

from reweave.evolution.infrastructure.ssh_channel import SshChannel
from reweave.utils.persistence import atomic_json


def bind_deployment(workspace: Path, client_id: str) -> None:
    """Record the local account deployment used for this invocation."""
    uuid.UUID(client_id)
    path = workspace / "author_deployment.json"
    deployment = {
        "transport": "ssh",
        "authentication": "local_chatgpt",
        "client_id": client_id,
    }
    atomic_json(path, deployment)


async def remote_host(args, reader, writer):
    from reweave.cli.evolve import EvolutionConfiguration
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.authoring.remote import (
        RemoteCodexAuthor,
        RemoteGraphAnalysis,
    )
    from reweave.evolution.bundle import AgentBundle
    from reweave.evolution.contracts import EvolutionProtocol
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
    from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator

    author = None
    analyzer = None
    run_started = False

    async def handle(method, params):
        nonlocal author, analyzer, run_started
        if method != "run":
            if author is None:
                raise ValueError("Evolution has not started")
            adapter = (
                analyzer
                if analyzer is not None and params.get("request_id") in analyzer.active
                else author
            )
            return await adapter.handle(method, params)
        if run_started:
            raise ValueError("One SSH connection owns exactly one evolution run")
        if set(params) != {"client_id"}:
            raise ValueError("Remote evolution uses local Codex account login only")
        run_started = True
        root = args.workspace.resolve()
        config = EvolutionConfiguration.model_validate_json(
            (root / "configuration.json").read_text()
        )
        policy = AuthorPolicy.model_validate(config.author)
        if policy.author != "codex":
            raise ValueError(
                "Remote account deployment requires configuration author=codex; existing Cursor runs cannot be switched"
            )
        workspace = EvolutionWorkspace(
            root,
            AgentBundle.model_validate(config.baseline),
            EvolutionProtocol.model_validate(config.protocol),
        )
        with workspace.transaction():
            bind_deployment(root, params["client_id"])

        runner = RolloutConfiguration.model_validate(config.runner)
        author = RemoteCodexAuthor(
            channel=channel,
            bundle_loader=workspace.bundle,
            policy=policy,
            evidence_workspace=root / "evidence",
            runtime_manifest=runner.runtime_manifest,
        )
        analyzer = RemoteGraphAnalysis(
            channel=channel,
            bundle_loader=workspace.bundle,
            policy=policy,
            evidence_workspace=root / "evidence",
            native_runtime_root=root / "graph_session/runtime",
            runtime_manifest=runner.runtime_manifest,
        )
        runtime_source = args.runtime_source.resolve()
        if not runtime_source.is_dir():
            raise ValueError(
                f"Runtime source directory is unavailable: {runtime_source}"
            )
        evaluator = RolloutEvaluator(
            runner,
            repository=runtime_source,
            workspace=root,
        )
        evolution = GraphEvolution(
            workspace, author, evaluator, analyzer=analyzer, recover=args.recover
        )
        state = await evolution.run()
        return {
            "completed": state["completed"],
            "stop_reason": state["stop_reason"],
            "versions": len(state["versions"]),
            "usage": evolution.accounting(state),
        }

    channel = SshChannel(
        reader,
        writer,
        handle,
        write_chunk_bytes=args.ssh_write_chunk_bytes,
        write_interval_s=args.ssh_write_interval_s,
    )
    await channel.run()


def ssh_command(args) -> list[str]:
    """SSH host is a single destination, and remote argv is shell-quoted once."""
    if (
        not args.host
        or args.host.startswith("-")
        or any(c.isspace() for c in args.host)
    ):
        raise ValueError("SSH host must be one explicit host alias/destination")
    for value in (
        args.python,
        args.repository,
        str(args.runtime_source),
        str(args.workspace),
    ):
        if not value.startswith("/"):
            raise ValueError(
                "Remote Python, repository, runtime source and workspace must be absolute paths"
            )
    if args.port is not None and (
        type(args.port) is not int or not 1 <= args.port <= 65535
    ):
        raise ValueError("SSH port must be an integer from 1 through 65535")
    argv = [
        args.python,
        "-u",
        "-m",
        "reweave.cli.remote_evolution",
        "serve",
        "--workspace",
        str(args.workspace),
        "--runtime-source",
        str(args.runtime_source),
        "--ssh-write-chunk-bytes",
        str(args.ssh_write_chunk_bytes),
        "--ssh-write-interval-s",
        str(args.ssh_write_interval_s),
    ]
    if args.recover:
        argv.append("--recover")
    command = f"cd {shlex.quote(args.repository)} && "
    if args.environment:
        if not args.environment.startswith("/"):
            raise ValueError(
                "Remote environment file must be an absolute operator-owned path"
            )
        command += f"set -a && . {shlex.quote(args.environment)} >&2 && set +a && "
    command += "exec " + shlex.join(argv)
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if args.port is not None:
        ssh.extend(("-p", str(args.port)))
    return [*ssh, args.host, command]


async def local_host(args):
    from reweave.evolution.authoring.local import LocalCodexAuthor
    from reweave.evolution.infrastructure.processes import drain_process_group
    from reweave.planner.codex_connection import codex_executable

    cli = codex_executable(args.codex_bin)
    command = ssh_command(args)
    root = args.local_state.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binding = root / "connection.json"
    connection = {
        "host": args.host,
        "port": args.port,
        "repository": args.repository,
        "runtime_source": str(args.runtime_source),
        "python": args.python,
        "workspace": str(args.workspace),
        "environment": args.environment,
        "codex_bin": cli,
        "ssh_write_chunk_bytes": args.ssh_write_chunk_bytes,
        "ssh_write_interval_s": args.ssh_write_interval_s,
    }
    if binding.exists():
        stored = json.loads(binding.read_text())
        client_id = stored["client_id"]
    else:
        client_id = str(uuid.uuid4())
    atomic_json(binding, {"client_id": client_id, "connection": connection})
    with (root / "ssh.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        channel = SshChannel(
            process.stdout,
            process.stdin,
            None,
            write_chunk_bytes=args.ssh_write_chunk_bytes,
            write_interval_s=args.ssh_write_interval_s,
        )
        author = LocalCodexAuthor(channel, root / "authors", codex_bin=cli)
        channel.handler = author.handle
        pump = asyncio.create_task(channel.run())
        try:
            result = await channel.call("run", {"client_id": client_id})
            atomic_json(root / "result.json", result)
            print(json.dumps(result, indent=2))
        finally:
            process.stdin.close()
            await drain_process_group(process)
            await pump


async def serve(args, output):
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
    )
    protocol = asyncio.streams.FlowControlMixin(loop=loop)
    transport, _ = await loop.connect_write_pipe(lambda: protocol, output.buffer)
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    await remote_host(args, reader, writer)


def _write_chunk_bytes(value: str) -> int:
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("SSH write chunk must be a positive integer")
    return size


def _write_interval_s(value: str) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval < 0:
        raise argparse.ArgumentTypeError(
            "SSH write interval must be finite and nonnegative"
        )
    return interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Run from the Mac; files and evaluation stay remote"
    )
    run.add_argument("--host", required=True, help="Explicit SSH host alias")
    run.add_argument(
        "--port",
        type=int,
        help="Explicit SSH port; omit to preserve the selected host's SSH configuration",
    )
    run.add_argument("--python", required=True, help="Remote Python interpreter")
    run.add_argument("--repository", required=True, help="Remote source checkout")
    run.add_argument(
        "--environment",
        help="Remote operator-owned Planner/runtime environment script; never downloaded",
    )
    run.add_argument(
        "--local-state",
        type=Path,
        required=True,
        help="Durable local Codex session registry",
    )
    run.add_argument(
        "--codex-bin",
        help="Explicit local native Codex executable; default is CODEX_BIN or codex on PATH",
    )
    remote = commands.add_parser(
        "serve", help="Private SSH stdio endpoint, launched by run"
    )
    for command in (run, remote):
        command.add_argument(
            "--workspace",
            type=Path,
            required=True,
            help="Initialized remote evolution workspace",
        )
        command.add_argument(
            "--runtime-source",
            type=Path,
            required=True,
            help="Repository directory imported by physical rollout workers",
        )
        command.add_argument(
            "--ssh-write-chunk-bytes",
            type=_write_chunk_bytes,
            default=65536,
            help="Maximum bytes per SSH write; applied to both endpoints",
        )
        command.add_argument(
            "--ssh-write-interval-s",
            type=_write_interval_s,
            default=0.0,
            help="Seconds between SSH frame chunks; zero disables pacing",
        )
        command.add_argument(
            "--recover",
            action="store_true",
            help="Explicit recovery with original identities; never automatic replay",
        )
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(local_host(args))
    else:
        output = sys.stdout
        sys.stdout = sys.stderr
        asyncio.run(serve(args, output))


if __name__ == "__main__":
    main()
