# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Fail-closed OS sandbox for coding-agent-authored Python packages."""

from __future__ import annotations

import asyncio
import ctypes.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from reweave.evolution.bundle import ProgramLimits, PythonProgram
from reweave.runtime.capability import PhysicalOutcomeUnknown, ProgramDeadlineExceeded
from reweave.runtime.model_budget import ModelBudgetExceeded, ModelServiceFailure
from reweave.runtime.source import validate_program_source

RuntimeInvoker = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]
SyncRuntimeInvoker = Callable[[str, str, dict[str, Any]], dict[str, Any]]

_WORKER = Path(__file__).with_name("sandbox_worker.py").read_text(encoding="utf-8")

_DARWIN_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(deny process-fork)
"""

_WORKER_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "PYTHONHASHSEED": "0",
    # Native numerical libraries otherwise attempt thread creation during
    # import, which the fail-closed Linux seccomp profile correctly denies.
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
}

# Runtime observations can legally carry up to eight 10 MiB image envelopes
# into a generated multimodal feedback loop.  asyncio's default StreamReader
# limit is only 64 KiB, which rejects those newline-delimited JSON messages
# before the trusted host can validate or route them.
_IPC_LINE_LIMIT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SandboxBackend:
    """Concrete OS isolation command selected for the current host."""

    name: str
    command_prefix: tuple[str, ...]
    worker_isolation: str


def sandbox_backend() -> SandboxBackend:
    """Return an OS sandbox or fail instead of executing generated code openly."""
    system = platform.system()
    if system == "Darwin":
        executable = shutil.which("sandbox-exec")
        if executable is None:
            raise RuntimeError("generated code requires macOS sandbox-exec")
        return SandboxBackend(
            name="sandbox-exec",
            command_prefix=(executable, "-p", _DARWIN_PROFILE),
            worker_isolation="darwin-sandbox",
        )
    if system == "Linux":
        if ctypes.util.find_library("seccomp") is None:
            raise RuntimeError("generated code requires libseccomp on Linux")
        return SandboxBackend(
            name="linux-seccomp",
            command_prefix=(),
            worker_isolation="linux-seccomp",
        )
    raise RuntimeError(f"generated code has no sandbox backend on {system}")


class ProgramSandbox:
    """Import and execute a real generated package through a protected runtime."""

    def __init__(self, backend: SandboxBackend | None = None) -> None:
        self._backend = backend or sandbox_backend()

    def _materialize_program(self, program: PythonProgram, root: Path) -> Path:
        validate_program_source(program)
        source_root = root / "src"
        source_root.mkdir()
        for item in program.files:
            target = source_root / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.source, encoding="utf-8")
            target.chmod(0o444)
        for directory in sorted(
            (p for p in source_root.rglob("*") if p.is_dir()), reverse=True
        ):
            directory.chmod(0o555)
        source_root.chmod(0o555)
        return source_root.resolve()

    def _program_read_roots(self, source_root: Path) -> list[str]:
        candidates = [
            source_root,
            Path(sys.executable).resolve(),
            Path(sys.base_prefix).resolve(),
            *(
                Path(sysconfig.get_path(key))
                for key in ("stdlib", "platstdlib", "purelib", "platlib")
            ),
            *(
                Path(path)
                for path in (
                    "/usr/lib",
                    "/lib",
                    "/lib64",
                    "/System/Library",
                    "/System/Volumes/Preboot/Cryptexes/OS/System/Library",
                    "/private/var/db/dyld",
                    "/dev/null",
                    "/dev/urandom",
                    "/etc/ld.so.cache",
                )
            ),
        ]
        return sorted(
            {str(path.resolve()) for path in candidates if path.exists()}
            | {
                str(path)
                for path in (Path(sys.executable), Path(sys.prefix) / "pyvenv.cfg")
                if path.exists()
            }
        )

    def _program_command(self, readable: list[str]) -> list[str]:
        if self._backend.worker_isolation == "darwin-sandbox":
            profile = (
                _DARWIN_PROFILE
                + '\n(deny file-read*)\n(allow file-read* (literal "/"))\n'
            )
            profile += "\n".join(
                f"(allow file-read* (subpath {json.dumps(path)}))" for path in readable
            )
            parents = {
                str(parent) for name in readable for parent in Path(name).parents
            }
            profile += "\n" + "\n".join(
                f"(allow file-read-metadata (literal {json.dumps(path)}))"
                for path in sorted(parents)
            )
            return [
                self._backend.command_prefix[0],
                "-p",
                profile,
                str(Path(sys.executable).resolve()),
                "-I",
                "-c",
                _WORKER,
            ]
        return [*self._backend.command_prefix, sys.executable, "-I", "-c", _WORKER]

    def validate_program_import(
        self, program: PythonProgram, *, wall_time_s: float
    ) -> None:
        """Import the candidate and bind its callable in the same OS read boundary."""
        with tempfile.TemporaryDirectory(prefix="harness-program-import-") as temporary:
            source_root = self._materialize_program(program, Path(temporary))
            readable = self._program_read_roots(source_root)
            request = {
                "package_root": str(source_root),
                "readable_roots": readable,
                "dependency_roots": [
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                ],
                "entrypoint": program.entrypoint,
                "worker_isolation": self._backend.worker_isolation,
                "validate_only": True,
                "program_input": {},
            }
            completed = subprocess.run(
                self._program_command(readable),
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
                timeout=wall_time_s,
                env=_WORKER_ENV,
                check=False,
            )
            lines = completed.stdout.splitlines()
            if completed.returncode != 0 or not lines:
                raise ValueError(f"candidate import failed: {completed.stderr}")
            response = json.loads(lines[-1])
            if response.get("kind") != "validation_result":
                raise ValueError(f"candidate callable validation failed: {response}")

    async def run_program(
        self,
        program: PythonProgram,
        limits: ProgramLimits,
        program_input: dict[str, Any],
        invoke: RuntimeInvoker,
        *,
        input_schema: dict[str, Any],
        exclude_runtime_time: bool = False,
    ) -> dict[str, Any]:
        """Run generated package until result, rejection, or hard deadline."""
        Draft202012Validator(input_schema).validate(program_input)
        with tempfile.TemporaryDirectory(prefix="harness-program-") as temporary:
            source_root = self._materialize_program(program, Path(temporary))
            readable = self._program_read_roots(source_root)
            process = await asyncio.create_subprocess_exec(
                *self._program_command(readable),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_IPC_LINE_LIMIT_BYTES,
                env=_WORKER_ENV,
            )
            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                process.kill()
                raise RuntimeError("sandbox process did not expose IPC streams")
            request = {
                "package_root": str(source_root),
                "readable_roots": readable,
                "dependency_roots": [
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                ],
                "entrypoint": program.entrypoint,
                "program_input": program_input,
                "worker_isolation": self._backend.worker_isolation,
            }

            stderr_capture = tempfile.SpooledTemporaryFile(max_size=64 * 1024)

            async def capture_stderr():
                while chunk := await process.stderr.read(64 * 1024):
                    stderr_capture.write(chunk)

            stderr_task = asyncio.create_task(capture_stderr())

            async def communicate() -> dict[str, Any]:
                process.stdin.write(
                    json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
                )
                await process.stdin.drain()
                calls = 0
                remaining = limits.wall_time_s
                while True:
                    started = time.monotonic()
                    if exclude_runtime_time:
                        # Bound a nonresponsive worker independently of provider
                        # and robot latency while the Host handles its RPC.
                        timeout = (
                            min(60.0, remaining) if remaining is not None else 60.0
                        )
                        try:
                            line = await asyncio.wait_for(
                                process.stdout.readline(), timeout
                            )
                        except TimeoutError as error:
                            raise ProgramDeadlineExceeded(
                                "generated worker exhausted its local execution allowance without a runtime response pending",
                                scope="program",
                            ) from error
                        if remaining is not None:
                            remaining -= time.monotonic() - started
                    else:
                        line = await process.stdout.readline()
                    if not line:
                        await asyncio.shield(stderr_task)
                        stderr_capture.seek(0)
                        stderr = stderr_capture.read().decode("utf-8", errors="replace")
                        raise RuntimeError(
                            "sandboxed program exited without a result"
                            + (f": {stderr.strip()}" if stderr.strip() else "")
                        )
                    message = json.loads(line)
                    kind = message.get("kind")
                    if kind == "program_error":
                        raise RuntimeError(
                            f"program {program.entrypoint!r} failed: {message.get('error')}"
                        )
                    if kind == "program_result":
                        result = message.get("result")
                        if not isinstance(result, dict):
                            raise RuntimeError("generated result must be an object")
                        return result
                    if kind != "runtime_call":
                        raise RuntimeError(
                            f"sandbox emitted unknown IPC kind: {kind!r}"
                        )
                    calls += 1
                    if (
                        limits.runtime_calls is not None
                        and calls > limits.runtime_calls
                    ):
                        raise RuntimeError("program exhausted its runtime-call budget")
                    request_id = message.get("request_id")
                    try:
                        result = await invoke(
                            str(message.get("operation")),
                            str(message.get("name")),
                            message.get("arguments"),
                        )
                    except (
                        PhysicalOutcomeUnknown,
                        ProgramDeadlineExceeded,
                        ModelServiceFailure,
                        ModelBudgetExceeded,
                    ):
                        raise
                    except Exception as exc:
                        response = {
                            "request_id": request_id,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    else:
                        response = {
                            "request_id": request_id,
                            "ok": True,
                            "result": result,
                        }
                    process.stdin.write(
                        json.dumps(response, separators=(",", ":")).encode("utf-8")
                        + b"\n"
                    )
                    await process.stdin.drain()

            failure = None
            try:
                return await asyncio.wait_for(
                    communicate(),
                    timeout=None if exclude_runtime_time else limits.wall_time_s,
                )
            except BaseException as error:
                failure = error
                raise
            finally:
                if process.returncode is None:
                    process.kill()

                async def drain(stream):
                    while await stream.read(64 * 1024):
                        pass

                # wait() also waits for pipe EOF. A full unread pipe pauses its
                # transport even after kill(), so drain both streams concurrently.
                cleanup = asyncio.gather(
                    drain(process.stdout),
                    stderr_task,
                    process.wait(),
                    return_exceptions=True,
                )
                interrupted = False
                try:
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            interrupted = True
                    for outcome in cleanup.result():
                        if isinstance(outcome, BaseException):
                            raise outcome
                except BaseException as cleanup_error:
                    if failure is not None:
                        if failure is cleanup_error:
                            raise
                        failure.add_note(
                            f"Sandbox cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}",
                        )
                        raise failure from cleanup_error
                    raise
                finally:
                    stderr_capture.close()
                if interrupted:
                    if failure is None:
                        raise asyncio.CancelledError
                    failure.add_note(
                        "Cancellation waited for sandbox pipe drainage"
                    )
