# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Fail-closed OS sandbox for coding-agent-authored Python packages."""

from __future__ import annotations

import asyncio
import ctypes.util
import inspect
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
from reweave.runtime.isolation import restrict_program_reads
from reweave.runtime.model_budget import ModelBudgetExceeded, ModelServiceFailure
from reweave.runtime.source import validate_program_source

RuntimeInvoker = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]
SyncRuntimeInvoker = Callable[[str, str, dict[str, Any]], dict[str, Any]]

_LINUX_DENIED_SYSCALLS = (
    "openat2",
    "creat",
    "link",
    "linkat",
    "symlink",
    "symlinkat",
    "unlink",
    "unlinkat",
    "rename",
    "renameat",
    "renameat2",
    "mkdir",
    "mkdirat",
    "rmdir",
    "mknod",
    "mknodat",
    "truncate",
    "ftruncate",
    "chmod",
    "fchmod",
    "fchmodat",
    "chown",
    "fchown",
    "fchownat",
    "lchown",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "open_by_handle_at",
    "name_to_handle_at",
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "clone",
    "clone3",
    "fork",
    "vfork",
    "execve",
    "execveat",
    "unshare",
    "setns",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_open",
    "pidfd_getfd",
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "io_uring_setup",
    "kexec_load",
    "finit_module",
    "init_module",
    "delete_module",
)

_WORKER = f"""
import asyncio
import ctypes
import errno
import importlib
import inspect
import json
import os
import resource
import sys

DENIED_SYSCALLS = {_LINUX_DENIED_SYSCALLS!r}

{inspect.getsource(restrict_program_reads)}

def json_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, dict):
        return {{str(key): json_value(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(type(value).__name__ + " is not JSON serializable")

def send(payload):
    sys.stdout.write(json.dumps(json_value(payload), separators=(",", ":")) + "\\n")
    sys.stdout.flush()

class Runtime:
    def __init__(self):
        self.request_index = 0
        self.async_lock = asyncio.Lock()
        self.input_buffer = bytearray()

    def _invoke(self, operation, name, arguments):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._blocking_invoke(operation, name, arguments)
        return self._async_invoke(operation, name, arguments)

    def _request(self, operation, name, arguments):
        if operation not in {{"observe", "act", "control"}}:
            raise RuntimeError("unknown trusted runtime operation")
        if not isinstance(name, str) or not name:
            raise RuntimeError("runtime operation name must be non-empty")
        if arguments is None:
            arguments = {{}}
        if not isinstance(arguments, dict):
            raise RuntimeError("runtime operation arguments must be an object")
        request_id = self.request_index
        self.request_index += 1
        send({{"kind": "runtime_call", "request_id": request_id,
              "operation": operation, "name": name, "arguments": arguments}})
        return request_id

    def _result(self, request_id, line):
        if not line:
            raise RuntimeError("trusted host closed the runtime channel")
        response = json.loads(line)
        if response.get("request_id") != request_id:
            raise RuntimeError("runtime response id mismatch")
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "runtime call rejected")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("runtime result must be an object")
        return result

    def _blocking_invoke(self, operation, name, arguments):
        request_id = self._request(operation, name, arguments)
        return self._result(request_id, sys.stdin.readline())

    async def _async_invoke(self, operation, name, arguments):
        # One ordered RPC stream per conversation. Other local tasks can progress
        # while the host works; independent effect streams use start/join.
        async with self.async_lock:
            request_id = self._request(operation, name, arguments)
            loop = asyncio.get_running_loop()
            ready = loop.create_future()

            def readable():
                try:
                    chunk = os.read(0, 65536)
                    if not chunk:
                        raise RuntimeError("trusted host closed the runtime channel")
                    self.input_buffer.extend(chunk)
                    if b"\\n" in self.input_buffer:
                        line, _, rest = self.input_buffer.partition(b"\\n")
                        self.input_buffer = bytearray(rest)
                        ready.set_result(line)
                        loop.remove_reader(0)
                except BaseException as error:
                    loop.remove_reader(0)
                    if not ready.done():
                        ready.set_exception(error)

            loop.add_reader(0, readable)
            # Cancellation cannot abandon a submitted RPC or leave its response
            # to be consumed by the next call on this stream.
            cancelled = False
            try:
                while not ready.done():
                    try:
                        await asyncio.shield(ready)
                    except asyncio.CancelledError:
                        cancelled = True
                result = self._result(request_id, ready.result())
            finally:
                loop.remove_reader(0)
            if cancelled:
                raise asyncio.CancelledError
            return result

    def observe(self, name, arguments=None):
        return self._invoke("observe", name, arguments)

    def act(self, name, arguments=None):
        return self._invoke("act", name, arguments)

    def control(self, name, arguments=None):
        return self._invoke("control", name, arguments)

    def snapshot(self):
        return self._invoke("control", "episode:snapshot", {{}})

    def plan(self, objective=None):
        return self._invoke("control", "model:plan", {{"objective": objective}})

    def execute(self, call):
        return self._invoke("control", "tool:execute", {{"call": call}})

    def feedback(self, call_id, result):
        return self._invoke("control", "tool:feedback", {{"call_id": call_id, "result": result}})

    def start(self, entrypoint, payload, permissions=None):
        return self._invoke("control", "task:start", {{"entrypoint": entrypoint, "payload": payload, "permissions": permissions}})

    def join(self, handle):
        return self._invoke("control", "task:join", {{"handle": handle}})

    def cancel(self, handle=None):
        return self._invoke("control", "task:cancel", {{"handle": handle}})

class PipeEventLoop(asyncio.SelectorEventLoop):
    # A local pipe supplies asyncio's wakeup FD without granting network access.
    def _make_self_pipe(self):
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._internal_fds += 1
        self._add_reader(self._wake_read, self._read_from_self)

    def _read_from_self(self):
        try:
            while os.read(self._wake_read, 4096):
                pass
        except BlockingIOError:
            pass

    def _write_to_self(self):
        try:
            os.write(self._wake_write, b"x")
        except (BlockingIOError, OSError):
            pass

    def _close_self_pipe(self):
        self._remove_reader(self._wake_read)
        os.close(self._wake_read)
        os.close(self._wake_write)
        self._internal_fds -= 1

class ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]

def apply_linux_seccomp():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("failed to set no_new_privs")
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ScmpArgCmp),
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError("failed to initialize seccomp")
    deny = 0x00050000 | errno.EPERM
    try:
        for name in DENIED_SYSCALLS:
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                raise RuntimeError("libseccomp cannot resolve syscall " + name)
            if seccomp.seccomp_rule_add(context, deny, number, 0) != 0:
                raise RuntimeError("failed to deny syscall " + name)
        # Imports need read-only open/openat. Deny both writable access modes.
        for name, flags_arg in (("open", 1), ("openat", 2)):
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                raise RuntimeError("libseccomp cannot resolve syscall " + name)
            for write_mode in (1, 2):
                comparison = ScmpArgCmp(flags_arg, 7, 3, write_mode)
                if seccomp.seccomp_rule_add_array(
                    context, deny, number, 1, ctypes.byref(comparison)
                ) != 0:
                    raise RuntimeError("failed to make " + name + " read-only")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("failed to install seccomp policy")
    finally:
        seccomp.seccomp_release(context)

def main():
    request = json.loads(sys.stdin.readline())
    package_root = request["package_root"]
    sys.dont_write_bytecode = True
    sys.path.insert(0, package_root)
    sys.path.extend(request.get("dependency_roots", []))
    if request["worker_isolation"] == "linux-seccomp":
        apply_linux_seccomp()
        if request.get("readable_roots"):
            restrict_program_reads(request["readable_roots"])
    module_name, function_name = request["entrypoint"].split(":", 1)
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, function_name)
    if not callable(entrypoint):
        raise RuntimeError("generated entrypoint is not callable")
    inspect.signature(entrypoint).bind(None, {{}})
    if request.get("validate_only"):
        send({{"kind": "validation_result", "result": {{"imported": True}}}})
        return
    result = entrypoint(Runtime(), request["program_input"])
    if inspect.isawaitable(result):
        loop = PipeEventLoop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(result)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    if not isinstance(result, dict):
        raise RuntimeError("generated entrypoint must return an object")
    send({{"kind": "program_result", "result": result}})

try:
    main()
except BaseException as exc:
    send({{"kind": "program_error", "error": type(exc).__name__ + ": " + str(exc)}})
"""

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
