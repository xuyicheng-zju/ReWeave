# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Sandboxed generated program execution and host IPC."""

import asyncio
import ctypes
import errno
import importlib
import inspect
import json
import os
import resource
import sys

DENIED_SYSCALLS = (
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


def restrict_program_reads(readable):
    """Apply Linux Landlock before importing any candidate-controlled module."""
    import ctypes
    import os

    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("generated programs require Linux Landlock read isolation")
    rights = (1 << 0) | (1 << 2) | (1 << 3)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ruleset = Ruleset(rights)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for name in readable:
            descriptor = os.open(name, os.O_PATH | os.O_CLOEXEC)
            try:
                access = rights if os.path.isdir(name) else rights & ~(1 << 3)
                rule = PathRule(access, descriptor)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(descriptor)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl")
        if libc.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)


def json_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(type(value).__name__ + " is not JSON serializable")


def send(payload):
    sys.stdout.write(json.dumps(json_value(payload), separators=(",", ":")) + "\n")
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
        if operation not in {"observe", "act", "control"}:
            raise RuntimeError("unknown trusted runtime operation")
        if not isinstance(name, str) or not name:
            raise RuntimeError("runtime operation name must be non-empty")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise RuntimeError("runtime operation arguments must be an object")
        request_id = self.request_index
        self.request_index += 1
        send(
            {
                "kind": "runtime_call",
                "request_id": request_id,
                "operation": operation,
                "name": name,
                "arguments": arguments,
            }
        )
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
                    if b"\n" in self.input_buffer:
                        line, _, rest = self.input_buffer.partition(b"\n")
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
        return self._invoke("control", "episode:snapshot", {})

    def plan(self, objective=None):
        return self._invoke("control", "model:plan", {"objective": objective})

    def execute(self, call):
        return self._invoke("control", "tool:execute", {"call": call})

    def feedback(self, call_id, result):
        return self._invoke(
            "control", "tool:feedback", {"call_id": call_id, "result": result}
        )

    def start(self, entrypoint, payload, permissions=None):
        return self._invoke(
            "control",
            "task:start",
            {"entrypoint": entrypoint, "payload": payload, "permissions": permissions},
        )

    def join(self, handle):
        return self._invoke("control", "task:join", {"handle": handle})

    def cancel(self, handle=None):
        return self._invoke("control", "task:cancel", {"handle": handle})


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
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ScmpArgCmp),
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
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
                if (
                    seccomp.seccomp_rule_add_array(
                        context, deny, number, 1, ctypes.byref(comparison)
                    )
                    != 0
                ):
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
    inspect.signature(entrypoint).bind(None, {})
    if request.get("validate_only"):
        send({"kind": "validation_result", "result": {"imported": True}})
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
    send({"kind": "program_result", "result": result})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        send({"kind": "program_error", "error": type(exc).__name__ + ": " + str(exc)})
