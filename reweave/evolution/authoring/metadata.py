# Modified for the ReWeave namespace.
"""Service native timestamp and mode changes on pinned, workspace-owned files.

Landlock does not mediate these changes. A seccomp notification supervisor performs
the operation on a checked descriptor; it never continues an unchecked syscall.
The native CLI keeps its own filesystem tools and lock implementation.
"""

from __future__ import annotations

import array
import ctypes
import errno
import json
import os
import select
import signal
import socket
import stat
from pathlib import Path

TIMESTAMP_CALLS = ("utime", "utimes", "futimesat", "utimensat")
MODE_CALLS = ("chmod", "fchmod", "fchmodat", "fchmodat2")
METADATA_CALLS = (*TIMESTAMP_CALLS, *MODE_CALLS)


def write_process_receipt(path: Path, *, descendants_drained: bool) -> None:
    """Publish supervisor identity and whether all native writers have exited."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(
            {"pid": os.getpid(), "descendants_drained": descendants_drained}, stream
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class _Data(ctypes.Structure):
    _fields_ = [
        ("nr", ctypes.c_int32),
        ("arch", ctypes.c_uint32),
        ("instruction_pointer", ctypes.c_uint64),
        ("args", ctypes.c_uint64 * 6),
    ]


class _Request(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("pid", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", _Data),
    ]


class _Response(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("val", ctypes.c_int64),
        ("error", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]


def _memory(pid: int, address: int, length: int) -> bytes:
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as stream:
        value = os.pread(stream.fileno(), length, address)
    if len(value) != length:
        raise OSError(errno.EFAULT, "incomplete metadata syscall arguments")
    return value


def _pathname(pid: int, address: int) -> str:
    value = bytearray()
    # Linux PATH_MAX, not a model or evidence limit. Avoid crossing an unmapped
    # page while reading a short NUL-terminated pathname.
    while len(value) < 4096:
        current = address + len(value)
        size = min(256, 4096 - len(value), 4096 - current % 4096)
        block = _memory(pid, current, size)
        if b"\0" in block:
            return os.fsdecode(bytes(value) + block.split(b"\0", 1)[0])
        value.extend(block)
    raise OSError(errno.ENAMETOOLONG, "metadata pathname exceeds PATH_MAX")


def _target(
    pid: int, directory: int, path_address: int, flags: int
) -> tuple[Path, bool]:
    if flags & ~(0x100 | 0x1000):  # AT_SYMLINK_NOFOLLOW, AT_EMPTY_PATH
        raise OSError(errno.EINVAL, "unsupported metadata flags")
    path = _pathname(pid, path_address) if path_address else None
    if path is None or (path == "" and flags & 0x1000):
        if directory == -100:
            raise OSError(errno.EFAULT, "metadata operation requires a path or fd")
        target = Path(f"/proc/{pid}/fd/{directory}")
    elif path.startswith("/"):
        target = Path(f"/proc/{pid}/root") / path.lstrip("/")
    else:
        base = (
            f"/proc/{pid}/cwd" if directory == -100 else f"/proc/{pid}/fd/{directory}"
        )
        target = Path(base) / path
    return target, bool(flags & 0x100)


def _timestamp_arguments(request: _Request, name: str):
    args, pid = request.data.args, request.pid
    if name in {"utimensat", "futimesat"}:
        directory = ctypes.c_int(args[0]).value
        path_address, time_address = args[1], args[2]
    else:
        directory, path_address, time_address = -100, args[0], args[1]
    flags = args[3] if name == "utimensat" else 0
    target, nofollow = _target(pid, directory, path_address, flags)
    times = None
    if time_address:
        count = 2 if name == "utime" else 4
        values = (ctypes.c_long * count).from_buffer_copy(
            _memory(pid, time_address, ctypes.sizeof(ctypes.c_long) * count)
        )
        if name == "utime":
            pairs = ((values[0], 0), (values[1], 0))
        else:
            scale = 1 if name == "utimensat" else 1000
            pairs = ((values[0], values[1] * scale), (values[2], values[3] * scale))
        times = (_Timespec * 2)(*(_Timespec(*pair) for pair in pairs))
    return target, times, nofollow


def _mode_arguments(request: _Request, name: str):
    args = request.data.args
    if name == "chmod":
        directory, path_address, mode, flags = -100, args[0], args[1], 0
    elif name == "fchmod":
        directory, path_address, mode, flags = (
            ctypes.c_int(args[0]).value,
            0,
            args[1],
            0,
        )
    else:
        directory, path_address, mode = ctypes.c_int(args[0]).value, args[1], args[2]
        flags = args[3] if name == "fchmodat2" else 0
    target, nofollow = _target(request.pid, directory, path_address, flags)
    return target, ctypes.c_uint(mode).value, nofollow


def _pin(target: Path, writable: tuple[Path, ...], nofollow: bool) -> int:
    descriptor = os.open(
        target, os.O_PATH | os.O_CLOEXEC | (os.O_NOFOLLOW if nofollow else 0)
    )
    try:
        info = os.fstat(descriptor)
        actual = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if not any(actual == root or actual.is_relative_to(root) for root in writable):
            raise PermissionError(
                errno.EACCES, "metadata target outside writable scope"
            )
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise PermissionError(
                errno.EACCES, "metadata target must be a file or directory"
            )
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise PermissionError(errno.EACCES, "metadata target has multiple links")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _serve(listener: int, child: int, writable: tuple[Path, ...], audit: Path) -> int:
    import ctypes.util

    lib = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
    request, response = ctypes.POINTER(_Request)(), ctypes.POINTER(_Response)()
    lib.seccomp_notify_alloc.argtypes = [
        ctypes.POINTER(type(request)),
        ctypes.POINTER(type(response)),
    ]
    lib.seccomp_notify_receive.argtypes = [ctypes.c_int, type(request)]
    lib.seccomp_notify_respond.argtypes = [ctypes.c_int, type(response)]
    lib.seccomp_notify_id_valid.argtypes = [ctypes.c_int, ctypes.c_uint64]
    lib.seccomp_notify_free.argtypes = [type(request), type(response)]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    names = {
        lib.seccomp_syscall_resolve_name(name.encode()): name for name in METADATA_CALLS
    }
    if lib.seccomp_notify_alloc(ctypes.byref(request), ctypes.byref(response)) != 0:
        raise RuntimeError("Cannot allocate seccomp metadata notifications")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.utimensat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(_Timespec),
        ctypes.c_int,
    ]
    libc.chmod.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    polling = select.poll()
    polling.register(listener, select.POLLIN | select.POLLHUP | select.POLLERR)
    try:
        with audit.open("a") as stream:
            while True:
                done, status = os.waitpid(child, os.WNOHANG)
                if done:
                    return os.waitstatus_to_exitcode(status)
                events = polling.poll(100)
                if not events:
                    continue
                flags = events[0][1]
                if not flags & select.POLLIN:
                    if flags & select.POLLHUP:
                        # Older kernels can block notify_receive forever after
                        # the last filtered process exits. HUP is not a request.
                        _, status = os.waitpid(child, 0)
                        return os.waitstatus_to_exitcode(status)
                    raise OSError(errno.EIO, "seccomp metadata listener failed")
                ctypes.memset(request, 0, ctypes.sizeof(_Request))
                received = lib.seccomp_notify_receive(listener, request)
                if received in (-errno.EINTR, -errno.ENOENT):
                    continue
                if received != 0:
                    raise OSError(-received, "seccomp metadata receive")
                item = request.contents
                error, target = 0, None
                try:
                    name = names[item.data.nr]
                    target, value, nofollow = (
                        _timestamp_arguments(item, name)
                        if name in TIMESTAMP_CALLS
                        else _mode_arguments(item, name)
                    )
                    descriptor = _pin(target, writable, nofollow)
                    try:
                        if lib.seccomp_notify_id_valid(listener, item.id) != 0:
                            continue
                        # All user memory is copied. Operate on this pinned inode,
                        # never SECCOMP_USER_NOTIF_FLAG_CONTINUE on mutable args.
                        path = os.fsencode(f"/proc/self/fd/{descriptor}")
                        result = (
                            libc.utimensat(-100, path, value, 0)
                            if name in TIMESTAMP_CALLS
                            else libc.chmod(path, value)
                        )
                        if result != 0:
                            raise OSError(ctypes.get_errno(), "metadata update")
                    finally:
                        os.close(descriptor)
                except OSError as failure:
                    error = failure.errno or errno.EIO
                response.contents.id = item.id
                response.contents.val = 0
                response.contents.error = -error
                response.contents.flags = 0
                sent = lib.seccomp_notify_respond(listener, response)
                if sent not in (0, -errno.ENOENT):
                    raise OSError(-sent, "seccomp metadata response")
                stream.write(
                    json.dumps(
                        {
                            "pid": item.pid,
                            "operation": names[item.data.nr],
                            "target": str(target),
                            "errno": error,
                        }
                    )
                    + "\n"
                )
                stream.flush()
    finally:
        lib.seccomp_notify_free(request, response)


def _drain_children() -> None:
    """Reap the CLI and every adopted background process before publication.

    The supervisor is a child subreaper. When an intermediate shell exits,
    even a child that created its own session is reparented here. Repeating
    after each reap handles descendants that are adopted during termination.
    """
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        try:
            reaped, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if reaped:
            continue
        for value in children_path.read_text().split():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                pass
        # No writer is abandoned just because shutdown is slow. SIGKILL has
        # already been sent; waiting releases authority only after actual exit.
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return


def _supervise_metadata(command, restrict, writable, audit) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "native author child subreaper")
    parent, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    supervisor = os.getpid()
    child = os.fork()
    if child == 0:
        parent.close()
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != supervisor:
            os._exit(125)
        try:
            os.setsid()
            listener = restrict()
            child_socket.sendmsg(
                [b"metadata-listener"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [listener]))],
            )
            os.close(listener)
            child_socket.close()
            os.execvpe(command[0], command, os.environ)
        except BaseException:
            import traceback

            traceback.print_exc()
            os._exit(125)
    child_socket.close()

    def forward(signum, _frame):
        try:
            os.killpg(child, signum)
        except ProcessLookupError:
            # A signal can arrive before the forked child establishes its group.
            try:
                os.kill(child, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)
    listener = None
    try:
        data, ancillary, flags, _ = parent.recvmsg(
            64, socket.CMSG_SPACE(array.array("i").itemsize)
        )
        if data != b"metadata-listener" or flags or len(ancillary) != 1:
            raise RuntimeError("Native metadata listener was not installed")
        level, kind, value = ancillary[0]
        if (level, kind) != (socket.SOL_SOCKET, socket.SCM_RIGHTS):
            raise RuntimeError("Invalid native metadata listener")
        descriptors = array.array("i")
        descriptors.frombytes(value)
        if len(descriptors) != 1:
            raise RuntimeError("Expected one native metadata listener")
        listener = descriptors[0]
        result = _serve(
            listener, child, tuple(Path(p).resolve() for p in writable), Path(audit)
        )
    except BaseException:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    finally:
        parent.close()
        if listener is not None:
            os.close(listener)
    return result


def exec_with_metadata_broker(command, restrict, writable, audit, process_receipt):
    """Keep the SDK-owned launcher alive until every native writer has exited."""
    try:
        result = _supervise_metadata(command, restrict, writable, audit)
    finally:
        _drain_children()
        write_process_receipt(Path(process_receipt), descendants_drained=True)
    os._exit(result if result >= 0 else 128 - result)
