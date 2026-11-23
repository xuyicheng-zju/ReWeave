# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""OS read boundary for generated workers, shared by validation and execution."""


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
