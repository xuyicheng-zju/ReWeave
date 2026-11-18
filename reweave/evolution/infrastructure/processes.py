# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Native author and rollout process-group ownership."""

import asyncio
import os
import signal
import time


async def drain_process_group(process, *, terminate: bool = False):
    """Drain an owned process group before publication or releasing its resources."""
    if terminate and process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=15.0)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
    # Services share the worker's process group. A normally exited leader can
    # leave descendants holding the inherited GPU lease, so stop and drain the
    # group before publishing its exit receipt or releasing the device.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        # Once the owned leader is reaped, EPERM means no signalable process
        # from this owner remains in that numeric group (notably on macOS).
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except (PermissionError, ProcessLookupError):
            return
        await asyncio.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except (PermissionError, ProcessLookupError):
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(
        "Owned process group did not close after SIGKILL; retain its quarantine"
    )
