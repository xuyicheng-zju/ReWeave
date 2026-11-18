"""Cross-workspace episode admission, with durable FIFO tickets and OS leases."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from reweave.utils.gpu_slots import decode_slot, encode_slot
from reweave.utils.persistence import atomic_json


def campaign_root(directory: Path) -> Path | None:
    directory = directory.resolve()
    for parent in (directory, *directory.parents):
        if (parent / "campaign_state.json").is_file():
            return parent
    return None


def task_for_owner(root: Path, owner: Path) -> str | None:
    try:
        parts = owner.relative_to(root / "tasks").parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else None


def transition_blocks(root: Path, owner: Path) -> bool:
    drain = root / "scheduler_drain.json"
    if not drain.exists():
        return False
    admissions = root / "scheduler_transition_admissions.json"
    if not admissions.exists():
        return True
    expected = json.loads(drain.read_text())
    release = json.loads(admissions.read_text())
    return not (
        expected.get("owner")
        and release.get("owner") == expected["owner"]
        and task_for_owner(root, owner) in release.get("refreshed_tasks", {})
    )


def reject_superseded(root: Path, owner: Path) -> None:
    path = root / "task_supersessions.json"
    if not path.exists():
        return
    task = task_for_owner(root, owner)
    replacement = json.loads(path.read_text()).get(task)
    if replacement is None:
        return
    receipt = json.loads((Path(replacement) / "rerun_provenance.json").read_text())
    if (
        receipt.get("supersedes_campaign") != str(root.resolve())
        or task not in receipt["tasks"]
    ):
        raise ValueError("Invalid task supersession provenance")
    raise RuntimeError(f"Task superseded by {replacement}; original evidence retained")


def wait_for_transition(directory: Path) -> None:
    """Park newly spawned workers before importing the physical runner."""
    root = campaign_root(directory)
    if root is None or not transition_blocks(root, directory):
        return
    atomic_json(directory / "admission.json", {"status": "waiting", "pid": os.getpid()})
    while transition_blocks(root, directory):
        time.sleep(0.5)
    atomic_json(
        directory / "admission.json", {"status": "admitted", "pid": os.getpid()}
    )


def enabled(root: Path) -> bool:
    path = root / "scheduler_policy.json"
    return path.exists() and json.loads(path.read_text()).get("mode") == "episode_pool"


def online_slots(pool: Path) -> list[int]:
    """Apply the shared pool's current operator allocation across all campaigns."""
    path = pool / "resource_policy.json"
    policy = json.loads(path.read_text()) if path.exists() else {}
    devices = policy.get("authorized_devices") if path.exists() else [0, 1, 2, 3]
    if (
        not isinstance(devices, list)
        or not devices
        or any(type(device) is not int or device not in range(8) for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("Invalid shared online GPU allocation")
    # Explicit per-device counts reserve space for models on only some GPUs.
    # Existing uniform policies remain authoritative for their live pools.
    counts = policy.get("slots_per_device")
    if counts is not None:
        if (
            "slots_per_gpu" in policy
            or not isinstance(counts, dict)
            or set(counts) != {str(gpu) for gpu in devices}
        ):
            raise ValueError("Invalid shared online slots per device")
    else:
        counts = {str(gpu): policy.get("slots_per_gpu", 4) for gpu in devices}
    if any(type(n) is not int or not 1 <= n <= 16 for n in counts.values()):
        raise ValueError("Invalid shared online slots per GPU")
    quarantined = policy.get("quarantined_gpus", [])
    if not isinstance(quarantined, list) or any(
        type(gpu) is not int or gpu not in devices for gpu in quarantined
    ):
        raise ValueError("Invalid quarantined GPU allocation")
    per_device = policy.get("slots_by_gpu")
    if per_device is not None:
        if "slots_per_device" in policy:
            raise ValueError("Conflicting per-device slot policy fields")
        if (
            not isinstance(per_device, dict)
            or set(per_device) != set(counts)
            or any(
                type(count) is not int or not 1 <= count <= 16
                for count in per_device.values()
            )
        ):
            raise ValueError("Invalid shared online slots_by_gpu allocation")
        counts = per_device
    return [
        encode_slot(gpu, lane)
        for lane in range(max(counts.values()))
        for gpu in devices
        if gpu not in quarantined and lane < counts[str(gpu)]
    ]


def allowed_online_slots(root: Path) -> set[int] | None:
    """Optional campaign-local subset of the unchanged shared physical pool."""
    path = root / "scheduler_policy.json"
    policy = json.loads(path.read_text()) if path.exists() else {}
    slots = policy.get("allowed_online_slots")
    if slots is None:
        return None
    if (
        not isinstance(slots, list)
        or not slots
        or any(type(slot) is not int or not 0 <= slot < 128 for slot in slots)
        or len(set(slots)) != len(slots)
    ):
        raise ValueError("Invalid campaign online slot allowlist")
    return set(slots)


@asynccontextmanager
async def episode_slot(
    root: Path, owner: Path, *, resource="online", capacity=16, devices=None
):
    """FIFO per eligible device; inherit the lease into the physical worker."""
    reject_superseded(root, owner)
    pool = root / "scheduler" / resource
    pool.mkdir(parents=True, exist_ok=True)
    ticket = pool / f"{time.time_ns():020d}-{os.getpid()}.ticket"
    # Ticket locks distinguish a waiting process from a stale receipt after a crash.
    pending = ticket.with_suffix(".pending")
    with pending.open("w+") as queued:
        fcntl.flock(queued, fcntl.LOCK_EX)
        queued.write(
            json.dumps(
                {"owner": str(owner), "devices": devices, "root": str(root.resolve())}
            )
        )
        queued.flush()
        pending.rename(ticket)
        lease = None
        try:
            while lease is None:
                if transition_blocks(root, owner):
                    await asyncio.sleep(0.2)
                    continue
                with (pool / "queue.lock").open("a") as mutex:
                    fcntl.flock(mutex, fcntl.LOCK_EX)
                    slots = (
                        online_slots(pool)
                        if resource == "online"
                        else list(range(capacity))
                    )
                    first = {}
                    for candidate in sorted(pool.glob("*.ticket")):
                        if candidate == ticket:
                            entry = {
                                "owner": str(owner),
                                "devices": devices,
                                "root": str(root.resolve()),
                            }
                        else:
                            with candidate.open("a+") as check:
                                try:
                                    fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    check.seek(0)
                                    value = check.read().strip()
                                    try:
                                        entry = json.loads(value)
                                    except json.JSONDecodeError:
                                        # A controller handoff can retain a live
                                        # legacy ticket containing only its owner.
                                        entry = {"owner": value, "devices": None}
                                    if not isinstance(entry, dict):
                                        entry = {"owner": value, "devices": None}
                                else:
                                    candidate.unlink(missing_ok=True)
                                    continue
                        waiting_owner = Path(entry["owner"])
                        waiting_root = (
                            Path(entry["root"])
                            if entry.get("root")
                            else campaign_root(waiting_owner)
                        )
                        if waiting_root is not None and transition_blocks(
                            waiting_root, waiting_owner
                        ):
                            continue
                        allowed = (
                            allowed_online_slots(waiting_root)
                            if resource == "online" and waiting_root is not None
                            else None
                        )
                        for slot in slots:
                            eligible = entry.get("devices")
                            if resource == "online" and (
                                (
                                    eligible is not None
                                    and decode_slot(slot)[0] not in eligible
                                )
                                or (allowed is not None and slot not in allowed)
                            ):
                                continue
                            first.setdefault(slot, candidate)
                    for slot in slots:
                        if first.get(slot) != ticket:
                            continue
                        exclusions = root / "scheduler_exclusions.json"
                        if resource == "online" and exclusions.exists():
                            reserved = json.loads(exclusions.read_text())
                            if (
                                Path(reserved["until_drain_removed"]).exists()
                                and slot in reserved["slots"]
                            ):
                                continue
                        handle = (pool / f"slot-{slot:02d}.lock").open("a")
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            handle.close()
                            continue
                        remote = pool / f"slot-{slot:02d}.remote.json"
                        try:
                            reserved = (
                                resource == "online"
                                and remote.exists()
                                and json.loads(remote.read_text())["status"] != "exited"
                            )
                        except BaseException:
                            handle.close()
                            raise
                        if reserved:
                            # Check after locking: the departing local worker
                            # may just have published a detached remote owner.
                            handle.close()
                            continue
                        lease = handle
                        atomic_json(
                            pool / f"slot-{slot:02d}.json",
                            {
                                "status": "leased",
                                "pid": os.getpid(),
                                "owner": str(owner),
                                "slot": slot,
                                "at": time.time(),
                            },
                        )
                        ticket.unlink(missing_ok=True)
                        break
                if lease is None:
                    await asyncio.sleep(0.2)
            try:
                device, lane = (
                    decode_slot(slot) if resource == "online" else divmod(slot, 4)
                )
                yield device, lane, lease.fileno()
            finally:
                atomic_json(
                    pool / f"slot-{slot:02d}.json",
                    {
                        "status": "released",
                        "owner": str(owner),
                        "slot": slot,
                        "at": time.time(),
                    },
                )
        finally:
            ticket.unlink(missing_ok=True)
            if lease is not None:
                lease.close()


@asynccontextmanager
async def offline_admission(directory: Path, *, required: bool):
    """Queue before starting an offline execution deadline; record waiting separately."""
    root = campaign_root(directory)
    if not required or root is None or not enabled(root):
        yield
        return
    policy = json.loads((root / "scheduler_policy.json").read_text())
    if policy.get("offline_admission") == "shared_request_gate_only":
        # The account-wide ProviderAdmission lease owns the model request limit.
        # An idle author/graph process must not reserve another offline quota.
        yield
        return
    started = time.monotonic()
    record = {"owner": str(directory), "queued_at": time.time(), "admitted_at": None}
    path = root / "reports/offline_admissions" / f"{time.time_ns()}-{os.getpid()}.json"
    try:
        async with episode_slot(root, directory, resource="offline", capacity=32):
            record.update(admitted_at=time.time(), waiting_s=time.monotonic() - started)
            yield
    finally:
        if record["admitted_at"] is None:
            record["waiting_s"] = time.monotonic() - started
        record["closed_at"] = time.time()
        atomic_json(path, record)
