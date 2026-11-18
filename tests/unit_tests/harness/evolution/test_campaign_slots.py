"""Global admission across tasks, cancellation, and inherited physical ownership."""

import asyncio
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reweave.evolution.infrastructure.campaign_slots import episode_slot, online_slots
from reweave.evolution.search.rollout import RolloutEvaluator
from reweave.utils.gpu_slots import decode_slot, encode_slot


def test_request_gate_only_does_not_reserve_idle_offline_process_slots(tmp_path):
    from contextlib import AsyncExitStack

    from reweave.evolution.infrastructure.campaign_slots import offline_admission

    (tmp_path / "campaign_state.json").write_text("{}")
    (tmp_path / "scheduler_policy.json").write_text(
        json.dumps(
            {"mode": "episode_pool", "offline_admission": "shared_request_gate_only"}
        )
    )

    async def run():
        async with AsyncExitStack() as stack:
            for n in range(40):
                await asyncio.wait_for(
                    stack.enter_async_context(
                        offline_admission(tmp_path / str(n), required=True)
                    ),
                    1,
                )
            assert not (tmp_path / "scheduler/offline").exists()

    asyncio.run(run())


@pytest.mark.parametrize("gpu0", [6, 7])
def test_asymmetric_allocation_admits_seven_plus_three_with_stable_leases(
    tmp_path, gpu0
):
    async def run():
        pool = tmp_path / "scheduler/online"
        pool.mkdir(parents=True)
        (pool / "resource_policy.json").write_text(
            json.dumps(
                {
                    "authorized_devices": [0, 1],
                    "slots_per_gpu": 3,
                    "slots_by_gpu": {"0": gpu0, "1": 3},
                }
            )
        )
        assert online_slots(pool) == [0, 4, 1, 5, 2, 6, 3, 16, 17, 18][: gpu0 + 3]
        contexts, assigned = [], []
        try:
            for index in range(gpu0 + 3):
                ctx = episode_slot(tmp_path, tmp_path / str(index))
                device, lane, _ = await ctx.__aenter__()
                contexts.append(ctx)
                assigned.append((device, lane))
            assert assigned.count((0, 0)) == 1
            assert sum(device == 0 for device, _ in assigned) == gpu0
            assert sum(device == 1 for device, _ in assigned) == 3
            blocked = episode_slot(tmp_path, tmp_path / "eleventh")
            waiter = asyncio.create_task(blocked.__aenter__())
            await asyncio.sleep(0.03)
            assert not waiter.done()
            await contexts.pop().__aexit__(None, None, None)
            assert (await asyncio.wait_for(waiter, 1))[:2] == (0, gpu0 - 1)
            await blocked.__aexit__(None, None, None)
        finally:
            for ctx in contexts:
                await ctx.__aexit__(None, None, None)

    asyncio.run(run())


@pytest.mark.parametrize(
    "counts",
    [
        {"0": 7},
        {"0": 7, "1": 3, "2": 1},
        {"0": 0, "1": 3},
        {"0": True, "1": 3},
        {"0": 17, "1": 3},
        [7, 3],
    ],
)
def test_asymmetric_allocation_rejects_invalid_or_unauthorized_gpus(tmp_path, counts):
    (tmp_path / "resource_policy.json").write_text(
        json.dumps(
            {
                "authorized_devices": [0, 1],
                "slots_per_gpu": 3,
                "slots_by_gpu": counts,
            }
        )
    )
    with pytest.raises(ValueError, match="slots_by_gpu"):
        online_slots(tmp_path)


@pytest.mark.parametrize("lanes", [8, 16])
def test_more_lanes_preserve_existing_leases_and_never_use_other_gpus(tmp_path, lanes):
    async def run():
        pool = tmp_path / "scheduler/online"
        pool.mkdir(parents=True)
        policy = pool / "resource_policy.json"
        policy.write_text('{"authorized_devices":[0,1],"slots_per_gpu":4}')
        contexts, assigned = [], []
        try:
            for index in range(8):
                ctx = episode_slot(tmp_path, Path(str(index)))
                device, lane, _ = await ctx.__aenter__()
                contexts.append(ctx)
                assigned.append((device, lane))
            legacy = assigned.copy()
            policy.write_text(
                json.dumps({"authorized_devices": [0, 1], "slots_per_gpu": lanes})
            )
            for index in range(8, 2 * lanes):
                ctx = episode_slot(tmp_path, Path(str(index)))
                device, lane, _ = await ctx.__aenter__()
                contexts.append(ctx)
                assigned.append((device, lane))
            assert assigned[:8] == legacy
            assert set(assigned) == {
                (gpu, lane) for gpu in (0, 1) for lane in range(lanes)
            }
            assert len({encode_slot(*pair) for pair in assigned}) == 2 * lanes
            assert all(decode_slot(encode_slot(*pair)) == pair for pair in assigned)
            assert all(
                encode_slot(gpu, lane) == gpu * 4 + lane
                for gpu in range(4)
                for lane in range(4)
            )
        finally:
            for ctx in contexts:
                await ctx.__aexit__(None, None, None)

    asyncio.run(run())


def test_root_owned_evaluators_share_global_admission_and_wait_without_overlap(
    tmp_path,
):
    (tmp_path / "campaign_state.json").write_text("{}")
    (tmp_path / "scheduler_policy.json").write_text('{"mode":"episode_pool"}')
    pool = tmp_path / "scheduler/online"
    pool.mkdir(parents=True)
    (pool / "resource_policy.json").write_text(
        '{"authorized_devices":[0,1],"slots_per_gpu":1}'
    )
    active, slots, completed = set(), set(), []
    maximum = 0

    async def run(request, bundle, directory, *, device, slot, **kwargs):
        nonlocal maximum
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "launch.json").write_text("{}")
        identity = (device, slot)
        assert identity not in active
        active.add(identity)
        slots.add(identity)
        maximum = max(maximum, len(active))
        await asyncio.sleep(0.03)
        active.remove(identity)
        completed.append(directory.name)

    async def storage(directory):
        pass

    async def scenario():
        evaluators = [
            RolloutEvaluator(
                SimpleNamespace(devices=(0, 1), rollouts_per_device=4),
                repository=tmp_path,
                workspace=tmp_path,
            )
            for _ in range(16)
        ]
        pending = []
        for index, evaluator in enumerate(evaluators):
            evaluator._run = run
            evaluator._finalize_storage = storage
            evaluator._close = lambda request, directory: directory.name
            pending.append(
                evaluator(
                    None,
                    None,
                    tmp_path / "heldout" / str(index),
                    before_launch=lambda: None,
                )
            )
        return await asyncio.gather(*pending)

    results = asyncio.run(scenario())
    assert len(results) == len(completed) == 16
    assert maximum == 2 and slots == {(0, 0), (1, 0)}
    assert not list(pool.glob("*.ticket"))


def test_draining_campaign_does_not_block_ready_shared_pool_peer(tmp_path):
    async def run():
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "campaign_state.json").write_text("{}")
        (new / "campaign_state.json").write_text("{}")
        (old / "scheduler_drain.json").write_text("{}")
        (old / "scheduler/online").mkdir(parents=True)
        (new / "scheduler").symlink_to(old / "scheduler", target_is_directory=True)
        parked = episode_slot(old, old / "tasks/a/evolution/attempt")
        waiting = asyncio.create_task(parked.__aenter__())
        await asyncio.sleep(0.05)
        try:
            peer = episode_slot(new, new / "tasks/b/evolution/attempt")
            await asyncio.wait_for(peer.__aenter__(), 1)
            assert not waiting.done()
            await peer.__aexit__(None, None, None)
        finally:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting

    asyncio.run(run())


def test_sixteen_slots_fifo_refill_and_cancel(tmp_path):
    async def run():
        leases = []
        try:
            for i in range(16):
                context = episode_slot(tmp_path, Path(str(i)))
                device, lane, _ = await context.__aenter__()
                leases.append((context, device, lane))
            assert {(d, s) for _, d, s in leases} == {
                (d, s) for d in range(4) for s in range(4)
            }
            first = episode_slot(tmp_path, Path("first"))
            second = episode_slot(tmp_path, Path("second"))
            pending = asyncio.create_task(first.__aenter__())
            await asyncio.sleep(0.03)
            cancelled = asyncio.create_task(second.__aenter__())
            await asyncio.sleep(0.03)
            assert not pending.done() and not cancelled.done()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            old, device, lane = leases.pop(5)
            await old.__aexit__(None, None, None)
            assert (await asyncio.wait_for(pending, 2))[:2] == (device, lane)
            await first.__aexit__(None, None, None)
        finally:
            for context, _, _ in leases:
                await context.__aexit__(None, None, None)
        assert not list((tmp_path / "scheduler/online").glob("*.ticket"))

    asyncio.run(run())


def test_child_inherits_exclusive_slot_after_parent_release(tmp_path):
    async def run():
        async with episode_slot(tmp_path, Path("child")) as (device, lane, fd):
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(fd,)
            )
        try:
            with (tmp_path / "scheduler/online/slot-00.lock").open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            async with episode_slot(tmp_path, Path("next")) as (d, s, _):
                assert (d, s) != (device, lane)
        finally:
            child.terminate()
            child.wait()

    asyncio.run(run())


def test_stale_tickets_do_not_block_and_offline_is_separate(tmp_path):
    async def run():
        pool = tmp_path / "scheduler/online"
        pool.mkdir(parents=True)
        (pool / "00000000-dead.ticket").write_text("dead")
        async with episode_slot(tmp_path, Path("online")) as (d, s, _):
            assert (d, s) == (0, 0)
            async with episode_slot(
                tmp_path, Path("offline"), resource="offline", capacity=1
            ):
                assert not (pool / "00000000-dead.ticket").exists()

    asyncio.run(run())


def test_drain_parks_before_any_physical_process(tmp_path):
    (tmp_path / "campaign_state.json").write_text("{}")
    (tmp_path / "scheduler_drain.json").write_text("{}")
    attempt = tmp_path / "tasks/a/evolution/attempt"
    attempt.mkdir(parents=True)
    code = 'from pathlib import Path; from reweave.evolution.infrastructure.campaign_slots import wait_for_transition; wait_for_transition(Path(__import__("sys").argv[1])); raise RuntimeError("must not start")'
    child = subprocess.Popen([sys.executable, "-c", code, str(attempt)])
    try:
        import time

        deadline = time.monotonic() + 5
        while not (attempt / "admission.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (
            json.loads((attempt / "admission.json").read_text())["status"] == "waiting"
        )
        assert child.poll() is None
        assert not (attempt / "process.json").exists()
    finally:
        child.terminate()
        child.wait()


def test_related_campaign_reserves_refresh_lanes_only_until_source_drain_closes(
    tmp_path,
):
    drain = tmp_path / "source_drain.json"
    drain.write_text("{}")
    (tmp_path / "scheduler_exclusions.json").write_text(
        json.dumps({"slots": [0, 4], "until_drain_removed": str(drain)})
    )

    async def run():
        async with episode_slot(tmp_path, Path("new")) as (device, lane, _):
            assert device * 4 + lane == 8
        async with episode_slot(tmp_path, Path("offline"), resource="offline") as (
            device,
            lane,
            _,
        ):
            assert device * 4 + lane == 0
        drain.unlink()
        async with episode_slot(tmp_path, Path("after")) as (device, lane, _):
            assert device * 4 + lane == 0

    asyncio.run(run())


def test_rolling_refresh_releases_one_task_without_unblocking_old_peer(tmp_path):
    from reweave.evolution.infrastructure.campaign_slots import transition_blocks

    async def run():
        (tmp_path / "campaign_state.json").write_text("{}")
        (tmp_path / "scheduler_drain.json").write_text('{"owner":"refresh"}')
        old_owner = tmp_path / "tasks/old/evolution/attempt"
        new_owner = tmp_path / "tasks/new/evolution/attempt"
        waiting_context = episode_slot(tmp_path, old_owner)
        waiting = asyncio.create_task(waiting_context.__aenter__())
        await asyncio.sleep(0.05)
        admission = tmp_path / "scheduler_transition_admissions.json"
        admission.write_text('{"owner":"foreign","refreshed_tasks":{"new":{}}}')
        assert transition_blocks(tmp_path, new_owner)
        admission.write_text('{"owner":"refresh","refreshed_tasks":{"new":{}}}')
        try:
            context = episode_slot(tmp_path, new_owner)
            await asyncio.wait_for(context.__aenter__(), 1)
            assert not waiting.done()
            assert transition_blocks(tmp_path, old_owner)
            await context.__aexit__(None, None, None)
        finally:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting

    asyncio.run(run())


def test_superseded_task_cannot_start_physical_or_offline_work(tmp_path):
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "rerun_provenance.json").write_text(
        json.dumps(
            {
                "supersedes_campaign": str(tmp_path),
                "tasks": ["old"],
            }
        )
    )
    (tmp_path / "task_supersessions.json").write_text(
        json.dumps({"old": str(replacement)})
    )

    async def run():
        for resource in ("online", "offline", "storage"):
            context = episode_slot(
                tmp_path, tmp_path / "tasks/old/evolution/attempt", resource=resource
            )
            with pytest.raises(RuntimeError, match="Task superseded"):
                await context.__aenter__()
        assert not (tmp_path / "scheduler").exists()

    asyncio.run(run())


def test_shared_gpu_allocation_applies_to_future_campaigns(tmp_path):
    async def run():
        original, future = tmp_path / "original", tmp_path / "future"
        pool = original / "scheduler/online"
        pool.mkdir(parents=True)
        (pool / "resource_policy.json").write_text('{"authorized_devices":[0,1]}')
        future.mkdir()
        (future / "scheduler").symlink_to(
            original / "scheduler", target_is_directory=True
        )
        leases = []
        try:
            for i in range(8):
                ctx = episode_slot(future, Path(str(i)))
                device, lane, _ = await ctx.__aenter__()
                leases.append(ctx)
                assert device in (0, 1)
            waiting = asyncio.create_task(
                episode_slot(future, Path("ninth")).__aenter__()
            )
            await asyncio.sleep(0.05)
            assert not waiting.done()
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        finally:
            for ctx in leases:
                await ctx.__aexit__(None, None, None)

    asyncio.run(run())


def test_offline_capacity_is_not_limited_by_the_gpu_slot_namespace(tmp_path):
    async def run():
        contexts = []
        try:
            for index in range(65):
                context = episode_slot(
                    tmp_path, Path(str(index)), resource="offline", capacity=65
                )
                await context.__aenter__()
                contexts.append(context)
            assert len(contexts) == 65
        finally:
            for context in contexts:
                await context.__aexit__(None, None, None)

    asyncio.run(run())


def test_campaign_allowlist_caps_24_requests_and_preserves_other_slots(tmp_path):
    from contextlib import AsyncExitStack

    pool = tmp_path / "scheduler/online"
    pool.mkdir(parents=True)
    (pool / "resource_policy.json").write_text(
        json.dumps({"authorized_devices": [0, 1], "slots_per_gpu": 8})
    )
    allowed = [encode_slot(gpu, lane) for lane in range(4) for gpu in (0, 1)]
    limited, unrestricted = [tmp_path / name for name in ("limited", "unrestricted")]
    for root in (limited, unrestricted):
        root.mkdir()
        (root / "campaign_state.json").write_text("{}")
        (root / "scheduler").symlink_to(
            tmp_path / "scheduler", target_is_directory=True
        )
    (limited / "scheduler_policy.json").write_text(
        json.dumps({"mode": "episode_pool", "allowed_online_slots": allowed})
    )
    (unrestricted / "scheduler_policy.json").write_text(
        json.dumps({"mode": "episode_pool"})
    )

    async def run():
        active = peak = completed = 0
        release = asyncio.Event()

        async def cell(index):
            nonlocal active, peak, completed
            async with episode_slot(limited, limited / f"cell-{index}") as (
                gpu,
                lane,
                _,
            ):
                assert encode_slot(gpu, lane) in allowed
                active += 1
                peak = max(peak, active)
                await release.wait()
                await asyncio.sleep(0.005)
                active -= 1
                completed += 1

        tasks = [asyncio.create_task(cell(index)) for index in range(24)]
        try:
            for _ in range(100):
                if active == 8:
                    break
                await asyncio.sleep(0.01)
            assert active == 8
            # Older restricted tickets cannot block the other eight global slots.
            async with AsyncExitStack() as stack:
                others = []
                for index in range(8):
                    gpu, lane, _ = await asyncio.wait_for(
                        stack.enter_async_context(
                            episode_slot(unrestricted, unrestricted / f"other-{index}")
                        ),
                        2,
                    )
                    others.append(encode_slot(gpu, lane))
                assert set(others).isdisjoint(allowed)
                assert set(others) | set(allowed) == set(online_slots(pool))
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
            assert peak == 8 and completed == 24
            # A remote owner remains reserved even when its local lease is gone.
            (pool / f"slot-{allowed[0]:02d}.remote.json").write_text(
                json.dumps({"status": "running"})
            )
            async with AsyncExitStack() as stack:
                for index in range(7):
                    gpu, lane, _ = await stack.enter_async_context(
                        episode_slot(limited, limited / f"remote-{index}")
                    )
                    assert encode_slot(gpu, lane) != allowed[0]
                blocked = asyncio.create_task(
                    episode_slot(limited, limited / "blocked").__aenter__()
                )
                await asyncio.sleep(0.03)
                assert not blocked.done()
                blocked.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await blocked
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("value", [[], [True], [0, 0], [128], "0"])
def test_invalid_campaign_allowlist_fails_closed(tmp_path, value):
    from reweave.evolution.infrastructure.campaign_slots import allowed_online_slots

    (tmp_path / "scheduler_policy.json").write_text(
        json.dumps({"allowed_online_slots": value})
    )
    with pytest.raises(ValueError, match="allowlist"):
        allowed_online_slots(tmp_path)
