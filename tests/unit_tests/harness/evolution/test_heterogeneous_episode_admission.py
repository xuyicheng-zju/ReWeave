"""Host-specific recipes share physical admission without crossing devices."""

import asyncio
import json
from types import SimpleNamespace

from reweave.evolution.infrastructure.campaign_slots import episode_slot
from reweave.evolution.search.rollout import RolloutEvaluator


def test_twelve_separate_evaluators_keep_their_six_lane_host(tmp_path):
    (tmp_path / "campaign_state.json").write_text("{}")
    (tmp_path / "scheduler_policy.json").write_text('{"mode":"episode_pool"}')
    pool = tmp_path / "scheduler/online"
    pool.mkdir(parents=True)
    (pool / "resource_policy.json").write_text(
        json.dumps({"authorized_devices": [0, 1, 4, 5], "slots_per_gpu": 3})
    )
    active = set()
    observed = set()

    async def scenario():
        all_running = asyncio.Event()

        class Evaluator(RolloutEvaluator):
            async def _run(self, request, bundle, directory, *, device, slot, **kwargs):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "launch.json").write_text("{}")
                assert device in self.configuration.devices
                assert (device, slot) not in active
                active.add((device, slot))
                observed.add((device, slot))
                if len(active) == 12:
                    all_running.set()
                await asyncio.wait_for(all_running.wait(), 3)
                active.remove((device, slot))

            async def _finalize_storage(self, directory):
                pass

            def _close(self, request, directory):
                return None

        work = []
        for index in range(12):
            devices = (0, 1) if index % 2 == 0 else (4, 5)
            evaluator = Evaluator(
                SimpleNamespace(devices=devices, rollouts_per_device=3),
                repository=tmp_path,
                workspace=tmp_path,
            )
            work.append(
                evaluator(
                    None,
                    None,
                    tmp_path / "heldout" / str(index),
                    before_launch=lambda: None,
                )
            )
        await asyncio.wait_for(asyncio.gather(*work), 5)

    asyncio.run(scenario())
    assert observed == {(device, slot) for device in [0, 1, 4, 5] for slot in range(3)}
    assert not active and not list(pool.glob("*.ticket"))


def test_busy_host_does_not_block_the_other_hosts_free_device(tmp_path):
    (tmp_path / "campaign_state.json").write_text("{}")
    pool = tmp_path / "scheduler/online"
    pool.mkdir(parents=True)
    (pool / "resource_policy.json").write_text(
        json.dumps({"authorized_devices": [0, 4], "slots_per_gpu": 1})
    )

    async def scenario():
        async with episode_slot(tmp_path, tmp_path / "busy", devices=(0,)):
            blocked = episode_slot(tmp_path, tmp_path / "older", devices=(0,))
            waiter = asyncio.create_task(blocked.__aenter__())
            await asyncio.sleep(0.03)
            async with episode_slot(tmp_path, tmp_path / "ready", devices=(4,)) as (
                device,
                slot,
                _,
            ):
                assert (device, slot) == (4, 0)
                assert not waiter.done()
        assert (await asyncio.wait_for(waiter, 1))[:2] == (0, 0)
        await blocked.__aexit__(None, None, None)

    asyncio.run(asyncio.wait_for(scenario(), 3))
