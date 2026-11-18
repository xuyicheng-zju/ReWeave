"""Slow durable receipt publication must not park the scheduler event loop."""

import asyncio
import threading

from reweave.evolution.search.rollout import RolloutEvaluator


def test_settlement_keeps_loop_responsive_and_drains_one_close_on_cancel(tmp_path):
    (tmp_path / "launch.json").write_text("{}")
    evaluator = object.__new__(RolloutEvaluator)
    entered, released, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    closes = []

    async def storage(directory):
        pass

    def close(request, directory):
        closes.append(directory)
        entered.set()
        assert released.wait(3)
        finished.set()
        return "published"

    evaluator._finalize_storage, evaluator._close = storage, close

    async def run():
        task = asyncio.create_task(evaluator._settle(None, tmp_path))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        # Cancellation cannot abandon a worker that owns the receipt files.
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and not finished.is_set()
        released.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert finished.is_set()

    asyncio.run(run())
    assert closes == [tmp_path]
