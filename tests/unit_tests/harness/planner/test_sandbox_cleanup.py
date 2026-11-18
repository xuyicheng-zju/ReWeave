# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Real protected workers drain pipe backpressure before joining the process."""

import asyncio
import json

import pytest

from reweave.evolution.bundle import ProgramLimits, PythonFile, PythonProgram
from reweave.runtime import sandbox
from reweave.runtime.capability import PhysicalOutcomeUnknown


def _program(source):
    return PythonProgram(
        entrypoint="probe:run", files=(PythonFile(path="probe.py", source=source),)
    )


async def _no_rpc(*args):
    raise AssertionError("no runtime call expected")


def test_protocol_failure_drains_full_stdout_pipe_before_join(monkeypatch):
    async def scenario():
        processes = []
        paused = asyncio.Event()
        original = asyncio.create_subprocess_exec

        class FullPipe:
            def __init__(self, stream):
                self.stream = stream

            async def readline(self):
                # Hold Host parsing until actual pipe backpressure is present.
                # The test then exercises cleanup after kill, not a timing race.
                while not self.stream._paused:
                    await asyncio.sleep(0.001)
                paused.set()
                return await self.stream.readline()

            async def read(self, size=-1):
                return await self.stream.read(size)

        async def spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            process.stdout = FullPipe(process.stdout)
            processes.append(process)
            return process

        monkeypatch.setattr(sandbox, "_IPC_LINE_LIMIT_BYTES", 1024)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            sandbox.ProgramSandbox().run_program(
                _program(
                    """def run(runtime, context):
    print("invalid protocol")
    for _ in range(10000):
        print("x" * 100)
    return {}
"""
                ),
                ProgramLimits(
                    runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=None
                ),
                {},
                _no_rpc,
                input_schema={"type": "object"},
            )
        )
        try:
            await asyncio.wait_for(paused.wait(), timeout=5)
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done, "killed worker remained blocked on a full IPC pipe"
            with pytest.raises(json.JSONDecodeError):
                await task
            assert processes[0].returncode is not None
            assert len(asyncio.all_tasks()) == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await process.communicate()

    asyncio.run(scenario())


def test_worker_stderr_backpressure_does_not_consume_a_local_deadline(monkeypatch):
    async def scenario():
        monkeypatch.setattr(sandbox, "_IPC_LINE_LIMIT_BYTES", 1024)
        result = await asyncio.wait_for(
            sandbox.ProgramSandbox().run_program(
                _program(
                    """from asyncio.log import logger
def run(runtime, context):
    for _ in range(1024):
        logger.warning("diagnostic " + "x" * 1024)
    return {"completed": True}
"""
                ),
                ProgramLimits(
                    runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=5
                ),
                {},
                _no_rpc,
                input_schema={"type": "object"},
            ),
            timeout=10,
        )
        assert result == {"completed": True}
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("unknown", [False, True])
def test_cancellation_during_pipe_cleanup_preserves_the_original_outcome(
    monkeypatch, unknown
):
    async def scenario():
        processes = []
        rpc_started = asyncio.Event()
        drainage_started = asyncio.Event()
        release_drainage = asyncio.Event()
        original = asyncio.create_subprocess_exec
        error = PhysicalOutcomeUnknown("submitted action lost acknowledgement")

        class DelayedDrain:
            def __init__(self, stream):
                self.stream = stream

            async def readline(self):
                return await self.stream.readline()

            async def read(self, size=-1):
                drainage_started.set()
                await release_drainage.wait()
                return await self.stream.read(size)

        async def spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            process.stdout = DelayedDrain(process.stdout)
            processes.append(process)
            return process

        async def invoke(*args):
            rpc_started.set()
            if unknown:
                raise error
            await asyncio.Event().wait()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            sandbox.ProgramSandbox().run_program(
                _program(
                    'def run(runtime, context):\n    return runtime.observe("probe", {})\n'
                ),
                ProgramLimits(
                    runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=None
                ),
                {},
                invoke,
                input_schema={"type": "object"},
            )
        )
        try:
            await asyncio.wait_for(rpc_started.wait(), timeout=5)
            if not unknown:
                task.cancel()
            await asyncio.wait_for(drainage_started.wait(), timeout=5)
            task.cancel()
            release_drainage.set()
            with pytest.raises(
                PhysicalOutcomeUnknown if unknown else asyncio.CancelledError
            ) as failed:
                await asyncio.wait_for(task, timeout=5)
            if unknown:
                assert failed.value is error
            assert processes[0].returncode is not None
            assert len(asyncio.all_tasks()) == 1
        finally:
            release_drainage.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await process.communicate()

    asyncio.run(scenario())
