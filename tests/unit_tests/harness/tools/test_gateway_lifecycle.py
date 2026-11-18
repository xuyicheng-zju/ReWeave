# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Fixed gateway cancellation joins physical work; capture preserves failures."""

import asyncio
import threading

import pytest

from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.kernel import EpisodeLimits
from reweave.tools.toolkit import readonly
from tests.test_active_execution import _RuntimeToolkit
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_bundle_runtime import Robot


def test_gateway_does_not_block_loop_and_joins_cancellation(tmp_path):
    entered, stopped = threading.Event(), threading.Event()

    class ControlledRobot(Robot):
        @readonly
        def inspect_scene(self):
            entered.set()
            try:
                while True:
                    self.raise_if_cancelled()
                    stopped.wait(0.001)
            finally:
                stopped.set()

    toolkit = ControlledRobot(tmp_path)
    runtime = BundleRuntime(
        baseline(),
        toolkit,
        "inspect",
        episode_limits=EpisodeLimits(
            gateway_calls=10, program_calls=10, wall_time_s=10
        ),
    )

    async def scenario():
        task = asyncio.create_task(runtime._dispatch("inspect_scene", {}))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert stopped.is_set()
        assert toolkit._active_operation is None

    asyncio.run(scenario())


def test_observation_cannot_erase_physical_handler_failure(tmp_path):
    class CapturingToolkit(_RuntimeToolkit):
        def get_env_state(self, **kwargs):
            super().get_env_state(**kwargs)
            return {
                "error": None,
                "interrupted": False,
                "terminated": False,
                "truncated": False,
            }

    toolkit = CapturingToolkit(
        tmp_path, handler=lambda: {"error": "servo failed", "interrupted": True}
    )
    feedback = toolkit.execute_tool("act", {}).result
    assert feedback["error"] == "servo failed"
    assert feedback["interrupted"] is True
    assert feedback["capability_result"]["error"] == "servo failed"


def test_libero_close_joins_actions_before_closing_recorder():
    robot = "libero"
    import importlib
    from types import SimpleNamespace

    from reweave.tools.toolkit import Toolkit

    module = importlib.import_module(f"reweave.benchmarks.{robot}.toolkit")
    cls = next(
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, Toolkit)
        and value is not Toolkit
    )
    toolkit = object.__new__(cls)
    events = []
    toolkit.cancel_active_and_wait = lambda: events.append("joined")
    toolkit._primitives = SimpleNamespace(
        stop_recording=lambda: events.append("recorder_closed")
    )
    toolkit.close()
    assert events == ["joined", "recorder_closed"]


def test_linux_program_launcher_has_no_retired_modification_dependency():
    import sys

    from reweave.runtime.sandbox import ProgramSandbox, SandboxBackend

    sandbox = ProgramSandbox(
        SandboxBackend(
            name="linux-seccomp", command_prefix=(), worker_isolation="linux-seccomp"
        )
    )
    command = sandbox._program_command(["/test/readable"])
    assert command[:3] == [sys.executable, "-I", "-c"]
    assert "def main():" in command[3]
