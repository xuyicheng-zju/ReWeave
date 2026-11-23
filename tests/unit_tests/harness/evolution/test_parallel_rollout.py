# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Real child-process isolation and leases; no robot or model is invoked here."""

import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import EvaluationRequest, EvolutionIncomplete
from reweave.evolution.runtime_manifest import RuntimeManifest, capture_baseline
from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
from tests.unit_tests.harness.evolution.test_rollout_recipe import (
    preflight,
    recipe_fixture,
)
from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture


@pytest.fixture
def evaluator(tmp_path):
    inputs = fixture(tmp_path)
    capture_baseline(tmp_path / "baseline", **inputs)
    manifest = RuntimeManifest.model_validate_json(
        (tmp_path / "baseline/runtime.json").read_text()
    )
    config = RolloutConfiguration(
        devices=(0, 1, 2, 3),
        arguments=("--robot", "libero", "--planner", "native"),
        memory_snapshot=str(inputs["memory"]),
        runtime_manifest=manifest,
        episode_limits=inputs["limits"]["episode"],
    )
    return RolloutEvaluator(
        config, repository=(tmp_path / "repo"), workspace=tmp_path / "search"
    )


def request():
    return EvaluationRequest(
        record_id="fixture-evaluation-request",
        bundle_id="a" * 64,
        instance=protocol().dev[0],
        split="dev",
        manifest_id="b" * 64,
        execution=protocol().execution_id,
        conditions=protocol().execution,
    )


def test_separate_evaluators_share_workspace_device_lease(evaluator):
    fresh = RolloutEvaluator(
        evaluator.configuration,
        repository=evaluator.repository,
        workspace=evaluator.workspace,
    )

    async def run():
        async with evaluator._slot() as (device, slot, _):
            assert (device, slot) == (0, 0)
            with pytest.raises(EvolutionIncomplete, match="still owned"):
                async with fresh._slot():
                    pytest.fail("a separate version evaluator reused the occupied slot")
        assert (
            evaluator.workspace / ".scratch/runtime/devices/cuda-0-slot-0.lock"
        ).exists()

    asyncio.run(run())


def test_gpu_first_allocation_rebalances_and_preserves_per_gpu_limit(evaluator):
    payload = evaluator.configuration.model_dump(mode="json")
    payload.update(
        devices=list(range(8)),
        rollouts_per_device=4,
        shared_inference_services=[
            {
                "device": device,
                "vla_endpoint": f"http://127.0.0.1:{8100 + device * 2}",
                "sam3_endpoint": f"http://127.0.0.1:{8101 + device * 2}",
            }
            for device in range(8)
        ],
    )
    evaluator = RolloutEvaluator(
        RolloutConfiguration.model_validate(payload),
        repository=evaluator.repository,
        workspace=evaluator.workspace,
    )

    async def run():
        leases = []

        async def acquire():
            lease = evaluator._slot()
            device, slot, _ = await lease.__aenter__()
            leases.append((device, slot, lease))
            return device, slot

        try:
            assignments = [await acquire() for _ in range(20)]
            assert [device for device, _ in assignments[:8]] == list(range(8))
            assert [sum(d == gpu for d, _ in assignments) for gpu in range(8)] == [
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
            ]
            # Completion order must not strand a GPU while filling other GPUs.
            released = [row for row in leases if row[0] == 7]
            for row in released:
                leases.remove(row)
                await row[2].__aexit__(None, None, None)
            assert (await acquire())[0] == 7
            assert (await acquire())[0] == 7
            for _ in range(12):
                await acquire()
            assert len({(d, s) for d, s, _ in leases}) == 32
            pending = asyncio.create_task(acquire())
            await asyncio.sleep(0)
            assert not pending.done()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            device, slot, lease = leases.pop()
            await lease.__aexit__(None, None, None)
            assert await asyncio.wait_for(acquire(), 1) == (device, slot)
        finally:
            for _, _, lease in reversed(leases):
                await lease.__aexit__(None, None, None)

    asyncio.run(run())


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


def test_extended_pooled_lane_reaches_the_child_as_the_same_remote_lock(
    evaluator, tmp_path
):
    root = evaluator.workspace
    root.mkdir()
    (root / "campaign_state.json").write_text("{}")
    (root / "scheduler_policy.json").write_text('{"mode":"episode_pool"}')
    directory = root / "attempts/extended"
    directory.mkdir(parents=True)
    child = "import os; from pathlib import Path; Path('slot.txt').write_text(os.environ['REWEAVE_CAMPAIGN_SLOT'])"

    async def run():
        with (tmp_path / "lease.lock").open("a") as lease:
            await evaluator._launch(
                request(),
                directory,
                [sys.executable, "-c", child],
                time.time(),
                1,
                8,
                lease.fileno(),
                lambda: None,
            )

    asyncio.run(run())
    from reweave.utils.gpu_slots import decode_slot

    assert decode_slot(int((directory / "slot.txt").read_text())) == (1, 8)


CHILD = """import json,time
from pathlib import Path
root = Path.cwd()
launch = json.loads((root / "launch.json").read_text())
(root / "ready.json").write_text(json.dumps({"cwd": str(root), "device": launch["cuda_device"], "slot": launch["device_slot"]}))
end = time.monotonic() + 5
while not (root.parent.parent / "release").exists() and time.monotonic() < end:
    time.sleep(0.01)
(root / "done").touch()
"""


@pytest.mark.parametrize("width", [4, 8])
def test_real_workers_have_private_cwd_and_devices(evaluator, tmp_path, width):
    evaluator = RolloutEvaluator(
        evaluator.configuration.model_copy(update={"devices": tuple(range(width))}),
        repository=evaluator.repository,
        workspace=evaluator.workspace,
    )

    async def run():
        directories = [tmp_path / "search/attempts" / str(i) for i in range(width + 1)]
        launches = []

        async def launch(directory):
            directory.mkdir(parents=True)
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", CHILD],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: launches.append(directory),
                )

        tasks = [asyncio.create_task(launch(directory)) for directory in directories]
        try:
            await wait_until(
                lambda: all(
                    (directory / "ready.json").exists()
                    for directory in directories[:width]
                )
            )
            rows = [
                json.loads((directory / "ready.json").read_text())
                for directory in directories[:width]
            ]
            assert {row["cwd"] for row in rows} == {
                str(directory) for directory in directories[:width]
            }
            assert {row["device"] for row in rows} == set(range(width))
            assert len(launches) == width
            assert not (directories[-1] / "ready.json").exists()
            assert not any(task.done() for task in tasks)
        finally:
            (tmp_path / "search/release").touch()
            await asyncio.gather(*tasks)
        assert all((directory / "done").exists() for directory in directories)
        assert all(
            json.loads((directory / "exit.json").read_text())["returncode"] == 0
            for directory in directories
        )

    asyncio.run(run())


def test_one_gpu_admits_multiple_environment_workers_with_shared_models(
    evaluator, tmp_path
):
    payload = evaluator.configuration.model_dump(mode="json")
    payload.update(
        devices=[0],
        rollouts_per_device=4,
        shared_inference_services=[
            {
                "device": 0,
                "vla_endpoint": "http://127.0.0.1:8101",
                "sam3_endpoint": "http://127.0.0.1:8102",
            }
        ],
    )
    evaluator = RolloutEvaluator(
        RolloutConfiguration.model_validate(payload),
        repository=evaluator.repository,
        workspace=evaluator.workspace,
    )

    async def run():
        directories = [tmp_path / "search/attempts" / str(i) for i in range(5)]

        async def launch(directory):
            directory.mkdir(parents=True)
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", CHILD],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: None,
                )

        tasks = [asyncio.create_task(launch(directory)) for directory in directories]
        try:
            await wait_until(
                lambda: all(
                    (directory / "ready.json").exists() for directory in directories[:4]
                )
            )
            rows = [
                json.loads((directory / "ready.json").read_text())
                for directory in directories[:4]
            ]
            assert {row["device"] for row in rows} == {0}
            assert {row["slot"] for row in rows} == {0, 1, 2, 3}
            assert not directories[-1].joinpath("ready.json").exists()
        finally:
            (tmp_path / "search/release").touch()
            await asyncio.gather(*tasks)

    asyncio.run(run())


def test_cancellation_joins_child_before_device_reuse(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/cancel"
        directory.mkdir(parents=True)

        async def launch():
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", CHILD],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: None,
                )

        task = asyncio.create_task(launch())
        await wait_until(lambda: (directory / "ready.json").exists())
        await wait_until(
            lambda: (
                json.loads((directory / "launch.json").read_text())["pid"] is not None
            )
        )
        pid = json.loads((directory / "launch.json").read_text())["pid"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not (directory / "done").exists()
        # A different evaluator has a fresh local queue; only the OS lease
        # can protect it from an orphan still running on the same device.
        fresh = RolloutEvaluator(
            evaluator.configuration,
            repository=evaluator.repository,
            workspace=evaluator.workspace,
        )
        async with fresh._slot() as (device, _, _):
            assert device == 0

    asyncio.run(run())


def test_normal_worker_exit_drains_owned_descendants(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/normal-descendant"
        directory.mkdir(parents=True)
        worker = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
"""
        async with evaluator._slot() as (device, slot, fd):
            await evaluator._launch(
                request(),
                directory,
                [sys.executable, "-c", worker],
                time.time(),
                device,
                slot,
                fd,
                lambda: None,
            )
        child_pid = int((directory / "child.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert json.loads((directory / "exit.json").read_text())["returncode"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("worker_exited", [False, True])
def test_recovery_rejects_live_process_group(
    evaluator, tmp_path, monkeypatch, worker_exited
):
    directory = tmp_path / "search/attempts/recovery"
    directory.mkdir(parents=True)
    (directory / "launch.json").write_text(
        json.dumps({"request": request().model_dump(mode="json"), "pid": 12345})
    )
    if worker_exited:
        (directory / "exit.json").write_text(json.dumps({"returncode": 0}))
    probes = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: probes.append((pid, sig)))

    async def run():
        with pytest.raises(EvolutionIncomplete, match="process group still exists"):
            await evaluator.recover(
                request(),
                baseline().model_copy(update={"record_id": request().bundle_id}),
                directory,
                before_launch=lambda: pytest.fail("relaunch"),
            )

    asyncio.run(run())
    assert probes == [(12345, 0)]
    assert not (directory / "result.json").exists()


def test_cancellation_drains_owned_descendants(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/cancel-descendant"
        directory.mkdir(parents=True)
        worker = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
Path('ready').touch()
time.sleep(60)
"""

        async def launch():
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", worker],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: None,
                )

        task = asyncio.create_task(launch())
        await wait_until(lambda: (directory / "ready").exists())
        child_pid = int((directory / "child.pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)

    asyncio.run(run())


def test_inherited_device_lease_survives_parent_exit(evaluator, tmp_path):
    directory = tmp_path / "search/attempts/orphan"
    directory.mkdir(parents=True)
    lock_path = tmp_path / "search/.scratch/runtime/devices/cuda-0-slot-0.lock"
    lock_path.parent.mkdir(parents=True)
    release = directory / "release"
    orphan = """import sys,time
from pathlib import Path
end=time.monotonic()+5
while not Path(sys.argv[1]).exists() and time.monotonic()<end:
    time.sleep(0.01)
"""
    parent = """import fcntl,os,subprocess,sys
fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX)
subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[3]],pass_fds=(fd,),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", parent, str(lock_path), orphan, str(release)], check=True
    )

    async def run():
        try:
            with pytest.raises(EvolutionIncomplete, match="still owned"):
                async with evaluator._slot():
                    raise AssertionError("orphan lease was released prematurely")
        finally:
            release.touch()

            def unlocked():
                with lock_path.open("a") as stream:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except BlockingIOError:
                        return False

            await wait_until(unlocked)

    asyncio.run(run())


def test_spawn_failure_is_durable_and_does_not_require_relaunch(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/spawn_error"
        directory.mkdir(parents=True)
        launches = []
        async with evaluator._slot() as (device, slot, fd):
            await evaluator._launch(
                request(),
                directory,
                [str(tmp_path / "missing-executable")],
                time.time(),
                device,
                slot,
                fd,
                lambda: launches.append(1),
            )
        exit_record = json.loads((directory / "exit.json").read_text())
        assert "FileNotFoundError" in exit_record["spawn_error"]
        assert launches == [1]
        record = evaluator._close(request(), directory)
        assert record.failure_origin == "infrastructure"
        assert record.usage.physical_steps == 0
        assert record.usage.physical_steps_complete
        assert record.usage.model_calls == 0
        assert record.usage.cost_usd == 0
        assert record.usage.wall_time_s >= 0

    asyncio.run(run())


@pytest.mark.parametrize("devices", [[], [0, 0], [True], [-1], [0.5]])
def test_device_pool_rejects_ambiguous_values(evaluator, devices):
    payload = evaluator.configuration.model_dump(mode="json")
    payload["devices"] = devices
    with pytest.raises(ValueError):
        RolloutConfiguration.model_validate(payload)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rollouts_per_device_requires_a_positive_integer(evaluator, value):
    payload = evaluator.configuration.model_dump(mode="json")
    payload["rollouts_per_device"] = value
    with pytest.raises(ValueError):
        RolloutConfiguration.model_validate(payload)


def test_multiple_rollouts_require_complete_shared_service_mapping(evaluator):
    payload = evaluator.configuration.model_dump(mode="json")
    payload["rollouts_per_device"] = 2
    with pytest.raises(ValueError, match="require shared inference services"):
        RolloutConfiguration.model_validate(payload)

    payload["shared_inference_services"] = [
        {
            "device": 0,
            "vla_endpoint": "http://127.0.0.1:8101",
            "sam3_endpoint": "http://127.0.0.1:8102",
        }
    ]
    with pytest.raises(ValueError, match="complete declared device pool"):
        RolloutConfiguration.model_validate(payload)


def test_shared_service_endpoints_are_runner_owned_deployment_inputs(tmp_path):
    runner = recipe_fixture(tmp_path)[2]
    payload = runner.model_dump(mode="json")
    payload["shared_inference_services"] = [
        {
            "device": device,
            "vla_endpoint": f"http://127.0.0.1:{8100 + 2 * device}",
            "sam3_endpoint": f"http://127.0.0.1:{8101 + 2 * device}",
        }
        for device in payload["devices"]
    ]
    runner = RolloutConfiguration.model_validate(payload)
    arguments = preflight(runner)
    assert arguments[-4:] == [
        "--vla-endpoint",
        "http://127.0.0.1:8100",
        "--sam3-endpoint",
        "http://127.0.0.1:8101",
    ]


@pytest.mark.parametrize(
    "flag", ["--cuda-device", "--env-endpoint", "--vla-endpoint", "--sam3-endpoint"]
)
def test_parallel_recipe_cannot_share_or_override_owned_services(evaluator, flag):
    payload = evaluator.configuration.model_dump(mode="json")
    payload["arguments"] += [flag, "0"]
    with pytest.raises(ValueError, match="runner-owned"):
        preflight(RolloutConfiguration.model_validate(payload))


def test_budget_rejection_leaves_no_launch_receipt(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/budget"
        directory.mkdir(parents=True)

        def reject():
            raise EvolutionIncomplete("budget exhausted")

        async with evaluator._slot() as (device, slot, fd):
            with pytest.raises(EvolutionIncomplete, match="budget exhausted"):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", CHILD],
                    time.time(),
                    device,
                    slot,
                    fd,
                    reject,
                )
        assert not (directory / "launch.json").exists()
        assert not (directory / "exit.json").exists()
        assert not (directory / "ready.json").exists()

    asyncio.run(run())


@pytest.mark.parametrize("extra", [("--robot", "libero"), ("--robot=libero",)])
def test_recipe_rejects_duplicate_adapter(evaluator, extra):
    payload = evaluator.configuration.model_dump(mode="json")
    payload["arguments"] += list(extra)
    with pytest.raises(ValueError, match="duplicate fixed rollout argument"):
        preflight(RolloutConfiguration.model_validate(payload))


def test_service_inherits_lease_after_worker_death(evaluator, tmp_path):
    directory = tmp_path / "search/attempts/services"
    directory.mkdir(parents=True)
    lock_path = tmp_path / "search/.scratch/runtime/devices/cuda-0-slot-0.lock"
    lock_path.parent.mkdir(parents=True)
    release = directory / "release"
    child = """import json,os,sys,time
from pathlib import Path
Path("service.json").write_text(json.dumps({"cwd":str(Path.cwd()),"lease":os.fstat(int(os.environ["REWEAVE_ROLLOUT_LEASE_FD"])).st_ino}))
end=time.monotonic()+5
while not Path(sys.argv[1]).exists() and time.monotonic()<end:
    time.sleep(0.01)
"""
    parent = """import fcntl,os,sys,time
from pathlib import Path
from reweave.utils.daemon import ProcessDaemon
fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX)
os.environ["REWEAVE_ROLLOUT_LEASE_FD"]=str(fd)
daemon=ProcessDaemon("service",[sys.executable,"-c",sys.argv[2],sys.argv[3]],cwd=sys.argv[4])
daemon.start()
end=time.monotonic()+5
while not (Path(sys.argv[4])/"service.json").exists() and time.monotonic()<end:
    time.sleep(0.01)
os._exit(0)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            parent,
            str(lock_path),
            child,
            str(release),
            str(directory),
        ],
        check=True,
    )

    async def run():
        try:
            row = json.loads((directory / "service.json").read_text())
            assert row["cwd"] == str(directory)
            assert row["lease"] == lock_path.stat().st_ino
            with pytest.raises(EvolutionIncomplete, match="still owned"):
                async with evaluator._slot():
                    raise AssertionError("service lost lease when worker exited")
        finally:
            release.touch()

            def unlocked():
                with lock_path.open("a") as stream:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except BlockingIOError:
                        return False

            await wait_until(unlocked)

    asyncio.run(run())


@pytest.mark.parametrize("spawn_fails", [False, True])
def test_cancel_while_spawning_retains_completion(
    evaluator, tmp_path, monkeypatch, spawn_fails
):
    async def run():
        directory = tmp_path / "search/attempts/spawning"
        directory.mkdir(parents=True)
        original = asyncio.create_subprocess_exec
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_spawn(*args, **kwargs):
            entered.set()
            await release.wait()
            if spawn_fails:
                raise OSError("spawn failed after cancellation")
            return await original(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)

        async def launch():
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", CHILD],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: None,
                )

        task = asyncio.create_task(launch())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = json.loads((directory / "exit.json").read_text())
        assert ("spawn_error" in record) == spawn_fails
        assert not (directory / "done").exists()

    asyncio.run(run())


def test_noncooperative_worker_is_killed_and_joined(evaluator, tmp_path):
    async def run():
        directory = tmp_path / "search/attempts/stubborn"
        directory.mkdir(parents=True)
        child = "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('ready').touch(); time.sleep(60)"

        async def launch():
            async with evaluator._slot() as (device, slot, fd):
                await evaluator._launch(
                    request(),
                    directory,
                    [sys.executable, "-c", child],
                    time.time(),
                    device,
                    slot,
                    fd,
                    lambda: None,
                )

        task = asyncio.create_task(launch())
        await wait_until(lambda: (directory / "ready").exists())
        pid = json.loads((directory / "launch.json").read_text())["pid"]
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 20)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert json.loads((directory / "exit.json").read_text())["returncode"] < 0

    asyncio.run(run())


def test_lost_ack_without_final_transcript_is_not_reset_as_infrastructure(
    evaluator, tmp_path
):
    directory = tmp_path / "unknown_attempt"
    episode = directory / "episode"
    episode.mkdir(parents=True)
    (directory / "exit.json").write_text(json.dumps({"elapsed_s": 3, "returncode": 1}))
    (episode / "execution_trace.jsonl").write_text(
        json.dumps({"event_type": "control_step"})
        + "\n"
        + json.dumps({"event_type": "action_outcome_unknown"})
        + "\n"
    )
    with pytest.raises(EvolutionIncomplete, match="never automatically rerun"):
        evaluator._close(request(), directory)
    assert not (directory / "result.json").exists()
    assert (episode / "execution_trace.jsonl").is_file()


@pytest.mark.parametrize(
    "termination,unknown_action,expected",
    [
        ("action_outcome_unknown", True, "unknown"),
        ("api_error", False, "infrastructure"),
        ("api_error", True, None),
        ("action_outcome_unknown", False, None),
    ],
)
def test_finalized_rollout_crosschecks_unknown_action_evidence(
    evaluator, tmp_path, termination, unknown_action, expected
):
    from reweave.evolution.contracts import Instance

    directory = tmp_path / "finalized_attempt"
    episode = directory / "episode"
    context = episode / "harness_context"
    context.mkdir(parents=True)
    initialization = {"suite": "fixture", "task": 0, "seed": 1, "reset_state_id": 0}
    evaluation = request().model_copy(
        update={
            "instance": Instance(
                record_id=new_record_id(),
                task="libero:fixture:0",
                initialization=initialization,
            )
        }
    )
    (context / "planner_context.json").write_text("{}")
    (context / "capability_catalog.json").write_text("{}")
    (directory / "exit.json").write_text(json.dumps({"elapsed_s": 3, "returncode": 0}))
    rows = [{"event_type": "control_step"}]
    if unknown_action:
        rows.append({"event_type": "action_outcome_unknown"})
    (episode / "execution_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (episode / "transcript_test.json").write_text(
        json.dumps(
            {
                "run_configuration": {"robot_name": "libero", **initialization},
                "bundle_id": evaluation.bundle_id,
                "termination_class": termination,
                "api_error_retryable": (True if termination == "api_error" else None),
                "official_success": False,
                "stats": {"requests": 2},
            }
        )
    )
    if expected is None:
        with pytest.raises(ValueError, match="classification disagree"):
            evaluator._close(evaluation, directory)
    else:
        record = evaluator._close(evaluation, directory)
        assert record.failure_origin == expected
        assert record.usage.physical_steps_complete == (not unknown_action)
        assert record.usage.model_calls is None
        assert record.episode is not None
        record.verify()


def test_campaign_device_selection_preserves_manifest_and_pool(evaluator):
    from reweave.cli.evolve import select_runner_device

    original = evaluator.configuration
    selected = select_runner_device(original, 3)
    assert selected.devices == (3,)
    assert original.devices == (0, 1, 2, 3)
    assert selected.runtime_manifest == original.runtime_manifest
    assert selected.arguments == original.arguments
    with pytest.raises(ValueError, match="published runner pool"):
        select_runner_device(original, 4)


def test_campaign_pool_releases_online_slot_before_storage(
    evaluator, monkeypatch, tmp_path
):
    root = tmp_path
    (root / "campaign_state.json").write_text("{}")
    (root / "scheduler_policy.json").write_text('{"mode":"episode_pool"}')
    directory = evaluator.workspace / "attempt"
    directory.mkdir(parents=True)
    (directory / "launch.json").write_text("{}")

    async def run(*args, **kwargs):
        with (root / "scheduler/online/slot-00.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    async def storage(path):
        with (root / "scheduler/online/slot-00.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(evaluator, "_run", run)
    monkeypatch.setattr(evaluator, "_finalize_storage", storage)
    monkeypatch.setattr(evaluator, "_close", lambda *a: "closed")
    assert (
        asyncio.run(evaluator(None, None, directory, before_launch=lambda: None))
        == "closed"
    )


def test_parked_worker_recovery_preserves_launch_without_replaying_physical(
    evaluator, monkeypatch
):
    directory = evaluator.workspace / "attempt"
    directory.mkdir(parents=True)
    evaluation = request()
    original = {"request": evaluation.model_dump(mode="json"), "pid": None}
    (directory / "launch.json").write_text(json.dumps(original))
    (directory / "admission.json").write_text('{"status":"waiting"}')
    (directory / "exit.json").write_text('{"returncode":-2}')

    async def admitted(self, *args, **kwargs):
        assert not (directory / "launch.json").exists()
        history = list((directory / "admission_history").glob("*/launch.json"))
        assert len(history) == 1 and json.loads(history[0].read_text()) == original
        return "admitted"

    monkeypatch.setattr(RolloutEvaluator, "__call__", admitted)
    assert (
        asyncio.run(
            evaluator.recover(
                evaluation,
                baseline().model_copy(update={"record_id": evaluation.bundle_id}),
                directory,
                before_launch=lambda: None,
            )
        )
        == "admitted"
    )
