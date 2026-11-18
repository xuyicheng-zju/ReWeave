"""Own finite independent evolution chains through a shared episode pool."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from reweave.utils.persistence import atomic_json


class Campaign:
    """Own complete chains, recover interruptions with bounded backoff, isolate failures."""

    def __init__(
        self,
        directory,
        repository,
        assignments,
        *,
        owner_fd,
        runtime_source=None,
        child_env=None,
        recover=False,
        stagger_s=5,
        retry_delays=(10, 30, 90),
        max_active_tasks=None,
        task_filter=None,
        requeue_tasks=(),
    ):
        self.child_env = dict(child_env or {})
        self.directory, self.repository = Path(directory), Path(repository)
        self.runtime_source = (
            Path(runtime_source) if runtime_source is not None else self.repository
        )
        self.assignments = {item["task"]: item for item in assignments}
        self.task_filter = (
            frozenset(task_filter)
            if task_filter is not None
            else frozenset(self.assignments)
        )
        if not self.task_filter or not self.task_filter <= self.assignments.keys():
            raise ValueError(
                "task_filter must name a nonempty subset of published tasks"
            )
        requeue = frozenset(requeue_tasks)
        if requeue and (not recover or not requeue <= self.task_filter):
            raise ValueError(
                "Explicit requeue requires --recover and declared selected tasks"
            )
        self.owner_fd, self.stagger_s = owner_fd, stagger_s
        self.retry_delays = retry_delays
        if max_active_tasks is not None and (
            type(max_active_tasks) is not int or max_active_tasks < 1
        ):
            raise ValueError("max_active_tasks must be a positive integer")
        self.max_active_tasks = max_active_tasks
        self.retries = {}
        self.path = self.directory / "campaign_state.json"
        self.processes = {}
        self.quarantined = set()
        from reweave.evolution.infrastructure.campaign_slots import enabled

        self.episode_pool = enabled(self.directory)
        self.stopping = False
        self.next_start = 0.0
        if self.path.exists():
            if not recover:
                raise ValueError(
                    "campaign already exists; inspect and explicitly --recover"
                )
            self.state = json.loads(self.path.read_text())
            if self.state["assignments"] != assignments:
                raise ValueError("recovery cannot change published task identities")
            if any(
                self.state["tasks"][task]["status"]
                not in {"blocked", "interrupted", "completed"}
                for task in requeue
            ):
                raise ValueError(
                    "Explicit requeue may only select settled blocked/interrupted tasks"
                )
            supersessions = self.directory / "task_supersessions.json"
            replacements = (
                json.loads(supersessions.read_text()) if supersessions.exists() else {}
            )
            for task, row in self.state["tasks"].items():
                if task in replacements and row["status"] != "completed":
                    replacement = Path(replacements[task]).resolve()
                    provenance = json.loads(
                        (replacement / "rerun_provenance.json").read_text()
                    )
                    if task not in provenance["tasks"] or provenance.get(
                        "supersedes_campaign"
                    ) != str(self.directory.resolve()):
                        raise ValueError(
                            "Replacement must explicitly supersede this task"
                        )
                    row.update(status="blocked", superseded_by=str(replacement))
                    continue
                if row["status"] == "completed":
                    if not self._completed(task):
                        raise ValueError(
                            f"{task}: completed evolution receipt is missing"
                        )
                elif task in requeue:
                    previous = row["status"]
                    row["status"] = "completed" if self._completed(task) else "pending"
                    row.setdefault("requeues", []).append(
                        {
                            "at": time.time(),
                            "previous_status": previous,
                            "status": row["status"],
                        }
                    )
                elif not (
                    self.episode_pool
                    and row["status"] == "blocked"
                    and (
                        self.state.get("scheduler_mode") == "episode_pool"
                        or task
                        in json.loads(
                            (self.directory / "scheduler_policy.json").read_text()
                        ).get("blocked_tasks_preserved", [])
                    )
                ):
                    row["status"] = "pending"
        else:
            if requeue:
                raise ValueError("Explicit requeue requires an existing campaign")
            if (self.directory / "launch.json").exists():
                raise ValueError(
                    "existing legacy launch must be reconciled before queue migration"
                )
            self.state = {
                "assignments": assignments,
                "tasks": {
                    task: {"status": "pending", "slot": None, "launches": []}
                    for task in self.assignments
                },
            }
        if self.episode_pool:
            self.state["scheduler_mode"] = "episode_pool"
        self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def _completed(self, task):
        path = Path(self.assignments[task]["workspace"]) / "evolution.json"
        if not path.is_file():
            return False
        value = json.loads(path.read_text())
        return isinstance(value, dict) and value.get("completed") is True

    def stop(self, *_):
        if self.stopping:
            return
        self.stopping = True
        for child in self.processes.values():
            if child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass

    def _next(self, slot):
        pending = [
            task
            for task, row in self.state["tasks"].items()
            if row["status"] == "pending" and task in self.task_filter
        ]
        if self.episode_pool:
            return next(iter(pending), None)
        pinned = [task for task in pending if self.state["tasks"][task]["slot"] == slot]
        return next(
            iter(
                pinned
                or [
                    task
                    for task in pending
                    if self.state["tasks"][task]["slot"] is None
                ]
            ),
            None,
        )

    def _start(self, task, slot):
        item, row = self.assignments[task], self.state["tasks"][task]
        command = [
            sys.executable,
            "-u",
            "-m",
            "reweave.cli.evolve",
            "run",
            "--workspace",
            item["workspace"],
            "--runtime-source",
            str(self.runtime_source),
            "--device",
            str(slot // 4),
        ]
        if self.episode_pool:
            command = command[:-2]  # Each episode obtains its own physical device.
        if row["launches"]:
            command.append("--recover")
        launch = {
            "slot": None if self.episode_pool else slot,
            "command": command,
            "started_at": time.time(),
            "pid": None,
        }
        row.update(status="starting", slot=None if self.episode_pool else slot)
        row["launches"].append(launch)
        self.save()  # Persist identity before a child could start physical work.
        try:
            with (Path(item["workspace"]).parent / "driver.log").open(
                "ab", buffering=0
            ) as log:
                child = subprocess.Popen(
                    command,
                    cwd=self.repository,
                    env={
                        **os.environ,
                        **self.child_env,
                        "REWEAVE_REPO_ROOT": str(self.repository),
                        "PYTHONPATH": os.pathsep.join(
                            filter(
                                None,
                                [str(self.repository), os.environ.get("PYTHONPATH")],
                            )
                        ),
                        "REWEAVE_CAMPAIGN_SLOT": str(slot),
                        **({"REWEAVE_EPISODE_POOL": "1"} if self.episode_pool else {}),
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(self.owner_fd,),
                )
        except OSError as error:
            launch.update(exit_code=None, finished_at=time.time())
            self._retry_or_block(task, f"{type(error).__name__}: {error}")
        else:
            self.processes[task] = child
            launch["pid"] = child.pid
            row["status"] = "running"
            if self.stopping:
                child.send_signal(signal.SIGINT)
        self.save()
        print(
            json.dumps(
                {
                    "task": task,
                    "gpu": None if self.episode_pool else slot // 4,
                    "slot": None if self.episode_pool else slot,
                    "status": row["status"],
                }
            ),
            flush=True,
        )

    def _retry_or_block(self, task, error):
        row = self.state["tasks"][task]
        count = self.retries.get(task, 0)
        row["error"] = error
        if self.stopping and self.episode_pool:
            row["status"] = "interrupted"
        elif not self.stopping and count < len(self.retry_delays):
            self.retries[task] = count + 1
            row.update(
                status="retry_wait", retry_at=time.time() + self.retry_delays[count]
            )
        else:
            row["status"] = "blocked"
            if not self.episode_pool:
                self.quarantined.add(row["slot"])

    def tick(self):
        for task, child in list(self.processes.items()):
            code = child.poll()
            if code is None:
                continue
            row = self.state["tasks"][task]
            row["launches"][-1].update(exit_code=code, finished_at=time.time())
            try:
                completed = code == 0 and self._completed(task)
            except (OSError, ValueError):
                completed = False
            row["status"] = "completed" if completed else "blocked"
            if not completed:
                self._retry_or_block(
                    task, "driver failed or lacks a completed evolution receipt"
                )
            del self.processes[task]
            self.save()
            print(
                json.dumps({"task": task, "status": row["status"], "exit_code": code}),
                flush=True,
            )
        if self.stopping or time.monotonic() < self.next_start:
            return
        waiting = [
            row
            for task, row in self.state["tasks"].items()
            if row["status"] == "retry_wait" and task in self.task_filter
        ]
        for row in waiting:
            if row["retry_at"] <= time.time():
                row["status"] = "pending"
        if self.episode_pool:
            if (
                self.max_active_tasks is not None
                and len(self.processes) >= self.max_active_tasks
            ):
                return
            task = self._next(0)
            if task is not None:
                self._start(task, 0)
                self.next_start = time.monotonic() + self.stagger_s
            return
        occupied = {self.state["tasks"][task]["slot"] for task in self.processes}
        occupied.update(row["slot"] for row in waiting if row["status"] == "retry_wait")
        # Reserve interrupted tasks' slots so a fresh task cannot steal their owner.
        reserved = {
            row["slot"]
            for row in self.state["tasks"].values()
            if row["status"] == "pending" and row["slot"] is not None
        }
        # Spread initial starts over all cards; all slots consume the same queue.
        for slot in [gpu * 4 + lane for lane in range(4) for gpu in range(4)]:
            if slot in occupied or slot in self.quarantined:
                continue
            task = self._next(slot)
            if slot in reserved and (
                task is None or self.state["tasks"][task]["slot"] != slot
            ):
                continue
            if task is not None:
                self._start(task, slot)
                self.next_start = time.monotonic() + self.stagger_s
                return

    def run(self):
        try:
            while True:
                self.tick()
                pending_can_start = not self.stopping and any(
                    slot not in self.quarantined and self._next(slot) is not None
                    for slot in range(16)
                )
                retry_waiting = not self.stopping and any(
                    row["status"] == "retry_wait"
                    for task, row in self.state["tasks"].items()
                    if task in self.task_filter
                )
                if not self.processes and not pending_can_start and not retry_waiting:
                    break
                time.sleep(0.2)
        finally:
            # Unexpected scheduler errors must not orphan already launched drivers.
            if self.processes:
                self.stop()
                for child in self.processes.values():
                    child.wait()
        return (
            0
            if all(
                self.state["tasks"][task]["status"] == "completed"
                for task in self.task_filter
            )
            else 1
        )
