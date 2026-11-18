# Modified for the ReWeave namespace.
"""Fixed command runner: isolated episodes, complete originals, explicit recovery."""

from __future__ import annotations

import asyncio
import fcntl
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from reweave.evolution.bundle import AgentBundle, Asset, new_record_id
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvaluationRequest,
    EvidenceFile,
    EvolutionIncomplete,
    Instance,
    Usage,
)
from reweave.evolution.evidence.records import (
    REQUIRED_DECISION_EVIDENCE,
    inspect_episode,
    inventory_episode,
)
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.processes import drain_process_group
from reweave.evolution.runtime_manifest import (
    RUNNER_OWNED_ARGUMENTS,
    RuntimeManifest,
    execution_configuration,
    validate_budget_arguments,
    validate_configuration,
)
from reweave.planner.base import ACTION_OUTCOME_UNKNOWN, API_ERROR
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json

# Exact initialization fields per robot, in their recorded transcript order.
_INITIALIZATION_FIELDS = {
    "libero": ("LIBERO", ("suite", "task", "seed", "reset_state_id")),
    "robodojo": ("RoboDojo", ("task", "layout_set", "layout")),
}


def _initialization_fields(robot: str) -> tuple[str, tuple[str, ...]]:
    if robot not in _INITIALIZATION_FIELDS:
        raise ValueError(f"unsupported evolution robot: {robot}")
    return _INITIALIZATION_FIELDS[robot]


class SharedInferenceServices(Asset):
    """Long-lived model services shared by rollout processes on one GPU."""

    device: Annotated[int, Field(ge=0, strict=True)]
    vla_endpoint: str = Field(min_length=1)
    sam3_endpoint: str | None = Field(
        default=None,
        description="Shared SAM3 endpoint for LIBERO or RoboDojo.",
    )
    graspgen_config: str | None = Field(
        default=None,
        description="Absolute Host YAML/JSON path binding this device's GraspGen-X service and calibrated grippers for LIBERO or RoboDojo. All devices must agree on non-route configuration.",
    )

    @field_validator("vla_endpoint", "sam3_endpoint")
    @classmethod
    def _endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip():
            raise ValueError("service endpoints must be non-empty and unpadded")
        return value

    @field_validator("graspgen_config")
    @classmethod
    def _graspgen_path(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip() or not Path(value).is_absolute()
        ):
            raise ValueError("shared GraspGen-X config requires an absolute Host path")
        return value


class RolloutConfiguration(Asset):
    """Host-owned execution recipe. Seed/task/output are supplied by the manifest."""

    devices: tuple[Annotated[int, Field(ge=0, strict=True)], ...] = Field(
        min_length=1,
        description="Distinct physical GPUs available to the rollout pool.",
    )

    @field_validator("devices")
    @classmethod
    def _devices(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("devices must be distinct non-negative integers")
        return values

    rollouts_per_device: Annotated[int, Field(ge=1, strict=True)] = Field(
        default=1,
        description="Independent environment processes admitted on each physical GPU.",
    )
    shared_inference_services: tuple[SharedInferenceServices, ...] = Field(
        default=(),
        description="Per-device long-lived inference endpoints; required when a GPU admits multiple rollouts.",
    )

    @model_validator(mode="after")
    def _parallel_topology(self):
        service_devices = tuple(item.device for item in self.shared_inference_services)
        if len(set(service_devices)) != len(service_devices):
            raise ValueError("shared inference services must name each device once")
        if self.shared_inference_services and set(service_devices) != set(self.devices):
            raise ValueError(
                "shared inference services must cover the complete declared device pool"
            )
        if self.rollouts_per_device > 1 and not self.shared_inference_services:
            raise ValueError(
                "multiple rollouts per device require shared inference services"
            )
        graspgen = [
            item.graspgen_config is not None for item in self.shared_inference_services
        ]
        if any(graspgen) and not all(graspgen):
            raise ValueError("shared GraspGen-X configuration must cover every device")
        return self

    def services_for(self, device: int) -> SharedInferenceServices | None:
        return next(
            (
                services
                for services in self.shared_inference_services
                if services.device == device
            ),
            None,
        )

    arguments: tuple[str, ...] = Field(
        description="Fixed Harness CLI arguments; no shell evaluation or per-instance overrides."
    )
    memory_snapshot: str = Field(
        description="Directory containing the immutable initial memory corpus, including an explicitly empty corpus."
    )
    runtime_manifest: RuntimeManifest = Field(
        description="Complete verified host/execution inputs captured with the baseline."
    )
    episode_limits: EpisodeLimits = Field(
        description="Total L/W/controller operation limits, fixed for all candidates."
    )

    def build_arguments(
        self, instance: Instance, *, directory: Path, memory: Path, device: int
    ) -> list[str]:
        """Build the exact worker argv from a real manifest instance."""
        initialization = instance.initialization
        robot = instance.task.partition(":")[0]
        label, fields = _initialization_fields(robot)
        if set(initialization) != set(fields):
            raise ValueError(
                f"{label} initialization requires exact {'/'.join(fields)} fields"
            )
        values = tuple(initialization[key] for key in fields)
        if robot == "libero":
            suite, task, seed, reset_state_id = values
            if not isinstance(suite, str) or not suite:
                raise ValueError("LIBERO suite must be a non-empty string")
            if any(
                type(value) is not int or value < 0
                for value in (task, seed, reset_state_id)
            ):
                raise ValueError(
                    "LIBERO reset identifiers must be non-negative integers"
                )
            if instance.task != f"libero:{suite}:{task}":
                raise ValueError(
                    "task identity must equal canonical libero:<suite>:<task>"
                )
            task_arguments = [
                "--suite",
                suite,
                "--task",
                str(task),
                "--seed",
                str(seed),
                "--reset-state-id",
                str(reset_state_id),
            ]
        elif robot == "robodojo":
            task, layout_set, layout = values
            if (
                not isinstance(task, str)
                or not task
                or task != task.strip()
                or Path(task).name != task
                or task in {".", ".."}
            ):
                raise ValueError(
                    "RoboDojo task must be a nonempty configuration basename"
                )
            if type(layout) is not int or layout < 0:
                raise ValueError("RoboDojo layout must be a nonnegative integer")
            if type(layout_set) is not int or layout_set < 0:
                raise ValueError("RoboDojo layout_set must be a nonnegative integer")
            if instance.task != f"robodojo:{task}":
                raise ValueError("task identity must equal canonical robodojo:<task>")
            from reweave.benchmarks.robodojo.tasks import native_step_limit

            physical_steps = self.runtime_manifest.conditions.physical_steps
            if physical_steps != native_step_limit(task):
                raise ValueError(
                    f"{task} physical budget is the official step_lim="
                    f"{native_step_limit(task)}; configured {physical_steps}"
                )
            task_arguments = [
                "--task",
                task,
                "--layout-set",
                str(layout_set),
                "--layout",
                str(layout),
            ]
        if type(device) is not int or device not in self.devices:
            raise ValueError("rollout device must belong to its declared pool")
        arguments = [
            *self.arguments,
            "--cuda-device",
            str(device),
            "--agent-bundle",
            str(directory / "bundle.json"),
            "--kernel-limits",
            str(directory / "kernel.json"),
            "--runtime-manifest",
            str(directory / "runtime.json"),
            *task_arguments,
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory),
            "--output-dir",
            str(directory / "episode"),
        ]
        services = self.services_for(device)
        if services is not None:
            arguments.extend(["--vla-endpoint", services.vla_endpoint])
            if services.sam3_endpoint is None:
                raise ValueError(f"{robot} shared inference requires a SAM3 endpoint")
            arguments.extend(["--sam3-endpoint", services.sam3_endpoint])
            if services.graspgen_config is not None:
                arguments.extend(["--graspgen-config", services.graspgen_config])
        return arguments

    def validate_arguments(
        self,
        instance: Instance,
        *,
        directory: Path,
        memory: Path,
        device: int,
        repository: Path | None = None,
    ) -> list[str]:
        """Preflight the exact CLI and frozen configuration, without starting services."""
        if repository is not None:
            # The selected physical source owns its CLI defaults. The offline
            # Host may evolve independently and must not reinterpret those defaults.
            payload = {
                "configuration": self.model_dump(mode="json"),
                "instance": instance.model_dump(mode="json"),
                "directory": str(directory),
                "memory": str(memory),
                "device": device,
            }
            code = """import json, sys
from pathlib import Path
from reweave.evolution.contracts import Instance
from reweave.evolution.search.rollout import RolloutConfiguration
p = json.load(sys.stdin)
arguments = RolloutConfiguration.model_validate(p['configuration']).validate_arguments(
    Instance.model_validate(p['instance']), directory=Path(p['directory']),
    memory=Path(p['memory']), device=p['device'])
print(json.dumps(arguments))
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=repository,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repository),
                    "REWEAVE_REPO_ROOT": str(repository),
                },
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=60,
            )
            if result.returncode:
                raise ValueError(
                    "physical runtime CLI preflight failed: " + result.stderr[-4000:]
                )
            return json.loads(result.stdout)
        from reweave.cli.main import parse_cli_arguments

        arguments = self.build_arguments(
            instance, directory=directory, memory=memory, device=device
        )
        diagnostics = io.StringIO()
        try:
            with redirect_stderr(diagnostics), redirect_stdout(diagnostics):
                parser, args, robot_spec = parse_cli_arguments(arguments)
        except SystemExit as error:
            raise ValueError(
                f"invalid fixed rollout CLI arguments (exit {error.code}): {diagnostics.getvalue().strip()}"
            ) from error
        owned = set(RUNNER_OWNED_ARGUMENTS)
        if any(item.graspgen_config for item in self.shared_inference_services):
            owned.add("graspgen_config")
        option_actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        seen = set()
        for token in self.arguments:
            option = token.split("=", 1)[0]
            action = option_actions.get(option)
            if action is None and token.startswith("-") and not token.startswith("--"):
                action = option_actions.get(token[:2])
                if action is not None and action.nargs == 0:
                    raise ValueError(
                        f"packed short options are not a fixed recipe option: {token}"
                    )
            if action is None:
                continue
            if action.dest in owned:
                raise ValueError(
                    f"fixed rollout arguments cannot override runner-owned field {token}"
                )
            if action.dest in seen:
                raise ValueError(
                    f"duplicate fixed rollout argument {token} ({action.dest})"
                )
            seen.add(action.dest)
        if (
            "robot_name" not in seen
            or args.robot_name != instance.task.partition(":")[0]
        ):
            raise ValueError(
                "rollout recipe must declare the instance robot exactly once"
            )
        if "planner" not in seen or args.planner != "native":
            raise ValueError(
                "rollout recipe must declare --planner native exactly once"
            )
        if (
            args.robot_name == "robodojo"
            and self.runtime_manifest.conditions.wait_steps != 0
        ):
            raise ValueError("RoboDojo resets own settling; wait_steps must be zero")
        robot_spec.parse_config(args)
        limits = {"episode": self.episode_limits.model_dump(mode="json")}
        validate_budget_arguments(self.runtime_manifest.conditions, vars(args), limits)
        validate_configuration(
            self.runtime_manifest.configuration,
            execution_configuration(vars(args), limits),
        )
        return arguments


class RolloutEvaluator:
    """Execute the real existing CLI without allowing candidates to classify failures."""

    def __init__(
        self, configuration: RolloutConfiguration, *, repository: Path, workspace: Path
    ):
        self.configuration, self.repository = configuration, repository.resolve()
        self.workspace = workspace.resolve()
        self._slot_changed = asyncio.Condition()
        self._occupied_slots = {device: set() for device in configuration.devices}

    @asynccontextmanager
    async def _slot(self):
        async with self._slot_changed:
            await self._slot_changed.wait_for(
                lambda: any(
                    len(slots) < self.configuration.rollouts_per_device
                    for slots in self._occupied_slots.values()
                )
            )
            # Spread over GPUs before adding another process to an occupied GPU.
            # Recompute on every acquisition, including after uneven completion.
            device = min(
                self._occupied_slots, key=lambda d: len(self._occupied_slots[d])
            )
            occupied = self._occupied_slots[device]
            slot = next(
                index
                for index in range(self.configuration.rollouts_per_device)
                if index not in occupied
            )
            occupied.add(slot)
        try:
            locks = self.workspace / ".scratch/runtime/devices"
            locks.mkdir(parents=True, exist_ok=True)
            with (locks / f"cuda-{device}-slot-{slot}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise EvolutionIncomplete(
                        f"CUDA device {device} slot {slot} is still owned by another rollout; refusing overlap"
                    ) from error
                # The child inherits the descriptor. Parent death must not make
                # this slot available while its old worker is still alive.
                yield device, slot, lock.fileno()
        finally:
            async with self._slot_changed:
                self._occupied_slots[device].remove(slot)
                self._slot_changed.notify()

    def _memory(self, directory: Path) -> Path:
        source = Path(self.configuration.memory_snapshot).resolve()
        destination = directory / "memory"
        if destination.exists():
            if destination.is_symlink():
                raise ValueError("private memory cannot be a symbolic link")
            return destination
        shutil.copytree(source, destination)
        return destination

    async def __call__(
        self,
        request: EvaluationRequest,
        bundle: AgentBundle,
        directory: Path,
        *,
        before_launch: Callable[[], None],
    ) -> EvaluationRecord:
        from reweave.evolution.infrastructure.campaign_slots import (
            campaign_root,
            enabled,
            episode_slot,
        )

        root = campaign_root(self.workspace)
        if root is not None and enabled(root):
            # The shared pool admits storage separately from physical slots.
            async with episode_slot(
                root, directory, devices=self.configuration.devices
            ) as (device, slot, lease_fd):
                await self._run(
                    request,
                    bundle,
                    directory,
                    device=device,
                    slot=slot,
                    lease_fd=lease_fd,
                    before_launch=before_launch,
                )
            return await self._settle(request, directory)
        async with self._slot() as (device, slot, lease_fd):
            await self._run(
                request,
                bundle,
                directory,
                device=device,
                slot=slot,
                lease_fd=lease_fd,
                before_launch=before_launch,
            )
            return await self._settle(request, directory)

    async def _settle(
        self, request: EvaluationRequest, directory: Path
    ) -> EvaluationRecord:
        published = await run_blocking(self._published_result, request, directory)
        if published is not None:
            return published
        await self._finalize_storage(directory)
        return await run_blocking(self._close_once, request, directory)

    def _published_result(self, request, directory):
        path = directory / "result.json"
        if path.is_symlink():
            raise ValueError("published rollout result must be a regular file")
        if not path.exists():
            return None
        record = EvaluationRecord.model_validate_json(path.read_text())
        if record.request != request:
            raise ValueError("published rollout result request changed")
        record.verify()
        return record

    def _close_once(self, request, directory):
        # Lock the existing immutable launch inode: no extra evidence file and
        # no concurrent closer may allocate a replacement receipt identity.
        with (directory / "launch.json").open("rb") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            published = self._published_result(request, directory)
            return (
                published if published is not None else self._close(request, directory)
            )

    async def _run(
        self,
        request: EvaluationRequest,
        bundle: AgentBundle,
        directory: Path,
        *,
        device: int,
        slot: int,
        lease_fd: int,
        before_launch: Callable[[], None],
    ) -> None:
        started = time.time()
        manifest = self.configuration.runtime_manifest
        if request.bundle_id != bundle.record_id:
            raise ValueError("evaluation bundle differs from its request")
        if request.conditions != manifest.conditions:
            raise ValueError(
                "evaluation conditions differ from the actual runner configuration"
            )
        memory = self._memory(directory)
        arguments = await run_blocking(
            self.configuration.validate_arguments,
            request.instance,
            directory=directory,
            memory=memory,
            device=device,
            repository=self.repository,
        )
        bundle_path = directory / "bundle.json"
        limits_path = directory / "kernel.json"
        atomic_json(bundle_path, bundle.model_dump(mode="json"))
        manifest_path = directory / "runtime.json"
        atomic_json(manifest_path, manifest.model_dump(mode="json"))
        atomic_json(
            limits_path,
            {
                "conditions": manifest.conditions.model_dump(mode="json"),
                "episode": self.configuration.episode_limits.model_dump(mode="json"),
            },
        )
        command = [
            sys.executable,
            "-m",
            "reweave.evolution.search.rollout",
            "--worker",
            str(directory),
        ]
        atomic_json(directory / "arguments.json", arguments)
        await self._launch(
            request,
            directory,
            command,
            started,
            device,
            slot,
            lease_fd,
            before_launch,
        )

    async def _launch(
        self,
        request,
        directory,
        command,
        started,
        device,
        slot,
        lease_fd,
        before_launch,
    ):
        from reweave.evolution.infrastructure.campaign_slots import (
            campaign_root,
            enabled,
        )
        from reweave.utils.gpu_slots import encode_slot

        root = campaign_root(directory)
        pooled = root is not None and enabled(root)
        launch = {
            "request": request.model_dump(mode="json"),
            "command": command,
            "started": started,
            "pid": None,
            "status": "prepared",
            "cuda_device": device,
            "device_slot": slot,
        }
        process = None

        def spawn_failed(error: OSError) -> None:
            atomic_json(
                directory / "exit.json",
                {
                    "returncode": None,
                    "spawn_error": f"{type(error).__name__}: {error}",
                    "elapsed_s": time.time() - started,
                },
            )

        with (directory / "runner.log").open("wb") as log:
            before_launch()
            atomic_json(directory / "launch.json", launch)
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    cwd=directory,
                    stdout=log,
                    env={
                        **os.environ,
                        "REWEAVE_REPO_ROOT": str(self.repository),
                        "REWEAVE_ROLLOUT_LEASE_FD": str(lease_fd),
                        **(
                            {"REWEAVE_CAMPAIGN_SLOT": str(encode_slot(device, slot))}
                            if pooled
                            else {}
                        ),
                        "PYTHONPATH": os.pathsep.join(
                            filter(
                                None,
                                (str(self.repository), os.environ.get("PYTHONPATH")),
                            )
                        ),
                    },
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lease_fd,),
                )
            )
            try:
                try:
                    process = await asyncio.shield(spawn)
                except OSError as error:
                    spawn_failed(error)
                    return
                except asyncio.CancelledError:
                    try:
                        process = await _drain(spawn)
                    except OSError as error:
                        spawn_failed(error)
                    raise
                launch.update(pid=process.pid, status="running")
                atomic_json(directory / "launch.json", launch)
                await process.wait()
                await drain_process_group(process)
            except BaseException:
                if process is not None:
                    await _drain(drain_process_group(process, terminate=True))
                raise
            finally:
                if (
                    process is not None
                    and process.returncode is not None
                    and not (directory / "exit.json").exists()
                ):
                    atomic_json(
                        directory / "exit.json",
                        {
                            "returncode": process.returncode,
                            "elapsed_s": time.time() - started,
                        },
                    )

    async def _finalize_storage(self, directory: Path) -> None:
        """Only ended RoboDojo output is eligible; failures never relaunch it."""
        output = directory / "episode"
        if (
            not (output / "server_evidence.json").exists()
            or (output / "episode_evidence.json").exists()
        ):
            return
        from reweave.evolution.evidence.sensor_retention import finalize_episode_sensors
        from reweave.evolution.infrastructure.campaign_slots import (
            campaign_root,
            enabled,
            episode_slot,
        )

        root = campaign_root(directory)
        try:
            if root is not None and enabled(root):
                async with episode_slot(
                    root, directory, resource="storage", capacity=4
                ):
                    await run_blocking(finalize_episode_sensors, directory)
            else:
                await run_blocking(finalize_episode_sensors, directory)
        except Exception as error:
            raise EvolutionIncomplete(
                "post-episode storage is incomplete; preserve the finished episode and recover storage only, never replay actions"
            ) from error

    def _close_partial(
        self,
        request: EvaluationRequest,
        directory: Path,
        *,
        infrastructure_retryable: bool,
    ) -> EvaluationRecord:
        """Retain incomplete startup/runner evidence without fabricating decisions."""
        output = directory / "episode"
        if not (directory / "exit.json").exists():
            raise EvolutionIncomplete(
                "runner completion is unknown; preserve launch/log and recover explicitly"
            )
        exit_record = json.loads((directory / "exit.json").read_text())
        never_started = "spawn_error" in exit_record
        if never_started and ((directory / "process.json").exists() or output.exists()):
            raise EvolutionIncomplete(
                "spawn failure contradicts recorded worker evidence; preserve the "
                "original attempt and reconcile explicitly, never automatically replay"
            )
        files = inventory_episode(directory)
        trace_path = output / "execution_trace.jsonl"
        observed_steps = 0
        unknown_action = False
        if trace_path.exists():
            # A missing newline alone does not invalidate a complete event.
            # An unparsable tail may hide an unacknowledged physical action.
            for line in trace_path.read_text().splitlines(keepends=True):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    if not line.endswith("\n"):
                        raise EvolutionIncomplete(
                            "incomplete final execution trace record may hide an unacknowledged action; preserve all evidence and reconcile explicitly, never automatically rerun the episode"
                        ) from error
                    raise
                event_type = event["event_type"]
                observed_steps += event_type == "control_step"
                unknown_action |= event_type == "action_outcome_unknown"
        if unknown_action:
            raise EvolutionIncomplete(
                "unacknowledged action lacks complete finalization evidence; restore finalization evidence, never automatically rerun the episode"
            )
        result = EvaluationRecord(
            record_id=new_record_id(),
            request=request,
            official_success=False,
            failure_origin="infrastructure",
            infrastructure_retryable=infrastructure_retryable,
            episode=None,
            usage=Usage(
                physical_steps=observed_steps,
                physical_steps_complete=never_started,
                # Visible native turns cannot certify hidden provider attempts.
                model_calls=0 if never_started else None,
                wall_time_s=exit_record["elapsed_s"],
                cost_usd=0 if never_started else None,
            ),
            evidence=tuple(
                EvidenceFile(path=str((directory / name).resolve()), **record)
                for name, record in files.items()
                if name != "result.json"
            ),
            initialization=request.instance.initialization,
        )
        atomic_json(directory / "result.json", result.model_dump(mode="json"))
        return result

    def _close(self, request: EvaluationRequest, directory: Path) -> EvaluationRecord:
        output = directory / "episode"
        transcripts = sorted(output.glob("transcript_*.json"))
        if len(transcripts) > 1:
            raise EvolutionIncomplete(
                "multiple rollout transcripts are ambiguous; preserve all evidence and reconcile explicitly, never automatically rerun the episode"
            )
        if not transcripts:
            return self._close_partial(
                request, directory, infrastructure_retryable=True
            )
        if not (directory / "exit.json").exists():
            raise EvolutionIncomplete(
                "runner completion timing is unavailable; preserve the complete transcript and reconcile the interrupted process"
            )
        transcript = json.loads(transcripts[0].read_text())
        config = transcript["run_configuration"]
        expected = request.instance.initialization
        robot = request.instance.task.partition(":")[0]
        if config["robot_name"] != robot:
            raise ValueError("actual rollout robot differs from its registered request")
        actual = {key: config[key] for key in _initialization_fields(robot)[1]}
        termination = transcript["termination_class"]
        success = transcript["official_success"]
        if not isinstance(success, bool):
            raise ValueError("official success verdict must be a boolean")
        retryable = transcript.get("api_error_retryable")
        if termination == API_ERROR:
            if not isinstance(retryable, bool):
                raise ValueError("API error transcript requires retry eligibility")
        elif retryable is not None:
            raise ValueError("only API error transcripts may declare retry eligibility")
        partial_startup = (
            termination == API_ERROR
            and not success
            and any(
                not (output / name).is_file() for name in REQUIRED_DECISION_EVIDENCE
            )
        )
        unbound_startup = partial_startup and transcript.get("bundle_id") is None
        if actual != expected or (
            not unbound_startup and transcript.get("bundle_id") != request.bundle_id
        ):
            raise ValueError(
                "actual rollout initialization or bundle differs from its registered request"
            )
        if partial_startup:
            return self._close_partial(
                request, directory, infrastructure_retryable=retryable
            )
        if robot == "robodojo":
            result_path = output / "result.json"
            if not result_path.is_file():
                raise ValueError("RoboDojo evaluator result is missing")
            cell = json.loads(result_path.read_text())
            if (
                any(cell.get(key) != value for key, value in actual.items())
                or cell.get("success_source") != "environment.is_success"
                or type(cell.get("valid")) is not bool
                or (
                    cell["valid"]
                    and (
                        type(cell.get("success")) is not bool
                        or cell["success"] != success
                    )
                )
                or (
                    not cell["valid"]
                    and (
                        termination not in {API_ERROR, ACTION_OUTCOME_UNKNOWN}
                        or cell.get("success") is not None
                        or success
                    )
                )
            ):
                raise ValueError(
                    "RoboDojo evaluator disagrees with rollout identity or official outcome"
                )
        evidence = inspect_episode(output)
        trace = [
            json.loads(line)
            for line in (output / "execution_trace.jsonl").read_text().splitlines()
        ]
        steps = sum(row["event_type"] == "control_step" for row in trace)
        unknown_action = any(
            row["event_type"] == "action_outcome_unknown" for row in trace
        )
        if unknown_action != (termination == ACTION_OUTCOME_UNKNOWN):
            raise ValueError(
                "unknown-action trace and termination classification disagree"
            )
        result = EvaluationRecord(
            record_id=new_record_id(),
            request=request,
            official_success=success,
            failure_origin="none"
            if success
            else "unknown"
            if unknown_action
            else "infrastructure"
            if termination == API_ERROR
            else "agent",
            infrastructure_retryable=retryable if termination == API_ERROR else None,
            usage=Usage(
                physical_steps=steps,
                physical_steps_complete=not unknown_action,
                model_calls=None,
                wall_time_s=json.loads((directory / "exit.json").read_text())[
                    "elapsed_s"
                ],
                cost_usd=None,
            ),
            evidence=tuple(
                EvidenceFile(path=str((output / name).resolve()), **record)
                for name, record in evidence.files.items()
            ),
            episode=evidence,
            initialization=actual,
        )
        result.verify()
        atomic_json(directory / "result.json", result.model_dump(mode="json"))
        return result

    async def verify_retry_closed(self, request, bundle, directory):
        """Certify closure of an existing retryable receipt without launching work."""

        def reject_launch():
            raise EvolutionIncomplete(
                "retry verification cannot launch missing original work"
            )

        record = await self.recover(
            request, bundle, directory, before_launch=reject_launch
        )
        if (
            record.failure_origin != "infrastructure"
            or not record.infrastructure_retryable
        ):
            raise EvolutionIncomplete(
                "only runner-classified retryable infrastructure can be retried"
            )
        if (directory / "episode/remote_environment.json").exists():
            closed = directory / "episode/remote_environment_closed.json"
            if not closed.exists():
                raise EvolutionIncomplete("remote environment closure is still unknown")
            receipt = json.loads(closed.read_text())
            if (
                receipt["status"] != "exited"
                or receipt["exit"].get("parent_closed") is not True
            ):
                raise EvolutionIncomplete("remote environment has not closed")
        return record

    async def recover(
        self,
        request: EvaluationRequest,
        bundle: AgentBundle,
        directory: Path,
        *,
        before_launch: Callable[[], None],
    ) -> EvaluationRecord:
        """Close completed physical work without starting another reset or rollout."""
        if request.bundle_id != bundle.record_id:
            raise ValueError("runner recovery bundle differs from its request")
        launch_path = directory / "launch.json"
        if not launch_path.exists():
            if (
                (directory / "episode").exists()
                or (directory / "process.json").exists()
                or (directory / "result.json").exists()
            ):
                raise EvolutionIncomplete(
                    "unregistered physical evidence exists; refusing to repeat initialization"
                )
            return await self(request, bundle, directory, before_launch=before_launch)
        launch = json.loads(launch_path.read_text())
        if launch["request"] != request.model_dump(mode="json"):
            raise ValueError("runner recovery request changed")
        process_path = directory / "process.json"
        pid = (
            json.loads(process_path.read_text())["pid"]
            if process_path.exists()
            else launch["pid"]
        )
        if pid is not None:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise EvolutionIncomplete(
                    "runner process group still exists; refusing to publish or repeat rollout"
                )
        published = await run_blocking(self._published_result, request, directory)
        if published is not None:
            return published
        admission = directory / "admission.json"
        if (
            admission.exists()
            and json.loads(admission.read_text()).get("status") == "waiting"
            and not process_path.exists()
            and not (directory / "episode").exists()
        ):
            # This worker parked before importing the runner: no reset, session,
            # or action could have occurred. Keep its launch evidence verbatim.
            history = directory / "admission_history" / str(time.time_ns())
            history.mkdir(parents=True)
            for name in ("launch.json", "exit.json", "admission.json", "runner.log"):
                path = directory / name
                if path.exists():
                    path.rename(history / name)
            return await self(request, bundle, directory, before_launch=before_launch)
        return await self._settle(request, directory)


async def _drain(work):
    """Join cleanup even if another cancellation arrives; caller re-raises its cancellation."""
    task = asyncio.ensure_future(work)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def worker(directory: Path) -> int:
    """Retain completion independently of the evolution-controller process lifetime."""
    from reweave.evolution.infrastructure.campaign_slots import wait_for_transition

    wait_for_transition(directory)
    launch = json.loads((directory / "launch.json").read_text())
    atomic_json(directory / "process.json", {"pid": os.getpid()})
    sys.argv = ["reweave", *json.loads((directory / "arguments.json").read_text())]
    result = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"rollout worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        from reweave.cli.main import main

        result = main()
        return result
    finally:
        atomic_json(
            directory / "exit.json",
            {"returncode": result, "elapsed_s": time.time() - launch["started"]},
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    sys.exit(worker(parser.parse_args().worker.resolve()))
