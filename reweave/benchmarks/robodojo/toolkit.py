# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""RoboDojo physical tools under the common episode execution lifecycle."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import RLock

from reweave.benchmarks.robodojo import tools
from reweave.benchmarks.robodojo.planner_observation import (
    PlannerObservation,
    contains_text,
)
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace
from reweave.runtime.capability import ModelInputRejected
from reweave.runtime.model_budget import ModelServiceFailure
from reweave.session.base import EnvState
from reweave.tools.toolkit import Toolkit
from reweave.utils.logging import get_output_dir
from reweave.utils.persistence import atomic_text


class RoboDojoState(EnvState):
    """Resolve named RGB artifacts from the exact retained server observation.

    Registering a state does not transfer pixels. Explicit image reads materialize
    only their selected camera; a later observation can never replace this ref.
    """

    def __init__(self, output_dir, *, env):
        self._env = env
        self._media_lock = RLock()
        super().__init__(output_dir)

    def register_observation(self, step, retained):
        record = self._record_for(step)
        ref = record.extras["observation_ref"]
        if not isinstance(retained, dict) or retained.get("observation_ref") != ref:
            raise ValueError(
                "retained observation does not match the recorded snapshot"
            )
        record.extras["server_observation"] = dict(retained)
        record.artifacts.update(artifact for _, artifact in tools.CAMERA_ARTIFACTS)
        if not self._step_open:
            self._write_manifest()

    def materialize_rgb(self, step, cameras):
        with self._media_lock:
            record = self._record_for(step)
            mapping = dict(tools.CAMERA_ARTIFACTS)
            if not cameras or any(camera not in mapping for camera in cameras):
                raise ValueError("RGB reads require known RoboDojo cameras")
            missing = [
                camera
                for camera in cameras
                if not self._artifact_file(mapping[camera], step).is_file()
            ]
            if not missing:
                return
            ref = record.extras["observation_ref"]
            images = self._env.read_rgb(ref, cameras=missing)
            if not isinstance(images, dict) or set(images) != set(missing):
                raise ValueError(
                    "server RGB response does not match the requested cameras"
                )
            for camera, image in images.items():
                if not isinstance(image, bytes) or not image.startswith(
                    b"\x89PNG\r\n\x1a\n"
                ):
                    raise ValueError("server RGB response must contain PNG bytes")
                destination = self._artifact_file(mapping[camera], step)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._temporary_file(destination)
                try:
                    temporary.write_bytes(image)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

    def cached_artifact_bytes(self, name, *, step=-1):
        """Return already materialized media for passive UI; never perform I/O RPC."""
        resolved = self._resolve_read_step(step)
        path = self._artifact_file(name, resolved)
        return path.read_bytes() if path.is_file() else None

    def artifact_path(self, name, *, step=-1):
        resolved = self._resolve_read_step(step)
        camera = next(
            (camera for camera, artifact in tools.CAMERA_ARTIFACTS if artifact == name),
            None,
        )
        if camera is not None and resolved is not None:
            self.materialize_rgb(resolved, [camera])
        return super().artifact_path(name, step=resolved)

    def load_bytes(self, name, *, step=-1):
        return self.artifact_path(name, step=step).read_bytes()

    def load(self, name, *, step=-1):
        self.artifact_path(name, step=step)
        return super().load(name, step=step)


class RoboDojoPrimitives:
    """Hold this episode's RPC clients and its shared action/cancellation owners."""

    def __init__(
        self,
        *,
        env,
        vla_client,
        sam3_client,
        execution,
        check_cancelled,
        graspgen_client=None,
    ):
        self.env, self.vla_client, self.sam3_client = env, vla_client, sam3_client
        self.graspgen_client = graspgen_client
        self.execution = execution
        self._check_cancelled = check_cancelled
        self._last_obs = env.last_obs
        self.model_budget = None

    @contextmanager
    def model_request(self, kind):
        with (
            self.model_budget.request(kind)
            if self.model_budget is not None
            else nullcontext()
        ):
            try:
                yield
            except Exception as error:
                # No retry: a dispatched inference may have consumed provider work.
                raise ModelServiceFailure(
                    f"RoboDojo {kind} request failed: {error}", retryable=False
                ) from error


class RoboDojoToolkit(Toolkit):
    """Expose public RGB-D/proprioception tools; settle each native action once."""

    def __init__(self, *, primitives_kwargs, dashboard_events, memory, task_name=None):
        self._task_name = task_name
        kwargs = dict(primitives_kwargs)
        frozen = kwargs.pop("frozen_bundle", False)
        filesystem = kwargs.pop("filesystem_tools_enabled", True) and not frozen
        maximum = kwargs.pop("frozen_episode_steps", kwargs["env"].max_episode_steps)
        state = RoboDojoState(get_output_dir(), env=kwargs["env"])
        state.reset()
        self._execution_trace = ExecutionTrace(
            Path(get_output_dir()) / "execution_trace.jsonl"
        )
        self._execution_controller = ActiveExecution(self._execution_trace)
        super().__init__(
            state=state,
            dashboard_events=dashboard_events,
            memory=memory,
            active_execution=self._execution_controller,
            filesystem_tools_enabled=filesystem,
        )
        self._primitives = RoboDojoPrimitives(
            execution=self._execution_controller,
            check_cancelled=self.raise_if_cancelled,
            **kwargs,
        )
        primitives = getattr(self, "_primitives", None)
        if getattr(primitives, "graspgen_client", None) is not None:
            atomic_text(
                Path(get_output_dir()) / "graspgen_config.json",
                json.dumps(
                    self._primitives.graspgen_client.config.model_dump(mode="json"),
                    indent=2,
                )
                + "\n",
            )
        self._planner_observation = PlannerObservation()
        self._primitives.env.bind_frozen_execution(
            self._execution_trace, self._execution_controller, max_episode_steps=maximum
        )
        self._publish_step(tools.dump_state(self._primitives, self._state))
        for spec in tools.TOOLS_SPEC:
            if (
                spec["name"] == "generate_grasps"
                and self._primitives.graspgen_client is None
            ):
                continue
            function = getattr(tools, spec["name"])
            arguments = {"state": self._state}
            if spec["name"] != "view_env_state":
                arguments["primitives"] = self._primitives
            self.add_tool(spec["name"], spec, partial(function, **arguments))

    def get_env_state(self, command, result, elapsed_s):
        record = tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        output = tools.view_env_state(
            record.step_idx, state=self._state, include_images=False
        )
        output["agent_elapsed_s"] = elapsed_s
        return output

    def bind_model_budget(self, budget):
        self._primitives.model_budget = budget

    def native_image_contract(self):
        return {
            "observation": len(tools.CAMERA_ARTIFACTS),
            "reads": {
                **super().native_image_contract()["reads"],
                "view_env_state": len(tools.CAMERA_ARTIFACTS),
                "locate": 0,
                "back_project": 0,
                "segment": 0,
                "generate_grasps": 0,
                "read_text_file": 0,
                "list_dir": 0,
            },
        }

    def prepare_model_context(self, context):
        record = self._state.latest_record()
        task_context = (
            "\nOfficial RoboDojo task instruction (verbatim):\n"
            + record.extras["official_task_instruction"]
        )
        supplement = record.extras["task_rule_supplement"]
        if supplement:
            task_context += "\n\nProject task-rule supplement:\n" + supplement
        context = {
            **context,
            "base_instructions": context["base_instructions"] + task_context,
        }
        observation = {
            "step": record.step_idx,
            "observation_ref": record.extras["observation_ref"],
            "physical_step": record.state["physical_step"],
            "state": record.state["state"],
            "terminated": record.terminated,
            "truncated": record.truncated,
        }
        self._state.materialize_rgb(
            record.step_idx, [camera for camera, _ in tools.CAMERA_ARTIFACTS]
        )
        images = [
            self._state.load_bytes(artifact, step=record.step_idx)
            for _, artifact in tools.CAMERA_ARTIFACTS
        ]
        # Refresh passive displays from already materialized pixels without a
        # second timeline action or any mutation of the archived StepRecord.
        self._publish_step(replace(record, command=None, result={}))
        return self._planner_observation.prepare(context, observation, images)

    def solved(self):
        return self._primitives.env.is_success()

    def finalize_model_context(self, context):
        """Bind the task's actual action budget after editable C has run."""
        from reweave.benchmarks.robodojo.vla_prior import task_prior

        prior = task_prior(self._task_name)
        budget = self.control_step_budget()
        limit, used, remaining = (
            budget[name] for name in ("limit", "used", "remaining")
        )
        one_percent = 100 * 50 / limit
        two_percent = 100 * 100 / limit
        guidance = (
            "RoboDojo native physical-action budget: this episode allows "
            f"{limit} environment control steps in total. This limit is specific "
            "to the current task; each episode has its own budget. "
            "pi0_pick max_chunks counts VLA forwards, not physical steps. "
            "One VLA chunk can execute up to 50 physical steps. "
            f"One chunk can therefore use {one_percent:.2f}% of this episode; "
            f"two chunks can use 100 steps ({two_percent:.2f}%). "
            "Read the remaining step count before each call. Prefer one chunk "
            "then inspect its actual effect before requesting another; do not "
            "blindly repeat an ineffective chunk. Reserve physical steps for "
            "correction, gripper release and both arms returning home when the "
            "task requires them. Atomic trajectories, gripper actions and home "
            "motions also use this same budget. Thinking and observation alone "
            "do not spend physical steps. A motion's max_steps is an admission "
            "cap, not a guarantee that a longer planned path executes a prefix."
        )
        reminder = (
            f"Current RoboDojo physical budget: total={limit}, used={used}, "
            f"remaining={remaining} steps. One VLA chunk costs up to 50 steps; "
            "two cost up to 100. Account for the remaining task and required "
            "release/home motions before spending another chunk."
        )
        if prior is not None:
            guidance += "\n\n" + prior["prompt"]
            atomic_text(
                Path(get_output_dir()) / "vla_baseline_prior.json",
                json.dumps(prior, indent=2) + "\n",
            )
        return {
            **context,
            "instructions": context["instructions"] + "\n\n" + guidance,
            "messages": [
                *context["messages"],
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": reminder}],
                },
            ],
        }

    def validate_model_context(self, context):
        super().validate_model_context(context)
        self._planner_observation.validate(context)
        # Editable C chooses evidence, but cannot withhold the official task text.
        official = self._state.latest_record().extras["official_task_instruction"]
        model_input = {
            "instructions": context["instructions"],
            "messages": context["messages"],
        }
        if not contains_text(model_input, official):
            raise ModelInputRejected(
                "native C omitted the verbatim official RoboDojo task instruction"
            )

    def execute_tool(self, name, input_dict):
        self._primitives.env.raise_if_invalid()
        result = super().execute_tool(name, input_dict)
        # The action and public state settle before invalidity aborts the episode.
        self._primitives.env.raise_if_invalid()
        return result

    def discard_active_vla_suffix(self):
        return self._execution_controller.discard_open_suffix(reason="tool_failed")

    def control_step_budget(self):
        env = self._primitives.env
        return {
            "limit": env.max_episode_steps,
            "used": env.elapsed_steps(),
            "remaining": env.steps_remaining(),
            "unit": "environment_control_step",
            "operation_costs": {
                "pi0_pick": {
                    "upper_bound_formula": "max_chunks * executable_actions_per_forward",
                    "executable_actions_per_forward": 50,
                },
                "move_to": {
                    "upper_bound_formula": "max_steps or remaining_episode_budget",
                },
                "move_pose": {
                    "upper_bound_formula": "max_steps or remaining_episode_budget",
                },
                "move_joints": {
                    "upper_bound_formula": "max_steps or remaining_episode_budget",
                },
                "set_gripper": {"upper_bound_formula": "max_steps (default 1)"},
            },
        }

    def harness_source_files(self):
        sources = super().harness_source_files()
        repository = Path(__file__).resolve().parents[3]
        for path in sorted(Path(__file__).parent.glob("*.py")):
            sources[path.relative_to(repository).as_posix()] = path.read_text()
        if self._primitives.graspgen_client is not None:
            path = repository / "reweave/robots/components/graspgen_client.py"
            sources[path.relative_to(repository).as_posix()] = path.read_text()
        return sources

    def close(self):
        # Join/cancel action owners before shutting down the owned environment.
        super().close()
        self._primitives.env.close()
