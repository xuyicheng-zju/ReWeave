# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""SIMULATED CPU demo of the real frozen RoboDojo Harness execution path.

Run with the repository interpreter: .venv/bin/python examples/robodojo_demo.py
No Isaac Sim, external model, credentials or physical robot is used. A scripted
Planner and policy prove interface execution only, never official physical success.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# Allow the documented direct-file invocation without requiring installation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient
from reweave.benchmarks.robodojo.observation_service import ObservationService
from reweave.benchmarks.robodojo.robot_spec import finalize_run
from reweave.benchmarks.robodojo.tasks import native_step_limit
from reweave.cli.main import parse_cli_arguments
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evaluation import RunFinalizationContext
from reweave.evolution.contracts import (
    EvaluationConditions,
    EvaluationRequest,
    Instance,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.runtime_manifest import (
    RuntimeManifest,
    RuntimeResources,
    capture_baseline,
)
from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator
from reweave.planner.base import API_ERROR, SUCCESS, classify_episode_termination
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.robots.base import get_toolkit
from reweave.robots.components.pi05_vla_client import Pi05VLAClient
from reweave.robots.robot_spec import RunConfig
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.logging import init_output_dir
from reweave.utils.persistence import atomic_json


def simulated_observation():
    """Public camera/proprioception fixture; private oracle bait must be filtered."""
    state = {}
    for arm in ("left", "right"):
        state[f"{arm}_arm_joint_state"] = np.zeros(6)
        state[f"{arm}_ee_joint_state"] = np.ones(1)
        state[f"{arm}_ee_pose"] = np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    state["object_pose"] = [999, 999, 999]
    return {
        "instruction": "Put the simulated bottle into the simulated dustbin.",
        "state": state,
        "oracle_success": True,
        "vision": {
            camera: {
                "color": np.full((4, 5, 3), index * 70, np.uint8),
                "depth": np.ones((4, 5)),
                "shape": [4, 5, 3],
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
                "object_id": "private-oracle",
            }
            for index, camera in enumerate(
                (
                    "cam_head",
                    "cam_left_wrist",
                    "cam_right_wrist",
                )
            )
        },
    }


class SimulatedEnvironmentRPC:
    """Deterministic RPC boundary for CPU integration and failure injection."""

    def __init__(
        self, *, limit=5, terminal_after=2, acknowledgement="valid", directory=None
    ):
        self.meta = {
            "task": "put_bottles_into_dustbin",
            "layout_set": 0,
            "layout": 0,
            "env_cfg_type": "arx_x5",
            "device_id": 0,
            "num_envs": 1,
            "max_episode_steps": limit,
            "observation_transport": "server_snapshot",
        }
        self.actions = 0
        self.terminal_after = terminal_after
        self.acknowledgement = acknowledgement
        self.after_action = None
        self.closed = False
        self.observation = simulated_observation()
        self._temporary_store = (
            tempfile.TemporaryDirectory(prefix="reweave-demo-observations-")
            if directory is None
            else None
        )
        self.observations = ObservationService(
            directory if directory is not None else self._temporary_store.name,
        )

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        if method == "env.get_env_meta":
            return dict(self.meta)
        if method in {"env.get_obs", "env.reset"}:
            if method == "env.reset" and self.actions:
                raise ValueError("SIMULATED scene cannot reset after a physical action")
            if not self.observations.settled:
                if self.actions:
                    raise ValueError("SIMULATED action observation is unsettled")
                self.observations.publish(self.observation, self.actions)
            return self.observations.compact()
        snapshot_reads = {
            "env.retain_observation": "retain_observation",
            "env.read_rgb": "read_rgb",
            "env.read_sensor_snapshot": "read_sensor_snapshot",
            "env.back_project_ref": "back_project_ref",
            "env.segment_ref": "segment_ref",
            "env.locate_ref": "locate_ref",
            "env.grasps_ref": "grasps_ref",
            "env.predict_ref": "predict_ref",
        }
        if method in snapshot_reads:
            if args:
                raise ValueError("Snapshot reads require explicit named references")
            return getattr(self.observations, snapshot_reads[method])(**(kwargs or {}))
        if method in {"env.render_camera", "env.chunk_step"}:
            raise ValueError(
                "SIMULATED Host cannot request raw-array RGBD or unacknowledged chunks"
            )
        if method == "env.get_task_language":
            return self.observation["instruction"]
        if method == "env.is_success":
            return self.actions >= self.terminal_after
        if method in {"env.plan_position_path", "env.plan_joint_path"}:
            values = kwargs or {}
            arm = values["arm"]
            target = (
                values["joints"]
                if method == "env.plan_joint_path"
                else np.asarray(
                    self.observation["state"][f"{arm}_arm_joint_state"]
                ).tolist()
            )
            return {
                "status": "Success",
                "planner_dt_s": 0.04,
                "control_period_s": 0.04,
                "joint_positions": [target],
                "joint_velocities": [[0.0] * 6],
            }
        if method == "env.step_trajectory":
            # CPU fixture models the measured endpoint of one window. Real
            # per-physics-sample queue execution has separate native SDK tests.
            values = kwargs or {}
            positions = np.asarray(values["positions"], dtype=np.float64)
            velocities = np.asarray(values["velocities"], dtype=np.float64)
            if (
                positions.ndim != 2
                or positions.shape[1:] != (6,)
                or not len(positions)
                or velocities.shape != positions.shape
            ):
                raise ValueError("SIMULATED invalid native trajectory window")
            arm = values["arm"]
            action = {
                name: np.asarray(self.observation["state"][name]).tolist()
                for name in (
                    "left_arm_joint_state",
                    "left_ee_joint_state",
                    "right_arm_joint_state",
                    "right_ee_joint_state",
                )
            }
            action[f"{arm}_arm_joint_state"] = positions[-1].tolist()
            gripper = values.get("gripper", 0)
            if gripper:
                action[f"{arm}_ee_joint_state"] = [0.0 if gripper > 0 else 1.0]
            obs, reward, done, info = self.call(
                "env.step", args=(action,), timeout_s=timeout_s
            )
            return (
                obs,
                reward,
                done,
                {
                    **info,
                    "trajectory_substeps": len(positions),
                    "trajectory_dt_s": 0.04,
                },
            )
        if method == "env.close":
            self.closed = True
            return None
        if method != "env.step":
            raise AssertionError(method)
        self.observations.invalidate()
        self.actions += 1
        if self.after_action:
            self.after_action()
        if self.acknowledgement == "missing":
            raise TimeoutError("SIMULATED lost acknowledgement")
        if self.acknowledgement == "malformed":
            return {"status": "not-an-acknowledgement"}
        return (
            self.observations.publish(self.observation, self.actions),
            0.0,
            self.actions >= self.terminal_after,
            {
                "environment_error": None,
                "status": {
                    "step": self.actions,
                    "step_limit": self.meta["max_episode_steps"],
                },
            },
        )


class SimulatedPolicyRPC:
    """Return a typed four-action forward through the real Pi0.5 client."""

    def __init__(self, *, action_dimension=14):
        self.calls = 0
        self.action_dimension = action_dimension
        self.requests = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        assert method == "vla.predict"
        self.calls += 1
        self.requests.append(args[0])
        return {
            "actions": np.zeros((1, 4, self.action_dimension), np.float32),
            "forward_id": f"simulated-forward-{self.calls}",
            "prediction_horizon_steps": 4,
            "execution_horizon_steps": 4,
            "inference_elapsed_s": 0.001,
        }


def create_simulated_toolkit(output: Path, *, rpc=None, policy=None):
    """Build real clients, Toolkit and ActiveExecution; only RPC responses are fake."""
    output.mkdir(parents=True, exist_ok=True)
    init_output_dir(output)
    rpc = rpc or SimulatedEnvironmentRPC(directory=output)
    policy = policy or SimulatedPolicyRPC()
    rpc.observations.vla = Pi05VLAClient(policy, embodiment="robodojo")
    env = RoboDojoEnvClient(rpc, expected_meta=rpc.meta)
    toolkit = get_toolkit(
        "robodojo",
        primitives_kwargs={
            "env": env,
            "vla_client": Pi05VLAClient(policy, embodiment="robodojo"),
            "sam3_client": None,
            "frozen_bundle": True,
        },
        dashboard_events=NullDashboardEventSink(),
        config=RunConfig(
            recipe_tag="simulated-robodojo",
            output_dir=output,
            prompt_vars={"memory_dir": output.parent / "memory"},
            task_desc={
                "task": rpc.meta["task"],
                "layout_set": 0,
                "layout": rpc.meta["layout"],
            },
        ),
    )
    return toolkit, rpc, policy


def run_demo(directory: Path):
    """Capture, execute and evaluate a SIMULATED immutable episode, with originals."""
    directory = directory.expanduser().resolve()
    if directory.exists():
        raise ValueError("demo requires a fresh output directory")
    episode = directory / "attempt/episode"
    memory = directory / "memory"
    memory.mkdir(parents=True)
    # The physical budget is the task's official step_lim, never a demo-local cap.
    limit = native_step_limit("put_bottles_into_dustbin")
    episode.mkdir(parents=True)
    toolkit, rpc, policy = create_simulated_toolkit(
        episode, rpc=SimulatedEnvironmentRPC(limit=limit, directory=episode)
    )
    limits = EpisodeLimits(program_calls=20, gateway_calls=20, wall_time_s=30)
    conditions = EvaluationConditions(
        controller_request_unit="native_turn",
        adaptation_mode="static",
        repetition=0,
        physical_steps=limit,
        wait_steps=0,
        model_requests=5,
        wall_time_s=30,
        program_calls=20,
        gateway_calls=20,
        model_sampling={"reasoning_effort": "low"},
    )
    fixed_arguments = (
        "--robot",
        "robodojo",
        "--planner",
        "native",
        "--reasoning-effort",
        "low",
        "--max-episode-steps",
        str(limit),
        "--wait-steps",
        "0",
        "--max-turns",
        "5",
        "--planner-timeout-s",
        "30",
    )
    _, parsed, spec = parse_cli_arguments(
        [
            *fixed_arguments,
            "--task",
            rpc.meta["task"],
            "--layout-set",
            "0",
            "--layout",
            "0",
            "--cuda-device",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory),
            "--output-dir",
            str(episode),
        ]
    )
    spec.parse_config(parsed)
    arguments = vars(parsed)

    note = toolkit.episode_notes.drafts / "task-family" / "conditions.md"
    note_text = """# SIMULATED RoboDojo observations

Source: view_env_state(step=0), received in this Planner session.
The current public observation supplies head and two wrist RGB-D views and
both arms' proprioception. The task language is resolved for this layout.
Use the policy's acknowledged actions and fresh feedback; discard its suffix
when the environment becomes terminal. A grasp motion heuristic does not
certify bottle possession. These are unverified interface observations from
this CPU fixture, with no claim about GPU behavior or physical success.
"""
    proposals = [
        {"name": "view_env_state", "arguments_json": json.dumps({"step": 0})},
        {
            "name": "write_text_file",
            "arguments_json": json.dumps({"path": str(note), "content": note_text}),
        },
        {"name": "read_text_file", "arguments_json": json.dumps({"path": str(note)})},
        {"name": "pi0_pick", "arguments_json": json.dumps({"max_chunks": 1})},
    ]
    requests = []

    class SimulatedNativeSession:
        """A scripted native session fixture; no model or network is invoked."""

        session_id = "SIMULATED-demo-native-session"
        context_generation = 0

        def __init__(self, directory):
            self.directory = directory

        async def start(self, instructions, cwd):
            self.cwd = cwd

        async def decide(self, text, images, output_schema, request_id):
            requests.append(
                {
                    "text": text,
                    "images": list(map(str, images)),
                    "request_id": request_id,
                }
            )
            index = len(requests) - 1
            return {
                "text": "SIMULATED interface exercise",
                "tool_calls": [proposals[index]] if index < len(proposals) else [],
            }

        async def close(self):
            pass

    planner = NativeBundlePlanner(
        output_dir=episode,
        model="SIMULATED-native",
        reasoning_effort="low",
        session_factory=SimulatedNativeSession,
    )
    fixture = directory / "simulated_environment.json"
    atomic_json(fixture, {"mode": "SIMULATED", "environment": rpc.meta})
    bundle = capture_baseline(
        directory / "baseline",
        arguments=arguments,
        limits={"episode": limits.model_dump(mode="json")},
        conditions=conditions,
        resources=RuntimeResources(
            models={},
            environment={"fixture": artifact_record(fixture)},
            routes={
                "planner": "SIMULATED scripted native session",
                "vla": "SIMULATED typed RPC",
            },
        ),
        memory=memory,
        toolkit=toolkit,
        planner=planner,
    )
    manifest = RuntimeManifest.model_validate_json(
        (directory / "baseline/runtime.json").read_text()
    )
    toolkit.export_harness_context(str(episode))
    planner.bind_bundle(bundle, episode_limits=limits)
    started = time.monotonic()
    result = planner.solve(
        system_prompt="SIMULATED robot demo; act using public observations.",
        user_message=rpc.observation["instruction"],
        toolkit=toolkit,
        max_turns=5,
    )
    toolkit.finalize_episode_evaluation()
    success = toolkit.solved()
    termination = classify_episode_termination(
        planner_termination_class=result.termination_class, official_success=success
    )
    elapsed = time.monotonic() - started
    finalize_run(
        RunFinalizationContext(
            output_dir=episode,
            robot_name="robodojo",
            task_desc={"task": rpc.meta["task"], "layout_set": 0, "layout": 0},
            environment_success=success,
            termination_class=termination,
            agent_error=result.error,
            elapsed_s=elapsed,
            planner="SIMULATED",
            model="scripted",
            reasoning_effort="low",
            max_turns=5,
            planner_timeout_s=30,
            finish_result=None,
            stats=result.stats,
        )
    )
    atomic_json(
        episode / "transcript_simulated.json",
        {
            "mode": "SIMULATED",
            "run_configuration": arguments,
            "bundle_id": bundle.record_id,
            "termination_class": termination,
            "api_error_retryable": result.api_error_retryable
            if termination == API_ERROR
            else None,
            "planner_termination_class": result.termination_class,
            "planner_api_error_retryable": result.api_error_retryable,
            "environment_success": success,
            "official_success": termination == SUCCESS,
            "stats": result.stats,
            "error": result.error,
        },
    )
    atomic_json(
        directory / "attempt/exit.json", {"returncode": 0, "elapsed_s": elapsed}
    )
    runner = RolloutConfiguration(
        devices=(0,),
        arguments=fixed_arguments,
        memory_snapshot=str(memory),
        runtime_manifest=manifest,
        episode_limits=limits.model_dump(mode="json"),
    )
    request = EvaluationRequest(
        record_id="demo-request",
        bundle_id=bundle.record_id,
        instance=Instance(
            record_id="demo-instance",
            task=f"robodojo:{rpc.meta['task']}",
            initialization={"task": rpc.meta["task"], "layout_set": 0, "layout": 0},
        ),
        split="dev",
        manifest_id="demo-manifest",
        execution=manifest.execution_id,
        conditions=conditions,
    )
    runner.validate_arguments(
        request.instance, directory=directory / "attempt", memory=memory, device=0
    )
    record = RolloutEvaluator(
        runner, repository=Path(__file__).parent.parent, workspace=directory / "search"
    )._close(request, directory / "attempt")
    toolkit.close()
    summary = {
        "mode": "SIMULATED",
        "physical_success_claim": False,
        "simulated_environment_success": record.official_success,
        "acknowledged_actions": rpc.actions,
        "policy_forwards": policy.calls,
        "evaluated_actions": record.usage.physical_steps,
        "planner_requests": len(requests),
        "planner_note": note.relative_to(episode).as_posix(),
        "archived_note_revisions": sum(
            name.startswith("planner_memory/revisions/")
            for name in record.episode.files
        ),
        "bundle_id": bundle.record_id,
        "termination_class": termination,
        "output": str(directory),
    }
    atomic_json(directory / "demo_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Fresh directory for SIMULATED evidence"
    )
    args = parser.parse_args()
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="robodojo-demo-")) / "run"
    print(json.dumps(run_demo(output), indent=2))
