# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Fixed physical controls preserve actual steps, horizons and chunk suffixes."""

from __future__ import annotations

import numpy as np
import pytest

from reweave.benchmarks.libero.tools import LiberoPrimitives
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace, merge_control_telemetry
from reweave.robots.vla_execution import VLAChunkPrediction


def _libero_critic_evidence() -> dict[str, object]:
    return {
        "privileged.available": True,
        "privileged.task.semantic_available": True,
        "privileged.task.goal.predicate_count": 1,
        "privileged.task.goal.evaluable_count": 1,
        "privileged.task.goal.satisfied_count": 0,
        "privileged.task.goal.predicate_fraction_available": True,
        "privileged.contact.robot.count": 0,
        "privileged.contact.gripper.count": 0,
        "privileged.contact.force_available": False,
        "privileged.contact.max_normal_force_n": 0.0,
        "privileged.joint.count": 0,
    }


def test_libero_fixed_horizon_preserves_predictions_and_every_action():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)

    class Env:
        terminated = False
        truncated = False

        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            states = np.zeros(8, dtype=np.float32)
            states[:3] = np.sum(np.asarray(self.actions)[:, :3], axis=0)
            return (
                {
                    "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
                    "states": states,
                },
                0.0,
                False,
                False,
                {"critic_evidence": _libero_critic_evidence()},
            )

        def elapsed_steps(self):
            return len(self.actions)

    class Model:
        def predict_action_batch(self, obs, mode="eval"):
            actions = np.ones((5, 7), dtype=np.float32)
            return VLAChunkPrediction(
                actions=actions,
                forward_id="controlled-forward",
                prediction_horizon_steps=20,
                execution_horizon_steps=5,
                inference_elapsed_s=0.1,
            )

    env = Env()
    primitives = LiberoPrimitives(
        env=env,
        model=Model(),
        sam3_client=object(),
        check_cancelled=lambda: None,
        execution_trace=trace,
        active_execution=active,
    )
    primitives.set_obs(
        {
            "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
            "states": np.zeros(8, dtype=np.float32),
            "task_descriptions": "task",
        }
    )
    active.begin("pi0_pick", {"prompt": "pick"})

    result = primitives._vlm_chunk("pick")
    assert result.executed_action_steps == 5
    assert len(env.actions) == 5
    assert env.actions[0].tolist() == pytest.approx([1.0] * 7)
    forward = next(
        event for event in trace.events if event["event_type"] == "vla_forward"
    )
    assert forward["prediction_horizon_steps"] == 20
    assert forward["execution_horizon_steps"] == 5
    control_steps = [
        event
        for event in merge_control_telemetry(trace.events)
        if event["event_type"] == "control_step"
    ]
    assert len(control_steps) == 5
    assert control_steps[0]["public_evidence"]["episode.step_index"] == 1
    assert control_steps[0]["public_evidence"]["robot.eef.motion_m"] > 0.0
    assert control_steps[0]["critic_private_evidence"] == (_libero_critic_evidence())
    assert not any(
        feature.startswith("privileged.")
        for feature in control_steps[0]["public_evidence"]
    )
    active.finish("completed")
