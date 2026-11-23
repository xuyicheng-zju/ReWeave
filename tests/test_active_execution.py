# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Physical tool ownership and causal execution trace contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from reweave.dashboard.events import NullDashboardEventSink
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace
from reweave.robots.vla_execution import VLAActionBuffer, VLAChunkPrediction
from reweave.session.base import EnvState
from reweave.tools.toolkit import (
    Toolkit,
    readonly,
)


def _prediction(*, forward_id: str = "forward-0", horizon: int = 3):
    return VLAChunkPrediction(
        actions=np.zeros((horizon, 7), dtype=np.float32),
        forward_id=forward_id,
        prediction_horizon_steps=horizon,
        execution_horizon_steps=horizon,
        inference_elapsed_s=0.1,
    )


def test_active_execution_binds_forwards_and_finishes_one_tool():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)
    execution_ref = active.begin("pi0_pick", {"prompt": "pick"})
    prediction = _prediction()
    active.record_forward(prediction, instruction="pick the blue bowl")

    for action in prediction.iter_actions():
        trace.record_control_step(
            action,
            terminated=False,
            truncated=False,
        )

    assert active.finish("completed") == execution_ref
    assert active.snapshot.phase == "idle"
    assert trace.events[0]["arguments"] == {"prompt": "pick"}
    controls = [row for row in trace.events if row["event_type"] == "control_step"]
    assert len(controls) == 3
    assert all(
        row["execution_ref"] == row["physical_call_ref"] == execution_ref
        and row["tool_name"] == "pi0_pick"
        for row in controls
    )


def test_episode_end_closes_suffix_without_an_external_cause():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)
    active.begin("pi0_pick")
    prediction = _prediction()
    active.record_forward(prediction, instruction="pick")
    action = next(prediction.iter_actions())
    trace.record_control_step(action, terminated=True, truncated=False)
    assert trace.record_action_suffix_discarded(action, reason="episode_end") == 2
    assert trace.events[-1]["cause_ref"] is None
    assert trace.events[-1]["first_discarded_action_index"] == 1
    active.finish("succeeded")


def test_control_step_trace_enforces_public_private_evidence_planes():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)
    active.begin("pi0_pick")
    prediction = _prediction(horizon=1)
    active.record_forward(prediction, instruction="pick")
    action = next(prediction.iter_actions())

    with pytest.raises(ValueError, match="public evidence cannot use"):
        trace.record_control_step(
            action,
            terminated=False,
            truncated=False,
            public_evidence={"privileged.task.success": False},
        )
    with pytest.raises(ValueError, match="must use privileged"):
        trace.record_control_step(
            action,
            terminated=False,
            truncated=False,
            critic_private_evidence={"robot.eef.motion_m": 0.0},
        )

    trace.record_control_step(
        action,
        terminated=False,
        truncated=False,
        public_evidence={"robot.eef.motion_m": np.float32(0.01)},
        critic_private_evidence={"privileged.task.success": np.bool_(False)},
    )
    event = trace.events[-1]
    assert event["public_evidence"] == {"robot.eef.motion_m": pytest.approx(0.01)}
    assert event["critic_private_evidence"] == {"privileged.task.success": False}
    active.finish("completed")


def test_active_execution_rejects_invalid_transitions():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)
    active.begin("pi05_act")
    with pytest.raises(RuntimeError, match="another continuation"):
        active.begin("pi05_act")
    active.record_forward(_prediction(), instruction="pick")
    with pytest.raises(ValueError, match="open action suffix"):
        active.finish("completed")
    assert active.discard_open_suffix(reason="tool_failed") == 3
    active.finish("failed")


def test_action_buffer_keeps_selected_action_until_confirmation():
    buffer = VLAActionBuffer(_prediction())
    action = buffer.pop_next()
    assert action is not None
    with pytest.raises(RuntimeError, match="before confirming"):
        buffer.pop_next()
    buffer.release_selected(action)
    assert buffer.discard_remaining() == 2


class _RuntimeToolkit(Toolkit):
    def __init__(
        self,
        tmp_path,
        *,
        handler,
    ):
        self.trace = ExecutionTrace(tmp_path / "execution_trace.jsonl")
        self.active = ActiveExecution(self.trace)
        self.discarded = 0
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(tmp_path),
            active_execution=self.active,
        )
        self.add_tool(
            "act",
            {
                "name": "act",
                "description": "act",
                "input_schema": {"type": "object", "properties": {}},
            },
            handler,
        )
        self.add_tool(
            "observe",
            {
                "name": "observe",
                "description": "observe",
                "input_schema": {"type": "object", "properties": {}},
            },
            self._observe,
        )

    @staticmethod
    @readonly
    def _observe():
        return {"observation": True}

    def get_env_state(self, *, command, result, elapsed_s):
        with self.state.record_step(
            state={"observed": True},
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return dict(result)

    def discard_active_vla_suffix(self):
        self.discarded += 1
        return 0

    def solved(self):
        return False


def test_toolkit_persists_active_execution_reference(tmp_path):
    toolkit = _RuntimeToolkit(tmp_path, handler=lambda: {"success": True})
    result = toolkit.execute_tool("act", {}).result
    assert result["active_execution_outcome"] == "succeeded"
    assert toolkit.trace.events[0]["execution_id"] == result["active_execution_ref"]


def test_readonly_tool_does_not_create_active_execution(tmp_path):
    toolkit = _RuntimeToolkit(tmp_path, handler=lambda: {"success": True})
    result = toolkit.execute_tool("observe", {}).result
    assert result["observation"] is True
    assert toolkit.active.snapshot.phase == "idle"


def test_author_public_control_history_is_complete_across_finished_calls(tmp_path):
    toolkit = _RuntimeToolkit(tmp_path, handler=lambda: {"success": True})
    for batch in range(2):
        toolkit.active.begin("act", arguments={"batch": batch})
        prediction = _prediction(forward_id=f"history-{batch}", horizon=11)
        toolkit.active.record_forward(prediction, instruction="evidence history")
        for action in prediction.iter_actions():
            toolkit.trace.record_control_step(
                action,
                terminated=False,
                truncated=False,
                public_evidence={"robot.motion": float(batch)},
                critic_private_evidence={"privileged.secret": 123.0},
            )
        toolkit.active.finish("completed")
    toolkit.active.begin("act")
    rows = toolkit.active.public_control_steps()
    assert len(rows) == 22
    assert rows[0]["control_step_index"] == 0
    assert rows[-1]["control_step_index"] == 21
    assert "privileged.secret" not in json.dumps(rows)
    assert "privileged.secret" in json.dumps(toolkit.trace.events)
    rows[0]["public_evidence"]["robot.motion"] = 999.0
    assert (
        toolkit.active.public_control_steps()[0]["public_evidence"]["robot.motion"]
        == 0.0
    )


def test_arguments_and_snapshots_cannot_mutate_live_state_or_trace():
    trace = ExecutionTrace()
    active = ActiveExecution(trace)
    arguments = {"target": {"position": [1, 2, 3]}}
    active.begin("move_to", arguments)
    arguments["target"]["position"][0] = 99
    snapshot = active.snapshot
    snapshot.arguments["target"]["position"][1] = 99
    events = trace.events
    events[0]["arguments"]["target"]["position"][2] = 99
    expected = {"target": {"position": [1, 2, 3]}}
    assert active.snapshot.arguments == expected
    assert trace.events[0]["arguments"] == expected
    active.finish("completed")
