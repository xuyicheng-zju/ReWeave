"""One SAM request, exact snapshot geometry, bounded measured control and no replay."""

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.robots.components.sam3_client import Sam3Result
from reweave.runtime.capability import PhysicalOutcomeUnknown


def test_locate_uses_one_server_sam_call_without_rgbd_transfer(tmp_path):
    rpc = SimulatedEnvironmentRPC(limit=20, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    calls = []

    class Sam:
        def segment(self, image, **kwargs):
            calls.append(kwargs)
            return Sam3Result(
                found=True, mask=np.ones((4, 5), bool), score=0.9, box=[0, 0, 5, 4]
            )

    rpc.observations.sam3 = Sam()
    try:
        result = toolkit.execute_tool("locate", {"text_prompt": "bottle"}).result
        assert len(calls) == 1 and rpc.actions == 0
        assert result["geometry"]["height_m"] is None
        assert (
            result["observation_ref"]
            == toolkit._primitives.env.last_obs["observation_ref"]
        )
        assert "mask" not in result and "vision" not in result
    finally:
        toolkit.close()


def measured_rpc(limit=20, acknowledgement="valid"):
    rpc = SimulatedEnvironmentRPC(
        limit=limit, terminal_after=99, acknowledgement=acknowledgement
    )
    rpc.observation["state"]["right_gripper_measured_opening"] = np.array([1.0])
    rpc.observation["state"]["left_gripper_measured_opening"] = np.array([1.0])
    rpc.after_action = lambda: rpc.observation["state"].update(
        right_gripper_measured_opening=np.array([max(0.2, 1 - rpc.actions * 0.3)])
    )
    return rpc


def test_gripper_finishes_measured_stroke_without_claiming_possession(tmp_path):
    rpc = measured_rpc()
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    try:
        result = toolkit.execute_tool(
            "set_gripper", {"arm": "right", "gripper": 1, "max_steps": 10}
        ).result["capability_result"]
        assert result["steps_used"] == rpc.actions == 4
        assert result["opening_stable"] is True
        assert result["stop_reason"] == "opening_stable"
        assert result["measured_opening"] == pytest.approx([0.2])
        assert "success" not in result and "holding" not in result
        assert toolkit._primitives.env.last_obs["state"]["right_ee_joint_state"] == [
            1.0
        ]
    finally:
        toolkit.close()


def test_measured_stroke_respects_last_physical_budget_ack(tmp_path):
    rpc = measured_rpc(limit=2)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    try:
        result = toolkit.execute_tool(
            "set_gripper", {"arm": "right", "gripper": 1, "max_steps": 10}
        ).result["capability_result"]
        assert rpc.actions == result["steps_used"] == 2
        assert result["stop_reason"] == "episode_budget_exhausted"
        assert toolkit.control_step_budget()["remaining"] == 0
    finally:
        toolkit.close()


def test_unknown_ack_inside_measured_stroke_never_issues_another_action(tmp_path):
    rpc = measured_rpc()
    previous = rpc.after_action

    def action():
        previous()
        if rpc.actions == 2:
            rpc.acknowledgement = "missing"

    rpc.after_action = action
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    try:
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit.execute_tool(
                "set_gripper", {"arm": "right", "gripper": 1, "max_steps": 10}
            )
        assert rpc.actions == 2
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit.execute_tool(
                "set_gripper", {"arm": "right", "gripper": 1, "max_steps": 10}
            )
        assert rpc.actions == 2
    finally:
        toolkit.close()
