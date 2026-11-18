"""Host retains full planning results while model feedback stays compact."""

import copy
import json

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.planner_observation import feedback_text
from reweave.utils.rpc.http_rpc import _from_json


@pytest.mark.parametrize("success", [False, True])
def test_complete_planning_response_is_archived_without_expanding_tool_feedback(
    tmp_path, success
):
    class PlanRPC(SimulatedEnvironmentRPC):
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            if method == "env.plan_joint_path":
                return copy.deepcopy(plan)
            return super().call(method, args=args, kwargs=kwargs, timeout_s=timeout_s)

    plan = {
        "status": "Success" if success else "Fail",
        "planner_status": "SOLVER_SUCCESS" if success else "IK_FAIL",
        "reason_known": True,
        "raw_planner_results": [
            {"status": "SOLVER_SUCCESS", "diagnostic": "native-detail-" * 20000}
        ],
        "planner_dt_s": 0.04,
        "control_period_s": 0.04,
        "joint_positions": np.linspace(np.zeros(6), np.ones(6), 5000).tolist(),
        "joint_velocities": np.zeros((5000, 6)).tolist(),
    }
    if not success:
        plan["error"] = "native collision-aware joint planning failed"
    rpc = PlanRPC(terminal_after=99, directory=tmp_path / "server")
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        result = toolkit.execute_tool(
            "move_joints",
            {
                "arm": "right",
                "joint_positions": [1] * 6,
                "tol_rad": 0.01,
                "max_steps": 200,
            },
        )
        files = list((tmp_path / "episode/native_plans").glob("*.json"))
        assert len(files) == 1
        receipt = _from_json(json.loads(files[0].read_text()))
        assert receipt["response"] == plan
        assert receipt["arguments"] == {"arm": "right", "joints": [1.0] * 6}
        assert (
            receipt["method"] == "env.plan_joint_path" and receipt["physical_step"] == 0
        )
        assert receipt["plan_id"] == files[0].stem and receipt["observation_ref"]
        feedback = json.dumps(feedback_text(result.result))
        assert len(feedback) < 2000
        assert (
            "native-detail-" not in feedback and "raw_planner_results" not in feedback
        )
        assert rpc.actions == 0
        assert len(files[0].read_bytes()) > 500000
    finally:
        toolkit.close()


def test_each_plan_has_distinct_immutable_evidence_including_typed_rpc_values(tmp_path):
    class PlanRPC(SimulatedEnvironmentRPC):
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            if method.startswith("env.plan_"):
                return copy.deepcopy(plan)
            return super().call(method, args=args, kwargs=kwargs, timeout_s=timeout_s)

    plan = {
        "status": "Fail",
        "error": "native failure",
        "raw_planner_result": {
            "array": np.array([1, 2], dtype=np.int16),
            "bytes": b"native-evidence",
        },
    }
    rpc = PlanRPC(terminal_after=99, directory=tmp_path / "server")
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        client = toolkit._primitives.env
        first = client.plan_position_path("right", [0, 0, 0.3])
        first["raw_planner_result"]["array"][0] = 99
        client.plan_pose_path("right", [[0, 0, 0.3, 1, 0, 0, 0]])
        files = list((tmp_path / "episode/native_plans").glob("*.json"))
        assert len(files) == 2 and len({file.stem for file in files}) == 2
        for file in files:
            original = _from_json(json.loads(file.read_text()))["response"]
            np.testing.assert_array_equal(
                original["raw_planner_result"]["array"], [1, 2]
            )
            assert original["raw_planner_result"]["array"].dtype == np.int16
            assert original["raw_planner_result"]["bytes"] == b"native-evidence"
        assert rpc.actions == 0
    finally:
        toolkit.close()
