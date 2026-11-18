"""Known perception/planning failures return to the same native decision loop."""

from functools import partial

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo import tools
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.robots.robodojo import test_demo
from tests.unit_tests.robots.robodojo.test_graspgen_tools import _Primitives
from tests.unit_tests.robots.robodojo.test_move_pose import (
    install_pose_solver,
    pose_arguments,
)

completed_demo = test_demo.completed_demo


@pytest.mark.parametrize("failure", ["sam_no_mask", "curobo_no_path"])
def test_known_grasp_chain_failure_reaches_next_decision_without_motion(
    completed_demo, tmp_path, failure
):
    captured, _, _, _, _ = completed_demo
    bundle = AgentBundle.model_validate_json(
        (captured / "baseline/bundle.json").read_text()
    )
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    if failure == "sam_no_mask":
        marker = "no target mask in this observed view"
        toolkit._primitives.graspgen_client = _Primitives().graspgen_client
        toolkit._primitives.env.segment_ref = lambda ref, **kwargs: {
            "found": False,
            "mask_ref": None,
            "reason": marker,
        }
        # Use the actual handler and read-only registration; only sensor/service
        # responses are synthetic. No grasp model or action should be dispatched.
        spec = next(s for s in tools.TOOLS_SPEC if s["name"] == "generate_grasps")
        toolkit.add_tool(
            "generate_grasps",
            spec,
            partial(
                tools.generate_grasps,
                primitives=toolkit._primitives,
                state=toolkit.state,
            ),
        )
        call = ToolCallPart(
            "generate_grasps", {"text_prompt": "red object", "arm": "right"}, "grasp"
        )
    else:
        marker = "no collision-free path to this candidate"
        install_pose_solver(rpc, result={"status": "Fail", "error": marker})
        call = ToolCallPart("move_pose", pose_arguments(xyz=[0.1, 0, 0.3]), "pose")
    requests = []

    def model(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            return ModelResponse(parts=[call])
        assert len(requests) == 2
        assert marker in ModelMessagesTypeAdapter.dump_json(messages).decode()
        assert rpc.actions == policy.calls == 0
        # Deliberate test termination only after proving a new decision occurs.
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish",
                    {"status": "stuck", "summary": "Synthetic feedback verified"},
                    "end",
                )
            ]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        bundle,
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=None
        ),
    )
    try:
        result = planner.solve(
            system_prompt="SIMULATED failure-feedback audit",
            user_message="Inspect a grasp candidate and replan on known failure.",
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None
        assert len(requests) == 2
        assert rpc.actions == policy.calls == 0
    finally:
        toolkit.close()
