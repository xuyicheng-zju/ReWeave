"""Passive Dashboard projections cannot trigger remote RoboDojo image reads."""

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.robot_spec import ROBODOJO_DASHBOARD_SPEC
from reweave.benchmarks.robodojo.toolkit import RoboDojoState
from reweave.dashboard.events import StepRecordEvent
from reweave.dashboard.state import DashboardState


def test_real_state_multiple_tools_stay_pixel_free_until_context_then_refresh_once(
    tmp_path,
):
    dashboard = DashboardState(
        output_dir=tmp_path, dashboard_spec=ROBODOJO_DASHBOARD_SPEC
    )
    rpc = SimulatedEnvironmentRPC(
        limit=1100, terminal_after=999, directory=tmp_path / "server"
    )
    requests = []
    original_call = rpc.call

    def call(method, *args, **kwargs):
        requests.append(method)
        return original_call(method, *args, **kwargs)

    rpc.call = call
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    assert isinstance(toolkit.state, RoboDojoState)
    toolkit._dashboard_events = dashboard
    try:
        dashboard.emit(StepRecordEvent(toolkit.state.latest_record(), toolkit.state))
        for gripper in (1, -1, 1):
            result = toolkit.execute_tool(
                "set_gripper", {"arm": "right", "gripper": gripper}
            )
            assert not result.result.get("error")
        assert "env.read_rgb" not in requests
        assert "env.read_sensor_snapshot" not in requests
        assert rpc.actions == 3
        before = dashboard.session_detail()["timeline"]
        assert len(before) == 3
        assert dashboard.frame("head") is None

        context = toolkit.prepare_model_context(
            {"base_instructions": "current state", "messages": []}
        )
        assert context["observation"]["physical_step"] == 3
        assert requests.count("env.read_rgb") == 1
        assert dashboard.frame("head").startswith(b"\x89PNG\r\n\x1a\n")
        assert dashboard.frame("wrist_left").startswith(b"\x89PNG\r\n\x1a\n")
        assert dashboard.frame("wrist_right").startswith(b"\x89PNG\r\n\x1a\n")
        assert dashboard.session_detail()["timeline"] == before
        assert toolkit.state.latest_record().command["action"] == "set_gripper"
        toolkit.prepare_model_context(
            {"base_instructions": "same state", "messages": []}
        )
        assert requests.count("env.read_rgb") == 1
        assert dashboard.session_detail()["timeline"] == before
        assert rpc.actions == 3
    finally:
        toolkit.close()
