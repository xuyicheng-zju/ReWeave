# Modified for the ReWeave namespace.
"""Only owned RoboDojo environments reserve shutdown time for video return."""

import sys
from types import SimpleNamespace

from reweave.benchmarks.robodojo import robot_spec
from reweave.utils.daemon import ProcessDaemon


def test_owned_environment_has_long_finite_drain_without_changing_model_services(
    tmp_path, monkeypatch
):
    created = []

    class UnstartedDaemon(ProcessDaemon):
        def start(self):
            created.append(self)

    monkeypatch.setattr(robot_spec, "ProcessDaemon", UnstartedDaemon)
    monkeypatch.setattr(robot_spec, "_runtime_overrides", lambda args: {})
    args = SimpleNamespace(
        env_endpoint=None,
        sam3_endpoint=None,
        vla_endpoint=None,
        sim_python=sys.executable,
        pi05_python=sys.executable,
        cuda_device=0,
        task="play_tic_tac_toe",
        layout_set=1,
        layout=0,
        env_cfg_type="arx_x5",
        max_episode_steps=1100,
    )
    try:
        environment, env_client = robot_spec._spawn_env_server(args, tmp_path)
        sam, sam_client = robot_spec._spawn_sam3_server(args, tmp_path)
        vla, vla_client = robot_spec._spawn_vla_server(args, tmp_path)
        assert environment.cmd[environment.cmd.index("--layout-set") + 1] == "1"
        assert environment.cmd[environment.cmd.index("--layout") + 1] == "0"
        assert environment.stop_timeout_s == 300.0
        assert sam.stop_timeout_s == vla.stop_timeout_s == 15.0
        assert [item.name for item in created] == [
            "robodojo_env_server",
            "sam3_server",
            "robodojo_vla_server",
        ]
        for client in (env_client, sam_client, vla_client):
            client.close()
    finally:
        for daemon in created:
            daemon.stop()


def test_attached_environment_does_not_create_a_shutdown_owner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Attached shared environment must not be owned or stopped")

    monkeypatch.setattr(robot_spec, "ProcessDaemon", forbidden)
    daemon, client = robot_spec._spawn_env_server(
        SimpleNamespace(env_endpoint="http://127.0.0.1:18930"), tmp_path
    )
    assert daemon is None
    client.close()
