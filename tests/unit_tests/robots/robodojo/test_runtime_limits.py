# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Native task termination and configured caps preserve exact client settlement."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit


@pytest.fixture
def agent_environment(monkeypatch):
    # Only the external simulator base is replaced; run the production adapter,
    # client, tools and trace against a deterministic native action boundary.
    module = ModuleType("rlinf.envs.sim.robodojo.robodojo_env")

    class SimulatorBase:
        def __init__(self, cfg, *args, **kwargs):
            del cfg, args, kwargs
            seed_manager = SimpleNamespace(
                config={"seed": 0}, eval_seed=0, seed_info={}
            )

            def init_eval():
                seed_manager.eval_seed = seed_manager.config["seed"]
                seed_manager.seed_info = {
                    index: {
                        "scene_layout": f"/layouts/{seed_manager.eval_seed}/general_pickup_{index}.json"
                    }
                    for index in range(3)
                }

            seed_manager.init_eval = init_eval
            seed_manager.get_seed_scene_info = lambda index: {
                "loaded_layout_index": index
            }
            native = SimpleNamespace(
                camera_manager=SimpleNamespace(cameras=[], camera_names=[]),
                seed_manager=seed_manager,
                step_lim=1100,
            )
            self._closed = False
            self._test_seed_manager = seed_manager
            self._test_reset_calls = []

            def reset(*, env_seeds):
                # A native task's reset/reward callback uses this field to find
                # its layout-set-specific support trajectory.
                assert native.eval_seed == seed_manager.eval_seed
                self._test_reset_calls.append(list(env_seeds))
                native.camera_manager.cameras = [[object()]]
                native.camera_manager.camera_names = [["cam_head"]]

            self.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)], reset=reset)

        def close(self, clear_cache=True):
            self._closed = True

    module.RoboDojoEnv = SimulatorBase
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    path = (
        Path(__file__).resolve().parents[4] / "reweave/benchmarks/robodojo/rlinf_env.py"
    )
    spec = importlib.util.spec_from_file_location("robodojo_limit_test", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter.RoboDojoAgentEnv


def test_constructor_initializes_then_calibrates_then_resets_same_layout(
    tmp_path, agent_environment, monkeypatch
):
    events = []
    module_globals = agent_environment.__init__.__globals__

    def synchronize(native):
        assert native.camera_manager.camera_names == [["cam_head"]]
        assert len(native.camera_manager.cameras[0]) == 1
        events.append("synchronized")
        return [{"camera": "cam_head", "verified": True}]

    def refresh(native):
        assert native.camera_manager.camera_names == [["cam_head"]]
        events.append("refreshed")
        return {"physical_actions": 0, "cameras": {"cam_head": {}}}

    monkeypatch.setitem(
        module_globals, "_synchronize_tiled_camera_apertures", synchronize
    )
    monkeypatch.setitem(module_globals, "_refresh_calibrated_camera_frames", refresh)
    cfg = SimpleNamespace(seed=0, task_config=SimpleNamespace(save_dir=str(tmp_path)))

    meta = {
        "task": "play_tic_tac_toe",
        "layout_set": 2,
        "layout": 1,
        "max_episode_steps": 1100,
    }
    server = agent_environment(cfg, meta=meta)

    assert server._test_seed_manager.eval_seed == 2
    assert server.layout_selection_receipt == {
        "layout_set": 2,
        "layout": 1,
        "available_layouts": 3,
        "scene_layout_path": "/layouts/2/general_pickup_1.json",
        "source": "native SeedManager eval layout directory",
    }
    assert server._test_reset_calls == [[1]]
    launch = json.loads((tmp_path / "server_launch.json").read_text())
    assert launch["task"] == "play_tic_tac_toe"
    assert launch["native_step_limit"] == 1100
    assert launch["layout_selection"] == server.layout_selection_receipt
    assert events == ["synchronized", "refreshed"]
    assert server.camera_calibration_receipt == [
        {"camera": "cam_head", "verified": True}
    ]
    assert server.camera_refresh_receipt == {
        "physical_actions": 0,
        "cameras": {"cam_head": {}},
    }


def test_constructor_rejects_budget_that_differs_from_official_step_limit(
    tmp_path, agent_environment, monkeypatch
):
    module_globals = agent_environment.__init__.__globals__
    monkeypatch.setitem(
        module_globals, "_synchronize_tiled_camera_apertures", lambda native: []
    )
    monkeypatch.setitem(
        module_globals, "_refresh_calibrated_camera_frames", lambda native: {}
    )
    cfg = SimpleNamespace(seed=0, task_config=SimpleNamespace(save_dir=str(tmp_path)))
    meta = {
        "task": "play_tic_tac_toe",
        "layout_set": 2,
        "layout": 1,
        "max_episode_steps": 2000,
    }
    with pytest.raises(ValueError, match="official play_tic_tac_toe step_lim=1100"):
        agent_environment(cfg, meta=meta)


def test_environment_cli_separates_official_seed_directory_from_layout_index():
    pytest.importorskip("omegaconf")
    from reweave.benchmarks.robodojo.env_server import build_config, build_parser

    for seed in (0, 1):
        args = build_parser().parse_args(
            [
                "--task",
                "play_tic_tac_toe",
                "--layout-set",
                str(seed),
                "--layout",
                "0",
                "--max-episode-steps",
                "1100",
            ]
        )
        assert build_config(args).seed == seed
        assert args.layout == 0
    with pytest.raises(ValueError):
        build_config(
            build_parser().parse_args(
                [
                    "--task",
                    "play_tic_tac_toe",
                    "--layout-set",
                    "-1",
                    "--layout",
                    "0",
                    "--max-episode-steps",
                    "1100",
                ]
            )
        )


def test_client_attachment_preserves_initialized_scene(agent_environment):
    native = SimpleNamespace(take_action_cnt=[0], unstable_envs=set())
    server = agent_environment.__new__(agent_environment)

    def unexpected_reset(**kwargs):
        pytest.fail("client attachment must not reinitialize the native scene")

    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native)], reset=unexpected_reset
    )
    observation = {"initial_scene": True}
    server._observation = lambda: observation
    assert server.reset_episode() is observation
    assert server.reset_episode() is observation
    native.take_action_cnt[0] = 1
    with pytest.raises(ValueError, match="after a physical action"):
        server.reset_episode()
    assert native.take_action_cnt[0] == 1


def test_client_attachment_cannot_clear_invalid_scene(agent_environment):
    from reweave.runtime.capability import EnvironmentInvalid

    server = agent_environment.__new__(agent_environment)
    native = SimpleNamespace(take_action_cnt=[0], unstable_envs={0})
    server.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)])
    with pytest.raises(EnvironmentInvalid):
        server.reset_episode()
    assert native.unstable_envs == {0}


def test_close_acknowledgement_finalizes_all_camera_videos(tmp_path, agent_environment):
    import imageio.v2 as imageio

    from reweave.benchmarks.robodojo.agent_support import _VideoRecorder
    from reweave.benchmarks.robodojo.env_facade import RoboDojoEnvFacade

    env = agent_environment.__new__(agent_environment)
    env._closed = False
    from reweave.benchmarks.robodojo.observation_service import ObservationService

    env.observations = ObservationService(tmp_path)
    env.recorder = _VideoRecorder(str(tmp_path))
    for i in range(12):
        frame = np.full((64, 64, 3), i * 15, dtype=np.uint8)
        env.recorder.record(
            {
                "vision": {
                    cam: {"color": frame}
                    for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                }
            }
        )
    paths = list(env.recorder.paths.values())
    assert list(env.recorder.paths) == ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    assert len(paths) == 3
    facade = RoboDojoEnvFacade(env)
    try:
        facade._dispatch("env.close", (), {})
        assert facade._shutdown_event.is_set()
        assert not env._closed  # Isaac teardown follows the recording acknowledgement.
        for path in paths:
            reader = imageio.get_reader(path)
            try:
                assert reader.count_frames() == 12, path
                assert reader.get_data(0).shape == (64, 64, 3)
                assert reader.get_data(11).mean() == pytest.approx(165, abs=2)
            finally:
                reader.close()
            assert b"avc1" in Path(path).read_bytes(), path
    finally:
        facade.close()
    assert env._closed
    facade.close()  # Main-thread final cleanup remains idempotent.


@pytest.mark.parametrize(
    (
        "configured_limit",
        "native_limit",
        "terminal_after",
        "native_failure_after",
        "expected_actions",
    ),
    [
        (2, 5, None, None, 2),
        (5, 2, None, None, 2),
        (5, 5, 1, None, 1),
        (5, 5, None, 1, 1),
    ],
)
def test_native_end_and_host_cap_stop_without_unknown_or_extra_actions(
    tmp_path,
    agent_environment,
    configured_limit,
    native_limit,
    terminal_after,
    native_failure_after,
    expected_actions,
):
    rpc = SimulatedEnvironmentRPC(limit=configured_limit, terminal_after=99)
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=native_limit,
        end_flag=[False],
        success=[True],
        unstable_envs=set(),
    )
    native.is_success = lambda **kwargs: False

    def apply_action(action, action_type, *, eval_fair):
        assert not eval_fair
        assert not native.end_flag[0]
        assert native.success[0]
        assert native.take_action_cnt[0] < native.step_lim
        native.take_action_cnt[0] += 1
        rpc.actions += 1
        if rpc.actions == terminal_after:
            native.end_flag[0] = True
        if rpc.actions == native_failure_after:
            native.success[0] = False

    server = agent_environment.__new__(agent_environment)
    server.meta = rpc.meta
    server.observations = rpc.observations
    server.venv = SimpleNamespace(
        envs=[SimpleNamespace(env=native, apply_action=apply_action)]
    )
    server.get_native_obs = lambda: rpc.observation
    original_call = rpc.call

    def call(method, args=(), kwargs=None, timeout_s=None):
        if method == "env.step":
            return server.step_native(*args)
        return original_call(method, args, kwargs, timeout_s)

    rpc.call = call
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    result = toolkit.execute_tool("pi0_pick", {"max_chunks": 3})
    client = toolkit._primitives.env
    assert result.result["error"] is None
    assert rpc.actions == client.elapsed_steps() == expected_actions
    assert policy.calls == 1
    assert client.terminated or client.truncated
    assert toolkit.solved() is False
    assert server.get_status()["step_limit"] == configured_limit
    trace = toolkit._execution_trace.events
    assert sum(row["event_type"] == "control_step" for row in trace) == expected_actions
    assert not any(row["event_type"] == "action_outcome_unknown" for row in trace)
    if native_failure_after is not None:
        assert native.end_flag == [False]  # CollectEnv has not latched EvalEnv's flag.
        assert client.terminated and not client.truncated
        assert (
            sum(
                row["discarded_action_steps"]
                for row in trace
                if row["event_type"] == "action_suffix_discarded"
            )
            == 3
        )
    with pytest.raises(ValueError, match="terminal"):
        client.step(np.zeros(14))
    with pytest.raises(RuntimeError, match="exhausted|terminal"):
        server.step_native(np.zeros(14))
    assert rpc.actions == expected_actions
    toolkit.close()


@pytest.mark.parametrize(
    ("native_success", "official_success", "expected_terminal"),
    [(True, False, False), (False, False, True), (True, True, True)],
)
def test_native_failure_flag_is_distinct_from_official_success(
    agent_environment, native_success, official_success, expected_terminal
):
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=5,
        end_flag=[False],
        success=[native_success],
        unstable_envs=set(),
        is_success=lambda **kwargs: official_success,
    )
    server = agent_environment.__new__(agent_environment)
    server.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)])
    assert server._episode_ended() is expected_terminal
    assert server.is_success() is official_success


@pytest.mark.parametrize(
    "options,model,effort",
    [
        ([], "gpt-5.4", "low"),
        (
            ["--model", "fixture-model", "--reasoning-effort", "high"],
            "fixture-model",
            "high",
        ),
    ],
)
def test_robodojo_native_defaults_preserve_explicit_settings(options, model, effort):
    from reweave.cli.main import parse_cli_arguments

    _, args, _ = parse_cli_arguments(
        [
            "--robot",
            "robodojo",
            "--task",
            "play_tic_tac_toe",
            "--layout-set",
            "1",
            "--layout",
            "0",
            "--max-episode-steps",
            "1100",
            *options,
        ]
    )
    assert (args.model, args.reasoning_effort, args.native_service_tier) == (
        model,
        effort,
        "default",
    )
