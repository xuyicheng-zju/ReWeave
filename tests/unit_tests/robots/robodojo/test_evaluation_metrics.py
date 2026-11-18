"""Partial native scores survive failed tasks without becoming an online oracle."""

import json
from types import SimpleNamespace

import pytest

from reweave.benchmarks.robodojo.evaluation import terminal_metrics
from tests.unit_tests.robots.robodojo import test_runtime_limits

agent_environment = test_runtime_limits.agent_environment


@pytest.mark.parametrize(
    "success,process_score,expected",
    [(False, 40, 0.4), (True, 40, 1.0), (False, 0, 0.0)],
)
def test_official_success_override_and_native_partial_score(
    success, process_score, expected
):
    calls = []
    env = SimpleNamespace(
        unstable_envs=set(),
        is_success=lambda **kwargs: calls.append("success") or success,
        get_score=lambda: pytest.fail("score registration must precede actions"),
        reward_manager=SimpleNamespace(
            get_score=lambda: calls.append("score") or [process_score]
        ),
    )
    assert terminal_metrics(env) == {
        "valid": True,
        "success": success,
        "score": expected,
    }
    assert calls == (["success"] if success else ["success", "score"])


def test_binary_task_and_unstable_scene_have_no_invented_partial_score():
    env = SimpleNamespace(unstable_envs=set(), is_success=lambda **kwargs: False)
    assert terminal_metrics(env)["score"] == 0.0
    env.unstable_envs.add(0)
    env.is_success = lambda **kwargs: pytest.fail("invalid scene has no verdict")
    assert terminal_metrics(env) == {"valid": False, "success": None, "score": None}


@pytest.mark.parametrize("score", [float("nan"), -1, 101])
def test_malformed_native_score_is_rejected(score):
    env = SimpleNamespace(
        unstable_envs=set(),
        is_success=lambda **kwargs: False,
        get_score=lambda: None,
        reward_manager=SimpleNamespace(get_score=lambda: [score]),
    )
    with pytest.raises(ValueError, match="native RoboDojo score"):
        terminal_metrics(env)


def test_close_publishes_partial_score_once_and_flushes_video(
    tmp_path, agent_environment
):
    server = agent_environment.__new__(agent_environment)
    native = SimpleNamespace(
        unstable_envs=set(),
        take_action_cnt=[7],
        is_success=lambda **kwargs: False,
        get_score=lambda: None,
        reward_manager=SimpleNamespace(get_score=lambda: [40]),
    )
    server.venv = SimpleNamespace(envs=[SimpleNamespace(env=native)])
    server.meta = {"task": "stack_bowls", "layout_set": 0, "layout": 1}
    server._evaluation_ready = True
    server._metrics_path = tmp_path / "official_metrics.json"
    flushed = []
    server.recorder = SimpleNamespace(close=lambda: flushed.append(True) or [])
    server.finish_recording()
    original = server._metrics_path.read_bytes()
    metrics = json.loads(original)
    assert metrics["score"] == 0.4 and metrics["success"] is False
    assert metrics["physical_steps"] == 7 and metrics["layout_set"] == 0
    native.reward_manager.get_score = lambda: pytest.fail("terminal score is immutable")
    server.finish_recording()
    assert server._metrics_path.read_bytes() == original and len(flushed) == 2


def test_incomplete_constructor_does_not_fabricate_terminal_metrics(
    tmp_path, agent_environment
):
    server = agent_environment.__new__(agent_environment)
    server._evaluation_ready = False
    server._metrics_path = tmp_path / "official_metrics.json"
    server.recorder = SimpleNamespace(close=lambda: [])
    server.finish_recording()
    assert not server._metrics_path.exists()


def test_score_checks_register_after_scene_reset_before_observation(
    tmp_path, agent_environment, monkeypatch
):
    namespace = agent_environment.__init__.__globals__
    original = namespace["_select_native_eval_layout"]
    events = []

    def select(native, layout_set, layout):
        def register():
            assert native.camera_manager.camera_names == [["cam_head"]]
            events.append("score_registered")

        native.get_score = register
        return original(native, layout_set, layout)

    def refresh(native):
        assert events == ["score_registered"]
        events.append("observation_refreshed")
        return {}

    monkeypatch.setitem(namespace, "_select_native_eval_layout", select)
    monkeypatch.setitem(
        namespace, "_synchronize_tiled_camera_apertures", lambda native: []
    )
    monkeypatch.setitem(namespace, "_refresh_calibrated_camera_frames", refresh)
    cfg = SimpleNamespace(seed=0, task_config=SimpleNamespace(save_dir=str(tmp_path)))
    server = agent_environment(
        cfg,
        meta={
            "task": "play_tic_tac_toe",
            "layout_set": 0,
            "layout": 1,
            "max_episode_steps": 1100,
        },
    )
    assert server._evaluation_ready is True
    assert events == ["score_registered", "observation_refreshed"]
