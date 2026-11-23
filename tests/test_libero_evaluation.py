# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""LIBERO-Pro evaluation protocol regression tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reweave.benchmarks.libero import robot_spec
from reweave.benchmarks.libero.env_server import assert_libero_backend, build_env_cfg
from reweave.benchmarks.libero.evaluation import (
    aspire_trial_partition,
    aspire_trial_to_reset_state_id,
    libero_horizon_contract,
)
from reweave.benchmarks.libero.tools import LiberoPrimitives
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace


def test_goal_swap_uses_zetta_openpi_control_contract():
    contract = libero_horizon_contract("libero_goal_swap")

    assert contract["policy_action_horizon"] == 300
    assert contract["wait_steps"] == 10
    assert contract["max_episode_steps"] == 310
    assert contract["actions_per_chunk"] == 5
    assert contract["is_standard"] is True


def test_aspire_trial_split_and_one_based_reset_mapping():
    assert aspire_trial_partition(51) == "development"
    assert aspire_trial_partition(1) == "heldout"
    assert aspire_trial_to_reset_state_id(51, 50) == 0
    assert aspire_trial_to_reset_state_id(52, 50) == 1

    with pytest.raises(ValueError, match="trial"):
        aspire_trial_partition(66)


def test_rlinf_libero_backend_fallback_is_rejected():
    assert_libero_backend("pro", "liberopro.liberopro.benchmark")

    with pytest.raises(RuntimeError, match="Refusing the RLinf fallback"):
        assert_libero_backend("pro", "libero.libero.benchmark")


def test_inner_robosuite_horizon_cannot_preempt_outer_evaluation_horizon(monkeypatch):
    class Node(dict):
        def __getattr__(self, key):
            return self[key]

    def convert(value):
        if isinstance(value, dict):
            return Node({key: convert(item) for key, item in value.items()})
        return value

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(OmegaConf=SimpleNamespace(create=convert)),
    )
    cfg = build_env_cfg(max_episode_steps=310)

    assert cfg.max_episode_steps == 310
    assert cfg.init_params.horizon == 1310
    assert cfg.init_params.ignore_done is True


@pytest.mark.parametrize(
    ("spawn", "module"),
    (
        (robot_spec._spawn_env_server, "reweave.benchmarks.libero.env_server"),
        (
            robot_spec._spawn_vla_server,
            "reweave.robots.components.pi05_vla_server",
        ),
        (
            robot_spec._spawn_sam3_server,
            "reweave.robots.components.sam3_server",
        ),
    ),
)
def test_libero_components_are_spawned_as_importable_modules(
    monkeypatch,
    tmp_path: Path,
    spawn,
    module: str,
):
    class Daemon:
        def __init__(self, **kwargs):
            self.cmd = kwargs["cmd"]

        def start(self):
            return None

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    monkeypatch.setattr(
        robot_spec,
        "reserve_free_port",
        lambda: SimpleNamespace(host="127.0.0.1", port=12345),
    )
    args = SimpleNamespace(
        env_endpoint=None,
        vla_endpoint=None,
        sam3_endpoint=None,
        suite="libero_spatial_task",
        task=0,
        seed=51,
        reset_state_id=0,
        max_episode_steps=230,
        cuda_device=None,
        libero_type="pro",
    )

    daemon, _ = spawn(args, tmp_path)

    assert daemon.cmd[:3] == [sys.executable, "-m", module]


def test_libero_reset_executes_counted_openpi_wait_actions():
    class Env:
        terminated = False
        truncated = False

        def __init__(self):
            self.actions = []

        @staticmethod
        def _obs():
            return {
                "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
                "states": np.zeros(8, dtype=np.float32),
            }

        @staticmethod
        def _info():
            return {
                "critic_evidence": {
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
                },
            }

        def reset(self):
            return self._obs(), self._info()

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return self._obs(), 0.0, False, False, self._info()

        def elapsed_steps(self):
            return len(self.actions)

    env = Env()
    trace = ExecutionTrace()
    primitives = LiberoPrimitives(
        env=env,
        model=object(),
        sam3_client=object(),
        check_cancelled=lambda: None,
        execution_trace=trace,
        active_execution=ActiveExecution(trace),
        wait_steps=10,
    )
    primitives.start_recording()

    primitives.reset()

    assert len(env.actions) == 10
    assert len(primitives.frame_slice(0)) == 10
    assert all(action.tolist() == [0.0] * 6 + [-1.0] for action in env.actions)


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("official", [True, False, None])
def test_comparison_video_uses_only_official_success(tmp_path, official):
    import json

    from reweave.benchmarks.libero.make_comparison import _outcome

    (tmp_path / "transcript_run.json").write_text(
        json.dumps({"official_success": official, "finish": {"status": "success"}})
    )
    (tmp_path / "states.json").write_text('{"steps": [{"terminated": true}]}')
    assert _outcome(tmp_path)[0] is official


def test_comparison_video_rejects_multiple_episodes(tmp_path):
    from reweave.benchmarks.libero.make_comparison import _outcome

    for name in ("first", "second"):
        (tmp_path / f"transcript_{name}.json").write_text('{"official_success": true}')
    with pytest.raises(ValueError, match="one episode"):
        _outcome(tmp_path)
