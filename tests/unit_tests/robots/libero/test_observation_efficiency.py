"""LIBERO reuses exact settled reads and converts only requested chunk images."""

from collections import Counter

import numpy as np
import pytest

from reweave.benchmarks.libero.env_client import LiberoEnvClient
from reweave.benchmarks.libero.env_server import LiberoEnvFacade


class _RPC:
    def __init__(self):
        self.meta = {"max_episode_steps": 100}
        self.calls = Counter()
        self.step_number = 0
        self.fail_step = False

    def call(self, method, *, kwargs=None, **unused):
        self.calls[method] += 1
        if method == "env.get_env_meta":
            return self.meta
        if method == "env.reset":
            self.step_number = 0
            return {"policy_image": np.zeros((2, 2, 3), np.uint8)}, {}
        if method in {"env.step", "env.chunk_step"}:
            if self.fail_step:
                raise TimeoutError("lost action response")
            self.step_number += 1
            obs = {"policy_image": np.full((2, 2, 3), self.step_number, np.uint8)}
            if method == "env.chunk_step" and kwargs["return_all_frames"]:
                obs = [obs, obs]
            return obs, 0, np.zeros(2, bool), np.zeros(2, bool), {}
        if method == "env.raw_obs":
            return {"joint": np.array([self.step_number])}
        if method == "env.get_camera_meta":
            return {"intrinsics": np.eye(3) * kwargs["width"]}
        if method == "env.get_task_language":
            return "Put the object in the bowl."
        if method == "env.render_camera":
            rgb = np.full(
                (kwargs["height"], kwargs["width"], 3), self.step_number, np.uint8
            )
            if kwargs["depth"]:
                return rgb, np.ones(rgb.shape[:2], np.float32)
            return rgb
        raise AssertionError(method)


def test_exact_camera_reads_reuse_rgbd_without_replacing_native_resolution():
    rpc = _RPC()
    env = LiberoEnvClient(rpc, expected_meta=rpc.meta)
    rgb, depth = env.render_camera("agentview", height=8, width=9, depth=True)
    rgb[:] = 255
    depth[:] = 99
    assert env.render_camera("agentview", height=8, width=9).shape == (8, 9, 3)
    rgb2, depth2 = env.render_camera("agentview", height=8, width=9, depth=True)
    assert not rgb2.any() and (depth2 == 1).all()
    assert rpc.calls["env.render_camera"] == 1
    env.render_camera("agentview", height=4, width=5)
    env.render_camera("robot0_eye_in_hand", height=8, width=9)
    assert rpc.calls["env.render_camera"] == 3
    assert env.last_obs["policy_image"].shape == (2, 2, 3)


@pytest.mark.parametrize("mutation", ["step", "chunk_step", "reset"])
def test_reads_refresh_at_each_settled_mutation_and_copies_cannot_pollute_cache(
    mutation,
):
    rpc = _RPC()
    env = LiberoEnvClient(rpc, expected_meta=rpc.meta)
    for _ in range(2):
        env.raw_obs()["joint"][:] = 77
        env.get_camera_meta(width=17)["intrinsics"][:] = 77
        env.get_task_language()
    assert env.raw_obs()["joint"].tolist() == [0]
    assert env.get_camera_meta(width=17)["intrinsics"][0, 0] == 17
    assert rpc.calls["env.raw_obs"] == rpc.calls["env.get_camera_meta"] == 1
    assert rpc.calls["env.get_task_language"] == 1
    if mutation == "reset":
        env.reset()
    else:
        getattr(env, mutation)(
            np.zeros((2, 7)) if mutation == "chunk_step" else np.zeros(7)
        )
    env.raw_obs()
    env.get_camera_meta(width=17)
    env.get_task_language()
    assert rpc.calls["env.raw_obs"] == rpc.calls["env.get_camera_meta"] == 2
    assert rpc.calls["env.get_task_language"] == 2


@pytest.mark.parametrize("mutation", ["step", "chunk_step"])
def test_lost_action_response_never_reuses_or_fetches_a_scene_read(mutation):
    rpc = _RPC()
    env = LiberoEnvClient(rpc, expected_meta=rpc.meta)
    env.raw_obs()
    env.render_camera(height=3, width=4, depth=True)
    rpc.fail_step = True
    with pytest.raises(TimeoutError):
        getattr(env, mutation)(
            np.zeros((2, 7)) if mutation == "chunk_step" else np.zeros(7)
        )
    for read in [
        env.raw_obs,
        env.render_camera,
        env.get_camera_meta,
        env.get_task_language,
    ]:
        with pytest.raises(RuntimeError, match="no settled action boundary"):
            read()
    assert rpc.calls["env.raw_obs"] == rpc.calls["env.render_camera"] == 1


class _Tensor:
    def __init__(self, number, *, forbidden=False):
        self.number = number
        self.forbidden = forbidden
        self.conversions = 0

    def detach(self):
        if self.forbidden:
            raise AssertionError("unused earlier chunk observation was materialized")
        return self

    def cpu(self):
        return self

    def numpy(self):
        self.conversions += 1
        return np.full((1, 8), self.number)


def _chunk_facade(observations):
    class Env:
        def chunk_step(self, actions):
            return (
                observations,
                np.zeros((1, 2)),
                np.zeros((1, 2), bool),
                np.zeros((1, 2), bool),
                [{"index": 0}, {"index": 1}],
            )

    facade = LiberoEnvFacade(Env(), meta={})
    facade._attach_critic_evidence = lambda info, observation: {
        **info,
        "critic_evidence": {"final_state": int(observation["states"][0])},
    }
    return facade


def test_final_only_chunk_does_not_materialize_unused_observations():
    before, final = _Tensor(1, forbidden=True), _Tensor(2)
    facade = _chunk_facade([{"states": before}, {"states": final}])
    obs, reward, term, trunc, info = facade.chunk_step(np.zeros((2, 7)))
    assert obs["states"].tolist() == [2] * 8
    assert before.conversions == 0 and final.conversions == 1
    assert reward.shape == term.shape == trunc.shape == (2,)
    assert info == [{"index": 0}, {"index": 1, "critic_evidence": {"final_state": 2}}]


def test_full_video_chunk_preserves_every_requested_observation():
    before, final = _Tensor(1), _Tensor(2)
    facade = _chunk_facade([{"states": before}, {"states": final}])
    obs, _, _, _, info = facade.chunk_step(np.zeros((2, 7)), return_all_frames=True)
    assert [frame["states"][0] for frame in obs] == [1, 2]
    assert before.conversions == final.conversions == 1
    assert info[-1]["critic_evidence"]["final_state"] == 2
