# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reweave.benchmarks.libero.env_server import LiberoEnvFacade
from reweave.benchmarks.libero.privileged_state import (
    attach_privileged_critic_state,
    collect_privileged_critic_state,
)


class _Model:
    body_names = ["world", "robot0_hand", "red_cup", "target_bowl"]
    geom_names = [
        "robot0_left_finger",
        "robot0_right_finger",
        "red_cup_geom",
        "target_bowl_geom",
    ]
    site_names = ["gripper0_grip_site"]
    joint_names = ["cabinet_slide", "stove_knob_hinge"]
    nbody = len(body_names)
    ngeom = len(geom_names)
    nsite = len(site_names)
    njnt = len(joint_names)
    body_parentid = np.asarray([0, 0, 0, 0])
    geom_bodyid = np.asarray([1, 1, 2, 3])
    jnt_limited = np.asarray([True, True])
    jnt_type = np.asarray([2, 3])
    jnt_qposadr = np.asarray([0, 1])
    jnt_range = np.asarray([[0.0, 0.4], [-1.0, 1.0]])

    def body_name2id(self, name: str) -> int:
        return self.body_names.index(name)

    def body_id2name(self, identifier: int) -> str:
        return self.body_names[identifier]

    def geom_name2id(self, name: str) -> int:
        return self.geom_names.index(name)

    def geom_id2name(self, identifier: int) -> str:
        return self.geom_names[identifier]

    def site_name2id(self, name: str) -> int:
        return self.site_names.index(name)

    def site_id2name(self, identifier: int) -> str:
        return self.site_names[identifier]

    def joint_id2name(self, identifier: int) -> str:
        return self.joint_names[identifier]


class _Contact:
    def __init__(self, first: int, second: int) -> None:
        self.geom1 = first
        self.geom2 = second


def _task_env() -> Any:
    model = _Model()
    data = SimpleNamespace(
        body_xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.1],
                [0.3, 0.0, 0.1],
            ]
        ),
        body_xquat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (4, 1)),
        site_xpos=np.asarray([[0.0, 0.0, 0.2]]),
        qpos=np.asarray([0.1, 0.5]),
        ncon=2,
        contact=[_Contact(0, 2), _Contact(1, 2)],
    )
    gripper = SimpleNamespace(
        contact_geoms=["robot0_left_finger", "robot0_right_finger"]
    )
    robot_model = SimpleNamespace(
        contact_geoms=["robot0_left_finger", "robot0_right_finger"]
    )
    red_cup = SimpleNamespace(
        name="red_cup",
        root_body="red_cup",
        contact_geoms=["red_cup_geom"],
    )
    target_bowl = SimpleNamespace(
        name="target_bowl",
        root_body="target_bowl",
        contact_geoms=["target_bowl_geom"],
    )
    raw = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[SimpleNamespace(gripper=gripper, robot_model=robot_model)],
        objects_dict={"red_cup": red_cup, "target_bowl": target_bowl},
        parsed_problem={"goal_state": [("In", "red_cup", "target_bowl")]},
        relation_satisfied=False,
    )
    raw._eval_predicate = lambda atom: raw.relation_satisfied and atom[0] == "In"
    raw._check_success = lambda: raw.relation_satisfied
    return raw


def test_privileged_state_exposes_goal_pose_contact_grasp_and_joints() -> None:
    state = collect_privileged_critic_state(SimpleNamespace(env=_task_env()))

    assert state["privileged.available"] is True
    assert state["privileged.task.semantic_available"] is True
    assert state["privileged.task.primary_relation"] == "In"
    assert state["privileged.task.manipulated_object.name"] == "red_cup"
    assert state["privileged.task.target.name"] == "target_bowl"
    assert state["privileged.task.manipulated_object.grasped"] is True
    assert state["privileged.task.manipulated_object.in_target"] is False
    assert state[
        "privileged.task.manipulated_object.distance_to_target_m"
    ] == pytest.approx(0.3)
    assert state["privileged.entity.red_cup.gripper_contact"] is True
    assert state["privileged.entity.target_bowl.position.x"] == pytest.approx(0.3)
    assert state[
        "privileged.joint.cabinet_slide.fraction_from_lower_limit"
    ] == pytest.approx(0.25)
    assert state[
        "privileged.joint.stove_knob_hinge.distance_to_upper"
    ] == pytest.approx(0.5)


def test_privileged_state_tracks_retention_release_and_target_progress() -> None:
    raw = _task_env()
    wrapped = SimpleNamespace(env=raw)
    attach_privileged_critic_state(wrapped)

    initial = wrapped.harness_privileged_critic_state(reset_tracker=True)
    expected_object_task_features = {
        "privileged.available",
        "privileged.task.semantic_available",
        "privileged.task.goal.predicate_count",
        "privileged.task.goal.evaluable_count",
        "privileged.task.goal.satisfied_count",
        "privileged.task.goal.predicate_fraction_available",
        "privileged.task.goal.predicate_fraction",
        "privileged.task.primary_relation",
        "privileged.task.primary_relation_satisfied",
        "privileged.task.target_identity_confidence",
        "privileged.task.manipulated_object.name",
        "privileged.task.manipulated_object.position.x",
        "privileged.task.manipulated_object.position.y",
        "privileged.task.manipulated_object.position.z",
        "privileged.task.manipulated_object.distance_to_eef_m",
        "privileged.task.manipulated_object.distance_to_target_m",
        "privileged.task.manipulated_object.gripper_contact",
        "privileged.task.manipulated_object.robot_contact",
        "privileged.task.manipulated_object.grasped",
        "privileged.task.manipulated_object.ever_grasped",
        "privileged.task.manipulated_object.retained",
        "privileged.task.manipulated_object.released_now",
        "privileged.task.manipulated_object.ever_released",
        "privileged.task.manipulated_object.in_target",
        "privileged.task.manipulated_object.target_progress_available",
        "privileged.task.target.name",
        "privileged.task.target.position.x",
        "privileged.task.target.position.y",
        "privileged.task.target.position.z",
        "privileged.task.target.distance_to_eef_m",
        "privileged.task.target.gripper_contact",
        "privileged.task.target.robot_contact",
        "privileged.task.target.is_nearest_entity",
        "privileged.task.target.distance_rank",
        "privileged.task.nearest_entity.name",
        "privileged.task.nearest_entity.distance_m",
        "privileged.contact.robot.count",
        "privileged.contact.gripper.count",
        "privileged.contact.force_available",
        "privileged.contact.max_normal_force_n",
        "privileged.joint.count",
    }
    assert expected_object_task_features <= set(initial)
    assert initial["privileged.task.manipulated_object.ever_grasped"] is True
    assert (
        initial["privileged.task.manipulated_object.target_progress_available"] is False
    )

    raw.sim.data.body_xpos[2, 0] = 0.1
    progressed = wrapped.harness_privileged_critic_state()
    assert progressed[
        "privileged.task.manipulated_object.target_progress_m"
    ] == pytest.approx(0.1)
    assert progressed["privileged.task.manipulated_object.retained"] is True

    raw.sim.data.ncon = 0
    raw.sim.data.contact = []
    released = wrapped.harness_privileged_critic_state()
    assert released["privileged.task.manipulated_object.released_now"] is True
    assert released["privileged.task.manipulated_object.ever_released"] is True
    assert released["privileged.task.manipulated_object.retained"] is False


def test_privileged_state_selects_the_unsatisfied_multi_goal_object() -> None:
    raw = _task_env()
    raw.parsed_problem["goal_state"] = [
        ("In", "red_cup", "target_bowl"),
        ("On", "target_bowl", "red_cup"),
    ]
    raw._eval_predicate = lambda atom: atom[0] == "In"

    state = collect_privileged_critic_state(SimpleNamespace(env=raw))

    assert state["privileged.task.goal.satisfied_count"] == 1
    assert state["privileged.task.primary_relation"] == "On"
    assert state["privileged.task.manipulated_object.name"] == "target_bowl"
    assert state["privileged.task.target.name"] == "red_cup"


def test_privileged_state_fails_when_no_simulator_exists() -> None:
    with pytest.raises(RuntimeError, match="MuJoCo simulator"):
        collect_privileged_critic_state(SimpleNamespace())


def test_env_facade_completes_eef_distances_without_actor_leakage() -> None:
    class Worker:
        def env_call(self, method, *, kwargs, target):
            assert method == "harness_privileged_critic_state"
            assert kwargs == {"reset_tracker": True}
            assert target == "self"
            return {
                "privileged.available": True,
                "privileged.task.manipulated_object.position.x": 0.1,
                "privileged.task.manipulated_object.position.y": 0.0,
                "privileged.task.manipulated_object.position.z": 0.2,
                "privileged.entity.red_cup.position.x": 0.1,
                "privileged.entity.red_cup.position.y": 0.0,
                "privileged.entity.red_cup.position.z": 0.2,
            }

    class Env:
        env = SimpleNamespace(workers=[Worker()])

        @staticmethod
        def reset():
            states = np.zeros((1, 8), dtype=np.float32)
            return {"states": states}, {}

    facade = LiberoEnvFacade(Env(), meta={})
    observation, info = facade.reset()

    assert not any(key.startswith("privileged.") for key in observation)
    assert info["critic_evidence"][
        "privileged.task.manipulated_object.distance_to_eef_m"
    ] == pytest.approx(np.sqrt(0.05))
    assert info["critic_evidence"][
        "privileged.entity.red_cup.distance_to_eef_m"
    ] == pytest.approx(np.sqrt(0.05))


def test_env_facade_preserves_chunk_info_sequence_and_enriches_last_step() -> None:
    class Worker:
        def env_call(self, method, *, kwargs, target):
            assert method == "harness_privileged_critic_state"
            assert kwargs == {"reset_tracker": False}
            assert target == "self"
            return {"privileged.available": True}

    class Env:
        env = SimpleNamespace(workers=[Worker()])

        @staticmethod
        def chunk_step(actions):
            assert actions.shape == (1, 2, 7)
            states = np.zeros((1, 8), dtype=np.float32)
            observations = [{"states": states}, {"states": states}]
            return (
                observations,
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 2), dtype=bool),
                np.zeros((1, 2), dtype=bool),
                [{"index": 0}, {"index": 1}],
            )

    facade = LiberoEnvFacade(Env(), meta={})
    _obs, _reward, _terminated, _truncated, infos = facade.chunk_step(
        np.zeros((2, 7), dtype=np.float32)
    )

    assert infos[0] == {"index": 0}
    assert infos[1]["index"] == 1
    assert infos[1]["critic_evidence"] == {"privileged.available": True}
