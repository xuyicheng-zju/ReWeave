# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""RoboDojo public observation, policy and exact episode contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedPolicyRPC, simulated_observation
from reweave.benchmarks.robodojo.access import (
    graph_features,
    native_action,
    public_observation,
    robot_state_features,
)
from reweave.benchmarks.robodojo.tasks import validate_instruction, validate_task
from reweave.evolution.contracts import Instance
from reweave.robots.components.pi05_vla_client import Pi05VLAClient
from reweave.robots.components.pi05_vla_server import Pi05VLAFacade, build_model_cfg
from tests.unit_tests.harness.evolution.test_rollout_recipe import recipe_fixture


def test_policy_encodes_exact_three_rgb_and_fourteen_joints():
    obs = simulated_observation()
    for offset, key in enumerate(
        (
            "left_arm_joint_state",
            "right_arm_joint_state",
            "left_ee_joint_state",
            "right_ee_joint_state",
        )
    ):
        obs["state"][key] += offset
    rpc = SimulatedPolicyRPC()
    client = Pi05VLAClient(rpc, embodiment="robodojo")
    prediction = client.predict_action_batch(obs)
    wire = rpc.requests[0]
    assert prediction.actions.shape == (4, 14)
    assert wire["states"].shape == (1, 14)
    np.testing.assert_equal(wire["states"][0], [0.0] * 6 + [1.0] * 6 + [3.0, 4.0])
    for key, pixel in (
        ("main_images", 0),
        ("wrist_images", 70),
        ("extra_view_images", 140),
    ):
        assert wire[key].shape == (1, 4, 5, 3) and wire[key].dtype == np.uint8
        assert int(wire[key][0, 0, 0, 0]) == pixel
    assert wire["task_descriptions"] == [obs["instruction"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cam_head", np.zeros((4, 5, 4))),
        ("cam_left_wrist", np.zeros((4, 5))),
        ("cam_right_wrist", np.zeros((0, 5, 3))),
        ("left_arm_joint_state", np.zeros(7)),
        ("right_arm_joint_state", np.full(6, np.nan)),
        ("left_ee_joint_state", np.zeros(2)),
        ("right_ee_joint_state", np.ones((1, 1))),
    ],
)
def test_encoder_rejects_malformed_rgb_and_joint_dimensions(field, value):
    obs = simulated_observation()
    if field.startswith("cam_"):
        obs["vision"][field]["color"] = value
    else:
        obs["state"][field] = value
    with pytest.raises(ValueError):
        Pi05VLAClient(SimulatedPolicyRPC(), embodiment="robodojo").encode_obs(obs)


def test_policy_rejects_wrong_action_dimension_before_physical_io():
    with pytest.raises(ValueError, match="14 joint action"):
        Pi05VLAClient(
            SimulatedPolicyRPC(action_dimension=7), embodiment="robodojo"
        ).predict_action_batch(simulated_observation())


def test_server_configuration_matches_robodojo_checkpoint_contract():
    pytest.importorskip("omegaconf")
    cfg = build_model_cfg(
        "/simulated/weights",
        embodiment="robodojo",
        norm_stats_path="/simulated/stats",
        repo_id="arx_x5",
    )
    assert cfg.pi05 is True and cfg.action_dim == 14 and cfg.num_action_chunks == 50
    assert cfg.openpi.config_name == "pi05_robodojo_arx_x5"
    assert cfg.openpi.num_images_in_input == 3 and cfg.openpi.action_env_dim == 14
    assert cfg.openpi.action_chunk == 50 and cfg.openpi.model_action_dim == 32
    assert cfg.openpi_data.norm_stats_path == "/simulated/stats"


@pytest.mark.parametrize(
    "field,value",
    [
        ("states", np.zeros((1, 13))),
        ("wrist_images", None),
        ("extra_view_images", None),
    ],
)
def test_server_rejects_incomplete_robodojo_input_before_inference(field, value):
    facade = object.__new__(Pi05VLAFacade)
    facade._embodiment = "robodojo"
    facade._prediction_horizon_steps = 50
    facade._inference_context = nullcontext
    facade._model = SimpleNamespace(
        predict_action_batch=lambda *a, **k: pytest.fail(
            "malformed request reached inference"
        )
    )
    wire = Pi05VLAClient(SimulatedPolicyRPC(), embodiment="robodojo").encode_obs(
        simulated_observation()
    )
    wire[field] = value
    with pytest.raises(ValueError):
        facade.predict(wire)


def test_public_evidence_excludes_oracle_fields_recursively():
    obs = public_observation(simulated_observation())
    assert set(obs) == {"state", "vision", "instruction"}
    assert "object_pose" not in obs["state"]
    assert all("object_id" not in camera for camera in obs["vision"].values())
    assert "oracle_success" not in obs


@pytest.mark.parametrize("defect", ["missing", "extra", "shape", "nonfinite"])
def test_graph_projection_rejects_incomplete_or_malformed_native_telemetry(defect):
    values = robot_state_features(simulated_observation())
    if defect == "missing":
        del values["left_ee_pose"]
    elif defect == "extra":
        values["oracle_success"] = [1.0]
    elif defect == "shape":
        values["left_ee_pose"] = [0.0] * 6
    else:
        values["right_arm_joint_state"][0] = float("inf")
    with pytest.raises(ValueError, match="RoboDojo"):
        graph_features(values)


@pytest.mark.parametrize(
    "value", [np.zeros(13), np.full(14, np.inf), {"right_arm_joint_state": [0.0] * 6}]
)
def test_native_action_rejects_partial_or_malformed_dual_arm_commands(value):
    with pytest.raises(ValueError):
        native_action(value)


@pytest.mark.parametrize("value", ["", "pick <object>", None])
def test_instruction_must_be_resolved(value):
    with pytest.raises(ValueError):
        validate_instruction(value)


@pytest.mark.parametrize("task", ["../task", " task", "", ".", ".."])
def test_task_rejects_noncanonical_basename(task):
    assert validate_task(task) is not None


@pytest.mark.parametrize(
    "initialization",
    [
        {"task": "put_bottles_into_dustbin", "layout": -1},
        {"task": "put_bottles_into_dustbin", "layout": True},
        {"task": "put_bottles_into_dustbin", "layout": 0, "seed": 0},
        {"task": "../task", "layout": 0},
    ],
)
def test_rollout_rejects_invalid_task_layout_before_start(tmp_path, initialization):
    runner = recipe_fixture(tmp_path)[2]
    instance = Instance(
        record_id="invalid-robodojo-condition",
        task="robodojo:put_bottles_into_dustbin",
        initialization=initialization,
    )
    with pytest.raises(ValueError):
        runner.build_arguments(
            instance,
            directory=tmp_path / "attempt",
            memory=tmp_path / "memory",
            device=0,
        )


@pytest.mark.parametrize(
    "field,value", [("task", "fill_pen_holder"), ("layout_set", 1), ("layout", 7)]
)
def test_client_rejects_different_episode_identity_before_reset(field, value):
    from examples.robodojo_demo import SimulatedEnvironmentRPC
    from reweave.benchmarks.robodojo.env_client import RoboDojoEnvClient

    rpc = SimulatedEnvironmentRPC()
    expected = {**rpc.meta, field: value}
    with pytest.raises(ValueError, match="env_meta mismatch"):
        RoboDojoEnvClient(rpc, expected_meta=expected)
    assert rpc.actions == 0


def test_rollout_rejects_physical_budget_that_differs_from_official_step_limit(
    tmp_path,
):
    from pathlib import Path

    from reweave.benchmarks.robodojo.tasks import native_step_limit
    from reweave.evolution.contracts import EvolutionProtocol

    path = (
        Path(__file__).resolve().parents[4]
        / "examples/graph_evolution/robodojo_protocol.json"
    )
    protocol = EvolutionProtocol.model_validate_json(path.read_text())
    runner = recipe_fixture(tmp_path)[2]
    assert runner.runtime_manifest.conditions.physical_steps != native_step_limit(
        protocol.instances[0].initialization["task"]
    )
    with pytest.raises(ValueError, match="official step_lim"):
        runner.build_arguments(
            protocol.instances[0],
            directory=tmp_path / "attempt",
            memory=tmp_path / "memory",
            device=0,
        )


def test_complete_robodojo_example_reaches_rollout_argument_builder(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from reweave.evolution.contracts import EvolutionProtocol

    path = (
        Path(__file__).resolve().parents[4]
        / "examples/graph_evolution/robodojo_protocol.json"
    )
    protocol = EvolutionProtocol.model_validate_json(path.read_text())
    runner = recipe_fixture(tmp_path)[2]
    # The shared fixture budget is not a RoboDojo step_lim; argv shape is under test.
    monkeypatch.setattr(
        "reweave.benchmarks.robodojo.tasks.native_step_limit",
        lambda task: runner.runtime_manifest.conditions.physical_steps,
    )
    for instance in protocol.instances:
        args = runner.build_arguments(
            instance,
            directory=tmp_path / "attempt",
            memory=tmp_path / "memory",
            device=0,
        )
        assert args[args.index("--layout") + 1] == str(
            instance.initialization["layout"]
        )
        assert "--seed" not in args
        assert args[args.index("--layout-set") + 1] == str(
            instance.initialization["layout_set"]
        )
    for invalid in (None, True, -1, "0"):
        values = protocol.model_dump(mode="json")
        values["dev"][0]["initialization"]["layout"] = invalid
        values["dev"][0]["initialization"]["seed"] = 0
        with pytest.raises(ValueError, match="nonnegative integer layout"):
            EvolutionProtocol.model_validate(values)
