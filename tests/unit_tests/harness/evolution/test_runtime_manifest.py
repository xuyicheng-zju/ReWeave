# Modified for the ReWeave namespace.
"""Actual configuration semantics and atomic baseline publication."""

import pytest

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.runtime_manifest import (
    RuntimeManifest,
    RuntimeResources,
    capture_baseline,
    dependency_inventory,
)
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import protocol
from tests.unit_tests.harness.evolution.test_bundle_runtime import Robot


def fixture(tmp_path):
    repository = tmp_path / "repo"
    (repository / "reweave/planner").mkdir(parents=True)
    (repository / "reweave/planner/base.py").write_text("SUCCESS = True\n")
    (repository / "reweave/benchmarks/libero").mkdir(parents=True)
    asset = repository / "reweave/benchmarks/libero/asset.json"
    asset.write_text('{"fixture": true}')
    memory = tmp_path / "memory"
    memory.mkdir()
    robot = Robot(tmp_path / "episode")
    planner = ScriptedNativePlanner(
        lambda *_: None,
        max_tokens=123,
        dashboard_events=NullDashboardEventSink(),
    )
    resources = RuntimeResources(
        models={},
        environment={str(asset): {"purpose": "synthetic environment fixture"}},
        routes={"controller": "deterministic fixture"},
    )
    conditions = protocol().execution.model_copy(
        update={
            "model_sampling": planner.sampling_settings(),
            "controller_request_unit": "native_turn",
        }
    )
    arguments = {
        "planner": "native",
        "adaptation_mode": "static",
        "robot_name": "libero",
        "max_episode_steps": 100,
        "wait_steps": 10,
        "max_turns": 20,
        "planner_timeout_s": 30,
        "memory_dir": str(memory),
        "suite": "libero_goal",
        "task": 0,
        "seed": 0,
    }
    limits = {
        "episode": {"program_calls": 20, "gateway_calls": 20, "wall_time_s": 30},
    }
    return {
        "arguments": arguments,
        "limits": limits,
        "conditions": conditions,
        "resources": resources,
        "memory": memory,
        "toolkit": robot,
        "planner": planner,
    }


def test_dependency_inventory_deduplicates_editable_distribution_metadata(monkeypatch):
    class Distribution:
        metadata = {"Name": "harness"}
        version = "0.0.0"

    import reweave.evolution.runtime_manifest as module

    monkeypatch.setattr(
        module.importlib.metadata,
        "distributions",
        lambda: (Distribution(), Distribution()),
    )

    assert dependency_inventory()["installed"] == [("harness", "0.0.0")]


def test_capture_records_config_without_freezing_or_checking_source_identity(tmp_path):
    kwargs = fixture(tmp_path)
    destination = tmp_path / "baseline"
    bundle = capture_baseline(destination, **kwargs)
    manifest = RuntimeManifest.model_validate_json(
        (destination / "runtime.json").read_text()
    )
    assert bundle.runtime == manifest.runtime_configuration()
    assert not (destination / "source").exists()
    assert "sources" not in RuntimeManifest.model_fields
    assert not hasattr(bundle.runtime, "sha256")
    assert manifest.memory_files == ()
    assert manifest.authoring_contract is not None
    assert "model:plan" in manifest.authoring_contract.operations
    assert "inspect_scene" in manifest.authoring_contract.operations
    assert (
        manifest.authoring_contract.native_image_contract
        == kwargs["toolkit"].native_image_contract()
    )
    assert kwargs["toolkit"].calls == 0
    manifest.validate_conditions(
        arguments={**kwargs["arguments"], "seed": 9, "source_root": "changed source"},
        limits=kwargs["limits"],
        planner=kwargs["planner"],
    )
    with pytest.raises(ValueError, match="configuration differs"):
        manifest.validate_conditions(
            arguments={**kwargs["arguments"], "max_turns": 21},
            limits=kwargs["limits"],
            planner=kwargs["planner"],
        )


def test_robodojo_execution_configuration_publishes_source_backed_task_rules():
    from reweave.benchmarks.robodojo.tasks import TASK_RULE_REVIEWS
    from reweave.evolution.runtime_manifest import execution_configuration

    configuration = execution_configuration(
        {"robot_name": "robodojo", "task": "play_tic_tac_toe"},
        {"episode": {"program_calls": 1}},
    )
    rules = configuration["task_rule_supplements"]
    selected = rules["robodojo:play_tic_tac_toe"]
    review = TASK_RULE_REVIEWS["play_tic_tac_toe"]
    assert selected == {
        "task": "robodojo:play_tic_tac_toe",
        "task_rule_supplement": review.supplement,
        "source_urls": list(review.source_urls),
        "review_status": "supplemented",
    }
    assert list(rules) == ["robodojo:play_tic_tac_toe"]


def test_capture_copy_failure_leaves_no_published_half_tree(tmp_path, monkeypatch):
    kwargs = fixture(tmp_path)
    destination = tmp_path / "baseline"
    import reweave.evolution.runtime_manifest as module

    original = module.shutil.copytree

    def fail(*args, **kwargs):
        raise OSError("injected interrupted copy")

    monkeypatch.setattr(module.shutil, "copytree", fail)
    with pytest.raises(OSError, match="interrupted copy"):
        capture_baseline(destination, **kwargs)
    assert not destination.exists()
    monkeypatch.setattr(module.shutil, "copytree", original)
    capture_baseline(destination, **kwargs)
    assert (destination / "bundle.json").is_file()


def test_empty_memory_is_valid_for_actual_frozen_cli(tmp_path):
    from reweave.benchmarks.libero.robot_spec import _add_cli_args, _parse_config
    from reweave.cli.main import _build_argparser

    parser = _build_argparser()
    _add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--robot",
            "libero",
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path),
            "--agent-bundle",
            str(tmp_path / "bundle.json"),
        ]
    )
    config = _parse_config(args)
    assert config.prompt_vars["frozen_bundle"] is True
    assert config.prompt_vars["memory_dir"] == str(tmp_path)


@pytest.mark.parametrize("robot", ["libero", "robodojo"])
def test_no_action_cap_requires_an_environment_that_supports_it(tmp_path, robot):
    from reweave.evolution.runtime_manifest import validate_conditions

    values = fixture(tmp_path)
    conditions = values["conditions"].model_copy(update={"physical_steps": None})
    with pytest.raises(ValueError, match="finite physical"):
        validate_conditions(
            conditions,
            {**values["arguments"], "robot_name": robot},
            values["limits"],
            values["planner"],
        )


def test_uncapped_online_contract_requires_explicit_zero_cli_and_finite_work(tmp_path):
    from reweave.evolution.contracts import EvaluationConditions
    from reweave.evolution.runtime_manifest import validate_conditions

    values = fixture(tmp_path)
    value = values["conditions"].model_dump(mode="json")
    value["wall_time_s"] = None
    conditions = EvaluationConditions.model_validate(value)
    arguments = {**values["arguments"], "planner_timeout_s": 0}
    limits = {"episode": {**values["limits"]["episode"], "wall_time_s": None}}
    validate_conditions(conditions, arguments, limits, values["planner"])
    with pytest.raises(ValueError, match="time budget"):
        validate_conditions(
            conditions,
            {**arguments, "planner_timeout_s": None},
            limits,
            values["planner"],
        )
    with pytest.raises(ValueError, match="finite physical"):
        EvaluationConditions.model_validate({**value, "physical_steps": None})


def test_runtime_configuration_is_plain_metadata_without_identity_gates(tmp_path):
    values = fixture(tmp_path)
    destination = tmp_path / "baseline"
    bundle = capture_baseline(destination, **values)
    manifest = RuntimeManifest.model_validate_json(
        (destination / "runtime.json").read_text()
    )
    revised = manifest.model_copy(update={"dependencies": {"different": True}})
    assert revised.runtime_configuration().dependencies == {"different": True}
    assert bundle.runtime.dependencies == manifest.dependencies
    revised.validate_conditions(
        arguments=values["arguments"],
        limits=values["limits"],
        planner=values["planner"],
    )
