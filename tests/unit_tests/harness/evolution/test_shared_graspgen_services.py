"""Per-device routes preserve one actual GraspGen-X geometry contract."""

import json
from pathlib import Path

import pytest

from reweave.cli.main import parse_cli_arguments
from reweave.evolution.runtime_manifest import (
    execution_configuration,
    validate_configuration,
)
from reweave.evolution.search.rollout import (
    RolloutConfiguration,
    SharedInferenceServices,
)
from tests.unit_tests.harness.evolution.test_rollout_recipe import recipe_fixture
from tests.unit_tests.robots.components.test_graspgen_client import _config


def configs(tmp_path):
    paths = []
    for device in (0, 1):
        path = tmp_path / f"graspgen-{device}.json"
        path.write_text(_config(f"tcp://127.0.0.1:{16655 + device}").model_dump_json())
        paths.append(str(path))
    return paths


def test_actual_graspgen_config_compares_content_across_routes_and_detects_edits(
    tmp_path,
):
    paths = configs(tmp_path)
    before = execution_configuration({"graspgen_config": paths[0]}, {})
    other_route = execution_configuration({"graspgen_config": paths[1]}, {})
    validate_configuration(before, other_route)
    assert "endpoint" not in before["graspgen"]
    assert "graspgen_config" not in before["arguments"]
    payload = json.loads(Path(paths[1]).read_text())
    payload["grippers"]["right"]["grasp_frame_from_tcp"][0][3] = 0.01
    Path(paths[1]).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration differs"):
        validate_configuration(
            before, execution_configuration({"graspgen_config": paths[1]}, {})
        )


def test_worker_uses_its_own_graspgen_route_and_rejects_fixed_override(tmp_path):
    _, _, original, protocol = recipe_fixture(tmp_path)
    paths = configs(tmp_path)
    services = tuple(
        SharedInferenceServices(
            device=device,
            vla_endpoint=f"http://127.0.0.1:{18800 + device}",
            sam3_endpoint=f"http://127.0.0.1:{18500 + device}",
            graspgen_config=paths[device],
        )
        for device in (0, 1)
    )
    runner = RolloutConfiguration.model_validate(
        {
            **original.model_dump(),
            "devices": (0, 1),
            "shared_inference_services": services,
        }
    )
    instance = protocol.dev[0]
    argv = runner.build_arguments(
        instance,
        directory=tmp_path / "worker0",
        memory=Path(runner.memory_snapshot),
        device=0,
    )
    _, actual, robot_spec = parse_cli_arguments(argv)
    robot_spec.parse_config(actual)
    manifest = runner.runtime_manifest.model_copy(
        update={
            "configuration": execution_configuration(
                vars(actual),
                {"episode": runner.episode_limits.model_dump(mode="json")},
            )
        }
    )
    runner = runner.model_copy(update={"runtime_manifest": manifest})
    for device in runner.devices:
        argv = runner.validate_arguments(
            instance,
            directory=tmp_path / f"worker{device}",
            memory=Path(runner.memory_snapshot),
            device=device,
        )
        assert argv[argv.index("--graspgen-config") + 1] == paths[device]
    duplicate = runner.model_copy(
        update={"arguments": (*runner.arguments, "--graspgen-config", paths[0])}
    )
    with pytest.raises(ValueError, match="runner-owned"):
        duplicate.validate_arguments(
            instance,
            directory=tmp_path / "duplicate",
            memory=Path(runner.memory_snapshot),
            device=0,
        )


def test_partial_graspgen_pool_and_relative_config_are_rejected(tmp_path):
    _, _, runner, _ = recipe_fixture(tmp_path)
    with pytest.raises(ValueError, match="absolute Host path"):
        SharedInferenceServices(
            device=0, vla_endpoint="http://127.0.0.1:18800", graspgen_config="x.json"
        )
    with pytest.raises(ValueError, match="every device"):
        RolloutConfiguration.model_validate(
            {
                **runner.model_dump(),
                "devices": (0, 1),
                "shared_inference_services": (
                    SharedInferenceServices(
                        device=0,
                        vla_endpoint="http://127.0.0.1:18800",
                        graspgen_config=str(tmp_path / "x.json"),
                    ),
                    SharedInferenceServices(
                        device=1, vla_endpoint="http://127.0.0.1:18801"
                    ),
                ),
            }
        )
