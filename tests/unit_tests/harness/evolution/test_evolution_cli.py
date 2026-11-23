# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""The CLI runs one declared matrix-guided task at a time."""

import json

import pytest
from pydantic import ValidationError

from reweave.cli.evolve import _evolution_command, _parser
from reweave.evolution.search.engine import EvolutionWorkspace
from tests.unit_tests.harness.evolution.test_rollout_recipe import (
    instance,
    recipe_fixture,
)


@pytest.mark.parametrize(
    "command",
    [
        "search-init",
        "search-run",
        "search-status",
        "create-h0",
        "record-dev",
        "evolve",
        "record-heldout",
    ],
)
def test_removed_commands_are_not_compatibility_aliases(command):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args([command])
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["--seeds", "--num-seeds"])
def test_removed_multi_seed_options_are_rejected(option):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["init", option, "1"])
    assert error.value.code == 2


def test_run_requires_an_explicit_runtime_directory():
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["run", "--workspace", "/tmp/evolution"])
    assert error.value.code == 2


@pytest.mark.parametrize("device", [None, 0])
def test_run_uses_the_selected_runtime_directory_for_rollout(
    tmp_path, monkeypatch, device
):
    _, _, runner, declared = recipe_fixture(tmp_path)
    assert _evolution_command(_init_arguments(tmp_path, runner, declared)) == 0
    source = tmp_path / "runtime-source"
    source.mkdir()
    observed = {}

    class Evaluator:
        def __init__(self, configuration, *, repository, workspace):
            observed["repository"] = repository
            observed["devices"] = configuration.devices

    class Evolution:
        def __init__(self, workspace, author, evaluator, *, analyzer, recover):
            pass

        async def run(self):
            return {"completed": True, "stop_reason": "test", "versions": {}}

        def accounting(self, state):
            return {}

    monkeypatch.setattr("reweave.evolution.search.rollout.RolloutEvaluator", Evaluator)
    monkeypatch.setattr("reweave.evolution.search.engine.GraphEvolution", Evolution)
    args = _parser().parse_args(
        [
            "run",
            "--workspace",
            str(tmp_path / "evolution"),
            "--runtime-source",
            str(source),
        ]
    )
    if device is not None:
        args.device = device
    assert _evolution_command(args) == 0
    assert observed["repository"] == source
    if device is not None:
        assert observed["devices"] == (device,)
        lease = tmp_path / "evolution/device_lease.json"
        assert json.loads(lease.read_text()) == {"device": device}
        args.device = None
        with pytest.raises(ValueError, match="original explicit --device"):
            _evolution_command(args)
        lease.write_text('{"device": 99}')
        args.device = device
        with pytest.raises(ValueError, match="cannot change"):
            _evolution_command(args)


def _init_arguments(tmp_path, runner, declared):
    runner_path = tmp_path / "runner.json"
    runner_path.write_text(runner.model_dump_json())
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(declared.model_dump_json())
    return _parser().parse_args(
        [
            "init",
            "--baseline",
            str(tmp_path / "baseline/bundle.json"),
            "--protocol",
            str(protocol_path),
            "--runner",
            str(runner_path),
            "--workspace",
            str(tmp_path / "evolution"),
        ]
    )


def test_init_and_nonblocking_status_use_same_immutable_protocol(tmp_path, capsys):
    _, bundle, runner, declared = recipe_fixture(tmp_path)
    args = _init_arguments(tmp_path, runner, declared)
    root = tmp_path / "evolution"
    assert _evolution_command(args) == 0
    capsys.readouterr()
    graph = EvolutionWorkspace(root, bundle, declared)
    with graph.transaction():
        status = _parser().parse_args(["status", "--workspace", str(root)])
        assert _evolution_command(status) == 0
        assert json.loads(capsys.readouterr().out)["protocol"] == declared.record_id

    configuration = json.loads((root / "configuration.json").read_text())
    configuration["runner"]["devices"] = [7]
    (root / "configuration.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="immutable"):
        _evolution_command(args)

    configuration["selection"] = []
    (root / "configuration.json").write_text(json.dumps(configuration))
    status = _parser().parse_args(["status", "--workspace", str(root)])
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _evolution_command(status)


def test_init_preflights_the_single_registered_instance_before_persisting(tmp_path):
    _, _, runner, declared = recipe_fixture(tmp_path)
    invalid = declared.model_copy(update={"dev": (instance(4, suite="invalid_suite"),)})
    args = _init_arguments(tmp_path, runner, invalid)
    with pytest.raises(ValueError):
        _evolution_command(args)
    assert not (tmp_path / "evolution").exists()


def test_protocol_file_preserves_all_development_instances(tmp_path):
    _, _, runner, declared = recipe_fixture(tmp_path)
    path = tmp_path / "protocol.json"
    payload = declared.model_dump(mode="json")
    payload["dev"] = [
        instance(1).model_dump(mode="json"),
        instance(2).model_dump(mode="json"),
    ]
    path.write_text(json.dumps(payload))
    runner_path = tmp_path / "runner.json"
    runner_path.write_text(runner.model_dump_json())
    args = _parser().parse_args(
        [
            "init",
            "--baseline",
            str(tmp_path / "baseline/bundle.json"),
            "--protocol",
            str(path),
            "--runner",
            str(runner_path),
            "--workspace",
            str(tmp_path / "evolution"),
        ]
    )
    _evolution_command(args)
    configuration = json.loads((tmp_path / "evolution/configuration.json").read_text())
    assert configuration["protocol"]["dev"] == payload["dev"]


def test_init_compares_actual_conditions_instead_of_runtime_identifiers(tmp_path):
    _, bundle, runner, declared = recipe_fixture(tmp_path)
    execution = declared.execution.model_copy(update={"wall_time_s": None})
    declared = declared.model_copy(update={"execution": execution})
    manifest_payload = runner.runtime_manifest.model_dump(mode="json")
    manifest_payload["conditions"] = execution.model_dump(mode="json")
    manifest_payload["configuration"]["arguments"]["planner_timeout_s"] = 0
    manifest_payload["configuration"]["limits"]["episode"]["wall_time_s"] = None
    manifest = type(runner.runtime_manifest).model_validate(manifest_payload)
    arguments = list(runner.arguments)
    arguments[arguments.index("--planner-timeout-s") + 1] = "0"
    runner = runner.model_copy(
        update={
            "arguments": tuple(arguments),
            "runtime_manifest": manifest,
            "episode_limits": runner.episode_limits.model_copy(
                update={"wall_time_s": None}
            ),
        }
    )
    assert _evolution_command(_init_arguments(tmp_path, runner, declared)) == 0
    configuration = json.loads((tmp_path / "evolution/configuration.json").read_text())
    assert configuration["runner"]["runtime_manifest"][
        "conditions"
    ] == execution.model_dump(mode="json")
    assert (
        configuration["runner"]["runtime_manifest"]["execution_id"]
        != configuration["protocol"]["execution_id"]
    )
