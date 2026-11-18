# Modified for the ReWeave namespace.
"""Preflight uses actual CLI semantics and never launches a model or service."""

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reweave.cli.main import parse_cli_arguments
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import EvaluationRequest, Instance
from reweave.evolution.runtime_manifest import RuntimeManifest, capture_baseline
from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator
from reweave.runtime.kernel import EpisodeLimits
from tests.unit_tests.harness.evolution.fixtures import protocol
from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture

RECIPE = (
    "--robot",
    "libero",
    "--planner",
    "native",
    "--max-episode-steps",
    "100",
    "--wait-steps",
    "10",
    "--max-turns",
    "20",
    "--planner-timeout-s",
    "30",
)


def instance(seed=1, *, task=0, suite="libero_goal"):
    return Instance(
        record_id=f"condition-{suite}-{task}-{seed}",
        task=f"libero:{suite}:{task}",
        initialization={
            "suite": suite,
            "task": task,
            "seed": seed,
            "reset_state_id": seed,
        },
    )


def recipe_fixture(tmp_path):
    inputs = fixture(tmp_path)
    # Capture every real default and robot-derived field, as the production CLI does.
    _, args, robot_spec = parse_cli_arguments(
        [
            *RECIPE,
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--seed",
            "1",
            "--reset-state-id",
            "1",
            "--cuda-device",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(inputs["memory"]),
            "--output-dir",
            str(tmp_path / "episode"),
            "--capture-bundle",
            str(tmp_path / "baseline"),
            "--runtime-resources",
            str(tmp_path / "resources.json"),
            "--kernel-limits",
            str(tmp_path / "kernel.json"),
        ]
    )
    robot_spec.parse_config(args)
    inputs["arguments"] = vars(args)
    inputs["limits"] = {
        "episode": EpisodeLimits.model_validate(inputs["limits"]["episode"]).model_dump(
            mode="json"
        ),
    }
    bundle = capture_baseline(tmp_path / "baseline", **inputs)
    manifest = RuntimeManifest.model_validate_json(
        (tmp_path / "baseline/runtime.json").read_text()
    )
    runner = RolloutConfiguration(
        devices=(0, 1, 2, 3),
        arguments=RECIPE,
        memory_snapshot=str(inputs["memory"]),
        runtime_manifest=manifest,
        episode_limits=inputs["limits"]["episode"],
    )
    declared = protocol().model_copy(
        update={
            "execution": manifest.conditions,
            "dev": (instance(1),),
        }
    )
    return inputs, bundle, runner, declared


@pytest.fixture
def recipe(tmp_path):
    return recipe_fixture(tmp_path)


def preflight(runner, actual_instance=None):
    return runner.validate_arguments(
        actual_instance or instance(),
        directory=Path(runner.memory_snapshot).parent / "attempt",
        memory=Path(runner.memory_snapshot),
        device=runner.devices[0],
    )


def test_preflight_preserves_real_instances_and_does_not_start_runtime(
    recipe, monkeypatch
):
    inputs, _, runner, declared = recipe
    from reweave.cli import main

    monkeypatch.setattr(
        main, "build_planner", lambda *a, **k: pytest.fail("planner initialized")
    )
    for actual in (*declared.dev, instance(9, task=1)):
        arguments = preflight(runner, actual)
        assert arguments == runner.build_arguments(
            actual,
            directory=inputs["memory"].parent / "attempt",
            memory=inputs["memory"],
            device=0,
        )
        _, parsed, spec = parse_cli_arguments(arguments)
        config = spec.parse_config(parsed)
        assert config.task_desc["seed"] == actual.initialization["seed"]
        assert parsed.reset_state_id == actual.initialization["reset_state_id"]
        assert parsed.task == actual.initialization["task"]
    assert not (inputs["memory"].parent / "attempt").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ("-i",),
        ("-ii",),
        ("--interactive",),
        ("--help",),
        ("-h",),
        ("--removed-option", "x"),
        ("--max-turns", "bad"),
        ("--cuda-dev", "9"),
        ("--planner=codex",),
        ("--libero-type", "invalid"),
        ("--online-modification",),
        ("--task-harness", "x"),
    ],
)
def test_actual_parser_rejects_invalid_fixed_recipe(recipe, extra):
    runner = recipe[2]
    runner = runner.model_copy(update={"arguments": (*runner.arguments, *extra)})
    with pytest.raises(ValueError, match="invalid fixed rollout CLI"):
        preflight(runner)


@pytest.mark.parametrize(
    "extra",
    [
        ("--cuda-device=0",),
        ("--seed", "1"),
        ("--task=0",),
        ("--reset-state-id", "1"),
        ("--output-dir=x",),
        ("--memory-dir=x",),
        ("--memory-profile=local",),
        ("--env-endpoint", "x"),
        ("--vla-endpoint=x",),
        ("--sam3-endpoint=x",),
        ("--agent-bundle=x",),
        ("--kernel-limits=x",),
        ("--runtime-manifest=x",),
    ],
)
def test_all_runner_owned_destinations_are_rejected(recipe, extra):
    runner = recipe[2]
    with pytest.raises(ValueError, match="runner-owned"):
        preflight(runner.model_copy(update={"arguments": (*runner.arguments, *extra)}))


@pytest.mark.parametrize(
    "extra",
    [
        ("--max-turns=20",),
        ("--robot=libero",),
        ("--planner=native",),
        ("--model=one", "--model=two"),
        ("--verbose", "--verbose"),
    ],
)
def test_duplicates_are_rejected_by_action_destination(recipe, extra):
    runner = recipe[2]
    with pytest.raises(ValueError, match="duplicate fixed rollout argument"):
        preflight(runner.model_copy(update={"arguments": (*runner.arguments, *extra)}))


def test_equals_syntax_uses_the_same_parser_contract(recipe):
    runner = recipe[2]
    arguments = ("--robot=libero", "--planner=native", *runner.arguments[4:])
    assert preflight(runner.model_copy(update={"arguments": arguments}))[:2] == list(
        arguments[:2]
    )


def test_physical_preflight_uses_selected_source_not_host_parser(recipe, monkeypatch):
    runner = recipe[2]
    expected = preflight(runner)

    def wrong_host_parser(*args, **kwargs):
        pytest.fail("the offline Host parser must not interpret the physical runtime")

    monkeypatch.setattr("reweave.cli.main.parse_cli_arguments", wrong_host_parser)
    actual = runner.validate_arguments(
        instance(),
        directory=Path(runner.memory_snapshot).parent / "attempt",
        memory=Path(runner.memory_snapshot),
        device=0,
        repository=Path(__file__).resolve().parents[4],
    )
    assert actual == expected


@pytest.mark.parametrize(
    "extra",
    [("--codex-auth-mode=api",), ("--model=another",), ("--libero-type=pro",)],
)
def test_full_execution_configuration_drift_fails_before_startup(recipe, extra):
    runner = recipe[2]
    with pytest.raises(ValueError, match="configuration differs"):
        preflight(runner.model_copy(update={"arguments": (*runner.arguments, *extra)}))


def test_robot_configuration_is_derived_before_comparing(recipe):
    runner = recipe[2]
    arguments = tuple("10" if value == "100" else value for value in runner.arguments)
    with pytest.raises(ValueError, match="policy action horizon must be positive"):
        preflight(runner.model_copy(update={"arguments": arguments}))
    with pytest.raises(ValueError, match="configuration differs"):
        preflight(runner, instance(5, suite="libero_spatial"))


def test_preflight_compares_numeric_budget_values(
    recipe,
):
    runner = recipe[2]
    configuration = json.loads(json.dumps(runner.runtime_manifest.configuration))
    configuration["limits"]["episode"]["wall_time_s"] = 30
    assert configuration == runner.runtime_manifest.configuration
    manifest = runner.runtime_manifest.model_copy(
        update={"configuration": configuration}
    )
    assert preflight(
        runner.model_copy(update={"runtime_manifest": manifest})
    ) == preflight(runner)


@pytest.mark.parametrize("extra", [("-i",), ("--no-images",), ("--max-turns=20",)])
def test_rejected_actual_worker_argv_never_reaches_launch_accounting(
    recipe, tmp_path, monkeypatch, extra
):
    inputs, bundle, runner, declared = recipe
    runner = runner.model_copy(update={"arguments": (*runner.arguments, *extra)})
    evaluator = RolloutEvaluator(
        runner, repository=(tmp_path / "repo"), workspace=tmp_path / "search"
    )
    request = EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=bundle.record_id,
        instance=declared.dev[0],
        split="dev",
        manifest_id="a" * 64,
        execution=declared.execution_id,
        conditions=declared.execution,
    )
    directory = tmp_path / "attempt"
    directory.mkdir()
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", lambda *a, **k: pytest.fail("spawned worker")
    )
    with pytest.raises(ValueError):
        asyncio.run(
            evaluator._run(
                request,
                bundle,
                directory,
                device=0,
                slot=0,
                lease_fd=0,
                before_launch=lambda: pytest.fail("registered launch"),
            )
        )
    assert not (directory / "launch.json").exists()
    assert not (directory / "arguments.json").exists()


@pytest.mark.parametrize("capture", [True, False])
def test_real_main_normalizes_string_memory_path_for_frozen_entrypoints(
    recipe, monkeypatch, tmp_path, capture
):
    from reweave.cli import main
    from reweave.evolution import runtime_manifest
    from reweave.robots import PromptBundle

    inputs, _, runner, _ = recipe
    spec = main.get_robot_spec("libero")
    monkeypatch.setattr(
        main,
        "get_robot_spec",
        lambda name: replace(
            spec,
            init_runtime=lambda *a: ([], {}),
            prompts=PromptBundle(system=lambda _: "system", user=lambda _: "task"),
        ),
    )
    monkeypatch.setattr(main, "build_planner", lambda *a, **k: object())
    monkeypatch.setattr(
        main,
        "get_toolkit",
        lambda *a, **k: SimpleNamespace(memory=None, close=lambda: None),
    )
    directory = tmp_path / "baseline"
    arguments = runner.build_arguments(
        instance(), directory=directory, memory=inputs["memory"], device=0
    )
    (directory / "kernel.json").write_text(
        json.dumps(
            {
                "conditions": runner.runtime_manifest.conditions.model_dump(
                    mode="json"
                ),
                "episode": {**inputs["limits"]["episode"], "wall_time_s": 30},
            }
        )
    )
    observed = []

    class VerifiedMemory(BaseException):
        pass

    if capture:
        index = arguments.index("--agent-bundle")
        arguments[index : index + 2] = ["--capture-bundle", str(tmp_path / "capture")]
        index = arguments.index("--runtime-manifest")
        arguments[index : index + 2] = [
            "--runtime-resources",
            str(tmp_path / "resources.json"),
        ]
        (tmp_path / "resources.json").write_text(inputs["resources"].model_dump_json())

        def capture_baseline(destination, **kwargs):
            observed.append(kwargs["memory"])
            assert type(kwargs["limits"]["episode"]["wall_time_s"]) is float

        monkeypatch.setattr(runtime_manifest, "capture_baseline", capture_baseline)
    else:

        def validate_conditions(self, *, arguments, limits, planner):
            observed.append(Path(arguments["memory_dir"]).resolve())
            raise VerifiedMemory

        monkeypatch.setattr(RuntimeManifest, "validate_conditions", validate_conditions)
    monkeypatch.setattr(sys, "argv", ["harness", *arguments])
    if capture:
        assert main.main() == 0
    else:
        with pytest.raises(VerifiedMemory):
            main.main()
    assert observed == [inputs["memory"].resolve()]
