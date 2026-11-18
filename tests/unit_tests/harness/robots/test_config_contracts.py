# Copyright 2026 The RPent Authors.
# Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reweave.robots import get_robot_spec


def _parser(robot_name: str, *, dashboard: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--memory-profile", choices=["local", "none"], default=None)
    parser.add_argument("--memory-dir", default=None)
    get_robot_spec(robot_name).add_cli_args(parser, use_dashboard=dashboard)
    return parser


@pytest.mark.parametrize(
    ("robot_name", "required_args", "identity_fields"),
    [
        ("libero", ["--suite", "libero_object_task", "--task", "2"], ("suite", "task")),
    ],
)
def test_robot_arguments_are_required_on_cli_but_deferred_for_dashboard(
    robot_name: str,
    required_args: list[str],
    identity_fields: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        _parser(robot_name).parse_args([])
    cli_args = _parser(robot_name).parse_args(required_args)
    assert all(getattr(cli_args, field) is not None for field in identity_fields)

    dashboard_args = _parser(robot_name, dashboard=True).parse_args([])
    assert all(getattr(dashboard_args, field) is None for field in identity_fields)


@pytest.mark.parametrize(
    ("robot_name", "message"),
    [
        ("libero", "--suite is required"),
    ],
)
def test_dashboard_identity_must_be_filled_before_config_parsing(
    robot_name: str,
    message: str,
) -> None:
    args = _parser(robot_name, dashboard=True).parse_args([])

    with pytest.raises(ValueError, match=message):
        get_robot_spec(robot_name).parse_config(args)


def test_libero_default_evaluation_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_object_task",
            "--task",
            "2",
            "--seed",
            "7",
            "--output-dir",
            str(output_dir),
        ]
    )

    config = get_robot_spec("libero").parse_config(args)

    assert config.recipe_tag == "object_task_t2_s7"
    assert config.output_dir == output_dir
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["reference_tag"] == "object_task_t2_s0"
    assert config.task_desc == {
        "suite": "libero_object_task",
        "task": 2,
        "seed": 7,
        "physical_step_budget": 10000,
        "vla_execution_horizon": 5,
    }


def test_libero_exploration_uses_local_memory_and_session_metadata(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_spatial",
            "--task",
            "1",
            "--explore",
            "--explore-sessions",
            "4",
            "--memory-dir",
            str(memory_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    config = get_robot_spec("libero").parse_config(args)

    assert args.memory_profile == "local"
    assert config.prompt_vars["mode"] == "explore"
    assert config.prompt_vars["session_number"] == 1
    assert config.prompt_vars["session_max"] == 4
    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())
    assert "memory_inbox" not in config.prompt_vars


def test_libero_local_evaluation_allows_absent_or_empty_reference_files(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory_dir),
        ]
    )

    assert (
        get_robot_spec("libero").parse_config(args).prompt_vars["memory_profile"]
        == "local"
    )

    memory_dir.mkdir()
    assert (
        get_robot_spec("libero").parse_config(args).prompt_vars["memory_profile"]
        == "local"
    )

    task_family = memory_dir / "task-family"
    task_family.mkdir()
    (task_family / "goal_conditions.md").write_text("Unverified conditions")
    config = get_robot_spec("libero").parse_config(args)
    assert config.prompt_vars["memory_profile"] == "local"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--explore", "--memory-profile", "none"], "run-local attempt archives"),
        (["--explore", "--explore-sessions", "0"], "greater than 0"),
    ],
)
def test_libero_rejects_invalid_mode_and_memory_combinations(
    extra_args: list[str],
    message: str,
) -> None:
    args = _parser("libero").parse_args(
        ["--suite", "libero_goal", "--task", "0", *extra_args]
    )
    with pytest.raises(ValueError, match=message):
        get_robot_spec("libero").parse_config(args)


def test_none_profile_reaches_reference_corpus_factory(tmp_path, monkeypatch):
    import importlib

    robot, toolkit_class = "libero", "LiberoToolkit"
    arguments = ["--suite", "libero_goal", "--task", "0"]

    root = tmp_path / "published"
    leaf = root / "global/lesson.md"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("published text")
    args = _parser(robot).parse_args(
        [*arguments, "--memory-profile", "none", "--memory-dir", str(root)]
    )
    spec = get_robot_spec(robot)
    config = spec.parse_config(args)
    assert config.prompt_vars["memory_profile"] == "none"
    toolkit_module = importlib.import_module(f"reweave.benchmarks.{robot}.toolkit")
    monkeypatch.setattr(
        toolkit_module, toolkit_class, lambda **kwargs: kwargs["memory"]
    )
    module = importlib.import_module(f"reweave.benchmarks.{robot}.robot_spec")
    corpus = module.get_toolkit(
        primitives_kwargs={}, dashboard_events=None, config=config
    )
    with pytest.raises(PermissionError, match="none profile"):
        corpus.get_common_tool_bindings()["read_text_file"][1](str(leaf))
