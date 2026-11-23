# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Offline contracts for the LIBERO toolkit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reweave.benchmarks.libero import robot_spec, toolkit
from reweave.dashboard.events import NullDashboardEventSink
from reweave.memory.corpus import ReadOnlyCorpus
from reweave.planner.base import accepted_finish_result
from reweave.robots.robot_spec import RunConfig
from reweave.tools.toolkit import Toolkit, _is_readonly
from reweave.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EVALUATION_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "move_to",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "view_camera_meta",
    "segment",
    "back_project",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


def test_toolkit_factory_keeps_reference_corpus_read_only_in_all_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit, "LiberoToolkit", fake_toolkit)
    memory_dir = tmp_path / "libero-memory"
    config = _run_config(memory_dir)

    evaluation = robot_spec.get_toolkit(
        primitives_kwargs={"env": "evaluation"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    exploration = robot_spec.get_toolkit(
        primitives_kwargs={"env": "exploration"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "state",
    )

    assert evaluation.memory.root == memory_dir.resolve()
    assert exploration.memory.root == memory_dir.resolve()
    evaluation_write = evaluation.memory.get_common_tool_bindings()["write_text_file"][
        1
    ]
    exploration_write = exploration.memory.get_common_tool_bindings()[
        "write_text_file"
    ][1]
    own_draft = memory_dir / "_internal" / "inbox" / config.recipe_tag / "draft.md"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        evaluation_write(str(own_draft), "draft")
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        exploration_write(str(own_draft), "draft")
    assert captured[0]["mode"] == "evaluation"
    assert captured[1]["mode"] == "exploration"
    assert captured[1]["attempts_per_session"] == 2


def test_toolkit_modes_construct_with_fake_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "LiberoPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    evaluation = toolkit.LiberoToolkit(
        primitives_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=ReadOnlyCorpus(tmp_path / "evaluation-memory"),
        mode="evaluation",
        state_output_dir=tmp_path / "evaluation",
    )
    exploration = toolkit.LiberoToolkit(
        primitives_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=ReadOnlyCorpus(
            tmp_path / "exploration-memory",
        ),
        mode="exploration",
        attempts_per_session=3,
        state_output_dir=tmp_path / "exploration",
    )
    runtime_enabled = toolkit.LiberoToolkit(
        primitives_kwargs={
            "env_client": object(),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=ReadOnlyCorpus(tmp_path / "runtime-memory"),
        mode="evaluation",
        state_output_dir=tmp_path / "runtime",
    )

    assert _tool_names(evaluation) == EVALUATION_TOOLS
    assert _tool_names(exploration) == EVALUATION_TOOLS | {"reset"}
    runtime_specs = {spec["name"]: spec for spec in runtime_enabled.get_tools_spec()}
    baseline_specs = {spec["name"]: spec for spec in toolkit.libero_tools.TOOLS_SPEC}
    for name in EVALUATION_TOOLS - COMMON_TOOLS:
        assert runtime_specs[name] == baseline_specs[name]
    assert _readonly_names(evaluation) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "segment",
        "back_project",
    }
    assert _readonly_names(exploration) == _readonly_names(evaluation)
    assert len(dumped) == 3
    assert all(
        instance.reset_calls == 1 for instance in fake_single_arm_primitives.instances
    )
    assert all(
        instance.recording_started for instance in fake_single_arm_primitives.instances
    )
    assert all(
        callable(instance.kwargs["check_cancelled"])
        for instance in fake_single_arm_primitives.instances
    )

    refused = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "first attempt"}
    )
    assert refused.result["error"] == "finish refused"
    assert accepted_finish_result(refused.result) is None

    exploration.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    assert (
        exploration.execute_tool("reset", {"reason": "new approach"}).result["attempt"]
        == 2
    )
    assert (
        exploration.execute_tool("reset", {"reason": "third approach"}).result[
            "attempt"
        ]
        == 3
    )
    allowed = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "budget spent"}
    )
    assert accepted_finish_result(allowed.result) == allowed.result
