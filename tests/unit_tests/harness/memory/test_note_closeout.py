# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Bounded Planner-authored notes without replaying or changing task outcomes."""

import asyncio
import json

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.memory.test_episode_notes import NoteRobot


def solve(
    tmp_path,
    model,
    *,
    max_turns=4,
    wall_time_s=10.0,
    note_closeout_s=3.0,
    robot_type=NoteRobot,
    bundle=None,
):
    robot = robot_type(tmp_path / "episode", tmp_path / "published")
    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline() if bundle is None else bundle,
        episode_limits=EpisodeLimits(
            program_calls=30,
            gateway_calls=30,
            wall_time_s=wall_time_s,
            note_closeout_s=note_closeout_s,
        ),
    )
    result = planner.solve(
        system_prompt="Use public evidence only.",
        user_message="Synthetic task",
        toolkit=robot,
        max_turns=max_turns,
    )
    status = json.loads((robot.episode_notes.root / "closeout.json").read_text())
    ledger = [
        json.loads(line)
        for line in (robot.state.output_dir / "native/native_turns.jsonl")
        .read_text()
        .splitlines()
    ]
    return result, robot, status, ledger


def test_environment_failure_gets_one_text_only_note_without_more_actions(tmp_path):
    seen = []

    def model(messages, info):
        seen.append(info)
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])
        assert "action phase has ended" in str(messages)
        assert "moved" in str(messages)
        return ModelResponse(
            parts=[TextPart("Observed move; task remains unverified.")]
        )

    result, robot, status, ledger = solve(tmp_path, model)
    assert result.error is None
    assert len(seen) == 2
    assert len(robot.capture_calls) == 1
    assert not robot.solved()
    assert status["status"] == "written"
    assert status["verification"] == "unverified"
    assert status["written"] is True
    assert [item["controller_kind"] for item in ledger] == ["planner", "note_closeout"]
    assert result.stats["visible_turns"] == 2
    revisions = list((robot.episode_notes.root / "revisions").glob("*.json"))
    assert len(revisions) == 1
    assert json.loads(revisions[0].read_text())["text"].startswith("Observed move")
    assert baseline().context.skills == ()


def test_active_failure_note_releases_request_reserve_and_skips_final_request(tmp_path):
    count = 0

    def model(messages, info):
        nonlocal count
        count += 1
        assert info.function_tools
        if count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "write_text_file",
                        {
                            "path": str(
                                tmp_path
                                / "episode/planner_memory/drafts/task-family/working-notes.md"
                            ),
                            "content": "Attempt 1 failed; mechanism unknown, next test unverified.",
                        },
                        "note-1",
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])

    result, robot, status, ledger = solve(tmp_path, model, max_turns=2)
    assert result.error is None
    assert count == 2
    assert status == {"status": "already_written"}
    assert len(ledger) == 2
    assert len(robot.capture_calls) == 1


def test_last_request_is_reserved_for_notes_and_never_exceeds_shared_cap(tmp_path):
    def model(messages, info):
        if info.function_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_text_file",
                        {
                            "path": str(
                                tmp_path
                                / "episode/planner_memory/drafts/task-family/missing.md"
                            )
                        },
                        "read-1",
                    ),
                    ToolCallPart("move", {}, "discarded-move"),
                ]
            )
        return ModelResponse(parts=[TextPart("No successful approach observed.")])

    result, robot, status, ledger = solve(tmp_path, model, max_turns=2)
    assert result.termination_class == "agent_execution_failure"
    assert status["status"] == "written"
    assert len(ledger) == result.stats["visible_turns"] == 2
    assert not robot.capture_calls


def test_unexecuted_pending_call_is_labelled_not_forged_or_replayed(tmp_path):
    from reweave.evolution.bundle import LoopPolicy, PythonFile, PythonProgram

    bundle = baseline().model_copy(
        update={
            "loop": LoopPolicy(
                program=PythonProgram(
                    entrypoint="main:run",
                    files=(
                        PythonFile(
                            path="main.py",
                            source=(
                                "async def run(runtime, context):\n"
                                "    await runtime.plan()\n"
                                "    raise RuntimeError('end before executing the suggested move')\n"
                            ),
                        ),
                    ),
                )
            )
        }
    )

    def model(messages, info):
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart("move", {}, "pending-move")])
        assert "unreturned_at_episode_end" in str(messages)
        return ModelResponse(
            parts=[TextPart("A move was suggested but never executed.")]
        )

    result, robot, status, ledger = solve(tmp_path, model, bundle=bundle)
    assert result.termination_class == "agent_execution_failure"
    assert status["status"] == "written"
    assert len(ledger) == 2
    assert not robot.capture_calls


def test_official_success_survives_note_generation(tmp_path):
    class SuccessRobot(NoteRobot):
        def solved(self):
            return self.terminal

    def model(messages, info):
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])
        return ModelResponse(
            parts=[TextPart("Observed the completed action; scope remains this scene.")]
        )

    result, robot, status, ledger = solve(tmp_path, model, robot_type=SuccessRobot)
    assert robot.solved()
    assert result.error is None
    assert result.termination_class is None
    assert status["status"] == "written"
    assert len(ledger) == 2


def test_unknown_physical_outcome_does_not_trigger_note_or_replay(tmp_path):
    class UnknownRobot(NoteRobot):
        def move(self):
            from reweave.runtime.capability import PhysicalOutcomeUnknown

            raise PhysicalOutcomeUnknown("physical acknowledgment missing")

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])

    result, robot, status, ledger = solve(tmp_path, model, robot_type=UnknownRobot)
    assert result.termination_class == "action_outcome_unknown"
    assert status["status"] == "not_written"
    assert len(ledger) == 1
    assert not robot.episode_notes.has_writes


def test_nonretryable_provider_failure_never_starts_note_request(tmp_path):
    count = 0

    def model(messages, info):
        nonlocal count
        count += 1
        from reweave.runtime.model_budget import ModelServiceFailure

        raise ModelServiceFailure("provider refused input", retryable=False)

    result, robot, status, ledger = solve(tmp_path, model)
    assert result.termination_class == "api_error"
    assert count == len(ledger) == 1
    assert status["status"] == "not_written"
    assert status["reason"] == "cancelled_or_unsafe_to_request_again"
    assert not robot.episode_notes.has_writes


@pytest.mark.parametrize(
    "failure, closeout_reason, ledger_error",
    [
        ("failed_turn", "ModelServiceBudgetExceeded", "NativeTurnFailed"),
        ("inflight_stall", "TimeoutError", "CancelledError"),
    ],
)
def test_note_failure_is_bounded_and_does_not_replace_environment_result(
    tmp_path, failure, closeout_reason, ledger_error
):
    count = 0

    async def model(messages, info):
        nonlocal count
        count += 1
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])
        if failure == "inflight_stall":
            # The closeout's existing timeout must cancel an in-flight request.
            # No arbitrary sleep duration or provider callback supplies the end.
            await asyncio.Event().wait()
            pytest.fail("An uncompleted closeout request escaped its timeout")
        from reweave.planner.providers.base import NativeTurnFailed

        raise NativeTurnFailed("failed", {"message": "summary connection lost"})

    result, robot, status, ledger = solve(tmp_path, model, note_closeout_s=1.0)
    assert result.error is None
    assert result.termination_class is None
    assert robot.episode_ended() and not robot.solved()
    assert count == len(ledger) == result.stats["visible_turns"] == 2
    assert status["status"] == "not_written"
    # A failed final allowed request exhausts the explicit closeout cost cap;
    # a still-running request instead ends at the existing closeout deadline.
    assert status["reason"] == closeout_reason
    assert [item["controller_kind"] for item in ledger] == ["planner", "note_closeout"]
    assert [item["status"] for item in ledger] == ["completed", "failed"]
    assert ledger[-1]["error"] == ledger_error
    assert ledger[0]["request_id"] != ledger[1]["request_id"]
    assert not (robot.state.output_dir / "native/recovery/blocked.json").exists()
    assert not robot.episode_notes.has_writes
    assert len(robot.capture_calls) == 1


def test_action_timeout_preserves_time_for_one_note_without_extra_decision(tmp_path):
    async def model(messages, info):
        if info.function_tools:
            await asyncio.sleep(20)
            raise AssertionError("action request should have been cancelled")
        return ModelResponse(
            parts=[TextPart("Action request interrupted; no motion occurred.")]
        )

    result, robot, status, ledger = solve(
        tmp_path, model, wall_time_s=4.0, note_closeout_s=3.0
    )
    assert result.termination_class == "planner_timeout"
    assert status["status"] == "written"
    assert len(ledger) == result.stats["visible_turns"] == 2
    assert ledger[0]["status"] == "failed"
    assert not robot.capture_calls


def test_closeout_cannot_execute_model_suggested_motion(tmp_path):
    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])

    result, robot, status, ledger = solve(tmp_path, model)
    assert result.error is None
    assert len(robot.capture_calls) == 1
    assert len(ledger) == 2
    assert status["status"] == "not_written"
    assert status["reason"] == "ValueError"


@pytest.mark.parametrize("reserve", [0.0, -1.0, 10.0, 11.0])
def test_note_reserve_must_leave_action_time(reserve):
    with pytest.raises(ValidationError):
        EpisodeLimits(
            program_calls=10,
            gateway_calls=10,
            wall_time_s=10.0,
            note_closeout_s=reserve,
        )


def test_uncapped_episode_waits_for_model_and_keeps_one_bounded_closeout(tmp_path):
    async def model(messages, info):
        if info.function_tools:
            await asyncio.sleep(1.5)
            return ModelResponse(parts=[ToolCallPart("move", {}, "move-1")])
        return ModelResponse(
            parts=[TextPart("Observed one move; no official success.")]
        )

    result, robot, status, ledger = solve(tmp_path, model, wall_time_s=None)
    assert result.error is None
    assert len(robot.capture_calls) == 1
    assert status["status"] == "written"
    assert len(ledger) == 2
    assert ledger[0]["wall_time_s"] >= 1.5
