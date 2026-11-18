"""Full audit originals stay out of ordinary native feedback, across tool envelopes."""

import copy
import json

import pytest

from reweave.benchmarks.robodojo.planner_observation import feedback_text
from reweave.runtime.capability import ModelInputRejected
from reweave.runtime.model_input import (
    project_diagnostic_feedback,
    validate_native_feedback,
)


@pytest.mark.parametrize(
    "operation",
    [
        "view_env_state",
        "back_project",
        "segment",
        "generate_grasps",
        "move_to",
        "move_pose",
        "move_joints",
        "set_gripper",
        "pi0_pick",
        "read_image",
        "read_text_file",
        "list_dir",
        "write_text_file",
        "finish",
    ],
)
@pytest.mark.parametrize(
    "envelope", ["tool_return", "public_feedback", "workflow_feedback"]
)
def test_all_tool_failure_envelopes_keep_error_without_full_stack(operation, envelope):
    error = {
        "error": "known inference failure",
        "traceback": "private stack\n" * 10000,
        "outcome_known": True,
        "reached": False,
        "position_error_m": 0.03,
    }
    if envelope == "tool_return":
        raw = {
            "part_kind": "tool-return",
            "tool_name": operation,
            "content": json.dumps(error),
        }
    elif envelope == "public_feedback":
        raw = {
            "history_index": 12,
            "feedback": {"operation": operation, "result": error},
        }
    else:
        raw = {"executed": {"name": operation}, "result": json.dumps(error)}
    original = copy.deepcopy(raw)
    for project in (project_diagnostic_feedback, feedback_text):
        selected = project(raw)
        text = json.dumps(selected)
        assert "known inference failure" in text
        assert "position_error_m" in text and "0.03" in text
        assert "private stack" not in text and len(text) < 1000
        validate_native_feedback([selected])
    assert raw == original and len(error["traceback"]) > 100000


@pytest.mark.parametrize(
    "encoding", ["object", "json", "nested_json", "jsonl", "prefixed"]
)
@pytest.mark.parametrize(
    "event",
    [
        {
            "event_type": "control_step",
            "control_step_index": 1,
            "execution_ref": "e",
            "action": {"joints": [0] * 14},
            "terminated": False,
            "truncated": False,
        },
        {
            "event_type": "control_telemetry",
            "control_step_index": 1,
            "control_event_index": 1,
            "execution_ref": "e",
            "public_evidence": {},
            "critic_private_evidence": {},
        },
        {
            "event_type": "vla_forward",
            "forward_id": "f",
            "execution_ref": "e",
            "prediction_horizon_steps": 50,
            "execution_horizon_steps": 50,
            "inference_elapsed_s": 0.5,
        },
    ],
)
def test_final_c_rejects_complete_execution_records_without_changing_selection(
    event, encoding
):
    value = copy.deepcopy(event)
    if encoding == "json":
        value = json.dumps(value)
    elif encoding == "nested_json":
        value = json.dumps({"feedback": json.dumps(value)})
    elif encoding == "jsonl":
        value = json.dumps(value) + "\n" + json.dumps(value)
    elif encoding == "prefixed":
        value = "Trace excerpt:\n" + json.dumps(value)
    original = copy.deepcopy(value)
    with pytest.raises(ModelInputRejected, match="complete execution audit record"):
        validate_native_feedback([value])
    assert value == original


def test_final_instructions_and_messages_share_the_same_audit_boundary():
    from reweave.benchmarks.robodojo.planner_observation import PlannerObservation
    from reweave.tools.toolkit import Toolkit

    context = {
        "instructions": 'Failure details: {"error":"failed", "traceback":"full stack"}',
        "messages": [],
    }
    with pytest.raises(ModelInputRejected, match="exception traceback"):
        Toolkit.validate_model_context(None, context)
    pool = {"support_rejections": [{"candidate_id": i} for i in range(4)]}
    for text in (
        json.dumps(pool),
        "Details:\n" + json.dumps(pool),
        json.dumps(pool) + "\n" + json.dumps(pool),
    ):
        with pytest.raises(ModelInputRejected, match="expanded GraspGen"):
            PlannerObservation().validate({"instructions": text, "messages": []})


def test_explicit_selected_file_content_is_exact_and_semantic_field_names_are_allowed():
    content = json.dumps(
        {
            "error": "example",
            "traceback": "documented example",
            "active_execution_ref": "chosen",
            "image_inputs": ["chosen"],
        }
    )
    page = {
        "path": "working-notes.md",
        "size": len(content),
        "content": content,
        "offset": 0,
        "next_offset": None,
        "has_more": False,
    }
    for project in (project_diagnostic_feedback, feedback_text):
        assert project(page) == page
        assert json.loads(project(json.dumps(page))) == page
    validate_native_feedback(
        [
            page,
            {
                "trace": "selected reference",
                "event_type": "control_step",
                "action": "task reasoning",
                "candidates": list(range(40)),
            },
        ]
    )


def test_generic_current_observation_projection_also_removes_automatic_exception_stack():
    from reweave.planner.native_context import prepare_observation

    context = {
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": json.dumps(
                            {
                                "error": "planner failed",
                                "traceback": "RAW_STACK",
                            }
                        ),
                    }
                ],
            }
        ],
        "observation": {"pose": [1, 2, 3]},
        "physical_budget": {},
        "remaining_wall_time_s": None,
        "objective": "continue",
    }
    original = copy.deepcopy(context)
    projected = prepare_observation(context)
    assert "RAW_STACK" not in json.dumps(projected)
    assert "planner failed" in json.dumps(projected)
    validate_native_feedback(projected["messages"])
    assert context == original


def test_fixed_assets_and_catalog_are_not_projected_as_tool_feedback():
    from reweave.planner.native_context import prepare_observation

    fixed = {
        "error": "documentation example",
        "traceback": "example, not a tool failure",
    }
    context = {
        "messages": [],
        "observation": {},
        "physical_budget": {},
        "remaining_wall_time_s": None,
        "objective": None,
        "assets": copy.deepcopy(fixed),
        "tools": [copy.deepcopy(fixed)],
    }
    projected = prepare_observation(context)
    assert projected["assets"] == context["assets"] == fixed
    assert projected["tools"] == context["tools"] == [fixed]


def test_feedback_without_diagnostic_changes_preserves_exact_text_format():
    original = '{\n    "observation": {"moved": true}\n}'
    assert project_diagnostic_feedback({"content": original}) == {"content": original}
