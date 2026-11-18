"""Which tool results invalidate the rest of a queued decision batch."""

import pytest

from reweave.planner.decisions import tool_failed


@pytest.mark.parametrize(
    "result",
    [
        {"error": "bad arguments"},
        {"capability_result": {"ik_error": "native planning failed", "steps_used": 0}},
        {"capability_result": {"ik_error": "path requires 140 acknowledged actions"}},
        {"active_execution_outcome": "failed"},
    ],
)
def test_rejected_plan_or_explicit_error_invalidates_queued_calls(result):
    assert tool_failed(result)


@pytest.mark.parametrize(
    "result",
    [
        {"capability_result": {"ik_error": None, "reached": False, "steps_used": 40}},
        {"active_execution_outcome": "completed"},
        {"value": 3},
    ],
)
def test_residuals_and_unreached_targets_do_not_invalidate_queued_calls(result):
    assert not tool_failed(result)
