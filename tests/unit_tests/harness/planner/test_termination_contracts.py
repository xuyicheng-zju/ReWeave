# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Episode termination vocabulary and PlannerResult pairing."""

from __future__ import annotations

import pytest

from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    AGENT_EXECUTION_FAILURE,
    API_ERROR,
    PHYSICAL_FAILURE,
    PLANNER_TIMEOUT,
    SUCCESS,
    PlannerResult,
    classify_episode_termination,
    episode_is_scored,
    task_evidence_is_complete,
)


def test_planner_timeout_and_program_failure_are_scored_but_incomplete():
    timeout = classify_episode_termination(
        planner_termination_class=PLANNER_TIMEOUT,
        official_success=False,
    )
    author = classify_episode_termination(
        planner_termination_class=AGENT_EXECUTION_FAILURE,
        official_success=False,
    )
    assert timeout == PLANNER_TIMEOUT
    assert author == AGENT_EXECUTION_FAILURE
    assert timeout != author
    assert episode_is_scored(timeout)
    assert episode_is_scored(author)
    assert task_evidence_is_complete(timeout) is False
    assert task_evidence_is_complete(author) is False


def test_completed_episode_uses_official_success():
    assert (
        classify_episode_termination(
            planner_termination_class=None,
            official_success=True,
        )
        == SUCCESS
    )
    assert (
        classify_episode_termination(
            planner_termination_class=None,
            official_success=False,
        )
        == PHYSICAL_FAILURE
    )
    assert task_evidence_is_complete(SUCCESS)
    assert task_evidence_is_complete(PHYSICAL_FAILURE)
    assert episode_is_scored(PHYSICAL_FAILURE)
    assert episode_is_scored(API_ERROR) is False


@pytest.mark.parametrize(
    "planner_termination", [PLANNER_TIMEOUT, AGENT_EXECUTION_FAILURE, API_ERROR]
)
def test_confirmed_official_success_survives_planner_abort(planner_termination):
    assert (
        classify_episode_termination(
            planner_termination_class=planner_termination,
            official_success=True,
        )
        == SUCCESS
    )


@pytest.mark.parametrize("official_success", [False, True])
def test_unknown_action_survives_independent_success_query(official_success):
    assert (
        classify_episode_termination(
            planner_termination_class=ACTION_OUTCOME_UNKNOWN,
            official_success=official_success,
        )
        == ACTION_OUTCOME_UNKNOWN
    )


@pytest.mark.parametrize("termination", [SUCCESS, PHYSICAL_FAILURE, "unexpected"])
@pytest.mark.parametrize("official_success", [False, True])
def test_official_success_does_not_hide_invalid_planner_class(
    termination, official_success
):
    with pytest.raises(ValueError, match="unknown planner termination class"):
        classify_episode_termination(
            planner_termination_class=termination,
            official_success=official_success,
        )


def test_planner_result_requires_paired_error_and_class():
    with pytest.raises(ValueError, match="must be set together"):
        PlannerResult(error="API planner timed out after 1s")
    with pytest.raises(ValueError, match="must be set together"):
        PlannerResult(termination_class=PLANNER_TIMEOUT)
    with pytest.raises(ValueError, match="cannot assign episode outcome"):
        PlannerResult(error="failed", termination_class=PHYSICAL_FAILURE)
    result = PlannerResult(
        error="API planner timed out after 1s",
        termination_class=PLANNER_TIMEOUT,
        messages=[{"role": "assistant", "content": "working"}],
        stats={"turns_used": 1},
    )
    assert result.termination_class == PLANNER_TIMEOUT


@pytest.mark.parametrize(
    "content",
    [
        {"ok": True},
        {"error": "rejected", "_finish": True},
        {"isError": True, "content": [{"type": "text", "text": '{"_finish":true}'}]},
        "accepted",
        [{"type": "image", "source": {"data": "not a sentinel"}}],
    ],
)
def test_transport_success_is_not_host_finish_acceptance(content):
    from reweave.planner.base import accepted_finish_result

    assert accepted_finish_result(content) is None
