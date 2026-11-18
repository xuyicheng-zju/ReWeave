"""Classify a settled episode without inventing an unavailable official verdict."""

from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    API_ERROR,
    classify_episode_termination,
)


def settled_episode_termination(
    *, planner_termination_class, environment_success, finalization_error
):
    """An unknown action outranks every later terminal-read or cleanup error."""
    if planner_termination_class == ACTION_OUTCOME_UNKNOWN:
        return ACTION_OUTCOME_UNKNOWN
    if finalization_error is not None:
        return API_ERROR
    return classify_episode_termination(
        planner_termination_class=planner_termination_class,
        official_success=environment_success is True,
    )
