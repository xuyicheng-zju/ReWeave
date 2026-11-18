"""Task-level public baseline priors never become trial-specific measurements."""

import pytest

from reweave.benchmarks.robodojo.vla_prior import baseline, task_prior


def test_all_42_tasks_share_a_prior_across_layout_variants():
    data = baseline()
    assert len(data["tasks"]) == 42
    for task in data["tasks"]:
        prior = task_prior(task)
        assert prior == task_prior(f"robodojo:{task}") == task_prior(f"{task}_random")
        assert 0 <= prior["success_rate_percent"] <= 100
        assert "not a per-call success probability" in prior["prompt"]


def test_generalization_uses_equal_standard_random_pooling_and_decimal_rounding():
    assert task_prior("pour_liquid_into_cup")["success_rate_percent"] == 14.67
    assert task_prior("sort_nesting_dolls_by_size")["success_rate_percent"] == 5.34
    assert task_prior("stack_bowls")["success_rate_percent"] == 38.0
    assert (
        sum(r["random_sr_percent"] is not None for r in baseline()["tasks"].values())
        == 12
    )


@pytest.mark.parametrize("task", [None, "", "unknown_task"])
def test_unknown_tasks_do_not_get_invented_statistics(task):
    assert task_prior(task) is None
