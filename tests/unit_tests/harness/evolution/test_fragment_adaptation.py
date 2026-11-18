"""Distillation suggestions permit explicit adaptation without weakening proofs."""

from reweave.evolution.distillation.adaptation import adaptation_proposals, pack
from reweave.evolution.distillation.fragment_logic import Fragment, compare, fact


def run(q, *references):
    units = (q, *references)
    bank = {u.source: [u] for u in units}
    inventory = {u.id: {"declared_result": pack(u.post)} for u in units}
    return adaptation_proposals(bank, inventory)


def test_partial_result_and_context_conflict_remain_actionable():
    q = Fragment(
        "q",
        (fact("color", "b", value="blue"),),
        (fact("held", "b"), fact("inside", "b", "drawer")),
        source="query",
    )
    r = Fragment(
        "r",
        (fact("color", "a", value="red"), fact("clear", "other")),
        (fact("held", "a"),),
        source="reference",
        assurance="observed",
    )
    assert compare(q, r) == []
    rows, metrics = run(q, r)
    assert len(rows) == 1
    row = rows[0]
    assert row["entity_mapping"] == {"a": "b"}
    assert row["entry_conflicts"] == pack((fact("color", "b", value="red"),))
    assert row["unbound_reference_entry"] == pack((fact("clear", "other"),))
    assert row["uncovered_results"] == pack((fact("inside", "b", "drawer"),))
    assert not row["callable"]
    assert metrics["supported_source_units"] == 1


def test_failed_donor_and_opposite_result_do_not_create_suggestions():
    q = Fragment("q", (), (fact("held", "b"),), source="q")
    failed = Fragment("r", (), (fact("held", "a"),), source="r")
    opposite = Fragment(
        "s", (), (fact("held", "a", value="no"),), source="s", assurance="observed"
    )
    assert run(q, failed, opposite)[0] == []


def test_support_metric_counts_results_once_across_teachers():
    q = Fragment("q", (), (fact("held", "b"),), source="q")
    r = Fragment("r", (), (fact("held", "a"),), source="r", assurance="observed")
    s = Fragment("s", (), (fact("held", "c"),), source="s", assurance="observed")
    rows, metrics = run(q, r, s)
    assert len([row for row in rows if row["query"] == "q"]) == 2
    assert metrics["guidance_coverage"] == 1
    assert metrics["result_support"] == 1
    assert metrics["supported_results"] == 3


def test_already_satisfied_outcome_is_not_a_new_reuse_need():
    q = Fragment("q", (fact("held", "b"),), (fact("held", "b"),), source="q")
    r = Fragment("r", (), (fact("held", "a"),), source="r", assurance="observed")
    rows, metrics = run(q, r)
    assert not any(row["query"] == "q" for row in rows)
    assert metrics["eligible_source_units"] == 1


def test_overall_success_is_an_evaluation_not_a_retrieval_anchor():
    q = Fragment("q", (), (fact("official_success", "task_q"),), source="q")
    r = Fragment(
        "r", (), (fact("official_success", "task_r"),), source="r", assurance="observed"
    )
    rows, metrics = run(q, r)
    assert rows == []
    assert metrics["eligible_source_units"] == 0
    assert metrics["guidance_coverage"] is None
