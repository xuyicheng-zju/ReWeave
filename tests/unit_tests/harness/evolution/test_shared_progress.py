"""Fixed retained progress cannot gain credit from branches or invented goals."""

import asyncio

import pytest

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.distillation.engine import SharedEvolution
from reweave.evolution.distillation.progress import TopologyObservation, progress_scores
from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator, plan


def graph():
    return SemanticGraph(
        nodes={
            name: {
                "meaning": name,
                "role": role,
                "predecessors": pred,
                "references": [],
                "uncertainty": "fixture",
            }
            for name, role, pred in (
                ("grounded", "milestone", []),
                ("picked", "milestone", ["grounded"]),
                ("retry", "branch", ["grounded"]),
                ("goal", "goal", ["picked"]),
            )
        },
        edges={},
        programs={},
        episodes={},
        unclassified=(),
    )


def fixed_plan():
    return plan().model_copy(
        update={
            "teacher_retention": False,
            "retention_weight": 0,
            "sequential_updates": True,
            "fixed_topology_scoring": True,
            "source_graphs": {"source": graph()},
        }
    )


def observation(record, node):
    return TopologyObservation(
        record_id=record.record_id,
        node_id=node,
        citations=(record.evidence[0].path,),
        explanation="end observation",
    )


@pytest.mark.parametrize(
    "node,expected",
    [("grounded", 1 / 3), ("picked", 2 / 3), ("retry", 1 / 3), (None, 0)],
)
def test_fixed_backbone_scores_retained_milestones_only(tmp_path, node, expected):
    p = fixed_plan()
    rows = asyncio.run(
        SharedEvolution(tmp_path, None, Evaluator()).evaluate(
            p, p.initial, p.sources, tmp_path / "dev"
        )
    )
    result = progress_scores(p, rows, [observation(rows[0], node)])
    assert result["topology_progress"]["source"] == expected
    assert result["topology_loss"] == 1 - expected


def test_projection_rejects_goal_unrelated_citation_and_duplicate(tmp_path):
    p = fixed_plan()
    rows = asyncio.run(
        SharedEvolution(tmp_path, None, Evaluator()).evaluate(
            p, p.initial, p.sources, tmp_path / "dev"
        )
    )
    for bad in (
        [observation(rows[0], "goal")],
        [observation(rows[0], "picked").model_copy(update={"citations": ("/other",)})],
        [observation(rows[0], None)] * 2,
    ):
        with pytest.raises(ValueError):
            progress_scores(p, rows, bad)


def test_last_candidate_and_previous_parent_survive_a_worse_final_loss(tmp_path):
    p = fixed_plan()
    author, evaluator = Author(), Evaluator()

    class Projector:
        async def score(self, plan, bundle, records, directory):
            return progress_scores(
                plan,
                records,
                [observation(r, "grounded") for r in records if not r.official_success],
            )

    frozen = asyncio.run(
        SharedEvolution(tmp_path, author, evaluator, topology_assessor=Projector()).run(
            p
        )
    )
    assert frozen["bundle"]["record_id"] == "bad"
    assert author.requests[1]["parent"]["record_id"] == "compact"
    assert frozen["score"]["topology_loss"] == pytest.approx(2 / 3)
    assert frozen["score"]["loss"] >= 1  # Compression cannot cancel the failure.
    assert len(evaluator.requests) == 3  # H0 + H1 + H2, no teachers/tests.


def test_global_loss_selects_checkpoint_without_changing_sequential_updates(tmp_path):
    p = fixed_plan().model_copy(update={"checkpoint_selection": "min_loss"})
    author, evaluator = Author(), Evaluator()

    class Projector:
        async def score(self, plan, bundle, records, directory):
            return progress_scores(
                plan,
                records,
                [observation(r, "grounded") for r in records if not r.official_success],
            )

    frozen = asyncio.run(
        SharedEvolution(tmp_path, author, evaluator, topology_assessor=Projector()).run(
            p
        )
    )
    assert frozen["round"] == 1
    assert frozen["bundle"]["record_id"] == "compact"
    assert frozen["last_evaluated_round"] == frozen["rounds_executed"] == 2
    assert author.requests[1]["parent"]["record_id"] == "compact"
    assert author.requests[0]["criteria"]["checkpoint_selection"] == "min_loss"
    assert len(frozen["checkpoints"]) == 3
    assert len(evaluator.requests) == 3


def test_full_loss_rejects_a_large_harness_with_higher_success(tmp_path):
    from reweave.evolution.contracts import Instance
    from tests.unit_tests.harness.evolution.test_shared import bundle

    p = fixed_plan()
    source = p.sources[0].model_copy(
        update={
            "dev": tuple(
                Instance(
                    record_id=f"dev-{i}", task="source", initialization={"seed": i}
                )
                for i in range(3)
            )
        }
    )
    p = p.model_copy(update={"sources": (source,), "checkpoint_selection": "min_loss"})
    small = bundle("small")
    big = bundle("big").model_copy(
        update={
            "context": small.context.model_copy(
                update={"instructions": "X" * 1_000_000}
            )
        }
    )

    class Updates:
        async def __call__(self, request, directory):
            return {
                "bundle": (small if request["round"] == 1 else big).model_dump(
                    mode="json"
                )
            }

    class Projector:
        async def score(self, plan, bundle, records, directory):
            return progress_scores(
                plan,
                records,
                [observation(r, "picked") for r in records if not r.official_success],
            )

    evaluator = Evaluator(
        lambda r, b: (
            b.record_id == "big"
            or (b.record_id == "small" and r.instance.initialization["seed"] < 2)
        )
    )
    frozen = asyncio.run(
        SharedEvolution(
            tmp_path, Updates(), evaluator, topology_assessor=Projector()
        ).run(p)
    )
    assert frozen["round"] == 1
    assert frozen["score"]["macro_success"] == pytest.approx(2 / 3)
    checkpoints = frozen["checkpoints"]
    assert checkpoints[2]["loss"] > checkpoints[1]["loss"]
    assert frozen["bundle"]["record_id"] == "small"


@pytest.mark.parametrize("initial_retains_old", [True, False])
def test_minimum_loss_has_no_old_task_retention_veto(tmp_path, initial_retains_old):
    from tests.unit_tests.harness.evolution.test_shared import task

    p = plan().model_copy(
        update={
            "mode": "continual",
            "sources": (task("source"), task("new-a"), task("new-b")),
            "new_tasks": ("new-a", "new-b"),
            "retention_weight": 0,
            "size_weight_per_kb": 0,
            "checkpoint_selection": "min_loss",
            "sequential_updates": True,
        }
    )

    def succeeds(request, candidate):
        if request.instance.task == "source":
            return candidate.record_id == "teacher" or (
                initial_retains_old and candidate.record_id == "initial"
            )
        return candidate.record_id == "compact"

    author, evaluator = Author(), Evaluator(succeeds)
    frozen = asyncio.run(SharedEvolution(tmp_path, author, evaluator).run(p))
    assert frozen["round"] == 1
    assert frozen["bundle"]["record_id"] == "compact"
    assert frozen["score"]["loss"] == pytest.approx(1 / 3)
    assert frozen["score"]["per_task_drop"]["source"] == 1
    assert frozen["new_task_acquisition"] == {"new-a": 1, "new-b": 1}
    assert author.requests[1]["parent"]["record_id"] == "compact"
    assert frozen["rounds_executed"] == 2


@pytest.mark.parametrize("large_at", ["initial", "first_update"])
def test_continual_success_does_not_stop_above_total_loss_threshold(tmp_path, large_at):
    from tests.unit_tests.harness.evolution.test_shared import bundle

    large = bundle("large").model_copy(
        update={
            "context": bundle("large").context.model_copy(
                update={"instructions": "X" * 1_000_000}
            )
        }
    )
    p = plan().model_copy(
        update={
            "mode": "continual",
            "new_tasks": ("source",),
            "initial": large if large_at == "initial" else bundle("initial"),
            "size_reference": bundle("initial"),
            "teachers": (),
            "teacher_retention": False,
            "retention_weight": 0,
            "checkpoint_selection": "min_loss",
            "loss_threshold": 0.1,
            "sequential_updates": True,
        }
    )

    class Updates:
        calls = 0

        async def __call__(self, request, directory):
            self.calls += 1
            value = (
                large
                if large_at == "first_update" and self.calls == 1
                else bundle("small")
            )
            return {"bundle": value.model_dump(mode="json")}

    author = Updates()
    evaluator = Evaluator(lambda r, b: b.record_id != "initial")
    frozen = asyncio.run(SharedEvolution(tmp_path, author, evaluator).run(p))
    assert author.calls == (1 if large_at == "initial" else 2)
    assert frozen["loss_threshold"] == 0.1
    assert frozen["score"]["loss"] < 0.1
    assert frozen["termination"] == "loss_threshold_reached"


def test_continual_lower_loss_stops_despite_old_task_drop(tmp_path):
    from tests.unit_tests.harness.evolution.test_shared import task

    p = plan().model_copy(
        update={
            "mode": "continual",
            "sources": (task("source"), task("new-a"), task("new-b")),
            "new_tasks": ("new-a", "new-b"),
            "retention_weight": 0,
            "size_weight_per_kb": 0,
            "checkpoint_selection": "min_loss",
            "sequential_updates": True,
            "loss_threshold": 0.4,
        }
    )
    author = Author()
    evaluator = Evaluator(
        lambda r, b: (
            b.record_id in {"initial", "teacher"}
            if r.instance.task == "source"
            else b.record_id == "compact"
        )
    )
    frozen = asyncio.run(SharedEvolution(tmp_path, author, evaluator).run(p))
    assert len(author.requests) == 1
    assert frozen["round"] == 1 and frozen["score"]["loss"] == pytest.approx(1 / 3)
    assert frozen["score"]["per_task_drop"]["source"] == 1
    assert frozen["loss_threshold_reached"] is True
