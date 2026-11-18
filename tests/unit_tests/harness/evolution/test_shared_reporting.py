"""Roundwise development reporting preserves observed losses and provenance."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.engine import SharedEvolution
from reweave.evolution.distillation.reporting import development_rounds, summarize
from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator, plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_real_initial_score_and_rejected_candidate_are_reported(tmp_path):
    result = asyncio.run(SharedEvolution(tmp_path, Author(), Evaluator()).run(plan()))
    initial = json.loads((tmp_path / "initial/score.json").read_text())
    rows = development_rounds(tmp_path)
    assert initial["macro_success"] == 0
    assert [r["round"] for r in rows] == [0, 1, 2]
    assert rows[0]["task_dev_sr"] == {"source": 0}
    assert rows[0]["task_loss"] == 1
    assert rows[2]["total_loss"] > rows[1]["total_loss"]
    assert rows[1]["selected"] and result["round"] == 1
    assert rows[2]["retention_feasible"] is False
    assert all(r["loss_components_consistent"] for r in rows)
    assert all(r["retention_loss_kind"] == "task_success_drop" for r in rows)


def test_conditional_components_are_weighted_without_clipping(tmp_path):
    write(tmp_path / "plan.json", {"retention_weight": 2, "size_weight_per_kb": 0.001})
    score = {
        "macro_success": 0.5,
        "scores": {"a": 1, "b": 0},
        "conditional_loss": 0.1,
        "mean_drop": 0.8,
        "delta_excess_kb": -10,
        "delta_size": {"size_kb": 20},
        "teacher_delta_mean_kb": 30,
        "loss": 0.69,
        "eligible": False,
    }
    write(tmp_path / "rounds/001/score.json", score)
    write(tmp_path / "rounds/002/score.json", {**score, "loss": 0.9})
    rows = development_rounds(tmp_path)
    assert [r["round"] for r in rows] == [1, 2]  # No invented R0.
    assert rows[0]["weighted_retention_loss"] == pytest.approx(0.2)
    assert rows[0]["complexity_penalty"] == pytest.approx(-0.01)
    assert rows[0]["retention_loss_kind"] == "conditional"
    assert rows[0]["loss_components_consistent"] is True
    assert rows[1]["total_loss"] == 0.9  # Preserve, flag, do not repair evidence.
    assert rows[1]["loss_components_consistent"] is False
    assert rows[0]["selected"] is None


def test_missing_weights_do_not_fabricate_components(tmp_path):
    assert development_rounds(tmp_path) == []
    write(tmp_path / "rounds/003/score.json", {"loss": 0.4, "macro_success": 0.6})
    row = development_rounds(tmp_path)[0]
    assert row["round"] == 3 and row["total_loss"] == 0.4
    assert row["retention_loss_kind"] is None
    assert row["weighted_retention_loss"] is None
    assert row["complexity_penalty"] is None
    assert row["loss_components_consistent"] is None


def test_experiment_summary_exports_every_recorded_round(tmp_path):
    write(
        tmp_path / "manifest.json",
        {
            "cohort": "source8",
            "selection_policy": "fixed",
            "groups": [
                {
                    "name": "source8",
                    "kind": "scaling",
                    "source_harnesses": 8,
                    "methods": ["aligned"],
                }
            ],
        },
    )
    directory = tmp_path / "source8/aligned"
    write(directory / "initial/score.json", {"loss": 1.0})
    write(directory / "rounds/001/score.json", {"loss": 0.2})
    write(
        directory / "test_report.json",
        {
            "panels": {
                "ood": {
                    f"task-{i}": {
                        "successes": 10 if i < 3 else 0,
                        "episodes": 10,
                        "settled": True,
                    }
                    for i in range(8)
                }
            }
        },
    )
    report = summarize(tmp_path)["A"]["source8"]["aligned"]
    rows = report["development_rounds"]
    assert [r["total_loss"] for r in rows] == [1, 0.2]
    assert len(report["ood_tasks"]) == 8
    assert report["ood_tasks"]["task-0"]["macro_sr"] == 1
    assert report["ood_tasks"]["task-7"]["macro_sr"] == 0
    assert report["ood"]["macro_sr"] == 3 / 8
