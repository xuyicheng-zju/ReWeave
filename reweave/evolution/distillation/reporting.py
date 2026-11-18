"""Observed shared results only; missing runs remain null, never synthetic zeroes."""

from __future__ import annotations

import json
import math
from pathlib import Path

from reweave.evolution.distillation.stream import work_report


def read_optional(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def success_summary(tasks):
    if not tasks:
        return {"macro_sr": None, "successes": 0, "episodes": 0, "complete": False}
    values = list(tasks.values())
    complete = all(v["settled"] and v["episodes"] > 0 for v in values)
    return {
        "macro_sr": sum(v["successes"] / v["episodes"] for v in values) / len(values)
        if complete
        else None,
        "successes": sum(v["successes"] for v in values),
        "episodes": sum(v["episodes"] for v in values),
        "complete": complete,
    }


def development_rounds(directory: Path):
    """Observed candidate scores in round order, including rejected candidates.

    R0 is the measured initialization, not a synthetic zero. Missing scores stay
    missing. Loss traces are not replaced by a running minimum or final score.
    """
    plan = read_optional(directory / "plan.json") or {}
    frozen = read_optional(directory / "frozen.json")
    paths = [(0, directory / "initial" / "score.json")]
    paths.extend(
        sorted(
            (int(p.parent.name), p)
            for p in directory.glob("rounds/*/score.json")
            if p.parent.name.isdecimal() and int(p.parent.name) > 0
        )
    )
    rounds = []
    for index, path in paths:
        score = read_optional(path)
        if score is None:
            continue
        task_loss = 1 - score["macro_success"] if "macro_success" in score else None
        conditional = "conditional_loss" in score
        retention = score.get("conditional_loss" if conditional else "mean_drop")
        weighted_retention = (
            plan["retention_weight"] * retention
            if "retention_weight" in plan and retention is not None
            else None
        )
        complexity = (
            plan["size_weight_per_kb"] * score["delta_excess_kb"]
            if "size_weight_per_kb" in plan and "delta_excess_kb" in score
            else None
        )
        components = (task_loss, weighted_retention, complexity)
        consistent = (
            math.isclose(sum(components), score["loss"], abs_tol=1e-9)
            if all(v is not None for v in components) and "loss" in score
            else None
        )
        rounds.append(
            {
                "round": index,
                "selected": index == frozen["round"]
                if frozen and "round" in frozen
                else None,
                "task_dev_sr": score.get("scores"),
                "dev_sr": score.get("macro_success"),
                "task_loss": task_loss,
                "retention_loss_kind": "conditional"
                if conditional
                else "task_success_drop"
                if "mean_drop" in score
                else None,
                "weighted_retention_loss": weighted_retention,
                "complexity_penalty": complexity,
                "total_loss": score.get("loss"),
                "loss_components_consistent": consistent,
                "delta_size_kb": score.get("delta_size", {}).get("size_kb"),
                "teacher_delta_mean_kb": score.get("teacher_delta_mean_kb"),
                "retention_feasible": score.get("eligible"),
                "score": score,
            }
        )
    return rounds


def summarize(root: Path):
    manifest = read_optional(root / "manifest.json")
    if manifest is None:
        raise ValueError("experiment manifest required")
    result = {
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "A": {},
        "B": [],
        "work": work_report(root),
        "metric_note": "Rates are fractions. Retention and conditional loss are development metrics; source/unseen SR uses frozen tests. Historical teacher acquisition cost is separate.",
    }
    for group in manifest["groups"]:
        if group["name"] == "continual":
            continue
        values = {}
        for method in group["methods"]:
            directory = root / group["name"] / method
            frozen = read_optional(directory / "frozen.json")
            tests = read_optional(directory / "test_report.json")
            values[method] = {
                "frozen": frozen is not None,
                "source": success_summary(tests.get("source", {})) if tests else None,
                "unseen": success_summary(tests.get("unseen", {})) if tests else None,
                "development": frozen["score"] if frozen else None,
                "development_rounds": development_rounds(directory),
                "work": work_report(directory) if directory.exists() else None,
            }
        values["baselines"] = read_optional(
            root / group["name"] / "baselines/baseline_report.json"
        )
        if group.get("kind") == "scaling":
            for method in group["methods"]:
                directory = root / group["name"] / method
                tests = read_optional(directory / "test_report.json")
                frozen = read_optional(directory / "frozen.json")
                panels = tests.get("panels", {}) if tests else {}
                ood = panels.get("ood", {})
                families = tests.get("families", {}) if tests else {}
                values[method].update(
                    source_harnesses=group["source_harnesses"],
                    fixed_seed=success_summary(panels.get("fixed_seed", {}))
                    if tests
                    else None,
                    ood=success_summary(ood) if tests else None,
                    ood_tasks={
                        name: success_summary({name: row}) for name, row in ood.items()
                    }
                    if tests
                    else None,
                    ood_families={
                        family: success_summary(
                            {
                                name: row
                                for name, row in ood.items()
                                if families.get(name) == family
                            }
                        )
                        for family in ("spatial", "object", "goal", "long")
                    }
                    if tests
                    else None,
                    round=frozen["round"] if frozen else None,
                    termination=frozen.get("termination") if frozen else None,
                    rounds_executed=frozen.get("rounds_executed") if frozen else None,
                    loss_threshold=frozen.get("loss_threshold") if frozen else None,
                    loss_threshold_reached=frozen.get("loss_threshold_reached")
                    if frozen
                    else None,
                )
        result["A"][group["name"]] = values
    stream = root / "continual/aligned"
    checkpoints = read_optional(stream / "checkpoints.json") or []
    reports = {
        r["stage"]: r["report"]
        for r in (read_optional(stream / "stage_test_report.json") or [])
    }
    previous_tasks = set()
    first_test_scores = {}
    cumulative = dict.fromkeys(
        (
            "development_rollouts",
            "development_steps",
            "development_work_s",
            "offline_work_s",
        ),
        0,
    )
    cumulative_complete = True
    for checkpoint in checkpoints:
        all_tasks = set(checkpoint["tasks"])
        new_tasks = all_tasks - previous_tasks
        report = reports.get(checkpoint["stage"], {})
        after = report.get("source", {})
        entry = report.get("stage_entry_records", [])
        before = {}
        for record in entry:
            name = record["request"]["instance"]["task"]
            item = before.setdefault(
                name, {"successes": 0, "episodes": 0, "settled": True}
            )
            item["successes"] += record["official_success"]
            item["episodes"] += 1
            item["settled"] &= record["failure_origin"] not in {
                "unknown",
                "infrastructure",
            }
        old_drops = {}
        if after:
            for name in previous_tasks:
                value = after.get(name)
                if (
                    name in first_test_scores
                    and value
                    and value["settled"]
                    and value["episodes"]
                ):
                    old_drops[name] = max(
                        0.0,
                        first_test_scores[name]
                        - value["successes"] / value["episodes"],
                    )
        for name in new_tasks:
            value = after.get(name)
            if value and value["settled"] and value["episodes"]:
                first_test_scores[name] = value["successes"] / value["episodes"]
        updates = (
            [stream / "initial"]
            if not previous_tasks
            else [
                stream / "updates" / f"{n:03d}"
                for n in range(len(previous_tasks) + 1, len(all_tasks) + 1)
            ]
        )
        work = [work_report(path) for path in updates]
        stage_work = {k: sum(w[k] for w in work) for k in cumulative}
        cumulative = {k: cumulative[k] + stage_work[k] for k in cumulative}
        stage_complete = all(
            w["physical_steps_complete"] and w["offline_work_complete"] for w in work
        )
        cumulative_complete &= stage_complete
        result["B"].append(
            {
                "stage": checkpoint["stage"],
                "stage_work": stage_work,
                "cumulative_work": dict(cumulative),
                "stage_work_complete": stage_complete,
                "cumulative_work_complete": cumulative_complete,
                "tasks": len(all_tasks),
                "new_before": success_summary(before),
                "new_after": success_summary(
                    {k: v for k, v in after.items() if k in new_tasks}
                ),
                "old_test_mean_drop": sum(old_drops.values()) / len(old_drops)
                if old_drops and set(old_drops) == previous_tasks
                else None,
                "old_test_worst_drop": max(old_drops.values())
                if old_drops and set(old_drops) == previous_tasks
                else None,
                "old_after": success_summary(
                    {k: v for k, v in after.items() if k in previous_tasks}
                ),
                "task_tests": {
                    name: success_summary({name: row}) for name, row in after.items()
                },
                "development": checkpoint["frozen"]["score"],
                "development_rounds": [
                    {
                        "update": str(path.relative_to(stream)),
                        "rounds": development_rounds(path),
                    }
                    for path in updates
                ],
            }
        )
        previous_tasks = all_tasks
    return result
