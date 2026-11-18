"""Public policy-family task priors; independent of this run's held-out results."""

import json
from decimal import ROUND_HALF_UP, Decimal
from functools import cache
from pathlib import Path


@cache
def baseline():
    return json.loads(Path(__file__).with_suffix(".json").read_text())


def task_prior(task):
    if not task:
        return None
    task = task.removeprefix("robodojo:").removesuffix("_random")
    source = baseline()
    row = source["tasks"].get(task)
    if row is None:
        return None
    value = Decimal(str(row["success_rate_percent"])).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return {
        "task": task,
        "success_rate_percent": float(value),
        "revision": source["revision"],
        "source": source["source_page"],
        "scope": source["checkpoint_scope"],
        "prompt": f"VLA capability prior: the organizer's published Pi0.5 RoboDojo end-to-end baseline success rate for this task is approximately {value}%. This is a task-level policy-family reference averaged across training seeds, not a per-call success probability for the current scene. Use it to calibrate reliance on VLA. Judge each short VLA call by observed task progress and the remaining physical-step budget; if it makes no useful progress, stop repeating it and replan with the available control primitives.",
    }
