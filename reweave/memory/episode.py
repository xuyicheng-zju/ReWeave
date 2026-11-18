"""Host-generated task-specific facts; never ask an agent to reconstruct actions."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.utils.persistence import atomic_json, atomic_text


def record_episode(transcript: Path) -> Path:
    """Index one settled episode and export its successful observed action sequence.

    Recipe rows retain exact control-step data and source event indices. They are
    evidence, not a robot replay entrypoint or a new evaluator verdict.
    """
    transcript = Path(transcript)
    root = transcript.parent
    result = json.loads(transcript.read_text())
    trace = root / "execution_trace.jsonl"
    events = (
        [json.loads(line) for line in trace.read_text().splitlines()]
        if trace.exists()
        else []
    )
    controls = [row for row in events if row["event_type"] == "control_step"]
    unknown = result["termination_class"] == "action_outcome_unknown" or any(
        row["event_type"] == "action_outcome_unknown" for row in events
    )
    success = result["official_success"] is True
    if success and unknown:
        raise ValueError("unknown physical outcomes cannot produce a successful recipe")
    if [row["control_step_index"] for row in controls] != list(range(len(controls))):
        raise ValueError("episode control steps are not an ordered complete prefix")
    sources = {
        name: {
            "path": name,
            "bytes": (root / name).stat().st_size,
        }
        for name in (
            transcript.name,
            "execution_trace.jsonl",
            "states.json",
            "task.json",
            "evaluation.json",
            "result.json",
            "episode.mp4",
            "harness_context/bundle.json",
            "native/adaptation/receipt.json",
        )
        if (root / name).is_file()
    }
    audit = {
        "owner": "Host",
        "source_root": "..",
        "official_success": result["official_success"],
        "termination_class": result["termination_class"],
        "bundle_id": result.get("bundle_id"),
        "control_steps": len(controls),
        "physical_outcome_unknown": unknown,
        "sources": sources,
        "recipe": "recipe.jsonl" if success and trace.exists() else None,
        "meaning": "Facts from this exact episode; no model-authored action reconstruction or replay authorization.",
    }
    directory = root / "task-specific"
    directory.mkdir(exist_ok=True)
    path = directory / "audit.json"
    if path.exists() and json.loads(path.read_text()) != audit:
        raise ValueError("refusing to replace an episode's recorded audit")
    recipe_path = directory / "recipe.jsonl"
    if audit["recipe"] is not None:
        recipe = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in controls)
        if recipe_path.exists() and recipe_path.read_text() != recipe:
            raise ValueError("refusing to replace an episode's recorded recipe")
        atomic_text(recipe_path, recipe)
    elif recipe_path.exists():
        raise ValueError("unsuccessful episodes cannot contain a winning recipe")
    atomic_json(path, audit)
    return path
