# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Read-only archive replay into a new, separately owned derived graph."""

import json
from pathlib import Path

from reweave.evolution.contracts import EvaluationRecord, GraphPolicy, VersionReference
from reweave.evolution.evidence.trajectory import build_graph, episode_from_record
from reweave.evolution.infrastructure.persistence import attempt_directory
from reweave.utils.persistence import atomic_json


def replay_graph(workspace: Path, output: Path, policy_path: Path):
    workspace = workspace.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_relative_to(workspace):
        raise ValueError("replay requires a fresh output outside the original archive")
    policy = GraphPolicy.model_validate_json(policy_path.read_text())
    state = json.loads((workspace / "evolution.json").read_text())
    versions = tuple(
        VersionReference(version_id=k, bundle_id=state["versions"][k]["bundle_id"])
        for k in state["order"]
    )
    episodes = []
    for key, attempt in state["attempts"].items():
        if attempt["kind"] != "evaluation" or attempt["status"] != "completed":
            continue
        path = attempt_directory(workspace, key, attempt) / "result.json"
        record = EvaluationRecord.model_validate_json(path.read_text())
        if record.failure_origin != "infrastructure" or record.episode is not None:
            episodes.append(episode_from_record(record, path, versions))
    graph = build_graph(episodes, policy)
    atomic_json(output, graph.model_dump(mode="json"))
    return graph
