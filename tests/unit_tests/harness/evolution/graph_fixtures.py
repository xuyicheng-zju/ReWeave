# Modified for the ReWeave namespace.
"""Explicit synthetic physical observations and graph judgments for orchestration tests."""

import json
from pathlib import Path

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import (
    EvidenceFile,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    StateSpan,
    TransitionWitness,
    UnclassifiedExecution,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.robots.physical_evidence import feature
from tests.unit_tests.harness.evolution.fixtures import artifact, usage


def physical_trace(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "execution_trace.jsonl"
    frames = [
        {
            "event_type": "initial_state",
            "event_index": 0,
            "physical_features": {
                "entity.object.distance_to_eef_m": feature(0.2, "m", "synthetic sensor")
            },
            "error": None,
        },
        {
            "event_type": "control_step",
            "event_index": 1,
            "control_step_index": 0,
            "execution_ref": "execution:fixture",
            "physical_call_ref": "physical:fixture",
            "tool_name": "move",
            "action": [0.1],
            "physical_features": {
                "entity.object.distance_to_eef_m": feature(
                    0.05, "m", "synthetic sensor"
                )
            },
        },
    ]
    path.write_text("".join(json.dumps(f) + "\n" for f in frames))
    return EvidenceFile(**artifact_record(path))


def semantic_fixture(raw):
    """Synthetic state semantics for testing Host lifecycle, not a physical claim."""
    from reweave.evolution.evidence.trajectory import consecutive_actions

    nodes = {
        "approach": SemanticNode(
            meaning="Synthetic required approach",
            role="milestone",
            predecessors=(),
            references=(),
            uncertainty="Synthetic unobserved backbone",
        ),
        "contact": SemanticNode(
            meaning="Synthetic required contact",
            role="milestone",
            predecessors=("approach",),
            references=(),
            uncertainty="Synthetic unobserved backbone",
        ),
        "complete": SemanticNode(
            meaning="Official evaluator goal",
            role="goal",
            predecessors=("contact",),
            references=(),
            uncertainty="Synthetic unobserved goal",
        ),
    }
    edges, episodes, gaps = {}, {}, []
    for episode in raw.episodes:
        if not episode.frames:
            episodes[episode.receipt_id] = ()
            continue
        terminal = (
            "complete"
            if episode.official_success
            else "frontier_" + episode.condition_id[:12]
        )
        path = [None] * (len(episode.frames) - 1) + [terminal]
        spans = []
        for i, node_id in enumerate(path):
            if node_id is not None:
                nodes[node_id] = SemanticNode(
                    meaning="Explicit synthetic acquired task condition " + node_id,
                    role="goal" if node_id == "complete" else "branch",
                    predecessors=("contact",),
                    references=episode.frames[i].references,
                    uncertainty="Scripted fixture; not empirical semantic evidence",
                )
            spans.append(
                StateSpan(
                    first_frame=i,
                    last_frame=i,
                    status="node" if node_id is not None else "background",
                    node_id=node_id,
                    uncertainty="Synthetic fixture",
                )
            )
            if i == 0:
                continue
            if (
                node_id is None
                or not consecutive_actions(episode.frames[i - 1], episode.frames[i])
                or "unacknowledged physical interval" in (episode.frames[i].error or "")
            ):
                gaps.append(
                    UnclassifiedExecution(
                        episode_id=episode.receipt_id,
                        first_frame=i - 1,
                        last_frame=i,
                        reason="Background interval without an acquired condition or noncontiguous source",
                    )
                )
                continue
            key = str(path[i - 1]) + "_to_" + node_id
            witness = TransitionWitness(
                episode_id=episode.receipt_id,
                first_frame=i - 1,
                last_frame=i,
                outcome="achieved",
                program_id=None,
                bindings={"target": "synthetic object"},
                uncertainty="Synthetic fixture has no isolated program attribution",
            )
            if key in edges:
                edges[key] = edges[key].model_copy(
                    update={"witnesses": (*edges[key].witnesses, witness)}
                )
            else:
                edges[key] = SemanticEdge(
                    source=path[i - 1],
                    target=node_id,
                    roles={"target": "object affected by the transition"},
                    entry_conditions="Supported synthetic prerequisite",
                    exit_conditions="Supported synthetic useful condition",
                    explanation="Fixture local transition, not physical task success",
                    references=episode.frames[i].references,
                    witnesses=(witness,),
                )
        episodes[episode.receipt_id] = tuple(spans)
    return SemanticGraph(
        nodes=nodes,
        edges=edges,
        programs={},
        episodes=episodes,
        unclassified=tuple(gaps),
    )


class FixtureGraphAnalysis:
    """Scripted semantic graph for controller tests; scheduling belongs to Host."""

    policy = AuthorPolicy()

    async def __call__(self, request, directory, *, before_launch):
        before_launch()
        raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
        result = GraphAnalysisResult(
            forecast={
                "outcomes": [{"progress": 100.0, "probability": 1.0}],
                "confidence": 0.8,
                "rationale": "Synthetic prediction; no physical efficacy claim.",
            },
            graph=semantic_fixture(raw),
            changes="Reproject all fixture episodes into one shared semantic graph",
        )
        return GraphAnalysisReceipt(
            record_id=request.record_id + "/fixture-receipt",
            request_id=request.record_id,
            analysis=result,
            session_id="fixture-global-graph-session",
            usage=usage().model_copy(update={"physical_steps": 0}),
            artifacts=(
                artifact(directory, "analysis.json", result.model_dump(mode="json")),
            ),
        )

    def verify_result(self, request, directory):
        receipt = GraphAnalysisReceipt.model_validate_json(
            (directory / "result.json").read_text()
        )
        if receipt.request_id != request.record_id:
            raise ValueError("fixture graph request changed")
        for evidence_file in receipt.artifacts:
            evidence_file.verify()
        return receipt

    async def recover(self, request, directory, *, before_launch):
        return await self(request, directory, before_launch=before_launch)
