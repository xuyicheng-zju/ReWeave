# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Admit complete semantic graphs against physical and executable evidence."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import ValidationError, validate

from reweave.evolution.authoring.references import AuthorReferences, history_originals
from reweave.evolution.contracts import (
    EvidenceFile,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
)
from reweave.evolution.evidence.trajectory import (
    consecutive_actions,
    read_originals,
)
from reweave.evolution.graph.programs import program_assets
from reweave.evolution.graph.projection import EpisodeProjection

PENDING_GRAPH_REFINEMENT = "Pending refinement against this round's new evidence."


def needs_matching(request):
    """Encode conditions for retrieval or the final graph consumed by distillation."""
    from reweave.evolution.evidence.archive_inputs import archived_author_policy

    return (
        archived_author_policy(request.policy).graph_retrieval is not None
        or request.purpose == "final_selection"
        or request.protocol.max_rounds == 0
    )


def analysis_result_schema(request):
    """Only ask the model to emit fields with a consumer in this phase."""
    schema = GraphAnalysisResult.model_json_schema()
    schema["$defs"]["SemanticEdge"]["properties"]["witnesses"]["minItems"] = 1
    if request.needs_forecast:
        schema["properties"]["forecast"] = {"$ref": "#/$defs/ProgressForecast"}
        schema["required"].append("forecast")
    else:
        schema["properties"]["forecast"] = {"type": "null", "default": None}
        schema["$defs"].pop("ProgressForecast", None)
        schema["$defs"].pop("ProgressOutcome", None)
    if not needs_matching(request):
        schema["$defs"]["SemanticGraph"]["properties"]["matching"] = {
            "type": "null",
            "default": None,
            "description": "Condition encoding is deferred to final analysis unless retrieval uses it.",
        }
    # Do not send disconnected definitions for fields omitted in this phase.
    definitions = schema.pop("$defs")
    pending = [schema]
    retained = {}
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            ref = value.get("$ref", "")
            if ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in retained:
                    retained[name] = definitions[name]
                    pending.append(definitions[name])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    schema["$defs"] = retained
    return schema


def validate_analysis_output(request, output):
    """Same output/coverage gate for in-session checks and final Host publication."""
    files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    allowed = {"result.json"}
    if history_originals(request)["navigation"].get("demonstrations"):
        allowed.add("demonstration_review.json")
    if "result.json" not in files or files - allowed:
        raise ValueError("graph output must contain exactly result.json")
    if "demonstration_review.json" in files and not isinstance(
        json.loads((output / "demonstration_review.json").read_text()), dict
    ):
        raise ValueError("demonstration review must be a JSON object")
    raw = (output / "result.json").read_bytes()
    references = AuthorReferences.load(output.parent)
    value = json.loads(raw)
    try:
        validate(value, references.schema(analysis_result_schema(request)))
    except ValidationError as error:
        raise ValueError(
            f"invalid author graph at {error.json_path}: {error.message}"
        ) from error
    result = GraphAnalysisResult.model_validate(references.decode(value))
    validate_refinement(request, result)
    if result.changes.strip() == PENDING_GRAPH_REFINEMENT:
        raise ValueError("complete the pending graph analysis before publication")
    return result


def validate_refinement(request, result):
    if request.needs_forecast and result.forecast is None:
        raise ValueError("dynamic width requires a candidate forecast")
    if (
        request.protocol.topology_mode == "fixed_initial"
        and request.previous_analysis is not None
    ):
        request.previous_analysis.verify()
        previous = GraphAnalysisReceipt.model_validate_json(
            Path(request.previous_analysis.path).read_text()
        ).analysis.graph
        before = topology_definition(previous)
        after = topology_definition(result.graph)
        if previous.matching is None:
            # Condition encoding is first added at final analysis, as in Full.
            after["matching"] = None
        if before != after:
            raise ValueError(
                "initial topology is frozen: node/edge identities and physical definitions cannot change"
            )
    request.graph.verify()
    evidence = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
    if evidence.policy != request.protocol.graph:
        raise ValueError("evidence graph differs from the frozen scaffold policy")
    graph = result.graph
    episodes = {e.receipt_id: e for e in evidence.episodes}
    if set(graph.episodes) != set(episodes):
        raise ValueError(
            "refinement must project every historical episode exactly once for refresh, and only all new receipts for first-observation"
        )
    allowed = {
        (f.path, f.artifact_id, f.bytes) for r in request.evidence for f in r.evidence
    }
    allowed.update(
        (e.receipt.path, e.receipt.artifact_id, e.receipt.bytes)
        for e in evidence.episodes
    )
    request.history.verify()
    allowed.update(
        (path, value["artifact_id"], value["bytes"])
        for path, value in history_originals(request)["files"].items()
    )
    projections = {}
    completed = graph.completed_nodes()
    used_nodes = set()
    for identity, episode in episodes.items():
        spans = graph.episodes[identity]
        projections[identity] = EpisodeProjection(spans, completed)
        if (spans[-1].last_frame + 1 if spans else 0) != len(episode.frames):
            raise ValueError("semantic projection must cover every original frame")
        for span in spans:
            node_id = span.node_id
            if node_id is None:
                continue
            used_nodes.add(node_id)
            node = graph.nodes[node_id]
            if not node.references:
                raise ValueError("observed semantic states require original evidence")
            if node.role == "goal" and not (
                episode.official_success
                and span.first_frame == span.last_frame == len(episode.frames) - 1
            ):
                raise ValueError(
                    "goal state requires the final official success witness"
                )
        if (
            episode.official_success
            and spans
            and (
                spans[-1].node_id is None
                or graph.nodes[spans[-1].node_id].role != "goal"
            )
        ):
            raise ValueError("official successful terminal frame must be a goal state")
    for node_id, node in graph.nodes.items():
        if node_id not in used_nodes:
            if node.role == "branch":
                raise ValueError("optional branch nodes require observed members")
            if not node.uncertainty.strip():
                raise ValueError(
                    "unobserved backbone nodes require explicit uncertainty"
                )
    refs = graph.original_references()
    for ref in refs:
        if (ref.path, ref.artifact_id, ref.bytes) not in allowed:
            raise ValueError("semantic evidence is outside the frozen originals")
        EvidenceFile(**ref.model_dump(exclude={"pointer"})).verify()
    selected = {ref for ref in refs if ref.pointer}
    selected.update(program.bundle for program in graph.programs.values())
    try:
        decoded = read_originals(selected)
    except (IndexError, KeyError, TypeError, UnicodeError) as error:
        raise ValueError(
            "semantic reference does not resolve in its original"
        ) from error
    programs = {
        key: program_assets(program, decoded[program.bundle])
        for key, program in graph.programs.items()
    }
    covered = {identity: set() for identity in episodes}

    def interval(witness):
        episode = episodes.get(witness.episode_id)
        if episode is None or witness.last_frame >= len(episode.frames):
            raise ValueError("semantic execution interval is outside its episode")
        indices = set(range(witness.first_frame + 1, witness.last_frame + 1))
        if covered[witness.episode_id] & indices:
            raise ValueError(
                "semantic execution witnesses must not double-count actions"
            )
        covered[witness.episode_id].update(indices)
        return episode, indices

    for edge in graph.edges.values():
        if not edge.witnesses:
            raise ValueError(
                "only inherited first-observation edges may lack new witnesses"
            )
        for witness in edge.witnesses:
            episode, indices = interval(witness)
            projection = projections[witness.episode_id]
            start, end = (
                projection.at(witness.first_frame),
                projection.at(witness.last_frame),
            )
            if (
                witness.program_id is not None
                and programs[witness.program_id]["bundle_id"] != episode.bundle_id
            ):
                raise ValueError(
                    "transition program belongs to a different executed Bundle"
                )
            if start.status == "unknown":
                raise ValueError(
                    "semantic edges cannot start from unknown observations"
                )
            if start.node_id != edge.source:
                raise ValueError(
                    "semantic edge source differs from its witnessed start state"
                )
            for i in indices:
                if not consecutive_actions(
                    episode.frames[i - 1], episode.frames[i]
                ) or any(
                    "unacknowledged physical interval"
                    in (episode.frames[j].error or "")
                    for j in (i - 1, i)
                ):
                    raise ValueError(
                        "semantic edges cannot bridge reset, missing or unknown physical actions"
                    )
            if witness.outcome == "achieved" and end.node_id != edge.target:
                raise ValueError("achieved semantic edge must witness its target state")
            if witness.outcome == "not_achieved" and end.node_id == edge.target:
                raise ValueError(
                    "failed semantic edge already reached its declared target"
                )
            if end.status == "unknown" and witness.outcome != "unknown":
                raise ValueError(
                    "unknown semantic post-state requires an unknown outcome"
                )
    for gap in graph.unclassified:
        interval(gap)
    for identity, episode in episodes.items():
        expected = set(range(1, len(episode.frames)))
        if covered[identity] != expected:
            raise ValueError(
                "every original frame interval needs a mechanism or explicit unclassified reason"
            )
    return result


def topology_definition(graph):
    """Freeze physical semantics while allowing new mappings and witnesses."""
    return {
        "matching": graph.matching.model_dump(exclude={"uncertainties"})
        if graph.matching is not None
        else None,
        "nodes": {
            key: node.model_dump(exclude={"references", "uncertainty"})
            for key, node in graph.nodes.items()
        },
        "edges": {
            key: edge.model_dump(exclude={"references", "witnesses", "explanation"})
            for key, edge in graph.edges.items()
        },
    }
