# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Derive task progress and reusable transition evidence from one semantic graph."""

from __future__ import annotations

from bisect import bisect_right

from reweave.evolution.evidence.trajectory import read_original
from reweave.evolution.graph.programs import program_assets
from reweave.evolution.search.matrix import evaluation_cells


def remap_episode_receipts(graph, aliases):
    """Relabel a source-learning view while retaining physical witnesses exactly."""
    if not aliases:
        return graph
    return graph.model_copy(
        update={
            "episodes": {
                aliases.get(key, key): value for key, value in graph.episodes.items()
            },
            "edges": {
                key: edge.model_copy(
                    update={
                        "witnesses": tuple(
                            witness.model_copy(
                                update={
                                    "episode_id": aliases.get(
                                        witness.episode_id, witness.episode_id
                                    )
                                }
                            )
                            for witness in edge.witnesses
                        )
                    }
                )
                for key, edge in graph.edges.items()
            },
            "unclassified": tuple(
                span.model_copy(
                    update={"episode_id": aliases.get(span.episode_id, span.episode_id)}
                )
                for span in graph.unclassified
            ),
        }
    )


def analysis_graph_receipts(request, evidence):
    """Select declared historical witnesses without changing their execution identity."""
    if request.retrospective_receipts is None:
        return selected_graph_receipts(
            request.protocol,
            [version.bundle_id for version in request.versions],
            request.evidence,
            evidence,
            execution_ids=frozenset(
                record.request.execution for record in request.evidence
            ),
        )
    records = {record.record_id: record for record in request.evidence}
    episodes = {episode.receipt_id: episode for episode in evidence.episodes}
    selected = {}
    for identity in request.retrospective_receipts:
        record = records[identity]
        key = (record.request.bundle_id, record.request.instance.record_id)
        episode = episodes.get(identity)
        if episode is None:
            if record.failure_origin != "infrastructure":
                raise ValueError("retrospective receipt has no physical episode")
            selected[key] = None
        elif (episode.bundle_id, episode.condition_id) != key:
            raise ValueError("retrospective witness belongs to another task cell")
        else:
            selected[key] = identity
    return selected


def selected_graph_receipts(protocol, bundles, records, evidence, *, execution_ids):
    """Reference the existing selector's physical observations, not official scores.

    Infrastructure evidence without a physical episode stays null. A complete
    selected episode can locate observed progress despite missing capability
    scoring; historical or better-progress attempts never replace the selection.
    """
    episodes = {episode.receipt_id: episode for episode in evidence.episodes}
    selected = {}
    for key, record in evaluation_cells(
        protocol, bundles, records, execution_ids=execution_ids
    ).items():
        episode = episodes.get(record.record_id)
        if episode is None:
            if record.failure_origin != "infrastructure":
                raise ValueError(
                    "selected evaluation receipt has no physical episode: "
                    + record.record_id
                )
            selected[key] = None
        elif (episode.bundle_id, episode.condition_id) != key:
            raise ValueError("selected receipt belongs to a different map cell")
        else:
            selected[key] = record.record_id
    return selected


class EpisodeProjection:
    """Index interval semantics without allocating one state per physical frame."""

    def __init__(self, spans, completed):
        self.spans = spans
        self.starts = tuple(span.first_frame for span in spans)
        self.completed = completed

    def at(self, index):
        span = self.spans[bisect_right(self.starts, index) - 1]
        if not span.first_frame <= index <= span.last_frame:
            raise ValueError("frame is outside the semantic projection")
        return span

    def achievements(self, span):
        if span.status == "unknown":
            return None
        if span.status == "background":
            return frozenset()
        return self.completed[span.node_id]


def execution_provenance(episode, first_frame, last_frame, starts):
    """Reference original invocations; null preserves missing historical arguments.

    invocation_tool identifies the recorded top-level call. control_tools lists
    its measured physical tools; arguments remain solely in the cited trace row.
    """
    tools = {}
    for frame in episode.frames[first_frame + 1 : last_frame + 1]:
        tools.setdefault(frame.execution_ref, {})[frame.tool] = None
    return [
        {
            "execution_ref": identity,
            "invocation": starts[identity]["reference"] if identity in starts else None,
            "invocation_tool": starts[identity]["event"]["tool_name"]
            if identity in starts
            else None,
            "control_tools": list(names),
        }
        for identity, names in tools.items()
    ]


def interval_cost(episode, first_frame, last_frame):
    """Only measured action cost belongs to a segment; Planner/author cost stays global."""
    frames = [
        f
        for f in episode.frames[first_frame + 1 : last_frame + 1]
        if f.control_step is not None
    ]
    times = [f.action_wall_time_s for f in frames]
    return {
        "physical_steps": len(frames),
        "action_wall_time_s": None if None in times else sum(times),
    }


def unresolved_transitions(
    graph, spans, transition_ids, transitions, interfaces, completed
):
    """Last still-unmet attempt per conditional interface, excluding later recovery.

    A local failure followed by acquisition of its target is historical evidence,
    not a current bottleneck. This is an observed graph relation, not a causal
    diagnosis. Repeated attempts and program variants do not create interfaces.
    """
    last_established = {}
    for span in spans:
        if span.node_id is not None:
            for node in completed[span.node_id] | {span.node_id}:
                last_established[node] = span.last_frame
    outstanding = {}
    for identity in transition_ids:
        attempt = transitions[identity]
        edge_id = attempt["edge_id"]
        edge = graph.edges[edge_id]
        if (
            attempt["outcome"] == "not_achieved"
            and last_established.get(edge.target, -1) < attempt["last_frame"]
        ):
            outstanding[interfaces[edge_id]] = identity
    return sorted(outstanding.values(), key=lambda key: transitions[key]["first_frame"])


def map_cell(value, version_id, condition_id, *, split="dev"):
    """Resolve one receipt reference; derive an absent dev cell's unknown facts.

    versions owns version-to-Bundle identity. cells and validation_cells own only
    condition-to-receipt references; episodes owns every complete physical cell.
    Nested cells from the removed representation are rejected, never migrated.
    """
    key = {"dev": "cells", "validation": "validation_cells"}[split]
    receipt = value[key][version_id][condition_id]
    if receipt is not None and not isinstance(receipt, str):
        raise ValueError("map cells require a receipt reference or null")
    bundle_id = value["versions"][version_id]
    if receipt is not None:
        cell = value["episodes"][receipt]
        if (
            cell["receipt_id"] != receipt
            or cell["bundle_id"] != bundle_id
            or cell["condition_id"] != condition_id
        ):
            raise ValueError("map receipt differs from its version or condition")
        return cell
    if split != "dev":
        raise ValueError("unmeasured validation conditions must remain absent")
    return {
        "receipt_id": None,
        "bundle_id": bundle_id,
        "condition_id": condition_id,
        "path_changes": [],
        "state_spans": [],
        "completed_nodes": None,
        "attained_nodes": [],
        "lost_nodes": None,
        "progress": None,
        "peak_progress": None,
        "frontier": None,
        "terminal_state_known": False,
        "official_success": None,
        "exit_type": None,
        "transition_ids": [],
        "unresolved_transition_ids": None,
        "remaining_frontier": None,
        "unclassified": [],
        "usage": None,
    }


def map_row(value, version_id, *, split="dev"):
    """Join a version's condition references to their sole episode authority."""
    key = {"dev": "cells", "validation": "validation_cells"}[split]
    return {
        condition: map_cell(value, version_id, condition, split=split)
        for condition in value[key][version_id]
    }


def project_map(
    evidence,
    graph,
    versions,
    conditions,
    *,
    validation_conditions=(),
    partial=False,
    cell_receipts=None,
    graph_id=None,
):
    """One interval projection feeds search, diagnosis and future teacher alignment.

    Transitions are stored once per physical witness, not once per version alias.
    Adjacent witnesses link only inside an observed episode; these are continuation
    observations, not a claim that arbitrary programs compose or transfer.
    cell_receipts explicitly selects current execution receipts. Null or absent
    cells have no physical episode under that execution. A selected infrastructure
    episode retains observed progress but no official capability score. Every
    historical episode remains in the projection and its witness provenance.
    """
    by_receipt = {e.receipt_id: e for e in evidence.episodes}
    if not set(graph.episodes) <= by_receipt.keys() or (
        not partial and set(graph.episodes) != set(by_receipt)
    ):
        raise ValueError("semantic map has a different set of physical receipts")
    completed = graph.completed_nodes()
    milestones = sorted(k for k, n in graph.nodes.items() if n.role == "milestone")
    total = len(milestones) + 1
    starts = {
        identity: {
            x["event"]["execution_id"]: x for x in by_receipt[identity].operation_starts
        }
        for identity in graph.episodes
    }
    originals, programs = {}, {}
    for key, program in graph.programs.items():
        if program.bundle not in originals:
            originals[program.bundle] = read_original(program.bundle)
        programs[key] = {
            **program_assets(program, originals[program.bundle]),
            "transition_ids": [],
        }

    # IDs/explanations/program lists are not physical interface distinctions.
    # Exact duplicate interfaces split across edge labels remain one exposure.
    interface_values = []
    interfaces = {}
    for key, edge in graph.edges.items():
        value = edge.model_dump(
            mode="json",
            include={
                "source",
                "target",
                "roles",
                "entry_conditions",
                "exit_conditions",
            },
        )
        if value not in interface_values:
            interface_values.append(value)
        interfaces[key] = f"interface:{interface_values.index(value) + 1:03d}"
    transitions = {}
    by_episode = {identity: [] for identity in graph.episodes}
    edge_stats = {}
    for edge_id, edge in graph.edges.items():
        counts = {"achieved": 0, "not_achieved": 0, "unknown": 0}
        bundles, episode_ids, condition_ids = set(), set(), set()
        for witness in edge.witnesses:
            identity = witness.episode_id
            episode = by_receipt[identity]
            transition_id = (
                f"{identity}/interval/{witness.first_frame}-{witness.last_frame}"
            )
            if transition_id in transitions:
                raise ValueError("duplicate physical transition witness")
            transitions[transition_id] = {
                "edge_id": edge_id,
                **witness.model_dump(mode="json"),
                "executions": execution_provenance(
                    episode, witness.first_frame, witness.last_frame, starts[identity]
                ),
                "cost": interval_cost(episode, witness.first_frame, witness.last_frame),
                "successor_id": None,
            }
            by_episode[identity].append(transition_id)
            if witness.program_id is not None:
                programs[witness.program_id]["transition_ids"].append(transition_id)
            counts[witness.outcome] += 1
            bundles.add(episode.bundle_id)
            episode_ids.add(identity)
            condition_ids.add(episode.condition_id)
        edge_stats[edge_id] = {
            **counts,
            "bundles": len(bundles),
            "episodes": len(episode_ids),
            "conditions": len(condition_ids),
        }
    gaps = {identity: [] for identity in graph.episodes}
    for gap in graph.unclassified:
        episode = by_receipt[gap.episode_id]
        gaps[gap.episode_id].append(
            {
                **gap.model_dump(mode="json"),
                "executions": execution_provenance(
                    episode, gap.first_frame, gap.last_frame, starts[gap.episode_id]
                ),
                "cost": interval_cost(episode, gap.first_frame, gap.last_frame),
            }
        )
    receipts_by_cell, episode_cells = {}, {}
    node_evidence = {key: {} for key in graph.nodes}
    for identity, spans in graph.episodes.items():
        episode = by_receipt[identity]
        projection = EpisodeProjection(spans, completed)
        achievements = [projection.achievements(span) for span in spans]
        terminal = (
            achievements[-1] if achievements and episode.terminal_state_known else None
        )
        attained = set().union(*(p for p in achievements if p is not None))
        transition_ids = sorted(
            by_episode[identity], key=lambda key: transitions[key]["first_frame"]
        )
        for left_id, right_id in zip(transition_ids, transition_ids[1:]):
            left, right = transitions[left_id], transitions[right_id]
            if left["last_frame"] == right["first_frame"]:
                left["successor_id"] = right_id
        path_changes = []
        for span in spans:
            if not path_changes or span.node_id != path_changes[-1]:
                path_changes.append(span.node_id)
        unresolved, remaining_frontier = None, None
        if episode.official_success:
            unresolved, remaining_frontier = [], []
        elif terminal is not None:
            unresolved = unresolved_transitions(
                graph, spans, transition_ids, transitions, interfaces, completed
            )
            remaining_frontier = sorted(
                key
                for key in milestones
                if key not in terminal and completed[key] - {key} <= terminal
            )
            if not remaining_frontier:
                remaining_frontier = sorted(
                    key for key, node in graph.nodes.items() if node.role == "goal"
                )
        cell = {
            "receipt_id": identity,
            "bundle_id": episode.bundle_id,
            "condition_id": episode.condition_id,
            "path_changes": path_changes,
            "state_spans": [span.model_dump(mode="json") for span in spans],
            "completed_nodes": sorted(terminal) if terminal is not None else None,
            "attained_nodes": sorted(attained),
            "lost_nodes": sorted(attained - terminal) if terminal is not None else None,
            "progress": len(terminal) / total if terminal is not None else None,
            "peak_progress": max(
                (len(p) / total for p in achievements if p is not None), default=0.0
            ),
            "frontier": spans[-1].node_id
            if spans and episode.terminal_state_known
            else None,
            "terminal_state_known": terminal is not None,
            "official_success": None
            if episode.exit_type in {"infrastructure", "unknown"}
            else episode.official_success,
            "exit_type": episode.exit_type,
            "transition_ids": transition_ids,
            # One actual still-unmet witness per conditional interface; null
            # means the final state cannot locate the remaining task frontier.
            "unresolved_transition_ids": unresolved,
            # Ready unmet required milestones (or goals), not a diagnosed cause.
            "remaining_frontier": remaining_frontier,
            "unclassified": sorted(gaps[identity], key=lambda gap: gap["first_frame"]),
            "usage": episode.usage.model_dump(mode="json"),
        }
        key = (episode.bundle_id, episode.condition_id)
        receipts_by_cell.setdefault(key, []).append(identity)
        episode_cells[identity] = cell
        for node_id in {span.node_id for span in spans} - {None}:
            item = node_evidence[node_id].setdefault(
                episode.bundle_id,
                {
                    "bundle_id": episode.bundle_id,
                    "version_ids": sorted(episode.version_ids),
                    "reached_episode_ids": [],
                    "completed_episode_ids": [],
                },
            )
            item["version_ids"] = sorted(
                set(item["version_ids"]) | set(episode.version_ids)
            )
            item["reached_episode_ids"].append(identity)
            if episode.official_success:
                item["completed_episode_ids"].append(identity)
    by_cell = {}
    for key, identities in receipts_by_cell.items():
        if cell_receipts is None:
            if len(identities) != 1:
                raise ValueError("multiple physical receipts for one logical map cell")
            by_cell[key] = identities[0]
            continue
        selected = cell_receipts.get(key)
        if selected is None:
            continue
        episode = by_receipt.get(selected)
        if episode is None or (episode.bundle_id, episode.condition_id) != key:
            raise ValueError("selected receipt belongs to a different map cell")
        if selected in episode_cells:
            by_cell[key] = selected
        elif not partial:
            raise ValueError("selected evaluation receipt is absent from semantic map")
    cells = {}
    validation_cells = {}
    for version in versions:
        row = {}
        for condition in conditions:
            receipt = by_cell.get((version.bundle_id, condition.record_id))
            if receipt is not None:
                row[condition.record_id] = receipt
            elif not partial:
                # An absent settled episode has no state or outcome witness.
                # Keep the full condition denominator without fabricating a
                # physical failure, frontier, cost, or positive progress.
                row[condition.record_id] = None
        cells[version.version_id] = row
        validation_cells[version.version_id] = {
            condition.record_id: by_cell[(version.bundle_id, condition.record_id)]
            for condition in validation_conditions
            if (version.bundle_id, condition.record_id) in by_cell
        }
    return {
        "graph_id": graph_id,
        "execution_graph_id": evidence.record_id,
        "milestone_nodes": milestones,
        "versions": {version.version_id: version.bundle_id for version in versions},
        "episodes": episode_cells,
        "cells": cells,
        "validation_cells": validation_cells,
        "transitions": transitions,
        "edge_interfaces": interfaces,
        "programs": programs,
        "node_evidence": {
            node_id: [
                {
                    **item,
                    "reached_episode_ids": sorted(item["reached_episode_ids"]),
                    "completed_episode_ids": sorted(item["completed_episode_ids"]),
                }
                for _, item in sorted(bundles.items())
            ]
            for node_id, bundles in node_evidence.items()
        },
        "edge_statistics": edge_stats,
        "pending_episode_ids": sorted(set(by_receipt) - set(graph.episodes)),
        "meaning": "One task graph supplies earned progress and conditional transitions linked to actual Harness fragments. Background is zero, unknown is null; branches and repeats add no score. Each physical transition is stored once. Successor links are adjacent observed attempts within the same episode, never counterfactual composition. Program pointers establish provenance, not causal sufficiency or standalone executability. Graph semantics remain revisable hypotheses.",
    }
