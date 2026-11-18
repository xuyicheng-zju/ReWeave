# Modified for the ReWeave namespace.
"""Host reports derived from exact execution receipts and mutation provenance."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.contracts import (
    BundleVersion,
    FinalProjection,
    RoundDecision,
    VersionReference,
)
from reweave.evolution.search.matrix import (
    best_version,
    changes,
    evaluation_cells,
    passes,
    result_matrix,
    success_set,
    target_passed,
)
from reweave.utils.persistence import atomic_json


def current_final_projection(state):
    """Earlier protocol revisions retain their own final analyses as history."""
    return next(
        (
            stage
            for stage in reversed(tuple(state.get("final_projections", {}).values()))
            if stage["request"]["protocol"]["record_id"] == state.get("protocol")
        ),
        None,
    )


def semantic_projection(state, records):
    """Expose exact map coverage; post-refinement episodes await the next analysis."""
    current = None
    covered = set()
    for _, round_record in sorted(
        state["rounds"].items(), key=lambda item: int(item[0]), reverse=True
    ):
        if round_record["decision"] is None:
            continue
        decision = RoundDecision.model_validate_json(
            Path(round_record["decision"]["path"]).read_text()
        )
        decision.graph.verify()
        decision.progress_map.verify()
        current = decision
        covered = set(
            json.loads(Path(decision.progress_map.path).read_text())["episodes"]
        )
        break
    final = current_final_projection(state)
    if final is not None and final["progress_map"] is not None:
        current = FinalProjection.model_validate(final)
        for evidence in (current.analysis, current.graph, current.progress_map):
            evidence.verify()
        covered = set(
            json.loads(Path(current.progress_map.path).read_text())["episodes"]
        )
    available = {
        r.record_id
        for r in records
        if r.failure_origin != "infrastructure" or r.episode is not None
    }
    return {
        "graph": current.graph.model_dump(mode="json") if current else None,
        "progress_map": current.progress_map.model_dump(mode="json")
        if current
        else None,
        "projected_episode_ids": sorted(covered),
        "pending_episode_ids": sorted(available - covered),
        "meaning": "The archive retains each receipt's first interpretation, including at final selection; new outcomes are projected once."
        if current is not None
        and "interpretations" in json.loads(Path(current.progress_map.path).read_text())
        else "The final graph reprojects the entire settled archive, including the last mutation outcomes, without additional authors or physical trials."
        if final is not None and final["progress_map"] is not None
        else "The latest frozen graph covers only its input receipts. Later physical outcomes remain pending until the next global refinement, including outcomes after the final search round.",
    }


def publish_report(workspace, state, records, accounting):
    """Publish development observations without claiming held-out generalization."""
    versions = {
        k: BundleVersion.model_validate(v) for k, v in state["versions"].items()
    }
    records = tuple(records)
    published = {v.bundle_id for v in versions.values()}
    # A round can stop after receipts were written but before its candidates
    # enter the version archive. Validate and retain that work without treating
    # an uncommitted candidate as a selectable, qualified version.
    evaluation_cells(
        workspace.protocol,
        published | {r.request.bundle_id for r in records},
        records,
        execution_ids=workspace.execution_ids(state),
    )
    published_records = tuple(r for r in records if r.request.bundle_id in published)
    rows = result_matrix(
        workspace.protocol,
        (v.bundle_id for v in versions.values()),
        published_records,
        execution_ids=workspace.execution_ids(state),
    )
    validation_rows = result_matrix(
        workspace.protocol,
        (v.bundle_id for v in versions.values()),
        published_records,
        split="validation",
        execution_ids=workspace.execution_ids(state),
    )
    no_topology = workspace.protocol.evolution_mode == "no_topology"
    best = best_version(
        [
            v
            for v in state["order"]
            if no_topology or None not in rows[versions[v].bundle_id]
        ],
        versions,
        rows,
    )
    qualified = [
        v
        for v in state["order"]
        if target_passed(
            workspace.protocol,
            rows[versions[v].bundle_id],
            validation_rows[versions[v].bundle_id],
        )
    ]
    if qualified:
        best = best_version(qualified, versions, rows)
    official_best = best
    final = current_final_projection(state)
    selection = None
    if final is not None and final["progress_map"] is not None:
        from reweave.evolution.graph.projection import map_row
        from reweave.evolution.graph.selection import parent_statistics, policy_parent

        stage = FinalProjection.model_validate(final)
        stage.progress_map.verify()
        progress_map = json.loads(Path(stage.progress_map.path).read_text())
        references = tuple(
            VersionReference(version_id=k, bundle_id=versions[k].bundle_id)
            for k in state["order"]
        )
        if progress_map["versions"] != {v.version_id: v.bundle_id for v in references}:
            raise ValueError("final selection map must cover every archived version")
        from reweave.evolution.distillation.metrics import harness_size

        sizes = {
            ref.version_id: harness_size(workspace.bundle(ref.bundle_id))["total_bytes"]
            for ref in references
        }
        selected = policy_parent(
            progress_map,
            references,
            workspace.protocol,
            purpose="final_selection",
            successful_sizes=sizes,
        )
        best = selected.version_id
        selection = {
            "criterion": "official_success_then_topology_progress_then_successful_harness_size",
            "version": selected.model_dump(mode="json"),
            "statistics": parent_statistics(map_row(progress_map, best)),
            "graph": stage.graph.model_dump(mode="json"),
            "progress_map": stage.progress_map.model_dump(mode="json"),
            "harness_size_bytes": sizes,
            "size_metric": "deployed_cwl_utf8_bytes",
            "meaning": (progress_map["meaning"] + " ")
            if "interpretations" in progress_map
            else "Globally selects over all versions on one final graph, independently of the search-parent ablation. Confirmed official dev successes come first; mean retained topology progress distinguishes equally successful candidates, followed by observed coverage and validation evidence. Fully successful ties prefer the smaller deployed C/W/L implementation, then the newest version. Size measures representation, not semantic or computational complexity. Qualification is reported separately.",
        }
    initial = rows[state["baseline"]]
    cells = evaluation_cells(
        workspace.protocol,
        rows,
        published_records,
        execution_ids=workspace.execution_ids(state),
    )
    execution_profiles = {
        bundle: {
            condition: record.request.execution
            for (observed_bundle, condition), record in cells.items()
            if observed_bundle == bundle and record.request.split == "dev"
        }
        for bundle in rows
    }
    condition_profiles = {
        bundle: {
            condition: record.request.conditions.model_dump(mode="json")
            for (observed_bundle, condition), record in cells.items()
            if observed_bundle == bundle and record.request.split == "dev"
        }
        for bundle in rows
    }

    def comparable(left, right):
        return (
            execution_profiles[left] == execution_profiles[right]
            and condition_profiles[left] == condition_profiles[right]
        )

    scores = {}
    for identity in state["order"]:
        version = versions[identity]
        row = rows[version.bundle_id]
        complete = None not in row
        score = {
            "bundle_id": version.bundle_id,
            "execution_ids": sorted(
                set(execution_profiles[version.bundle_id].values())
            ),
            # Paired comparisons require budget agreement at each condition,
            # not merely the same set of budgets somewhere in the two rows.
            "execution_by_condition": execution_profiles[version.bundle_id],
            "conditions_by_condition": condition_profiles[version.bundle_id],
            "historical_execution_ids": sorted(
                {
                    record.request.execution
                    for record in records
                    if record.request.bundle_id == version.bundle_id
                    and record.request.split == "dev"
                }
            ),
            "comparable_to_baseline": comparable(version.bundle_id, state["baseline"]),
            "complete": complete,
            "success_rate": len(success_set(row)) / len(row) if complete else None,
            "development_qualified": passes(
                row, workspace.protocol.target_success_rate
            ),
            "target_reached": target_passed(
                workspace.protocol, row, validation_rows[version.bundle_id]
            ),
        }
        validation = validation_rows[version.bundle_id]
        if workspace.protocol.validation is not None:
            validation_complete = None not in validation
            score["validation_complete"] = validation_complete
            score["validation_success_rate"] = (
                sum(validation) / len(validation) if validation_complete else None
            )
        if complete and None not in initial and score["comparable_to_baseline"]:
            score["baseline_delta"] = (sum(row) - sum(initial)) / len(row)
        if version.proposal_id is not None and complete:
            proposal = state["proposals"][version.proposal_id]
            references = proposal["references"]
            score["parent"] = proposal["parent"]
            parent = workspace.bundle(proposal["parent"]["bundle_id"])
            score["parent_changes"] = (
                changes(rows[parent.record_id], row)
                if comparable(parent.record_id, version.bundle_id)
                else None
            )
            score["changed_scopes"] = sorted(
                workspace.bundle(version.bundle_id).novel_scopes((parent,))
            )
            score["reference_changes"] = {
                ref["version_id"]: changes(rows[ref["bundle_id"]], row)
                if comparable(ref["bundle_id"], version.bundle_id)
                else None
                for ref in references
            }
            score["novel_scopes"] = sorted(
                workspace.bundle(version.bundle_id).novel_scopes(
                    tuple(workspace.bundle(ref["bundle_id"]) for ref in references)
                )
            )
            if any(
                None in rows[ref["bundle_id"]]
                or not comparable(ref["bundle_id"], version.bundle_id)
                for ref in references
            ):
                score["integration_missing"] = None
                score["beyond_references"] = None
            else:
                union = set()
                for ref in references:
                    union.update(success_set(rows[ref["bundle_id"]]))
                actual = success_set(row)
                score["integration_missing"] = sorted(union - actual)
                score["beyond_references"] = sorted(actual - union)
        scores[identity] = score
    allocation = {}
    for index, round_record in state["rounds"].items():
        if round_record["decision"] is not None:
            decision = RoundDecision.model_validate_json(
                Path(round_record["decision"]["path"]).read_text()
            )
            allocation[index] = {
                "parent": decision.parent.model_dump(mode="json"),
                "width": decision.width,
                "statistics": decision.statistics,
                "candidate_cost": decision.candidate_cost.model_dump(mode="json"),
                "alternatives": decision.alternatives,
                "analysis": decision.analysis.model_dump(mode="json"),
                "interpretation": "Width alternatives are finite-horizon Bayesian planning values from the forecast completion probability, not physical results. Dollar nulls remain unknown; explicit planning prices are identified in statistics.",
            }
            if decision.search_review:
                allocation[index]["search_review"] = decision.search_review
    value = {
        "method": "physical_graph_evolution",
        "mutation_scopes": workspace.protocol.mutation_scopes,
        "width_mode": workspace.protocol.width_mode,
        "max_authors_per_round": workspace.protocol.max_authors_per_round,
        "success_consolidation": "enabled"
        if workspace.protocol.mutation_scopes == ("W", "C", "L")
        else "skipped_to_preserve_frozen_components",
        "protocol_id": workspace.protocol.record_id,
        "execution_id": workspace.protocol.execution_id,
        "execution": workspace.protocol.execution.model_dump(mode="json"),
        "configuration": json.loads(workspace.require_configuration().read_text()),
        "conditions": [i.model_dump(mode="json") for i in workspace.protocol.dev],
        "matrix": rows,
        "validation": {
            **workspace.protocol.validation.model_dump(mode="json"),
            "matrix": validation_rows,
            "interpretation": "Feedback validation is adaptively reused for diagnosis after failure; it is not an independent held-out test.",
        }
        if workspace.protocol.validation
        else None,
        "versions": state["versions"],
        "order": state["order"],
        "proposals": state["proposals"],
        "graph_rounds": state["rounds"],
        "success_publications": {
            path.stem: json.loads(path.read_text())
            for path in sorted((workspace.root / "publications").glob("*.json"))
        },
        "search_allocation": allocation,
        "semantic_projection": None
        if no_topology
        else semantic_projection(state, records),
        "scores": scores,
        "best_version": best,
        "completed": state["completed"],
        "stop_reason": state["stop_reason"],
        "target_reached": bool(qualified),
        "evaluations": [r.model_dump(mode="json") for r in records],
        "attempts": state["attempts"],
        "total_usage_including_failed_work": accounting,
        "interpretation": "Development observations on one fixed task. Reference coverage is not inherited success; no held-out or generalization claim.",
    }
    if workspace.protocol.evolution_mode == "no_topology":
        value["method"] = "full_without_topology"
        value["selection"] = {
            "criterion": "global_official_success_then_coverage_then_newest",
            "meaning": "Full author/history/engineering and global search retained. Search parents and final output rank confirmed official development successes, then official measurement coverage, then the newest version. Complete rows therefore use SR then recency. Missing outcomes remain unknown and cannot qualify; their confirmed successes only define a lower bound on the fixed development denominator. No topology progress, graph analysis or probability forecast is used.",
        }
        value["round_outcomes"] = {
            str(p["round_index"]): json.loads(path.read_text())
            for p in state["proposals"].values()
            if (
                path := workspace.root
                / "outcomes"
                / f"round-{p['round_index']:06d}.json"
            ).exists()
        }
    uncommitted = [r.record_id for r in records if r.request.bundle_id not in published]
    if uncommitted:
        value["uncommitted_evaluation_ids"] = uncommitted
    outcomes = {
        index: json.loads(Path(record["outcome"]["path"]).read_text())
        for index, record in state["rounds"].items()
        if allocation.get(index, {}).get("search_review")
        and record["outcome"] is not None
    }
    # Old decisions did not observe these diagnostics. Preserve their immutable
    # report shape and do not read unrelated legacy outcome attachments.
    if outcomes:
        value["round_outcomes"] = outcomes
    if state.get("final_projections"):
        value["final_projections"] = state["final_projections"]
    if selection is not None:
        value["selection"] = selection
        value["official_best_version"] = official_best
    # Normalize tuples before immutable comparison.
    value = json.loads(json.dumps(value))
    path = workspace.root / ("report.json" if state["completed"] else "progress.json")
    if path.is_symlink():
        raise ValueError("report must not be a symlink")
    if state["completed"] and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("immutable terminal report changed")
    else:
        atomic_json(path, value)
