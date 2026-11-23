# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Deterministic global search over the shared semantic progress map."""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import cache
from itertools import product
from statistics import median

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvaluationRequest,
    RoundDecision,
    SearchCost,
    Usage,
)
from reweave.evolution.graph.projection import map_row
from reweave.evolution.infrastructure.persistence import attempt_directory
from reweave.evolution.search.matrix import official_rank


def unique_versions(versions):
    """One version per executable content, in archive order; repeats keep the latest."""
    latest = {version.bundle_id: version for version in versions}
    return [version for version in versions if latest[version.bundle_id] is version]


def parent_statistics(row):
    """All-condition terminal progress; unknowns contribute only their lower bound.

    project_map supplies a complete development row under one current graph.
    Unknown physical states remain in the progress denominator. A known physical
    state does not fill an infrastructure-missing official capability score.
    """
    return {
        "parent_mean_progress": math.fsum(c["progress"] or 0 for c in row.values())
        / len(row),
        "parent_unknown_conditions": float(
            sum(c["progress"] is None for c in row.values())
        ),
        "parent_success_rate": None
        if any(c["official_success"] is None for c in row.values())
        else sum(c["official_success"] is True for c in row.values()) / len(row),
    }


def select_parent(progress_map, versions, *, successful_sizes=None):
    """Globally maximize official successes, then mean retained topology progress.

    The fixed development denominator makes confirmed success counts comparable;
    missing outcomes do not become complete success rates. Among equally
    successful Harnesses, the freshly reprojected map distinguishes retained
    partial progress and frontiers for further revision. Prefer observed
    terminal/official coverage on exact ties, then feedback-validation evidence.
    Final selection may prefer a smaller deployed implementation when all
    development and measured validation conditions succeed. This never trades
    success or progress for size. Remaining ties use archive recency.
    """
    available = unique_versions(versions)
    if not available:
        raise ValueError("parent selection requires historical Harnesses")

    def rank(version):
        row = map_row(progress_map, version.version_id)
        successes, official_coverage = official_rank(
            tuple(c["official_success"] for c in row.values())
        )
        # All rows share the same graph denominator and development conditions.
        # Compare exact node counts so float rounding cannot break a real tie.
        score = (
            successes,
            sum(len(c["completed_nodes"] or ()) for c in row.values()),
            sum(c["progress"] is not None for c in row.values()),
            official_coverage,
        )
        feedback = (
            map_row(progress_map, version.version_id, split="validation")
            if progress_map.get("validation_cells", {}).get(version.version_id)
            else {}
        )
        return (
            *score,
            sum(c["official_success"] is True for c in feedback.values()),
            sum(len(c["completed_nodes"] or ()) for c in feedback.values()),
            sum(c["progress"] is not None for c in feedback.values()),
            sum(c["official_success"] is not None for c in feedback.values()),
            -successful_sizes[version.version_id]
            if successful_sizes is not None
            and row
            and all(c["official_success"] is True for c in row.values())
            and all(c["official_success"] is True for c in feedback.values())
            else 0,
        )

    # max keeps the first maximum, so scanning newest first breaks ties by recency.
    return max(reversed(available), key=rank)


def current_round_versions(state):
    """Freeze all siblings in the latest published round using proposal provenance."""
    rounds = {}
    for identity in state["order"]:
        proposal = state["versions"][identity]["proposal_id"]
        rounds[identity] = (
            -1 if proposal is None else state["proposals"][proposal]["round_index"]
        )
    latest = max(rounds.values())
    return tuple(
        identity for identity, round_index in rounds.items() if round_index == latest
    )


def policy_parent(
    progress_map,
    versions,
    protocol,
    *,
    purpose="search",
    successful_sizes=None,
    current_versions=(),
    audit=None,
):
    """Apply the protocol's search-parent rule; final selection is always global."""
    if successful_sizes is not None and purpose != "final_selection":
        raise ValueError("implementation size is only a final-selection preference")
    if purpose == "search" and protocol.parent_selection == "latest_evaluated":
        # Requests preserve archive order. A settled infrastructure failure is
        # not a complete development measurement and cannot replace the parent.
        for version in reversed(tuple(versions)):
            row = map_row(progress_map, version.version_id)
            if row and all(c["official_success"] is not None for c in row.values()):
                return version
        raise ValueError("latest-evaluated parent requires a complete development row")
    winner = select_parent(progress_map, versions, successful_sizes=successful_sizes)
    if purpose != "search" or not current_versions:
        return winner
    current = [
        version
        for version in versions
        if version.version_id in current_versions
        and (row := map_row(progress_map, version.version_id))
        and all(c["official_success"] is not None for c in row.values())
    ]
    if not current:
        # Missing official evidence is not an equal-SR comparison.
        return winner
    anchor = select_parent(progress_map, current)
    winner_row = map_row(progress_map, winner.version_id)
    anchor_row = map_row(progress_map, anchor.version_id)
    winner_successes = sum(c["official_success"] is True for c in winner_row.values())
    anchor_successes = sum(c["official_success"] is True for c in anchor_row.values())
    gap = (
        parent_statistics(winner_row)["parent_mean_progress"]
        - parent_statistics(anchor_row)["parent_mean_progress"]
    )
    threshold = protocol.parent_rollback_min_progress
    equal_sr = winner_successes == anchor_successes and all(
        c["official_success"] is not None for c in winner_row.values()
    )
    historical = winner.version_id not in current_versions
    enough_progress = gap >= threshold or math.isclose(
        gap, threshold, rel_tol=0, abs_tol=1e-12
    )
    blocked = historical and equal_sr and not enough_progress
    selected = anchor if blocked else winner
    if audit is not None:
        audit.update(
            current_round_versions=list(current_versions),
            current_round_best=anchor.version_id,
            unrestricted_best=winner.version_id,
            selected=selected.version_id,
            min_progress=threshold,
            progress_gap=gap,
            equal_official_success=equal_sr,
            historical_winner=historical,
            rollback_blocked=blocked,
        )
    return selected


def forecast_context(progress_map, parent):
    """Bind the forecast to the parent, with bounds on the same 0–100 scale."""
    row = {
        **map_row(progress_map, parent.version_id),
        **map_row(progress_map, parent.version_id, split="validation"),
    }
    known = math.fsum(
        1.0 if c["official_success"] else c["progress"] or 0.0 for c in row.values()
    )
    unknown = sum(
        c["progress"] is None and not c["official_success"] for c in row.values()
    )
    return {
        "parent_version_id": parent.version_id,
        "conditions": sorted(row),
        "current_progress_lower": 100.0 * known / len(row),
        "current_progress_upper": 100.0 * (known + unknown) / len(row),
    }


def completion_probability(forecast):
    """Forecast mass on the topology's completion node: one candidate succeeds."""
    total = math.fsum(o.probability for o in forecast.outcomes)
    return (
        math.fsum(o.probability for o in forecast.outcomes if o.progress >= 100.0)
        / total
    )


def log_beta_ratio(alpha, beta, successes, failures):
    """log B(alpha + successes, beta + failures) / B(alpha, beta).

    A zero parameter is a point mass at theta = 0 or 1, under which an outcome of
    the opposite kind is impossible.
    """
    if alpha == 0:
        return 0.0 if successes == 0 else -math.inf
    if beta == 0:
        return 0.0 if failures == 0 else -math.inf
    return (
        math.lgamma(alpha + successes)
        + math.lgamma(beta + failures)
        - math.lgamma(alpha + beta + successes + failures)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(alpha + beta)
    )


def condition_success_rate(completion, gap, strength):
    """Prior mean m with E[theta**gap] = completion under Beta(n0 m, n0 (1 - m))."""
    if gap <= 1 or completion in (0.0, 1.0):
        return completion
    low, high = 0.0, 1.0
    for _ in range(60):
        rate = (low + high) / 2
        moment = math.prod((strength * rate + i) / (strength + i) for i in range(gap))
        low, high = (rate, high) if moment < completion else (low, rate)
    return (low + high) / 2


@cache
def sibling_outcomes(gap, siblings):
    """Binomial weights of the siblings' new successes, grouped by (total, best)."""
    groups = Counter()
    for gains in product(range(gap + 1), repeat=siblings):
        groups[sum(gains), max(gains)] += math.prod(math.comb(gap, g) for g in gains)
    return tuple(groups.items())


def compute_width(
    forecast, rounds_remaining, maximum, solved, conditions, cost, policy
):
    """Finite-horizon Bayesian width over development conditions.

    Siblings share one unknown per-condition success rate
    theta ~ Beta(n0 m, n0 (1 - m)). From a parent solving `solved` of `conditions`,
    each candidate adds Binomial(gap, theta) successes; the archive keeps the best,
    and success means solving every condition. m is set so that E[theta**gap] is
    the forecast's completion mass. In units of one candidate's planned dollars,

        J_R(s, a, b) = max_K -K + E[kappa 1{best = gap}
                                    + 1{best < gap} J_{R-1}(s + best, a + S, b + F)],

    where S and F count every sibling's new successes and failures, J_0 = 0. One
    condition reduces this to J_R = max_K kappa (1 - psi_K) - K + psi_K J_{R-1}
    with psi_K = B(a, b + K) / B(a, b). With unlimited rounds one author is always
    optimal; extra siblings are bought only by the finite round budget.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum width must be a positive integer")
    if rounds_remaining < 1:
        raise ValueError("width requires at least one remaining round")
    q = completion_probability(forecast)
    strength = policy.prior_strength
    rate = condition_success_rate(q, conditions - solved, strength)
    alpha, beta = strength * rate, strength * (1.0 - rate)
    declared_rollouts = conditions * policy.rollout_cost_usd
    rollout_usd = (
        cost.rollout_usd if cost.rollout_usd is not None else declared_rollouts
    )
    candidate_usd = policy.author_cost_usd + rollout_usd
    kappa = (
        policy.success_value
        * (policy.author_cost_usd + declared_rollouts)
        / candidate_usd
    )
    values = {}

    def outcomes(solved, successes, failures, k):
        gap = conditions - solved
        for (gained, best), weight in sibling_outcomes(gap, k):
            lost = k * gap - gained
            log_p = log_beta_ratio(alpha + successes, beta + failures, gained, lost)
            if log_p > -math.inf:
                yield weight * math.exp(log_p), gap, gained, lost, best

    def choice(rounds, solved, successes, failures, k):
        total = -float(k)
        for p, gap, gained, lost, best in outcomes(solved, successes, failures, k):
            total += p * (
                kappa
                if best == gap
                else value(
                    rounds - 1, solved + best, successes + gained, failures + lost
                )
            )
        return total

    def value(rounds, solved, successes, failures):
        if rounds == 0:
            return 0.0
        key = rounds, solved, successes, failures
        if key not in values:
            values[key] = max(
                choice(rounds, solved, successes, failures, k)
                for k in range(1, maximum + 1)
            )
        return values[key]

    def alternative(k):
        rows = tuple(outcomes(solved, 0, 0, k))
        return {
            "width": float(k),
            "round_success_probability": math.fsum(p for p, g, *_, b in rows if b == g),
            "round_progress_probability": math.fsum(p for p, *_, b in rows if b > 0),
            "value": choice(rounds_remaining, solved, 0, 0, k),
        }

    alternatives = tuple(alternative(k) for k in range(1, maximum + 1))
    best = alternatives[0]
    for candidate in alternatives[1:]:
        # Ties, including roundoff, keep the smaller width.
        if candidate["value"] > best["value"] + 1e-12:
            best = candidate
    statistics = {
        "completion_probability": q,
        "condition_success_rate": rate,
        "prior_alpha": alpha,
        "prior_beta": beta,
        "sibling_correlation": 1.0 / (1.0 + strength),
        "parent_successes": float(solved),
        "development_conditions": float(conditions),
        "rounds_remaining": float(rounds_remaining),
        "candidate_planning_usd": candidate_usd,
        "rollout_uses_declared_price": float(cost.rollout_usd is None),
        "success_value_ratio": kappa,
        "maximum_width": float(maximum),
    }
    return int(best["width"]), statistics, alternatives


def decide_round(
    request,
    receipt,
    analysis_file,
    graph_file,
    map_file,
    progress_map,
    candidate_cost,
    state,
):
    if receipt.request_id != request.record_id:
        raise ValueError("analysis receipt answers another request")
    protocol = request.protocol
    dynamic = protocol.width_mode == "dynamic" and protocol.max_authors_per_round > 1
    # A published receipt has passed Host graph admission. Selection consumes
    # that result; it must not rerun raw-witness validation or graph analysis.
    parent_audit = {}
    parent = policy_parent(
        progress_map,
        request.versions,
        request.protocol,
        current_versions=request.current_round_versions,
        audit=parent_audit,
    )
    if dynamic:
        if receipt.analysis.forecast is None:
            raise ValueError("dynamic width requires a candidate forecast")
        width, statistics, alternatives = compute_width(
            receipt.analysis.forecast,
            protocol.max_rounds - request.round_index,
            protocol.max_authors_per_round,
            sum(
                c["official_success"] is True
                for c in map_row(progress_map, parent.version_id).values()
            ),
            len(protocol.dev),
            candidate_cost,
            protocol.search,
        )
    else:
        width = protocol.max_authors_per_round
        statistics, alternatives = {}, ()
    statistics.update(parent_statistics(map_row(progress_map, parent.version_id)))
    from reweave.evolution.search.review import review_search

    parent_reason = (
        "Parent prioritizes confirmed official dev successes, then mean retained "
        "topology progress over the fixed development conditions. At equal "
        "success counts, graph progress distinguishes partial completion and "
        "frontiers for further revision. Observed terminal and official "
        "measurement coverage follow; the newest version wins remaining ties. "
        "Only the parent Bundle is copied; every historical program remains "
        "accessible through the shared graph. No version donor selection or "
        "code composition is imposed. "
    )
    if parent_audit:
        parent_reason += (
            f"At equal official dev SR, rollback to an older round requires at least "
            f"{100 * protocol.parent_rollback_min_progress:g} percentage points more "
            "mean progress than the best completely evaluated current-round sibling. "
            f"Rollback blocked: {parent_audit['rollback_blocked']}. "
        )
    if request.protocol.parent_selection == "latest_evaluated":
        parent_reason = "Parent is the newest version with a complete development evaluation. Only the parent Bundle is copied; every historical program remains accessible through the shared graph. Final archive selection remains global. "

    return RoundDecision(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=analysis_file,
        graph=graph_file,
        progress_map=map_file,
        parent=parent,
        references=(parent,),
        width=width,
        candidate_cost=candidate_cost,
        alternatives=alternatives,
        statistics=statistics,
        parent_selection_audit=parent_audit,
        search_review=review_search(progress_map, request.versions, state, parent),
        reason=parent_reason
        + (
            f"Width is fixed by the protocol at K={width}; no probability forecast or dynamic recommendation is computed. Every sibling completes before stopping at round success; all author and rollout work is charged."
            if protocol.width_mode == "fixed"
            else "Width is the smallest maximizer of the finite-horizon value of solving every development condition. Siblings share a per-condition success rate theta ~ Beta(n0 m, n0 (1 - m)), with m set so that E[theta**gap] equals the forecast's completion mass; partial gains advance the parent and the search continues. Each K trades K candidates' planned author and rollout dollars against that value given the remaining rounds. No repair is assigned to authors."
            if dynamic
            else "Width is one: the protocol allows a single author."
        ),
    )


def attempt_dollars(root, key, attempt):
    """Measured dollars of one settled attempt; an unmeasured attempt stays null."""
    directory = attempt_directory(root, key, attempt)
    receipt = directory / "result.json"
    source = receipt if receipt.exists() else directory / "audit/usage.json"
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    return Usage.model_validate(
        payload["usage"] if "usage" in payload else payload
    ).cost_usd


def estimate_cost(state, root, protocol):
    """Measured rollout dollars per candidate; null while any part is unmeasured."""
    jobs, validation, evaluated = {}, [], set()
    for key, attempt in state["attempts"].items():
        if attempt["kind"] != "evaluation" or attempt["status"] not in {
            "completed",
            "timed_out",
        }:
            continue
        request = EvaluationRequest.model_validate(attempt["request"])
        evaluated.add(request.bundle_id)
        dollars = attempt_dollars(root, key, attempt)
        if request.split == "validation":
            validation.append(dollars)
        else:
            group = jobs.setdefault(request.instance.record_id, {})
            group.setdefault(request.record_id, []).append(dollars)

    def total(values):
        return None if None in values else math.fsum(values)

    per_condition = []
    for condition in protocol.dev:
        job_dollars = [total(d) for d in jobs[condition.record_id].values()]
        per_condition.append(None if None in job_dollars else median(job_dollars))
    validation_dollars = total(validation)
    per_condition.append(
        None if validation_dollars is None else validation_dollars / len(evaluated)
    )
    return SearchCost(rollout_usd=total(per_condition))
