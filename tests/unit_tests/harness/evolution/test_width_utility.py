# Modified for the ReWeave namespace.
"""Finite-horizon Bayesian width and measurable cost boundaries."""

import json
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvaluationRequest,
    EvolutionProtocol,
    ProgressForecast,
    SearchCost,
    SearchPolicy,
)
from reweave.evolution.graph.projection import map_row
from reweave.evolution.graph.selection import (
    completion_probability,
    compute_width,
    condition_success_rate,
    estimate_cost,
    forecast_context,
    log_beta_ratio,
    select_parent,
)
from tests.unit_tests.harness.evolution.fixtures import protocol, usage
from tests.unit_tests.harness.evolution.test_semantic_graph import width_map

DECLARED = SearchCost(rollout_usd=None)
# Hand-computed values and the single-condition pilot replay use kappa = 30.
PILOT = SearchPolicy(success_value=30.0)
LEGACY = {
    "breakthrough_reward": 1.0,
    "progress_per_usd": 0.01,
    "progress_per_rollout_second": 0.001,
    "author_cost_estimate_usd": 1.0,
    "rollout_cost_estimate_usd": 2.0,
}


def forecast(q):
    outcomes = [
        {"progress": p, "probability": w}
        for p, w in ((0.0, 1 - q), (100.0, q))
        if w > 0
    ]
    return ProgressForecast(outcomes=outcomes, confidence=0.5, rationale="Synthetic")


def width(q, rounds, maximum=3, cost=DECLARED, policy=None, solved=0, conditions=1):
    return compute_width(
        forecast(q), rounds, maximum, solved, conditions, cost, policy or PILOT
    )


def test_completion_probability_is_forecast_mass_on_the_completion_node():
    value = ProgressForecast(
        outcomes=(
            {"progress": 0.0, "probability": 0.2},
            {"progress": 50.0, "probability": 0.5},
            {"progress": 100.0, "probability": 0.3},
        ),
        confidence=0.5,
        rationale="Synthetic",
    )
    assert completion_probability(value) == pytest.approx(0.3)


def test_log_beta_ratio_is_the_beta_binomial_product():
    # B(a, b + k) / B(a, b) = prod_j (b + j) / (a + b + j).
    assert math.exp(log_beta_ratio(1.5, 1.5, 0, 3)) == pytest.approx(
        0.5 * 2.5 / 4 * 3.5 / 5
    )
    assert log_beta_ratio(0.0, 3.0, 0, 2) == 0.0
    assert log_beta_ratio(0.0, 3.0, 1, 0) == -math.inf
    assert log_beta_ratio(3.0, 0.0, 0, 1) == -math.inf


def test_last_round_buys_siblings_when_success_outweighs_their_dollars():
    # q = 0.5, n0 = 3: psi = 0.5, 0.3125, 0.21875; value = 30 (1 - psi) - K.
    chosen, stats, alternatives = width(0.5, 1)
    assert chosen == 3
    assert [a["value"] for a in alternatives] == pytest.approx([14, 18.625, 20.4375])
    assert stats["sibling_correlation"] == 0.25
    assert stats["success_value_ratio"] == pytest.approx(30)
    assert stats["candidate_planning_usd"] == 28
    assert stats["rollout_uses_declared_price"] == 1


def test_two_round_value_matches_the_hand_computed_bellman_recursion():
    chosen, _, alternatives = width(0.1, 2)
    assert chosen == 2
    assert [a["value"] for a in alternatives] == pytest.approx(
        [3.997325, 4.15565392857, 4.10601065625], abs=1e-9
    )


@pytest.mark.parametrize("q", [0.05, 0.3, 0.6])
def test_unlimited_rounds_make_one_author_optimal(q):
    assert width(q, 60)[0] == 1


@pytest.mark.parametrize(
    "forecasts, widths",
    [
        # Pilot dynamic_k3 rounds (logs/ablation_deep_audit_20261028): Spatial-T04
        # rising chance widens late; Goal-P05 falling chance stays narrow.
        ((0.10, 0.20, 0.15, 0.14, 0.10), [1, 2, 2, 3, 3]),
        ((0.25, 0.12, 0.08, 0.06, 0.05), [1, 1, 1, 1, 2]),
    ],
)
def test_five_round_pilot_states_reproduce_the_frozen_design(forecasts, widths):
    assert [width(q, 5 - t)[0] for t, q in enumerate(forecasts)] == widths


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_certain_or_impossible_completion_needs_one_author(q):
    assert width(q, 3)[0] == 1


def test_success_is_valued_in_declared_candidates():
    assert width(0.5, 1, policy=SearchPolicy(success_value=1.0))[0] == 1
    scaled = PILOT.model_copy(
        update={"author_cost_usd": 16.0, "rollout_cost_usd": 40.0}
    )
    expected = width(0.5, 1)[2]
    for actual, reference in zip(
        width(0.5, 1, policy=scaled)[2], expected, strict=True
    ):
        assert actual == pytest.approx(reference)


def test_measured_rollout_dollars_replace_the_declared_price():
    measured = SearchCost(rollout_usd=812.0)
    chosen, stats, _ = width(0.5, 1, cost=measured)
    assert chosen == 1
    assert stats["candidate_planning_usd"] == 820
    assert stats["success_value_ratio"] == pytest.approx(30 * 28 / 820)
    assert stats["rollout_uses_declared_price"] == 0


def test_conditions_set_the_declared_candidate_price_not_the_value_ratio():
    _, stats, _ = width(0.3125, 1, conditions=2)
    assert stats["candidate_planning_usd"] == 48
    assert stats["success_value_ratio"] == pytest.approx(30)


def test_completion_mass_calibrates_the_per_condition_rate():
    for q, gap in ((0.3125, 2), (0.05, 3), (0.4, 3)):
        rate = condition_success_rate(q, gap, 3.0)
        moment = math.prod((3 * rate + i) / (3 + i) for i in range(gap))
        assert moment == pytest.approx(q, abs=1e-12)
    assert condition_success_rate(0.3, 1, 3.0) == 0.3


def test_two_condition_last_round_matches_the_hand_computed_moments():
    # m = 0.5 (a = b = 1.5): E[t^2] = 0.3125, E[t^4] = 0.1640625,
    # E[t^6] = 0.104736328125; P(some sibling solves both) is inclusion-exclusion.
    chosen, stats, alternatives = width(0.3125, 1, conditions=2)
    assert stats["condition_success_rate"] == pytest.approx(0.5)
    assert chosen == 3
    assert [a["round_success_probability"] for a in alternatives] == pytest.approx(
        [0.3125, 0.4609375, 0.550048828125]
    )
    assert [a["value"] for a in alternatives] == pytest.approx(
        [8.375, 11.828125, 13.50146484375]
    )
    assert alternatives[0]["round_progress_probability"] == pytest.approx(0.6875)


def test_one_remaining_condition_reduces_to_the_single_condition_recursion():
    single = width(0.1, 2)[2]
    for solved in (1, 2):
        alternatives = width(0.1, 2, solved=solved, conditions=solved + 1)[2]
        for actual, reference in zip(alternatives, single, strict=True):
            assert actual == pytest.approx(reference)


@pytest.mark.parametrize("maximum", [0, -1, 1.5, True])
def test_operator_width_limit_is_explicit(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        width(0.5, 1, maximum=maximum)


def test_width_requires_a_remaining_round():
    with pytest.raises(ValueError, match="remaining round"):
        width(0.5, 0)


def test_default_search_policy_is_the_frozen_bayesian_width():
    assert protocol().search == SearchPolicy()
    assert SearchPolicy().model_dump() == {
        "author_cost_usd": 8.0,
        "rollout_cost_usd": 20.0,
        "success_value": 50.0,
        "prior_strength": 3.0,
    }


def test_superseded_policy_fields_are_rejected():
    payload = protocol().model_dump(mode="json")
    payload["search"] = LEGACY
    with pytest.raises(ValidationError):
        EvolutionProtocol.model_validate(payload)


def test_dynamic_width_requires_development_success_to_be_forecast_completion():
    payload = protocol().model_dump(mode="json")
    payload["validation"] = {
        "conditions": [
            {"record_id": "feedback", "task": "drawer", "initialization": {"seed": 9}}
        ],
        "target_success_rate": 1.0,
    }
    with pytest.raises(ValidationError, match="completion event"):
        EvolutionProtocol.model_validate(payload)
    payload["width_mode"] = "fixed"
    EvolutionProtocol.model_validate(payload)
    payload = protocol().model_dump(mode="json")
    payload["dev"].append(
        {"record_id": "second", "task": "drawer", "initialization": {"seed": 2}}
    )
    payload["target_success_rate"] = 0.5
    with pytest.raises(ValidationError, match="completion event"):
        EvolutionProtocol.model_validate(payload)
    payload["max_authors_per_round"] = 1
    EvolutionProtocol.model_validate(payload)


def test_missing_confidence_is_rejected():
    with pytest.raises(ValidationError, match="confidence"):
        ProgressForecast(
            outcomes=({"progress": 50.0, "probability": 1.0},),
            rationale="No certainty estimate",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, True, float("nan")])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValidationError):
        ProgressForecast(
            outcomes=({"progress": 50.0, "probability": 1.0},),
            confidence=confidence,
            rationale="Synthetic",
        )


def test_forecast_target_is_all_seed_terminal_mean_with_unknown_bounds():
    value = width_map(completed=0)
    row = map_row(value, "0")
    row["0"].update(official_success=True, progress=1.0)
    row["1"].update(progress=0.5)
    row["2"].update(progress=None)
    # Peak and historical successes from other candidates are not the target mean.
    for cell in row.values():
        cell["peak_progress"] = 1.0
    context = forecast_context(value, SimpleNamespace(version_id="0"))
    assert len(context["conditions"]) == 4
    assert context["current_progress_lower"] == 37.5
    assert context["current_progress_upper"] == 62.5


def dev_request(config):
    return EvaluationRequest(
        record_id=new_record_id(),
        bundle_id="a" * 64,
        instance=config.dev[0],
        split="dev",
        manifest_id=config.manifest_id,
        conditions=config.execution,
        execution=config.execution_id,
    )


def test_author_history_does_not_change_the_rollout_cost(tmp_path):
    config = protocol()
    state = {"attempts": {}}
    directory = tmp_path / "attempts/dev"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps({"usage": usage().model_dump(mode="json")})
    )
    state["attempts"]["dev"] = {
        "kind": "evaluation",
        "status": "completed",
        "request": dev_request(config).model_dump(mode="json"),
    }
    before = estimate_cost(state, tmp_path, config)
    state["attempts"]["author"] = {"kind": "author", "status": "completed"}
    assert estimate_cost(state, tmp_path, config) == before


def test_cost_keeps_unknown_dollars_retries_and_conditional_validation(tmp_path):
    config = protocol()
    state = {"attempts": {}}

    def add(key, kind, seconds, dollars, request=None):
        directory = tmp_path / "attempts" / key
        directory.mkdir(parents=True)
        value = usage().model_copy(update={"wall_time_s": seconds, "cost_usd": dollars})
        (directory / "result.json").write_text(
            json.dumps({"usage": value.model_dump(mode="json")})
        )
        state["attempts"][key] = {
            "kind": kind,
            "status": "completed",
            "request": request,
        }

    request = dev_request(config)
    add("author", "author", 99999.0, None)
    add("failed_service", "evaluation", 10.0, 2.0, request.model_dump(mode="json"))
    add("dev", "evaluation", 20.0, 3.0, request.model_dump(mode="json"))
    validation = request.model_copy(update={"split": "validation"})
    add("validation", "evaluation", 5.0, None, validation.model_dump(mode="json"))
    result = estimate_cost(state, tmp_path, config)
    assert result.rollout_usd is None  # one unmeasured validation attempt
    path = tmp_path / "attempts/validation/result.json"
    payload = json.loads(path.read_text())
    payload["usage"]["cost_usd"] = 4.0
    path.write_text(json.dumps(payload))
    assert estimate_cost(state, tmp_path, config).rollout_usd == 9.0


def test_exact_parent_ties_go_to_the_newest_version():
    value = width_map(completed=0)
    value["versions"]["1"] = "b"
    value["cells"]["1"], value["validation_cells"]["1"] = {}, {}
    for condition, cell in map_row(value, "0").items():
        receipt = f"new:{condition}"
        value["cells"]["1"][condition] = receipt
        value["episodes"][receipt] = {**cell, "receipt_id": receipt, "bundle_id": "b"}
    old = SimpleNamespace(version_id="0", bundle_id=value["versions"]["0"])
    new = SimpleNamespace(version_id="1", bundle_id="b")
    assert select_parent(value, (old, new)).version_id == "1"
    assert select_parent(value, (new, old)).version_id == "0"
