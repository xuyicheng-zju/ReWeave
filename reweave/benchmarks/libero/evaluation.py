# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Auditable LIBERO-Pro evaluation contracts shared by runners and tests.
"""

from __future__ import annotations

from typing import Any

LIBERO_POLICY_ACTION_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBERO_WAIT_STEPS_DEFAULT = 10
LIBERO_ACTIONS_PER_CHUNK = 5

ASPIRE_DEVELOPMENT_TRIALS = tuple(range(51, 66))
ASPIRE_HELDOUT_TRIALS = tuple(range(1, 51))
LIBERO_PRO_PERTURBATION_SUFFIXES = {
    "object": "object",
    "position": "swap",
    "semantic": "lan",
    "task": "task",
    "environment": "env",
}
LIBERO_PRO_DEBUG_PERTURBATIONS = (
    "object",
    "position",
    "semantic",
    "task",
)


def libero_base_suite(suite: str) -> str:
    """Return the base family that determines the OpenPI action horizon."""

    normalized = str(suite).strip()
    for base_suite in LIBERO_POLICY_ACTION_HORIZONS:
        if normalized == base_suite or normalized.startswith(f"{base_suite}_"):
            return base_suite
    raise ValueError(f"no OpenPI action horizon for LIBERO suite {suite!r}")


def libero_horizon_contract(
    suite: str,
    *,
    policy_action_horizon: int | None = None,
    wait_steps: int = LIBERO_WAIT_STEPS_DEFAULT,
) -> dict[str, Any]:
    base_suite = libero_base_suite(suite)
    standard_policy_horizon = LIBERO_POLICY_ACTION_HORIZONS[base_suite]
    policy_horizon = (
        standard_policy_horizon
        if policy_action_horizon is None
        else int(policy_action_horizon)
    )
    wait = int(wait_steps)
    if policy_horizon <= 0:
        raise ValueError("LIBERO policy action horizon must be positive")
    if wait < 0:
        raise ValueError("LIBERO wait steps must be non-negative")
    return {
        "schema_version": 1,
        "protocol": "aspire_split_zetta_openpi_vla_v1",
        "suite": str(suite).strip(),
        "base_suite": base_suite,
        "standard_policy_action_horizon": standard_policy_horizon,
        "policy_action_horizon": policy_horizon,
        "wait_steps": wait,
        "max_episode_steps": policy_horizon + wait,
        "actions_per_chunk": LIBERO_ACTIONS_PER_CHUNK,
        "is_standard": (
            policy_horizon == standard_policy_horizon
            and wait == LIBERO_WAIT_STEPS_DEFAULT
        ),
    }


def aspire_trial_partition(trial: int) -> str:
    trial = int(trial)
    if trial in ASPIRE_DEVELOPMENT_TRIALS:
        return "development"
    if trial in ASPIRE_HELDOUT_TRIALS:
        return "heldout"
    raise ValueError(
        "ASPIRE LIBERO trial must be in held-out 1..50 or development 51..65"
    )


def aspire_trial_to_reset_state_id(trial: int, state_count: int) -> int:
    aspire_trial_partition(trial)
    state_count = int(state_count)
    if state_count <= 0:
        raise ValueError("LIBERO reset-state count must be positive")
    return (int(trial) - 1) % state_count
