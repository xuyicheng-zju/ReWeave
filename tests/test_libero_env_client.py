# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""LIBERO RPC client physical-horizon accounting."""

from __future__ import annotations

import numpy as np

from reweave.benchmarks.libero.env_client import LiberoEnvClient
from reweave.benchmarks.libero.tools import TOOLS_SPEC


class _Rpc:
    def __init__(self, meta: dict) -> None:
        self._meta = meta

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        del kwargs, timeout_s
        if method == "env.get_env_meta":
            return dict(self._meta)
        if method == "env.reset":
            return ({"states": np.zeros(8)}, {})
        if method == "env.step":
            return ({"states": np.zeros(8)}, 0.0, False, False, {})
        if method == "env.chunk_step":
            actions = np.asarray(args[0])
            count = int(actions.shape[0])
            return (
                {"states": np.zeros(8)},
                np.zeros(count),
                np.zeros(count, dtype=bool),
                np.zeros(count, dtype=bool),
                {},
            )
        raise AssertionError(f"unexpected RPC method: {method}")


def test_libero_client_counts_control_steps_and_reset_restores_budget():
    meta = {
        "suite": "libero_spatial_lan",
        "task": 4,
        "seed": 51,
        "max_episode_steps": 230,
        "reset_state_id": 0,
        "libero_type": "pro",
    }
    client = LiberoEnvClient(_Rpc(meta), expected_meta=meta)

    assert client.elapsed_steps() == 0
    assert client.steps_remaining() == 230
    client.step(np.zeros(7))
    client.chunk_step(np.zeros((5, 7)))
    assert client.elapsed_steps() == 6
    assert client.steps_remaining() == 224

    client.reset()
    assert client.elapsed_steps() == 0
    assert client.steps_remaining() == 230


def test_view_state_contract_does_not_treat_fixture_names_as_presence_evidence():
    spec = next(item for item in TOOLS_SPEC if item["name"] == "view_env_state")

    assert "movable task objects only" in spec["description"]
    assert "absence from object_names is not evidence" in spec["description"]
