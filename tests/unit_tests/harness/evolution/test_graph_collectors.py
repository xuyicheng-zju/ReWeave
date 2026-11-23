# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""The LIBERO adapter exposes measured graph state without evaluator side effects."""

from types import SimpleNamespace

import pytest

from reweave.benchmarks.libero.privileged_state import collect_privileged_critic_state
from reweave.robots.execution_trace import ExecutionTrace
from reweave.robots.physical_evidence import libero_features
from tests.test_libero_privileged_state import _task_env


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_trace_rejects_invalid_action_timing_before_persisting(value):
    trace = ExecutionTrace()
    with pytest.raises(ValueError, match="acknowledgement time"):
        trace.record_telemetry(0, action_wall_time_s=value)
    assert not trace.events


def test_libero_physics_does_not_call_success_or_invent_coupling():
    env = _task_env()
    env._check_success = lambda: pytest.fail("collector must not call evaluator")
    measured = collect_privileged_critic_state(SimpleNamespace(env=env))
    assert "privileged.task.success" not in measured
    assert not any(
        "coupled" in k or "mechanical_engagement" in k or ".stage." in k
        for k in measured
    )
    features = libero_features(measured)
    assert features["privileged.entity.red_cup.distance_to_eef_m"]["unit"] == "m"
    assert features["privileged.joint.cabinet_slide.position"]["unit"] == "m"
    assert features["privileged.joint.stove_knob_hinge.position"]["unit"] == "rad"
    assert features["privileged.contact.max_normal_force_n"]["value"] is None
