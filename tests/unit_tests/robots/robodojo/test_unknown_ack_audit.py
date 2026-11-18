# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Audit failures cannot downgrade an unacknowledged physical action."""

import asyncio

import numpy as np
import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.runtime.capability import PhysicalOutcomeUnknown


@pytest.mark.parametrize("acknowledgement", ["missing", "malformed"])
@pytest.mark.parametrize("audit_error_type", [OSError, asyncio.CancelledError])
@pytest.mark.parametrize("tool", ["set_gripper", "pi0_pick"])
def test_unknown_action_survives_audit_failure(
    tmp_path, monkeypatch, acknowledgement, audit_error_type, tool
):
    rpc = SimulatedEnvironmentRPC(acknowledgement=acknowledgement, terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    audit_error = audit_error_type("SIMULATED unknown-action audit failure")
    original_append = toolkit._execution_trace._append

    def fail_audit(event):
        if event["event_type"] == "action_outcome_unknown":
            raise audit_error
        return original_append(event)

    monkeypatch.setattr(toolkit._execution_trace, "_append", fail_audit)
    try:
        arguments = (
            {"arm": "right", "gripper": 1}
            if tool == "set_gripper"
            else {"max_chunks": 1}
        )
        with pytest.raises(PhysicalOutcomeUnknown) as raised:
            toolkit.execute_tool(tool, arguments)
        unknown = raised.value
        assert toolkit.physical_outcome_error is unknown
        assert isinstance(
            unknown.__cause__,
            TimeoutError if acknowledgement == "missing" else ValueError,
        )
        assert any(
            type(audit_error).__name__ in note and str(audit_error) in note
            for note in unknown.__notes__
        )
        assert toolkit._primitives.env._outcome_unknown is True
        assert toolkit._primitives.env.elapsed_steps() == 0
        with pytest.raises(PhysicalOutcomeUnknown) as replay:
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert replay.value is unknown
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit._primitives.env.step(np.zeros(14))
        with pytest.raises(ValueError, match="cannot reset"):
            toolkit._primitives.env.reset()
        assert rpc.actions == 1
        assert not any(
            row["event_type"] == "control_step"
            for row in toolkit._execution_trace.events
        )
        if tool == "pi0_pick":
            # Only the three provably unsubmitted actions may be discarded.
            assert (
                sum(
                    row["discarded_action_steps"]
                    for row in toolkit._execution_trace.events
                    if row["event_type"] == "action_suffix_discarded"
                )
                == 3
            )
    finally:
        toolkit.close()


def test_proven_unsubmitted_action_is_infrastructure_without_unknown_or_budget_reset(
    tmp_path, monkeypatch
):
    from reweave.runtime.capability import EnvironmentServiceFailure
    from reweave.utils.rpc.rpc_client import RpcTransportError

    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    env = toolkit._primitives.env
    previous = env.last_obs
    original = rpc.call

    def unavailable(method, *args, **kwargs):
        if method == "env.step":
            raise RpcTransportError(
                method, "route probe failed", recovery_state="not_submitted"
            )
        return original(method, *args, **kwargs)

    monkeypatch.setattr(rpc, "call", unavailable)
    try:
        with pytest.raises(EnvironmentServiceFailure):
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert rpc.actions == env.elapsed_steps() == 0
        assert not env._outcome_unknown
        assert env.last_obs is previous
        assert not any(
            row["event_type"] == "action_outcome_unknown"
            for row in toolkit._execution_trace.events
        )
    finally:
        toolkit.close()
