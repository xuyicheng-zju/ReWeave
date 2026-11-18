# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Acknowledged, uncertain, and unused robot actions have distinct evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from reweave.benchmarks.libero.tools import LiberoPrimitives
from reweave.dashboard.events import NullDashboardEventSink
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace, merge_control_telemetry
from reweave.robots.vla_execution import VLAChunkPrediction
from reweave.runtime.capability import EnvironmentServiceFailure, PhysicalOutcomeUnknown
from reweave.tools.toolkit import Toolkit
from reweave.utils.rpc.rpc_client import RpcTransportError

KEYS = (
    "available",
    "task.semantic_available",
    "task.goal.predicate_count",
    "task.goal.evaluable_count",
    "task.goal.satisfied_count",
    "task.goal.predicate_fraction_available",
    "contact.robot.count",
    "contact.gripper.count",
    "contact.force_available",
    "contact.max_normal_force_n",
    "joint.count",
)

INFO = {"critic_evidence": {"privileged." + k: 0 for k in KEYS}}


class Env:
    terminated = False
    truncated = False

    def __init__(self):
        self.steps = 0
        self.bad_info = False

    def obs(self):
        return {
            "states": np.zeros(8),
            "main_images": np.zeros((2, 2, 3)),
            "task_descriptions": "original",
        }

    def reset(self):
        self.steps = 0
        return self.obs(), INFO

    def elapsed_steps(self):
        return self.steps

    def step(self, action):
        self.steps += 1
        return self.obs(), 0, False, False, ({} if self.bad_info else INFO)


class Model:
    def __init__(self):
        self.calls = 0

    def predict_action_batch(self, *args, **kwargs):
        self.calls += 1
        return VLAChunkPrediction(
            actions=np.zeros((5, 7)),
            forward_id=f"f{self.calls}",
            prediction_horizon_steps=50,
            execution_horizon_steps=5,
            inference_elapsed_s=0,
        )


class Kit(Toolkit):
    def __init__(self, *, wait=0):
        self.trace = ExecutionTrace()
        self.active = ActiveExecution(self.trace)
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=SimpleNamespace(latest_record=lambda: None),
            active_execution=self.active,
            filesystem_tools_enabled=False,
        )
        self.env = Env()
        self.p = LiberoPrimitives(
            env=self.env,
            model=Model(),
            sam3_client=None,
            check_cancelled=self.raise_if_cancelled,
            execution_trace=self.trace,
            active_execution=self.active,
            wait_steps=wait,
        )
        self.p.reset()
        self.add_tool(
            "act",
            {"name": "act", "input_schema": {"type": "object"}},
            self.p._vlm_chunk,
        )

    def get_env_state(self, *, command, result, elapsed_s):
        return result

    def discard_active_vla_suffix(self):
        return self.p.discard_active_vla_suffix()


def counts(kit):
    rows = kit.trace.events
    return (
        kit.env.steps,
        sum(row["event_type"] == "control_step" for row in rows),
        sum(row.get("discarded_action_steps", 0) for row in rows),
    )


def test_reset_waits_are_recorded_and_exploration_reuses_its_invocation():
    kit = Kit(wait=10)
    assert counts(kit) == (10, 10, 0)
    kit.execute_tool("act", {"instruction": "test"})
    assert counts(kit) == (15, 15, 0)
    kit.active.begin("reset")
    kit.p.reset()
    kit.active.finish("completed")
    assert counts(kit) == (10, 25, 0)
    assert (
        len(
            [
                row
                for row in kit.trace.events
                if row["event_type"] == "active_execution_started"
            ]
        )
        == 3
    )


@pytest.mark.parametrize("fault", ["telemetry", "video", "rpc"])
def test_post_step_fault_cannot_reclassify_applied_action_as_unused(fault):
    kit = Kit()
    arguments = {"instruction": "test"}
    if fault == "telemetry":
        kit.env.bad_info = True
    elif fault == "video":
        kit.p.start_recording()

        def failed_video(_):
            raise OSError("video failure")

        kit.p.record_frame = failed_video
    else:

        def lost_response(*args, **kwargs):
            kit.env.steps += 1
            raise OSError("lost response after physical execution")

        kit.env.step = lost_response
    if fault == "rpc":
        with pytest.raises(PhysicalOutcomeUnknown):
            kit.execute_tool("act", arguments)
        assert counts(kit) == (1, 0, 4)
        unknown = [
            row
            for row in kit.trace.events
            if row["event_type"] == "action_outcome_unknown"
        ]
        assert len(unknown) == 1 and unknown[0]["action_index"] == 0
        with pytest.raises(PhysicalOutcomeUnknown):
            kit.execute_tool("act", arguments)
        assert kit.env.steps == 1
    else:
        result = kit.execute_tool("act", arguments).result
        assert result["error"]
        assert counts(kit) == (1, 1, 4)
        controls = [
            row
            for row in merge_control_telemetry(kit.trace.events)
            if row["event_type"] == "control_step"
        ]
        if fault == "telemetry":
            assert controls[0]["telemetry_error"]
        assert kit.active.snapshot.phase == "idle"


@pytest.mark.parametrize("acknowledged", [0, 2])
@pytest.mark.parametrize("scripted", [False, True])
def test_unsubmitted_step_preserves_ack_prefix_and_discards_the_whole_unused_suffix(
    acknowledged, scripted
):
    kit = Kit()
    original_step = kit.env.step
    transport = RpcTransportError(
        "env.step", "route unavailable", recovery_state="not_submitted"
    )
    calls = []

    def step(action):
        calls.append(action)
        if kit.env.steps == acknowledged:
            raise transport
        return original_step(action)

    kit.env.step = step
    if scripted:

        def move():
            for index in range(5):
                kit.p._scripted_control_step(np.zeros(7), action_index=index)

        kit.add_tool("move", {"name": "move", "input_schema": {"type": "object"}}, move)
        name, arguments = "move", {}
    else:
        name, arguments = "act", {"instruction": "test"}

    def forbidden_state_capture(**kwargs):
        pytest.fail("a known transport failure must stop subsequent environment I/O")

    kit.get_env_state = forbidden_state_capture
    with pytest.raises(EnvironmentServiceFailure) as failed:
        kit.execute_tool(name, arguments)
    assert failed.value.retryable is True
    assert failed.value.__cause__ is transport
    assert kit.physical_outcome_error is None
    assert counts(kit) == (
        acknowledged,
        acknowledged,
        0 if scripted else 5 - acknowledged,
    )
    assert kit.active.snapshot.phase == "idle"
    assert kit.p.leftover_vla_suffix_steps() == 0
    assert not any(
        row["event_type"] == "action_outcome_unknown" for row in kit.trace.events
    )
    with pytest.raises(EnvironmentServiceFailure) as blocked:
        kit.execute_tool(name, arguments)
    assert blocked.value is failed.value and len(calls) == acknowledged + 1


@pytest.mark.parametrize(
    "recovery_state", ["unavailable", "unknown", "instance_changed"]
)
def test_unconfirmed_submission_stays_unknown_despite_error_message(recovery_state):
    kit = Kit()

    def step(action):
        kit.env.steps += 1
        raise RpcTransportError(
            "env.step",
            "not_submitted appeared in an untrusted diagnostic",
            recovery_state=recovery_state,
        )

    kit.env.step = step
    with pytest.raises(PhysicalOutcomeUnknown):
        kit.execute_tool("act", {"instruction": "test"})
    assert counts(kit) == (1, 0, 4)
    assert (
        sum(row["event_type"] == "action_outcome_unknown" for row in kit.trace.events)
        == 1
    )


@pytest.mark.parametrize("corruption", ["unknown", "owner", "duplicate", "source"])
def test_telemetry_join_rejects_wrong_identity(corruption):
    kit = Kit(wait=1)
    rows = json.loads(json.dumps(kit.trace.events))
    event = next(row for row in rows if row["event_type"] == "control_telemetry")
    if corruption == "unknown":
        event["control_step_index"] = 999
    elif corruption == "owner":
        event["execution_ref"] = "another action"
    elif corruption == "source":
        event["control_event_index"] = -1
    else:
        rows.append(dict(event, event_index=len(rows)))
    with pytest.raises(ValueError):
        merge_control_telemetry(rows)


@pytest.mark.parametrize("plane", ["public_evidence", "critic_private_evidence"])
def test_telemetry_join_still_rejects_overwriting_nonempty_evidence(plane):
    kit = Kit(wait=1)
    rows = json.loads(json.dumps(kit.trace.events))
    control = next(row for row in rows if row["event_type"] == "control_step")
    telemetry = next(row for row in rows if row["event_type"] == "control_telemetry")
    key = "existing" if plane == "public_evidence" else "privileged.existing"
    control[plane] = {key: 1}
    telemetry[plane] = {key: 2}
    with pytest.raises(ValueError, match="overwrite"):
        merge_control_telemetry(rows)


def test_uncertain_gateway_failure_cannot_be_caught_by_generated_code():
    import asyncio

    from reweave.evolution.bundle import ProgramLimits, PythonFile, PythonProgram
    from reweave.runtime.sandbox import ProgramSandbox

    program = PythonProgram(
        entrypoint="workflow:run",
        files=(
            PythonFile(
                path="workflow.py",
                source="""def run(runtime, context):
    try:
        runtime.act("move", {})
    except Exception:
        return {"status": "completed"}
""",
            ),
        ),
    )

    async def gateway(*args):
        raise PhysicalOutcomeUnknown("unacknowledged motion")

    async def scenario():
        with pytest.raises(PhysicalOutcomeUnknown, match="unacknowledged"):
            await ProgramSandbox().run_program(
                program,
                ProgramLimits(
                    runtime_calls=1, physical_calls=1, model_calls=0, wall_time_s=10
                ),
                {},
                gateway,
                input_schema={"type": "object"},
            )

    asyncio.run(scenario())


def test_rpc_uncertainty_takes_priority_over_concurrent_cancellation():
    import asyncio
    import threading
    from contextvars import ContextVar

    from reweave.planner.bundle_runtime import BundleRuntime

    async def scenario():
        kit = Kit()
        entered, release = threading.Event(), threading.Event()

        def step(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            kit.env.steps += 1
            raise OSError("lost action acknowledgement")

        kit.env.step = step
        runtime = object.__new__(BundleRuntime)
        runtime.toolkit = kit
        runtime._submitted = set()
        runtime._memory_bindings = {}
        runtime.history = []
        runtime.settled_operation = ContextVar("settled_operation", default=None)
        pending = asyncio.create_task(runtime._dispatch("act", {"instruction": "test"}))
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        release.set()
        with pytest.raises(PhysicalOutcomeUnknown):
            await pending
        assert counts(kit) == (1, 0, 4)

    asyncio.run(scenario())


def test_cleanup_failure_cannot_mask_rpc_uncertainty():
    kit = Kit()

    def lost_response(*args, **kwargs):
        kit.env.steps += 1
        raise OSError("lost action acknowledgement")

    kit.env.step = lost_response

    def failed_discard():
        raise RuntimeError("cleanup failed")

    kit.p.discard_active_vla_suffix = failed_discard
    with pytest.raises(PhysicalOutcomeUnknown):
        kit.execute_tool("act", {"instruction": "test"})
    assert kit._active_operation is None
    assert any(
        row["event_type"] == "action_outcome_unknown" for row in kit.trace.events
    )
