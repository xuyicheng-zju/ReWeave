"""Same-session recovery through real C/L/Kernel; no model or simulator calls."""

import asyncio
import json
import uuid
from functools import wraps

import pytest

from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers.base import (
    NativeCompletedInvalidDecision,
    NativeProtocolError,
    NativeRecoveryBlocked,
    NativeTransportError,
    NativeTurnFailed,
    NativeTurnRecoveryRequired,
)
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


DECISION = {
    "text": "Execute once.",
    "tool_calls": [{"name": "advance", "arguments_json": "{}"}],
}


class ActionRobot(Robot):
    def __init__(self, path):
        super().__init__(path)
        self.add_tool(
            "advance",
            {
                "name": "advance",
                "description": "Synthetic acknowledged effect",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.advance,
        )

    def advance(self):
        self.calls += 1
        self.terminal = True
        return {"acknowledged": True}


class RecoverySession:
    def __init__(self, directory, outcomes):
        self.directory = directory
        self.session_id = "SAME-SID"
        self.context_generation = 0
        self.outcomes = iter(outcomes)
        self.inputs = []
        self.starts = 0
        self.closed = False
        self.authorizations = []

    async def start(self, instructions, cwd):
        self.starts += 1
        self.cwd = cwd

    async def decide(self, text, images, schema, request_id):
        self.inputs.append((text, images, request_id))
        item = next(self.outcomes)
        if isinstance(item, BaseException):
            raise item
        return item

    def recovery_status(self, request_id):
        return {
            "session_id": self.session_id,
            "request_id": request_id,
            "turn_id": "FAILED-TURN",
            "phase": "recovery_blocked",
        }

    async def authorize_recovery(self, request_id, turn_id, nonce, *, session_id):
        self.authorizations.append((session_id, request_id, turn_id, nonce))
        return {"authorized": True, "will_submit": True}

    async def close(self):
        self.closed = True


def planner_for(path, outcomes, *, maximum=5, wall=None, session_type=RecoverySession):
    sessions = []

    def factory(directory):
        session = session_type(directory, outcomes)
        sessions.append(session)
        return session

    planner = NativeBundlePlanner(output_dir=path, session_factory=factory)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=wall, program_calls=100, gateway_calls=100
        ),
    )
    robot = ActionRobot(path)
    return planner, robot, sessions, maximum


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def blocked_record(path):
    for _ in range(1000):
        if path.exists():
            value = json.loads(path.read_text())
            if value["status"] == "blocked":
                return value
        await asyncio.sleep(0.005)
    pytest.fail("Host did not publish blocked recovery")


def control(blocked, nonce=None):
    return {
        "schema_version": 1,
        "action": "resume",
        **{key: blocked[key] for key in ("session_id", "request_id", "turn_id")},
        "nonce": nonce or uuid.uuid4().hex,
    }


def test_actual_continuation_charges_again_but_transport_reconnect_does_not(
    tmp_path, monkeypatch
):
    from reweave.planner import native_loop

    original_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await original_sleep(0)

    monkeypatch.setattr(native_loop.asyncio, "sleep", fast_sleep)
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [
            NativeTurnRecoveryRequired({}, reason="transient", continuation=1),
            NativeTransportError("wire dropped"),
            DECISION,
        ],
    )
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=maximum
    )
    assert result.error is None
    assert result.stats["visible_turns"] == 2
    assert robot.calls == 1 and robot.terminal
    assert sessions[0].starts == 1 and sessions[0].closed
    assert len(sessions[0].inputs) == 3
    assert sessions[0].inputs[0] == sessions[0].inputs[1] == sessions[0].inputs[2]
    ledger = rows(tmp_path / "native/native_turns.jsonl")
    assert [row["status"] for row in ledger] == ["failed", "completed"]
    assert [row["error"] for row in ledger] == ["NativeTurnRecoveryRequired", None]
    assert [row["request_attempt"] for row in ledger] == [1, 2]
    assert {row["request_id"] for row in ledger} == {sessions[0].inputs[0][2]}


@pytest.mark.parametrize("compaction", [False, True])
def test_failed_native_provider_recovers_before_one_host_action(
    tmp_path, monkeypatch, compaction
):
    from reweave.planner import native_loop
    from reweave.planner.providers.codex import CodexDecisionSession
    from tests.unit_tests.harness.planner.test_native_codex_recovery import (
        BAD_REQUEST,
        STREAM_ERROR,
        fail_turn,
    )
    from tests.unit_tests.harness.planner.test_native_provider_codex import (
        FakeTransport,
        count,
    )

    original_sleep = asyncio.sleep

    async def yield_only(seconds):
        await original_sleep(0)

    monkeypatch.setattr(native_loop.asyncio, "sleep", yield_only)
    transport = FakeTransport(tmp_path / "native/workspace")
    starts = 0

    async def respond(params):
        nonlocal starts
        starts += 1
        if starts == 1:
            if compaction:
                turn = fail_turn(
                    transport,
                    params,
                    error={
                        "codexErrorInfo": "other",
                        "message": "Error running remote compact task: stream disconnected before completion: error sending request",
                    },
                    effects=[{"id": "compact", "type": "contextCompaction"}],
                )
                transport.emit(
                    "error",
                    {
                        "threadId": "thread-one",
                        "turnId": turn["id"],
                        "error": STREAM_ERROR,
                    },
                )
            else:
                turn = fail_turn(transport, params, error=BAD_REQUEST)
        else:
            turn = transport.make_turn(params, result=DECISION)
            transport.history.append(turn)
        return {"turn": turn}

    transport.on_turn_start = respond
    planner = NativeBundlePlanner(
        output_dir=tmp_path,
        session_factory=lambda directory: CodexDecisionSession(
            directory, transport=transport
        ),
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=None, program_calls=100, gateway_calls=100
        ),
    )
    robot = ActionRobot(tmp_path)
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=5
    )
    assert result.error is None
    assert robot.calls == 1 and robot.terminal
    assert count(transport, "thread/start") == 1
    assert count(transport, "thread/resume") == 1
    assert count(transport, "turn/start") == 2
    ledger = rows(tmp_path / "native/native_turns.jsonl")
    assert [row["status"] for row in ledger] == ["failed", "completed"]
    assert [row["request_attempt"] for row in ledger] == [1, 2]
    assert len({row["request_id"] for row in ledger}) == 1


@async_test
async def test_blocked_episode_waits_without_sampling_then_identity_bound_resume(
    tmp_path,
):
    planner, robot, sessions, maximum = planner_for(
        tmp_path, [NativeRecoveryBlocked({}, reason="limit", continuations=2), DECISION]
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    blocked = await blocked_record(path)
    session = sessions[0]
    assert not session.closed and robot.calls == 0
    # A parked episode is explicitly not a model turn in progress.
    assert blocked["state"] == "parked_operator_required"
    assert blocked["error_type"] == "NativeRecoveryBlocked"
    assert "provider_error" in blocked and blocked["parked_at"] > 0
    bad = control(blocked)
    bad["session_id"] = "FOREIGN-SID"
    atomic_json(path.with_name("resume.json"), bad)
    await asyncio.sleep(0.3)
    assert len(session.inputs) == 1 and not session.authorizations
    atomic_json(path.with_name("resume.json"), control(blocked))
    result = await asyncio.wait_for(task, 10)
    assert result.error is None and robot.calls == 1
    assert result.stats["visible_turns"] == 2 and session.starts == 1
    assert len(session.authorizations) == 1
    assert session.inputs[0] == session.inputs[1]
    assert json.loads(path.read_text())["status"] == "completed"
    assert any(
        row["kind"] == "native_recovery_control_rejected"
        for row in rows(tmp_path / "harness_context/bundle_events.jsonl")
    )


@async_test
async def test_completed_original_read_on_resume_is_not_charged_or_executed_twice(
    tmp_path,
):
    class Completed(RecoverySession):
        async def authorize_recovery(self, *args, **kwargs):
            self.authorizations.append(args)
            return {"authorized": False, "will_submit": False, "result": DECISION}

    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="read required", continuations=2)],
        session_type=Completed,
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    blocked = await blocked_record(path)
    atomic_json(path.with_name("resume.json"), control(blocked))
    result = await asyncio.wait_for(task, 10)
    assert result.error is None and robot.calls == 1
    assert result.stats["visible_turns"] == 1 and len(sessions[0].inputs) == 1
    assert len(rows(tmp_path / "native/native_turns.jsonl")) == 1


@async_test
async def test_nonce_is_consumed_once_even_if_next_turn_blocks_again(tmp_path):
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [
            NativeRecoveryBlocked({}, reason="first", continuations=2),
            NativeRecoveryBlocked({}, reason="second", continuations=3),
            DECISION,
        ],
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    first = await blocked_record(path)
    atomic_json(path.with_name("resume.json"), control(first))
    for _ in range(1000):
        value = json.loads(path.read_text())
        if value.get("reason") == "second":
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("Second failed turn was not parked")
    await asyncio.sleep(0.3)
    assert len(sessions[0].inputs) == 2 and len(sessions[0].authorizations) == 1
    atomic_json(path.with_name("resume.json"), control(value))
    result = await asyncio.wait_for(task, 10)
    assert result.error is None and robot.calls == 1
    assert result.stats["visible_turns"] == 3
    assert len(list((path.parent / "resume_receipts").glob("*.json"))) == 2


@async_test
async def test_unsafe_authorization_stays_blocked_without_new_sampling(tmp_path):
    class Unsafe(RecoverySession):
        async def authorize_recovery(self, *args, **kwargs):
            self.authorizations.append(args)
            raise NativeProtocolError("Original submission is not uniquely identified")

    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="unknown", continuations=2)],
        session_type=Unsafe,
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    blocked = await blocked_record(path)
    atomic_json(path.with_name("resume.json"), control(blocked))
    await asyncio.sleep(0.3)
    assert len(sessions[0].inputs) == 1 and robot.calls == 0 and not sessions[0].closed
    task.cancel()
    result = await asyncio.wait_for(task, 10)
    assert result.termination_class == "api_error" and sessions[0].closed
    assert result.api_error_retryable is False
    assert len(rows(tmp_path / "native/native_turns.jsonl")) == 1
    closed = json.loads(path.read_text())
    assert closed["status"] == "terminated" and closed["cancelled"] is True


def test_original_deadline_still_ends_parked_episode(tmp_path):
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="permanent", continuations=2)],
        wall=2.0,
    )
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=maximum
    )
    assert result.termination_class == "planner_timeout"
    assert robot.calls == 0 and sessions[0].closed
    assert (tmp_path / "native/recovery/blocked.json").exists()
    assert len(sessions[0].inputs) == 1 and result.stats["visible_turns"] == 1


def test_shared_cost_cap_still_terminates_without_granting_extra_turn(tmp_path):
    planner, robot, sessions, _ = planner_for(
        tmp_path,
        [NativeTurnRecoveryRequired({}, reason="transient", continuation=1), DECISION],
    )
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=1
    )
    assert result.termination_class == "api_error"
    assert result.stats["visible_turns"] == 1 and len(sessions[0].inputs) == 1
    assert robot.calls == 0 and sessions[0].closed
    assert not (tmp_path / "native/recovery/blocked.json").exists()


@async_test
async def test_start_failure_without_sid_parks_before_any_model_charge(tmp_path):
    from reweave.planner.providers.base import NativeAuthenticationError

    class StartBlocked(RecoverySession):
        def __init__(self, *args):
            super().__init__(*args)
            self.session_id = None

        async def start(self, instructions, cwd):
            self.starts += 1
            if self.starts == 1:
                raise NativeAuthenticationError("External account repair required")
            self.session_id, self.cwd = "SAME-SID", cwd

        def recovery_status(self, request_id):
            pytest.fail("No decision request exists before session start")

    planner, robot, sessions, maximum = planner_for(
        tmp_path, [DECISION], session_type=StartBlocked
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    blocked = await blocked_record(path)
    assert blocked["session_id"] is None and blocked["turn_id"] is None
    assert blocked["controller_attempts"] == 0
    atomic_json(path.with_name("resume.json"), control(blocked))
    result = await asyncio.wait_for(task, 10)
    assert result.error is None and robot.calls == 1
    assert sessions[0].starts == 2 and not sessions[0].authorizations
    assert result.stats["visible_turns"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "foreign_request",
        "foreign_turn",
        "extra",
        "bool_schema",
        "oversized",
        "symlink",
        "fifo",
    ],
)
def test_resume_control_rejects_invalid_or_nonregular_file(tmp_path, kind):
    import os

    from reweave.planner.native_loop import _read_recovery_control

    identity = {"session_id": "SID", "request_id": "request", "turn_id": "turn"}
    value = control(identity)
    if kind == "foreign_request":
        value["request_id"] = "old-request"
    elif kind == "foreign_turn":
        value["turn_id"] = "old-turn"
    elif kind == "extra":
        value["command"] = "forbidden"
    elif kind == "bool_schema":
        value["schema_version"] = True
    path = tmp_path / "resume.json"
    if kind == "symlink":
        target = tmp_path / "other.json"
        target.write_text(json.dumps(value))
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "oversized":
        path.write_text(" " * 16385)
    else:
        path.write_text(json.dumps(value))
    with pytest.raises((OSError, ValueError)):
        _read_recovery_control(path, identity)


def test_continuation_respects_actual_nested_workflow_model_cap(tmp_path):
    from reweave.evolution.bundle import AgentBundle
    from tests.unit_tests.harness.evolution.test_author_interaction import candidate

    value = candidate(
        'def run(runtime, context):\n    runtime.plan("Inspect current evidence")\n    return {"status":"completed","summary":"done","output":{}}\n',
        loop='def run(runtime, context):\n    runtime.execute(runtime.plan()["tool_calls"][0])\n    return {}\n',
    ).model_dump(mode="json")
    item = value["workflows"][0]
    item["permissions"], item["observations"] = ["model:plan"], []
    item["limits"].update(model_calls=1, runtime_calls=20, wall_time_s=None)
    root_decision = {
        "text": "Nested decision",
        "tool_calls": [
            {"name": "workflow_probe", "arguments_json": '{"arguments":{}}'}
        ],
    }
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [
            root_decision,
            NativeTurnRecoveryRequired({}, reason="transient", continuation=1),
            DECISION,
        ],
    )
    planner.bind_bundle(
        AgentBundle.model_validate(value),
        episode_limits=EpisodeLimits(
            wall_time_s=None, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task", user_message="Inspect", toolkit=robot, max_turns=maximum
    )
    assert result.termination_class == "agent_execution_failure"
    assert "workflow model budget exhausted" in result.error
    assert len(sessions[0].inputs) == 2 and result.stats["visible_turns"] == 2
    assert robot.calls == 0
    ledger = rows(tmp_path / "native/native_turns.jsonl")
    assert ledger[0]["request_id"] != ledger[1]["request_id"]


def test_recovery_policy_is_explicit_runtime_metadata(tmp_path):
    configuration = NativeBundlePlanner(output_dir=tmp_path).runtime_configuration()
    policy = configuration["native_recovery"]
    assert policy["revision"] == "same-session-recovery-v1"
    assert policy["automatic_failed_turn_continuations"] == 2
    assert policy["continuation_charges_controller_attempt"] is True
    assert policy["transport_reconnect_charges_controller_attempt"] is False
    assert policy["budget_and_deadline_reset"] is False


@async_test
async def test_authorization_transport_reconnect_does_not_charge_an_extra_turn(
    tmp_path,
):
    class LostAuthorization(RecoverySession):
        async def authorize_recovery(self, request_id, turn_id, nonce, *, session_id):
            self.authorizations.append((session_id, request_id, turn_id, nonce))
            if len(self.authorizations) == 1:
                raise NativeTransportError("Authorization reply disconnected")
            return {"authorized": False, "will_submit": True}

    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="limit", continuations=2), DECISION],
        session_type=LostAuthorization,
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    path = tmp_path / "native/recovery/blocked.json"
    blocked = await blocked_record(path)
    atomic_json(path.with_name("resume.json"), control(blocked))
    result = await asyncio.wait_for(task, 10)
    assert result.error is None and robot.calls == 1
    assert sessions[0].authorizations[0] == sessions[0].authorizations[1]
    assert result.stats["visible_turns"] == 2
    assert len(rows(tmp_path / "native/native_turns.jsonl")) == 2


def test_budget_request_identity_is_optional_and_distinct_from_service_ordinal(
    tmp_path,
):
    from reweave.runtime.model_budget import ModelCallBudget, ModelServiceBudgetExceeded

    evidence = tmp_path / "ledger.jsonl"
    budget = ModelCallBudget(2, evidence)
    with budget.request("sam"):
        pass
    with budget.request("controller", request_id="logical-request"):
        pass
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with budget.request("controller", request_id="logical-request"):
            raise RuntimeError("synthetic failure")
    with pytest.raises(ModelServiceBudgetExceeded):
        with budget.request("controller", request_id="logical-request"):
            pytest.fail("Shared cost cap was bypassed")
    ledger = rows(evidence)
    assert [row["request_id"] for row in ledger] == [
        None,
        "logical-request",
        "logical-request",
    ]
    assert [row["request_attempt"] for row in ledger] == [1, 2, 3]
    assert budget.controller_attempts == budget.maximum == 2


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [
        (NativeCompletedInvalidDecision("Invalid completed JSON"), None),
        (NativeProtocolError("Native tool is outside this protocol"), False),
    ],
)
def test_invalid_decision_closes_without_parking_or_replaying(
    tmp_path, failure, retryable
):
    planner, robot, sessions, _ = planner_for(tmp_path, [failure])
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=5
    )
    assert result.termination_class == (
        "agent_execution_failure"
        if isinstance(failure, NativeCompletedInvalidDecision)
        else "api_error"
    )
    assert result.api_error_retryable is retryable
    assert robot.calls == 0 and sessions[0].closed
    assert len(sessions[0].inputs) == 1
    assert not (tmp_path / "native/recovery/blocked.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        NativeTurnFailed("failed", {"message": "provider refusal"}),
        NativeRecoveryBlocked({}, reason="permanent failure", continuations=2),
        NativeProtocolError("Invalid completed proposal"),
    ],
)
def test_last_failed_controller_attempt_ends_at_cost_cap_instead_of_parking(
    tmp_path, failure
):
    planner, robot, sessions, _ = planner_for(tmp_path, [failure], wall=2.0)
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=1
    )
    assert result.termination_class == "api_error"
    assert result.api_error_retryable is False
    assert "cost cap exhausted" in result.error
    assert result.stats["visible_turns"] == 1 and len(sessions[0].inputs) == 1
    assert robot.calls == 0 and sessions[0].closed
    assert not (tmp_path / "native/recovery/blocked.json").exists()
    ledger = rows(tmp_path / "native/native_turns.jsonl")
    assert len(ledger) == 1 and ledger[0]["status"] == "failed"


@pytest.mark.parametrize("starting", [False, True])
@pytest.mark.parametrize("phase", ["failed", "recovery_blocked", "recovery_prepared"])
def test_known_failed_campaign_wait_expires_without_another_model_request(
    tmp_path, starting, phase
):
    from reweave.planner.providers.base import NativeAuthenticationError

    class Failed(RecoverySession):
        def recovery_status(self, request_id):
            return {**super().recovery_status(request_id), "phase": phase}

        async def start(self, instructions, cwd):
            if starting:
                raise NativeAuthenticationError("Account route unavailable")
            await super().start(instructions, cwd)

    atomic_json(tmp_path / "campaign_state.json", {})
    atomic_json(
        tmp_path / "scheduler/online/resource_policy.json",
        {"native_failed_wait_s": 0.02},
    )
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="failed", continuations=1)],
        session_type=Failed,
    )
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=maximum
    )
    assert (
        result.termination_class == "api_error" and result.api_error_retryable is True
    )
    assert "Known-failed" in result.error
    assert robot.calls == 0 and sessions[0].closed
    assert not sessions[0].authorizations
    assert len(sessions[0].inputs) == (0 if starting else 1)
    assert (
        json.loads((tmp_path / "native/recovery/blocked.json").read_text())["status"]
        == "terminated"
    )


@pytest.mark.parametrize(
    "phase,reason",
    [("submitted", None), ("recovery_blocked", "unsettled_native_effects")],
)
@async_test
async def test_unsettled_turn_is_never_expired_by_failed_wait_policy(
    tmp_path, phase, reason
):
    class Submitted(RecoverySession):
        def recovery_status(self, request_id):
            return {
                **super().recovery_status(request_id),
                "phase": phase,
                "reason": reason,
            }

    atomic_json(tmp_path / "campaign_state.json", {})
    atomic_json(
        tmp_path / "scheduler/online/resource_policy.json",
        {"native_failed_wait_s": 0.01},
    )
    planner, robot, sessions, maximum = planner_for(
        tmp_path,
        [NativeRecoveryBlocked({}, reason="uncertain", continuations=1)],
        session_type=Submitted,
    )
    task = asyncio.create_task(planner._solve("Task", "Move once", robot, maximum))
    blocked = await blocked_record(tmp_path / "native/recovery/blocked.json")
    assert blocked["infrastructure_wait_limit_s"] is None
    await asyncio.sleep(0.06)
    assert not task.done() and not sessions[0].authorizations and robot.calls == 0
    task.cancel()
    await task


def test_session_without_recovery_hooks_closes_failed_turn_without_replay(tmp_path):
    """A terminal Claude failure releases the episode without resubmission."""

    class NoRecoveryHooks:
        session_id = "CLAUDE-SID"
        context_generation = 0

        def __init__(self, directory, outcomes):
            self.inputs = []
            self.closed = False

        async def start(self, instructions, cwd):
            pass

        async def decide(self, text, images, schema, request_id):
            self.inputs.append(request_id)
            raise NativeTurnFailed("error", {"errors": ["overloaded"]})

        async def close(self):
            self.closed = True

    planner, robot, sessions, maximum = planner_for(
        tmp_path, [], wall=2.0, session_type=NoRecoveryHooks
    )
    result = planner.solve(
        system_prompt="Task", user_message="Move once", toolkit=robot, max_turns=maximum
    )
    assert not (tmp_path / "native/recovery/blocked.json").exists()
    assert "overloaded" in result.error
    assert robot.calls == 0 and len(sessions[0].inputs) == 1 and sessions[0].closed
    assert result.termination_class == "api_error"
    assert result.api_error_retryable is True
