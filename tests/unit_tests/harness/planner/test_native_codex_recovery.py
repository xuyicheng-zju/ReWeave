"""Same-SID failed-turn continuation is explicit, durable and separately budgeted."""

from __future__ import annotations

import asyncio
import copy

import pytest

from reweave.planner.providers import (
    CodexDecisionSession,
    NativeProtocolError,
    NativeRecoveryBlocked,
    NativeSubmissionUnknown,
    NativeTurnFailed,
    NativeTurnRecoveryRequired,
)
from tests.unit_tests.harness.planner.test_native_provider_codex import (
    DECISION,
    SCHEMA,
    FakeTransport,
    count,
    run,
    state,
)

STREAM_ERROR = {
    "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": None}},
    "message": "stream disconnected",
}
BAD_REQUEST = {"codexErrorInfo": "other", "message": '{"detail":"Bad Request"}'}


def fail_turn(transport, params, *, error=STREAM_ERROR, effects=()):
    turn = transport.make_turn(params, status="failed")
    turn["items"] = turn["items"][:1] + list(effects)
    turn["error"] = copy.deepcopy(error)
    transport.history.append(turn)
    return turn


@pytest.mark.parametrize(
    "error",
    [
        STREAM_ERROR,
        {"codexErrorInfo": "other", "message": "request timed out"},
        {
            "codexErrorInfo": "other",
            "message": "stream disconnected before completion: idle timeout waiting for SSE",
        },
    ],
)
def test_transport_failure_continues_same_sid_with_short_text_only_and_durable_mapping(
    tmp_path,
    error,
):
    async def exercise():
        image = tmp_path / "current.png"
        image.write_bytes(b"snapshot")
        directory = tmp_path / "provider"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0, result={"turn": fail_turn(transport, params, error=error)}
        )
        with pytest.raises(NativeTurnRecoveryRequired) as caught:
            await session.decide(
                "ORIGINAL OBSERVATION", (image,), SCHEMA, "host-request"
            )
        assert caught.value.continuation == 1
        assert session.recovery_status("host-request")["phase"] == "recovery_prepared"
        assert count(transport, "thread/read") == 1
        assert count(transport, "turn/start") == 1
        await session.close()
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = None
        assert (
            await session.decide(
                "ORIGINAL OBSERVATION", (image,), SCHEMA, "host-request"
            )
            == DECISION
        )
        starts = [
            params for method, params in transport.calls if method == "turn/start"
        ]
        assert len(starts) == 2
        assert starts[0]["threadId"] == starts[1]["threadId"] == "thread-one"
        assert starts[1]["input"] == [{"type": "text", "text": session.RECOVERY_TEXT}]
        assert starts[1]["outputSchema"] == SCHEMA
        assert starts[0]["clientUserMessageId"] != starts[1]["clientUserMessageId"]
        record = state(directory)["requests"]["host-request"]
        assert len(record["attempts"]) == 2
        assert (
            len(
                list(
                    (directory / "requests").glob("*/attempts/*/native_completion.json")
                )
            )
            == 2
        )
        assert (
            await session.decide(
                "ORIGINAL OBSERVATION", (image,), SCHEMA, "host-request"
            )
            == DECISION
        )
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


@pytest.mark.parametrize("stream_evidence", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_generic_bad_request_has_bounded_recovery_without_claiming_a_network_cause(
    tmp_path, stream_evidence, compact
):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def failure(params):
            error = copy.deepcopy(BAD_REQUEST)
            if compact:
                error["message"] = (
                    "Error running remote compact task: " + error["message"]
                )
            turn = fail_turn(transport, params, error=error)
            if stream_evidence:
                transport.emit(
                    "error",
                    {
                        "threadId": "thread-one",
                        "turnId": turn["id"],
                        "error": STREAM_ERROR,
                        "willRetry": True,
                    },
                )
            return {"turn": turn}

        transport.on_turn_start = failure
        with pytest.raises(NativeTurnRecoveryRequired) as caught:
            await session.decide("obs", (), SCHEMA, "request")
        assert caught.value.reason == (
            "stream_failure_then_generic_bad_request_root_cause_unknown"
            if stream_evidence
            else "generic_bad_request_root_cause_unknown"
        )
        assert count(transport, "thread/read") == 1
        assert count(transport, "turn/start") == 1
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        starts = [
            params for method, params in transport.calls if method == "turn/start"
        ]
        assert len(starts) == 2
        assert starts[0]["threadId"] == starts[1]["threadId"]
        assert starts[1]["input"] == [{"type": "text", "text": session.RECOVERY_TEXT}]
        assert transport.connections == 2
        assert count(transport, "thread/start") == 1
        assert count(transport, "thread/resume") == 1
        await session.close()

    run(exercise())


def test_persistent_generic_bad_request_parks_after_one_continuation(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0, result={"turn": fail_turn(transport, params, error=BAD_REQUEST)}
        )
        with pytest.raises(NativeTurnRecoveryRequired):
            await session.decide("obs", (), SCHEMA, "request")
        with pytest.raises(NativeRecoveryBlocked) as caught:
            await session.decide("obs", (), SCHEMA, "request")
        assert caught.value.reason == "continuations_exhausted"
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


@pytest.mark.parametrize("stream_evidence", [False, True])
def test_wrapped_compaction_disconnect_uses_same_turn_stream_evidence(
    tmp_path, stream_evidence
):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def failure(params):
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
                    "turnId": turn["id"] if stream_evidence else "earlier-turn",
                    "error": STREAM_ERROR,
                    "willRetry": True,
                },
            )
            return {"turn": turn}

        transport.on_turn_start = failure
        expected = NativeTurnRecoveryRequired if stream_evidence else NativeTurnFailed
        with pytest.raises(expected):
            await session.decide("obs", (), SCHEMA, "request")
        if stream_evidence:
            transport.on_turn_start = None
            assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        else:
            assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


@pytest.mark.parametrize("kind", ["serverOverloaded", "internalServerError"])
def test_known_service_failure_can_continue_same_session(tmp_path, kind):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0,
            result={
                "turn": fail_turn(transport, params, error={"codexErrorInfo": kind})
            },
        )
        with pytest.raises(NativeTurnRecoveryRequired):
            await session.decide("obs", (), SCHEMA, "request")
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        await session.close()

    run(exercise())


@pytest.mark.parametrize(
    "error",
    [
        BAD_REQUEST,
        STREAM_ERROR,
        {"codexErrorInfo": "other", "message": "request timed out"},
    ],
)
def test_unsettled_native_command_blocks_automatic_and_manual_continuation(
    tmp_path, error
):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def failure(params):
            turn = fail_turn(
                transport,
                params,
                error=error,
                effects=[
                    {
                        "type": "commandExecution",
                        "id": "command",
                        "status": "inProgress",
                    }
                ],
            )
            return {"turn": turn}

        transport.on_turn_start = failure
        for _ in range(2):
            with pytest.raises(NativeRecoveryBlocked, match="failed") as caught:
                await session.decide("obs", (), SCHEMA, "request")
            assert caught.value.reason == "unsettled_native_effects"
        with pytest.raises(NativeRecoveryBlocked):
            await session.authorize_recovery(
                "request", "turn-1", "nonce", session_id="thread-one"
            )
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_failed_notification_uses_completed_original_without_continuation(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def delayed(params):
            original = transport.make_turn(params)
            transport.history.append(original)
            notification = copy.deepcopy(original)
            notification.update(
                status="failed", error=STREAM_ERROR, items=[], itemsView="notLoaded"
            )
            return {"turn": notification}

        transport.on_turn_start = delayed
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_missing_original_submission_never_creates_replacement(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def missing(params):
            return {
                "turn": {
                    "id": "lost",
                    "status": "failed",
                    "error": STREAM_ERROR,
                    "items": [],
                }
            }

        transport.on_turn_start = missing
        with pytest.raises(NativeSubmissionUnknown):
            await session.decide("obs", (), SCHEMA, "request")
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_exhaustion_parks_and_exact_nonce_authorizes_only_one_extra_turn(tmp_path):
    async def exercise():
        directory = tmp_path / "provider"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0, result={"turn": fail_turn(transport, params)}
        )
        for continuation in (1, 2):
            with pytest.raises(NativeTurnRecoveryRequired) as caught:
                await session.decide("obs", (), SCHEMA, "request")
            assert caught.value.continuation == continuation
        for _ in range(2):
            with pytest.raises(NativeRecoveryBlocked) as caught:
                await session.decide("obs", (), SCHEMA, "request")
            assert caught.value.continuations == 2
        assert count(transport, "turn/start") == 3
        for sid, turn in (("foreign", "turn-3"), ("thread-one", "turn-2")):
            with pytest.raises(NativeProtocolError, match="identity"):
                await session.authorize_recovery(
                    "request", turn, "token", session_id=sid
                )
        authorized = await session.authorize_recovery(
            "request", "turn-3", "token", session_id="thread-one"
        )
        assert authorized["authorized"] and authorized["will_submit"]
        duplicate = await session.authorize_recovery(
            "request", "turn-3", "token", session_id="thread-one"
        )
        assert not duplicate["authorized"] and duplicate["will_submit"]
        with pytest.raises(NativeRecoveryBlocked) as caught:
            await session.decide("obs", (), SCHEMA, "request")
        assert caught.value.continuations == 3
        assert count(transport, "turn/start") == 4
        assert state(directory)["requests"]["request"]["phase"] == "recovery_blocked"
        await session.close()

    run(exercise())


def test_auth_failure_requires_explicit_repair_authorization(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0,
            result={
                "turn": fail_turn(
                    transport,
                    params,
                    error={
                        "codexErrorInfo": "unauthorized",
                        "message": "authentication required",
                    },
                )
            },
        )
        with pytest.raises(NativeTurnFailed) as caught:
            await session.decide("obs", (), SCHEMA, "request")
        assert not isinstance(caught.value, NativeTurnRecoveryRequired)
        assert count(transport, "turn/start") == 1
        authorized = await session.authorize_recovery(
            "request", "turn-1", "external-repair", session_id="thread-one"
        )
        assert authorized["will_submit"]
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


@pytest.mark.parametrize("client_id", [None, "another-input"])
def test_short_continuation_requires_current_user_message_in_history(
    tmp_path, client_id
):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def missing_input(params):
            turn = fail_turn(transport, params)
            if client_id is None:
                turn["items"] = []
            else:
                turn["items"][0]["clientId"] = client_id
            return {"turn": turn}

        transport.on_turn_start = missing_input
        with pytest.raises(NativeRecoveryBlocked) as caught:
            await session.decide("current observation", (), SCHEMA, "request")
        assert caught.value.reason == "missing_current_input"
        with pytest.raises(NativeRecoveryBlocked):
            await session.authorize_recovery(
                "request", "turn-1", "token", session_id="thread-one"
            )
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_verified_original_in_progress_waits_without_new_model_turn(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)

        async def still_running(params):
            original = transport.make_turn(params, status="inProgress")
            transport.history.append(original)
            terminal = copy.deepcopy(original)
            terminal["status"] = "completed"
            transport.emit(
                "turn/completed", {"threadId": "thread-one", "turn": terminal}
            )
            stale = copy.deepcopy(original)
            stale.update(status="failed", error=STREAM_ERROR)
            return {"turn": stale}

        transport.on_turn_start = still_running
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_completed_native_command_edits_are_preserved_without_replay(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        edited = tmp_path / "task-family-note.txt"

        async def settled(params):
            edited.write_text("completed native edit")
            return {
                "turn": fail_turn(
                    transport,
                    params,
                    effects=[
                        {
                            "type": "commandExecution",
                            "id": "command",
                            "status": "completed",
                        }
                    ],
                )
            }

        transport.on_turn_start = settled
        with pytest.raises(NativeTurnRecoveryRequired):
            await session.decide("obs", (), SCHEMA, "request")
        assert edited.read_text() == "completed native edit"
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert edited.read_text() == "completed native edit"
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


def test_blocked_state_survives_reopen_without_automatic_sampling(tmp_path):
    async def exercise():
        directory = tmp_path / "provider"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0, result={"turn": fail_turn(transport, params)}
        )
        for _ in range(2):
            with pytest.raises(NativeTurnRecoveryRequired):
                await session.decide("obs", (), SCHEMA, "request")
        with pytest.raises(NativeRecoveryBlocked):
            await session.decide("obs", (), SCHEMA, "request")
        await session.close()
        resumed = CodexDecisionSession(directory, transport=transport)
        await resumed.start("contract", tmp_path)
        with pytest.raises(NativeRecoveryBlocked):
            await resumed.decide("obs", (), SCHEMA, "request")
        assert resumed.session_id == "thread-one"
        assert count(transport, "turn/start") == 3
        await resumed.close()

    run(exercise())


def test_lost_authorization_reply_keeps_pending_turn_budget_requirement(
    tmp_path, monkeypatch
):
    async def exercise():
        from reweave.planner.providers import NativeTransportError

        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        transport.on_turn_start = lambda params: asyncio.sleep(
            0,
            result={
                "turn": fail_turn(
                    transport, params, error={"codexErrorInfo": "unauthorized"}
                )
            },
        )
        with pytest.raises(NativeTurnFailed):
            await session.decide("obs", (), SCHEMA, "request")
        original = session._record

        def lost_reply(kind, **values):
            original(kind, **values)
            if kind == "decision_recovery_authorized":
                raise NativeTransportError("reply lost after durable authorization")

        monkeypatch.setattr(session, "_record", lost_reply)
        with pytest.raises(NativeTransportError):
            await session.authorize_recovery(
                "request", "turn-1", "token", session_id="thread-one"
            )
        receipt = await session.authorize_recovery(
            "request", "turn-1", "token", session_id="thread-one"
        )
        assert not receipt["authorized"] and receipt["will_submit"]
        assert count(transport, "turn/start") == 1
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


def test_http_disconnect_during_compaction_recovers_automatically_in_same_session(
    tmp_path,
):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "provider", transport=transport)
        await session.start("contract", tmp_path)
        error = {
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": None}},
            "message": "Error running remote compact task: Connection failed",
        }
        transport.on_turn_start = lambda params: asyncio.sleep(
            0,
            result={
                "turn": fail_turn(
                    transport,
                    params,
                    error=error,
                    effects=[{"id": "compact", "type": "contextCompaction"}],
                )
            },
        )
        with pytest.raises(NativeTurnRecoveryRequired):
            await session.decide("observation", (), SCHEMA, "request")
        transport.on_turn_start = None
        assert await session.decide("observation", (), SCHEMA, "request") == DECISION
        assert count(transport, "thread/start") == 1
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())


def test_interrupted_turn_requires_explicit_identity_bound_budgeted_continuation(
    tmp_path,
):
    async def exercise():
        directory = tmp_path / "provider"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("contract", tmp_path)

        async def interrupted(params):
            turn = fail_turn(transport, params, error=None)
            turn["status"] = "interrupted"
            return {"turn": turn}

        transport.on_turn_start = interrupted
        with pytest.raises(NativeTurnFailed):
            await session.decide("retained observation", (), SCHEMA, "request")
        assert count(transport, "turn/start") == 1
        assert session.recovery_status("request")["phase"] == "failed"
        receipt = await session.authorize_recovery(
            "request", "turn-1", "explicit-resume", session_id="thread-one"
        )
        assert receipt["authorized"] and receipt["will_submit"]
        assert count(transport, "turn/start") == 1
        duplicate = await session.authorize_recovery(
            "request", "turn-1", "explicit-resume", session_id="thread-one"
        )
        assert not duplicate["authorized"]
        transport.on_turn_start = None
        assert (
            await session.decide("retained observation", (), SCHEMA, "request")
            == DECISION
        )
        record = state(directory)["requests"]["request"]
        assert record["continuations"] == 1
        assert len(record["attempts"]) == 2
        assert count(transport, "turn/start") == 2
        assert (
            await session.decide("retained observation", (), SCHEMA, "request")
            == DECISION
        )
        assert count(transport, "turn/start") == 2
        await session.close()

    run(exercise())
