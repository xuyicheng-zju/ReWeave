from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from reweave.planner.providers import (
    CodexDecisionSession,
    NativeAuthenticationError,
    NativeProtocolError,
    NativeRpcError,
    NativeSessionError,
    NativeSubmissionUnknown,
    NativeTransportError,
    NativeTurnFailed,
)

SCHEMA = {
    "type": "object",
    "properties": {"tool_calls": {"type": "array"}},
    "required": ["tool_calls"],
    "additionalProperties": False,
}
DECISION = {"tool_calls": [{"name": "move_pose", "arguments": {"arm": "left"}}]}


class FakeTransport:
    def __init__(
        self, cwd: Path, *, account="chatgpt", auto_complete=True, model="gpt-5.4"
    ):
        self.cwd = cwd
        self.model = model
        self.account = account
        self.auto_complete = auto_complete
        self.calls = []
        self.events = asyncio.Queue()
        self.history = []
        self.observer = None
        self.connections = 0
        self.closes = 0
        self.on_turn_start = None
        self.start_response = None
        self.turn_started = asyncio.Event()

    @property
    def metadata(self):
        return {"protocol": "fake-jsonrpc", "version": "codex-cli 0.162.1"}

    def set_observer(self, observer):
        self.observer = observer

    async def connect(self, cwd):
        assert cwd == self.cwd
        self.connections += 1
        self.events = asyncio.Queue()

    def observe(self, direction, message):
        self.observer({"direction": direction, "message": copy.deepcopy(message)})

    def emit(self, method, params):
        event = {"jsonrpc": "2.0", "method": method, "params": params}
        self.observe("incoming", event)
        self.events.put_nowait(event)

    async def notify(self, method, params):
        self.observe("outgoing", {"jsonrpc": "2.0", "method": method, "params": params})

    def thread_response(self):
        return {
            "thread": {"id": "thread-one"},
            "model": self.model,
            "modelProvider": "openai",
            "cwd": str(self.cwd),
        }

    def make_turn(self, params, *, status="completed", result=DECISION):
        return {
            "id": f"turn-{len(self.history) + 1}",
            "status": status,
            "itemsView": "full",
            "items": [
                {
                    "type": "userMessage",
                    "id": "user-item",
                    "clientId": params["clientUserMessageId"],
                    "content": params["input"],
                },
                {
                    "type": "agentMessage",
                    "id": "final-item",
                    "phase": "final_answer",
                    "text": json.dumps(result),
                },
            ],
        }

    async def request(self, method, params):
        self.calls.append((method, copy.deepcopy(params)))
        identity = len(self.calls)
        self.observe(
            "outgoing",
            {"jsonrpc": "2.0", "id": identity, "method": method, "params": params},
        )
        if method == "initialize":
            response = {"userAgent": "codex-cli/0.162.1"}
        elif method == "account/read":
            response = {
                "account": None if self.account is None else {"type": self.account},
                "requiresOpenaiAuth": True,
            }
        elif method == "skills/list":
            response = {"data": [{"cwd": str(self.cwd), "skills": [], "errors": []}]}
        elif method == "thread/start":
            response = self.start_response or self.thread_response()
        elif method == "thread/resume":
            response = self.thread_response()
        elif method == "thread/read":
            response = {"thread": {"id": "thread-one"}}
        elif method == "thread/turns/list":
            response = {"data": copy.deepcopy(self.history), "nextCursor": None}
        elif method == "turn/start":
            self.turn_started.set()
            if self.on_turn_start is not None:
                response = await self.on_turn_start(params)
            else:
                turn = self.make_turn(params)
                self.history.append(turn)
                response = {
                    "turn": {"id": turn["id"], "status": "inProgress", "items": []}
                }
                if self.auto_complete:
                    self.emit(
                        "thread/tokenUsage/updated",
                        {
                            "threadId": "thread-one",
                            "turnId": turn["id"],
                            "tokenUsage": {"total": {"totalTokens": 17}},
                        },
                    )
                    self.emit(
                        "thread/compacted",
                        {"threadId": "thread-one", "turnId": turn["id"]},
                    )
                    self.emit(
                        "turn/completed", {"threadId": "thread-one", "turn": turn}
                    )
        elif method == "turn/interrupt":
            turn = next(turn for turn in self.history if turn["id"] == params["turnId"])
            turn.update(status="interrupted", error={"message": "cancelled"})
            self.emit("turn/completed", {"threadId": "thread-one", "turn": turn})
            response = {}
        else:
            raise AssertionError(f"Unexpected RPC {method}")
        self.observe("incoming", {"jsonrpc": "2.0", "id": identity, "result": response})
        return copy.deepcopy(response)

    async def next_event(self):
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self):
        self.closes += 1
        self.events.put_nowait(NativeTransportError("fake connection closed"))


def run(coroutine):
    return asyncio.run(coroutine)


def state(directory):
    return json.loads((directory / "session.json").read_text())


def count(transport, method):
    return sum(call[0] == method for call in transport.calls)


def test_experiment_fast_policy_applies_only_to_new_turns_without_replay(tmp_path):
    async def exercise():
        directory = tmp_path / "online/native"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("fixed instructions", tmp_path)
        await session.decide("first", (), SCHEMA, "one")
        policy = tmp_path / "native_request_policy.json"
        policy.write_text(
            json.dumps(
                {
                    "scope_root": str(tmp_path),
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "service_tier": "fast",
                    "operator_authorization": "explicit operator choice",
                }
            )
        )
        await session.decide("first", (), SCHEMA, "one")
        await session.decide("second", (), SCHEMA, "two")
        turns = [p for m, p in transport.calls if m == "turn/start"]
        assert [p["serviceTier"] for p in turns] == ["default", "fast"]
        assert count(transport, "thread/start") == 1
        routing = state(directory)["requests"]["two"]["service_tier_routing"]
        assert routing["configured_service_tier"] == "default"
        assert routing["policy_sha256"]
        await session.close()

    asyncio.run(exercise())


def test_fast_policy_cannot_change_a_frozen_model(tmp_path):
    policy = tmp_path / "native_request_policy.json"
    policy.write_text(
        json.dumps(
            {
                "scope_root": str(tmp_path),
                "model": "different-model",
                "reasoning_effort": "medium",
                "service_tier": "fast",
                "operator_authorization": "operator",
            }
        )
    )
    session = CodexDecisionSession(
        tmp_path / "native", transport=FakeTransport(tmp_path)
    )
    with pytest.raises(NativeProtocolError, match="frozen model"):
        session._request_service_tier()


def test_one_thread_multiple_current_observations_and_native_images(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        image = tmp_path / "head.png"
        image.write_bytes(b"immutable-image")
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("Published C instructions", tmp_path)
        first = await session.decide("obs-1 initial", (image,), SCHEMA, "request-one")
        second = await session.decide("obs-2 tool ACK", (image,), SCHEMA, "request-two")
        assert first == second == DECISION
        assert session.session_id == "thread-one"
        assert count(transport, "thread/start") == 1
        turns = [params for method, params in transport.calls if method == "turn/start"]
        assert [turn["threadId"] for turn in turns] == ["thread-one", "thread-one"]
        assert turns[1]["input"][0] == {"type": "text", "text": "obs-2 tool ACK"}
        assert turns[1]["input"][1]["type"] == "localImage"
        snapshot = Path(turns[1]["input"][1]["path"])
        assert snapshot.is_relative_to(directory / "launcher/images")
        assert snapshot.read_bytes() == image.read_bytes()
        assert snapshot != image
        archived = next((directory / "requests").glob("*/images/0.png"))
        assert snapshot.stat().st_ino != image.stat().st_ino
        assert snapshot.read_bytes() == archived.read_bytes()
        assert "obs-1" not in json.dumps(turns[1])
        assert all(turn["outputSchema"] == SCHEMA for turn in turns)
        assert all(turn["effort"] == "medium" for turn in turns)
        assert all(turn["serviceTier"] == "default" for turn in turns)
        assert all(
            "input_hash" not in request
            for request in state(directory)["requests"].values()
        )
        start = next(
            params for method, params in transport.calls if method == "thread/start"
        )
        assert start["model"] == "gpt-5.4"
        assert start["modelProvider"] == "openai"
        assert start["developerInstructions"] == "Published C instructions"
        assert start["baseInstructions"] == CodexDecisionSession.BASE_INSTRUCTIONS
        assert start["config"]["project_doc_max_bytes"] == 0
        assert start["config"]["skills.max_context_tokens"] == 1
        assert start["allowProviderModelFallback"] is False
        assert start["dynamicTools"] == []
        assert "environments" not in start
        assert state(directory)["token_usage"] == {"total": {"totalTokens": 17}}
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        assert any(
            event.get("message", {}).get("method") == "thread/compacted"
            for event in events
        )
        assert {method for method, _ in transport.calls} <= {
            "initialize",
            "account/read",
            "skills/list",
            "thread/start",
            "turn/start",
        }
        await session.close()

    run(exercise())


def test_context_generation_tracks_only_this_sessions_incoming_compaction(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)
        assert session.context_generation == 0
        transport.emit("thread/compacted", {"threadId": "unrelated-thread"})
        assert session.context_generation == 0
        transport.observe(
            "outgoing",
            {"method": "thread/compacted", "params": {"threadId": session.session_id}},
        )
        assert session.context_generation == 0
        transport.emit("thread/compacted", {"threadId": session.session_id})
        assert session.context_generation == 1
        await session.decide("current observation", (), SCHEMA, "one")
        assert session.context_generation == 2
        assert count(transport, "turn/start") == 1
        await session.close()
        resumed = CodexDecisionSession(
            tmp_path / "native", transport=FakeTransport(tmp_path)
        )
        assert resumed.session_id == "thread-one"
        assert resumed.context_generation == 0
        await resumed.close()

    run(exercise())


def test_fast_and_reasoning_are_sent_independently_on_start_resume_and_turn(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        first = FakeTransport(tmp_path)
        session = CodexDecisionSession(
            directory, transport=first, reasoning_effort="medium", service_tier="fast"
        )
        await session.start("instructions", tmp_path)
        await session.decide("obs-1", (), SCHEMA, "one")
        await session.close()
        second = FakeTransport(tmp_path)
        resumed = CodexDecisionSession(
            directory, transport=second, reasoning_effort="medium", service_tier="fast"
        )
        await resumed.start("instructions", tmp_path)
        await resumed.decide("obs-2", (), SCHEMA, "two")
        for transport, start_method in (
            (first, "thread/start"),
            (second, "thread/resume"),
        ):
            start = next(
                params for method, params in transport.calls if method == start_method
            )
            assert start["serviceTier"] == "fast"
            assert start["baseInstructions"] == CodexDecisionSession.BASE_INSTRUCTIONS
            assert start["developerInstructions"] == "instructions"
            assert start["config"] == {
                "model_context_window": 500_000,
                "features.fast_mode": True,
                "project_doc_max_bytes": 0,
                "project_root_markers": ["AGENTS.md"],
                "skills.max_context_tokens": 1,
                "skills.config": [],
            }
            turn = next(
                params for method, params in transport.calls if method == "turn/start"
            )
            assert (turn["effort"], turn["serviceTier"]) == ("medium", "fast")
        assert resumed.session_id == session.session_id
        await resumed.close()

    run(exercise())


def test_native_skill_catalog_budget_keeps_explicit_task_experience_and_file_access(
    tmp_path,
):
    async def exercise():
        instructions = (
            "Task-family experience: measure the actual grasp before tilting."
        )
        current = "C selected skill: return upright immediately after the short pulse."
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start(instructions, tmp_path)
        await session.decide(current, (), SCHEMA, "one")
        start = next(
            params for method, params in transport.calls if method == "thread/start"
        )
        turn = next(
            params for method, params in transport.calls if method == "turn/start"
        )
        assert start["developerInstructions"] == instructions
        assert turn["input"] == [{"type": "text", "text": current}]
        assert start["config"]["skills.max_context_tokens"] == 1
        assert "environments" not in start
        assert "features.shell_tool" not in start["config"]
        assert "features.unified_exec" not in start["config"]
        assert start["cwd"] == str(tmp_path)
        assert start["sandbox"] == "workspace-write"
        await session.close()

    run(exercise())


def test_selected_astra_model_is_durable_on_start_resume_and_each_turn(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        first = FakeTransport(tmp_path, model="gpt-5.4")
        session = CodexDecisionSession(
            directory,
            transport=first,
            model="gpt-5.4",
            reasoning_effort="low",
            service_tier="default",
        )
        await session.start("instructions", tmp_path)
        await session.decide("obs-1", (), SCHEMA, "one")
        await session.close()
        second = FakeTransport(tmp_path, model="gpt-5.4")
        resumed = CodexDecisionSession(
            directory,
            transport=second,
            model="gpt-5.4",
            reasoning_effort="low",
            service_tier="default",
        )
        await resumed.start("instructions", tmp_path)
        await resumed.decide("obs-2", (), SCHEMA, "two")
        assert resumed.session_id == session.session_id
        assert state(directory)["model"] == "gpt-5.4"
        for transport, operation in (
            (first, "thread/start"),
            (second, "thread/resume"),
        ):
            thread = next(
                params for method, params in transport.calls if method == operation
            )
            turn = next(
                params for method, params in transport.calls if method == "turn/start"
            )
            assert thread["model"] == turn["model"] == "gpt-5.4"
            assert thread["serviceTier"] == turn["serviceTier"] == "default"
            assert thread["config"]["features.fast_mode"] is False
            assert turn["effort"] == "low"
        await resumed.close()

    run(exercise())


def test_historical_session_cannot_silently_change_selected_model(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        original = CodexDecisionSession(directory, transport=FakeTransport(tmp_path))
        stale_transport = FakeTransport(tmp_path, model="different-model")
        stale = CodexDecisionSession(
            directory, transport=stale_transport, model="different-model"
        )
        await original.start("instructions", tmp_path)
        await original.decide("obs", (), SCHEMA, "one")
        await original.close()
        historical = (directory / "session.json").read_bytes()
        with pytest.raises(NativeProtocolError, match="selected model"):
            CodexDecisionSession(directory, model="different-model")
        with pytest.raises(NativeProtocolError, match="selected model"):
            await stale.start("instructions", tmp_path)
        assert not stale_transport.calls
        assert stale._lease is None
        assert (directory / "session.json").read_bytes() == historical
        compatible = CodexDecisionSession(directory, transport=FakeTransport(tmp_path))
        await compatible.start("instructions", tmp_path)
        assert compatible.session_id == original.session_id
        await compatible.close()
        await stale.close()

    run(exercise())


@pytest.mark.parametrize("model", ["", " ", " gpt-5.4", 1, False])
def test_invalid_explicit_model_does_not_select_the_default(tmp_path, model):
    with pytest.raises(ValueError, match="Native model"):
        CodexDecisionSession(tmp_path / "native", model=model)


@pytest.mark.parametrize(
    "settings", [{"reasoning_effort": "fast"}, {"service_tier": "medium"}]
)
def test_invalid_native_reasoning_and_speed_fail_before_connect(tmp_path, settings):
    with pytest.raises(ValueError, match="Unsupported native"):
        CodexDecisionSession(tmp_path / "native", **settings)


def test_cached_decision_is_repeatable_after_restart_and_does_not_repeat_query(
    tmp_path,
):
    async def exercise():
        directory = tmp_path / "native"
        first = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=first)
        # Constructing the second owner early must not retain stale session state.
        second = FakeTransport(tmp_path)
        restored = CodexDecisionSession(directory, transport=second)
        await session.start("instructions", tmp_path)
        result = await session.decide("obs", (), SCHEMA, "same-request")
        result["tool_calls"].clear()
        assert await session.decide("obs", (), SCHEMA, "same-request") == DECISION
        await session.close()
        await restored.start("instructions", tmp_path)
        assert await restored.decide("obs", (), SCHEMA, "same-request") == DECISION
        assert count(first, "turn/start") == 1
        assert count(second, "thread/start") == count(second, "turn/start") == 0
        assert count(second, "thread/resume") == 1
        await restored.close()

    run(exercise())


def test_default_local_environment_is_not_disabled_on_start_turn_or_resume(tmp_path):
    async def exercise():
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        directory = tmp_path / "native"
        first = FakeTransport(candidate)
        session = CodexDecisionSession(directory, transport=first)
        await session.start("Edit the private candidate with native tools", candidate)
        await session.decide("first observation", (), SCHEMA, "first")
        identity = session.session_id
        await session.close()

        second = FakeTransport(candidate)
        resumed = CodexDecisionSession(directory, transport=second)
        await resumed.start("Edit the private candidate with native tools", candidate)
        await resumed.decide("settled feedback", (), SCHEMA, "second")
        assert resumed.session_id == identity
        assert count(first, "thread/start") == 1
        assert count(second, "thread/start") == 0
        assert count(second, "thread/resume") == 1
        assert count(first, "turn/start") == count(second, "turn/start") == 1
        for method, params in [*first.calls, *second.calls]:
            if method in {"thread/start", "thread/resume", "turn/start"}:
                # Installed Codex selects its default local environment when the
                # field is omitted; [] explicitly removes native read/edit/exec.
                assert "environments" not in params
            if method in {"thread/start", "thread/resume"}:
                assert params["cwd"] == str(candidate)
                assert params["sandbox"] == "workspace-write"
                assert params["approvalPolicy"] == "never"
        await resumed.close()

    run(exercise())


@pytest.mark.parametrize("changed", ["text", "schema", "image"])
def test_same_id_with_changed_input_is_rejected_without_new_query(tmp_path, changed):
    async def exercise():
        image = tmp_path / "image.png"
        image.write_bytes(b"first-image")
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)
        await session.decide("obs", (image,), SCHEMA, "same-request")
        text = "changed obs" if changed == "text" else "obs"
        schema = {"type": "object"} if changed == "schema" else SCHEMA
        if changed == "image":
            image.write_bytes(b"different-image")
        with pytest.raises(NativeProtocolError, match="different input"):
            await session.decide(text, (image,), schema, "same-request")
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


@pytest.mark.parametrize("account", [None, "apiKey", "amazonBedrock"])
def test_missing_chatgpt_login_never_falls_back(tmp_path, account):
    async def exercise():
        transport = FakeTransport(tmp_path, account=account)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        with pytest.raises(NativeAuthenticationError):
            await session.start("instructions", tmp_path)
        assert count(transport, "thread/start") == 0
        assert all(
            method not in {"account/login/start", "turn/start"}
            for method, _ in transport.calls
        )
        assert next(
            params for method, params in transport.calls if method == "account/read"
        ) == {"refreshToken": False}
        await session.close()

    run(exercise())


def test_response_lost_after_submission_recovers_by_client_id_without_resend(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("instructions", tmp_path)

        async def lose_response(params):
            transport.history.append(transport.make_turn(params))
            raise NativeTransportError("response lost after actual submission")

        transport.on_turn_start = lose_response
        with pytest.raises(NativeTransportError):
            await session.decide("obs", (), SCHEMA, "request")
        assert state(directory)["requests"]["request"]["phase"] == "submitting"
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 1
        assert count(transport, "thread/resume") == 1
        assert state(directory)["event_stream_complete"] is False
        await session.close()

    run(exercise())


def test_known_turn_completed_during_event_disconnect_is_recovered(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        transport = FakeTransport(tmp_path, auto_complete=False)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("instructions", tmp_path)
        pending = asyncio.create_task(session.decide("obs", (), SCHEMA, "request"))
        await transport.turn_started.wait()
        transport.events.put_nowait(NativeTransportError("event connection lost"))
        with pytest.raises(NativeTransportError):
            await pending
        assert state(directory)["requests"]["request"]["turn_id"] == "turn-1"
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_unresolved_submission_blocks_resend_and_later_request(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)

        async def ambiguous(params):
            raise NativeTransportError("submission outcome unknown")

        transport.on_turn_start = ambiguous
        with pytest.raises(NativeTransportError):
            await session.decide("obs", (), SCHEMA, "request")
        with pytest.raises(NativeSubmissionUnknown, match="uniquely"):
            await session.decide("obs", (), SCHEMA, "request")
        with pytest.raises(NativeSubmissionUnknown, match="earlier"):
            await session.decide("next obs", (), SCHEMA, "another-request")
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_confirmed_not_sent_request_can_be_submitted_once_on_resume(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("instructions", tmp_path)

        async def not_sent(params):
            raise NativeTransportError("failed before write", request_sent=False)

        transport.on_turn_start = not_sent
        with pytest.raises(NativeTransportError):
            await session.decide("obs", (), SCHEMA, "request")
        assert state(directory)["requests"]["request"]["phase"] == "prepared"
        transport.on_turn_start = None
        assert await session.decide("obs", (), SCHEMA, "request") == DECISION
        assert len(transport.history) == 1
        await session.close()

    run(exercise())


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_native_failure_is_retained_and_never_requeried(tmp_path, status):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)

        async def fail(params):
            turn = transport.make_turn(params, status=status)
            turn["error"] = {
                "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 400}},
                "message": "original failure",
            }
            transport.history.append(turn)
            return {"turn": turn}

        transport.on_turn_start = fail
        for _ in range(2):
            with pytest.raises(NativeTurnFailed) as caught:
                await session.decide("obs", (), SCHEMA, "request")
            assert caught.value.status == status
            assert caught.value.error["message"] == "original failure"
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_explicit_rpc_refusal_is_not_retried_or_exposed_in_exception_string(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)
        error = {"code": 401, "message": "private-diagnostic-token"}

        async def refuse(params):
            raise NativeRpcError("turn/start", error)

        transport.on_turn_start = refuse
        for _ in range(2):
            with pytest.raises(NativeRpcError) as caught:
                await session.decide("obs", (), SCHEMA, "request")
            assert caught.value.error == error
            assert "private-diagnostic-token" not in str(caught.value)
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


@pytest.mark.parametrize("result", [[], {"unexpected": True}, "not-json-object"])
def test_invalid_completed_output_is_archived_without_correction_query(
    tmp_path, result
):
    async def exercise():
        directory = tmp_path / "native"
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("instructions", tmp_path)

        async def invalid(params):
            turn = transport.make_turn(params, result=result)
            transport.history.append(turn)
            return {"turn": turn}

        transport.on_turn_start = invalid
        for _ in range(2):
            with pytest.raises(NativeProtocolError):
                await session.decide("obs", (), SCHEMA, "request")
        assert count(transport, "turn/start") == 1
        assert len(list((directory / "requests").glob("*/native_completion.json"))) == 1
        await session.close()

    run(exercise())


def test_parallel_duplicate_reads_submit_one_turn(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)
        results = await asyncio.gather(
            *[session.decide("obs", (), SCHEMA, "request") for _ in range(3)]
        )
        assert results == [DECISION] * 3
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4"])
def test_effective_model_mismatch_keeps_thread_identity_without_replacement(
    tmp_path, model
):
    async def exercise():
        transport = FakeTransport(tmp_path, model=model)
        transport.start_response = {
            **transport.thread_response(),
            "model": "other-model",
        }
        session = CodexDecisionSession(
            tmp_path / "native", transport=transport, model=model
        )
        with pytest.raises(NativeProtocolError, match="model or provider"):
            await session.start("instructions", tmp_path)
        assert session.session_id == "thread-one"
        await session.start("instructions", tmp_path)
        assert count(transport, "thread/start") == 1
        assert count(transport, "thread/resume") == 1
        await session.close()

    run(exercise())


def test_owner_lock_prevents_two_native_threads_for_one_episode(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        first = CodexDecisionSession(directory, transport=FakeTransport(tmp_path))
        second_transport = FakeTransport(tmp_path)
        second = CodexDecisionSession(directory, transport=second_transport)
        await first.start("instructions", tmp_path)
        with pytest.raises(NativeSessionError, match="active owner"):
            await second.start("instructions", tmp_path)
        assert second_transport.calls == []
        await first.close()
        await second.start("instructions", tmp_path)
        assert count(second_transport, "thread/start") == 0
        await second.close()

    run(exercise())


def test_cancel_known_turn_waits_for_native_terminal_event_and_keeps_identity(tmp_path):
    async def exercise():
        transport = FakeTransport(tmp_path, auto_complete=False)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("instructions", tmp_path)
        pending = asyncio.create_task(session.decide("obs", (), SCHEMA, "request"))
        await transport.turn_started.wait()
        await session.cancel()
        with pytest.raises(NativeTurnFailed) as caught:
            await pending
        assert caught.value.status == "interrupted"
        assert session.session_id == "thread-one"
        assert count(transport, "turn/start") == 1
        await session.close()

    run(exercise())


def test_close_drains_active_waiter_before_releasing_owner(tmp_path):
    async def exercise():
        directory = tmp_path / "native"
        transport = FakeTransport(tmp_path, auto_complete=False)
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("instructions", tmp_path)
        pending = asyncio.create_task(session.decide("obs", (), SCHEMA, "request"))
        await transport.turn_started.wait()
        await asyncio.wait_for(session.close(), timeout=2)
        with pytest.raises(NativeTransportError):
            await pending
        assert state(directory)["requests"]["request"]["phase"] == "submitted"
        restored = CodexDecisionSession(directory, transport=FakeTransport(tmp_path))
        await restored.start("instructions", tmp_path)
        await restored.close()

    run(exercise())


class SkillTransport(FakeTransport):
    def __init__(self, cwd, *, refuse_disable=False, inventory=None):
        super().__init__(cwd)
        self.skills = (
            inventory
            if inventory is not None
            else [
                {
                    "name": "host-skill",
                    "path": str(cwd / "global/SKILL.md"),
                    "enabled": True,
                },
                {
                    "name": "builtin-skill",
                    "path": str(cwd / "system/SKILL.md"),
                    "enabled": True,
                },
            ]
        )
        self.refuse_disable = refuse_disable

    @property
    def metadata(self):
        return {**super().metadata, "native_home": str(self.cwd / "private-runtime")}

    async def request(self, method, params):
        if method == "skills/list":
            self.calls.append((method, copy.deepcopy(params)))
            return {
                "data": [
                    {
                        "cwd": str(self.cwd),
                        "skills": copy.deepcopy(self.skills),
                        "errors": [],
                    }
                ]
            }
        if method == "config/batchWrite":
            self.calls.append((method, copy.deepcopy(params)))
            if not self.refuse_disable:
                self.skills = [{**item, "enabled": False} for item in self.skills]
            return {"status": "ok"}
        return await super().request(method, params)


def test_all_native_skills_disabled_in_private_config_before_first_thread(tmp_path):
    async def exercise():
        transport = SkillTransport(tmp_path)
        directory = tmp_path / "native"
        session = CodexDecisionSession(directory, transport=transport)
        await session.start("Complete Harness task-family experience", tmp_path)
        methods = [method for method, _ in transport.calls]
        assert methods.index("config/batchWrite") < methods.index("thread/start")
        assert methods.count("skills/list") == 2
        write = next(
            params
            for method, params in transport.calls
            if method == "config/batchWrite"
        )
        assert write == {
            "filePath": str(tmp_path / "private-runtime/config.toml"),
            "reloadUserConfig": True,
            "edits": [
                {
                    "keyPath": "skills.config",
                    "mergeStrategy": "replace",
                    "value": [
                        {"path": str(tmp_path / "global/SKILL.md"), "enabled": False},
                        {"path": str(tmp_path / "system/SKILL.md"), "enabled": False},
                    ],
                }
            ],
        }
        start = next(
            params for method, params in transport.calls if method == "thread/start"
        )
        assert (
            start["developerInstructions"] == "Complete Harness task-family experience"
        )
        assert "environments" not in start
        assert start["sandbox"] == "workspace-write"
        assert state(directory)["disabled_native_skills"] == write["edits"][0]["value"]
        assert count(transport, "turn/start") == 0
        await session.close()

    run(exercise())


def test_failed_native_skill_disabling_never_starts_model_thread(tmp_path):
    async def exercise():
        transport = SkillTransport(tmp_path, refuse_disable=True)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        with pytest.raises(NativeProtocolError, match="did not disable"):
            await session.start("instructions", tmp_path)
        assert count(transport, "thread/start") == 0
        assert count(transport, "turn/start") == 0
        await session.close()

    run(exercise())


@pytest.mark.parametrize(
    "skill",
    [
        {"path": "relative/SKILL.md", "enabled": True},
        {"path": "/skill/SKILL.md", "enabled": "false"},
        {"path": "/skill/SKILL.md"},
    ],
)
def test_malformed_skill_inventory_cannot_silently_enable_native_catalog(
    tmp_path, skill
):
    async def exercise():
        transport = SkillTransport(tmp_path, inventory=[skill])
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        with pytest.raises(NativeProtocolError, match="invalid native skill inventory"):
            await session.start("instructions", tmp_path)
        assert count(transport, "thread/start") == 0
        await session.close()

    run(exercise())


def test_rate_limit_interrupts_one_turn_and_closes_shared_admission(tmp_path):
    from reweave.utils.provider_admission import ProviderAdmission

    async def exercise():
        transport = FakeTransport(tmp_path, auto_complete=False)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("Instructions", tmp_path)
        gate = ProviderAdmission("test", root=tmp_path / "limits", start_interval_s=0)
        session._admission = gate

        async def limited(params):
            turn = transport.make_turn(params, status="inProgress")
            transport.history.append(turn)
            for _ in range(2):
                transport.emit(
                    "error",
                    {
                        "threadId": "thread-one",
                        "turnId": turn["id"],
                        "error": {
                            "codexErrorInfo": {
                                "responseStreamDisconnected": {"httpStatusCode": 429}
                            }
                        },
                        "willRetry": True,
                    },
                )
            return {"turn": {"id": turn["id"], "status": "inProgress", "items": []}}

        transport.on_turn_start = limited
        try:
            with pytest.raises(NativeSessionError):
                await session.decide("observation", (), SCHEMA, "request")
            assert count(transport, "turn/start") == 1
            assert count(transport, "turn/interrupt") == 1
            assert gate._acquire()[0] is None
        finally:
            await session.close()

    run(exercise())


def test_cached_decision_does_not_probe_or_clear_a_provider_cooldown(tmp_path):
    from reweave.utils.provider_admission import ProviderAdmission

    async def exercise():
        transport = FakeTransport(tmp_path)
        session = CodexDecisionSession(tmp_path / "native", transport=transport)
        await session.start("Instructions", tmp_path)
        await session.decide("observation", (), SCHEMA, "request")
        gate = ProviderAdmission("test", root=tmp_path / "limits", start_interval_s=0)
        session._admission = gate
        gate.note_rate_limit("HTTP 429 Too Many Requests")
        assert (
            await asyncio.wait_for(
                session.decide("observation", (), SCHEMA, "request"), 1
            )
            == DECISION
        )
        assert count(transport, "turn/start") == 1
        assert gate._acquire()[0] is None
        await session.close()

    run(exercise())
