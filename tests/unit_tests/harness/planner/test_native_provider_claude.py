"""Claude Code decision sessions: API routing, idempotent turns, no replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import claude_agent_sdk
import pytest

from reweave.planner import codex_connection
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers.claude import ClaudeDecisionSession
from reweave.planner.providers.base import (
    NativeAuthenticationError,
    NativeProtocolError,
    NativeSubmissionUnknown,
    NativeTransportError,
    NativeTurnFailed,
)

KEY = "fixture-anthropic-secret"
SCHEMA = {
    "type": "object",
    "properties": {"tool_calls": {"type": "array"}},
    "required": ["tool_calls"],
    "additionalProperties": False,
}
DECISION = {"tool_calls": [{"name": "move_pose"}]}


def result(*, structured=DECISION, is_error=False, status=None, usage=None):
    return claude_agent_sdk.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=2,
        session_id="fixture",
        usage=usage or {"input_tokens": 3},
        structured_output=structured,
        api_error_status=status,
        terminal_reason="completed",
    )


class FakeClaude:
    """Stands in for the CLI; the transcript is what a real session leaves behind."""

    def __init__(self):
        self.options = []
        self.messages = []
        self.script = [lambda: [result()]]
        self.fail_before_send = None
        self.write_transcript = True

    def __call__(self, options):
        self.options.append(options)
        return FakeClient(self, options)


class FakeClient:
    def __init__(self, owner, options):
        self.owner, self.options = owner, options

    async def __aenter__(self):
        if self.owner.fail_before_send is not None:
            raise self.owner.fail_before_send
        return self

    async def __aexit__(self, *error):
        return False

    async def query(self, prompt, session_id):
        async for message in prompt:
            self.owner.messages.append(message)
            if self.owner.write_transcript:
                config = Path(self.options.env["CLAUDE_CONFIG_DIR"])
                transcript = config / "projects/workspace" / f"{session_id}.jsonl"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                with transcript.open("a") as stream:
                    stream.write(
                        json.dumps({"type": "user", "uuid": message["uuid"]}) + "\n"
                    )

    async def receive_response(self):
        step = self.owner.script.pop(0)
        for frame in step():
            yield frame


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Contract\nUse move_pose.\n")
    fake = FakeClaude()
    image = tmp_path / "head.png"
    image.write_bytes(b"\x89PNG fixture")

    def build(**settings):
        return ClaudeDecisionSession(
            tmp_path / "provider", client_factory=fake, **settings
        )

    return build, fake, workspace, image, tmp_path


def run(coroutine):
    return asyncio.run(coroutine)


def test_planner_selects_claude_provider_with_anthropic_runtime(tmp_path):
    planner = NativeBundlePlanner(
        output_dir=tmp_path, provider="claude", model="claude-opus-5"
    )
    session = planner.session_factory(tmp_path / "provider")
    assert isinstance(session, ClaudeDecisionSession)
    assert session.model == "claude-opus-5" and session.editable_paths == ()
    configuration = planner.runtime_configuration()
    assert configuration["provider"] == "claude"
    assert configuration["model_provider"] == "anthropic"
    # Thinking is always off for Claude; the recorded sampling setting is the
    # decision effort, so it matches the evaluation conditions of later runs.
    assert configuration["thinking"] == "disabled"
    assert planner.sampling_settings() == {"reasoning_effort": "medium"}
    assert session.reasoning_effort == "medium"
    adaptive = NativeBundlePlanner(
        output_dir=tmp_path, provider="claude", adaptation_mode="adaptive"
    )
    assert adaptive.session_factory(tmp_path / "adaptive").editable_paths == (
        "harness/task-family/**",
    )
    assert NativeBundlePlanner(output_dir=tmp_path).provider == "codex"
    with pytest.raises(ValueError, match="No native session adapter"):
        NativeBundlePlanner(output_dir=tmp_path, provider="other")


def test_decision_uses_private_anthropic_environment_and_read_only_tools(episode):
    build, fake, workspace, image, tmp_path = episode
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        return await session.decide("observe", (image,), SCHEMA, "r1")

    assert run(scenario()) == DECISION
    options = fake.options[0]
    assert "ANTHROPIC_API_KEY" not in options.env
    assert options.env["OPENAI_API_KEY"] == ""
    assert options.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert Path(options.env["CLAUDE_CONFIG_DIR"]) == session.config_directory
    assert options.tools == ["Read", "Glob", "Grep"]
    assert options.allowed_tools == ["Read", "Glob", "Grep"]
    assert options.permission_mode == "dontAsk"
    assert options.thinking == {"type": "disabled"}
    assert options.setting_sources == []
    assert options.output_format == {"type": "json_schema", "schema": SCHEMA}
    assert options.cwd == str(workspace.resolve())
    assert options.session_id == session.session_id and options.resume is None
    assert "Use move_pose." in options.system_prompt
    content = fake.messages[0]["message"]["content"]
    assert content[0] == {"type": "text", "text": "observe"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert fake.messages[0]["uuid"] == session._state["requests"]["r1"]["client_id"]


def test_adaptive_session_may_edit_only_declared_source(episode):
    build, fake, workspace, image, _ = episode
    session = build(editable_paths=("harness/**",))

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        await session.decide("observe", (), SCHEMA, "r1")

    run(scenario())
    options = fake.options[0]
    assert options.tools == ["Read", "Glob", "Grep", "Edit", "Write"]
    assert options.allowed_tools == [
        "Read",
        "Glob",
        "Grep",
        "Edit(harness/**)",
        "Write(harness/**)",
    ]


def test_same_request_is_cached_and_changed_input_is_rejected(episode):
    build, fake, workspace, image, _ = episode
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        first = await session.decide("observe", (image,), SCHEMA, "r1")
        again = await session.decide("observe", (image,), SCHEMA, "r1")
        assert first == again
        with pytest.raises(NativeProtocolError, match="different input"):
            await session.decide("changed", (image,), SCHEMA, "r1")

    run(scenario())
    assert len(fake.messages) == 1


def test_later_decision_resumes_the_same_session_after_restart(episode):
    build, fake, workspace, image, _ = episode
    fake.script = [lambda: [result()], lambda: [result()]]

    async def first():
        session = build()
        await session.start("Follow AGENTS.md.", workspace)
        await session.decide("one", (), SCHEMA, "r1")
        await session.close()
        return session.session_id

    identity = run(first())

    async def second():
        resumed = build()
        await resumed.start("Follow AGENTS.md.", workspace)
        await resumed.decide("two", (), SCHEMA, "r2")
        assert resumed.session_id == identity
        # The cached first decision survives the restart without a new query.
        assert await resumed.decide("one", (), SCHEMA, "r1") == DECISION

    run(second())
    assert fake.options[1].resume == identity and fake.options[1].session_id is None
    assert len(fake.messages) == 2


def test_compaction_boundary_advances_context_generation(episode):
    build, fake, workspace, _, _ = episode
    boundary = claude_agent_sdk.SystemMessage(subtype="compact_boundary", data={})
    fake.script = [lambda: [boundary, result()]]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        assert session.context_generation == 0
        await session.decide("observe", (), SCHEMA, "r1")

    run(scenario())
    assert session.context_generation == 1


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credential_is_an_authentication_error_and_never_requeried(
    episode, status
):
    build, fake, workspace, _, _ = episode
    fake.script = [lambda: [result(structured=None, is_error=True, status=status)]]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        with pytest.raises(NativeAuthenticationError):
            await session.decide("observe", (), SCHEMA, "r1")
        with pytest.raises(NativeTurnFailed):
            await session.decide("observe", (), SCHEMA, "r1")

    run(scenario())
    assert len(fake.messages) == 1


def test_failed_turn_is_retained_and_never_requeried(episode):
    build, fake, workspace, _, _ = episode
    fake.script = [lambda: [result(structured=None, is_error=True, status=529)]]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        for _ in range(2):
            with pytest.raises(NativeTurnFailed) as caught:
                await session.decide("observe", (), SCHEMA, "r1")
        return caught.value.error

    assert run(scenario())["api_error_status"] == 529
    assert len(fake.messages) == 1


@pytest.mark.parametrize("output", [None, {"tool_calls": "no"}, ["list"]])
def test_invalid_output_is_archived_without_correction_query(episode, output):
    build, fake, workspace, _, _ = episode
    fake.script = [lambda: [result(structured=output)]]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        for _ in range(2):
            with pytest.raises(NativeProtocolError):
                await session.decide("observe", (), SCHEMA, "r1")

    run(scenario())
    assert len(fake.messages) == 1


def test_connection_failure_before_send_allows_one_later_submission(episode):
    build, fake, workspace, _, _ = episode
    fake.fail_before_send = claude_agent_sdk.CLIConnectionError("refused")
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        with pytest.raises(NativeTransportError) as caught:
            await session.decide("observe", (), SCHEMA, "r1")
        assert caught.value.request_sent is False
        fake.fail_before_send = None
        return await session.decide("observe", (), SCHEMA, "r1")

    assert run(scenario()) == DECISION
    assert len(fake.messages) == 1


def test_stream_loss_after_recorded_submission_is_never_replayed(episode):
    build, fake, workspace, _, _ = episode

    def lost():
        raise claude_agent_sdk.ProcessError("stream closed", exit_code=1)
        yield

    fake.script = [lost]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        with pytest.raises(NativeTransportError) as caught:
            await session.decide("observe", (), SCHEMA, "r1")
        assert caught.value.request_sent is True
        with pytest.raises(NativeSubmissionUnknown):
            await session.decide("observe", (), SCHEMA, "r1")
        with pytest.raises(NativeSubmissionUnknown):
            await session.decide("later", (), SCHEMA, "r2")

    run(scenario())
    assert len(fake.messages) == 1


def test_stream_loss_without_a_transcript_entry_resubmits_once(episode):
    build, fake, workspace, _, _ = episode
    fake.write_transcript = False

    def lost():
        raise claude_agent_sdk.ProcessError("stream closed", exit_code=1)
        yield

    fake.script = [lost, lambda: [result()]]
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        with pytest.raises(NativeTransportError):
            await session.decide("observe", (), SCHEMA, "r1")
        return await session.decide("observe", (), SCHEMA, "r1")

    assert run(scenario()) == DECISION
    assert len(fake.messages) == 2


def test_credential_is_never_persisted(episode):
    build, fake, workspace, image, tmp_path = episode
    session = build()

    async def scenario():
        await session.start("Follow AGENTS.md.", workspace)
        await session.decide("observe", (image,), SCHEMA, "r1")

    run(scenario())
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert KEY.encode() not in file.read_bytes(), file


def test_owner_lock_prevents_two_owners_for_one_episode(episode):
    build, _, workspace, _, _ = episode
    first, second = build(), build()

    async def scenario():
        await first.start("Follow AGENTS.md.", workspace)
        with pytest.raises(Exception, match="already has an active owner"):
            await second.start("Follow AGENTS.md.", workspace)

    run(scenario())


def test_historical_session_cannot_silently_change_the_model(episode):
    build, _, workspace, _, tmp_path = episode
    run(build().start("Follow AGENTS.md.", workspace))
    with pytest.raises(NativeProtocolError, match="does not match"):
        ClaudeDecisionSession(tmp_path / "provider", model="claude-opus-5")


def test_routing_without_config_directory_leaves_the_launcher_home_alone(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    routing = codex_connection.claude_environment()
    assert "CLAUDE_CONFIG_DIR" not in routing
    assert "ANTHROPIC_API_KEY" not in routing
    assert routing["CODEX_API_KEY"] == ""
