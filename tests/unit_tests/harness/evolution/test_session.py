# Modified for the ReWeave namespace.
"""Lossless native sessions keep branch identity across processes and relocation."""

from __future__ import annotations

import asyncio
import json
import shutil

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal import session_resume

from reweave.evolution.infrastructure.session import ClaudeSessionStore

_SESSION = "11111111-1111-4111-8111-111111111111"
_OTHER = "22222222-2222-4222-8222-222222222222"


def _key(*, session_id=_SESSION, project="old-location", subpath=None):
    key = {"session_id": session_id, "project_key": project}
    if subpath is not None:
        key["subpath"] = subpath
    return key


def _entry(identity="message-1", text="native message"):
    return {
        "uuid": identity,
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def test_revision_journal_retains_every_original_and_loads_latest_uuid(tmp_path):
    store = ClaudeSessionStore(tmp_path)
    original = _entry(text="完整原始消息" * 5000)
    second = _entry("message-2", "follow-up")
    revised = {
        **original,
        "message": {"content": "edited revision", "usage": {"tokens": 28}},
    }
    marker = {"type": "tag", "value": "opaque metadata"}
    asyncio.run(store.append(_key(), [original, second, marker]))
    prefix = store.path.read_bytes()
    asyncio.run(store.append(_key(project="new-location"), [revised, marker]))
    assert store.path.read_bytes().startswith(prefix)
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert [row["entry"] for row in rows] == [original, second, marker, revised, marker]
    assert asyncio.run(store.load(_key(project="new-location"))) == [
        revised,
        second,
        marker,
        marker,
    ]
    loaded = asyncio.run(store.load(_key()))
    loaded[0]["message"]["content"] = "mutated caller copy"
    assert asyncio.run(store.load(_key()))[0] == revised


def test_subagent_uuid_never_overwrites_main_transcript(tmp_path):
    store = ClaudeSessionStore(tmp_path)
    main = _entry(text="main")
    child = _entry(text="child with identical UUID")
    asyncio.run(store.append(_key(), [main]))
    asyncio.run(store.append(_key(subpath="subagents/agent-a"), [child]))
    assert asyncio.run(store.load(_key())) == [main]
    assert asyncio.run(store.load(_key(subpath="subagents/agent-a"))) == [child]
    assert asyncio.run(store.list_subkeys(_key(project="moved"))) == [
        "subagents/agent-a"
    ]


def test_bound_store_rejects_wrong_provider_session_and_missing_main(tmp_path):
    store = ClaudeSessionStore(tmp_path)
    store.bind(_SESSION)
    with pytest.raises(ValueError, match="missing"):
        asyncio.run(store.load(_key()))
    with pytest.raises(ValueError, match="different session"):
        asyncio.run(store.append(_key(session_id=_OTHER), [_entry()]))
    with pytest.raises(ValueError, match="different session"):
        asyncio.run(store.load(_key(session_id=_OTHER)))


@pytest.mark.parametrize(
    "subpath",
    [
        "../other",
        "/outside",
        "C:other",
        "subagents/../outside",
        "subagents\\..\\outside",
    ],
)
def test_subpath_that_sdk_would_silently_skip_is_rejected(tmp_path, subpath):
    with pytest.raises(ValueError, match="safely materialized"):
        asyncio.run(
            ClaudeSessionStore(tmp_path).append(_key(subpath=subpath), [_entry()])
        )


def test_native_event_serialization_preserves_complete_sdk_fields():
    from dataclasses import dataclass

    from pydantic import BaseModel

    from reweave.evolution.infrastructure.session import serialize_native_event

    @dataclass
    class Event:
        text: str
        values: tuple[int, ...]

    class Message(BaseModel):
        text: str

    text = "complete native reasoning " * 10000
    assert serialize_native_event(Event(text, (1, 2))) == {
        "text": text,
        "values": [1, 2],
    }
    assert serialize_native_event(Message(text=text)) == {"text": text}
    with pytest.raises(TypeError, match="unsupported SDK audit"):
        serialize_native_event(object())


def test_actual_sdk_materializes_main_and_subagent_from_relocated_store(
    tmp_path, monkeypatch
):
    # The SDK materializer normally copies local authentication files. No auth
    # or model access is needed to test the actual resume/store protocol.
    monkeypatch.setattr(session_resume, "_copy_auth_files", lambda *_args: None)
    original = tmp_path / "source"
    store = ClaudeSessionStore(original)
    store.bind(_SESSION)
    main = _entry()
    child = _entry("child", "subagent content")
    asyncio.run(store.append(_key(), [main]))
    asyncio.run(store.append(_key(subpath="subagents/agent-a"), [child]))
    moved = tmp_path / "moved"
    shutil.move(original, moved)
    resumed = ClaudeSessionStore(moved)
    resumed.bind(_SESSION)
    options = ClaudeAgentOptions(
        cwd=moved,
        resume=_SESSION,
        session_store=resumed,
        session_store_flush="eager",
    )

    async def materialize():
        materialized = await session_resume.materialize_resume_session(options)
        assert materialized is not None
        try:
            files = list((materialized.config_dir / "projects").rglob("*.jsonl"))
            by_name = {
                path.name: [json.loads(line) for line in path.read_text().splitlines()]
                for path in files
            }
            assert by_name == {f"{_SESSION}.jsonl": [main], "agent-a.jsonl": [child]}
            assert materialized.resume_session_id == _SESSION
        finally:
            await materialized.cleanup()

    asyncio.run(materialize())


def test_query_checkpoint_preserves_failed_persistence(tmp_path):
    store = ClaudeSessionStore(tmp_path)
    store.bind(_SESSION)
    with pytest.raises(TypeError):
        asyncio.run(
            store.append(_key(), [{"type": "assistant", "unsupported": object()}])
        )
    asyncio.run(store.append(_key(), [_entry()]))
    with pytest.raises(ValueError, match="persistence failed"):
        store.checkpoint()


@pytest.mark.parametrize("rewrite", [False, True])
def test_codex_mirror_follows_provider_journal_without_byte_gate(tmp_path, rewrite):
    from reweave.evolution.infrastructure.session import CodexSessionStore

    store = CodexSessionStore(tmp_path / "mirror")
    store.bind("thread")
    native = tmp_path / "native.jsonl"
    meta = {"type": "session_meta", "payload": {"id": "thread"}}
    native.write_text(json.dumps(meta) + "\n")
    store.capture(native)
    # Resume appends settings; providers may also rewrite metadata formatting.
    if rewrite:
        meta["payload"]["cli_version"] = "updated"
        native.write_text(json.dumps(meta) + "\n")
    with native.open("a") as output:
        output.write(
            json.dumps(
                {"type": "event_msg", "payload": {"type": "thread_settings_applied"}}
            )
            + "\n"
        )
    store.capture(native)
    assert store.path.read_bytes() == native.read_bytes()
    assert store.checkpoint()["bytes"] == native.stat().st_size
