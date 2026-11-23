# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Lossless bounded Codex history transport in both SSH directions."""

import asyncio
import base64
import copy
import json
import sys
from uuid import uuid4

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.local import LocalCodexAuthor
from reweave.evolution.infrastructure.native_history import (
    HISTORY_CHUNK_BYTES,
    NativeHistorySource,
    receive_history,
)
from reweave.evolution.infrastructure.session import CodexSessionStore
from tests.unit_tests.harness.evolution.test_native_history import fork_journal, journal


def large_journal(path, identity):
    journal(path, identity)
    with path.open("a") as output:
        output.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"text": "x" * (2 * HISTORY_CHUNK_BYTES)},
                }
            )
            + "\n"
        )
    return path


def pull(source, directory, *, existing=None, request=None, checkpoint=None):
    async def read(method, params):
        return source.handle(method, params)

    return asyncio.run(
        receive_history(
            checkpoint if checkpoint is not None else source.checkpoint,
            source.checkpoint["session_id"],
            directory,
            request or read,
            existing_journal=existing,
        )
    )


def test_small_manifest_bounded_chunks_preserve_complete_lineage(tmp_path):
    child, parent = str(uuid4()), str(uuid4())
    (tmp_path / "source/lineage").mkdir(parents=True)
    ancestor = journal(tmp_path / f"source/lineage/{parent}.jsonl", parent)
    source_path = fork_journal(tmp_path / "source/transcript.jsonl", child, parent)
    with source_path.open("a") as output:
        output.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"text": "x" * (2 * HISTORY_CHUNK_BYTES)},
                }
            )
            + "\n"
        )
    source = NativeHistorySource(source_path, child)
    packets = []

    async def request(method, params):
        result = source.handle(method, params)
        packets.append((method, params, result))
        return result

    target = pull(source, tmp_path / "receives", request=request)
    assert target.read_bytes() == source_path.read_bytes()
    assert (
        target.parent / f"lineage/{parent}.jsonl"
    ).read_bytes() == ancestor.read_bytes()
    assert len(json.dumps(source.checkpoint)) < 512
    chunks = [response for method, _, response in packets if method == "history_chunk"]
    assert len(chunks) == 4
    assert all(
        len(base64.b64decode(chunk["data"])) <= HISTORY_CHUNK_BYTES for chunk in chunks
    )
    assert all(len(json.dumps(response)) < 360_000 for _, _, response in packets)
    assert (target.parent.parent / "received.json").is_file()
    packets.clear()
    assert pull(source, tmp_path / "receives", request=request) == target
    assert packets == []


def test_unchanged_prefix_compares_bytes_and_keeps_published_mirror(tmp_path):
    identity = str(uuid4())
    native = large_journal(tmp_path / "native.jsonl", identity)
    store = CodexSessionStore(tmp_path / "published")
    store.bind(identity)
    store.capture(native)
    old = store.path.read_bytes()
    tail = (
        json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "thread_settings_applied"},
            }
        ).encode()
        + b"\n"
    )
    with native.open("ab") as output:
        output.write(tail)
    source = NativeHistorySource(native, identity)
    chunks = []

    async def request(method, params):
        assert store.path.read_bytes() == old
        result = source.handle(method, params)
        if method == "history_chunk":
            chunks.append(result)
        return result

    received = pull(source, tmp_path / "receives", existing=store.path, request=request)
    assert b"".join(base64.b64decode(chunk["data"]) for chunk in chunks) == old + tail
    assert chunks[0]["offset"] == 0
    assert store.path.read_bytes() == old
    store.capture(received)
    assert store.path.read_bytes() == old + tail


def test_cancellation_retains_full_chunks_and_explicit_receive_resumes(tmp_path):
    async def run():
        identity = str(uuid4())
        native = journal(tmp_path / "native.jsonl", identity)
        store = CodexSessionStore(tmp_path / "published")
        store.bind(identity)
        store.capture(native)
        old = store.path.read_bytes()
        with native.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"text": "x" * (2 * HISTORY_CHUNK_BYTES)},
                    }
                )
                + "\n"
            )
        source = NativeHistorySource(native, identity)
        second = asyncio.Event()
        offsets = []

        async def blocked(method, params):
            if method == "history_chunk":
                offsets.append(params["offset"])
                if len(offsets) == 2:
                    second.set()
                    await asyncio.Event().wait()
            return source.handle(method, params)

        task = asyncio.create_task(
            receive_history(
                source.checkpoint,
                identity,
                tmp_path / "receives",
                blocked,
                existing_journal=store.path,
            )
        )
        await second.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.path.read_bytes() == old
        offsets.clear()

        async def request(method, params):
            if method == "history_chunk":
                offsets.append(params["offset"])
            return source.handle(method, params)

        received = await receive_history(
            source.checkpoint,
            identity,
            tmp_path / "receives",
            request,
            existing_journal=store.path,
        )
        assert offsets[0] == 0
        assert received.read_bytes() == native.read_bytes()
        assert store.path.read_bytes() == old
        assert len(list((tmp_path / "receives").glob("*/manifest.json"))) == 1

    asyncio.run(asyncio.wait_for(run(), 5))


def test_changed_root_prefix_is_preserved_and_exact_source_is_received(tmp_path):
    identity = str(uuid4())
    native = large_journal(tmp_path / "native.jsonl", identity)
    store = CodexSessionStore(tmp_path / "published")
    store.bind(identity)
    store.capture(native)
    old = store.path.read_bytes()
    native.write_bytes(native.read_bytes().replace(b"02:03:04", b"02:03:05"))
    source = NativeHistorySource(native, identity)
    target = pull(source, tmp_path / "receives", existing=store.path)
    assert target.read_bytes() == native.read_bytes()
    assert store.path.read_bytes() == old
    rejected = list(target.parent.glob("transcript.jsonl.rejected-*"))
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == old


@pytest.mark.parametrize(
    "problem",
    [
        "offset_bool",
        "offset",
        "identity",
        "short",
        "oversize",
        "base64",
        "content",
    ],
)
def test_damaged_transport_never_publishes_or_silently_restarts(tmp_path, problem):
    identity = str(uuid4())
    native = large_journal(tmp_path / "native.jsonl", identity)
    source = NativeHistorySource(native, identity)
    calls = []

    async def request(method, params):
        calls.append(method)
        result = source.handle(method, params)
        if method == "history_chunk":
            if problem == "offset_bool":
                result["offset"] = False
            elif problem == "offset":
                result["offset"] += 1
            elif problem == "identity":
                result["rollout_id"] = str(uuid4())
            elif problem == "short":
                result["data"] = base64.b64encode(b"short").decode()
            elif problem == "oversize":
                result["data"] += "AAAA"
            elif problem == "base64":
                result["data"] = "!" + result["data"][1:]
            elif problem == "content":
                raw = bytearray(base64.b64decode(result["data"]))
                raw[0] ^= 1
                result["data"] = base64.b64encode(raw).decode()
        return result

    with pytest.raises(ValueError):
        pull(source, tmp_path / "receives", request=request)
    assert not list((tmp_path / "receives").glob("*/received.json"))
    assert not list((tmp_path / "receives").rglob("*.rejected-*"))


@pytest.mark.parametrize(
    "problem",
    [
        "legacy",
        "wrong_target",
        "unsafe_id",
        "bytes_bool",
        "negative_bytes",
        "checkpoint_id",
        "extra_field",
    ],
)
def test_invalid_checkpoint_is_rejected_before_any_rpc(tmp_path, problem):
    identity = str(uuid4())
    native = journal(tmp_path / "native.jsonl", identity)
    source = NativeHistorySource(native, identity)
    checkpoint = copy.deepcopy(source.checkpoint)
    if problem == "legacy":
        checkpoint = {identity: base64.b64encode(native.read_bytes()).decode()}
    elif problem == "wrong_target":
        checkpoint["session_id"] = str(uuid4())
    elif problem == "unsafe_id":
        checkpoint["files"]["../../outside"] = checkpoint["files"][identity]
    elif problem == "bytes_bool":
        checkpoint["files"][identity]["bytes"] = True
    elif problem == "negative_bytes":
        checkpoint["files"][identity]["bytes"] = -1
    elif problem == "checkpoint_id":
        checkpoint["checkpoint_id"] = "../unsafe"
    else:
        checkpoint["files"][identity]["extra"] = 1

    async def forbidden(*args):
        pytest.fail("invalid checkpoint must not issue an RPC")

    with pytest.raises(ValueError):
        pull(source, tmp_path / "receives", request=forbidden, checkpoint=checkpoint)


def test_source_rejects_invalid_offset(tmp_path):
    identity = str(uuid4())
    source = NativeHistorySource(journal(tmp_path / "native.jsonl", identity), identity)
    for offset in (-1, True, None, "0", 10**9):
        with pytest.raises(ValueError, match="offset"):
            source.handle("history_chunk", {"rollout_id": identity, "offset": offset})


def test_duplicate_chunk_cannot_replace_or_append_wrong_bytes(tmp_path):
    from reweave.evolution.infrastructure.native_history import _append_history_chunk

    target = tmp_path / "partial"
    target.touch()
    _append_history_chunk(target, 0, b"first")
    with pytest.raises(ValueError, match="already received"):
        _append_history_chunk(target, 0, b"duplicate")
    assert target.read_bytes() == b"first"


@pytest.mark.parametrize("problem", ["missing", "unrelated"])
def test_transport_manifest_requires_native_lineage(tmp_path, problem):
    identity, ancestor = str(uuid4()), str(uuid4())
    (tmp_path / "lineage").mkdir()
    source = fork_journal(tmp_path / "transcript.jsonl", identity, ancestor)
    parent = journal(tmp_path / f"lineage/{ancestor}.jsonl", ancestor)
    history = {identity: source.read_bytes(), ancestor: parent.read_bytes()}
    if problem == "missing":
        del history[ancestor]
    else:
        extra = str(uuid4())
        history[extra] = journal(tmp_path / "extra.jsonl", extra).read_bytes()
    checkpoint = {
        "checkpoint_id": str(uuid4()),
        "session_id": identity,
        "files": {key: {"bytes": len(raw)} for key, raw in history.items()},
    }

    async def request(method, params):
        key, offset = params["rollout_id"], params["offset"]
        return {
            "rollout_id": key,
            "offset": offset,
            "data": base64.b64encode(
                history[key][offset : offset + HISTORY_CHUNK_BYTES]
            ).decode(),
        }

    with pytest.raises(ValueError, match=problem):
        asyncio.run(
            receive_history(checkpoint, identity, tmp_path / "receives", request)
        )
    assert not list((tmp_path / "receives").glob("*/received.json"))


def test_retired_inquiry_cannot_start_a_native_session(tmp_path):

    class NoCalls:
        async def call(self, *args):
            raise AssertionError("retired inquiry reached native transport")

    author = LocalCodexAuthor(NoCalls(), tmp_path, codex_bin=sys.executable)
    with pytest.raises(ValueError, match="unsupported native job role"):
        asyncio.run(
            author.query(
                {
                    "policy": AuthorPolicy(author="codex").model_dump(mode="json"),
                    "role": "session_inquiry",
                    "session_key": "removed",
                    "request_id": "removed",
                }
            )
        )
