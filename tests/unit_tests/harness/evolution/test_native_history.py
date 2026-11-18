# Modified for the ReWeave namespace.
"""Native ancestry is a frozen history fork, never shared writable identity."""

import asyncio
import json
from uuid import uuid4

import pytest

from reweave.evolution.infrastructure.native_history import (
    NativeHistorySource,
    native_history,
    receive_history,
)
from reweave.evolution.infrastructure.session import CodexSessionStore


def journal(path, identity, final="task_complete"):
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {
                    "type": "session_meta",
                    "timestamp": "2026-10-08T02:03:04.000Z",
                    "payload": {"id": identity},
                },
                {"type": "event_msg", "payload": {"type": "task_started"}},
                {"type": "event_msg", "payload": {"type": final}},
            ]
        )
        + "\n"
    )
    return path


def fork_journal(path, identity, ancestor, ordinal=2):
    journal(path, identity)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["payload"].update(
        forked_from_id=ancestor, forked_from_ordinal_exclusive=ordinal
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def transfer_history(source, identity, directory, *, checkpoint=None):
    sender = NativeHistorySource(source, identity)
    store = CodexSessionStore(directory)
    store.bind(identity)

    async def request(method, params):
        return sender.handle(method, params)

    staged = asyncio.run(
        receive_history(
            checkpoint if checkpoint is not None else sender.checkpoint,
            identity,
            directory.parent / (directory.name + "_transfers"),
            request,
            existing_journal=store.path if store.path.exists() else None,
        )
    )
    store.capture(staged)
    return store.path


def test_capture_transports_three_generation_native_lineage(tmp_path):
    identities = [str(uuid4()) for _ in range(3)]
    sessions = tmp_path / "original/sessions/2026/10/08"
    sessions.mkdir(parents=True)
    paths = [sessions / f"rollout-{identity}.jsonl" for identity in identities]
    journal(paths[0], identities[0])
    for index in (1, 2):
        fork_journal(paths[index], identities[index], identities[index - 1], 3 * index)
    originals = [path.read_bytes() for path in paths]
    store = CodexSessionStore(tmp_path / "mirror")
    store.bind(identities[2])
    store.capture(paths[2])
    transported = transfer_history(store.path, identities[2], tmp_path / "remote")
    imported = native_history(transported, identities[2])
    assert set(imported.values()) == set(originals)
    assert [path.read_bytes() for path in paths] == originals
    assert len(imported) == 3


@pytest.mark.parametrize("problem", ["missing", "cycle", "conflicting", "ordinal"])
def test_invalid_lineage_never_imports_or_forks(tmp_path, problem):
    parent, ancestor = str(uuid4()), str(uuid4())
    source = fork_journal(tmp_path / "transcript.jsonl", parent, ancestor)
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    ancestor_path = lineage / f"{ancestor}.jsonl"
    if problem == "cycle":
        fork_journal(ancestor_path, ancestor, parent)
        (lineage / f"{parent}.jsonl").write_bytes(source.read_bytes())
    elif problem != "missing":
        journal(ancestor_path, ancestor)
    if problem == "conflicting":
        with source.open("a") as output:
            output.write(
                json.dumps({"type": "session_meta", "payload": {"id": parent}}) + "\n"
            )
    if problem == "ordinal":
        fork_journal(source, parent, ancestor, -1)

    with pytest.raises(ValueError):
        native_history(source, parent)


def test_capture_rejects_conflicting_source_rollouts(tmp_path):
    from reweave.evolution.infrastructure.session import CodexSessionStore

    parent, ancestor = str(uuid4()), str(uuid4())
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    source = fork_journal(sessions / f"rollout-{parent}.jsonl", parent, ancestor)
    journal(sessions / f"rollout-a-{ancestor}.jsonl", ancestor)
    other = journal(sessions / f"rollout-b-{ancestor}.jsonl", ancestor)
    other.write_text(other.read_text().replace("task_complete", "turn_aborted"))
    store = CodexSessionStore(tmp_path / "mirror")
    store.bind(parent)
    with pytest.raises(ValueError, match="conflicting"):
        store.capture(source)
    assert not store.path.exists()


def test_checkpoint_updates_target_but_keeps_ancestor_snapshots_immutable(tmp_path):
    parent, ancestor = str(uuid4()), str(uuid4())
    (tmp_path / "source/lineage").mkdir(parents=True)
    source = fork_journal(tmp_path / "source/transcript.jsonl", parent, ancestor)
    ancestor_path = journal(tmp_path / f"source/lineage/{ancestor}.jsonl", ancestor)
    target = transfer_history(source, parent, tmp_path / "receiver")
    with source.open("a") as output:
        output.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"type": "thread_settings_applied"},
                }
            )
            + "\n"
        )
    transfer_history(source, parent, tmp_path / "receiver")
    assert target.read_bytes() == source.read_bytes()
    ancestor_path.write_text(
        ancestor_path.read_text().replace("task_complete", "turn_aborted")
    )
    with pytest.raises(ValueError, match="ancestor snapshot changed"):
        transfer_history(source, parent, tmp_path / "receiver")


def test_checkpoint_rejects_unrelated_and_missing_source_rollouts(tmp_path):
    parent, ancestor, unrelated = (str(uuid4()) for _ in range(3))
    (tmp_path / "source/lineage").mkdir(parents=True)
    source = fork_journal(tmp_path / "source/transcript.jsonl", parent, ancestor)
    journal(tmp_path / f"source/lineage/{ancestor}.jsonl", ancestor)
    checkpoint = NativeHistorySource(source, parent).checkpoint
    with pytest.raises(ValueError):
        transfer_history(
            source,
            parent,
            tmp_path / "missing",
            checkpoint={
                "checkpoint_id": checkpoint["checkpoint_id"],
                "session_id": parent,
                "files": {parent: checkpoint["files"][parent]},
            },
        )
    extra = journal(tmp_path / "extra.jsonl", unrelated)
    checkpoint["files"].update(
        NativeHistorySource(extra, unrelated).checkpoint["files"]
    )
    with pytest.raises(ValueError):
        transfer_history(source, parent, tmp_path / "extra", checkpoint=checkpoint)
