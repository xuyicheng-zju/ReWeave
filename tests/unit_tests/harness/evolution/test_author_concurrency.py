# Modified for the ReWeave namespace.
"""Native adapter cancellation and durable workspace ownership."""

import asyncio
import json
import shutil
import threading
from types import SimpleNamespace

import pytest

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import Proposal
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline, mutated, usage
from tests.unit_tests.harness.evolution.test_bundle_author import (
    Journal,
    host_input,
    native_completion,
    workspace,
)


class QueryObserved(RuntimeError):
    """Stop a test query without any provider or physical execution."""


def setup_author(tmp_path, monkeypatch, policy=None):
    template = tmp_path / "template"
    template.mkdir()
    source = workspace(template)
    proposal = Proposal.model_validate_json(
        host_input(source, "proposal.json").read_text()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=policy or AuthorPolicy(),
    )
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.claude_workspace_options",
        lambda *a, **kw: {},
    )

    def prepare(proposal, destination):
        shutil.copytree(source, destination)
        private = destination / ".host_inputs"
        shutil.copytree(host_input(source, "references.json").parent, private)
        atomic_json(private / "proposal.json", proposal.model_dump(mode="json"))

    return author, proposal, prepare


@pytest.mark.anyio
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_preparation_keeps_lock_until_worker_settles(
    tmp_path, monkeypatch, fail
):
    author, proposal, prepare = setup_author(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    completed = threading.Event()
    launches = []
    directory = tmp_path / "branch"

    def slow_prepare(request, destination):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3), "test failed to release worker"
        try:
            if fail:
                raise ValueError("archive checksum mismatch")
            prepare(request, destination)
        finally:
            completed.set()

    monkeypatch.setattr(author, "_prepare", slow_prepare)
    task = asyncio.create_task(
        author(proposal, directory, before_launch=lambda: launches.append(1))
    )
    try:
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not completed.is_set()
        with pytest.raises(RuntimeError, match="already owned"):
            await author.recover(
                proposal, directory, before_launch=lambda: launches.append(1)
            )
        assert launches == []
        release.set()
        if fail:
            with pytest.raises(ValueError, match="archive checksum mismatch") as error:
                await task
            assert isinstance(error.value.__context__, asyncio.CancelledError)
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert completed.is_set()
        assert not (directory / "result.json").exists()
        assert launches == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    if not fail:
        # Preparation finished atomically before cancellation. Recovery reuses it
        # and only now registers the first native query.
        async def query(**kwargs):
            raise QueryObserved("first launch after completed preparation")
            yield

        monkeypatch.setattr("claude_agent_sdk.query", query)
        with pytest.raises(QueryObserved):
            await author.recover(
                proposal, directory, before_launch=lambda: launches.append(1)
            )
        assert launches == [1]


@pytest.mark.anyio
async def test_completion_verification_runs_off_event_loop_and_preserves_receipt(
    tmp_path, monkeypatch
):
    workspace(tmp_path)
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.ClaudeSessionStore", Journal
    )
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    original = author._load_completion
    loop = asyncio.get_running_loop()
    owner = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()

    def verify(directory):
        assert threading.get_ident() != owner
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3), "event loop blocked by input verification"
        return original(directory)

    monkeypatch.setattr(author, "_load_completion", verify)
    task = asyncio.create_task(author._publish(tmp_path))
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert not task.done()
        release.set()
        result = await task
        assert result.candidates[0].bundle.same_harness(mutated(baseline(), "C"))
        assert (
            json.loads((tmp_path / "result.json").read_text())["session_id"]
            == "native-id"
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_cancelled_completion_commits_receipt_and_recovers_without_new_query(
    tmp_path, monkeypatch
):
    import uuid

    from claude_agent_sdk import ResultMessage, SystemMessage

    from reweave.evolution.authoring import author as bundle_author

    author, proposal, prepare = setup_author(tmp_path, monkeypatch)
    monkeypatch.setattr(author, "_prepare", prepare)
    directory = tmp_path / "branch"
    identity = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    completing = asyncio.Event()
    release = threading.Event()
    launches = []

    async def query(*, prompt, options):
        await options.session_store.append(
            {"session_id": identity, "project_key": "concurrent-review"},
            [
                {
                    "uuid": "completed",
                    "type": "assistant",
                    "message": {"content": "Completed the requested analysis."},
                }
            ],
        )
        yield SystemMessage(subtype="init", data={"session_id": identity})
        yield ResultMessage(
            subtype="success",
            duration_ms=5,
            duration_api_ms=2,
            is_error=False,
            num_turns=1,
            session_id=identity,
            total_cost_usd=1,
        )

    original = bundle_author.ClaudeSessionStore.checkpoint

    def checkpoint(store):
        loop.call_soon_threadsafe(completing.set)
        assert release.wait(3), "test failed to release checkpoint"
        return original(store)

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(bundle_author.ClaudeSessionStore, "checkpoint", checkpoint)
    task = asyncio.create_task(
        author(proposal, directory, before_launch=lambda: launches.append(1))
    )
    try:
        await asyncio.wait_for(completing.wait(), 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert (directory / "native_result.json").exists()
    assert not (directory / "result.json").exists()
    result = await author.recover(
        proposal, directory, before_launch=lambda: launches.append(1)
    )
    assert result.candidates[0].bundle.same_harness(mutated(baseline(), "C"))
    assert result.session_id == identity
    assert launches == [1]


@pytest.mark.anyio
async def test_claude_author_uses_the_anthropic_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-key")
    author, proposal, prepare = setup_author(
        tmp_path,
        monkeypatch,
        AuthorPolicy(author="claude_code"),
    )
    environments = []

    async def query(**kwargs):
        environments.append(kwargs["options"].env)
        raise QueryObserved("child environment observed")
        yield

    monkeypatch.setattr(author, "_prepare", prepare)
    monkeypatch.setattr("claude_agent_sdk.query", query)
    with pytest.raises(QueryObserved):
        await author(proposal, tmp_path / "branch", before_launch=lambda: None)
    (environment,) = environments
    assert environment["REWEAVE_AUTHOR_TOOLS_URL"]
    assert environment["OPENAI_API_KEY"] == ""
    assert "ANTHROPIC_API_KEY" not in environment
