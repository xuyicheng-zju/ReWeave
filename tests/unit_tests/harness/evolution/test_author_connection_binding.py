"""Native author connections remain bound before preparation or query writes."""

import asyncio
import json

import pytest

from reweave.evolution.authoring.author import NativeSession
from reweave.evolution.authoring.config import AuthorPolicy


def session(tmp_path, policy):
    return NativeSession(
        bundle_loader=lambda _: pytest.fail("bundle loaded"),
        policy=policy,
        evidence_workspace=tmp_path / "evidence",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"codex_auth_mode": "api"},
        {"model": "another-model"},
        {"author": "claude_code"},
    ],
)
def test_existing_author_connection_cannot_change_before_preparation(
    tmp_path, monkeypatch, change
):
    original = AuthorPolicy(author="codex")
    directory = tmp_path / "author"
    context = directory / "audit/context.json"
    context.parent.mkdir(parents=True)
    context.write_text(json.dumps({"policy": original.model_dump(mode="json")}))
    before = context.read_bytes()
    selected = AuthorPolicy.model_validate(
        {**original.model_dump(mode="json"), **change}
    )
    author = session(tmp_path, selected)
    monkeypatch.setattr(
        author, "_prepare_inputs", lambda *a, **k: pytest.fail("preparation ran")
    )
    with pytest.raises(ValueError, match="connection or model differs"):
        asyncio.run(author._execute_once(None, directory, resume=True))
    assert context.read_bytes() == before


def test_historical_auth_default_is_normalized_without_rewriting_binding(
    tmp_path, monkeypatch
):
    original = AuthorPolicy(author="codex")
    policy = original.model_dump(mode="json")
    del policy["codex_auth_mode"]
    directory = tmp_path / "author"
    context = directory / "audit/context.json"
    context.parent.mkdir(parents=True)
    context.write_text(json.dumps({"policy": policy}))
    before = context.read_bytes()
    author = session(tmp_path, original)

    def boundary(*args, **kwargs):
        raise RuntimeError("preparation boundary reached")

    monkeypatch.setattr(author, "_prepare_inputs", boundary)
    with pytest.raises(RuntimeError, match="preparation boundary reached"):
        asyncio.run(author._execute_once(None, directory, resume=True))
    assert context.read_bytes() == before


def test_existing_author_session_requires_its_connection_binding(tmp_path):
    directory = tmp_path / "author"
    identity = directory / "session/identity.json"
    identity.parent.mkdir(parents=True)
    identity.write_text('{"session_id":"original-thread"}')
    before = identity.read_bytes()
    author = session(tmp_path, AuthorPolicy(author="codex", codex_auth_mode="api"))
    with pytest.raises(ValueError, match="lacks its connection binding"):
        asyncio.run(author._execute_once(None, directory, resume=True))
    assert identity.read_bytes() == before
