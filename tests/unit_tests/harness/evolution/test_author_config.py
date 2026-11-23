# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Offline model selection must not borrow the online Planner's environment."""

import pytest

from reweave.evolution.authoring.config import (
    CODEX_AUTHOR_MODEL,
    CURSOR_LONG_CONTEXT_MODEL,
    AuthorPolicy,
)


def test_offline_codex_model_is_independent_of_online_planner(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "online-only-model")
    monkeypatch.delenv("REWEAVE_CODEX_AUTHOR_MODEL", raising=False)
    assert AuthorPolicy(author="codex").model == CODEX_AUTHOR_MODEL == "gpt-5.4"

    monkeypatch.setenv("REWEAVE_CODEX_AUTHOR_MODEL", "offline-author-model")
    assert AuthorPolicy(author="codex").model == "offline-author-model"
    assert (
        AuthorPolicy(author="codex", model="explicit-model").model == "explicit-model"
    )


def test_codex_reasoning_survives_transport_serialization():
    effort = "xhigh"
    from openai_codex.generated.v2_all import ReasoningEffort

    policy = AuthorPolicy(author="codex", model="gpt-5.4", reasoning_effort=effort)
    received = AuthorPolicy.model_validate_json(policy.model_dump_json())
    assert received.model == "gpt-5.4"
    assert ReasoningEffort(received.reasoning_effort) == ReasoningEffort(effort)


@pytest.mark.parametrize("backend", ["claude_code", "codex", "cursor"])
@pytest.mark.parametrize("effort", ["medium", "high"])
def test_every_offline_backend_requires_xhigh(backend, effort):
    assert AuthorPolicy(author=backend).reasoning_effort == "xhigh"
    with pytest.raises(ValueError):
        AuthorPolicy(author=backend, reasoning_effort=effort)


@pytest.mark.parametrize("mode", ["auth", "api"])
def test_codex_connection_mode_survives_policy_serialization(mode):
    policy = AuthorPolicy(author="codex", codex_auth_mode=mode)
    assert (
        AuthorPolicy.model_validate_json(policy.model_dump_json()).codex_auth_mode
        == mode
    )


@pytest.mark.parametrize("backend", ["cursor", "claude_code"])
def test_codex_api_mode_requires_the_codex_backend(backend):
    with pytest.raises(ValueError, match="requires the Codex backend"):
        AuthorPolicy(author=backend, codex_auth_mode="api")


def test_cursor_defaults_to_256k_xhigh_and_keeps_context_session_binding():
    from reweave.evolution.authoring.config import connection_changed
    from reweave.planner.cursor_connection import cursor_model_spec

    policy = AuthorPolicy(author="cursor")
    assert policy.model == "grok-4.7-xhigh"
    selected = cursor_model_spec(policy.model)
    assert selected["context"] == "256k"
    assert selected["reasoning_effort"] == "xhigh"
    assert selected["fast"] is False
    assert policy.codex_auth_mode == "auth"
    assert connection_changed(AuthorPolicy(author="codex").model_dump(), policy)
    assert not connection_changed(policy.model_dump(), policy)
    with pytest.raises(ValueError, match="xhigh catalog"):
        AuthorPolicy(author="cursor", model="grok-4.7-high")
    assert connection_changed(
        {**policy.model_dump(), "model": CURSOR_LONG_CONTEXT_MODEL}, policy
    )
    with pytest.raises(ValueError, match="xhigh catalog"):
        AuthorPolicy(
            author="cursor", model="grok-4.7[context=500k,effort=high,fast=false]"
        )


def test_explicit_500k_cursor_policies_remain_readable():
    original = AuthorPolicy(author="cursor", model=CURSOR_LONG_CONTEXT_MODEL)
    restored = AuthorPolicy.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.model != AuthorPolicy(author="cursor").model
