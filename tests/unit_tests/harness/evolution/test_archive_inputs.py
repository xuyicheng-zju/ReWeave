"""Retired author controls are archive-only data, never a current launch option."""

import copy

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import Proposal
from reweave.evolution.evidence.archive_inputs import (
    archived_author_policy,
    archived_proposal,
)
from tests.unit_tests.harness.evolution.test_author_concurrency import setup_author


@pytest.mark.parametrize("retired", [False, True])
def test_current_author_policy_rejects_retired_inheritance_and_archive_reader_preserves_bytes(
    retired,
):
    payload = AuthorPolicy(author="codex").model_dump(mode="json") | {
        "inherit_parent_session": retired
    }
    before = copy.deepcopy(payload)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AuthorPolicy.model_validate(payload)
    assert archived_author_policy(payload) == AuthorPolicy(author="codex")
    assert payload == before
    assert (
        "inherit_parent_session" not in AuthorPolicy.model_json_schema()["properties"]
    )


def test_current_proposal_rejects_retired_parent_and_archive_reader_keeps_evidence(
    tmp_path, monkeypatch
):
    _, request, _ = setup_author(tmp_path, monkeypatch)
    payload = request.model_dump(mode="json") | {"parent_author": None}
    before = copy.deepcopy(payload)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Proposal.model_validate(payload)
    assert archived_proposal(payload) == request
    assert payload == before
    assert "parent_author" not in Proposal.model_json_schema()["properties"]
