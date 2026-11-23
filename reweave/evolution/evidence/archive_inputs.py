# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Read immutable contracts predating removal of author session inheritance.

Current public models reject the retired fields. Only archive readers discard
obsolete provenance when decoding saved records; original bytes stay untouched
and no field here can enable inheritance or select a native source session.
"""

from __future__ import annotations

from typing import Any


def archived_proposal_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    current = dict(value)
    current.pop("parent_author", None)
    return current


def archived_proposal(value: Any):
    from reweave.evolution.contracts import Proposal

    return Proposal.model_validate(archived_proposal_value(value))


def archived_author_policy(value: Any):
    from reweave.evolution.authoring.config import AuthorPolicy

    if not isinstance(value, dict):
        return AuthorPolicy.model_validate(value)
    current = dict(value)
    current.pop("inherit_parent_session", None)
    return AuthorPolicy.model_validate(current)
