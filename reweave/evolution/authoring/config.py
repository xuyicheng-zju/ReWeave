# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Author model selection and observed cost accounting, with no Host budgets."""

import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reweave.evolution.evidence.archive_inputs import archived_author_policy

CLAUDE_CODE_AUTHOR_MODEL = "claude-sonnet-4-6"
CODEX_AUTHOR_MODEL = "gpt-5.4"
CURSOR_AUTHOR_MODEL = "grok-4.7-xhigh"
CURSOR_LONG_CONTEXT_MODEL = "grok-4.7[context=500k,effort=xhigh,fast=false]"
AUTHOR_REASONING_EFFORT = "xhigh"
AuthorBackend = Literal["claude_code", "codex", "cursor"]


def default_connection(data):
    if data["author"] != "codex":
        return "auth"
    from reweave.planner.codex_connection import default_codex_auth_mode

    return default_codex_auth_mode()


def author_model(author: AuthorBackend) -> str:
    """Resolve the selected author's model without borrowing another backend's settings."""
    if author == "claude_code":
        return (
            os.environ.get("REWEAVE_CLAUDE_CODE_AUTHOR_MODEL")
            or CLAUDE_CODE_AUTHOR_MODEL
        )
    if author == "codex":
        return os.environ.get("REWEAVE_CODEX_AUTHOR_MODEL") or CODEX_AUTHOR_MODEL
    if author == "cursor":
        return CURSOR_AUTHOR_MODEL
    raise ValueError(f"unsupported author backend: {author}")


class AuthorPolicy(BaseModel):
    """One native model configuration shared by all repair mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    author: AuthorBackend = Field(
        default="claude_code",
        description="Native coding-agent backend for harness analysis and source authoring.",
    )
    codex_auth_mode: Literal["auth", "api"] = Field(
        default_factory=default_connection,
        description="Codex connection via local login or an API key. Credentials stay outside author configuration.",
    )
    model: str = Field(
        default_factory=lambda data: author_model(data["author"]),
        min_length=1,
        description="Coding model used for evidence analysis and source authoring.",
    )
    reasoning_effort: Literal["xhigh"] = Field(
        default=AUTHOR_REASONING_EFFORT,
        description="Offline graph/author reasoning effort: xhigh for every backend. Independent of online Planner settings.",
    )
    graph_retrieval: Path | None = Field(
        default=None,
        description="Optional absolute Host path to a GraphRetrievalConfig selecting one final graph per reference task and the local Qwen embedding cache. Enables author-initiated retrieval and on-demand evidence reads in every graph-analysis/mutation round; does not change parent selection or generate H0 automatically.",
    )

    @model_validator(mode="after")
    def validate_reasoning(self):
        if self.codex_auth_mode != "auth" and self.author != "codex":
            raise ValueError("Codex API mode requires the Codex backend")
        if self.author == "cursor" and not (
            self.model.endswith("-xhigh") or self.model == CURSOR_LONG_CONTEXT_MODEL
        ):
            raise ValueError(
                "Cursor offline model must select its xhigh catalog variant"
            )
        if self.graph_retrieval is not None and not self.graph_retrieval.is_absolute():
            raise ValueError("graph_retrieval configuration path must be absolute")
        return self


_CONNECTION_FIELDS = ("author", "model", "reasoning_effort", "codex_auth_mode")


def connection_changed(recorded: Any, policy: AuthorPolicy) -> bool:
    """Whether a recorded session binding names another backend, model or route."""
    previous = archived_author_policy(recorded)
    return any(
        getattr(previous, field) != getattr(policy, field)
        for field in _CONNECTION_FIELDS
    )


def recorded_sdk_cost_usd(result_message: Any | None) -> tuple[float | None, bool]:
    """Return the SDK query estimate and whether that estimate is complete.

    A missing or invalid ``total_cost_usd`` is recorded as incomplete. This
    never invents ``0`` and is not a generation stop condition.
    """
    cost = (
        getattr(result_message, "total_cost_usd", None)
        if result_message is not None
        else None
    )
    complete = (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    )
    if complete:
        return float(cost), True
    return None, False
