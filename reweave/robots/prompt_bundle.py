# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Prompt bundle dataclass for robot-contributed LLM prompts.

Lives in :mod:`reweave.robots` so each robot's
``prompt_bundle.py`` (e.g. :mod:`reweave.benchmarks.libero.prompt_bundle`)
can import it without depending on the RPC transport layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from reweave.prompt.utils import PromptNode, format_prompt

PromptFactory = Callable[..., PromptNode]


@dataclass(frozen=True)
class PromptBundle:
    """Python-defined prompt factories for one robot."""

    system: PromptFactory
    user: PromptFactory

    def render(
        self,
        variant: str,
        *,
        variables: Mapping[str, object] | None = None,
    ) -> str:
        """Render one prompt variant (``"system"`` or ``"user"``)."""
        prompt = getattr(self, variant)(variables)
        return format_prompt(prompt, variables=variables)
