# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Shared protocol for high-level reasoning backends."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Any, Literal, Protocol

from reweave.dashboard.interaction import DashboardInteractionPort
from reweave.tools.toolkit import Toolkit

#: MCP namespace prefix for Harness tools (``mcp__<server>__<tool>``).
#: Toolkits expose plain tool names; planners add/strip this prefix.
MCP_TOOL_PREFIX = "mcp__harness__"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")

SUCCESS = "success"
PHYSICAL_FAILURE = "physical_failure"
PLANNER_TIMEOUT = "planner_timeout"
API_ERROR = "api_error"
AGENT_EXECUTION_FAILURE = "agent_execution_failure"
ACTION_OUTCOME_UNKNOWN = "action_outcome_unknown"

TerminationClass = Literal[
    "success",
    "physical_failure",
    "planner_timeout",
    "api_error",
    "agent_execution_failure",
    "action_outcome_unknown",
]
TERMINATION_CLASSES: frozenset[str] = frozenset(
    {
        SUCCESS,
        PHYSICAL_FAILURE,
        PLANNER_TIMEOUT,
        API_ERROR,
        AGENT_EXECUTION_FAILURE,
        ACTION_OUTCOME_UNKNOWN,
    }
)
TASK_EVIDENCE_COMPLETE_CLASSES: frozenset[str] = frozenset({SUCCESS, PHYSICAL_FAILURE})
PLANNER_ABORT_CLASSES: frozenset[str] = (
    TERMINATION_CLASSES - TASK_EVIDENCE_COMPLETE_CLASSES
)


def classify_episode_termination(
    *,
    planner_termination_class: str | None,
    official_success: bool,
) -> TerminationClass:
    """Assign the single episode termination class.

    The planner owns abort classes: wall-clock ``planner_timeout``, fixed
    program ``agent_execution_failure``, and provider/infrastructure
    ``api_error``. Confirmed official success takes precedence over these aborts;
    the original planner error remains separate diagnostic evidence. An
    unacknowledged submitted action always remains ``action_outcome_unknown``,
    even if a later independent success query returns true: the missing action
    acknowledgement must not be erased or replayed. Completeness of task evidence
    is derived from this class, not stored beside it.
    """
    if planner_termination_class is not None:
        if planner_termination_class not in PLANNER_ABORT_CLASSES:
            raise ValueError(
                f"unknown planner termination class: {planner_termination_class!r}"
            )
        if planner_termination_class == ACTION_OUTCOME_UNKNOWN or not official_success:
            return planner_termination_class  # type: ignore[return-value]
    return SUCCESS if official_success else PHYSICAL_FAILURE


def require_termination_class(termination_class: str) -> TerminationClass:
    """Reject any token outside the episode termination vocabulary."""
    if termination_class not in TERMINATION_CLASSES:
        raise ValueError(f"unknown termination class: {termination_class!r}")
    return termination_class  # type: ignore[return-value]


def task_evidence_is_complete(termination_class: str) -> bool:
    """True only for a finished official success or physical/horizon failure."""
    return (
        require_termination_class(termination_class) in TASK_EVIDENCE_COMPLETE_CLASSES
    )


def episode_is_scored(termination_class: str) -> bool:
    """True when the episode counts in the end-to-end rate.

    Timeouts and author exhaustion remain scored system failures. Provider
    ``api_error`` stays unscored infrastructure. Unacknowledged submitted actions
    count conservatively as failures and are not automatically replayed.
    """
    return require_termination_class(termination_class) != API_ERROR


def strip_mcp_prefix(name: str) -> str:
    """Return the bare tool name, dropping the MCP namespace if present."""
    return name.removeprefix(MCP_TOOL_PREFIX)


def accepted_finish_result(content: Any) -> dict[str, Any] | None:
    """Read the Host's finish sentinel from API JSON or MCP content blocks.

    Transport completion and submitted arguments never establish acceptance.
    """
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if isinstance(content, dict):
        if content.get("error") or content.get("isError"):
            return None
        if content.get("_finish") is True:
            return content
        content = content.get("content")
    if isinstance(content, list):
        results = [
            accepted_finish_result(block.get("text"))
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "TextBlock"}
        ]
        accepted = [result for result in results if result is not None]
        if len(accepted) == 1:
            return accepted[0]
    return None


class PlannerResult:
    """Result returned by a planner invocation."""

    __slots__ = (
        "finish_result",
        "messages",
        "stats",
        "error",
        "termination_class",
        "api_error_retryable",
    )

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
        termination_class: str | None = None,
        api_error_retryable: bool | None = None,
    ):
        """Initialize a serializable planner result.

        ``messages`` and ``stats`` are the observer history collected so far,
        including on abort. ``termination_class`` is the planner-owned abort
        class and must be paired with ``error``. ``None`` means the planner
        finished without aborting; the CLI then assigns ``success`` or
        ``physical_failure`` from official task evidence.
        """
        if (error is None) != (termination_class is None):
            raise ValueError(
                "PlannerResult.error and termination_class must be set together"
            )
        if (
            termination_class is not None
            and termination_class not in PLANNER_ABORT_CLASSES
        ):
            raise ValueError(
                f"planner cannot assign episode outcome {termination_class!r}"
            )
        if (termination_class == API_ERROR) != (api_error_retryable is not None):
            raise ValueError("only API errors require explicit retry eligibility")
        if api_error_retryable is not None and type(api_error_retryable) is not bool:
            raise ValueError("API retry eligibility must be a boolean")
        self.finish_result = finish_result
        self.messages = messages or []
        self.stats = stats or {}
        self.error = error
        self.termination_class = termination_class
        self.api_error_retryable = api_error_retryable


class Planner(Protocol):
    """A planner solves a task by conversing with an LLM/VLM backend.

    It is given one system prompt, one initial user message, and a set of
    tool definitions.  It returns a ``PlannerResult`` after the task is
    finished or the turn budget is exhausted.
    """

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            toolkit: The full :class:`~reweave.tools.toolkit.Toolkit`
                (common + robot tools). Backends derive ``tools_spec`` via
                ``toolkit.get_tools_spec()`` and dispatch calls via
                ``toolkit.execute_tool()``.
            max_turns: Maximum LLM turns before giving up.
            input_queue: Optional queue of user-typed lines for interactive steering.
            dashboard_interaction: Optional Dashboard interaction channel.

        Returns:
            ``PlannerResult`` with finish status, conversation transcript,
            token-usage stats, optional abort, and the planner termination
            class.
        """
        ...


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def build_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    model: str | None = None,
    planner_timeout_s: int | None = None,
    reasoning_effort: str | None = None,
    native_provider: str = "codex",
    codex_auth_mode: str = "auth",
    native_service_tier: str = "default",
    adaptation_mode: str = "static",
    online_workflows: bool = False,
):
    """Construct the sole W/C/L execution path with its selected native provider."""
    if planner_type != "native":
        raise ValueError(f"unsupported planner backend: {planner_type}")
    from reweave.planner.native_loop import NativeBundlePlanner

    return NativeBundlePlanner(
        output_dir=output_dir,
        provider=native_provider,
        codex_auth_mode=codex_auth_mode,
        service_tier=native_service_tier,
        model=model,
        reasoning_effort=reasoning_effort or "medium",
        adaptation_mode=adaptation_mode,
        online_workflows=online_workflows,
        timeout_s=planner_timeout_s
        if planner_timeout_s and planner_timeout_s > 0
        else None,
    )
