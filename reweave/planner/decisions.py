# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Planner decision batches, separate from VLA action chunks and horizons."""

MAX_TOOLS_PER_DECISION = 5

DECISION_INSTRUCTIONS = f"""Choose 1–{MAX_TOOLS_PER_DECISION} tool calls per decision using current evidence/budget. When confidence is high and a useful short sequence is already determined, prefer multiple calls to reduce decision latency; do not default to one call. Calls execute serially in returned order. Batch only already-determined arguments: never reference an earlier call's pending output/observation ID. If feedback is needed, return fewer calls and replan afterward. Each call costs its normal time/budget and delays reconsideration. Explicit error/cancellation/interruption skips remaining calls and returns feedback; residuals, tolerances and target_reached=false alone do not. Unknown physical outcomes are fatal: never replay them. Put finish last, justified by already observed evidence, never assumed queued results. Over-limit batches or calls after finish are rejected in full, never truncated. Counts are tool calls, not VLA chunks or action steps."""


def decision_error(
    names: list[str], *, image_contract=None, image_limit=10
) -> str | None:
    """Validate the whole provider batch before any call can execute."""
    if len(names) > MAX_TOOLS_PER_DECISION:
        return f"A decision may contain at most {MAX_TOOLS_PER_DECISION} tool calls; this batch has {len(names)}. Nothing was executed."
    if "finish" in names[:-1]:
        return "finish must be the final tool call of a decision. Nothing was executed."
    if "write_workflow" in names and len(names) != 1:
        return "write_workflow must be the only call in a decision; include arguments to run the written W immediately. Nothing was executed."
    if image_contract and image_contract.get("observation") is not None:
        unknown = [
            name
            for name in names
            if name in image_contract["reads"] and image_contract["reads"][name] is None
        ]
        if unknown:
            return f"Image-return bounds are undeclared for {unknown}. Nothing was executed."
        read_images = sum(image_contract["reads"].get(name, 0) for name in names)
        count = image_contract["observation"] + read_images
        if read_images and count > image_limit:
            return (
                f"This batch can return {count} distinct images including the current "
                f"observation; the single-request ceiling is {image_limit}. "
                "Request fewer image reads and replan. Nothing was executed."
            )
    return None


def tool_failed(result):
    """Explicit execution failures require replanning; residuals do not.

    ``ik_error`` is a rejected plan (planner failure or over-budget path): no
    action ran, and later queued calls were planned against its outcome.

    Keep this classifier self-contained: baseline L freezes its source with the
    loop, so API and plugin execution use the same failure semantics.
    """
    if not isinstance(result, dict):
        return False
    if (
        result.get("error")
        or result.get("ik_error")
        or result.get("interrupted")
        or result.get("status") in ("failed", "cancelled", "interrupted")
        or result.get("active_execution_outcome")
        in ("failed", "cancelled", "interrupted")
    ):
        return True
    if tool_failed(result.get("capability_result")):
        return True
    return result.get("execution_performed") is False and tool_failed(
        result.get("feedback")
    )
