# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Replay recorded C inputs through production construction and Host validation.

Public observations stay historical. No model, robot, or fabricated read response
is available. A new retrieval requirement is explicitly untested, never passed.
Full W/C/L orchestration remains covered by the separate interaction suite.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from reweave.evolution.authoring.candidate import read_proposal, read_references
from reweave.evolution.evidence.records import restore_record
from reweave.planner.native_context import prepare_observation, project_native_feedback
from reweave.runtime.execution import build_context
from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits
from reweave.runtime.model_input import unique_image_count

_EVENTS = "harness_context/bundle_events.jsonl"


def recorded_feedback_sources(context, events):
    """Recover read ownership from old Host lineage, never from an operation name.

    Older workflow scopes inherited the caller's planner_call_id. Matching that
    ID *and* its parent program to the workflow start proves an internal read;
    a different ID belongs to a model decision inside the workflow. Unknown
    ownership stays unknown and retains its pixels. Only a replay copy changes.
    """
    context = deepcopy(context)
    workflows = set()
    operations = []
    for event in events:
        if event.get("kind") == "workflow_started":
            workflows.add((event.get("planner_call_id"), event.get("program_call_id")))
        elif event.get("kind") == "operation_settled":
            source = event.get("feedback_source")
            if source is None and "planner_call_id" in event:
                source = (
                    "program"
                    if (
                        event["planner_call_id"] is None
                        or (
                            event["planner_call_id"],
                            event.get("parent_program_call_id"),
                        )
                        in workflows
                    )
                    else "model"
                )
            operations.append(
                (event.get("operation"), source, event.get("program_call_id"))
            )
    for message in context.get("messages", []):
        for part in message.get("parts", []):
            content = part.get("content")
            text = content[0] if isinstance(content, list) and content else content
            if not isinstance(text, str):
                continue
            try:
                value = json.loads(text)
            except ValueError:
                continue
            if not isinstance(value, dict) or not isinstance(
                value.get("feedback"), dict
            ):
                continue
            index = value.get("history_index")
            if type(index) is not int or not 0 <= index < len(operations):
                continue
            operation, source, owner = operations[index]
            feedback = value["feedback"]
            if operation != feedback.get("operation") or source is None:
                continue
            feedback.setdefault("feedback_source", source)
            if owner is not None:
                feedback.setdefault("program_call_id", owner)
            if isinstance(content, list):
                content[0] = json.dumps(value)
            else:
                part["content"] = json.dumps(value)
    return context


def recorded_context_inputs(record):
    """First/last successful boundary and every recorded failed input, once.

    Selection is deterministic and references exact original row positions.
    It covers startup/null state, the latest accumulated state, and known errors
    without replaying every repeated successful decision of an unchanged parent.
    """
    original = next(
        (item for item in record.evidence if item.path.endswith("/" + _EVENTS)),
        None,
    )
    if original is None:
        return
    original.verify()
    source = Path(original.path)
    first = last = None
    failed = []
    with source.open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            kind = event.get("kind")
            if "input" not in event:
                continue
            if kind == "context_built":
                if first is None:
                    first = index
                last = index
            elif kind == "context_rejected" or (
                kind == "context_build_failed"
                and event.get("error_type") != "ModelInputRejected"
            ):
                failed.append(index)
    selected = set(failed) | ({first, last} - {None})
    lineage = []
    with source.open() as stream:
        for index, line in enumerate(stream):
            raw = json.loads(line)
            if index in selected:
                event = restore_record(raw, source.parent)
                yield {
                    "receipt_id": record.record_id,
                    "original": {**original.model_dump(), "pointer": f"/{index}/input"},
                    "original_event": event["kind"],
                    "input": recorded_feedback_sources(event["input"], lineage),
                }
            if raw.get("kind") in {"workflow_started", "operation_settled"}:
                lineage.append(
                    {
                        key: raw[key]
                        for key in (
                            "kind",
                            "operation",
                            "planner_call_id",
                            "program_call_id",
                            "parent_program_call_id",
                            "feedback_source",
                        )
                        if key in raw
                    }
                )


async def check_recorded_contexts(workspace, candidate, contract, *, case_results=None):
    """Check current parent's recorded C boundaries; retain all replay outcomes."""
    proposal = read_proposal(workspace)
    parent = next(
        r
        for r in read_references(workspace)
        if r.record_id == proposal.parent.bundle_id
    )
    limits = EpisodeLimits.model_validate(contract["episode_limits"])
    reports = []

    async def forbidden(*args):
        raise PermissionError("historical context replay cannot perform live I/O")

    for record in proposal.evidence:
        if (
            record.request.bundle_id != proposal.parent.bundle_id
            or record.request.execution != proposal.task_protocol.execution_id
        ):
            continue
        for case in recorded_context_inputs(record):
            report = {
                "scope": "historical_context",
                "physical_execution_tested": False,
                "full_loop_tested": False,
                "receipt_id": case["receipt_id"],
                "original": case["original"],
                "original_event": case["original_event"],
                "validation_status": "running",
                "new_reads": [],
            }
            reports.append(report)
            if case_results is not None:
                case_results.append(report)
            context = deepcopy(case["input"])
            if (
                context.get("state")
                and candidate.context.program != parent.context.program
            ):
                # A different C starts its own private state in a new episode.
                # Requiring it to accept an old program's internal state would
                # impose an artificial compatibility constraint on evolution.
                report.update(
                    validation_status="not_tested",
                    reason="Historical private C state belongs to a different context program.",
                )
                continue
            context["assets"] = candidate.context.model_dump(mode="json")
            context["tool_effects"] = {
                name: operation["effect"]
                for name, operation in contract["operations"].items()
            }
            # Historical observations/counters remain exact; absence stays unknown.
            context.setdefault("controller_budget", None)
            kernel = EpisodeKernel(
                candidate,
                operations=contract["operations"],
                gateway=forbidden,
                cancel=forbidden,
                emit=lambda event: None,
                episode_limits=limits,
            )

            async def retrieve(method, name, arguments):
                if contract["operations"].get(name, {}).get("effect") != "read":
                    raise PermissionError("C replay only allows public read evidence")
                report["new_reads"].append(
                    {"method": method, "name": name, "arguments": arguments}
                )
                raise ValueError(
                    "this historical input has no recorded fixture for the new read"
                )

            try:
                if {
                    "observation",
                    "physical_budget",
                    "remaining_wall_time_s",
                    "objective",
                } <= context.keys():
                    context = prepare_observation(project_native_feedback(context))
                    report["native_input_contract"] = context.get(
                        "native_input_contract", "adapter-specific"
                    )
                else:
                    report["native_input_contract"] = (
                        "historical-input-without-complete-snapshot"
                    )
                prepared = await build_context(
                    kernel,
                    candidate.context.program,
                    context,
                    retrieve,
                    context["tools"],
                    native=True,
                )
                report["request_images"] = unique_image_count(prepared["messages"])
            except Exception as error:
                report["error"] = {"type": type(error).__name__, "message": str(error)}
                if report["new_reads"]:
                    report["validation_status"] = "not_tested"
                    continue
                report["validation_status"] = "failed"
                raise ValueError(
                    f"historical C input rejected at {case['original']['path']}"
                    f"{case['original']['pointer']}: {error}"
                ) from error
            report["validation_status"] = "passed"
    return reports
