# Copyright 2026 PyRUA-Lean Contributors.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
#
# Prompt text adapted from PyRUA-Lean src/pyrualean/arms.py PROGRAM_CONTRACT,
# IMAGE_NOTES and NVlabs/ASPIRE iterative-debugging/subagent-prompt.md.
# Interface substitutions and omissions are documented in
# docs/method/online_workflows.md. This ReWeave adaptation is behaviorally unvalidated.
"""Opt-in online authoring of ordinary W assets, using the existing W contract."""

from __future__ import annotations

from reweave.evolution.bundle import WorkflowSpec


def write_workflow_tool_spec():
    schema = WorkflowSpec.model_json_schema()
    schema["description"] = "An ordinary W, authored privately during this episode."
    schema["properties"]["program"]["description"] = (
        "The ordinary executable PythonProgram, written online."
    )
    definitions = schema.pop("$defs", {})
    return {
        "name": "write_workflow",
        "description": (
            "Write or revise one episode-private task-family W using the ordinary "
            "WorkflowSpec contract. Supply arguments to execute that exact W immediately "
            "in this call; omit arguments to only publish it. Reuse it afterward through "
            "workflow_<workflow_id>. Call this tool alone in a decision. Global W, C, L, "
            "fixed primitives and all episode budgets remain protected. Writing a W is "
            "not evidence of task success or permission for cross-episode inheritance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow": schema,
                "arguments": {
                    "type": "object",
                    "description": "When present, run the newly written W immediately with these arguments.",
                },
            },
            "required": ["workflow"],
            "additionalProperties": False,
            "$defs": definitions,
        },
    }


ONLINE_WORKFLOW_INSTRUCTIONS = """# Online W / Code-as-Policy contract

Write a complete Python W that defines run(runtime, context). The Host runs it
through the same WorkflowSpec, sandbox and Kernel as offline-authored W. Pass
that WorkflowSpec to write_workflow; include arguments to write and run it in
one call, or omit arguments to publish it for later workflow_<id> calls.
Every invocation starts from a fresh Python namespace; helper functions live
in the W source tree. W source persists within this episode. Continuation state
persists only through the existing yielded/state/continuation_id contract.

Decisions inside the program are made in code from documented return values.
Prefer programs that carry out several steps with in-code checks (localization,
motion result, gripper feedback), and look when a decision depends on the picture.
Write short W invocations when you need to look before deciding, and loops when
you do not. The original direct tools and already-written W remain available.
An explicit runtime.plan call is an ReWeave extension: it runs the same model/session
and C, consuming the shared controller budget. Code itself does not call a model.

Use runtime.observe, runtime.act and declared workflow dependencies, with the
actual APIs below. Return the usual WorkflowResult: status, summary, output,
and state when yielding. Return the public results needed for the next decision.
ReWeave's C supplies the current observation and selects returned images; this is
not PyRUA-Lean's print/display-only feedback or persistent notebook namespace.

A program that raises stops there; the episode continues with the robot where
it stopped, unless a fixed deadline, infrastructure failure or unknown physical
outcome ends it. Fix the smallest source defect using its recorded error.
Distinguish program, provider, perception, planning, execution and task-completion
failures. Every task-dependent object choice and target must be derived at runtime
through the allowed public APIs; do not copy scene coordinates or seed dispatch
into code. Debug generic offsets/tolerances, not individual seed identities.

This tool writes the current episode's private W (called task-family W in the
repository's ownership metadata). It can contain complete programs, not just
parameter changes. C/L editing still follows the separate adaptation mode.
Global W and task-specific facts remain read-only.
All physical actions use the Kernel; do not use the native shell to reach robot
services. Original shared physical/model/time limits remain cumulative. A local
completed W is not the benchmark's success verdict. Online W is episode-private;
the offline author reviews its source and measured evidence before inheritance.
"""


ONLINE_WORKFLOW_AUTHOR_GUIDANCE = """\n## Online-authored W evidence

This condition enables online W authoring. Online code is an ordinary
WorkflowSpec. read_trace(view="online_workflows") indexes write/run events and
exact adaptation source references. Diagnose failure modes rather than seed
identities: distinguish program errors, provider failures, perception, planning,
execution and incomplete task subgoals from the actual traces and observations.
Fix the smallest source defect; task-dependent geometry must come from the
allowed public perception APIs. Generic offsets/tolerances may be tuned.

Integrate useful W into the complete candidate and run the existing code and
interaction admission. Keep every measured revision and its regressions. Select
one complete measured candidate; do not call an untested combination of pieces
a measured solution. An online revision executed midway through an episode was
not independently evaluated from its initial state. Shared knowledge follows
proposal, verification, then promotion; a successful transmission or one scene
does not establish transfer. Existing editable scopes, development/test splits,
graph selection, budgets and experiment ownership remain authoritative.
"""
