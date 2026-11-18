# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Offline author guidance; complete contracts remain readable on disk."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from reweave.evolution.authoring.candidate import (
    SCOPE_DIRECTORIES,
    authoring_scope_contract,
)
from reweave.memory.experience import EXPERIENCE_FORMAT

if TYPE_CHECKING:
    from reweave.evolution.contracts import Proposal


_METHOD_ROLES = {
    "C": """- C shapes the evidence, instructions, skills and registered tools available at a
  model decision. Use it when relevant information is missing, poorly organized,
  misleading or not usable for the current judgment. Better grounding and feedback
  interpretation can improve choices and avoid repeated reasoning or recovery.
  C does not itself execute physical work or own the episode loop; more guidance
  can add tokens, conflicts or distraction. Check what actually reaches the model
  and whether its decisions change, rather than crediting text merely for existing.""",
    "W": """- W implements callable, reusable computation or feedback-dependent local work.
  Use it when inputs, local transitions and stopping conditions can be expressed
  as a useful executable procedure. A Planner-selected W can replace repeated root
  decisions with bounded observe/act/verify work under its declared permissions
  and shared budgets. Internal model calls and extra sensing still cost resources;
  an unused helper or a helper adding calls need not improve efficiency. Preserve
  necessary feedback, ground parameters in current evidence, and return partial
  progress, uncertainty or failure when further judgment is needed. Check actual
  invocation, behavior and usable returned results, not just tool availability.""",
    "L": """- L controls the episode's planning, execution, observation and feedback boundaries.
  Use it when the timing or grouping of decisions and settled outcomes causes
  stale actions, lost feedback, redundant replanning or poor continuation. Suitable
  boundaries can reduce unnecessary model calls and preserve useful feedback.
  Too much batching can miss changed conditions; too frequent replanning can add
  latency. L must retain the existing Planner authorization and terminal contracts.
  Check that the relevant boundary is reached and changes subsequent behavior,
  rather than crediting an unexercised branch.""",
}


_METHOD_DEFAULTS = {
    "parent_selection": "For a mutation, the parent is selected by confirmed official development success first,\nthen mean retained graph progress. At equal success, reverting to an older round\nrequires the configured progress margin over the latest evaluated round.",
    "history_access": """Every historical implementation remains readable through history_originals.json,
including unsuccessful versions. context/EVOLUTION_HISTORY.md indexes measured
results, parentage and changes; original diagnoses and SKILL.md notes are available
on demand. Earlier explanations are revisable beliefs, not established causes.
The batch uses frozen evidence; siblings' drafts and live online notes are private.
""",
    "success_extraction": "Keep mutation changes in task-family; existing global assets are read-only.",
    "component_choices": "Choose components by the supported mechanism, without an edit quota. "
    "Editable components are permissions, not required changes; a change to one component is sufficient.\n\n"
    + "\n".join(_METHOD_ROLES.values()),
    "online_revisions": "necessary targeted online W/C/L revision or a specific missing historical fact.",
    "consolidation_retention": "",
    "feedback_validation": "",
    "task_model": "",
}

# Sections delivered only to the configuration that uses them.
_CONSOLIDATION_FIELDS = {
    "success_extraction": """During a success-consolidation assignment, factor reusable task-bound Workflow logic
and C/L into global implementations, retaining task bindings in task-family.
An empty W stays empty. Preserve the winning behavior and its evidence; this
reorganization has no measured score until separately evaluated.""",
    "consolidation_retention": " Success consolidation may retain already organized content when justified.",
}

_FEEDBACK_VALIDATION = """Feedback validation is measured only after dev qualification and contributes to
later learning; it is not held-out evidence. """
_AUTHOR_METHOD_TEMPLATE = """## Objective and evidence

Improve one complete candidate under the objective and budgets in context/input.json.
Read the actual task instructions and named current rules; historical verdicts
retain their original criteria. Official success across the development conditions
is the objective. Successes from different candidates cannot be combined into one
candidate's score. Infrastructure-missing outcomes remain unknown.
{{feedback_validation}}
{{history_access}}
{{task_model}}
## Choose and implement

Start from the observed execution problem and inspect the original model inputs,
choices and feedback needed to understand it. For physical claims, inspect relevant
video or images alongside the trace. Check authoritative task constraints before
turning a failed action into a prohibition. Keep observed facts and causal hypotheses
distinct; record a material unresolved ambiguity in plan.json.

Use your own diagnosis to choose a coherent change within the editable components.
There is no required repair strategy, checklist of hypothetical failures or minimum
amount of investigation. Test the changed interface and a relevant failure path;
expand investigation when actual evidence or validation requires it. Consider the
remaining task and budgets when choosing the repair. A short C edit, an existing W
revision or a different L boundary can each be a complete candidate.

Integrate the change into the existing implementation; remove superseded rules
and conflicting patches. Preserve demonstrated behavior and its conditions, not
all inherited code. Geometry and task bindings must be grounded in current public
observations; no task/seed dispatch, copied action scripts or privileged online inputs.
{{success_extraction}}

Keep plan.json concise: what changed, why, actual checks and useful
uncertainty. SKILL.md is optional for additional reusable lessons, not a second report. Use
plan.probes only for useful public event activation/effect signals; these are
observations of normal evaluation, not additional model or physical experiments.
Original evidence, interface contracts and code checks remain available on demand.
"""


# Engineering reference shared byte-for-byte by every author arm; method
# layers (graph diagnosis, history use) are delivered separately.
_ENGINEERING_TEMPLATE = """## Engineering reference

{{candidate_layout}}

{{component_boundary}}

### Component roles

{{component_choices}}

### Online agent

Do not reconstruct the provider conversation or routinely reload already-delivered
images to maintain history. Each ordinary observation includes three camera views,
but additional observations must not be copied as whole three-view groups by
default: select only the views that address a concrete uncertainty or comparison,
and explain their relevance. C output may contain at most 10 distinct images,
including images omitted by image_selection. This is a ceiling, not a target.
Choose evidence deliberately; do not fill the budget with an automatic
history window or replay a long W's action images merely to preserve memory.
Host messages contain current pixels and model-requested image reads; intermediate
physical feedback and other programs’ internal control reads keep text and archive references.
source_messages retains the complete
newly settled feedback for deliberate C selection, not accumulated session history.
Preserve actionable concise feedback and report infrastructure failures with their
original input and receipt.
Default current frames belong to current_camera_observation.observation_id rather
than an older state-changing feedback's observation_id. Deliberately included
historical or transformed frames must carry their own source observation identity
and mapping; do not relabel them as the current observation.

For native online decisions, the episode's fixed system/task/tool instructions
are loaded from its clean workspace AGENTS.md. Raw traces, full joint vectors, static hardware configuration,
per-joint telemetry and audit originals are experiment evidence outside that
workspace, not routine model input. Do not copy them back into C messages or
duplicate a parsed tool result as its original JSON string. Preserve actionable
errors, known outcome, current pose/gripper state and explicitly requested
geometry. Fixed guidance or tool-contract revisions are delivered at settled
boundaries; unchanged text is not another per-turn message.

The online native session is a task decision model, not a repository development
job. Ordinary turns consume the supplied messages, images and loaded task
experience and return structured proposals directly. Do not require repository
onboarding, development-skill discovery, routine log/source rereads or coding
preflight before task decisions. Native file tools remain available only for a
{{online_revisions}}
Online task-family experience recording and offline review remain required.

### Interfaces and execution

Exact W/C/L interfaces and execution rules are in context/runtime.schema.json;
fixed operations and output schemas are in context/gateway.json and
context/capability_catalog.json. context/examples.json demonstrates interface
formats, not a recommended algorithm. Use the current registered contract to
determine available capabilities; an older episode's missing tool does not mean
the current deployment lacks it. A source implementation or running model service
does not establish that the tool is configured and registered for this candidate.
For geometric proposal tools, preserve source observation identity, coordinate
frames, quaternion order and already-applied calibration transforms when composing
them with motion. Model scores or limited geometric checks do not establish path
feasibility, possession or task success.

Root actions must consume the unchanged call from runtime.plan(); local failures
must remain feedback for subsequent planning. Preserve tool-call/return pairing,
physical ownership, cancellation and shared budgets. The fixed runtime, models,
evaluator and execution limits are not editable. Report a foundation fault as such
rather than hiding it with a candidate workaround. Do not edit live episodes,
install dependencies, write across proposals or invent validation evidence.

### Reading evidence

Read identical component bytes once, while considering their different executions.
Keep native investigation outputs targeted: request tool names or one relevant
schema, not complete tool catalogs. Read source files and exact original JSON
pointers needed for the current question; do not print whole Bundles, contract
cases, schemas or image-bearing snapshots merely to locate a field. Use indexed
image/video tools for pixels rather than emitting Base64 as text. Forward native
MCP text/images directly instead of JSON-encoding their complete response wrappers
again. Keep originals on disk; narrow oversized or truncated reads to the missing
fields rather than treating truncation as complete evidence. These are output
selection rules, not permission to omit evidence that affects the diagnosis.

### Verify and leave useful feedback

Run validate_candidate(phase="code") while implementing the executable change.
Before returning, run validate_candidate(phase="interaction") and repair failures
in this same turn. It previews the production C/W/L interaction checks as well as
component cases; the Host repeats final admission. Reuse existing fixtures and
extend only coverage affected by the change. Use a few discriminating cases for
the affected interface and failure path. See DEVELOPMENT.md for .scratch/tests/interaction.json,
including recoverable W failure coverage when applicable. Synthetic responses test
interfaces and feedback, not real-model adoption, physical success or preservation
of historical trajectories. An old trajectory's suffix is not the measured result
of a newly changed action. Only actual rollout outcomes assess task improvement.
For feedback-dependent mechanisms, test the original observed failure wording and
the first decision where that feedback is visible. A judgment and action can share
one Planner response: guidance added by the next C call cannot affect an action
already proposed in that response. Test this ordering and mechanism retirement,
not just a fixture whose wording already matches the implementation.
Do not repeat successful checks on unchanged code or weaken checks to obtain a pass.

Use candidate/plan.json for a concise diagnosis, actual change and checks.
No pre-edit report or hypothetical failure checklist is required. The Host derives
changed components and enforces write authority from actual candidate bytes.
Optional plan.probes can identify relevant activation/effect events in normal
evaluation; they do not launch extra trials or certify causal improvement.
A mutation candidate must differ from every reference.{{consolidation_retention}}
Add candidate/SKILL.md only for useful lessons not already in the plan.
It may be empty or absent; offline notes do not change C or the executable Bundle.
Keep temporary analysis/tests/media in .scratch/ and only deployable assets in
candidate/. Native tools are available within the workspace boundary; no approval
requests. Finish when the candidate and required checks are complete.
"""

_SHORT_ROLES = {
    "C": "- C shapes usable evidence, instructions, skills and registered tools at decisions.",
    "W": "- W implements callable feedback-dependent work.",
    "L": "- L controls planning, execution and feedback boundaries.",
}


_SHORT_DEFAULTS = {
    "component_choices": "Choose C, W, L or a coherent combination, without quotas. "
    "Editable components may remain unchanged; a change to one component is sufficient:\n\n"
    + "\n".join(_SHORT_ROLES.values()),
    "consolidation_entry": "",
}

_MUTATION_TEMPLATE = """# Develop one complete Harness

Improve ONE complete candidate under context/input.json's objective. Read
task_instructions.json and DEVELOPMENT.md; consult affected runtime.schema.json
interfaces as needed. {{consolidation_entry}}candidate/ already contains the parent.
Use your own diagnosis from supplied evidence; no investigation sequence is assigned.
{{component_choices}}
Preserve fixed runtime, tool, observation and budget contracts. Editable L may change
when to plan, execute and return feedback. Online provider history stays native;
C sends incremental context with at most 10 images. No task/seed dispatch,
privileged inputs, physical trials, external model probes or service launches.

Write concise diagnosis/change/checks in plan.json; SKILL.md is optional for additional
lessons. Run validate_candidate(phase="code") while implementing, then run
validate_candidate(phase="interaction") on .scratch/tests/interaction.json and
repair failures before returning. Reuse existing fixtures and update affected
coverage; the Host repeats final admission. Keep scratch work in .scratch/;
finish once the candidate and checks are complete. Each assignment is independent;
only correction/recovery continues this session.
"""


PLUGIN_CONTRACTS = {
    "W": """W is a callable program: runtime.observe/act invoke fixed operations;
runtime.plan(objective=None) requests an LLM decision through the same C used
everywhere. Declare model:plan permission when needed. Public settled tool feedback,
including images and history, is supplied by the runtime; W does not assemble a
second model request or copy image-bearing history into an argument.
runtime.control('workflow:<id>', {'arguments': ..., 'continuation_id': ...})
invokes declared dependencies. Local loops, verification, parameters and model
judgment are editable. Runtime operations require declared permissions. Optional
task:start/join/cancel permissions enable concurrent local callables. Capability metadata is
not an automatic applicability guard; implement any necessary checks in W.
Reuse parameterized global logic where it exists; task wrappers may call it via
declared W dependencies. Do not duplicate implementations or require paired W versions.
A verifier or critic can be an ordinary W: inspect public evidence, optionally
call runtime.plan, and return judgments or repair suggestions in its declared
output. It may check a local subgoal or the whole episode; no physical action
is required, and read-only checks can declare physical_calls=0. Keep its prompt
and checking logic in its program. A completed check is not a satisfied goal:
represent the judgment and insufficient evidence explicitly in output, separately
from WorkflowResult.status. Its judgment never replaces the fixed evaluator.
""",
    "C": """C is build(runtime, context) plus instructions.md and
SKILL.md assets. The online backend is a persistent native Codex session. One
logical episode has one provider session, including root and W-local decisions.
Native history and compaction belong to that provider; do not rebuild a stateless
API conversation, append an initial/recent image window, or reread records to
maintain history.
PydanticAI-shaped message dictionaries are an internal Host encoding, not evidence
that the deployed agent is a stateless API caller. Do not infer history-replay
requirements from the schema format or inherited parent source. C private state
may hold reasoning state and deliberate keyframe references, but is not a second
conversation or a replacement for native compaction.

By default the Host supplies the current observation and not-yet-delivered settled
feedback on every plan(). This is an incremental starting point, not a restriction
on C's evidence choices. C freely organizes text and images, selects useful camera
views, retrieves admitted public evidence, and deliberately re-emphasizes relevant
historical keyframes or intermediate observations when needed. There is no fixed
camera set or current-only rule. Although an ordinary observation contains three
camera views, do not blindly attach all three views of additional observations.
Choose only the relevant extra images needed for the current question. Identify
each image's source camera/observation,
time and any pixel transform; historical geometry must not be passed off as current.
Do not routinely rebuild provider history or fill an initial/recent image window.
A short delta does not mean the agent forgot earlier observations.

C may also select task-family skills, revise task guidance and retain private
reasoning state. Return instructions, messages, tools, state and optionally
image_selection. The complete C output may contain at most 10 distinct images,
including any hidden by image_selection; identical byte payloads count once.
This ceiling applies per request, not to the session or offline author history.
Fewer or zero images are allowed when
appropriate. Deployment input budgets also apply. Host rejects oversized C output
before submitting that request and never silently clips evidence. A known native
input-budget rejection is recorded as unsent. In adaptive mode, Host sends a
text-only correction turn in the same session, discards every task proposal from
that turn, then rebuilds C at the next settled boundary. C receives
context_rejection (status/error/instruction and correction text) until valid
request delivery. Use it to correct evidence selection; adaptive native edits may
repair task-family C. Static source remains read-only and stops for offline repair
without an ineffective correction turn. Correction turns use the
same decision/time budgets; repeated failure never resets counters or replays
physical actions. If another rejection occurs with unchanged C after correction,
Host stops the known program failure for offline repair instead of spending more
correction turns on identical code. Unknown physical outcomes remain terminal. The naive baseline
simply forwards current observation and new feedback; do not confuse this baseline
with the expressive limit of C. Tools may only select unchanged permitted
interfaces. C may retrieve admitted public read evidence but cannot act, spawn or
call a model recursively. Changing C never changes the official evaluator,
physical owner, resource accounting or source history.

The clean episode workspace AGENTS.md carries fixed system/task/tools and selected
experience. Later native turns add current observations and newly settled concise
feedback by default. C may deliberately emphasize relevant evidence. Fixed
guidance changes are delivered by Host at settled boundaries.
W/C/L edits activate only there. Keep raw traces, full telemetry and audit/video
outside online input. Use admitted originals for local archived-input
experiments; preserve every source receipt and discard all physical proposals.
""",
    "L": """L is a replaceable run(runtime, context) program owning the ENTIRE episode loop.
It may use snapshot(), plan(objective=None), execute(call), feedback(call_id, result), observe,
start(entrypoint, payload, permissions=None), join(handle), cancel(handle=None), and
registered W. feedback resolves a pending suggestion without executing it; this
is recorded as authored feedback, not an environment result. There is no Decision menu or mandatory plan/tool/plan sequence.
Every root physical or Workflow execution must consume the exact, unchanged tool
call returned by runtime.plan(); L cannot manufacture, rewrite or directly dispatch
an executable call, including indirectly through a local task. Deterministic triggers and checks may shape plan(objective), but
the Planner remains on every root execution path. For a registered
workflow_<id>, call["arguments"] is the wrapper
{"arguments": <W input>, "continuation_id": <optional>}; the W input is not placed
directly at call["arguments"]. runtime.control("workflow:<id>") is available only
inside W for declared Workflow dependencies, never as a root-L bypass. L may use
model-confirmed triggers, per-action planning, Planner-confirmed batched execution or
another authored strategy; none is mandatory. Every W outcome is recorded back into
the same Planner conversation and remains local evidence for the continuing root
control flow. Returning from L is valid only after snapshot().terminal is true,
including an accepted explicit finish operation. finish is a task-level Planner
decision. A local W failure or the fact that W has run must not select a finish-only
objective or disable alternative recovery paths. Replan from the actual W result;
keep its diagnostic feedback in the same conversation. Before replanning, execute
or explicitly reject with feedback every tool call from the preceding Planner
response so the provider conversation never contains an unmatched call.
""",
}

RUNTIME_CONTRACT = """There is no subagent registry or mandatory agent architecture. A plugin can
implement agent behavior as ordinary functions combining model calls, tools
and its own control flow. start executes a callable from the SAME frozen source
package in a private worker and returns a handle immediately. Permissions can
only narrow; parent exit drains/cancels outstanding tasks. Use join to collect
results. Serial helper functions need no special runtime mechanism.
Workers support synchronous entrypoints and async entrypoints on a real asyncio
loop (Future, Event, sleep and gather). The deployed worker uses Python 3.10;
do not use asyncio.TaskGroup or other Python 3.11-only APIs. In async code await every
runtime call; calls share one ordered RPC stream per worker. Local tasks can
compute or wait concurrently, but independent model/tool streams use start/join.
Cancelling an async RPC waits for its submitted result before reusing the stream.
No network sockets, subprocesses or threads are granted by asyncio support.
The input observation is the latest settled state-changing tool feedback, not
the last arbitrary read and not a live view during an in-flight action. Capture
errors and unknown outcomes remain explicit. last_tool_result holds the latest
settled public tool operation; history holds all such results and failures.
Reads of images, memory or old state do not overwrite observation. Nested W
invocations receive fresh snapshots of these fields, not their parent's old input.
remaining_wall_time_s is the earliest remaining program/episode wall budget; null means no applicable time limit. Request and physical budgets still apply.
physical_budget is the robot adapter's authoritative {unit, limit, used, remaining,
operation_costs}, or null when unavailable. Individual unknown counters stay null.
Observation step indices and tool-call counts are not physical budget counters.
controller_budget is the shared native-turn {unit, limit, used, remaining,
failed_attempts}, sampled before the next charged turn; null means unavailable.
Root, local, correction and provider-continuation turns share this allowance;
transport reads do not create another charged turn.
Every plan() supplies this current budget and wall time to C, including a fresh
fact in messages. Use measured consumption and motion response to plan continuation;
the Host does not choose reserves, shorten authored actions or claim feasibility.
Workflow continuation resumes the same workflow with IDENTICAL yielded
resume_arguments and caller ancestry. This includes max_steps and tolerance.
The kernel retains cumulative counters and remaining wall time: continuation
never grants a new budget. Changed arguments require a fresh call without the
token. Unknown/consumed/mismatched tokens return failed/invalid_continuation before
program execution; keep that result in the Planner conversation and replan.
request_limits applies to every plan() call. Image selection belongs only to C;
L/W call plan with the question, without managing history or image transport.
Text and original image references remain available; selected pixels are unchanged.
Omitting C.image_selection retains all images. C owns selection and the Host
rejects oversized input; W does not assemble another request or copy history.
An invalid input belongs to the candidate, not an outage.
Before any state-changing operation, observation is {} even if a read has already
returned task details or an image. The top-level task_language is the program's
initial task message; a benchmark may use a generic message rather than the
environment's actual instruction. Obtain actual task facts from public tool
results when needed. Missing evidence is unknown, not a negative applicability
fact. Acquire it or return a precise missing-evidence result to the caller.
Model requests, physical actions and plugin invocations share episode budgets;
all evidence is recorded with explicit program-call/request identities. Models and
low-level tools are fixed interfaces, not candidate-selected backends.
"""


def runtime_contract(controller_request_unit: str) -> str:
    """Expose transport ownership for the deployed online backend only."""
    if controller_request_unit != "native_turn":
        raise ValueError("unknown online controller request unit")
    start = RUNTIME_CONTRACT.index("request_limits applies to every plan() call.")
    end = RUNTIME_CONTRACT.index("Before any state-changing operation", start)
    return (
        RUNTIME_CONTRACT[:start]
        + "Every plan() uses the same native provider session. The Host owns "
        "default observation projection and incremental delivery. C freely selects "
        "views and relevant keyframes; no mandatory camera triple. Native C output "
        "contains at most 10 images before image_selection. Do not rebuild provider "
        "history. Deployment budgets also apply; raw action images stay archived.\n"
        + RUNTIME_CONTRACT[end:]
    )


def _render_prompt(template: str, fields: dict[str, str]) -> str:
    for name, value in fields.items():
        template = template.replace("{{" + name + "}}", value)
    return template


AUTHOR_METHOD_REFERENCE = _render_prompt(_AUTHOR_METHOD_TEMPLATE, _METHOD_DEFAULTS)
WORKFLOW_BOUNDARY_GUIDANCE = (
    "Workflow API descriptions, applicability and argument/return schemas belong to W, even though they contain text; they are not C edits. "
    "Keep these interface descriptions available for reading and calling when W is read-only. C may select immutable tool definitions but cannot rewrite their APIs or descriptions. "
    "Workflow behavior may be implemented inline in the Loop without adding a callable workflow when L is editable; changes to Loop source count as L changes. "
    "Inlined root actions still consume unchanged Planner calls and retain the fixed Kernel permissions. "
    "Choose the implementation from the supported mechanism. Do not require a new tool or count calls as evidence of improvement."
)


def author_scope_guidance(scopes: tuple[str, ...]) -> str:
    """Describe actual authority without exposing treatment names or expected gains."""
    authority = authoring_scope_contract(scopes)
    editable = ", ".join(authority["editable_components"])
    frozen = ", ".join(authority["read_only_components"])
    paths = ", ".join(
        "candidate/" + path for path in authority["candidate_write_paths"]
    )
    return (
        f"Editable components: {editable}. Read-only components: {frozen}. "
        f"Candidate write paths: {paths}. "
        "Keep each read-only component identical to the parent, including source, metadata and contract cases. "
        "Read-only interfaces remain available for diagnosis and integration tests. "
        "A historical implementation or an interface example never expands write authority. "
        "Use .scratch/ for analysis and tests. The Host enforces these same permissions."
    )


def _component_choices(scopes: tuple[str, ...], *, detailed: bool) -> str:
    roles = _METHOD_ROLES if detailed else _SHORT_ROLES
    frozen_roles = {
        "C": "- C is read-only: inspect its current evidence projection, instructions and skills; do not rewrite them.",
        "W": "- W is read-only: inspect the registered workflow interfaces and existing behavior; do not create, remove or modify workflows, metadata or code.",
        "L": "- L is read-only: inspect its existing decision, execution and feedback ordering; do not alter loop source, boundaries or contract cases.",
    }
    return (
        "Choose a mechanism within the editable components ("
        + ", ".join(scopes)
        + "), without quotas or a prescribed repair. Inspect the full agent to understand its interfaces.\n\n"
        + "\n".join(
            roles[scope] if scope in scopes else frozen_roles[scope]
            for scope in ("C", "W", "L")
        )
    )


def _robot(proposal: Proposal) -> str:
    """Robot family from the canonical task identity, e.g. robodojo:<task>."""
    return proposal.task_protocol.dev[0].task.split(":", 1)[0]


def author_method_reference(
    scopes: tuple[str, ...],
    *,
    purpose: str = "mutation",
    feedback_validation: bool = False,
    parent_selection: str = "global",
    topology: bool = True,
) -> str:
    """A small task-model supplement to the common coding/evidence contract."""
    fields = dict(_METHOD_DEFAULTS)
    if purpose == "success_consolidation":
        fields.update(_CONSOLIDATION_FIELDS)
    elif topology:
        selection = (
            "For a mutation, the parent is the newest historical Harness with a complete development evaluation."
            if parent_selection == "latest_evaluated"
            else fields["parent_selection"]
        )
        fields["task_model"] = (
            "## Task model\n\n"
            "The topology is a revisable interpretation of execution evidence, not an assigned repair. "
            "Use graph/map summaries to locate useful witnesses when they help your diagnosis. "
            "You may question its explanations and investigate original trajectories directly. "
            "A graph transition need not become a Workflow. In plan.topology_feedback, "
            "record only material claims you actually reviewed: target, stance "
            "(agree / modify / refuse), reason and inspected evidence. agree means "
            "this specific claim is supported; modify states the revised interpretation; "
            "refuse explains why the claim should not guide this candidate. "
            "Unreviewed claims need no entry and do not imply agreement. "
            "Do not duplicate this record in SKILL or audit every graph element.\n"
            "Graph progress diagnoses partial achievement; it does not establish success "
            "probability or require preserving an unsuccessful implementation. "
            "context/search_review.json provides same-condition comparisons on demand.\n"
            + selection
            + "\n"
        )
    else:
        fields["task_model"] = (
            "## Parent selection\n\n"
            "The parent is selected globally from all historical Harnesses: highest official development success rate, then newest version on equal success rates. Final output uses the same rule. "
            "If official outcomes are missing, the shared search rule ranks confirmed successes over the fixed development denominator, then official measurement coverage, then recency. Missing outcomes remain unknown, never failures or a complete success rate, and do not qualify for success stopping. "
            "All original history remains available. No other quality score is used to break ties.\n"
        )
    if feedback_validation:
        fields["feedback_validation"] = _FEEDBACK_VALIDATION
    if scopes != ("W", "C", "L"):
        fields["success_extraction"] = (
            "Edit only "
            + ", ".join(scopes)
            + " in task-family. Other components, global "
            "assets and task-specific facts remain read-only. Report a required change "
            "outside that authority as a limitation."
        )
    return _render_prompt(_AUTHOR_METHOD_TEMPLATE, fields)


def _mutation_instructions(
    scopes: tuple[str, ...],
    *,
    purpose: str = "mutation",
) -> str:
    fields = dict(_SHORT_DEFAULTS)
    if purpose == "success_consolidation":
        fields["consolidation_entry"] = (
            "For success consolidation, start from winning episode originals; search views\n"
            "are absent. "
        )
    if scopes != ("W", "C", "L"):
        fields.update(
            component_choices=_component_choices(scopes, detailed=False),
        )
    return _render_prompt(_MUTATION_TEMPLATE, fields)


MUTATION_INSTRUCTIONS = _mutation_instructions(("W", "C", "L"))


def author_instructions(proposal: Proposal) -> str:
    writable = ", ".join(
        f"candidate/{SCOPE_DIRECTORIES[scope]}/" for scope in proposal.scopes
    )
    phase = (
        "AFTER FIXED-EVALUATOR SUCCESS: organize this complete Harness into global/, "
        "task-family/ and Host-owned task-specific/. Extract reusable W/C/L into global; "
        "retain task bindings and necessary differences in task-family. Reuse source "
        "through module imports or Workflow dependencies, never duplicate whole implementations. "
        "Both C and L need reusable global source; an empty W stays empty. Review the "
        "online task-family experience and extract general skills only where supported. "
        "The winning physical receipt belongs to the original Harness; your reorganized "
        "content has synthetic admission only until separately evaluated. Do not edit "
        "task-specific facts or invent a successful recipe. Global source ownership is "
        "declared by its directory and persisted in PythonFile.layer/WorkflowSpec.layer."
        if proposal.purpose == "success_consolidation"
        else "DURING EVOLUTION: write and edit only task-family/. Global assets are read-only; task-specific facts belong to the Host."
    )
    authority = ""
    # Component-boundary semantics are delivered at entry only when authority is
    # restricted; full-scope authors read them in DEVELOPMENT.md.
    boundary = ""
    if proposal.scopes != ("W", "C", "L"):
        authority = author_scope_guidance(proposal.scopes) + "\n\n"
        phase = "DURING EVOLUTION: edit only the declared component directories under task-family/. Global and task-specific assets are read-only. Retain the measured winner without global source extraction."
        boundary = WORKFLOW_BOUNDARY_GUIDANCE + "\n\n"
    return (
        authority
        + _mutation_instructions(
            proposal.scopes,
            purpose=proposal.purpose,
        )
        + "\n\n"
        + phase
        + "\n\n"
        + boundary
        + (
            "Writable reusable source: candidate/global/. "
            if proposal.purpose == "success_consolidation"
            else ""
        )
        + "Writable plugin directories: "
        + writable
        + "."
    )


def author_request(proposal: Proposal) -> str:
    if proposal.purpose == "success_consolidation":
        return (
            "Read context/input.json, context/task_instructions.json and the actual winning episode originals. "
            "This is success consolidation, not another search round: graph/map allocation views are absent. "
            "Organize the whole winning W/C/L and online experience into global/ and task-family/, "
            "preserving Host-owned task-specific/evidence.json. Extract parameterized shared code, "
            "retain task entrypoints and dependencies, and justify reuse conditions without claiming transfer tests. "
            "Record the diagnosis and actual checks in candidate/plan.json and task-family experience in candidate/SKILL.md. "
            "Reuse existing code and fixtures, run validate_candidate phase=interaction, and repair its failures before returning; never run physical actions or inherit the winning score."
        )
    history = "context/EVOLUTION_HISTORY.md"
    graph = (
        "Graph interpretations are revisable evidence, not assigned repair methods. "
        "Use read_graph summaries or relevant witnesses when helpful; complete views "
        "are context/graph.json and context/progress_map.json. "
        if proposal.task_protocol.evolution_mode != "no_topology"
        else ""
    )
    request = (
        f"Round {proposal.round_index + 1}, independent author {proposal.author_index + 1}: "
        "read context/input.json and context/task_instructions.json. "
        f"Historical outcomes and original evidence are indexed in {history}. "
        + graph
        + (
            "Improve the complete parent in candidate/ toward the evaluation objective. "
            if proposal.scopes == ("W", "C", "L")
            else "Improve the complete parent by editing only "
            + ", ".join(proposal.scopes)
            + "; keep all other components unchanged. Read authoring_authority in context/input.json. "
        )
        + "Use your own investigation to produce a complete improvement. Read relevant "
        "DEVELOPMENT.md sections, implement and verify; leave the actual diagnosis, "
        "checks and uncertainty in plan.json. Begin with the supplied parent and relevant failure evidence; preserve unaffected mechanisms and avoid regenerating unchanged files. Reuse parent interaction fixtures when provided, adapt affected coverage, then run validate_candidate phase=interaction and repair failures before returning. Additional reusable experience in SKILL.md is optional."
    )
    return request


_EVIDENCE_HEAD = """## Evidence access

context/input.json indexes original episodes; context/task_instructions.json holds
the recorded public task instructions. Use exact admitted filenames; a denied
parent-directory glob may appear empty. Earlier author notes and complete versions
are indexed by context/history_originals.json; context/EVOLUTION_HISTORY.md is a
measured history index, not a required reading of every earlier diagnosis.
Use issued handles such as episode:001, trace:001 and file:001#/23.

- read_original(reference): exact file/JSON-pointer content; images return as images.
- read_trace(evidence_id, view): executions, planner_feedback, workflow_events,
  timeline, telemetry and diagnostics. Read-only calls appear in feedback/timeline
  even without physical executions. A completed call can return failure; inspect
  its result. kernel_actions links actual runtime.act returns; planner_feedback
  links what the Planner received. Missing links and unscored outcomes stay unknown.
- read_video(evidence_id, path, frames): frames=[] gives metadata; explicit frame
  indices return images and source references. Video frames are not physical steps.
- read_sensor_snapshot(evidence_id, observation_ref): use the exact sensor snapshot
  ID from read_trace, not a file: handle. For local observation files use read_original.
  Retained depth/calibration remain available through their original indexed paths;
  video alone cannot provide missing RGB-D.
- read_source(source_id, path): one historical file, indexed by
  context/source_inventory.json. Fixed-tool implementation is read-only reference.
"""

_GRAPH_EVIDENCE = """- read_graph(layer, view, identity): semantic/execution/map summaries and targeted
  node, program, transition, witness or episode views. Map comparison pairs a
  version's same-condition outcomes, including unknowns and official disagreements.
  Semantic node/program/edge IDs are literal; witness IDs are issued handles.
"""


_RETRIEVAL_TOOLS = """- retrieve_graphs(query=None, top_k=5, exclude_tasks=[]), when configured: calling
  it without a query searches with this round's COMPLETE canonical graph,
  including its matching conditions. This refreshes each call; a previous local
  query never becomes the default. Nothing triggers a mandatory search.
  An explicit query contains catalog definitions, global entities, node conditions,
  directed dependencies and conditional interfaces (source/target/entry/exit).
  Use the same representation for a cold-start hypothesis, a local repair, or one
  state pair. Object identity must remain consistent across the entire query.
  Read the shared catalog before encoding. Unknown is an omitted condition;
  mutually exclusive values are explicit, never inferred from embedding similarity.
  Phase-dependent requirements include the declared phase as an entity argument;
  reuse its identity throughout the query. Do not phase-qualify physical facts
  to evade conflicts or encode cross-sample evaluation labels as state facts.
  Select references yourself or exclude inspected tasks for another batch.
  context/reference_matches.json, when present, contains THIS round's analyst
  results shared by all authors. Authors can independently query again without
  overwriting those results. Distances measure coverage under fixed vocabulary;
  inspect unsupported conditions, unmatched_entities and solver bounds. A missing
  object/phase correspondence leaves that condition unknown; contradictions are
  checked under a complete parameter correspondence, not from opposing words alone.
- read_reference_graph(task, view, identity), when configured: inspect selected
  nodes, transitions, programs or view=conditions. With no arguments list corpus
  tasks. view=catalog with no task reads shared definitions; optional text and
  top_k use Qwen to suggest relevant coordinates, not certify opposite values or
  entailment. Reuse existing definitions exactly; add only genuinely new meanings.
  Search returns correspondence and witness-path IDs, not source or trajectories;
  read originals only when useful. A path is historical evidence, not a composed
  executable precondition. Retrieval does not establish transfer or causal credit.
"""

_EVIDENCE_TAIL = """
Tool schemas describe query arguments; context/dev.py --help exposes the same tools
from the shell. Python: `from context.dev import call`; `call("read_original",
reference=ref)["value"]` resolves the original without a separate loader.
Telemetry fields are named column arrays under data.public_evidence and
 data.critic_private_evidence, aligned with control_step_indices. Choose feature
names from the trace index, preserve full names/units and leave missing values null.

Use native shell/Python for targeted archived-input inspection and local code checks
under .scratch/; do not start additional model sessions or inference-service probes.
Physical actions, resets, live capture and evaluation belong to the fixed evaluator.
context/capability_catalog.json and gateway.json specify currently registered tools.
"""

EVIDENCE_GUIDE = _EVIDENCE_HEAD + _GRAPH_EVIDENCE + _RETRIEVAL_TOOLS + _EVIDENCE_TAIL


_HOST_TOOLS = """## Host tools

- validate_candidate(phase="code"): for mutations, checks the candidate's file set, actual
  allowed scope, schemas, imports and declared contract cases using the production
  isolated workers. It does not require a finished plan. Call this while writing.
- validate_candidate(phase="interaction"): runs those component checks plus the full
  production C/W/L synthetic interaction suite before the turn ends. Repair any
  failures here using the existing fixtures; this preview grants no final admission.
- Host final admission checks the complete plan, actual changes relative to the parent and
  .scratch/tests/interaction.json once after the native turn. If it fails, the full
  error returns to this same session for correction; invalid code never advances
  to rollout. Fix failures without bypassing validation or changing fixed inputs.

On an infrastructure failure, retain the error and report it; do not monkeypatch
the validator, guess an NFS explanation or turn a failed check into success.
"""


CASE_GUIDE = """Write a few discriminating cases for the mechanism and likely regression. Specify
expected behavior independently of the implementation; do not produce assertions
by recording the candidate's own output. Reuse supplied fixture formats and the
Host runner. Fixture responses match calls in strict order, with exact operation
names and argument dictionaries (including numeric values); all must be consumed.
W cases check dispatches, workflow input/output schemas and expected_status. They
do not compare specific output values or validate synthetic operation responses
against the capability catalog. Match those responses to real envelopes, and
assert output values explicitly in scratch tests when the
mechanism depends on them. C/L PluginCase.expected compares the full returned
object. A passing status alone does not test a stop reason or endpoint predicate.
Put analysis scripts in .scratch/analysis/ and tests in .scratch/tests/; Python
cache is redirected to .scratch/runtime/pycache. Do not compile into candidate/. Finish the same
plan.json with the actual change, observed checks and untested limitations.
Replaying recorded observations can test what the new program would decide on
those observations. Once it issues a different action, later recorded states are
not that action's measured consequences. Report such replays as decision tests,
not physical steps saved or proof that the new controller preserves trajectories.
"""


PLUGIN_INPUTS = {
    "W": """The example entrypoint run(runtime, context) receives a public snapshot dictionary:
context["arguments"] is the schema-validated caller argument dictionary, and
context["state"] is the continuation state. Caller argument keys are not top-level
snapshot keys. Read context["observation"] for the settled public observation.
Entrypoints receive two positional arguments; naming the second one input instead
of context does not change its shape or the keys above.
W returns
WorkflowResult (status/summary/output/state as declared by its schema). The kernel
owns continuation IDs. Use gateway.json to select declared permissions and exact
operation arguments. Local synthetic responses must match those arguments.
runtime.schema.json includes workflow_result; bundle.schema.json defines the W
manifest, permissions, local limits and contract_cases. The release_checked
example already has that complete editable directory shape.
The kernel validates caller arguments against input_schema before W runs and
validates WorkflowResult.output afterwards. Put argument types and bounds in the
schema; schema defaults are annotations, so implement optional defaults in code.
Do not silently replace an explicitly supplied value with a different default.

An accepted W is automatically exposed in the next bundle's tool catalog as
workflow_<id>, with arguments and optional continuation_id. Its description and
applicability are shown to the Planner; no C/L edit or registration script is
needed for the default C/L to select and execute it. Authored C/L can choose which
registered tools to use. This exposure does not force the Planner to call W.
Registering W does not require changing C/L. Their own interfaces below cover
context construction and the complete decision loop independently of Workflows.
Check the actual model-visible request and returned feedback in the interaction report.

context/capability_catalog.json describes physical result envelopes in
output_schema; gateway.json lists dispatch permissions and arguments.
capability_result contains the handler's actual result; captured public state and
error/terminated/truncated/interrupted fields accompany it. For example, a motion
endpoint flag belongs to result['capability_result'], not directly to the state
or workflow status. Tool-specific fields are optional: inspect the exact recorded
result and its coverage before relying on them. Read operations return their own
operation-specific shape. Use kernel_actions and read_original for complete
recorded runtime.act envelopes; planner_feedback only projects their reports.
For LIBERO pose conventions, quaternion order is x,y,z,w; target_pitch is the projected
EEF-z tilt, not Euler rotation about world Y. Do not infer a different convention
from the parameter name.

""",
    "C": """C shapes what the Planner can see and how it is instructed to reason about the
task: evidence selection, grounding, uncertainty, feedback interpretation and
continuation. It applies to direct tool decisions as well as W selection.
C receives the public snapshot plus base_instructions, messages, assets, state,
and objective for every plan call. It returns
NativeModelContext: instructions, messages, tools, state and optional image_selection.
Messages use internal PydanticAI-shaped dictionaries with incremental current
observations and newly settled feedback by default; native history remains in the
provider session. Tools retain registered schemas. C freely organizes evidence and
may deliberately emphasize relevant keyframes. At most 10 distinct images are
allowed in its complete output before selection. The context_built event records
exact input/output references when you need the actual runtime shape.

""",
    "L": """L owns the complete episode's planning/execution/feedback loop, including when
to request a decision, consume its calls and replan from settled outcomes.
It receives the public snapshot and owns run(runtime, context). snapshot() returns
program_call_id, parent_program_call_id, task_language, terminal, observation,
history, last_tool_result, tools and epoch. plan() returns text and tool_calls;
execute(call) actually dispatches. Read the L example for their executable shape.

""",
}


PLAN_GUIDE = """## One concise edit record

Finish candidate/plan.json against context/plan.schema.json:
- diagnosis: the observed problem and evidence motivating the edit;
- summary: what actually changed and why;
- validation: checks actually performed, outcomes and material limitations.

These fields may be incomplete while coding. No pre-edit report, exhaustive risk
list or separate multi-component justification is required. The Host derives changed
components from code and enforces write authority. Optional probes describe public
activation/effect events to inspect in normal evaluation; omit them when unhelpful.
The final admission receipt records its own checks; do not rewrite the plan to copy it.

candidate/SKILL.md is optional, for additional reusable lessons or open questions
not already in the plan. Keep evidence and uncertainty clear. Empty or absent means
no additional note. Neither this note nor the plan changes executable C/W/L or
certifies physical success. Online C skills remain separate deployed assets.
"""

CASE_RESULTS_GUIDE = """## Inspect actual case results without a second runner

For a shell check, persist the full report once (the command below uses the
interpreter shown in this guide):

```sh
{python} context/dev.py validate --output .scratch/tests/check.json
```

The file contains the full MCP result, including structuredContent and complete
errors. The terminal prints a status projection and the report path; failed checks
exit nonzero. Read that file's structuredContent.contract_case_results for output
assertions instead of rerunning validation to inspect fields. A direct tool call's
case list is only the status/error projection. The Host keeps every original check in
its audit, including if this scratch report is later replaced. This report is not
a completion receipt. Actual candidate edits still require another check.

From a scratch Python script, use the same production checks. The direct return
is intentionally a small status receipt so routine checks do not fill the native
conversation with repeated tool transcripts. It always binds the complete report:

```python
from context.dev import call, DevelopmentToolError
try:
    receipt = call("validate_candidate", phase="code")
except DevelopmentToolError as exc:
    receipt = exc.result["structuredContent"]
    # Preserve/report the failure; inspecting evidence does not turn it into pass.
    print(receipt["contract_case_results"])
    raise
assert receipt["structurally_valid"] is True
entry = call("read_original", reference=receipt["complete_report"])["value"]
report = entry["result"]
cases = report["contract_case_results"]
print([(case["scope"], case["case_index"], case["validation_status"])
       for case in cases])
```

The success receipt uses structurally_valid, not an ok field. Author-side
plan_checked remains false; interaction_tested is true only for phase="interaction".
Final admission is a separate Host receipt. Save the returned report in .scratch/tests/ when
you need several assertions; another inspection of it needs no validation rerun.
validation_status is the case-check outcome, not the candidate's result.status:
an expected failed workflow may pass its case. There is no per-case passed field.
Each case includes scope, case_index, W workflow_id when applicable,
validation_status, ordered calls and unconsumed_responses. The actual result is
present only when the worker returned; otherwise inspect the explicit error. Inspect
the relevant case's result and write explicit semantic assertions in scratch
against expectations derived independently from the mechanism. This is the real
Host sandbox run, so there is no reason to build FakeRuntime/NoChmodSandbox just
to access a return value. The full results and failure are also in the audit.
Do not derive expected answers by copying the observed output. A fixture remains
synthetic and cannot establish a physical counterfactual.

For an additional recorded-feedback decision test, add a contract case to the
relevant candidate manifest, with the new program's independently expected calls
and the selected original response fields as explicitly hypothetical fixtures.
Run the same validator and assert its returned output. A separate positional Stub
can silently give one operation another operation's response; the production
fixture matcher already checks both operation and arguments. Original response
references establish provenance, not what would happen after a changed action.
"""


INTERACTION_GUIDE = """The component contract_cases are unit fixtures. W unit cases
provide the declared observation and a placeholder task, not a production snapshot;
C/L unit cases call their entrypoints without the complete conversation machinery.
Use them for local mechanisms; the Host final gate tests the actual combination.

Write a JSON array to .scratch/tests/interaction.json. Each scenario follows
runtime.schema.json.interaction_scenario: name, system_prompt, task_message,
model_responses, operations, assertions. The Host creates the initial snapshot
and runs the candidate's actual L/C/W. You cannot inject observation/history.
If a workflow requires a prior action or read, script those actual calls first.
There is no required opening ritual, successful W status or physical call.
Every changed C/L and each new/modified W must actually be exercised across the
suite; a text-only empty run cannot cover a new W. Deleted W must stay absent.
An edited C or L with Workflows also requires a scenario marked recovery=true. Supply
a genuinely recoverable nonterminal W failure and a later Planner-selected
recovery action/W. Host independently checks the failed feedback survives C into
the next root plan and its new action executes; a queued pre-failure action,
read-only inspection, or finish does not establish recovery. This fixture does
not require arbitrary retries in real episodes or prove physical efficacy.

model_responses contains ordered objects with text and/or tool_calls; each call
has name and arguments. Optional instructions_contains, required_tools and
forbidden_tools check the C-built model request. For a visual model response, set
required_images to the declared artifact filenames: Host checks actual binary
image bytes, not filenames or text. Text-only reasoning needs no images.
For a camera task, at least one validated model request must exercise C on real
image inputs; all-text input fixtures cannot establish visual input validity.
C may still deliberately select a text-only output.
Model-requested reads and reads preceding a program’s own plan() keep their images.
Other programs’ internal reads keep text feedback; C can deliberately select their
original pixels from source_messages.
Exercise repeated internal reads when a W uses them, then request another model turn.
Image labels or filenames alone do not supply pixels.
operations contains ordered objects with name, arguments and result. These substitute external responses,
not the loop, context construction, workflow body or feedback machinery. Use the
exact gateway.json operation schema. The fixed catalog retains original metadata
with episode output paths expressed as {{output_dir}}. Unscripted/mismatched or
unused responses fail even if the candidate catches the resulting error.

assertions is a nonempty list of {"path": [keys/indices...], "equals": value}
against the full report. For example, ["calls", 0, "name"] checks the first
public operation; ["workflows", 0, "result", "output", "reason"] checks a
particular workflow reason. Assert the main mechanism and meaningful failure
feedback, not just a constant field or invocation count. The report contains
requests, events, calls, workflows and result without clipping. result is the
PlannerResult projection (error, termination_class, finish_result, messages, stats), not the
dictionary returned by L: ["result", "summary"] is invalid. Inspect L's return
in a program_completed event, or assert the actual workflow/tool behavior.
operations may specify environment_state={"official_success": false,
"terminated": false, "truncated": true} to model horizon exhaustion, or
{"official_success": false, "terminated": true, "truncated": false} for a
valid unsuccessful terminal episode. Tool result flags alone do not
change the environment. Include every model request, including W-internal plan()
and the subsequent root replanning request; all share model_responses in order.
Choose expectations from the hypothesis before checking the actual output.

Host final admission executes the settled suite and checks coverage against the
actual changed components. If it fails, the same session receives the exact report
reference for correction. Reports and complete scenario values stay in Host evidence. Keep
temporary work under .scratch/ and final assets only in candidate/.

This is deterministic integration testing. Model responses are scripted: adoption
by a real Planner is still untested. Environmental responses are hypothetical:
neither contact nor task success has been measured. A recorded response keeps
its provenance but does not become the consequence of a newly changed action.
A concurrent scenario exercises one concrete ordering, not every interleaving.

For read_image, declare scenario.artifacts entries with name, step, media_type
and data_base64 plus initial_artifact_step. Only these verified image bytes may
be read through the real artifact interface; no host file paths are accepted.
Do not also put read_image in operations. artifact_reads and the unified calls
record the actual reads; unconsumed artifacts fail. Recorded image bytes retain
their original provenance. Prefer the actual initial image when it affects a
decision, without treating later recorded frames as a new action's consequence.
operations.environment_state may explicitly set synthetic official_success,
terminated and truncated flags to test the loop's terminal path. These are mocked external
evaluator flags, separate from any tool result fields and never a measured score.
"""


_CANDIDATE_LAYOUT = (
    "Candidate has three scope directories: global/, task-family/ and Host-owned task-specific/. "
    "Each global/workflows/<id>/ or task-family/workflows/<id>/ contains workflow.json, program.json and src/. "
    "Task bindings are task-family/context/{context.json,instructions.md,program.json} and task-family/loop/{loop.json,program.json}. "
    "C/L src/ files are split between global/<component>/src/ and task-family/<component>/src/ and compiled as one package; module paths must be unique. "
    "Each durable skill is <scope>/context/skills/<skill_id>/SKILL.md, with scope derived from its directory. "
    "task-specific/evidence.json is written only by the Host during successful consolidation and points at original episode facts. "
    "These files are the deployable bundle—there is no registration script or hidden source directory."
)


def engineering_documents(proposal: Proposal, interpreter: str) -> dict[str, str]:
    """Engineering shared by author arms with the same robot and authority.

    It carries the candidate layout, component roles, online-agent contract,
    output rules, verification, plan/experience, contract cases and interaction
    admission. Diagnosis methods (topology, history) are separate layers.
    """
    scopes = proposal.scopes
    restricted = scopes != ("W", "C", "L")
    fields = dict(_METHOD_DEFAULTS)
    if proposal.purpose == "success_consolidation":
        fields.update(_CONSOLIDATION_FIELDS)
    fields.update(
        candidate_layout=_CANDIDATE_LAYOUT,
        # The startup entry carries the Workflow boundary for restricted authority.
        component_boundary=author_scope_guidance(scopes)
        if restricted
        else WORKFLOW_BOUNDARY_GUIDANCE,
        component_choices=_component_choices(scopes, detailed=True)
        if restricted
        else _METHOD_DEFAULTS["component_choices"],
    )
    case_guide = CASE_GUIDE
    if restricted:
        fields["online_revisions"] = (
            "specific missing historical fact. The deployment is static: offline write authority does not permit editing a running online episode."
        )
        case_guide = (
            "Contract cases inside read-only components are frozen too. Add or revise component cases only in editable components; use .scratch/ tests and the full interaction scenario to inspect integration with unchanged interfaces.\n\n"
            + case_guide
        )
    return {
        "ENGINEERING.md": _render_prompt(_ENGINEERING_TEMPLATE, fields)
        + "\n\n"
        + "Interfaces: context/runtime.schema.json (interfaces and execution_contract); synthetic formats: context/examples.json.\n",
        "VALIDATION.md": _HOST_TOOLS
        + "\n\n"
        + f"Shell interpreter: `{interpreter}`. Run scratch scripts from the workspace root. CLI: `{interpreter} context/dev.py --help`.\n\n"
        + CASE_RESULTS_GUIDE.replace("{python}", shlex.quote(interpreter)),
        "PLAN_AND_EXPERIENCE.md": PLAN_GUIDE,
        "SKILL_FORMAT.md": EXPERIENCE_FORMAT,
        "CONTRACT_CASES.md": "## Contract cases\n\n" + case_guide,
        "INTERACTION.md": "## Full candidate interaction\n\n" + INTERACTION_GUIDE,
    }


def engineering_reference(proposal: Proposal, interpreter: str) -> str:
    """Shared concise index; full engineering documents are materialized once."""
    return """## Engineering reference

Read the relevant reference before editing or making a claim:
- context/ENGINEERING.md: layout, C/W/L boundaries, session/history, image selection,
  online experience, interfaces and evidence limits.
- context/PLAN_AND_EXPERIENCE.md: plan.json, diagnosis, probes and SKILL.md writing convention.
- context/SKILL_FORMAT.md: format when editing deployed C skills or consolidating reusable skills.
- context/CONTRACT_CASES.md: discriminating component cases.
- context/INTERACTION.md: .scratch/tests/interaction.json and complete candidate admission.
- context/VALIDATION.md: Host tools, code checks and inspection of case results.

context/runtime.schema.json is the authoritative interface/execution contract;
context/examples.json contains synthetic formats. Preserve one provider session per episode,
at most 10 C images before selection, exact source observation/pixel mappings and shared budgets.
Only declared component directories are writable; frozen components remain read-only.
Use validate_candidate(phase="code") while coding, then phase="interaction" before
returning. Fix failures within this turn; the Host repeats final admission.
The common engineering references grant no additional evidence, probe or physical permissions.
"""


def author_reference_documents(proposal: Proposal, interpreter: str) -> dict[str, str]:
    """Shared contracts only; no duplicate author workflow or method checklist."""
    return engineering_documents(proposal, interpreter)


def development_guide(
    proposal: Proposal, interpreter: str, *, graph_retrieval: bool = False
) -> str:
    """The single detailed authoring source: ReWeave method layer plus engineering."""
    restricted = proposal.scopes != ("W", "C", "L")
    availability = "All historical programs and source Bundles remain available through history_originals.json and read_original, independently of version scores or the reference list. "
    modify = (
        "Modify only "
        + ", ".join(proposal.scopes)
        + " in the inherited complete candidate,"
        if restricted
        else "Modify the inherited W/C/L candidate,"
    )
    entry = (
        "# Develop one Harness mutation\n\n"
        "candidate/ is a complete independent copy of the parent. "
        f"{availability}{modify} using your own diagnosis. "
        "The objective, budgets and original evidence are in context/input.json. "
        "Read current task instructions and the interfaces affected by your change. "
        "Use the index below to find further evidence when needed."
    )

    method = author_method_reference(
        proposal.scopes,
        purpose=proposal.purpose,
        feedback_validation=proposal.task_protocol.validation is not None,
        parent_selection=proposal.task_protocol.parent_selection,
        topology=proposal.task_protocol.evolution_mode != "no_topology",
    )
    evidence = (
        _EVIDENCE_HEAD
        + (
            _GRAPH_EVIDENCE
            if proposal.task_protocol.evolution_mode != "no_topology"
            else ""
        )
        + (_RETRIEVAL_TOOLS if graph_retrieval else "")
        + _EVIDENCE_TAIL
    )
    if proposal.task_protocol.execution.online_workflows:
        from reweave.runtime.online_workflow import ONLINE_WORKFLOW_AUTHOR_GUIDANCE

        evidence += ONLINE_WORKFLOW_AUTHOR_GUIDANCE
    return (
        "\n\n".join(
            [
                entry,
                "## Complete authoring reference\n\n"
                + method.replace("\n## ", "\n### ").rstrip("\n"),
                evidence.rstrip("\n"),
            ]
        )
        + "\n\n"
        + engineering_reference(proposal, interpreter)
    )


def runtime_interfaces(
    controller_request_unit: str = "native_turn",
    *,
    scopes: tuple[str, ...] = ("W", "C", "L"),
) -> dict[str, str]:
    """The same complete interface text used by scope-specific guides."""
    runtime_contract(controller_request_unit)
    return {
        scope: (
            ("EDITABLE" if scope in scopes else "READ-ONLY")
            + f" {scope} interface. Component write authority is recorded under authoring_authority.\n\n"
            if scopes != ("W", "C", "L")
            else ""
        )
        + contract
        + "\n"
        + PLUGIN_INPUTS[scope]
        for scope, contract in PLUGIN_CONTRACTS.items()
    }
