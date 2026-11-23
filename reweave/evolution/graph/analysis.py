# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Independent per-round graph sessions, with immutable graph handoff receipts."""

from __future__ import annotations

import fcntl
import json
import sys
import time
from pathlib import Path

from reweave.evolution.authoring.author import (
    AuthorCorrectionRequired,
    NativeSession,
    _artifact,
    _publication_paths,
)
from reweave.evolution.authoring.candidate import authoring_scope_contract
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    GraphAnalysisReceipt,
    GraphAnalysisRequest,
    GraphAnalysisResult,
    GraphSnapshot,
)
from reweave.evolution.evidence.archive_inputs import archived_author_policy
from reweave.evolution.graph.projection import analysis_graph_receipts, project_map
from reweave.evolution.graph.refinement import (
    PENDING_GRAPH_REFINEMENT,
    analysis_result_schema,
    needs_matching,
    validate_analysis_output,
)
from reweave.utils.persistence import atomic_json, atomic_text

_RETRIEVAL = """When retrieve_graphs is configured, decide whether cross-task references would
help. Calling it without query uses your current complete graph draft, or the
latest shared graph before a draft exists; you may instead select local states
and relations. The Host passes this round's compact retrieval results to every
mutation author. No search is automatic, no old round's results become defaults,
and authors may independently query other matches. """
_MATCHING = """When cross-task retrieval is configured, maintain graph.matching on this SAME
canonical graph, covering exactly its node and edge IDs. Read the shared catalog
with read_reference_graph(view="catalog"); optional text lookup locates relevant
coordinates. Reuse each definition, ordered argument meaning and mutually exclusive
value domain verbatim. Add a fluent only for a genuinely unexpressible property,
not a synonym. State conditions are finite allowed-value sets. Unknown is omission,
not a physical value. Intersection expresses repeated constraints on one variable.
Use one graph-wide entity table: names are display text; explicit qualifications
and declared task requirements are constraints. Keep intended function distinct
from actual behavior. Graph-wide requirements may qualify an entity; temporary
roles belong to their node/interface. Do not infer exclusive roles from names.
If a requirement depends on a task phase, include that declared phase as an
explicit entity argument, e.g. inspection_required(object, phase); do not hide
it in "current stage". Reuse the same phase ID throughout that phase. Do not add
phase arguments to physical properties merely to avoid a same-frame conflict.
The physical frame may be shared while two task-phase requirements differ.
Separate current physical facts, acquired information/history and official goals.
Project every scope onto relevant catalog coordinates directly entailed by its
stated evidence and the definitions. A specific property does not replace a
coarser entailed coordinate already in the catalog. Ground only established
objects/relations in that scope; do not invent combinations, transient persistence,
secure/safe/success values or numerical thresholds. Record unresolved implications
and identity in uncertainties. Entry/exit refine their endpoints and must be
consistent with them. Background entrance conditions stay on that edge; identical
entry texts do not establish shared progress-state identity. This encoding does
not change witnesses, historical bundles, task progress or evaluator outcomes.
Cross-sample evaluation instructions and equivalence labels are not graph facts.

"""
# Structural-change permissions. A fixed initial topology omits exactly these;
# every non-structural evidence, witness, history and forecast rule remains.
_STRUCTURE = {
    "refine_structure": """Refine earned progress and historical mechanisms: split states when different
task-relevant capabilities require the distinction, merge equivalent achievements,
split transition interfaces when actual preconditions or continuation requirements
explain different outcomes. """,
    "split_interfaces": """Split interfaces only when evidence supports a
material entry or continuation distinction; naming differences and symptoms alone
do not create different mechanisms. Conversely, equal progress nodes can hide
different witnessed entry conditions and must not force all attempts into one edge.
Keep the target condition precise enough that its later acquisition can be
recognized as recovery, including root-Planner continuation after a failed W.
""",
    "state_granularity": """State refinements
must explain physical distinctions, not manufacture additional search width.
Preserve observable differences in earned goal-relevant conditions and their
retention, including partial fulfillment of repeated or conjunctive requirements.
Do not collapse distinct fulfilled requirements into a single first-versus-all
state when it hides measured progress. Use the same condition and dependency
representation for discrete, relational, temporal and continuous tasks; no fixed
inventory, task-name dispatch, stage quota or hand-written score is implied.
Current control and continuation conditions belong in branch/interface detail
when they are not additional task achievements. """,
    "refine_edges": """When equal achievements
hide different continuation outcomes, refine edge entry/exit conditions or a
meaningful branch; do not invent extra milestones to reward state detail. """,
    "backbone_revision": """Subsequent analyses may add, remove, split, merge or reclassify both backbone and
branches. """,
    "backbone_contradiction": """ Contradicting successful
routes require revision of the backbone and reprojection of all history.""",
    "granularity": """Refine state/interface granularity when it
changes these distinctions; extra labels alone are not evidence of learning.
""",
    "revise_interpretations": """Revise earlier interpretations when evidence
warrants.""",
}
_VALIDATION = {
    "validation_conditions": " plus already measured feedback validation",
    "validation_history": """Feedback validation episodes, when present, also belong to this history: interpret
their failures and transitions without inventing outcomes for untested versions.
""",
    "validation_pool": " Validation remains sparse observed feedback. This adaptively reused pool is not an independent test set.",
}
_ANALYSIS_OPERATIONS = "The deployed native online agent retains one Codex session per episode. The naive\nbaseline adds current observations and newly settled feedback only; a short input\ndelta does not imply lost memory. C freely organizes evidence and may select camera\nviews or deliberately re-emphasize relevant historical/intermediate keyframes.\nAdditional observations contain three views; do not blindly copy whole groups\nwhen only particular views are relevant. There is no mandatory camera triple.\nAt most 10 images may appear in C output before\nselection; the ceiling is not a target. Do not recommend automatic history windows\nor routine conversation replay. Session history/compaction, physical ownership and\nresource accounting remain fixed infrastructure outside W/C/L authorship. A long W\nmay archive many images without automatically sending them all at its next plan().\n\n\nAim to complete a well-supported graph within about ten minutes; this is an\nefficiency target, not a deadline or permission to omit evidence. Form an early\ncomplete draft, then call validate_graph(). It uses the final Host structural\nvalidator, including exact pointers, all-history coverage and action continuity;\ndo not reimplement these checks in ad-hoc scripts. Repair reported errors and\nend the turn once the graph is supported and the check passes. Continue reading\nonly to resolve a material uncertainty. No repeated validation of unchanged output.\nUse scripts to copy exact references and construct spans from original indices."

_FORECAST_INSTRUCTIONS = """Alongside the graph, supply forecast.outcomes, forecast.confidence (0–1), and
forecast.rationale. outcomes is a finite distribution of {progress, probability}
for ONE complete child candidate after ONE independent author modifies the
Host-selected parent with the same available graph evidence, author time budget and episode limits.
Each progress is the arithmetic mean TERMINAL graph progress (0–100) across the
parent's full dev{{validation_conditions}} conditions, equally
weighted on THIS graph's scale. Include currently successful seeds and potential
regressions. Never average only failing seeds, use peak progress, predict the best
of several authors, or splice different candidates' best seeds. Progress values
must be strictly increasing with no duplicates; each probability is positive and
the probabilities sum to one. 100 predicts completion throughout that set; it is a
prediction, not a measured result. Explain the probability
of regression, retention and improvement, and the evidence or uncertainty behind
both tails. A single outcome is permitted when justified; do not invent precision.
Confidence audits your certainty in the evidence supporting this distribution;
it is not a gain discount or a variance estimate. {{width_use}} These predictions never change measured progress, parent selection
or stopping. Do not allocate repairs or authors.
validate_graph returns the deterministically selected parent, starting-reference IDs, target
conditions and current progress bounds. Check them against your rationale and
revise the forecast if its target changed. context/analysis_request.json supplies
the frozen author budget and execution limits. Estimate jointly from HISTORY and
CURRENT FAILURES: inspect what previous complete candidates actually changed,
which target seeds improved or regressed, which apparent fixes repeatedly failed,
and what remains unresolved in the current parent's graph. Use actual successful
root continuations, calls, relevant video frames, measured motion response and
remaining budget to assess whether progress can be retained and continued within
one author/episode budget. Historical success on different candidates is evidence,
not proof that their behavior can be combined in one candidate. Account for that
integration risk and regression risk in the candidate-level outcome distribution. Justify confidence
separately using evidence coverage, consistency and comparable earlier prediction
errors; missing evidence or contradictory outcomes should reduce certainty.
Cite relevant graph interfaces or original evidence in the rationale.
Distinguish observed regression from an assumption that the next author will repeat
the same intervention. {{author_space}} When the distribution's expected progress
does not exceed the parent, explain the supported bottleneck and the assumed loss
of retained ability, and what evidence could overturn that judgment. Separate
recoverable interface faults, missing information, model decisions and execution
limits. Do not manufacture a numerical derivation: label a subjective estimate as
such. Put transferable entry/exit conditions and counterexamples in the shared
graph, without assigning future repairs. This diagnosis must inform the next author,
not merely become a pessimistic forecast.
Difficult, diverse failures alone do not establish that more authors will help. Independent
sessions can converge on the same
mechanism. Compare actual sibling changes and outcomes when judging what another
round can accomplish; session count is not measured mechanism diversity. Keep unknown
progress explicit; do not invent a physical witness or assign repairs to authors.
Re-estimate the outcome distribution AND confidence each round; the seeded previous forecast is
historical, not a default. Compare earlier predictions with observed outcomes only
on comparable graph scales and condition sets; retain uncertainty otherwise."""

_GRAPH_INSTRUCTIONS = """You are the physical graph refiner. Each new round starts a fresh native session.
The previous COMPLETE graph, progress map and evidence index carry forward the run's
analysis. Use that structured handoff and this round's new evidence; do not rebuild
native conversation history or reread all historical evidence by default. Recovery
within this same round retains its existing session and draft. Your output is the
COMPLETE next shared semantic graph, not a report on a fixed graph and not an
evolution plan. Start with context/input.json and read_graph(view="summary");
If input.json names current_task, read that active fixed task contract. Historical
verdicts retain their original criteria; do not relabel them under revised goals.
read the prior semantic summary and progress map when available. Use targeted
episode/frame/feature selections and original trace/image/video/code witnesses to answer
concrete semantic questions. context/execution_graph.json preserves the full
scaffold for scripted indexing; do not print its entire contents or all partition
signatures into the conversation. {{revise_interpretations}} The current request supersedes the prior input snapshot.
Write {{exactly}}analysis/result.json using context/analysis.schema.json. You may
write only analysis/ and .scratch/. No candidate authoring or physical execution.
{{retrieval}}Read original programs and
trajectories only when needed. Retrieved examples do not select the complete parent.
For a physical attribution such as ring retention, slipping, obstruction or an
unrecognized placement, inspect the relevant head-video frames or ordered original
images and correlate them with actual calls, telemetry and remaining budget.
read_video(frames=[]) reads metadata only; it does not inspect physical behavior.
Record the visual witnesses and competing explanations. Occlusion stays uncertain;
video frame numbers and nominal playback seconds are not physical-step indices.
The online session, image-selection and fixed-infrastructure contract is in context/ANALYSIS_OPERATIONS.md; read it before interpreting C/L behavior.
Diagnose infrastructure failure separately from task strategy and evidence choice.
An infrastructure failure leaves official capability scoring missing; it does not
make an acknowledged physical state unknown. With no settled physical episode,
keep the condition unknown and do not fabricate a graph path. When the final
physical state is known from acknowledged actions and public observations,
retain supported non-goal achievements that still hold. An API failure or
exhausted model-attempt budget alone does not erase observed progress or establish
a full-decision-opportunity agent failure. Unacknowledged physical actions still
leave terminal state unknown. Only an official success witness permits a goal
node. Historical observations do not supply current-execution scores. The Host
keeps all dev conditions in the denominator; original receipts and costs remain.
The episode index includes recorded_outcome with the Planner's original finish
claim, official verdict, termination and request limits/counts. If the Planner
claims success but the official verdict is false, expose that completion gap.
Recheck the hypothesized backbone against admitted task requirements and original
witnesses: visible placement/contact/material may establish partial progress
without establishing required orientation, quantity, sequence or registration.
These are possible checks, not new universal requirements. Do not erase supported
physical achievements or invent a hidden evaluator cause. Explain what remains
unobserved and what evidence could distinguish missing requirements from failed
execution. Separate actual resource exhaustion from repeated ineffective work.

Use context/ANALYSIS_OPERATIONS.md for draft/validation workflow and exact-reference construction; validate_graph() is the final Host structural gate.
Semantic distinctions require task-progress and acquisition evidence; measurement-bin
changes alone do not require new semantic nodes. Represent consecutive observations
with spans, retaining every required original frame and interval.
{{projection_contract}}

EVERY node, including every optional branch, must represent meaningful task
progress that was not obtained for free. Name the positive condition acquired,
explain how it enables or completes this task, and cite evidence of its acquisition
through actual observation, grounding, tool use or control. Initial object placement,
an open gripper, "not yet grasped", waiting, movement alone, wrong-object contact,
and merely issuing a call are not task-progress nodes. Do not turn every physical
state or failure category into a node. Failure mechanisms and wasted work remain
in witnessed attempts, state spans and original traces. Recovery qualifies only
when it actually restores a task-relevant capability or condition.

The first node must be the first meaningful acquired condition, not a synthetic
start node. Correct target grounding can qualify when the agent actually localizes
or disambiguates the task object in execution evidence; a task-provided name,
simulator identity, initial visibility or an analyst recognizing it is not evidence
that the agent acquired this grounding. This is an example, not a required first
stage for every task. Do not impose a physical-action minimum: useful perception
can be earned without robot motion. Repeated checks add no new achievement.

{{refine_structure}}Programs are separate from physical transitions.
Represent shared physical interfaces by shared edges even when programs, seeds
or repeated attempts differ. {{split_interfaces}}Retain acquisition and regression spans so Host can derive current unfulfilled
transitions and remaining task progress from graph structure. Do not add a separate
failure-label taxonomy or author count. Unattempted backbone
relations are hypotheses, never invented failed executions. {{state_granularity}}A later outcome can challenge an
earlier interpretation, but never defines an earlier state as eventually successful.
Inspect actual tools, arguments, W/C/L code and original control intervals to
understand why A->B succeeded or failed. Preserve uncertainty in causal claims.
For physical-process claims, use read_video or ordered original images around the
relevant transition, align them with calls and telemetry, and compare actual
successful continuations when available. A final image or visual proximity does
not establish reachability, an obstruction's cause, or that extra budget suffices.
Keep missing/occluded visual evidence explicit. Do not require a full media tour.
Start with input.json's new_episode_ids and author_topology_feedback. The latter
records agree / modify / refuse assessments of specific claims in the preceding
round's graph, with reasons and inspected evidence. Check these against original
witnesses and subsequent outcomes; agreement is local, and silence is unreviewed.
In the existing changes field, briefly state which material assessments you adopted
or left unresolved and why. No separate review report or graph-wide re-audit is needed.
Accepted corrections must update or remove the affected graph claims themselves;
mentioning a correction only in changes does not revise an obsolete explanation.
{{projection_feedback}} Older assessments
and full author notes are indexed in context/history_originals.json when needed.
Do not adopt a diagnosis because an author asserted it. {{refine_edges}}Compare
the same implementation under different conditions and different implementations
under comparable conditions. Preserve unresolved alternatives when evidence cannot
distinguish them. This feedback revises the graph, not future authors' repair plans.
Do not prescribe a next repair, select this task's parent Harness, supply a separate
official-breakthrough probability, or determine search width. Host selects the parent and width under
the frozen protocol. There is no version donor selection: all historical
mechanisms remain accessible through program/witness views and their source
Bundles. Program attribution remains evidence, not causal proof.

{{forecast}}

The FIRST graph analysis establishes the task's initial backbone from the task
instruction and H0 evidence, even if every H0 episode failed. Humans supply no
milestone list, weights, or repair plan. The same graph contains role=milestone
for necessary earned achievements, role=branch for optional earned achievements
or useful execution modes, and role=goal for evaluator completion. All three
roles obey the same meaningful, non-free progress requirement.
{{backbone_revision}}Necessity before observed success is an explicit hypothesis, not a
claim that all valid routes have already been tested.{{backbone_contradiction}}

Node predecessors reference immediate MILESTONE NODE IDs in this same graph,
whose achievements remain satisfied at this state. They form an acyclic task
backbone, not an extra fact vocabulary or a prescribed method sequence. All
predecessors are jointly satisfied, so alternate implementations are branches,
not additional mandatory achievements. A milestone adds itself, a branch adds
nothing, and each goal includes the entire milestone backbone. Do not subdivide
a branch or rename a physical detail merely to increase progress. Regressed
states link to the earlier milestones whose effects still hold. Normal release
after successful placement must not erase completed task progress.

Keep earned conditions distinct where useful. Unobserved milestone/goal
nodes are allowed with explicit uncertainty; optional branches need observed
members. Only officially successful final frames may use a goal node. Host
computes effective progress from these graph relations, never from model-provided
numbers or a separately defined progress_points/achieved_points vocabulary.

Reproject EVERY historical episode under this one graph.
{{validation_history}}The selection map compares complete dev rows. cells and validation_cells map each
version and condition to an episode handle or null; episodes stores each measured
projection once.{{validation_pool}}
All original frame indices must be tiled exactly once. Use status=node with a node_id only
when its earned progress condition is supported. Use status=background and
node_id=null for known absence of retained task progress; it is zero progress,
not a node or an achievement. Use status=unknown and node_id=null with a reason
when evidence is insufficient; Host preserves null progress, not an invented zero.
Do not discard an already earned condition merely because the gripper is empty:
retain the supported progress node while its task-relevant achievement still holds.
Semantic edges describe actually attempted mechanisms and carry contiguous
execution witnesses, with achieved/not_achieved/unknown LOCAL outcomes. A null
edge source means known background, allowing attempts to the first meaningful
node without creating a free start node. The target is a real progress node;
failed attempts may end in background or at earlier retained progress. Unknown
start states and intervals without an evidenced mechanism remain unclassified.
Every original frame interval must
belong to one edge witness or an explicit unclassified interval. No overlapping
witnesses, no invented methods and no positive edge across an unknown action or
reset. Keep complete originals and underlying internal state paths. Graph path
concatenation across different programs or conditions is not proven composition.

{{matching}}Each edge declares object roles, entry_conditions and exit_conditions. Entry
includes relevant physical prerequisites and publicly available history; exit
describes the acquired condition AND what matters for continuing afterwards.
Do not label an entrance by a future success/failure. Interface conditions are
evidence-backed interpretations, not proven universal pre/postconditions.

Mechanisms include online Planner decisions: what evidence was available, which
action was selected, how feedback changed the plan and what supported continuation.
Ground these in recorded model requests/responses and tool feedback. C/L-mediated
direct-tool behavior can realize an edge without any W. A preceding failed W may
have contributed useful state or evidence, but does not by itself explain root
recovery; an unused helper cannot have performed it. Separate observed decisions
from hypotheses about which instructions caused them.
Preserve reusable conditions and relations while keeping episode-specific values
in witness bindings and original calls. Do not abstract away a condition that
changes applicability, or promote one observed strategy into a mandatory policy.

Store actually used Harness fragments once in graph.programs. Each program binds
one COMPLETE archived AgentBundle from context/history_originals.json and sorted
JSON pointers into its workflows/context/loop assets. Keep code in its original;
do not copy it into graph prose. Include coupled W/C/L dependencies when supported,
and record uncertain isolation/causal assumptions. A program may support multiple
edges or a multi-node path; an edge may have multiple programs. Do not force one
edge per function or per version. Each witness names the actual program_id and
binds exactly the edge's object roles to observed entities. Unresolved program or
entity attribution is explicitly null with uncertainty, not an invented binding.
The Host verifies that each bound program belongs to that witness's executed
Bundle. This establishes provenance, not an independently callable plugin.

Explain outcome differences using actual entry/exit conditions, program and
arguments, including failures and regressions. A successful local transition may
still leave an unsuitable next input. {{granularity}}Do not prescribe the next author's repair or extract speculative future plugins.

Support semantic claims with exact evidence references, including original code
or media when relevant. Empty pointers denote the complete file; JSON pointers
select exact original JSON/JSONL values. Host verifies references and derives backbone progress
but cannot certify your physical interpretation. Explain the changes and their
basis in changes; retain the full updated graph, not a patch or instructions.
Use read_original(reference=...) with an issued file handle and optional JSON
pointer for graph, trace and history originals. The Host retains exact provenance.
read_original returns indexed image originals as native images; use read_video
for selected original video frames.
"""


_GRAPH_ENTRY = """Update the shared, evidence-grounded task model in ONE independent round session.
Read context/input.json, task_instructions.json and analysis.schema.json. Start from
the seeded graph, new_episode_ids and author_topology_feedback, rather than replaying
past conversations. Use graph summaries and targeted original traces/images/code;
context/GRAPH_REFERENCE.md explains detailed representations when needed.
Only correction/recovery continues this session. Write {{exactly}}analysis/result.json.

The graph represents earned task conditions, dependencies, conditional transitions
and their observed Harness implementations. Use role=milestone for necessary earned
achievements, branch for alternative/control details and goal only with official
success evidence. Initial scene facts are not earned progress. Preserve retained
achievements, losses, recoveries and unresolved alternatives; visual proximity alone
does not establish contact, possession or completion. Use ordered visual evidence
for physical claims and distinguish facts from causal hypotheses.
{{backbone_revision}}{{revise_interpretations}} {{refine_structure}}
Programs are separate from physical transitions: reference actual archived W/C/L
code and witnessed invocation/continuation, not proposed repairs. Shared state labels
do not establish compatible entry/exit conditions. Preserve useful distinctions;
do not inflate labels or impose one Workflow per edge. {{refine_edges}}

Author agree / modify / refuse records assess specific claims of the preceding
graph. Verify their evidence and subsequent outcomes; silence is unreviewed.
Do not adopt a diagnosis because an author asserted it. In changes, briefly record
which material feedback you adopted, rejected or left unresolved and why.
Accepted corrections must update the affected graph claims, not only changes.
{{projection_contract}}
Do not reconstruct unchanged history from memory or copy code/trajectories into prose.
Original notes and older feedback remain indexed in history_originals.json.

Infrastructure failure leaves official capability unknown; it does not make an
acknowledged physical state unknown. Retain supported non-goal achievements that
still hold. Unacknowledged physical actions still leave terminal state unknown.
Historical observations do not supply current-execution scores. Current task rules
do not rewrite historical verdicts. {{validation_history}}
Host derives retained progress and handles SR-first parent selection and width.
Do not prescribe a next repair, select this task's parent Harness or assign authors.
{{forecast}}

{{retrieval}}{{matching}}
Read ANALYSIS_OPERATIONS.md for native-history, evidence and validation contracts.
Use validate_graph() on your complete draft; repair reported errors and finish once
the supported output passes. No candidate editing, model probes or physical trials.
Write only analysis/ and .scratch/. Validation checks provenance and consistency,
not the truth of a physical interpretation.
"""


def _graph_instructions(request, workspace, *, detailed=False):
    """Compose the analyst prompt from the sections this protocol uses."""
    protocol = request.protocol
    fields = {
        **_STRUCTURE,
        "exactly": "exactly ",
        "retrieval": "",
        "matching": "",
        "validation_conditions": "",
        "validation_history": "",
        "validation_pool": "",
        "author_space": "Authors may change the complete W/C/L Harness; poor past C\nrewrites do not establish the limit of that search space.",
        "width_use": "Host reads the probability on 100\nas one candidate's chance of completing the development set and combines it with\nplanning costs and the remaining rounds to select width.",
        "projection_contract": "Return the complete, coherently reprojected graph with every historical episode/frame/action interval covered. Revisit old interpretations affected by new evidence; preserve supported assignments. The draft is seeded with the previous complete graph, and context/input.json lists new_episode_ids. Unresolved intervals remain explicitly unclassified.",
        "projection_feedback": "Revisit older evidence where findings change its interpretation; return a complete, coherently reprojected graph.",
    }
    if archived_author_policy(request.policy).graph_retrieval is not None:
        fields.update(retrieval=_RETRIEVAL, matching=_MATCHING)
    elif needs_matching(request):
        fields["matching"] = _MATCHING.replace(
            "When cross-task retrieval is configured, maintain graph.matching on this SAME",
            "Prepare graph.matching for downstream shared-harness distillation on this SAME",
        ).replace(
            'Read the shared catalog\nwith read_reference_graph(view="catalog"); optional text lookup locates relevant\ncoordinates. Reuse each definition, ordered argument meaning and mutually exclusive\nvalue domain verbatim. Add a fluent only for a genuinely unexpressible property,\nnot a synonym.',
            "Use a consistent task-local catalog of evidence-grounded properties, ordered arguments and value domains; reuse equivalent definitions.",
        )
    else:
        fields["matching"] = "Omit graph.matching in this phase; encode it in final analysis for distillation.\n"
    if protocol.validation is not None:
        fields.update(_VALIDATION)
    if (workspace / "context/demonstrations.json").exists():
        fields["exactly"] = ""
    if protocol.mutation_scopes != ("W", "C", "L"):
        fields["author_space"] = (
            "Authors may change only "
            + ", ".join(protocol.mutation_scopes)
            + ". Poor past outcomes alone do not establish the limit of the permitted search space."
        )
        if request.needs_forecast:
            fields["author_space"] += (
                " The forecast must describe that permitted candidate space; exclude improvements that require modifying a read-only component."
            )
    if (
        protocol.topology_mode == "fixed_initial"
        and request.previous_analysis is not None
    ):
        # Only structural-change permissions are omitted; the evidence, witness,
        # history-objection, no-repair-prescription and forecast rules remain.
        fields.update(dict.fromkeys(_STRUCTURE, ""))
        fields["revise_interpretations"] = (
            "Update physical evidence assignments under the fixed topology definitions."
        )
    fields["forecast"] = (
        _render(_FORECAST_INSTRUCTIONS, fields)
        if request.needs_forecast
        else "No probability forecast is needed for this analysis; omit forecast."
    )
    return _render(_GRAPH_INSTRUCTIONS if detailed else _GRAPH_ENTRY, fields)


def _render(template, fields):
    for name, value in fields.items():
        template = template.replace("{{" + name + "}}", value)
    if "{{" in template:
        raise ValueError("graph analysis prompt has an unrendered section")
    return template


class NativeGraphAnalysis(NativeSession):
    """New session per round; same-session recovery within an unfinished round."""

    async def _run_native(self, request, directory, *, resume, before_launch):
        if self.native_runtime_root is None:
            raise ValueError(
                "graph analysis requires its run-owned native runtime directory"
            )
        self.native_runtime_root.parent.mkdir(parents=True, exist_ok=True)
        with (self.native_runtime_root.parent / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "graph analysis is already active for this run"
                ) from error
            try:
                return await super()._run_native(
                    request, directory, resume=resume, before_launch=before_launch
                )
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _read_request(workspace):
        return GraphAnalysisRequest.model_validate_json(
            (workspace.parent / "audit/inputs/analysis_request.json").read_text()
        )

    def _prepare_inputs(self, request, directory, *, resume):
        if archived_author_policy(request.policy) != self.policy:
            raise ValueError("graph analysis native policy differs from frozen request")
        if request.round_index == 0 and request.previous_analysis is not None:
            raise ValueError("initial graph analysis cannot have a previous analysis")
        if request.round_index > 0 and request.previous_analysis is None:
            raise ValueError("later graph analysis requires its previous graph receipt")
        super()._prepare_inputs(request, directory, resume=resume)
        if request.previous_analysis is not None:
            request.previous_analysis.verify()

    @staticmethod
    def _prompts(request, _workspace):
        instructions = _graph_instructions(request, _workspace)
        current_rules = _workspace / "context/current_task_rules.json"
        if current_rules.is_file():
            instructions += (
                "\nCurrent published task contract for this request (historical verdicts stay unchanged):\n"
                + current_rules.read_text()
                + "\nRead context/current_task_rules.json before interpreting old guidance."
            )
        if (_workspace / "context/demonstrations.json").exists():
            instructions += (
                "\nThis is an explicitly demonstration-conditioned round. Original expert videos are available at the read-only paths below. "
                "Choose how to inspect the original files yourself with native media tools, shell/Python and the native image viewer. "
                "No frames or viewing sequence have been preselected. "
                "Also write analysis/demonstration_review.json with core_solution_steps, observed_demo_facts (video handles and frame indices), "
                "task_rule_checks, possible_task_misunderstandings, failure_comparison, proposed_CWL_changes, and uncertainties. "
                "Compare the demonstrated contact/manipulation mode and each finishing condition with public rules and actual failed episodes. "
                "Identify applicable task constraints and possible prior misconceptions; do not transplant another task's restrictions. "
                "A demonstration suggests a method; it is not a scored agent episode or proof of improvement. Never create goal witnesses or scores from it. "
                "Keep observed facts, causal hypotheses and remaining ambiguity separate. The review is a diagnostic handoff, not a substitute for the complete graph."
                " The output contract accepts result.json plus this optional diagnostic review. In-session validate_graph uses the same output contract "
                "and declared retrospective witnesses as final admission; unmeasured latest-parent cells remain unknown."
                + "\nOriginal demonstration videos:\n"
                + (_workspace / "context/demonstrations.json").read_text()
            )
        if request.protocol.mutation_scopes != ("W", "C", "L"):
            authority = authoring_scope_contract(request.protocol.mutation_scopes)
            instructions = (
                "The next mutation author can edit "
                + ", ".join(authority["editable_components"])
                + "; its read-only components are "
                + ", ".join(authority["read_only_components"])
                + ". These are the next author's permissions. Your own output remains analysis/ only; do not edit a candidate. "
                "Keep the same physical goal, topology semantics, evidence coverage and measurement rules; never redefine progress to make the permitted edits look better.\n\n"
                + instructions
            )
        fixed = (
            request.protocol.topology_mode == "fixed_initial"
            and request.previous_analysis is not None
        )
        if fixed:
            instructions += "\nThe initial topology is fixed. Preserve every node/edge ID, node meaning/role/predecessors and edge source/target/roles/entry/exit conditions exactly. Update only evidence, uncertainty, explanations, programs, witnesses and complete episode mappings. Unmatched intervals remain unclassified; do not add, delete, split or merge nodes or edges. Reproject every historical trajectory on this same fixed topology."
        prompt = f"Analyze frozen round {request.round_index + 1}. Continue from the structured graph handoff using current immutable inputs."
        if request.purpose == "final_selection":
            topology = "the same fixed topology" if fixed else "one updated topology"
            evidence = "include the last mutation outcomes and reproject ALL settled historical episodes"
            prompt = (
                f"Perform final archive selection analysis on {topology}: {evidence}. "
                "This analysis creates no authors or physical trials and is not another evolution round. No candidate forecast or search-width allocation is required."
            )
        return (
            instructions,
            prompt,
        )

    def _prepare(self, request, workspace):
        context = workspace / "context"
        context.mkdir(parents=True)
        private = workspace / ".host_inputs"
        private.mkdir()
        atomic_json(private / "analysis_request.json", request.model_dump(mode="json"))
        navigation, implementation = self._prepare_evidence(request, workspace)
        catalog = AuthorReferences.load(workspace)
        with catalog.batch():
            self._publish_graph_inputs(
                request, workspace, navigation, implementation, catalog
            )

    def _publish_graph_inputs(
        self, request, workspace, navigation, implementation, catalog
    ):
        context = workspace / "context"
        history = json.loads(Path(request.history.originals.path).read_text())
        author_feedback = [
            review
            for review in history["navigation"].get("latest_topology_feedback", [])
            if review["round_index"] == request.round_index - 1
        ]
        request.graph.verify()
        raw = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
        atomic_json(
            context / "execution_graph.json",
            catalog.encode(raw.model_dump(mode="json")),
        )
        previous_graph = None
        previous_result = None
        previous_map = None
        if request.previous_analysis is not None:
            request.previous_analysis.verify()
            previous_receipt = GraphAnalysisReceipt.model_validate_json(
                Path(request.previous_analysis.path).read_text()
            )
            previous_result = previous_receipt.analysis
            from reweave.evolution.graph.projection import remap_episode_receipts

            previous_graph = remap_episode_receipts(
                previous_result.graph, history["navigation"].get("receipt_aliases", {})
            )
            previous_map = project_map(
                raw,
                previous_graph,
                request.versions,
                request.protocol.dev,
                graph_id=previous_receipt.record_id + "/analysis/graph",
                cell_receipts=analysis_graph_receipts(request, raw),
                validation_conditions=(
                    request.protocol.validation.conditions
                    if request.protocol.validation
                    else ()
                ),
                partial=True,
            )
        atomic_json(
            context / "graph.json",
            catalog.encode(previous_graph.model_dump(mode="json"))
            if previous_graph is not None
            else None,
        )
        atomic_json(context / "progress_map.json", catalog.encode(previous_map))
        atomic_json(
            context / "analysis_request.json",
            catalog.encode(
                {
                    "record_id": request.record_id,
                    "round_index": request.round_index,
                    **(
                        {"purpose": request.purpose}
                        if request.purpose == "final_selection"
                        else {}
                    ),
                    "conditions": [
                        c.model_dump(mode="json") for c in request.protocol.dev
                    ],
                    "validation_conditions": [
                        c.model_dump(mode="json")
                        for c in (
                            request.protocol.validation.conditions
                            if request.protocol.validation
                            else ()
                        )
                    ],
                    "versions": [v.model_dump(mode="json") for v in request.versions],
                    "author_wall_time_s": request.protocol.author_wall_time_s,
                    "execution": request.protocol.execution.model_dump(mode="json"),
                    **(
                        {
                            "authoring_authority": authoring_scope_contract(
                                request.protocol.mutation_scopes
                            )
                        }
                        if request.protocol.mutation_scopes != ("W", "C", "L")
                        else {}
                    ),
                    **(
                        {
                            "width_policy": {
                                "mode": "fixed",
                                "authors_per_round": request.protocol.max_authors_per_round,
                                "forecast_use": "disabled",
                            }
                        }
                        if request.protocol.width_mode == "fixed"
                        else {}
                    ),
                    "execution_graph_id": raw.record_id,
                }
            ),
        )
        schema = catalog.schema(analysis_result_schema(request))
        if (
            request.protocol.topology_mode == "fixed_initial"
            and previous_graph is not None
        ):
            from reweave.evolution.graph.refinement import topology_definition

            definition = topology_definition(previous_graph)
            for section, model in (
                ("nodes", "SemanticNode"),
                ("edges", "SemanticEdge"),
            ):
                schema["$defs"]["SemanticGraph"]["properties"][section] = {
                    "description": "Initial topology definitions are immutable; only evidence and witness fields may change.",
                    "type": "object",
                    "required": list(definition[section]),
                    "additionalProperties": False,
                    "properties": {
                        key: {
                            "allOf": [
                                {"$ref": f"#/$defs/{model}"},
                                {
                                    "properties": {
                                        field: {"const": value}
                                        for field, value in item.items()
                                    }
                                },
                            ]
                        }
                        for key, item in definition[section].items()
                    },
                }
            matching = previous_graph.matching
            if matching is not None:
                schema["$defs"]["GraphSemantics"]["properties"].update(
                    {
                        key: {"const": value}
                        for key, value in definition["matching"].items()
                    }
                )
        atomic_json(context / "analysis.schema.json", schema)
        atomic_json(
            context / "input.json",
            catalog.encode(
                {
                    "kind": "graph_analysis",
                    "request": request.record_id,
                    "implementation": implementation,
                    "index": navigation,
                    "task_instructions": "context/task_instructions.json",
                    "graph": "context/graph.json",
                    "execution_graph": "context/execution_graph.json",
                    "progress_map": "context/progress_map.json",
                    "new_episode_ids": sorted(
                        {episode.receipt_id for episode in raw.episodes}
                        - (
                            set(previous_map["episodes"])
                            if previous_map is not None
                            else set()
                        )
                    ),
                    "author_topology_feedback": author_feedback,
                    "output": "analysis/result.json",
                    "history": {
                        "summary": "context/EVOLUTION_HISTORY.md",
                        "originals": "context/history_originals.json",
                    },
                }
            ),
        )
        from reweave.evolution.authoring.client import development_client

        client = context / "host_client.py"
        client.write_bytes(
            (Path(__file__).parent.parent / "authoring/client.py").read_bytes()
        )
        atomic_text(context / "dev.py", development_client())
        atomic_text(
            context / "DEVELOPMENT.md",
            f"""Use {sys.executable} context/dev.py --help for the workspace-bound Host tools.
Python: from context.dev import call
Start with call('read_graph', view='summary'). For an episode, select its receipt_id
as identity, with inclusive first_frame/last_frame and optional features. summary
lists feature groups; view='features', identity='<prefix>' lists exact names.
The response reports original indices and selected coverage. node reads
one partition signature; semantic node reads one state with its complete references.
Semantic program reads its original Bundle slice and dependencies; map summary
is compact, and map version/episode/program selects exact evidence. Map transitions
selects a literal edge ID; witness selects an issued transition handle.
read_trace, read_original and read_source preserve exact evidence access.
read_original restores journal image_payloads; read_graph(view="episode") resolves
frame feature_source references to original measurements, including units and sources.
Write analysis/result.json, then call('validate_graph') or run
{sys.executable} context/dev.py validate-graph. A passing receipt binds the checked
bytes; errors and complete reports stay under audit/development_reports. This is
the same structural gate as final publication, not certification of semantics.
Finish the native turn after the supported result passes; the Host independently
rechecks final bytes and completion provenance. No candidate validation or physical actions.
""",
        )
        atomic_text(context / "ANALYSIS_OPERATIONS.md", _ANALYSIS_OPERATIONS)
        atomic_text(
            context / "GRAPH_REFERENCE.md",
            _graph_instructions(request, workspace, detailed=True),
        )
        (workspace / "analysis").mkdir()
        if previous_graph is not None:
            atomic_json(
                workspace / "analysis/result.json",
                catalog.encode(
                    GraphAnalysisResult(
                        graph=previous_graph
                        if needs_matching(request)
                        else previous_graph.model_copy(update={"matching": None}),
                        changes=PENDING_GRAPH_REFINEMENT,
                        forecast=previous_result.forecast
                        if request.needs_forecast
                        else None,
                    ).model_dump(mode="json")
                ),
            )
        (workspace / ".scratch/runtime").mkdir(parents=True)

    def _correction_prompt(self, feedback):
        return (
            "Continue this same graph analysis session. Correct analysis/result.json using the Host feedback, preserving the full shared graph, original evidence, and physical state semantics. "
            + json.dumps(feedback)
        )

    async def _publish(self, directory, *, timing_start=None, before_launch=None):
        request = self._read_request(directory / "workspace")
        if (directory / "result.json").exists():
            return self.verify_result(request, directory)
        started, started_at = timing_start or (time.monotonic(), time.time())
        audit = directory / "audit"
        checks = audit / "checks"
        checks.mkdir(exist_ok=True)
        check = checks / f"{len(list(checks.glob('*.json'))):06d}.json"
        atomic_json(
            check,
            {
                "started_at": started_at,
                "wall_time_s": 0,
                "wall_time_complete": False,
                "validation_status": "running",
            },
        )
        completion = self._load_completion(directory)
        status = "failed"
        try:
            result = validate_analysis_output(request, directory / "workspace/analysis")
            status = "passed"
        except (ValueError, KeyError, OSError, StopIteration) as error:
            self._save_correction(directory, check, error, case_results=[])
            raise AuthorCorrectionRequired(str(error)) from error
        finally:
            atomic_json(
                check,
                {
                    "started_at": started_at,
                    "wall_time_s": time.monotonic() - started,
                    "wall_time_complete": True,
                    "validation_status": status,
                },
            )
            usage = self._update_usage(directory)
        receipt = GraphAnalysisReceipt(
            record_id=new_record_id(),
            request_id=request.record_id,
            analysis=result,
            session_id=completion["session_id"],
            usage=usage,
            artifacts=tuple(_artifact(p) for p in _publication_paths(directory)),
        )
        atomic_json(directory / "result.json", receipt.model_dump(mode="json"))
        return receipt

    def verify_result(self, request, directory):
        """Reuse the settled analysis for its actual saved request."""
        self._check_request(request, directory / "workspace")
        result = GraphAnalysisReceipt.model_validate_json(
            (directory / "result.json").read_text()
        )
        if result.request_id != request.record_id:
            raise ValueError("graph analysis receipt belongs to another request")
        return result
