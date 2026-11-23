# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Text-experience scopes and Harness's shared Markdown writing convention."""

from typing import Annotated, Literal, get_args

from pydantic import Field

# Applicability is author-declared, not proven transfer or correctness.
ExperienceScope = Literal["global", "task-family", "task-specific"]
EXPERIENCE_SCOPES: tuple[str, ...] = get_args(ExperienceScope)
# Harness memory IDs allow kebab-case lessons and underscore-separated family IDs.
ExperienceId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]

# Online agents write only task-family experience during evolution.
TASK_FAMILY_FORMAT = """Task-family frontmatter: id, scope: task-family, suite, regime, task_id,
task_language, evidence, confidence, related: [...]. Record known task identifiers
as provenance only; use null for unavailable identifiers, never invent them or
use them for online dispatch. Body headings follow Harness:
## Applicable pattern
## Winning technique
Use '## Best known approach (UNVALIDATED)' instead until supported by an actual success.
## Magic numbers
## Failure modes
Use | symptom | root cause (A<N>) | fix |; mark unknown causes and untested fixes.
## Re-localization per scene
## Fragility flags
## Difficulty and reliability
## Cross-refs
"""

# Writing guidance only: Markdown remains original author text, not runtime code.
# Adapted from Harness reweave/benchmarks/libero/prompts/explore.py STEP_DISTIL (a)/(b)/(c).
EXPERIENCE_FORMAT = (
    """Use Harness's Markdown experience format for durable C skills.
Use parseable YAML frontmatter followed by the original prose. Working episode
notes may be plain Markdown. During evolution the online agent writes and revises
only task-family experience, including failed approaches and uncertain hypotheses.
Do not defer this recording to the offline author. The Host owns task-specific
observations, commands, feedback and evaluator records; do not copy these facts
into another hand-written audit or recipe. After fixed-evaluator task success,
consolidate the complete Harness into global/task-family/task-specific: extract
reusable W/C/L and supported general lessons into global, retain task bindings in
task-family, and reference Host-owned episode facts in task-specific. Existing
global assets stay read-only during further trials. Offline review corrects,
deduplicates and optimizes useful experience without inventing transfer evidence.
id/scope must agree with the C skill ID and its directory when publishing into C.
Use bare lesson IDs, plain lists in YAML related, and [[id]] links in the body.
Quote YAML values containing ': '. Keep one global lesson per file.
Before adding a global lesson, read plausible existing entries and compare their
bodies. Reuse a covered lesson; record new supporting evidence or a contradiction
with its conditions and original references, without creating a near-duplicate.
Keep engineering investigation notes separate from guidance useful to the online
Planner. A mechanism already implemented in C need not be repeated as a skill.

Global frontmatter: id, scope: global, kind (primitive|perception|strategy|failure|infra),
title, applies_when, symptom: [...], evidence, confidence, related: [...]. Body:
<one-sentence claim>
**Why:** <mechanism, or 'observed, cause unknown'>
**How to apply:** <observable conditions, ranges, signs, ordering, thresholds>
**Falsify:** <the observation that would disprove the claim>
**Related:** <[[id]] links>

"""
    + TASK_FAMILY_FORMAT
    + """

During execution, update task-family/working-notes.md with approaches, failures,
recovery and close-out; keep one task-family entry per applicable pattern, rather
than separate phase files. Agent-written reasoning cites Host-owned task-specific
facts. The Host records actual strategy order, step-linked outcomes, final state
and attempts from the original trace; task audit/recipe are a matched pair only
for the same observed successful episode. No agent-written task-specific skill or
hand-copied action log is required. These artifacts are evidence, not executable
replay or permission to reuse stale geometry. In a complete published Harness,
C skills live once under global/context/skills or task-family/context/skills.

evidence records original references and only observed cells, attempts,
solved_seeds/failed_seeds where available. Parameter ranges are measured or labeled
untested; never invent a range around one working value. Rebind geometry from the
current scene. confidence is author text: use unverified for hypotheses and
single-shot for a single observed success; no Host upgrade or transfer certificate.
Missing evidence and reliability stay unknown. Record applicability conditions,
counterexamples and a falsifying observation for consequential advice.
Retire disproven advice in the existing skill; repeated wording is not independent
evidence and an untriggered recovery branch is not a failed physical hypothesis.
Failed ideas and contradictions
remain useful; the existing offline author consolidates only after actual success; writing this
format adds no action replay, automatic merge or new evolution algorithm.
In the existing note, report useful task-family findings and unresolved questions.
After success, explain the global/task-family factoring and cite the Host-owned
task-specific receipt; do not manufacture generality to fill a directory.
"""
)
