# LIBERO baseline execution contract

This guide describes the current tools. It does not prescribe a task-specific
solution or assume a particular suite, seed, table height or object offset.

## Decision loop

Use the supplied scene and `task_language`; call `view_env_state(step=0)` only
if the initial observation is unavailable. Localization and file reads are not
prerequisites to every motion. A visually grounded VLA instruction can be enough;
coordinate tools need a justified estimate, which may already be available.

One Planner response may contain several tool calls, executed in order. Batch
independent reads and fully specified operations whose validity does not depend
on unseen intermediate results. A batch has no intervening model decision; do
not precommit a feedback-dependent action. A workflow may implement conditional
control internally. Physical tools return observations and `log.result`; use
those directly, with no fixed extra view/perception/model round after each call.
The saved `step` numbers observations, not physics steps.

Use additional perception to resolve a specific uncertainty. Wrist refinement
is useful near contact or occlusion, but is not a required tour of every object.
For repeated handles, inspect the original image and use a point prompt when
text segmentation keeps selecting the wrong one. Once location is sufficiently
clear, distinguish motion failure from perception failure: collisions, approach
angle, an obstructing object or the wrong tool require different actions.

## Tools and feedback

- `move_to` controls world XYZ and optional yaw. `move_pose` additionally accepts
  a pitch goal; both share the same servo and controller. Check `target_reached`,
  `stop_reason`, `final_dist_m`, orientation errors and `steps_used` in `log.result`.
  `motion_response` measures EEF path length and target-distance reduction over
  the last <=20 acknowledged steps, with the measured start/end positions. Low
  motion with residual error supports replanning, but does not identify its cause.
  An already-reached target executes zero steps, including zero gripper commands.
- `rotate_wrist` and `rotate_pitch` rotate while correcting XYZ drift. Their
  angle definitions are in the schemas. Contact or reach constraints can prevent
  convergence; invocation completion is not evidence the target was reached.
- `set_gripper` and `release` apply close/open commands for actual physics steps.
  +1 closes, -1 opens. Explicitly preserve +1 during carrying motions; defaults
  differ between tools. To open or close at rest, call a gripper tool.
- `pi0_pick` runs VLA forwards with a grasp instruction. `max_chunks` limits
  forwards inside this call, not Planner turns. The grasp heuristic uses
  chunk-end samples: `descent_thresh` descent (default 0.10 m) from this call's
  start, then `lift_thresh` ascent and finger separation in
  `[gripper_open_thresh, gripper_closed_thresh)`, default `[0.0, 0.06)` m.
  The descent threshold must be nonnegative and the opening bounds must satisfy
  `0 <= gripper_open_thresh < gripper_closed_thresh`. These tune the local
  heuristic, not the official task evaluator.
  It can miss a valid grasp begun close to the object, or accept an empty closed
  gripper. Confirm possession and identity visually. Repeatedly increasing the
  budget may let the VLA continue into placement; the tool is not a guaranteed
  grasp-only policy. Do not send a full pick-and-place instruction when only
  grasping is intended.
- `pi0_doubled` runs non-pick contact skills. `success=false` can coexist with a
  successfully opened drawer when the whole task is still unfinished. Inspect
  intermediate effects before repeating the contact command.

A call can execute many control steps before Planner supervision resumes.
Use adequate tool budgets; do not split every operation into short invocations
to force frequent Planner calls. Additional supervision can be implemented by
evolved W/C/L programs. H0 has no learned monitor; do not assume a useful intermediate stopping condition.
An exposed workflow may manage intermediate observations and decisions itself.
Use its declared behavior; do not assume an automatic modifier or replay.

## Geometry and evaluation

Use the calibration-frame image, camera, resolution and step together. See
[calibration](env_calibration.md). `segment` estimates visible surfaces; its red
mask does not describe object color. A surface median is not necessarily a
handle axis, grasp point or container interior.

Choose carry clearance from the visible path, and placement height from the
current destination and measured held-object offset. A previously published
constant height or bowl offset is not a universal rule. After release, first
inspect where the object actually is; descending an empty gripper cannot by
itself correct a misplaced object. Never move again after termination/truncation.

Only top-level `terminated=true` establishes official success. The environment
determines task success. Local failure requires feedback-driven recovery while
the episode remains active. `finish` reports a task-level conclusion; merely
calling or failing one workflow does not justify ending the episode.
Evaluation has one episode and no reset. If filesystem tools are available,
write the declared audit at the end; the Host exports the command recipe. With
`memory-profile=none`, no file reading/writing is available; the Host retains
the full evidence archive.
