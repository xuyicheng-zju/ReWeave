# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Public RoboDojo observations and generic dual-arm control contract."""

from collections.abc import Mapping

from reweave.prompt.utils import PromptNode

_GRASPGEN_GUIDANCE = """When generate_grasps is registered in the current tool contract, use it for measured
GraspGen-X grasp geometry when useful. It performs SAM3 internally on the same RGB-D
snapshot; a separate segment call is not required. Check the target identity and provenance.
Its isaac_world candidates already include the configured grasp-frame-to-control-frame
transform: do not apply another fingertip/TCP offset. Convert quaternion_xyzw [x,y,z,w]
to move_pose quaternion_wxyz [w,x,y,z]; approach_direction is a world vector, not an Euler angle.
GraspGen-X ranks proposals using its discriminator. A proposal is not a collision
or reachability approval. Plan an open-gripper pregrasp, reach the required orientation,
then use the native motion contract for approach, closure and lift; verify possession from
new observations. Scores and branch tags are model proposals, not calibrated success probabilities.
"""


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    memory_guidance = (
        "Published memory is disabled for this episode. Use current observations and private episode notes."
        if (variables or {}).get("memory_profile") == "none"
        else "Reference text under {{memory_dir}} is prior knowledge, never a fact about the current layout."
    )
    graspgen = bool((variables or {}).get("graspgen_enabled"))
    return (
        """You control a RoboDojo dual-arm robot using public RGB-D cameras and robot proprioception.
Three cameras are available: cam_head, cam_left_wrist and cam_right_wrist.
Use the resolved task language from view_env_state, never unresolved object placeholders.
back_project uses each camera's depth and calibration to return the visible surface world point.
This surface point is not converted into an EE target or grasp/contact offset.
*_ee_pose and move_to/move_pose use the same native configured EE frame; its origin is not
guaranteed to be a fingertip/pinch contact point or calibrated TCP.
segment.mask_point_rc is an integer foreground pixel for depth lookup; its true centroid_rc
may lie in a ring's hole or outside a concave mask. A mask sample is not a verified grasp point.
locate combines SAM3 and PyRUA-Lean footprint geometry on one exact RGB-D observation.
Use support_z only for an object known to stand on that support plane; otherwise height
is unavailable and the center describes visible surfaces. closing_axis_deg is a world XY
axis, not an EE quaternion. Re-localise after contact or occlusion; inspect identity.
@@GRASPGEN@@After a known failure, replan from the latest settled state; never replay an action with unknown ACK.
Missing segmentation/depth@@EMPTY_POOL@@ is a perception limitation: try another
view or supported measured geometry/control method when justified, rather than declaring task failure.
move_to and set_gripper act on the selected arm. Tool gripper arguments are 1=close, -1=open;
native *_ee_joint_state values use 1=open and 0=closed and are command-derived opening
targets, not measured aperture. Independent *_gripper_measured_opening values use the
actual gripper joint and runtime scale/sign (dimensionless, nominally 0=closed/1=open;
finite overshoot is retained). Neither channel proves grip force, contact or possession;
verify possession with images/depth. Holding a gripper and VLA input retain the native
command target rather than replacing it with a contact-blocked measured opening.
Native control still limits each queued gripper target relative to the actual joint
and stores the limited value for the next window. Thus gripper=0 reuses native
command history but does not guarantee an unchanged opening target during motion.
Check both opening channels and object retention after transport or reorientation;
a keep command alone does not certify a retained grasp.
set_gripper holds both observed arm joint
targets. move_to executes an explicit gripper command even if XYZ is already reached.
move_to plans one native collision-aware position trajectory with XYZ-only criteria;
orientation may change. move_pose takes world xyz and quaternion_wxyz, normalizes the
nonzero quaternion and plans a native full-pose trajectory. linear=true selects the
native linear-motion criterion along link6 +X. move_joints takes six explicit public
arm_joint_state angles in radians and plans a native joint-space trajectory.
All three execute every original position/velocity sample once through the official
control queue, with one acknowledgement per control window. They do not jump directly
to IK goals, search extra orientations, or replay the final target to settle residuals.
Supply tol_m/tol_rad for pose and tol_rad for joints; tolerances classify final measured
errors. max_steps is optional and defaults to the remaining episode budget.
Explicit values are preserved and the shared
remaining episode budget always applies. The complete path must fit both budgets before
any action. A rejected path or residual error is feedback for replanning, not a reason to
blindly increase max_steps. A completed trajectory returns even if a tiny tolerance is
not met; do not use tiny tolerances to consume time waiting for the opponent. Waiting
requires an explicit holding workflow with the gripper command semantics checked.
No generic no-effect hold tool is provided. A residual error is not an unknown ACK.
set_gripper reports the commanded gripper state; an ACK does not establish actual opening,
contact or possession. Verify measured opening and current images for those facts.
For a complete measured gripper stroke, set max_steps=10; it stops on measured opening
stability after at least two actions. The default remains one ACK for explicit holds.
An opening_stable result can mean contact, an obstruction or an empty closed gripper;
verify that the intended object moves with the arm. Every stroke action consumes budget.
Pi0.5 controls both arms with 14 joint values; pi0_pick's arm labels a reference arm only.
It stops only at its explicit max_chunks, official termination, budget or cancellation;
height and command-derived gripper closure are not possession or success tests.
Every native action consumes the episode's shared physical budget. Remaining actions in a
prediction are discarded on a tool return, cancellation or environment termination.
Base decisions on current observations and settled feedback; use finish only with task-level evidence.
Robot configuration: {{env_cfg_type}}.""".replace(
            "@@SENSING@@", "SAM, GraspGen, VLA, " if graspgen else "SAM, VLA, "
        )
        .replace("@@GRASPGEN@@", _GRASPGEN_GUIDANCE if graspgen else "")
        .replace("@@FAILURE_LIST@@", "An empty grasp pool," if graspgen else "A")
        .replace("@@EMPTY_POOL@@", " or an empty candidate pool" if graspgen else "")
        + "\n"
        + memory_guidance
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    return "Task: {{task}}. Read the initialized scene and its resolved instruction."
