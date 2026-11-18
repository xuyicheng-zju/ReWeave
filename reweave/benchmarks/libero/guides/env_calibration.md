# LIBERO camera and controller coordinates

Object geometry comes from current RGB and depth, not BDDL, simulator object
poses or task-number lookup tables. Robot proprioception and camera/controller
calibration are exposed by the tools.

## Images and pixels

`image_inputs` identifies attached images in order. `agentview_policy.png` has
the VLA's orientation; do not select back-projection pixels from it.
`agentview.png`, `agentview_high.png`, `wrist.png` and `wrist_high.png` use the
calibration frame. Select `[row, col]` in that image, with row from the top and
col from the left. Use its exact saved step, camera and resolution:

```json
{"row": 500, "col": 400, "step": 2, "camera": "agentview", "resolution": "high"}
```

The numbers above illustrate the schema, not a task target. `high` uses the
matching high-resolution world map; `low` uses the native-resolution map.
`view_camera_meta` exposes the calibration. Wrist extrinsics change with robot
motion, so old wrist pixels must not be paired with a new step. All returned
world XYZ share one world frame; consistency does not prove that both cameras
selected the correct object.

`back_project` returns the visible surface under a pixel. It cannot recover a
hidden object center. A rectangular region includes every visible surface in
that rectangle; its bounding center can be background-biased. `segment` returns
a mask-based surface estimate; confirm identity and select contact geometry
separately. Errors and missing artifacts are not usable coordinates.

## World motion

XYZ and translation clips are in meters, rotation targets and clips in radians.
The Host queries the actual OSC input/output limits and controller reference
frame, then converts world displacement into normalized actions. `control_spec`
in a Cartesian tool result records that conversion. The model cannot set an
`action_scale`.

Yaw is `atan2(R[1,0], R[0,0])`. The tool's historical `pitch` name denotes
`atan2(R[1,2], -R[2,2])`, the projected gripper-axis tilt toward world +y from
world -z; it is not the conventional Euler rotation about world Y. Rotation
increments for this tilt use world X. These projected angles have singular
orientations; use small, observed changes rather than assuming arbitrary 3D
pose reachability. Position and requested angle tolerances determine servo
arrival; a controller or contact limit can exhaust the budget first.

Table/support height must be measured in the current scene. Initial EEF height
alone does not establish table height, reachable workspace, safe carry height
or container interior. Use visible clearance and current tool residuals.

## Optional GraspGen-X proposals

Pass `--graspgen-config /absolute/path/to/graspgen.yaml` to register the
read-only `generate_grasps` tool. The config must use a `tcp://` endpoint and
contain a `right` Panda gripper binding with an explicitly calibrated
`grasp_frame_from_tcp` rigid transform. No Panda transform is inferred by the
adapter. A request applies SAM3 to a recorded agentview or wrist RGB frame and
uses only the aligned measured depth/world-map artifact from that same recorded
step. Returned TCP poses use `libero_world` and include the step, camera, image,
depth, and mask prompt as provenance. The proposals do not certify IK
reachability, scene collision, or path collision. Use `move_pose` with
`target_quaternion_xyzw` to command the complete candidate orientation. The
tool reports the measured shortest SO(3) residual, including roll; pitch/yaw
mode remains available separately and must not be substituted for a generated
quaternion.
