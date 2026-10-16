# SO-101 simulated embodiment (S1, TASK-097)

Phase S1 of the SO-101 plan (`docs/SO101_PLAN.md`, DECISIONS 2026-10-16 (bp)). **Software only,
simulation only.** Nothing here is an experiment, a bar or a claim about manipulation; the
scripted reach below reads simulator truth and is never a learned result. The G1 + Dex3 files
(`simulation.py`, `embodiment.py`, `constraints.py`, `configs/g1_sim_action.json` and every other
hash-pinned file) are byte-identical.

## Setup

```sh
uv run --no-sync python scripts/fetch_so101.py      # pinned MuJoCo Menagerie robotstudio_so101
MUJOCO_GL=egl JEPA_TEST_RENDER=1 uv run --no-sync pytest tests/test_so101_embodiment.py
MUJOCO_GL=egl uv run --no-sync python scripts/dev_so101_reach.py --seeds 0 1 2 --frames <dir>
```

`scripts/fetch_assets.py` runs the same fetch after the G1 assets, so the integration CI job
(which calls it) has the model. The fetch writes into the ignored `third_party/mujoco_menagerie_so101` (a sparse checkout of
this one subtree at this one pin, so another Menagerie model gets its own directory; the same convention as the G1
assets: the manifest is committed under `assets/`, the checkout is not). Revision, licence and
per-file hashes: [DEPENDENCIES.md](DEPENDENCIES.md#so-101-simulation-model-task-097-2026-10-16).

## Scene (`so101_simulation.py`)

- Menagerie's `so101.xml` loaded unchanged (six `sts3215` position actuators, 5 ms physics step,
  implicitfast), extended in memory with: a table whose top is z = 0 with the robot base at the
  origin (+x is the arm's straight-out direction at zero joint angles); a free 3 cm cube (20 g,
  red); a flat 8 cm blue target marker without collision; a directional light and a sky
  gradient.
- Cameras, exposed as `wrist_rgb` and `overview_rgb` (default 224 × 224):
  - `wrist_rgb` is Menagerie's own `wrist_cam` on the gripper's `camera_mount` (fovy 61), reused
    rather than new geometry. It looks down past the fixed jaw, whose tip is in the lower part of
    the frame.
  - `overview_rgb` is a fixed camera at (0.52, −0.14, 0.34) m aimed at (0.14, 0.02, 0.05) m
    (fovy 55), front-right of the robot, covering the arm, the cube and the target.
- `read()` returns both images, joint positions and velocities and the simulator time from one
  instant; `task_truth()` (cube and target poses, cube velocity, jaw contacts, the gripper site)
  is for scorers and scripted references only and never enters an observation.
- Resets take explicit `cube_xy`, `cube_yaw`, `target_xy` and `joint_positions`; anything left
  out is drawn with the seed from a *development default* region (cube x 0.17–0.27 m, |y| ≤
  0.08 m, yaw ±45°; target at least 9 cm from the cube). S2's task modules (TASK-098) declare
  their own reset distributions.

## Embodiment (`so101.py`, registered as `so101`)

`joint_delta_so101_v0`, six normalized components in LeRobot `so_follower` order; physical
meaning from `configs/so101_sim_action.json` (`so101_sim_action_v0`):

| Channel | Mapping |
| --- | --- |
| `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex` | delta of the joint-position target, 1/16 rad per step at ±1 |
| `wrist_roll` | delta of the joint-position target, 1/8 rad per step at ±1 |
| `gripper` | absolute target: −1 → −0.17453 rad (closed), +1 → 1.7453292 rad (fully open); moves at most 0.25 rad per step |

- Control step 0.05 s (10 physics substeps, targets interpolated linearly). Deltas add to the
  current *target* (command history), not the measured pose.
- Limits are the intersection of each MJCF joint range and actuator control range, recorded in
  the manifest and checked against the compiled model. A clip (joint limit or gripper rate) gives
  status `clipped` with the applied normalized action and the reason; the requested action is
  returned unchanged. Out-of-range targets sent straight to the transport are refused.
- Arm scales are powers of two, so the arm channels round-trip exactly in float32; the gripper's
  affine map round-trips to within 2 × 10⁻⁷.
- A zero gripper command is half-open, not a hold; `hold_action()` gives the action that keeps
  the current targets. As with G1, `execute` stops and holds the measured pose on a missing,
  stale or consumed observation, an expired command deadline, or a measured joint speed above
  8 rad/s.
- State: `so101_proprio_v0`, the six joint positions then the six velocities.
- `config.py` registers the embodiment and its schema, and refuses an `so101` config at load
  until S2 registers SO-101 tasks (`EMBODIMENT_TASKS`): the registered tasks and `benchmark.py`
  are G1-only, so an SO-101 config must not reach them.

## Development observations (2026-10-16; not gated, no claim)

Frames rendered at 224 px under `MUJOCO_GL=egl` and inspected by eye:

- Home pose (`[0, −1, 1, 0.8, 0, 0.6]` rad): the wrist view shows the table, the red cube in the
  lower left and the fixed jaw's tip at the bottom edge; the overview shows the whole arm, the
  cube in front of it and the blue target to its left.
- During the scripted reach, at pre-grasp (7 cm above) the cube is in the upper left of the
  wrist view; at grasp height it is large in the lower centre of the frame beside the fixed jaw,
  and it stays in view while lifted (the plan's "cube visible in the wrist view at grasp
  distance"). In the overview the cube is about 15–20 px wide.

`dev_so101_reach.py` (module `so101_scripted.py`): privileged damped least-squares IK on the
cube pose from simulator truth, top-down approach with the jaw closing across a cube face,
driven only through `joint_delta_so101_v0` actions (pre-grasp 7 cm above, descend, close, lift
8 cm; 130 steps). On development seeds 0–7: 7 of 8 lifted the cube about 8 cm with both jaws in
contact; the jaw stalls at about 0.16 rad on the 3 cm cube. Seed 4 (cube at x = 0.264 m, near
the far edge of the development region) failed: the strictly vertical pre-grasp left an 8 mm IK
residual, and on descent the fixed jaw landed on the cube instead of beside it. Each run reported
11 `clipped` steps, all from the gripper rate limit. This shows that the scene is reachable and
the gripper can hold the cube; it is not a success rate. S2 measures the scripted follower's rate
on declared seeds.

## Not done here

Task modules, scoring and reset distributions (S2, TASK-098); a collector and corpus (S3); any
model, planner run or closed loop; real hardware (S6, owner ruling). The models' default
camera key is G1's `onboard_rgb` (`models/base.py`); choosing `wrist_rgb` and/or `overview_rgb`
belongs to the first SO-101 data step.
