# SO-101 cube tasks: declarations and development measurement (S2, TASK-098)

**Development only. No preregistration, no gate, no claim.** Every success counted here is the
**privileged scripted follower's** (it reads the cube pose and its in-hand offset from simulator
truth), in simulation, on development seeds. None is a learned result, and none is an Apple→Plate
result: R7 does not change (R28.9). The numbers say what the scene and the scorers allow; they set
what later bars *can* be (SO101_PLAN.md, lesson 1). They do not set bars.

Plan: [SO101_PLAN.md](../SO101_PLAN.md) S2; rulings: [DECISIONS](../DECISIONS.md) 2026-10-16 (bp),
R28.1–R28.9; scene and embodiment: [SO101_SIMULATION.md](../SO101_SIMULATION.md) (S1, TASK-097).

## What was built

- `src/embodied_jepa/so101_tasks.py`: `so101_cube_lift_v0` and `so101_cube_place_v0`. Each owns
  its declared reset distribution, goal, termination and scorer. Both scorers read simulator
  truth only (`SO101Simulation.task_truth()`), and nothing from them reaches an observation, a
  model input or a planning cost. The pure functions `lift_outcome` and `place_outcome` score
  per-control-step truth arrays. The same code scores a live episode (`evaluate()` once per
  control step) and the synthetic trajectories in `tests/test_so101_tasks.py`.
- Registered lazily in `config.py` (`TASKS`: `"module:attr"` strings; `EMBODIMENT_TASKS` and a
  new `TASK_EMBODIMENTS` route them to the `so101` embodiment only). An SO-101 *experiment
  config* is still refused at load (`EMBODIMENT_CONFIG_REFUSED`), because `benchmark.py` is
  G1-only and no SO-101 entry point exists yet (the S1 review finding of #232 still holds). The
  tasks also refuse a non-SO-101 embodiment.
- `task_truth()` gains `cube_angular_velocity` and `gripper_contact` (either jaw or the camera
  mount touching the cube). Nothing else in `so101_simulation.py` changes.
- `src/embodied_jepa/so101_followers.py`: privileged scripted lift and place followers that
  extend S1's `so101_scripted.py`. They take rigid grasp-aim and place-aim offsets (the K-style
  doses) and provide a kinematic reach map.
- `scripts/measure_so101_tasks.py`, with subcommands `reach`, `measure` and `frames`.
- Every hash-pinned G1 file is byte-identical: the diff against `origin/main` touches only
  `config.py` (additive registrations and checks), `so101_simulation.py` and new files.

## Declarations

These were fixed before the reported measurement and were not changed after any follower
episode was scored. The order of events is listed under *History* below.

### Reset distribution (`ResetDistribution`, `RESETS_V0`)

| Item | Declared |
|---|---|
| Cube xy | uniform, x ∈ [0.18, 0.23] m, y ∈ [−0.10, 0.10] m (world; robot base at the origin, +x straight out) |
| Cube yaw | uniform on [−45°, 45°] (covers every face orientation of a cube) |
| Target centre (place) | uniform, x ∈ [0.17, 0.21] m, y ∈ [−0.12, 0.12] m, redrawn until ≥ 0.10 m from the cube |
| Lift target marker | fixed at (0.30, −0.20) m, away from every lift cube (the scene always has a marker) |
| Streams | `default_rng([9801, seed])` for lift and `[9802, seed]` for place, so the two tasks draw unrelated resets for the same seed |

A centre distance of at least 0.10 m is more than √2 × 0.055 m, so the starting cube lies wholly
outside the 8 cm target square (half-size 4 cm plus the cube's 1.5 cm).

**How the boxes were chosen: from a kinematic reach map, not from the follower's outcomes.** The
map's collision check was itself prompted by two development-seed lift failures (History,
item 1); the final boxes were then read from the map, not from any follower outcome.
`measure_so101_tasks.py reach` solves the follower's own IK on a grid (x 0.12–0.29 m in 1 cm
steps; y in 2 cm steps), with no physics stepping. For the grasp map it uses the cube at 7 yaws
from −45° to 45°, the grasp height and the 6 cm pre-grasp height, with the jaw open. For the
place map it uses the release pose (the jaw closing across the base-to-target ray) and 6 cm
above it. A cell is *ok* when every pose's IK residual is below 1 mm and no pose has a
robot–robot or robot–table contact (for the grasp map, also no jaw more than 1 mm inside the
cube). Each declared box is a rectangle of ok cells:

- The grasp map's ok region is bounded near the base by the **wrist camera mount colliding with
  the upper arm/shoulder**. At x ≤ 0.17 m with |y| ≤ 0.06 m every yaw collides at some pose.
  Toward the far edge it is bounded by **IK residuals** (≥ 1 mm beyond about x = 0.23 m at
  |y| = 0.10 m and x = 0.26 m at y = 0). Every grid point of the cube box is ok in the v1 map
  (`so101-s2-reach-ff09958`).
- The place map is ok over x 0.17–0.21 m for |y| ≤ 0.16 m. The box stops at x = 0.21 m because
  (0.22, +0.12) has a 1.3 mm residual. A draft upper bound of 0.22 m was narrowed to 0.21 m from
  this map before any measurement.

### `so101_cube_lift_v0` (`LiftThresholds`, `LIFT_V0`)

| Threshold | Value | Rationale |
|---|---|---|
| Lift height | cube centre ≥ 0.05 m above its resting height (0.015 m) | Clearly off the table: 1.7 cube sizes. No support other than the gripper exists at that height. |
| Hold | every step of 20 consecutive control steps (1.0 s at 20 Hz) | Long enough that a flick or a toss cannot count. |
| Gripper contact | required on every step of the hold | Not thrown or tossed. `gripper_contact` is true when *any one* of the fixed jaw, the moving jaw or the camera mount touches the cube, so a cube held by a single jaw, or resting on the camera mount, also counts. The scripted follower always grips with both jaws; a later version for learned policies may require both jaws (`fixed_jaw_contact` and `moving_jaw_contact` are already in `task_truth()`). The same contact definition applies to the place task's pick. |
| Lost | cube centre below z = −0.05 m or off the table top | Terminal failure. |
| Episode | at most 200 control steps (10 s) | The follower needs about 120. |

Success is the first step that completes the hold. The episode terminates there, or on a lost
cube, or at 200 steps. A hold that breaks (a drop) restarts the count, and separate holds do not
add up.

### `so101_cube_place_v0` (`PlaceThresholds`, `PLACE_V0`)

| Threshold | Value | Rationale |
|---|---|---|
| Target region | cube centre inside the 8 cm axis-aligned square (half-size 0.04 m) drawn by the blue marker | The visible marker defines the region. With the centre inside, at least half the cube's footprint is over the marker. |
| Pick (precondition) | centre ≥ 0.02 m above rest with a gripper contact on 5 consecutive steps (0.25 s) | A pick-and-place must pick: pushing the cube into the square does not count. |
| At rest | on every step of 20 consecutive steps (1.0 s), all after the pick: centre inside the square; centre height within 3 mm of rest (on the table, flat); linear speed ≤ 5 mm/s; angular speed ≤ 0.1 rad/s; no gripper contact (released) | The TASK-067 lesson: no latched transient crossing can count, and the cube must have been let go. |
| Drift | the centre's xy range over those 20 steps ≤ 2 mm | Catches a slow creep that stays under the speed bar at every sampled step. |
| Lost | as for lift | |
| Episode | at most 320 control steps (16 s) | The follower needs about 220. |

Success is the first step that completes such a window. The episode terminates there, or on a
lost cube, or at 320 steps. The gripper may still hover near the cube at that step; the scorer
asks only that it has let go. The release height is not bounded: after a pick, a cube dropped
from any height that comes to rest inside the square counts. Only drops that end outside the
square, or that never settle, fail.

The unit tests (`tests/test_so101_tasks.py`, synthetic per-step truth) check these cases:

- **Lift.** A 4-step flick above the bar does not count, and a hold one step short of the
  window followed by a drop does not count. Two holds separated by a drop do not add up. A cube
  above the bar without a gripper contact does not count. A lost cube fails.
- **Place.** A cube that slides through the square at 0.2 m/s and stops outside does not count.
  At rest inside for 19 steps and then knocked out does not count. Neither does a cube dropped
  so that it lands outside, a cube pushed in without a pick, a pick too short for the pick
  window, or a cube still held, held 1 cm above the table, sliding at 10 mm/s, spinning at
  0.5 rad/s or creeping 4 mm over the window. Success falls on the step that completes the window, and inputs are
  validated.

## The scripted followers (`so101_followers.py`, `FollowerConfig`)

- **Lift.** The follower uses S1's top-down grasp (damped least-squares IK on a scratch copy of
  the model; the jaw closes across the cube face most perpendicular to the base-to-cube ray).
  It moves to a pre-grasp 6 cm above the cube (40 steps), descends through 6 Cartesian waypoints
  (30 steps), closes (20 steps), lifts 8 cm through 3 waypoints (30 steps) and then holds. All
  commands are `joint_delta_so101_v0` actions through `SO101Embodiment.execute`.
- **Place.** The pick is the same. After the lift the follower reads the cube's offset from the
  fingertip site in the site frame (privileged). It turns the wrist so the jaw closes across the
  base-to-target ray, and aims the site so the cube centre lands over the aim with its bottom
  4 mm above the table. It then transports at lift height (6 waypoints, 50 steps), lowers
  (6 waypoints, 30 steps) and opens to about 60 % (15 steps). It backs the fixed jaw 8 mm off
  the cube face (10 steps), rises 7 cm (7 waypoints, 35 steps) and holds.
- **v1 (reported) versus v0.** v1 leaves **5 mm between the fixed jaw and the cube face** on the
  descent. The moving jaw, which opens toward the site's +z, pushes the cube onto the fixed jaw
  as it closes. v0, which is S1's alignment, put the fixed jaw tangent to the cube face.
- **Doses (K-style, after TASK-094's K follower).**
  - A *grasp-aim offset* is a rigid error added to every grasp-phase IK target (pre-grasp,
    descent, close, lift), as a gripper calibration error would be. It is expressed in the
    zero-offset grasp pose's frame. `+closing` is the site's +z: the gripper moves toward the
    moving jaw's side, so the **fixed jaw moves toward the cube**. `−closing` moves the fixed
    jaw away from the cube. `±along` is the site's ±y (horizontal, along the cube face), and
    `±up` is world ±z.
  - A *place-aim offset* is a world-xy error of the place aim. The place follower corrects for
    a grasp offset through its privileged in-hand reading, so on place a grasp offset tests the
    pick only.
- 11 steps per lift episode (of 122) and 15 per place episode (of 221–222) report the gripper's
  rate clip as status `clipped`. No step was stopped.

## History (in order)

1. **Draft and development, seeds 1000–1015.** The followers were built and debugged on seeds
   1000–1015, first with a draft cube box x ∈ [0.14, 0.24] m. Two lift failures there
   (x ≈ 0.15 m) came from the camera mount hitting the upper arm, which led to the collision
   check in the reach map. One place episode showed the fixed jaw flicking the cube about 3 cm
   on the retreat, which led to the 8 mm back-off before rising. The thresholds above were
   written before any of this and were never changed.
2. **Reach map** (`so101-s2-reach-e733220`, then `-925153f`). The boxes were declared from it,
   and the target x upper bound was narrowed to 0.21 m.
3. **First measurement, follower v0, `925153f`, seeds 0–63**
   (`so101-s2-measure-925153f`). Both baselines were 64/64. Its envelope exposed two follower
   defects:
   - The closing-axis offset was applied along a cube-face normal of arbitrary sign, so its
     cells mixed both jaw sides. They scored 27–35/64 at every size from 0.5 to 3 cm, roughly
     flat.
   - The fixed jaw was tangent to the cube, so any offset toward it was fatal.

   These v0 cells are kept below as recorded.
4. **Follower v1** (5 mm fixed-jaw clearance, offsets in the jaw frame; `2c90672`, plus the
   reach-map alignment in `ff09958`) was developed on seeds 1000–1015 again. It was then
   measured **once on fresh seeds 100–163**, which no follower version had used before, with
   finer doses. This run, `so101-s2-measure-ff09958`, is the reported measurement.

## Results (follower v1, `ff09958`, development seeds 100–163, n = 64 per cell)

Exact (Clopper–Pearson) 95 % intervals. Every cell uses the same 64 resets per task, so cells
are paired by reset. Each cell is one run of a deterministic simulation; the CPU wall time for
all 128 cells (8 192 episodes, 12 worker processes) was 294 s.

### Baselines (no offset)

| Task | Successes | Exact 95 % | Notes |
|---|---|---|---|
| `so101_cube_lift_v0` | **64/64** | 0.944–1.000 | every episode terminated on success at step 121 |
| `so101_cube_place_v0` | **64/64** | 0.944–1.000 | all picked; success at step 220 or 221; final distance of the cube centre from the target centre: median 4.5 mm, max 7.4 mm |

The place follower's landing error (cube centre against its aim) is 4.4–4.6 mm in median (max
8.0 mm) in the baseline and in each of the eight 10 mm place-aim cells. It is a systematic error of this follower's
release (the 4 mm drop and the opening jaw), not of the reach.

### Grasp-aim envelope (successes out of 64)

| Direction | 2.5 mm | 5 mm | 7.5 mm | 10 mm | 15 mm | 20 mm | 30 mm |
|---|---|---|---|---|---|---|---|
| **Lift** `+closing` (fixed jaw toward the cube) | 64 | 64 | 56 | 0 | 0 | 0 | 0 |
| Lift `−closing` (fixed jaw away) | 64 | 64 | 64 | 64 | 64 | 64 | 18 |
| Lift `+along` | 64 | 64 | 64 | 64 | 17 | 0 | 0 |
| Lift `−along` | 64 | 64 | 64 | 64 | 42 | 0 | 0 |
| Lift `+up` | 64 | 64 | 64 | 64 | 0 | 0 | 0 |
| Lift `−up` | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| **Place** `+closing` | 64 | 64 | 55 | 2 | 0 | 0 | 0 |
| Place `−closing` | 64 | 64 | 64 | 64 | 64 | 64 | 18 |
| Place `+along` | 64 | 64 | 64 | 64 | 9 | 0 | 0 |
| Place `−along` | 64 | 64 | 64 | 64 | 34 | 0 | 0 |
| Place `+up` | 64 | 64 | 64 | 64 | 0 | 0 | 0 |
| Place `−up` | 64 | 64 | 64 | 64 | 64 | 64 | 64 |

64/64 has the interval 0.944–1.000; 0/64 has 0.000–0.056. Every failure is a timeout: the cube
was never held at the lift height, or never at rest in the square.

How to read the grasp envelope:

- **The tolerance depends on direction.** Toward the fixed jaw, the lift holds to 5 mm (the
  declared 5 mm clearance) and is gone at 10 mm. Along the face and upward it holds to 10 mm and
  fails by 15–20 mm, where the fingertips pass the cube's edge or top (the fingertip site aims
  at the cube centre, and the cube is 3 cm). Away from the fixed jaw it holds to 20 mm, because
  the moving jaw closes the remaining gap.
- **Downward offsets are not doses.** The table stops the fingertips, so `−up` cannot move the
  grasp below the table top.
- **Place mirrors lift** in the pick phase. The privileged in-hand reading removes the grasp
  offset from the place aim. At 15 mm along the face the cube more often slips or tilts and does
  not end flat and at rest: 9/64 and 34/64 on place, against 17/64 and 42/64 on lift.

### Place-aim envelope (`so101_cube_place_v0`, world-xy aim error; successes out of 64)

| Direction | 10 mm | 20 mm | 30 mm | 35 mm | 40 mm | 45 mm | 50 mm |
|---|---|---|---|---|---|---|---|
| 0° (+x) | 64 | 64 | 64 | 64 | 64 | 1 | 0 |
| 45° | 64 | 64 | 64 | 64 | 64 | 64 | 63 |
| 90° (+y) | 64 | 64 | 64 | 64 | 36 | 6 | 0 |
| 135° | 64 | 64 | 64 | 64 | 64 | 64 | 58 |
| 180° (−x) | 64 | 64 | 63 | 57 | 22 | 4 | 6 |
| 225° | 64 | 64 | 64 | 64 | 64 | 64 | 50 |
| 270° (−y) | 64 | 64 | 64 | 60 | 29 | 0 | 0 |
| 315° | 64 | 64 | 64 | 64 | 64 | 64 | 59 |

How to read the place envelope:

- **It is mostly the target square's geometry.** Along an axis the half-size is 4 cm, so an aim
  error of 3 cm keeps at least 63/64 in every axis direction. The 4–5 mm landing error and its
  direction then decide where the cliff falls between 3.5 and 4.5 cm. A diagonal offset reaches
  the square's edge only at 4√2 ≈ 5.7 cm, hence 50–64/64 at 5 cm.
- **The cliff toward the base is earlier and not monotone.** At 180° the count is 63/64 at 3 cm,
  57/64 at 3.5 cm, 4/64 at 4.5 cm and 6/64 at 5 cm. An aim 4–5 cm toward the base takes the
  release pose to x = 0.12–0.17 m, partly into the region where the place map shows collisions
  or IK residuals near the base, so the landing there need not follow the aim. This was not
  examined further.

### First measurement (follower v0, `925153f`, seeds 0–63), recorded as run

| Cell | Lift | Place |
|---|---|---|
| Baseline | 64/64 | 64/64 (final distance median 4.4 mm, max 7.1 mm) |
| `±closing` (face-normal sign, so both jaw sides mixed), 0.5–3 cm | 27–35/64 | 21–40/64 |
| `±along`, 0.5 / 1 / 1.5 / 2 / 3 cm | 64, 64, 43–64, 0, 0 | 64, 63–64, 50–64, 0–1, 0 |
| `+up`, 0.5 / 1 / 1.5 / 2 / 3 cm | 64, 64, 61, 0, 0 | 64, 64, 13, 0, 0 |
| Place aim 4 cm at 0° / 90° / 180° / 270° | — | 64 / 31 / 21 / 35 |
| Place aim 5 cm at 45° / 135° / 225° / 315° | — | 63 / 57 / 58 / 64 |

v0's baselines and place-aim cells agree with v1's. Its `±closing` cells measure the face-normal
sign more than a jaw-side tolerance, and v1 replaces them.

## Rendering check (both cameras, 224 px, `MUJOCO_GL=egl`, under the GPU lock)

`measure_so101_tasks.py frames` rendered `wrist_rgb` and `overview_rgb` at 224 × 224, the size
S1 renders by default and the size the plan's later steps assume. It rendered seeds 100 and 101
of each task at steps 0, 45, 70 and 90 (lift) and 0, 70, 120, 170 and 200 (place), with v1
(`so101-s2-frames-ff09958`; contact sheets `sheet_lift.png` and `sheet_place.png`). Inspected by
eye:

- **Wrist view.**
  - At the home pose the cube is at most partly in view, at the upper-left edge: partly for
    seed 100, a sliver for lift seed 101, and not at all for place seed 101.
  - At pre-grasp the whole cube is in the upper middle; at grasp height and while lifted it fills
    the lower centre beside the fixed jaw.
  - While lowering for the place, the blue target is visible around the held cube. At the
    release (step 200) the target square fills much of the frame behind the cube.
- **Overview.**
  - The whole arm, the cube (about 15–20 px wide) and the target square are in view.
  - At the release the arm stands between the camera and the target region in these two
    episodes and occludes most of the cube and part of the square.
  - In lift episodes the fixed marker at (0.30, −0.20) m shows at the lower-left edge of the
    overview and in the home-pose wrist view.
- **For the first data step.** These are observations, not measurements. Relying on the
  overview alone at the release would hide the placement, and relying on the wrist view alone
  at the home pose would hide the cube. Choosing the cameras belongs to S3/S4.

## Caveats

- **Privileged scripted references, not controllers.** The followers read simulator truth: the
  cube pose at the reset, and for place the cube's in-hand offset. Their 64/64 numbers are
  ceilings for this scene and these scorers, not achievable rates for a controller that sees
  only images and joint states.
- **Development seeds; one deterministic run per cell.** n = 64 per cell; 64/64 still allows a
  true rate as low as 0.944. Cells are paired, but no paired test was run, because nothing is
  compared against a bar.
- **The follower changed once after a measurement (v0 → v1).** That change was made because v0's
  envelope exposed the two defects above, and v1 was measured on seeds no follower version had
  seen. The task thresholds and reset boxes were not changed after any measurement.
- **The envelope belongs to this follower,** which has a 5 mm fixed-jaw clearance, a fingertip
  site aimed at the cube centre and a 4 mm release height. Another controller has another
  envelope. A rigid offset is one kind of error. Noisy, drifting or perception-coupled errors
  were not measured.
- **The G1 comparison is direction-dependent.** On G1, TASK-094's K follower lost about two
  thirds of its lifts (88/128 to 31/128) under one 1.2 cm palm offset, in one direction. Here a
  1 cm offset keeps 64/64 in four of the six grasp directions (`−closing`, `±along`, `+up`),
  but toward the fixed jaw it gives 0/64. The robots, objects, followers and scorers differ, so
  this is a development observation. It is not a test of the plan's expectation that a jaw
  gripper tolerates more placement error.
- **Simulation details that matter:**
  - The cube is a 20 g, 3 cm box with friction 1 against the table; the jaw stalls on it at
    about 0.16 rad.
  - The gripper rate is clipped on some steps (11 per lift episode and 15 per place episode,
    recorded as `clipped`).
  - Contact is MuJoCo's soft contact on Menagerie's primitive collision geometry.
  - Nothing here was checked against real hardware (S6 needs an owner ruling).

## Reproduction and artifacts

From a worktree at `ff09958` (`scripts/new_worktree.sh`):

```sh
uv run --no-sync python scripts/fetch_so101.py
uv run --no-sync python scripts/measure_so101_tasks.py reach   --output outputs/<new>
uv run --no-sync python scripts/measure_so101_tasks.py measure --output outputs/<new> --workers 12
MUJOCO_GL=egl scripts/gpu_run.sh --wait --board -- \
  uv run --no-sync python scripts/measure_so101_tasks.py frames --output outputs/<new> \
  --frame-seeds 100 101 </dev/null
```

The v0 run used `925153f` with `--seeds 0 … 63` (the default at that revision).

After `ff09958` the review fixes renamed the follower configuration constant `FOLLOWER_V0` to
`FOLLOWER_V1` (it always held `version="so101_follower_v1"`), in `so101_followers.py` and
`measure_so101_tasks.py`. That is a rename only, with no behavioural change, but those two
files' hashes at the PR head therefore differ from the ones `summary.json` records; reproduce
the recorded hashes from `ff09958`.

Outputs (git-ignored) are under `/mnt/data/emai/projects/open-embodied-jepa/outputs/`. Each
`summary.json` records the revision, the dirty flag (false for all runs), the file hashes, the
thresholds, the resets of every seed, the follower configuration and the platform: Linux PC,
Python 3.12.3, MuJoCo 3.13.0, NumPy 2.5.3, CPU only except the frames.

| Run | File | SHA-256 |
|---|---|---|
| v1 measurement (`ff09958`, seeds 100–163) | `so101-s2-measure-ff09958/summary.json` | `c96cfbee8eba6b36723192735a10255005c9dccff206ae214a9215a7d4236315` |
| | `so101-s2-measure-ff09958/episodes.jsonl` | `ecda4d24d82a3421e5ae7c06d8afdafc1300ca876706d63dbafc540cfb60717e` |
| v0 measurement (`925153f`, seeds 0–63) | `so101-s2-measure-925153f/summary.json` | `e9e8c34211263f2d7b3978e888c20141aa3758f6146d7b4ae5fd1ae47d47e074` |
| | `so101-s2-measure-925153f/episodes.jsonl` | `67b27443177a9e5fcc0c8d2b9fc13151932dd0ea44fe8b72e8ec043d6f920359` |
| Reach map, v1 grasp alignment (`ff09958`) | `so101-s2-reach-ff09958/reach.json` | `ac20df3b79c601f99ed782fb68bc0bf7eba7679e6c0ff6109b59ccd75632d7bd` |
| Reach map, v0 grasp alignment (`925153f`) | `so101-s2-reach-925153f/reach.json` | `5150f5910cba81fa6b6bb27276f69834c1a89396c6cb9f2a912a76991b2b91d3` |
| Frames (`ff09958`) | `so101-s2-frames-ff09958/` (PNGs and `frames.json`) | — |

`so101-s2-reach-e733220` is the same map at the first commit, before the target box was
narrowed; it is superseded but kept.
