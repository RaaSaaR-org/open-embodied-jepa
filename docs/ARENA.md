# Isaac Lab-Arena: the GR00T tutorial's G1 pick-and-place scene (development spike, 2026-10-01)

A development bring-up spike on the Linux PC (RTX 5080), under TASK-025. It is not a gated run, not a preregistered comparison, and not a manipulation, policy or task result. The only "policies" here are a joint-hold command, Arena's built-in zero action and scripted teleports of the apple. No GR00T policy server was started, contacted or touched: the owner's server (pid 14247, `/home/huhn/Isaac-GR00T`) kept running throughout, and no request was sent to port 5555. **Learned Apple→Plate is still 0 successes**, and nothing here changes the v1 benchmark, the v2 task or any gate.

The aim is a third simulator setup next to MuJoCo v2 ([SIMULATION.md](SIMULATION.md), `src/embodied_jepa/apple_to_plate_v2.py`) and our Isaac v2 scene ([ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md), [ISAAC_E9_REPLAY.md](ISAAC_E9_REPLAY.md)). The scene is the one NVIDIA's GR00T end-to-end tutorial evaluates in: [Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-evaluation.html).

## Summary

- **The scene runs headless in the local Arena container, configured exactly as in the tutorial.** That means environment `galileo_g1_static_pick_and_place`, object `apple_01_objaverse_robolab`, destination `clay_plates_hot3d_robolab`, embodiment `g1_wbc_agile_joint`, cameras on. It loads, steps, terminates and reports Arena's metrics, both through Arena's own unchanged `policy_runner.py` (`zero_action` policy, `--num_steps 600`) and through our probe. A scripted drop of the apple onto the plate (teleported over its centre, bottom 1 cm above the plate's top) fires the success term, and holding still times out at step 299 (the 300th step).
- **The local image is exactly the tutorial's Arena.** It was built from IsaacLab-Arena `8b4a3a47fc`, which is the head of `release/0.2.1`, the branch the tutorial's setup page clones (GitHub compare: identical).
- **The plate is not white.** The plate is HOT3D `clay_plates` (`obj_000003`), an OmniPBR material with a diffuse texture and a white tint. The head camera renders it **cream/beige**, with concentric turned rings: the centre patch reads RGB (206, 190, 153) after 100 hold steps (`arena-probe-3`). Ours is solid blue (0.15, 0.35, 0.85).
- **There is one camera.** `robot_head_cam` is a `TiledCamera` on `head_link`, 640 × 480 RGB, pinhole with focal length 15 mm and horizontal aperture 20.955 mm (HFOV 69.9°, VFOV 55.3°). It sits 0.353 m above the pelvis origin and is pitched 35° down. Ours is 112 × 112 with a 75° vertical FOV, on `torso_link`, about 0.39 m above the pelvis and pitched 60° down.
- **The action space is 50-D: absolute joint targets, not end-effector deltas.**
  - `[0:43]` are joint-position targets in the articulation's breadth-first joint order. The WBC uses only the 28 upper-body ones: 14 arm and wrist joints and 14 Dex3 joints.
  - `[43:46]` is `navigate_cmd` (vx, vy, wz). `[46]` is `base_height_command` (m). `[47:50]` is the torso RPY command, which the AGILE backend ignores.
  - The legs and waist are driven by the AGILE recurrent ONNX lower-body policy.
  - The control rate is 50 Hz (sim dt 0.005 s, decimation 4).
- **The base is floating, balanced by the WBC.** It is not fixed like ours. Under a hold command (base height 0.75) the pelvis sinks 4.5 cm and drifts +1.2 cm forward and −0.6 cm sideways in the first 50 steps, then stays still.
- **Success is one per-step test.** The apple–plate contact force must exceed 0.5 N while the apple's speed is below 0.1 m/s. There is no position bound and no settle window, and (inferred from the code, not tested) no release requirement. It is much looser than our `apple_at_rest_v0`.
- **Cross-simulator checks (§7, 2026-10-01, development).** The arm kinematics of NVIDIA's USD agree with our MJCF: the wrist is within 0.8 mm and 0.0002° of our FK at Arena's measured joints. The Dex3 fingers do not: their joint origins sit up to 2.3 cm closer to the wrist, and the fingertips differ by up to 3.1 cm. e9 does not transfer to Arena with our layout. It left the apple at rest on 0 of 16 development seeds by both verdicts, against 16/16 in MuJoCo on the same seeds. **Blocker:** e9 never grasps. Two causes were found:
  - The arms stall without a gravity offset, because Arena's ideal-PD drive has no bias compensation.
  - With the offset, the arm follows e9's targets to within millimetres of MuJoCo. Then, in the close phase, the floating base steps back 4–12 cm, and e9's pelvis-frame targets end up behind the apple.
- **Plan.** The cheapest meaningful cross-simulator check is an e9 replay in Arena through a mirror adapter (§5). It has to deal with three differences: the floating pelvis, 50 Hz against our 20 Hz, and the apple sitting on the robot's left (e9 is right-handed).

## 1. Environment and provenance

| Item | Value |
| --- | --- |
| Image | local `isaaclab_arena:latest`, `sha256:2588b52605d77552d4480196501c4b8b61774ef6d70d9b3d59291622d6856d9c` (39.5 GB, built 2026-09-23; not pulled or rebuilt for this spike) |
| Arena source | `8b4a3a47fc53de23e8205089d71109a2e2348acd` = head of `release/0.2.1` (the tutorial's `git clone --branch release/0.2.1`) |
| Isaac Sim / Lab | 6.0.0-rc.22 (a release candidate, [ISAAC_BRINGUP_SPIKE.md](ISAAC_BRINGUP_SPIKE.md)); `isaaclab` 4.5.24 and `isaaclab_arena` 1.0.0 per package metadata; torch 2.10.0+cu128; warp 1.12.0 |
| Physics | PhysX GPU pipeline (Arena's default for this env; no `--presets`), `cuda:0`, one environment |
| Network assets fetched at run time (unpinned) | robot `…/Isaac/6.0/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd`, background `…/IsaacLab/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd`, apple `…/object_library/srl_robolab_assets/objects/objaverse/apple_01.usd`, plate `…/objects/hot3d/clay_plates.usd` (all on `omniverse-content-staging.s3-us-west-2.amazonaws.com`); AGILE ONNX `unitree_g1_velocity_height_recurrent_student.onnx` from `github.com/nvidia-isaac/WBC-AGILE` at commit `7259792c` |
| GR00T | not used. The tutorial pins Isaac-GR00T `4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e` for its server; nothing here imports or contacts it. |

### Runs (`outputs/`, git-ignored, in the spike worktree)

| Run | Code | What | Result |
| --- | --- | --- | --- |
| `arena-probe-1` | `3434538` + uncommitted first draft | probe (facts, frames, hold, drop) | built in 420 s. Its success-path teleport copied Arena's test offset (plate origin + 2 cm), which puts the apple inside the plate's collision geometry, and success fired on the first step. That check is **void**; the facts and frames are the same as in run 2. |
| `arena-probe-2` | `89854c2`, clean | probe; the drop teleports the apple over the plate centre with its bottom 1 cm above the plate's AABB top (rim height) | **reference run for the facts and committed frames**: built in 327 s, hold episode `time_out` at step 299, drop episode `success` at step 4, metrics `success_rate 0.5, object_moved_rate 0.5, num_episodes 2`. The rows it logged on a terminating step are read after Isaac Lab's automatic reset, so they show the next episode's state; force and speed at the success step were **not observed** in this run. |
| `arena-probe-review-1` | `3777d60`, clean (the reviewer's run) | the same probe | same hold and drop traces as run 2 (bit-identical); the step-0 head-camera frame was **entirely black**, so its step-0 colour samples were (0, 0, 0) |
| `arena-probe-3` | `d64abfe`, clean | probe after the review fixes: records the success term's own inputs before the auto-reset, samples colour at hold step 100, flags blank frames | built in 320 s; hold `time_out` at step 299, drop `success` at step 4, metrics as run 2; no blank frame; success-step inputs: force 1.07 N, apple speed 0.038 m/s, apple at (0.578, 0.060, −0.008), on the plate |
| `arena-runner-zero-1` | `89854c2`, clean | Arena's `policy_runner.py`, `zero_action`, 600 steps | completed 600/600, but its prints were lost: stdout is a block-buffered pipe and Kit exits without flushing. Fixed in the wrapper. |
| `arena-runner-zero-2` | `16f4ef9` (untracked `docs/arena/` only) | the same, line-buffered | 600/600 steps; episode 1 `time_out` at step 300 (runner-reported progress); episode 2 ended between steps 306 and 339 (6–39 steps in; the runner's second Metrics print lies between those progress ticks) by a non-success, non-timeout term, which can only be `object_dropped` (not printed by the runner, so inferred); metrics `success_rate 0.0, object_moved_rate 1.0, num_episodes 2` |

`zero_action` is a real Arena policy, but it is not a hold for this embodiment. It commands base height 0 and every upper-body joint to 0. The robot crouches, the arms swing down, and the apple moved in both episodes. The probe's hold is `wbc_action(default joint positions, base_height=0.75)`. That is Arena's own standing-idle pattern (`actions[:, -4] = 0.75` in its tests), but with the env's open-arm default pose instead of zeros.

**Repeatability.** The hold and drop traces (pelvis, apple position and velocity) were identical in every probe run that logged them: runs 1–3 and the reviewer's run, four processes. Rendering was not repeatable:
- between runs 1 and 2, frames differ by a mean absolute difference of 1.0/255 at step 0 and 0.55/255 at step 200 (1.08/255 at step 200 against the reviewer's run);
- in 1 of 4 runs (the reviewer's) the **first** head-camera frame after the scene was built was entirely black, although `num_rerenders_on_reset = 1`. Every later frame rendered. Any evaluation must expect that the first observation can be blank; the probe now flags blank frames and samples colour at step 100.

**Timing and memory** (`arena-probe-2` unless noted):
- **Start-up.** AppLauncher takes 106–151 s, and the scene is built 327–420 s after start, most of it fetching assets into a fresh per-run cache.
- **Step time.** The env step with the 640 × 480 camera has a median of 48 ms and a p95 of 107 ms (about 0.4× real time). The runner reported 15–20 steps/s.
- **GPU memory** (`nvidia-smi` samples about 1 s apart, so peaks are lower bounds). The Isaac process peaked at 6885–7181 MiB (6.7–7.0 GiB). The device peaked at 13767–14189 MiB (13.4–13.9 GiB) of 16303 MiB (15.9 GiB), with GR00T's 6626 MiB (6.5 GiB) resident. That leaves about 2 GiB of headroom: do not run two Arena processes next to GR00T.

## 2. Commands

All heavy runs were wrapped in the shared lock, after a check that load ≤ 2.0 and at least 7 GiB of GPU memory was free. `run_isaac.sh` records the image id, code revision and status and samples GPU memory; it refuses to overwrite an output. `PXR_WORK_THREAD_LIMIT=1` is pinned inside the process (PR #112's workaround, now `arena_transport.pin_pxr_work_thread_limit`). The log shows Kit's write of 16 being replaced, and `pxr.Work.GetConcurrencyLimit()` read 1. No start-up hang occurred in the 5 runs made for this spike (6 with the reviewer's) (that hang was a Newton issue; this env runs PhysX). With longer start-ups, `serve_with_watchdog.sh`'s pattern of a per-run limit plus stopping only the run's own container would need a limit above about 600 s.

```sh
# from the repo root; plain `docker` needs `sg docker -c` in sessions that predate the group change
export MJCF_DIR=$PWD/third_party/unitree_mujoco/unitree_robots/g1 PXR_WORK_THREAD_LIMIT=1
flock <lock> sg docker -c "scripts/isaac/run_isaac.sh arena_probe.py outputs/<new>"
# Arena's own runner, the tutorial's command with zero_action instead of
# Gr00tRemoteClosedloopPolicy and --headless instead of --viz kit:
flock <lock> sg docker -c "scripts/isaac/run_isaac.sh arena_policy_runner.py outputs/<new> \
  --headless --policy_type zero_action --num_steps 600 --enable_cameras \
  galileo_g1_static_pick_and_place --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab --embodiment g1_wbc_agile_joint"
```

For comparison, the tutorial's own command, which was **not run** here:

```sh
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --viz kit \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
  --remote_host localhost --remote_port 5555 --num_steps 600 --enable_cameras \
  galileo_g1_static_pick_and_place --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab --embodiment g1_wbc_agile_joint
```

`arena_policy_runner.py` refuses any argument that mentions `remote` or `gr00t`.

Code added (opt-in; nothing in the package imports it):
- `src/embodied_jepa/arena_transport.py`:
  - NumPy only at import: the tutorial constants, `tutorial_argv`, `wbc_action` (the 50-D layout, mapped by joint name), ROS-convention projection, and the PXR pin;
  - `ArenaScene`, which imports Isaac lazily and builds the env exactly as `policy_runner.py` does, with `reset`, `step`, `frames`, `teleport_apple` (harness only), an evaluator-only `truth`, and `metrics`;
  - tests in `tests/test_arena_transport.py`, including the core-import check.
- `scripts/isaac/arena_probe.py`: writes `facts.json`, `episodes.json` and `frames/`.
- `scripts/isaac/arena_policy_runner.py`: runs Arena's runner unchanged.

## 3. Scene facts, side by side

Arena values were read back at run time (`arena-probe-2/run/facts.json`) unless marked *source*, which means read from the Arena source at `8b4a3a47fc`. Arena positions are env-local metres; after the WBC settles, the pelvis origin is at (0.262, 0.074, −0.045). "Rel. pelvis" is relative to that settled pelvis.

| | MuJoCo v2 (`apple_to_plate_v2`) | Our Isaac v2 ([ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md)) | Arena tutorial scene |
| --- | --- | --- | --- |
| **Robot model** | pinned Unitree `g1_29dof_with_hand.xml` (43 actuated joints) | that MJCF converted to USD (`g1_29dof_with_hand-ref`, hash-pinned) | NVIDIA `g1_29dof_with_hand_rev_1_0.usd` (43 joints; same names; Dex3 limits are a strict subset of the MJCF's on 10 of 14 joints, as in [ISAAC_BRINGUP_SPIKE.md](ISAAC_BRINGUP_SPIKE.md); here e.g. `left_hand_index_0` [−1.571, 0]) |
| **Base** | pelvis fixed at (0, 0, 0.793), free joint removed | fixed root at the same pose | **floating** (`is_fixed_base = False`); spawned at (0.25, 0.08, 0), standing on AGILE WBC; with base height 0.75 under a hold it settles to z −0.045 (4.5 cm lower), +1.2 cm in x, −0.6 cm in y within about 50 steps, then moves < 1 mm |
| **Legs, waist** | held by PD | held | AGILE recurrent student ONNX (WBC-AGILE `7259792c`) outputs the 12 leg joints. The waist is in the WBC's lower-body group, so the action's waist targets are not used. The ONNX policy does not control the waist either (*source*: `g1_agile.yaml`); it is held by PD with stiffness 300. |
| **Arm drive** | clipped PD + bias compensation (`configs/g1_sim_action.json`) | joint targets through Isaac PD (parity in [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md) §2) | Isaac Lab `IdealPDActuator`: stiffness 20/40/100, damping 2/5, effort limit 5/25 N m |
| **Hand** | Dex3-1, 7 joints per hand, open/closed grasp synergy | the same, converted | Dex3-1, same 14 joint names; `IdealPDActuator` stiffness **4**, damping 0.5, effort 5. A prestartup event binds a high-friction material (static 6.0, dynamic 5.0) to finger prims. |
| **Action / control** | `ee_delta_grasp_v0` (14-D: per-arm Δpose + 2 grasp scalars) → IK → joint targets at **20 Hz** (dt 0.002, 25 substeps) | joint targets from our IK at 20 Hz | **50-D** joint targets + nav/height/RPY at **50 Hz** (dt 0.005, decimation 4, render interval 2) |
| **Camera** | `onboard_rgb` on `torso_link` at (0.08, 0, 0.35): about 0.39 m above and 0.08 m ahead of the pelvis; **60° down**; 112 × 112, fovy 75°; fixed-function GL | same pose and intrinsics, pinned 64-spp path tracer; geometry parity passes, photometry fails | `robot_head_cam` on `head_link` (whose origin coincides with the pelvis) at offset (0.04485, 0, 0.35325), ROS quaternion (−0.62721, 0.62721, −0.32651, 0.32651); **35° down**; **640 × 480**, f 15 mm, aperture 20.955 × 15.716 mm → HFOV 69.9°, VFOV 55.3°, K = [[458.1, 0, 320], [0, 458.1, 240]]; clip 0.1–5 m; RTX real-time (DLSS warned that a 320 × 240 render was below its minimum; upscaling of this camera is inferred, not checked); RGB only |
| **Other cameras** | none | none | none. The env's `rgb_array` viewer renders black when headless. |
| **Table** | brown box 0.64 × 0.90 m, top z 0.74 (**5.3 cm below** the pelvis), friction 1 | same | dark, glossy `galileo_locomanip` warehouse shelf. Objects rest on an invisible support cuboid 0.8 × 1.5 × 0.04 m with its top at z −0.030 (*source*: the shelf mesh's collision is perforated), which is **1.5 cm above** the settled pelvis. Three background boxes are deactivated. |
| **Apple** | sphere r 0.027 m, **0.080 kg**, friction (1, 0.01, 0.001), **condim 6** (rolling friction), red | sphere, same mass and inertia; PhysX: no rolling friction (Newton: as MuJoCo) | Objaverse `apple_01` mesh × 0.009: AABB **6.1 × 5.9 × 6.1 cm**, **0.097 kg** (USD density 980 kg/m³), inertia diag (3.17, 3.25, 3.41) × 10⁻⁵, **convex decomposition (253 shapes)**, friction 0.8/0.8, restitution 0; textured red/yellow (`apple_01.jpg`) |
| **Plate** | static body, base cylinder r 0.07 m (top 1.2 cm above the table) + 16 rim capsules (top 1.9 cm); **blue** (0.15, 0.35, 0.85) | kinematic; same geometry (base as a convex hull) | HOT3D `clay_plates` × 0.5: AABB **15.0 × 15.1 × 2.4 cm**; a **dynamic** rigid body of **0.5 kg** (not kinematic); convex decomposition (256 shapes); friction 2.0/2.0, restitution 0.1; **cream/beige**, textured (`obj_000003.png`, tint white), rendered centre patch (5 × 5 px) RGB (206, 190, 153) at hold step 100 in `arena-probe-3`, with 8 ring samples at a quarter of the plate's width spanning 204–215, 187–200, 149–162 (at step 0 in run 2 the centre patch was (205, 189, 150)) |
| **Layout, rel. pelvis** | apple (0.34, −0.18) ± 0.03, robot's **right**; plate (0.49, −0.09) ± 0.02 | same (`isaac_scene.reset_layout`) | apple (0.316, **+0.195**), robot's **left**; plate (0.316, −0.014), straight ahead |
| **Reset randomisation** | wide jitter ±3 cm apple, ±2 cm plate, seeded | same | **none**. `APPLE_SPAWN_XY_RANGE_M = 0.0` (the reset event's range is a single point: x 0.5785, y 0.27, z −0.0079), the plate pose is fixed, the robot joints return to their defaults and the WBC state is reset. Every episode starts the same. |
| **Episode** | e9: look + up to 740 commands + a 60-step settle (about 40 s) | same | **6.0 s = 300 steps**; `--num_steps 600` = 2 episodes |
| **Success** | `apple_at_rest_v0`: apple centre within 4 cm of the plate centre, supported, speed ≤ 0.001 m/s for the final 20 steps of a 60-step settle, no hand contact | same, from `task_truth()` | `object_on_destination`, every step: a contact sensor on the apple, filtered to the plate, reads force **> 0.5 N** and apple speed is **< 0.1 m/s** → terminate as success. Also `object_dropped` (apple z < −0.2) and `time_out`. Metrics: `success_rate` (episodes ended by `success`) and `object_moved_rate`. |

### What the Arena success test does and does not require

Read from `isaaclab_arena/tasks/terminations.py` and checked with one drop.

- **It does not bound position.** Any apple–plate contact above 0.5 N counts: on the rim, tilted, or half off the plate.
- **It does not require release** (inferred from the code; not tested). An apple held in the hand and pressed slowly onto the plate would satisfy it.
- **It has no settle window.** The apple was teleported over the plate centre with its bottom 1 cm above the plate's AABB top (rim height), and fell 3.3 cm in all. It was at 0.78 m/s one step before contact, and `success` fired on the first step in contact (step 4 after the teleport). In `arena-probe-2` the force and speed at that step were not observed: the row was read after the automatic reset (its 1.07 N is a post-reset sensor value). `arena-probe-3` records the term's own inputs before the reset: force 1.07 N and apple speed 0.038 m/s at the success step, so the apple had slowed from 0.78 to 0.04 m/s within one 20 ms step of touching the plate.
- **The contrast with ours.** Our `apple_at_rest_v0` would need a whole second of near-zero speed within 4 cm of the centre. Any cross-simulator report must give both verdicts.

## 4. Sample frames

All frames are from `robot_head_cam` in `arena-probe-2`. The full 640 × 480 PNGs are in `outputs/arena-probe-2/run/frames/` (git-ignored, 186–210 KB each): `robot_head_cam_hold_step000.png` (sha256 `5a451209cbb855c1…`), `_hold_step100`, `_hold_step200` (`469ec5664bdd03b1…`), `_hold_after_end_step299` and `_drop_step000` (`1dec5624a213ba29…`). Small copies are committed under `docs/arena/`, each under 65 KB:

| File | Shows |
| --- | --- |
| [`head_cam_reset_320x240.png`](arena/head_cam_reset_320x240.png) | the first step after reset. The apple sits left, on the dark glossy shelf, and the cream ringed plate is straight ahead. Behind them are white pillars and magenta shelving: the background's MDL materials failed to resolve ("EntityResolver FAILED" in the log), so this is a rendering defect, not the authored colour. The hands are out of view. |
| [`head_cam_hold_step200_320x240.png`](arena/head_cam_hold_step200_320x240.png) | after 200 hold steps. The view has moved with the settled pelvis, and the Dex3 fingertips just enter the left and right edges. |
| [`head_cam_drop_on_plate_320x240.png`](arena/head_cam_drop_on_plate_320x240.png) | the first step after the scripted teleport above the plate |
| [`head_cam_reset_centrecrop_112.png`](arena/head_cam_reset_centrecrop_112.png) | the reset frame, centre-cropped to 480 × 480 and resized to 112: roughly what a 112 px pipeline would see from this camera |
| [`mujoco_v2_onboard_reset_112.png`](arena/mujoco_v2_onboard_reset_112.png) | for contrast, our MuJoCo v2 `onboard_rgb` at reset (development seed 50200, wide-jitter layout): the steep 60° view, brown table, blue plate and both Dex3 hands in frame |

## 5. What it takes to evaluate our stack here

### Action mapping

- **What Arena takes.** `g1_wbc_agile_joint` takes absolute joint targets. Our stack already produces those: `ee_delta_grasp_v0` goes through `G1Embodiment`'s IK on the MuJoCo model, and the arm and Dex3 joint targets come out (the synergy maps the grasp scalars to the 14 Dex3 joints).
- **The adapter.** It is the e9 replay's mirror pattern ([ISAAC_E9_REPLAY.md](ISAAC_E9_REPLAY.md) §1).
  - Run our IK on a MuJoCo mirror whose joint state comes from Arena.
  - Take the 28 upper-body targets and map them by name into the 50-D action (`arena_transport.wbc_action`).
  - Fill the leg slots with anything (they are ignored), set `base_height` to 0.75 and set `navigate` to 0.
- **The rate.** Arena steps at 50 Hz, and our commands are at 20 Hz: 0.05 s is 2.5 Arena steps.
  - Option 1: hold each command for 2 or 3 steps alternately (exact over every 2 commands).
  - Option 2: interpolate the targets.
  - Option 3: run a variant env with decimation 10. That leaves the tutorial's configuration, and it changes the WBC's 50 Hz loop.
  - Options 1 and 2 keep the tutorial env unchanged.
- **The drive differs.** Arena's arms and hands are ideal PD with low gains: Dex3 stiffness 4 against our clipped PD. The same targets will track differently, as in the bring-up spike, and grip force in particular will differ.

### Fixed against floating base

- **What changes.** Our IK, e9 and P-3's palm-pose feature all assume a pelvis fixed at (0, 0, 0.793). In Arena the pelvis settles 4.5 cm lower and stands under a learned balance controller, which will also react to arm loads (the reaction is untested here).
- **The adapter.** It must write Arena's pelvis pose into the mirror's base each step, so that FK, IK and the palm pose are pelvis-relative. Arena already exposes the wrist poses in the pelvis frame (`left/right_wrist_pose_pelvis_frame`), which allows a cheap check that the two kinematic models agree.
- **The fallback.** Pinning the root (`fix_root_link`, as in the bring-up spike) would remove the WBC, so it is no longer the tutorial's embodiment.

### Camera matching

- **Pose and intrinsics.** The two cameras differ in mount, height, pitch (35° against 60°), FOV and resolution. `G1WBCAgileJointEmbodiment` accepts a `camera_offset`. A variant env could put a second camera at our `onboard_rgb` pose, then crop and resize to 112. That is not the tutorial's observation.
- **Appearance.** The appearance gap is larger than in our Isaac v2 scene: a photoreal warehouse, a dark glossy shelf, a textured apple, a cream plate, RTX real-time rendering (with DLSS warnings), and magenta material failures in the background.
- **Consequence.** Any model trained on MuJoCo pixels is far out of distribution here.

### What could run here

| Component | Can it run? | What is missing |
| --- | --- | --- |
| **e9** (privileged scripted expert) | yes, with the adapter | Truth is available from Arena (`ArenaScene.truth`: apple and plate pose and velocity, contact force). e9 is right-handed and the tutorial puts the apple 0.195 m to the robot's left, so it needs either a left-handed (y-mirrored) e9 or a variant layout. e9's height constants assume our table 5.3 cm below the pelvis; Arena's is 1.5 cm above. Its scorer must be computed from Arena truth. |
| **P-3** (BC/DAgger MLP on a frozen DINOv2 readout, TASK-072) | not meaningfully | Its readout was fit on MuJoCo 112 px post-look frames, and its inputs include 86-D proprioception and FK palm poses under a fixed pelvis. With this camera and appearance its estimates would be out of distribution. A first step is offline only: measure the readout's apple and plate estimate error on Arena frames from scripted resets, with no control. Note that Arena has no reset jitter, so it has one layout. |
| **LeWM planner** | nothing to run | No learned LeWM controller exists. Arena could later supply head-camera data, for example from e9 rollouts, for world-model training. |

### Staged plan (development, cheapest first)

0. **Done (this spike).** The scene comes up in the tutorial configuration with a local policy, the facts above are recorded, and the success term fires on a drop.
1. **Done (§7.1). Kinematic agreement, no contact.** Drive Arena with scripted upper-body joint targets: the TASK-025 `free_space_v1` arm trajectory, mapped by name. After each step, compare Arena's pelvis-frame wrist poses with FK on our MuJoCo model at Arena's measured joints. This checks that the NVIDIA USD and our MJCF agree kinematically, that name mapping works, and that tracking holds under the ideal-PD gains on a floating base. Cost: one container run (about 7 min). Requires only an `ArenaScene` scripted-target loop.
2. **Run; blocked (§7.2). e9 in Arena, our layout.** Run a variant env (Arena `--external_environment_class_path`, our own class) that keeps the robot, WBC, assets, shelf and success term, but places the apple and plate at our v2 positions relative to the settled pelvis, with the apple on the right. Run e9 closed loop through the mirror adapter (step 1, plus the 20→50 Hz hold and the pelvis pose), on development seeds only (TASK-070 dev seeds 50200–50215, as in the e9 replay). Report Arena's success term and our `apple_at_rest_v0` side by side. Labelled as a cross-simulator scripted-expert check, never a learned result. This isolates the robot asset, floating base, WBC, drive and object differences from the layout. Because the tutorial resets are deterministic, the variant would apply our seeded jitter itself.
3. **e9 on the tutorial layout.** The same, but with a y-mirrored, left-handed e9 and the table-height constants re-derived for Arena's shelf, both declared before running. This is the like-for-like "e9 replay success in Arena" on the tutorial's own scene.
4. **Perception transfer, offline.** Collect head-camera frames, and optionally a second camera at our `onboard_rgb` pose, from step 2 or 3 resets. Measure the frozen DINOv2 readout's estimate error on them. Only if that error is within the place tolerance is a P-3 closed loop worth planning.
5. **Later.** Arena as a data source for LeWM training, and a GR00T-versus-ours comparison on the same success definitions. The latter requires the owner's GR00T server and therefore the owner's go-ahead.

## 6. Caveats and blockers

**No hard blocker.** Caveats:

- **Unpinned network assets.** The robot, background, apple and plate USDs come from NVIDIA's staging S3 and the AGILE ONNX from GitHub, all fetched on every run into a per-run cache of 0.3–0.4 GB. Mirror them and record hashes before any recorded experiment. A shared persistent cache would also cut the 5.5–7 min start-up.
- **Simulator version.** Isaac Sim 6.0.0-rc.22 is a release candidate. The tutorial asks for 6.0.0.
- **Rendering.** Background materials fail to resolve (magenta), and DLSS is active at this resolution. Frames are not repeatable across processes although physics was: later frames differ by a mean absolute difference of about 1/255, and the first frame after the build was entirely black in 1 of 4 runs. The headless viewer is black. The camera pose read from the `TiledCamera` sensor (`pos_w`) was the same at step 0 and step 100 although the pelvis had moved, so it may not refresh each step (not investigated); the colour projection still landed on the plate.
- **GPU headroom.** About 2 GiB remains next to GR00T. One Arena process at a time.
- **Small sample.** Each episode type was run once per run (one hold, one drop, two zero-action episodes). Nothing here is a rate.
- **Tutorial's own policy.** The tutorial reports an expected success rate of 1.0 for its GR00T N1.7 policy over its 600-step command. That was not reproduced or tested here.

## 7. Cross-simulator checks: kinematics, and e9 on our layout (development, 2026-10-01)

These are development runs under TASK-025 on the Linux PC. None of them is a gated run or a preregistered comparison, and no bar was declared before them.
- e9 is TASK-070's privileged scripted expert, which reads simulator truth by design. A scripted expert's outcome in any simulator is not a learned result. **Learned Apple→Plate is still 0 successes.**
- Only TASK-070 development seeds 50200–50215 (plate error 0) were simulated, through `isaac_e9.check_seeds`.
- The v1 task, the v2 task and the TASK-070 gate are unchanged.
- No GR00T server was started, contacted or touched.

### Code (opt-in; nothing in the package imports it)

- `src/embodied_jepa/arena_e9.py`. NumPy only at import. It holds the adapter:
  - `FrameMap`: Arena's env-local frame → our world, relative to Arena's **live** pelvis.
  - `layout_in_arena`: our v2 reset xy, placed at the same offsets from Arena's settled pelvis, rotated by its yaw only.
  - `command_action`: e9's 43 targets mapped by name to the 50-D `g1_wbc_agile_joint` action, with navigate 0 and base height 0.75.
  - `arena_steps`: 20 → 50 Hz. Each command is held for 2 and 3 Arena steps alternately, which is exact over every pair of commands.
  - `mirror_state`, `ArenaMirrorSimulation` and `ArenaEndpoint`: the e9-replay mirror pattern of PR #110 (`isaac_e9.MirrorSimulation`).
  - `gravity_offset`: opt-in, see below.
  - `arena_verdicts`: both verdicts on Arena's world state.
  - The kinematic trajectory and FK helpers.
  - `MuJoCoArenaSham`: the server protocol over a displaced, joint-reordered MuJoCo scene.
- `scripts/isaac/arena_layout_env.py` (container only). The env variant `oej_g1_static_pick_and_place`, loaded through Arena's own `--external_environment_class_path`. It keeps the tutorial's robot USD, AGILE WBC, background, shelf support, apple, plate, finger friction and success term. Its opt-in additions are:
  - an apple–hand contact sensor (wrist-yaw, palm and 7 finger links per hand);
  - the robot's initial position;
  - an invisible static platform under the robot, for matching the table height;
  - object spawn positions;
  - the episode length.
- `scripts/isaac/arena_e9_server.py` (container only). It has two modes:
  - `--mode kinematics`;
  - `--mode serve`, a Unix-socket server.
  
  In both modes cameras are off, and every termination term is evaluated and recorded but held at False, so an e9 attempt of about 40 s never auto-resets.
- `scripts/isaac/arena_e9.py` (host). Subcommands `kinematics`, `arena` and `rescore`. The MuJoCo reference is `scripts/isaac/e9_replay.py mujoco`, unchanged.
- `src/embodied_jepa/arena_transport.py`: `ArenaScene` accepts a variant environment, `enable_cameras`, `hold_terminations`, `set_object_pose` and `raw_state`.
- `scripts/isaac/arena_policy_runner.py` now refuses any argument that starts with `--pol` unless it is exactly `--policy_type` or `--policy_config_yaml_path` (bare or `=value`). This closes the argparse-abbreviation bypass of the allowlist (`--policy_t pkg.X`) that the PR #119 review found.
- Tests are in `tests/test_arena_e9.py` and `tests/test_arena_transport.py`. They cover:
  - the frames, layout, rate and name mapping;
  - both verdicts, the gravity offset and the runner guard;
  - a plumbing test. With the sham server, the mirror reproduces a plain MuJoCo e9 attempt for seed 50200. Through the reset and the look it is exact (< 1e-9 rad). After that the joints are within 5 mrad and the final apple position within 1 mm, with the same at-rest verdict.
    - The remaining difference is the known `mj_step` layout effect of docs/ISAAC_E9_REPLAY.md §1.
    - The Arena adapter sends no `pre` state, because PhysX gives none.

### Runs (`outputs/`, git-ignored, in the worktree)

| Run | Code | What | Report sha256 |
| --- | --- | --- | --- |
| `arena-kin-1` | `a78a9e5` + an uncommitted addition to `arena_e9.py` (the sham class only; the trajectory code the container ran was as committed) | kinematic check, tutorial scene unchanged except the passive hand sensor and held terminations | compare `9cee9181…` |
| `e9-mujoco-ref-arena-1` | `b585b8a`, clean | MuJoCo e9 reference, seeds 50200–50215, plate error 0 (`e9_replay.py mujoco`) | `d1062b3d…` |
| `arena-e9-server-layout-1` + `arena-e9-layout-1` | `b585b8a`, clean | e9 in the Arena layout variant, **plain** adapter (no gravity offset) | `a3509418…`; rescore at `34bbfcf`: `e3cdc65e…` |
| `arena-e9-server-grav-1` + `arena-e9-grav-1` | `e75540f`, clean | the same with `--gravity_offset` | `9e05f46d…`; rescore at `34bbfcf`: `c09cff41…` |

**Environment.** Image `isaaclab_arena:latest` `sha256:2588b526…` (§1). Host MuJoCo 3.13.0. PhysX GPU, one environment.
- Every Arena run held the shared lock, after a check that load ≤ 2.0 and at least 7 GiB of GPU memory was free. One Arena process ran at a time.
- The scene built in 130–238 s. Kinematic run: 1650 Arena steps in 61 s.
- e9 runs: median 0.09–0.10 s per 20 Hz command (2–3 Arena steps plus the round trip), and 19–22 min per 16 attempts.

**Variant settings** for both e9 runs, chosen before the e9 runs and unchanged between them:
- Robot initial position (0.08, 0.08, 0.068) on a platform with its top at −0.727. The floor is at −0.795, read from the background's floor collider, so the robot stands 6.8 cm higher, which is what `shelf_height_match` gives from the spike's settled pelvis.
- The robot moved 17 cm back from the tutorial's x = 0.25, so its legs clear the invisible shelf support, which starts at x = 0.22.
- After the settle the pelvis stood at (0.024, 0.083, 0.031): **6.1 cm** above the shelf top against our 5.3 cm, leaning back about 4°. Table height is therefore matched to 0.8 cm.
- Apple and plate are placed per seed at our v2 offsets from the settled pelvis. The placement error is ≤ 0.9 mm in `grav-1`.
- The Arena apple (Objaverse mesh, 6.1 cm, 0.097 kg) and plate (cream HOT3D, 15 cm, dynamic 0.5 kg) are Arena's, not ours.

### 7.1 Kinematic check (step 1)

**What ran.** Arena's default (open-arm) pose, then 16 scripted segments: one per arm joint, both arms mirrored, ±0.4–0.8 rad; a combined arm pose; the closed Dex3 synergy; and each finger joint at 70 % of Arena's range. Each segment was a 25-step ramp out, a 25-step hold, a 25-step ramp back and a 25-step hold. FK was computed on our MJCF at Arena's **measured** joints at every one of the 1650 steps, and compared with Arena's link poses. Both are relative to the pelvis. FK at measured joints is unaffected by tracking error or contact.

**Joint names, order and limits**
- The 43 joint names are identical as sets.
- The order differs from index 1 on: Arena's articulation is breadth-first, ours follows MJCF actuator order. All mapping is by name.
- 10 Dex3 limits differ, as the bring-up spike found. Arena's are a strict subset: for example `*_index_0`/`*_middle_0` [−1.571, 0] against [−1.833, 0.192], and the distal joints ±1.745 against ±2.094. Our open and closed synergies lie inside Arena's limits.

**Pose difference, pelvis frame, all 1650 steps**

| Link | Position, max (median) | Rotation, max |
| --- | --- | --- |
| shoulder, elbow, wrist links, both arms | 0.78 mm (0.18 mm) | 0.0002° |
| `*_wrist_yaw_link` (our IK frame) | 0.78 mm (0.19 mm) | 0.0002° |
| our `*_ee` site (wrist-yaw + 0.12 m x) | 0.78 mm (0.18 mm) | — |
| `torso_link` | 10.0 mm (constant) | 0° |
| `*_hand_thumb_0_link` | 3.2 mm | 0.0002° |
| `*_hand_thumb_2_link` | 16.9 mm | 0.0001° |
| `*_hand_index_0` / `middle_0_link` | 23.2 mm | 0.0002° |
| `*_hand_index_1` / `middle_1_link` | 31.3 mm | 0.0002° |
| legs (e.g. `*_ankle_roll_link`) | 0.001 mm | 0.0001° |

**Per-joint origins** (each link relative to its parent; the rotation difference is 0 for every joint, so every joint axis agrees)
- **Torso and waist.** The frames are placed differently but compose to the same chain:
  - waist_roll above waist_yaw: 44 against 35 mm;
  - torso above waist_roll: 0 against 19 mm;
  - shoulder above torso: 247.8 against 237.8 mm.
  
  The sum is the same, 291.8 mm.
- **Dex3: a real geometric difference.** In the wrist-yaw frame:
  - `index_0`/`middle_0` joint origins: x 119.2 mm in Arena against 141.5 mm in our MJCF (2.2 cm closer to the wrist);
  - the next phalanx: 45.8 against 52.8 mm;
  - `thumb_0`: 67.0 against 69.5 mm.
  
  Arena has a `*_hand_palm_link` 41.5 mm ahead of the wrist yaw. Our MJCF has no palm body; its geometry sits on the wrist-yaw link.
- **Conclusion.** The arm kinematics agree, so our IK and the `*_ee` site are valid on Arena's robot. NVIDIA's Dex3 is a different finger geometry: its fingers are shorter and sit closer to the wrist.

**Tracking under Arena's drive (ideal PD, no gravity compensation; WBC upper-body passthrough)**
- The WBC passes the upper-body targets through **exactly**: the maximum difference between Arena's applied PD target and our action is 0.
- Error at the end of each 0.5 s hold, max (median):
  - shoulder pitch 0.06 (0.03) rad and shoulder roll 0.12 (0.03) rad;
  - elbow 0.19 (0.08) rad;
  - wrist pitch 0.31 (0.06) rad, wrist yaw 0.27 rad and wrist roll 0.17 rad;
  - Dex3 ≤ 0.006 rad, except `middle_0` (max 0.63, median 0) and `middle_1` (max 0.31) in one segment each.
- At the default pose the elbow already sags by 0.08 rad.
- The pelvis leans 4–5° when standing (7.7° at most) and moves 1.4 cm in x over the run.
- The apple–hand sensor builds and reports forces.

### 7.2 e9 on our layout in Arena (step 2)

**Results.** 16 attempts each, plate error 0, the same seeds and resets in every column.

| Seeds 50200–50215 | MuJoCo v2 (`e9-mujoco-ref-arena-1`) | Arena, plain adapter (`layout-1`) | Arena, + gravity offset (`grav-1`) |
| --- | --- | --- | --- |
| `apple_at_rest_v0` (ours) | **16/16** | **0/16** on Arena's world state; 0/16 in the mirror | **0/16** on Arena's world state; 0/16 in the mirror |
| Arena contact success (`object_on_destination` at any step) | n/a (latched scorer: 16/16) | **0/16** | **0/16** |
| attempts completed (not stopped by a guard) | 16 | 15 (1 joint-velocity guard) | 14 (2 joint-velocity guards) |
| apple lifted > 2 cm in Arena | 16 (grasped and carried) | 2 (knocked, not grasped) | 0 |
| attempts with apple–hand contact (Arena's sensor) | 16 | 11, mostly at placement (see below) | 4 (≤ 31 steps; no grasp) |
| pelvis drift > 5 cm during the attempt | — (fixed) | 14 | 15 (max 3.6–14.1 cm) |
| final apple–plate distance | 2.3–3.9 cm | 10.0–42.1 cm | 15.2–25.4 cm |

**How the verdicts are computed**
- **Arena success** is Arena's own success term, evaluated at every Arena step. Arena would terminate at the first step it fires.
- **`apple_at_rest_v0` on Arena's world state:**
  - It uses the last 20 commands of the 60-step settle.
  - The apple's centre of mass must be within 4 cm of the plate origin in xy, and within 1.2 cm of the plate's z plus a resting height of 3.29 cm. The resting height comes from a drop onto the plate in each run, read after 100 or 150 steps.
  - Its speed must be ≤ 0.001 m/s.
  - There must be no apple–hand force.
- **Which speed.** The speed is the displacement of the centre of mass per command interval. On an apple resting on the plate whose pose does not change (sub-µm over 0.4 s), PhysX's reported velocity reads 5–10 mm/s. That reported velocity gives a second, labelled reading, which is also 0/16 in both runs.
- **Mirror verdict.** "Mirror" is `run_attempt`'s own verdict on the pelvis-relative mirror state.
- **Rescoring.** Both runs were rescored with the final scorer (`arena_e9.py rescore`). The verdict counts are the same as in the run reports.

**Why the plain adapter fails: the arms stall**
- The commanded translation saturates at ±0.4 (6 mm per command) for hundreds of commands, yet the palm moves < 1 mm per command. In seed 50200 the palm stays 13–15 cm short of e9's orient and descend targets.
- Our embodiment re-bases its IK on the **measured** pose every command. Our MuJoCo actuator adds bias (gravity) compensation; Arena's `IdealPDActuator` does not. The per-command joint increment (0.01–0.04 rad) is then smaller than the sag Arena's gains need to hold the arm (`qfrc_bias / kp` is 0.03–0.07 rad at shoulder pitch and elbow). The arm settles where gravity balances the increment.
- A second defect of that run: the reset pose sagged onto the apple's spawn position. Apples were knocked away during placement in several seeds (for example 147 N of hand force at the first command in seed 50202). This is why `grav-1` also holds the reset pose with the offset; its placements had no hand contact and errors ≤ 0.9 mm.

**Why the gravity-offset adapter fails: the base steps back during the grasp**
- `gravity_offset` adds `qfrc_bias(q) / kp_arena` to each arm and hand target. The bias is computed on our MJCF at Arena's joints, with gravity in Arena's pelvis frame, and is at most 0.087 rad. This gives Arena's PD the holding torque our actuator has.
- With it the arm follows e9. In seed 50200 the palm is within 1–5 mm of MuJoCo's at the end of transfer, lower, steady, open and retreat.
- The grasp fails in every attempt:
  1. In MuJoCo, e9's descent stops when the fingers land on the apple: the palm sits 11.6 cm above the apple centre.
  2. In Arena no finger touches the apple at that height. The fingers are 2.2–3.1 cm shorter (§7.1). In seed 50200 the apple also sits 2.4 cm lower in the pelvis frame, because the pelvis leans back about 4° and stands 0.8 cm high.
  3. In seed 50200 the palm keeps descending to 5 cm above the apple centre. FK on our MJCF at Arena's joints puts the lowest hand geometry centre at −0.02 to −0.03 m world z, the height of the shelf top (−0.030).
  4. During the close phase the floating base then steps **back**, in every attempt: by 3.6–11.7 cm (median about 6.5 cm; more than 5 cm in 14 of 16). By the end of the attempt the pelvis is 4.6–13.1 cm behind where it started.
  5. e9's targets are fixed in the pelvis frame at reset, so after the step the hand closes 5–12 cm behind the apple. No attempt lifted the apple more than 4 mm.
  6. e9 then runs its transfer, release and retreat with an empty hand, and finishes "complete".
- Seed 50200 shows the step clearly: between close commands 210 and 255, Arena's pelvis x goes from −0.010 to −0.103.
- The two joint-velocity guard stops (seeds 50205 and 50207) come from the WBC's leg joints (ankle roll > 5 rad/s while stepping). That guard was written for our fixed-pelvis robot and checks all 43 joints.

**The exact blocker**
- e9's top-down grasp depends on stopping on the apple's top with our Dex3 finger geometry and a fixed pelvis.
- In Arena the fingers are shorter. The hand reaches the shelf instead of the apple, and the AGILE WBC answers that contact by stepping the base back several centimetres. e9 is open loop in the pelvis frame after reset, so it cannot recover.
- **This is a cross-simulator mismatch of the scripted expert, not evidence about any learned controller.**

### Caveats

- **Sample and scope.** The sample is small: 16 development seeds, one plate level, one Arena process per configuration, and no repeat run. The 0/16 results come from one mechanism that recurs in every attempt; they are not a rate estimate.
- **The layout is not the tutorial's.**
  - It is our v2 layout relative to the settled pelvis, with the robot moved back 17 cm and raised 6.8 cm on an invisible platform.
  - Table height is matched to 0.8 cm, but the pelvis leans back about 4°, so in the pelvis frame the shelf is about 2.4 cm lower at the apple than our table.
  - The objects are Arena's: the apple is larger and heavier, the plate is dynamic and shaped differently.
- **The gravity offset changes the commands.** It is a controller-side change to the targets Arena receives. It reproduces our MuJoCo actuator's bias compensation; it is not part of Arena's embodiment. The plain-adapter run is the like-for-like mapping.
- **Pelvis-relative mapping.** Mapping relative to the live pelvis means a base step moves the apparent apple in the mirror. That is the physical truth for IK, but e9's reset-time targets do not follow it.
- **Arena-side scoring.**
  - The resting height is calibrated per run from one drop.
  - Arena's apple–hand sensor covers the wrist-yaw, palm and finger links only.
  - The mirror's own contact flags use our sphere apple and our hand geometry.
- **Unpinned assets.** The network assets are unpinned, as in §6.

### Next steps (development, cheapest first)

1. **Decide whether e9 may be adapted to Arena.** Any change would be declared before the run, and its result labelled an Arena-adapted scripted expert, not e9. Two candidate changes:
   - stop the descent at a measured height above the apple (no reliance on finger contact);
   - re-plan e9's phase targets from the current apple position relative to the live pelvis, rather than freezing them at reset.
2. **Measure base compliance alone.** A short scripted press of the hand on the shelf, with no apple, would show how the WBC steps under arm contact. Alternatively a fixed-root variant (`fix_root_link`), which is no longer the tutorial's embodiment.
3. **Then step 3 of §5** (the tutorial layout with a left-handed e9). It inherits both blockers above, so it is not worth running first.
