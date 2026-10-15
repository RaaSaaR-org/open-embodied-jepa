# SO-101 branch plan (opened 2026-10-16)

**Status: PLAN, no code yet.** On 2026-10-16 the owner opened the SO-101 branch
([DECISIONS 2026-10-16 (bp)](DECISIONS.md), R28.1–R28.9). The defaults below were chosen by
Claude under owner delegation; the owner can overrule any of them. This page is a plan: it admits
no experiment, sets no bar and makes no claim. Each research step still needs its own
preregistration, independent review and ruling ([AGENTS.md](../AGENTS.md)).

## What the branch is

- **Beside G1 + Dex3, as a stepping stone.** The LeRobot SO-101 (five Feetech STS3215 arm joints
  and a jaw gripper) runs beside the G1 + dual Dex3 line. G1 code, configs, checkpoints, results,
  fired clauses and the R7 canonical sentence are untouched. The G1 grasp line stays paused
  (R27.211) and TASK-095 stays on hold (R27.210).
- **Simulation first.** No real-hardware execution until a separate owner ruling. Buying the
  arm (follower, and a leader for teleoperation) is the owner's decision (S6).
- **Why SO-101.** The binding constraint on the G1 line was centimetre-level perception and
  placement at contact (TASK-075 OBS-NONE; TASK-094's 1.2 cm +y palm offset cut the scripted
  follower from 88/128 to 31/128 lifts; R27.214). The expectation, **not measured**, is that a jaw
  gripper with a wrist camera tolerates more placement error than a dexterous hand seen from one
  112 px onboard camera, and that a cheap arm makes real data possible later. S1–S3 measure that
  tolerance before any model is trained.

## Chosen defaults (Claude, under owner delegation; R28.3)

| Item | Default |
| --- | --- |
| Relation to G1 | Beside it, a stepping stone; G1 line and R7 unchanged |
| Execution | Simulation only until an owner ruling on hardware |
| Action schema | `joint_delta_so101_v0`: 5 arm joint-position deltas + 1 gripper command |
| Motor order | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (LeRobot `so_follower` order) |
| Cameras | A wrist camera on the gripper and a fixed front/overview camera |
| First tasks | Cube pick-and-lift (`so101_cube_lift_v0`), then pick-and-place into a target (`so101_cube_place_v0`) |
| Community Hub datasets | Deferred; a licence check per dataset before any use |

## Lessons carried from the G1 line

1. **Calibrate bars from measured ceilings.** TASK-074's O2 bar (1.0 cm) sat below the measured
   error of the readout it applied to, so the gate was unreachable whatever the predictor did.
   Every SO-101 bar is set from a measured ceiling (the scripted follower, a privileged readout,
   a clock prior) on development seeds, not carried over.
2. **Centimetre perception at contact was the binding constraint.** Hence the wrist camera from
   the start, and a measured placement-tolerance envelope (a TASK-094 K-style dose: scripted
   follower with rigid gripper offsets) before any learned controller is designed.
3. **Measure before gating.** Corpora are measured (coverage, lift windows, action ranges) before
   training (TASK-085, TASK-090 Stage C); offline gates (collapse, action sensitivity, copy-last,
   no-action baseline) come before any closed loop (TASK-065, TASK-066, TASK-077).
4. **A scripted reference arm in every closed loop.** On G1 the scripted IK follower reached or
   lifted far more often than every world-model arm; "LeWM needed" was never shown. The SO-101
   studies keep a scripted follower and non-learned rule arms beside the learned ones.
5. **Implementation hashes are a contract.** The TASK-072 edit to `models/base.py` made older
   checkpoints load only at their recorded revision. S0 must not change the hash of any G1 model
   path, or must record a compatibility manifest like
   `benchmarks/manifests/task072-checkpoint-compatibility.json`.
6. **Training budgets that do not saturate** (`last_two_triggered`) weakened several G1 readings;
   S4 declares a saturation check.

## Phases

Effort figures are **estimates**, not measurements, in agent-assisted working days (d) or GPU
hours (GPU h) on the shared RTX 5080.

### S0 — Foundations: action-dimension generalisation and `joint_delta_so101_v0` (TASK-096)

- **Delivers.** The core takes the action dimension from the action schema instead of the module
  constant `ACTION_DIM = 14`: `contracts.py` (an `ActionSchema` registry with
  `ee_delta_grasp_v0` unchanged and `joint_delta_so101_v0` added, names and order fixed),
  `config.py` (lines 201–206 accept a registered schema instead of rejecting all but
  `ee_delta_grasp_v0`), `planning.py` (CEM bounds and sample shapes from the schema), `data.py`
  (feature shapes and the final-row padding from the manifest's schema), `policy.py`, and the
  literal 14s in `models/native.py`, `models/lewm.py` (`input_dim=14`), `models/frozen_encoder.py`
  and `models/latent_critic.py`. G1 task-line modules stay as they are.
- **Check.** Every existing test passes unchanged; new tests cover a 6-D schema through config
  validation, data round-trip, CEM and both backends' `train_step`/`predict` on synthetic data;
  the core-import check (no `torch`/`mujoco` in `sys.modules` after `import embodied_jepa`) still
  passes; G1 model implementation hashes are unchanged, or a compatibility manifest is committed
  and reviewed; a G1 config still resolves to the same 14-D path.
- **Effort (estimate).** 2–4 d.

### S1 — Pinned SO-101 MuJoCo embodiment (TASK-097)

- **Upstream (verified on 2026-10-16 via the GitHub API; not yet fetched).**
  - MuJoCo Menagerie `robotstudio_so101/` (`so101.xml`, `scene.xml`, `scene_box.xml`, `assets/`),
    Apache-2.0 per its `LICENSE`; last change to that directory at commit
    `3abca2e302625cec9429c49806121ea2855cef81` (2026-09-28). Derived from TheRobotStudio's
    `so101_new_calib.xml`; adds primitive collision geometry, gripper contact parameters and a
    camera mount. Joint and actuator names match the LeRobot motor order; six position actuators.
    Requires MuJoCo ≥ 3.1.3 (the lock has 3.13.0). **Preferred.**
  - TheRobotStudio `SO-ARM100` `Simulation/SO101/` (`so101_new_calib.xml`,
    `so101_new_calib_camera.xml`, URDFs, `scene.xml`), Apache-2.0, `main` at
    `a758567c3978dfeefe282ede0500085a48fe8f78` (2026-10-06). The source; fallback.
  - Menagerie's `trs_so_arm100` is the SO-100, not the SO-101; not used.
  - The pinned SHA, file hashes and licence go into [DEPENDENCIES.md](DEPENDENCIES.md) only when
    the fetch script actually fetches them; the SHAs above are re-checked then.
- **Delivers.** `scripts/fetch_so101.py` (pinned revision, per-file SHA-256, into ignored
  `assets/`); an SO-101 tabletop scene (table, cube, target region, a wrist camera on the gripper
  mount and a fixed front/overview camera); `configs/so101_sim_action.json` (joint limits, delta
  scales, gripper range, control rate); an `so101` embodiment adapter implementing the
  `Embodiment` protocol (`observe`, `state`, `execute`, `normalize_action`,
  `denormalize_action`, `stop`) with **IK-free joint-position control** (deltas added to the
  current target and clipped to joint limits, which are recorded, never silently applied to the
  stored action); a MuJoCo transport; registry entries as `"module:attr"` strings, imported
  lazily.
- **Check.** Fetch is reproducible from a clean checkout and hash-verified; the scene loads and
  steps under `MUJOCO_GL=egl`; both cameras render and their frames are inspected (cube visible
  in the wrist view at grasp distance); action round-trip (normalize/denormalize) is exact;
  joint-limit and rate guards are tested; a zero action holds pose; determinism with a fixed seed;
  core-import check still passes.
- **Effort (estimate).** 3–5 d.

### S2 — Tasks `so101_cube_lift_v0` and `so101_cube_place_v0` (TASK-098)

- **Delivers.** Task modules owning resets (cube pose, and target pose for place, from declared
  distributions), goals, termination and scoring. Scoring uses simulator truth only and never
  reaches model inputs or planning costs. Lift: the cube is held above a declared height for a
  declared duration. Place: the cube **at rest** inside the target region after release (no
  latched transient crossings; the TASK-067 scorer defect is not repeated).
- **Check.** Scorer unit tests on synthetic trajectories (transient crossing does not count;
  a dropped cube does not count); a privileged scripted follower's success rate on development
  seeds is **measured and reported** (it sets what later bars can be); a K-style placement
  tolerance envelope (the follower with rigid gripper offsets of a few sizes) is measured on
  development seeds.
- **Effort (estimate).** 3–5 d plus CPU runs.

### S3 — Scripted collector and play corpus, measured before training

- **Delivers.** A scripted collector (privileged, like `play_corpus.py`) that records reach,
  approach, close, lift and place segments with injected perturbations into a `DatasetStore`
  with both cameras; episode-grouped splits; a corpus manifest with hashes.
- **Check.** Corpus measures before any training: action-range coverage per joint, gripper
  open/close balance, lift and place window counts, camera frame checks, split leakage check.
- **Effort (estimate).** 3–5 d to write and tune; collection itself probably short (G1's
  `play-v1`, 3 194 episodes, took 1 025 s with 12 workers), but the SO-101 scene is untimed.

### S4 — LeWM and `native_jepa` training with offline gates

- **Delivers.** A preregistered offline study: both backends (a one-line backend swap) on the S3
  corpus, likely with frozen DINOv2 token latents as the reference encoder plus a random-init
  floor, three model seeds.
- **Check (offline gates, bars calibrated on development data).** No-collapse rank; action
  sensitivity; beats copy-last; beats an equally trained no-action predictor; a readout of the
  cube and gripper from predicted latents, with its bar set from the encoded-frame readout's
  measured error; a training-saturation check.
- **Effort (estimate).** 3–5 d plus 10–20 GPU h, every GPU job under
  `scripts/gpu_run.sh --wait --board`.

### S5 — First preregistered closed-loop study in simulation

- **Delivers.** A preregistration fixing fresh seeds, bars, margins and row order before
  anything runs: a LeWM-driven controller (planning, or LeWM choosing goals for a local
  controller) on `so101_cube_lift_v0` first, with the scripted follower as the reference arm and
  action-blind, scene-blind and random controls.
- **Check.** Its own gated stage and ruling. A pass is one run in simulation; "LeWM needed" is a
  separate claim against the scripted and rule arms.
- **Effort (estimate).** 1–2 weeks including Stage 0 and the gated stage; a few GPU h.

### S6 — Real hardware (only after an owner ruling)

- Needs the owner's purchase decision and a ruling allowing real execution. Then: a LeRobot-backed
  real transport (disabled by default, mock-tested like `hardware.py`), calibration against the
  simulated joint conventions (Menagerie uses TheRobotStudio's "new calibration", zero at the
  middle of each range; LeRobot's gripper is 0–100), safety limits and a teleoperated
  demonstration corpus with a leader arm. Community Hub datasets enter only after a per-dataset
  licence check.

## What does not change

R7's canonical sentence; the frozen v1 benchmark and its 0/150 per backend; every fired clause
(TASK-054, TASK-057, TASK-062, TASK-065, TASK-075, R26.33, R27.71, R27.149); the paused G1 grasp
line and TASK-095's hold; the backend-swap invariant and the core-import check.
