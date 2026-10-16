# SO-101 branch plan (opened 2026-10-16)

**Status: PLAN; S0 done (#232, `bcd837c`); S1 done (#233, `cf2d4b1`); S2 done (#234; TASK-098, development measurement only, [so101_cube_tasks_dev.md](experiments/so101_cube_tasks_dev.md)); S3 done (#235; TASK-099, development data only, [so101_play_corpus_v1.md](experiments/so101_play_corpus_v1.md)); S4 DRAFT preregistration (TASK-100, [so101_world_model_v1.md](experiments/so101_world_model_v1.md); nothing run).** On 2026-10-16 the owner opened the SO-101 branch
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
  follower from 88/128 to 31/128 lifts in one simulation-only run with a scripted offset of one
  size; R27.214). The expectation, **not measured**, is that a jaw
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
4. **A scripted reference arm in every closed loop.** On G1 the scripted IK follower lifted far
   more often than every world-model grasp arm (and reached 64/64, against 62/64 and 53/64 for
   the TASK-088 reach arms); "LeWM needed" was never shown. The SO-101
   studies keep a scripted follower and non-learned rule arms beside the learned ones.
5. **Implementation hashes are a contract.** The TASK-072 edit to `models/base.py` made older
   checkpoints load only at their recorded revision. S0 keeps every G1 hash-pinned file
   byte-identical (see S0's coupling note); the fallback is a compatibility manifest like
   `benchmarks/manifests/task072-checkpoint-compatibility.json` plus a ruling.
6. **Training budgets that do not saturate** (`last_two_triggered`) weakened several G1 readings;
   S4 declares a saturation check.

## Phases

Effort figures are **estimates**, not measurements, in agent-assisted working days (d) or GPU
hours (GPU h) on the shared RTX 5080.

### S0 — Foundations: action-dimension generalisation and `joint_delta_so101_v0` (TASK-096)

- **Status: done in #232 (`bcd837c`, TASK-096).** Implemented in new files only:
  `action_schemas.py` (schema registry, `joint_delta_so101_v0`), `models/schema_backends.py`
  (`SchemaNativeJEPA`, `SchemaLeWM`), `schema_planning.py` (`SchemaCEMPlanner`) and
  `schema_data.py` (`SchemaDatasetStore`); `config.py` changed additively. Every pinned file
  named below is byte-identical, and the G1 implementation hashes and all four
  `frozen_files_check` results matched `origin/main` (`dc5c686`), so no compatibility manifest
  was needed. Not yet schema-sized (left to the phase that needs them, per `docs/MODELS.md`):
  the frozen-encoder and frozen-token mixins, `policy.py`, `models/latent_critic.py`, the
  `jepa_wms` and sensor backends, `audit`, the `training` and `benchmark` CLIs and encoded
  dataset forks. The text below is the plan as written before #232.

- **Coupling with G1 hashes (named by the independent review of #231).** Two kinds of byte
  pins reach the files a naive generalisation would edit:
  - *Model implementation hashes.* `VisualModel.implementation_sha256` (`models/base.py`) hashes
    `base.py`, the concrete backend's file (`models/native.py` or `models/lewm.py`), `readout.py`
    and `readout_labels.py`; `models/frozen_encoder.py` and `models/frozen_tokens.py` add their
    own files; `policy.py` hashes itself; TASK-087's `grounded_wm_model.implementation_sha256()`
    hashes `grounded_wm_model.py`, `grounded_wm.py` and **`models/lewm.py`**, and
    `zero_shot_runtime.py` and `grasp_corpus_wm_runtime.py` check it. A byte change to any of
    these files invalidates every native/LeWM checkpoint since TASK-072 (TASK-077/080/081/083's
    W, N and twin checkpoints included) or TASK-087's G checkpoints, the paused grasp line's
    resume path (TASK-095).
  - *Frozen-protocol file pins.* TASK-091–TASK-094's frozen modules pin the SHA-256 of
    `contracts.py`, `planning.py`, `models/lewm.py`, `embodiment.py`, `simulation.py`,
    `constraints.py`, `devices.py`, `run_tools.py` and `configs/g1_sim_action.json`
    (`FROZEN_FILES_SHA256`), and `tests/test_grasp_*.py` assert those pins, so editing any of
    them fails CI as well as the G1 runners' pre-stage check.
  - *Earlier frozen-manifest pins.* `data.py` is byte-pinned in the `hashes`,
    `source_hashes` or `input_identities` of `apple-look-corpus-v1` and `apple-look-corpus-v1-results`,
    `apple-token-dynamics-v1`, `apple-latent-dynamics-v1`, `apple-first-policy-v1`,
    `apple-goal-alignment-v1`, `apple-aligned-cost-v1` and `apple-wide-object-ceiling-v1`;
    `audit.py` and `benchmark.py` are byte-pinned in `apple-control-development-v1`,
    `apple-goal-alignment-v1`, `apple-aligned-cost-v1` and `apple-wide-object-ceiling-v1`
    (all under `benchmarks/manifests/`). The TASK-064/066/067 runners
    (`collect_apple_look.py`, `train_apple_token_dynamics.py`, `run_first_policy.py`) check
    their manifest's `hashes` at preflight, and
    `tests/test_latent_dynamics.py::test_committed_pins_match_the_tree` asserts TASK-065's pins,
    so editing `data.py` fails CI and breaks those runners' reproduction path.
- **Design (chosen).** Every file above stays **byte-identical**. `ee_delta_grasp_v0` stays on
  the existing classes and constants. The SO-101 path lives in new files: a schema module that
  registers `joint_delta_so101_v0` (names, order, dimension 6) beside the frozen 14-D one,
  schema-parametrised subclasses of the native and LeWM backends in new files that override
  only the action-input construction (the action embedder width), and a schema-aware CEM
  wrapper or subclass in a new file; the schema-aware dataset path also lives in a new file,
  so `data.py`, `audit.py` and `benchmark.py` stay byte-identical too. Only `config.py`, which
  no frozen manifest pins at its current revision, may change additively, with the
  `ee_delta_grasp_v0` path unchanged. **Fallback**, only if a
  pinned file cannot stay byte-identical: a TASK-072-style compatibility manifest naming the
  invalidated checkpoint families (the VisualModel backends and TASK-087's G), the hash-checking
  runtimes and frozen protocols, and the last compatible revision, recorded by a ruling before
  merge; frozen modules' pins are never edited to make tests pass.
- **Delivers.** Configs, data, planning and both backends take the action dimension from the
  action schema for the SO-101 path, including but not limited to the hard-coded 14s and
  `ACTION_*` constants used by `contracts.py` (`ACTION_NAMES`, `ACTION_DIM` and the
  validators), `config.py` (lines 201–206 accept only `ee_delta_grasp_v0`), `planning.py` (CEM
  bounds and sample shapes), `data.py` (feature shapes, the final-row padding, the `[T,14]`
  messages at lines 94 and 127), `policy.py`, `audit.py` (`ACTION_NAMES`), `models/native.py`,
  `models/lewm.py` (`input_dim=14`), `models/frozen_encoder.py`, `models/frozen_tokens.py`
  (`[..., 14]`) and `models/latent_critic.py` (itself byte-pinned by the TASK-073–076 v2
  manifests), reached through the new files where the file is pinned. G1 task-line modules stay
  as they are.
- **Check.** Every existing test passes unchanged; new tests cover a 6-D schema through config
  validation, data round-trip, CEM and both backends' `train_step`/`predict` on synthetic data;
  the core-import check (no `torch`/`mujoco` in `sys.modules` after `import embodied_jepa`) still
  passes; every G1 hash-pinned runtime check still passes at HEAD (each `frozen_files_check`,
  and the G1 model implementation hashes of native, LeWM, frozen-encoder, frozen-token, policy
  and TASK-087's G are unchanged, compared against `origin/main` before the change); a G1 config
  still resolves to the same 14-D path.
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
  `assets/`); an SO-101 tabletop scene (table, cube, target region, the wrist camera that
  Menagerie's `so101.xml` already defines (`wrist_cam` on its `camera_mount` body, reused rather
  than new geometry) and a fixed front/overview camera); `configs/so101_sim_action.json` (joint
  limits, delta scales, gripper range, control rate); an `so101` embodiment adapter implementing
  the `Embodiment` protocol (`action_schema`, `state_schema`, `observe`, `state`, `execute`,
  `normalize_action`, `denormalize_action`, `stop`) with **IK-free joint-position control**
  (deltas added to the current target and clipped to joint limits, which are recorded, never
  silently applied to the stored action); a MuJoCo transport in a new module (the pinned
  `simulation.py` and `embodiment.py` stay byte-identical); registry entries as `"module:attr"`
  strings, imported lazily.
- **Check.** Fetch is reproducible from a clean checkout and hash-verified; the scene loads and
  steps under `MUJOCO_GL=egl`; both cameras render and their frames are inspected (cube visible
  in the wrist view at grasp distance); action round-trip (normalize/denormalize): arm channels
  exact, gripper within 2e-7; joint-limit and rate guards are tested; a zero action holds pose;
  determinism with a fixed seed;
  core-import check still passes.
- **Effort (estimate).** 3–5 d.

### S2 — Tasks `so101_cube_lift_v0` and `so101_cube_place_v0` (TASK-098)

- **Status: done (#234, TASK-098; development only, no gate, no claim).**
  `so101_tasks.py` (declared resets, held-window scorers on simulator truth), `so101_followers.py`
  and `scripts/measure_so101_tasks.py`. The privileged scripted follower (v1) scored 64/64 on
  each task on development seeds 100–163 (exact 95 % 0.944–1.000). Its grasp-offset envelope is
  direction-dependent (5 mm toward the fixed jaw, 10 mm along the face or upward, 20 mm away
  from the fixed jaw, each keeping 64/64), and a place-aim error up to 3 cm along any axis keeps
  at least 63/64. Results, history (a v0 run on seeds 0–63) and caveats are in
  [so101_cube_tasks_dev.md](experiments/so101_cube_tasks_dev.md). The text below is the plan as
  written before.
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

- **Status: done (#235, TASK-099; development data, no gate,
  no claim).** `so101_play_corpus.py` (the privileged scripted collector `so101_play_policy_v0`),
  `scripts/collect_so101_play.py` and `scripts/measure_so101_play.py`. The corpus
  `so101-play-v1` has 3 000 episodes of 400 commands at 224 px for both cameras, split
  2 700 / 150 / 150 by episode, 26.5 GiB, collected in 14.4 min with 12 workers. Its measures
  before training: every action bin covered, no split leakage, 3 262 lift windows, 65.4 % of
  pick attempts grasped, and the joint coverage partial (wrist_flex and shoulder_pan). Results
  and caveats are in [so101_play_corpus_v1.md](experiments/so101_play_corpus_v1.md). The text
  below is the plan as written before.

- **Delivers.** A scripted collector (privileged, like `play_corpus.py`) that records reach,
  approach, close, lift and place segments with injected perturbations into a `DatasetStore`
  with both cameras; episode-grouped splits; a corpus manifest with hashes.
- **Check.** Corpus measures before any training: action-range coverage per joint, gripper
  open/close balance, lift and place window counts, camera frame checks, split leakage check.
- **Effort (estimate).** 3–5 d to write and tune; collection itself probably short (G1's
  `play-v1`, 3 194 episodes, took 1 025 s with 12 workers), but the SO-101 scene is untimed.

### S4 — LeWM and `native_jepa` training with offline gates

- **Status: DRAFT preregistration (TASK-100, [so101_world_model_v1.md](experiments/so101_world_model_v1.md);
  DECISIONS 2026-10-17 (bq), R28.10–R28.26).** Nothing has been featurised, trained or evaluated;
  the protocol freezes at its Stage 0 merge. The text below is the plan as written before.

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
