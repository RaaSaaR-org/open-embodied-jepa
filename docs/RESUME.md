# Resuming the project (paused 2026-10-15)

**Status: PAUSED.** On 2026-10-15 the owner paused the project
([DECISIONS 2026-10-15 (bo)](DECISIONS.md), R27.209–R27.215). Nothing is running or queued.
TASK-095 is recommended but **not confirmed and not started**. This page is the handoff: what
works, what does not, where the evidence is, how to work on the machine, and how to resume each
option, including a branch to the LeRobot SO-101 arm.

**Update 2026-10-16: the SO-101 branch is open.** The owner opened it
([DECISIONS 2026-10-16 (bp)](DECISIONS.md), R28.1–R28.9); its plan is
[SO101_PLAN.md](SO101_PLAN.md). The G1 grasp line stays paused and TASK-095 stays on hold.

**Update 2026-10-21: paused again ([DECISIONS 2026-10-21 (ce)](DECISIONS.md), R28.148–R28.152).**
The SO-101 branch stops at TASK-102 amendment A1. To resume the SO-101 line:

1. **Get PR #247 independently reviewed, then merge it** (branch `docs/task102-a1`, worktree
   `worktrees/task102-a1`; CI was green, but no review was posted). It carries the A1 record
   `experiments/so101_hybrid_lift_v1_a1_planner.md` (row **A1-MODEL-BOUND**, unreviewed) and
   this pause entry.
2. **Where it stands** (simulation, development only, no gated closed loop has run):
   - TASK-100: WM-PASS offline with `native_jepa` (J) only; LeWM failed only G6, the
     training-saturation check, at a capped budget.
   - TASK-101: closed with no claim.
   - TASK-102: the learned wrist-camera grasp G_loc lifts after an exact approach (30/32, and
     28/32 at 0.08 rad off). With the jaw scripted open, the privileged exact-pose cost
     hands over 8/8 in 15 of 16 planner configurations, but J's predicted cost hands over at
     most 1/8, and 3/8 with a 4× larger search (8 seeds per cell). So J's cost, not the planner,
     is the limit.
3. **Next (needs a ruling):** a model change. Either R28.57's LeWM longer-budget retrain of
   TASK-100's L, or a cost that resolves the arm pose better. Each needs its own
   preregistration. TASK-102 stays in progress, not frozen; its reserved cohorts P′, A′, D′ and
   the gated seeds 103200–103327 are unused.
4. Evidence: `~/develop/emai/evidence/task100-*`, `task101-*`, `task102-*` (each with
   `SHA256SUMS`); TASK-100's checkpoints `checkpoints/task100-run/` and feature stores
   `outputs/task100-features-*` on `/mnt/data`.

This page summarises; it is not the record. Every number below is copied from its results
document or decision with its qualifiers, and those documents win if anything here disagrees.
The canonical status sentence is R7 (DECISIONS 2026-10-02, as updated by R18.28, R18.31, R19.18,
R19.21 and R21.20). It is quoted verbatim in [CLAUDE.md](../CLAUDE.md), [PLAN.md](PLAN.md) and
the [experiment index](experiments/README.md), and it is unchanged by the pause.

## Current state in plain terms

### What works (all in MuJoCo simulation; nothing runs on the real robot)

- **The software stack.** Two world-model backends (`native_jepa` and a LeWM adapter over pinned
  upstream source) share one data format (LeRobot v3 PNG-in-Parquet plus a JEPA manifest), one
  CEM planner and one evaluator. A backend swap is a one-line config change. Green CI and smoke
  runs are software evidence, not working manipulation.
- **The scripted expert e9 on `apple-to-plate-v2`** left the apple at rest on 32/32 fresh gated
  seeds with the plate exact and 30/32 at 1.0 cm plate error (TASK-070). It is privileged and
  scripted, not learned.
- **A behaviour-cloning pick (P-3)**, an MLP on a frozen DINOv2 readout trained on e9's
  demonstrations (not a world model), scored 40/40 counted successes on cohort C against 39/40
  for its random-init encoder control R-3 (TASK-072 M2: one run, one training seed per arm, 40
  resets, one camera at 112 px onboard, a narrow reset distribution). The row is M2-FAIL on G3:
  encoder pretraining contributed nothing measurable. Cohort C is no longer held out.
- **LeWM choosing one place aim on v2.** In TASK-081, a LeWM token predictor on frozen DINOv2
  features chose only the single place aim between P-3's learned pick and e9's scripted place,
  under the declared simulation-only moving-plate condition C1-M. On 128 fresh gated resets it
  scored 118/128 (exact 95 % interval 0.861–0.962; one run, one model seed whose
  `last_two_triggered` flag is set): L-PASS. "LeWM needed" is not shown: the non-learned rule
  H-rule, given the simulator's plate law, scored 125/128 and was measurably better (paired
  difference −7/128, 95 % interval −13/128 to −2/128). The solver's effect is not shown. TASK-083
  replicated L-PASS with the two other model seeds of the same training run, 124/128 and 120/128
  on 128 further fresh gated resets (one run, simulation only; H-rule 123/128 on that cohort, not
  measurably better than either seed; W-66800, reported only, 123/128 on the same resets). The
  three seeds share one corpus, recipe and encoder. The privileged look-ahead ceiling scored
  126/128 on both cohorts, and TASK-081's 128-reset cohort was chosen after TASK-080's L-NEAR
  (58/64 against H-rule's 63/64). R7 carries the full qualifiers.
- **World-model reach.** In TASK-088, CEM over TASK-087's models reached new goals: the gated
  arm P (plain LeWM, an image-only cost) 53/64 (exact 95 % 0.71–0.91; bar 52/64, cleared by one
  reset), and the declared second arm G (image + pose) 62/64 (0.89–1.00), on 64 fresh gated
  resets, one model seed gated, simulation only. The row is Z3-REACH. "LeWM needed" is not shown:
  the scripted IK follower, with the same goal poses and no world model, reached 64/64.

### What does not work

- **Whole-task world-model planning.** Image-goal CEM on the frozen v1 MVP benchmark (TASK-020)
  scored 0/150 per backend (`native_jepa` and LeWM); every learned episode ended on a joint-rate
  guard stop (the zero-shot plan reads that failure as confounded). LeWM's only closed-loop Apple→Plate runs on v1 have 0
  successes.
- **World-model grasping.** Every gated world-model grasp-and-lift closed loop failed its bar,
  while the scripted IK follower, which uses the same goal poses and no world model, lifted far
  more often; the offline gates and the diagnostics on the grasp line did not open a way past
  that. Each is one run, in simulation only:
  - TASK-088: 0/64 for every world-model arm (0.00–0.06; bar 26/64) against 50/64 for the
    follower.
  - TASK-089 (B-FAIL): G (seed 87100) 0/64 (0.00–0.06; bar 26/64) against 48/64; G's other two
    seeds 8/64 and 7/64, reported only. Its clause fired (R26.33): no more planner, cost or
    horizon variants over TASK-087's checkpoints for grasping.
  - TASK-090 (A′-FAIL) and TASK-091 (B-FAIL): grasp-enriched play data lifted 22/384 pooled over
    three seeds against 26/384 for the equal-size control (sign-flip p = 0.77); follower 82/128.
    TASK-090 Stage C's data measures apply: the added data held 2.38× the lift windows of the
    equal-size control (C1) and 1.69× in the combined training sets (C2).
    Its clause fired (R27.71): no more scripted grasp-and-lift play of that collector's kind.
  - TASK-092 (A‴-FAIL, an offline no-regression gate; no gated closed loop ran). Its
    development reading D-RATE′ (not gated, 32 resets, which cannot establish an effect) ended
    D-FLAT, 0/32 against 0/32.
  - TASK-093 (B-FAIL): rate-projected planning 20/384 against 24/384 unprojected (bootstrap 95 %
    interval of the pooled rate difference −0.039 to +0.018); follower 91/128. Its clause fired
    (R27.149).
  - TASK-094 (K-NULL; a measurement, not a world-model attempt; no model in K): at the follower's exact pose, partial
    closure is not shown to cause the lost lifts (K-1.0 88/128, exact 95 % 0.600–0.766; losses
    +0.031 and +0.055, both bounded below 0.15). A rigid 1.2 cm +y palm offset cut the follower to
    31/128 (Δ +0.445, reset-bootstrap 95 % +0.352 to +0.539; a scripted dose of one size, not the
    world-model arms' errors).
- **Centimetre-level perception from the onboard camera.** TASK-075 ended OBS-NONE: with frozen
  DINOv2 features and linear or kernel-ridge readouts of single frames, no view (onboard 112 and
  224 px, the hand crop, overview 224 px) read the apple-minus-plate offset within τ = 1.0 cm
  (upper bounds 2.66–3.44 cm for pooled tokens; best 2.02 cm). Trained detectors, fine-tuned
  features, temporal aggregation and other sensors were not tested.

### The assessment at the pause (an opinion, not a ruling)

Recorded as R27.214. End-to-end world-model dexterous grasping from the one 112 px onboard
camera is unlikely on the current path. The binding constraint keeps being centimetre-level
perception and placement at contact: TASK-075's OBS-NONE, and TASK-094's 1.2 cm +y offset
(88/128 → 31/128 for the scripted follower), to which the frozen planning cost's pose term is
nearly blind (a median change of 0.0065, against 1.09 for a full closure change; R27.203,
arithmetic, a best case for G). The WM arms' held palm error during the close was a median 2.62 cm
(R27.200; from run reports, pooled over arms, seeds and cohorts for orientation only, read after
the results; its direction and orientation were not logged). The realistic directions
are a hybrid controller, better sensing, or a different embodiment. None is decided.

### Open MC tasks

- `TASK-095` (todo, **on hold**): the grasp placement-resolution diagnostic; recommended by R27.205,
  not confirmed by the owner.
- `TASK-025` (Isaac / Arena transport, development only) and `TASK-026` (commission SDK2 on the
  physical G1, future) stay in todo.

## Where things live

- **Code and records:** this repository, `main`. Protocols and results are in
  [docs/experiments/](experiments/README.md), rulings in [DECISIONS.md](DECISIONS.md) (newest
  first), manifests in `benchmarks/manifests/`, tasks in `.mc/tasks/`.
- **Evidence (SSD):** `~/develop/emai/evidence/<task-stage>/`, one folder per gated stage, each
  with a `SHA256SUMS` file whose hash the decision entry records. The newest are
  `task094-gated/`, `task094-gated-recompute/` and `grasp-next-v2/`. The folder's `README.md`
  explains the copies restored from the failing USB disk on 2026-10-04.
- **Data, checkpoints and run outputs (HDD):** since 2026-10-12 the repository's `data/`,
  `checkpoints/` and `outputs/` are symlinks to
  `/mnt/data/emai/projects/open-embodied-jepa/{data,checkpoints,outputs}` (a rotational ext4 HDD;
  its `README.md` describes it). Other projects share `/mnt/data/emai/`; never touch their
  directories. Copy a dataset to the SSD before I/O-heavy training.
- **USB archive disk:** `/media/huhn/INTENSO` is failing; see [STORAGE.md](STORAGE.md) before
  reading from it.
- **Never overwrite evidence** under `data/`, `checkpoints/`, `outputs/` or the evidence folders;
  add, do not replace.

## Working on the machine

- **Setup.** The Linux PC with the RTX 5080 is the working platform ([SETUP.md](SETUP.md)).
  `uv sync --locked` with every extra (see [CLAUDE.md](../CLAUDE.md)), then always
  `uv run --no-sync`. Set `MUJOCO_GL=egl` for rendering over ssh.
- **Worktrees.** One worktree per branch under `~/develop/emai/worktrees/<name>`, created with
  `scripts/new_worktree.sh <dir> --branch <name>` (its own `.venv`) and removed with
  `scripts/remove_worktree.sh <dir>`. Gated runs go in a clean, detached run tree:
  `scripts/new_worktree.sh <dir>-run --from <frozen sha> --run`. Runners hash-pinned by a
  manifest are never edited. Merged remote branches are not deleted (R6).
- **GPU sharing.** The GPU is shared with other projects. Every GPU job goes through
  `scripts/gpu_run.sh --wait --board -- <cmd>`, which takes the machine-wide lock
  `~/.local/state/gpu/lock` and records the holder in `~/.local/state/gpu/BOARD.md`. Before a GPU
  job, read the board's "Requests" section. It works both ways: anyone may add a
  `- <who>: <what>, <how long>` line, and the V2D queue of another project pauses between jobs
  until that line is removed; when the V2D session has posted a `- v2d: ...` line, start with
  `--wait` so that its job goes first. Tell that session in advance about any multi-hour job. Resident services
  (the GR00T server, other queues) are never stopped or reconfigured. CPU-only stages do not take
  the lock.
- **Workflow.** [AGENTS.md](../AGENTS.md): branch → commit → PR → independent review → merge.
  Every research step needs a preregistration that fixes seeds, bars, margins and row order
  before anything runs, and a ruling in DECISIONS.md.

## How to resume

First steps for any option: `git pull` on `main`; `uv sync --locked` with every extra;
`uv run --no-sync pytest`; `mc task board`; read the newest DECISIONS entries ((bo), (bn), (bm));
check the GPU board. Every option below needs an owner ruling first. The GPU-hour figures are
estimates from R27.204, not measurements.

### TASK-095, the placement-resolution diagnostic (measurement only; about 0.6 GPU h)

1. The owner confirms TASK-095, which lifts the pause for this measurement only. Record the
   ruling in DECISIONS.md and remove the hold note from the card.
2. Write the DRAFT preregistration `docs/experiments/grasp_placement_resolution_v1.md` from the
   card and [grasp_line_next_v2.md](experiments/grasp_line_next_v2.md): K′ (21 palm-offset arms),
   R (cost ranking at R1 roots), S′ and V (a CPU-only command replay). Seeds go through the seed
   check; bars, margins and row order are fixed. Get an independent review, then merge.
3. Stage 0: opt-in code (TASK-094's runners and summarizer stay unedited; a new summarizer carries
   R27.195's size-check fix), tests and a debug smoke; the protocol freezes at that merge.
4. The gated stage in a run tree at the frozen revision; evidence in
   `~/develop/emai/evidence/task095-*`. Its rows only route to further rulings; none is a
   controller.

### D, a Phase 4 precursor (world-model reach plus a scripted closure rule; about 10–15 GPU h)

Not admitted: R26.33 applies, because the planner would no longer command the grasp, and its
condition (evidence of a different kind) is not met. The owner would have to rule it in. R27.204
judged its chance low, since it inherits the 2.6 cm held error.

### X, back to the v2 Apple→Plate LeWM line

Every gated v2 LeWM success is under C1-M, which the owner stopped after TASK-083. Reopening it is
the owner's call. A new task needs fresh seeds and its own preregistration; R18.32 recommends
against repeating TASK-080's Stage S design.

### A hybrid controller or a sensing change

Two shapes, which can combine; they fall under different rulings:

- **Hybrid:** LeWM chooses the goal, approach and placement; a local grasp controller (scripted
  or learned, possibly with contact feedback) closes the last centimetre. TASK-081's structure,
  where LeWM chooses one aim and other components execute, is the precedent. Over TASK-087's
  checkpoints this is a new grasp controller: R26.33 and R27.157 apply, as for D, and its
  admissibility needs its own ruling citing evidence of a different kind.
- **Sensing (W in R27.204; open under R27.71; about 30–60 GPU h plus collection, an estimate):**
  a wrist camera on the Dex3 hand, a higher-resolution camera, or contact signals. The current
  scenes have no wrist camera; one has to be added to the MJCF and its frames checked before any
  corpus is collected.

First step: a design note that states which centimetre the new piece must resolve, measured
against TASK-094's K-style tolerance envelope, before anything is trained.

## Branching to SO-101

**Opened on 2026-10-16** ([DECISIONS (bp)](DECISIONS.md), R28.1–R28.9). The plan, with the
chosen defaults and phases S0–S6, is [SO101_PLAN.md](SO101_PLAN.md); it supersedes the outline
and the open decisions below, which are kept as written at the pause.

The LeRobot SO-101 is a low-cost arm with six Feetech STS3215 servos: five arm joints
(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`) and a jaw gripper
(per `so_follower.py` in the pinned LeRobot v0.4.4 checkout). Wrist cameras are a common mount in
LeRobot setups. The expectation, not measured here, is that its jaw gripper is more forgiving of
centimetre placement errors than a dexterous hand, and that the hardware is cheap enough to own,
so that real data and real closed loops become possible. It is **a change of the product goal** (LeWM on G1 + dual Dex3), so it needs an owner
ruling, its own plan and its own first preregistration (R27.213). Its results would transfer to
G1 + Dex3 only in part.

### What already supports it

- **Registries.** `EMBODIMENTS`, `TASKS`, `MODELS` and `PLANNERS` in `config.py` hold
  `"module:attr"` strings, so an `so101` embodiment and its tasks can be registered next to
  `unitree_g1_dex3`.
- **The `Embodiment` and `Transport` protocols** (`contracts.py`) already separate the robot (IK,
  limits, scaling, frames) from the model and planner. The SO-101 adapter implements
  `observe`, `state`, `execute`, `normalize_action`, `denormalize_action` and `stop`.
- **The backend-swap invariant.** Both world-model backends, the CEM planner (normalized actions
  only, no backend branches) and the evaluator are robot-agnostic in design intent, but **not yet
  in code**: they assume the 14-D `ee_delta_grasp_v0` (see item 1 below).
- **Data.** `data.py` stores LeRobot v3 datasets, and the pinned LeRobot checkout reads and
  records SO-101 datasets. `real_g1_convert.py` (TASK-086) is a precedent for converting external
  LeRobot data into our profile.
- **Frozen DINOv2 features, LeWM token predictors and the preregistration machinery**
  (`run_tools`, seed checks, run trees, evidence folders) carry over unchanged.

### What must be built

1. **An action schema.** `ActionSchema` accepts only `ee_delta_grasp_v0` (14-D, two hands) or an
   experimental `joint_delta_*` schema. A 5-joint arm cannot follow a full 6-D end-effector
   delta, so the choice is a joint-delta schema (`joint_delta_so101_v0`: five joint deltas plus a
   jaw command) or a new reduced end-effector schema, which needs a contract change and a
   versioned migration. LeRobot records absolute joint-position targets, so either needs a
   conversion. The core code assumes 14-D actions and must take the action dimension from the
   schema: `contracts.py`, `config.py` (lines 201–206 reject any schema other than
   `ee_delta_grasp_v0`), `planning.py` (`ACTION_DIM`), `data.py` and `policy.py`, plus literal
   14s in `models/native.py` (`dim + 14`), `models/lewm.py` (`input_dim=14`),
   `models/frozen_encoder.py` and `models/latent_critic.py`. Of the 13 files that match
   `ACTION_DIM|ee_delta_grasp_v0`, the rest are G1-specific task-line modules, which stay as
   they are.
2. **An embodiment adapter** with the arm's limits, scales and frames in its own config (like
   `configs/g1_sim_action.json`), a MuJoCo transport, and later a LeRobot-backed real transport,
   disabled by default.
3. **A MuJoCo scene** from an upstream model: MuJoCo Menagerie's `trs_so_arm100` (SO-100), or the
   SO-101 URDF/MJCF in TheRobotStudio's SO-ARM100 repository. A fetch script must pin the exact
   revision and record the licence in [DEPENDENCIES.md](DEPENDENCIES.md); neither has been
   checked here. Add an overview and a wrist camera, a table and objects.
4. **A task:** resets, goals, termination and scoring (for example cube pick-and-place), with
   simulator truth used only for scoring.
5. **A scripted collector and a play corpus,** like `play_corpus.py`, measured before training.
6. **Training:** LeWM (and `native_jepa`) on the corpus with the existing recipe, with offline
   gates for collapse, action sensitivity and copy-last baselines.
7. **A first preregistration,** for example reach and then pick in simulation, with the scripted
   follower as the reference arm, before any real-hardware step.

### Open owner decisions

- Whether SO-101 replaces the G1 + Dex3 goal, runs beside it, or is a stepping stone back to it.
- Simulation only, or buy hardware (a follower arm, and a leader arm for teleoperated
  demonstrations), and when real-robot execution is allowed.
- The action schema (joint deltas or a reduced end-effector schema) and the cameras (wrist,
  overview, resolution).
- The first task and its success definition.
- Whether to start from community SO-101 datasets on the Hugging Face Hub, which needs licence
  checks per dataset.

### Rough effort (an estimate, not a measurement)

- Schema, adapter, scene, task and tests: roughly 1–2 weeks of agent-assisted work.
- Scripted collector and a corpus of a few thousand episodes: writing and tuning the collector
  dominates; the collection itself is likely short. For scale, TASK-085's `play-v1` (3 194
  episodes, 16.06 h) took 1 025 s of wall time with 12 workers and EGL rendering on the GPU, but
  the SO-101 scene is new and untimed.
- Training: TASK-087's Stage T (18 training jobs) was projected at 9.3 GPU h and used about 10 GPU
  h; a comparable first
  SO-101 study is plausibly 10–20 GPU h.
- First gated simulation result: plausibly 3–6 weeks from the ruling, with real hardware
  adding procurement, calibration and safety commissioning on top.

## What did not change

- R7's canonical sentence. The frozen v1 benchmark and its 0/150 per backend.
- Every fired clause (TASK-054, TASK-057, TASK-062, TASK-065, TASK-075, R26.33, R27.71, R27.149).
- The product goal, until the owner rules otherwise.
