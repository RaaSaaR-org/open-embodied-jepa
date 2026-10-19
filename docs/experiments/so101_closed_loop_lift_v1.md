# TASK-101 — SO-101 S5: a first closed loop on `so101_cube_lift_v1` (amended from `so101_cube_lift_v0` by A1) in simulation (CEM over TASK-100's `native_jepa` token model J, toward image subgoals from a per-reset demonstration)

**Status: DRAFT preregistration (docs and card only; nothing has run).** Opened by Claude under
owner delegation ([DECISIONS](../DECISIONS.md) 2026-10-19 (bt), R28.41–R28.58; the design choices
are R28.43–R28.56 in §17). Freeze path: independent review, then merge of this DRAFT, then a
Stage 0 PR (code in new files, development closed loops, calibration of the planner and of the
bar on development seeds only), independently reviewed. **The protocol freezes when the Stage 0
PR merges.** No gated reset is run before then.

**Stage 0′ (2026-10-19, [record](so101_closed_loop_lift_v1_stage0b.md), [DECISIONS](../DECISIONS.md)
(bw), R28.76–R28.82): row S0-NULL.** Every §7.2 check passed before any count, K′ scored 0/32 in all
four configurations, D′ did not run, and the diagnostic Δ ([record](so101_closed_loop_lift_v1_null_diagnostic.md))
read Δ-MIXED. Escalated for a ruling. **Not frozen** (R28.81): there is no gated stage under
S0-NULL, and nothing may run on S.

**Amendment A2 (2026-10-19, [DECISIONS](../DECISIONS.md) (bx), R28.83–R28.89; decided by Claude
under owner delegation).** The ruling on Δ-MIXED: the 0.04 rad switching tolerance was declared,
not calibrated, and Δ puts J's cost minimum about 0.05 rad off the subgoal pose. A2 measures the
task's pose-error envelope on the privileged follower on fresh seeds, sets the switching tolerance
τ_s by a rule fixed before the measurement, and reruns the calibration on fresh cohorts K″ and D″
(Stage 0″). Not frozen. §19 holds the amendment.

**Amendment A1 (2026-10-19, [DECISIONS](../DECISIONS.md) (bv), R28.65–R28.75; decided by Claude
under owner delegation).** Stage 0 ended S0-VOID ([record](so101_closed_loop_lift_v1_stage0.md),
R28.59): the lift scene's inert target marker lay outside the play corpus's marker range. A1 moves
the study to **`so101_cube_lift_v1`**, which is v0 with the marker drawn per reset from the corpus's
marker distribution. It runs Stage 0′ on K′ and D′, with every §7.2 check enforced before any count
is read, and declares a diagnostic if K′ is null. **The protocol freezes when the Stage 0′ PR
merges.** §18 holds the amendment and lists every section it changes; the changed sections below
point to it.

This is phase S5 of [SO101_PLAN.md](../SO101_PLAN.md) ("first preregistered closed-loop study in
simulation"), admitted by TASK-100's row WM-PASS with arm J only
([results](so101_world_model_v1_results.md); R28.34–R28.40). **It is a simulation-only control
study on the SO-101 cube-lift task. It is not an Apple→Plate experiment and no row of it changes
R7** (R28.9). The controller's model is `native_jepa` (J), not LeWM, because J is the arm that
passed S4 (R28.35); whether LeWM's token predictor passes S4 at a saturating budget is a separate,
recommended and unstarted task (R28.57).

Closest precedents, and what is taken from each:

- **TASK-088** ([zero_shot_reach_grasp_v1.md](zero_shot_reach_grasp_v1.md)): per-reset goals made
  by a privileged demonstrator that is never a counted arm; four image subgoals for a
  grasp-and-lift (end of hover, descent, close and lift); subgoal switching on proprioception and
  the goal pose only; CEM replanned every command; a non-learned follower given the same goal
  information as the "needed" comparator; the bar capped at 0.9 × the follower's calibration rate
  (TASK-074's lesson). Its world-model grasp arms scored 0/64 against the follower's 50/64.
- **TASK-089** ([zero_shot_grasp_v1.md](zero_shot_grasp_v1.md)): the step-averaged cost (the mean
  of the per-step cost over the horizon, so the first commands, and the absolute gripper channel
  in particular, count). It is one of the two cost forms the calibration chooses between (§5.4).
  Its closed loop also lifted 0/64.
- **TASK-092 and TASK-093** ([grasp_rate_aware_wm_v1.md](grasp_rate_aware_wm_v1.md),
  [grasp_rate_projected_planner_v1.md](grasp_rate_projected_planner_v1.md)): the rate lesson. On G1
  the planner scored grasp requests the embodiment could not execute at the planned rate.
  Here the model was trained on the executed, already clipped actions (TASK-099 stores
  `applied_action`), so the planner scores each candidate as the embodiment will execute it
  (§5.2). That is a consistency requirement, not an experimental variable.
- **TASK-080, TASK-081 and TASK-083** ([apple_lewm_commit_precision_v2.md](apple_lewm_commit_precision_v2.md)):
  the row ladder (a bar, exact one-sided McNemar tests against an action-blind twin and
  scene-blind controls at p < 0.01, non-inferiority against a non-learned arm with the allocated
  margin δ = 16/128), 128 fresh gated resets, and every claim restated with its qualifiers.
- **[RESUME.md](../RESUME.md) and the plan's lessons:** bars from measured ceilings, not carried
  over (TASK-074's O2); measure before gating (a development closed loop calibrates before the
  freeze); a scripted reference arm in every closed loop; budgets with a hard cap.

## 0. What S4 handed over, and what it constrains

From TASK-100's gated record (offline, one run, simulation, a privileged scripted corpus):

- **J passed G1–G6 on all three seeds at h = 8 and 16** on `so101-play-v1`'s test split: no
  collapse, below copy-last (ratio 0.74 at h = 8, 0.66 at h = 16) and below its no-action twin
  (0.93, 0.90), action-sensitive (sensitivity 1.17–1.26, lower bounds 1.16–1.25, against 1.10), and the cube and jaw are
  readable from its predictions within the relative bars. Its budget saturated (kept checkpoints at
  14 000–16 000 of 20 000 updates).
- **The predicted cube is read to about 1.6–1.8 cm in median**, and only 3.4–5.4 % of test roots
  read within S2's 0.5 cm grasp-offset tolerance. R28.39: a controller should not take its grasp
  point from this readout alone. **This controller reads no readout at all** (§5): it compares
  predicted latents with encoded goal frames.
- **§9's planning measure** (offline, against the true future frame, which a planner does not
  have): J's latent cost ranks the executed sequence first among 16 candidates on 10–15 % of roots
  at h = 8 and 16 (chance 6.25 %) with normalised rank 0.24–0.27 (chance 0.5). It separates
  other-episode and σ 0.3 candidates far better than σ 0.1 ones. **J ranks coarse command
  differences, not fine ones.** That is the main risk to a grasp, whose measured tolerance (S2,
  follower v1) is 5 mm toward the fixed jaw, 10 mm along the face or upward and 20 mm away from
  the fixed jaw (§13).
- **The corpus's test split has been read.** Nothing here is evaluated on corpus episodes; every
  gated reset is a fresh draw from the S2 task stream (§2), so the closed loop does not depend on
  the corpus's held-out status.

## 1. Question and claim

**Question.** In simulation, does MPC by CEM over TASK-100's J model lift the cube on
`so101_cube_lift_v0`, unchanged (since A1, on `so101_cube_lift_v1`: v0 with the marker drawn
inside the corpus's range; §18), on fresh resets, when the controller sees only the two camera
frames, its own joint state and image subgoals of a demonstration of the same reset? And does its
success depend on the actions, the scene and the goal, as it should?

**The primary claim**, made only by row CL-PASS (§9): **"J-driven closed-loop lift success toward
image subgoals of a privileged per-reset demonstration"** on `so101_cube_lift_v0` (A1: `v1`) in simulation,
with every qualifier of §14. The goal source is part of the claim's name and is never dropped when
the claim is restated. **Since A1 (§18.4) the claim's name ends "…, on `so101_cube_lift_v1` (the lift
scene with its inert target marker drawn inside the play corpus's marker range)"**, and that part is
not dropped either. It is not "LeWM-driven", it is not
"J needed" (the privileged follower and a non-learned pose follower given the same goal pose are
reported beside it), and it is not a placement claim.

## 2. Task, scene and scorer (amended by A1: the marker)

**A1 (§18.2):** the task is **`so101_cube_lift_v1`**: everything below, plus the inert target
marker drawn per reset by the play corpus's rule (x ∈ [0.16, 0.22] m, |y| ≤ 0.14 m, at least 8 cm
from the cube) from `default_rng([10107, seed])`, in place of v0's fixed x = 0.30 m, y = −0.20 m.
The cube draws, the scorer and everything else are as written here.


`so101_cube_lift_v0` exactly as TASK-098 declared it ([so101_cube_tasks_dev.md](so101_cube_tasks_dev.md);
`so101_tasks.py`, `LIFT_V0`, `RESETS_V0`): cube xy uniform on x ∈ [0.18, 0.23] m,
y ∈ [−0.10, 0.10] m, yaw uniform on ±45°, from `default_rng([9801, seed])`; arm at `home_rad`.
Success: the cube centre ≥ 5 cm above its resting height with a gripper contact on every step of
20 consecutive control steps; lost below z = −0.05 m or off the table; **at most 200 control steps
(10 s at 20 Hz)**. Held-window scorer on simulator truth; no latching. The scene, the cameras
(`wrist_rgb`, `overview_rgb`, 224 × 224), `configs/so101_sim_action.json` and the
`SO101Embodiment` are unchanged. A `stopped` execution (for example the 8 rad/s measured-velocity
guard) ends the episode as a failure and is reported; it is not a VOID.

## 3. The model

- **J**, TASK-100's `native_jepa` residual token step over frozen DINOv2 ViT-S/14 tokens of both
  cameras, grid g4 (32 tokens × 384), as frozen at `f0da5bf` and trained in TASK-100's gated stage
  T. The kept checkpoints are used unchanged: `checkpoints/task100-run/J-<seed>/model.pt`, SHA-256
  `41fa52ad…cd16` (100100), `0e04d495…0061` (100101) and `0ae2b6cc…f3c7` (100102), as recorded in
  the val report (`b3c3edde…aa31`). Each load checks the file hash and the model's implementation
  SHA-256 (`78ba9dbe00eb…`, TASK-100's compute table); a mismatch is a VOID. No model is trained,
  fine-tuned or re-selected.
- **The primary seed, by a rule fixed now:** the J seed whose kept checkpoint has the **lowest val
  criterion** (`training.kept_criterion` in its TASK-100 job report, the value checkpoint selection
  used, written before the test split was featurised). The values are 0.53142 (100100), **0.53021
  (100101)** and 0.53383 (100102), so the **primary model is J-100101**; J-100100 and J-100102 are
  replication arms, reported only. *Disclosed:* this rule is written after TASK-100's test numbers
  were published. It reads no test number, and all three seeds passed every gate with close values.
  *Also disclosed:* J-100101 happens to have the best test values among the J seeds on G2, G3 and
  G4, by small margins (h = 8 copy-last ratio 0.740 against 0.741 and 0.744; h = 16 no-action
  ratio 0.899 against 0.903 and 0.909; h = 16 action sensitivity 1.264 against 1.253 and 1.236).
  The rule does not read these values and would pick the same seed without them; the coincidence
  is stated so that the post-hoc disclosure is complete.
- **The action-blind twin N** is N_J-100101 (`a8fe58a4…0c3c`), trained in TASK-100 on the same
  windows with zero commands; as in TASK-100 it receives zero commands for every candidate.
- **The live latent** is TASK-100's pipeline, unchanged: each camera's 224 px frame through the
  pinned frozen DINOv2 (weights checked against their pins), final patch tokens average-pooled to
  4 × 4, wrist tokens then overview tokens, **rounded to float16 as the feature store was**, then
  standardised with the frozen fit-sample moments of TASK-100's pretrained feature store (fit
  report hash checked). Stage 0 checks that this live path reproduces the stored latents (§7.2).
- **Distance:** TASK-100's latent error, the mean squared difference over the 12 288 standardised
  values.
- The model reads no proprioception (J has no state input). Only the subgoal switching rule (§5.3)
  and the non-learned pose follower read joint states.

## 4. Goals: a per-reset demonstration, image subgoals

**How goals are made.** For each reset seed, before any arm runs on it, the task's **goal
demonstration** runs the privileged scripted follower v1 (`so101_followers.follow_lift`,
`FOLLOWER_V1`, no offset) on the same declared reset in its own `SO101Simulation` instance with
rendering on. Its four **subgoals** are the state after demonstration commands **40, 70, 90 and
120**, which by `FOLLOWER_V1`'s schedule are the ends of its pre-grasp (40 commands, the jaw open
6 cm above the grasp point), descent (30, the jaw open at the grasp point), close (20) and lift (30,
the cube about 8 cm up) phases. For each subgoal the goal store keeps both camera frames (uint8,
hashed), their encoded standardised latent (§3) and the demonstration's measured joint positions
(5 arm joints and the jaw). The goal store of a cohort is written by one job before any arm runs,
hashed, and checked by every arm's runner, so every arm sees the same goals (paired).

- **Rejection and replacement.** The reset is rejected if the demonstration does not succeed by
  the task's scorer, or stops, **or terminates (success, lost cube or timeout) or stops before
  command 120 has executed**, so that subgoal 4 would not exist. (In S2 follower v1's success came
  at step 121, a margin of one step.) A rejected reset is replaced by the next unused seed of the
  cohort's reserve range (§12), recorded with the reason; every arm then runs the replaced cohort.
  Follower v1 lifted on 64/64 development seeds (S2), so replacement is expected to be rare. Its
  count per cohort is reported, and on D it is the measured ceiling check of §7.6 (S0-CEILING).
- **Why demonstration goals.** The lift task's goal is a state of the cube ("held 5 cm up"), not an
  image. An image-goal planner needs a concrete instance of that state for this reset, and an
  image-goal benchmark supplies it the same way: a goal observation from a demonstration of the
  same scene (TASK-020's frozen goals, TASK-088's demonstrator; image-goal world-model planning
  evaluations commonly take their goals from recorded trajectories of the same environment). A goal without the
  reset's arm pose, such as a generic "a lifted cube" image, would ask the planner to localise and
  grasp the cube from a goal that does not show where. That is a harder, later question.
- **Why four subgoals, not one.** The follower needs about 120 commands, while the planner looks
  16 at most ahead (J's longest gated horizon). A single final goal would ask a 16-step search to
  find a grasp that is 100 steps away; TASK-020's single image-goal CEM scored 0/150 per backend.
  The phase ends are the follower's own, so they are fixed by the follower's schedule, not chosen.
  A single-goal variant is a development diagnostic only (§7.4), not an arm.

**What each controller arm reads.** The current `wrist_rgb` and `overview_rgb` frames; its own
measured joint positions and its own joint-position targets (the command history, which it sent);
and the goal: the subgoal latents and the subgoal joint positions. **No controller and no planning
cost reads object truth.** Object truth is read by the goal demonstration (to make goals), by the
scorer, and by the privileged follower arm F. The runner logs task truth at every step for the
reported measures (§10); the controller object never receives it, and a test checks that.

**What the goal gives away (fixed now).** A subgoal shows the arm at the demonstration's grasp
pose, and the subgoal joint positions put the fingertips there. A success therefore shows that the
planner drove the arm and jaw to match a demonstration of this reset closely enough to lift. It
does not show that the planner localised the cube on its own. The pose follower H-pose (§6) uses
the same joint positions without images or a model. It measures how much of the task the goal
pose alone solves, and it is the comparator for "J needed".

## 5. The controller W

### 5.1 Planner

`schema_planning.SchemaCEMPlanner`, **unchanged**, over `joint_delta_so101_v0`: bounds [−1, 1] on
all six channels; **256 samples, 6 iterations, 32 elites**, minimum std 0.05, action penalty 0;
replanned from scratch every command (MPC), the first action of the best candidate executed. One
planner per episode, seeded `10101 · 10⁶ + reset seed` (salt 10101), the same seed for every CEM
arm on a reset. The sample, iteration and elite counts are fixed now to keep the gated stage inside
its budget (§13). The calibration (§5.4) chooses the horizon and the cost form only.

### 5.2 Planning on what is executable (projection)

`project_candidates` is on. The projector is the embodiment's own `SO101Embodiment.preview`,
chained over the horizon: starting from the current joint-position targets, each candidate step is
replaced by the applied action that `preview` returns (arm targets clipped to the joint limits, the
gripper target moved at most 0.25 rad a step), and the next step starts from the resulting
targets. Every candidate is feasible. The model is rolled out on the projected (executed)
candidates, and CEM's distribution update uses the requested ones (the planner's existing
behaviour). Why: the corpus stores executed actions (26.4 % of them clipped), so J has only ever
seen executable commands. A requested gripper jump from +1 to −1 in one step would be out of its
training distribution. The projector reads only the embodiment's own targets and limits, no sensor
beyond the measured state and no object truth (`constraints.CandidateProjection`'s contract). A
Stage 0 test checks that the chained preview equals the applied actions `execute` returns on
random sequences.

### 5.3 Cost and subgoal switching

- **Cost.** For candidate sequence a over horizon H, the predicted latents ẑ₁ … ẑ_H from the
  encoded current frame, and the current subgoal's latent g: either **last**, d(ẑ_H, g), or
  **mean**, (1/H) Σₕ d(ẑₕ, g), with d TASK-100's latent error. The mean form is a wrapper that
  hands `SchemaCEMPlanner` the running mean in place of the per-step distance (TASK-089's
  wrapper; the planner reads the last step), so the planner is not edited.
- **Switching (every arm that uses subgoals, alike).** The controller works on subgoal k
  (k = 1 … 4) and moves to k + 1 when, for **3 consecutive commands**, the measured arm joints are
  within **0.04 rad RMS** (over the five arm joints) of subgoal k's and the measured jaw is within
  **0.15 rad** of subgoal k's; or after **T_k = 60, 45 and 30 commands** on subgoals 1, 2 and 3.
  Subgoal 4 (lifted) is kept to the end of the episode. The timeouts are 1.5 × the demonstration's
  phase lengths (40, 30, 20); their sum, 135, leaves 65 of the task's 200 commands for the lift
  (the demonstration's takes 30) and the 20-step hold. The tolerances are not precision
  requirements: the image cost on each subgoal drives the precision, and the rule only decides when
  to move on. 0.04 rad RMS over five joints at the arm's 0.2–0.3 m reach is of the order of a
  centimetre of fingertip motion. The jaw tolerance tells a jaw stalled on the cube (the
  demonstration's closed jaw stalls at about 0.16 rad) from one closed on nothing (−0.17 rad).
  The rule reads proprioception and the goal pose only (TASK-088 §5.3's form, with this robot's
  units).

### 5.4 Calibration grid (Stage 0, on calibration seeds only)

Four configurations: **H ∈ {8, 16}** (TASK-100's gated horizons, §6 of its protocol: h = 8 spans
one full jaw closure, h = 16 a closure plus the start of a lift) × **cost ∈ {last, mean}**. The
rule is in §7.3.

## 6. Arms (paired: the same resets, the same goal store)

| Arm | What acts | Role |
|---|---|---|
| **W** | CEM over **J-100101**, §5's controller at the calibrated configuration | **primary** |
| W-100100, W-100102 | the same with J-100100 / J-100102 | replication, reported only |
| **N** | the same planner over **N_J-100101** (zero commands to the model for every candidate) | action-blind twin |
| **W-shuf** | W with the whole goal (subgoal latents and joints) of another reset of the same cohort, by a fixed derangement (salt 10103); under A1 the borrowed goal frames show the other reset's marker position (§18.4) | goal-blind control |
| **W-mean** | W with the current-frame latent replaced at every planning call by the fit-sample mean latent (zero in the standardised space); goals unchanged | scene-blind control |
| **rand** | every channel uniform on [−1, 1], fresh every command (salt 10102); the embodiment clips as usual | floor |
| **hold** | `SO101Embodiment.hold_action()` every command | floor |
| **F** | the privileged scripted follower v1 (`follow_lift`, no offset); reads simulator truth | determinism guard on the goal pipeline (not learned; scores every kept reset by construction, §6) |
| H-pose | non-learned pose follower: each command moves the arm targets toward subgoal k's measured arm joints with the follower's own law, `clip((q_k − target) / scale, ±1)`, and commands the jaw +1 if subgoal k's jaw angle is above 0.785 rad (mid-range), else −1; §5.3's switching; no image, no model | "J needed" comparator, reported only |

- **N is a weak control by construction, disclosed now.** N's predictions do not depend on the
  candidate, so every candidate ties; `SchemaCEMPlanner` then keeps its initial mean, and N
  executes the zero command (arm held, jaw half open) at every step. This is what an action-blind
  model can do with this planner. W-shuf and W-mean are the informative controls, and W > N adds
  little beyond W > hold. **Exact ties are made by construction, not left to the GPU:** because
  every candidate gives N the same inputs (the same encoded frame and zero commands), N's adapter
  rolls the model out once per planning call and broadcasts that roll-out to every candidate, so
  the costs are bit-identical whatever the GPU's batched arithmetic does. A Stage 0 test (§7.1)
  checks the ties on real CUDA outputs. If the ties were nevertheless broken (a defect), N would
  still be action-blind, since its costs carry no information about the commands, but it would
  execute an argmin over float noise rather than the zero command; the Stage 0 record states which
  N ran.
- **F is not a measured ceiling on the kept resets, by construction.** F runs the same
  deterministic program as the goal demonstration, and §4 replaces every reset whose
  demonstration fails, so on the kept resets k_F equals the number kept and r_F = 1 by
  construction. F is run and counted anyway, as a determinism check of the goal pipeline: **F's
  executed actions or outcome differing from the goal demonstration's on any kept reset is a
  guard failure** (S0-VOID in Stage 0, VOID in S; §12), never a ceiling finding. The measured
  ceiling quantity is the demonstration's rejection count on the drawn seeds (§4, §7.6). The
  measured non-trivial reference is **H-pose**, a non-learned follower given the same goal
  information (TASK-088's analogue); its D count is reported beside the bar in the Stage 0 record,
  but it does not enter the bar.
- **No hybrid arm.** A world model for the arm with a hand-written jaw rule, or a world-model
  approach with a scripted grasp, was considered and is not an arm here. The first SO-101 closed
  loop should say what the model-driven controller does alone. A hybrid becomes a candidate only
  after a decomposition record (§9) names the failing phase, and needs its own preregistration.

## 7. Stage 0 (development; before the freeze)

Stage 0 runs on **calibration seeds K** (101010–101041, reserve 101042–101059) and **development
seeds D** (101100–101163, reserve 101164–101199) only, on the fresh replacement cohorts K′ and D′
of §7.6 if a rule there calls for them, and on debug seeds 101900–101999 for plumbing (never
counted). No gated seed is drawn, rendered or run before the freeze. **A1:** Stage 0's K and D are
spent. Stage 0′ runs K′ and D′ on `so101_cube_lift_v1` under §18.5's ordering rule, and the
diagnostic cohort X (101700–101715) is used only by §18.6.

### 7.1 Code and tests (new files only)

New opt-in modules (no registry entry; `import embodied_jepa` unchanged and torch-free, checked by
a test): `so101_closed_loop.py` (NumPy: cohorts, seeds, salts, the goal-store format and hashes,
the switching rule, the derangement, the non-learned arms, statistics, the bar rule and the rows)
and `so101_closed_loop_runtime.py` (torch: the live featuriser, the J/N planner-model adapter with
the last and mean cost forms and the scene-blind substitution, the chained-preview projector, the
episode loop and the worker pool); scripts `run_task101.py` (subcommands `goals`, `episodes`,
`check`), `summarize_task101.py` and `run_task101_stage.sh` (one `scripts/gpu_run.sh --wait
--board` lock per GPU job). `schema_planning.py`, `so101.py`, `so101_simulation.py`,
`so101_tasks.py`, `so101_followers.py`, every TASK-100 file pinned by
`benchmarks/manifests/so101-world-model-v1.json` and every G1 byte-pinned file stay byte-identical;
`config.py` is not changed (it refuses an `so101` experiment config; the run configuration is a
module constant, TASK-066's and TASK-100's precedent). A new manifest
`benchmarks/manifests/so101-closed-loop-lift-v1.json` pins this task's files at the freeze.

Tests: the switching rule in both directions and its timeouts; the derangement has no fixed point
and is fixed by its salt; the chained projector equals `execute`'s applied actions on random
sequences; the mean-cost wrapper; ties in N's planner return the initial mean, checked on real
CUDA outputs of N_J-100101 (bit-identical costs across the batch) as well as on synthetic costs;
F's executed actions equal the goal demonstration's on the debug seeds; the controller
object never receives task truth (a sentinel); seeds outside a cohort are refused; a cohort cannot
be run twice into the same output; checkpoints and goal stores are refused on a hash mismatch; the
bar rule, the McNemar tests, the bootstrap and every row in both directions on synthetic counts;
the core-import check; TASK-100's `frozen_files_check` and the G1 frozen-file checks still pass.
**A1 adds** the new module `so101_tasks_lift_v1.py` (no registry entry; `config.py` stays
byte-identical) with its tests, and the check-ledger tests of §18.5's ordering rule.

### 7.2 Checks (each failure is S0-VOID until fixed under its own record)

**A1 (§18.5):** 7.2.0 (scene comparison) and 7.2.6 (the marker is inert) are added. 7.2.2 runs on
each cohort's own goal store before that cohort's episodes, and 7.2.3 runs on K′'s reset states.
7.2.4 is read in the conditional form written there. **Every check runs before any count of a
cohort is read, and the runner refuses otherwise (the ordering rule).**

1. **Live latent = stored latent.** 64 stored frames per camera from 8 val episodes of
   `so101-play-v1` are encoded by the live path; every float16-rounded raw value must equal the
   stored float16 value or differ from it by at most 2 float16 ulps at its magnitude. (Val frames,
   because Stage F of TASK-100 featurised them; reading them here selects nothing.)
2. **The live latent reads the scene.** TASK-100's pretrained R-enc readout (`renc_pretrained.npz`,
   fitted on every fifth fit-sample frame in TASK-100's readouts pass; file hash checked) applied to the live-encoded frames of D's goal demonstrations reads
   the cube within B_cube = 2.397 cm in median (R-enc's test median was 1.39 cm). A miss means a
   pipeline or rendering mismatch, not a model result.
3. **Projector:** the test of §7.1 on 1 000 random 16-step sequences from D's first frames.
4. **Determinism:** one W episode and one N episode, each run twice, give identical executed
   action sequences and identical outcomes; and the same four W episodes run in a pool of 1
   worker and in a pool of the size used for K, D and S give identical executed action sequences
   and outcomes (pool-size independence, §13).
5. **Timing:** median and 95th-percentile planning seconds per command for each configuration of
   §5.4, the worker count used, and the projection of K, D and S (§13). Every command must finish
   inside the embodiment's 30 s wall-clock deadline (`max_wall_snapshot_age_s`).

### 7.3 K: choosing the horizon and the cost form

*A1: applied to K′ (101060–101091) on `so101_cube_lift_v1`; K's counts were read under the void
condition and set nothing (§18.5).*

W (J-100101) runs all four configurations on the 32 K resets. **The configuration with the most
successes is chosen**; ties go to **mean** over last (TASK-089's reason: the first commands, which
are the ones executed, count), then to **H = 8** over H = 16 (half the compute). A configuration
whose projected Stage S cost (§13) does not fit the cap is not eligible. If every configuration
scores 0/32, the row is S0-NULL (§7.6). K's numbers are development numbers; K seeds are never
used again.

### 7.4 D: the development closed loop

*A1: applied to D′ (101600–101663); §7.6 row 1's clause is not waived (§18.5). The bar (§7.5) reads
W(D′).*

Every arm of §6 except the replication seeds runs on the 64 D resets at the chosen configuration:
W, N, W-shuf, W-mean, rand, hold, F and H-pose. Reported in the Stage 0 record with exact 95 %
intervals, per-phase decomposition (§10) and timing. **Development diagnostic only (not an arm,
read by nothing):** W with the single final subgoal (4) from the first command, on the first 16 D
resets, to record what the phasing contributes. D changes no knob of §5 by itself. When a fix
may be rerun on the same seeds and when it needs fresh ones is fixed in §7.6, row 1.

### 7.5 The bar (rule fixed now, applied to D's numbers)

**b = min(0.9 × r_F(D), max(0.50, LCB(W(D))))**, and the gated bar is **⌈b × 128⌉ successes of
128**, where r_F(D) is F's success rate on D's kept resets and LCB(W(D)) the lower end of the
exact (Clopper–Pearson) 95 % interval of W's D count out of 64. **Because F replays the goal
demonstration and failed demonstrations are replaced (§4, §6), r_F(D) = 1 by construction, so the
cap is the constant 0.90, and the binding calibration is LCB(W(D)) between the 0.50 floor and
that cap.** F does not measure a ceiling here; the formula keeps TASK-088's form only so that the
cap is written down.

- **0.9 × r_F(D) = 0.90, the cap:** no bar may ask more than 90 % of what the privileged follower
  achieves on the same goals (TASK-074's lesson; TASK-088's form). On the kept resets that is
  0.90; the measured check on the follower is S0-CEILING's rejection count (§7.6).
- **0.50, the claim floor:** "closed-loop lift success" means at least half of the resets (TASK-088
  used half as its stop level). A lower rate is reported as what it is (CL-LOW), never as success.
- **LCB(W(D)), the development level:** between the floor and the cap, the gated stage asks W to
  confirm on fresh resets the level its development run showed, at the conservative end of that
  run's interval, so that a correct development reading passes with high probability.

What the rule gives (exact binomial; power is P(X ≥ bar) at a true rate p):

| W(D) | LCB | b | bar of 128 | power at p = W(D) | power at p = W(D) − 0.05 |
|---|---|---|---|---|---|
| 34/64 | 0.40 | 0.50 | 64 | 0.79 | 0.37 |
| 36/64 | 0.43 | 0.50 | 64 | 0.94 | 0.65 |
| 40/64 | 0.50 | 0.50 | 64 | > 0.99 | 0.96 |
| 48/64 | 0.63 | 0.63 | 81 | > 0.99 | 0.96 |
| 56/64 | 0.77 | 0.77 | 99 | > 0.99 | 0.95 |
| 60/64 | 0.85 | 0.85 | 109 | > 0.99 | 0.92 |
| 62/64 | 0.89 | 0.89 | 115 | > 0.99 | 0.84 |
| 64/64 | 0.94 | 0.90 | 116 | > 0.99 | 0.99 |

(r_F(D) = 1, as it is by construction; computed for this draft, recomputed in the Stage 0 record.)
The last column is not monotone in W(D) (0.84 at 62/64, 0.99 at 64/64). That is not an error: the
0.90 cap binds only at 64/64, where it lowers the bar from ⌈0.94 × 128⌉ to 116 while the
evaluated rate stays 0.95.

### 7.6 Stage 0 rows (first match) and the freeze

1. **S0-VOID** — a check of §7.2 failed, a guard of §12 failed (including F diverging from the
   goal demonstration), or a run did not complete. Fixed under its own record. **Same-seed reruns
   are allowed only for a VOID raised before any success or failure count of the affected cohort
   (K or D) was read**: a §7.2 check, a crash, a guard. **Once K's counts have been read, any fix
   that changes code or configuration is rerun on the fresh calibration cohort K′ (101060–101091,
   reserve 101092–101099); once D's counts have been read, on the fresh development cohort D′
   (101600–101663, reserve 101664–101699).** K and D set the configuration and the bar, so a fix
   made after their counts were seen and rerun on the same seeds would be a knob turned on seen
   seeds. K′ and D′ are used at most once each; a second post-count fix needs a ruling that names
   further seeds. The Stage 0 record states which cohorts set the configuration and the bar.
2. **S0-CEILING** — the goal demonstration was rejected (§4) on **8 or more** of the D seeds drawn
   to keep 64 (its rate on the drawn seeds, 64 / (64 + rejections), below 0.9). This is the
   measured follower check, since F itself scores every kept reset by construction (§6): the
   scene, the goal procedure or the task is in doubt, not the model. Escalate to a ruling; no
   gated stage. (F diverging from the demonstration is not this row but S0-VOID.)
3. **S0-NULL** — every K configuration scored 0/32. No gated stage; the next step is a development
   decomposition record (§9's form) on K's and D's logs. **A1:** read on K′. D′ is not run, and
   the decomposition is §18.6's diagnostic Δ, whose rows (Δ-VOID, Δ-MODEL, Δ-PLANNER, Δ-MIXED)
   are declared there.
4. **S0-LOW** — the bar's power at W's D rate is below 0.80 (by the table, W(D) ≤ 34/64). Escalate:
   a ruling either runs the gated stage as frozen, with its power stated, or closes with a
   development decomposition. No knob changes on these seeds.
5. **S0-OVER** — the projected gated stage does not fit the cap even without the replication arms
   (§13). Escalate (cloud GPU time is the owner's decision).
6. **S0-DONE** — otherwise. The Stage 0 record states the chosen configuration, D's counts, b, the
   bar count, H-pose's D count beside the bar (the measured non-trivial reference, §6; it does not
   enter b), the demonstration's rejection counts on K and D, the recomputed power table (bar and
   McNemar, from D's rates) and the projection.
   **The protocol freezes when the Stage 0 PR merges**; the gated stage then runs without a further
   ruling unless a declared rule escalates.

## 8. Gated stage S

- **Cohort:** **128 fresh resets, seeds 101200–101327** of the lift stream; reserve for §4's
  replacement 101400–101463. Goals made first (one GPU job), then every arm of §6 including the
  replication seeds, at the frozen revision from a clean run tree
  (`scripts/new_worktree.sh <dir>-run --from <frozen sha> --run`), one run. Each cohort and arm is
  run once; a second run into a new output name is refused unless a VOID ruling names fresh seeds.
- **Why 128.** The bar alone would not need it: at its 0.50 floor a true rate of 0.60 meets the
  bar with probability 0.99 at 128 resets (64/128) and 0.96 at 64 (32/64); 0.55 gives 0.89 and
  0.82. The McNemar tests need it, because the goal-blind and scene-blind controls may lift on
  some resets. Simulated for this draft (exact one-sided test at p < 0.01, 4 000 trials,
  independent coupling), W at 0.5 against a control at 0.3 has power 0.80 at 128 and 0.41 at 64.
  W at 0.6 against 0.3 has 0.99 and 0.81, and W at 0.5 against 0.2 has 1.00 and 0.86. TASK-081/083
  used 128 resets as well. The power is recomputed from D's rates in the Stage 0 record.

## 9. Rows (first match) and what each leads to

k_X is arm X's successes out of 128; "beats" is an exact one-sided McNemar test on the paired
resets at **p < 0.01**. The five tests of row 2 are conjunctive (all must pass), an
intersection-union test, so no multiplicity correction is needed.

1. **VOID** — a guard of §12 failed (including F's executed actions or outcome differing from the
   goal demonstration's on any kept reset), a run did not complete, a stage exceeded 1.5 × its
   measured projection, or a cohort was run twice. Nothing is read. One repeat on fresh seeds
   named by a ruling; a second VOID closes the task **INCONCLUSIVE**.
2. **CL-PASS** — k_W ≥ bar, **and** W beats each of **N, W-shuf, W-mean, rand and hold**. The
   primary claim, "J-driven closed-loop lift success toward image subgoals of a privileged
   per-reset demonstration", with §14's qualifiers.
3. **CL-UNSPECIFIC** — k_W ≥ bar, but W does not beat at least one of the five. W lifts, but its
   success is not shown to depend on what the failed control removes (the commands, the goal or
   the scene). Escalate; no claim.
4. **CL-LOW** — k_W < bar, and W beats rand and hold. Escalate; no claim.
5. **CL-FAIL** — otherwise (W does not beat rand or hold).

The rows are exhaustive: 2–3 split k_W ≥ bar by the tests, 4–5 split k_W < bar by the floors.
There is no ceiling row in S. An earlier draft had S-VOID-CEILING (k_F < bar), but it could fire
only if F diverged from the demonstration, which is a determinism defect and is now a VOID guard.
The demonstration's rejection count on S's drawn seeds is reported (§10).

What each row leads to (recommendations; the owner, or Claude under delegation, rules):

- **CL-PASS:** S5's place study on `so101_cube_place_v0` is the next preregistration (its own
  goals, bars and controls). The LeWM long-budget re-run (R28.57) becomes the route to a
  LeWM-driven SO-101 claim. "J needed" is not claimed: k_F and k_H-pose stand beside k_W.
- **CL-UNSPECIFIC, CL-LOW, CL-FAIL:** **a development decomposition record comes before any second
  controller variant** (horizon, samples, cost, subgoals, switching, hybrid, another model) is
  preregistered (TASK-077's precedent): from S's own episode logs and from new development seeds
  only, no training. It reads, per arm, which subgoals were reached and how (reached or timed
  out), the fingertip-to-cube offset at the close in the grasp frame against S2's envelope, the jaw
  closure and contact, the maximum rise, and the executed gripper commands. It names the failing
  phase before any variant is chosen.
- **No abandonment clause.** This is the first SO-101 closed loop: one model family, one
  configuration and one goal procedure. A clause here would close a line on one attempt. On G1,
  clauses fired only after a declared variant had also failed (TASK-089 after TASK-088; TASK-093
  after TASK-092), and that record is what justified closing. The binding consequence above (a
  decomposition before a variant) takes the clause's place.
- **Every row:** R7 does not change (R28.9); nothing is re-thresholded, re-run, re-selected or
  re-chosen after S's numbers are seen; every arm's count is reported with its interval, including
  those of rows further down.

## 10. Reported only (no gate)

- Every arm's count with its exact 95 % interval; every paired difference against W with its
  discordant counts and McNemar p (and N, W-shuf, W-mean against rand).
- **Non-inferiority against H-pose:** the paired difference k_W − k_H-pose with a reset-bootstrap
  95 % interval (10 000 resamples, salt 10104); "not detectably inferior beyond δ" if its lower
  bound is above **−16/128** (TASK-081's allocation, carried as a fraction; not calibrated here).
  The same against F, with the note that F scores every kept reset by construction (§6), so this
  comparison is W's failure count against a constant arm. Reported, read by no row.
- **The demonstration's rejection count** on S's drawn seeds, with reasons (§4).
- **Action dependence beyond N (reported only, not an arm).** Because N executes the zero command,
  W > N adds little to W > hold. As a non-degenerate reading, on every 10th planning call of W on
  S, J's cost of the executed best candidate is compared with J's cost of the same candidate with
  its H steps randomly permuted (8 permutations, salt 10106) and re-projected; reported as the
  fraction of calls on which the unpermuted candidate is cheaper than every permutation, per
  phase. It reads W's own state at those calls and changes nothing W executes.
- **The replication seeds** W-100100 and W-100102: counts against the bar, and McNemar against
  rand, hold, N and W-shuf.
- **Per-phase decomposition**, per arm: subgoals reached or timed out and when; the fingertip site's
  offset from the cube centre in the grasp frame (closing, along, up) at the first command on
  subgoal 3 and at its end, against S2's envelope (5 mm toward the fixed jaw, 10 mm along and up,
  20 mm away); any gripper contact, both-jaw contact, the maximum rise, the step of success; lost
  cubes; `clipped` and `stopped` counts and their reasons.
- **The model inside the loop:** J's open-loop prediction error at h = 1, 4, 8 and 16 against the
  encoded frames the episode then produced, per phase; the chosen cost at every command.
- Planning seconds per command (median, 95th percentile), GPU hours per job, worker counts.
- D's single-final-goal diagnostic (§7.4) beside S's W.

## 11. Statistics

Exact (Clopper–Pearson) 95 % intervals for every count. Exact one-sided McNemar tests on the
discordant pairs (binomial with p = 0.5). The reset-bootstrap interval of a paired difference
resamples the 128 resets with replacement (10 000 resamples, salt 10104) and takes the 2.5th and
97.5th percentiles of the summed difference. Power for the Stage 0 record by exact binomial sums
(bar) and simulation with salt 10105 (McNemar; 20 000 trials per cell; overlap, half and
independent couplings, TASK-081's form).

## 12. Seeds, salts, guards

- **Seeds** (the `so101_cube_lift_v0` stream, `default_rng([9801, seed])`): calibration K
  101010–101041 (reserve 101042–101059); fresh calibration K′ 101060–101091 (reserve
  101092–101099); development D 101100–101163 (reserve 101164–101199); gated S 101200–101327 (S's
  reserve 101400–101463); fresh development D′ 101600–101663 (reserve 101664–101699); debug
  101900–101999. K′ and D′ are used only under §7.6 row 1. None appears in `docs/`, `src/`,
  `scripts/`, `tests/`, `benchmarks/`, `configs/` or `.mc/` on any branch (`git grep -w` over every
  remote ref, 2026-10-19). S2 used 100–163, 1000–1015 and 0–63.
- **Salts:** 10101 (CEM), 10102 (rand), 10103 (derangement), 10104 (bootstrap), 10105 (power
  simulation), 10106 (the permutation reading of §10). None appears anywhere in the same check.
  TASK-100's 10011–10015 are not reused. **A1:** 10107 (the v1 marker stream) and 10108 (§18.6's
  random candidates and planner seeds), unused on every remote ref (`git grep -w`, 2026-10-19).
- **A2 (§19):** envelope cohort E 102000–102031 (reserve 102032–102047), salt **10109** (the
  envelope's offset directions), and Stage 0″'s K″ 102100–102131 (reserve 102132–102139) and D″
  102200–102263 (reserve 102264–102299), all unused on every remote ref (`git grep -w`, 2026-10-19).
- **A1 cohorts:** K′ and D′ are in use (§18.5). Diagnostic cohort **X 101700–101715 (reserve
  101716–101731)**, used only by §18.6 and unused on every remote ref (2026-10-19).
- **Guards (each a VOID):** the tracked tree clean and at the frozen revision at every job's start
  and end; this task's manifest and TASK-100's `frozen_files_check`; the checkpoint file hashes and
  implementation hashes (§3); the DINOv2 weight pins; the fit-moments hash; the goal-store hash
  read by every arm; seeds inside their cohort; no cohort run twice; every latent, cost and action
  finite; F's executed action sequence and outcome equal to the goal demonstration's on every kept
  reset; every GPU job under `scripts/gpu_run.sh --wait --board`.

## 13. Budget, disk and evidence

**Hard cap: 20 GPU h in total, Stage 0 included, Stage 0 itself capped at 6 GPU h.** GPU hours are
lock time. CPU arms (F, H-pose, rand, hold) need no frames to act and run without rendering
outside the lock, **except their first 8 episodes on each cohort**, which render frames for the
evidence (§13, disk). **A1:** they run in that cohort's episode jobs under a lock, after the
cohort's checks, and not in the goal job (§18.5's ordering rule). CEM arms run as a pool
of worker processes inside one lock, each episode on its own seed and model state, so the pool
size does not change any episode (checked by §7.2.4 at two pool sizes). **A1:** that check is read
in its conditional form (§18.5). Because the EGL renderer is not bit-stable on some states, **S's
runs are not guaranteed to be bit-reproducible.** The controller and the physics are deterministic
given the frames. Stage 0′'s budget is §18.7.

| Stage | Episodes on the GPU | Estimate (GPU h; estimates, not measurements) |
|---|---|---|
| Stage 0 checks, debug, goals for K and D | — | 0.3–0.6 |
| K | 4 configurations × 32 | 0.3–1.5 |
| D | 4 CEM arms (W, N, W-shuf, W-mean) × 64, plus the 16-reset diagnostic | 0.6–3 |
| S | goals, then 6 CEM arms × 128 = 768 | 1.5–9 |

The estimates assume 0.1–0.4 s of planning a command at 256 × 6 samples (J has 177 536 parameters
and took 0.074 s per training update of 64 sixteen-step windows), 120–200 commands an episode, and a
pool of 4–6 workers. Stage 0 replaces them with measurements. If the measured projection of S does
not fit the cap's remainder, the replication arms are dropped first (they are reported only) and
the drop is recorded. If S still does not fit, the row is S0-OVER. Exceeding a stage's measured
projection by more than 1.5 × stops the run (VOID). Before any job longer than an hour, the session
tells the V2D session and respects the `- v2d:` lines under "Requests" on
`~/.local/state/gpu/BOARD.md`.

**Disk and evidence.** Goal stores (four subgoals × two 224 px frames a reset) and per-step episode
logs (actions requested and applied, statuses, chosen costs, task truth for the reported measures)
are small. Frames are kept only for the first 8 episodes of each arm on each cohort. Run outputs
go under `outputs/task101-*`; evidence goes to `~/develop/emai/evidence/task101-*/` with
`SHA256SUMS`. Every name is new; nothing under `data/`, `checkpoints/` or `outputs/` is
overwritten; TASK-100's checkpoints are read only.

## 14. What this cannot show

- **Simulation only, one run.** One scene (a red 3 cm cube on a tan table, fixed lighting), one
  task, one robot model, MuJoCo soft contacts; nothing about hardware (S6 needs an owner ruling).
- **The goals come from a privileged demonstration of the same reset.** They show the arm at the
  grasp pose. A success shows that the planner matched a demonstration's image subgoals closely
  enough to lift, not that it localised the cube or chose a grasp on its own (§4). The switching
  rule reads the demonstration's joint positions.
- **`native_jepa`, not LeWM.** Whether LeWM's token predictor would do as well is untested here
  (R28.57). Three J seeds share one corpus, recipe, encoder and grid; the primary seed is one.
- **One planner configuration**, chosen from four on 32 calibration resets; the sample, iteration
  and elite counts are fixed by budget, not tuned.
- **The corpus is privileged scripted play** (`so101_play_policy_v0`): the model has seen many
  scripted top-down grasps like the demonstration's, with partial joint coverage and 26 % clipped
  commands. Whether it generalises to other grasp styles is not tested.
- **"J needed" is not tested by any row.** F (privileged) and H-pose (the goal pose, no images, no
  model) stand beside W; if H-pose lifts as often, the goal pose alone solves what W solved.
- **N is weak by construction** (§6): it executes the zero command, so W > N adds little beyond
  W > hold; the permutation reading of §10 is the reported-only action-dependence measure.
- **F is not a measured ceiling** (§6): it replays the goal demonstration, so it scores every kept
  reset by construction; the bar's cap is 0.90 by construction, and the follower's measured check
  is the demonstration's rejection count.
- **Non-inferiority uses a carried allocation** (δ = 16/128) and is reported only.
- **SO-101 results are not Apple→Plate results**; R7 does not change.
- **A1 (§18.4): in-distribution scenes only.** The task is `so101_cube_lift_v1`, whose marker is
  drawn inside the play corpus's range. Nothing is shown for `so101_cube_lift_v0` or for scenes with
  distractors outside the corpus's range. Moving one inert element out of range moved the
  pipeline's linear cube readout from 1.048 to 6.106 cm, a recorded sensitivity.
- **A1: K's 0/128 ran under the void condition and is attributed to nothing.** In particular, that
  the marker caused it is not shown (§18.4).
- **A1: runs are not guaranteed to be bit-reproducible** (the renderer; §7.2.4's conditional form).

## 15. Considered and not adopted

- **LeWM (L) as the model:** it failed S4's G6 on two seeds (R28.35). Its re-run is R28.57's
  separate recommendation.
- **A goal without the reset's arm pose, or a single final goal:** §4's reasons; the single-goal
  form is a D diagnostic only.
- **Proprioception or a readout in the planning cost:** J has no state input, and R28.39 advises
  against taking the grasp point from the predicted-cube readout. The cost is image-latent only,
  and only the switching rule reads joints.
- **Planning on requested (unprojected) actions:** out of J's training distribution (§5.2).
- **The place task first:** the plan orders lift before place (SO101_PLAN S5), and place needs a
  working pick.
- **A hybrid arm:** §6.

## 16. Process

DRAFT (this PR: docs and card only) → independent review → merge → Stage 0 PR (code, tests,
checks, K, D, the bar and the projection; the Stage 0 record
`docs/experiments/so101_closed_loop_lift_v1_stage0.md` and a DECISIONS entry) → independent review
→ **freeze at its merge** → Stage S in a clean run tree → results document
`docs/experiments/so101_closed_loop_lift_v1_results.md` and a DECISIONS entry with the row →
review → merge. The executing agent reports rows; it does not choose the next study.
**A1:** Stage 0 ended S0-VOID and was not frozen. The A1 docs PR (§18) is reviewed and merged.
Then a Stage 0′ PR (code, tests, checks, K′, D′, the record
`docs/experiments/so101_closed_loop_lift_v1_stage0b.md` and a DECISIONS entry) is independently
reviewed, and **the protocol freezes at its merge**.

## 17. Rulings this document records (R28.43–R28.56, DRAFT; decided by Claude under owner delegation)

*A1's rulings, R28.65–R28.75, are in §18.10.*

- **R28.43** — TASK-101 is S5: a closed loop on `so101_cube_lift_v0` in simulation; task,
  resets, scorer and the 200-command budget unchanged (§2).
- **R28.44** — the model is TASK-100's J, unchanged and hash-checked; the primary seed is the one
  with the lowest val criterion at its kept checkpoint (J-100101); the other two are replication
  arms; N is N_J-100101 (§3).
- **R28.45** — the live latent is TASK-100's g4 pipeline with float16 rounding and the frozen
  moments, checked against the stored latents at Stage 0 (§3, §7.2).
- **R28.46** — goals: a per-reset demonstration by follower v1, four image subgoals after commands
  40, 70, 90 and 120 with their joint positions, written once per cohort and hashed; replacement
  from a reserve if the demonstration fails or terminates or stops before command 120; what each
  arm may read (§4).
- **R28.47** — the controller: `SchemaCEMPlanner` unchanged, MPC every command, 256 samples, 6
  iterations, 32 elites; candidates projected by the embodiment's chained `preview` (plan on what
  is executable); latent cost, last or step-mean; H ∈ {8, 16} (§5).
- **R28.48** — subgoal switching on proprioception and the goal pose: 0.04 rad RMS arm, 0.15 rad
  jaw, 3 consecutive commands, or timeouts 60 / 45 / 30 (§5.3).
- **R28.49** — arms W, W-100100, W-100102, N, W-shuf, W-mean, rand, hold, F and H-pose; no hybrid
  arm; N's ties exact by construction (one roll-out broadcast to every candidate); F replays the
  goal demonstration and is a determinism guard, not a measured ceiling; H-pose is the measured
  non-trivial reference (§6).
- **R28.50** — Stage 0 on calibration and development seeds only: checks, K's configuration rule,
  D's development closed loop, rows S0-VOID, S0-CEILING (the demonstration rejected on 8 or more
  drawn D seeds), S0-NULL, S0-LOW, S0-OVER and S0-DONE; same-seed reruns only for a VOID raised
  before the cohort's counts were read, otherwise the fresh cohorts K′ or D′; the freeze at the
  Stage 0 merge (§7).
- **R28.51** — the bar: b = min(0.9 × r_F(D), max(0.50, LCB(W(D)))), ⌈b × 128⌉ of 128, where
  r_F(D) = 1 by construction, so the cap is 0.90 and LCB(W(D)) is the binding calibration (§7.5).
- **R28.52** — the gated stage: 128 fresh resets, one run, every arm (§8).
- **R28.53** — rows VOID (including F diverging from the demonstration), CL-PASS ("J-driven
  closed-loop lift success toward image subgoals of a privileged per-reset demonstration"),
  CL-UNSPECIFIC, CL-LOW and CL-FAIL (first match), exact one-sided McNemar at p < 0.01 against N,
  W-shuf, W-mean, rand and hold; a decomposition record before any second variant; no
  abandonment clause (§9).
- **R28.54** — reported only: non-inferiority against H-pose and F at δ = 16/128, the
  demonstration's rejection counts, a permutation reading of action dependence, replication
  seeds, the per-phase decomposition and the model-in-the-loop measures (§10).
- **R28.55** — seeds, salts and guards as §12.
- **R28.56** — budget: a 20 GPU h hard cap with Stage 0 ≤ 6 GPU h, replication arms dropped first
  if S does not fit, every GPU job under `scripts/gpu_run.sh --wait --board` (§13).

## 18. Amendment A1 (2026-10-19): the ruling on Stage 0's S0-VOID, and Stage 0′

**Decided by Claude under owner delegation** ([DECISIONS](../DECISIONS.md) 2026-10-19 (bv),
R28.65–R28.75). It answers the escalation R28.63 of the
[Stage 0 record](so101_closed_loop_lift_v1_stage0.md) (row S0-VOID, R28.59). The protocol is still
not frozen: it freezes when the Stage 0′ PR merges. Every section not listed in §18.9 is unchanged,
and every rule below is fixed before any Stage 0′ seed is drawn.

### 18.1 What failed, and what the ruling does

Check 7.2.2 failed. On the live-encoded subgoal frames of D's 64 goal demonstrations, TASK-100's
R-enc read the cube at 6.106 cm in median, against B_cube = 2.397 cm. `so101_cube_lift_v0` draws its
inert blue target marker at the fixed point x = 0.30 m, y = −0.20 m. `so101-play-v1`, the corpus on
which J, the feature moments and R-enc were fitted, drew the marker per episode inside
x ∈ [0.16, 0.22] m, |y| ≤ 0.14 m, at least 8 cm from the cube. The same 256 states, re-rendered
with only the marker moved into that box, read 1.048 cm (`check marker`).

**Ruling (R28.65): a declared condition change.** TASK-101 runs on a new task variant,
**`so101_cube_lift_v1`**. It is `so101_cube_lift_v0` with one difference: the inert target marker
is drawn per reset from the play corpus's declared marker distribution, by the corpus's rule and
from a salt of its own (§18.2). The lift task's frames then come from inside the scene distribution
that TASK-100's J and readout were fitted on, in this element as in the others compared in
§18.4. `so101_cube_lift_v0`, its code and TASK-098's numbers on it are unchanged.

### 18.2 `so101_cube_lift_v1`, exactly

- **Cube and arm.** As v0: cube xy uniform on x ∈ [0.18, 0.23] m, y ∈ [−0.10, 0.10] m and yaw
  uniform on ±45°, from `default_rng([9801, seed])`, the same draws in the same order. For every
  seed, v1's cube pose equals v0's bit for bit. The arm starts at `home_rad`.
- **Marker.** From its own stream `default_rng([10107, seed])` (salt **10107**, new): x uniform on
  [0.16, 0.22] m, y uniform on [−0.14, 0.14] m, redrawn until the marker centre is at least 0.08 m
  from the cube centre (at most 1 000 draws, else an error). These are the corpus's own constants,
  `so101_play_corpus.TARGET_REGION` and `MIN_CUBE_TARGET_M`, imported rather than copied, and
  `so101_play_corpus.sample_reset`'s rule: uniform on the box, redrawn on the distance. The marker
  comes from a separate stream so that the cube draws stay those of v0. Given the cube, the
  marker's distribution is exactly the corpus's. The lift cube box lies inside the corpus's cube box
  (§18.4), so every (cube, marker) pair v1 can draw lies inside the corpus's support.
- **Scorer, thresholds, step budget, cameras, embodiment.** `LIFT_V0` and `lift_outcome`,
  unchanged (the marker plays no part in lift scoring), and §2's scene, cameras and embodiment.
- **Code (Stage 0′).** `so101_tasks.py` and `so101_simulation.py` are pinned by
  `benchmarks/manifests/so101-play-v1.json` (TASK-099) and stay byte-identical. The variant goes in
  a new NumPy-only module, **`src/embodied_jepa/so101_tasks_lift_v1.py`**. It holds
  `LIFT_V1_TASK = "so101_cube_lift_v1"`, `MARKER_SALT = 10107`, `sample_reset_lift_v1(seed)` (it
  calls `so101_tasks.sample_reset(LIFT_TASK, seed)` for the cube and replaces `target_xy`) and
  `SO101CubeLiftTaskV1(SO101CubeLiftTask)`, which overrides only `name` and `sample_reset`.
  **No registry entry.** `config.py` is byte-pinned by G1 manifests (`apple-aligned-cost-v1`,
  `apple-control-development-v1`, `apple-goal-alignment-v1`, `apple-policy-v1`,
  `apple-wide-object-ceiling-v1`, `task072-checkpoint-compatibility`), so an additive `TASKS.register` would change pinned bytes.
  As §7.1 already fixes, the run configuration names the task as a module constant (the precedent
  of TASK-066 and TASK-100). `config.py` keeps refusing an `so101` experiment config. Tests: v1's
  cube draws equal v0's for 1 000 seeds; every marker lies in the box and at least 8 cm from its
  cube; the marker stream is fixed by its salt and independent of v0's; the module imports no torch
  or MuJoCo; `so101_tasks.py`, `so101_simulation.py` and `config.py` keep their `origin/main`
  hashes.

### 18.3 Why this option (R28.66)

- **The marker is inert for lift.** Its geom has `contype = 0` and `conaffinity = 0`, so it
  touches nothing, and `lift_outcome` never reads it. Moving it changes what the cameras see and
  nothing the scorer or the physics sees. Check 7.2.6 (§18.5) tests the physics part.
- **It is the smallest change that removes the measured mismatch.** It touches neither the model
  nor the readout, the corpus, the planner, the goals or the scorer. The diagnostic shows the
  mismatch falling from 6.106 cm to 1.048 cm when only this element moves.
- **Retraining on a corpus with the marker at the task pose (not adopted).** That needs a new
  corpus and a rerun of TASK-100's gated study, whose WM-PASS for J is what admitted TASK-101.
  It would change the model under test to fit one task's scene, and it costs a corpus collection
  and a full S4 rerun.
- **Removing the marker (not adopted).** Every corpus episode shows a marker, so a scene without
  one is also outside the training distribution, and nothing has measured how far.
- **One fixed in-box point (not adopted).** A single point lies inside the support but is a
  degenerate draw from it. A per-reset draw matches the corpus's distribution, including markers
  as close as 8 cm to the cube, which the corpus showed too.
- **Closing INCONCLUSIVE (not adopted).** The cause was measured and the fix is cheap and
  declared. The lesson is recorded either way (§18.8).

### 18.4 What A1 narrows, and the honest risks (R28.67)

- **The claim is narrowed to scenes inside the corpus's distribution.** The primary claim (§1)
  becomes "J-driven closed-loop lift success toward image subgoals of a privileged per-reset
  demonstration, on `so101_cube_lift_v1` (the lift scene with its inert target marker drawn inside
  the play corpus's marker range)". The parenthesis is part of the claim's name. Nothing in
  TASK-101 says anything about `so101_cube_lift_v0`.
- **Sensitivity to an off-distribution distractor is a finding, recorded, not hidden.** One inert
  scene element, moved about 10 cm outside its training range, shifted the frozen-DINOv2 g4
  latent's linear cube readout from 1.048 cm to 6.106 cm in median, mostly as a constant bias
  (+2.8 cm in x, −5.3 cm in y). The pipeline's readout is not robust to an off-distribution
  distractor. A v1 result does not transfer to scenes with distractors outside the corpus's range,
  and §14 states so.
- **K's 0/128 may not be due to the marker, or not only to it.** No K episode reached even
  subgoal 1. Subgoal 1 is the pre-grasp pose, a free-space move with no contact, and on H8-mean the
  median closest arm RMS to it was 0.41 rad against the 0.04 rad tolerance. In this task the marker
  appears both in the current frame and in the goal frames, at the same place, so a latent distance
  partly cancels a static offset. The marker therefore explains the readout miss more directly than
  it explains a planner that does not reach a free-space pose. It is the only off-distribution
  element measured so far, and A1 removes it. **That it caused K's null is not shown.** K′ may be
  null again, and §18.6 declares in advance what is read then.
- **The goal design is kept (H-pose 8/8 on D).** R28.63 asked the ruling to weigh H-pose, which
  lifted on all 8 D slots it ran using only the subgoal joint positions. A1 keeps the goal design:
  changing the goals together with the scene would confound two changes in one rerun. §1 already
  claims no "J needed", and §14 already says that a success shows tracking of a demonstration's
  pose subgoals through images. H-pose's D′ count is reported beside the bar as before. A goal
  redesign would need its own preregistration.
- **W-shuf now sees a marker mismatch, disclosed (§6).** In v0 every reset's marker was at the same
  point, so another reset's goal frames showed the marker where the current scene had it. In v1,
  W-shuf's borrowed goal frames show the marker at the other reset's position. This adds a
  roughly action-independent term to W-shuf's cost. W-shuf remains goal-blind, and its definition
  is unchanged.

**Scene comparison (fixed now; Stage 0′'s check 7.2.0 records it before any goal job).** It covers
every declared scene element, so that 7.2.2 is not again the first place a mismatch shows up:

| element | `so101-play-v1` (training corpus) | `so101_cube_lift_v1` | inside? |
|---|---|---|---|
| target marker xy | U(x 0.16–0.22, \|y\| ≤ 0.14), ≥ 8 cm from the cube | the same rule (salt 10107) | yes, by construction |
| cube xy | x 0.16–0.25, \|y\| ≤ 0.12 | x 0.18–0.23, \|y\| ≤ 0.10 | yes (sub-box) |
| cube yaw | U(±45°) | U(±45°) | yes |
| arm at reset | `home_rad` ± 0.06 rad uniform per arm joint | `home_rad` exactly (centre of the box) | yes (interior point) |
| jaw at reset | U(0.2, 1.2) rad | 0.6 rad (`home_rad`) | yes |
| table, cube size and colour, marker size and colour, lighting | `so101_simulation.py` at its TASK-099 pin | the same file, byte-identical | yes (a hash check) |
| cameras (`wrist_rgb`, `overview_rgb`), 224 × 224, renderer | the same file and MJCF, EGL on this RTX 5080 | the same | yes (a hash check; renderer nondeterminism as in R28.61) |
| motion | scripted play `so101_play_policy_v0` | follower v1 demonstrations; CEM's own commands | **not compared by construction**; §14's caveat stands |

Motion is the remaining known difference, and it is not a scene element: the corpus's scripted play
is not follower v1, and CEM's commands are its own. 7.2.2, re-run per cohort (§18.5), is the
measured check that the goal frames are readable.

### 18.5 Stage 0′ (R28.68–R28.71)

**Cohorts (R28.68).** **K′ = 101060–101091 (reserve 101092–101099)** and **D′ = 101600–101663
(reserve 101664–101699)**, both declared in §12 for exactly this use and never drawn, rendered or
run. Both are disjoint from K, D, S, the debug seeds, S2's seeds and the diagnostic cohort X below.
K's and D's goal stores were made in the v0 scene and cannot serve v1. D′ is used, not D: §7.6
row 1's text applies, since four D arm counts have been read (F 8/8, H-pose 8/8, rand 0/8,
hold 0/8), and D's goal store would have to be regenerated anyway. **The clause is not waived.** K's
and D's records stay as they are. K's 0/128 is reported as run under the void condition and is
never pooled with K′. Debug seeds 101900–101999 stay plumbing seeds, never counted. A second
post-count fix needs a ruling that names further seeds (§7.6 row 1, unchanged).

**Checks, amended (R28.69).** Each failure is S0-VOID, as before.

- **7.2.0 (new) scene comparison.** §18.4's table is filled in from the code. For each element it
  records the value or range on each side, whether the lift value lies inside the corpus's
  support, and the SHA-256 of `so101_simulation.py`, the MJCF and `configs/so101_sim_action.json`
  against their TASK-099 pins. Any "no" is S0-VOID. It reads no frame and no count.
- **7.2.1** unchanged (it reads corpus frames only).
- **7.2.2, per cohort.** R-enc on the live-encoded subgoal frames of **each cohort's own goal
  store**: K′'s before any K′ episode, D′'s before any D′ episode. Pass: median ≤ B_cube =
  2.397 cm, unchanged. The per-subgoal medians are reported, and so is the median on each goal
  demonstration's first frame (the reset frame).
- **7.2.3** on K′'s reset states, before any K′ episode.
- **7.2.4 determinism, written in its conditional form (R28.61 adopted into the text).** One W
  episode and one N episode are each run twice, and four W episodes are run at pool sizes 1 and
  6. **Pass:** in every pair, the executed actions never diverge before the live latent does, and
  wherever the live latents are identical at every step, the executed actions and the outcome are
  identical. The literal form (identical executed sequences and outcomes) is reported beside it.
  Reason: on this RTX 5080, MuJoCo's EGL renderer returns 2–3 distinct images for one physics
  state on some states (up to 19 pixels, one intensity level). This cannot be removed without
  changing the observation (for example by turning off multisampling, which would move the frames
  away from the corpus's rendering). So the controller and the physics are deterministic, and the
  renderer is the only nondeterministic input. **S's runs are not guaranteed to be
  bit-reproducible** (§13, §14). Each arm and cohort is still run once, and no row reads a rerun.
  F reads no frames and must still replay the demonstration exactly (§12's guard).
- **7.2.5** unchanged.
- **7.2.6 (new) the marker is inert.** On 16 debug seeds (101900–101915), the goal demonstration
  runs twice with rendering off, once with v0's reset and once with v1's for the same seed. The
  arrays compared, after the reset and after every command, are named exactly: the arm and jaw
  joint positions and velocities (`read()`'s `qpos` and `qvel`), the transport's joint control
  targets (`SO101Simulation.targets`), the demonstration's executed commands, and from
  `task_truth()` the cube's `cube_position`, `cube_quaternion`, `cube_velocity` and
  `cube_angular_velocity`, the `gripper_site_position`, the three contact flags
  (`fixed_jaw_contact`, `moving_jaw_contact`, `gripper_contact`) and `timestamp`. Each must be
  bit-identical between v0 and v1 at every step. **The marker position (`task_truth()`'s
  `target_position`) is excluded from that comparison, because it is the one intended
  difference;** instead it must equal `(target_xy, TABLE_TOP_Z)` exactly at every step, with
  `target_xy` the drawn point in v1 and (0.30, −0.20) in v0 (the marker is a fixed child of the
  world body, so its position is set, never integrated). It reads no frame and no count of any
  cohort, and debug seeds are never counted.

**The ordering rule (R28.70; fixes deviation 7, enforced by the runner, not by prose).**

1. **Order per cohort** c ∈ {K′, D′}: (i) c's goal job writes c's goal store, and **only** the goal
   store. The CPU arms' frame-rendering evidence episodes (§13) move out of the goal job into c's
   episode jobs. (ii) The checks run: 7.2.0, 7.2.1, 7.2.3, 7.2.4 and 7.2.6 once, before K′'s first
   episode, and 7.2.2 on c's goal store. (iii) Only then do c's episodes run, for every arm, CPU
   arms included. 7.2.5 (timing) is read from K′'s episodes; it gates eligibility (§7.3), not the
   order.
2. **A check ledger.** Every `run_task101.py check` writes a report into the run's output
   directory with the check's name, its pass or fail, the revision and, for 7.2.2, the SHA-256 of
   the goal store it read. `run_task101.py episodes` refuses (a `GuardError`, raised before any
   controller, simulator or seed is constructed) unless the ledger holds a **passing** report at
   the same revision for every check listed in (ii) and a passing 7.2.2 report naming the hash of
   the goal store those episodes read. Any failing report blocks every later `episodes` job of
   every cohort until a ruling.
3. **The summariser enforces it again.** `summarize_task101.py` refuses to print or write a count
   of any cohort whose first episode started before the latest required passing check report
   (timestamps and revisions from the ledger and the episode logs).
4. Tests (Stage 0′): `episodes` refused with no ledger, with a failing report, with a report at
   another revision and with a 7.2.2 report for another goal store's hash; the goal job writes no
   episode; the summariser refuses an out-of-order cohort.

**K′ and D′ (R28.71).** §7.3's configuration rule is unchanged, applied to K′ (32 resets, the four
configurations). §7.4 is unchanged, applied to D′ (every arm except the replication seeds, plus the
16-reset single-goal diagnostic). §7.5's bar is unchanged and reads W(D′). §7.6's rows are
unchanged, except row 3 (S0-NULL, §18.6), and read K′ and D′ in place of K and D. S0-CEILING counts
the demonstration's rejections on the D′ seeds drawn to keep 64. Because the marker is inert
(7.2.6), follower v1 is expected to behave exactly as on v0, and the count measures this.

### 18.6 If K′ is null again: a declared diagnostic Δ (R28.72)

**S0-NULL (§7.6 row 3) is amended.** If every K′ configuration scores 0/32, there is no gated
stage. D′ is not run. Before any controller variant, a development diagnostic Δ runs under its own
record (`docs/experiments/so101_closed_loop_lift_v1_null_diagnostic.md`). Δ reads K′'s logs and a
new **diagnostic cohort X = 101700–101715 (reserve 101716–101731)**. It trains nothing and changes
no knob. Its purpose is to say whether the null is on the planner's side or on the model's side,
read in advance by the rows below.

- **States.** On each X seed, the v1 goal demonstration (the same job and rejection rule as §4) is
  run, and the planner is queried, not executed, at the demonstration's states after commands
  t ∈ {0, 10, 20, 30} (on subgoal 1: free-space reach) and t ∈ {40, 50, 60} (subgoal 2: descent),
  with the demonstration's own subgoal latent as the goal. That makes 7 × 16 = 112 planning states.
  The configuration is the one §7.3's rule chose on K′ (H8-mean when all tie at 0).
- **Candidates at each state.** a_demo, the demonstration's next H executed commands (executable,
  since they were executed); a_cem, W's CEM best candidate from that state (planner seed
  10108 · 10⁶ + 1 000 · (seed − 101700) + t); and 255 random candidates uniform on [−1, 1],
  projected by §5.2's projector (salt **10108**, new). Every candidate is scored by J's cost as W
  scores it.
- **The a_demo window, fixed now.** The subgoal boundaries are fixed command indices of the
  demonstration (§4: subgoal 1 is the state after command 40, subgoal 2 after command 70), not
  per-seed switch points, and every kept demonstration executes at least 120 commands (§4's
  rejection rule). a_demo is always the demonstration's commands t … t + H − 1, untruncated and
  unpadded: the last state (t = 60) needs commands up to 75 at H = 16, so a full window always
  exists, and no state is skipped. The goal is the subgoal the state belongs to (subgoal 1 for
  t ≤ 30, subgoal 2 for t ≥ 40), whatever the window's end. At H = 16 the windows at t = 30 and
  t = 60 run past that subgoal's boundary (into commands 40–45 and 70–75), so a_demo overshoots the
  goal frame there; those 2 × 16 = 32 states are kept and counted in m1 and m2, and m1 and m2 are
  also reported without them (reported only; the rows read the full 112). At H = 8 no window
  crosses a boundary.
- **Measures.** m1, the fraction of states where J's cost of a_demo is **lower** than that of
  a_cem (the model prefers what the search did not find). m2, the median normalised rank of a_demo
  among a_demo and the 255 random candidates under J's cost (0 = best, chance 0.5). Reported only:
  m3, the fraction of states where executing a_cem open-loop for H steps in a copy of the
  simulation lowers the arm RMS to subgoal k's joints. Also reported: m1 to m3 per subgoal, and
  from K′'s logs the per-subgoal closest arm RMS and the chosen cost over time.
- **Rows (first match).**
  1. **Δ-VOID**: a guard failed, or fewer than 12 X seeds kept a demonstration. Escalate.
  2. **Δ-MODEL**: m2 ≥ 0.25. J's cost does not rank the demonstration's own commands toward its own
     subgoal ahead of most random commands. The cost landscape does not point at the subgoal, so
     a planner change alone is not expected to fix it. **Leads to:** no planner variant on J with this goal form is
     preregistered. The next step is a model, representation or goal-form change (for example
     R28.57's LeWM long-budget rerun, or another goal representation), under its own
     preregistration.
  3. **Δ-PLANNER**: m2 ≤ 0.10 and m1 ≥ 0.50. J ranks the demonstration's commands well, but CEM at
     256 × 6 does not find commands as cheap. **Leads to:** a planner-variant preregistration
     (samples, iterations, warm start or initialisation; J and the goals unchanged) is admitted.
  4. **Δ-MIXED**: otherwise. Escalate with the record; no variant is admitted by default.
- **Why these cut-offs, and what Δ cannot show.** The cut-offs are declared, not calibrated. 0.25
  is half of chance and about TASK-100 §9's offline normalised rank (0.24–0.27) against harder,
  perturbed candidates. 0.10 asks for clear separation. Δ is optimistic for the model: a_demo
  leads exactly to the subgoal frame it is scored against, from states on the demonstration's own
  path, so a Δ-MODEL row is strong evidence and a Δ-PLANNER row is weaker. Δ is a development
  reading on 16 seeds. It is not a gate, and it makes no claim. **Budget:** at most 1.5 GPU h,
  counted against **the Stage 0 cap of 6 GPU h cumulative** (§18.7), and so also inside the 20 GPU h
  total cap (S does not run under S0-NULL), under `gpu_run.sh --wait --board`. Under S0-NULL D′
  does not run, so the projection is 1.66 (used) + about 0.3 (goals and checks) + 1.1 (K′) + 1.5
  (Δ) ≈ 4.6 GPU h, inside the cap; if it would not fit, the row is Δ-VOID (escalate) and the cap
  is not extended silently.

If K′ is not null, Δ does not run, and D′ follows as §18.5 says.

### 18.7 Budget (R28.73)

Stage 0 used 1.66 GPU h of its 6. K's measured throughput (1.07 h of lock time for 128 CEM
episodes) projects Stage 0′ at about 0.3 h for goals and checks, 1.1 h for K′, and 2.3 h for D′
(256 CEM episodes and the 16-reset diagnostic): about 3.7 h, or 5.4 h cumulative. **The Stage 0 cap
stays 6 GPU h cumulative** (§13). If the projection made after K′ does not fit, the 16-reset
single-goal diagnostic is dropped first and the drop is recorded. If it still does not fit, the
row is S0-OVER (escalate). The cap is not extended silently. The 20 GPU h total cap and S's
projection (about 5.9–6.9 GPU h) are unchanged.

### 18.8 Lessons (R28.74; also for RESUME.md's list)

1. Before Stage 0, compare every scene element of a closed-loop task with the training corpus's
   scene distribution (7.2.0's table). Here one inert element was enough to move a linear readout
   by 5 cm.
2. Run every §7.2 check before any count of a cohort is read, and have the runner refuse
   otherwise; prose ordering is not enough (deviation 7).
3. Keep a cohort's evidence episodes out of its goal job, so that writing goals reads no count.

### 18.9 Sections changed by A1 (everything else is unchanged)

| section | change |
|---|---|
| title, status | the task is `so101_cube_lift_v1`; A1 is noted |
| §1 | the claim's name carries `so101_cube_lift_v1` and its parenthesis (§18.4) |
| §2 | the task is v1: v0 plus the marker rule of §18.2 |
| §6 | W-shuf's marker mismatch is disclosed (§18.4) |
| §7 (intro), §7.1 | K′, D′ and X; the new module `so101_tasks_lift_v1.py`; the ledger tests |
| §7.2 | 7.2.0 and 7.2.6 added; 7.2.2 per cohort; 7.2.3 on K′; 7.2.4 in its conditional form; the ordering rule (§18.5) |
| §7.3, §7.4, §7.5 | read on K′ and D′ |
| §7.6 | row 3 (S0-NULL) leads to Δ (§18.6) |
| §12 | salts 10107 (marker) and 10108 (Δ); cohort X; K′ and D′ in use |
| §13 | CPU-arm evidence episodes leave the goal job; S's runs are not guaranteed to be bit-reproducible; Stage 0′ budget (§18.7) |
| §14 | the claim narrowed to in-distribution scenes; the distractor sensitivity; K's null is not attributed |
| §16 | Stage 0′ in the process |
| §17 | points to §18's rulings |

Unchanged, among others: the model and checkpoints (§3), the goal procedure (§4), the controller
and its calibration grid (§5), the arms and their roles (§6, apart from the disclosure), the bar
rule (§7.5), rows S0-VOID, S0-CEILING, S0-LOW, S0-OVER and S0-DONE, the gated stage and its seeds
(§8), the gated rows (§9), the reported measures (§10), the statistics (§11), the guards (§12) and
the 20 GPU h cap.

### 18.10 Rulings A1 records (R28.65–R28.75; decided by Claude under owner delegation)

- **R28.65**: the condition change to `so101_cube_lift_v1` (§18.1–18.2).
- **R28.66**: why this option over retraining, removing the marker, one fixed point or closing (§18.3).
- **R28.67**: the narrowed claim, the distractor finding, K's null not attributed, the goal design
  kept, and W-shuf's disclosure (§18.4).
- **R28.68**: K′ and D′ (the D′ clause not waived) and the debug seeds (§18.5).
- **R28.69**: checks 7.2.0 and 7.2.6 added, 7.2.2 per cohort, and 7.2.4 in its conditional form
  (R28.61 adopted) (§18.5).
- **R28.70**: the ordering rule, enforced by the runner's check ledger and by the summariser (§18.5).
- **R28.71**: §7.3–7.6 read on K′ and D′ (§18.5).
- **R28.72**: the S0-NULL diagnostic Δ, with its cohort X, salt 10108 and rows Δ-VOID, Δ-MODEL,
  Δ-PLANNER and Δ-MIXED (§18.6).
- **R28.73**: budget (§18.7).
- **R28.74**: lessons (§18.8).
- **R28.75**: the process: this amendment is a docs PR, reviewed and merged; the Stage 0′ PR then
  brings the code, the tests, the checks, K′ and D′ and the record
  `docs/experiments/so101_closed_loop_lift_v1_stage0b.md`, and is independently reviewed. **The
  protocol freezes when the Stage 0′ PR merges.** R7 does not change (R28.9).

## 19. Amendment A2 (2026-10-19): the ruling on Δ-MIXED, a calibrated switching tolerance, and Stage 0″

**Decided by Claude under owner delegation** ([DECISIONS](../DECISIONS.md) 2026-10-19 (bx),
R28.83–R28.89). It answers the escalation R28.82 of the [Stage 0′ record](so101_closed_loop_lift_v1_stage0b.md)
(row S0-NULL, Δ-MIXED). The protocol is still not frozen. **This section is committed before any
envelope episode runs, and the commit order is the evidence:** the measurement of §19.3 runs at
the revision that adds this section, and its record (§19.8) is a later commit.

### 19.1 What Δ showed, and the ruling (R28.83)

Δ ([record](so101_closed_loop_lift_v1_null_diagnostic.md)) read m2 = 0.000 and m1 = 0.116: J's
cost ranks the demonstration's own commands ahead of every projected random candidate at the
median state, and CEM finds commands that J scores as cheaper than the demonstration's on 99 of
112 states. At the pre-grasp pose itself (t = 20, 30, arm RMS 0.000 to subgoal 1), CEM's cheaper
commands moved the arm to 0.048–0.055 rad RMS. K′'s closed loops came within 0.26–0.48 rad of
subgoal 1 in median (the best single episode within 0.035 rad) but no episode held §5.3's
0.04 rad for 3 consecutive commands, so every switch was a timeout.

**The 0.04 rad / 3-command switching tolerance was declared, not calibrated** (§5.3: "of the order
of a centimetre of fingertip motion"). TASK-074's lesson is that bars and tolerances are set from
measured ceilings. Δ places the cost's minimum about 0.05 rad off the subgoal pose, just outside
that tolerance, and nothing has measured how much pose error the task can absorb at a switch.

**Ruling (R28.83): measure the task's pose-error envelope on the privileged follower, set the
switching tolerance τ_s from it by a rule declared now (§19.4), and rerun the calibration on fresh
seeds (K″, §19.5).** J, the goals, the cost, the planner, the projection, the timeouts, the jaw
tolerance, the 3-command hold, the scorer and every arm are unchanged. Only `SWITCH_ARM_RMS_RAD`
can change, and only to the value §19.4's rule gives.

Not adopted, and why:

- **A planner variant** (more samples, warm start): Δ-MIXED admits none by default, and Δ's m1
  says CEM already finds what J scores cheapest; more search would find the off-pose minimum more
  reliably, not the pose.
- **A goal-form or cost change** (for example joints in the cost): J has no state input, and a
  cost change is a new controller needing its own preregistration. It stays a candidate if A2
  escalates.
- **A hybrid local grasp controller, or R28.57's LeWM longer-budget retrain:** larger changes. They
  are the two options §19.4 puts to the owner if the tolerance turns out not to be the binding
  constraint.
- **Closing TASK-101 now:** the switching tolerance is a declared, uncalibrated knob that Δ
  implicates, and calibrating it costs CPU minutes. Closing first would leave that untested.
- **Simply loosening the tolerance to, say, 0.06 rad:** that would be a knob turned on K′'s seen
  logs. The value must come from a measurement on fresh seeds, by a rule fixed before it runs.

### 19.2 What the switching tolerance controls

The rule of §5.3 moves the controller from subgoal k to k + 1 when the measured arm is within the
tolerance of subgoal k's joints (and the jaw within 0.15 rad) for 3 consecutive commands. A looser
tolerance lets the controller move on with a larger pose error. The task can absorb that error only
if the remaining phases still lift the cube from there. That is what §19.3 measures, with the
privileged follower standing in for the remaining phases. **Loosening the switching rule changes
when the controller moves on. It does not change what counts as a success: `lift_outcome` and
`LIFT_V0`'s thresholds, read on simulator truth, are unchanged.**

### 19.3 The envelope measurement (R28.84)

- **Seeds.** Fresh development cohort **E = 102000–102031 (reserve 102032–102047)** of the lift
  stream on `so101_cube_lift_v1`. E is disjoint from K, K′, D, D′, S, X, the debug seeds and S2's
  seeds, and was unused on every remote ref (`git grep -w`, 2026-10-19). For each seed the
  unperturbed goal demonstration (follower v1, `FOLLOWER_V1`, no offset) runs first, under §4's
  rejection rule, with replacement from E's reserve. Its measured joints after commands 40 and 70
  are the reference poses q₁ and q₂ (the goal store's subgoal 1 and 2 joints).
- **Offsets.** For seed s and subgoal k ∈ {1, 2}, a direction u = n / RMS(n), with n a standard
  normal draw over the five arm joints from `default_rng([10109, s, k])` (salt **10109**, new,
  unused on every remote ref). δ = r · u, so δ's RMS over the five arm joints is exactly r. The
  same direction is used at every level, so a seed's levels differ only in size. The jaw is not
  perturbed. **Levels r ∈ {0.02, 0.04, 0.06, 0.08, 0.12, 0.16} rad, N = 32 seeds each.**
- **Hand-over, exactly** (`so101_switch_envelope.perturbed_lift`):
  - **"pre" (subgoal 1, a rule reading).** The pre-grasp phase drives to q_pre + δ instead of
    q_pre for its 40 commands, jaw open. The follower then runs its own schedule unchanged: the
    descent's Cartesian waypoints, the close and the lift, aimed from simulator truth as always.
    This models a switch off subgoal 1 at pose error δ, after which the next phase tracks subgoal 2
    (whose own switch is governed by the same tolerance, read by "carry").
  - **"carry" (subgoal 2, a rule reading).** The descent's six waypoint targets are offset by
    δ · j / 6 (j = 1 … 6), so the arm arrives at q_grasp + δ at command 70. The close phase holds
    the arm there while the jaw closes (its arm target is q_grasp + δ, not q_grasp). The lift's
    waypoints are the follower's own. This models a switch off subgoal 2 at pose error δ, after
    which the jaw closes where the arm is.
  - **"reaim" (subgoal 2, reported only).** As "carry", but the close phase drives the arm back to
    the follower's own q_grasp while the jaw closes.
  - With δ = 0 every variant executes exactly the goal demonstration's commands. That is guard
    G-zero, checked on every kept seed for every variant.
- **Recorded per episode:** success by `lift_outcome` (unchanged), termination reason, success
  step, commands, the measured arm RMS and jaw error to q_k after the hand-over command (the
  achieved perturbation, which joint limits or contact can change), δ and the SHA-256 of the
  executed commands. Per level and variant: the count of 32 with its exact (Clopper–Pearson) 95 %
  interval and the median achieved hand-over RMS.
- **Compute:** CPU and MuJoCo only, rendering off, no GPU, so no lock is taken. 32 demonstrations
  plus 32 × 3 × 7 follower episodes, a few minutes. Script `scripts/measure_task101_envelope.py`,
  module `src/embodied_jepa/so101_switch_envelope.py` (NumPy only at import; no registry entry),
  tests `tests/test_so101_switch_envelope.py`, all new files. No G1, TASK-100 or TASK-101 Stage 0′
  file changes. The script refuses a dirty tracked tree and an existing output.
- **Guards (each A2-VOID):** fewer than 32 seeds kept from E with its reserve; G-zero fails; the
  tracked tree is dirty or the revision changes during the run; a run does not complete.

### 19.4 The rule for τ_s (R28.85; fixed now, applied mechanically to §19.3's counts)

A level **passes** when **both** rule readings lift on **at least 30 of 32**: "pre" (subgoal 1) and
"carry" (subgoal 2). Both are needed because the one tolerance governs every switch. "reaim" is
reported only.

1. **A2-VOID**: a guard of §19.3 failed. Escalate.
2. Let L be the largest level such that **every level up to and including L passes** (a level
   above a failing one does not count, so a non-monotone count cannot pick a larger tolerance).
   **A2-NOT-BINDING**: no level passes, or L < 0.06 rad. **τ_s stays 0.04 rad**, and A2 records
   that **the switching tolerance is not the binding constraint**. K″ does not run. Escalate to the
   owner with the choice: **a hybrid local grasp controller** (its own preregistration) **or R28.57's
   LeWM longer-budget retrain**. If 0.04 itself fails the 30/32 bar, that is recorded too.
3. **A2-TAU**: otherwise. **τ_s = L** (0.06, 0.08, 0.12 or 0.16 rad), and Stage 0″ (§19.5) runs
   with `SWITCH_ARM_RMS_RAD = τ_s`.

The bar 30/32 means a level passes with probability 0.79 at a true lift rate of 0.95 and 0.97 at
0.98 (0.62 and 0.95 for both readings together, if independent). It asks the remaining phases to
be nearly as reliable from the perturbed pose as the demonstration is from its own (64/64 in S2;
32/32 on K′ and 16/16 on X). Levels are not interpolated.

### 19.5 Stage 0″ under A2-TAU (R28.86)

- **Cohorts.** Fresh calibration cohort **K″ = 102100–102131 (reserve 102132–102139)** and fresh
  development cohort **D″ = 102200–102263 (reserve 102264–102299)**, both unused on every remote
  ref (2026-10-19) and disjoint from E and every TASK-101 cohort. This ruling is the one §7.6 row 1
  requires to name further seeds for a second post-count change. K′, D′ (never drawn) and X are
  not used again.
- **What changes.** Only `SWITCH_ARM_RMS_RAD`, from 0.04 to τ_s, for every arm that uses §5.3's
  rule (W, its controls and H-pose alike), and the cohort table. The jaw tolerance (0.15 rad), the
  3-command hold and the timeouts are unchanged.
- **Order and checks.** §18.5's ordering rule and check ledger, unchanged, at the Stage 0″
  revision: 7.2.0, 7.2.1, 7.2.3 (on K″'s reset states), 7.2.4 and 7.2.6 once, and 7.2.2 on each of
  K″'s and D″'s own goal stores, all passing in the ledger before that cohort's first episode.
- **K″.** §7.3's rule, unchanged: W (J-100101) runs **the same four configurations** (H ∈ {8, 16}
  × cost ∈ {last, mean}) on K″'s 32 resets; the most successes wins, ties to mean, then to H = 8.
- **Rows.** §7.6's rows, unchanged, read on K″ and D″ in place of K′ and D′: **if K″ and D″ meet the
  existing Stage 0 rule (S0-DONE), the protocol freezes at the Stage 0″ PR's merge**, with the bar
  of §7.5 from W(D″), and S runs as §8 says. **Any other row escalates.** In particular, S0-NULL on
  K″ (every configuration 0/32) does not run D″ and does not rerun Δ (Δ is spent); it escalates
  with the record, and the candidates are those of §19.4 row 2.

### 19.6 Budget (R28.87)

The envelope uses no GPU. Stage 0 has used 2.87 GPU h of its 6 GPU h cap. K′'s measured throughput
projects Stage 0″ at about 0.3 h (goals and checks), 1.1 h (K″) and 2.3 h (D″ with the 16-reset
diagnostic), about 3.7 h, or 6.6 h cumulative. **A2 raises the Stage 0 cap explicitly from 6 to
7.5 GPU h cumulative.** The reason: Stage 0′ ended at K′, and a new calibration is a second
Stage 0 pass. The 20 GPU h total cap is unchanged; S's projection (5.86–6.94 GPU h) still fits its
remainder. §18.7's drop order applies (the single-goal diagnostic first, then S0-OVER). Every GPU job
runs under `scripts/gpu_run.sh --wait --board` and respects `- v2d:` lines.

### 19.7 What A2 cannot show (R28.88)

- **τ_s is task-calibrated on the follower, a privileged controller.** The follower re-aims from
  simulator truth after a "pre" hand-over, and its descent, close and lift are scripted. A tolerance
  the follower absorbs is necessary for the closed loop, not sufficient: the CEM controller must
  still reach the next subgoals on its own.
- **A looser tolerance changes when the controller moves on, not what counts as a lift.** The
  scorer and its thresholds, on simulator truth, are unchanged. It can let W leave subgoal 1 from
  where Δ found J's cost minimum; whether the later phases then lift is what K″ measures.
- **One direction per seed and subgoal, 32 seeds per level, two hand-over points.** The switch off
  subgoal 3 (close to lift) is not perturbed separately. "carry" lifts from a perturbed grasp,
  which covers the pose the controller holds through the close.
- **The levels and the 30/32 bar are declared.** The envelope is a measured ceiling for the
  tolerance, under this follower and this perturbation model, and nothing more.
- **K′'s logs are not re-read to choose or check τ_s.** No count of a spent cohort selects anything.
- Simulation only; not Apple→Plate; R7 does not change (R28.9).

### 19.8 Process and record (R28.89)

One PR (branch `docs/task101-a2`): **first commit**, this section, the DECISIONS entry (bx) and the
envelope code and tests; **second commit**, the measurement at the first commit's revision from a
clean tracked tree, the rule applied mechanically, and the record
[`so101_closed_loop_lift_v1_envelope.md`](so101_closed_loop_lift_v1_envelope.md) (counts with exact
intervals per level and reading, seeds, revision, hashes; evidence under
`~/develop/emai/evidence/task101-envelope/` with `SHA256SUMS`) with τ_s and the row. Independent
review, then merge. Under A2-TAU, a Stage 0″ PR then sets `SWITCH_ARM_RMS_RAD = τ_s` and the
cohorts, runs §19.5, and is independently reviewed; **the protocol freezes at its merge only if
§19.5's rule gives S0-DONE**. Under A2-NOT-BINDING or A2-VOID, A2 escalates and nothing further
runs.
