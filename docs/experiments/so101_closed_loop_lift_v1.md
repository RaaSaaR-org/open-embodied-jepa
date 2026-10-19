# TASK-101 — SO-101 S5: a first closed loop on `so101_cube_lift_v0` in simulation (CEM over TASK-100's `native_jepa` token model J, toward image subgoals from a per-reset demonstration)

**Status: DRAFT preregistration (docs and card only; nothing has run).** Opened by Claude under
owner delegation ([DECISIONS](../DECISIONS.md) 2026-10-19 (bt), R28.41–R28.58; the design choices
are R28.43–R28.56 in §17). Freeze path: independent review, then merge of this DRAFT, then a
Stage 0 PR (code in new files, development closed loops, calibration of the planner and of the
bar on development seeds only), independently reviewed. **The protocol freezes when the Stage 0
PR merges.** No gated reset is run before then.

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
`so101_cube_lift_v0`, unchanged, on fresh resets, when the controller sees only the two camera
frames, its own joint state and image subgoals of a demonstration of the same reset? And does its
success depend on the actions, the scene and the goal, as it should?

**The primary claim**, made only by row CL-PASS (§9): **"J-driven closed-loop lift success toward
image subgoals of a privileged per-reset demonstration"** on `so101_cube_lift_v0` in simulation,
with every qualifier of §14. The goal source is part of the claim's name and is never dropped when
the claim is restated. It is not "LeWM-driven", it is not
"J needed" (the privileged follower and a non-learned pose follower given the same goal pose are
reported beside it), and it is not a placement claim.

## 2. Task, scene and scorer (unchanged)

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
| **W-shuf** | W with the whole goal (subgoal latents and joints) of another reset of the same cohort, by a fixed derangement (salt 10103) | goal-blind control |
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
counted). No gated seed is drawn, rendered or run before the freeze.

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

### 7.2 Checks (each failure is S0-VOID until fixed under its own record)

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

W (J-100101) runs all four configurations on the 32 K resets. **The configuration with the most
successes is chosen**; ties go to **mean** over last (TASK-089's reason: the first commands, which
are the ones executed, count), then to **H = 8** over H = 16 (half the compute). A configuration
whose projected Stage S cost (§13) does not fit the cap is not eligible. If every configuration
scores 0/32, the row is S0-NULL (§7.6). K's numbers are development numbers; K seeds are never
used again.

### 7.4 D: the development closed loop

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
   decomposition record (§9's form) on K's and D's logs.
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
  TASK-100's 10011–10015 are not reused.
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
evidence (§13, disk) and run inside that cohort's goal job under its lock. CEM arms run as a pool
of worker processes inside one lock, each episode on its own seed and model state, so the pool
size does not change any episode (checked by §7.2.4 at two pool sizes).

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

## 17. Rulings this document records (R28.43–R28.56, DRAFT; decided by Claude under owner delegation)

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
