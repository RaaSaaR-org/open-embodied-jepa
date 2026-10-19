# TASK-102 — SO-101 S5, second study: a hybrid closed loop on `so101_cube_lift_v1` in simulation (CEM over TASK-100's `native_jepa` model J approaches the pre-grasp image subgoal; a learned local grasp-and-lift policy G_loc grasps and lifts)

**Status: DRAFT preregistration (docs and card only; nothing has run).** Opened by Claude under
owner delegation ([DECISIONS](../DECISIONS.md) 2026-10-19 (by), R28.91–R28.97; the design choices
are R28.98–R28.109 in §17). **The owner can override the ruling that opened it (R28.97).** Freeze
path: independent review, then merge of this DRAFT, then a Stage 0 PR (code in new files, G_loc's
training, development closed loops, calibration of τ_h and of the bar on development seeds only),
independently reviewed. **The protocol freezes when the Stage 0 PR merges, and only on S0-DONE
(§7.7).** No gated reset is drawn, rendered or run before then.

This is phase S5 of [SO101_PLAN.md](../SO101_PLAN.md) after TASK-101
([so101_closed_loop_lift_v1.md](so101_closed_loop_lift_v1.md)) closed with no claim (R28.93). **It
is a simulation-only control study on the SO-101 cube-lift task. It is not an Apple→Plate
experiment, and no row of it changes R7** (R28.9). Its world model is `native_jepa` (J), not LeWM
(R28.92, R28.96).

Closest precedents, and what is taken from each:

- **TASK-101** ([protocol](so101_closed_loop_lift_v1.md), [envelope](so101_closed_loop_lift_v1_envelope.md),
  [Δ](so101_closed_loop_lift_v1_null_diagnostic.md)): the task `so101_cube_lift_v1`, the per-reset
  goal demonstration and its goal store, J-100101 and N_J-100101 with their hash checks, the live
  latent, the projected `SchemaCEMPlanner` at H8-mean, the switching form, the checks and the
  check ledger with its ordering rule, F as a determinism guard, H-pose, the statistics and the
  row ladder. TASK-101's code is imported, never edited.
- **TASK-071/072, P-3** ([apple_first_policy_v2.md](apple_first_policy_v2.md), [v1 §5.2](apple_first_policy_v1.md)):
  behaviour cloning of an MLP on frozen DINOv2 features plus proprioception and a clock, MSE to the
  demonstrated command, AdamW for 30 000 updates, selection on val MSE with a per-dimension
  output-std eligibility rule, and a live-against-offline equality check. P-3 scored 40/40 on
  `apple-to-plate-v2` (one run, one seed per arm; M2-FAIL on G3, because pretraining contributed
  nothing measurable). P-3 added DAgger with a privileged expert. G_loc does not (§15), so it is
  plain behaviour cloning on a fixed corpus.
- **TASK-088** ([zero_shot_reach_grasp_v1.md](zero_shot_reach_grasp_v1.md)): the bar is capped by
  measured ceilings (TASK-074's lesson), and a non-learned follower given the same goal
  information is reported beside the learned arm.
- **TASK-080/081/083**: exact one-sided McNemar tests at p < 0.01 against controls that remove one
  ingredient each, 128 fresh gated resets, and every claim restated with its qualifiers.

## 0. What TASK-101 handed over, and what it constrains

- **J cannot drive the whole lift by itself under TASK-101's design.** On K′, CEM over J scored
  0/32 in each of four configurations, and no subgoal was reached by tolerance (R28.77).
- **J's cost points the right way, coarsely.** Δ (R28.79): at the median state J ranks the
  demonstration's next commands ahead of every projected random candidate (m2 = 0.000). But CEM
  finds commands J scores cheaper than the demonstration's on 99 of 112 states (m1 = 0.116). At
  the pre-grasp pose those commands move the arm 0.048–0.055 rad RMS off it.
- **The pre-grasp pose is tolerant; the grasp is not** (the envelope, R28.90; privileged follower
  v1, 32 seeds per level). With the hand-over at subgoal 1 ("pre") the follower lifted 32/32 at
  every level up to 0.16 rad RMS. With the jaw closing where the arm stands ("carry") it lifted
  32/32 at 0.02 rad, 20/32 at 0.04 rad and 2/32 at 0.16 rad. With the pose corrected while closing
  ("reaim", reported only) it lifted 31/32 at 0.04 rad and 25/32 at 0.16 rad.
- **What has not been shown: that CEM over J reaches the pre-grasp neighbourhood from the reset
  pose.** On K′ the median closest arm RMS to subgoal 1 was 0.26–0.48 rad, and the best single
  episode came within 0.035 rad (R28.77). Δ's 0.05 rad offset was measured from the pose itself.
  This is the hybrid's main risk on the approach side. Arm W+F measures it on development seeds,
  and row S0-APPROACH (§7.7) stops the study before any gated stage if it fails.
- **The grasp precision S2 measured** (TASK-098's grasp-offset envelope for follower v1): 5 mm toward the fixed
  jaw, 10 mm along the face or upward, 20 mm away from the fixed jaw. J's predicted cube reads to
  about 1.6–1.8 cm (R28.39). This is why the close is given to a policy that reads the current
  frames at every step, not to a planner's prediction.
- **`so101-play-v1`'s test split has been read** (TASK-100). G_loc never opens it (§5.4). Its
  closed-loop evaluation uses fresh task resets, so no row depends on the corpus's held-out status.

## 1. Question and claim

**Question.** In simulation, on `so101_cube_lift_v1` and on fresh resets, does a hybrid controller
lift the cube? Its first phase is MPC by CEM over J toward the pre-grasp image subgoal of a per-reset
demonstration. Its second phase is a learned closed-loop grasp-and-lift policy G_loc that reads the
two camera frames and its own joint state. And does its success depend on the world-model approach,
that is, does it beat G_loc acting alone from the reset pose and G_loc after an action-blind
approach?

**The primary claim**, made only by row HY-PASS (§9): **"a hybrid with a `native_jepa`
world-model approach and a learned grasp policy succeeds"** at lifting the cube on
`so101_cube_lift_v1` (the lift scene with its inert target marker drawn inside the play corpus's
marker range) in simulation, with every qualifier of §14. Two qualifiers are part of the claim and
are never dropped when it is restated: **the approach is aimed at an image subgoal of a privileged
per-reset demonstration, and G_loc is cloned from privileged scripted play.** It is not "LeWM", not
"LeWM-driven", not "the world model grasps" and not "J needed" (R28.96).

## 2. Task, scene and scorer

`so101_cube_lift_v1` exactly as TASK-101's A1 declared it (TASK-101 §18.2;
`so101_tasks_lift_v1.py`, imported unchanged): `so101_cube_lift_v0`'s cube draws (x ∈ [0.18, 0.23] m,
y ∈ [−0.10, 0.10] m, yaw ±45°, `default_rng([9801, seed])`), the inert target marker from
`default_rng([10107, seed])` inside the play corpus's marker box, and the arm at `home_rad`.
**Success:** the cube centre at least 5 cm above its resting height with a gripper contact on every
step of 20 consecutive control steps; lost below z = −0.05 m or off the table; **at most 200 control
steps (10 s at 20 Hz)** per episode, whatever the controller. The scorer is the held-window scorer
on simulator truth, with no latching. The scene, the cameras (`wrist_rgb`, `overview_rgb`,
224 × 224), `configs/so101_sim_action.json` and `SO101Embodiment` are unchanged. A `stopped`
execution ends the episode as a failure and is reported. It is not a VOID.

## 3. The models

- **J-100101 and N_J-100101** exactly as TASK-101 §3: TASK-100's kept checkpoints, read only and
  hash-checked (file SHA-256 and implementation SHA-256; a mismatch is a VOID); the live latent is
  TASK-100's g4 pipeline (frozen DINOv2 ViT-S/14, each camera average-pooled to 4 × 4, wrist then
  overview, rounded to float16, standardised with the frozen fit-sample moments: 12 288 values).
  The primary J is J-100101 by TASK-101's rule. J-100100 and J-100102 are not used (§15).
- **G_loc**, the learned local grasp-and-lift policy, is new (§5.3–§5.4). It reads the same live
  latent, so one encoder pass per command serves both phases.

## 4. Goals

TASK-101 §4's goal procedure, unchanged: per reset, before any arm runs, the privileged follower v1
(`follow_lift`, `FOLLOWER_V1`, no offset) runs on the same reset with rendering on. Its subgoals are
the states after commands 40, 70, 90 and 120. The goal store keeps both frames, the encoded
latents and the measured joints. The reset is rejected and replaced from the cohort's reserve if the
demonstration fails, stops or terminates before command 120. One job writes and hashes a cohort's
goal store before any arm runs, and every arm's runner checks it.

**What the hybrid reads from the goal: subgoal 1 only** (the pre-grasp pose, the jaw open about
6 cm above the grasp point): its latent, as the planner's target, and its joints, for the hand-over
rule. Subgoals 2–4 are not read by H, N_J+G or W+F. They stay in the store, because the store's
format and hash are TASK-101's. G_loc reads no goal at all. It must find the cube in its own frames.

**What the goal gives away (fixed now).** Subgoal 1 shows the arm at the demonstration's pre-grasp
pose, and its joints place the jaw above the cube, so the approach does not localise the cube on its
own. H-pose+G (§6) uses the same joints with no image and no model, and it is the reported
comparator for "J needed" in the approach. From the hand-over on, nothing privileged is read: G_loc
sees frames and proprioception only.

## 5. The controller H

### 5.1 Phase 1: the world-model approach

TASK-101's controller W, **unchanged in every planner setting**, aimed at subgoal 1 only:
`SchemaCEMPlanner` over `joint_delta_so101_v0`, 256 samples, 6 iterations, 32 elites, minimum std
0.05, action penalty 0, replanned every command (MPC), with candidates projected by the
embodiment's chained `preview` (TASK-101 §5.2). **Configuration H8-mean** (horizon 8, step-mean
latent cost): the configuration TASK-101 §7.3's rule chose on K′, by its tie rule, and the one Δ was
measured at. It is not recalibrated here (§15). Planner seed `10110 · 10⁶ + reset seed` (salt 10110;
the same seed for every CEM arm on a reset).

### 5.2 The hand-over rule and τ_h

**Hand-over.** Phase 1 ends and G_loc takes over when, **for 3 consecutive commands**, the measured
arm joints are within **τ_h rad RMS** (over the five arm joints) of subgoal 1's joints and the
measured jaw is within **0.15 rad** of subgoal 1's jaw, **or after T₁ = 60 commands** (TASK-101's
timeout for subgoal 1), whichever comes first. On a timeout G_loc takes over from wherever the arm
is. The episode keeps the task's single 200-command budget, so G_loc has 200 minus the commands
phase 1 used. The rule reads proprioception and the goal pose only.

**τ_h, by a rule fixed now, in two steps.**

1. **The envelope's limit (already measured; R28.90).** τ_env is the largest envelope level L with
   every level up to and including L lifting at least 30/32 in the "pre" column (TASK-101 §19.4's
   pass bar, applied to "pre" alone). On the recorded counts, **τ_env = 0.16 rad**. The cap is then
   **τ_max = the next lower declared level, 0.12 rad**: one level of margin, because the follower
   re-aims its descent from simulator truth and G_loc must re-aim from images. *Disclosed:* this
   rule is written after the envelope's counts were published. It reads only the "pre" column, the
   column the hand-over at subgoal 1 corresponds to, and it takes the margin below the measured
   limit, never above it.
2. **G_loc's own tolerance (Stage 0, cohort P; §7.4).** The privileged scripted approach drives the
   arm to subgoal 1's joints plus an offset δ of RMS exactly r, for r ∈ {0, 0.04, 0.08, 0.12} rad
   (directions as the envelope's, salt 10112), and G_loc runs from there. Let k(r) be the lifts
   of 32. **τ_h is the largest r ∈ {0.04, 0.08, 0.12} such that every level from 0.04 up to r has
   k(r) ≥ k(0) − 3.** If no level qualifies, τ_h = 0.04 rad (TASK-101's value), and the Stage 0
   record says so. τ_h never exceeds τ_max.

The margin of 3 of 32 allows sampling noise at the level k(0) reaches. It does not make a looser
tolerance free: a looser τ_h lets phase 1 hand over earlier at a larger error, and the P counts
measure what that error costs G_loc.

### 5.3 Phase 2: G_loc

**Inputs, at every command, from what a real controller would have:**

- **z**, the live latent of the current frames (12 288 standardised values; §3), the same array
  phase 1 encodes;
- **proprioception**: the 6 measured joint positions and 6 measured joint velocities (the
  `so101` state vector) and the 6 current joint-position targets (the controller's own command
  history);
- **a clock**: t, the commands since the hand-over, as t / 100 plus sin and cos at 8 geometric
  periods from 4 to 512 commands (17 values; P-3's form, scaled to this task's phase lengths).

The non-latent inputs (35 values) are standardised by G_loc's training-window moments (std floored
at 10⁻³) and frozen with the checkpoint. z is already standardised. **No object truth, contact
truth, goal or phase label is an input.** A test checks that the policy object never receives task
truth (a sentinel, TASK-101's form).

**Head** (P-3's): LayerNorm → Linear(12 323, 512) → SiLU → Linear(512, 512) → SiLU → Linear(512,
512) → SiLU → Linear(512, 6). The 6 outputs are the normalised `joint_delta_so101_v0` command (5 arm
deltas and the absolute gripper channel). They are clipped to [−1, 1] by the policy before sending,
because the contract validates and never clips. The embodiment then applies its own joint-limit
and gripper-rate clipping as for every arm.

**G_loc must correct the pose while it closes.** That is not an input or a rule. It is a property
the training data must contain and the closed loop measures. The corpus contains corrections: the
collector's descents track an IK goal under Ornstein–Uhlenbeck noise and, in the perturbed mode,
bursts of random joint deltas, with 2 mm aim noise at the grasp (TASK-099). What the corpus does
not contain is the arm moving during the close itself: the collector holds the arm's target while
the jaw closes. G_loc may therefore learn to place the jaw before closing rather than to re-aim
while closing. Both satisfy the envelope's requirement (about 0.02 rad at the close, or correction
while closing). Which one G_loc does is reported (§10), not gated.

### 5.4 G_loc's training (behaviour cloning; Stage 0, before any closed loop is counted)

- **Data: `so101-play-v1`, train split only, episode-grouped** (the corpus's frozen
  `sha256("9902:<seed>")` split; 2 700 train episodes). **Val (150 episodes) is used only for
  checkpoint and seed selection. The test split is never opened**: the loader refuses the test
  split's files, and a test checks it.
- **Windows.** One window per pick attempt that **grasped** (the corpus's per-attempt fact: at the
  end of its lift, the cube at least 2 cm up with both jaws in contact). Regrasp attempts count. The
  window runs from the **first command of the attempt's last contiguous `descend` run** (the
  collector's pre-grasp pose, 5–8 cm above the grasp point; t = 0 for the clock) to the attempt's
  `lift_end_step`, plus the immediately following `hold` commands, if any, up to 40. Attempts
  injected with `early_close` are excluded (their close is a scripted error), as are attempts
  with no `lift_end_step`. The success label selects windows only. It is never an input. *This is
  filtered behaviour cloning, disclosed:* G_loc imitates the collector's grasps that worked.
  Expected size, from TASK-099's counts: about 3 300 windows of about 60–100 commands each,
  roughly 200 000–300 000 frames. Stage 0 records the exact counts.
- **Features.** The latents are read from TASK-100's pretrained g4 feature store
  (`outputs/task100-features-pretrained`, HDD; read only; every file hash-checked against TASK-100's
  feature report). Proprioception, targets, phase labels and attempt facts are read from the corpus
  (`data/so101-play-v1`, hash-checked against `so101-play-v1.json`). A guard checks, for every
  window, that the store's actions and phase labels equal the corpus's on the same frames.
  Nothing is featurised again, and no new GPU featurisation runs.
- **Label:** the stored executed (`applied_action`) command at that step. The model learns
  executable commands (TASK-101 §5.2's reason).
- **Loss:** mean squared error over the 6 channels.
- **Training:** 30 000 updates, AdamW, learning rate 3 × 10⁻⁴ cosine to 3 × 10⁻⁵, weight decay
  10⁻⁴, gradient clip 1.0, batch 256, uniform over training frames, on CUDA under
  `scripts/gpu_run.sh --wait --board`.
- **Three seeds:** G-0, G-1 and G-2, with initialisation and batch order from
  `default_rng([10115, i])`, i = 0, 1, 2.
- **Selection.** Every 1 000 updates, by val MSE on the val windows (built by the same rule). A
  checkpoint is eligible if every output dimension's std over val frames is at least 0.02 and its
  update is > 0. **The primary G_loc is the seed whose selected checkpoint has the lowest val MSE.**
  The other two are replication policies, reported only. A seed with no eligible checkpoint is
  reported as such. If all three have none, the row is S0-GLOC-WEAK (§7.7).
- **Saturation (reported only).** Whether the selected update is one of the last two evaluation
  points (`last_two_triggered`, TASK-074's flag). Not a gate: closed-loop success, not val MSE, is
  what the stages read. *Declared weakness:* val action error is a weak proxy for closed-loop
  success (P-3's §5.2).

## 6. Arms (paired: the same resets, the same goal store)

| Arm | Phase 1 (from the reset pose) | Phase 2 | Role |
|---|---|---|---|
| **H** | CEM over J-100101 toward subgoal 1, §5.2's hand-over | primary G_loc | **primary** |
| **G_loc-only** | none | primary G_loc from the reset pose, clock from 0, for all 200 commands | control: no approach |
| **N_J+G** | the same planner over N_J-100101 (zero commands to the model for every candidate), §5.2's hand-over | primary G_loc | control: action-blind approach |
| **S+G** | privileged scripted approach: the goal demonstration's own first 40 commands, replayed | primary G_loc, from command 41 | G_loc's ceiling given an exact approach (reported; enters the bar's cap) |
| **W+F** | as H | privileged follower v1 from the hand-over pose: its descent waypoints, close and lift, aimed from simulator truth (the envelope's "pre" path) | the world-model approach's ceiling given a privileged grasp (reported; enters the bar's cap) |
| **F** | the privileged follower v1 for the whole episode | — | determinism guard on the goal pipeline (TASK-101 §6); scores every kept reset by construction |
| H-pose+G | TASK-101's non-learned pose follower toward subgoal 1's joints (no image, no model), §5.2's hand-over | primary G_loc | "J needed" for the approach; reported only |
| **rand** | every channel uniform on [−1, 1] each command (salt 10111) | — | floor |
| **hold** | `hold_action()` every command | — | floor |
| H-G1, H-G2 | as H | replication G_loc seeds | reported only; dropped first if the budget requires |

- **N_J+G is weak by construction, disclosed now (TASK-101 §6).** N_J's predictions do not depend on
  the candidate, so every candidate ties, the planner keeps its initial mean, and N_J executes the
  zero command (arm held, jaw half open). With the arm held at home, the hand-over fires only if the
  home pose is already within τ_h of subgoal 1, so N_J+G is in practice G_loc from the home pose
  after a 60-command timeout, with 140 commands left. The
  ties are exact by construction (one roll-out broadcast to every candidate; TASK-101's adapter).
  G_loc-only is the stronger "no approach" control, because it has all 200 commands.
- **What H's two controls isolate.** H against G_loc-only asks whether the world-model approach
  helps compared with G_loc doing everything from the reset pose. H against N_J+G asks whether the
  approach's actions matter. G_loc was trained only from pre-grasp poses, so both controls may be
  near zero. Then H beating them shows that an approach to the pre-grasp pose is needed and that
  J's approach supplies it. It does not show that J's approach is better than a non-learned one:
  that is H against H-pose+G, reported only (§10).
- **F is not a measured ceiling, by construction** (TASK-101 §6): it replays the goal demonstration,
  and failed demonstrations are replaced, so it lifts on every kept reset. F's executed actions or
  outcome differing from the demonstration's on any kept reset is a guard failure.
- **S+G and W+F are the measured component ceilings.** H is in practice limited by both its
  approach (which W+F measures with a privileged grasp) and its grasp (which S+G measures with a
  privileged approach). This is not a strict bound: a reset where W+F fails could still be one where
  G_loc lifts. That is why the bar uses them only as a cap with a 0.9 factor (§7.6).

## 7. Stage 0 (development; before the freeze)

Stage 0 runs on **development cohort P** (103000–103031, reserve 103032–103047), **development
cohort D** (103100–103163, reserve 103164–103199), the fresh replacements P′ and D′ of §7.7 row 1 if
called for, and debug seeds 103900–103999 for plumbing (never counted). No gated seed is drawn,
rendered or run before the freeze.

### 7.1 Code and tests (new files only)

New opt-in modules (no registry entry; `import embodied_jepa` unchanged and torch-free, checked by a
test): `so101_hybrid.py` (NumPy: cohorts, seeds, salts, the hand-over rule, the τ_h rule, the
window rule, the arms' composition, statistics, the bar rule and the rows),
`so101_grasp_policy.py` (torch: G_loc's head, input assembly, clock, training loop and selection)
and `so101_hybrid_runtime.py` (torch: the hybrid episode loop, phase 1 through TASK-101's imported
runtime, G_loc's live path, the follower-from-hand-over path, the worker pool); scripts
`train_task102.py`, `run_task102.py` (subcommands `goals`, `episodes`, `check`),
`summarize_task102.py` and `run_task102_stage.sh` (one `scripts/gpu_run.sh --wait --board` lock per
GPU job). A new manifest `benchmarks/manifests/so101-hybrid-lift-v1.json` pins this task's files at
the freeze. **Byte-identical:** every G1 byte-pinned file, every TASK-100 file pinned by
`so101-world-model-v1.json`, every TASK-101 file pinned by `so101-closed-loop-lift-v1.json` (its
manifest stays `CANDIDATE`), `so101_switch_envelope.py`, `schema_planning.py`, `so101.py`,
`so101_simulation.py`, `so101_tasks.py`, `so101_followers.py`, `so101_play_corpus.py` and
`config.py`. If a TASK-102 test needs an exemption in an existing test, that is a change to an
existing file, and the Stage 0 record discloses it (TASK-101's deviation 1 precedent).

Tests: the hand-over rule in both directions and its timeout; the τ_h rule and the bar rule on
synthetic counts, in every branch; the window rule on synthetic attempts (grasped or not, regrasps,
`early_close`, missing `lift_end_step`, a following `hold` run longer than 40); the loader refuses
the test split; train, val and test windows share no episode; the store-corpus alignment guard;
G_loc's live `act()` equals the batched offline prediction within 10⁻⁴; the policy never receives
task truth (sentinel); the follower-from-hand-over path, handed over at the demonstration's exact
command-40 state, reproduces the demonstration's commands 41 onward bit for bit (guard G-zero);
S+G's replayed approach equals the demonstration's first 40 commands; seeds outside a cohort are
refused; a cohort cannot be run twice into one output; hash mismatches are refused; McNemar,
bootstrap and every row of §7.7 and §9 in both directions on synthetic counts; the core-import
check; the byte-identity checks above.

### 7.2 G_loc's training (§5.4)

One GPU job per seed, before any closed loop of P or D. The Stage 0 record states the window counts
per split, the selected update and val MSE per seed, the eligibility outcome, `last_two_triggered`,
the primary seed and the checkpoint hashes. Checkpoints go to `checkpoints/task102-stage0/G-<i>/`
(new names only).

### 7.3 Checks (each failure is S0-VOID until fixed under its own record)

TASK-101's **ordering rule and check ledger** apply (TASK-101 §18.5): every check below is in the
ledger as passing, at the running revision and from a clean tracked tree, before a cohort's first
episode, and the runner refuses otherwise.

1. **Live latent = stored latent** (TASK-101 7.2.1's test on 64 val frames per camera).
2. **The live latent reads the scene**, on each cohort's own goal store (TASK-101 7.2.2: R-enc
   within B_cube = 2.397 cm in median).
3. **Projector** (TASK-101 7.2.3, on P's reset states).
4. **G_loc's live path** equals its batched offline prediction within 10⁻⁴ on 256 stored val frames.
5. **Determinism**, read in TASK-101 7.2.4's conditional form as A1 amended it (TASK-101 §18.5),
   because EGL rendering is not bit-stable on some states: one H episode and one S+G episode, each
   run twice, and four H episodes in a pool of 1 and in the pool size used.
6. **Timing:** median and 95th-percentile seconds per command in phase 1 and in phase 2, the worker
   count, and the projection of P, D and S (§13). Every command finishes inside the embodiment's
   30 s wall-clock deadline.

### 7.4 P: G_loc's tolerance at the hand-over, and τ_h

On P's 32 resets: S+G with the scripted approach driven to subgoal 1's joints plus δ, for
r ∈ {0, 0.04, 0.08, 0.12} rad (32 × 4 = 128 G_loc episodes; δ = r · u with u = n / RMS(n),
n ~ N(0, I₅) from `default_rng([10112, seed])`, one direction per seed for every level). The
approach is the demonstration's 40 commands with the pre-grasp target offset by δ, which is the
envelope's "pre" construction. Guard: at r = 0 it equals the demonstration's commands exactly.
Recorded: k(r) of 32 with exact 95 % intervals, and the achieved hand-over RMS. **τ_h follows from
§5.2 step 2, mechanically.** P's counts set τ_h and nothing else. If k(0) < 16/32, the row is
S0-GLOC-WEAK (§7.7), and D does not run.

### 7.5 D: the development closed loop

Every arm of §6 except H-G1 and H-G2 runs on D's 64 resets with τ_h from P: H, G_loc-only, N_J+G,
S+G, W+F, F, H-pose+G, rand and hold. The Stage 0 record reports each count with its exact 95 %
interval, the per-phase decomposition (§10) and timing. **D changes no knob.** It sets the bar
(§7.6) and the Stage 0 row (§7.7).

### 7.6 The bar (rule fixed now, applied to D's numbers)

Let r_SG and r_WF be S+G's and W+F's success rates on D's 64 kept resets, and LCB(H(D)) the lower end
of the exact 95 % interval of H's D count of 64.

**c = 0.9 × min(r_SG, r_WF); b = min(c, max(0.50, LCB(H(D)))); the gated bar is ⌈b × 128⌉ of 128.**

- **c, the cap, from measured component ceilings:** no bar may ask more than 90 % of the weaker of
  H's two components, each measured with its other half privileged (TASK-074's lesson: bars from
  measured ceilings, TASK-088's 0.9 factor). Unlike TASK-101's cap, this one is measured, not 0.90
  by construction.
- **0.50, the claim floor:** "succeeds" means at least half of the resets (TASK-101 §7.5).
- **LCB(H(D)):** between the floor and the cap, the gated stage asks H to confirm the level its
  development run showed, at the conservative end of that run's interval.
- If c < 0.50, no bar at or above the claim floor is admissible: row S0-CEILING (§7.7).

What the rule gives when the cap does not bind (exact binomial; power = P(X ≥ bar) at a true rate p):

| H(D) | LCB | b | bar of 128 | power at p = H(D) | power at p = H(D) − 0.05 |
|---|---|---|---|---|---|
| 34/64 | 0.40 | 0.50 | 64 | 0.79 | 0.37 |
| 40/64 | 0.50 | 0.50 | 64 | > 0.99 | 0.96 |
| 48/64 | 0.63 | 0.63 | 81 | > 0.99 | 0.96 |
| 56/64 | 0.77 | 0.77 | 99 | > 0.99 | 0.95 |

(Carried from TASK-101 §7.5's identical binomial computation. The Stage 0 record recomputes the
table with c applied.)

### 7.7 Stage 0 rows (first match) and the freeze

1. **S0-VOID** — a check of §7.3 failed, a guard of §12 failed, G_loc's training job failed, or a
   run did not complete. Fixed under its own record. Same-seed reruns are allowed only for a VOID
   raised before any success count of the affected cohort was read. **Once P's counts have been
   read, a fix that changes code, configuration or G_loc is rerun on P′ (103060–103091, reserve
   103092–103099); once D's counts have been read, on D′ (103600–103663, reserve 103664–103699).**
   P′ and D′ are used at most once each. A second post-count fix needs a ruling that names further
   seeds.
2. **S0-GLOC-WEAK** — no G_loc seed has an eligible checkpoint, or k(0) on P is below 16/32, or
   S+G on D is below 32/64. The learned grasp does not lift half the time even after an exact
   approach. Escalate. Named candidates, each needing its own amendment or preregistration: DAgger
   with the privileged follower's "reaim" path as the labeller (P-3's precedent), more or different
   training windows, or a wrist-only input. No gated stage.
3. **S0-APPROACH** — W+F on D is below 32/64. The world-model approach does not deliver the
   pre-grasp pose within τ_h, even with a privileged grasp after it. Escalate. Named candidates:
   R28.57's LeWM retrain, a cost or goal change for the approach, a planner variant. No gated stage.
4. **S0-CEILING** — c < 0.50 although neither row 2 nor row 3 fired (min(r_SG, r_WF) between 0.50
   and 0.555). Escalate. No gated stage.
5. **S0-LOW** — the bar's power at H's D rate is below 0.80. Escalate. A ruling either runs the
   gated stage as frozen, with its power stated, or closes with a development decomposition. No knob
   changes on these seeds.
6. **S0-OVER** — the projected gated stage does not fit the cap even without H-G1 and H-G2 (§13).
   Escalate (cloud GPU time is the owner's decision).
7. **S0-DONE** — otherwise. The Stage 0 record states G_loc's training outcome and the primary seed,
   τ_h with P's counts, D's counts for every arm, c, b and the bar count, H-pose+G's D count beside
   the bar, the demonstration's rejection counts on P and D, the recomputed power table (bar and
   McNemar, from D's rates) and the projection. **The protocol freezes when the Stage 0 PR
   merges**, and the gated stage then runs without a further ruling unless a declared rule escalates.

## 8. Gated stage S

- **Cohort: 128 fresh resets, seeds 103200–103327** of the lift stream on `so101_cube_lift_v1`;
  reserve for §4's replacement 103400–103463. Goals first (one GPU job), then every arm of §6
  including H-G1 and H-G2, at the frozen revision from a clean run tree
  (`scripts/new_worktree.sh <dir>-run --from <frozen sha> --run`), **one run**. Each cohort and arm
  runs once. A second run into a new output name is refused unless a VOID ruling names fresh seeds.
- **Why 128 (power).** The bar alone would need fewer resets: at the 0.50 floor, a true rate of 0.60
  meets the bar with probability 0.96 at 64 resets and 0.99 at 128; 0.55 gives 0.82 and 0.89. The
  McNemar tests decide the size. Computed for this draft (exact one-sided test at p < 0.01, 20 000
  trials, independent coupling, `default_rng(10114)`):

  | H's rate vs a control's | n = 64 | n = 96 | n = 128 |
  |---|---|---|---|
  | 0.5 vs 0.3 | 0.41 | 0.64 | 0.79 |
  | 0.6 vs 0.3 | 0.82 | 0.96 | 0.99 |
  | 0.5 vs 0.2 | 0.86 | 0.98 | 1.00 |
  | 0.6 vs 0.1 | 1.00 | 1.00 | 1.00 |

  G_loc-only and N_J+G are expected to be low, since G_loc was trained only from pre-grasp poses,
  and then 64 would suffice. But that expectation is exactly what the tests check, and a control
  near 0.3 needs 128 for power near 0.8. TASK-081/083 and TASK-101 used 128 as well. The Stage 0
  record recomputes the power from D's rates.

## 9. Rows (first match) and what each leads to

k_X is arm X's successes out of 128. "Beats" is an exact one-sided McNemar test on the paired
resets at **p < 0.01**. The four tests of row 2 are conjunctive (an intersection-union test), so no
multiplicity correction is needed.

1. **VOID** — a guard of §12 failed (including F's executed actions or outcome differing from the
   goal demonstration's on any kept reset), a run did not complete, a stage exceeded 1.5 × its
   measured projection, or a cohort was run twice. Nothing is read. One repeat on fresh seeds named
   by a ruling. A second VOID closes the task **INCONCLUSIVE**.
2. **HY-PASS** — k_H ≥ bar, **and** H beats each of **G_loc-only, N_J+G, rand and hold**. The
   primary claim, "a hybrid with a `native_jepa` world-model approach and a learned grasp policy
   succeeds", with §14's qualifiers.
3. **HY-UNSPECIFIC** — k_H ≥ bar, but H does not beat at least one of the four. The hybrid lifts, but
   its success is not shown to depend on the approach (G_loc-only or N_J+G) or on acting at all
   (rand, hold). The record names which test failed. Escalate. No claim.
4. **HY-LOW** — k_H < bar, and H beats rand and hold. Escalate. No claim.
5. **HY-FAIL** — otherwise (H does not beat rand or hold).

The rows are exhaustive: rows 2–3 split k_H ≥ bar by the tests, and rows 4–5 split k_H < bar by
the floors.

What each row leads to (recommendations; the owner, or Claude under delegation, rules):

- **HY-PASS:** the place study on `so101_cube_place_v0` (or its v1 counterpart with the marker in
  the corpus's range) with the hybrid as its pick, under its own preregistration with its own goals,
  bars and controls. R28.57's LeWM retrain stays the route to a LeWM-driven SO-101 claim. If LeWM
  passes S4, it can replace J in phase 1 as a one-key swap, under its own study. "J needed" is not
  claimed: k_H-pose+G and k_S+G stand beside k_H.
- **HY-UNSPECIFIC, HY-LOW, HY-FAIL:** **a development decomposition record comes before any second
  variant** is preregistered (TASK-101 §9's form). It uses S's own episode logs and new development
  seeds only, with no training. It reads, per arm: hand-over reached or timed out, and the
  hand-over RMS; G_loc's jaw offset from the cube at the close, in the grasp frame, against S2's
  envelope; the arm's motion during the close (re-aimed or not); contact, rise and the step of
  success. It reads S+G and W+F beside H to name the failing component before any variant is
  chosen.
- **No abandonment clause.** This is the first hybrid on SO-101: one world model, one policy recipe,
  one configuration. TASK-101 §9's reasoning carries: on G1, clauses fired only after a declared
  variant had also failed.
- **Every row:** R7 does not change (R28.9). Nothing is re-thresholded, re-run, re-selected or
  re-chosen after S's numbers are seen. Every arm's count is reported with its interval.

## 10. Reported only (no gate)

- Every arm's count with its exact 95 % interval. Every paired difference against H with its
  discordant counts and McNemar p.
- **Non-inferiority against F:** k_H − k_F with a reset-bootstrap 95 % interval (10 000 resamples,
  salt 10113); "not detectably inferior beyond δ" if its lower bound is above **−16/128**
  (TASK-081's allocation, carried, not calibrated). F scores every kept reset by construction, so
  this is H's failure count against a constant arm. Read by no row.
- **"J needed" for the approach:** H against H-pose+G, paired difference with bootstrap interval and
  two-sided exact McNemar p. Read by no row.
- **Component readings:** S+G and W+F on S, against H.
- **Replication:** H-G1 and H-G2 against the bar, and McNemar against G_loc-only, rand and hold.
- **Per-phase decomposition**, per arm: phase-1 commands used, hand-over by tolerance or timeout,
  hand-over arm RMS and jaw error; in phase 2, the jaw's offset from the cube centre in the grasp
  frame (closing, along, up) at the first command with the jaw command below 0 and at the first
  both-jaw contact, against S2's envelope; arm RMS motion between the start of closing and the
  first both-jaw contact (whether G_loc re-aims while closing, §5.3); contacts, maximum rise, the
  step of success, lost cubes, `clipped` and `stopped` counts.
- **The demonstration's rejection count** on S's drawn seeds, with reasons.
- Seconds per command per phase (median, 95th percentile), GPU hours per job, worker counts.

## 11. Statistics

Exact (Clopper–Pearson) 95 % intervals for every count. Exact one-sided McNemar tests on discordant
pairs (binomial with p = 0.5). The reset-bootstrap interval of a paired difference resamples the
128 resets with replacement (10 000 resamples, salt 10113) and takes the 2.5th and 97.5th
percentiles. Power for the Stage 0 record by exact binomial sums (bar) and simulation with salt
10114 (McNemar; 20 000 trials per cell; overlap, half and independent couplings).

## 12. Seeds, salts, guards

- **Seeds** (the lift stream, `default_rng([9801, seed])`, with the v1 marker from
  `default_rng([10107, seed])`): P 103000–103031 (reserve 103032–103047); P′ 103060–103091
  (reserve 103092–103099); D 103100–103163 (reserve 103164–103199); **gated S 103200–103327**
  (reserve 103400–103463); D′ 103600–103663 (reserve 103664–103699); debug 103900–103999. P′ and D′
  are used only under §7.7 row 1. **Checked on 2026-10-19 with `git grep -w` over every remote ref**
  (`docs/`, `src/`, `scripts/`, `tests/`, `benchmarks/`, `configs/`, `.mc/`): no six-digit number in
  103000–103999 appears as a seed. The only matches are the digits of two decimals in
  `apple_arrival_feedback_results_v1.md` ("0.103209", "0.103606"). Every TASK-101 range (101010–102299)
  is retired (R28.93) and disjoint from these. S2 used 100–163, 1000–1015 and 0–63.
- **Salts:** 10110 (CEM planner seeds), 10111 (rand), 10112 (P's offset directions), 10113
  (bootstrap), 10114 (power simulation), 10115 (G_loc initialisation and batch order). None appears
  on any remote ref in the same check. TASK-101's 10101–10109 and TASK-100's 10011–10015 are not
  reused, except 10107, which is part of `so101_cube_lift_v1`'s definition.
- **Guards (each a VOID):** the tracked tree clean and at the frozen revision at every job's start
  and end; this task's manifest, TASK-100's `frozen_files_check` and TASK-101's pinned-file hashes;
  J's, N_J's and G_loc's checkpoint file and implementation hashes; the DINOv2 weight pins; the
  fit-moments hash; TASK-100's feature-store hashes and the corpus manifest (training only); the
  store-corpus alignment of every training window; the goal-store hash read by every arm; seeds
  inside their cohort; no cohort run twice; every latent, cost, policy output and action finite;
  F's executed actions and outcome equal to the goal demonstration's on every kept reset; G-zero for
  the follower-from-hand-over path and for P's r = 0 approach; every GPU job under
  `scripts/gpu_run.sh --wait --board`.

## 13. Budget, disk and evidence

**Hard cap: 12 GPU h in total, Stage 0 included; Stage 0 itself capped at 5 GPU h.** GPU hours are
lock time. CPU arms (F, rand, hold) need no frames to act and run without rendering, except their
first 8 episodes on each cohort, which render for the evidence inside that cohort's episode job.
Arms with a CEM phase or G_loc run as a pool of worker processes inside one lock, each episode on
its own seed and model state.

| Stage | Work | Estimate (GPU h; estimates, not measurements) |
|---|---|---|
| G_loc training | 3 seeds × 30 000 updates on stored features | 0.3–1.0 |
| Checks, debug, goals for P and D | — | 0.3–0.6 |
| P | 128 S+G episodes (G_loc only, no CEM) | 0.2–0.5 |
| D | 64 × (H, G_loc-only, N_J+G, S+G, W+F, H-pose+G) with frames | 0.7–2.0 |
| S | goals; 128 × (H, G_loc-only, N_J+G, S+G, W+F, H-pose+G, H-G1, H-G2) | 2.0–4.5 |

The estimates use TASK-101's measured throughput: K′'s CEM commands projected Stage S at 5.86 GPU h
for 768 episodes of up to 200 commands, about 0.14 s of lock time per CEM command with a pool. Here
CEM runs for at most 60 commands per episode, and a G_loc command costs one encoder pass and an MLP,
estimated at 0.02–0.05 s with a pool. Stage 0 replaces the estimates with measurements. If S's
measured projection does not fit the cap's remainder, H-G1 and H-G2 are dropped first (reported only),
and the drop is recorded. If S still does not fit, the row is S0-OVER. Exceeding a stage's measured
projection by more than 1.5 × stops the run (VOID). Before any job longer than an hour, the session
tells the V2D session and respects the `- v2d:` lines under "Requests" on
`~/.local/state/gpu/BOARD.md`.

**Disk and evidence.** G_loc's training reads about 5–8 GB of stored float16 latents from the HDD
through memory maps (30 GB RAM on the PC). Checkpoints are small (about 25 MB each). Goal stores and
per-step logs are small, and frames are kept only for the first 8 episodes of each arm on each
cohort. Run outputs go under `outputs/task102-*`, checkpoints under `checkpoints/task102-*`, and
evidence to `~/develop/emai/evidence/task102-*/` with `SHA256SUMS`. Every name is new. Nothing under
`data/`, `checkpoints/` or `outputs/` is overwritten. TASK-100's feature stores and checkpoints and
TASK-101's outputs are read only.

## 14. What this cannot show

- **Simulation only, one run.** One scene (a red 3 cm cube on a tan table, fixed lighting), one
  task, one robot model, MuJoCo soft contacts. Nothing about hardware.
- **The approach's goal comes from a privileged demonstration of the same reset.** Subgoal 1 shows
  the arm at the demonstration's pre-grasp pose, and the hand-over rule reads its joints. A success
  shows that the planner drove the arm near that pose and that G_loc grasped from there. It does not
  show that the world model localised the cube.
- **G_loc is cloned from privileged scripted play.** Its demonstrations come from a collector that
  read simulator truth, filtered to grasps that worked. G_loc reads no truth at run time, but its
  competence is the collector's, within the corpus's distribution (top-down grasps, the corpus's
  reset and marker boxes, partial joint coverage).
- **`native_jepa`, not LeWM.** Whether LeWM would do as well in phase 1 is untested (R28.57).
- **"J needed" is not tested by any row.** H-pose+G (the goal pose, no image, no model) and S+G (a
  privileged approach) stand beside H. If H-pose+G lifts as often, the goal pose alone supplies what
  J's approach supplied.
- **The controls are weak where they are expected to be weak.** N_J+G executes the zero command and
  then G_loc from home (§6). G_loc-only starts from a pose G_loc never trained from. H beating them
  shows that an approach is needed and that J's supplies one, not more.
- **One configuration of each part:** H8-mean, carried from TASK-101 and not recalibrated; one
  G_loc recipe; τ_h from one development cohort of 32 resets per level.
- **τ_h's cap was written after the envelope's counts were published** (§5.2). It reads only the
  "pre" column and takes a margin below it.
- **F is not a measured ceiling** (§6). S+G and W+F are measured with one half privileged each, and
  they bound H only approximately.
- **Not bit-reproducible** (the EGL renderer; §7.3's conditional determinism).
- **In-distribution scenes only** (`so101_cube_lift_v1`; TASK-101 §18.4).
- **SO-101 results are not Apple→Plate results.** R7 does not change.

## 15. Considered and not adopted

- **R28.57's LeWM retrain first:** R28.92. It stays a separate recommendation.
- **DAgger for G_loc** (P-3 had three rounds): it would label G_loc's own visited states with a
  privileged expert. It is the first named candidate under S0-GLOC-WEAK. It is not adopted up front,
  because the first question is whether filtered cloning from a fixed, sealed corpus already
  grasps, which keeps the training data independent of any closed loop.
- **A non-learned grasp (the follower) in H:** that is W+F, a ceiling arm. The study's point is a
  learned grasp.
- **A hand-written jaw rule on top of J for the whole task:** TASK-101 §6 considered it. A rule
  cannot correct the pose while closing from images.
- **Recalibrating phase 1 (horizon, cost form, samples) or using J-100100/J-100102:** that would turn
  TASK-101's knobs on new seeds. The approach is measured as TASK-101 left it, and W+F tells whether
  it suffices.
- **Wrist camera only for G_loc:** the wrist view shows the cube large between the jaws at the grasp,
  but the overview shows height and the arm. The two-camera latent is the one whose live path has been
  checked bit for bit (TASK-101 7.2.1), and it is shared with phase 1. Wrist-only is a named
  S0-GLOC-WEAK candidate.
- **A grasp target from J's predicted-cube readout:** R28.39 advises against it (1.6–1.8 cm).
- **Training G_loc from the reset pose, so that it could do the whole task:** then H against
  G_loc-only would compare two full controllers, and the hybrid's division of labour, which the
  envelope motivates, would not be tested.
- **An abandonment clause:** §9.

## 16. Process

DRAFT (this PR: docs and cards only) → independent review → merge → Stage 0 PR (code, tests, G_loc's
training, checks, P, D, τ_h, the bar and the projection; the record
`docs/experiments/so101_hybrid_lift_v1_stage0.md` and a DECISIONS entry) → independent review →
**freeze at its merge, on S0-DONE only** → Stage S in a clean run tree → results document
`docs/experiments/so101_hybrid_lift_v1_results.md` and a DECISIONS entry with the row → review →
merge. The executing agent reports rows. It does not choose the next study.

## 17. Rulings this document records (R28.98–R28.109, DRAFT; decided by Claude under owner delegation)

- **R28.98** — TASK-102: a hybrid closed loop on `so101_cube_lift_v1` in simulation; task, resets,
  scorer and the 200-command budget as TASK-101's A1 (§2).
- **R28.99** — models: J-100101 and N_J-100101, unchanged and hash-checked; the live latent shared by
  both phases (§3); TASK-101's goal store format, with only subgoal 1 read by the approach (§4).
- **R28.100** — phase 1: TASK-101's planner at H8-mean, unchanged, toward subgoal 1; hand-over by
  τ_h RMS arm, 0.15 rad jaw, 3 commands, or a 60-command timeout (§5.1, §5.2).
- **R28.101** — τ_h: capped at 0.12 rad (one level below the envelope's "pre" limit of 0.16 rad),
  then set on P by "every level up to r within 3 of k(0)", with 0.04 rad as the fallback (§5.2,
  §7.4).
- **R28.102** — G_loc: P-3's head and recipe on the J latent plus 18 proprioceptive values and a
  17-value clock; executed commands as labels; filtered cloning on grasped attempt windows of the
  train split; val for selection; the test split never opened; three seeds, the primary by the
  lowest val MSE (§5.3, §5.4).
- **R28.103** — arms H, G_loc-only, N_J+G, S+G, W+F, F, H-pose+G, rand, hold, H-G1 and H-G2, with
  their roles (§6).
- **R28.104** — Stage 0: training, checks under TASK-101's ledger, P for τ_h, D for the bar;
  rows S0-VOID, S0-GLOC-WEAK, S0-APPROACH, S0-CEILING, S0-LOW, S0-OVER and S0-DONE; P′ and D′ for
  post-count fixes; the freeze only on S0-DONE (§7).
- **R28.105** — the bar: c = 0.9 × min(r_SG, r_WF), b = min(c, max(0.50, LCB(H(D)))), ⌈b × 128⌉ of
  128 (§7.6).
- **R28.106** — the gated stage: 128 fresh resets, one run, every arm (§8).
- **R28.107** — rows VOID, HY-PASS, HY-UNSPECIFIC, HY-LOW and HY-FAIL (first match); exact one-sided
  McNemar at p < 0.01 against G_loc-only, N_J+G, rand and hold; a decomposition before any second
  variant; no abandonment clause; the pass's name (§9, R28.96).
- **R28.108** — reported only: non-inferiority to F at δ = 16/128, H against H-pose+G, the component
  arms, the replication seeds and the per-phase decomposition (§10).
- **R28.109** — seeds, salts, guards and budget: §12 and §13; a 12 GPU h hard cap with Stage 0 at
  most 5.
