# TASK-089 — zero-shot grasp-and-lift: an offline diagnosis (Part A), then a closed-loop test with one declared change (Part B)

**Status: FROZEN at the merge of its Stage 0 PR** (R26.1–R26.23, decided by Claude under owner
delegation; the owner said "go" on 2026-10-11; the DRAFT was reviewed and merged at #197, `5a7c82a`; Stage 0's
record is §16). After that merge nothing in §2–§11 changes without its own reviewed ruling.

This is the grasp-focused task that DECISIONS R25.27 recommends after TASK-088 closed **Z3-REACH**
([zero_shot_reach_grasp_v1_results.md](zero_shot_reach_grasp_v1_results.md): reach P 53/64 and
G 62/64; grasp-and-lift 0/64 for every world-model arm; the scripted IK follower 50/64; one run,
one model seed gated, simulation only).

**This is not an Apple→Plate experiment.** It plans grasp-and-lifts of `play-v1`'s objects with
G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not change,
whatever row this task ends in** (R26.18); a change would need its own ruling.

## 1. Question

TASK-088's world-model arms brought the palm near the grasp subgoals but never closed and lifted.
R25.27 asks first, offline, whether the models' predictions separate a closing-and-lifting hand from
an open or non-lifting hand at the object over the planning horizon, and only then a closed-loop
grasp test with a change of a different kind (a longer horizon, a grasp-phase cost, or more grasp
data). So:

- **Part A (offline, gates Part B):** from fresh states with the open hand at the object and with
  the closed hand at the object, do TASK-087's models (G and P, model seed 87100) predict a lower
  goal cost for the closing (or lifting) commands than for the open (or non-lifting) commands, as
  often as the same cost computed on the simulator's real frames does? And does the CEM planner,
  run once at those states, choose to close and lift, under TASK-088's cost and under the candidate
  changes of §6?
- **Part B (closed loop, only if Part A admits it):** with the one change Part A selects, does CEM
  over G grasp and lift `play-v1`'s objects on fresh gated resets, at the bar of §7.2
  (the plan's 0.40, lowered only if the IK follower's development rate is below 0.445)?

"Zero-shot" means as in TASK-088: no training, fine-tuning or selection on this task, its goals or
its resets; TASK-087's checkpoints are used unchanged.

## 2. What TASK-088's records already show (read before this draft, reported)

From TASK-088's gated grasp run files (`~/develop/emai/evidence/task088-gated/gated-grasp/`,
read-only; the computation is `an088b.py` in `~/develop/emai/evidence/task089-prototype/`): the
executed right-grasp command (−1 open, +1 closed; an absolute synergy target) per subgoal phase.
The first four columns are the **median over resets of each episode's mean command in that phase**;
the last is the **share of all commands in the phase, pooled over the resets, above 0.5**. Every
arm has 64 resets in each phase except the lift phase of G (61), G-pose (59) and ik (62), whose
other episodes ended (stopped, or for ik succeeded) before it.

| Arm | hover | pre-grasp | grasp | lift | pooled share > 0.5 (grasp phase, lift phase) |
| --- | --- | --- | --- | --- | --- |
| P-87100 | −0.07 | −0.29 | +0.05 | +0.12 | 0.08, 0.16 |
| G-87100 | +0.06 | −0.16 | −0.30 | −0.08 | 0.01, 0.10 |
| G-lat-87100 | −0.17 | −0.23 | +0.04 | +0.10 | 0.09, 0.15 |
| G-pose-87100 | −0.06 | +0.10 | −0.32 | −0.15 | 0.02, 0.04 |
| ik | −1.00 | −0.98 | +0.90 | +1.00 | 0.88, 1.00 |

So the world-model arms rarely commanded a closed hand: their typical grasp command stayed near the
CEM's initial mean of 0, a half-closed hand, and at most 16 % of their lift-phase commands were
above 0.5. Whether that is because the models cannot predict what
closing does, or because the planner does not use what they predict, is Part A's question.

## 3. Models and cost (reused, not retrained)

- TASK-087's checkpoints, loaded by TASK-088's guarded loader unchanged
  (`zero_shot_runtime.load_model`: `T-JOB-DONE`, recorded checkpoint sha256, TASK-087's feature
  report, the current `grounded_wm_model.py` implementation hash), and TASK-087's latent (frozen
  DINOv2 ViT-S/14 tokens of the onboard 112 px frame, 4 × 4 pooled, projected to `[16, 192]`).
- **Primary model: G-87100** (+ state token, joint-change and inverse-dynamics losses), with
  TASK-088's G cost: latent MSE + λ · standardised joint-position MSE of the 14 right arm and hand
  joints, **λ = 2.0552** (TASK-088's R25.18; recomputed and checked). TASK-088's results document
  (§6, R25.27) recommends G as the default model of this task, because it reached better than P (62 against 53 of 64,
  9 to 0 discordant, p = 0.004); this task adopts it (R26.2).
- **Second arm: P-87100** (plain LeWM, latent MSE only), reported against the same bars, so that
  plain and grounded are compared on the same states and resets.
- **Reported only:** G-lat-87100 (G, latent term only), the no-action model N-87100 (Part A
  control), and in Part B G-87101 and G-87102 with the selected change (replication).
- The models' maximum roll-out is 16 steps (`max_horizon`), so the horizons here are 8 (TASK-088's
  planning horizon) and 16.

## 4. Scene, actions, objects

`play_scene_v0`, `ee_delta_grasp_v0`, the right arm and right grasp planned, the left arm held,
execution through the embodiment's projection, the 10 settle commands, and "stopped" as a failure:
all exactly as TASK-088 §3. Layouts by TASK-088's grasp sampler (`zero_shot.sample_layout("grasp",
seed, attempt)`: the target uniform over the four objects plus 0–2 distractors in TASK-085's Phase 3
test object region) and goals by TASK-088's grasp demonstrator (`zero_shot.make_grasp_goal`: e9's
pick without noise, hover 60, descend open 50, close 35, lift 12 cm in 50, hold 20; rejected unless
it meets the lift criterion itself; the four subgoals hover, pre-grasp, grasp and lift are its frame
and pose at the end of those phases). Their salts (8801, 8802) are TASK-088's; the seeds (§10) are
new, so every layout is fresh. Object truth is read only by the demonstrator, the root builder (§5)
and the scorer, never by a controller or a cost.

## 5. Part A — offline, on fresh branched grasp windows

### 5.1 Roots and branches

Held-out `play-v1` windows cannot give what this question needs: few test episodes lift anything
(about 10 % of `play-v1`'s episodes do, TASK-085), and a recorded episode has only one future, so
there is no same-state contrast and no simulator ceiling. So the windows are **recorded fresh**:

For each root seed, the demonstrator's draw is made exactly as `make_grasp_goal` makes it (same
layout, yaw and acceptance; a draw it rejects is replaced as there). On the accepted draw, two
**roots** are taken on its trajectory:

- **R1 — open at the object:** the end of the open descent (the pre-grasp subgoal's state);
- **R2 — closed at the object:** the end of the close (the grasp subgoal's state).

From each root, 16-command **branches** are run in the simulator from the same state (the episode is
re-run from its reset with the same commands, which MuJoCo reproduces exactly, then branched), each
with e9's tracking law (arm command (target − palm) / 1.5 cm clipped at 0.4, so a lift moves the
palm at most 0.6 cm a command; the target itself is not ramped) and the grasp ramped by 0.2 a command,
as the demonstrator does. No branch presses beyond the demonstrator: its close phase targets the
aim, which the palm reaches open at R1:

| Root | Branch | Palm target | Grasp | Role |
| --- | --- | --- | --- | --- |
| R1 | **close** | the aim (as the demonstrator's close phase) | → +1 | true (the demonstrator's own continuation) |
| R1 | open | the aim | −1 | alternative: the hand stays open |
| R2 | **lift** | the aim + 12 cm (as its lift phase) | +1 | true |
| R2 | release | the aim + 12 cm | → −1 | alternative: lifts the hand while opening it |
| R2 | stay | the aim | +1 | alternative: closed but not lifted |

Every branch's executed commands, its frames (onboard, 112 px) and measured states at steps 1–16,
and the target's rise and grasp contact are stored. Before each branch the re-run's state at the
root is checked to equal the demonstrator's recorded state there (the pre-grasp subgoal's 28-D
state for R1, the grasp subgoal's for R2); a mismatch stops the run (A-VOID). A root whose branch is
stopped by the embodiment is replaced like a rejected draw (and counted).

**The goal** of each comparison is the subgoal the planner was given there in TASK-088: at R1 the
grasp subgoal (the frame and pose at the end of the close), at R2 the lift subgoal (the end of the
hold).

### 5.2 A1 — do the predictions separate the branches? (gates Part B)

For a cost family c (G's: latent + λ · pose; P's and G-lat's: latent only), a comparison
(root type, true branch t, alternative a) and a horizon h ∈ {8, 16}, the root r is a **win** when

    c(prediction_h of t) < c(prediction_h of a)

against the goal, where prediction_h of a branch is the model's roll-out from the root's encoded
frame (and, for G, its measured 28-D state) under that branch's executed commands, read at step h.
Ties are not wins. The three comparisons are R1 close-vs-open, R2 lift-vs-release and R2
lift-vs-stay. The **win rate** is wins / roots. A1 is read on the per-step cost at h (the cost
TASK-088's planner reads at its last step); the same wins with the step-averaged cost of §6 (the mean
over h = 1 … 8) are reported beside it, not gated.

- **Ceiling (measured, non-learned):** the same win rate with the simulator in place of the model:
  c computed on the encoded real frame (and the measured state) at step h of each branch.
- **Controls:**
  - *action-shuffled*: for each comparison separately, the 32 commands of its two branches are
    pooled and dealt at random (generator keyed by salt 8901, the root seed and the comparison's
    index 0–2) into two 16-command sequences; the model's win rate of the first over the
    second is a floor (expected one half). Paired against the model's win indicator per root.
  - *no-action*: N-87100 (actions zeroed), whose predictions cannot depend on the commands, so every
    comparison is a tie (0 wins). A plumbing check, reported.

### 5.3 A2 — does the planner, run once at the root, close and lift? (selects Part B's change)

At every R1 root (goal: the grasp subgoal) and R2 root (goal: the lift subgoal), one CEM call with
TASK-088's planner (`CEMPlanner` unchanged: 300 samples, 10 iterations, 30 elites, minimum std 0.05,
the left arm fixed, CEM seed `zero_shot.cem_seed(root seed)`), the model G-87100 with G's cost, for
TASK-088's setting and for each candidate change of §6. Measured on the **first planned command**,
the one MPC executes:

- **close@R1**: its right grasp > 0.5;
- **lift@R2**: its right grasp > 0.5 **and** its right arm's vertical command (dimension 8, the right
  palm's dz) > 0.

P-87100 with its cost is run the same way, reported. The rates are over the gated roots.

### 5.4 Calibration A-K0 and the bars (Stage 0, before the freeze)

TASK-074's lesson: bars come from measured references. **A-K0** builds 32 calibration roots
(§10) and computes only the **simulator ceiling** of A1 (no model is run on them) for each cost
family, comparison and horizon. Then, fixed in the Stage 0 record:

- A comparison is **gated** for a cost family and horizon when its K0 ceiling is at least 0.75
  (24 / 32): the cost, on real frames, separates those branches. A comparison below that is reported
  only (a planner cannot be asked to separate what the cost on real frames does not).
- Its **bar** is `0.5 + 0.9 × (ceiling − 0.5)`, as a count `⌈bar × 64⌉` of 64 gated roots.
- A horizon is **qualifying** when, for G's cost family, close-vs-open **and** at least one R2
  comparison are gated at it. **If neither h = 8 nor h = 16 qualifies,** the row is **A-CEILING**
  (§8): the latent and pose cost does not see the grasp on real frames at any one horizon, and the
  task escalates to a ruling before the freeze. A-FAIL (§8) is judged only on qualifying horizons,
  so a ceiling problem can never fire the clause.

**A2's bars are declared, not calibrated** (they are a selection rule, not a claim): close@R1 and
lift@R2 each at least **48 / 64** (0.75) **and** above TASK-088's setting on the same roots (exact
one-sided McNemar, p < 0.01). The references are known: the demonstrator and the IK follower close
and lift on every such root by construction, and TASK-088's setting did so rarely (§2, §13).

### 5.5 A's gated run

**Stage A** (after the freeze, from a clean tree at the frozen revision): 64 fresh roots (§10);
A1 for G-87100 (primary), P-87100, G-lat-87100, N-87100, the action-shuffled control and the
ceiling, at h = 8 and 16; A1 also for G and P with model seeds 87101 and 87102 (reported); A2 as
§5.3. Stage A also computes the ceiling on its own roots; it is reported only (the gate and the bars
are A-K0's). A1 **passes at h** for a model when close-vs-open and at least one R2 comparison are gated at
h for its cost family, and every comparison gated at h has at least its bar's count of wins **and**
beats the action-shuffled control (exact one-sided McNemar on the per-root win indicators,
p < 0.01).

## 6. The candidate changes for Part B, in order

R25.27 named a longer horizon, a grasp-phase cost, or more grasp data; the owner's brief for this
task (2026-10-11; recorded in DECISIONS (ah) and the TASK-089 MC task) asks for the cheapest one
that Part A supports. More grasp data means a new corpus and retraining; it is not a candidate here
(it is the recommendation if this task ends without a pass, §9). The candidates need no retraining:

1. **Step-averaged cost (a grasp-phase cost):** on every subgoal of a grasp episode, the planning
   cost of a candidate is the **mean of the per-step cost over h = 1 … 8** instead of the cost at
   h = 8 alone. Motivation (§2, §13): the executed command is the first one, but a last-step cost
   barely depends on the first command's grasp value, because the later commands set the hand's
   final state; averaging over every step makes the first commands count.
2. **Longer horizon:** H = 16 (the models' maximum), the cost at h = 16, otherwise TASK-088's
   setting.
3. **Held grasp:** the planner's candidates hold their first step's grasp value over the whole
   horizon (the model is rolled out with the right grasp of step 1 repeated; the other dimensions
   unchanged; the cost as TASK-088's). An action-parametrisation change, outside R25.27's list;
   admitted as the last candidate by R26.6.

**Selection rule:** the Part B change is the **first** candidate, in this order, for which (i) the
candidate's horizon is qualifying (§5.4) and A1 passes for G-87100 at it (h = 8 for 1 and 3, h = 16 for 2) and (ii) A2's two
bars are met for G-87100 under that candidate. Exactly one change goes into Part B. Each is
implemented around TASK-088's `PlannerModel` and `CEMPlanner` without changing either.

## 7. Part B — closed loop, admitted only by A-PASS

Only the selected change differs from TASK-088's grasp setting: the scene, goals (four subgoals),
subgoal switching (§5.3 of TASK-088: 1 cm, 0.17 rad and 0.15 rad RMS for 3 commands, or 100
commands), budget (400 commands), execution, stop rule and the success scorer (target ≥ 5 cm above
its start height and in grasp contact for 20 consecutive commands; no latching) are TASK-088's code,
unchanged.

### 7.1 Arms (all on the same goals and resets, paired)

| Arm | Model | Planner | Role |
| --- | --- | --- | --- |
| **G-chg** | G-87100 | the selected change, G's cost | **primary** |
| P-chg | P-87100 | the selected change, P's cost | second arm (plain against grounded), reported against the bar |
| G-base | G-87100 | TASK-088's setting | control: what the change adds, on the same resets |
| ik | — | TASK-088's IK follower with e9's press | ceiling; its Stage BD rate enters the bar rule (§7.2) |
| ik-nopress | — | the same without the press | reported |
| hold, random | — | TASK-088's | floors |
| G-chg-87101, G-chg-87102 | G-87101 / 87102 | the selected change | replication, gated cohort only, reported |

### 7.2 Stages, bar and rows

- **Stage BD — development, not gated** (32 resets, §10): every arm of §7.1 except the
  replications. Its numbers are read and recorded; **the bar is bar = min(0.40, 0.9 × r)**, with r
  ik's success rate on BD and 0.40 the plan's example grasp bar (used in TASK-088). **What sets it,
  honestly:** ik's earlier rates (TASK-088: 24/32, 12/16, 50/64) give 0.9 × r ≥ 0.67, so the rule
  returns the declared 0.40 unless ik does markedly worse on BD; ik's rate binds only when
  0.9 × r < 0.40 (r below 0.445, at most 14 of 32). If r < 0.30 (at most 9 of 32) the bar is not set
  and the task escalates to a ruling before Stage BS (the goals, not the planners, would be in
  doubt). BD can show a defect (a fix needs its own ruling) but changes no knob of §2–§7 by itself.
- **GO:** an independent reviewer reads BD's record and posts **GO** (or NO-GO) as a PR comment
  before Stage BS starts.
- **Stage BS — gated:** 64 fresh resets (§10), every arm of §7.1, at the frozen revision from a
  clean tree. Power: a true rate of 0.55 meets a 0.40 bar (26 / 64) with probability about 0.99; at
  a true rate equal to the bar about one half.
- **Rows** (first match), on G-chg (model seed 87100), k = its successes of 64:
  1. **B-VOID** — a run did not complete under the rules, a goal, checkpoint, λ or seed check
     failed, or a cohort was run twice.
  2. **B-PASS** — k ≥ ⌈bar × 64⌉ **and** G-chg beats random (exact one-sided McNemar, p < 0.01).
  3. **B-LOW** — below the bar, but G-chg beats random (p < 0.01).
  4. **B-FAIL** — G-chg does not beat random.
- **Reported with every row:** G-chg against G-base (exact one-sided McNemar, p < 0.01: whether
  the change itself mattered; a B-PASS without it shows that G-chg grasps, not that the change is
  why), P-chg against G-chg (two-sided), ik against G-chg, per-object results, grasp contact,
  maximum rise, subgoals reached or timed out, executed grasp commands per phase (as §2), stopped
  episodes, planning seconds, and the replication seeds. Exact (Clopper–Pearson) 95 % intervals
  for every count.
- **"LeWM needed" is not claimed by any row.** The IK follower uses the same goal information and
  no world model; if it scores at least as well, the record says so. A B-PASS would show that CEM
  over these play-trained models grasps and lifts at the plan's bar with this change, in
  simulation, on goals a scripted pick can do; not that a world model is needed for it.

## 8. Rows of the task (Part A first, then Part B)

1. **A-VOID** — a Stage A check failed (a root, branch, checkpoint, λ or seed check, a cohort run
   twice, a dirty tree).
2. **A-CEILING** (Stage 0) — no horizon qualifies (§5.4): escalate before the freeze; no clause.
3. **A-FAIL** — A1 fails for G-87100 at every qualifying horizon (§5.4): the predictions do not
   separate the grasp as the real frames do. No Part B; **the abandonment clause fires** (§9).
4. **A-PLAN** — A1 passes at some qualifying horizon, but no candidate meets the selection rule: the
   predictions separate the grasp but the planner, under every candidate, does not choose it.
   Escalate; no Part B; no clause.
5. **A-PASS(k)** — candidate k is selected; Part B runs with it, and the task's row is then Part
   B's (B-VOID, B-PASS, B-LOW or B-FAIL).

After **A-VOID** or **B-VOID** the defect is fixed under its own reviewed ruling and the stage
runs again on fresh seeds named by that ruling (a voided cohort is never re-run). After a **NO-GO**
on Stage BD, the reviewer's defect is fixed under its own ruling and Stage BD is recorded again on
fresh development seeds before a new GO.

Escalations (A-CEILING, A-PLAN, B-LOW, ik's BD rate below 0.30) go to a ruling in
`docs/DECISIONS.md`; each closes the task with a recommendation unless the ruling says otherwise.
B-PASS admits a Phase 4 preregistration (pick-and-place with subgoals); it does not preregister it.

## 9. Abandonment clause

**Fires on A-FAIL or B-FAIL.** It closes **CEM planning of grasp-and-lift over TASK-087's
checkpoints** (trained on `play-v1` as collected, TASK-087's latent): no further planner, cost,
horizon or action-parametrisation variant over these checkpoints is preregistered for grasping
without new evidence of a different kind. The next grasp step then changes the data (more
grasp-and-lift in the play corpus) or the model. **Not closed:** LeWM, DINOv2, `play-v1` as a base,
CEM in general, reach planning (TASK-088's Z3-REACH stands), Phase 4 as a goal, and the product
goal.

## 10. Seeds and salts

A-K0 roots 89000–89031; Stage A roots 890000–890063; Stage BD resets 89100–89131; Stage BS resets
891000–891063; debug 89900–89999 (plumbing only, never counted). Layouts and goals use TASK-088's
salts 8801 and 8802 with these seeds; CEM 8803 (`cem_seed`) and random 8804 as TASK-088; new salt
**8901** (A1's action-shuffled deal); 8902 reserved. Checked free on every branch of the repository
(`git grep` over all refs in `src/`, `scripts/`, `docs/`, `.mc/`: no use of 89000–89999,
890000–891063 or 8901–8902; only `range(88950, 89000)`'s end bound matched).

## 11. What this cannot show

One scene, one arm and hand, one camera at 112 px, simulation only, one encoder and latent, one
corpus, one model seed gated (two more reported), one planner. Goals come from a scripted
demonstrator, so they are reachable by construction and the cohort is biased to layouts a scripted
pick can do (TASK-088: 57 rejected draws for 64 goals); the grasp subgoal's pose places the palm at
the object, so success does not show that a planner localised the object from the image. Every
A1 comparison also differs in the robot's own commanded motion, so a win can come from predicting
the hand's and arm's own consequences (G's pose term reads predicted joints directly) rather than
the object; lift-vs-stay and lift-vs-release are the comparisons where the object's rise differs,
and the branches' recorded rise and contact are reported beside the wins. Part A's
roots are on the demonstrator's own trajectory (palm exactly at the aim), which a closed loop
reaches only approximately. A B-PASS would be the first world-model grasp-and-lift in this project,
on these terms only; "LeWM needed" is not shown by any row.

## 12. Runtime, disk and evidence

- Every GPU use (rendering, encoding, CEM) is one `scripts/gpu_run.sh --wait --min-free-gib 8
  --board` job, the lock taken per job, not per queue; at most 6 processes per world-model run.
- Estimates: A-K0 and Stage A under 1 GPU hour together; Stage BD about 2 GPU hours; Stage BS about
  7 (5 world-model runs × 64 × at most 400 commands at about 1 s per command with 6 processes; H = 16
  doubles a run's time if selected).
- Disk: Part A stores 2 roots × (2 + 3 branches) × 16 frames of 112 px per root (a few MB); nothing
  is written below 10 GiB free (the runner refuses). Outputs in `outputs/task089-*` of the run
  worktree; evidence in `~/develop/emai/evidence/task089-*/` with `SHA256SUMS`; two MP4 replays of
  Part B (one success if any, one failure) in `~/develop/emai/evidence/task089-*/videos/` with
  their own `SHA256SUMS`. TASK-087's checkpoints and features and TASK-088's records are read, never
  written.

## 13. Disclosed development before this draft

On debug seeds 89900–89911 (12 roots; layouts drawn with the seed + 10⁶, so not the formal
debug layouts; never counted), a prototype of §5 (not the Stage 0 code) ran under the GPU lock:

- **A1 (wins of 12):** close-vs-open at h = 8 and 16: ceiling (latent and latent + pose) 12, P 12,
  G 12, G-lat 12. Lift-vs-stay at h = 8: ceiling latent 11, latent + pose 12; P, G, G-lat 12.
  Lift-vs-release at h = 8: ceiling latent 6, latent + pose 12; P 3, G 12, G-lat 1; at h = 16:
  ceiling latent 11, P 10, G 12, G-lat 10. N: ties everywhere.
- **A2 (first planned command, 12 roots × 4 CEM seeds):** with TASK-088's setting the first grasp
  command was above 0.5 on 2 % (R1) and 29 % (R2) of G's plans; with H = 16 on 0 % and 40 %; with
  the step-averaged cost on 94 % and 94 %; with the held grasp on 100 % and 100 %. P was lower
  throughout (step-averaged 23 % and 56 %, held 48 % and 42 %).
- These numbers shaped the candidate list, its order (the step-averaged cost first, as the one in
  R25.27's list that the probe supports), A2's 48 / 64 bar (set after seeing these rates) and the
  choice of G as primary; they are not results. Stage A runs on fresh roots. The prototype's
  scripts and outputs are kept in `~/develop/emai/evidence/task089-prototype/` (`SHA256SUMS`).

## 14. Stage 0, freeze and what Stage 0 may change

- **Stage 0** (its own PR, before the freeze): opt-in modules `zero_shot_grasp.py` (NumPy at import:
  cohorts, roots and branches, A1/A2 statistics, bars, rows) and `zero_shot_grasp_runtime.py`
  (torch: the three candidate changes around TASK-088's `PlannerModel`, A1 roll-outs, A2 CEM),
  `scripts/run_task089.py`, a summary script, a stage script and tests (the branch definitions, the
  win and bar rules, the shuffled deal, the selection rule, the rows, the changes' cost and
  parametrisation, the seed checks, the core-import check). No existing file changes, so TASK-087's
  implementation hash and TASK-088's frozen code stay as they are. λ recomputed; a debug smoke of
  every arm (A1, A2 and a few Part B episodes on debug seeds); **A-K0** and the bars it gives.
- Stage 0 may change only the per-run process count and, if the smoke projects Stage BS above
  20 GPU hours, drop the two replication runs (each recorded).
- Frozen at the Stage 0 merge; Stage A, BD and BS run at the merged revision from a clean tree.

## 15. Rulings this document records (R26.1–R26.18; decided by Claude under owner delegation)

- **R26.1** — TASK-089 follows R25.27: Part A offline first, Part B only if A admits it.
- **R26.2** — G-87100 is the primary model (R25.27's recommendation adopted), P-87100 the second
  arm; TASK-087's checkpoints are reused unchanged; λ = 2.0552.
- **R26.3** — Part A uses fresh branched grasp windows (§5.1), not held-out `play-v1` windows.
- **R26.4** — A1's metric, ceiling and controls (§5.2).
- **R26.5** — A2's planner probe (§5.3).
- **R26.6** — the three candidate changes and their order (§6), the third admitted although outside
  R25.27's list; more grasp data is not a candidate here.
- **R26.7** — A-K0 and the bar rule (§5.4): gated comparisons from the measured ceiling, bars
  `0.5 + 0.9 × (ceiling − 0.5)`; A2's bars declared.
- **R26.8** — Stage A as §5.5 and the selection rule (§6).
- **R26.9** — Part B's arms (§7.1).
- **R26.10** — Part B's bar min(0.40, 0.9 × r) with r ik's Stage BD rate: the declared 0.40 unless
  r < 0.445; escalation if r < 0.30.
- **R26.11** — Stage BD, an independent GO, Stage BS (§7.2).
- **R26.12** — the rows (§7.2, §8).
- **R26.13** — the abandonment clause (§9).
- **R26.14** — seeds and salts (§10).
- **R26.15** — reported quantities, including the two MP4s (§7.2, §12).
- **R26.16** — Stage 0 and the freeze (§14).
- **R26.17** — runtime, disk and evidence (§12): every GPU use under `gpu_run.sh`, per job.
- **R26.18** — R7 does not change, whatever the row.

## 16. Stage 0 record (R26.19–R26.23)

- **Code** (opt-in; no registry entry; `import embodied_jepa` stays torch- and MuJoCo-free, checked
  by a test): `src/embodied_jepa/zero_shot_grasp.py` (cohorts, the roots and branches re-run from
  TASK-088's demonstrator, A1's wins, gating, bars and pass rule, the shuffled deal, A2's measures
  and bars, the selection rule, Part B's arms, bar and rows; NumPy at import),
  `src/embodied_jepa/zero_shot_grasp_runtime.py` (the step-averaged and held-grasp wrappers around
  TASK-088's `PlannerModel`, the H = 16 planner, A1's roll-out and ceiling costs through the
  planner's own `distance`, A2's single CEM call, Part B's controller), `scripts/run_task089.py`
  (lambda, roots, ceiling, a1, a2, goals, run, video), `scripts/summarize_task089.py`,
  `scripts/run_task089_stage.sh` (one `gpu_run.sh` job per step; only gpu_run's refusal, exit 76, is
  retried) and `tests/test_zero_shot_grasp.py` (15 tests; one graphics opt-in, which checks that a
  root's demonstrator draw equals `zero_shot.make_grasp_goal`'s on the same seed). No existing file
  changes: TASK-087's implementation hash and TASK-088's frozen code are untouched.
- **The re-run root check** (review nit of #197) is bitwise: the 28-D measured state at the root,
  re-run from the reset with the same commands, must equal the recorded subgoal state exactly
  (`numpy.array_equal`). It held on every root built so far (8 debug, 32 calibration).
- **A-K0** (calibration roots 89000–89031, at `72a766e` (kept reachable as the branch
  `run/task089-k0`; the same code as this PR's Stage 0 code commit before its rebase), clean tree, NVIDIA EGL, under the GPU
  lock; 40 rejected draws; targets can 17, cube 9, apple 6, banana 0 — the demonstrator lifts no
  banana, as in TASK-088): the simulator ceiling's win rates (of 32) and the bars by §5.4's rule,
  frozen as `zero_shot_grasp.K0_CEILING` / `A1_BARS`:

  | Cost family | h | close-vs-open | lift-vs-release | lift-vs-stay |
  | --- | --- | --- | --- | --- |
  | latent + pose (G) | 8 | 1.000 → 61 / 64 | 1.000 → 61 / 64 | 0.969 → 59 / 64 |
  | latent + pose (G) | 16 | 1.000 → 61 / 64 | 1.000 → 61 / 64 | 1.000 → 61 / 64 |
  | latent (P, G-lat) | 8 | 0.938 → 58 / 64 | 0.469: **not gated** | 0.969 → 59 / 64 |
  | latent (P, G-lat) | 16 | 0.938 → 58 / 64 | 0.844 → 52 / 64 | 1.000 → 61 / 64 |

  Both horizons qualify for G's cost, so the row cannot be A-CEILING. On real frames the latent cost
  alone does not tell "lift while opening" from "lift while closed" 8 commands ahead (0.469); with
  the pose term it always does. Evidence `~/develop/emai/evidence/task089-stage0/` (`SHA256SUMS`
  sha256 `293cb6f1…1864`; `k0/summary-k0.json` sha256 `c8a84453…6f22`; `k0/ceiling-k0-roots.json`
  sha256 `57138e1d…28a5`).
- **λ** is TASK-088's 2.0552; Stage A recomputes it and the runner refuses another value.
- **Debug smoke** (debug seeds, not results; the debug roots were built at `440a7b6` with the
  Stage 0 modules not yet committed, the other debug runs at `07affcd` and `72a766e`; debug runs may
  be dirty):
  - Part A on 8 debug roots (89900–89907; 17 rejected draws). A1 wins of 8 at h = 8 / 16: G-87100,
    G-87101, G-87102 8 on every comparison; P-87100 close 8 / 8, release 4 / 7, stay 8 / 8;
    G-lat release 2 / 6; N 0 everywhere (ties). A2 (close@R1, lift@R2 of 8) for G: base 1, 4;
    step-mean 7, 8; h16 0, 3; held 8, 8. For P: base 3, 0; step-mean 3, 6; h16 2, 0; held 5, 6.
  - Part B on 8 debug resets (89950–89957; goals for the 50 debug seeds with 43 rejected draws),
    with the step-averaged cost given by hand (`--setting`, debug only): ik 7 / 8, ik-nopress
    7 / 8, hold and random 0 / 8; **G-chg 0 / 8, G-base 0 / 8, P-chg 0 / 8**. With the change, G's
    hand stayed open on the approach and closed on the grasp and lift subgoals (share of commands
    above 0.5: hover 0.00, pre-grasp 0.00, grasp 0.75, lift 0.81, against 0.00 and 0.02 for G-base
    on grasp and lift), and it left the pre-grasp subgoal by "reached" on 6 of 8 resets (G-base
    0); but it touched the target with the grasp contact on 3 and never lifted it (maximum rise
    0.6 cm); 2 of its episodes ended on the measured-velocity stop, and 2 of G-base's (one
    velocity stop, one "projection infeasible"). P-chg closed on 6 % of grasp-phase commands.
    Planning about 0.65 s per command with 4 processes.
  - These smoke numbers were read after the protocol was reviewed and merged (#197) and change
    nothing in §2–§11: the rule selects Part B's change from Stage A, and Part B's row is read on
    fresh gated resets. They are disclosed because they make a B-LOW or B-FAIL more likely than
    the DRAFT's §13 suggested.
- **Knobs:** the process count stays 8 (baselines) and 6 (world-model arms) per run, and the
  replication runs stay. Projection from the smoke: Stage A under 1 GPU hour; Stage BD about
  1.5 GPU hours (3 world-model runs × 32 × at most 400 commands); Stage BS about 5–6 (5 runs × 64);
  H = 16 would double a run. Inside §14's 20-hour limit, so nothing changes.
- **Phase alignment (review of #198).** The subgoal switch is updated before each command, and the
  command of the step on which it switches already serves the next subgoal, so subgoal k ≥ 1 is
  served by exactly its recorded step count and the first subgoal by one fewer. §2's table (in the
  merged DRAFT) assigned one command per switch to the earlier subgoal. Recomputed with the right
  alignment (`summarize_task089._phase_share`; the evidence copy of `an088b.py` is the original),
  the changes are at most 0.02 and change no reading: G hover +0.07 (§2: +0.06) and lift-phase
  share 0.09 (0.10); G-lat grasp +0.03 (+0.04); G-pose grasp −0.31 (−0.32); ik pre-grasp −1.00
  (−0.98), grasp +0.88 (+0.90) and grasp-phase share 0.87 (0.88). §2 is context, not a gated
  quantity; it stays as merged, with this correction recorded here.
- **Guards added after the review of #198:** the Stage BS summary refuses (VOID) a Stage BD summary
  that is not a clean `bd-grasp` summary at the same revision; Part B's runner refuses a Stage A
  summary made at another revision; `lambda` needs a clean tree, and the Stage A summary checks
  `lambda.json` (value, revision, clean).
- **Review nits of #197** applied here: §7.1's ik row now says its Stage BD rate enters the bar rule
  (it does not set the bar by itself); §6 says where the owner's brief is recorded.
- **R26.19** — the Stage 0 code above. **R26.20** — A-K0 and the bars in the table above; both
  horizons qualify. **R26.21** — the debug smoke is not evidence; it is disclosed and changes
  nothing. **R26.22** — frozen at this PR's merge; Stage A, BD and BS run at the merged revision
  from a clean tree with `scripts/run_task089_stage.sh`, evidence in
  `~/develop/emai/evidence/task089-*`. **R26.23** — R7 does not change.

## 17. Stage A record (gated, offline; R26.24–R26.26)

At `c40919d` (the frozen revision, clean tree), NVIDIA EGL, `scripts/run_task089_stage.sh a`, one
GPU lock per step, 2026-10-11 03:09–03:15 UTC. λ recomputed 2.0552 (9 941 pairs, every G seed). 64
roots on the cohort's seeds 890000–890063 (55 rejected draws; targets apple 22, can 21, cube 21,
banana 0); the bitwise root check held on every branch. Evidence
`~/develop/emai/evidence/task089-a/` (`SHA256SUMS` sha256 `9b4e8953…6c98`; `summary-a.json` sha256
`26acfe8e…86c1`; `a1-a-roots.json` `fd3337c1…04f9`; `a2-a-roots.json` `f9d9aa3e…57bb`).

**Row: A-PASS(step-mean).** A1 passes for G-87100 at h = 8 and h = 16 (both qualifying), and the
first candidate, the step-averaged cost, meets A2's bars. Part B runs with the step-averaged cost.
The clause does not fire.

**A1** (wins of 64; gated bars from A-K0 in brackets; exact one-sided McNemar against the
action-shuffled control):

| Model (cost) | h | close-vs-open | lift-vs-release | lift-vs-stay | shuffled wins | A1 |
| --- | --- | --- | --- | --- | --- | --- |
| **G-87100** (latent + pose) | 8 | **64** [61] | **64** [61] | **64** [59] | 27, 31, 31 | **pass** |
| **G-87100** (latent + pose) | 16 | **64** [61] | **64** [61] | **64** [61] | 33, 29, 31 | **pass** |
| G-87101, G-87102 | 8, 16 | 64 | 64 | 64 | 28–36 | pass |
| P-87100 (latent) | 8 | 64 [58] | 18 (not gated) | 64 [59] | 28, 33, 27 | pass |
| P-87100 (latent) | 16 | 64 [58] | **50** [52] | 64 [61] | 31, 32, 35 | **fail** |
| P-87101 | 8 / 16 | 64 / 64 | 29 (n.g.) / 57 | 64 / 64 | | pass / pass |
| P-87102 | 8 / 16 | 62 / 64 | 36 (n.g.) / 52 | 64 / 64 | | pass / pass |
| G-lat-87100 (latent) | 8 / 16 | 62 / 64 | 15 (n.g.) / 52 | 64 / 64 | | pass / pass |
| N-87100 | 8, 16 | 0 (ties) | 0 | 0 | 0 | — |

Every McNemar p for a passing cell is below 10⁻⁵ except P-87102's and G-lat's lift-vs-release at
h = 16 (p = 0.006 and 9 × 10⁻⁵). 64 / 64 has an exact 95 % interval of 0.944–1.000. Reported only:
the ceiling on these roots (G's cost 63–64 of 64 everywhere; the latent cost 62, 32 and 63 at h = 8
and 63, 56 and 64 at h = 16); wins under the step-averaged cost (G-87100 64 everywhere; P-87100 63, 15 and 64). N's ties hold by
construction (its actions are zeroed). G-87102's checkpoint carries TASK-087's `last_two_triggered`
flag.
Within the 16 branch commands the lift branch raised the target by at least 5 cm on 21 of 64 roots
(median maximum rise 3.0 cm); no other branch did.

**A2** (first planned command, of 64; bars 48 and above TASK-088's setting at p < 0.01):

| Setting | G-87100 close@R1 | G-87100 lift@R2 | meets | P-87100 close@R1 | P-87100 lift@R2 |
| --- | --- | --- | --- | --- | --- |
| base (TASK-088) | 4 | 17 | — | 11 | 12 |
| **step-mean** | **62** | **61** | **yes** (p < 10⁻¹⁷, < 10⁻¹³) | 17 | 31 |
| h16 | 5 | 12 | no | 8 | 6 |
| held | 64 | 64 | yes (not reached by the rule) | 38 | 33 |

**What this says, and what it does not.** G's predictions separate a closing from an open hand and
a lifting from a releasing or staying hand at the object on every root (three model seeds, h = 8
and 16), at least as often as the simulator's own frames do under G's cost (63–64 of 64); plain P separates closing
and staying, but at h = 16 not lifting-while-closed from lifting-while-opening as often as its
latent ceiling allows. With TASK-088's last-step cost the planner's first command at these roots
rarely closes (4 of 64) or lifts closed (17); averaging the cost over the horizon makes it do so on
62 and 61. This is offline, on the demonstrator's own states (palm exactly at the aim), and G's
wins rest partly, and for lift-vs-release at h = 8 wholly, on the robot's own predicted joints (G's
pose term): there the latent cost on real frames is at chance (32 of 64) and G-lat wins 15; and only
21 of the 64 lift branches raised the object 5 cm within 16 commands. It does not show that a closed
loop grasps. The debug smoke of §16 (0 / 8 for G with this change) is the reason for caution.

- **R26.24** — Stage A is recorded: row **A-PASS(step-mean)**; G-87100 passes A1 at h = 8 and 16,
  and the step-averaged cost meets A2 (62 and 61 of 64).
- **R26.25** — Part B runs as frozen with the step-averaged cost: Stage BD (development; started at
  `c40919d` from a clean tree right after Stage A's summary), then an independent GO, then Stage BS.
- **R26.26** — R7 does not change.

## 18. Stage BD record (development, not gated; R26.27–R26.30)

At `c40919d` (the frozen revision, clean tree), NVIDIA EGL, `scripts/run_task089_stage.sh bd` with
Stage A's summary (A-PASS(step-mean); the runner checked its revision), one GPU lock per run,
2026-10-11 03:15–05:03 UTC. Stage BD was started right after Stage A's summary was written, before
the Stage A record (#199) was reviewed; the protocol allows that (it is development) and no knob
depends on the review. Goals: 32 (89100–89131; 29 rejected draws; targets can 16, cube 10, apple
6, banana 0). Evidence `~/develop/emai/evidence/task089-bd/` (`SHA256SUMS` sha256
`b72db5d7…4e60`; `summary-bd.json` sha256 `2dc0e9c4…d01e`).

| Arm | grasp-and-lift (32) | exact 95 % | grasp contact | stopped | share of commands > 0.5 (grasp, lift phase) |
| --- | --- | --- | --- | --- | --- |
| **G-chg-87100** (step-averaged cost) | **2** | 0.008–0.208 | 17 | 5 | 0.73, 0.83 |
| G-base-87100 (TASK-088's cost) | 0 | 0.000–0.109 | 4 | 3 | 0.00, 0.04 |
| P-chg-87100 (step-averaged cost) | 0 | 0.000–0.109 | 2 | 0 | 0.23, 0.42 |
| ik | 28 | 0.710–0.965 | 32 | 0 | 0.85, 1.00 |
| ik-nopress | 28 | 0.710–0.965 | 32 | 0 | 0.84, 1.00 |
| hold, random | 0 | 0.000–0.109 | 0 | 0 | |

- **The bar.** ik's BD rate r = 28 / 32 = 0.875, so bar = min(0.40, 0.9 × 0.875) = **0.40, 26 / 64**
  on Stage BS (the declared value; ik's rate does not bind), no escalation.
- **What BD shows.** With the step-averaged cost, G's planner keeps the hand open on the approach
  and closes it on the grasp and lift subgoals; it left the pre-grasp subgoal by "reached" on 14 of
  32 resets (hover on 9; the grasp subgoal never by "reached"), touched the target with grasp
  contact on 17 and lifted it on 2 (both cans, maximum rise 7.9 and 8.3 cm; every other episode
  rose at most 1.02 cm). These are the first grasp-and-lifts by a world-model planner in this
  project, on development resets: 2 of 32 is far below the bar (26 of 64 on BS), and G-base and
  P-chg lifted nothing. 5 G-chg episodes were stopped (4 on the measured-velocity stop, 1
  "projection infeasible"), and 3 of G-base's (1 and 2).
- Planning about 0.98 s per command (G) and 0.93 s (P) with 6 processes; a world-model run took 2 081–2 177 s
  for 32 resets, so Stage BS (5 runs × 64) projects to about 6 GPU hours.
- **No defect was found**: every run completed under the rules at one revision from a clean tree,
  the goal digests and the scorer's start-height check held, one change (step-mean) throughout.
  (Every run report's `change` field names the selected change, also for G-base and the
  baselines; G-base's episodes record `setting: base`, which is what it ran.)
- **R26.27** — Stage BD is recorded as development: G-chg 2 / 32, G-base 0 / 32, P-chg 0 / 32,
  ik 28 / 32; not gated, not a claim. **R26.28** — the Stage BS bar is 0.40 (26 / 64).
  **R26.29** — Stage BS runs as frozen after an independent GO posted on this PR. **R26.30** — R7
  does not change.
