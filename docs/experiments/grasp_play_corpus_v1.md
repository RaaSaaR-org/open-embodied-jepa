# TASK-090 — grasp-enriched play corpus: does more scripted grasp-and-lift play data make CEM over a retrained grounded LeWM grasp and lift?

**Status: FROZEN at the merge of its Stage 0 PR** (R27.1–R27.28, decided by Claude under owner
delegation, 2026-09-30; the DRAFT was reviewed and merged at #202, `664df1e`; Stage 0's record is
§17; the collection and Stage C record, C-PASS, is §18). After that merge nothing in §2–§11 changes without its own reviewed ruling. Stage 0 changed,
as §14 allows and §17 records: Stage C's C1 and C2 bars (§4, R27.21), Stage BS's size (n = 128,
§7.2, R27.22) and Stage T's store mode (§12, R27.24).

This is the grasp step that DECISIONS R26.34 recommends after TASK-089 closed **A-PASS(step-mean)
→ B-FAIL** ([zero_shot_grasp_v1_results.md](zero_shot_grasp_v1_results.md)). In TASK-089,
G-chg-87100 lifted **0 / 64** (exact 95 % 0.000–0.056; bar 26 / 64) and did not beat random. The
IK follower lifted 48 / 64. Reported only: G-chg-87101 lifted 8 / 64 and G-chg-87102 lifted 7 / 64,
"the first world-model grasp-and-lifts on gated resets here, below the bar". That run was
simulation only, with one scene and camera and one model seed gated, and "LeWM needed" is not shown.
TASK-089's abandonment clause fired (R26.33). It closed CEM planning of grasp-and-lift over
TASK-087's checkpoints. Per R26.34, "The next grasp step changes the data or the model, as §9 says:
for example more grasp-and-lift in the play corpus …". This task takes the **data** option.

**This is not an Apple→Plate experiment.** It plans grasp-and-lifts of the play scene's objects
with G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not change,
whatever row this task ends in** (R27.20). Any change would need its own ruling.

## 1. Question

TASK-089's results document puts the remaining gap "in executing a stable grasp (where the palm is
while the hand closes and the arm rises), not in choosing to close. Why is not tested here." Two
facts about `play-v1` point at the data:

- **Recorded (TASK-085):** Lift 10.4 % of train episodes lift anything. In random mode it is 0 %.
- **Development reading, not recorded or gated (§13):** about 3.1 % of TASK-087's 8-command
  training windows contain a lift state of any object.

So most of what the model learns about a closed hand comes from a hand that is closed away from
any object.

**Question.** Hold TASK-087's recipe, latent and budget fixed, and TASK-089's planner and cost
fixed. Add episodes to `play-v1`'s train split. Does a model trained with **grasp-enriched**
episodes grasp and lift more often in closed loop than one trained with an equal number of
**unchanged** play episodes? And does it reach the plan's bar?

## 2. The changed factor: corpus composition, at equal size

Two new collections of the same size. Each is added to `play-v1`'s 2 874 train episodes.

| Corpus | Added episodes | Collector | Role |
| --- | --- | --- | --- |
| **play-v2g** | 3 200 (seeds 920000–923199) | `play_policy_grasp_v0` (new, §2.1) | the changed data |
| **play-v1p** | 3 200 (seeds 910000–913199) | `play_policy_v0` (TASK-085's, unchanged) | the size control |

The comparison that answers this task's question is **play-v2g against play-v1p**. Both corpora
double the train set under the same update budget, so the only difference between them is what the
added episodes contain. Beating TASK-087's checkpoints, which were trained on `play-v1` alone, would
show only that *more* data helped. It is reported, not gated (§7).

### 2.1 `play_policy_grasp_v0` (new; `play_policy_v0` stays frozen)

This is a new module. TASK-085's `play_corpus.py` and its constants are not edited.

- **Modes:** scripted 0.55 and perturbed 0.45. No random mode, because random mode lifts 0 %
  (TASK-085). The noise σ values are TASK-085's (0.10 and 0.25), as are the burst rate and burst
  lengths.
- **Skills:** pick_place 0.85, push 0.10, wander 0.05 (no poke). Push and wander are kept so that
  open-hand contact and free motion stay in the data.
- **Pick recipe:** e9's recipe as in `play_policy_v0`, with `CLOSURE`, `AIM_SIGMA`, `GRASP_RAMP`,
  `GRASP_OFFSET`, the waypoint step ranges and the 0.10–0.20 m lift range unchanged. Two additions:
  1. **A lifted hold.** After the lift waypoint, the hand holds at the lift height with the grasp
     closed for U(40, 80) commands, then transfers and places as in `play_policy_v0`.
  2. **One retry.** If the collector's privileged check at the end of the lift waypoint finds no
     lift state (target ≥ 3 cm above its start with grasp contact), it re-attempts the same object
     once: hover, descend, close and lift again, with fresh aim noise and failure mode `none`.
- **`FAILURE_WEIGHTS` unchanged** (none 0.65, miss 0.10, early_close 0.07, drop 0.08, no_close
  0.05, abort 0.05). Failed grasps stay in the data as contrasts.
- **The pick target** is uniform over the objects on the table, as in `play_policy_v0`. There is
  no apple bias, because TASK-088 and TASK-089 draw the grasp target uniformly over the four
  objects. `sample_layout`, the plate and the camera are unchanged.
- **Unchanged:** the embodiment, the 5 rad/s velocity stop, the 400-command episode, the 10 settle
  commands, the 50-command minimum, the stored fields (including the per-frame `grasp_contact` and
  `object_position` sidecars) and the per-shard split rule (5 val and 5 test per 100 seeds, by
  TASK-085's rank rule with a new salt).
- **Policy salt 9003.** play-v1p uses `play_policy_v0` with TASK-085's policy salt 8701 on its own
  new seeds.

**What the factor is, honestly.** It is a bundle: skill mix (including no poke; `play_policy_v0`
draws poke at 0.10), mode mix, hold and retry. It is one
*kind* of change, more and longer scripted grasp-and-lift by the same pick recipe. It is not a
single knob, and the task cannot say which part of the bundle mattered.

### 2.2 Everything else is held fixed

- **Latent.** TASK-087's latent: frozen DINOv2 ViT-S/14, `onboard_rgb` at 112 px, 4 × 4 pooled,
  projected to `[16, 192]` with the **pinned projection** (`PROJECTION_SHA256` `c196fc48…e33f`; not
  refitted). The new episodes are featurised with it.
- **Model and training.** TASK-087's arm G and its losses and code (`grounded_wm_model.py`,
  `grounded_wm.py` and `models/lewm.py`, unedited, so the implementation hash is unchanged). Also:
  - the uniform window sampler over all valid (episode, start) pairs of the combined train set;
  - batch 64, AdamW at lr 3e-4, gradient clip 1.0;
  - **U = 60 000 updates**, TASK-087's checkpoint-selection rule and tolerance.
- **State standardisation pinned.** G standardises its 28-D state (and palm) input by train moments
  (`FeatureStore.moments()`, set with `set_moments`), and the planner's pose term and A-K0's
  ceiling cost use the same standardisation. With the code unchanged, each combined train set would
  give its own moments, so the two arms' inputs and their λ-weighted costs would differ from each
  other and from the setting where λ = 2.0552 was calibrated (R25.18). So **every new model (G and
  N, both corpora) is given `play-v1` train's state and palm moments, TASK-087's**, not its own
  corpus's. The training driver computes them from `play-v1`'s train store and checks the resulting
  `state_mean`, `state_scale`, `palm_mean` and `palm_scale` buffers against TASK-087's G-87100
  checkpoint's bitwise; each Stage T report and each closed-loop run records their sha256. A-K0's ceiling cost uses the same moments. The combined-set moments are computed and
  reported only.
- **Selection set.** Both corpora's models select on **the same** val windows: `play-v1` val plus
  the val episodes of both new collections (160 + 160 + 160). Neither arm selects on a val set the
  other lacks.
- **Test splits are never decoded.** That covers `play-v1`'s and both new ones.
- **Planner and cost.** TASK-089's G-chg exactly: `CEMPlanner` with H = 8, 300 samples,
  10 iterations, 30 elites and min std 0.05, MPC every command, the left arm fixed. The cost is
  G's step-averaged cost (mean over h = 1 … 8 of latent MSE + λ · standardised joint-position MSE),
  with **λ = 2.0552 for every arm**. R25.18's λ rule is recomputed for the new models and reported
  only.
- **Scene and scoring.** Unchanged from TASK-088 and TASK-089: the scene, the grasp goals
  (TASK-088's demonstrator; four subgoals), subgoal switching, the 400-command budget, execution,
  the stop rule and the success scorer. Success means the target is ≥ 5 cm above its start height
  and in grasp contact for 20 consecutive commands. Crossings are not latched.

Under TASK-087's rule, the training budget did not saturate. The cap bound and "the val curves
were still falling slowly" ([grounded_lewm_v1_results.md](grounded_lewm_v1_results.md)), and
doubling the data under the same U halves how often each window is seen. Both new arms share the
same budget; whether undertraining interacts with composition is not tested. `last_two_triggered`
is recorded per model and **is not an escalation**. TASK-074's train stage escalated twice on its budget rules and never reached its
gates; this task does not repeat that.

## 3. Models

| Model | Corpus | Seeds | Role |
| --- | --- | --- | --- |
| **M-v2g** | play-v2g | G-90100, G-90101, G-90102 | **primary** (all three gated, pooled) |
| M-v1p | play-v1p | G-90100, G-90101, G-90102 | size control (gated comparison) |
| M-v1 | `play-v1` | TASK-087's G-87100, G-87101, G-87102, unchanged | reported: "more data" against "no new data" |
| N-v2g, N-v1p | each corpus | 90100 only | no-action twins, validity (Stage E) |

The same model seed is used for both corpora. The initialisation and the sampler seed's derivation
are therefore shared, and seed i of M-v2g is paired with seed i of M-v1p.

**All three seeds are gated, pooled.** TASK-089's three seeds lifted 0, 8 and 7 of 64 (8 and 7
reported only, replication runs, below the bar), and its gated seed was the weakest; the protocol had fixed it before any closed loop. A single gated seed is
fragile, so the primary count here is the sum over three seeds × 64 resets.

**M-v1 is a control re-run on fresh resets, not a new variant.** It is TASK-087's checkpoints under
TASK-089's selected planner, unchanged. R26.33 closed further variants over those checkpoints; it
did not close re-running the closed setting as a reference. No row of this task can claim anything
from M-v1.

## 4. Stage C — the data gate (before any training)

Stage C runs after the collection and before featurisation. It is measured with a frozen label
script over the sidecars (`grasp_contact`, `object_position`), on the train episodes of each corpus.
A **lift window** is an 8-command window (TASK-087's design: 9 frames inside one episode) that
contains at least one lift state of any object. A lift state is the target ≥ 3 cm above its start
height with grasp contact (thumb plus index or middle), TASK-085's lift definition applied per
frame.

- **C1, enrichment of the added episodes:** the lift-window share of play-v2g's added train
  episodes must be **≥ 2.0×** that of play-v1p's added train episodes (declared 2.5×; revised at
  the freeze by §4's revision rule, R27.21, §17).
- **C2, enrichment of the training set:** the lift-window share of play-v2g's combined train set
  (`play-v1` train plus the additions) must be **≥ 1.5×** that of play-v1p's combined train set
  (declared 1.75×; revised with C1 by the same (1 + C1) / 2 derivation, R27.21, §17).
- **C3, the size control matches `play-v1`:** play-v1p's added train episodes must have a
  lift-window share within ±25 % of `play-v1` train's. Same policy, new seeds.
- **Reported:**
  - per-object lift episodes and windows;
  - grasp-contact windows;
  - closed-hand-near-object windows;
  - the per-episode maximum rise distribution;
  - the share of lift states with a rise ≥ 5 cm;
  - early ends, with the joint that tripped the velocity stop;
  - the mode and skill counts;
  - the collector's retry count and retry lift rate.

**The bars are declared, not calibrated.** They are set from a rough estimate, not a measurement.
Removing random mode and raising pick_place's weight should give about 1.35 × 1.55 (from TASK-085's
per-mode lift rates). The hold should give about 1.5 more lift-state steps per lift. Together that
is roughly 3× in the added episodes, so C1 is set below that point estimate, at 2.5×. Because the
two halves of each combined set are about equal in size, C2's 1.75× is the combined-set equivalent
of C1's 2.5× ((1 + 2.5) / 2); C2 is the binding bar for what the model sees.

Stage 0's debug collection (§14) measures both ratios on debug seeds: **300** episodes of
`play_policy_grasp_v0` (930000–930299) and 100 of `play_policy_v0` (930300–930399). The debug C1
ratio uses `play-v1` train's lift-window share as its denominator (2 874 episodes of the same
policy; the 100 `play_policy_v0` debug episodes check C3's direction only), with an
episode-bootstrap 95 % interval. **Revision rule:** if the debug C1 or C2 point ratio is below its
bar, the collector or the bar is revised **at the freeze**, with the debug readings disclosed, and
never after. A debug ratio whose interval merely includes the bar is disclosed and is not by itself
a reason to revise.

**C-FAIL** (any of C1–C3 fails) stops the task before any GPU training and escalates. No clause
fires, because the hypothesis was not tested.

## 5. Stage T — training

Eight jobs, one `gpu_run.sh` lock each:

- G × {play-v2g, play-v1p} × seeds 90100–90102;
- N × {play-v2g, play-v1p} × seed 90100.

Each job runs TASK-087's training loop through a new driver (§14) over a combined feature store. Each
job writes a report with:

- the checkpoint sha256;
- the implementation hash, which must equal TASK-087's;
- the feature-store hashes (play-v1's report `f5648a35…7eae` plus the new stores);
- the kept update;
- `last_two_triggered`.

**T-VOID:** a job fails a check or is run twice.

## 6. Offline stages (gate the closed loop)

### 6.1 Stage E — validity on the common val windows

For each new G model, measured on the common val windows at h = 4 and h = 8, with intervals from
TASK-087's episode bootstrap:

- **(i)** its recursive roll-out error is below copy-last (upper bound below 1 on the ratio);
- **(ii)** its action sensitivity (wrong-action over true-action roll-out error, TASK-087's
  definition) is above 1, with the lower bound above 1;
- **(iii)** for seed 90100, its roll-out error is below its corpus's N twin's.

**E-FAIL:** any M-v2g or M-v1p model fails any of (i)–(iii) that applies to it ((iii) applies to
seed 90100 only). The task escalates and no clause fires.

**Reported:** the roll-out error on lift windows and on grasp-contact windows separately, for M-v1,
M-v1p and M-v2g.

### 6.2 A-K0 and Stage A′ — TASK-089's offline grasp probes on the new models

TASK-089's Part A is reused unchanged, as its code (§5.1–§5.4 of
[zero_shot_grasp_v1.md](zero_shot_grasp_v1.md)): the R1 and R2 roots, the 16-command branches, the
A1 win rule and the A2 single CEM call.

- **A-K0** (Stage 0, before the freeze) builds 32 calibration roots, 90000–90031. Only the simulator
  ceiling is computed on them; no model runs. The gating and bars follow TASK-089 §5.4: a
  comparison is gated when its ceiling is ≥ 0.75, and its bar is `0.5 + 0.9 × (ceiling − 0.5)`.
  The ceiling's pose term uses `play-v1` train's pinned state moments (§2.2). **If h = 8 does not
  qualify** under TASK-089 §5.4's rule (G's close-vs-open and at least one R2 comparison gated at
  h = 8), the row is **A-CEILING**: the task escalates to a ruling before the freeze, and no clause
  fires.
  A2's bars are declared: close@R1 and lift@R2 each ≥ 48 / 64. TASK-089's second A2 condition
  (above TASK-088's last-step setting on the same roots, exact one-sided McNemar at p < 0.01) is
  **dropped**: it served TASK-089's selection among cost settings, and here the setting is already
  selected and A′ only checks that nothing regressed.
- **Stage A′** runs on 64 fresh roots, 903000–903063, for every model in §3:
  - A1 with G's cost, at h = 8 and 16;
  - A2 with the step-averaged cost;
  - the action-shuffled control.

**A′ is a no-regression gate, not the test.** TASK-089's G-87100 already won 64 / 64 on every
comparison and passed A2 (62 and 61 of 64). So A′ cannot show that new data helps; it can only show
that the new models have not lost what TASK-087's had. A′ passes for a model when it meets TASK-089
§5.5's pass rule at h = 8 and A2's two bars.

**A′-FAIL:** M-v2g fails A′ on two or more of its three seeds, **or** M-v1p fails A′ on two or more
of its three seeds (a regressed control could otherwise be "beaten" in the closed loop). The task
escalates and no clause fires. Below that, a seed that fails A′ alone stays in its arm's pooled
count; its A′ result is reported beside its closed-loop count. M-v1's A′ results are reported
only.

**Reported only (diagnostics adopted from the model-first design, §15):**

- **The grasp-outcome readability probe.**
  - Label: on Stage A′'s branches, the outcome "the target is in a lift state at step h"
    (h = 8, 16).
  - Probe: a cross-fitted logistic probe (5 folds by root, salt 9004).
  - Fitted three ways:
    - on the encoded real frame plus measured state at h (**ceiling**);
    - on the state alone (**proprio floor**);
    - applied to each model's **predicted** latent and state at h, fitted on that model's
      predictions over the other folds.
  - Reported: AUROC with a root-bootstrap interval, for M-v1, M-v1p and M-v2g.
  - Use: offline only. Simulator truth is the label of an offline probe here. It never enters a
    model input or a planning cost.
- **Latent-only wins.** The latent-only (G-lat) win rates on lift-vs-release and lift-vs-stay. On
  lift-vs-release, TASK-089's latent cost on real frames was at chance at h = 8 (32 / 64); on
  lift-vs-stay its latent ceiling was 63 / 64.
- **Rise in the branches.** The lift branch's rise within 16 commands. In TASK-089's Stage A
  record ([zero_shot_grasp_v1.md](zero_shot_grasp_v1.md)), "the lift branch raised the target by
  at least 5 cm on 21 of 64 roots".

## 7. Closed loop

### 7.1 Arms (all on the same goals and resets, paired)

| Arm | Model | Planner | Role |
| --- | --- | --- | --- |
| **M-v2g-{90100,90101,90102}** | M-v2g | TASK-089's G-chg (step-averaged cost) | **primary, pooled** |
| M-v1p-{90100,90101,90102} | M-v1p | the same | size control, gated comparison |
| M-v1-{87100,87101,87102} | TASK-087's G | the same | reported ("more data"), no claim; in Stage BS 87100 only (87101 and 87102 dropped by §14's rule, R27.25, §17) |
| ik | — | TASK-088's IK follower with e9's press | ceiling; its BD rate sets the bar |
| ik-nopress | — | the same without the press | reported |
| hold, random | — | TASK-088's | floors |

### 7.2 Stages, bar and rows

- **Stage BD (development, not gated):** 32 resets, 90200–90231.
  - Arms: ik, ik-nopress, hold, random, and seed 90100 of M-v2g and M-v1p, plus M-v1-87100.
  - The bar is **bar = min(0.40, 0.9 × r)**, with r ik's success rate on BD (TASK-089's rule). ik's
    earlier rates were 24/32, 12/16, 50/64, 28/32 and 48/64, so the rule returns the declared 0.40
    unless r < 0.445. If r < 0.30 the bar is not set and the task escalates.
  - BD can show a defect, but by itself it changes no knob of §2–§7.
- **GO:** an independent reviewer posts GO (or NO-GO) on BD's record before Stage BS.
- **Stage BS (gated):** n = 128 fresh resets, 904000–904127, disjoint from TASK-088's and
  TASK-089's cohorts. (Declared n = 64, 904000–904063; Stage 0's power rule below enlarged it,
  R27.22, §17.) Every arm
  of §7.1 runs at the frozen revision from a clean tree.
- **Notation.** k_g is M-v2g's pooled successes of 3n (three seeds × n resets; 192 at n = 64), and
  k_c is M-v1p's. The **pooled bar** is 3 × ⌈bar × n⌉, which is 78 / 192 at bar 0.40 and n = 64.
  The **per-seed floor** is ⌈⌈bar × n⌉ / 2⌉, which is 13 / 64 at bar 0.40 and n = 64. At the
  frozen n = 128 and bar 0.40 they are 156 / 384 and 26 / 128.
- **Gating tests are at the reset level**, because the three seeds of an arm share each reset and
  only the n resets are independent. For each reset r, let G_r be M-v2g's successes over its three
  seeds (0–3), C_r M-v1p's, and R_r random's (0 or 1).
  - "M-v2g beats M-v1p": an exact one-sided sign-flip permutation test at p < 0.01 over the n
    per-reset differences d_r = G_r − C_r (the exact null distribution of Σ d_r under independent
    sign flips of the nonzero d_r, computed by enumeration/convolution, not sampled).
  - "M-v2g beats random": the same test over d_r = G_r − 3 R_r. This use of the sign-flip test
    is approximate: G_r and 3 R_r do not have the same support, so the null of symmetric d_r holds
    only approximately; with random lifting nothing (TASK-088 and TASK-089: 0 lifts) d_r ≥ 0 on
    almost every reset and the test is conservative.
  - This removes the arbitrary pairing of seed i with seed i and the triple reuse of random's
    outcomes.
- **Reported beside every test, not gating except as named in row 5:** the exact one-sided McNemar
  test over the 3n (seed index, reset) pairs, and a reset-cluster bootstrap 95 % percentile interval
  of the pooled rate difference (k_g − k_c) / 3n (10 000 resamples of the n resets, salt 9006).
- **Power.** The declared alternative is per-seed success rates of 0.42 for M-v2g against 0.25 for
  M-v1p. Stage 0 estimates the reset-level test's power at it by simulation, with independent
  per-arm, per-reset logit effects of SD 0, 1 and 2, and records all three. **If the power at SD 1
  is below 0.80 with n = 64, Stage BS uses n = 128** (904000–904127). If it is below 0.80 at
  n = 128 too, this is disclosed and escalated to a ruling before the freeze.
- **Rows** (first match):
  1. **B-VOID** — a run did not complete under the rules, a check failed (goal, checkpoint,
     implementation hash, pinned state moments, λ, seed or feature store), or a cohort was run
     twice.
  2. **B-PASS** — k_g ≥ the pooled bar, **and** every M-v2g seed ≥ the per-seed floor, **and**
     M-v2g beats random, **and** M-v2g beats M-v1p.
  3. **B-COMP** — M-v2g beats M-v1p, but k_g is below the pooled bar, a seed is below the per-seed
     floor, or M-v2g does not beat random. The composition helped, but the bar was missed. The task
     escalates; no clause fires.
  4. **B-SIZE** — k_g ≥ the pooled bar and M-v2g beats random, but it does not beat M-v1p. Recorded
     as **size, not composition**: more data of either kind may help, and the composition is not
     shown. The task escalates; **no clause fires**. B-SIZE does not by itself admit a Phase 4
     preregistration; that is for a ruling. B-SIZE does not apply the per-seed floor, because it
     carries no claim.
  5. **B-FAIL** — M-v2g does not beat M-v1p, k_g is below the pooled bar or M-v2g does not beat
     random, **and** the reset-cluster bootstrap 95 % upper bound of (k_g − k_c) / 3n is below
     δ = +0.10. **The clause fires** (§9).
  6. **B-UNDER** — as B-FAIL, but that upper bound is ≥ δ: the cohort cannot rule out a composition
     gain of δ. The task escalates; no clause fires.
- **Reported with every row:**
  - each seed's count with its exact 95 % interval (a restatement of B-PASS quotes each seed's
    count, not only the pooled one);
  - M-v2g against M-v1, and M-v1p against M-v1 (the "more data" readings);
  - ik against M-v2g;
  - per-object results;
  - grasp contact, maximum rise, and subgoals reached or timed out;
  - executed grasp commands per phase (as TASK-089 §2);
  - stopped episodes and planning seconds;
  - two MP4 replays (one success if any, one failure);
  - a determinism re-run of 4 resets of M-v2g-90100 (the first four BS seeds), which must match
    bitwise. A mismatch is B-VOID.
- **"LeWM needed" is not claimed by any row.** The IK follower uses the same goal information and
  no world model.
- **B-PASS admits a Phase 4 preregistration** (pick-and-place with subgoals). It does not
  preregister it.

## 8. Rows of the task, in order

| Row | When | Next step |
| --- | --- | --- |
| C-FAIL | Stage C fails | escalate; no clause |
| T-VOID | a training job fails a check or runs twice | fix under its own ruling |
| E-FAIL | Stage E fails | escalate; no clause |
| A-CEILING | A-K0: h = 8 does not qualify (Stage 0, before the freeze) | escalate; no clause |
| A′-FAIL | Stage A′ fails (§6.2) | escalate; no clause |
| B-VOID | §7.2 row 1 | fix under its own ruling |
| B-PASS | §7.2 row 2 | admits a Phase 4 preregistration |
| B-COMP | §7.2 row 3 | escalate; no clause |
| B-SIZE | §7.2 row 4 | escalate; no clause |
| B-FAIL | §7.2 row 5 | the clause fires |
| B-UNDER | §7.2 row 6 | escalate; no clause |

- After any VOID, the defect is fixed under its own reviewed ruling, and the stage runs again on
  fresh seeds named by that ruling. A voided cohort is never re-run.
- After a NO-GO, BD is recorded again on fresh development seeds.
- Escalations go to a ruling in `docs/DECISIONS.md`. Each closes the task with a recommendation
  unless the ruling says otherwise.

## 9. Abandonment clause

**Fires on B-FAIL only.**

**It closes:** adding more scripted grasp-and-lift play to `play-v1` (e9's pick recipe run more
often, held longer or retried, as `play_policy_grasp_v0` does) as the grasp fix, under TASK-087's
recipe and latent and TASK-089's step-averaged planner. After it fires, no further variant of this
kind is preregistered for grasping without new evidence of a different kind. That covers the skill
mix, mode mix, hold length, retries, target weighting or added amount of this collector.

**The next grasp step is then a model change.** For example, a model trained to predict the grasp's
outcome (contact and object rise) over the planning horizon, the model-first design of §15. It
needs its own preregistration.

**Not closed:**

- LeWM, DINOv2 and CEM;
- `play-v1` and play-v2g as bases;
- other kinds of data: real G1 data, a different demonstrator or pick recipe, other sensors;
- model changes;
- reach planning (TASK-088's Z3-REACH stands);
- Phase 4 as a goal, and the product goal.

**Neither does it fire on** C-FAIL, A-CEILING, E-FAIL, A′-FAIL, B-COMP, B-SIZE or B-UNDER.
M-v1's counts and the replication of TASK-089's seeds are recorded with the clause, not as a way
around R26.33.

## 10. Seeds and salts

| Use | Seeds |
| --- | --- |
| Added play-v1p episodes | 910000–913199 |
| Added play-v2g episodes | 920000–923199 |
| Model seeds | 90100–90102 |
| A-K0 roots | 90000–90031 |
| Stage A′ roots | 903000–903063 |
| Stage BD resets | 90200–90231 |
| Stage BS resets | 904000–904063 (904000–904127 if §7.2's power rule enlarges BS) |
| Stage 0 debug collection (never counted) | 930000–930299 (`play_policy_grasp_v0`), 930300–930399 (`play_policy_v0`) |
| Debug (plumbing and smoke; never counted) | 90900–90999 |

**Salts:**

- Layouts and goals use TASK-088's 8801 and 8802; CEM 8803 and random 8804, as TASK-088.
- New salts:
  - **9002**: A′'s action-shuffled deal;
  - **9003**: `play_policy_grasp_v0`'s policy salt;
  - **9004**: the readability probe's folds;
  - **9005**: the new collections' split rank;
  - **9006**: the reset-cluster bootstrap of §7.2.
- play-v1p uses TASK-085's policy salt 8701.

**Checked:** `git grep -w` on every remote branch over `src/`, `scripts/`, `docs/`, `.mc/` and
`tests/` finds no seed or salt use of these ranges or salts (904064–904127 and 930000–930399
re-checked over the same paths plus `benchmarks/`). The hits that are not seed or salt uses:

- 90000 appears as the exclusive end bound of TASK-089's `range(89950, 90000)`;
- 9001 appears as a test's `max_seconds` value, so 9001 is skipped;
- 0.90999 appears as a loss value in `mvp_training_results.md` (line 62).

Labels 900000–901999 are TASK-077's and are avoided.

## 11. What this cannot show

- **Scope.** One scene, one arm and hand, one camera at 112 px, simulation only, one encoder and
  latent, one training recipe and budget, one planner. The added data comes from the same privileged
  scripted pick recipe as `play-v1`, so the model sees more of one demonstrator's grasps, not
  different grasps.
- **The bundle.** §2.1's changes act together (skill mix including no poke, mode mix, hold,
  retry). A B-PASS would not say which of them mattered.
- **Pooled seeds.** The pooled count can hide a weak seed; the per-seed floor limits this, and every
  restatement quotes each seed's count.
- **Cohort bias.** Goals come from TASK-088's scripted demonstrator, so the cohort is biased to
  layouts a scripted pick can do (no banana).
- **What the model cannot see.** The model has no object, contact or tactile input. If the gap is in
  how the cost reads a grasp (TASK-089: A1's wins rest partly on predicted joints, "and for
  lift-vs-release at h = 8 wholly"), more data may not move it. That would be the clause's case.
- **The velocity stop.** It ended 611 of TASK-085's 628 early-ended episodes, and it also truncates
  play-v2g's post-grasp phases. It is not changed, because changing it would change the embodiment.
- **Undertraining.** U is TASK-087's capped budget, so both new arms may be undertrained alike.
- **What a pass shows.** A B-PASS would be the first gated world-model grasp-and-lift at the plan's
  bar in this project, on these terms only.

- **The manipulation is weaker than designed.** The DRAFT designed for about 3× the lift windows
  in the added episodes and about 2× in the training set. Stage 0 measured about 2.3× and 1.6× on
  debug episodes (§17), and the bars were revised to C1 ≥ 2.0 and C2 ≥ 1.5 (R27.21). Any
  restatement of a B row quotes Stage C's measured C1 and C2 beside it. If the clause fires (§9),
  its closure of "more scripted grasp-and-lift of this kind" rests on this weaker enrichment, not
  on the designed one.

## 12. Runtime, disk, paths and evidence

**GPU.** Every GPU use is one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` job, with the
lock taken per job: collection (EGL rendering), featurisation, training, A-K0, A′, and each
closed-loop run.

**Estimates**, scaled from TASK-085, TASK-087 and TASK-089's records:

| Step | Estimate |
| --- | --- |
| Collection, 2 × 3 200 episodes (TASK-085: 1 025 s for 3 200 with 12 workers) | about 0.6 h |
| Featurisation of 6 400 episodes (TASK-087's Stage F: 1 804 s for 3 034) | about 1.1 h |
| Training, 8 jobs at 31–37 min (TASK-087, stores on the SSD; see the disk note below) | about 4.5–5 h |
| Stage E, A-K0 and Stage A′ | about 2 h |
| Stage BD, 3 world-model runs × 32 resets plus baselines | about 2 h |
| Stage BS, 9 world-model runs × 64 resets (TASK-089's Stage BS, every arm with five world-model runs × 64, ran 2026-10-11 05:07–10:46 UTC: R26.31, `~/develop/emai/evidence/task089-bs/stage.log`) | about 10–11 h (about double at n = 128) |
| **Total** | **about 20–22 GPU hours** as drafted; about 25–27 after Stage 0's n = 128 (§17, R27.22, R27.25); about 5–7 days wall-clock with reviews and rulings |

**Disk.** About 25 GB for the two collections (TASK-085: 12.3 GB per 3 194 episodes), about 14 GB
for the new feature stores, and small checkpoints. All large artifacts live on the data disk
(`/mnt/data/emai/projects/open-embodied-jepa/{data,checkpoints,outputs}`). They are reached through
the repository's `data/`, `checkpoints/` and `outputs/` links, because TASK-087's code and
`run_task087.sh` use repo-relative paths. **The data disk is rotational** (`/mnt/data` is an HDD),
while TASK-087's 31–37 ms per update was measured with stores on the SSD. `FeatureStore` memory-maps
the latents by default and each update gathers 64 random windows of 9 frames, so a cold read from
the HDD costs a seek per window. The training driver therefore opens each combined train store with
`mmap=False` (about 12.8 GB of latents per arm, against 30 GB of RAM, one training job at a time
under the lock), or stages the store on the SSD; Stage 0's smoke times ms per update on the real
path and records which. (Stage 0 chose the in-memory store, `mmap=False`: 37.0 ms per update
against 40.7 ms memory-mapped, R27.24, §17.) Before any stage runs, Stage 0 checks two things. First,
that those links resolve to the data disk. Second, that `play-v1`, TASK-087's feature report,
projection and 18 checkpoints still verify by sha256 after the move. The runner refuses to write
below 10 GiB free.

**Evidence.** Stored in `~/develop/emai/evidence/task090-*/` with `SHA256SUMS`. Existing evidence
under `data/`, `checkpoints/` and `outputs/` is read and never written.

## 13. Disclosed development before this draft

On `play-v1`'s 2 874 train episodes (1 040 588 commands, 1 020 470 windows), a read-only script
over the sidecars measured shares that are **not recorded, gated or reviewed anywhere**. They
shaped this design and the Stage C bars, and they are not results. "Near" means the right palm is
within 8 cm of the object centre.

| Measure | Share |
| --- | --- |
| Steps: requested grasp > 0.5 | 26.1 % |
| Steps: grasp > 0.5 and palm near any object | 0.78 % |
| Steps: grasp contact, any object | 4.1 % |
| Steps: lift state, any object | 2.9 % |
| Windows: containing grasp contact, any object | 4.65 % |
| Windows: containing a lift state, any object | 3.14 % |
| Windows: containing a lift state of the apple | 0.76 % |

So most closed-hand steps are away from any object. The script is
`/tmp/claude-1000/task090-scratch/steps.py` (ephemeral); **its output was not stored**, and the
shares above are copied from its console output. Stage 0 copies the script to
`~/develop/emai/evidence/task090-prototype/`, re-runs it there with its output saved, and checks the
frozen Stage C label script against it on `play-v1` before the label script is frozen. No GPU job, collection or training was run for this draft.

## 14. Stage 0, freeze and what Stage 0 may change

**Stage 0** is its own PR, merged before the freeze. It adds only opt-in modules and changes no
existing file, so TASK-087's implementation hash and TASK-085, TASK-088 and TASK-089's frozen code
stay as they are:

- `play_corpus_grasp.py`: `play_policy_grasp_v0` and its collection, reusing `play_corpus.py`'s
  scene, storage and sidecars;
- `grasp_corpus_wm.py`: the combined feature stores, the common selection set, Stage C's labels and
  bars, cohorts and rows; NumPy at import;
- a training driver that calls TASK-087's model and loss code over the combined store;
- a planner loader for the new checkpoints (TASK-088's guards, with the new feature-report pins);
- `scripts/{collect,measure,train,run,summarize}_task090.py` and a stage script;
- tests: the policy's skill and mode draws, hold and retry, the label and window rules, the bars and
  rows, the seed checks, and the core-import check.

**Stage 0 also runs:**

- a debug collection of 300 `play_policy_grasp_v0` and 100 `play_policy_v0` episodes on the debug
  collection seeds (§10), with the Stage C ratios and their intervals measured as §4 says and
  disclosed;
- the path and hash checks of §12;
- A-K0;
- a debug smoke of every arm, including a timed training smoke on the real store path;
- the power estimate of §7.2.

**Stage 0 may change only:**

- the per-run process count;
- the collector or the Stage C bars, if the debug ratios fall short (§4);
- the BS cohort size, only by §7.2's power rule (64 → 128);
- the train store's location (data disk or SSD) and mmap mode;
- the dropping of M-v1's 87101 and 87102 runs, if the smoke projects Stage BS above 14 GPU hours.

Each such change is recorded. The protocol is frozen at the Stage 0 merge. Every later stage runs at
the merged revision from a clean tree.

## 15. Why the data-first design, and what it takes from the model-first design

Two designs were drafted. One was **data-first**: this document. The other was **model-first**: a
grasp-outcome head (contact and object rise) on G, trained on `play-v1`, with a cost term on its
prediction. R27.2 chose data-first, for five reasons:

1. **It was the recommendation given to the owner**, and R26.34 lists it first.
2. **The data is upstream of both options.** Lift-state windows are about 3 % of the training
   windows (§13, a development reading). A head trained on those positives would learn from the
   thin data that this task would thicken. If the data change fails, the model change is still
   open, and it gets the better corpus.
3. **It needs no new rule exception.** The model-first cost term acts on a learned prediction of
   simulator truth. That needs its own ruling against the project rule that simulator truth must
   not reach planning cost. The data-first design keeps TASK-089's cost unchanged.
4. **Attribution is clean.** The size control separates "what the data contains" from "how much
   data there is". The model-first design could only compare against TASK-087's checkpoints.
5. **The odds are similar, and the cost is acceptable.** The drafters judged a gated pass at about
   15–25 % (data-first) and 10–15 % (model-first); these are judgements, not measurements. This design costs about 20–22 GPU hours (about 25–27 after Stage 0, §17) against about 13–15, and it yields a reusable
   corpus either way.

**Taken from the model-first design:**

- the measured-ceiling-first rule and the proprio-only floor, as the reported readability probe of
  §6.2. It is not a cost term;
- the "beats the unchanged control" requirement inside the pass, as M-v2g against M-v1p;
- a NEAR-style escalation row, B-COMP;
- all three seeds gated and pooled, against the seed variance both drafts named as a risk.

## 16. Rulings this document records (R27.1–R27.20; DRAFT; decided by Claude under owner delegation)

- **R27.1** — TASK-090 follows R26.34's data option, as a fresh preregistration. It is admitted by
  R26.33, because it changes the data and trains new checkpoints. No planner, cost, horizon or
  action-parametrisation variant over TASK-087's checkpoints is introduced.
- **R27.2** — data-first is chosen over model-first, for §15's reasons. The model-first design is
  the declared next step if the clause fires.
- **R27.3** — the changed factor is corpus composition at equal size, play-v2g against play-v1p
  (§2).
- **R27.4** — `play_policy_grasp_v0` as §2.1. `play_policy_v0` stays frozen.
- **R27.5** — everything in §2.2 is held fixed, including the pinned projection, `play-v1` train's
  state and palm moments for every new model and the planner cost, U = 60 000, the common selection
  set and λ = 2.0552 for every arm.
- **R27.6** — `last_two_triggered` is reported, not an escalation (§2.2).
- **R27.7** — the models and seeds of §3, with all three M-v2g seeds gated and pooled. M-v1 is a
  reported control re-run that cannot carry a claim.
- **R27.8** — Stage C, the data gate: C1 ≥ 2.5×, C2 ≥ 1.75× and C3 within ±25 %. The bars are
  declared and may be revised only at the freeze, from a 300-episode debug collection (§4).
- **R27.9** — Stage T (§5).
- **R27.10** — Stage E (§6.1).
- **R27.11** — A-K0 (with the A-CEILING row) and Stage A′ as a no-regression gate on both M-v2g
  and M-v1p, with the reported readability probe and diagnostics (§6.2).
- **R27.12** — the closed-loop arms (§7.1).
- **R27.13** — the bar min(0.40, 0.9 × r), the pooled bar 3 × ⌈bar × n⌉, the per-seed floor
  ⌈⌈bar × n⌉ / 2⌉, reset-level exact sign-flip tests at p < 0.01, and the power rule (§7.2).
- **R27.14** — Stage BD, an independent GO, then Stage BS (§7.2).
- **R27.15** — the rows: C-FAIL, T-VOID, E-FAIL, A-CEILING, A′-FAIL, B-VOID, B-PASS, B-COMP,
  B-SIZE, B-FAIL and B-UNDER (§7.2, §8).
- **R27.16** — the abandonment clause, firing on B-FAIL only (§9).
- **R27.17** — seeds and salts (§10).
- **R27.18** — reported quantities, the determinism re-run and the two MP4s (§7.2).
- **R27.19** — Stage 0, the freeze, and what Stage 0 may change. Runtime, disk, paths and evidence
  (§12, §14), with every GPU use under `gpu_run.sh` and the lock taken per job.
- **R27.20** — R7 does not change, whatever the row.
- **R27.21–R27.28** — the Stage 0 record, §17.

## 17. Stage 0 record (R27.21–R27.28; decided by Claude under owner delegation)

Stage 0 ran on 2026-10-12 in the worktree of the Stage 0 PR (branch `feat/task090-stage0`), every
GPU use one `scripts/gpu_run.sh` job. Evidence: `~/develop/emai/evidence/task090-stage0/`
(`SHA256SUMS` sha256 `d9c52b02…2de0`) and `~/develop/emai/evidence/task090-prototype/`. Large
artefacts are on the data disk: the debug collections `data/task090-debug-{grasp,v0}`, their
features `data/task090-debug-features-{grasp,v0}`, the debug models
`checkpoints/task090-debug/`.

- **Code** (opt-in; no registry entry; `import embodied_jepa` stays torch- and MuJoCo-free, checked
  by a test; no existing file changes, so TASK-087's implementation hash `e2ec0096…49b1` and
  TASK-085's, TASK-088's and TASK-089's code are untouched):
  `src/embodied_jepa/play_corpus_grasp.py` (`play_policy_grasp_v0`, its episode and storage),
  `src/embodied_jepa/grasp_corpus_wm.py` (cohorts and seeds; the frozen Stage C label script and
  bars; the combined feature store; the pinned-moments check; Stage E's and A′'s rules; the
  readability probe; the exact sign-flip test, the reset-cluster bootstrap, the pooled bar, the
  per-seed floor and the rows; the power rule; NumPy at import),
  `src/embodied_jepa/grasp_corpus_wm_runtime.py` (the loader for the new checkpoints with TASK-088's
  guards plus the feature-report pins and the pinned moments; featurisation with the pinned
  projection; TASK-089's G-chg controller), `scripts/collect_task090.py`,
  `scripts/measure_task090.py`, `scripts/train_task090.py`, `scripts/run_task090.py`,
  `scripts/summarize_task090.py`, `scripts/run_task090_stage.sh` and
  `tests/test_grasp_play_corpus.py` (21 tests).
- **§12's checks** (`readings/verify.json`): `data/`, `checkpoints/` and `outputs/` resolve to
  `/mnt/data/emai/projects/open-embodied-jepa/` on `/dev/sda1`, which is rotational. After the
  move, `play-v1` verifies (corpus.json against TASK-085's sha256, every shard's files re-hashed
  by `DatasetStore`, 75 s), and so do TASK-087's feature report (`f5648a35…7eae`), its train and val
  store files, the projection file and arrays (`c196fc48…e33f`) and all 18 checkpoints (each
  `model.pt` against its report, each report byte-identical to its copy in
  `evidence/task087-val`). Disclosed: TASK-087's feature store itself was never on the moved
  directories; it lives in TASK-087's run worktree (`worktrees/task087-run/outputs/`, on the SSD),
  and Stage T reads `play-v1`'s part of each combined store from there in place. **That worktree
  (`/home/huhn/develop/emai/worktrees/task087-run`) must not be archived or removed until TASK-090
  closes**; Stage T, E, A′, BD and BS read its feature store and projection.
- **§2.2's moments check:** the buffers `set_moments` makes from `play-v1` train's moments equal
  G-87100's `state_mean`, `state_scale`, `palm_mean` and `palm_scale` bitwise (sha256
  `1e960227…81b5c` both). The training driver repeats the check before training and on the kept
  checkpoint, and the loader checks every new model's buffers against it.
- **§13's prototype check** (`readings/prototype-check.json`): the development script, copied to
  `evidence/task090-prototype/steps.py` and re-run there with its output saved
  (`steps.output.txt`), reproduces the DRAFT's shares (windows with a lift state 3.14 %, steps
  2.87 %, grasp-contact windows 4.65 %). The frozen label script, run in the script's convention,
  matches it exactly on `play-v1` train (2 874 episodes, 1 040 588 steps, 29 841 lift steps,
  1 020 470 windows, 32 066 lift windows). In the protocol's convention (each window's 9 frames,
  the episode's last frame included; the on-table mask from the layout) the count is the same,
  32 066 (3.142 %).
- **The debug collection** (`readings/debug-readings.json`; 300 `play_policy_grasp_v0` episodes,
  930000–930299, at `ffb521a`; 100 `play_policy_v0` episodes, 930300–930399, at `d1a3878`; all
  stored, none short or failed; every stored episode counted, not only the train splits). Stage C's
  ratios with episode-bootstrap 95 % intervals (10 000 resamples of both `play-v1` train and the
  debug episodes):

  | Reading | Point | 95 % interval | Declared bar |
  | --- | --- | --- | --- |
  | C1 (debug grasp share / `play-v1` train share) | 2.31 | 1.68–3.03 | ≥ 2.5 |
  | C2 (projected to equal-size additions) | 1.64 | 1.33–1.99 | ≥ 1.75 |
  | C3 direction (debug v0 share / `play-v1` train share) | 0.50 | 0.12–1.00 | 0.75–1.25 |

  Lift-window shares: `play-v1` train 3.14 %, debug grasp 7.26 %, debug v0 1.57 %. Reported: the
  debug grasp episodes lifted something in 61 of 300 (`play-v1` train: 299 of 2 874, debug v0: 5 of
  100); their collector's first-attempt lift check passed on 41 of 235 picks that reached it, so it
  retried 194 times; 56 retries reached their own check (the others were cut by the episode's
  end or the velocity stop) and 7 of those lifted (12.5 %); 86 of 300 ended early, 83 of them on the
  measured-velocity stop at `right_hand_thumb_1_joint` (`play-v1` train: 564 of 2 874); modes
  scripted 164 and perturbed 136; skills pick_place 326, pick_retry 194 (the retry's log entries,
  one per failed first check), push 46, wander 23; per object (lift
  episodes) apple 19, cube 21, can 20, banana 1; grasp-contact windows 10.0 % (`play-v1` 4.65 %).
  The C3 direction reading rests on 5 lifting episodes; no file under `play_policy_v0` (the play
  scene, embodiment, simulation, v2 contacts, the resting expert, contracts, configs) changed since
  `play-v1`'s revision `3ee2557`, so play-v1p's policy is TASK-085's exactly.
- **R27.21 — Stage C's bars are revised at the freeze, by §4's revision rule; the collector is not
  changed.** Both debug point ratios are below their declared bars, so §4 requires a revision. At
  full size (`readings/debug-projection.json`: 2 000 draws of 2 880 added episodes each, play-v2g's
  from the debug grasp episodes and play-v1p's from `play-v1` train, the same policy) the declared
  bars would be met in 15 % (C1) and 7.5 % (C2) of draws, so Stage C would almost surely end
  C-FAIL. The collector is kept because what limits enrichment is the pick recipe's own success
  rate (17 % of first attempts lift), which §2.1 holds fixed; changing it would be a different
  collector, designed after seeing debug data. The bars are re-set by the DRAFT's own derivation
  applied to the measurement: **C1 ≥ 2.0** (below the debug point 2.31, as 2.5 was below the
  estimated 3) and **C2 ≥ 1.5 = (1 + C1) / 2**; C3 stays at ±25 %, unrevised (§4's revision rule
  covers C1 and C2 only). **C3 is a real risk.** The projection's C3 (0.85–1.18) draws play-v1p
  from `play-v1` train itself, so it assumes what C3 tests and is not evidence for it. The direct
  debug reading, 0.50 (0.12–1.00), is outside ±25 %: 5 of 100 debug v0 episodes lifted
  something against 10.4 % of `play-v1` train (binomial P(X ≤ 5 | 100, 0.104) ≈ 0.045). The code is
  unchanged, so chance is the likely reading, but a C-FAIL on C3 is possible; it stops the task
  before any GPU training (§4, §8), and no clause fires. At the revised bars the projection meets C1 in 98 % and C2 in 98 % of draws (an
  optimistic projection: the 300 debug episodes stand in for the policy, and C3's projected draws
  are circular, as above). **What this costs:** the
  manipulation is weaker than designed: about 2.3× the lift windows in the added episodes and
  about 1.6× in the training set, not about 3× and 2×. A null result (B-FAIL) is therefore a
  weaker statement about composition than the DRAFT anticipated; the clause's scope (§9) is
  unchanged.
- **R27.22 — Stage BS has n = 128 (904000–904127), by §7.2's power rule** (`readings/power.json`;
  20 000 simulations per cell; per-seed marginal rates 0.42 and 0.25, the logit intercepts set so
  that the marginal rates hold under the per-arm, per-reset N(0, SD²) logit effects, each effect
  shared by the arm's three seeds; the generator keyed by [9006, n, 10·SD]):

  | n | SD 0 | SD 1 | SD 2 |
  | --- | --- | --- | --- |
  | 64 | 0.852 | **0.727** | 0.589 |
  | 128 | 0.995 | **0.974** | 0.918 |

  The power at SD 1 is below 0.80 at n = 64 and above it at n = 128, so n = 128; nothing escalates.
  At bar 0.40 the pooled bar is 156 / 384 and the per-seed floor 26 / 128.
- **R27.23 — A-K0** (`k0/`; calibration roots 90000–90031 at `23d01ce`, clean tree, NVIDIA EGL,
  under the GPU lock; 45 rejected draws; targets cube 17, can 11, apple 4, banana 0): the
  simulator ceiling's win rates (of 32) and the bars by TASK-089 §5.4's rule, frozen as
  `grasp_corpus_wm.K0_CEILING` / `A1_BARS`:

  | Cost family | h | close-vs-open | lift-vs-release | lift-vs-stay |
  | --- | --- | --- | --- | --- |
  | latent + pose (G) | 8 | 1.000 → 61 / 64 | 1.000 → 61 / 64 | 1.000 → 61 / 64 |
  | latent + pose (G) | 16 | 1.000 → 61 / 64 | 1.000 → 61 / 64 | 1.000 → 61 / 64 |
  | latent (G-lat, N) | 8 | 0.906 → 56 / 64 | 0.563: **not gated** | 0.969 → 59 / 64 |
  | latent (G-lat, N) | 16 | 0.969 → 59 / 64 | 0.844 → 52 / 64 | 1.000 → 61 / 64 |

  h = 8 qualifies (and h = 16), so the row is not A-CEILING. The ceiling's pose term uses G-87100's
  buffers, which are the pinned moments (checked bitwise; the model's weights are not used).
  `k0/ceiling-k0-roots.json` sha256 `7de45892…98bd`, `k0/summary-k0.json` `80324b3f…a3a9`.
- **R27.24 — Stage T's store: read into memory, from where it is.** The timed smoke trained G for
  1 000 updates on the real store path (`play-v1` train from TASK-087's store plus the debug grasp
  store on the data disk), each mode from a cold page cache (`posix_fadvise` DONTNEED): read into
  memory, 37.0 ms per update (median; p90 37.1; TASK-087's G-87100 36.5 ms) after an 8.2 s load;
  memory-mapped, 40.7 ms (mean 43.2, p90 47.3) with only 0.6 GB on the HDD (a real store puts
  about 6.4 GB there). So Stage T uses `mmap=False` (`TRAIN_MMAP`); no store is staged on the SSD.
  The per-job cap is TASK-087's rule, 3 × seconds per update × U, with 40 ms: 7 200 s
  (`TRAIN_CAP_SECONDS`). `train_task090.py` enforces both in non-debug runs.
- **R27.25 — M-v1's 87101 and 87102 runs are dropped from Stage BS, by §14's rule; the process
  counts are unchanged** (8 for baselines, 6 for world-model arms). TASK-089's Stage BS world-model
  runs took 3 623–4 321 s per 64 resets at 6 processes (planning 0.95–1.01 s per command); the
  smoke's took 0.89–0.91 s per command. At n = 128 a world-model run is about 2.2 GPU hours, so the
  nine runs of §7.1 project to about 20 GPU hours, above 14. Without the two dropped runs, seven
  runs project to about 15.5 GPU hours (plus a few minutes of baselines, goals and the
  four-reset determinism re-run): still above 14, disclosed. §14's 14 GPU hours is the trigger for
  this drop, not a cap on Stage BS; of §14's knobs only the process count could still change, and
  it stays (more processes would contend for the one GPU and not shorten the run). M-v1's
  closed-loop reading in Stage BS is 87100 alone (its "more data" comparisons scale its per-reset
  successes by 3); Stage BD and Stage A′ keep their M-v1 runs.
- **R27.26 — readings of points the DRAFT left open** (each the reading closest to TASK-089's
  Stage 0 or to TASK-085/087's code):
  1. The hold is U(40, 80) inclusive (40–80 commands) at the lift waypoint's target with the pick's
     grasp command (so a `no_close` pick holds open). It follows the lift waypoint of a first
     attempt and of the retry; a `drop` pick holds, then drops as in `play_policy_v0`.
  2. The retry check runs on the first command after the lift waypoint ends: the target's height
     against its height on the episode's first recorded frame (the label's start) with grasp
     contact. The retry aims at the target's current position; it keeps the pick's closure, lift
     height, destination and yaw, redraws the aim noise and step counts, and is logged as its own
     skill entry `pick_retry`. A retry that fails its check is not retried. `abort` picks never
     reach a check.
  3. Stage C's denominators are the pooled window counts (the share is Σ lift windows / Σ
     windows); a window is TASK-087's (9 frames, starts 0 … F − 9); "near" for the reported
     closed-hand windows is the right palm within 8 cm of an on-table object's centre over the
     window's 8 commands.
  4. Stage 0's debug ratios use every stored debug episode; the debug C2 projects each added set to
     2 874 episodes at the debug per-episode means.
  5. Stage E: the common val windows are TASK-087's val roots (stride 10, 16 commands) over the
     three val stores; intervals are TASK-087's episode cluster bootstrap (salt 8713) at its gate
     level, 97.5 %; the wrong commands are TASK-087's table (salt 8711). (i) is the copy ratio's
     upper bound below 1; (ii) the sensitivity's lower bound above 1; (iii) G's over N's summed
     error, upper bound below 1; each at h = 4 and 8.
  6. Stage A′: A1's model and shuffled wins use TASK-089's code with the shuffled deal keyed by
     9002; A2 uses the step-averaged setting with TASK-088's CEM seeds; the readability probe
     standardises, projects to at most 32 principal components and fits an L2 logistic regression
     (penalty 1.0), all on the training folds; AUROC intervals use 2 000 root-bootstrap draws.
  7. The new feature stores are `data/task090-features-{v2g,v1p}` (train and val only); the loader
     accepts a new model only with the current sha256 of `play-v1`'s and both new feature reports.
  8. Debug seeds: model 90900–90909, roots 90910–90949, resets 90950–90999 (inside §10's debug
     range).
- **Debug smoke** (`smoke/`; debug seeds and debug models, not results, at `23d01ce`; it changes
  nothing): the four debug models (G and N on each debug store, 600 updates, debug seed 90900)
  trained and loaded under every guard; Stage E on them ran and read E-FAIL (expected after 600
  updates; the common val windows: 180 episodes, 6 340 roots, 274 with a lift state); 8 debug
  roots (15 rejected draws) and A′ ran end to end for all nine G models and both N models (no
  model passes A1 at h = 8 on 8 roots); the closed loop on 8 debug resets (7 rejected draws): ik
  5 / 8, ik-nopress 5 / 8, hold 0 / 8, random 0 / 8, debug M-v2g 0 / 8, debug M-v1p 2 / 8,
  M-v1-87100 0 / 8; the BD summary and an MP4 replay (which matched the run's palm distances)
  worked. The determinism re-run and the BS summary are covered by tests only.
- **Compute.** About 0.7 GPU hours in all (collection 543 s, debug featurisation about 3 min, A-K0
  under 1 min, the timed smoke and four debug models about 3 min, the debug Stage E, roots and A′
  about 2 min, the debug closed loop about 23 min) and a few CPU minutes (§12's checks 86 s,
  labels, power 4 s).
- **R27.27 — the debug smoke is not evidence** and changes nothing in §2–§11.
- **R27.28 — frozen at this PR's merge.** Collection, Stage C, featurisation, Stage T, E, A′, BD and
  BS run at the merged revision from a clean tree with `scripts/run_task090_stage.sh`, evidence in
  `~/develop/emai/evidence/task090-*`. R7 does not change.
- **Review nits of #202** applied here: §10's "Checked" note lists 0.90999 as a hit that is not a
  seed or salt use; §1 quotes TASK-085's lift share without quote marks; §7.2 says the "beats
  random" sign-flip test is approximate (conservative); §7.2 says B-SIZE does not apply the
  per-seed floor because it carries no claim.

## 18. Collection and Stage C record (gated, CPU; R27.29–R27.31; decided by Claude under owner delegation)

At `53f0b72` (the frozen revision, clean tracked tree), worktree `worktrees/task090-run`,
`scripts/run_task090_stage.sh collect` then `scripts/run_task090_stage.sh c`, 2026-10-12. The two
collections each ran as one `scripts/gpu_run.sh` job (NVIDIA EGL, RTX 5080, 12 workers): play-v2g
about 11:42:11–12:07:44 UTC (1 533 s; the stage started at 11:40:29 UTC and waited about 2 min for
the lock first), play-v1p 12:07:44–12:28:46 UTC (1 258 s, no wait). Stage C's labels and gate ran on the CPU at 12:28:56–12:29:25 UTC (29 s). Nothing in
the frozen code, seeds, salts or bars was changed. Evidence `~/develop/emai/evidence/task090-c/`
(`SHA256SUMS` sha256 `9ff11727…c7ac`, 77 files; `stage-c.json` `9610ce6f…82d8`; labels
`labels-play-v1-train.json` `dcf4306d…600d`, `labels-play-v2g-train.json` `69ad0765…742c`,
`labels-play-v1p-train.json` `be60d244…7c8e`; `stage-c-intervals.json` `20b8b545…b877`; the
collectors' manifests and per-shard reports under `collect-play-{v2g,v1p}/`). The corpora are
`data/play-v2g` (11 GB) and `data/play-v1p` (12 GB) on the data disk.

**The collections.** Every one of the 32 shards of each collection completed and was sealed by the
per-shard split rule (salt 9005); no episode raised an error.

| Corpus | Seeds | Policy (salt) | Stored | Discarded short (< 50 commands) | Train / val / test | `corpus.json` sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| play-v2g | 920000–923199 | `play_policy_grasp_v0` (9003) | 3 199 | 1 (922373, a velocity stop at command 49) | 2 879 / 160 / 160 | `9cf51db3…b06a` |
| play-v1p | 910000–913199 | `play_policy_v0` (8701) | 3 197 | 3 (910198, 910795, 913159: projection infeasible) | 2 877 / 160 / 160 | `4935ef49…c19d` |

The test splits were not read.

**Row: C-PASS.** Each check is measured by the frozen label script (`measure_task090.py`) over the
train episodes' sidecars (every sidecar's sha256 checked against its manifest). The row is decided
on the point ratios, as `grasp_corpus_wm.stage_c` and §4 say. The intervals are episode-bootstrap
95 % percentile intervals (10 000 resamples, each of the three episode sets resampled
independently, generator seeded 910000). They are **reported only**: they were computed after the
gate by `stage_c_intervals.py`, which is stored with the evidence and is not part of the frozen
code, with the same formulas as the frozen function.

| Check | Bar | Measured | 95 % interval (reported) | Pass |
| --- | --- | --- | --- | --- |
| C1: play-v2g added / play-v1p added | ≥ 2.0 | **2.383** | 2.080–2.736 | yes |
| C2: combined play-v2g set / combined play-v1p set | ≥ 1.5 | **1.693** | 1.560–1.840 | yes |
| C3: play-v1p added / `play-v1` train | 0.75–1.25 | **1.045** | 0.893–1.228 | yes |

Lift-window shares, as pooled window counts: `play-v1` train 3.142 % (32 066 of 1 020 470 windows,
2 874 episodes; the same count as Stage 0's prototype check), play-v2g added train 7.826 %
(76 818 of 981 555, 2 879 episodes) and play-v1p added train 3.284 % (33 829 of 1 030 186, 2 877
episodes). The full-size readings sit near Stage 0's debug points (C1 2.31, C2 1.64), although those are not like-for-like: the debug C1 used `play-v1` train as its denominator and the debug C2 was a projection to equal-size additions. C3, which
Stage 0 flagged as a real risk from its 100-episode debug reading (0.50), reads 1.045. So the debug
reading was low by chance, as R27.21 judged likely.

**Reported (§4)** (the train episodes of each set; play-v1 train / play-v1p added / play-v2g added):

- **Lift episodes (anything lifted):** 299 (10.4 %), 320 (11.1 %) and 617 (21.4 %). Per object
  (apple, cube, can, banana): 65, 119, 111, 4 / 73, 138, 106, 3 / 145, 257, 202, 13.
- **Lift windows per object** (apple, cube, can, banana): 7 740, 12 306, 11 598, 422 /
  9 526, 14 503, 9 759, 41 / 18 623, 32 297, 24 750, 1 148.
- **Window shares:** grasp-contact windows 4.65 %, 4.84 % and 10.78 %. Closed-hand-near-object
  windows 0.84 %, 0.92 % and 1.89 %.
- **Lift states and rise:** the share of lift states with a rise ≥ 5 cm is 83.5 %, 83.4 % and
  85.0 %. The per-episode maximum rise has p90 0.69, 0.67 and 0.92 cm, and p99 16.1, 15.9 and
  18.6 cm (the median is 0 in all three).
- **Early ends:** 564, 523 and 806. Most were the measured-velocity stop at
  `right_hand_thumb_1_joint` (469, 453 and 774); projection infeasible accounts for 16, 13 and 9.
- **Modes:** scripted 1 128 / 1 202 / 1 617, perturbed 1 042 / 1 025 / 1 262, random 704 / 650 / 0.
- **Skill log entries:** pick_place 2 762 / 2 865 / 3 205; pick_retry – / – / 1 811; push 1 272 /
  1 283 / 412; wander 3 490 / 3 314 / 206; poke 521 / 508 / 0.
- **play-v2g's collector checks:** the first-attempt lift check passed on 439 of 2 250 picks that
  reached it (19.5 %), so it retried 1 811 times. 593 retries reached their own check, and 62 of
  those lifted (10.5 %).

**What this says, and what it does not.** The data gate passes at the bars revised by Stage 0
(R27.21). In its added train episodes, play-v2g holds about 2.4× the lift windows of the
equal-size control. Its combined training set holds about 1.7× those of the control's combined
set. The control's added episodes match `play-v1` train within the ±25 % band. As §11 says, this
is weaker than the DRAFT's designed 3× and 2×. Every restatement of a B row quotes C1 2.38 and
C2 1.69 beside it. This is a data measurement only. It is not a model, offline or closed-loop
result, and it shows nothing about whether training on play-v2g changes grasping. Most of the extra lift
comes from the pick recipe's own success rate: first attempts lift 19.5 % of the time and retries
10.5 %. More lifting episodes also end early on the thumb's velocity stop (806 against 523), which
§11 discloses.

- **R27.29** — Collection and Stage C are recorded: row **C-PASS** (C1 2.383 ≥ 2.0, C2 1.693 ≥ 1.5,
  C3 1.045 within 0.75–1.25). The intervals are reported only.
- **R27.30** — Featurisation and Stage T may run as frozen (`run_task090_stage.sh features`, then
  `t`, at `53f0b72` from a clean tree, one GPU lock per job) **only after this record is reviewed
  and merged**. This record starts neither.
- **R27.31** — R7 does not change.
