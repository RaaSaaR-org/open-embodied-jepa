# TASK-090 — grasp-enriched play corpus: does more scripted grasp-and-lift play data make CEM over a retrained grounded LeWM grasp and lift?

**Status: DRAFT** (R27.1–R27.20, decided by Claude under owner delegation, 2026-09-30). Nothing
here is frozen. The protocol freezes when its Stage 0 PR merges (§14). After that merge nothing in
§2–§11 changes without its own reviewed ruling.

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

- **Recorded (TASK-085):** "Lift 10.4 %" of train episodes lift anything. In random mode it is 0 %.
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
  episodes must be **≥ 2.5×** that of play-v1p's added train episodes.
- **C2, enrichment of the training set:** the lift-window share of play-v2g's combined train set
  (`play-v1` train plus the additions) must be **≥ 1.75×** that of play-v1p's combined train set.
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
| M-v1-{87100,87101,87102} | TASK-087's G | the same | reported ("more data"), no claim |
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
- **Stage BS (gated):** n = 64 fresh resets, 904000–904063, disjoint from TASK-088's and
  TASK-089's cohorts; n = 128 (904000–904127) if Stage 0's power rule below enlarges it. Every arm
  of §7.1 runs at the frozen revision from a clean tree.
- **Notation.** k_g is M-v2g's pooled successes of 3n (three seeds × n resets; 192 at n = 64), and
  k_c is M-v1p's. The **pooled bar** is 3 × ⌈bar × n⌉, which is 78 / 192 at bar 0.40 and n = 64.
  The **per-seed floor** is ⌈⌈bar × n⌉ / 2⌉, which is 13 / 64 at bar 0.40 and n = 64.
- **Gating tests are at the reset level**, because the three seeds of an arm share each reset and
  only the n resets are independent. For each reset r, let G_r be M-v2g's successes over its three
  seeds (0–3), C_r M-v1p's, and R_r random's (0 or 1).
  - "M-v2g beats M-v1p": an exact one-sided sign-flip permutation test at p < 0.01 over the n
    per-reset differences d_r = G_r − C_r (the exact null distribution of Σ d_r under independent
    sign flips of the nonzero d_r, computed by enumeration/convolution, not sampled).
  - "M-v2g beats random": the same test over d_r = G_r − 3 R_r.
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
     preregistration; that is for a ruling.
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
`tests/` finds no use of these ranges or salts (904064–904127 and 930000–930399 re-checked over the
same paths plus `benchmarks/`). There are two exceptions, neither of them a seed or
salt use:

- 90000 appears as the exclusive end bound of TASK-089's `range(89950, 90000)`;
- 9001 appears as a test's `max_seconds` value, so 9001 is skipped.

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
| **Total** | **about 20–22 GPU hours**, and about 5–7 days wall-clock with reviews and rulings |

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
path and records which. Before any stage runs, Stage 0 checks two things. First,
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
   15–25 % (data-first) and 10–15 % (model-first); these are judgements, not measurements. This design costs about 20–22 GPU hours against about 13–15, and it yields a reusable
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
