# TASK-092 — a rate-aware grounded world model: train G to predict what the embodiment executes for any grasp request, then TASK-091's closed-loop grasp-and-lift comparison against the unchanged G

**Status: FROZEN at the Stage 0 PR's merge** (R27.74–R27.88 as the DRAFT, #214; Stage 0 record
§15, R27.89–R27.94; decided by Claude under owner delegation, 2026-10-14; DECISIONS (ax), (ay)).
Stage 0 changes none of the three items §13 lets it change. Every later stage runs at the merged
revision from a clean tree with `scripts/run_task092_stage.sh`.

This is the model change that R27.71 names. TASK-091 ([grasp_play_corpus_v1_planned_close.md](grasp_play_corpus_v1_planned_close.md))
closed at **B-FAIL** (R27.70, R27.72) and its §7 clause fired (R27.71). The clause closes more
scripted grasp-and-lift play of `play_policy_grasp_v0`'s kind as the grasp fix, under TASK-087's
recipe and latent and TASK-089's step-averaged planner. It says the next grasp step is a model
change, with its own preregistration. Model changes are explicitly not closed (R27.71; TASK-089's
R26.33 also leaves them open).

**This is not an Apple→Plate experiment.** It plans grasp-and-lifts of the play scene's objects
with G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not
change, whatever row this task ends in** (R27.88).

## 1. What the records say about why the world-model controllers fail at grasp-and-lift

Every number below is quoted with the qualifiers of its record. The records are TASK-089's results
([zero_shot_grasp_v1_results.md](zero_shot_grasp_v1_results.md)), TASK-090's record
([grasp_play_corpus_v1.md](grasp_play_corpus_v1.md) §19–§20) and TASK-091's record
([grasp_play_corpus_v1_planned_close.md](grasp_play_corpus_v1_planned_close.md) §15–§17).

### 1.1 Recorded

- **Closed loop, TASK-091 Stage BS** (one run, simulation only, 128 fresh gated resets per arm;
  TASK-090 Stage C's C1 2.38 and C2 1.69 apply to the M-v2g and M-v1p corpora). The six world-model
  seeds lifted 6–11 of 128 each. M-v1-87100 (TASK-087's G, reported only) lifted 1 / 128. The
  scripted IK follower, which uses the same goals and no world model, lifted 82 / 128. Where the
  world-model arms fall short:
  - **grasp contact** on 39–87 of 128 resets per seed, against ik's 122;
  - **median maximum rise** 0.18–0.34 cm per seed, against ik's 6.2 cm;
  - velocity stops on 13–45 resets per seed, against ik's 12.
  The record says this "describes what happened. It is not a diagnosis of the cause."
- **TASK-089 Stage BS** (one run, simulation only, 64 fresh gated resets, one gated model seed):
  G-chg-87100 lifted 0 / 64 with grasp contact on 39. The replication seeds 87101 and 87102 lifted
  8 / 64 and 7 / 64 (reported only). ik lifted 48 / 64. The record locates the gap "in executing a
  stable grasp (where the palm is while the hand closes and the arm rises), not in choosing to
  close. Why is not tested here."
- **The grasp channel is rate-limited.** The embodiment clips the right-grasp synergy target to
  the previous target ± `joint_speed_limit_rad_s × control_dt / max slope` =
  2.0 × 0.05 / 0.55 = **2 / 11 ≈ 0.182 per command** (`embodiment.py`, `_prepare_side`, the
  "grasp rate" clip after the arm IK; `configs/g1_sim_action.json`). From an open hand (−1) the
  executed channel first exceeds 0.5 at about the 9th closing command. TASK-091's deferral
  measures are consistent with this: in Stage BS (one run, simulation only) the median first
  executed close in the grasp phase came at command 8–10 in every world-model arm, and in Stage BD
  (development, 32 resets, three world-model arms) the p10 was 8 in every arm and the observed
  minimum 7, "about 0–2 commands after the earliest possible".
- **A planned close is not an executed close.** On reset 90319 (TASK-091 Stage BD, development)
  M-v1-87100 planned the close at step 1 on 46 % of its re-plans but never executed one: "its
  requests alternated close and open, and the rate limit never let the channel build up".
- **The offline separation passed.** TASK-089's A1 (one run, offline, simulation only, 64 fresh
  roots): G's predictions separated a closing from an open hand and a lifting from a releasing or
  staying hand at the object on 64 / 64 roots at h = 8 and 16 (bars 59–61 from the simulator's
  measured ceiling; action-shuffled control 27–33). TASK-090's A′ and TASK-091's A″ reproduced A1
  on every new model. But G's wins "rest partly, and for lift-vs-release at h = 8 wholly, on the
  robot's own predicted joints (G's pose term)": the latent cost on real frames is at chance
  there (32 / 64) and G-lat wins 15 / 64. And the branches A1 compares are executed, rate-feasible
  command sequences.
- **Prediction quality near contact, under recorded commands.** TASK-090 Stage E (one run, offline,
  simulation only; common val windows): G's h = 8 roll-out error on lift windows (0.419–0.431) and on grasp-contact windows
  (0.435–0.449) is not higher than on all windows (0.466–0.473); TASK-087's G 0.435–0.439 and
  0.451–0.456. TASK-087 Stage E (one run, offline, simulation only): G's state head predicts the standardised right arm and hand state with
  MSE 0.012–0.013 at h = 1 and 0.036–0.037 at h = 16 under the executed commands. TASK-090 A′'s
  readability probe (reported only; 16 positives at h = 8): the outcome "the target is in a lift
  state at h" reads from the models' predicted latent and state at AUROC 0.740–0.809, from the
  encoded frame plus state (ceiling) at 0.803, and from the state alone (proprio floor) at 0.974.
  No model's interval separated from another's. So, **under the commands the corpus contains,**
  the records show no special loss of prediction quality near contact.
- **The data change did not move the closed loop.** TASK-091: M-v2g 22 / 384 against M-v1p
  26 / 384 ("M-v2g beats M-v1p" p = 0.767; bootstrap upper bound +0.023 below δ = +0.10).

### 1.2 New in this draft (development, not gated; §11)

The planner does not query the models under the commands the corpus contains.

- **Training support.** In `play-v1` train (2 874 episodes, 1 040 588 executed commands), **no**
  executed right-grasp step exceeds the rate limit (largest |Δg| 0.18182); 7.3 % of steps sit at
  it. The models have never seen a grasp request that the embodiment clips.
- **The planner's queries.** TASK-089's `CEMPlanner` (as in TASK-090 and TASK-091) samples every
  first iteration from N(0, 1) clipped to [−1, 1], with no candidate projection
  (`project_candidates` is off in TASK-088's `ModelController`). In a first iteration 83 % of
  within-plan grasp steps exceed 2 / 11 and no 8-step candidate is rate-feasible.
- **What the chosen plans looked like** (TASK-091 Stage BS run reports, `plan_log`, read-only; one
  run, simulation only): per world-model seed, **28–45 % of the steps of the chosen plans** lie
  off the training support (|Δg| > 2 / 11 against the previous executed or planned grasp), only
  **11–25 % of chosen plans** are wholly on it, and the grasp of the executed first command was
  clipped by the rate limit on **69–88 % of grasp-phase commands**. Successful episodes had a
  lower off-support share (median 0.14–0.36; the upper end is M-v1-87100's single success) than
  failed ones (0.24–0.44), a correlation that shorter, successful episodes confound.
- **What the models predict there.** On 826 val roots of `play-v1` whose eight executed grasp
  commands contain at least one at-limit step, each model was rolled out (CPU only) under three
  command sequences that **the embodiment executes identically**: the executed one; a *saturated*
  one, with every at-limit step's request set to ±1; and a *random* one, with each at-limit step's
  request drawn uniformly from the requests the clip maps onto it. All three lead to the same
  executed commands and so the same real future. Against that future, at h = 8:

  | Models | Hand-state MSE (standardised, right-hand joints): executed | saturated | random | Latent error: saturated / executed |
  | --- | --- | --- | --- | --- |
  | TASK-087's G-87100, 87101, 87102 | 0.044, 0.050, 0.055 | 0.335, 0.476, 0.408 | 0.152, 0.172, 0.172 | 1.08, 1.08, 1.10 |
  | TASK-090's six G models | 0.042–0.047 | 0.294–0.520 | 0.148–0.223 | 1.06–1.12 |

  For requests the embodiment executes exactly as recorded commands, the models' predicted hand
  state is **about 7–11 times** further from the truth than for the recorded commands. G's pose
  term (λ = 2.0552 × the standardised joint-position MSE) reads exactly these predicted joints.

### 1.3 Reading

The planner asks the model about grasp requests the model never saw, and the model answers wrongly
about the one thing the grasp needs: how fast the hand actually closes. A world model that predicts
the hand closing faster than it can lets CEM score a plan that touches, closes and lifts within a
few steps. In closed loop the hand closes at 2 / 11 per command, so the arm lifts or moves on before
the fingers hold. That fits the records' symptoms: a close that registers no earlier than command
8–9, contact without rise, requests that alternate and never build up, and A1 passing on executed
branches while closed loop fails. **This is a hypothesis.** The development measurements of §1.2
show the mismatch. They do not show that it causes the failures.

## 2. Candidate model changes, ranked

Each candidate must fit the plan's research rules and stay outside the closed scopes:

- TASK-062: training a LeWM-family encoder on `apple-wide-v1`;
- TASK-065: an action-conditioned predictor on frozen pooled CLS latents on `apple-look-v1`;
- TASK-089 (R26.33): planner, cost, horizon or action-parametrisation variants over TASK-087's
  checkpoints;
- TASK-091 (R27.71): more scripted grasp-and-lift play of `play_policy_grasp_v0`'s kind.

| Rank | Change | Evidence for | Evidence against, cost | Scope |
| --- | --- | --- | --- | --- |
| **1 (chosen)** | **Rate-equivalent request augmentation (G-req).** Train TASK-087's G so that its grasp input is, on part of the windows, a request the embodiment's rate clip maps onto the recorded command. The target stays the recorded future. | §1.2: no training step beyond the limit; 28–45 % of chosen plan steps beyond it; hand-state error 7–11× under execution-identical requests. It targets the recorded symptoms (§1.3). | It covers the grasp channel only, not the arm's projection; it may fix the prediction and still not the contact rate (39–87 of 128). Cost: 3 training jobs, about 2 GPU h. | Open: a model change, new checkpoints, same corpus, same latent, planner unchanged. |
| 2 | **Grasp-outcome head and cost term** (TASK-090 §15's model-first design, R27.71's example): predict contact and object rise over the horizon and add a cost on it. | Lift outcome reads from state at AUROC 0.974 (proprio floor, TASK-090 A′, 16 positives). | Thin positives (about 3 % of windows contain a lift state, TASK-090 §13, development). The head reads predicted latents and states, so it inherits the off-support error of §1.2. A learned prediction of simulator truth in the planning cost needs its own ruling against the project rule (TASK-090 §15 reason 3). | Open. Natural next step if G-req installs rate fidelity but the closed loop stays low. |
| 3 | **Hand-crop or higher-resolution latent.** | Latent-only lift-vs-release at chance on real frames at h = 8 (TASK-089: 32 / 64); the frame-plus-state probe (0.803) does not beat the state alone (0.974). | `play-v1` stores only the 112 px onboard frame, so the corpus must be re-rendered and re-featurised (several GPU hours before any training); TASK-075 found no view reading the place offset within τ with frozen DINOv2 (a different phase and object). Weaker link to the symptoms. | Open (TASK-062 and TASK-065 do not cover it). |
| 4 | **Contact- or lift-weighted loss** (up-weighting the 3 % of lift windows). | — | Close in effect to TASK-091's enrichment, which raised the lift-window share 1.6× (C2) with no measurable closed-loop gain. | Open in letter, but weak after R27.70. |

**Not a model change, and not taken:** projecting CEM's candidates through the embodiment
(`CEMConfig.project_candidates`, or a grasp-only rate clip in the planner). It would remove the
mismatch on the planner's side. It is a planner change: over TASK-087's checkpoints R26.33 closes
it "without new evidence of a different kind", and over TASK-090's checkpoints it would confound
TASK-091's frozen comparison. §1.2 may be such new evidence, but taking it would need its own ruling,
and this task holds the planner fixed so that the model change is the only factor (R27.75).
Full candidate projection would also run the embodiment's IK about 300 × 8 × 10 times per command
(not measured; likely far above the 7 200 s job cap at 400 commands). §7 does not close it.

**Why rank 1.** It is the only candidate with a measured, mechanistic link to the recorded
symptoms; it changes one thing (what the model learns about the grasp input), so the unchanged G
is an exactly paired control (same corpus, latent, recipe, budget, model seeds, initialisation and
training windows); it needs no exception to the simulator-truth rule, because the rate limit is the
robot's own declared command limit (`configs/g1_sim_action.json`) and enters training data
construction only; and it is cheap. The drafter judges a gated B-PASS (at the plan's 0.40) unlikely,
about 5–15 %, and a measurable gain over the control (B-PASS or B-GAIN) more likely, about
30–45 %. These are judgements, not measurements.

## 3. Question and hypotheses

**Question.** With TASK-087's corpus, latent, recipe and budget and TASK-089's step-averaged planner
held fixed, does a G trained to predict what the embodiment executes for any grasp request grasp
and lift more often in closed loop than the unchanged G, and does it reach the plan's bar?

- **H1 (offline, mechanism).** G-req predicts the hand state under execution-identical saturated
  requests almost as well as under the recorded commands (R1), and loses no accuracy under the
  recorded commands (R2).
- **H2 (closed loop).** M-req (G-req, three seeds pooled) beats M-v1 (TASK-087's G, the same three
  seeds) on grasp-and-lift on the same resets, and reaches the plan's bar.

## 4. The model change

### 4.1 G-req

- **Architecture, losses and weights:** TASK-087's arm G unchanged (`GroundedWM("G")`, its config,
  the latent, joint-change and inverse-dynamics losses at weight 1). TASK-087's files are not edited,
  so its implementation hash stays as it is.
- **What changes: the grasp channel of the model's action input** (index 13) on a share of the
  training windows. For a window with executed grasps e₁ … e₈ and the executed grasp e₀ of the
  command before the window (same episode):
  - a step k is **at the limit upward** when e_k − e_{k−1} is within 10⁻⁴ of +2 / 11, and
    **downward** when it is within 10⁻⁴ of −2 / 11;
  - with probability **p = 0.5 per window**, every at-limit step's input is replaced by a request
    drawn **uniformly** from the requests the clip maps onto e_k: [e_{k−1} + 2 / 11, 1] upward,
    [−1, e_{k−1} − 2 / 11] downward. Every other step keeps e_k;
  - when the window starts at an episode's first command, e₀ is unknown and step 1 is never
    replaced.
  - The draws use their own generator, `default_rng(SeedSequence([9008, m]))` for model seed m,
    so TASK-087's window sampler (`SeedSequence([8714, m])`) and initialisation
    (`torch.manual_seed(m)`) are untouched.
- **Targets do not change.** The latent targets, the measured states of the joint-change loss and
  the inverse-dynamics loss's target (the **executed** seven played dimensions) stay as recorded.
  Only the predictor's action input is augmented. Because TASK-087's `GroundedWM.loss` uses one
  tensor for both, Stage 0 adds a subclass in a new module that passes the augmented input to
  `step` and `rollout` and the executed actions to the inverse head (§13).
- p = 0.5, the uniform draw and the grasp-only scope are **declared, not tuned**. Half the windows
  stay exactly as recorded, so the on-support accuracy has the same data as the control's.
- **Why this teaches the clip.** Every at-limit transition in the data now pairs many requests with
  one executed outcome, so the model learns that requesting more than 2 / 11 beyond the current
  grasp has the effect of requesting exactly 2 / 11. From an open hand at rest, requests anywhere in
  [−0.818, 1] map to −0.818, the first step of the collector's own close. The model must infer the
  previous grasp target from the measured hand joints (in the 28-D state) and, in a roll-out, from
  its own predicted state, as it already does.

### 4.2 Training

TASK-087's training loop, as TASK-090's driver runs it (§13): `play-v1` train (2 874 episodes,
TASK-087's feature store and projection), batch 64, AdamW at 3 × 10⁻⁴, weight decay 10⁻⁴,
gradient clip 1.0, **U = 60 000 updates**, selection every 3 000 updates on `play-v1` val under
the **executed** commands with TASK-087's rule (`select_checkpoint`, tolerance 0.01),
`last_two_triggered` recorded and not an escalation. The state and palm moments are `play-v1`
train's, which is what TASK-087's driver computes; Stage T checks the buffers bitwise against
TASK-087's G-87100 (TASK-090 §2.2's check). The training store is held in memory (`mmap=False`),
as TASK-090's Stage 0 chose.

**Model seeds 87100, 87101, 87102 → G-req-87100, G-req-87101, G-req-87102.** These are TASK-087's
seeds on purpose: with the same seed, G-req and TASK-087's G start from the same weights and see
the same windows in the same order, and differ only in the augmentation. Stage 0 checks this
(§13). Test, `play-v2g` and `play-v1p` are not read in training.

## 5. Arms and controls

| Arm | Model | Planner | Role |
| --- | --- | --- | --- |
| **M-req-{87100, 87101, 87102}** | G-req | TASK-089's G-chg: `CEMPlanner` H = 8, 300 samples, 10 iterations, 30 elites, min std 0.05, MPC every command, left arm fixed, step-averaged cost, λ = 2.0552 | **primary, pooled** |
| M-v1-{87100, 87101, 87102} | TASK-087's G, unchanged | the same | **control: the unchanged G** (gated comparison) |
| ik | — | TASK-088's IK follower with e9's press | ceiling; its BD rate sets the bar |
| ik-nopress | — | the same without the press | reported |
| hold, random | — | TASK-088's | floors |

- **Why M-v1 is the control, not M-v2g or M-v1p.** M-v1 is the same model trained on the same
  corpus with the same seeds; only the augmentation differs. TASK-090's models were trained on
  twice the data, so comparing against them would add corpus size to the factor.
- **R26.33 and M-v1.** R26.33 closed further planner, cost, horizon and action-parametrisation
  *variants* over TASK-087's checkpoints. Running them unchanged under the planner TASK-089 selected
  is a control re-run, as TASK-090 §3 and TASK-091 did. M-v1 carries no claim of its own; the
  claims are about M-req.
- **Planner, scene and scoring unchanged:** TASK-089's G-chg exactly (the table), TASK-088's scene,
  grasp goals (its demonstrator; four subgoals), subgoal switching, the 400-command budget,
  execution through the embodiment's projection, the 5 rad/s velocity stop and the scorer (target
  ≥ 5 cm above its start height and in grasp contact for 20 consecutive commands; no latching).
  **No planner knob changes, now or after the freeze, whatever any stage shows.**
- **Simulator truth.** Object truth is read by the demonstrator, the root builder, the offline
  probes' labels and the scorer only. It never enters a model input, the augmentation or a
  planning cost.

## 6. Stages and gates

### 6.1 Stage T — training

Three jobs, one `gpu_run.sh` lock each, each under the 7 200 s per-job cap (TASK-087's 31–37 min
per job). Each report records the checkpoint sha256, TASK-087's implementation hash (which must
match), the new module's sha256, the feature-report pin (`f5648a35…7eae`), the moments' sha256,
the augmentation counts (windows and steps replaced) and `last_two_triggered`.
**T-VOID:** a job fails a check, reaches the cap, or is run twice.

### 6.2 Stage R — rate fidelity and no regression (offline, gates the closed loop)

**Roots.** `play-v1`'s **test** split under TASK-087's root rule (start frames 0, 10, 20, … with
start + 16 ≤ T): 5 634 roots in 160 episodes. The test split was featurised and read once, for
TASK-087's Stage E. No model has trained on it and no checkpoint was selected on it. **R-roots**
are the test roots that have a predecessor command in their episode and at least one at-limit grasp
step among their eight executed commands: **821 roots in 154 episodes** (counted from the
executed actions only; no model was run on test for this draft).

**Measures** (h = 8, the planning horizon; h = 1 and 4 reported), for each of the six models of §5:

- **H_x:** Σ over R-roots of the standardised right-hand joint-position MSE (state dimensions
  7–13, predicted against measured at h) under command sequence x;
- **L_x:** Σ over roots of the latent error (TASK-087's MSE over 16 × 192) at h;
- x is *exec* (the executed commands) or *sat* (§1.2's saturated, execution-identical requests);
  *rnd* (§1.2's random one, generator salt 9009) is reported.
- **Ratios of sums with intervals** from TASK-087's episode-cluster bootstrap (10 000 resamples,
  salt 8713, the 97.5 % level).

**Gates, per G-req seed i, paired with TASK-087's G of the same seed:**

- **R1, rate fidelity.** The measured ceiling is 1: the sequences are executed identically, so a
  model that knows the clip predicts the same future for both. The reference is the paired
  control's ratio ρ_ctrl,i = H_sat / H_exec of G-i on the same roots (development on val: about
  7.4–9.5, §11). **Bar: the upper bound of ρ_req,i ≤ 1 + 0.25 × (ρ_ctrl,i − 1)**, that is,
  G-req removes at least three quarters of its control's excess over the ceiling. If ρ_ctrl,i is
  below 2 on the test roots, the premise of §1.2 does not hold there: the row is R-PREMISE
  (§7), before any closed loop.

  **Attainability (Stage 0, reported, val, CPU).** Before the freeze, Stage 0 measures the ratio
  H_sat / H_exec at h = 8 on the 826 val R-roots for each control G-i wrapped with an *inferred
  clip*. At each roll-out step the request is clipped to ± 2 / 11 around the grasp inferred
  (`zero_shot.synergy_grasp`'s least squares on the synergy line) from the measured hand joints at
  step 1, and from G-i's own predicted hand joints afterwards. This measures how well the previous
  target can be recovered from the joints G-req sees. If any seed's upper bound exceeds that seed's
  R1 bar computed on val, the bar is not attainable from the hand state, and the draft escalates
  before the freeze. The wrapper is an offline reference only, never an arm or a planner setting.

  **The reference, measured for this draft** (development, CPU only, `r1_reference.py`, §11):

  | Control | ρ_ctrl on val (97.5 % interval) | R1 bar on val | Inferred-clip ratio (97.5 % interval) | Inferred-clip H_exec (bare control's) |
  | --- | --- | --- | --- | --- |
  | G-87100 | 7.59 (4.56–14.97) | 2.65 | 0.999 (0.997–1.003) | 0.327 (0.044) |
  | G-87101 | 9.45 (5.45–19.23) | 3.11 | 1.001 (0.998–1.004) | 0.350 (0.050) |
  | G-87102 | 7.39 (4.07–16.73) | 2.60 | 0.999 (0.996–1.003) | 0.372 (0.055) |

  The ratio's bar is attainable from the hand state: every upper bound is far below its bar, so
  the bar is not re-derived. But the reference reaches a ratio near 1 by being equally wrong for
  both sequences. Its H_exec is 6.6–7.4 times the bare control's, so it would fail R2 by far. The
  joint angles lag the synergy target: the inferred previous grasp is within 2 / 11 of the
  executed one on 88 % of val frames (median error 0.002, p90 0.235). **So R1 alone can be met
  degenerately, and R1 and R2 are gated together.** No reference shows that both can be met at
  once; only training G-req can. That is a declared risk (§10), not a reason to change the bar.
- **R2, no regression under the recorded commands.** δ is measured on the same roots from the
  three control seeds, as TASK-087 §6.4 measured it on val: the largest, over the six ordered pairs
  (i, j) with i ≠ j, of the upper 97.5 % bound of the ratio X(G-i) / X(G-j), minus 1 (0 if
  negative). It is computed once for
  the latent error on all 5 634 test roots (δ_L) and once for the hand-state MSE on R-roots (δ_H).
  **Bars: the upper bound of L_exec(G-req-i) / L_exec(G-i) ≤ 1 + δ_L, and of
  H_exec(G-req-i) / H_exec(G-i) ≤ 1 + δ_H.** Measured the same way on val for this draft
  (development, `r2_delta_val.py`, §11): δ_L = 0.013 and δ_H = 0.51. So R2's latent margin is
  tight, but its hand-state margin is loose: a G-req whose executed-command hand error is up to
  about 1.5 times its control's passes. Stage 0 repeats the val measurement and records it.
  Together, R1 and R2 bound the saturated error by H_sat(G-req-i) ≤ (1 + 0.25 (ρ_ctrl,i − 1)) ×
  (1 + δ_H) × H_exec(G-i). On val that is about 3.9–4.7 times the control's executed-command
  error, against the control's own 7.4–9.5, so the jointly guaranteed reduction of the saturated
  error's excess is about 55 %, not the 75 % R1 states in ratio terms. δ_H is not tightened: it
  is the measured seed-to-seed spread. The
  test values set the bars. H_sat(G-req-i) / H_exec(G-i), the saturated error against the
  control's executed-command error, is reported beside R1 and R2.
- **E, validity** (TASK-090 §6.1's checks, on the same test roots): (i) roll-out error below
  copy-last at h = 4 and 8 (upper bound of the ratio < 1); (ii) action sensitivity (TASK-087's
  wrong-command / true-command ratio with TASK-087's wrong table, salt 8711) with lower bound > 1
  at h = 4 and 8; (iii) for G-req-87100, roll-out error below TASK-087's N-87100 (upper bound < 1)
  at h = 4 and 8.

**Reported only:** the latent ratios L_sat / L_exec, every measure under *rnd*, h = 1 and 4, and the
same measures for TASK-090's six G models (as §1.2's reference).

**R passes for a seed** when R1, R2 and E hold for it.

### 6.3 Stage A‴ — TASK-091's offline grasp probes as a no-regression gate

TASK-091 §4's Stage A″ exactly (TASK-089 §5.1–§5.5's code through TASK-091's frozen modules), for
the six models of §5, on **64 fresh roots, 908000–908063**: A1 with G's cost at h = 8 and 16
(bars: A-K0's frozen `A1_BARS`, 61 / 64 on each gated comparison at h = 8, and beating the
action-shuffled control, salt 9002), **close≤4@R1 ≥ 48 / 64** and **lift@R2 ≥ 48 / 64** with the
step-averaged cost. close@R1, the readability probe (salt 9004), latent-only wins and the lift
branch's rise are reported. The ceiling on these roots is computed and reported only. **A‴ passes
for a seed** when it meets A1 at h = 8 and both bars. A‴ is a no-regression check: the bars were
set for models without the augmentation, and A″ passed for every G at TASK-091.

Reported on the A‴ roots, for every model: the plan's *executed* grasp (the plan's requests passed
through the rate clip from the root's grasp) at steps 1–8 beside the model's predicted hand
synergy, so a reader can see whether G-req's plans predict a hand that closes at the rate it can.

### 6.4 Stage BD, GO and Stage BS — TASK-091 §5 on fresh resets

The machinery of TASK-090 §7 and TASK-091 §5, with M-req in M-v2g's place and M-v1 in M-v1p's:

- **Stage BD (development, not gated):** 32 fresh resets, **90400–90431**. Arms: ik, ik-nopress,
  hold, random, M-req-87100 and M-v1-87100. **bar = min(0.40, 0.9 × r)**, r ik's BD rate; if
  r < 0.30 the bar is not set and the task escalates. BD can show a defect, whose fix needs its own
  ruling, but changes no knob by itself.
- **GO:** an independent reviewer posts GO (or NO-GO) on BD's record before Stage BS. After a
  NO-GO, BD is recorded again on fresh development seeds named by a ruling.
- **Stage BS (gated): n = 128 fresh resets, 909000–909127**, every arm of §5 (all six
  world-model runs), at the frozen revision from a clean tree. Each world-model run is split into
  two 64-reset jobs (909000–909063, 909064–909127) under the 7 200 s cap, combined before any
  summary. **Determinism re-run:** M-req-87100 on 909000–909003, which must match bitwise.
- **Notation.** k_g is M-req's pooled successes of 384, k_c M-v1's. Pooled bar 3 × ⌈bar × 128⌉ and
  per-seed floor ⌈⌈bar × 128⌉ / 2⌉; at bar 0.40, **156 / 384** and **26 / 128**.
- **Gating tests (reset level),** as TASK-090 §7.2: for each reset r, G_r is M-req's successes
  over its three seeds, C_r M-v1's, R_r random's. "M-req beats M-v1": the exact one-sided sign-flip
  test at p < 0.01 over d_r = G_r − C_r. "M-req beats random": the same over d_r = G_r − 3 R_r
  (approximate, conservative, as there). Reported: the pooled exact McNemar test and the
  reset-cluster bootstrap 95 % interval of (k_g − k_c) / 384 (10 000 resamples, salt 9006).
- **Power.** The declared alternative is per-seed success rates of **0.15 for M-req against 0.05
  for M-v1** (M-v1's recorded rates: TASK-089 0, 8 and 7 of 64; TASK-091 87100 1 / 128 in BS and
  2 / 32 in BD). With TASK-090's simulation (independent per-arm, per-reset N(0, SD) logit
  effects, 2 000 simulations, development; §11), the reset-level test's power at n = 128 is 0.99,
  0.98 and 0.91 at SD 0, 1 and 2 (at n = 64: 0.79, 0.71, 0.56). So n = 128. Stage 0 recomputes it
  with 20 000 simulations and records it; if the power at SD 1 is below 0.80 at n = 128, it is
  disclosed and escalated before the freeze.
- **Rows** (first match):
  1. **B-VOID** — a run did not complete under the rules (including the cap), a check failed (goal,
     checkpoint, implementation hash, module hash, moments, λ, seed), the determinism re-run did
     not match, or a cohort was run twice.
  2. **B-PASS** — k_g ≥ the pooled bar, **and** every M-req seed ≥ the per-seed floor, **and**
     M-req beats random, **and** M-req beats M-v1.
  3. **B-GAIN** — M-req beats M-v1, but k_g is below the pooled bar, a seed is below the floor, or
     M-req does not beat random. The model change helped, but the bar was missed. Escalate; no
     clause.
  4. **B-BAR** — k_g ≥ the pooled bar and M-req beats random, but M-req does not beat M-v1. The bar
     is met, but not because of the change. Escalate; no clause; no Phase 4 by itself.
  5. **B-FAIL** — M-req does not beat M-v1, **and** (k_g is below the pooled bar or M-req does
     not beat random), **and** the bootstrap 95 % upper bound of (k_g − k_c) / 384 is below **δ = +0.10**.
     **The clause fires** (§8).
  6. **B-UNDER** — as B-FAIL, but that upper bound is ≥ δ. Escalate; no clause.
- **Reported with every row:** each seed's count with its exact 95 % interval; ik against M-req;
  per-object results; grasp contact, maximum rise, subgoals reached or timed out; executed grasp
  commands per phase; stopped episodes and planning seconds; TASK-091 §5.3's three deferral
  measures per seed; **the share of grasp-phase first commands whose grasp the rate clip changed,
  and the share of chosen plan steps off the training support** (§1.2's reading, from the runs'
  `plan_log`); Stage R's and A‴'s results beside each seed; two MP4 replays (one success if any,
  one failure).
- **"LeWM needed" is not claimed by any row.** The IK follower uses the same goal information and
  no world model.
- **B-PASS admits a Phase 4 preregistration** (pick-and-place with subgoals). It does not
  preregister it.

## 7. Rows of the task, in order

| Row | When | Next step |
| --- | --- | --- |
| T-VOID | §6.1 | fix under its own ruling; re-run on the same model seeds only if the job did not complete. A completed job that failed a check is not replaced by a fresh model seed without a ruling that also says how M-v1's pairing (§4.2) is kept, since fresh seeds break it |
| R-VOID | a Stage R check fails (checkpoint, hashes, roots, the equivalence check of §13) | fix under its own ruling |
| R-PREMISE | any control seed's ρ_ctrl,i < 2 on the test roots | escalate; no clause: §1.2's mismatch does not hold on held-out roots |
| E-FAIL | any G-req seed fails E | escalate; no clause |
| R-FAIL | two or more G-req seeds fail R1 or R2 (A‴-FAIL then also counts a seed that failed R together with one that fails A‴) | escalate; no clause: the augmentation did not install rate fidelity without regression, so H2 was not tested |
| A‴-VOID | as TASK-091's A″-VOID, on A‴ | fix under its own ruling; fresh roots |
| A‴-FAIL | two or more G-req seeds fail R1, R2 or A‴, counted together (a seed failing any of them counts once) | escalate; no clause |
| B-VOID … B-UNDER | §6.4 | as §6.4 |

- A G-req seed that fails R or A‴ alone stays in M-req's pooled count; its results are reported
  beside its closed-loop count.
- After any VOID the stage runs again only on fresh seeds named by its ruling. A voided cohort is
  never re-run.
- Escalations go to a ruling in `docs/DECISIONS.md`. Each closes the task with a recommendation
  unless the ruling says otherwise.

## 8. Abandonment clause

**Fires on B-FAIL only.**

**It closes:** rate-equivalent request augmentation of the right-grasp channel as the grasp fix:
training a grounded LeWM to predict the recorded outcome for grasp requests that the embodiment's
rate clip maps onto the recorded command, under TASK-087's recipe, latent and corpus and TASK-089's
step-averaged planner. After it fires, no further variant of it (augmentation share, request
distribution, which steps are augmented, budget) is preregistered for grasping without new
evidence of a different kind.

**Not closed:**

- LeWM, DINOv2 and CEM;
- the other model changes of §2 (a grasp-outcome head and cost, a hand-crop or higher-resolution
  latent, a weighted loss) and any other model change;
- an arm-channel feasibility model (the arm's projection is not tested here);
- planner-side candidate projection: still outside this task and still under R26.33 for
  TASK-087's checkpoints; it needs its own ruling;
- the corpora (`play-v1`, play-v2g, play-v1p) as bases, and the kinds of data R27.71 leaves open;
- reach planning (TASK-088's Z3-REACH stands); Phase 4 as a goal; the product goal.

**It does not fire on** T-VOID, R-VOID, R-PREMISE, E-FAIL, R-FAIL, A‴-VOID, A‴-FAIL, B-VOID,
B-GAIN, B-BAR or B-UNDER. M-v1's counts are recorded with the clause, not as a way around R26.33.

## 9. Seeds and salts

| Use | Seeds |
| --- | --- |
| Model seeds | **87100, 87101, 87102** (TASK-087's, reused on purpose for pairing, §4.2); checkpoints `checkpoints/task092-run/G-req-<seed>` |
| Stage 0 pairing check (§13) | 87900 (TASK-087's debug range 87900–87999; never a result) |
| Stage R roots | `play-v1` test under TASK-087's root rule (no seed) |
| Stage A‴ roots | **908000–908063** |
| Stage BD resets | **90400–90431** |
| Stage BS resets | **909000–909127** (two jobs: 909000–909063, 909064–909127) |
| Determinism re-run | 909000–909003 (inside BS) |
| Debug (Stage 0 plumbing and smoke; never counted) | **90500–90599** |

**Salts.** New: **9008**, the augmentation draws (`SeedSequence([9008, m])`); **9009**, Stage R's
random-equivalent requests. Reused: TASK-087's 8711 (wrong table) and 8713 (episode bootstrap);
TASK-088's 8801 and 8802 (layouts and goals), 8803 (CEM) and 8804 (random), with the new seeds;
TASK-090's 9002 (A‴'s shuffled deal), 9004 (the readability probe) and 9006 (the reset-cluster
bootstrap). A root's or reset's rejected draws are further attempts under the same seed.

**Checked (DRAFT, 2026-10-14).**

- **Remote branches.** On all 109 remote-tracking refs (`git for-each-ref refs/remotes` after
  `git fetch`, 2026-10-14; the count includes stale tracking refs, and `git ls-remote --heads origin`
  lists fewer), over `src/`, `scripts/`, `docs/`, `.mc/`,
  `tests/`, `benchmarks/` and `configs/`, `git grep` finds no token in 908000–909999,
  90400–90599, 9008 or 9009. 9007 is D-A2's salt (TASK-090 §20), so it is skipped. The nearest
  ranges in use are TASK-091's 906000–906063 and 907000–907127, TASK-090's 903000–905063 and
  90200–90231, TASK-091's 90300–90331 and 90700–90799, and TASK-090's play-v1p and play-v2g episode
  seeds 910000–913199 and 920000–923199.
- **Evidence.** In `~/develop/emai/evidence/` (`*.json`, `*.txt`, `*.log`, `*.md`, `*.csv`,
  `*.yaml`, `*.py`, `*.sh`), the tokens in these ranges are decimal fractions (for example
  `11.90409`), frame offsets in TASK-087's episode tables (for example `908352`) and a hexadecimal
  fragment. None is a seed (no `"seed"`, `"reset"` or `"root"` field holds one).
- **Stage 0** repeats both checks at its own revision and records them.

## 10. What this cannot show, and its risks

- **Scope.** One scene, arm, hand and camera at 112 px, simulation only; one encoder, latent,
  corpus, recipe, budget and planner; goals from a scripted demonstrator (biased to layouts a
  scripted pick can do; almost no banana); seeds pooled.
- **The mechanism is a hypothesis.** §1.2 shows a mismatch between the planner's queries and the
  model's training support, and wrong hand predictions there. It does not show that the mismatch
  causes the closed-loop failures. A B-GAIN or B-PASS would be consistent with it. A B-FAIL with R
  passed would say that fixing the grasp prediction alone does not move the closed loop.
- **Grasp channel only.** The arm channels are also executed through a projection (backtracking
  of the arm delta and the joint-rate limit), and the records cannot show how often the planner's
  arm requests were reduced (`plan_log` keeps the grasp only). The contact gap (39–87 against 122)
  may come from the arm, the latent's resolution or the cost, which this task does not change.
- **The model must infer the previous grasp target.** The 28-D state holds the hand joints, which
  track the synergy target with a lag; the target itself is not an input. On val, the grasp
  inferred from the measured joints is within 2 / 11 of the executed one on 88 % of frames (§6.2,
  development). If G-req cannot infer it, R1 or R2 fails (R-FAIL, no clause).
- **R2's hand-state margin is loose.** On val δ_H = 0.51 (development), so R2 tolerates a hand error
  up to about 1.5 times the control's under recorded commands.
- **A flat cost over equivalent requests.** A rate-aware model gives the same prediction for every
  request beyond the limit, so CEM's choice among them is arbitrary. That changes nothing executed,
  but the off-support share of §6.4 is then not a measure of success; it is reported only.
- **R's bar.** "Removes three quarters of the control's excess" is a declared fraction, calibrated
  against a measured ceiling (1) and the measured control on the same roots, not against
  closed-loop success. Offline plan measures did not predict closed loop before (TASK-089: A1 and
  A2 passed, then 0 / 64).
- **Undertraining.** U is TASK-087's capped budget (its val curves were still falling slowly). Both
  M-req and M-v1 share it.
- **What a pass shows.** A B-PASS would be the first gated world-model grasp-and-lift at the plan's
  bar in this project, on these terms only. It would not show "LeWM needed".

## 11. Disclosed development before this draft

All of it is development, **not gated**, at `64dad40`, CPU only (no GPU job, simulation, rendering
or training was run for this draft). Scripts and outputs are in
`~/develop/emai/evidence/task092-prototype/` (`SHA256SUMS` sha256 `fd972ef4…910c`). The last four
scripts were added after the independent review of #214, at `32ff61c`.

- `grasp_steps.py`: the executed grasp steps of `play-v1` train (TASK-087's feature store) and a
  first CEM iteration's candidates (§1.2).
- `bs_offsupport.py`: TASK-091 Stage BS's run reports, read-only (§1.2).
- `rate_equiv.py`: nine G checkpoints (TASK-087's three, TASK-090's six) on 826 **val** roots
  (150 episodes) of `play-v1` (§1.2). The table in §1.2 is its output. Val, not test, was used so
  that Stage R's test roots stay unseen by any model run.
- `power.out`: §6.4's power figures (2 000 simulations).
- `count_r_roots.py`: Stage R's root and R-root counts on `play-v1` test, from `actions_test.npy`
  and `episodes_test.json` only (no latent, state or model read) (§6.2).
- `power.py`: the command that produced `power.out` (re-run, output identical).
- `r1_reference.py`: R1's inferred-clip attainability reference on the 826 val R-roots (§6.2).
- `r2_delta_val.py`: R2's δ_L and δ_H measured on val (§6.2).

These numbers chose the candidate, R1's form and the declared alternative. R1's bar is not read
off them: it is a declared fraction of the paired control's excess, measured again on the test
roots at Stage R.

## 12. Runtime, disk, paths and evidence

**GPU.** Every GPU use is one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` job, with the
shared lock taken per job and the 7 200 s per-job cap. Before each job the `## Requests` section of
`~/.local/state/gpu/BOARD.md` is checked; a `- v2d: …` line means the job waits (the 2026-10-12
agreement). The V2D session is told in advance about Stage T, BD and BS.

| Step | Estimate | Basis |
| --- | --- | --- |
| Stage 0 (pairing check, debug smoke, timing) | about 0.5 GPU h | TASK-091 Stage 0 |
| Stage T (3 jobs) | about 1.9 GPU h | TASK-087: 31–37 min per job |
| Stage R (6 models, test roots) | under 0.2 h | TASK-090 Stage E: under 1 min for 11 models; R adds two roll-out sets |
| Stage A‴ (roots, 6 models) | under 0.5 h | TASK-090 A′ 0.12 h for 11 models; D-A2 0.26 h |
| Stage BD (2 world-model runs × 32 plus baselines) | about 1.2 h | TASK-091 BD: 1 674–2 079 s per run |
| Stage BS (6 world-model runs × 128 as 12 jobs, baselines, determinism) | about 12.8 GPU h | TASK-091 BS: 3 184–4 218 s per 64-reset job |
| **Total** | **about 17 GPU h**; about 4–6 days wall-clock with reviews | |

**Disk.** No new corpus or feature store; three checkpoints (small), goals, runs and MP4s go to
`checkpoints/task092-run/` and `outputs/task092-*` on the data disk through the repository's links.

**Paths that must stay.** TASK-087's feature store and projection in
`worktrees/task087-run/outputs/`, and `checkpoints/task087-run/`. **That worktree must not be
archived or removed until TASK-092 closes.**

**Evidence.** `~/develop/emai/evidence/task092-*/` with `SHA256SUMS`. Existing evidence under
`data/`, `checkpoints/`, `outputs/` and `~/develop/emai/evidence/task08*`/`task09[01]-*` is read and
never written.

## 13. Stage 0, freeze and what Stage 0 may change

**Stage 0** is its own PR, merged before the freeze. It adds only opt-in modules and new scripts
and changes no existing file, so TASK-087's implementation hash and TASK-088–TASK-091's frozen code
stay as they are:

- **A new module** (NumPy at import), for example `grasp_rate_aware.py`: the at-limit rule, the
  equivalence sets and the augmentation draw, Stage R's root filter, measures, δ and bars, the R
  and A‴ rows, the cohorts and seeds, and the B rows (reusing TASK-090's sign-flip test, bootstrap
  and power code, with §6.4's names).
- **A torch module**, for example `grasp_rate_aware_runtime.py`: a `GroundedWM` subclass whose loss
  passes the augmented input to the predictor and the executed actions to the inverse head; the
  training driver over TASK-087's store; a guarded loader for G-req checkpoints (TASK-087's
  implementation hash, the new module's hash, the feature report, the moments); Stage R's
  roll-outs; and the closed loop through TASK-091's frozen runtime unchanged.
- **Scripts** `scripts/{train,run,summarize}_task092.py` and `scripts/run_task092_stage.sh`.
- **Tests:** the at-limit rule and the equivalence sets (the clip maps every drawn request back to
  the recorded command); the generator separation (with p = 0 the training batches and their
  order equal TASK-087's); the subclass's loss equals TASK-087's when the input equals the
  executed actions; Stage R's ratios, δ and bars on synthetic data; the rows; the seed ranges and
  their disjointness; the core-import check.

**Stage 0 also runs:**

- **the pairing check:** TASK-087's driver and the new driver with p = 0, each for 300 updates on
  debug seed 87900, must give bitwise-identical weights; and with p = 0.5 the first batch differs
  only in index 13 of at-limit steps;
- **the pairing check against the historical run:** the new driver with p = 0 on model seed 87100
  up to the first selection point (3 000 updates). Its val criterion there must equal the value
  recorded in TASK-087's G-87100 report. This ties the pairing to the actual G-87100, not only to
  the debug seed (a few GPU minutes);
- the inferred-clip attainability reference of §6.2 R1, R2's δ on val, and a grasp-blind
  reference (each control G-i with index 13 held at the step-1 inferred grasp for both sequences;
  its R1 ratio is 1 by construction), reporting each reference's H_exec / H_exec(G-i) against
  1 + δ_H, so the record shows that R2 excludes both degenerate ways of meeting R1. (Review of
  #214, development, val: 15.1–19.0, lower bounds 8.3–11.3, against 1.51; script archived in
  `~/develop/emai/evidence/task092-review214/`.)
- §9's seed checks and the hash checks (TASK-087's checkpoints, feature report and projection,
  N-87100);
- the power simulation of §6.4 with 20 000 simulations;
- a debug smoke on 90500–90599 of a short training job, Stage R, A‴ and every closed-loop arm,
  timed per command.

**Stage 0 may change only:** the per-run process count; the BS job split (more, smaller jobs) if a
64-reset job projects above 5 400 s; and n if the power rule of §6.4 requires escalation. Each change
is recorded. The protocol is frozen at the Stage 0 merge, and every later stage runs at the merged
revision from a clean tree.

## 14. Rulings this document records (R27.74–R27.88; DRAFT; decided by Claude under owner delegation)

- **R27.74** — The next grasp step after R27.71 is **TASK-092**, a model change: rate-equivalent
  request augmentation of the grasp channel (G-req, §4). Rationale: §1.2's development
  measurements (not gated) show the planner queries the models off their grasp-rate training
  support, where their hand-state predictions are about 7–11 times worse; this is the only ranked
  candidate with a measured link to the recorded symptoms, and it is exactly paired with the
  unchanged G. The ranking of §2 is recorded.
- **R27.75** — **The planner stays frozen**: TASK-089's step-averaged CEM, λ = 2.0552, no candidate
  projection. Planner-side projection is not taken and not closed (§2, §8).
- **R27.76** — **Training** (§4.2): TASK-087's recipe, corpus, budget and moments, model seeds
  87100–87102 reused for pairing, augmentation p = 0.5 with salt 9008, targets unchanged.
- **R27.77** — **Arms** (§5): M-req ×3 primary, pooled; M-v1 (TASK-087's G ×3, unchanged) the
  gated control; ik, ik-nopress, hold, random. M-v1 is a control re-run, not a variant under R26.33.
- **R27.78** — **Stage R** (§6.2) on `play-v1` test: R1 (rate fidelity, bar 1 + 0.25 × (ρ_ctrl − 1)
  per paired seed, from the measured ceiling 1 and the measured control), R2 (no regression within
  δ measured from the control seeds) and E (validity). R-PREMISE if a control's ratio is below 2.
- **R27.79** — **Stage A‴** (§6.3): TASK-091's A″ on fresh roots 908000–908063 as a no-regression
  gate.
- **R27.80** — **Closed loop** (§6.4): BD 90400–90431, an independent GO, BS n = 128 on
  909000–909127, the bar rule min(0.40, 0.9 r), the pooled bar and per-seed floor, the reset-level
  sign-flip tests and rows B-VOID, B-PASS, B-GAIN, B-BAR, B-FAIL (δ = +0.10), B-UNDER.
- **R27.81** — **Power** (§6.4): declared alternative 0.15 against 0.05; n = 128 (power 0.98 at
  SD 1, development); Stage 0 recomputes it.
- **R27.82** — **Rows** (§7) and their order. A‴-FAIL counts G-req seeds failing R1, R2 or A‴
  together, so the closed loop never runs with fewer than two clean seeds. A T-VOID on a completed
  job is not replaced by a fresh model seed without a ruling on pairing.
- **R27.83** — **The abandonment clause** (§8): fires on B-FAIL only, with the scope stated there.
- **R27.84** — **Seeds and salts** (§9), checked on 109 remote-tracking refs and in the evidence; new
  salts 9008 and 9009.
- **R27.85** — **Budget** (§12): about 17 GPU h, every job under the 7 200 s cap and the shared
  lock; `worktrees/task087-run` stays until TASK-092 closes.
- **R27.86** — **Disclosed development** (§11) is recorded, with its scripts and outputs in
  `~/develop/emai/evidence/task092-prototype/`.
- **R27.87** — **Status.** DRAFT in this PR; a Stage 0 PR freezes it at its merge; gated stages
  only after that, and BS only after an independent GO on BD's record.
- **R27.88** — **R7 does not change**, whatever row TASK-092 ends in. TASK-092 is not an
  Apple→Plate experiment.

## 15. Stage 0 record (R27.89–R27.94; decided by Claude under owner delegation)

Everything below is Stage 0: checks, references on **val** and a debug smoke on debug seeds. No
gated stage ran, no model read `play-v1` test, and nothing here is evidence of any outcome.
Evidence: `~/develop/emai/evidence/task092-stage0/` (`SHA256SUMS` sha256 `2b517483…0a86`). Code at `bc95576`, with two
fixes (`67837c8`: a reported train-split reading; `c1ccf47`: Stage R rolls N-87100 out with
TASK-087's `predict_at`, since N predicts no state) and the CLAUDE.md entry (`66595ab`).

### 15.1 Code

- `src/embodied_jepa/grasp_rate_aware.py` (NumPy at import): the at-limit rule (|Δg| within 10⁻⁴
  of 2 / 11, the half-width checked against `configs/g1_sim_action.json`), the equivalence sets,
  the augmenter (`SeedSequence([9008, m])`; per batch one uniform per window, then one per step,
  whatever p is), Stage R's root filter, saturated and random requests (salt 9009), measures, δ,
  bars and rows, the A‴ and task rows, the B rows of §6.4, the reported rate measures, the power
  recompute, the cohorts and seeds, and sha256 pins of every earlier file TASK-092 imports
  (TASK-087's model, data and driver code, TASK-090's frozen files, TASK-091's modules and
  scripts). Every runner refuses to start when one of them differs.
- `src/embodied_jepa/grasp_rate_aware_runtime.py` (torch): `GReqWM`, TASK-087's `GroundedWM`
  with the loss split (augmented input to `step` and `rollout`, executed actions to the inverse
  head; otherwise TASK-087's loss line for line); the guarded G-req loader (T-JOB-DONE, kind
  `job`, checkpoint hash, TASK-087's implementation hash, the TASK-092 module hashes recorded at
  training, p and salt, the feature report, the pinned moments); Stage R's roll-outs; the two
  references.
- `scripts/train_task092.py` (Stage T; `--pairing` and `--debug` kinds), `scripts/run_task092.py`
  (`verify`, `seeds`, `power`, `pairing-batch`, `references`, `stage-r`, `roots`, `a-third`,
  `goals`, `run`, `combine`, `video`), `scripts/summarize_task092.py` (stages t, r, a, bd, bs,
  timing) and `scripts/run_task092_stage.sh` (stages t, r, a, bd, bs).
- `tests/test_grasp_rate_aware.py`, 24 tests: the at-limit rule and the equivalence sets; p = 0
  bitwise and p = 0.5 changing only index 13 of at-limit steps; the augmenter's stream and salt;
  the sampler unchanged by the augmenter; `GReqWM`'s parameters, loss and gradients equal TASK-087's
  bitwise when the input equals the executed actions, and its predictor and inverse target read
  the right tensors; Stage R's equivalence checks, δ, R1 and R2 (including the degenerate R1
  that R2 excludes); the R, A‴ and B rows; the rate measures; the power rule; the seed ranges; the
  frozen-file pins; the training driver's refusals; the core-import check. The whole suite: 2 420
  passed, 33 skipped (graphics opt-in).
- No existing file changes except CLAUDE.md's entry points and this record.

### 15.2 Checks

| Check | Result |
| --- | --- |
| **Pairing, debug seed 87900, 300 updates** | TASK-087's driver and TASK-092's driver with p = 0 give **bitwise-identical weights** (state sha256 `2d20d913…31c4` both) and the same val criterion (0.7744575799350353) |
| **First batch, seed 87900** | p = 0: bitwise the executed actions. p = 0.5: 30 of 52 at-limit steps changed (31 of 64 windows drawn), nothing outside index 13, no step that is not at the limit; the clip maps every drawn request back within 6.5 × 10⁻⁸ |
| **Pairing against the historical run, seed 87100, p = 0, 3 000 updates** | val criterion **0.5292273672193022**, equal bitwise to G-87100's recorded value at update 3 000 (report sha256 `4bc6a14c…4641`) |
| **Hashes** (`verify`) | TASK-087's G-87100–87102 and N-87100 against their pins, reports and evidence copies; the train/val and test feature reports (`f5648a35…7eae`, test `a6c380ff…d650`) and every store file; the projection; the pinned moments; TASK-087's implementation hash; TASK-090's six G checkpoints and three feature reports; λ = 2.0552; the rate limit; every frozen file: **all pass** |
| **Seeds** (`seeds`, 110 remote-tracking refs and HEAD) | no token in 908000–909999, 90400–90599, 9008 or 9009 outside TASK-092's own files. In the evidence (22 tokens): TASK-077's training-update counts 90400 and 90500, a byte count (90536), an `arm_out` count (90406) and frame offsets in TASK-087's episode table (90585, 908352–909982); none is a seed |
| **Train split** (reported) | largest executed |Δg| 0.181818 (the limit); 7.26 % of steps at it, as §1.2 |

### 15.3 References on val (§6.2, §13; GPU, at `c1ccf47`, clean tree)

On the 826 val R-roots (150 episodes; 5 643 val roots), with the execution-identical requests
checked (the clip maps saturated and random requests back onto the executed grasps):

| Control | ρ_ctrl (97.5 %) | R1 bar on val | Inferred clip: ratio (97.5 %); H_exec / bare H_exec (97.5 %) | Grasp-blind: ratio; H_exec / bare H_exec (97.5 %) |
| --- | --- | --- | --- | --- |
| G-87100 | 7.590 (4.564–14.969) | 2.648 | 0.999 (0.997–1.003); 7.41 (4.81–14.15) | 1.000; 19.00 (11.34–38.36) |
| G-87101 | 9.447 (5.453–19.230) | 3.112 | 1.001 (0.998–1.004); 6.95 (4.23–13.72) | 1.000; 16.56 (9.51–34.11) |
| G-87102 | 7.386 (4.069–16.731) | 2.597 | 0.999 (0.996–1.003); 6.73 (3.93–15.14) | 1.000; 15.07 (8.31–35.08) |

- **R2's δ on val:** δ_L = **0.0133**, δ_H = **0.513** (the DRAFT's values, reproduced).
- Both references meet R1's ratio bar on val (every upper bound ≤ its bar), and **R2 excludes
  both**: every lower bound of H_exec / bare H_exec (3.93–11.34) is above 1 + δ_H = 1.513. So R2
  rules out both degenerate ways of meeting R1, as §13 asks the record to show.
- The grasp inferred from the measured joints is within 2 / 11 of the executed one on 88.2 % of
  val frames (median error 0.0023, p90 0.235).
- The ratios equal the DRAFT's CPU measurements (§6.2) to three decimals. The R1 bars on test are
  set by Stage R from the test roots.

### 15.4 Power (§6.4, 20 000 simulations)

At the declared alternative (0.15 against 0.05 per seed), the reset-level sign-flip test's power
is **0.988, 0.976 and 0.915** at n = 128 and SD 0, 1 and 2 (n = 64: 0.789, 0.707, 0.560). The
power at SD 1 is above 0.80, so **n = 128 stands**; no escalation.

### 15.5 Debug smoke (debug seeds only; never counted; not evidence)

- **Training.** A G-req job on debug model seed 87901 (TASK-087's debug range), p = 0.5, 3 000
  updates: 0.0366 s per update (median; 115 s), the in-memory store loaded in 0.6 s from the page
  cache. Augmentation: 96 037 of 192 000 windows selected, **14 738 (7.7 %) changed**, 54 230 of
  109 198 at-limit steps replaced (3.5 % of all steps). Only windows with an at-limit step change,
  as §4.1 declares.
- **Stage R on val** with that debug model in all three G-req slots (13 s; test not read): the
  plumbing ran end to end, with the equivalence checks passing and TASK-090's six models reported.
  Its row, E-FAIL (E (iii) at h = 4 against N-87100, upper bound 1.114), is that of a 3 000-update
  debug model, not a result; so are its ratio of 1.038 and its R2 failure. The first attempt
  stopped at N-87100's roll-out (N predicts no state); fixed in `c1ccf47`, re-run.
- **A‴** on 8 debug roots (90500–90507, 9 rejected draws) for all six models: every A2-seed plan's
  first command equalled `first_planned`'s bitwise; the executed plan grasp and predicted hand
  synergy were recorded.
- **Closed loop** on 6 debug resets (90540–90545): ik 6 / 6, ik-nopress 5 / 6, hold 0 / 6, random
  0 / 6, M-v1-87100 1 / 6, the debug M-req 1 / 6 (run as two 3-reset jobs, combined). The
  determinism re-run of the debug M-req on the first four resets matched the combined run
  **bitwise** (applied commands and outcomes), across a different job split and process count.
  The deferral and rate measures were recorded on every world-model episode.
- **Timing.** At 6 processes M-v1-87100 took **1.088 s per command per process**, so a 64-reset
  BS job projects to at most **4 786 s** at the full 400-command budget, below the 5 400 s
  trigger (TASK-091's smoke: 1.08–1.09). A Stage T job projects to about 60 000 × 0.0366 s plus
  20 val selections, about 37–39 min (TASK-087's G jobs took 2 222–2 229 s).
- **GPU use:** 19 jobs, **0.39 GPU h**, all under `gpu_run.sh --wait --board` after a BOARD.md
  check (no `- v2d:` request was present). Two jobs stopped at once and were re-run: `references`
  refused a dirty tree (a CLAUDE.md edit, committed) and the Stage R crash above.

### 15.6 Projected budget

Stage T about 1.9 GPU h; Stage R under 0.1; Stage A‴ about 0.2; Stage BD about 1.2; Stage BS
about 11–14 at TASK-091's measured 3 184–4 218 s per 64-reset job, at most about 16 at the full
budget (12 jobs × 4 786 s), plus baselines and the determinism re-run. Total about 15–19 GPU h.

### 15.7 Rulings (R27.89–R27.94)

- **R27.89 — Stage 0 is recorded** (§15.1–§15.5): every check of §13 passes; the pairing is
  bitwise on debug seed 87900 and against G-87100's recorded val criterion at update 3 000.
- **R27.90 — Stage 0 changes none of §13's three items.** Process counts stay 8 for baselines, 6
  for world-model runs, 4 for the determinism re-run. The BS split stays two 64-reset jobs
  (projection 4 786 s ≤ 5 400 s). n stays 128 (power 0.976 at SD 1).
- **R27.91 — readings of points the DRAFT left open:** the debug split 90500–90539 (roots) and
  90540–90599 (resets); the pairing's state comparison by TASK-087's `state_sha256`; the per-batch
  draw order of the augmenter; ρ_ctrl's point value sets R1's bar; R-PREMISE reads ρ_ctrl's point
  value; L_sat / L_exec is reported on R-roots; Stage R's §6.2 root counts (5 634 roots, 160
  episodes, 821 R-roots, 154 episodes) are checked and a mismatch is R-VOID; the plan's executed
  grasp in A‴ starts from the demonstrator's grasp before each root (−1 at R1, +1 at R2); the
  rate measures take the settled open hand (−1) as the first previous grasp.
- **R27.92 — the debug smoke is not evidence** and changes nothing in §2–§12.
- **R27.93 — frozen at the Stage 0 PR's merge.** Stage T, R, A‴, BD and BS run at the merged
  revision from a clean tree with `scripts/run_task092_stage.sh`; BS only after an independent GO
  on BD's record. The V2D session is told before Stage T, BD and BS.
- **R27.94 — R7 does not change.**
