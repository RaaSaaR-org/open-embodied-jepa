# TASK-087 — Phase 2: robot-grounded LeWM on `play-v1`, offline

**Status: FROZEN when the Stage 0 PR merges** (R24.1–R24.19, decided by Claude under owner
delegation); after that merge nothing in §2–§10 changes without its own reviewed ruling.
This is Phase 2 of the proposal [JEPA_ZERO_SHOT_PLAN.md](../JEPA_ZERO_SHOT_PLAN.md)
("Robot-grounded LeWM, offline"), on the play corpus `play-v1` (TASK-085, P1-PASS, R23.24). The
plan's second data arm (sim play + open real G1 / Dex3 data) was dropped at TASK-086
(R-DROP-GRASP, R23.28), so this task trains on simulated play only.

**This is an offline world-model comparison, not a control result and not an Apple→Plate
experiment.** Nothing here runs a planner or a closed loop. R7's canonical status sentence does
not change, whatever row this task ends in.

## 1. Question

On frozen DINOv2 patch tokens of `play-v1`, does grounding a LeWM token predictor in the robot's
own state (proprioceptive input, a joint-change head and an inverse-dynamics loss) make its
predictions **more action-sensitive** and its **short-step ranking of command sequences** better
than plain LeWM, on every training seed? This is the plan's Phase 2 gate. The plain-LeWM baseline
of TASK-084 part B (R22.13; another corpus) was weak at short horizons: wrong / true commands
1.04 at one step, and the executed commands ranked first among 16 on 14–19 % of roots at one
step and 40–44 % at eight.

## 2. Data

- **Corpus:** `data/play-v1/` as sealed by TASK-085 (32 shards; `corpus.json` and every shard's
  manifest hash checked against TASK-085's evidence). Splits as frozen: train 2 874, val 160,
  test 160 episodes. **Train** fits everything (features' normalisation and projection, state
  and label moments, the models). **Val** selects checkpoints and sets δ (§6.4). **Test** is not
  read, decoded or featurised until Stage F′ (§12), which starts only after the val report is
  written and its sha256 recorded.
- **Frames:** `onboard_rgb`, 112 × 112, 20 Hz; T stored commands, T + 1 frames per episode.
- **Actions:** the stored executed `ee_delta_grasp_v0` actions (14-D, normalised to [−1, 1]). The
  left arm's six dimensions (constant 0) and left grasp (constant −1) stay in the input; models
  see all 14.
- **Robot state (model input for S, G, C):** the right arm's 7 and the right Dex3's 7 joint
  positions and their 14 velocities from `observation.state` (`g1_dex3_proprio_v0`), 28-D,
  standardised by train moments with a floor of 0.01 on the std and clipped at ±5. The other 58
  dimensions are near-constant in this corpus (fixed pelvis, left arm held; development max std
  of the other positions 0.006 rad, §11) and are not used.
- **Labels (never model inputs):** the right palm position in the pelvis frame
  (`palms[:, 0:3]` of the TASK-085 sidecar, verified by its sha256) for arm C's readout head,
  standardised by train moments.

## 3. The latent: frozen DINOv2 tokens, pooled and projected

- **Encoder:** the pinned frozen DINOv2 ViT-S/14 (`pretrained_encoder.load_pretrained`, TASK-063's
  input handling: bicubic resize to 224, ImageNet normalisation), final patch tokens (16 × 16 ×
  384), average-pooled to a **4 × 4 grid** (`frozen_tokens.pool_tokens`, the TASK-066 grid).
  Featurisation runs on the GPU in float32 with eager attention and a fixed batch of 64 (the
  development check in §11 agrees with the CPU path to 2.8 × 10⁻⁵, against a mean |value| of 1.37).
- **Standardisation:** per dimension of the 4 × 4 × 384 vector, with mean and std (floor 10⁻⁶)
  from the **fit sample**: the train episodes whose seed is divisible by 4 (about 720 episodes,
  about 260 000 frames).
- **Projection:** principal components of the standardised 384-channel tokens, pooled over the
  16 grid positions, fitted on the same fit sample; the first **k = 192** components are kept,
  without whitening. Each frame's latent is `[16, 192]`, stored as float16 (about 6.1 kB a frame,
  about 7 GB for train and val). On the development sample k = 192 kept 95 % of the standardised
  variance (§11); the measured fraction on the fit sample, and the fraction of frame-to-frame
  change it keeps, are reported.
- Why a projection: the unprojected 4 × 4 latent would take about 14 GB in float16 on a disk with
  24 GB free and a 10 GB floor (§12). The projection is fixed and linear, fitted on train only,
  and shared by every arm, so every arm predicts the **same target space**; prediction errors are
  therefore directly comparable across arms here (one encoder, one projection), unlike raw latent
  errors across different encoders.
- The distance used everywhere is the mean squared difference over the 16 × 192 latent values
  ("latent error").

## 4. Arms

One predictor architecture for every trained arm: the pinned upstream LeWM `ARPredictor`
(AdaLN-zero conditioned transformer) with the token grid as its sequence axis, bidirectional
attention within the frame and the action embedding (upstream `Embedder`) conditioning every
token, then upstream `pred_proj` (a BatchNorm MLP) — TASK-066's token-predictor composition,
re-implemented in a new opt-in module so no existing model file changes. Width 192 (the latent
width), depth 4, 4 heads of 48, MLP width 768, `pred_proj` hidden width 768.

| Arm | Inputs | Losses | Role |
| --- | --- | --- | --- |
| **P** — plain LeWM | latent, action | latent loss (§5) | the baseline |
| **S** — + robot state | latent + one **state token** (an MLP embedding of the 28-D state) as a 17th token, action | latent loss | grounded, **eligible** |
| **G** — + state, joint-change and inverse-dynamics losses | as S | latent loss + joint-change loss + inverse-dynamics loss | grounded, **eligible** |
| **C** — control: state fusion + readout heads | latent with the state embedding added to every token, action | latent loss + palm-position readout loss | control (the apple WM v2–v4 recipe, adapted), not eligible |
| **I** — inverse dynamics only | as P | latent loss + inverse-dynamics loss | decomposition, reported only |
| **N** — no action | latent, actions replaced by 0 | latent loss | floor (trained) |
| **copy-last** | — | — | floor (the start latent at every horizon) |

The two training passes of §5 are a **teacher-forced** pass (every frame t of the window
predicts frame t + 1 from the encoded latent of t) and a **recursive** pass (an 8-step roll-out
from the window's first frame on the model's own predictions). Measured robot state of later
frames is used **in training only**, where this list says; at evaluation (§7) a roll-out reads the
root's encoded latent and the root's measured state and nothing later.

- **S.** Teacher-forced: the state token of frame t is the embedding of the measured state at t.
  Recursive: at the first step the embedding of the measured state at the window's first frame;
  at later steps the predictor's own output in the state slot at the previous step (after the
  transformer's final LayerNorm; `pred_proj` acts on the 16 visual tokens only), carried forward
  and never supervised. No loss reads the state slot.
- **G.** The state slot's output goes through a head (MLP 192 → 256 → 28) that predicts the
  change of the standardised state. Teacher-forced: the state token of frame t is the embedding of
  the measured state at t, and the head's output is compared with the measured change from t to
  t + 1. Recursive: the first state token embeds the measured state at the first frame; the
  predicted state of step k + 1 is the predicted state of step k plus the head's change (the
  measured state at step 0), and the next state token is its embedding; it is compared with the
  measured state at step k + 1. **Joint-change loss:** the mean of the two mean squared errors
  (teacher-forced changes, recursive states), in standardised units, weight 1.
  **Inverse-dynamics loss** (AD-WM's action recovery): an MLP (2 × 16 × 192 → 512 → 7) reads an
  input latent and the *predicted* next latent and predicts the executed action's seven played
  dimensions (right arm 6–11, right grasp 13). Teacher-forced pairs: (encoded latent of t,
  prediction of t + 1 from it). Recursive pairs: (the latent the step started from — the encoded
  first frame at step 0, the previous prediction later — and the step's prediction). The loss is
  the mean of the two pairs' mean squared errors, weight 1. Applied to predicted latents, so it
  shapes the predictor; the frozen encoder cannot change.
- **C, adapted.** The repository's `state_fusion` adds a learned state embedding to the latent and
  makes the fused latent the prediction target; here the target must stay the shared visual
  latent (§3), so the embedding is added to the input tokens only: teacher-forced, the embedding
  of the measured state at t is added to every token of frame t; recursive, it is added (the
  measured state at the first frame) at the first step only, and nothing is added at later steps.
  `readout_heads` regressed declared physical readouts; here the readout is the right palm
  position (a training label from the sidecar), read from the encoded latents of every window
  frame and from the recursive predictions (MLP 3 072 → 256 → 3); the loss is the sum of the two
  mean squared errors in standardised units, weight 1. C is a control: it cannot produce the pass
  row.
- **I** is G's inverse-dynamics loss (both pair kinds) on P, without state. It is reported to separate the loss from
  the state input; it cannot produce the pass row.
- Loss weights are 1 and were not tuned; no arm has a hyper-parameter search. S, G and C carry
  extra parameters (the state embedding and heads) and G and I extra compute per update; the
  update budget U is the same for every arm and is set from P (§6.1). Per-arm parameter counts
  are reported.

## 5. Training (identical for every trained arm)

- **Windows:** 8 commands and their 9 frames, wholly inside one train episode; start drawn
  uniformly over all valid (episode, start) pairs. **Batch** 64 windows. The sampler of model
  seed m is `default_rng(SeedSequence([8714, m]))`, so every arm of a seed sees the same windows
  in the same order (paired arms).
- **Latent loss** (LeWM's, with SIGReg off because the encoder is frozen): the one-step
  teacher-forced mean squared error over the 8 steps plus the recursive 8-step roll-out's mean
  squared error (weight 1), both on the 16 visual tokens only.
- AdamW, learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, gradient-norm clip 1.0, no schedule.
- **Updates U** (equal for every arm) by the rule of §6.1; selection every U / 20 updates.
- **Model seeds** 87100, 87101, 87102 for every trained arm (3 per arm, 18 jobs). Each model is
  initialised under `torch.manual_seed(seed)`.
- Selection checkpoints are kept in host memory (as TASK-077's trainer); only the kept checkpoint
  is written.
- **Device:** the RTX 5080, one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` invocation
  per job (the lock is released between jobs), the repository's deterministic CUDA set-up
  (`devices.py`). A job that exceeds its cap (§6.1) stops; it may be re-run once, as
  `<arm>-<seed>-r2`, only if the first run did not complete.

## 6. Selection, budget and the val report

### 6.1 Budget rule (Stage B, before any gated job)

A scale probe trains P with the debug seed 87900 for 60 000 updates, selecting every 2 000 on val.
u\* is the first selection point whose val criterion is within 1 % of the probe's minimum.
**U = max(20 000, min(60 000, ⌈1.5 u\* / 5 000⌉ × 5 000)).** The per-job cap is 3 × the probe's
measured seconds per update × U. If 18 × U × the measured seconds per update exceeds 14 h, the
budget escalates to a ruling (a cloud GPU is allowed by the owner; the plan prefers to fit
locally) before any gated job. The probe is development, not a result.

### 6.2 Val roots and the criterion

Val roots: in every val episode, start frames 0, 10, 20, … with start + 16 ≤ T. The selection
criterion of every arm is the mean latent error of its recursive roll-out at steps 1–8 from every
val root under the root's executed commands (N: zero commands). The kept checkpoint is
`run_tools.select_checkpoint(curve, 0.01)` (the earliest point within 1 % of the minimum).
`last_two_triggered` is recorded as a flag; it does not escalate (TASK-074's lesson: a budget that
keeps escalating never reaches its gate).

### 6.3 The val report (Stage V)

Before the test split is read, every kept model is evaluated on the val roots with §7's
statistics, and the val report is written and its sha256 recorded in the task file and the run
log. Val numbers are not gated: val selected the checkpoints.

### 6.4 δ, the accuracy margin, measured on val with the gate's own statistic

For h ∈ {1, 4, 8}: on the val roots, for each ordered pair (i, j) of P's three seeds (i ≠ j, six
pairs), compute the statistic that (c) uses on test — the upper 97.5 % cluster-bootstrap bound
(over val episodes, §7's bootstrap with salt 8713) of acc_h(P_i) / acc_h(P_j). δ_h is the largest
of the six, minus 1 (0 if that is negative). It is how far the gate's upper bound reaches when two
equally trained plain models are compared, so an arm as accurate as another plain seed passes (c)
by construction of the bar. It is the only margin in the gate, is written into the val report and
is fixed before the test split is opened.

## 7. Statistics (Stage E, test split)

- **Test roots:** in every test episode, start frames 0, 10, 20, … with start + 16 ≤ T. Each root
  has its executed commands for the next 16 steps.
- **Wrong commands:** for each root, the 16 commands of one root from **another episode**, drawn
  once (salt 8711). **Candidates:** for each root, its own commands and those of 15 roots from
  other episodes (distinct, drawn once, salt 8712). The same tables serve every arm and seed.
- For each arm, seed and h ∈ {1, 2, 4, 8, 16}, every model rolls out from the root's encoded latent
  (and, for S, G and C, the root's measured state) — under the root's own commands, the wrong
  commands and every candidate alike — and every error is measured against **the root's own**
  encoded latent h steps later:
  - **action sensitivity** `sens_h` = Σ_roots wrong-command latent error at h / Σ_roots
    true-command latent error at h (TASK-084's wrong / true, ratio of sums);
  - **short-step ranking** `top1_h`: each candidate is scored by the latent error at h against the
    encoded latent at h; top-1 is the share of roots whose own commands score strictly best (ties
    count against; chance 1 / 16); the normalised rank (0 best, chance 0.5) is reported;
  - **accuracy** `acc_h` = Σ true-command latent error at h; **copy ratio** = acc_h / Σ copy-last
    error at h; zero / true is reported.
- **Intervals:** a cluster bootstrap over **test episodes** (10 000 resamples, salt 8713); ratio of
  sums per resample; paired differences on the same resamples. Gate intervals are two-sided
  **97.5 %**: the claim is that S *or* G passes, so the per-arm level is Bonferroni-corrected over
  the two eligible arms; within an arm the conditions are conjunctive (all must hold), which needs
  no correction. The others are reported at 95 %.

## 8. Gate and rows (first match)

For an eligible arm X ∈ {S, G}, seed m and h, against P of the same seed m:

- **(a) sensitivity:** Δsens = sens_h(X) − sens_h(P), lower 97.5 % bound > 0;
- **(b) ranking:** Δtop1 = top1_h(X) − top1_h(P), lower 97.5 % bound > 0;
- **(c) accuracy:** acc_h(X) / acc_h(P), upper 97.5 % bound ≤ 1 + δ_h;
- **(d) dynamics:** X's copy ratio at h, upper 97.5 % bound < 1.

Rows:

1. **P2-VOID** — a job or the evaluation did not complete under the rules, a hash, split, seed or
   salt check failed, or the test split was read before the val report's hash was recorded.
2. **P2-PLAIN-INVALID** — on some seed, P's copy ratio has an upper 95 % bound ≥ 1 at some
   h ∈ {1, 4, 8}, or at h = 8 P's acc / N's acc has an upper 95 % bound ≥ 1 or P's sens has a
   lower 95 % bound ≤ 1. Plain LeWM does not show learned, action-using dynamics here, so beating
   it means nothing; escalate, no claim.
3. **P2-PASS** — some X ∈ {S, G} meets (a)–(d) on **all three seeds at every h ∈ {1, 4, 8}**.
4. **P2-SHORT** — not P2-PASS, but some X meets (a)–(d) on all three seeds at h = 1. Grounding
   helps at the shortest step only; escalate, carried to Phase 3 only by ruling.
5. **P2-FAIL** — otherwise.

What each row leads to (the plan's stop rule for this phase):

- **P2-PASS:** the passing arm is Phase 3's model (if S and G both pass, the one with the larger
  mean Δtop1 at h = 1 over seeds); Phase 3 needs its own preregistration.
- **P2-SHORT, P2-PLAIN-INVALID:** escalate to a ruling.
- **P2-FAIL:** the grounded recipes S and G, as specified here on `play-v1` with this latent, are
  not carried into Phase 3 as "the grounded model"; what Phase 3 uses instead (plain LeWM, another
  grounding, more data) needs a ruling. This closes nothing else: LeWM, DINOv2, the play corpus
  and the product goal are not abandoned.
- C passing (a)–(d) is reported; it cannot produce P2-PASS (it is the recipe that failed in apple
  WM v2–v4). The same holds for I.

**Why these bars.** Every bar is relative to measured floors or spreads, not a guessed absolute
number (the lesson of TASK-074, whose readout bar sat below the readout's own measured error):
(a) and (b) compare with plain LeWM on the same roots and seeds; (c)'s margin is how far the same
upper bound reaches between two equally trained plain seeds on val (§6.4); (d) and row 2 use the copy-last and no-action floors. Requiring all
three seeds and all three horizons is conjunctive and therefore strict; with 160 test episodes
the per-seed intervals are expected to be narrow for the latent errors and wider for top-1, which
is a proportion over a few thousand roots clustered in 160 episodes.

## 9. Reported only (no gate)

- The absolute separation Σ wrong-command error − Σ true-command error per arm and its paired
  difference to P: wrong / true rises when the true-command error falls, so an arm with state
  input could gain on (a) through accuracy alone; the separation shows whether wrong commands are
  also told apart in absolute terms.
- Every statistic at h ∈ {2, 16} (16 is outside the training horizon), zero / true, the normalised
  rank, and all of §7 for C, I and N.
- **Collapse diagnostic:** the effective rank (exponential of the entropy of the normalised
  singular values) of each arm's predicted latents at h = 8 over the test roots, against that of
  the encoded latents at the same frames. With a frozen encoder the target cannot collapse; a
  predictor that collapsed to its mean would fail (d), so this is a diagnostic, not a gate.
- **Measured ceilings:** the share of roots where some other candidate's commands equal the root's
  own over the first h steps (to 10⁻⁶; no model can rank those) — the ranking ceiling is one minus
  it; and an inverse-dynamics probe (ridge on the train fit sample, its penalty chosen on val) from
  encoded (z_t, z_{t+1}) to the seven played action dimensions, its R² on test — how much of one
  command is visible in one encoded frame pair. The ranking ceiling is reported separately for
  roots at start frame 0 and for later roots.
- G's joint-change error against no change at h, C's palm readout error, G's and I's
  inverse-dynamics R² on encoded and on predicted pairs. A predicted-pair R² well above the
  encoded-pair probe's R² would mean the inverse-dynamics loss has put action information into the
  predicted latent beyond what frames show.
- **How a G pass is worded (fixed now).** If G produces P2-PASS or P2-SHORT and I also meets
  (a)–(d) on all seeds at the same horizons, the result says the gain is not shown to need the
  state input (the inverse-dynamics loss alone may carry it); if the predicted-pair R² exceeds the
  encoded-pair probe's, the result says the sensitivity gain may partly be action information
  placed in the prediction rather than better dynamics. Neither changes the row.
- Per-update time, GPU memory, parameter counts, `last_two_triggered`, kept updates.

## 10. Seeds and salts

Model seeds 87100–87102; debug and probe seeds 87900–87999; salts 8711 (wrong commands), 8712
(candidates), 8713 (bootstrap), 8714 (window sampler). None of these is used by an earlier task
(checked in `docs/` and `src/`).

## 11. Disclosed development before this draft

On 24 **train** episodes of `play-v1` shard 00 (8 268 frames; no val or test episode read), on the
RTX 5080 under the GPU lock: GPU featurisation at about 817 frames a second; GPU and CPU pooled
tokens agree to 2.8 × 10⁻⁵ (64 frames); PCA of the standardised 4 × 4 tokens kept 0.90 of the
variance at k = 128 and 0.95 at k = 192, and 0.77 of the frame-to-frame change variance at
k = 128 (not computed at 192); the right arm and hand joints vary (std 0.008–0.69 rad) and every
other joint position has std ≤ 0.006 rad. No model was trained before this draft.

## 12. Stage 0, freeze, runtime and evidence

- **Stage 0** (before the freeze): the opt-in modules (`grounded_wm*.py`; NumPy at import, torch
  lazily, no registry entry, `import embodied_jepa` unchanged), the scripts, tests (shapes, the
  recursion reads no future state, the losses, the sampler stays inside episodes, the statistics
  on a known case, the row logic, the test-split guard), and a debug smoke on debug seeds: a
  featurisation of a few train and val episodes, every arm for a few hundred updates, the val and
  the evaluation path on val episodes standing in for test, and a CUDA determinism check (one job
  twice, identical weights). Smoke numbers are not results.
- **Frozen** when the Stage 0 PR merges; the gated stages run at the merged revision from a clean
  tree.
- **Stages after the freeze:** F (featurise train and val; one GPU job in two passes: pass A
  featurises the fit sample and fits the standardisation and projection, pass B featurises every
  train and val episode again and writes the projected latents, so no unprojected store is
  written), B (the budget probe, §6.1, which also reports P's copy ratio at h = 1 on val), T (18 jobs), V (the val report; δ), then F′ (featurise test with the frozen projection,
  its hash checked) and E (the gated evaluation and the row).
- **Disk:** about 24 GB free before this task; features about 7 GB; nothing is written while `/`
  has less than 10 GiB free (the run stops and records it). The converted real-data stores of
  TASK-086 (`data/real-g1-v1`, unused since R23.28) may be deleted only if space is needed, and
  the deletion is recorded.
- **Outputs** under new names: `outputs/task087-*`, `checkpoints/task087-*`; evidence in
  `~/develop/emai/evidence/task087-*/` with `SHA256SUMS`.
- **GPU time:** featurisation about 30 min; training 18 × U updates (estimated 3–8 h at U = 40 000,
  to be measured by the probe).

## 13. What this cannot show

It is offline: a better ranking against the true future frame is not a planner (a planner does
not have that frame) and not a closed loop. One corpus (scripted play, right arm and hand,
simulation), one camera at 112 px, one encoder, one pooling and projection, one architecture and
budget, three seeds; a pass is evidence for this grounding on this data, not in general. The
grounded arms have more parameters (state embedding, heads) and, for G and I, more compute per
update than P at the same number of updates; a pass does not separate grounding from that extra
capacity. The
latent errors are comparable across arms only because every arm shares the frozen encoder and
projection.

## 14. Rulings this document records (R24.1–R24.14, DRAFT; decided by Claude under owner delegation)

- **R24.1** — TASK-087 is Phase 2, on `play-v1` only (the real arm was dropped, R23.28; no
  Teleop-only real arm now, revisited only if Phase 3 suggests it).
- **R24.2** — the latent: frozen DINOv2 ViT-S/14 tokens, 4 × 4 pooled, train-only standardisation
  and a k = 192 projection fitted on the fit sample (§3), float16 storage.
- **R24.3** — state: the right arm and hand's 28-D positions and velocities; labels: the right palm
  position, for C only.
- **R24.4** — arms P, S, G (eligible), C (control), I (decomposition), N and copy-last (floors);
  one architecture (§4); loss weights 1, untuned.
- **R24.5** — training as §5; 3 model seeds per arm (87100–87102), paired windows.
- **R24.6** — U by the probe rule of §6.1; escalation above 14 h of GPU time.
- **R24.7** — selection on val by the true-command criterion, tolerance 1 %; `last_two_triggered`
  is a flag only.
- **R24.8** — δ_h from the gate's own bootstrap statistic over pairs of P's seeds on val (§6.4),
  fixed before the test split is opened.
- **R24.9** — test roots, wrong-command and candidate tables, statistics and the cluster bootstrap
  as §7; 97.5 % gate intervals.
- **R24.10** — the gate and rows as §8; P2-PASS needs all seeds at h = 1, 4 and 8.
- **R24.11** — the stop rule as §8: P2-FAIL takes S and G off Phase 3's table; nothing else closes.
- **R24.12** — seeds and salts as §10.
- **R24.13** — disk and evidence as §12; the TASK-086 stores may be deleted only when needed and
  recorded.
- **R24.14** — R7 does not change, whatever the row.

## 15. Stage 0 record (R24.15–R24.19)

- **Code** (opt-in; no registry entry; `import embodied_jepa` unchanged and torch-free, checked by
  a test): `src/embodied_jepa/grounded_wm.py` (constants, tables, statistics, δ, rows; NumPy),
  `grounded_wm_data.py` (corpus reading, featurisation, the projection, the stores, the sampler;
  NumPy at import), `grounded_wm_model.py` (the arms; torch); scripts `featurise_task087.py`,
  `train_task087.py`, `evaluate_task087.py` and the stage runner `run_task087.sh` (one
  `gpu_run.sh` lock per job); `tests/test_grounded_wm.py` (29 tests: windows stay inside episodes,
  the root rule, the wrong and candidate tables use other episodes and are fixed, ties count
  against, the cluster bootstrap, the statistics on a known predictor, the budget rule, δ, the
  criteria and every row, the projection against a direct PCA, the stores and the sampler, every
  arm's loss and roll-out, the roll-out reads only the start state, S carries the state slot after
  the final LayerNorm, N ignores commands, chunk-independent prediction). No existing file changes.
- **Evaluation device (decided here, before the freeze):** V and E run on the RTX 5080 under
  `gpu_run.sh` with the repository's strict deterministic CUDA set-up, in chunks of 2 048 roots. On
  the CPU the 18 models' roll-outs were estimated (from operation counts, not measured) at
  roughly 1–2 h; on the GPU the smoke took seconds.
- **Debug smoke** (debug seeds 87901–87902, not results; `outputs/task087-smoke/` in the Stage 0
  worktree): Stage F on 12 fit-sample train episodes and 3 val episodes (0.953 of the
  standardised variance and 0.869 of the frame-to-frame change variance kept at k = 192, about
  0.6 s per episode); every arm for 400 updates on that store (31–37 ms per update; parameters
  P and N 3.12 M, S 3.18 M, C 3.97 M, I 6.27 M, G 6.39 M; peak CUDA allocation 1.7 GB); the val
  report and a test evaluation with val standing in for test (row logic exercised; with 400
  updates on 12 episodes P does not beat copy-last, so the smoke's row is P2-PLAIN-INVALID, as
  expected of an untrained model). The smoke showed the case §9 guards against: G's
  inverse-dynamics head read its *predicted* pairs (R² 0.48) far better than encoded pairs
  (−0.48).
- **Determinism:** P-87901 trained twice for 400 updates gave identical weights (state sha256
  `ad8ed85a2219…` both times).
- **Projection of cost:** about 35 min for Stage F; at about 31–37 ms per update, U = 40 000 would
  take about 7 h for the 18 jobs, inside R24.6's 14 h.
- **R24.15–R24.19** — the Stage 0 code above (R24.15); the evaluation device (R24.16); the smoke
  and determinism results are not evidence (R24.17); frozen at this PR's merge, the gated stages
  run at the merged revision from a clean tree with evidence in `~/develop/emai/evidence/task087-*`
  (R24.18); R7 does not change (R24.19).
