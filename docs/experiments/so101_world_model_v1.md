# TASK-100 — SO-101 S4: an offline world-model study on `so101-play-v1` (LeWM and `native_jepa` token predictors over frozen DINOv2 tokens of both cameras)

**Status: DRAFT preregistration** (R28.12–R28.26 in §15, decided by Claude under owner delegation;
opened by R28.10–R28.11, [DECISIONS](../DECISIONS.md) 2026-10-17 (bq)). It becomes **FROZEN when
the Stage 0 PR merges**; after that merge nothing in §2–§12 changes without its own reviewed
ruling. Nothing has been trained, featurised or evaluated for this document.

This is phase S4 of [SO101_PLAN.md](../SO101_PLAN.md) ("LeWM and `native_jepa` training with
offline gates") on the S3 corpus `so101-play-v1` ([so101_play_corpus_v1.md](so101_play_corpus_v1.md),
TASK-099). **It is an offline world-model study, not a control result and not an Apple→Plate
experiment.** No planner, controller or closed loop runs. SO-101 results do not enter R7 (R28.9);
R7's canonical status sentence does not change, whatever row this task ends in.

Closest precedents, and what is taken from each:

- **TASK-066** ([apple_token_dynamics_v1.md](apple_token_dynamics_v1.md)): the token predictor
  over frozen, pooled DINOv2 patch tokens (`models/frozen_tokens.py`), the arms W / N / copy-last,
  the gate family G1–G5 and G1's calibrated rank bar with the truncation check.
- **TASK-087** ([grounded_lewm_v1.md](grounded_lewm_v1.md)): train / val / test discipline (test
  not featurised before the val report's hash is recorded), the budget probe and U rule, the
  cluster bootstrap over test episodes, the wrong-command and candidate tables, Bonferroni over
  eligible arms, and a train-only channel projection (PCA) of the tokens.
- **TASK-077** ([apple_lewm_c1m_v2.md](apple_lewm_c1m_v2.md)) and its decomposition record
  ([apple_lewm_c1m_v2_decomposition.md](apple_lewm_c1m_v2_decomposition.md)): G5 failed because a
  readout fitted on encoded frames was applied to predicted latents (a distribution shift; a
  readout refitted on the predictor's own outputs read the plate within bar). Here both readouts
  are declared, and the gated one is fitted on predicted latents (§7.3).
- **[RESUME.md](../RESUME.md) and the plan's lessons:** every bar is calibrated from a measured
  ceiling on development data (TASK-074's O2 bar sat below its readout's own measured error);
  measure before gating; budgets declare a saturation check (TASK-066/081 `last_two_triggered`)
  with an escalation that cannot loop (TASK-074).

## 1. Question

On `so101-play-v1`, does an action-conditioned token predictor over frozen DINOv2 ViT-S/14 patch
tokens of **both cameras** (wrist and overview), trained on `joint_delta_so101_v0` commands,
learn dynamics that (i) do not collapse, (ii) beat copy-last and an equally trained no-action
predictor, (iii) use the commands, and (iv) keep the **cube position** and the **gripper
opening** readable from the predicted latent at h = 8 and h = 16 control steps, on the held-out
test split, for every model seed, with a training budget that saturates? This is what S5 (a
LeWM-driven closed loop on `so101_cube_lift_v0`) needs before it can be preregistered: a model
whose roll-outs carry the quantities a lift controller would be judged on.

## 2. Data

- **Corpus:** `data/so101-play-v1/` as sealed by TASK-099 (`corpus.json` SHA-256
  `c81571de2bebe76704129d2814f39959d341ec8cba169fc310f7c22d8dd45b69`; every shard's manifest
  hash checked against [`benchmarks/manifests/so101-play-v1.json`](../../benchmarks/manifests/so101-play-v1.json)).
  3 000 episodes of 400 commands and 401 frames at 20 Hz; splits as frozen: train 2 700, val 150,
  test 150 episodes.
- **Use of the splits.**
  - **Train** fits everything: feature standardisation and projection, the readouts, the models.
  - **D0, the development subset of train:** the train episodes of shards 00–02 (seeds
    991000–991299; 270 episodes), split by shard into **D0-fit** (shards 00–01, 180 episodes)
    and **D0-eval** (shard 02, 90 episodes). Stage 0 calibrates every bar on D0 only (§10). D0
    stays part of train for the gated models (it is training data, never evaluation data).
  - **Val** selects checkpoints, decides G6's extension and is reported in the val report
    (§8.3); val numbers are not gated.
  - **Test** is not read, decoded or featurised until Stage F′, which starts only after the val
    report is written and its SHA-256 recorded in the task card and the run log (TASK-087's rule).
- **Frames:** `wrist_rgb` and `overview_rgb`, 224 × 224 (DINOv2's native input size; no resize).
- **Actions:** the stored executed `joint_delta_so101_v0` actions (6-D, normalised to [−1, 1]; the
  gripper channel is an absolute command). The corpus stores the clipped, executed action
  (26.4 % of commands clipped, mostly at the wrist_flex limit and the gripper rate), so at the
  wrist_flex limit a positive and a zero command have the same effect; this caps action
  sensitivity and is disclosed (§13).
- **Labels (never model inputs, never planning inputs):** the cube centre position (world frame,
  from the TASK-099 sidecar truth, reported in cm) and the measured gripper jaw angle (rad, from
  `observation.state`, `so101_proprio_v0`). They are used only by the readouts of §7.3 and the
  subset definitions of §7.4.

## 3. The latent: frozen DINOv2 tokens of both cameras

- **Encoder:** the pinned frozen DINOv2 ViT-S/14 (`pretrained_encoder.load_pretrained`, weights
  checked against their pins), ImageNet normalisation, final patch tokens (16 × 16 × 384) per
  camera. GPU float32, eager attention, a fixed batch of 64 frames per camera.
- **Grid, chosen at Stage 0 by the rule of §10.3** between two declared candidates of equal
  storage (24.6 kB a frame for both cameras in float16):
  - **g4:** each camera's tokens average-pooled to 4 × 4 (`frozen_tokens.pool_tokens`, TASK-066's
    grid), 384 channels: 32 tokens × 384.
  - **g8:** each camera pooled to 8 × 8, then the 384 channels projected to **k = 96** by principal
    components of the standardised channels (pooled over cameras and grid positions), fitted on
    the fit sample, without whitening (TASK-087's form): 128 tokens × 96. *(Clarified at Stage
    0, review note 4.)* g8 takes **two GPU passes over the fit sample**: pass A accumulates the
    per-dimension moments and the per-slot channel second moments of the pooled 8 × 8 tokens and
    fits the projection; pass B featurises every train and val episode (the fit sample again) and
    writes the projected tokens. The flat-latent standardisation below is then fitted on the CPU
    from the stored latents of the fit sample. g4 takes one pass.
- **The frame latent** is the wrist tokens followed by the overview tokens, flat, 12 288 values
  for either grid. Camera identity is carried by the predictor's per-token position embedding.
- **Standardisation:** per latent dimension, mean and std (floor 10⁻⁶) from the **fit sample**:
  train episodes whose seed is divisible by 4 (about 675 episodes, about 270 000 frames).
- **Random-init floor latent:** the same pipeline with `pretrained_encoder.random_init` (the
  architecture at its seed-0 initialisation, TASK-063's floor), its own standardisation and (g8)
  its own projection fitted on the same fit sample. One floor seed only (§13).
- **Distance:** the mean squared difference over the 12 288 standardised values ("latent error").
  Latent errors are comparable between arms that share the pretrained latent (L, J, N_L, N_J);
  the floor arm R lives in another latent space, so it is compared with the others only through
  physical readout errors and rankings (§7.3, §9), never through latent errors.

## 4. Arms

| Arm | Latent | Predictor (backend) | Commands | Role |
| --- | --- | --- | --- | --- |
| **L** | pretrained | LeWM token predictor (`leworldmodel`) | executed | **eligible** (the product-goal backend) |
| **J** | pretrained | native residual token step (`native_jepa`; the one-key backend swap of L) | executed | **eligible** |
| **N_L** | pretrained | as L | all zero, in training and evaluation | no-action floor for L (trained) |
| **N_J** | pretrained | as J | all zero | no-action floor for J (trained) |
| **R** | random-init DINOv2 | as L | executed | encoder floor (trained); reported, not eligible |
| **copy-last** | either | — | — | floor: the start latent at every horizon |

- **Implementation (Stage 0, new files only).** A schema-sized, two-camera frozen-token class in a
  new module `models/schema_frozen_tokens.py`: TASK-066's `FrozenTokenMixin` composition (the
  pinned upstream LeWM `ARPredictor`, `Embedder` and `pred_proj` with the token grid as the
  sequence axis, bidirectional within the frame, the action embedding conditioning every token;
  or the native residual token step) placed in front of S0's `SchemaLeWM` / `SchemaNativeJEPA`
  ([schema_backends.py](../../src/embodied_jepa/models/schema_backends.py)), with the action
  width taken from `joint_delta_so101_v0` (6) instead of `EE_DELTA_GRASP_V0.dimension` and the
  token count and width from §3. Training runs through the backend's own `train_step` on cached
  features (`train_step_features`), with SIGReg off (`sigreg_weight` 0; the encoder is frozen),
  as TASK-066. The backend is one key of the run's model configuration; nothing else differs
  between L and J. `models/frozen_tokens.py`, `models/frozen_encoder.py`, `models/base.py`, the
  backend files, `pretrained_encoder.py`, `schema_backends.py` and every G1 byte-pinned file stay
  byte-identical (SO101_PLAN S0); `config.py` is not changed (it refuses an `so101` experiment
  config, so the run configuration is a module constant, TASK-066's `MODEL_CONFIG` precedent).
- **Configuration:** the adapters' defaults otherwise; `max_horizon` 32. Predictor sizes are
  recorded with parameter counts; there is no hyper-parameter search.
- **Pairing.** Every arm of a model seed draws the same windows in the same order (§5).

## 5. Training (identical for every trained arm)

- **Windows:** 16 commands and their 17 frames, wholly inside one train episode, start drawn
  uniformly over all valid (episode, start) pairs; batch 64. The sampler of model seed m is
  `default_rng(SeedSequence([10011, m]))`, shared by every arm of seed m. *(Clarified at Stage
  0, review note 1.)* Model seed m also initialises each arm's predictor. Calibration N_L
  (§10.4) initialises from its own seed 100911 but draws from calibration L's stream,
  `SeedSequence([10011, 100900])`, so the two calibration models see the same windows in the same
  order; every other job's stream is its own seed.
- **Loss and optimiser:** the backend's own `train_step` (TASK-066's recipe: teacher-forced
  one-step and the backend's multistep term over the window), AdamW at the adapter's defaults,
  no schedule.
- **Updates U**, equal for every arm, by the rule of §8.1; a selection point every U / 20 updates.
- **Model seeds** 100100, 100101, 100102 for every trained arm: 5 arms × 3 seeds = **15 jobs**.
- **Selection:** the val criterion is the mean latent error of the recursive roll-out at steps
  1–16 from every val root (§7.1) under the root's executed commands (N arms: zero commands; R in
  its own latent). The kept checkpoint is `run_tools.select_checkpoint(curve, 0.01)` (the earliest
  point within 1 % of the minimum). Each job also writes its final full state (weights, optimiser,
  RNG, sampler position) so that G6's one extension (§6) can resume it.
- **Device:** the RTX 5080, one `scripts/gpu_run.sh --wait --board` invocation per job (the lock is
  released between jobs; the V2D requests on `~/.local/state/gpu/BOARD.md` are respected), the
  repository's deterministic CUDA set-up (`devices.py`). A job that did not complete may be re-run
  once as `<arm>-<seed>-r2`.

## 6. Gates (per eligible arm X ∈ {L, J}, per model seed, at h = 8 and h = 16)

**Why h = 8 and h = 16.** Control runs at 20 Hz (`control_dt_s` 0.05). The gripper target moves
at most 0.25 rad a step, so a full open-to-closed command (1.92 rad) takes 8 steps: h = 8 (0.4 s)
spans one closure and up to 0.5 rad of arm joint motion. The S2 follower's close phase is 20
steps and its lift 30 steps, the lift scorer's hold window is 20 steps, and the collector's lift
segments are 6–12 cm: h = 16 (0.8 s) spans a closure plus the start of a lift, which is the
look-ahead an S5 planner on `so101_cube_lift_v0` would need at the least. Training windows are 16
commands, so h = 16 is the longest trained horizon; h ∈ {1, 4, 32} are reported only.

Gate intervals are two-sided **97.5 %** cluster-bootstrap intervals (Bonferroni over the two
eligible arms; within an arm the gates are conjunctive and need no correction). "ub" and "lb" are
their upper and lower bounds.

| Gate | Passes when (at h) | Bar source |
| --- | --- | --- |
| **G1 no collapse** | (i) the collapsed fraction (latent dimensions whose std of X's predictions over the test roots is < 0.01 × the std of the encoded latents at the same frames) ≤ 0.05; **and** (ii) the effective-rank ratio of X's predicted to the encoded latents ≥ **B_rank** and the std ratio ≥ **B_std**; **and** (iii) lb of (rank ratio of X − rank ratio of N_X) **> 0**, in the projected basis below | (i) **carried** (TASK-065/066/077); (ii) **calibrated at Stage 0** (§10.4); (iii) **definitional** (TASK-066's owner Ruling 1, carried as TASK-077's G1 (iii)) |
| **G2 beats copy-last** | ub of acc_h(X) / acc_h(copy-last) **< 1.0** | **definitional** (TASK-087's (d)); the TASK-066 allocation 0.8 is reported beside it |
| **G3 beats no-action** | ub of acc_h(X) / acc_h(N_X) **< 1.0** | **definitional** |
| **G4 action sensitivity** | lb of sens_h(X) = Σ wrong-command latent error / Σ true-command latent error **≥ 1.10** | **carried** (TASK-065/066/077), an allocation, not calibrated on this corpus; calibration L's value is disclosed at Stage 0 and does not move it, and if its lb on D0-eval is below 1.10 the owner may rule before the freeze (§10.4) |
| **G5 cube and gripper readable from the prediction** | with the predicted-latent readout R-pred of X (§7.3), for each label ℓ ∈ {cube, jaw}: (a) ub of the median error on all test roots ≤ **B_ℓ**; (b) on the moving subset M_ℓ,h (§7.4), ub of median error(X) / median error(copy-last readout) **< 1.0**; (c) on M_ℓ,h, ub of median error(X) / median error(N_X's R-pred) **< 1.0** | (a) **calibrated at Stage 0** from the encoded-frame readout's measured error (§10.5); (b), (c) **definitional** |
| **G6 training saturation** | for X **and** N_X of this seed: the val criterion's best value over the last quarter of training is not more than **2 %** below its best value up to 0.75 × the job's final update count, after at most one extension (below) | **declared** |

- **G1 details.** The effective rank is the exponential of the entropy of the normalised
  eigenvalues of the centred roots × roots Gram matrix (TASK-077's exact form; float64 on the CPU).
  B_rank and B_std come with TASK-066's owner condition: the combined G1 must **fail** the rank-1,
  rank-2 and rank-4 truncations of calibration L's D0-eval predictions, or the bar is raised (to
  the smallest 0.01 step that fails all three) and the raise is disclosed before the freeze.
- **G1 (iii), the comparative part.** TASK-066's owner ruled (§6.4, Ruling 1) that a rank bar set
  from the tested architecture's own calibration alone is circular, and added the requirement that
  W keep more rank than its no-action twin; TASK-077 carried it as G1 (iii). It is carried here
  for each eligible X against N_X of the same seed. A cluster bootstrap that recomputes a 5 550 ×
  5 550 Gram eigendecomposition 10 000 times is too costly, so (iii) is computed, as in TASK-066,
  in a **projected basis**: the top 256 principal directions of the standardised encoded latents
  of the fit sample (fitted at Stage F before any model is trained; its SHA-256 recorded). In that
  basis each test episode contributes the sums and cross-products of its roots' projected
  predictions (of X and of N_X) and projected encoded targets; each resample of §7.5's bootstrap
  (the same 10 000 resamples, salt 10014) adds the moments of its episodes, forms the 256 × 256
  covariances and recomputes both rank ratios, and (iii) reads the lb of their difference. The
  point value is the same statistic on all test episodes. **This is an approximation** (the
  effective rank of a 256-direction projection, not of the full 12 288-value latent) and is
  disclosed as one; the full-width difference is reported beside it. G1 (ii) is unchanged and
  stays full-width.
- **G4 uses wrong commands only.** The zero command is not a hold for this schema: the gripper
  channel is absolute, so zero commands the jaw half open. zero / true and hold / true (arm zero,
  gripper at the root's commanded value) are reported, not gated.
- **G5 is gated on R-pred, not R-enc** (TASK-077's decomposition lesson; it is the readout an S5
  controller would use on predictions). R-enc applied to X's predictions is reported beside it as
  the distribution-shift diagnostic. (b) asks that the prediction carry more than the start frame
  where the label moves; (c) asks that it carry the commands' effect on the label. *(Clarified
  at Stage 0, review note 3.)* For the jaw, (c) is partly met by the schema rather than by
  learned dynamics: X receives the absolute gripper command, which names the jaw's target
  directly, while N_X receives zero commands. A pass of G5 (c) on the jaw therefore shows that
  the prediction carries the commanded target, not that the model learned the jaw's motion from
  images (disclosed in §13).
- **G6's one extension.** At the end of Stage T, for each eligible arm X: if any of the six models
  X-m and N_X-m (the three seeds m) fails G6, **all six** resume from their final state to 1.5 U, with the
  same sampler stream, and selection is redone over the whole curve (Stage T′). There is no second
  extension: a model that still fails after T′ fails G6. R gets no extension; its G6 is reported.
  `last_two_triggered` is recorded for every model as a flag. (TASK-074's lesson: a budget rule
  that keeps escalating never reaches its gate; G6 escalates once, then decides.)

**A seed passes for X at h** if G1–G6 all pass. G6 is per seed, not per h.

## 7. Roots, tables, readouts and statistics

### 7.1 Roots

In every episode of a split, start frames 0, 10, 20, … with start + 32 ≤ 400: 37 roots per
episode, 5 550 test roots and 5 550 val roots. Each root has its next 32 executed commands. Every
model rolls out from the root's encoded latent only; nothing later is read except as a target.

### 7.2 Tables

- **Wrong commands:** for each root, the 32 commands of one root from another episode of the
  same split, drawn once (salt 10012).
- **Candidates (§9's planning measure):** for each root, 16 command sequences: the executed one;
  4 with N(0, 0.1²) added to every channel and 4 with N(0, 0.3²), clipped to [−1, 1]; 3 with
  only the gripper channel perturbed by N(0, 0.5²), clipped; and 4 from roots of other episodes
  (salt 10013). The same tables serve every arm and seed.

### 7.3 Readouts (ridge regression; float64; CPU)

All readouts map one standardised 12 288-value latent to the label at the latent's frame: the
cube centre (3-D, cm) and the jaw angle (rad). The ridge penalty is chosen per readout from
{10⁻¹, 10⁰, …, 10⁵} by 5-fold episode-grouped cross-validation on its own fitting data (folds
salt 10015).

- **R-enc** (the encoded-frame readout): fitted on encoded latents of every fifth frame of the
  fit-sample episodes. Applied to encoded test frames at t + h, it is the **ceiling** (reported),
  and applied to X's predictions it is the distribution-shift diagnostic (reported).
- **R-pred(X, h)** (the gated readout; one per model and h): fitted on X's own predicted latents
  at h from the **train** roots of the fit-sample episodes (§7.1's rule) under their executed
  commands (N arms: zero commands), against the truth at t + h; applied to X's predictions on the
  test roots. *(Clarified at Stage 0, review note 2.)* The fit-sample episodes are training data
  of every model, so R-pred is fitted on **in-sample predictions** (roll-outs from windows the
  model trained on), as in TASK-077's decomposition. If X's predictions are better on train than
  on test, R-pred's fit is made on a cleaner input than it is applied to; that would make G5
  harder to pass, not easier, but it also means R-pred's own fit error is not an estimate of its
  test error. Disclosed in §13.
- **Copy-last readout(h)** (one per h): fitted on the encoded latent of each **train** root of the
  fit-sample episodes (§7.1's rule) against the truth at t + h, with the same 5-fold
  episode-grouped cross-validation and fold salt (10015) as R-pred; applied to the encoded latent
  of each test root (the best a readout of the start frame alone can do); the floor of G5 (b).
- R's readouts are fitted the same way in R's latent space; they enter only §9's comparison.
- **G5 (a) has no absolute anchor.** B_ℓ is set relative to R-enc's own error on D0 (§10.5), and
  test episodes are drawn from the same distribution as D0, so R-enc on encoded test frames would
  miss B_ℓ only if its test error were more than 1.5 × its D0 error. A "ceiling flag" defined
  against B_ℓ could therefore not fire, and none is declared. What G5 (a) tests is that the
  prediction keeps what the encoded frame shows, within 1.5 ×; whether that is precise enough for
  a task is not gated. Reported beside it, against an absolute reference declared now: R-enc's and
  R-pred's median cube error on the test roots against **0.5 cm** (S2's measured grasp-offset
  tolerance toward the fixed jaw, the tightest measured direction) and against **3 cm** (S2's
  place-aim tolerance). The jaw has no measured task tolerance; its errors are reported in rad
  and degrees without a reference.

### 7.4 Moving subsets

M_cube,h: test roots where the cube centre moves ≥ **1.0 cm** between t and t + h.
M_jaw,h: test roots where the jaw angle changes ≥ **0.2 rad** between t and t + h. Stage 0
measures their shares on D0-eval; if either share × 5 550 is below 300 roots at h = 8, its
threshold is halved (once) before the freeze and the change is disclosed. A subset with fewer
than 100 test roots at E makes that G5 sub-gate uncomputable. That depends only on the test split,
so a repeat would give the same count: the task closes **INCONCLUSIVE** directly, with no `-r2`
repeat (§11).

### 7.5 Statistics

- **acc_h(X)** = Σ over roots of the true-command latent error at h; **copy ratio** =
  acc_h(X) / acc_h(copy-last); **sens_h(X)** as in §6 (ratio of sums, TASK-084/087's form).
- **Readout errors:** Euclidean cube error in cm and absolute jaw error in rad (also in degrees);
  medians over roots.
- **Intervals:** a cluster bootstrap over **test episodes** (10 000 resamples, salt 10014); ratios
  of sums or of medians per resample; paired differences on the same resamples.

## 8. Stages, budget and the val report

### 8.1 Budget rule (Stage B, before any gated job)

A scale probe trains L with probe seed 100920 for 60 000 updates, selecting every 2 000 on val.
u\* is the first selection point whose val criterion is within 1 % of the probe's minimum.
**U = max(20 000, min(60 000, ⌈1.5 u\* / 5 000⌉ × 5 000))** (TASK-087's rule). The per-job cap is
3 × the probe's measured seconds per update × U. The probe is development, not a result.

### 8.2 GPU budget and hard cap

| Stage | Estimate (GPU h; estimates, not measurements) |
| --- | --- |
| Stage 0 (featurisation probe on D0 with both encoders, grid rule, calibration L and N_L, debug smoke, determinism) | 2.5–3.5 |
| F: featurise train + val, both encoders (about 2.29 M images per encoder, both cameras; TASK-087 measured about 817 frames a second for ViT-S/14) | 1.5–2.5 |
| B: budget probe (60 000 updates) | 1–1.5 |
| T: 15 jobs × U updates (about 60–90 ms an update projected for 32–128 tokens at batch 64, 16-step windows; U = 30 000 gives 0.5–0.75 h a job) | 7.5–11 |
| T′: G6's extension, at most 12 jobs × 0.5 U (only if triggered; at U = 30 000 and 60–90 ms an update, 12 × 15 000 updates) | 0, or 3.0–4.5 |
| V, F′, E: val report, featurise test, gated evaluation and readouts | 1–1.5 |

The upper ends of these estimates sum to more than the cap (about 24.5 GPU h with a full T′);
the U rule below lowers U before any gated job if the measured projection does not fit.

**Hard cap: 20 GPU h in total** (the plan's 10–20 GPU h), Stage 0 included, Stage 0 itself capped
at 4 GPU h. After Stage B, if the projection of T plus a full T′ plus V–E exceeds the remainder of
the cap, U is lowered in steps of 5 000 (not below 20 000) until it fits; if it does not fit at
20 000, the task escalates to a ruling before any gated job (cloud GPU time is the owner's
decision). Exceeding a stage's measured projection by more than 1.5 × stops the run (VOID).
CPU-only stages (readouts, statistics) do not take the GPU lock. Every GPU job goes through
`scripts/gpu_run.sh --wait --board`; the session running the stages tells the V2D session
before any job longer than an hour.

### 8.3 The val report (Stage V)

Before the test split is read, every kept model is evaluated on the val roots with §7's
statistics (readouts fitted on train as declared), G6's status is recorded, and the val report is
written and its SHA-256 recorded in the task card and run log. Val numbers are not gated.

### 8.4 Stages after the freeze

F (featurise train and val, both encoders; the fit-sample pass fits standardisation, for g8 the
projections, and G1 (iii)'s 256-direction basis, before any model is trained; nothing unprojected
is stored), B (§8.1), T (15 jobs), T′ (only if G6 triggers), V
(§8.3), F′ (featurise test with the frozen standardisation and projections, hashes checked), E
(readouts on train, then the gated evaluation and the row). All run at the merged revision from a
clean run tree (`scripts/new_worktree.sh <dir>-run --from <frozen sha> --run`).

**Disk.** About 29.6 GB of float16 features per encoder for all splits, 59 GB for both, written
under `outputs/task100-features-*` (HDD, `/mnt/data`), with a hashed working copy on the SSD for
training (I/O-heavy random windows; RESUME.md's rule). The working copy holds train and val for
both encoders, about **56 GB** (2 850 of 3 000 episodes × 29.6 GB × 2); `/` had about 118 GB free
on 2026-10-17. The free space is checked **before the copy starts**: the copy proceeds only if
`/` would keep at least 10 GiB free after it, otherwise the run stops before copying and records
it (training then reads the HDD store, and the measured per-update time is re-projected against
the cap). Nothing is written while `/` has less than 10 GiB free; the run stops and records it. Checkpoints under `checkpoints/task100-*`, run outputs
under `outputs/task100-*`, evidence in `~/develop/emai/evidence/task100-*/` with `SHA256SUMS`.
Nothing is overwritten; every name is new. Prior evidence under `data/`, `checkpoints/`,
`outputs/` is never touched; the SSD working copy is deleted after Stage E and its deletion
recorded.

## 9. Reported only (no gate)

- **Planning usefulness** (what S5 needs). For each arm, seed and h ∈ {4, 8, 16}: every candidate
  of §7.2 is scored by the latent error between its predicted latent at h and the root's encoded
  latent at t + h (the goal frame). Reported: the share of roots whose executed sequence scores
  strictly best (ties count against; chance 1/16) and its normalised rank (0 best, chance 0.5),
  overall and per candidate class (σ 0.1, σ 0.3, gripper-only, other episode); the same with a
  physical goal cost (R-pred's cube and jaw estimate against the truth at t + h, cube error in cm
  plus jaw error in rad × 1 cm/0.1 rad, a declared weighting); and the **ranking ceiling** (one
  minus the share of roots where some candidate equals the executed sequence over h steps to
  10⁻⁶, and, separately, where every candidate leads to the same jaw target because of the rate
  clip). Copy-last ties on every root by construction; N_X is reported for reference. This is
  offline: a planner does not have the goal frame, and a ranking against it is not a closed loop.
- **The encoder floor R:** R's G1–G6 quantities (in its own latent space), its R-pred errors and
  its planning measure. Declared reading: if L's median cube error under R-pred is below R's on
  all three seeds at h = 8 and h = 16 (ub of the paired difference L − R < 0, 95 %), the results
  say "pretrained features read the predicted cube better than the random-init floor"; otherwise
  "pretraining is not shown to matter here" (the TASK-072/076 reading). Neither changes the row.
- Every gate quantity at h ∈ {1, 4, 32}; the TASK-066 G2 allocation (ub ≤ 0.8); G1 (iii)'s rank
  difference X − N_X at full width beside the gated projected value; zero / true and hold / true;
  R-enc on predictions (the distribution-shift diagnostic) and its difference to R-pred; G5 split
  by phase label (reach, descend, close, lift, transport, release) and by camera visibility of the
  cube; the readout errors against S2's measured envelope (§7.3: a 5 mm grasp offset toward the
  fixed jaw kept 64/64, 10 mm did not; a 3 cm place-aim error kept ≥ 63/64): **G5 passing does not
  mean a readout is precise enough to grasp**, which is S5's question.
- Per-update time, peak GPU memory, parameter counts, kept updates, every val curve,
  `last_two_triggered`, GPU hours per stage.

## 10. Stage 0 (development; before the freeze)

Stage 0 reads **D0 only** (train episodes of shards 00–02) for every calibration, plus val
episodes only where §10.6's smoke stands val in for test to exercise the code path (its numbers
are not read as results, and they set nothing). Test is never opened.

**The Stage 0 fit set is D0-fit.** Stage 0 cannot use the fit sample of §3 (most of its about 675
episodes lie outside D0), so at Stage 0 every fit that the gated stages make on the fit sample is
made on **D0-fit** (shards 00–01, 180 episodes) instead, by the same rules: the standardisation
(every D0-fit frame), the g8 projection, G1 (iii)'s 256-direction basis, R-enc (every fifth
D0-fit frame), and calibration L's R-pred and copy-last readouts (D0-fit roots, §7.1's rule, the
same cross-validation and fold salt). Everything is then applied to D0-eval (shard 02). These
Stage 0 fits are recorded with their hashes and are not reused by the gated stages, which refit
on the fit sample at Stage F.

### 10.1 Code and tests

New opt-in modules (no registry entry; `import embodied_jepa` unchanged and torch-free, checked by
a test): `so101_wm.py` (constants, roots, tables, statistics, gates, rows; NumPy),
`so101_wm_data.py` (corpus reading for both cameras, featurisation, pooling and projection,
feature stores, the sampler; NumPy at import), `models/schema_frozen_tokens.py` (§4; torch);
scripts `featurise_task100.py`, `train_task100.py`, `evaluate_task100.py` and the stage runner
`run_task100_stage.sh` (one `gpu_run.sh` lock per job). Tests: windows stay inside episodes; the
root rule; the wrong and candidate tables use other episodes and are fixed by their salts; ties
count against; the cluster bootstrap on a known case; every gate and row in both directions; the
test-split guard; G6 and its single extension; the moving-subset rule; R-pred and the copy-last
readout are fitted on train roots only; G1 (iii)'s per-episode moments reproduce the direct
projected rank ratios on a known case; a 6-D schema through the frozen-token class for both backends (`train_step_features`,
prediction shapes); checkpoints refuse other implementation hashes; every G1 byte-pinned file,
the G1 model implementation hashes and every `frozen_files_check` match `origin/main`.

### 10.2 Featurisation probe (GPU)

Both encoders on D0: throughput (frames a second, both cameras), GPU against CPU agreement on 64
frames, two featurisations of one episode bit-identical at the fixed batch, the PCA variance kept
at k = 96 for g8. The measured rate gives F's projection.

### 10.3 The grid rule (declared now; applied to D0 numbers)

R-enc is fitted on D0-fit (every fifth frame) and applied to every frame of D0-eval, for g4 and
g8 with the pretrained encoder. **g8 is chosen if and only if** its median cube error on D0-eval
is at most **0.85 ×** g4's **and** g4's median cube error exceeds **0.5 cm** (S2's smallest
measured grasp-offset tolerance, toward the fixed jaw); otherwise **g4** (TASK-066's grid, fewer
tokens). The jaw readout is reported, not used by the rule. The choice and both numbers go into
the Stage 0 record.

### 10.4 G1's bars

A **calibration L** (debug seed 100900, the chosen grid, 20 000 updates on D0-fit, §5's recipe
otherwise) and a **calibration N_L** (debug seed 100911, the same windows in the same order as
calibration L, zero commands) are rolled out on D0-eval roots (§7.1's rule) at h = 8 and h = 16.
Then, as TASK-066: **B_rank = max(0.10, ½ × calibration L's effective-rank ratio)** and
**B_std = max(0.25, ½ × its std ratio)**, each the smaller over h ∈ {8, 16}, rounded down to
0.01, followed by the truncation check of §6, applied to the combined G1 including (iii) against
calibration N_L (as TASK-066's truncation table). Calibration L's G1 (iii), G2–G5 values on
D0-eval are disclosed in the Stage 0 record and move no bar. **G4:** if calibration L's lb of
sens_h on D0-eval is below 1.10 at h = 8 or h = 16, that is disclosed in the Stage 0 record, the
bar does not move by itself, and the owner (or Claude under delegation, in a recorded ruling) may
rule on G4 before the freeze; the inert positive wrist_flex commands at the limit and the absolute
gripper channel make such a miss plausible.

### 10.5 G5's bars

**B_ℓ = 1.5 × the median error of R-enc on D0-eval's encoded frames** (fitted on D0-fit), for
ℓ ∈ {cube, jaw}, at the chosen grid. The ceiling it is set from is the encoded-frame readout's
measured error (lesson 1); the factor 1.5 leaves room for prediction and comes from TASK-077's
decomposition, where a readout refitted on predicted latents read 1.36–1.41 × the encoded-frame
readout's error at h = 60 (here h ≤ 16). The bars are recorded in the Stage 0 record; if
calibration L's R-pred misses one on D0-eval, that is disclosed, the bar does not move, and the
owner may rule before the freeze.

**Disclosed mismatch.** B_ℓ comes from an R-enc fitted on D0-fit (180 episodes), while the gated
R-enc and R-pred are fitted on the fit sample (about 675 episodes). A readout fitted on less data
has a larger error, so B_ℓ is probably **looser** than 1.5 × the gated encoded-frame ceiling. The
gated R-enc's test median is reported beside B_ℓ, and so is the ratio of R-pred's test median to
it, so a reader can see how much of the 1.5 × the pass used.

### 10.6 Debug smoke and determinism

Every arm for a few hundred updates on D0 with debug seeds 100901–100910 (separate from the
calibration seeds 100900 and 100911 and the probe seed 100920); the readouts, the val
report and the evaluation path with val standing in for test (the row logic exercised; numbers
not results). CUDA determinism: one job trained twice gives identical weights (state SHA-256 both
times). Cost projection for F, B, T, T′ and E from the measured rates against §8.2's cap.

### 10.7 Freeze

The Stage 0 PR records §10.2–§10.6's numbers, the chosen grid, B_rank, B_std, B_cube, B_jaw and
the subset thresholds, and is independently reviewed. **The protocol freezes when it merges.**

## 11. Rows (first match) and what each leads to

1. **VOID** — a guard failed (corpus, shard, DINOv2-weight, feature-store, checkpoint
   implementation-hash or split check; test read before the val report's hash was recorded); a job
   failed twice; a stage exceeded its cap; a G5 subset had fewer than 100 test roots; or the
   evaluation did not complete. Nothing is read. One from-scratch repeat of the failed stage is
   allowed under new output names (`-r2`); a second VOID closes the task **INCONCLUSIVE**. A VOID
   for a G5 subset under 100 test roots is fixed by the test split, so it gets no repeat and closes
   the task **INCONCLUSIVE** at once (§7.4).
2. **WM-PASS** — some eligible arm X ∈ {L, J} passes G1–G6 on **all three seeds at h = 8 and
   h = 16**.
3. **WM-PARTIAL** — not WM-PASS, and some eligible arm passes **G1–G4 on all three seeds at h = 8**.
   The results name, per arm, which of G5, G6, h = 16 or which seeds failed.
4. **WM-FAIL** — otherwise.

What each row leads to (recommendations; the owner, or Claude under delegation, rules):

- **WM-PASS:** the passing arm is S5's model (L if both pass: it is the product-goal backend; J is
  reported). S5 needs its own preregistration: fresh reset seeds from the S2 task streams, the
  scripted follower as the reference arm, action-blind, scene-blind and random controls, bars set
  from measured ceilings, and §9's planning measure as its design input. A WM-PASS is an offline
  result: it is **not** a control result and makes no claim about a closed loop.
- **WM-PARTIAL:** escalate to a ruling. S5 may be admitted by that ruling with the named
  limitation (for example, an S5 design that does not read the cube from predictions if G5 failed,
  or a re-run with a longer budget if G6 failed), or a diagnosis may come first.
- **WM-FAIL:** escalate to a ruling. **No abandonment clause fires**: this is the first SO-101
  world-model study, on one corpus with a measured partial joint coverage and frequent clipped
  commands, so a failure does not justify closing a line. The one binding consequence: **no second
  training variant of this recipe on `so101-play-v1` is preregistered before a development
  decomposition record** (TASK-077's precedent; train and val only, no new training) names which
  gate failed and the declared reason. LeWM, `native_jepa`, DINOv2, the corpus, the SO-101 branch
  and the product goal are not closed by any row.
- **Every row:** R7 does not change (R28.9); nothing is re-thresholded, retrained, re-selected or
  re-chosen after the test numbers are seen; every gate quantity is reported with its interval
  per arm, seed and h, including those of rows further down.

## 12. Seeds, salts and guards

- **Model seeds** 100100–100102; **debug, calibration and probe seeds** 100900–100999; **salts**
  10011 (window sampler), 10012 (wrong commands), 10013 (candidates), 10014 (bootstrap), 10015
  (readout folds). None appears in `docs/`, `src/`, `scripts/`, `tests/`, `benchmarks/` or
  `configs/` (checked with `grep -w` on 2026-10-17). The fit sample, D0 and the roots are
  deterministic rules with no seed. The random-init floor uses `pretrained_encoder.random_init`'s
  fixed seed 0.
- **Guards (each a VOID):** the corpus and shard hashes (§2); the DINOv2 weight pins and the loaded
  digest; the tracked tree is clean and at the frozen revision at every stage's start and end; the
  feature stores re-hashed at every stage that reads them; checkpoints refused unless their
  implementation SHA-256 is the frozen code's; Stage V refuses anything but the 15 (or, after T′,
  extended) models of §5; Stage E refuses checkpoints other than the val report's; the test split
  is not decoded before Stage F′; every feature, loss, prediction and statistic finite; the disk
  floor.

## 13. What this cannot show

- **Offline only.** A ranking against the true future frame is not a planner, and readable
  predictions are not a controller; S5 is where control is tested, with a scripted reference arm.
- **One corpus, one collector, one scene.** A privileged scripted collector, one red cube on a tan
  table under fixed lighting (a model may lean on colour), partial joint coverage (wrist_flex
  positive, shoulder_pan within about ±0.88 rad outside wander), half the arm commands near zero,
  and 26 % clipped commands. Val and test are random episodes of the same distribution, not
  held-out conditions.
- **One encoder family, one floor seed, one projection, two declared grids, one architecture per
  backend, one budget rule, three model seeds.** A pass is evidence for this recipe on this data.
- **G5's bars are relative to the encoded-frame readout and have no absolute anchor** (§7.3), so
  a pass says the prediction keeps what the latent shows, within 1.5 ×; it does not say that what
  the latent shows is precise enough for a grasp (S2's envelope is reported beside it, §7.3, §9).
  The bar is set from a readout fitted on 180 episodes and is probably looser than 1.5 × the gated
  ceiling (§10.5).
- **G1 (iii) is computed in a 256-direction projection** of the latent, an approximation
  disclosed as in TASK-066 (§6).
- **R-pred is fitted on in-sample predictions** (§7.3), and **G5 (c) on the jaw is partly met by
  the schema**: X sees the absolute gripper command, which names the jaw's target, and N_X does
  not (§6).
- **G4's bar is an allocation** carried from the G1 line, not calibrated on this corpus; the
  inert positive wrist_flex commands at the limit and the absolute gripper channel make action
  sensitivity smaller than on the G1 corpora; calibration L's value is disclosed at Stage 0, and a
  miss there may be ruled on before the freeze (§10.4).
- **The random-init floor shares the recipe**; a difference between L and R is about the features,
  not about prediction alone.

## 14. Process

DRAFT (this PR, docs and card only) → independent review → merge → Stage 0 PR (code, tests,
D0 calibration, smoke, determinism; independent review) → **freeze at its merge** → the gated
stages of §8.4 in a clean run tree → results document
`docs/experiments/so101_world_model_v1_results.md` and a DECISIONS entry with the row → review →
merge. The executing agent reports rows; it does not choose S5.

## 15. Rulings this document records (R28.12–R28.26, DRAFT; decided by Claude under owner delegation)

- **R28.12** — TASK-100 is S4: offline only, on `so101-play-v1` as sealed; train fits, D0 (train
  shards 00–02) calibrates, val selects, test is gated and opened only after the val report's hash.
- **R28.13** — the latent: frozen DINOv2 ViT-S/14 tokens of both cameras at 224 px, grid g4 or g8
  (k = 96 projection) by §10.3's rule, train-only standardisation.
- **R28.14** — arms L, J (eligible), N_L, N_J, R and copy-last (§4); new files only; the G1
  byte-pinned files, `config.py` and the existing model files unchanged.
- **R28.15** — training as §5: 16-command windows, paired samplers, three model seeds per arm.
- **R28.16** — U by §8.1's probe rule; checkpoint selection on val at 1 % tolerance.
- **R28.17** — horizons h = 8 and h = 16 gated; 1, 4 and 32 reported (§6's justification).
- **R28.18** — G1: (i) carried; (ii) calibrated from calibration L's ratios on D0-eval with
  TASK-066's truncation check; (iii) X keeps more rank than N_X (lb of the rank-ratio difference
  > 0, 256-direction projected-basis bootstrap, disclosed as an approximation; TASK-066's owner
  Ruling 1, TASK-077's G1 (iii)); G2 and G3 definitional; G4 carried at 1.10 on wrong commands
  only, with calibration L's value disclosed and an owner ruling possible before the freeze.
- **R28.19** — G5 on the predicted-latent readout R-pred, with (a) B_ℓ = 1.5 × R-enc's median error
  on D0-eval (Stage 0 fits on D0-fit; the mismatch with the gated fit sample disclosed; no
  absolute anchor and no ceiling flag; the cube error reported against S2's 0.5 cm and 3 cm) and
  (b), (c) against the copy-last readout (fitted on fit-sample train roots) and N_X on moving
  subsets; R-enc on predictions reported (TASK-077's lesson).
- **R28.20** — G6: a 2 % last-quarter saturation criterion for X and N_X, one extension to 1.5 U,
  then decided; `last_two_triggered` a flag.
- **R28.21** — roots, tables, readouts and the cluster bootstrap as §7; 97.5 % gate intervals.
- **R28.22** — rows VOID, WM-PASS, WM-PARTIAL, WM-FAIL (first match) and their consequences as
  §11; a G5-subset VOID closes INCONCLUSIVE without a repeat; no abandonment clause; WM-FAIL
  requires a decomposition record before another variant.
- **R28.23** — §9's planning-usefulness measure and the encoder-floor reading are reported only.
- **R28.24** — budget: a 20 GPU h hard cap including Stage 0 (Stage 0 ≤ 4 GPU h), U lowered to fit
  and escalation below U = 20 000; every GPU job under `scripts/gpu_run.sh --wait --board`.
- **R28.25** — seeds, salts, guards, disk and evidence as §8.4 and §12.
- **R28.26** — R7 does not change, whatever the row; SO-101 results are not Apple→Plate results
  (R28.9).
