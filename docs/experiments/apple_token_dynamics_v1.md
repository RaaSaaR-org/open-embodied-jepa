# Apple→Plate token dynamics v1: does an action-conditioned predictor over frozen DINOv2 patch-token latents avoid rank collapse on `apple-look-v1`, and keep the apple readable through multi-step prediction? (TASK-066)

**Status: preregistration. No gated predictor has been trained on `apple-look-v1` token latents,
and no readout has been fitted on any predicted latent.** The only corpus-trained predictors are
the runner smoke's four 20-update models with noise targets (§14), which nothing reads. The
numbers here come from three kinds of source:
- published results of TASK-063/064/065;
- the latent-space calibration on a disjoint pilot (§6);
- the pre-freeze checks: the label-free anchor check (§14) and two owner-ruled readability checks
  of **encoded** 4 × 4 grids (§6.4). One ran on the pilot. The other ran on the corpus's 190
  train + val roots, and it is the only corpus readout fitted before the freeze.

**The test split of `apple-look-v1` is never decoded.
Cohorts C (45300–45339) and D (45000–45007, 45100–45107) are never simulated; nothing is
simulated at all. `exemption_spent` (`benchmarks/manifests/apple-policy-diagnostics-v1.json`)
stays `false`.**

This is a **world-model test only.** It trains an action-conditioned latent predictor and measures
it offline, on stored frames. It preregisters **no control formulation**, and nothing in it
implies one. CEM over the world-model cost was abandoned as the primary control line at TASK-054,
and the behaviour-cloning line stopped at TASK-057 (`docs/DECISIONS.md`). Both clauses still hold.
**Learned Apple→Plate is still 0 successes.** The corpus comes from the **privileged scripted
collector**, whose successes are scripted, not learned.

Predecessors:
- [`apple_latent_dynamics_v1.md`](apple_latent_dynamics_v1.md) /
  [`_results.md`](apple_latent_dynamics_v1_results.md) (TASK-065, outcome **WM-NO-DYNAMICS**; the
  harness, the gates and the clause this task builds on);
- [`apple_look_corpus_v1.md`](apple_look_corpus_v1.md) /
  [`_results.md`](apple_look_corpus_v1_results.md) (TASK-064, outcome **C-ACCEPT**; the corpus
  and its pilot);
- [`apple_pretrained_encoder_v1.md`](apple_pretrained_encoder_v1.md) /
  [`_results.md`](apple_pretrained_encoder_v1_results.md) (TASK-063, outcome **O-PT-POOLED**; the
  encoder and the P-tok read-out point).

Manifest: `benchmarks/manifests/apple-token-dynamics-v1.json`. Code that fixes the design:
`src/embodied_jepa/token_dynamics.py` (the latent, the calibration rules and results, the
calibrated G1 bars, the budget, the rows) and the model option in
`src/embodied_jepa/models/frozen_tokens.py`, with the calibration script
`scripts/calibrate_token_dynamics.py`. All are committed with this document and pinned by hash.

---

## 1. The question, and the claim it can make

TASK-065's results recommended this task: a preregistered patch-token latent predictor on
`apple-look-v1`, in the style of DINO-WM, with its budget and its rank bar calibrated before the
freeze. TASK-065's clause closed only the pooled-CLS predictor line; patch-token latents were
recorded as untested, not refuted.

**Question.** Trained on `apple-look-v1` train frames over the pinned frozen DINOv2 ViT-S/14
**final patch tokens**, average-pooled to a 4 × 4 grid, does an action-conditioned predictor:
- avoid rank collapse, against a bar **calibrated** on disjoint pilot data (§6);
- beat copying the last latent forward, and the same predictor trained without actions;
- depend on the actions, by a margin fixed now;
- and **keep the apple position readable through multi-step prediction**, by the TASK-063 probe
  fitted on encoded grids of train roots only?

**Claim scope, fixed now.** A pass (WM-TOK-DYNAMICS, §11) would say: on this corpus, at horizons
8 and 16 steps (0.4 s and 0.8 s at 20 fps), on train-split sessions each held out from the
predictor that predicts them, *this* predictor on *this* pooled frozen token grid learns
action-conditioned latent dynamics that do not collapse by the calibrated bar, beat both
baselines, and keep the post-look apple position readable to within the TASK-059 T1 bar and
0.5 cm of the encoded target. It would **not** say anything about control, a planner,
closed-loop behaviour, horizons beyond 16, other task phases as a readability claim, the test
split, another grid, or another encoder.

## 2. What is reused unchanged

| piece | source | how |
|---|---|---|
| corpus | `data/apple-look-v1`, manifest sha256 `81d760d1…db64` (TASK-064) | read-only; G-data |
| splits, halves, folds | TASK-064's 170 / 20 / 10 sessions; TASK-065's halves `26709b93…5061` and probe folds `44a3f267…86a7` | `latent_dynamics.session_halves`, `info_ceiling.fold_of`, unchanged; test never decoded |
| encoder | `facebook/dinov2-small` @ `ed25f3a3`, pinned files, digest `3a697b87…2af27` | `pretrained_encoder`, unchanged: 112 → 224 px bicubic, no crop, ImageNet normalisation, CPU float32; read-out point **`tokens`** (the 256 final patch tokens after the final LayerNorm) |
| frozen-encoder mixin | TASK-065's `FrozenEncoderMixin` | subclassed, not edited: its train-only normalisation, feature injection, `train_step_features` and `predict_features` run unchanged (§4.1) |
| predictor components | pinned upstream `le-wm` @ `8edfeb33`: `ARPredictor`, `Embedder`, `pred_proj` (`MLP` + BatchNorm), `JEPA.predict` | loaded through `models/lewm.load_upstream` (hash-checked), not edited; adapted to a token sequence (§4.2) |
| training step | the LeWM adapter's `train_step` (one-step + recursive multistep loss, SIGReg weight 0) | unchanged, on cached features |
| sampler | TASK-065's `SeedSequence([6500, seed, half])`, shared by W and N | unchanged: W and N here draw the windows TASK-065's models drew, in the same order |
| harness | TASK-065's runner `scripts/train_apple_latent_dynamics.py` | imported read-only for its reader (`TrainValReader`, Q-split), its guard helpers and its window/gather code; its bytes are pinned |
| statistics, gates G2–G5, baselines | `latent_dynamics`, `info_ceiling`, `observation_reprobe` | unchanged: cluster bootstrap (6501), cross-session shuffle (6502), paired bootstrap (5901), copy-last, no-action N, wrong and zero actions, G2 ≤ 0.8, G3 < 1.0, G4 ≥ 1.10, G5 (T1 bar and 0.5 cm) |
| guards | TASK-065 §9 including the amended G-cache (determinism check plus the 1e-3 anchor bound) | extended to the tokens (§10) |

## 3. Context carried forward (reported, not decisional)

### 3.1 TASK-065's result

Outcome **WM-NO-DYNAMICS**; its abandonment clause fired and closed *"an action-conditioned
LeWM-family predictor on frozen, externally pretrained pooled (CLS) latents on `apple-look-v1`"*.
The row name overstates the failure (its results document says so):
- **G2–G4 passed on all three seeds at h = 8 and 16.** W / copy-last 0.746 (h = 8) and
  0.647–0.652 (h = 16); W / N 0.879–0.910; wrong / W 1.78–1.98.
- **G1 failed on effective rank alone**: ratio 0.365–0.399 against an **uncalibrated** bar of 0.5
  (std ratio 0.81–0.85 and collapsed fraction 0 passed).
- **G5 failed**: W read the apple to 0.94–1.19 cm against 0.68–0.83 cm encoded; the 0.5 cm
  non-inferiority margin failed at h = 16 on every seed.
- **Checkpoints were selected late** (updates 8000–10 000 of 10 000), so the budget may not have
  saturated.

This task answers the two process lessons directly: **the G1 bars and the budget are calibrated
before the freeze (§6)**, on disjoint data and from latent-space quantities only.

### 3.2 What TASK-063/064 say about the token read-out point

On this corpus's 190 train + val post-look frames (TASK-064, reported rows):
- **P-tok** (all 256 final tokens, 98 304-d) read the apple to **0.456 cm [0.406, 0.528]**,
  183/190; on TASK-063's roots, 0.394 cm, 187/190.
- **P-mean** (the mean of the 256 tokens, i.e. a 1 × 1 pooled grid) read it to **0.589 cm
  [0.549, 0.655]**, 180/190.
- P-cls read it to 0.550 cm [0.478, 0.642]. R-tok (random-init tokens) met every bar too
  (0.572 cm).

The 4 × 4 grid of this task contains P-mean as a linear function (the mean of its 16 cells).
That is a published fact about the encoder, not a new readout. Before the freeze, by owner
ruling, the **encoded** 4 × 4 grid was read on the pilot and on the corpus's 190 train + val roots
(§6.4). It meets the T1 bar on the corpus: 0.504–0.607 cm. In the run, whether the encoded grid
meets the T1 bar is measured again (the ceiling, §9), and a failure has its own row
(WM-TOK-CEILING, §11).

### 3.3 Caveats carried forward, unchanged

- **Random-init features also read the apple** (R-tok met every bar). Readability is not evidence
  that pretraining matters.
- **One encoder, one input size, one floor seed.** The cause of readability is not identified.
- **The apple is static over every E-post horizon read so far** (TASK-065: 0 of 190 roots moved
  more than 1 mm by frame 8 + 64). E-post measures whether prediction keeps a static object
  readable while the arm moves; it does not measure object dynamics.
- **MPS training is not bit-reproducible.** The seeds and the budget are fixed, not the bits.
- **Render nondeterminism (TASK-064 §16).** This task renders nothing; its features are features
  of the accepted corpus bytes.
- **A deterministic float32 batch-size effect** (TASK-065 run-1): batched CPU inference of the
  same frame differs by up to 3.8e-5 between batch sizes. Every pre-freeze check here covers
  partial batches (§14).

## 4. The latent and the predictor (fixed now)

### 4.1 The latent: the final patch tokens, average-pooled to a 4 × 4 grid

A new option, `frozen_encoder: dinov2_small_tokens` with `token_grid: 4`, written once in a new
module, `models/frozen_tokens.py`. `frozen_token_model(backend)` returns the backend class with
`FrozenTokenMixin` in front of it; `FrozenTokenMixin` **subclasses TASK-065's
`FrozenEncoderMixin`**, so its train-only normalisation, its feature injection,
`train_step_features` and `predict_features` run unchanged. The plain backends and TASK-065's CLS
class reject the new key as unknown, and the token class refuses to run without it. **The option
is off unless a run selects it.**

- **The feature.** The 256 final patch tokens of the pinned DINOv2 (after the final LayerNorm,
  `last_hidden_state[:, 1:]`, the TASK-063/064 P-tok read-out point), reshaped to their 16 × 16
  grid and **averaged over each 4 × 4 block of patches** (`pool_tokens`, a float64 mean cast to
  float32). The latent is the 4 × 4 grid of 384-d tokens, **6144-d**, row-major over (grid row,
  grid column, channel), standardised per dimension by train-only moments (std floored at 1e-3).
  There is no projector: a prediction maps back to the raw pooled grid exactly, so the probe
  fitted on encoded grids reads predictions.
- **Why 4 × 4, fixed before the freeze** (compute, measured on synthetic inputs; §14):

  | grid | latent | feature cache (192 789 frames, float32) | LeWM update, batch 64 × 16, MPS | G1 covariance (float64) |
  |---|---|---|---|---|
  | 16 × 16 (all tokens) | 98 304 | 75.8 GB | not run | 77 GB |
  | 8 × 8 | 24 576 | 18.9 GB | 0.313 s | 4.8 GB each |
  | **4 × 4** | **6144** | **4.74 GB** | **0.095 s** | **0.30 GB each** |

  The full grid does not fit in memory on this machine (48 GiB). The 8 × 8 grid would put about
  19 GB of features in memory, cost 3.3× the training time per update and 16× the G1 covariance
  memory; with the calibrated budget the run would not fit a working day. The 4 × 4 grid keeps
  the spatial layout at 56 px cells of the 224 px input (28 px of the 112 px frame), and it
  contains P-mean, which already reads the apple on this corpus (§3.2).
- **The encoder never trains** (as in TASK-065): not a submodule, not in the optimizer, never
  moved to MPS, never in the checkpoint's weights; its digest is in the model metadata and a load
  under a different encoder is refused.
- **Nothing of the backend's own image pathway is built.** In TASK-065 the backend's encoder was
  built and unused; with a 6144-d latent, LeWM's ViT encoder at that width would be about 0.9 B
  parameters, so the token class builds only the envelope (`VisualModel`), the token predictor and
  (for LeWM) SIGReg.
- **No existing file changes.** `models/frozen_encoder.py`, `models/base.py`, `models/lewm.py`,
  `models/native.py`, `models/readout.py`, `readout_labels.py` and `pretrained_encoder.py` keep
  their bytes (tested against TASK-065's pins). So TASK-054's E0 checkpoint (guarded by
  `tests/test_policy_preflight.py`) and TASK-065's twelve checkpoints stay loadable.
  `implementation_sha256` of a token model covers the new module, `frozen_encoder.py`, the
  backend's file and `pretrained_encoder.py`.
- It refuses `state_fusion`, `readout_heads`, a `cameras` list and every TASK-054 prediction-step
  option (none is preregistered with it).
- Known gap, not fixed: `sigreg_projections` is type-checked but not required to be ≥ 1, as the
  plain LeWM adapter requires. The frozen configuration uses the default of 128. Fixing it would
  change the implementation hash, and the calibration checkpoints would no longer load.

### 4.2 The predictor: the pinned upstream LeWM predictor, adapted to a token sequence

**Choice: the pinned upstream LeWM predictor adapted to token sequences, not native code.**
- The product goal is LeWM on G1 + dual Dex3, so the predictor under test is the one the product
  would use.
- TASK-065 tested the same upstream predictor on the CLS latent. Keeping it makes the latent the
  change between the two tasks, rather than the latent and the predictor at once.
- DINO-WM's predictor is a ViT over the patch tokens of each frame; the upstream `ARPredictor` is
  an AdaLN-zero conditioned transformer, so a token sequence is a small adaptation of it, not a
  new model.
- Native code is one key away (`token_dynamics.BACKEND = "native_jepa"`: a residual token step, a
  shared MLP over each token, its position embedding, the frame's mean token and the action). It
  is tested but **not run**: a second backend would be a second arm, and it is not preregistered.

**What the token predictor is.** One prediction step maps the 16 tokens of one frame and that
frame's action to the 16 tokens of the next frame (history one, as in TASK-065):
- the upstream `Embedder` embeds the 14-D action to 384-d, **once per frame**;
- the upstream `ARPredictor` is built with `num_frames = 16`, so its learned position embedding
  has **one vector per token**, and its sequence axis is the token grid;
- every token of the frame is conditioned on the frame's action embedding through upstream's
  AdaLN-zero modulation (the embedding is broadcast over the tokens);
- the upstream `pred_proj` (MLP with BatchNorm) is applied per token, through the upstream
  `JEPA.predict`, which already flattens its sequence axis for it.

**The two adaptations, and why.** Both live in the new module; no upstream byte changes, and the
pinned source hashes still guard the upstream files.
1. **The sequence axis is the token grid, not time.** With history one there is no time axis;
   the grid is what the attention should mix.
2. **Attention is bidirectional within the frame.** Upstream's `Attention` masks causally over
   its sequence axis (it predicts time step t from steps ≤ t). Over a raster of tokens, that mask
   would let the top-left token see only itself, so it could not see the arm or the apple
   elsewhere in the frame. DINO-WM attends across all tokens of a frame. The upstream attention
   modules are used with their upstream weights and initialisation; only their mask argument is
   fixed to `causal=False`. The tests show both directions: with the adaptation a change in the
   last token moves the first token's prediction, and with the upstream mask restored it does
   not.

**Kept from TASK-065, not tuned here.** The adapter's local-budget capacity (depth 2, 2 heads ×
24, MLP 128, 384-d tokens: 2.84 M parameters), zero dropout, SIGReg weight 0 (the encoder is
frozen), the recursive multistep loss (weight 1.0), learning rate 3e-4 constant, weight decay
1e-4, gradient clip 1.0, batch 64 windows of 16 transitions. **Not adopted from DINO-WM**, and
disclosed: its larger predictor (depth 6, 16 heads), its history of 3 frames and its frame skip
of 5. Each would be a second change next to the latent.

### 4.3 Configuration (`token_dynamics.MODEL_CONFIG`)

`camera: onboard_rgb`, `frozen_encoder: dinov2_small_tokens`, `token_grid: 4`,
`latent_dim: 6144`, `sigreg_weight: 0.0`, `max_horizon: 64`. Everything else is the adapter's
default (as in TASK-065). The backend swap is one key (`token_dynamics.BACKEND`).

### 4.4 The arms and the baselines (TASK-065's, unchanged)

| name | what | trained |
|---|---|---|
| **W** | the token predictor, on the true executed actions | yes |
| **N** (no-action baseline) | the same model, same seed, same windows, **every action zero** in training and evaluation | yes |
| **copy-last** | the start latent, repeated for every step | no |

## 5. Data (fixed now; TASK-065's, unchanged)

- **Train and val episodes only** (679 + 80), through TASK-065's `TrainValReader`, which refuses
  every other episode id before a file is opened (Q-split). The test split's 40 episodes are never
  decoded.
- Frames, actions, `action_valid`; label sidecars (`privileged__apple_position_world`,
  `collector__phase_index`) for train + val roots only, as analysis targets and the look check.
  They are never model inputs, loss weights or selection filters.
- **Cross-fitting halves** A / B of the 170 train sessions (TASK-065's pin). Every model trains on
  one half; every evaluated train session is predicted by the model that never trained on it; the
  probe is cross-fitted by TASK-065's 10 folds over the 170 train roots.
- **Normalisation**: each model fits per-dimension moments of the raw pooled grid over every frame
  of its own training half, once, before its first update (computed in float64 in chunks). The
  **metric scale** is the per-dimension std over every frame of all 170 train sessions, floored at
  1e-3, identical for every model.
- **Features.** Every frame of every train and val episode is featurised **once**, CPU float32,
  6 threads, each episode in batches of 16 from frame 0 (the cache layout), into a cache under the
  run's output directory (sha256 recorded; never committed).

## 6. Calibration before the freeze (latent-space only, on disjoint data)

### 6.1 Leakage rules (binding, as given by the task owner)

- Calibration uses **only latent-space quantities**: training-loss curves, held-out criterion
  curves, and the effective rank and spread of encoded against predicted latents.
- It **never fits or evaluates an apple readout**, and it opens no label sidecar.
- It **never touches the test split** of `apple-look-v1`.
- It uses **data disjoint from the gate-evaluation roots**: TASK-064's pilot, below. No
  `apple-look-v1` episode is opened by the calibration. (The label-free anchor check of §14 reads
  corpus frames for features only; it computes no model quantity.)
- **Two disclosed exceptions, by owner ruling (§6.4).** Neither is part of the calibration of
  the budget or of G1. Both read **encoded** 4 × 4 grids only, never a prediction.
  - The pilot readability check reads the pilot's label sidecars
    (`privileged__apple_position_world`, `collector__phase_index`).
  - The corpus ceiling pre-check reads the same sidecars of the 190 train + val roots of
    `apple-look-v1`. These are gate-evaluation roots, read for the ceiling quantity only; test
    split excluded.

### 6.2 The pilot

`outputs/task064-scratch/pilot-d/dataset` (manifest sha256 `0a41d302…f50c`): TASK-064's pilot on
the separate reset range **47900–47931** (32 roots, 32 sessions, 127 episodes, 33 878 frames),
collected by TASK-064's committed collector and **never part of the corpus** (TASK-064 §13). It
is the kind of data the corpus is (the same collector, look prefix and camera). The 32 sessions,
sorted, are permuted by `default_rng(6600)`: the first 24 are **pilot-train**, the other 8
**pilot-held-out**. The pilot's own split labels are ignored; the pilot is not the corpus.

### 6.3 The procedure and the rules (fixed before the calibration ran)

- Pilot frames are featurised exactly as the gated run will (the same featurizer and cache
  layout); two whole pilot episodes whose last batch is partial are re-featurised and must be
  bit-identical.
- **Three calibration models** (`CAL_RUNS`): W seed 0, W seed 1, N seed 0, each trained on
  pilot-train with the frozen recipe (§4) for **30 000 updates** (`CAL_UPDATES`), with the
  held-out criterion (TASK-065's val criterion: normalised MSE of the recursive prediction over
  steps 1…16, windows at stride 4) every 500 updates, the training loss every 100, and G1's
  collapse statistics (h = 8 and 16, predicted against encoded, divided by the metric scale) at
  every 5000 updates and at the selected checkpoint. Batches come from
  `SeedSequence([6600, seed])`, disjoint from the gated runs' sampler. They run as three parallel
  MPS processes.
- **Budget rule** (`budget_rule`). For each model, `u_sat` is the first evaluation point whose
  held-out criterion is within 1 % of that run's minimum. The gated budget is
  `U = clamp(5000 × ceil(2 × max u_sat / 5000), 10 000, 30 000)`. If `2 × max u_sat` exceeds
  30 000, the calibration **escalates to the owner** and nothing is frozen. The factor 2 covers
  that a half (85 sessions) holds about 3.5 times the pilot-train data, which can move saturation
  later; the val-based checkpoint selection (§7.3) makes an over-long budget cost time, not
  validity.
- **G1 bar rule** (`bar_rule`). At each W calibration model's selected checkpoint, on the
  pilot-held-out windows, the effective-rank ratio and the std ratio of predicted against encoded
  latents at h = 8 and h = 16. The reference is the smallest of the four values (two seeds × two
  horizons). **The relative bar is half the reference, rounded down to the nearest 0.01.** If the
  rank reference is below 0.10 the calibration escalates to the owner (the recipe itself may be
  collapsing).
- **Why half of the recipe's own ratio.** A predictor that regresses to a conditional mean
  loses rank against its targets even when it is working; TASK-065's uncalibrated bar compared
  predictions with the targets directly and could not tell that apart from collapse. The reference
  here is the ratio the same recipe reaches on disjoint data of the same kind, so it already
  contains the regression-to-the-mean loss. **Keeping less than half of that is collapse.** The
  factor one half is TASK-065's own reading of "collapse" (halving the rank), now applied to a
  calibrated reference instead of to the raw targets.
- **The relative bar alone would be circular**, because it is set from the tested architecture's
  own pilot behaviour: a recipe that collapsed on the pilot would lower its own bar. The owner's
  ruling (§6.4) therefore adds two parts that do not come from W's pilot behaviour:
  - **Absolute floors, fixed before the calibration's results were seen.** The rank bar is never
    below **0.10** and the std bar never below **0.25**: `bar = max(floor, relative bar)`. What a
    collapsed predictor scores: the mean predictor 0; a rank-1 prediction about 1 / (encoded
    effective rank), so about 0.025 at the encoded effective rank of about 40 seen in the pilot
    smoke; a rank-4 prediction at most 0.10.
  - **A comparative part: W keeps more rank than the no-action model N.** The lower 95 % bound of
    (rank ratio W − rank ratio N) must be **above 0** (margin 0). N collapsed hardest in TASK-065
    (0.28–0.33 against W's 0.37–0.40). A W whose predictions are no more diverse than an
    action-blind model's fails.
    - *How it is computed, and the approximation.* A session bootstrap of the full 6144-d
      effective rank needs a 6144 × 6144 eigendecomposition per resample (about 13 s each), which
      is infeasible. So this part is computed in a fixed, model-free basis: the top 256 principal
      directions of the train-split encoded latents, fitted on training frames before any model
      is trained. Per-session moments of W, N and the encoded targets are resampled together
      (2000 session-clustered resamples, seed 6603). **It is an approximation of the full-width
      statistic, and is disclosed as one.** On the pilot the bootstrap has only 8 held-out
      sessions.
- **The bars must bind: synthetic collapse controls (owner condition).** On the pilot-held-out
  windows, W's predictions are truncated to their own top k principal directions (the mean
  kept), for k = 1, 2, 4, 8. **The combined G1 (every part, with the calibrated bars) must fail
  k = 1, 2 and 4.** If any passes, the bar is raised and that is disclosed; a gate that lets a
  truncated predictor through is not shipped. Where k = 8 lands is reported either way (§6.5).
- The collapsed-fraction part of G1 (≤ 0.05 of dimensions with std < 0.01) is an absolute check
  and is kept unchanged.
- Nothing from the calibration is read beyond these rules: the rules' inputs, the curves and the
  collapse statistics are reported (§6.5). **Disclosed:** the bars are set after seeing pilot
  behaviour of the same architecture. The calibration started at 03:24Z (revision `7fa8183`; its
  training subprocesses at 03:32Z) before the floors and the comparative part were added
  (revision `753c42c`, 03:43Z). Its training code did not change, and the floors and the
  comparative part were fixed before any calibration result was read.

### 6.4 Owner rulings before the freeze, and the pooling checks

Rulings are relayed by the coordinator and recorded at the UTC time they were received.

**Ruling 1, received about 03:35Z; condition received about 03:40Z (G1 design).** The first
design derived G1's bars from the tested architecture's own pilot ratios alone. The owner found
that circular and possibly lax, and ruled for:
- absolute floors (rank 0.10, std 0.25);
- the comparative W-over-N rank part (margin 0, projected 256-direction bootstrap, disclosed as an
  approximation);
- synthetic truncation controls that the combined G1 must fail at k = 1, 2, 4 (§6.3);
- an encoded-only 4 × 4 readability check on pilot-d, with its label read disclosed.

The rejected alternative: a fixed fraction of the encoded rank would repeat TASK-065's flaw,
because the conditional mean loses rank against its targets.

**The pilot-d pooling check (encoded latents only), and its rule.**
- The rule, fixed before it ran: reconsider the pooling if the 4 × 4 grid misses the TASK-059 T1
  bar at h = 8 or 16 while full P-tok meets it. If both miss, the pilot is too small to inform.
- 30 pilot roots were read (the pilot's 2 test roots have no recorded occlusion flag), with
  TASK-063's probe and 10-fold CV.
- The occlusion flags come from rendered pixels and are real. The pilot report's `apple_xy` are
  its smoke noise targets and were not used. A first attempt (`readability-1`) stopped on that
  before any readout.
- Report: `outputs/task066-calibration/readability-3/report.json`, revision `4b34ea2`.
  `readability-2` has the same numbers, without the per-root errors.

Reading the table: median T1 in cm, with the 95 % interval of the ratio to B-occ in brackets.
The T1 bar needs a median ≤ 1.5 cm and the upper end of that interval ≤ 0.6.

| h | 4 × 4 grid | full P-tok 16 × 16 | P-mean 1 × 1 |
|---|---|---|---|
| 0 (frame 8) | 1.069 [0.369, 0.711] | 0.848 [0.276, 0.664] | 1.320 [0.434, 0.875] |
| 8 (frame 16) | 1.061 [0.387, **0.644**] | 0.989 [0.302, 0.588] | 1.238 [0.397, 0.761] |
| 16 (frame 24) | 0.871 [0.242, **0.618**] | 0.815 [0.268, 0.540] | 1.038 [0.313, 0.709] |

- **The rule fired** ("reconsider pooling"): the 4 × 4 grid misses at h = 8 and 16 while P-tok
  meets the bar.
- **The pilot is too weak to decide the question.**
  - The paired median difference (4 × 4 − P-tok; reported only) is +0.071 cm [−0.061, 0.300] at
    h = 8 and +0.056 [−0.198, 0.317] at h = 16. Both intervals include 0.
  - P-mean, which the 4 × 4 grid contains linearly, fails every pilot bar. On the corpus's 190
    roots it passed with a ratio upper bound of 0.376 (TASK-064).
  - Full P-tok itself fails at h = 0.

**Ruling 2, received 2026-09-27T03:46Z (pooling).** Option A: keep 4 × 4, subject to one more
pre-freeze check that decides whether it stays. Neither an 8 × 8 grid (about 14 h more) nor a
channel-PCA 8 × 8 grid (unvalidated design). Its grounds:
- the rule fired correctly, but the pilot cannot decide the question;
- the informative check is available now.

**The corpus ceiling pre-check (owner-ruled; written here before it ran).**
- **What it reads.** The **encoded** latents only, of the 190 train + val roots of
  `apple-look-v1`, at frames 8, 16 and 24 (h = 0, 8, 16). The sources are the 4 × 4 grid, as the
  run's cache will compute it (each root's frames 0–31 in batches of 16), and full 16 × 16 P-tok
  as the reference. Targets are the apple labels at those frames, and B-occ uses TASK-064's
  reset-occlusion flags.
- **What it does not do.** No predictor, no predicted latent and no test-split episode.
- **The probe.** TASK-063's (linear + RBF kernel ridge, nested CV), with TASK-064's 10 folds over
  the 190 roots (`default_rng(59)`; the fold hash must equal TASK-064's `da6b5b5a…486b`).
- **Reported only.** The same numbers on the 170 train roots with the run's own folds
  (`44a3f267…86a7`), which is the run's exact ceiling quantity.
- **Decision rule** (the owner's):
  - If the 4 × 4 encoded grid meets the TASK-059 T1 bar (median ≤ 1.5 cm and ratio-to-B-occ upper
    bound ≤ 0.6) at h = 8 **and** h = 16 on the 190 roots, 4 × 4 stays and the protocol freezes.
  - If it misses at either horizon while full P-tok meets it, the agent stops and reports to the
    owner before freezing, and the owner chooses between the 8 × 8 options with these numbers.
  - If both miss, the agent reports to the owner.
  - Added by the agent: if the 170-root version disagrees with the 190-root verdict, the agent
    also reports before freezing.
- **Disclosure.** This check reads gate-evaluation roots, for encoded features only. Those
  features are the ceiling quantity. TASK-064 already read these roots at the frame-8 anchor
  (P-cls, P-tok). The check says nothing about the outcome: it cannot bias G1–G4, or the
  W-versus-encoded comparison in G5.

**The corpus ceiling pre-check: result.** Report `outputs/task066-calibration/ceiling-1/report.json`,
revision `96ed14c`, clean tree, 286 s.
- Both fold hashes matched: TASK-064's 190-root folds and the run's 170-root folds.
- Frame-0 apple labels matched TASK-064's reset truth within 1.5e-8 m.
- No predictor was built, and `test_split_decoded` is false.

Reading the table: median T1 in cm, with its 95 % interval in the first brackets and the 95 %
interval of the ratio to B-occ in the second. B-occ's median is 1.866 cm on the 190 roots and
1.859 cm on the 170. Every entry here meets the T1 bar.

| roots | h | 4 × 4 grid | full P-tok 16 × 16 | 4 × 4 − P-tok (paired median difference, cm) |
|---|---|---|---|---|
| 190 (decisional) | 0 | 0.504 [0.453, 0.581], [0.237, 0.331] | 0.456 [0.406, 0.528], [0.213, 0.300] | +0.049 [−0.003, 0.111] |
| 190 (decisional) | 8 | 0.607 [0.542, 0.655], [0.287, 0.381] | 0.517 [0.451, 0.611], [0.242, 0.336] | +0.090 [0.009, 0.137] |
| 190 (decisional) | 16 | 0.492 [0.424, 0.564], [0.222, 0.320] | 0.456 [0.397, 0.501], [0.211, 0.291] | +0.035 [−0.025, 0.097] |
| 170 train (reported; the run's ceiling) | 8 | 0.632 [0.559, 0.703], [0.290, 0.400] | 0.572 [0.483, 0.643], [0.252, 0.366] | +0.059 [−0.011, 0.134] |
| 170 train (reported; the run's ceiling) | 16 | 0.503 [0.443, 0.571], [0.232, 0.329] | 0.439 [0.381, 0.523], [0.200, 0.291] | +0.064 [0.004, 0.118] |

- **Verdict: keep 4 × 4 and freeze.** The 190-root and 170-root verdicts agree.
- Full P-tok at h = 0 on the 190 roots reproduces TASK-064's published 0.456 cm
  [0.406, 0.528].
- The pooling costs 0.04–0.09 cm of readability, detectable at some horizons. That loss is part
  of the latent this task tests, and it is disclosed.
- **Consequence for G5, stated before any model is trained.** The encoded 4 × 4 ceiling reads
  about 0.50–0.63 cm at the gated horizons.

### 6.5 Results (recorded; report `outputs/task066-calibration/run-1/report.json`)

**The run.**
- Report sha256 `a1d9fc17…53e1`, revision `7fa8183`, clean tree, 7283 s. Outcome `calibrated`.
- Pilot: 127 episodes, 33 878 frames, 24 / 8 sessions, 95 / 32 episodes by part.
- Pilot-train gave 23 485 training windows; pilot-held-out gave 2101 windows at stride 4.
- Featurisation took 14.3 ms per frame.
- Two whole pilot episodes re-featurised bit-identically: `look-47900`, whose final batch is 14,
  and `look-47900-b0-noise_only`, whose final batch is 4.
- The models ran at about 0.225 s per update each, as three parallel MPS processes.
- No label was read, no apple readout was fitted, and no corpus episode was opened.

**Held-out criterion curves**, sampled every 2500 of the 500-update points; all points are in the
report.

| model | 2500 | 5000 | 7500 | 10 000 | 15 000 | 20 000 | 25 000 | 30 000 | selected (criterion) | u_sat |
|---|---|---|---|---|---|---|---|---|---|---|
| W s0 | 0.405 | 0.388 | 0.380 | 0.375 | 0.380 | 0.384 | 0.386 | 0.388 | 9500 (0.3749) | 6500 |
| W s1 | 0.402 | 0.381 | 0.378 | 0.374 | 0.376 | 0.380 | 0.389 | 0.397 | 9000 (0.3709) | 7000 |
| N s0 | 0.442 | 0.423 | 0.405 | 0.401 | 0.414 | 0.412 | 0.407 | 0.409 | 28 000 (0.3942) | 16 500 |

- W saturates by about 7000 updates on the pilot, and it overfits the 24 pilot-train sessions
  after about 10 000.
- N's curve is flat and noisy from 10 000 on, between 0.394 and 0.414. Its minimum is 0.3942 at
  28 000, and five points from 10 000 on lie below 0.400: 11 500, 16 500, 18 500, 27 000 and
  28 000. Its 1 % saturation point, 16 500, reflects that noise.
- **The budget rule escalates.** 2 × max u_sat = 33 000 > 30 000, so the rule asks for the
  owner (Ruling 3, below).

**G1 quantities at the selected checkpoints** (pilot-held-out windows, predicted against encoded,
full width).

| model (selected update) | h | effective-rank ratio | std ratio | predicted / encoded effective rank |
|---|---|---|---|---|
| W s0 (9500) | 8 | 0.336 | 0.814 | 13.5 / 40.3 |
| W s0 (9500) | 16 | **0.332** | 0.819 | 13.3 / 40.2 |
| W s1 (9000) | 8 | 0.340 | **0.796** | 13.7 / 40.3 |
| W s1 (9000) | 16 | 0.337 | 0.803 | 13.5 / 40.2 |
| N s0 (28 000), reported | 8 | 0.294 | 0.764 | 11.8 / 40.3 |
| N s0 (28 000), reported | 16 | 0.263 | 0.771 | 10.6 / 40.2 |

- The collapsed fraction was 0 everywhere. Copy-last's rank ratio was 1.006 at h = 8 and 1.010 at
  h = 16.
- **The G1 references and bars.**
  - The rank reference is **0.3315**, from W s0's val-selected checkpoint (update 9500) at h = 16:
    `checkpoints/task066-calibration/run-1/W-s0.pt`, sha256 `5dbf18a9…3f81`.
    The relative bar is 0.16, so **B_rank = max(0.10, 0.16) = 0.16**.
  - The std reference is **0.7963**, from W s1's val-selected checkpoint (update 9000) at h = 8:
    `checkpoints/task066-calibration/run-1/W-s1.pt`, sha256 `cd7ccec5…a7d6`. The N reference is
    `N-s0.pt`, sha256 `e31b3d4a…0213`. Each checkpoint holds the selected update's weights, while
    its `updates` field records 30 000 (the final count), as in TASK-065.
    The relative bar is 0.39, so **B_std = max(0.25, 0.39) = 0.39**.
  - Neither escalates: the rank reference is well above 0.10.
- **W's rank ratio rises with training while its held-out criterion worsens.** Along W s0's
  curve it is 0.31 at 5000, 0.34 at 10 000, 0.38 at 20 000 and 0.39 at 30 000 (h = 8). W s1
  behaves alike. N's rises too, from 0.27 to 0.32.
  - So the budget and the checkpoint move G1.
  - **G1 is read at the val-selected checkpoint, in the calibration and in the run. This is
    fixed now.** The references above are the val-selected checkpoints'.

**The truncation controls** (`outputs/task066-calibration/controls-1/report.json`, revision
`2c32bdd`; see the disclosure below). They use the bars above, and the W − N comparative against N
s0 on the 8 held-out sessions.

| predictor (both W seeds, h = 8 and 16) | effective-rank ratio | std ratio | W − N (point; lower 95 % bound for the untruncated row) | combined G1 |
|---|---|---|---|---|
| untruncated W | 0.332–0.340 | 0.796–0.819 | +0.033 to +0.062 (0.029 to 0.059) | **passes** every part |
| k = 1 | 0.025 | 0.375–0.395 | −0.28 to −0.31 | **fails** (rank, W − N; std too at 3 of 4) |
| k = 2 | 0.047–0.048 | 0.515–0.532 | −0.25 to −0.28 | **fails** (rank, W − N) |
| k = 4 | 0.085–0.086 | 0.630–0.652 | −0.20 to −0.22 | **fails** (rank, W − N) |
| k = 8 | 0.139–0.141 | 0.710–0.731 | −0.12 to −0.15 | **fails** (rank, W − N) |

- **The owner's condition is met.** The combined G1 fails k = 1, 2 and 4, so no bar was raised.
  k = 8 fails too.
- The rank bar alone (0.16) already fails every control. The comparative part fails each of them
  independently.
- **Disclosed:** controls-1 ran on a tree that was dirty only in the task card (Markdown; no code).

### 6.6 What the calibration fixes

**Ruling 3, received 2026-09-27T05:36Z (budget escalation).** Option A.
- **U = 30 000 updates** (the ceiling), with **selection on val every 1500 updates** (20 points).
- **Caps:** 4500 s per model, 57 600 s (16 h) for the whole run.
- The owner's grounds:
  - an under-trained N would bias both G3 and G1 (iii) towards W, so N reaching its plateau is a
    validity requirement;
  - selection on val protects W from overfitting;
  - each corpus half has about 3.5× the pilot-train data, so its saturation may come later;
  - about 12 h is acceptable.

**Frozen values** (`token_dynamics`, tested against the rules in `tests/test_token_dynamics.py`):

| quantity | value | source |
|---|---|---|
| updates per model | 30 000 | budget rule escalated (wanted 33 000); Ruling 3 |
| selection | every 1500 updates (20 points), lowest val criterion, ties to the earlier | Ruling 3; TASK-065's rule |
| G1 rank bar B_rank | 0.16 | reference 0.3315 (W s0 at 9500, h = 16); floor 0.10 |
| G1 std bar B_std | 0.39 | reference 0.7963 (W s1 at 9000, h = 8); floor 0.25 |
| G1 comparative | lower 95 % bound of (rank ratio W − rank ratio N) > 0 | Ruling 1 |
| G1 collapsed fraction | ≤ 0.05 | TASK-065, unchanged |
| per-model cap | 4500 s | Ruling 3 (expected about 3100 s: 30 000 × 0.095 s plus selection and evaluation) |
| global cap | 57 600 s | Ruling 3 (expected about 12 h) |

## 7. Training (fixed now)

### 7.1 Runs

2 arms (W, N) × 3 model seeds (0, 1, 2) × 2 halves (A, B) = **12 models**. Each model:
- is constructed with `frozen_token_model("leworldmodel")(state_schema, device="mps", seed=s,
  config=MODEL_CONFIG)`;
- fits its input normalisation on its half's frames;
- takes **30 000 updates** (§6.6) of batch 64 windows of 16 transitions (17 frames), through
  `train_step_features`;
- samples windows with TASK-065's stream `SeedSequence([6500, s, half index])`, shared by W and N;
- runs at a constant learning rate.

### 7.2 Device and determinism

MPS for training and prediction; CPU for features and every statistic. MPS training is not
bit-reproducible run to run; this is disclosed, not a guard.

### 7.3 Model selection, on val only

After every 1500 updates (20 points), the **val criterion** (TASK-065's) is computed over
the windows of all 80 val episodes starting at every 4th frame (for N, with zero actions). The
checkpoint with the lowest criterion is kept (ties to the earlier one) and evaluated. **Reported
only:** whether a model selected one of its last two points (a saturation diagnostic, not a gate).

## 8. Evaluation sets and metrics (fixed now; TASK-065's, unchanged)

- **E-all**: every window of 16 transitions starting at every 4th frame of every train episode,
  predicted by the half model that did not train on it. Per window, the normalised MSE (the mean
  over the 6144 dimensions of ((prediction − target) / metric scale)²) at h ∈ {1, 2, 4, 8, 16},
  for W (true, wrong and zero actions), N and copy-last. Ratios of summed errors with a
  session-clustered bootstrap (10 000 resamples of the 170 sessions, seed 6501).
- **G1 on E-all** pools both halves' W predictions at h = 8 and 16 (divided by the metric scale)
  against the encoded targets of the same windows. The statistics are computed from streamed
  first and second moments (`token_dynamics.Moments`), which equal TASK-065's
  `collapse_statistics` on the concatenated rows (tested), without holding 40 000 × 6144 arrays.
- **E-post**: the 170 train roots from the post-look frame 8, rolled forward with their own
  executed actions by the half model that did not train on them; the apple's xy at frame 8 + h
  read by TASK-063's probe (linear + RBF kernel ridge, nested CV) fitted on the **encoded** grids of
  frame 8 + h of train roots, 10-fold cross-fitted; TASK-059's T1 quantities with B-occ.
- **Reported only** (none decides anything; TASK-065 §7.3, plus): every E-all ratio at h = 1, 2, 4
  and on the E-post roots; E-post at h = 32 and 64; the literal frame-8 probe; the val-root
  secondary estimate; collapse statistics of N and copy-last; the val curves and losses; apple
  movement; and, new here, **the encoded grid's own T1 numbers at every h** and a side-by-side
  with TASK-065's CLS numbers (cm only: latent MSE is not comparable across latents).

## 9. Gates (fixed now)

Each gate is computed **per model seed** (W_A and W_B of that seed together), at **each of h = 8
and h = 16**; a gate passes for a seed only if it passes at both.

| gate | set | passes when (at h = 8 and h = 16) |
|---|---|---|
| **G1 no collapse** | E-all | (i) collapsed fraction ≤ 0.05; **and** (ii) effective-rank ratio ≥ **0.16** and std ratio ≥ **0.39** (calibrated, never below 0.10 and 0.25; full-width statistic, both halves pooled); **and** (iii) lower 95 % bound of (rank ratio W − rank ratio N) **> 0** (projected 256-direction basis, 2000 session-clustered resamples, seed 6603, no undefined resample) |
| **G2 beats copy-last** | E-all | upper 95 % bound of MSE(W) / MSE(copy-last) **≤ 0.8** |
| **G3 beats no-action** | E-all | upper 95 % bound of MSE(W) / MSE(N) **< 1.0** |
| **G4 action sensitivity** | E-all | lower 95 % bounds of MSE(W, wrong) / MSE(W, true) and MSE(W, zero) / MSE(W, true) **≥ 1.10**, with no undefined resample |
| **G5 the apple stays readable** | E-post | W's predicted grid under the frame-(8 + h) probe: median T1 **≤ 1.5 cm**, upper bound of its ratio to B-occ **≤ 0.6**, and upper bound of (median W − median encoded frame 8 + h) **≤ 0.5 cm** |

**The ceiling condition (not a gate; it selects a row).** In the run, on the 170 train roots
with the run's folds, the encoded 4 × 4 grid of frame 8 + h, read by the cross-fitted
frame-(8 + h) probe, meets the TASK-059 T1 bar (median ≤ 1.5 cm **and** upper bound of the
ratio to B-occ ≤ 0.6) at h = 8 **and** at h = 16 (`token_dynamics.ceiling_passes`). If it misses
at either horizon, G5 cannot test prediction on this latent (row WM-TOK-CEILING). The pre-check
(§6.4) read 0.632 and 0.503 cm, with upper bounds 0.400 and 0.329, on the same quantity.

**Why these thresholds.**
- **G1's bars are calibrated** (§6). This is the one change of substance to TASK-065's gates.
- **G2–G4 are TASK-065's**, unchanged. They are tests of the latent error against baselines that
  need no learned dynamics; TASK-065 found them informative.
- **G5 is TASK-065's**, unchanged, including its 0.5 cm margin. That margin was not calibrated in
  TASK-065 and **it cannot be calibrated here**: calibrating it would need an apple readout, which
  the leakage rules forbid before the freeze. It is kept for comparability with TASK-065 and
  disclosed as uncalibrated.

**Per seed:** the seed passes if G1–G5 all pass. Its **dynamics** hold if G1–G4 pass; its
**latent gates** hold if G2–G4 pass.

## 10. Guards: any failure voids the run (outcome V)

| guard | condition |
|---|---|
| **G-hash** | every file in the manifest's `hashes` matches at preflight (recorded then) and after the last stage; the tracked tree is clean at both |
| **G-frozen** | `token_dynamics.require_frozen()`: the calibrated bars, budget and caps are set |
| **G-data** | the corpus opens (every recorded hash verified), manifest `81d760d1…db64`; 170 / 20 / 10 sessions, 679 / 80 / 40 episodes |
| **Q-split** | only train and val episodes are decoded, through TASK-065's reader; `test_split_decoded: false` is recorded |
| **G-split** | the halves hash to TASK-065's pin; each model trains on exactly its half; normalisation episodes = training episodes |
| **G-labels** | every read root's first 8 phase labels are the look; its frame-0 apple label equals TASK-064's reset truth within 1e-6 m |
| **G-weights** | the DINOv2 files match their pins; the loaded digest is `3a697b87…2af27` |
| **G-anchor** | the 190 post-look frames hash to TASK-064's `20b67967…3612`; featurised as TASK-064 did (its root order, batches of 16), their CLS hashes to `P_cls` `24bc50f5…feec` **and their tokens to `P_tok` `d702fa64…f424`** |
| **G-repro** | a second featurisation of the 190 post-look frames is bit-identical (CLS and tokens) |
| **G-cache** (TASK-065 amended, extended) | (a) a second featurisation of the 190 post-look frames in the cache's own layout equals the cache rows bit for bit; (b) **two whole train episodes whose last batch is partial, re-featurised, equal their cache rows bit for bit**; (c) the pooled cache rows of the 190 roots are within **1e-3** absolute of the pooled anchor rows (the actual max absolute and relative difference and the differing roots are reported) |
| **G-folds** | the probe fold hash is `44a3f267…86a7` |
| **G-finite** | every feature, loss, prediction and statistic is finite |
| **G-cap** | featurisation ≤ 5400 s; each model (30 000 updates, 20 val evaluations and its evaluation) ≤ 4500 s; the whole run ≤ 57 600 s |
| **G-device** | MPS available and used for every model; CPU for features and statistics |

Any other exception is a crash, and a crash is V; `report.json` is written with outcome V and a
`void_reason` whatever stage failed, preflight included. PR 2's tests exercise every guard, every
gate and every row in both directions, the reader's refusal of a test episode, a crash at
preflight and mid-run, and the partial-batch determinism check.

## 11. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§10), or the run stops before a complete report | nothing is read | one from-scratch repeat (§13) |
| **WM-TOK-DYNAMICS** | every seed (0, 1, 2) passes G1–G5 | On `apple-look-v1`, a LeWM predictor on frozen, pooled DINOv2 patch tokens learns action-conditioned dynamics that do not collapse by the calibrated bar, beat copy-last and a no-action predictor, use the actions, and keep the post-look apple readable through 16-step prediction. | A separately preregistered **held-out confirmation on the corpus's test split** (never decoded until then), and/or a harder world-model test (later phases, longer horizons, a moving apple). **No control formulation is implied**; control needs its own preregistration and must answer the TASK-054 and TASK-057 clauses. |
| **WM-TOK-UNSTABLE** | one or two seeds pass | The result does not hold across seeds. | **The clause does not fire.** The owner decides whether a disclosed amendment adds seeds (counted with these three, rule unchanged, reported as exploratory). |
| **WM-TOK-CEILING** | no seed passes; G1–G4 pass on at least two seeds; the encoded grid misses the T1 bar at h = 8 or 16 | The predictor learns dynamics, but the pooled grid itself does not carry the apple to the T1 bar, so G5 did not test prediction. | **The clause does not fire.** The pooling, not prediction, is the finding; the owner decides (a finer grid needs its own compute plan). |
| **WM-TOK-APPLE-LOST** | no seed passes; G1–G4 pass on at least two seeds | The predictor learns dynamics, the encoded grid carries the apple, and prediction loses it. | **The clause fires.** |
| **WM-TOK-COLLAPSE** | no seed passes; G2–G4 pass on at least two seeds | The latent-error gates pass, but the predictions collapse against the calibrated bar. | **The clause fires.** |
| **WM-TOK-NO-DYNAMICS** | otherwise | The predictor does not beat the baselines or does not use the actions. | **The clause fires.** |

The row names are chosen so that a rank-only failure is not called "no dynamics" (TASK-065's
lesson): WM-TOK-COLLAPSE is the row when G2–G4 pass and G1 fails.

**INCONCLUSIVE (task-level, not a row of a run).** A second V closes TASK-066 as INCONCLUSIVE
(§13). Nothing is read, the clause does not fire, and the owner decides what follows.

**Abandonment clause (WM-TOK-APPLE-LOST, WM-TOK-COLLAPSE, WM-TOK-NO-DYNAMICS).**
- **What closes:** the line *"an action-conditioned LeWM-family predictor on frozen, externally
  pretrained DINOv2 patch-token latents on `apple-look-v1`"*. No further predictor variant on this
  corpus with frozen pretrained patch-token latents is preregistered without new evidence of a
  different kind. That covers the pooling grid and resolution, history length, attention masking,
  action chunking, step embeddings, loss weighting, capacity, residual parameterisation and
  budget. Together with TASK-065's clause, **the frozen-pretrained-DINOv2-latent predictor line on
  this corpus is then closed** at both read-out points.
- **What does not close:** the LeWM backend, the encoder as a component, the product goal, and the
  corpus (kept, sealed; its test split still unread).
- **Recorded as untested, not refuted:** a trainable encoder initialised from the pretrained one
  (fine-tuning); other encoders; another observation (the fixed `overview` camera of TASK-061,
  0.183 cm [0.168, 0.207], which is a hardware/workspace decision and the owner's).

**Always reported, whatever the row:** every gate quantity with its interval, per seed and horizon;
the ceiling numbers; every reported-only quantity (§8); every model's val curve, selected update
and saturation diagnostic; the rows that also match further down.

**In every outcome:** nothing is re-thresholded, retrained, re-selected or re-chosen after the
numbers are seen; learned Apple→Plate stays at 0 successes and `exemption_spent` stays `false`;
the executing agent recommends a next task and does not choose it.

**Stated in advance.**
- **Every failing row is a live possibility.** TASK-054 and TASK-065 history is of predictors that
  either did not beat persistence or lost rank and the apple.
- **G5 can pass while G2–G4 fail** (the apple is static on E-post, so a predictor that kept its
  input would read it). G5 alone decides nothing.
- **A pass is 170 cross-fitted train sessions and one corpus**, not a test-split result.
- **Random-init tokens also read the apple** (TASK-064), and no random-init or raw-pixel predictor
  is run: a pass would not show that pretraining matters for dynamics.
- **G1 (iii) alone can decide WM-TOK-COLLAPSE.** If only the comparative part fails (W no more
  diverse than N), the row is still WM-TOK-COLLAPSE. The results document states which G1 part
  failed.
  - The rank ratio rises with training (§6.5). So a W selected early and an N selected late
    handicap W on (iii). The selected updates are reported.
- **The G1 bars come from a 32-root pilot** (24 sessions of training data). A gated model trains on
  85 sessions; its ratios may differ from the pilot's for that reason alone. The bar is half the
  pilot reference to leave room for that.

## 12. Budget, device, seeds, recording

- **Device:** MPS for the 12 models; CPU (6 threads, float32) for features and every statistic.
- **Caps:** featurisation 5400 s; each model 4500 s; the whole run 57 600 s (Ruling 3).
  Expected: featurisation about 45 min (192 789 frames at about 14 ms), each model about
  52 min (30 000 × 0.095 s plus selection and evaluation), statistics about 30 min; about
  12 h in total.
- **Seeds:** model seeds 0, 1, 2; halves 65; sampler `SeedSequence([6500, seed, half])`; cluster
  bootstrap 6501; cross-session shuffle 6502; probe folds 59, inner 5900 + k, bootstrap 5901.
  Calibration: pilot partition 6600, calibration sampler `SeedSequence([6600, seed])`. Nothing
  else is random.
- **Memory.** Not a guard. The feature cache is 4.74 GB. Per seed, the G1 moments are eight
  6144 × 6144 float64 accumulators of 0.30 GB each, plus the encoded references and the
  projected per-session moments. The runner smoke on a subset peaked at 6.98 GB RSS. The
  expected peak is about 12–20 GB on a 48 GiB machine; TASK-065 peaked at 6.38 GB. The report
  records the peak.
- **Recorded** (`outputs/task066-token-dynamics/run-<k>/report.json`): as TASK-065 §11, plus the
  token anchor, the partial-batch determinism facts, the encoded-grid ceiling and the calibration
  report's sha256.
- **Output:** `outputs/task066-token-dynamics/run-1/`, checkpoints
  `checkpoints/task066-token-dynamics/run-1/`. The runner refuses to overwrite either. Nothing
  under `data/` is written. Checkpoints and features are never committed.

**Frozen run command** (from a clean checkout of the reviewed commit, after the pre-run GO):

```sh
uv run --no-sync python scripts/train_apple_token_dynamics.py run \
    --output outputs/task066-token-dynamics/run-1 \
    --checkpoints checkpoints/task066-token-dynamics/run-1
```

## 13. Void rule

- A run that stops early for any reason (a guard, a crash, a cap) is **V**, with a `void_reason`.
  Nothing in it is read.
- **Exactly one from-scratch repeat** is allowed, into `run-2`: the same seeds, caps and device, a
  clean tree, the features recomputed. A void run's directories are kept as evidence.
- **A second void closes TASK-066 as INCONCLUSIVE.**
- **Any guard change needs an owner ruling.** The executing agent asks and does not decide. A fix
  between runs is otherwise limited to runner mechanics, goes through a reviewed PR and a fresh
  pre-run GO, and is disclosed.
- **Every ruling is recorded with a UTC timestamp** in the task card before the run it affects.

## 14. Pre-freeze facts (disclosed; they fit nothing on the corpus)

**The anchor holds, for all 190 roots, partial batches included.**
- Report `outputs/task066-calibration/anchor-1/report.json`: revision `f78257c`, clean tree,
  183 s, features only.
- The 190 post-look frames, featurised as TASK-064 did (its root order, batches of 16), hash to
  TASK-064's recorded values:
  - the frames to `20b67967…3612`;
  - the CLS to `P_cls` `24bc50f5…feec`;
  - **the tokens to `P_tok` `d702fa64…f424`**.
  A second featurisation is bit-identical.
- **Pooled cache rows against pooled anchor rows, all 190 roots.**
  - Max absolute difference 2.93e-5; max relative to the row norm 3.6e-6.
  - The differing roots are exactly the 14 that formed TASK-064's partial final anchor batch,
    47186–47199. That is TASK-065 run-1's batch-size effect, as expected, and 34× under the
    1e-3 bound.
  - The cache layout is itself bit-identical on a second featurisation.
- **Whole episodes with partial final batches** (`look-47000`, 510 frames, final batch 14;
  `look-47001`, 529 frames, final batch 1) re-featurise bit-identically. The runner's G-cache (b)
  repeats this check in the run.

**Timing and memory (synthetic inputs, MPS, batch 64 × 16).**
- A token update takes 0.095 s at 4 × 4 (0.281 s on CPU) and 0.313 s at 8 × 8.
- Prediction of 2048 windows × 16 steps takes 0.85 s.
- Three concurrent MPS processes run at about 0.19–0.22 s per update each.
- Featurisation takes about 14.3 ms per frame on 6 CPU threads (calibration run-1, pilot, 33 878
  frames).
- One 6144 × 6144 eigendecomposition takes about 13 s (float64).

- **Smoke runs before the freeze (mechanics only; nothing in them is read).**
  - `cal-smoke-a` (03:10Z–03:24Z) ran 1000 updates of the calibration models on the pilot. §6.3
    quotes one number from it: the encoded effective rank of about 40.
  - The runner's `smoke-a` (03:52Z–04:04Z, revision `96ed14c`, before the freeze) trained four
    token predictors for 20 updates each on 64 corpus train episodes (32 per half, 8 sessions
    each), with 8 val episodes for selection. That is 72 episodes read by the model stages.
    - It decoded 244 train + val episodes in all: the anchor stage decodes every one of the 190
      roots.
    - It did not decode the test split.
    - On its 18 roots it read the phase labels and the apple labels at frames 0 and 8–72, as
      G-labels and the E-post targets do.
    - The probes' targets were replaced by seeded noise. The one real-label quantity it reported
      is `apple_moved_more_than_1mm_roots`, which was 0 at every h. TASK-065 had already
      disclosed that the apple stays static until frame ≥ 117.
  - Any later smoke run (PR 2) is disclosed with it.
  - These are the only predictors trained on the corpus before the gated run. No quantity from
    them informed any choice.
- **Labels read before the freeze.** Only the apple and phase labels were read. Beyond the
  smokes' G-labels checks above, all reads were owner-ruled (§6.4), for encoded-grid readability
  checks:
  - the pilot's 30 roots at frames 0–24;
  - the corpus's 190 train + val roots at frames 0–24 (the ceiling pre-check).
  TASK-065's disclosed inspection of 40 train roots' labels (the apple static until frame ≥ 117)
  is carried forward, not repeated.

## 15. Not done (declared)

- No control formulation, planner or controller; no closed-loop attempt; no simulation at all.
- The test split is not decoded; cohorts C and D are not simulated.
- No encoder is trained or fine-tuned. No second grid, read-out point, encoder or input size; no
  raw-pixel or random-init predictor; no native-backend arm; no DINO-WM capacity, history or frame
  skip.
- No change to any existing model file, `pretrained_encoder.py`, `frozen_encoder.py`,
  `info_ceiling.py`, `observation_reprobe.py`, `latent_dynamics.py`, TASK-065's runner, the corpus,
  any earlier checkpoint or any earlier protocol.

## 16. Process

1. **PR 1:** this document, the manifest, `token_dynamics.py`, the `frozen_tokens` option (with
   `docs/MODELS.md`), the calibration script and its recorded results, the tests and the task card.
   It merges on an independent reviewer's **reported** APPROVE and green CI, before any predictor
   is trained on the corpus.
2. **PR 2:** the runner (`scripts/train_apple_token_dynamics.py`), its guard tests and its pin in
   the manifest. It merges on a reported APPROVE and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3:** the results document, the results manifest, a summarize script and, if a clause
   fires, a `docs/DECISIONS.md` entry. A reviewer checks every restated number against
   `report.json` by script.
5. **One task, one agent.** A protocol defect found after the freeze is escalated to the task owner
   and fixed only through a disclosed amendment.

## 17. Changes made in PR 2 (disclosed; no gated run yet)

PR 2 commits the runner (`scripts/train_apple_token_dynamics.py`), its guard tests
(`tests/test_token_dynamics_runner.py`) and the runner's pin in the manifest. The rows, gates,
thresholds, seeds, budget and caps are unchanged. What the runner does, all mechanics:

- **Reuse.** It imports TASK-065's runner read-only, for its reader (Q-split before any file is
  opened), its guard helpers, its window, gather and val-criterion code and its report writer.
  That file's bytes are pinned by this manifest.
- **Preflight.** G-hash is recorded before any import. G-frozen (`token_dynamics.require_frozen`)
  runs before torch is imported. Any exception from the first line of `run` onwards writes a V
  report.
- **G-anchor on tokens.** The 190 post-look frames' CLS and tokens must hash to TASK-064's
  `P_cls` and `P_tok`. G-repro covers both.
- **G-cache (a/b/c).**
  - (a) is TASK-065's amended determinism check.
  - (b) re-featurises the first two train episodes whose length is not a multiple of 16, and
    checks them bit for bit against the cache.
  - (c) is the 1e-3 bound on the pooled rows, which reports the differing roots and names
    TASK-064's partial anchor batch.
- **Memory.**
  - Normalisation and the metric scale are computed in float64 in chunks, not as a copy of the
    cache.
  - G1's full-width statistics are streamed: `Moments` per model and horizon, and `SessionMoments`
    in the projected basis.
  - Each seed's moments are reduced as soon as its four models exist, then released.
  - The projected basis is fitted on every train frame's encoded latent before any model is
    trained.
- **Recorded beyond §12:**
  - every model's `selected_in_last_two_points`;
  - every seed's `G1_parts` per horizon;
  - `rank_W_over_N` with its interval;
  - the ceiling per gated horizon;
  - the projection basis sha256;
  - the frozen values themselves.
- **Smoke runs (not evidence, not rows).** `smoke` mode uses a subset: 8 train sessions per half,
  8 val episodes, seed 0 only, 20 updates, selection every 10. **The apple targets are replaced by
  seeded noise**, and G1 is not evaluated. Its gates are meaningless by construction and were not
  read.
  - `outputs/task066-scratch/smoke-a`: an earlier draft with a provisional manifest.
  - `outputs/task066-scratch/smoke-b`, with the committed code (`581baa1`. The later rebases onto
    PR 1's fixes left the runner's bytes unchanged, sha256 `7a89aee1…878b`):
    - report sha256 `b8d075f4…ab1ca`, 422 s;
    - every section was written, `non_finite_fields` was empty, and `test_split_decoded` was
      false;
    - G-cache (b) held on `look-47000` (final batch 14) and `look-47000-b0-noise_only` (final
      batch 4);
    - **peak RSS 14.2 GB.** smoke-a peaked at 6.98 GB on the same subset. The difference is not
      explained. MPS allocations count towards the process footprint on unified memory.
- **The memory expectation of §12 is therefore widened, not as a guard: about 15–30 GB** on the
  48 GiB machine. The report records the peak.
