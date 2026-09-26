# Apple→Plate latent dynamics v1: does a LeWM predictor on frozen DINOv2 CLS latents learn real dynamics on `apple-look-v1`, and does the apple stay readable through prediction? (TASK-065)

**Status: preregistration. No predictor has been trained on `apple-look-v1`, and no readout has
been fitted on any feature of it that this study introduces.** The only numbers here are TASK-064's
published results and the label-free pre-freeze checks of §13. **The test split of `apple-look-v1`
is never decoded. Cohorts C (45300–45339) and D (45000–45007, 45100–45107) are never simulated;
nothing is simulated at all. `exemption_spent`
(`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays `false`.**

This is a **world-model test only.** It trains an action-conditioned latent predictor and measures
it offline, on stored frames. It preregisters **no control formulation**, and nothing in it
implies one. CEM over the world-model cost was abandoned as the primary control line at TASK-054,
and the behaviour-cloning line stopped at TASK-057 (`docs/DECISIONS.md`). Both clauses still hold.
**Learned Apple→Plate is still 0 successes.** The corpus comes from the **privileged scripted
collector**, whose successes are scripted, not learned.

Predecessors:
- [`apple_look_corpus_v1.md`](apple_look_corpus_v1.md) /
  [`_results.md`](apple_look_corpus_v1_results.md) (TASK-064, outcome **C-ACCEPT**; the corpus);
- [`apple_pretrained_encoder_v1.md`](apple_pretrained_encoder_v1.md) /
  [`_results.md`](apple_pretrained_encoder_v1_results.md) (TASK-063, outcome **O-PT-POOLED**; the
  encoder);
- [`apple_encoder_study_v1_results.md`](apple_encoder_study_v1_results.md) (TASK-062,
  O-ENC-ARCH) and [`apple_observation_reprobe_v1_results.md`](apple_observation_reprobe_v1_results.md)
  (TASK-061, O-LOOK-RAW; the look and the probe);
- [`apple_world_model_v4_results.md`](apple_world_model_v4_results.md) (TASK-054; the persistence
  problem this test is built to expose).

Manifest: `benchmarks/manifests/apple-latent-dynamics-v1.json`. Code that fixes the design:
`src/embodied_jepa/latent_dynamics.py` (partition, configuration, windows, statistics, gates,
rows) and the model option `frozen_encoder` in `src/embodied_jepa/models/frozen_encoder.py`,
committed with this document and pinned by hash.

---

## 1. The question, and the claim it can make

TASK-064's C-ACCEPT row pre-declared the next task: *a separately preregistered world-model task on
`apple-look-v1` with this frozen encoder as a candidate image encoder, with its own hypothesis,
controls, collapse and action-sensitivity diagnostics, and its own gates.* This is that task.

**Question.** Trained on `apple-look-v1` train frames over the pinned frozen DINOv2 ViT-S/14 CLS
feature, does a LeWM action-conditioned predictor learn **real dynamics**, meaning:
- its predictions do not collapse;
- they beat copying the last latent forward;
- they beat the same predictor trained without actions;
- they depend on the actions, by a margin fixed now;
- and **the apple position stays readable, through multi-step prediction,** by the TASK-063 probe
  fitted on encoder latents of train roots only?

**Claim scope, fixed now.** A pass (WM-DYNAMICS, §10) would say: on this corpus, at horizons 8 and
16 steps (0.4 s and 0.8 s at 20 fps), on train-split sessions each held out from the predictor
that predicts them, *this* predictor on *this* frozen feature learns action-conditioned latent
dynamics that beat both baselines, and it keeps the post-look apple position readable to within the
TASK-059 T1 bar and 0.5 cm of the encoded target. It would **not** say anything about control, a
planner, closed-loop behaviour, horizons beyond 16, other phases of the task as a readability
claim, the test split, or another encoder.

## 2. What is reused unchanged

| piece | source | how |
|---|---|---|
| corpus | `data/apple-look-v1`, dataset manifest sha256 `81d760d1…db64` (TASK-064) | read-only; the manifest hash is a guard (G-data) |
| splits | TASK-064's whole-reset partition: 170 train / 20 val / 10 test sessions | unchanged; test never decoded |
| encoder | `facebook/dinov2-small` @ `ed25f3a3`, pinned files, pretrained digest `3a697b87…2af27` | `pretrained_encoder`, unchanged: 112 → 224 px bicubic, no crop, ImageNet normalisation, CLS after the final LayerNorm (`pooler_output`), CPU float32 |
| predictor | LeWM adapter over pinned upstream `le-wm` @ `8edfeb33`: `ARPredictor` + `Embedder` + `pred_proj` | `models/lewm.py` `next_embedding`, unchanged; history one (§4) |
| probe | TASK-061's L readouts as TASK-063/064 ran them: linear + RBF kernel ridge, nested 5-fold inner CV, λ grid 1e-4…1e4, grouped 10-fold outer CV with `default_rng(59)` | `info_ceiling.nested_cv`, `observation_reprobe.predict_with_fold_readouts`, unchanged |
| statistics | 10 000 paired bootstrap resamples (seed 5901), T1 bar (median ≤ 1.5 cm, ratio to B-occ upper bound ≤ 0.6), B-occ | `info_ceiling`, unchanged |
| reset-occlusion flags, reset apple xy | TASK-064 `collection_report.json` `readability.per_root` (sha256 `f3d2d3d5…6680`) | read; the label sidecars are checked against them (G-labels) |

## 3. Context carried forward (reported, not decisional)

### 3.1 TASK-064's result

Outcome **C-ACCEPT**. On the corpus's 190 train + val post-look frames (fresh resets), P-cls read
the apple to **0.550 cm [0.478, 0.642]**, dx sign **178/190**, beating its random-init floor R-cls
by −0.298 cm [−0.454, −0.178]. The corpus: 200 roots + 599 branches, 200 345 transitions; train
679 episodes (171 556 transitions), val 80 (20 474), test 40 (8315; never decoded).

### 3.2 Caveats carried forward, unchanged

They apply to this task and to anything built on it.
- **P-cls is not detectably different from raw pixels** (P-cls − L-raw −0.034 cm [−0.125, 0.054]
  on this corpus). No equivalence margin was preregistered.
- **Random-init features also read the apple.** R-tok met every bar on this corpus; R-cls missed
  only the T3 bar.
- **One encoder, one input size, one floor seed.** Floor-seed variance is not measured.
- **The cause is not identified** (pretraining data, width and input size changed at once).
- **Readability is not prediction.** This task is the first to test prediction.

### 3.3 Carried-forward context from TASK-064's results

- **Render nondeterminism (TASK-064 protocol §16).** A few mid-episode frames can differ by
  1–2 px, 1–2 levels, between collection runs. This task renders nothing: it reads the accepted
  corpus bytes (manifest `81d760d1…db64`), so its features are features of those bytes. A
  re-collection would not reproduce the dataset hash, and this task's results are tied to it.
- **Open non-blocking review items of TASK-064's runner**, and how this runner is built against
  them (PR 2's tests must show each):
  - the gate script's digest was checked at finalize, not recorded at load. **Here** every pinned
    file's sha256 is recorded at preflight, before any stage, and re-checked at the end;
  - a non-guard exception inside preflight wrote no report. **Here** any exception from the first
    line of `run` onwards writes `report.json` with outcome V and a `void_reason`;
  - the frozen `check_pins` branch had no unit test. **Here** the pin check is a tested helper,
    in both directions.

### 3.4 What the world-model history says this test must expose

- TASK-054's G2a (rollout readout error ÷ the model's own persistence readout) never passed:
  0.835 at v2, 0.831 at best in v3, 0.86–0.90 in v4. Copying the last latent forward was hard to
  beat. That is why **copy-last is a gate here, not a report.**
- TASK-058's audit found that several v4 gate passes were cleared by a constant or by a
  true-state copy-last. That is why **every gate here is paired with a baseline that needs no
  learned dynamics**, and why low prediction loss alone is not read as dynamics.

## 4. The predictor (fixed now)

### 4.1 Architecture: a frozen external encoder feeding the pinned LeWM predictor

A new backend-agnostic option, **`frozen_encoder`**, written once in a new module,
`models/frozen_encoder.py`. `frozen_encoder_model(backend)` returns the backend class with a mixin
in front of it. The plain backends do not know the key (they reject it as unknown), and the
frozen class refuses to run without it. So **the option is off unless a run selects it.**
Enabling it is the research choice this document preregisters (`docs/MODELS.md`).

**No existing model file changes.** `models/base.py`, `models/lewm.py`, `models/native.py`,
`models/readout.py` and `readout_labels.py` keep their bytes. That matters here: TASK-054's E0
checkpoint (and so TASK-062/063's anchors) enforces an `implementation_sha256` over those files,
and `tests/test_policy_preflight.py` guards it. A first draft of this option lived in
`models/base.py` and failed that guard, so it was moved out before anything was committed. Every
earlier model and checkpoint is therefore bit-for-bit what it was.

With `frozen_encoder: dinov2_small_cls`:
- **The image pathway is replaced by the pinned frozen DINOv2 CLS feature** (384-d), standardised
  per dimension by moments fitted on the model's **training frames only**
  (`fit_frozen_feature_normalization`, std floored at 1e-3). The latent *is* this standardised
  feature: no projector, no learned encoder layer. That is deliberate. The TASK-063 probe was
  fitted on the raw CLS; a learned projector would make the latent a learned function of it, and
  predictions could no longer be read by that probe. With an affine, train-fitted map, a
  prediction maps back to the raw CLS space exactly (`predict_features`).
- **The encoder is never trained**, never a submodule, never in the optimizer, never moved to MPS,
  never in the checkpoint's weights. Its weights digest goes into the model's metadata, which the
  checkpoint saves and `load` compares, so a load under a different frozen encoder is refused.
  `implementation_sha256` also covers the new module, the backend's file and
  `pretrained_encoder.py`. Features are always computed on CPU in float32 (bit-reproducible, and
  equal to TASK-064's P-cls: §13).
- **The backend's own training step and predictor run unchanged on it** (`train_step`,
  `next_embedding`, `rollout`). `train_step_features` hands cached features to the backend's own
  `train_step` in place of the frames' features. Because the encoder is frozen, this is the same
  update: `tests/test_frozen_encoder.py` checks it bit for bit (weight digest and loss) for both
  backends. The backend's own image encoder and projector are still constructed but unused (no
  gradient reaches them), so the predictor's initialisation stream is the adapter's.
- It refuses `state_fusion`, `readout_heads` and a `cameras` list. It is image-only, like the
  adapters' defaults.
- **The backend swap stays one key** (`latent_dynamics.BACKEND`); `tests/test_frozen_encoder.py`
  trains both backends on the same configuration.

**The predictor is the pinned upstream LeWM predictor, not native code.** The product goal is LeWM
on G1 + dual Dex3, so the predictor under test is the one the product would use: upstream
`ARPredictor` (AdaLN-conditioned transformer), `Embedder` (the action encoder) and `pred_proj`
(the BatchNorm MLP after the predictor), through the adapter's unchanged `next_embedding`. The
native predictor is one key away (`BACKEND = "native_jepa"`) but is **not run**: a second backend
would be a second arm, and it is not preregistered.

**What differs from upstream LeWM, and why.**
- The encoder is frozen and external (the point of the task), so upstream's `encode` and its
  projector are bypassed.
- **SIGReg weight 0.** SIGReg shapes a trainable encoder's embedding distribution. The encoder is
  frozen, so it has no gradient path here.
- **History one, zero dropout, the adapter's small predictor** (depth 2, 2 heads × 24, MLP 128).
  These are the adapter's declared local-budget choices (`docs/MODELS.md`), kept unchanged. None
  of them is tuned here.
- The adapter's extra recursive loss (`multistep_weight` 1.0) is kept, as in every earlier LeWM
  run in this repository.

### 4.2 Configuration (`latent_dynamics.MODEL_CONFIG`)

`camera: onboard_rgb`, `frozen_encoder: dinov2_small_cls`, `latent_dim: 384`, `sigreg_weight: 0.0`,
`max_horizon: 64`. Everything else is the adapter's default: `hidden_dim` 128, predictor depth 2,
heads 2, head dim 24, learning rate 3e-4, weight decay 1e-4, gradient clip 1.0, `multistep_weight`
1.0. Every TASK-050/052/054 option stays at its inactive default.

### 4.3 The arms and the baselines

| name | what | trained |
|---|---|---|
| **W** | the predictor above, on the true executed actions | yes |
| **N** (no-action baseline) | the same model, same seed, same windows, with **every action replaced by zero**, in training and in evaluation | yes |
| **copy-last** | the start latent, repeated for every step | no |

N is the "no-action baseline" of the brief: an action-blind predictor with the same capacity and
budget. It can still learn the average motion from the image, and on this corpus the expert's
commands depend on the visible scene. So beating N is a test that the actions add information
beyond what the image predicts.

## 5. Data (fixed now)

### 5.1 What is read

- **Train and val episodes only** (679 + 80). A reader that refuses every other episode id is the
  only path to the episode files (Q-split). The test split's 40 episodes are never decoded.
- Frames (`observation.images.onboard_rgb`, PNG), actions (`action`, the applied normalised
  `ee_delta_grasp_v0`, 14-D) and `action_valid`. Proprioception is not read by any model.
- **Label sidecars, train + val only, for analysis targets only:**
  `privileged__apple_position_world` (the probe's target) and `collector__phase_index` (to check
  the look). They are never model inputs, loss weights or selection filters.

### 5.2 Cross-fitting halves (probe roots are cross-fitted, not held out)

- The 170 train sessions (a root and its branches stay together), sorted, are permuted by
  `default_rng(65)`: the first 85 are **half A**, the other 85 **half B**. Pinned:
  `halves_sha256 = 26709b93…5061`. Half A: 339 episodes, 85 521 transitions; half B: 340
  episodes, 86 035 transitions.
- **Every model is trained on one half.** For each model seed and arm there are two models:
  W_A (trained on half A) and W_B.
- **Every evaluated train session is predicted by the model that never trained on it**: half-A
  sessions by W_B and half-B sessions by W_A. So all 170 train sessions are evaluated out of the
  predictor's training data.
- **The probe is cross-fitted too.** It is fitted on *encoder* latents of train roots only, by
  10-fold CV over the 170 train roots, and each root is read by the fold readout that held it out.
  The probe's training roots include roots of the predictor's own training half. That is not
  leakage: the probe maps an encoded latent to the apple and never sees a prediction, and the
  evaluated root is held out from both the predictor and the probe fold that reads it.
- **Val (20 sessions) is used only for model selection** (§6.3), plus one reported-only secondary
  readability estimate (§7.3).

### 5.3 Normalisation (train only)

- **Model input normalisation**: each model fits the per-dimension mean and std of the raw CLS
  over **every frame of its own training half** (float64 moments), once, before its first update.
- **The metric scale** (for every normalised MSE of §7): the per-dimension std of the raw CLS over
  every frame of **all 170 train sessions**, floored at 1e-3, fitted once. It is a train-only
  constant, identical for every model, so MSE values are comparable across halves, arms and
  seeds. It is never a model input.

### 5.4 Features

- Every frame of every train and val episode is featurised **once**, through the model's own
  `frozen_features` (CPU, float32, 6 torch threads, batches of 16 frames per episode in frame
  order) into a feature cache under the run's output directory (sha256 recorded; never
  committed). Training reads only this cache. Given the same features, the cached-feature update
  equals the frame update bit for bit (`tests/test_frozen_encoder.py`). Batched float32 CPU
  inference could in principle depend on how frames are batched; G-cache bounds that on the
  post-look frames, and the pre-freeze check found 0.0 (§13).
- **G-anchor** (§9): the 190 train + val post-look frames, featurised as TASK-064 did (its root
  order, batches of 16), hash to TASK-064's `feature_sha256.P_cls`, `24bc50f5…`. **G-cache**: the
  cache's frame-8 rows equal those anchor rows within 1e-5.

## 6. Training (fixed now)

### 6.1 Runs

2 arms (W, N) × 3 model seeds (0, 1, 2) × 2 halves (A, B) = **12 models**. Each model:
- is constructed with
  `frozen_encoder_model("leworldmodel")(state_schema, device="mps", seed=s, config=MODEL_CONFIG)`;
- fits its input normalisation on its half's frames;
- takes **10 000 updates** of batch 64 windows of 16 transitions (17 frames), through
  `train_step_features`;
- samples windows uniformly over every start of every valid 16-transition window of its half's
  episodes (a window never crosses an episode boundary, and never includes the final row, whose
  `action_valid` is false), with `default_rng(SeedSequence([6500, s, half index]))`;
  - **the stream does not depend on the arm**: W and N of the same seed and half draw the identical
    windows in the identical order, so G3 compares them paired;
- runs at a constant learning rate (no scheduler).

### 6.2 Device and determinism

MPS for training and prediction; CPU for features and every statistic. MPS training is not
bit-reproducible run to run. This is disclosed, not a guard: the seeds and the budget are fixed,
not the bits.

### 6.3 Model selection, on val only

- After every 500 updates (20 points), the **val criterion** is computed: the normalised MSE of
  the recursive prediction, averaged over steps 1…16, over the windows of all 80 val episodes
  starting at every 4th frame. For N the actions are zero here too.
- The checkpoint with the lowest criterion is kept (ties go to the earlier one), and is the model
  evaluated in §7. Nothing else is selected, and no train-split or evaluation number is used for
  selection.
- Checkpoints go to `checkpoints/task065-latent-dynamics/run-<k>/` and are never committed.

## 7. Evaluation sets and metrics (fixed now)

### 7.1 E-all: dynamics, every phase

- For each seed and arm, **every window of 16 transitions starting at every 4th frame of every
  train episode**, predicted by the half model that did not train on it. This covers all 170 train
  sessions, roots and branches, and every phase.
- For each window, start frame t, the prediction at step h is compared with the encoded frame
  t + h. The per-window error is the **normalised MSE**: the mean over the 384 dimensions of
  ((prediction − target) / metric scale)².
- Predictions compared at the gated horizons **h ∈ {8, 16}** (reported also at 1, 2, 4):
  - **W** with the true actions;
  - **W** with **wrong actions**: each window's actions are those of another window from a
    different session (a fixed cross-session shuffle, seed 6502);
  - **W** with **zero actions**;
  - **N** (zero actions, as trained);
  - **copy-last**.
- **Ratios** are ratios of summed per-window errors, with a 95 % percentile interval from a
  **session-clustered bootstrap**: 10 000 resamples of the 170 sessions with replacement (seed
  6501), the same resamples for every ratio.

### 7.2 E-post: the apple through prediction

- **The 170 train roots, starting at the post-look frame 8** (TASK-064's decision frame), rolled
  forward with the root's own executed actions, each by the half model that did not train on it.
- **Target:** the apple's xy at frame 8 + h (`privileged__apple_position_world`), the label of
  the frame whose latent is being predicted.
- **Probe:** for each horizon h, TASK-063's readouts (linear + RBF kernel ridge, nested CV) are
  fitted by 10-fold CV (`default_rng(59)`, fold hash `44a3f267…86a7` over the 170 train roots
  sorted by seed) on the **encoded** raw CLS of frame 8 + h of train roots. Each root's
  prediction is read by the fold readout that held that root out.
- **What is read by that probe**, at h ∈ {1, 2, 4, 8, 16}:
  - the encoded frame itself (the ceiling);
  - W's predicted feature;
  - copy-last (the encoded frame 8);
  - N's prediction;
  - W with wrong or zero actions.
- **Statistics:** TASK-059's T1 quantities (median error in cm with its bootstrap CI, ratio to
  B-occ; B-occ uses TASK-064's reset-occlusion flags and the same folds). A "median difference"
  is the difference of the two medians (`info_ceiling.paired_difference` with `_median`), with a
  paired percentile interval over `ic.bootstrap_indices(170)` (seed 5901).
- **What the pre-freeze label inspection says about this target (§13):** on the 40 train roots
  inspected, the apple did not move before frame 117, and the right palm moved 1.2–3.4 cm in the
  first 8 post-look steps (0.9–7.1 cm by frame 40). **So, on those 40 roots, the apple is static
  over h ≤ 16 (and over the extended h ≤ 64), and copy-last carries it by construction.** This is
  a 40-root observation; the run reports the count for all roots (§7.3). Beating copy-last on
  apple readability is therefore *not* a dynamics test on these windows, and it is not a gate. The dynamics gates are on E-all.
  The readability gate asks whether prediction **keeps** the apple: an absolute bar and a
  non-inferiority margin to the encoded target.

### 7.3 Reported only (none decides anything)

- Every E-all ratio at h = 1, 2, 4, and the same ratios on the E-post windows.
- E-post readability at the **extended horizons 32 and 64** (beyond the training horizon), and
  the literal variant: the probe fitted on encoded **frame 8** only, applied to predictions at
  every h.
- **Secondary estimate on val**: the probe fitted on all 170 train roots' encoded frame 8 + h,
  applied to the 20 val roots' predictions from W_A and W_B separately.
- Collapse statistics of N and copy-last; the val criterion curve of every model; training losses.
- How many roots' apple moved more than 1 mm by frame 8 + h.

## 8. Gates (fixed now)

Each gate is computed **per model seed** (W_A and W_B of that seed together, over all 170 train
sessions), at **each of h = 8 and h = 16**. A gate passes for a seed only if it passes at both
horizons.

| gate | set | passes when (at h = 8 and h = 16) |
|---|---|---|
| **G1 no collapse** | E-all | W's predictions (divided by the metric scale): collapsed fraction (per-dimension std < 0.01) ≤ 0.05, **and** effective rank ≥ 0.5 × the encoded targets' effective rank on the same windows, **and** mean per-dimension std ≥ 0.5 × the encoded targets' |
| **G2 beats copy-last** | E-all | upper 95 % bound of MSE(W) / MSE(copy-last) **≤ 0.8** |
| **G3 beats no-action** | E-all | upper 95 % bound of MSE(W) / MSE(N) **< 1.0** |
| **G4 action sensitivity** | E-all | lower 95 % bound of MSE(W, wrong actions) / MSE(W, true) **≥ 1.10**, **and** lower bound of MSE(W, zero actions) / MSE(W, true) **≥ 1.10** |
| **G5 the apple stays readable** | E-post (170 roots) | W's predicted feature under the frame-(8 + h) probe: median T1 error **≤ 1.5 cm** **and** upper bound of its ratio to B-occ **≤ 0.6** (TASK-059's T1 bar, unchanged), **and** upper bound of the median difference (median W − median encoded frame 8 + h, paired interval) **≤ 0.5 cm** |

**Why these thresholds** (fixed before any predictor was trained on the corpus; none was tuned):
- **G2's 0.8** is TASK-054's G2a threshold (rollout ÷ persistence ≤ 0.8), applied here to the
  latent itself. A predictor that cannot remove a fifth of copy-last's error at 0.4–0.8 s has not
  learned the motion that the frames show.
- **G3 < 1.0.** "Beats" means the whole interval lies below parity. No margin is added, because
  N can legitimately predict much of the expert's motion from the image.
- **G4's 1.10** is the action-sensitivity margin the brief asks to be fixed in advance: wrong or
  absent actions must cost at least 10 % more error, at the lower bound, at both horizons. Zero
  is out of distribution for most steps; the cross-session shuffle is in distribution. So both are
  required.
- **G1's 0.5 ratios.** Predictions regress to a conditional mean, so some loss of spread is
  expected. Halving the spread or the rank is collapse.
- **G5.** The absolute bar is the one every readability result since TASK-059 met. The 0.5 cm
  margin is about 38 % of the 1.316 cm gap between the encoded P-cls (0.550 cm on this corpus) and
  B-occ (1.866 cm). It asks that prediction keep most of what the encoder reads, not merely stay
  above the prior.

**Per seed:** the seed passes if G1–G5 all pass. Its **dynamics** hold if G1–G4 all pass.

## 9. Guards: any failure voids the run (outcome V)

| guard | condition |
|---|---|
| **G-hash** | every file in the manifest's `hashes` matches at preflight (recorded then, per file), the tracked tree is clean, and both hold again after the last stage; a change during the run is V |
| **G-data** | the dataset opens (`DatasetStore` verifies every recorded hash) and its manifest sha256 is `81d760d1…db64`; the sessions are 170 / 20 / 10 and the episodes 679 / 80 / 40 by split |
| **Q-split** | only train and val episodes are decoded, through a reader that refuses every other id; `test_split_decoded: false` is recorded |
| **G-split** | the halves hash to the pin; each model's training episodes are exactly its half's train episodes; no episode trains both halves; normalisation episodes = training episodes |
| **G-labels** | every read root's first 8 phase labels are the look (−1); its frame-0 apple label equals TASK-064's recorded reset apple xy within 1e-6 m |
| **G-weights** | the DINOv2 files match their pins and the loaded digest is `3a697b87…2af27` |
| **G-anchor** | the 190 post-look frames hash to TASK-064's `stored_post_look_frames_sha256` (`20b67967…3612`), and their features to `feature_sha256.P_cls` (`24bc50f5…feec`) |
| **G-repro** | a second featurisation of the 190 post-look frames is bit-identical |
| **G-cache** | the cache's frame-8 rows equal the anchor rows within 1e-5 (the pre-freeze check found 0.0, §13) |
| **G-folds** | the probe fold hash over the 170 train roots equals `44a3f267…86a7` |
| **G-finite** | every feature, loss, prediction and statistic is finite (a non-finite value is mechanical, V) |
| **G-cap** | featurisation ≤ 5400 s; each model (10 000 updates + 20 val evaluations) ≤ 1800 s; the whole run ≤ 21 600 s |
| **G-device** | MPS is available and used for every model; CPU for features and statistics |

- **Any other exception is a crash, and a crash is V.** `report.json` is written with outcome V and
  a `void_reason` whatever stage failed, preflight included.
- **PR 2's tests must exercise**, in both directions: every guard; every gate through
  `latent_dynamics`; every row of `decide`; that the reader refuses a test episode; that a crash
  at preflight and one mid-run each write a V report; the pin check; that non-finite values are
  written as `null` and listed in `non_finite_fields`.

## 10. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§9), or the run stops before writing a complete report | nothing is read | one from-scratch repeat (§12) |
| **WM-DYNAMICS** | every seed (0, 1, 2) passes G1–G5 | On `apple-look-v1`, a LeWM predictor on frozen DINOv2 CLS latents learns action-conditioned dynamics: no collapse, beats copy-last and a no-action predictor, uses the actions, and keeps the post-look apple readable through 16-step prediction. | A separately preregistered **held-out confirmation on the corpus's test split** (never decoded until then), and/or a harder world-model test (later task phases, longer horizons, what the apple does once touched). **No control formulation is implied**; any control use needs its own preregistration and must answer the TASK-054 and TASK-057 clauses. |
| **WM-UNSTABLE** | one or two seeds pass | The result does not hold across seeds. Nothing is established either way. | **The abandonment clause does not fire.** The owner decides whether a disclosed amendment adds seeds. If it does, the added seeds are counted together with these three, the rule stays "every seed passes", and the amended result is reported as exploratory. |
| **WM-APPLE-LOST** | no seed passes, and G1–G4 pass on at least two seeds | The predictor learns dynamics, but the apple does not survive prediction on this latent. | **The abandonment clause fires** (below). |
| **WM-NO-DYNAMICS** | otherwise (no seed passes, and G1–G4 fail on at least two seeds) | The predictor on this latent does not learn action-conditioned dynamics that beat the baselines (or collapses). | **The abandonment clause fires** (below). |

**INCONCLUSIVE (task-level, not a row of a run).** A second V closes TASK-065 as INCONCLUSIVE
(§12). Nothing is read, the abandonment clause does not fire, and the owner decides what follows.

**Abandonment clause (WM-APPLE-LOST, WM-NO-DYNAMICS).**
- **What closes:** the line *"an action-conditioned LeWM-family predictor on frozen, externally
  pretrained **pooled (CLS)** latents on `apple-look-v1`"*. No further predictor variant on this
  corpus with frozen pooled pretrained latents is preregistered without new evidence of a
  different kind. That covers history length, action chunking, step embeddings, loss weighting,
  capacity, residual parameterisation and budget.
- **What does not close:** the LeWM backend, the encoder as a component, the product goal, and the
  corpus (kept, sealed; its test split still unread).
- **Recorded as untested, not refuted:** token-grid latents (P-tok) as a world-model latent; a
  trainable encoder initialised from the pretrained one; other encoders.
- **Context, recorded here and not preregistered as a follow-up:** the fixed `overview` camera
  (TASK-061 O-raw, 0.183 cm [0.168, 0.207], 185/190) remains the observation change on record. It
  is a hardware/workspace change, the owner's decision, and would need its own preregistration.

**Always reported, whatever the row:** every gate quantity with its interval, per seed and horizon;
every reported-only quantity of §7.3; every model's val curve and selected update; the rows that
also match further down.

**In every outcome:**
- nothing is re-thresholded, retrained, re-selected or re-chosen after the numbers are seen;
- learned Apple→Plate stays at 0 successes, and `exemption_spent` stays `false`;
- the executing agent recommends a next task and does not choose it.

**Stated in advance.**
- **Every failing row is a live possibility.** The history (§3.4) is of predictors that did not
  beat persistence.
- **G5 can pass while G2–G4 fail.** The apple is static over h ≤ 16 on E-post, so a predictor that
  learned nothing but kept its input would read the apple well. That is why G5 alone decides
  nothing.
- **A pass is 170 cross-fitted train sessions and one corpus.** It is not a test-split result.
- **P-cls is still not detectably different from raw pixels** as a readout. A pass would not show
  that pretraining matters for dynamics; no raw-pixel or random-init predictor is run.

## 11. Budget, device, seeds, recording

- **Device:** MPS for the 12 models (training, selection, prediction); CPU for features (6
  threads, float32) and every statistic. The runner refuses to start without MPS (a guard, V).
- **Caps:** featurisation **5400 s**; each model **1800 s**; the whole run **21 600 s** (6 h).
  Expected (§13): featurisation about 45 min (192 789 frames at about 13 ms), each model about
  10 min (about 54 ms per update on MPS), evaluation about 10 min; about 3 h in total.
- **Seeds:** model seeds 0, 1, 2; halves 65; sampler `SeedSequence([6500, seed, half])` (shared by W and N);
  cluster bootstrap 6501; cross-session shuffle 6502; probe folds 59, inner 5900 + k, bootstrap
  5901. Nothing else is random.
- **Recorded** (`outputs/task065-latent-dynamics/run-<k>/report.json`):
  - the code revision (clean tracked tree required), every pinned hash at preflight and at the end;
  - the environment (torch, transformers, threads, MPS);
  - the feature cache's sha256 and the anchor comparison;
  - the halves, every model's training episodes hash, normalisation hash, val curve, selected
    update, losses, elapsed time, checkpoint sha256;
  - every E-all and E-post quantity per seed, arm and horizon, with intervals, and per-root
    readouts;
  - the gates, the per-seed rule and the decision;
  - elapsed times and peak RSS.
- **Output:** `outputs/task065-latent-dynamics/run-1/`, checkpoints
  `checkpoints/task065-latent-dynamics/run-1/`. The runner refuses to overwrite either. Nothing
  under `data/` is written. Checkpoints and features are never committed.

**Frozen run command** (from a clean checkout of the reviewed commit, after the pre-run GO):

```sh
uv run --no-sync python scripts/train_apple_latent_dynamics.py run \
    --output outputs/task065-latent-dynamics/run-1 \
    --checkpoints checkpoints/task065-latent-dynamics/run-1
```

## 12. Void rule

- A run that stops early for any reason (a guard, a crash, a cap) is **V**, with a `void_reason`.
  Nothing in it is read.
- **Exactly one from-scratch repeat** is allowed, into `run-2` (outputs and checkpoints): the same
  seeds, caps and device, a clean tree, and the features recomputed (a V run's cache is not
  reused). A void run's directories are kept as evidence.
- **A second void closes TASK-065 as INCONCLUSIVE.**
- A fix between the runs is limited to runner mechanics. It goes through a reviewed PR and a fresh
  pre-run GO, and it is disclosed in the results.
- **Any ruling (a void, a repeat, an owner decision) is recorded with a UTC timestamp** in the task
  card before the run it affects.

## 13. Pre-freeze facts (disclosed; they fit nothing on the corpus)

- **No existing model file changes** (§4.1). The TASK-054 E0 implementation-digest guard
  (`tests/test_policy_preflight.py`) passes. A first draft that edited `models/base.py` failed
  it and was withdrawn before commit.
- **The anchor holds.** The 190 post-look frames, decoded from the corpus and featurised through
  `pretrained_encoder` exactly as TASK-064 did, hash to TASK-064's recorded
  `stored_post_look_frames_sha256` and `feature_sha256.P_cls` (`24bc50f5…`). Featurised in the
  cache's layout (each root's frames 0–15 in one batch), the frame-8 rows of 20 roots differ from
  the anchor by **0.0**. No readout was fitted.
- **Timing (synthetic inputs):** a frozen-encoder LeWM update of batch 64 × 16 takes about 54 ms on
  MPS and 107 ms on CPU; featurisation about 13 ms per frame on 6 CPU threads.
- **Label inspection, disclosed.** While designing the horizons, the author read the
  `privileged__apple_position_world`, `privileged__palm_position_world` and
  `collector__phase_index` labels of **40 train roots** (`look-47000` to `look-47045`, in order).
  - The apple first moved (more than 1 mm from its frame-0 xy) at frame 117–234; at frame 222 or
    later on 37 of 40.
  - The collector's phase 1 started at frame 138 on all 40.
  - The right palm moved 1.2–3.4 cm between frames 8 and 16, and 0.9–7.1 cm between 8 and 40.
  - This informed §7.2's reading (the apple is static on E-post) and the choice of gated horizons
    within the 16-step training window. **No feature, prediction or readout was computed**, and no
    threshold was set from a model number.
- **The partition's pins** (from the dataset manifest's split lists only): halves
  `26709b93…5061` (85 / 85 sessions); probe folds `44a3f267…86a7`; the shortest train root has
  227 frames, so every root covers frame 8 + 64.

## 14. Not done (declared)

- No control formulation, planner or controller; no closed-loop attempt; no simulation at all.
- The test split is not decoded; cohorts C and D are not simulated.
- No encoder is trained or fine-tuned. No second encoder, read-out point (P-tok) or input size;
  no raw-pixel or random-init predictor; no native-backend arm.
- No change to any existing model file (`models/base.py`, `lewm.py`, `native.py`, `readout.py`,
  `readout_labels.py`), `pretrained_encoder.py`, `info_ceiling.py`, `observation_reprobe.py`, the
  corpus, any earlier checkpoint or any earlier protocol. The option is a new file
  (`models/frozen_encoder.py`).
- Every TASK-050/052/054 model option stays at its inactive default.

## 15. Process

1. **PR 1:** this document, the manifest, `latent_dynamics.py`, the `frozen_encoder` option (with
   `docs/MODELS.md`), their tests and the task card. It merges on an independent reviewer's
   **reported** APPROVE and green CI, before any predictor is trained on the corpus.
2. **PR 2:** the runner (`scripts/train_apple_latent_dynamics.py`), the guard tests of §9 and the
   manifest's pin of the runner. It merges on a reported APPROVE and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3:** the results document, the results manifest (every restated number), a summarize script
   and the recommended next task. A reviewer checks every restated number against `report.json` by
   script. The card goes to done.
5. **One task, one agent.** A protocol defect found after the freeze is escalated to the task owner
   and fixed only through a disclosed amendment.

## 16. Changes made in PR 2 (disclosed; no gated run yet)

PR 2 commits the runner (`scripts/train_apple_latent_dynamics.py`), its guard tests
(`tests/test_latent_dynamics_runner.py`) and the runner's pin in the manifest. The rows, gates,
thresholds, seeds and caps are unchanged. Additions, all mechanics:

- **G4 and an undefined resample.** `cluster_ratio` counts a zero-denominator resample as the
  largest float. That fails every upper bound, but it would help a lower bound. So the runner's G4
  also requires `undefined_resamples == 0` for both of its ratios (PR 1 review, non-blocking).
- **The preflight order.** The manifest is read and every pinned file hashed and recorded before
  torch or any model is imported. A crash anywhere, preflight included, writes a V report
  (§3.3).
- **What the report records beyond §11:**
  - the cross-session shuffle is applied per evaluated half;
  - the val-root secondary estimate uses a probe selected and fitted on all 170 train roots;
  - the optimizer state in a checkpoint is the final one, and its weights are the selected
    update's.
- **Smoke runs (not evidence, not rows).**
  - What a smoke run is: `smoke` mode on a subset (8 train sessions per half, 8 val episodes,
    seed 0 only, 20 updates, selection every 10), with **the apple targets replaced by seeded
    noise**. Its gates are meaningless by construction and were not read as results.
  - `outputs/task065-scratch/smoke-a/` (report sha256 `904c3515…ff2b`):
    - preflight, G-anchor (frames `20b67967…`, features `24bc50f5…`) and G-repro passed;
    - G-cache 0.0;
    - labels within 1.4e-8 m;
    - 17 620 frames featurised in 236 s (13.4 ms per frame);
    - four models trained;
    - every report section was written, `non_finite_fields` empty, `test_split_decoded` false;
    - 266 s in total.
  - smoke-a ran the runner before the preflight reorder above.
- **After PR 2's first review** (all mechanics; no row, gate, threshold, seed or cap changed):
  - The reader checks Q-split before any import, so a test id is refused even without the
    optional imaging packages.
  - **G-finite now covers every statistic.** A non-finite statistic before the decision is V, not
    a failed gate. A NaN interval would otherwise fail a gate and could fire the clause.
  - Two quantities §7.3 preregisters are now computed; both are reported only:
    - the latent-MSE ratios on the E-post roots (W/copy-last, W/N, wrong/W, zero/W);
    - the collapse statistics of N and copy-last.
  - G4's undefined-resample rule is a tested helper (`g4_gate`).
  - The normalisation episodes are asserted equal to the training episodes (G-split).
  - `look_corpus.py` is pinned.
  - Copy-last reads only frames 0 and h.
  - **smoke-b**, with the committed code, same subset and noise targets:
    - report sha256 `88c038ba…8363`;
    - 266 s in total;
    - every section written, `non_finite_fields` empty.
- **After PR 2's second review:** in the reader, both optional imports (`pyarrow`, `PIL`) now come
  after Q-split and the file-hash check. Before this, the core-only CI job, which has neither
  package, failed the hash-mismatch test on the import. This is an import-order change only.
