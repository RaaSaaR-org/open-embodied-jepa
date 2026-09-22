# Apple world model v2: preregistration (TASK-050)

This is the prospective protocol for roadmap step T3. It is committed before the
frozen training runs. **Everything here is OFFLINE evaluation of learned models on
recorded validation data.** No closed-loop control is run, and no number here is a
learned Apple→Plate result: learned Apple→Plate stays at 0 successes.

The frozen values live in `benchmarks/manifests/apple-world-model-v2.json`
(`frozen`). The code is `src/embodied_jepa/world_model_v2.py`,
`src/embodied_jepa/models/readout.py`, `src/embodied_jepa/readout_labels.py` and
the shared additions in `src/embodied_jepa/models/base.py`.

## Question

Can a learned world model, driven only by the robot's own camera, proprioception
and executed actions, **predict the quantities the object-aware phase cost needs**
(`embodied_jepa.object_ceiling`) at planning horizons, from its own predicted
latents, well enough to be worth a closed-loop test?

The quantities are the palm–apple offset, the apple height above its rest, the
apple–plate offset and whether the apple is held. At run time no simulator truth
is available to the model or the planner.

The product backend is **LeWM** (`models/lewm.py` over the pinned upstream
revision `8edfeb33`). The native JEPA is the comparison. LeWM must pass for T4
(TASK-051) to start.

## Data

- **Corpus.** `data/apple-wide-v1` (TASK-048), dataset manifest SHA-256
  `028e130576c052437f7753d74dd64d80dabc1edb8085247e7f56bc71a0412184`. 797
  episodes (200 wide-jitter roots, 597 grasp-phase branches), 205,519
  transitions, 20 fps.
- **Splits (whole resets, frozen by TASK-048).** train 677 / val 80 / test 40
  episodes (170 / 20 / 10 resets).
  - Updates use **train** only.
  - Checkpoint selection and every gate use **val**.
  - **test is never decoded** (the loader refuses it; reports record
    `test_episodes_decoded: 0`). It stays reserved for later confirmation.
- **Model inputs.** `onboard_rgb` at 112×112 (one camera per run), the 86-D
  proprioception and the executed 14-D normalized actions.
  - Proprioception is normalized with the corpus's train-only moments
    (`meta/jepa_normalization.json`, fit on the 677 train episodes; std floored
    at 0.01).
  - `hand_crop_rgb` is **not** used in this task (see *Not done*).
- **Privileged labels.** The gated sidecars are read with
  `acknowledge_privileged_training_labels=True`. They serve only two roles:
  - **readout-head training targets;**
  - **val scoring references.**
  They are never a model input, a loss weight or a sampling/selection filter.
  The collector phase index is used only to define val *scoring* cohorts.

### Readout targets (`readout_labels.targets`)

| Readout | Width | Definition |
|---|---|---|
| `palm_minus_apple` | 3 | palm site minus apple centre, world metres |
| `apple_height` | 1 | apple z minus its resting z at the reset (the root's first frame) |
| `apple_minus_plate` | 3 | apple centre minus plate centre, world metres |
| `hand_contact` | 1 | hand/wrist–apple contact (probability head) |
| `apple_held` | 1 | contact **and** `apple_height ≥ 2 cm` (probability head) |

## Models

### Shared, backend-agnostic additions (`models/base.py`, `models/readout.py`)

Both are off by default, so every earlier model keeps its image-only latent.

- **State fusion (`state_fusion: true`).** latent = backend image embedding +
  a shared MLP embedding of the normalized proprioception.
  - The predictor rolls the fused latent forward, so predicted steps need no
    future robot state.
  - Image-only goal encoding is refused for fused models; planning uses the
    declared readouts.
- **Readout heads (`readout_heads: true`).** A shared LayerNorm + MLP maps a
  latent to the five readouts.
  - Loss: smooth-L1 on 5 cm-scaled regressions and BCE on the probabilities.
  - It is applied to the encoded latents of every frame of a window **and** to
    the recursively predicted latents (steps 1…16), with weight 1 each.
  - The gradient trains the encoder, the predictor and the heads.
- **Interface.**
  - `Capabilities.readouts` declares the names.
  - `model.readout(z)` returns physical values for an encoded `[B,D]` or a
    predicted `[B,K,T,D]` latent.
  - `model.latent_statistics(z…)` gives collapse diagnostics.
  - These are the only ways an evaluator or planner derives anything from a
    latent. The `ReadoutWorldModel` protocol in `contracts.py` documents this.

**Honest label.** The LeWM arm is the pinned upstream encoder, predictor,
projector and SIGReg objective, **plus** the shared state fusion and the
auxiliary readout loss. That is not the unmodified LeWM objective. The native
arm receives exactly the same additions.

### Backends (`configs/apple_wm_v2.yaml`, `configs/apple_wm_v2_lewm.yaml`)

The LeWM config is `extends: apple_wm_v2.yaml` plus `world_model.backend:
leworldmodel`. That is the one-line swap. The shared keys are identical for both
backends: `camera onboard_rgb`, `latent_dim 128`, `hidden_dim 256`,
`max_horizon 64`, `state_fusion`, `readout_heads`, `readout_weight 1.0` and
`readout_hidden_dim 256`.

| | LeWM (`leworldmodel`) | native (`native_jepa`) |
|---|---|---|
| Image size | 112 (patch 14 → 8×8 tokens) | 128 (see below) |
| Encoder | ViT, depth 4, 4 heads, dim 128 | conv stride-8 → 4×4 pool → MLP |
| Predictor | upstream AR predictor, depth 4, 4 heads × 32 | residual MLP |
| Anti-collapse | SIGReg (weight 0.09) | variance + covariance, EMA target |
| Parameters | 2,163,103 | 931,145 |

The native encoder's stride-8 grid must divide its 4×4 adaptive pool on MPS.
It therefore upsamples the 112 px camera to 128 px. No resolution is discarded.
The difference is backend-owned preprocessing.

## Training (frozen; `frozen.training`)

| | Value |
|---|---|
| Steps | 15,000 per backend |
| Batch | 32 windows of 16 transitions (uniform over all train windows, with replacement) |
| Optimizer | the model's AdamW, lr 3e-4, weight decay 1e-4, grad clip 1.0 (constant) |
| Seed | 0 (model init and window sampler) |
| Device | MPS (Apple M5 Pro, 48 GB), one backend at a time: LeWM first, then native |
| Wall-clock cap | 6,000 s per backend, including decoding; hitting it stops with status `time_budget` |
| Validation | every 1,000 steps and at the last step, on the fixed selection cohort |

**Selection rule (val only).**

- **Cohort.** 1,024 val windows, drawn with seed 10001 from the stride-4
  horizon-8 windows.
- **Score.** The median palm–apple readout error at h = 8 on **predicted**
  latents, over the cohort's *moving* windows (defined under G1). Lower is
  better.
- **Eligibility.** Encoded latents must have collapsed fraction ≤ 0.05, mean
  per-dim std ≥ 0.1 and effective rank ≥ 2.
- **Choice.** The best eligible checkpoint is `<backend>.pt`. The last state is
  always saved as `<backend>.latest.pt`.
- **If no checkpoint is eligible.** The run reports `selection_failed`, and the
  gates are evaluated on `.latest.pt` (declared now).

**Disclosure.** The selection cohort and the gate cohort are both val, so
selection is optimistic for G1. Test stays untouched for a later unbiased check.

**Ensemble.** No ensemble and no second seed in this task. The budget is one
seed per backend.

**Artifacts** (git-ignored, under the main checkout):

- `checkpoints/task050-wm-v2/<backend>{.pt,.latest.pt,.run.json,.metrics.jsonl}`
- `outputs/task050-wm-v2/<backend>-val-gates.json`

## Offline gates (frozen; `frozen.gates`; val only)

**Cohorts.**

- **All windows.** Every val window of 16 transitions at start stride 4.
- **Encoding.** Each window's start observation is encoded once. It is then
  rolled out with the window's executed actions. Readouts are taken from the
  predicted latent h steps ahead.
- **Moving windows.** The true palm–apple offset changes by ≥ 1 cm between the
  start and h. Most frames are holds, so on all windows true persistence
  already has a median h = 8 error of 0.9 cm (train labels). An absolute gate
  on all windows would be vacuous.
- **Grasp cohort.** The target frame's collector phase is `close` or `lift`.
- **Approach cohort.** The target frame's phase is `orient` or `descend`.

**Controls.** All controls use the same model and readout head.

- **Persistence.** The readout of the encoded *start* latent.
- **Shuffled actions.** A fixed derangement gives each window the executed
  actions of a window from another episode.
- **Zero actions** are reported only.
- **Siblings.** Each val root's continuation from its branch frame and its
  branches share one pre-grasp state (the labels verify equal starts).
  - All siblings are rolled out from the root's encoded branch-frame
    observation, each with its own executed actions.
  - In the "swapped" control, sibling *i*'s truth is predicted with sibling
    *j*'s actions.

| Gate | Metric at h = 8 unless stated | Threshold |
|---|---|---|
| **G1** | median palm–apple error, moving windows | ≤ 1.5 cm |
| **G2a** | G1 median ÷ persistence-control median (moving) | ≤ 0.8 |
| **G2b** | G1 median ÷ shuffled-action-control median (moving) | ≤ 0.8 |
| **G3** | median apple–plate error, all windows | ≤ 2.0 cm |
| **G4** | median \|apple height error\|, grasp cohort | ≤ 1.0 cm |
| **G5** | AUROC of `apple_held`, grasp cohort (needs ≥ 20 of each class, else fail) | ≥ 0.85 |
| **G6** | cost calibration: median \|ln(predicted ÷ true approach cost)\| on approach-cohort windows with true cost ≥ 2 cm | ≤ ln 1.5 |
| **G7a** | siblings at h = 16: fraction of ordered pairs (true divergence ≥ 1 cm) where own-action error < swapped-action error (needs ≥ 20 pairs, else fail) | ≥ 0.70 |
| **G7b** | siblings at h = 16: Spearman ρ between predicted and true palm–apple divergence over unordered pairs | ≥ 0.5 |
| **G8a** | collapsed fraction (per-dim std < 0.01) of encoded val latents (every 4th frame) | ≤ 0.05 |
| **G8b** | effective rank (exp entropy of the singular-value energy) | ≥ 4 |
| **G8c** | mean per-dim latent std | ≥ 0.1 |

- **Approach cost.** It is the object-aware ceiling's approach term,
  ‖palm_minus_apple − (−1.5, 0, 13) cm‖.
- **Pass rule.** A backend **passes** only if every gate passes.
- **Missing values.** A gate that cannot be evaluated counts as failed.

**Descriptive, non-gating** (reported for both backends):

- all metrics at h = 1, 4 and 16;
- all-window medians;
- the zero-action control;
- the shuffled `apple_held` AUROC;
- contact AUROC;
- approach-cost ratio and Spearman;
- transport-cost calibration (`apple_minus_plate` xy, transfer phase);
- sibling metrics at h = 8 and 32;
- **grasp outcome from the pre-grasp state:** AUROC of the maximum predicted
  `apple_held` over h = 32…64 against the episode's scorer grasp label, for
  siblings with ≥ 64 steps;
- training curves, time and memory.

## Pre-declared readings and next steps

- **LeWM passes all gates →** start TASK-051. It is closed-loop LeWM on the wide
  development resets 45000–45007, with the object-aware phase cost computed from
  `model.readout` of predicted latents, against shuffle and replay controls.
  Passing is necessary evidence, not success.
- **LeWM fails only precision gates (G1, G3, G4, G6) but passes G2, G7 and
  G8 →** the dynamics use the actions but readouts are too coarse. Next: the
  `hand_crop_rgb` arm (a two-camera or crop-only model), longer training or a
  larger encoder, preregistered anew.
- **LeWM fails G2 or G7 →** the prediction does not depend on the actions
  enough to plan with. **Do not** start closed-loop. Next: redesign the action
  conditioning (for example action-chunk tokens, or stronger multistep
  weighting) and rerun this protocol.
- **LeWM fails G8 →** representation collapse. Fix the objective before
  anything else.
- **Native passes and LeWM fails →** report both honestly. Whether to proceed
  with native is a product decision for the user. The backend is not swapped
  silently.
- **Either way,** no failed run is re-tuned against val and re-reported as the
  frozen result. Any rerun is a new, disclosed protocol version.

## Not done in this task (declared)

- **`hand_crop_rgb`.** No second camera arm. It remains the first follow-up if
  the precision gates fail.
- **Ensembles and extra seeds.** Not run.
- **Test split and closed-loop control.** Neither is touched.

## Pilots before freezing (disclosed)

All pilots are under `outputs/task050-scratch/` and were run from uncommitted or
draft code:

- **`smoke-a`.** 8 train and 8 val episodes, 60 steps, plus a smoke of
  `evaluate` on those 8 val episodes. It checked the pipeline only.
- **`smoke-full`.** Full decode (11.5 s for 757 episodes, peak host RSS
  9.0 GB) and 200 steps per backend. It measured about 0.23 s per LeWM step
  and about 0.3 s per native step.
  - The encoded-latent effective rank fell from about 12–15 at initialization
    to 2.2–3.2 by step 200 in both backends.
  - Because of this, the selection-eligibility rank floor was set to 2 (commit
    `17bf5dd`), below gate G8b's 4, so that selection does not fail wholesale
    early in training.
  - G8b itself was not changed.
- **`pilot-a`.** LeWM, 2,000 steps, run after the gate thresholds were
  committed in `17bf5dd`. Its only purpose was checking latent-rank behaviour
  and wall-clock over a longer run. Its val selection metric was visible to the
  author. No threshold was changed after it.
