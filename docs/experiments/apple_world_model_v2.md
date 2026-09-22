# Apple world model v2: preregistration (TASK-050)

This is the prospective protocol for roadmap step T3. It is committed before the
frozen training runs. **Everything here is OFFLINE evaluation of learned models on
recorded validation data.** No closed-loop control is run, and no number here is a
learned Apple→Plate result: learned Apple→Plate stays at 0 successes.

**Results: `apple_world_model_v2_results.md`. All three arms failed the gate
set, and the closed loop was not started.**

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

The runs are the LeWM product backend on the onboard camera (primary), the
same backend on the hand crop (secondary), and the native backend on the
onboard camera (comparison). Every arm is judged by the same gates.

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
  - `hand_crop_rgb` (the 112 px robot-kinematics crop) is the **secondary
    preregistered arm**: one more LeWM run that differs only in the camera.
- **Privileged labels.** The gated sidecars are read with
  `acknowledge_privileged_training_labels=True`. They serve only two roles:
  - **readout-head training targets;**
  - **val scoring references.**
  They are never a model input, a training-sample weight or a training-sampling
  filter. Two label-derived masks are part of the target and scoring
  definitions, and are declared here:
  - the **regression target validity mask**: regression readouts carry no loss
    on frames whose apple has fallen off the table (see below);
  - the **val scoring cohorts**: moving/grasp/lift/approach windows, the
    exclusion of dropped-apple windows and the collector phase index. The
    selection score uses the same moving-window definition, so val truth
    shapes the *selection* score as it shapes the gates. This is val scoring,
    never a train input.

### Readout targets (`readout_labels.targets`)

| Readout | Width | Definition |
|---|---|---|
| `palm_minus_apple` | 3 | palm site minus apple centre, world metres |
| `apple_height` | 1 | apple z minus its resting z at the reset (the root's first frame) |
| `apple_minus_plate` | 3 | apple centre minus plate centre, world metres |
| `hand_contact` | 1 | hand/wrist–apple contact (probability head) |
| `apple_held` | 1 | contact **and** `apple_height ≥ 2 cm` (probability head) |
| `apple_dropped` | 1 | the simulator's own drop rule, apple centre below 0.70 m (probability head) |

- **Dropped apples.** A failed grasp often knocks the apple off the table. In
  the train split 32,961 of 175,451 frames (19 %, in 242 episodes) have
  `apple_dropped`. There the palm–apple offset is up to ~2.6 m and the apple is
  outside the workspace and the camera view.
  - The pilot `pilot-a` (below) showed these frames dominating both the
    regression loss and the moving-window metric.
  - The regression readouts (`palm_minus_apple`, `apple_height`,
    `apple_minus_plate`) are therefore **masked out of the loss** where
    `apple_dropped` is 1.
  - `apple_dropped` itself is a declared readout trained on every frame, so a
    planner can detect a lost apple.
- **Rest height.** The rest height is the root's first-frame apple z, the spawn
  height. The apple settles about 3.4 mm lower within 5 frames. A resting apple
  therefore has `apple_height ≈ −3.4 mm`, and `apple_held` effectively needs a
  rise of about 2.34 cm above the settled apple.
- **Phase index.** The collector phase at an observation is the phase of the
  command issued there. The final observation row repeats the last phase.

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
  latent to the six readouts.
  - Loss: smooth-L1 on 5 cm-scaled regressions (masked where the apple is dropped) and BCE on the probabilities.
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

**Other notes.**

- **Native target path.** Native's EMA target path adds the *online* state
  embedding under `no_grad`; there is no EMA copy of the state encoder.
- **Checkpoint compatibility.** `implementation_sha256` now also covers
  `models/readout.py` and `readout_labels.py`. With the edit to `base.py`,
  every historical checkpoint is incompatible with this code. That is the
  repository's policy, and the historical results stay tied to their
  revisions.

**Honest label.** The LeWM arm is the pinned upstream encoder, predictor,
projector and SIGReg objective, **plus** the shared state fusion and the
auxiliary readout loss. That is not the unmodified LeWM objective. The native
arm receives exactly the same additions.

### Backends (`configs/apple_wm_v2.yaml`, `configs/apple_wm_v2_lewm.yaml`)

The LeWM config is `extends: apple_wm_v2.yaml` plus `world_model.backend:
leworldmodel`. That is the one-line swap. The secondary arm
`apple_wm_v2_lewm_hand_crop.yaml` extends the LeWM config and changes only the
camera (and the checkpoint path). The shared keys are identical for both
backends: `camera onboard_rgb`, `latent_dim 128`, `hidden_dim 256`,
`max_horizon 64`, `state_fusion`, `readout_heads`, `readout_weight 1.0` and
`readout_hidden_dim 256`.

| | LeWM (`leworldmodel`) | native (`native_jepa`) |
|---|---|---|
| Image size | 112 (patch 14 → 8×8 tokens) | 128 (see below) |
| Encoder | ViT, depth 4, 4 heads, dim 128 | conv stride-8 → 4×4 pool → MLP |
| Predictor | upstream AR predictor, depth 4, 4 heads × 32 | residual MLP |
| Anti-collapse | SIGReg (weight 0.09) | variance + covariance, EMA target |
| Parameters | 2,163,360 | 931,402 |

The parameter counts include the `apple_dropped` head added with the
dropped-apple mask before the freeze (257 parameters more than the counts first
written down when the configs were sized).

The native encoder's stride-8 grid must divide its 4×4 adaptive pool on MPS.
It therefore upsamples the 112 px camera to 128 px. No resolution is discarded.
The difference is backend-owned preprocessing.

## Training (frozen; `frozen.training`)

| | Value |
|---|---|
| Steps | 15,000 per backend |
| Batch | 32 windows of 16 transitions (uniform over all train windows, with replacement) |
| Optimizer | the model's AdamW, lr 3e-4 with cosine decay to 3e-5 over the 15,000 steps (runner-owned, identical for both backends), weight decay 1e-4, grad clip 1.0 |
| Seed | 0 (model init and window sampler) |
| Device | MPS (Apple M5 Pro, 48 GB), one run at a time, in this order: LeWM/onboard, LeWM/hand-crop, native/onboard |
| Wall-clock cap | 6,000 s per run, including decoding; hitting it stops with status `time_budget` |
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
- **Step 0.** The untrained step-0 state is validated and logged but is never
  selectable.
- **If no checkpoint is eligible.** The run reports `selection_failed`, and the
  gates are evaluated on `.latest.pt` (declared now).

**Disclosure.** The selection cohort and the gate cohort are both val, so
selection is optimistic for G1. Test stays untouched for a later unbiased check.

**Ensemble.** No ensemble and no second seed in this task. The budget is one
seed per backend.

**Frozen commands.** Both run from a clean committed worktree;
`--require-clean` refuses a dirty tree.

```sh
uv run --no-sync python -m embodied_jepa.world_model_v2 train \
  --config configs/apple_wm_v2_lewm.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v2.json \
  --device mps --workers 10 --require-clean --acknowledge-privileged-training-labels
# secondary arm: --config configs/apple_wm_v2_lewm_hand_crop.yaml
# comparison:   --config configs/apple_wm_v2.yaml
uv run --no-sync python -m embodied_jepa.world_model_v2 evaluate \
  --config configs/apple_wm_v2_lewm.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v2.json --device mps \
  --acknowledge-privileged-training-labels \
  --output outputs/task050-wm-v2/leworldmodel-val-gates.json
# native: --config configs/apple_wm_v2.yaml --output .../native_jepa-val-gates.json
```

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
- **Valid windows.** Every cohort excludes windows whose apple is dropped at the
  start or at the target. Sibling pairs are excluded when either sibling's
  apple is dropped at h.
- **Grasp cohort.** The target frame's collector phase is `close` or `lift`.
- **Lift cohort.** The target frame's collector phase is `lift`.
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
| **G3** | median apple–plate error, all valid windows | ≤ 2.0 cm |
| **G4** | median \|apple height error\|, grasp cohort | ≤ 1.0 cm |
| **G5** | AUROC of `apple_held`, lift cohort (needs ≥ 20 of each class, else fail) | ≥ 0.85 |
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
- the shuffled-action `apple_held` AUROC (lift cohort);
- the grasp-cohort (`close` + `lift`) `apple_held` AUROC;
- contact AUROC;
- `apple_dropped` AUROC over all windows;
- collapse statistics of the **image pathway alone** (before state fusion),
  so the proprio embedding cannot mask an image collapse;
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
  G8 →** the dynamics use the actions but readouts are too coarse. Next: a
  two-camera model, longer training or a larger encoder, preregistered anew.
- **The hand-crop arm passes gates the onboard arm fails →** TASK-051 uses the
  hand-crop model, and the onboard result is still reported. The same rule
  holds in reverse: a hand-crop failure does not weaken an onboard pass.
- **No arm passes →** T4 does not start. The failed gates and the pre-declared
  reading for the earliest failed group decide the next protocol.
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

- **Two cameras at once.** Each run uses one camera. No two-camera model is
  built here.
- **Ensembles and extra seeds.** Not run.
- **Test split and closed-loop control.** Neither is touched.

## Pilots before freezing (disclosed)

All pilots are under `outputs/task050-scratch/` and were run from uncommitted or
draft code:

- **`smoke-a`.** 8 train and 8 val episodes, 60 steps, plus a smoke of
  `evaluate` on those 8 val episodes. It checked the pipeline only.
- **`smoke-b` and `smoke-c`.** 12 train and 12 val episodes, 40 steps, one per
  backend, run from the reviewed revision with `--require-clean` plus the
  matching `evaluate`. They checked the frozen command path only; `smoke-c`
  also caught the NumPy-float learning rate described below.
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
  committed in `17bf5dd`. It checked latent-rank behaviour and wall-clock over
  a longer run. Its val selection metric was visible to the author.
  - Rank recovered to about 4.4–4.7 by steps 1,500–2,000.
  - The moving-window h = 8 error stayed at about 25 cm, and so did its own
    persistence control.
  - The cause was the dropped-apple frames described above. This led to the
    regression validity mask, the `apple_dropped` readout and the exclusion of
    dropped windows from the scoring cohorts.
- **Pre-run review.** A fresh-context review then found one blocker: the step-0
  state was selectable. It is fixed. On the review's advice G5 moved from the
  close+lift cohort to the **lift** cohort, because every close-phase frame is
  a trivial negative. The grasp-cohort AUROC remains descriptive. The review
  also asked for:
  - image-only collapse statistics;
  - a clean-tree refusal;
  - checkpoint-versus-corpus provenance checks in `evaluate`;
  - clearer wording on label-derived masks.
  All were added. The cosine learning-rate decay was added at the same time,
  before any frozen run.
- **After the review cleared the code**, one more fix was needed: the
  scheduled learning rate was a NumPy float, which `torch.load(weights_only=True)`
  refuses. It is cast to a Python float (`293b331`), with a test. The secondary
  hand-crop arm was preregistered at the same time, before any frozen run.
- **`pilot-b`.** LeWM, 2,000 steps with the mask and constant lr, run after
  these changes.
  - The h = 8 moving-window median was 7.4–8.3 cm from step 500 on, against
    persistence at 7.8–9.2 cm and shuffled actions at 9.1–11.1 cm.
  - The lift-cohort held AUROC was 0.95–1.00.
  - Effective rank was 4.2–5.6.
  - These val numbers were visible. **No gate threshold was changed after
    either pilot.** Every threshold is the value committed in `17bf5dd`.
    Two *definitions* did change after a pilot, both disclosed above: the
    dropped-window exclusion and the regression mask (after `pilot-a`), and
    G5's cohort (close+lift → lift, from the review, not from a result).
