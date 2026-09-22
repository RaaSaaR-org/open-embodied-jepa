# Apple world model v3: preregistration (TASK-052)

This is the prospective protocol for the follow-up to TASK-050. It is committed before
the frozen training runs. **Everything here is OFFLINE evaluation of learned models on
recorded validation data.** No closed-loop control is run, and no number here is a
learned Apple→Plate result: learned Apple→Plate stays at **0 successes**.

The frozen values live in `benchmarks/manifests/apple-world-model-v3.json` (`frozen`).
The code is `src/embodied_jepa/world_model_v3.py`, the generalized
`src/embodied_jepa/world_model_v2.py` runner, `src/embodied_jepa/models/base.py`,
`src/embodied_jepa/models/readout.py` and `src/embodied_jepa/readout_labels.py`.

## What TASK-050 measured, and what this attacks

All three v2 arms failed (`apple_world_model_v2_results.md`). The failure was not
diffuse:

| Measurement (LeWM/onboard, val, h = 8) | Value |
|---|---|
| palm–apple error, **all** valid windows | 0.77 cm |
| palm–apple error, **moving** windows (1,462 of 3,961) | **3.65 cm** (gate ≤ 1.5 cm) |
| readout of a **directly encoded** target-frame observation, same windows | 3.26 cm |
| ratio to the model's own persistence readout (G2a) | 0.835 (gate ≤ 0.8) |
| ratio to the shuffled-action control (G2b) | 0.699 **pass** |
| cross-window Spearman of predicted vs true approach cost | 0.16 |

About **89 % of the rollout error is already in the encoding**, the actions clearly carry
information (G2b and the sibling-divergence control pass), and the planning-relevant
number — how well the predicted cost *orders* candidate actions — was never gated.

v3 therefore changes three things and measures a fourth:

1. **Readout precision under motion.** A two-camera model, a readout loss weighted
   towards moving frames, auxiliary absolute-position targets, a wider readout head, and
   a finer encoder patch grid.
2. **The gate that matters for CEM.** v2's absolute cost-calibration gate G6 is replaced
   by a within-state ranking gate over sibling candidate actions. v2's calibration number
   is still reported, descriptively.
3. **The proprioception scale defect** the TASK-050 review raised.
4. Everything that already worked is kept: the persistence, shuffled-action, zero-action
   and sibling-swap controls, the collapse diagnostics, and the sealed test split.

## Data

Identical to TASK-050. Nothing about the corpus or the splits changes.

- **Corpus.** `data/apple-wide-v1` (TASK-048), dataset manifest SHA-256
  `028e130576c052437f7753d74dd64d80dabc1edb8085247e7f56bc71a0412184`. 797 episodes
  (200 wide-jitter roots, 597 grasp-phase branches), 205,519 transitions, 20 fps.
- **Splits (whole resets, frozen by TASK-048).** train 677 / val 80 / test 40 episodes
  (170 / 20 / 10 resets). Updates use **train** only. Checkpoint selection and every gate
  use **val**. **test is never decoded** (the loader refuses it; every report records
  `test_episodes_decoded: 0`).
- **Model inputs.** `onboard_rgb` and/or `hand_crop_rgb` at 112×112, the 86-D
  proprioception and the executed 14-D normalized actions. `hand_crop_rgb` is the
  FK-placed crop of TASK-048: its window comes from robot kinematics alone
  (`src/embodied_jepa/hand_crop.py`), so it is a legal model input.
- **Privileged labels.** Read with `acknowledge_privileged_training_labels=True`, and
  used for exactly three things, all declared here:
  - **readout-head training targets;**
  - **the readout loss's per-frame weight** (see below) — new in v3, and disclosed as a
    change of policy from v2, which declared labels were never a training-sample weight.
    They still never reach a model input, a planner input or a planning cost, and they
    still never filter which windows are sampled;
  - **val scoring references and cohorts** (moving / grasp / lift / approach windows, the
    dropped-apple exclusion, the collector phase index, and the sibling grouping).

## Models

### Shared, backend-agnostic changes (`models/base.py`, `models/readout.py`)

Each is off by default, so every earlier model keeps its exact loss and latent.

- **Two cameras (`cameras: [onboard_rgb, hand_crop_rgb]`).** Each camera is embedded by
  the backend's *own* encoder, then passed through its own learned
  `Linear(latent_dim, latent_dim)` (only the first carries a bias) and summed. That is a
  concatenate-then-project fusion written once in shared code. **No backend file has a
  two-camera branch**, so `world_model.backend` stays a one-line swap. With one camera no
  fusion module is built and the image pathway is exactly v2's `embed(pixels)`.
  - Native's EMA target path embeds each camera with the target encoder and applies the
    *online* fusion projections under `no_grad`, exactly as v2 treated state fusion.
  - The fusion weights are drawn **last** in the initialization stream, so a one-camera
    and a two-camera model with the same seed share every other weight bit for bit
    (verified by a test). The camera ablation therefore differs by the second camera
    alone, not by a shifted random stream.
- **Motion-weighted readout loss (`readout_moving_weight: 3.0`,
  `readout_moving_threshold_m: 0.01`).** A frame whose true palm–apple offset has moved
  ≥ 1 cm from its window's first frame gets regression weight 4; a still frame keeps 1.
  The threshold is the gate cohort's own definition. Probability heads are left
  unweighted: they are already near-perfect and are not the failure. The weight comes
  from the targets, so it is a label-derived loss weight (disclosed above).
- **Auxiliary position readouts (`readout_auxiliary_weight: 0.5`).** Two new regression
  readouts, `apple_position` and `palm_position`, both relative to a fixed declared
  origin `(0.30, 0.00, 0.75) m`. Supervising the apple's absolute position forces the
  latent to localize the object rather than only the palm-relative offset that the
  proprioception branch can partly explain. At weight 0 they contribute no loss term and
  no gradient, are not declared in `Capabilities.readouts`, and are not returned by
  `model.readout` — an untrained head is never exposed.
- **Wider readout head (`readout_hidden_dim: 512`, v2 used 256).** The head runs on
  latents, not pixels; the measured cost is negligible. It applies to every v3 arm
  equally.
- **Proprioception scaling fix.** v2 floored every dimension's train standard deviation
  at 0.01. 43 of the 86 dimensions have a train std below that (the smallest is
  1.7e-6), so a later deviation in those dimensions entered the model up to 100× larger
  than in a normally varying dimension. The TASK-050 review flagged this before any
  closed-loop run. v3 scales a dimension with train std < 0.01 by **1.0 (its physical
  unit)** instead, and clips every normalized value to ±10. On the train split both are
  inert; in a later closed loop an unseen deviation enters at physical scale instead of
  amplified. The policy and the number of such dimensions are recorded in the checkpoint
  metadata. **This applies to all four arms**, so it does not confound the comparisons
  *within* v3; it is one of the reasons a v3 arm is not a perfectly controlled successor
  of a v2 arm.

**Honest label.** Every arm is the pinned upstream LeWM encoder, predictor, projector and
SIGReg objective (revision `8edfeb33`) **plus** the shared state fusion, the shared
camera fusion and the auxiliary readout loss. That is not the unmodified LeWM objective.

### Arms (frozen)

All four are the **LeWM** product backend, one seed, run once each, sequentially.

| Arm | Config | Cameras | Readout | Patch | Role |
|---|---|---|---|---|---|
| **A** | `apple_wm_v3_lewm.yaml` | onboard + hand crop | v3 | 14 | **primary** |
| **B** | `apple_wm_v3_lewm_onboard.yaml` | onboard | v3 | 14 | camera ablation |
| **C** | `apple_wm_v3_lewm_v2_readout.yaml` | onboard + hand crop | uniform, no auxiliary | 14 | readout ablation |
| **D** | `apple_wm_v3_lewm_fine.yaml` | onboard | v3 | **8** | encoder-resolution ablation |

Arm C keeps v3's *wider* readout head and the proprioception fix and drops only the
motion weighting and the auxiliary targets, so its name refers to v2's loss shaping,
not to a v2 model.

Each ablation is **one factor** from its neighbour: A↔B is the second camera, A↔C is the
readout shaping, B↔D is the encoder's spatial resolution (patch 8 gives 14×14 = 196
tokens per camera instead of 8×8 = 64). The missing cell of the A/B/C design (one camera,
v2 readout) is TASK-050's LeWM/onboard arm, which is **not** a controlled member of this
design: it predates the readout-head width, the proprioception fix and this code
revision, and its number is quoted for orientation only.

`apple_wm_v3.yaml` is the native backend on the same shared settings; the LeWM configs
are `extends:` plus `world_model.backend`. **The native backend is not a frozen arm of
this protocol** — the budget below does not fit a fifth run. It is exercised only by the
smoke, so the one-line-swap invariant is demonstrated as a software property rather than
as a second trained comparison. TASK-050's native comparison stands.

| | LeWM arms A, B, C | LeWM arm D |
|---|---|---|
| Image size | 112 | 112 |
| Patch / tokens | 14 → 8×8 = 64 | 8 → 14×14 = 196 |
| Encoder | ViT, depth 4, 4 heads, dim 128 | the same |
| Predictor | upstream AR predictor, depth 4, 4 heads × 32 | the same |
| Anti-collapse | SIGReg (weight 0.09) | the same |
| Parameters | 2,431,782 (A, C) / 2,398,886 (B) | 2,365,094 |

## Training (frozen; `frozen.training`)

| | Value |
|---|---|
| Steps | 15,000 per arm (the same budget as TASK-050) |
| Batch | 32 windows of 16 transitions (uniform over all train windows, with replacement) |
| Optimizer | the model's AdamW, lr 3e-4 with cosine decay to 3e-5 over the 15,000 steps, weight decay 1e-4, grad clip 1.0 |
| Seed | 0 (model init and window sampler) |
| Device | MPS (Apple M5 Pro, 48 GB), one run at a time, in the order A, B, C, D |
| Wall-clock cap | 10,800 s per run, including decoding; hitting it stops with status `time_budget` |
| Validation | every 1,000 steps and at the last step, on the fixed selection cohort |

**Measured budget** (from the disclosed smokes, steady-state seconds per update × 15,000):
A 1.45 h, B 0.96 h, C 1.45 h, D 2.02 h — **about 5.9 h of MPS training in total**, plus
about 15 s of decoding and about 1 min of evaluation per arm. The 10,800 s cap is a
runaway guard with about 48 % headroom on the slowest arm, not the expected time.
Two-camera decoding holds both cameras in memory: peak host RSS 16.0 GB, against 9.2 GB
for a one-camera arm.

**Longer training is not attempted.** The budget above is already the task's ceiling.
Every arm's validation curve (16 points) is reported, so whether the step budget binds is
answerable descriptively, and a longer run is one of the pre-declared next steps.

**Selection rule (val only).** Unchanged from TASK-050, so the selection score stays
comparable across the two protocols.

- **Cohort.** 1,024 val windows, drawn with seed 10001 from the stride-4 horizon-8
  windows.
- **Score.** The median palm–apple readout error at h = 8 on **predicted** latents, over
  the cohort's *moving* windows. Lower is better.
- **Eligibility.** Encoded latents must have collapsed fraction ≤ 0.05, mean per-dim std
  ≥ 0.1 and effective rank ≥ 2.
- **Choice.** The best eligible checkpoint is the arm's `.pt`. The last state is always
  saved as `.latest.pt`. The untrained step-0 state is validated and logged but is never
  selectable. If no checkpoint is eligible the run reports `selection_failed` and the
  gates are evaluated on `.latest.pt` (declared now).

**Disclosure.** Selection and the gates both use val, so G1 is flattered. Test stays
untouched for a later unbiased check.

**Frozen commands.** Each runs from a clean committed worktree; `--require-clean` refuses
a dirty tree.

```sh
uv run --no-sync python -m embodied_jepa.world_model_v3 train \
  --config configs/apple_wm_v3_lewm.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v3.json \
  --device mps --workers 8 --require-clean --acknowledge-privileged-training-labels
# arm B: --config configs/apple_wm_v3_lewm_onboard.yaml
# arm C: --config configs/apple_wm_v3_lewm_v2_readout.yaml
# arm D: --config configs/apple_wm_v3_lewm_fine.yaml
uv run --no-sync python -m embodied_jepa.world_model_v3 evaluate \
  --config configs/apple_wm_v3_lewm.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v3.json --device mps \
  --acknowledge-privileged-training-labels \
  --output outputs/task052-wm-v3/leworldmodel-val-gates.json
```

**Artifacts** (git-ignored, under the main checkout):

- `checkpoints/task052-wm-v3/<name>{.pt,.latest.pt,.run.json,.metrics.jsonl}`
- `outputs/task052-wm-v3/<name>-val-gates.json`

## The new measurement: within-state candidate ranking

A CEM never compares one state's cost to another state's. It takes **one** state, rolls
out several candidate action sequences, and picks the best. v2 gated the wrong thing: an
absolute calibration of the predicted cost across unrelated windows. v3 gates the ranking.

- **Candidate sets.** Each val root contributes one pre-grasp state and the executed
  action sequences of its continuation and its three branches — 20 groups of 4 in val.
  Every candidate is rolled out from the **same encoded observation** of the anchor's
  branch frame, so the only thing that differs is the actions. The labels verify that the
  siblings' start states coincide (`start_state_max_label_mismatch_m`).
- **Cost.** The object-aware ceiling's approach term,
  ‖`palm_minus_apple` − (−1.5, 0, 13) cm‖, from the readout of the predicted latent.
- **Ranked groups.** A group counts only when it has ≥ 3 candidates whose apple is not
  dropped at the horizon and whose **true** costs span ≥ 5 mm. Below that the candidates
  are physically indistinguishable and no ranking exists to test.
- **Horizon.** h = 16, reported also at h = 8 and h = 32. The horizon was chosen from
  **val labels alone, before any v3 model existed** (disclosed): at h = 16, 18 of 20
  groups clear the 5 mm spread, against 10 of 20 at h = 8. h = 16 also matches the
  existing sibling gate horizon. The planner's own horizon is 8; this is a limitation of
  the val cohort and is recorded as such.
- **Statistics.**
  - `within_state_spearman_pooled`: each group's predicted and true costs are converted
    to within-group ranks in [0,1] and pooled across groups; the Spearman of the pooled
    vectors. This is the headline. (The mean of per-group Spearman over 4-candidate
    groups is also reported, and is noisier.)
  - `top1_regret`: the true cost of the candidate the model ranks best, minus the best
    true cost, in metres. Median and mean; also normalized by the group's true spread.
- **Controls.**
  - **Shuffled actions**: candidate *i* ranked by the prediction made with candidate
    *i*+1's actions. Same rollouts, no extra compute. Expected ρ ≈ 0.
  - **Persistence cannot be reported here at all**, because the persistence readout is
    identical for every candidate of a state. That is precisely why an absolute
    calibration gate could not test this, and it is reported as such.
  - **Random choice**: the expected regret of picking uniformly at random, computed from
    val **labels only**. Measured before any v3 model existed: median 8.44 mm absolute,
    0.531 normalized, over the 18 ranked groups at h = 16.
- **Power.** 18 groups of about 3.8 candidates, about 68 pooled points. This is thin, and
  a pass is **necessary, not sufficient** evidence. It is recorded here in advance.

## Offline gates (frozen; `frozen.gates`; val only)

G1–G5 and G7–G8 are TASK-050's thresholds, **copied verbatim** so the two protocols are
directly comparable. G6 is the replacement described above.

| Gate | Metric | Threshold |
|---|---|---|
| **G1** | median palm–apple error at h = 8, moving windows | ≤ 1.5 cm |
| **G2a** | G1 ÷ persistence-control median (moving) | ≤ 0.8 |
| **G2b** | G1 ÷ shuffled-action-control median (moving) | ≤ 0.8 |
| **G3** | median apple–plate error at h = 8, all valid windows | ≤ 2.0 cm |
| **G4** | median \|apple height error\| at h = 8, grasp cohort | ≤ 1.0 cm |
| **G5** | AUROC of `apple_held` at h = 8, lift cohort (needs ≥ 20 of each class) | ≥ 0.85 |
| **G6a** | pooled within-state ranking Spearman at h = 16 (needs ≥ 12 ranked groups) | ≥ 0.5 |
| **G6b** | median top-1 regret at h = 16 (same cohort) | ≤ 4 mm |
| **G7a** | siblings at h = 16: fraction of ordered pairs (true divergence ≥ 1 cm) where own-action error < swapped-action error (needs ≥ 20 pairs) | ≥ 0.70 |
| **G7b** | siblings at h = 16: Spearman between predicted and true palm–apple divergence | ≥ 0.5 |
| **G8a** | collapsed fraction (per-dim std < 0.01) of encoded val latents | ≤ 0.05 |
| **G8b** | effective rank | ≥ 4 |
| **G8c** | mean per-dim latent std | ≥ 0.1 |

- **G6b's threshold** is below half the label-derived random-pick baseline (8.44 mm). It
  was set from that baseline, which is a property of the val labels and of no model.
- **Pass rule.** An arm **passes** only if every gate passes.
- **Missing values.** A gate that cannot be evaluated counts as failed.

**Descriptive, non-gating** (reported for every arm): all metrics at h = 1, 4 and 16;
all-window medians; the zero-action control; **v2's absolute approach-cost calibration
(G6 of the old protocol) and its cross-window Spearman**, so the protocols stay
comparable; the shuffled-action `apple_held` AUROC; the grasp-cohort AUROC; contact
AUROC; `apple_dropped` AUROC; image-only collapse statistics; transport-cost calibration;
sibling metrics at h = 8 and 32; grasp outcome from the pre-grasp state (AUROC of the
maximum predicted `apple_held` over h = 32…64); the ranking metrics at h = 8 and 32; and
training curves, time and memory.

## Pre-declared readings and next steps

The decision rule is applied as written, not re-interpreted afterwards.

- **Arm A passes every gate →** the closed loop becomes the next task (TASK-053): LeWM
  with the object-aware phase cost computed from `model.readout` of predicted latents, on
  the wide development resets 45000–45007, against shuffle and replay controls. Passing
  is necessary evidence, not success.
- **An ablation arm passes and A does not →** that arm becomes the closed-loop candidate,
  and A's failure is reported unchanged. Any pass is still a pass of *this* protocol.
- **No arm passes → the closed loop does not start**, and the next protocol is chosen by
  the earliest failing group, in this order:
  1. **G8 fails (collapse)** → fix the objective before anything else.
  2. **G2b or G7 fails (action conditioning)** → the prediction does not use the actions.
     Redesign the conditioning (action-chunk tokens, stronger multistep weighting) and
     rerun. Do **not** attack readouts again.
  3. **G1/G3/G4 fail but G2a improves materially over v2's 0.835 →** readout precision is
     moving in the right direction and the attack was correct but insufficient. Next:
     **more of the same, bigger** — longer training and a larger encoder at the
     resolution the arm-B↔D comparison favours, preregistered anew.
  4. **G1 fails and G2a does not improve over v2 (≥ 0.8), while G2b and G7 still pass →**
     the models keep failing to beat their own persistence readout even with two cameras,
     motion weighting and a finer grid. The evidence would then point at the **data**, not
     the architecture: 112 px onboard frames of a ~2 cm apple, recorded by a scripted
     collector, may simply not determine the palm–apple offset to 1.5 cm. Next: measure
     the information ceiling directly (train a readout on a *single* frame at 224 px and
     at the native render resolution, no dynamics at all) before spending more on models.
  5. **G6a/G6b fail while G1 passes →** the readouts are precise enough but the cost they
     imply does not order candidates. That is an argument about the **planning
     formulation**, not the model.
- **Plainly, in advance:** if outcome 4 or 5 holds — that is, if after v3 the model still
  cannot beat its own persistence readout under motion, or can read out positions
  accurately yet still cannot rank candidate actions — then **CEM over this cost is
  probably the wrong control formulation for this task**, and the next task should test a
  learned policy instead: behaviour cloning on the scripted corpus, with the world model
  used as a critic or as a residual, rather than as the forward model of a sampling
  planner. The corpus already contains the expert actions that such a policy needs. This
  reading is committed now so that it cannot be read as a post-hoc excuse.
- **Either way,** no failed run is re-tuned against val and re-reported as the frozen
  result. Any rerun is a new, disclosed protocol version.

## Not done in this task (declared)

- **The closed loop.** Not started, under any outcome of this protocol.
- **The test split.** Never decoded.
- **The native backend as a frozen arm.** Only smoked.
- **Ensembles, extra seeds, longer training.** Not run.
- **An apple-pixel auxiliary target.** Considered and dropped: projecting the apple into
  the onboard image needs the camera's per-frame extrinsics, which the corpus does not
  record. The absolute world-position targets are the substitute.
- **The hand-crop window as a model input.** Considered and dropped: `robot__hand_crop_window`
  is a legal robot-derived label, but feeding it would add a per-corpus model input
  outside the canonical `Episode`, and it is already a deterministic function of the
  proprioception the model receives.

## Pilots before freezing (disclosed)

All pilots are under `checkpoints/task052-scratch/` and `outputs/task052-scratch/`. In
the order they were run, all from the committed revision `63f9cd2`:

- **Label-only cohort measurements** (no model, before any training): the sibling
  candidate spread at h = 8/16/32/64 and the random-pick regret baseline. These set the
  ranking gate's horizon, its 5 mm spread rule and G6b's threshold, and are quoted above.
- **`smoke-a`.** Arm A on 8 train and 8 val episodes, 40 steps, plus `evaluate` on 12 val
  episodes. Plumbing only. It confirmed the two-camera path, the v3 gate wiring and that
  the ranking metric is computable end to end.
- **`smoke-b`.** Arm A, full corpus, 200 steps. It measured decoding (16.8 s for both
  cameras), peak host RSS (16.0 GB) and 0.348 s per update. Its val selection score at
  steps 0 and 200 (0.197 m and 0.0889 m) **was visible to the author**. No threshold was
  changed after seeing it; every gate threshold is either copied from TASK-050 or set
  from the label-only measurements above.
- **`smoke-onboard`, `smoke-fine`.** Arms B and D, full corpus, 120 steps, for wall clock
  only: 0.229 s and 0.486 s per update. These fixed the arm budget table and the
  10,800 s cap.
- **One bug found by the smokes and fixed before freezing:** the run report counted
  cameras instead of observations (`train_observations: 2`) after the runner became
  multi-camera.
