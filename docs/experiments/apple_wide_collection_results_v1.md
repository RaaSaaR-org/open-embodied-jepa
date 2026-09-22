# Apple wide-jitter TRAIN corpus v1: results (TASK-048)

**Every preregistered acceptance check passed (A1–A13).** The corpus is
accepted as the TASK-050 world-model-v2 training corpus.

It was produced by the **privileged scripted collector** with injected
perturbations and grasp-phase branches. No number here is a learned-control
result: learned Apple→Plate stays at zero successes. The protocol is
`apple_wide_collection_v1.md`.

## Run

| | |
|---|---|
| Command | the frozen command, executed once |
| Code | clean tracked checkout `32865f57a69655c574de444030204f6be561998c` (docs-only on top of the reviewed collector `d023a6b`); `tracked_tree_dirty: false` |
| Collector SHA-256 | `fa3ef67d60ca24c1ff1fb1c054b90a992000c24e61047d46573e48fd93b17952` |
| Protocol SHA-256 | `bd27f11fc2f197e1982226487e0d151f3c856dc2a3a250d56af8c6f94d82da52` |
| Plan SHA-256 | `15ed1a99e45114a5cec6013d345804ec561fad859dc3f0dd89dd93ec1e33062c` (the frozen value) |
| Runtime | Python 3.12.13, macOS 26.5.1 arm64, NumPy 2.5.3, MuJoCo 3.13.0; `uv.lock` `23ad3e6e…09b9` |
| Budget use | 12 CPU workers, one thread each, all exit 0, no supervisor timeout |
| Time | collection 2,346 s; assembly 883 s; audit and acceptance included in 3,345 s total |
| Per-root time | 34–216 s (mean 133 s under 12-way contention) |

Outputs are git-ignored under the main checkout:

- **Corpus.** `data/apple-wide-v1/`: 3.40 GB (3,403,248,236 bytes). Dataset
  manifest SHA-256
  **`028e130576c052437f7753d74dd64d80dabc1edb8085247e7f56bc71a0412184`**.
- **Shards, plan, provenance, worker logs and `collection_report.json`.**
  `data/apple-wide-v1-work/` (2.9 GB).

## Corpus

| | Root | Branch | Total |
|---|---|---|---|
| Stored episodes | 200 / 200 | 597 / 600 | 797 |
| Transitions | 114,107 | 91,412 | 205,519 |

Three planned branches were not stored because they stopped before 8
commands, all by `guard_refused`: `wide-48078-b2-shift_close` (7), `wide-48187-b2-noise_only`
(7) and `wide-48167-b1-noise_only` (6).

- **Streams.** `onboard_rgb` 112×112×3 and `hand_crop_rgb` 112×112×3 (PNG in
  Parquet), 86-D proprioception, 14-D applied actions and physical actions,
  at 20 fps.
- **Frame intervals.** Every interval is 0.05 s (audit min/max
  0.049999999999972 / 0.050000000000061).

## Outcomes (privileged scorer labels)

| | Count |
|---|---|
| Root full-task successes | **115 / 200** |
| Root grasps (contact lift ≥ 5 cm) | 126 / 200 |
| Grasp-phase attempts (all stored episodes) | 797 |
| Grasp-phase successes | **329** |
| Grasp-phase failures | **468 (0.587)** |
| Branch grasps | 203 / 597 |
| Branches dropped after grasp | 44 |

- **Terminations:** 115 success, 457 branch_complete, 160 guard_refused and
  65 policy_complete (a root that ran its full budget without success).
  The guard stops split into 20 roots and 140 branches.
- **First missing stage:** none 115; reach 11; grasp 457; transport 206; place 8.
  Branches stop after lift, so a successful branch grasp shows up as
  "transport".

**Roots by noise level (descriptive):**

| Level | Grasp | Success |
|---|---|---|
| 0 | 41/50 | 41/50 |
| 1 | 36/50 | 32/50 |
| 2 | 29/50 | 27/50 |
| 3 | 20/50 | 15/50 |

Each level includes 10 aim-offset roots.

**Branch grasp rate by kind (descriptive):**

| Kind | Grasp |
|---|---|
| `noise_only` | 73/118 |
| `shift_close` | 42/119 |
| `weak_close` | 9/120 |
| `early_lift` | 40/120 |
| `open_during_lift` | 39/120 |

**By split (sessions train 170, val 20, test 10):**

| Split | Episodes | Root success | Root grasp | Branch grasp | Transitions |
|---|---|---|---|---|---|
| train | 677 | 99/170 | 108/170 | 180/507 | 174,774 |
| val | 80 | 11/20 | 13/20 | 17/60 | 20,326 |
| test | 40 | 5/10 | 5/10 | 6/30 | 10,419 |

## Acceptance checks (all passed)

| Check | Threshold | Result |
|---|---|---|
| A1 root episodes | ≥ 190 | 200 ✔ |
| A2 branch episodes | ≥ 450 | 597 ✔ |
| A3 root successes | ≥ 80 | 115 ✔ |
| A4 grasp-phase failure fraction | [0.20, 0.80] | 0.587 ✔ |
| A5 grasp-phase successes | ≥ 250 | 329 ✔ |
| A6 right-arm dims 6–11: fraction > +0.1 and < −0.1 | ≥ 0.03 each | min 0.063 (+, dim 9) and 0.095 (−, dim 11) ✔ |
| A7 off-script fraction | ≥ 0.50 | 0.874 ✔ |
| A8 sibling pairs RGB-distinct at step 16 | ≥ 90% | 499/499 ✔ |
| A9 split leakage | none | 0 sessions spanning splits, 0 off plan, normalisation = 677 train episodes ✔ |
| A10 audit | succeeds, 0.05 s intervals | 797 episodes, 205,519 transitions ✔ |
| A11 label sidecars | verified and gated | 0 errors; privileged load without acknowledgement refused ✔ |
| A12 resolution | 112×112×3 × 2 | ✔ |
| A13 integrity | all completed and exact | 200/200 roots completed, 200/200 restored exactly, 0 runtime errors, no source or input change ✔ |

**Action coverage (all transitions).**

- The standard deviations of the right-arm dimensions 6–11 are 0.151, 0.154,
  0.287, 0.190, 0.186 and 0.118. The right grasp's is 0.865, and 34.7% of its
  values are interior (|g| < 0.9).
- The left arm and left grasp are constant (std 0), by design.
- As preregistered, A7 and A8 only verify that the perturbations were
  injected and are visible. They are not evidence of a model's action
  sensitivity.

## Interpretation and limits

- **What the corpus is.** It has 200 wide-jitter resets with 115 full
  successful demonstrations and a dense set of grasp-phase outcomes on both
  sides of the success boundary: 329 grasps and 468 failures, from 200
  distinct pre-grasp states.
- **What it gives a world model.** Images in which the hand is better
  resolved than in the 24-px model's input, and action variation around the
  grasp.
  - `onboard_rgb` is 4.7× the 24-px resolution in each direction.
  - `hand_crop_rgb` is cut from a 320-px render, about 13× the 24-px
    resolution.

Post-run verification by a fresh subagent recomputed every number in this
document and in the manifest from the corpus, the sidecars and the worker
reports. It found no discrepancy. This clarification and the guard-stop split
above were its only (cosmetic) notes.
- **What it does not show.** It shows nothing about learned dynamics or
  learned control. The actions come from a privileged oracle and its
  perturbations.
- **Unchanged.** Every protocol limitation still applies: oracle actions, a
  left arm that is never excited, occlusion from the top-down camera, scorer
  stage labels, and branches that stop after lift.
- **Privileged metadata.** `privileged_outcome_labels`, `termination` and
  LeRobot `next.terminated` are privileged. They must not become model
  inputs, loss weights or selection filters.

## For TASK-050 (model v2)

- **Training data.** Train on the `train` split. Select on `val`. Do not
  decode `test` during training or selection.
- **Configuration.**
  - Set `image_size: 112` (LeWM patch 14 gives 8×8).
  - Choose `camera: onboard_rgb` or `hand_crop_rgb`. Using both needs a new
    two-camera model path.
  - Render the evaluator at 112 px, and compute the crop with
    `embodied_jepa.hand_crop.HandCrop`.
- **Diagnostics.** Include collapse and action-sensitivity diagnostics, for
  example matched sibling branches from the same pre-grasp state. Compare
  backends by physical success on the frozen wide development cohort
  45000–45007.
