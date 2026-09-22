# Apple world model v2: results (TASK-050)

**All three arms FAIL the preregistered gate set.** No arm passes, so by the
protocol's pre-declared readings **TASK-051 does not start the closed loop.**
Learned Apple→Plate remains at **0 successes**; nothing here is a control
result. The protocol is `apple_world_model_v2.md`, frozen at `3b6af0b`.

## Runs

Three runs, each executed once, each from a clean committed checkout, in this
order on one machine (no two runs overlapped).

| | LeWM / onboard (primary) | LeWM / hand-crop (secondary) | native / onboard (comparison) |
|---|---|---|---|
| Config | `apple_wm_v2_lewm.yaml` | `apple_wm_v2_lewm_hand_crop.yaml` | `apple_wm_v2.yaml` |
| Revision | `3b6af0b` | `9fa3d9f` | `9fa3d9f` |
| Python source SHA-256 | `6b1b1f1ecc3a…` | the same | the same |
| Steps | 15,000 / 15,000 | 15,000 / 15,000 | 15,000 / 15,000 |
| Selected step | 13,000 | 11,000 | 12,000 |
| Selection score (val) | 3.63 cm | 3.89 cm | 5.24 cm |
| Parameters | 2,163,360 | 2,163,360 | 931,402 |
| Wall clock | 3,557 s | 3,930 s | 4,502 s |
| Process CPU | 1,582 s | 1,983 s | 266 s |
| Peak host RSS | 9.10 GB | 9.12 GB | 9.00 GB |
| Checkpoint SHA-256 | `68031e15d7ba…` | `0100aa750293…` | `f3179b9f016a…` |
| Implementation SHA-256 | `885cf114885d…` | the same | `624f05e83fea…` |

The two revisions differ only in `.mc` task notes; the Python source hash
`6b1b1f1ecc3a98e805e6d53bfc1d07adc65f2202aab7d71352ca752a81149364` is identical
for all three runs. Every run reports `dirty: false` (`--require-clean`).

- **Device** MPS, Apple M5 Pro, 48 GB; macOS 26.5.1 arm64; torch 2.14.0; NumPy 2.5.3.
- **Data** `data/apple-wide-v1`, manifest SHA-256 `028e130576c0…`, 677 train
  episodes (175,451 observations, 164,822 horizon-16 windows) and 80 val
  episodes (20,406 observations). Split hash `51f8e09df1c0…`, action hash
  `da987bb0798c…`. **`test_episodes_decoded: 0`** in every report: the test
  split was never opened.
- **Artifacts** `checkpoints/task050-wm-v2/{leworldmodel,leworldmodel_hand_crop,native_jepa}.{pt,latest.pt,run.json,metrics.jsonl}`
  and `outputs/task050-wm-v2/*-val-gates.json` (both git-ignored).

## Gate table (val, h = 8 unless stated)

| Gate | Threshold | LeWM / onboard | LeWM / hand-crop | native / onboard |
|---|---|---|---|---|
| G1 palm–apple, moving windows | ≤ 1.5 cm | **3.65 cm FAIL** | **4.08 cm FAIL** | **5.84 cm FAIL** |
| G2a ÷ persistence control | ≤ 0.8 | **0.835 FAIL** | **0.858 FAIL** | **0.844 FAIL** |
| G2b ÷ shuffled-action control | ≤ 0.8 | 0.699 PASS | 0.684 PASS | 0.614 PASS |
| G3 apple–plate, valid windows | ≤ 2.0 cm | 1.93 cm PASS | **2.24 cm FAIL** | **2.16 cm FAIL** |
| G4 apple height, grasp cohort | ≤ 1.0 cm | 0.24 cm PASS | 0.16 cm PASS | 0.16 cm PASS |
| G5 `apple_held` AUROC, lift cohort | ≥ 0.85 | 0.9996 PASS | 0.9997 PASS | 0.9995 PASS |
| G6 approach-cost calibration | ≤ ln 1.5 = 0.405 | **1.132 FAIL** | **1.075 FAIL** | **0.802 FAIL** |
| G7a siblings own < swapped, h = 16 | ≥ 0.70 | **0.679 FAIL** | **0.689 FAIL** | 0.745 PASS |
| G7b sibling divergence ρ, h = 16 | ≥ 0.5 | 0.589 PASS | 0.502 PASS | 0.737 PASS |
| G8a collapsed fraction | ≤ 0.05 | 0.000 PASS | 0.000 PASS | 0.000 PASS |
| G8b effective rank | ≥ 4 | 8.13 PASS | 8.00 PASS | 14.74 PASS |
| G8c mean latent std | ≥ 0.1 | 0.907 PASS | 0.935 PASS | 1.055 PASS |
| **Overall** | all gates | **FAIL** | **FAIL** | **FAIL** |

## What the numbers say

**The models are accurate on still frames and inaccurate exactly where planning
needs them.** At h = 8 the palm–apple error over *all* valid windows is 0.77 cm
(LeWM onboard), 0.86 cm (hand crop) and 0.85 cm (native). On the 1,462 of 3,961
windows where the true offset moves by ≥ 1 cm, it is 3.65 / 4.08 / 5.84 cm,
against a true displacement of 2.49 cm.

| h | valid | moving | predicted | persistence | shuffled | zero-action | true displacement |
|---|---|---|---|---|---|---|---|
| 1 | 4,010 | 131 | 2.75 cm | 3.35 | 3.49 | 2.88 | 1.30 cm |
| 4 | 3,991 | 1,004 | 4.03 cm | 4.49 | 5.00 | 4.86 | 1.67 cm |
| 8 | 3,961 | 1,462 | 3.65 cm | 4.37 | 5.22 | 5.31 | 2.49 cm |
| 16 | 3,901 | 1,943 | 3.32 cm | 4.27 | 5.37 | 6.10 | 3.14 cm |

(LeWM / onboard. The other two arms behave the same way; native is uniformly
worse, hand crop is between.)

- **The actions matter, but not enough.** Every arm beats its shuffled-action
  control by a wide margin (G2b) and tracks sibling divergence (G7b). But the
  model barely beats *its own* persistence readout (G2a fails on all three),
  because most of the error is already in the encoded readout of a moving
  state, not in the rollout.
- **Sibling discrimination is real but short of the bar.** Among the 106
  qualifying ordered sibling pairs at h = 16, the own-action prediction is
  closer to the truth than the swapped-sibling prediction in 67.9 % (LeWM),
  68.9 % (hand crop) and 74.5 % (native) of cases, against a 70 % threshold and
  a 50 % chance level. Median own error 1.76 / 1.41 / 1.38 cm versus swapped
  2.15 / 1.85 / 2.14 cm.
- **The object-aware cost would be badly ranked.** G6 fails on every arm, and
  the descriptive Spearman ρ between the predicted and true approach cost is
  only 0.16 / 0.16 / 0.06 at h = 8. A CEM cost built on these readouts would
  order candidates nearly arbitrarily near the grasp point. This is the most
  direct evidence against starting the closed loop now.
- **Held/lift detection is near-perfect, and that is partly trivial.** The lift
  cohort AUROC is ≈ 1.0 for every arm. With proprioception fused into the
  latent, a closed hand at height is almost sufficient to call `apple_held`;
  the 3.6 cm palm–apple error shows the model does not know the apple nearly as
  well as it knows itself. Treat G5 as passed but weak evidence.
- **No collapse.** Every arm passes G8. The descriptive image-only statistics
  are more interesting: the effective rank of the image pathway alone is 7.60
  (LeWM onboard), 7.19 (hand crop) but **2.19** for native. The native arm's
  latent variety comes mostly from the proprioception branch, which is
  consistent with its worse palm–apple numbers.
- **Grasp outcome from the pre-grasp state** (descriptive, h = 64, 63 siblings,
  29 positive): AUROC 0.846 (LeWM), 0.819 (hand crop), 0.911 (native).
- **`apple_dropped`** AUROC over all windows: 0.976 / 0.947 / 0.931.
- **The hand crop did not help.** It is slightly better on apple height and on
  sibling own-error, and worse on palm–apple, apple–plate and cost
  calibration. The crop follows the hand, so the apple leaves it exactly when
  the offset is large.

## Reading, as pre-declared

The protocol's readings are applied as written, not re-interpreted after the
fact:

- **LeWM fails G2 → do not start the closed loop.** TASK-051 is therefore not
  the T4 closed-loop task. No arm passed, so the hand-crop and native
  fallbacks do not apply either.
- G1, G3 and G6 (the precision gates) also fail, and G7a fails for both LeWM
  arms.

**What the failure is, precisely.** The failing quantity in G2a is the ratio to
the model's *own* persistence readout, and G2b passes; so the evidence is that
the action conditioning does carry information (shuffled actions are clearly
worse, sibling divergence correlates), but the readout is too imprecise on
moving and occluded states for the ratio to clear 0.8. This diagnosis is
offered as an explanation, not as a re-reading of the gate: the gate failed.

## Recommendation for TASK-051

Not the closed loop. A revised offline protocol (`v3`) that attacks the
measured failure, in this order:

1. **Readout precision under motion.** The all-window/moving-window gap (0.8 cm
   versus 3.6 cm) is the core defect. Candidates: a two-camera model (onboard
   plus hand crop), a larger encoder or longer training, an explicit apple-pixel
   auxiliary target, and weighting the readout loss towards moving frames.
2. **Cost ranking as the headline metric.** Replace the absolute calibration
   gate with a candidate-ranking gate measured the way a CEM would use it:
   Spearman ρ between predicted and true phase cost across the *sibling*
   candidates from one state, plus top-1 regret in metres. The current
   ρ ≈ 0.16 is the number to beat.
3. **Keep the controls that already work** (shuffled actions, persistence,
   sibling swap) and keep test sealed.
4. Carried over from the pre-run review: the 0.01 std floor on the 43
   near-constant proprioception dimensions amplifies closed-loop deviations up
   to 100×; fix it before any closed-loop run.

## Honest labelling and limits

- The LeWM arms are the pinned upstream encoder, predictor, projector and
  SIGReg objective **plus** the shared state fusion and the auxiliary readout
  loss. That is not the unmodified LeWM objective.
- Selection and the gates both used val, which flatters G1 slightly; test was
  never decoded, so an unbiased check is still available.
- One seed per arm, no ensemble. The differences between arms are single-run
  differences.
- Latent MSEs are not compared across backends; the comparison is in physical
  readout units and compute.
- The corpus is privileged scripted-collector data with injected perturbations,
  and every limitation of `apple_wide_collection_results_v1.md` still applies.
