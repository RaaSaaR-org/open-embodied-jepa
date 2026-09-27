---
id: TASK-066
aliases:
- TASK-066
title: Preregister and run a LeWM-family predictor on frozen DINOv2 patch-token latents over apple-look-v1 (world-model test only)
slug: preregister-and-run-a-lewm-family-predictor-on-frozen-dinov2-patch-token-latents
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- world-model
- lewm
sprint: ''
depends_on:
- "[[TASK-065]]"
due_date: ''
created: 2026-09-27
updated: 2026-09-28
---



# Preregister and run a LeWM-family predictor on frozen DINOv2 patch-token latents over apple-look-v1 (world-model test only)

## Description

TASK-065 ended in **WM-NO-DYNAMICS**, and its clause closed only the pooled-CLS predictor line.
- G2–G4 passed on every seed.
- G1 failed on an uncalibrated rank bar.
- G5 failed.
- Checkpoints were selected late.

TASK-066 asks: does an action-conditioned predictor over frozen DINOv2 ViT-S/14 **patch-token**
latents (in the style of DINO-WM) on `apple-look-v1` avoid rank collapse and keep the apple
readable through multi-step prediction?

**This is a world-model test only.** No control formulation is preregistered or implied: the
TASK-054 (CEM) and TASK-057 (BC) clauses hold. **Learned Apple->Plate is still 0 successes.**

- **Latent.** The final patch tokens, average-pooled 16×16 → 4×4 (6144-d). It is a new option,
  `models/frozen_tokens.py`, which subclasses TASK-065's `FrozenEncoderMixin`. No existing model
  file changes.
- **Predictor.** The pinned upstream LeWM `ARPredictor`/`Embedder`/`pred_proj` over the token
  sequence, with bidirectional attention within each frame. The native backend is one key away
  and is not run.
- **Harness.** TASK-065's harness is reused: the reader, the guards (including the amended G-cache
  plus a partial-batch check), the baselines, cross-fitting, and G2–G5.
- **G1.** Calibrated on TASK-064's disjoint pilot-d from latent-space quantities only. The bars
  have absolute floors, plus a comparative W-over-N rank part. The G1 bars must fail the
  truncation controls at k = 1, 2, 4.
- **Rows.** V, WM-TOK-DYNAMICS, WM-TOK-UNSTABLE, WM-TOK-CEILING, WM-TOK-APPLE-LOST,
  WM-TOK-COLLAPSE, WM-TOK-NO-DYNAMICS. The last three fire the clause for the patch-token line.
  A second V is INCONCLUSIVE.

Protocol: `docs/experiments/apple_token_dynamics_v1.md`.
Manifest: `benchmarks/manifests/apple-token-dynamics-v1.json`.

## Acceptance Criteria
- [x] PR 1 merges on an independent reviewer's reported APPROVE and green CI, before any
      gated predictor is trained on the corpus. (The runner smoke-a trained four 20-update models
      on corpus train episodes with noise targets before the freeze; it is disclosed in protocol
      §14.) It contains the protocol, the manifest,
      `token_dynamics.py`, the `frozen_tokens` option with `docs/MODELS.md`, the calibration
      script and its recorded results, the tests and this card. (#74, `eeb12a7`.)
- [x] PR 2 merges on a reported APPROVE and green CI. It contains the runner, the guard tests and
      the runner pin. (#75, `87f3f81`.)
- [x] The gated run starts only on the pre-run reviewer's reported verdict, from a clean tree.
      (run-1 at `87f3f81` on a reported PRE-RUN: GO.)
- [ ] PR 3 states the outcome row and every gate quantity with its interval. A reviewer checks
      every restated number against `report.json` by script. It contains the results document,
      the results manifest, the summarize script and, if a clause fires, a DECISIONS entry.
- [x] The test split is never decoded. Cohorts C and D are untouched. `exemption_spent` stays
      false.

## Notes
- **Worktree.** `data`, `outputs`, `checkpoints` and `third_party` are ignored convenience
  symlinks to the main checkout. `.venv` is the worktree's own.
- **Void rule.** An early stop is V, with a void_reason. One from-scratch repeat into `run-2`
  with the same seeds, caps and device. A second V closes the task as INCONCLUSIVE. Any guard
  change needs an owner ruling.

## Rulings and pre-freeze log (UTC)
- 2026-09-27T03:10Z–03:24Z: the calibration smoke (`outputs/task066-scratch/cal-smoke-a`, 1000 updates)
  checked the mechanics only; it is not read.
- 2026-09-27T03:14Z–03:17Z: the anchor pre-check (`outputs/task066-calibration/anchor-1`, `f78257c`)
  ran on features only. P_cls and P_tok reproduce TASK-064's hashes. The pooled cache differs
  from the pooled anchor by at most 2.93e-5, on exactly the 14 partial-batch roots. Whole episodes
  with partial final batches re-featurise bit-identically.
- 2026-09-27T03:24Z: the calibration `run-1` started at `7fa8183` on pilot-d. Its training
  subprocesses started at 03:32Z.
- **About 03:35Z, owner ruling 1** (G1 design, received via the coordinator; approximate).
  - Absolute floors: rank 0.10, std 0.25.
  - A comparative W-over-N rank part: lower 95 % bound > 0, projected 256-direction bootstrap,
    disclosed as an approximation.
  - An encoded-only 4×4 readability check on pilot-d, with its label read disclosed.
- **About 03:40Z, owner condition.** The combined G1 must fail the truncation controls at
  k = 1, 2, 4 before the freeze; k = 8 is reported. Implemented at `753c42c` (03:43Z), before any
  calibration result was read.
- 03:44Z: the pilot readability check.
  - `readability-1` stopped before any readout: the pilot report's `apple_xy` are smoke noise.
  - Fixed at `7820ea5`.
  - `readability-2` and `readability-3` (`4b34ea2`, with the paired differences) ran.
  - **The pooling rule fired: reconsider pooling.** 4×4 misses the T1 ratio bar at h = 8 (upper
    bound 0.644) and h = 16 (0.618), while P-tok meets it (0.588, 0.540).
  - The pilot is insensitive: the paired 4×4 − P-tok intervals include 0, and P-mean fails
    every bar.
  - Reported to the owner.
- **2026-09-27T03:46Z, owner ruling 2** (pooling): option A, keep 4×4, subject to an
  encoded-only ceiling pre-check on the 190 train + val corpus roots. The decision rule was
  written into the prereg before the check ran (`96ed14c`).
- 03:47Z–03:52Z: the ceiling pre-check (`outputs/task066-calibration/ceiling-1`) says **keep 4×4 and
  freeze**. The 190 roots read 0.607 and 0.492 cm at h = 8 and 16, with upper bounds 0.381 and
  0.320. The run's 170-root ceiling agrees.
- 03:52Z–04:04Z: the runner smoke (`outputs/task066-scratch/smoke-a`) ran with noise targets. It
  is not evidence. Every section was written, and the peak RSS was 6.98 GB.
- 05:25Z: calibration `run-1` complete (7283 s, sha256 `a1d9fc17…53e1`).
  - u_sat: W s0 6500, W s1 7000, N s0 16 500. **The budget rule escalated**: it wanted 33 000,
    above the 30 000 ceiling.
  - G1 bars 0.16 (rank reference 0.3315, W s0 at update 9500, h = 16) and 0.39 (std reference
    0.7963, W s1 at update 9000, h = 8).
- 05:3xZ: truncation controls (`controls-1`, tree dirty in this card only). The combined G1 fails
  k = 1, 2, 4 and 8; the untruncated pilot W passes. The owner's condition is met.
- **2026-09-27T05:36Z, owner ruling 3** (budget): option A.
  - U = 30 000, selection every 1500.
  - Caps: 4500 s per model, 57 600 s global.
  - G1 is read at the val-selected checkpoint (fixed).
- 2026-09-27T11:54:38Z: #74 merged (`eeb12a7`, by the user). #75 was rebased onto main
  (head `9d85222`, tree identical to the approved `16dbc6e`; a reviewer reported APPROVE on the
  diff-of-diffs) and merged (`87f3f81`, by the user from the main session; the classifier blocked
  the agent's merge twice).
- 2026-09-27T12:08:23Z: **run-1 started** at `87f3f81` on the pre-run reviewer's reported
  PRE-RUN: GO. Python PID 50916.
- 2026-09-27T23:00:57Z: **run-1 complete**, 39 153 s, **outcome WM-TOK-DYNAMICS** (report sha256
  `e6b28e07…0b60`). No clause fires.

## Results (2026-09-28)

**Outcome WM-TOK-DYNAMICS**: every seed passes G1–G5 at h = 8 and 16, and the encoded grid
meets the T1 bar (0.632 and 0.503 cm).
- G1: rank ratio 0.325–0.344 (bar 0.16), std ratio 0.82–0.84, W − N rank lower bounds
  0.038–0.076.
- G2: W / copy-last upper bounds 0.790–0.792 at h = 8, 0.652–0.654 at h = 16.
- G3: W / N upper bounds ≤ 0.916. G4: lower bounds ≥ 1.67.
- G5: W reads 0.75–0.93 cm; ratio-to-B-occ upper bounds ≤ 0.581; excess upper bounds
  ≤ 0.423 cm.

**Caveats:** (1) the budget did not saturate: 9 of 12 models, every N, selected a late
checkpoint, which may favour W on G3 and G1 (iii); (2) narrow margins at G2 h = 8 and G5 seed 2
h = 8; (3) the token rank ratio is not better than TASK-065's CLS; (4) train split only, no control
claim.

**Recommended (the owner chooses):** a preregistered held-out confirmation on the test split,
with the saturation caveat fixed first. **Learned Apple->Plate is still 0 successes.**

Results: `docs/experiments/apple_token_dynamics_v1_results.md`,
`benchmarks/manifests/apple-token-dynamics-v1-results.json`; decision in `docs/DECISIONS.md`.
%% mc-links: [[TASK-065]] %%
