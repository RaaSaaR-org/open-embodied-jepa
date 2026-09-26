---
id: TASK-062
aliases:
- TASK-062
title: 'Preregister and run an encoder study: can a frozen encoder expose the post-look apple?'
slug: preregister-and-run-an-encoder-study-can-a-frozen-encoder-expose-the-post-look-a
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- perception
- evaluation
sprint: ''
depends_on:
- "[[TASK-061]]"
due_date: ''
created: 2026-09-26
updated: 2026-09-26
---




# Preregister and run an encoder study: can a frozen encoder expose the post-look apple?

## Description

TASK-061 ended in **O-LOOK-RAW**. After a fixed 8-command look, raw 112 px onboard pixels read
the apple position and the expert's first reset-dependent command (L-raw 0.469 cm, ratio 0.242;
T2 180/190). The frozen E0 encoder does not (1.273 cm, 147/190). A random-init encoder of E0's
architecture is about as good as E0 on position and better on the sign (176/190).

The task owner chose the encoder work before the look-prefix corpus (2026-09-26). **Learned
Apple→Plate is still 0 successes.** This task trains image encoders, not controllers. It collects
no corpus and preregisters no control formulation: the TASK-054 (CEM) and TASK-057 (BC) clauses
hold.

The study is preregistered, with four decisional arms, each tied to one explanation of E0's
failure. They are cross-fitted, so that every root is probed with an encoder that never saw it.

- **A-tok** (pooling): read the recipe model's final patch tokens instead of the pooled latent.
- **A-sig** (collapse regime): an extra SIGReg term, weight 1.0, on the image feature itself.
- **A-rec** (objective): add a pixel-reconstruction term on the image feature.
- **A-plain** (supervision): turn the privileged readout heads off.

Each arm must pass the unchanged TASK-059 bars on TASK-061's post-look frames, **beat its own
random-init floor**, survive Holm over the four arms, and pass the apple-hidden check. Before
any training, TASK-061's L-raw, L-E0 and L-random must reproduce bit for bit (G-anchor).

Protocol: `docs/experiments/apple_encoder_study_v1.md`.
Manifest: `benchmarks/manifests/apple-encoder-study-v1.json`.
Calibration (label-free, fits nothing): `scripts/calibrate_encoder_study.py`.

## Acceptance Criteria
- [x] The preregistration, manifest and calibration script merge on an independent reviewer's
      reported APPROVE, with green CI, before any encoder is trained (PR 1, #57, f23f755).
- [x] The runner merges with the tests of protocol §11 (PR 2, #58, 67a92e9), on a reported
      APPROVE with green CI.
- [x] The gated run starts only on the pre-run reviewer's reported verdict, with a clean tree
      (PRE-RUN: GO; started 2026-09-26T10:24:55Z at 67a92e9).
- [x] The results document, the results manifest and the summarize script (PR 3) state:
  - the outcome row, and the numbers with CIs against their baselines and floors;
  - every negative result;
  - a recommended (not chosen) next task.
- [x] The test split, cohort C and cohort D are untouched, and `exemption_spent` stays false.

## Notes
- Void rule: an early stop is V. One from-scratch repeat is allowed; a second void is
  INCONCLUSIVE. Owner rulings are recorded with a UTC timestamp before the run they affect.
- PR #57's merge was blocked by the permission classifier; the task owner merged it.

## Results (2026-09-26)

**Outcome O-ENC-ARCH; the abandonment clause fired** (run-1, 67a92e9, 28 274 s; all guards
passed; G-anchor exact; report sha256 `fc9d5e52…962f`).
- A-tok meets the bars: 0.463 cm [0.422, 0.536], ratio 0.239 [0.212, 0.291], 178/190. It does
  not beat its floor F-tok (179/190 by itself). Qualifier: "meets the bars, not the floor".
- A-sig (147/190), A-rec (160/190) and A-plain (159/190) fail the bars and their floors. Holm
  rejects nothing. Every arm is apple-caused under the apple-hidden check.
- Reported only: E0-tok 0.415 cm, 180/190 (not held out); R0-cls 1.223 cm, 162/190;
  A-tok − L-raw −0.006 cm [−0.068, 0.074].
- Recommended next task: a preregistered test of an externally pretrained frozen encoder against
  its random-init floor. Learned Apple→Plate is still 0 successes.
- Results: `docs/experiments/apple_encoder_study_v1_results.md`,
  `benchmarks/manifests/apple-encoder-study-v1-results.json`.
%% mc-links: [[TASK-061]] %%
