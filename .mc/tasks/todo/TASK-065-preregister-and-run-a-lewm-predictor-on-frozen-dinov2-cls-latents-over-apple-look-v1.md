---
id: TASK-065
aliases:
- TASK-065
title: Preregister and run a LeWM predictor on frozen DINOv2 CLS latents over apple-look-v1 (world-model test only)
slug: preregister-and-run-a-lewm-predictor-on-frozen-dinov2-cls-latents-over-apple-look-v1
status: in-progress
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
- "[[TASK-064]]"
due_date: ''
created: 2026-09-26
updated: 2026-09-26
---




# Preregister and run a LeWM predictor on frozen DINOv2 CLS latents over apple-look-v1 (world-model test only)

## Description

TASK-064 ended in **C-ACCEPT**: `apple-look-v1` (dataset manifest `81d760d1…db64`, 170/20/10
train/val/test sessions, test never decoded) is accepted, and the frozen DINOv2 CLS reads the
post-look apple on it (0.550 cm [0.478, 0.642], 178/190, beating its random-init floor). Its row
pre-declared this task: a separately preregistered world-model task with this frozen encoder as a
candidate image encoder.

**Question:** does a LeWM action-conditioned predictor, trained on `apple-look-v1` over frozen
DINOv2 CLS latents, learn real dynamics, and does the apple position stay readable through
multi-step prediction?

**This is a world-model test only.** No control formulation is preregistered or implied: the
TASK-054 (CEM) and TASK-057 (BC) clauses hold. **Learned Apple->Plate is still 0 successes.**

- Architecture: a new option `frozen_encoder`, written once in `models/frozen_encoder.py` (a
  mixin in front of an unchanged backend; the plain backends reject the key; no existing model
  file changes, so the TASK-054 E0 checkpoint stays valid). It feeds the pinned upstream LeWM
  predictor (`ARPredictor` + `Embedder` + `pred_proj`, history one) with the train-standardised
  frozen CLS; SIGReg weight 0.
- Arms: W (true actions) and N (no-action baseline: all actions zero, same batch stream as W),
  3 seeds, 2 cross-fitting halves of the 170 train sessions; copy-last as an untrained baseline.
  Selection on val only.
- Gates at h = 8 and 16, per seed: G1 collapse, G2 beats copy-last (ratio upper bound ≤ 0.8), G3
  beats no-action (< 1.0), G4 action sensitivity (wrong/zero-action ratio lower bound ≥ 1.10), G5
  the apple stays readable (TASK-059 T1 bar and ≤ 0.5 cm over the encoded target) under the
  TASK-063 probe fitted on encoder latents of train roots, cross-fitted.
- Rows: V, WM-DYNAMICS, WM-UNSTABLE, WM-APPLE-LOST, WM-NO-DYNAMICS (the last two fire the
  abandonment clause for the frozen-pooled-pretrained-latent predictor line); a second V is
  INCONCLUSIVE.
- Budget: MPS for 12 models (1800 s each), CPU features (5400 s), global 21 600 s.

Protocol: `docs/experiments/apple_latent_dynamics_v1.md`.
Manifest: `benchmarks/manifests/apple-latent-dynamics-v1.json`.
Design code: `src/embodied_jepa/latent_dynamics.py`; model option in
`src/embodied_jepa/models/frozen_encoder.py`.

## Acceptance Criteria
- [ ] PR 1 (protocol, manifest, `latent_dynamics.py`, `frozen_encoder` option with
      `docs/MODELS.md`, tests, task card) merges on an independent reviewer's reported APPROVE
      and green CI, before any predictor is trained on the corpus.
- [ ] PR 2 (runner, guard tests, runner pin) merges on a reported APPROVE and green CI.
- [ ] The gated run starts only on the pre-run reviewer's reported verdict, from a clean tree.
- [ ] PR 3 (results document, results manifest, summarize script) states the outcome row, every
      gate quantity with its interval per seed and horizon, every negative result and a
      recommended (not chosen) next task; a reviewer checks every restated number against
      `report.json` by script. Checkpoints and features are never committed.
- [ ] The test split is never decoded; cohorts C and D are untouched; `exemption_spent` stays
      false.

## Notes
- Void rule: an early stop is V (with a void_reason). One from-scratch repeat into `run-2` with
  the same seeds, caps and device; a second void closes the task as INCONCLUSIVE. Rulings are
  recorded here with a UTC timestamp before the run they affect.
- Pre-freeze (disclosed in protocol §13): no existing model file changes (a first draft that
  edited `models/base.py` failed the E0 guard and was withdrawn before commit); the anchor
  (190 post-look P-cls features) reproduces TASK-064's hash; labels of 40 train roots were read to
  choose horizons (apple static until frame ≥ 117); no feature, prediction or readout computed.
- Worktree: `data`, `outputs`, `checkpoints` and `third_party` are ignored convenience symlinks to
  the main checkout; `.venv` is the worktree's own. Nothing under them is overwritten.
%% mc-links: [[TASK-064]] %%
