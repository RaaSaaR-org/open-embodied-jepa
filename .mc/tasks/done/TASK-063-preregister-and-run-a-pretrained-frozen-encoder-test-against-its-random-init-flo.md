---
id: TASK-063
aliases:
- TASK-063
title: Preregister and run a pretrained frozen encoder test against its random-init floor on the post-look apple probe
slug: preregister-and-run-a-pretrained-frozen-encoder-test-against-its-random-init-flo
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
- "[[TASK-062]]"
due_date: ''
created: 2026-09-26
updated: 2026-09-26
---



# Preregister and run a pretrained frozen encoder test against its random-init floor on the post-look apple probe

## Description

TASK-062 ended in **O-ENC-ARCH** and its abandonment clause fired: the in-corpus
encoder-training line is closed. The apple survives in ViT patch tokens and is lost in the pooled
128-d latent, but A-tok (0.463 cm, 178/190) did not beat its random-init floor F-tok
(0.601 cm, 179/190). Its recommended next task is this one: **one preregistered test of an
externally pretrained frozen encoder against its own random-init floor, on the same probe, before
any corpus.**

**Learned Apple→Plate is still 0 successes.** This task tests encoder **readability**, not
control. It trains nothing, collects no corpus and preregisters no control formulation: the
TASK-054 (CEM) and TASK-057 (BC) clauses hold.

- Encoder: DINOv2 ViT-S/14 (`facebook/dinov2-small` @ `ed25f3a3`, safetensors sha256
  `ae1e99fc…4be1`, Apache-2.0 code and weights, bit-identical to FAIR's original checkpoint).
- Arms (Holm over two, one-sided α 0.025): **P-cls** (CLS, 384-d) and **P-tok** (final patch
  tokens, 98 304-d), each against the same architecture's seed-0 random init read at the same
  point (R-cls, R-tok).
- TASK-061's probe, frames (renderer warm-up), readouts, folds, bars, floor rule and apple-hidden
  check, unchanged; G-anchor against the TASK-061 and TASK-062 run-1 reports.

Protocol: `docs/experiments/apple_pretrained_encoder_v1.md`.
Manifest: `benchmarks/manifests/apple-pretrained-encoder-v1.json`.
Calibration (label-free, fits nothing): `scripts/calibrate_pretrained_encoder.py`.

## Acceptance Criteria
- [x] PR 1 (preregistration, manifest, encoder module, fetch + calibration scripts, dependency
      entry, `pretrained` extra) merges on an independent reviewer's reported APPROVE that
      explicitly covers the licence and provenance review, with green CI, before any readout (#61, 5c04c81).
- [x] PR 2 (runner, decision code, guard tests) merges on a reported APPROVE with green CI (#62, 198d1f1).
- [x] The gated run starts only on the pre-run reviewer's reported verdict, with a clean tree
      (PRE-RUN: GO; started 2026-09-26T19:21:32Z at 198d1f1).
- [x] PR 3 (results document, results manifest, summarize script) states the outcome row, the
      numbers with CIs against floors and baselines, every negative result, and a recommended
      (not chosen) next task; a reviewer checks every restated number against report.json by
      script.
- [x] The test split, cohort C and cohort D are untouched; `exemption_spent` stays false.

## Notes
- Void rule: an early stop is V (with a void_reason). One from-scratch repeat into run-2 with the
  same seeds, caps and device; a second void closes the task as INCONCLUSIVE. Owner rulings are
  recorded with a UTC timestamp before the run they affect.
- Design fixed before calibration: commit 74dfe6b (encoder module, input, read-out points,
  floor); the calibration ran afterwards
  (`outputs/task063-pretrained-encoder/calibration-v1.json`).
- Worktree: `data`, `outputs`, `checkpoints` and `third_party` are git-ignored convenience
  symlinks to the main checkout; nothing under them is overwritten.

## Results (2026-09-26)

**Outcome O-PT-POOLED; the abandonment clause does not fire** (run-1, 198d1f1, 199 s, CPU; all
guards passed; both G-anchor sets exact; report sha256 `575ae650…ee90b`).
- P-cls (frozen DINOv2 CLS): 0.538 cm [0.487, 0.611], ratio 0.277 [0.249, 0.329], 183/190; beats
  R-cls by −0.316 cm [−0.456, −0.214] (R-cls 169/190). Holm p = 1.2e-22 (threshold 0.0125).
- P-tok (frozen tokens): 0.394 cm [0.363, 0.447], 187/190; beats R-tok by −0.106 cm
  [−0.182, −0.041] (R-tok 181/190). Holm p = 0.0013 (threshold 0.025). P-tok also passes.
- Both apple-caused (apple-hidden: 2.704 cm / 79/190 and 2.592 cm / 69/190).
- Both random-init floors meet the bars on their own (reported only).
- Readability only; **learned Apple→Plate is still 0 successes.**
- Recommended (owner chooses): preregister the look-prefix 112 px corpus with this frozen encoder
  as the candidate image encoder; no control formulation is preregistered.
- Results: `docs/experiments/apple_pretrained_encoder_v1_results.md`,
  `benchmarks/manifests/apple-pretrained-encoder-v1-results.json`.
%% mc-links: [[TASK-062]] %%
