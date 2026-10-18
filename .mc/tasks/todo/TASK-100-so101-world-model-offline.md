---
id: TASK-100
aliases:
- TASK-100
title: 'SO-101 S4: offline world-model study on so101-play-v1 (LeWM and native_jepa token predictors over frozen DINOv2 tokens of both cameras, gates G1–G6)'
slug: so101-world-model-offline
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- world-model
- lewm
- preregistration
sprint: ''
depends_on:
- "[[TASK-099]]"
due_date: ''
created: 2026-10-17
updated: 2026-10-17
---


# SO-101 S4: offline world-model study on so101-play-v1 (LeWM and native_jepa token predictors over frozen DINOv2 tokens of both cameras, gates G1–G6)

## Description

Phase S4 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md), opened by Claude under owner delegation
(DECISIONS 2026-10-17 (bq), R28.10–R28.26). On the S3 corpus `so101-play-v1` (TASK-099), train a
LeWM token predictor (L) and its `native_jepa` one-key swap (J) over frozen DINOv2 ViT-S/14 patch
tokens of the wrist and overview cameras, action-conditioned on `joint_delta_so101_v0`, with
no-action twins, a random-init DINOv2 floor and copy-last, three model seeds each; gate on the
test split at h = 8 and h = 16: G1 no collapse, G2 copy-last, G3 no-action, G4 action
sensitivity, G5 cube and jaw readable from the prediction (readout fitted on predicted latents),
G6 training saturation; report a planning-usefulness measure for S5.

**The preregistration (FROZEN at the Stage 0 PR's merge, #237):**
[`docs/experiments/so101_world_model_v1.md`](../../../docs/experiments/so101_world_model_v1.md).
Offline only; no control claim; not an Apple→Plate experiment; R7 does not change (R28.9).

## Acceptance Criteria

- [x] DRAFT preregistration independently reviewed and merged (#236)
- [x] Stage 0 (#237; S0-DONE recorded in `docs/experiments/so101_world_model_v1_stage0.md`): code in new files only (G1 byte-pinned files, `config.py` and existing model files
  unchanged; core-import check passes), tests, D0 featurisation probe, grid rule, G1 and G5 bar
  calibration on D0 only, debug smoke and CUDA determinism; independently reviewed and merged
  (the freeze)
- [ ] Stages F, B, T (T′ only if G6 triggers), V, F′, E at the frozen revision in a clean run tree,
  every GPU job under `scripts/gpu_run.sh --wait --board`, within the 20 GPU h cap; val report
  hash recorded here before the test split is read; evidence under
  `~/develop/emai/evidence/task100-*` with `SHA256SUMS`
- [ ] Results (VOID / WM-PASS / WM-PARTIAL / WM-FAIL) and DECISIONS entry, reviewed and merged;
  R7 unchanged

## Notes

- 2026-10-17, created with the DRAFT preregistration (branch `docs/so101-s4-draft`). Nothing
  featurised, trained or evaluated.
%% mc-links: [[TASK-099]] %%
- 2026-10-17, DRAFT merged (#236). Stage 0 run on branch `feat/so101-s4-stage0` (D0 only; test
  never opened): S0-DONE. Grid g4; B_rank 0.11, B_std 0.37, B_cube 2.397 cm, B_jaw 0.1666 rad;
  moving subsets 1.0 cm and 0.2 rad (not halved); calibration L's G4 lower bounds 1.769 / 2.083
  (h = 8 / 16) and R-pred medians inside both G5 bars, so no ruling needed; CUDA determinism
  bit-identical; 1.98 GPU h used of the 4 GPU h Stage 0 cap; gated stages projected at 12.1–17.7
  GPU h against 18.0 left. Evidence `~/develop/emai/evidence/task100-stage0/` (SHA256SUMS
  `7bee61cf…`). DECISIONS 2026-10-17 (br), R28.27–R28.33. The protocol froze at the Stage 0 PR's
  merge (#237); the gated stages have not run.
- 2026-10-18, gated stages F, B, T, T′ and V run at `f0da5bf` from the clean run tree
  `worktrees/task100-run`. **Val report SHA-256
  `b3c3eddef04da3e29a1fad952ce619af6f65714574a0e749f8f8f4c6f633aa31`**
  (`outputs/task100-run/val_report.json`), recorded here and in `outputs/task100-run/runlog.txt`
  before Stage F′ featurises the test split.
