---
id: TASK-080
aliases:
- TASK-080
title: 'LeWM committed aim under C1-M, read by a readout fitted on predicted latents'
slug: lewm-committed-aim-under-c1m-read-by-a-readout-fitted-on-predicted-latents
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- preregistration
- draft
sprint: ''
depends_on:
- "[[TASK-077]]"
due_date: ''
created: 2026-10-07
updated: 2026-10-07
---

# LeWM committed aim under C1-M, read by a readout fitted on predicted latents

## Description

TASK-077 closed **G-NO-BAR** (`docs/experiments/apple_lewm_c1m_v2_results.md`, R17.52): its
predictor passed G1–G4, but R8, a plate readout fitted on encoded frames, read W's predicted plate
at r at 1.58–1.87 cm against τ_commit = 1.0 cm. The decomposition record
(`docs/experiments/apple_lewm_c1m_v2_decomposition.md`, R17.53–R17.54, #152) ended **D-READOUT**:
a ridge refitted on W's own stand-in predictions reads the plate at 0.556–0.578 cm in val median,
an optimistic development reading. This task is the preregistration that row names.

**The question.** Under C1-M, does W's committed aim at 405, with the predicted plate read by
R-S (a dual ridge fitted on W's own predicted latents under the kinematic stand-in's chunks), reach
R9.8's primary claim, "LeWM-driven closed-loop success", on a fresh gated cohort? TASK-077's six
Stage T checkpoints are reused (no training). A fresh 500-root corpus (65300–65799) gives a gate
split whose aims are built from R-plate's reading, as the closed loop builds its grid, and a
true-plate contrast half for the corpus-aim confound. Stage R gates the readout on those fresh
roots and the offline aims before any closed loop; D and S carry TASK-077's bars and rows (56/64,
non-inferiority within δ = 8/64, McNemar wins over N, L-shuf, L-mean and L-rand). "LeWM needed" is
reported only.

**The protocol (DRAFT, not frozen):**
[`docs/experiments/apple_lewm_c1m_v2_pred_readout.md`](../../../docs/experiments/apple_lewm_c1m_v2_pred_readout.md).
Rulings R18.1–R18.14 (`docs/DECISIONS.md`, decision 2026-10-07), decided by Claude under owner
delegation (2026-09-30). The plan is `docs/PLAN.md`.

Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7. This task does not
change it unless an L-PASS is followed by its own reviewed ruling.

## Acceptance Criteria
- [ ] DRAFT preregistration reviewed independently (this card's first PR; docs only, plus
      #151's five minor nits).
- [x] Stage 0 PR: stage code in new modules, frozen block, manifest, tests (seed ranges against
      every forbidden range including 66000–68999; the reused artifacts' sha256s; W reads R-S and
      N reads R-N, never R8; gate-P aims built from p̂; the `"not evaluated"` sentinel; no
      privileged read in W or a twin), debug smokes, scale probes and R18.13's development dry run.
- [ ] K0′ on a reported GO; K0′'s values in the frozen block.
- [ ] The freeze merged on an independent reviewer's reported APPROVE.
- [ ] Stages C′, R, D and S, each on its own reported GO, from a clean worktree of the merged
      revision, after a full `pytest` at that revision (G-tests).
- [ ] Results PR, with every restated number checked by an independent reviewer.

## Notes
- 2026-10-07: card opened under R18.1 on branch `docs/task080-prereg-draft`. DRAFT protocol only;
  nothing run. Open questions for the freeze are the protocol's §15.
- 2026-10-07: Stage 0 (R18.15–R18.21; `docs/experiments/apple_lewm_c1m_v2_pred_readout_stage0.md`):
  new modules `lewm_pr_v2{,_runtime,_offline}.py` and `scripts/run_lewm_pr_v2.py`, DRAFT manifest,
  31 tests, debug smokes (65900–65999), caps, and R18.13's dry run at `e4babc1` (development, optimistic, no bar;
  seed 66800 flagged `last_two_triggered`; W 56.88/64 offline, N 25.94, W − N +30.9, predictions
  not closed-loop counts; report sha256 `e3becdfa…aed5`). §15 questions 1–5 ruled;
  6 recommended. Next: K0′ on a reviewer's GO, then the freeze.
