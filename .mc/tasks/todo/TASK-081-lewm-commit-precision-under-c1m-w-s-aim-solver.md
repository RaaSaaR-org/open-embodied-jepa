---
id: TASK-081
aliases:
- TASK-081
title: "LeWM commit precision under C1-M: W's aim solver"
slug: lewm-commit-precision-under-c1m-w-s-aim-solver
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- design-note
- preregistration
- frozen
sprint: ''
depends_on:
- "[[TASK-080]]"
due_date: ''
created: 2026-10-08
updated: 2026-10-08
---

# LeWM commit precision under C1-M: W's aim solver

## Description

TASK-080 closed **L-NEAR** (`docs/experiments/apple_lewm_c1m_v2_pred_readout_results.md`,
R18.30–R18.32): W 58/64 passed G-bar and the four twin tests but failed G-NI against the
non-learned rule controller H-rule 63/64 (W − H-rule −5/64, 95 % [−10, 0] against −8/64). R18.33
(decided by Claude under owner delegation) pursues R18.32's direction (b), W's commit precision,
first; (a), a condition in which no hand-written arm is given the plate law, stays the documented
alternative (a task change).

**The design note (DRAFT):**
[`docs/experiments/apple_lewm_commit_precision_v2_design.md`](../../../docs/experiments/apple_lewm_commit_precision_v2_design.md).
Diagnosis: W's single-evaluation refinement oscillates on a predicted-plate map that is rough
below about 1 cm and, at its cap, commits one extreme of the oscillation. Development check
(seeds 70000–70063, CPU, one run, not gated; R18.35): W:frozen 57/64, W:damped 58/64,
W:affine_local 62/64 against H-rule 64/64 and the privileged H-final(commit) 63/64;
affine_local was picked as the best of three executed variants after seeing them, so its 62/64
is optimistic and needs fresh seeds.

**The preregistration (DRAFT, R19.1–R19.11):**
[`docs/experiments/apple_lewm_commit_precision_v2.md`](../../../docs/experiments/apple_lewm_commit_precision_v2.md).
W's solver after the grid becomes affine_local (no retraining, no refit; R-S and R-N from TASK-080's
Stage R); the twins carry the same solver; W-frozen (TASK-080's W) is reported only; no K0, corpus or
offline gate (τ_commit 1.0 cm carried); D 16 resets (70100–70115) with TASK-080's stops; S 128
resets (70200–70327) with G-bar 112/128, G-NI within δ = 16/128 against the better of H-rule and
H-sysid, and the four McNemar tests; debug 71900–71999; salts 8302–8304.

**The question for the preregistration.** Under C1-M, with W-66800 and R-S unchanged and only W's
solver after the grid changed to affine_local (the clipped fixed point of a least-squares affine
fit to W's own grid predictions near the argmin), does W reach R9.8's primary claim (G-bar, G-NI
within δ = 8/64 against the better of H-rule and H-sysid, and the four twin tests) on fresh
gated resets? Seeds 70100–71999 (R18.34), salts 8301–8312.

Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7 (as updated by
R18.31 and R19.18). R19.18 corrected its last clause with this task's development counts (a
factual correction, not a success claim); a gated LeWM success needs an L-PASS followed by its own
reviewed ruling.

## Acceptance Criteria
- [x] DRAFT design note with the diagnosis, candidate fixes, a development feasibility check on
      development seeds only, G-NI's power and the margin/cohort discussion (this card's first
      PR).
- [x] DRAFT preregistration reviewed independently (solver, arms including a reported-only
      W-frozen, δ, cohort size, seeds 70100–71999, rows and clause).
- [x] Stage 0: code in new modules, frozen block, manifest, tests, debug smokes (R19.12–R19.14,
      `docs/experiments/apple_lewm_commit_precision_v2_stage0.md`; this card's third PR).
- [x] The freeze merged on an independent reviewer's reported APPROVE (R19.15; this card's fourth
      PR; ticked in the PR, in force on merge).
- [ ] Stages D and S, each on its own reported GO, from a clean worktree of the merged revision.
      Stage D: **D-PASS** (R19.17; W 16/16, W-frozen 15/16, N 5/16, L-shuf 5/16, L-mean 8/16,
      H-rule 16/16, privileged H-final(commit) 16/16; development, not gated). Stage S: plan R19.19.
- [ ] Results PR, with every restated number checked by an independent reviewer.

## Notes
- 2026-10-08: Stage D ran on #163's GO at `2a6b633` (report `b58ae61f…e920`): D-PASS; R7's last
  clause corrected (R19.18); Stage S's plan in protocol §6.6 (R19.19).
- 2026-10-08: #162 (Stage 0) merged at `1ca1258`; the freeze (R19.15–R19.16) on branch
  `feat/task081-freeze`, frozen sha `77ccc636…705a`; Stage D's plan in protocol §6.4.
- 2026-10-08: Stage 0 on branch `feat/task081-stage0`: code, 43 tests, debug smokes on
  71900–71999 only, caps and power (salt 8304, report `5043c3ab…a56c`); frozen-block candidate
  `857a3078…32ef`; nothing of D or S simulated.
- 2026-10-08: #161 (the DRAFT) merged at `74a4674` on an independent reviewer's APPROVE at
  `33edc80` (nine non-blocking nits, applied or answered in R19.13–R19.14).
- 2026-10-08: DRAFT preregistration written on branch `docs/task081-prereg` (R19.1–R19.11,
  DECISIONS 2026-10-08 (f)); nothing simulated.
- 2026-10-08: card opened under R18.33–R18.35 on branch `docs/task081-commit-precision-design`.
  Development check at `3a24abb` (report `outputs/task081-dev-1/report.json`, sha256
  `febe7280…fb7f`; evidence copy `~/develop/emai/evidence/task081-dev/`). Nothing gated was run;
  no seed of TASK-076/077/080's ranges was simulated.
