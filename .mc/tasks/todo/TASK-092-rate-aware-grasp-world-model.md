---
id: TASK-092
aliases:
- TASK-092
title: 'Rate-aware grounded world model: train G on rate-equivalent grasp requests, gate it on rate fidelity, then the closed-loop grasp-and-lift comparison against the unchanged G'
slug: rate-aware-grasp-world-model
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- model
- planning
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-091]]"
due_date: ''
created: 2026-10-14
updated: 2026-10-14
---


# Rate-aware grounded world model: train G on rate-equivalent grasp requests, gate it on rate fidelity, then the closed-loop grasp-and-lift comparison against the unchanged G

## Description

R27.71's model change after TASK-091 closed at **B-FAIL** and its clause fired. The development
measurements behind it (protocol §1.2, §11; not gated, CPU only) are these:

- The right-grasp channel is rate-limited to 2 / 11 per command, and no `play-v1` training step
  exceeds that.
- In TASK-091 Stage BS (one run, simulation only), 28–45 % of the chosen plans' grasp steps per
  seed did exceed it.
- Under grasp requests the embodiment executes identically to the recorded ones, the G models
  predict the right-hand state about 7–11 times worse at h = 8 (826 val roots).

TASK-092 trains TASK-087's G with rate-equivalent grasp requests on half of the windows (**G-req**).
Everything else stays TASK-087's: the corpus, latent, recipe, budget and model seeds 87100–87102,
so G-req is exactly paired with the unchanged G. The stages are:

- **Stage T:** three training jobs.
- **Stage R** (offline, `play-v1` test, gates the closed loop):
  - R1, rate fidelity: the hand-state ratio's upper bound must be ≤ 1 + 0.25 × (the paired
    control's ratio − 1). The measured ceiling is 1.
  - R2, no regression: within δ measured from the control seeds.
  - E, validity.
- **Stage A‴:** TASK-091's A″ on fresh roots 908000–908063, as a no-regression gate.
- **Closed loop:** BD (90400–90431), an independent GO, then BS (n = 128, 909000–909127). M-req
  runs against M-v1 (TASK-087's G, unchanged), with TASK-089's step-averaged planner frozen. The
  bar is min(0.40, 0.9 r). Rows: B-VOID, B-PASS, B-GAIN, B-BAR, B-FAIL and B-UNDER. The clause
  fires on B-FAIL only.

This is not Apple→Plate. R7 does not change.

Protocol: [grasp_rate_aware_wm_v1.md](../../../docs/experiments/grasp_rate_aware_wm_v1.md)
(**DRAFT**, R27.74–R27.88; DECISIONS (ax)).

## Acceptance criteria

- [ ] DRAFT preregistration reviewed independently and merged
- [ ] Stage 0, reviewed, with the protocol frozen at its merge. It covers:
  - opt-in modules for the augmentation, the loss subclass, the training driver, the guarded
    loader, Stage R, A‴ and the rows;
  - the pairing check (p = 0 gives TASK-087's weights bitwise on debug seed 87900);
  - the seed and hash checks;
  - the power recomputed with 20 000 simulations;
  - a timed debug smoke on 90500–90599.
- [ ] Stage T recorded (T-VOID on a failed check or a cap hit)
- [ ] Stage R recorded (R-PREMISE, E-FAIL or R-FAIL escalate, with no clause)
- [ ] Stage A‴ recorded (A‴-FAIL escalates, with no clause)
- [ ] Stage BD recorded, the bar set, and an independent GO posted before Stage BS
- [ ] Stage BS run; the record, row and DECISIONS rulings reviewed and merged
- [ ] Evidence in `~/develop/emai/evidence/task092-*/` with `SHA256SUMS`, and two MP4s (one
  success if any, one failure)
- [ ] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds. Model seeds 87100–87102 (TASK-087's, reused for pairing). A‴ roots 908000–908063. BD
  90400–90431. BS 909000–909127, as two 64-reset jobs per world-model run. Debug 90500–90599. The
  pairing check uses 87900.
- Salts. New: 9008 (the augmentation) and 9009 (Stage R's random requests). Reused: TASK-087's
  8711 and 8713, TASK-088's 8801–8804 and TASK-090's 9002, 9004 and 9006.
- Budget, about 17 GPU h: Stage T 1.9, BD 1.2 and BS 12.8, the rest under 1. Every job runs under
  the 7 200 s cap and one `gpu_run.sh --wait` lock, and BOARD.md's Requests are checked first.
- `worktrees/task087-run` (TASK-087's feature store and projection) must not be archived or
  removed until TASK-092 closes.
- The development evidence behind the DRAFT is `~/develop/emai/evidence/task092-prototype/`
  (`SHA256SUMS` sha256 `9d9c3fa9…d1e8`).
