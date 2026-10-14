---
id: TASK-093
aliases:
- TASK-093
title: 'Rate-projected planner: project CEM candidates onto the grasp-rate limit before scoring, gate it offline on executed measures, then the closed-loop grasp-and-lift comparison against the unprojected planner on TASK-087''s G'
slug: rate-projected-grasp-planner
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- planning
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-092]]"
due_date: ''
created: 2026-10-14
updated: 2026-10-14
---


# Rate-projected planner: project CEM candidates onto the grasp-rate limit before scoring, gate it offline on executed measures, then the closed-loop grasp-and-lift comparison against the unprojected planner on TASK-087's G

## Description

This is the planner-side fix that R27.104 and R27.113 route to after D-RATE′ read D-FLAT
(development, one run, simulation only, 32 resets: 0 / 32 against 0 / 32). It is a planner variant
over TASK-087's checkpoints, so it falls under R26.33. R27.115 rules that R26.33's condition
("new evidence of a different kind") is met for this one variant only. The evidence is about the
planner–model interface:

- TASK-092 Stage R (one run, offline, simulation only): G's hand-state error under
  execution-identical saturated requests is 15–20 times its error on the recorded commands.
- TASK-092's A‴ record: G's plans predict a hand that closes about twice as fast as it can.
- D-RATE′ (development): 0.47 of M-v1-87100's chosen plan steps were off support.

The design:

- **The change.** Each CEM candidate's right-grasp channel is clipped cumulatively onto the
  embodiment's rate set (2 / 11 per command, `_prepare_side`'s own formula, from the last accepted
  hand targets) before the model rollout and the cost. It uses `CEMPlanner`'s existing `projector`
  hook, and no frozen file is edited.
- **Frozen:** TASK-087's G-87100–87102, the latent, the step-averaged cost, λ = 2.0552, the CEM
  budget, the goals and the scorer.
- **Arms:** M-proj ×3 (primary, pooled) against M-v1 ×3 (the same checkpoints, unprojected; the
  gated control), with ik, ik-nopress, hold and random.
- **Stage 0** calibrates the executed-measure offline gate (C8@R1, H8@R2) on development roots
  916000–916063, with degeneracy checks.
- **Stage P** is the gate, on fresh roots 917000–917063.
- **Closed loop:** BD (91900–91931), an independent GO, then BS (n = 128, 918000–918127). The bar
  is min(0.40, 0.9 r), with reset-level sign-flip tests. Rows: B-VOID, B-PASS, B-GAIN, B-BAR,
  B-FAIL (δ = +0.10) and B-UNDER. The clause fires on B-FAIL only.
- **Power** (development): 0.97 at SD 1 for 0.15 against 0.05 per seed. The smallest effect
  detected at about 0.80 is 0.05 → 0.12; a doubling to 0.10 is not reliably detected, and ends in B-FAIL (the
  clause fires) about half the time (development row simulation, SD 1, 400 simulations). n = 128
  stands by declaration: δ = +0.10 is the smallest pooled gain worth keeping the line open for.
- **Budget:** about 13–16 GPU h.

This is not Apple→Plate. R7 does not change.

Protocol: [grasp_rate_projected_planner_v1.md](../../../docs/experiments/grasp_rate_projected_planner_v1.md)
(DRAFT R27.115–R27.131, DECISIONS (bd)).

## Acceptance criteria

- [ ] DRAFT preregistration and the R26.33 ruling reviewed independently and merged
- [ ] Stage 0, reviewed, with the protocol frozen at its merge. It covers:
  - opt-in modules for the projector, Stage P, the rows and the controller;
  - the projector's bitwise equivalence with `robot.project_candidates` and with executed
    commands;
  - the pairing check (an identity projector reproduces the unprojected plan bitwise);
  - Stage P's development calibration and degeneracy checks on 916000–916063;
  - the seed and hash checks;
  - the power recomputed with 20 000 simulations;
  - a timed debug smoke on 91940–91999.
- [ ] Stage P recorded (P-FAIL escalates, with no clause)
- [ ] Stage BD recorded, the bar set, and an independent GO posted before Stage BS
- [ ] Stage BS recorded with its row; the clause fires on B-FAIL only; R7 unchanged

## Notes

- 2026-10-14, DRAFT (branch `docs/task093-draft`; R27.115–R27.131, DECISIONS (bd)). Development
  before the draft: code reading, the power table and the seed checks only, CPU only. Evidence
  `~/develop/emai/evidence/task093-draft/`.
- 2026-10-14, independent review of #220 applied: required edits 1–6 (base rate, honest reading,
  n = 128 by declaration with the clause risk below δ, a failable closed-loop equivalence check,
  float64 bounds as `_prepare_side`, the evidence seed statements) and the nits.
