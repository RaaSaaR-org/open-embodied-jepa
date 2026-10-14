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

- [x] DRAFT preregistration and the R26.33 ruling reviewed independently and merged (#220)
- [x] Stage 0, reviewed, with the protocol frozen at its merge (#221). It covers:
  - opt-in modules for the projector, Stage P, the rows and the controller;
  - the projector's bitwise equivalence with `robot.project_candidates` and with executed
    commands;
  - the pairing check (an identity projector reproduces the unprojected plan bitwise);
  - Stage P's development calibration and degeneracy checks on 916000–916063;
  - the seed and hash checks;
  - the power recomputed with 20 000 simulations;
  - a timed debug smoke on 91940–91999.
- [x] Stage P recorded (P-FAIL escalates, with no clause)
- [ ] Stage BD recorded, the bar set, and an independent GO posted before Stage BS
- [ ] Stage BS recorded with its row; the clause fires on B-FAIL only; R7 unchanged

## Notes

- 2026-10-14, DRAFT (branch `docs/task093-draft`; R27.115–R27.131, DECISIONS (bd)). Development
  before the draft: code reading, the power table and the seed checks only, CPU only. Evidence
  `~/develop/emai/evidence/task093-draft/`.
- 2026-10-14, independent review of #220 applied: required edits 1–6 (base rate, honest reading,
  n = 128 by declaration with the clause risk below δ, a failable closed-loop equivalence check,
  float64 bounds as `_prepare_side`, the evidence seed statements) and the nits.
- 2026-10-14, Stage 0 (branch `feat/task093-stage0`; R27.132–R27.138, DECISIONS (be); record
  §16): **S0-PASS**. Projector bitwise against `robot.project_candidates` (15 360 steps, 0
  mismatches) and in the debug closed loop (M-proj 0 of 2 400 commands differ; M-v1 1 454 of
  2 206); calibration on cal-roots 916000–916063 sets Stage P's bars C8@R1 ≥ 39, H8@R2 ≥ 48 of 64,
  all degeneracy checks pass; power 0.976 at SD 1 (20 000 simulations), n = 128 stands; 0.54 GPU h.
  Evidence `~/develop/emai/evidence/task093-stage0/`. The protocol freezes at the Stage 0 PR's
  merge; Stage P is next.
- 2026-10-14, Stage P and Stage BD (branch `docs/task093-p-bd-record`; R27.139–R27.146, DECISIONS
  (bf); record §17–§18), both at `4d3491e` from a clean tree in `worktrees/task093-run`. **Stage P:
  P-PASS** (gated, offline, one run, 64 roots 917000–917063): M-proj C8@R1 39, 48, 54 (bar 39),
  H8@R2 58, 61, 58 (bar 48); M-v1 reported C8 43, 55, 54, H8 62, 64, 62; 281 s GPU. **Stage BD**
  (development, not gated, one run, 32 resets 91900–91931): ik 21 / 32, so the bar is 0.40 (BS
  pooled bar 156 / 384, per-seed floor 26 / 128); M-proj-87100 0 / 32, M-v1-87100 0 / 32, ik-nopress
  21 / 32, hold 0 / 32, random 0 / 32; projector equivalence held on all 11 650 M-proj commands;
  4 203 s GPU; one job waited 120 s for a V2D request on BOARD.md; the V2D session was not told
  separately before BD, a disclosed deviation from §10 and R27.137 that affects no result, and it is
  told explicitly before BS. Evidence `~/develop/emai/evidence/task093-{p,bd}/`. Stage BS waits for
  an independent GO on §18.
