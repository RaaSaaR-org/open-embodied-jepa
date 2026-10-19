---
id: TASK-101
aliases:
- TASK-101
title: 'SO-101 S5: first closed loop on so101_cube_lift_v0 in simulation (CEM over TASK-100''s native_jepa token model J toward image subgoals of a per-reset demonstration)'
slug: so101-closed-loop-lift
status: in-progress
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- world-model
- closed-loop
- preregistration
sprint: ''
depends_on:
- "[[TASK-100]]"
due_date: ''
created: 2026-10-19
updated: 2026-10-19
---


# SO-101 S5: first closed loop on so101_cube_lift_v0 in simulation (CEM over TASK-100's native_jepa token model J toward image subgoals of a per-reset demonstration)

## Description

Phase S5 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md), opened by Claude under owner delegation
(DECISIONS 2026-10-19 (bt), R28.41–R28.58) after TASK-100's row WM-PASS with arm J only
(R28.34–R28.40). MPC by `SchemaCEMPlanner` (unchanged) over `joint_delta_so101_v0`, with
candidates projected by the embodiment's own chained `preview` so the planner scores what will be
executed, using TASK-100's J checkpoints (primary J-100101 by the lowest val criterion; J-100100
and J-100102 reported). The cost is the latent error to four image subgoals from a privileged
per-reset demonstration by follower v1, and switching reads proprioception and the goal pose. Arms:
W, the action-blind twin N, goal-blind W-shuf, scene-blind W-mean, rand, hold, the privileged
follower F (it replays the goal demonstration, so it scores every kept reset by construction and is a
determinism guard, not a measured ceiling) and the non-learned pose follower H-pose (the measured
reference). Stage 0 calibrates the horizon and
the cost form on calibration seeds and sets the bar from a development closed loop:
b = min(0.9 × r_F(D), max(0.50, LCB(W(D)))), where r_F(D) = 1 by construction, so the cap is 0.90
and LCB(W(D)) binds. The gated stage runs 128 fresh resets. Rows: VOID /
CL-PASS / CL-UNSPECIFIC / CL-LOW / CL-FAIL. There is no abandonment clause; a
development decomposition comes before any second variant.

**The preregistration (DRAFT):**
[`docs/experiments/so101_closed_loop_lift_v1.md`](../../../docs/experiments/so101_closed_loop_lift_v1.md).
Simulation only. Not an Apple→Plate experiment; R7 does not change (R28.9). Not LeWM: J passed S4,
and the LeWM long-budget re-run is a separate recommendation (R28.57) that has not been started.

## Acceptance Criteria

- [x] DRAFT preregistration independently reviewed and merged (#239, `d628fae`)
- [ ] Stage 0 PR: code in new files only (`so101_closed_loop.py`, `so101_closed_loop_runtime.py`,
  `run_task101.py`, `summarize_task101.py`, `run_task101_stage.sh`, manifest
  `benchmarks/manifests/so101-closed-loop-lift-v1.json`). G1 byte-pinned files, TASK-100's pinned
  files, `schema_planning.py`, `so101*.py` and `config.py` stay unchanged, and the core-import check
  passes. Tests; the checks of §7.2 (live latent, readout sanity, projector, determinism, timing);
  K's configuration rule on 101010–101041; D's development closed loop on 101100–101163; bar,
  power and GPU projection; Stage 0 row and record
  `docs/experiments/so101_closed_loop_lift_v1_stage0.md`; independently reviewed and merged (the
  freeze)
- [ ] Gated stage S on 101200–101327 (reserve 101400–101463) at the frozen revision in a clean run
  tree, every GPU job under `scripts/gpu_run.sh --wait --board`, within the 20 GPU h cap (Stage 0
  ≤ 6); evidence under `~/develop/emai/evidence/task101-*` with `SHA256SUMS`
- [ ] Results document and DECISIONS entry with the row, reviewed and merged; R7 unchanged

## Notes

- Seeds and salts (10101–10106) checked unused across all remote refs on 2026-10-19.
- TASK-100's checkpoints are read only (`checkpoints/task100-run/`, HDD); nothing is retrained.
- 2026-10-19 Stage 0 (branch `feat/so101-s5-stage0`): code, tests and checks done; row **S0-VOID,
  escalated** (R28.59–R28.64). Check 7.2.2 failed: R-enc reads the cube at 6.106 cm on D's goal
  frames (bar 2.397 cm) because the lift scene's inert target marker (x 0.30, y −0.20 m) lies
  outside the play corpus's marker box; with only the marker moved, 1.048 cm. K scored 0/32 in all
  four configurations under that condition (not attributed to J); K ran before checks 7.2.2 and
  7.2.3 had been executed (a deviation; it spent K, so a rerun uses K′). D's goals (64/64 kept) and
  CPU-arm evidence ran (F 8/8, H-pose 8/8, rand 0/8, hold 0/8); D's CEM closed loop did not. By
  §7.6 row 1's text a rerun uses D′ (recommended, unless the ruling waives it with a reason). Fixes under §7.6 row 1 before any count: the live
  latent uses TASK-100's batch of 64 (bit-identical), a fresh process per episode, and the
  determinism check in its conditional form (EGL rendering is not bit-stable). 1.66 GPU h used.
  Record `docs/experiments/so101_closed_loop_lift_v1_stage0.md`; evidence
  `~/develop/emai/evidence/task101-stage0` (SHA256SUMS `9ee6638d…`). Awaiting a ruling (R28.63).
