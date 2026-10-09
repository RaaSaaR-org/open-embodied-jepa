---
id: TASK-084
aliases:
- TASK-084
title: 'Phase 0: calibrate on Meta''s released JEPA-WM (Push-T) with its planner and ours; LeWM action-sensitivity baseline'
slug: phase-0-calibrate-on-meta-s-released-jepa-wm-and-baseline-lewm-action-sensitivity
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- jepa-wms
- calibration
- planning
- preregistration
- phase-0
- p0-pass
sprint: ''
depends_on:
- "[[TASK-083]]"
due_date: ''
created: 2026-10-09
updated: 2026-10-09
---



# Phase 0: calibrate on Meta's released JEPA-WM (Push-T) with its planner and ours; LeWM action-sensitivity baseline

## Description

Phase 0 of [JEPA_ZERO_SHOT_PLAN.md](../../../docs/JEPA_ZERO_SHOT_PLAN.md) ("Close and calibrate";
the plate line was closed by TASK-083, #177). Part A (gated): reproduce the published Push-T
success rate of Meta's released JEPA-WM checkpoint (arXiv 2512.24497, 70.2 %, CEM L2) on this
machine with Meta's planner, and run the same model, cost and episodes with this repository's
`embodied_jepa.planning.CEMPlanner`; gate: upstream within ±10 points of 70.2 and ours at most 10
points below upstream, else fix the planner first. Part B (reported only): action sensitivity and
short-step command ranking of TASK-077's W checkpoints (seeds 66800–66802) on the val split, a
baseline for Phase 2.

**The preregistration (R22.1–R22.9, decided by Claude under owner delegation; frozen at merge):**
[`docs/experiments/jepa_wms_pusht_calibration.md`](../../../docs/experiments/jepa_wms_pusht_calibration.md).
Not an Apple→Plate experiment; R7 does not change.

## Acceptance Criteria

- [x] Preregistration, runner, decision script, part B probe and tests, independently reviewed
  (APPROVE) and merged (#178, `176258b`)
- [x] Smoke at the merged revision (both arms, debug seed 9902), then part A's eight jobs (96
  paired episodes per arm) and part B, evidence under `~/develop/emai/evidence/task084-*` with
  `SHA256SUMS`
- [x] Results document and DECISIONS entry with the measured row, independently reviewed and
  merged; R7 unchanged, stated explicitly

## Outcome

**P0-PASS** (R22.10–R22.13). Part A: Meta's released Push-T JEPA-WM with Meta's planner 67/96
(69.8 %, Wilson 60.0–78.1 %) against the published 70.2 %; `embodied_jepa.planning.CEMPlanner`
83/96 on the same 96 episodes (ours − upstream +16.7 points, paired 95 % [+7.3, +26.0]; not a
preregistered superiority claim). Part B (reported only; offline; val roots that were used for checkpoint selection): W-66800..66802 wrong/true 1.04 at h = 1,
top-1 of 16 at h = 1 14–19 %, at h = 60 90–94 %. Evidence `~/develop/emai/evidence/task084-run/`,
`task084-probe/`. R7 unchanged. Results:
[`jepa_wms_pusht_calibration_results.md`](../../../docs/experiments/jepa_wms_pusht_calibration_results.md).
%% mc-links: [[TASK-083]] %%
