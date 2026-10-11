---
id: TASK-089
aliases:
- TASK-089
title: 'Zero-shot grasp-and-lift: offline diagnosis (Part A), then a closed loop with one declared change (Part B)'
slug: zero-shot-grasp-and-lift
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
- "[[TASK-088]]"
due_date: ''
created: 2026-10-11
updated: 2026-10-11
---

# Zero-shot grasp-and-lift: offline diagnosis (Part A), then a closed loop with one declared change (Part B)

## Description

The grasp-focused task DECISIONS R25.27 recommends after TASK-088's Z3-REACH (grasp-and-lift 0/64
for every world-model arm; IK follower 50/64). Part A, offline on fresh branched grasp windows: do
TASK-087's G and P (seed 87100) predict a lower goal cost for closing and lifting than for an open
or non-lifting hand at the object, against the simulator's measured ceiling, and does CEM, run once
there, choose to close and lift? Part B, only if Part A admits it: a closed-loop grasp-and-lift
test on fresh gated resets with the one change Part A selects (step-averaged cost, H = 16 or a held
grasp channel), the bar from the IK follower's development rate. Not Apple→Plate; R7 does not
change.

Protocol: [zero_shot_grasp_v1.md](../../../docs/experiments/zero_shot_grasp_v1.md) (DRAFT,
R26.1–R26.18).

## Acceptance criteria

- [ ] DRAFT preregistration reviewed independently and merged
- [ ] Stage 0: code, tests, λ, debug smoke, A-K0 calibration and bars; reviewed; protocol frozen at merge
- [ ] Stage A (gated, 64 roots) run; row recorded; Part B admitted or the task closed with results and a recommendation
- [ ] If Part B: Stage BD (development) recorded and the bar set; independent GO posted before Stage BS
- [ ] If Part B: Stage BS (gated, 64 resets) run; results document, row and DECISIONS rulings reviewed and merged
- [ ] Evidence in `~/develop/emai/evidence/task089-*/` with `SHA256SUMS`; if Part B, two MP4s (one success if any, one failure)
- [ ] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds 89000–89031 (A-K0), 890000–890063 (Stage A), 89100–89131 (BD), 891000–891063 (BS),
  89900–89999 (debug); salt 8901 (8902 reserved); TASK-088's salts 8801–8804 reused with new seeds.
