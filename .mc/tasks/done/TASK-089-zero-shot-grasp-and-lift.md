---
id: TASK-089
aliases:
- TASK-089
title: 'Zero-shot grasp-and-lift: offline diagnosis (Part A), then a closed loop with one declared change (Part B)'
slug: zero-shot-grasp-and-lift
status: done
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
grasp channel), the bar min(0.40, 0.9 × the IK follower's development rate). Not Apple→Plate; R7 does not
change.

Protocol: [zero_shot_grasp_v1.md](../../../docs/experiments/zero_shot_grasp_v1.md) (FROZEN at
`c40919d`, R26.1–R26.30); results
[zero_shot_grasp_v1_results.md](../../../docs/experiments/zero_shot_grasp_v1_results.md)
(R26.31–R26.36): **A-PASS(step-mean) → B-FAIL**; the abandonment clause fired.

## Acceptance criteria

- [x] DRAFT preregistration reviewed independently and merged (#197, `5a7c82a`)
- [x] Stage 0: code, tests, λ, debug smoke, A-K0 calibration and bars; reviewed; protocol frozen at merge (#198, `c40919d`)
- [x] Stage A (gated, 64 roots) run; row A-PASS(step-mean) recorded (#199); Part B admitted
- [x] Stage BD (development) recorded, bar 26/64; independent GO posted on #200 before Stage BS
- [x] Stage BS (gated, 64 resets) run; results document, row B-FAIL and DECISIONS (al) reviewed and merged
- [x] Evidence in `~/develop/emai/evidence/task089-*/` with `SHA256SUMS`; MP4s in `~/develop/emai/evidence/task089-bs/videos/`
- [x] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds 89000–89031 (A-K0), 890000–890063 (Stage A), 89100–89131 (BD), 891000–891063 (BS),
  89900–89999 (debug); salt 8901 (8902 reserved); TASK-088's salts 8801–8804 reused with new seeds.
%% mc-links: [[TASK-088]] %%
