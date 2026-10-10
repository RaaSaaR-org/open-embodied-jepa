---
id: TASK-088
aliases:
- TASK-088
title: 'Phase 3: zero-shot reach and grasp-and-lift by short-horizon planning'
slug: phase-3-zero-shot-reach-and-grasp
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- z3-reach
- world-model
- lewm
- planning
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-087]]"
due_date: ''
created: 2026-10-10
updated: 2026-10-10
---

# Phase 3: zero-shot reach and grasp-and-lift by short-horizon planning

## Description

Phase 3 of [JEPA_ZERO_SHOT_PLAN.md](../../../docs/JEPA_ZERO_SHOT_PLAN.md): CEM planning
(`embodied_jepa.planning.CEMPlanner`) over TASK-087's world models (plain LeWM P primary, G second
arm; R24.23 adopted) to new image + pose goals on fresh resets, for reach and grasp-and-lift of
`play-v1`'s objects with G1's right arm and right Dex3 in MuJoCo. Baselines hold, random and a
scripted IK follower. Not Apple→Plate; R7 does not change.

Protocol: [zero_shot_reach_grasp_v1.md](../../../docs/experiments/zero_shot_reach_grasp_v1.md)
(FROZEN at `028e6f3`, R25.1–R25.24; results R25.25–R25.29, row Z3-REACH).

## Acceptance criteria

- [x] DRAFT preregistration reviewed independently and merged (#191, `2271667`)
- [x] Stage 0: code, tests, λ, debug smoke, K0 calibration and bars; reviewed; protocol frozen at merge (#192, `028e6f3`)
- [x] Stage D (development) run and recorded (§16)
- [x] Independent GO posted as a PR comment before Stage S (#193)
- [x] Stage S (gated, 64 + 64) run; results document, row (Z3-REACH) and DECISIONS ruling (ag) reviewed and merged
- [x] Evidence in `~/develop/emai/evidence/task088-*/` with `SHA256SUMS`; two MP4s (one reach, one grasp)

## Notes

- Seeds 88000–88131 (K0), 88200–88315 (D), 880000–880063 / 881000–881063 (S), 88900–88999 (debug);
  salts 8801–8806.
