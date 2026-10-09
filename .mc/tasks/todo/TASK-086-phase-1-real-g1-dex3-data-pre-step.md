---
id: TASK-086
aliases:
- TASK-086
title: 'Phase 1 real-data pre-step: license check and conversion of open real G1 + Dex3 data'
slug: phase-1-real-g1-dex3-data-pre-step
status: in-progress
priority: 2
owner: ''
projects: []
customers: []
tags:
- real-data
- data
- licensing
- preregistration
- phase-1
sprint: ''
depends_on:
- "[[TASK-084]]"
due_date: ''
created: 2026-10-09
updated: 2026-10-09
---


# Phase 1 real-data pre-step: license check and conversion of open real G1 + Dex3 data

## Description

The pre-step of [JEPA_ZERO_SHOT_PLAN.md § Open real G1 data](../../../docs/JEPA_ZERO_SHOT_PLAN.md#open-real-g1-data):
confirm licenses (Unitree G1_Dex3 sets Apache-2.0, NVIDIA GR00T-Teleop-G1 CC-BY-4.0; NVIDIA
GR00T-N1.7-AppleToPlate held out as a real-image test set only; Humanoid Everyday G1 not used
while its license is unresolved), convert states and actions to `ee_delta_grasp_v0` with our G1
forward kinematics and Dex3 synergy projection, convert images for the sets that fit the disk
budget, and apply the plan's stop rule (drop the real-data arm if most converted steps fall
outside [−1, 1] or the grasp fit is poor).

**The preregistration (R23.10–R23.15, DRAFT; frozen at the converter PR's merge):**
[`docs/experiments/real_g1_dex3_prestep.md`](../../../docs/experiments/real_g1_dex3_prestep.md).
Not an Apple→Plate experiment and not a learned result; R7 does not change.

## Acceptance Criteria

- [x] DRAFT preregistration independently reviewed (APPROVE at afa9c74) and merged (#182)
- [ ] Converter, tests and license record; independently reviewed and merged (freezes it)
- [ ] Downloads within the disk budget, conversion run, evidence under
  `~/develop/emai/evidence/task086-*` with `SHA256SUMS`
- [ ] Results (R-KEEP / R-DROP-RANGE / R-DROP-GRASP / R-VOID) and DECISIONS entry, reviewed and
  merged; R7 unchanged
