---
id: TASK-098
aliases:
- TASK-098
title: 'SO-101 S2: tasks so101_cube_lift_v0 and so101_cube_place_v0, scored from simulator truth, with the scripted follower and a placement-tolerance envelope measured'
slug: so101-cube-tasks
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- task
- simulation
sprint: ''
depends_on:
- "[[TASK-097]]"
due_date: ''
created: 2026-10-16
updated: 2026-10-16
---

# SO-101 S2: tasks so101_cube_lift_v0 and so101_cube_place_v0, scored from simulator truth, with the scripted follower and a placement-tolerance envelope measured

## Description

Phase S2 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md) (DECISIONS 2026-10-16 (bp), R28.6–R28.7).
Task modules own resets (declared cube and target distributions), goals, termination and scoring.
Scoring uses simulator truth only and never reaches model inputs or planning costs. Lift: the cube
held above a declared height for a declared duration. Place: the cube at rest inside the target
region after release; transient crossings are not latched (the TASK-067 scorer defect is not
repeated).

A privileged scripted follower is measured on development seeds, as are rigid gripper offsets of
a few sizes (a TASK-094 K-style placement-tolerance envelope). These measurements set what later
bars can be; they are scripted, not learned results, and gate nothing.

## Acceptance Criteria

- [ ] Both tasks registered lazily; scorer unit tests on synthetic trajectories (a transient
  crossing and a dropped cube do not count).
- [ ] The scripted follower's success rates on development seeds are measured and reported with
  exact intervals, labelled scripted and development-only.
- [ ] The placement-tolerance envelope is measured and reported on development seeds.
- [ ] ruff and pytest stay green; PR reviewed independently and merged.

## Notes

- 2026-10-16, created by R28.6 (branch `docs/so101-plan`).
