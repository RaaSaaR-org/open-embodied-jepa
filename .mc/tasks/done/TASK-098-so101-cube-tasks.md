---
id: TASK-098
aliases:
- TASK-098
title: 'SO-101 S2: tasks so101_cube_lift_v0 and so101_cube_place_v0, scored from simulator truth, with the scripted follower and a placement-tolerance envelope measured'
slug: so101-cube-tasks
status: done
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

- [x] Both tasks registered lazily; scorer unit tests on synthetic trajectories (a transient
  crossing and a dropped cube do not count).
- [x] The scripted follower's success rates on development seeds are measured and reported with
  exact intervals, labelled scripted and development-only.
- [x] The placement-tolerance envelope is measured and reported on development seeds.
- [x] ruff and pytest stay green; PR reviewed independently and merged.

## Notes

- 2026-10-16, created by R28.6 (branch `docs/so101-plan`).
%% mc-links: [[TASK-097]] %%
- 2026-10-16, in review on branch `feat/so101-cube-tasks`: `so101_tasks.py` (declared resets from
  a kinematic reach map; held-window lift and at-rest place scorers on simulator truth; no
  latched crossings; synthetic-trajectory tests), `so101_followers.py` (privileged scripted lift
  and place followers with K-style grasp-aim and place-aim offsets), `config.py` additive (tasks
  registered lazily for `so101` only; SO-101 experiment configs still refused at load, since
  `benchmark.py` is G1-only), `scripts/measure_so101_tasks.py` and
  `docs/experiments/so101_cube_tasks_dev.md`. Development only: follower v1 64/64 on each task on
  seeds 100–163 (exact 95 % 0.944–1.000), direction-dependent grasp envelope, place-aim error up
  to 3 cm along any axis keeps ≥ 63/64; a first v0 run on seeds 0–63 is recorded. Scripted, not
  learned; gates nothing. Every G1 hash-pinned file is byte-identical.
- 2026-10-16, done: PR #234 independently reviewed (APPROVE, no blocking findings). Review
  fixes: the single-contact `gripper_contact` semantics and the unbounded release height are
  stated in the results doc, the `clipped` and home-pose frame wording corrected, a test asserts
  `so101_tasks.py`'s repeated scene constants match the scene, `FOLLOWER_V0` renamed
  `FOLLOWER_V1` (rename only), and the reach map's development-prompted collision check noted.
  Merged after green CI.
