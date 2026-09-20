---
id: TASK-036
aliases:
- TASK-036
title: Align candidate feasibility with exact MJCF joint-limit acceptance
slug: align-candidate-feasibility-with-exact-mjcf-joint-limit-acceptance
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- planner
- actuator-constraints
- apple-pnp
sprint: ''
depends_on:
- "[[TASK-030]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---



# Align candidate feasibility with exact MJCF joint-limit acceptance

## Description

The first six-run apple control diagnostic exposed three projected-feasible commands that failed the exact MJCF joint-limit guard during execution. Reproduce from saved snapshots without WM inference and align the shared command preparation with the runtime guard while preserving physical limits and original negative results.

## Acceptance Criteria

- [x] All three saved late snapshots reproduce the original defect and accept their unchanged projected actions after the repair.
- [x] Scratch IK handles measured soft-limit penetration without altering live state; every returned command is validated before marking a candidate feasible.
- [x] Float32 transport endpoints remain inside exact MJCF limits; genuinely invalid targets remain rejected.
- [x] Regression tests cover the late physical state, both quantized endpoints, and an invalid solver return; existing embodiment/projection checks pass.
- [x] Preregistered bounded reproduction, checks and review evidence are recorded; original six-run result remains unchanged.

## Notes

Protocol and evidence: `docs/experiments/joint_limit_precision_v1.md`. No full control rerun is part of this repair. Independent implementation-agent review found no blocking defects; the coordinator also reviewed the final diff and passed 40 projection/embodiment/supervisor tests. All three saved snapshots accepted unchanged projected actions after the fix. Deliverable criteria are satisfied; PR #5 merge remains pending.
%% mc-links: [[TASK-030]] %%
