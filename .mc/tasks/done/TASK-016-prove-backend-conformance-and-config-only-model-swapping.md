---
id: TASK-016
aliases:
- TASK-016
title: Prove backend conformance and config-only model swapping
slug: prove-backend-conformance-and-config-only-model-swapping
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m3
- mvp
sprint: ''
depends_on:
- "[[TASK-014]]"
- "[[TASK-015]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Prove backend conformance and config-only model swapping

## Description

Run one conformance suite and the identical CEM/MPC reach harness against native_jepa and leworldmodel.

## Acceptance Criteria

- [x] Both pass encode/predict/goal/distance/train/save/load shape, finiteness, action-sensitivity, and round-trip checks.
- [x] Switch only world_model.backend in a common input config; all dataset, robot, task, and planner source remains unchanged.
- [x] Resolved configs record checkpoint/default differences while common data/actions/goals/seeds/planning budgets remain equal.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Both adapters pass shared API, action-sensitivity, causal prediction and checkpoint tests. The final configurations share data, actions, goals, seeds and planning budgets. A fresh checkout changed only world_model.backend in an inherited YAML and completed simulation records for both models; see benchmarks/manifests/clean-reproduction.json.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-014]] [[TASK-015]] %%
