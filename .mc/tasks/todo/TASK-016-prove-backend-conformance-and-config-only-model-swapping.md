---
id: TASK-016
aliases:
- TASK-016
title: Prove backend conformance and config-only model swapping
slug: prove-backend-conformance-and-config-only-model-swapping
status: backlog
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

- [ ] Both pass encode/predict/goal/distance/train/save/load shape, finiteness, action-sensitivity, and round-trip checks.
- [ ] Switch only world_model.backend in a common input config; all dataset, robot, task, and planner source remains unchanged.
- [ ] Resolved configs record checkpoint/default differences while common data/actions/goals/seeds/planning budgets remain equal.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
