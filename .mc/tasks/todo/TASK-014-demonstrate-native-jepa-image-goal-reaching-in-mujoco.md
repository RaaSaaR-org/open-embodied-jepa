---
id: TASK-014
aliases:
- TASK-014
title: Demonstrate native JEPA image-goal reaching in MuJoCo
slug: demonstrate-native-jepa-image-goal-reaching-in-mujoco
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m2
- mvp
sprint: ''
depends_on:
- "[[TASK-010]]"
- "[[TASK-011]]"
- "[[TASK-013]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Demonstrate native JEPA image-goal reaching in MuJoCo

## Description

Train on the pilot, then run the first complete data-to-model-to-planner reach slice on the Mac.

## Acceptance Criteria

- [ ] Archive checkpoint, data hashes, complete config, commands, and fixed reset/goal cohort.
- [ ] Report one/multi-step prediction, collapse diagnostics, action ablations, CPU/MPS memory and synchronized planning latency.
- [ ] Demonstrate reach progress and compare success with random/hold and scripted controls; diagnose failures before adding grasp complexity.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
