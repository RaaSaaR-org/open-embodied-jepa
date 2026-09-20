---
id: TASK-009
aliases:
- TASK-009
title: Build MuJoCo G1 tabletop scene and embodiment adapter
slug: build-mujoco-g1-tabletop-scene-and-embodiment-adapter
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
- "[[TASK-003]]"
- "[[TASK-006]]"
- "[[TASK-008]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Build MuJoCo G1 tabletop scene and embodiment adapter

## Description

Provide reset, timestamped RGB/state observations, normalized action execution, and ground-truth task instrumentation in a stabilized G1/Dex3 MuJoCo scene.

## Acceptance Criteria

- [ ] Exercise both hands/arms through the common Embodiment API and confirm camera/state synchronization.
- [ ] Reset from a seed with recorded object/plate poses; create image goals without exposing simulator truth to the model or planner.
- [ ] Record a scripted reach rollout and validate contact/pose evaluator access separately from policy observations.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
