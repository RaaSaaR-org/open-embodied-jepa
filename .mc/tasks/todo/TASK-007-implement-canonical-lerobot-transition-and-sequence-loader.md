---
id: TASK-007
aliases:
- TASK-007
title: Implement canonical LeRobot transition and sequence loader
slug: implement-canonical-lerobot-transition-and-sequence-loader
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m1
- mvp
sprint: ''
depends_on:
- "[[TASK-004]]"
- "[[TASK-005]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Implement canonical LeRobot transition and sequence loader

## Description

Pin the supported LeRobot format and map its storage to the canonical schema. Include a tiny license-cleared fixture independent of real recordings.

## Acceptance Criteria

- [ ] Load image/state/action transitions and masked multi-step sequences without crossing episode boundaries.
- [ ] Validate timestamps, joint ordering, actual-action alignment, missing required fields, and terminal windows with diagnostic errors.
- [ ] Episode/session splits and training-only normalization are deterministic; fixture and manifest hashes reproduce across loaders.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
