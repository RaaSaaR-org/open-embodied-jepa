---
id: TASK-008
aliases:
- TASK-008
title: Implement G1 action normalization, IK, hand synergies, and limits
slug: implement-g1-action-normalization-ik-hand-synergies-and-limits
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
- "[[TASK-003]]"
- "[[TASK-004]]"
- "[[TASK-005]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Implement G1 action normalization, IK, hand synergies, and limits

## Description

Implement robot-specific conversion behind the embodiment interface, using the verified MuJoCo robot. Calibrate physical scales rather than inventing deployment values.

## Acceptance Criteria

- [ ] Document and test both arm/hand mappings, base frame, units, normalization round trips, and saturation.
- [ ] Reachable EE targets solve through IK and calibrated Dex3 synergies; unreachable targets and stale commands fail predictably.
- [ ] Joint/workspace/rate limits, control interpolation, and applied-action reporting have boundary checks; experimental joint actions use a separate schema.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
