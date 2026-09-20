---
id: TASK-008
aliases:
- TASK-008
title: Implement G1 action normalization, IK, hand synergies, and limits
slug: implement-g1-action-normalization-ik-hand-synergies-and-limits
status: done
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

- [x] Document and test both arm/hand mappings, base frame, units, normalization round trips, and saturation.
- [x] Reachable EE targets solve through IK and calibrated Dex3 synergies; unreachable targets and stale commands fail predictably.
- [x] Joint/workspace/rate limits, control interpolation, and applied-action reporting have boundary checks; experimental joint actions use a separate schema.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Named mappings, physical units, simulation scales, SO(3) IK and hand synergy limits are implemented and tested for both sides, including stop/resume and 180-degree orientation regressions. Physical hardware calibration remains disabled. The planner’s requested/applied grasp mismatch is disclosed separately in TASK-029; see docs/SIMULATION.md.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-003]] [[TASK-004]] [[TASK-005]] %%
