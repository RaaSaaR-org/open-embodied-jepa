---
id: TASK-007
aliases:
- TASK-007
title: Implement canonical LeRobot transition and sequence loader
slug: implement-canonical-lerobot-transition-and-sequence-loader
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

- [x] Load image/state/action transitions and masked multi-step sequences without crossing episode boundaries.
- [x] Validate timestamps, joint ordering, actual-action alignment, missing required fields, and terminal windows with diagnostic errors.
- [x] Episode/session splits and training-only normalization are deterministic; fixture and manifest hashes reproduce across loaders.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

The canonical data API preserves T+1 observations, T applied actions, masks, timestamps and episode boundaries. All 36 dataset tests passed, including the pinned official LeRobot reader. The final leakage audit verifies preserved source splits, training-only normalization and zero exact frame/goal overlap across splits; see benchmarks/manifests/mvp-leakage-audit.json.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-004]] [[TASK-005]] %%
