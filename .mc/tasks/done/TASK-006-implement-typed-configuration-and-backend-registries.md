---
id: TASK-006
aliases:
- TASK-006
title: Implement typed configuration and backend registries
slug: implement-typed-configuration-and-backend-registries
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




# Implement typed configuration and backend registries

## Description

Implement validated experiment configuration from configs/*.example.yaml and registries for models, embodiments, planners, and tasks. Resolve backend-specific defaults inside adapters.

## Acceptance Criteria

- [x] Unknown names, unresolved required paths/scales, and incompatible schema/history settings fail before execution.
- [x] Resolve inheritance, seeds, device selection and backend checkpoint mapping; save complete resolved configuration per run.
- [x] Changing only world_model.backend selects the alternate adapter without editing planner, data, or task code.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Strict inherited configuration and lazy registries reject invalid settings before execution. Tests cover paths, schema requirements, device selection and backend-only overrides. All six final configurations have matching common settings; see docs/reviews/runtime-review.md.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-004]] [[TASK-005]] %%
