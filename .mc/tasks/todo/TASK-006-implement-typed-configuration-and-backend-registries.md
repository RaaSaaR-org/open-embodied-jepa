---
id: TASK-006
aliases:
- TASK-006
title: Implement typed configuration and backend registries
slug: implement-typed-configuration-and-backend-registries
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

# Implement typed configuration and backend registries

## Description

Implement validated experiment configuration from configs/*.example.yaml and registries for models, embodiments, planners, and tasks. Resolve backend-specific defaults inside adapters.

## Acceptance Criteria

- [ ] Unknown names, unresolved required paths/scales, and incompatible schema/history settings fail before execution.
- [ ] Resolve inheritance, seeds, device selection and backend checkpoint mapping; save complete resolved configuration per run.
- [ ] Changing only world_model.backend selects the alternate adapter without editing planner, data, or task code.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
