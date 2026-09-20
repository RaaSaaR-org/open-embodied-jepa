---
id: TASK-012
aliases:
- TASK-012
title: Implement model-independent CEM planning
slug: implement-model-independent-cem-planning
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
- "[[TASK-004]]"
- "[[TASK-006]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Implement model-independent CEM planning

## Description

Sample and refine bounded action sequences against WorldModel prediction costs. Start with terminal goal distance and configurable shared action penalties.

## Acceptance Criteria

- [ ] Known-dynamics tests show better goal progress than random/hold candidates and preserve action bounds.
- [ ] Seeded sampling, elite selection, candidate chunking, finite-cost handling, and inference-only execution are verified.
- [ ] No model/robot-specific branches; horizon, samples, iterations, and measured memory/latency are logged.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
