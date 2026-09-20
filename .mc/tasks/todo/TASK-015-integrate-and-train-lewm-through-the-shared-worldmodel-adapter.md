---
id: TASK-015
aliases:
- TASK-015
title: Integrate and train LeWM through the shared WorldModel adapter
slug: integrate-and-train-lewm-through-the-shared-worldmodel-adapter
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
- "[[TASK-002]]"
- "[[TASK-010]]"
- "[[TASK-011]]"
- "[[TASK-006]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Integrate and train LeWM through the shared WorldModel adapter

## Description

Wrap the pinned external architecture with internal preprocessing/batch conversion and train on the same canonical G1 corpus. Preserve upstream attribution.

## Acceptance Criteria

- [ ] Implement all WorldModel methods, action-conditioned multi-step prediction, compatible goal distance, and checkpoint validation.
- [ ] Run finite-gradient training and save/load on CPU or validated MPS; record any operator fallback and compute cost.
- [ ] No upstream dataset, planner, or simulator replaces shared common-mode components; no unavailable future state enters inference.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
