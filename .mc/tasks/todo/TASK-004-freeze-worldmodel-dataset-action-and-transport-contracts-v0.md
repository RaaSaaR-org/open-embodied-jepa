---
id: TASK-004
aliases:
- TASK-004
title: Freeze WorldModel, dataset, action, and transport contracts v0
slug: freeze-worldmodel-dataset-action-and-transport-contracts-v0
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- m0
- mvp
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Freeze WorldModel, dataset, action, and transport contracts v0

## Description

Turn docs/ARCHITECTURE.md into an explicit contract specification with examples. Preserve every PRD API method and opaque model-owned latents.

## Acceptance Criteria

- [ ] Fix RGB/state/sequence axes, rollout and cost shapes, timestamps, validity masks, and backend capability negotiation.
- [ ] Specify EE frame/rotation composition, 14-component candidate action ordering, hand synergy semantics, and separate joint-space schema.
- [ ] Specify MuJoCo / future Isaac / SDK2 transport boundary and image-goal comparison without requiring goal or future proprioception.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
