---
id: TASK-011
aliases:
- TASK-011
title: Implement native JEPA training, collapse checks, and checkpoints
slug: implement-native-jepa-training-collapse-checks-and-checkpoints
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
- "[[TASK-006]]"
- "[[TASK-007]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Implement native JEPA training, collapse checks, and checkpoints

## Description

Implement a compact original RGB/state/action-conditioned latent model. Start with EMA targets and explicit anti-collapse regularization as a candidate; no pixel reconstruction requirement.

## Acceptance Criteria

- [ ] All WorldModel methods work and recursive multi-step prediction uses no ground-truth future state.
- [ ] A small CPU/MPS fixture trains with finite gradients; latent variance and action-shuffle controls detect collapsed/action-ignoring behavior.
- [ ] Save/load resumes weights, optimizer, target network, normalization, schema, and RNG state with matching evaluation outputs.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
