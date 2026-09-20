---
id: TASK-015
aliases:
- TASK-015
title: Integrate and train LeWM through the shared WorldModel adapter
slug: integrate-and-train-lewm-through-the-shared-worldmodel-adapter
status: done
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

- [x] Implement all WorldModel methods, action-conditioned multi-step prediction, compatible goal distance, and checkpoint validation.
- [x] Run finite-gradient training and save/load on CPU or validated MPS; record any operator fallback and compute cost.
- [x] No upstream dataset, planner, or simulator replaces shared common-mode components; no unavailable future state enters inference.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

The adapter imports pinned upstream LeWM JEPA/ARPredictor/SIGReg classes with a compact randomly initialized ViT. Actual CPU/MPS forward/backward, causal recursive prediction, goal distance and checkpoint tests pass. Shared data, planner and simulator remain unchanged; inference has no future robot state. See docs/MODELS.md and docs/LEWM_SPIKE.md.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-002]] [[TASK-010]] [[TASK-011]] [[TASK-006]] %%
