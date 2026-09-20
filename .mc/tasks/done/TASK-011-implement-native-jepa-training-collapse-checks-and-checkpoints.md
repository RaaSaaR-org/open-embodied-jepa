---
id: TASK-011
aliases:
- TASK-011
title: Implement native JEPA training, collapse checks, and checkpoints
slug: implement-native-jepa-training-collapse-checks-and-checkpoints
status: done
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

- [x] All WorldModel methods work and recursive multi-step prediction uses no ground-truth future state.
- [x] A small CPU/MPS fixture trains with finite gradients; latent variance and action-shuffle controls detect collapsed/action-ignoring behavior.
- [x] Save/load resumes weights, optimizer, target network, normalization, schema, and RNG state with matching evaluation outputs.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Native JEPA implements recursive action-conditioned prediction, EMA targets, train-only distance statistics and strict checkpoint provenance. CPU/MPS gradients, causal prediction, collapse diagnostics, target-space persistence and exact CPU resume pass tests. The collapsed v0 and invalid mixed-space v1 pilots are preserved. Current final training outcomes remain separate research evidence.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-006]] [[TASK-007]] %%
