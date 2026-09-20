---
id: TASK-004
aliases:
- TASK-004
title: Freeze WorldModel, dataset, action, and transport contracts v0
slug: freeze-worldmodel-dataset-action-and-transport-contracts-v0
status: done
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

- [x] Fix RGB/state/sequence axes, rollout and cost shapes, timestamps, validity masks, and backend capability negotiation.
- [x] Specify EE frame/rotation composition, 14-component candidate action ordering, hand synergy semantics, and separate joint-space schema.
- [x] Specify MuJoCo / future Isaac / SDK2 transport boundary and image-goal comparison without requiring goal or future proprioception.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## Implementation evidence — 2026-09-20

Executable NumPy-only schemas and Protocols in `src/embodied_jepa/contracts.py`; v0 frames, action order, data axes, validity masks, synchronization, opaque rollout/costs and transport semantics frozen in `docs/ARCHITECTURE.md`. Contract tests exercise malformed input, episode boundaries, masks, snapshots and capability negotiation. LeWM agent independently reviewed contract implementation/tests; no blockers.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/2. Local final foundation suite: 76 passed. GitHub macOS and Linux core jobs passed on `f733be5`. Independent agent cross-review plus author integration review found no unresolved blocker. Deliverable acceptance is satisfied; merge pending at this task-note revision.
