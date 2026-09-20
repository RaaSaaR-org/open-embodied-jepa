---
id: TASK-002
aliases:
- TASK-002
title: Spike LeWM compatibility, action conditioning, and licenses on Mac
slug: spike-lewm-compatibility-action-conditioning-and-licenses-on-mac
status: review
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



# Spike LeWM compatibility, action conditioning, and licenses on Mac

## Description

Audit the official lucas-maes/le-wm source and the minimum dependencies needed for an adapter. Check code, weights, encoders, assets, and data separately. Verify a canonical G1-shaped batch rather than only an upstream demo.

## Acceptance Criteria

- [x] Pin source revision and record source/transitive/artifact license evidence in docs/DEPENDENCIES.md.
- [x] Run or document concrete blockers for forward/backward, arbitrary action dimension, history, and multi-step rollout on CPU and available MPS.
- [x] Produce an adapter mapping that does not change the shared dataset or CEM; identify any requirement for unavailable future robot state.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## Implementation evidence — 2026-09-20

Pinned upstream `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Full 18M-parameter architecture and small smoke profile run CPU/MPS forward/backward, 14D actions, recursive four-step rollout, image-goal costs and action-sensitivity checks. Both H1 and H3 tested; H1 supports shared current-observation contract. See `docs/LEWM_SPIKE.md`, source/license audit `docs/DEPENDENCIES.md`, and `scripts/spikes/lewm/probe.py`. These are synthetic compatibility results, not learned robot behavior.
