---
id: TASK-020
aliases:
- TASK-020
title: Run both models on the full common-mode MuJoCo benchmark
slug: run-both-models-on-the-full-common-mode-mujoco-benchmark
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m4
- mvp
sprint: ''
depends_on:
- "[[TASK-017]]"
- "[[TASK-018]]"
- "[[TASK-019]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Run both models on the full common-mode MuJoCo benchmark

## Description

Run final Apple-to-Plate and stage evaluations locally under a shared measured Mac budget; retrain on the final shared corpus when expanded.

## Acceptance Criteria

- [ ] Archive both checkpoints and matched common-mode configurations; run all planned seeds/episodes or explicitly document incomplete work.
- [ ] Report full-task/stage success, confidence intervals, latency, memory, replans, controls, and zero-shot outcomes without omitting failures.
- [ ] Produce a reproducible comparison report and rollout references; clearly separate engineering completeness from research success.

## Notes

- Milestone: m4.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
