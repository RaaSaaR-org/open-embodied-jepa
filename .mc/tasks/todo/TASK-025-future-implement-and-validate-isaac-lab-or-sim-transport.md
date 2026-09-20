---
id: TASK-025
aliases:
- TASK-025
title: 'Future: implement and validate Isaac Lab or Sim transport'
slug: future-implement-and-validate-isaac-lab-or-sim-transport
status: backlog
priority: 4
owner: ''
projects: []
customers: []
tags:
- future_platform
- optional
sprint: ''
depends_on:
- "[[TASK-022]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Future: implement and validate Isaac Lab or Sim transport

## Description

Execute this task only when suitable Isaac compute and assets are available. It is not a prerequisite for local MuJoCo MVP acceptance.

## Acceptance Criteria

- [ ] Pin a supported Isaac environment and corresponding G1/Dex3 assets, then implement the prepared transport interface.
- [ ] Run reset/observation/action/timing parity checks and a separately labeled cross-simulator benchmark.
- [ ] Reuse models, canonical data, CEM/MPC and evaluator interfaces without simulator-specific branches outside adapters.

## Notes

- Milestone: future_platform.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## 2026-09-20 full-task request audit

No supported Isaac compute/runtime is available on the current Mac; runtime parity and cross-simulator results require that external resource. Prepared contracts/runbooks remain available; acceptance is not complete. Continue all locally executable apple-MVP work independently.
