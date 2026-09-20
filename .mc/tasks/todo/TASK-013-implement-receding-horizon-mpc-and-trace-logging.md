---
id: TASK-013
aliases:
- TASK-013
title: Implement receding-horizon MPC and trace logging
slug: implement-receding-horizon-mpc-and-trace-logging
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
- "[[TASK-009]]"
- "[[TASK-012]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Implement receding-horizon MPC and trace logging

## Description

Connect observations, goal encoding, CEM, first-action execution, and replanning with task termination and trace logging.

## Acceptance Criteria

- [ ] Run a deterministic simulated closed-loop scenario through the declared interfaces and log requested/applied actions.
- [ ] Observation staleness, failed planning, deadline misses, solver rejection, and stop/timeout paths produce explicit outcomes.
- [ ] One action is executed before re-observation by default; model latents and simulator state do not leak into unrelated layers.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
