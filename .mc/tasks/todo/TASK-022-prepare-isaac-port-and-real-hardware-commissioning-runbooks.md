---
id: TASK-022
aliases:
- TASK-022
title: Prepare Isaac port and real-hardware commissioning runbooks
slug: prepare-isaac-port-and-real-hardware-commissioning-runbooks
status: backlog
priority: 3
owner: ''
projects: []
customers: []
tags:
- m5
- mvp
sprint: ''
depends_on:
- "[[TASK-004]]"
- "[[TASK-009]]"
- "[[TASK-021]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Prepare Isaac port and real-hardware commissioning runbooks

## Description

Prepare future Isaac Lab/Sim and physical deployment without blocking the local MVP. Record exact seams, artifact parity requirements, and execution prerequisites.

## Acceptance Criteria

- [ ] Document future reset/observe/execute/close transport APIs, MJCF-to-Isaac asset correspondence, cameras, action scales, and parity tests.
- [ ] Provide configuration templates and a staged hardware runbook: replay, calibration, bounded single-joint/hand checks, then supervised task execution.
- [ ] Separate checks possible on the Mac from future machine/robot checks; link TASK-025 and TASK-026 for actual execution.

## Notes

- Milestone: m5.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
