---
id: TASK-022
aliases:
- TASK-022
title: Prepare Isaac port and real-hardware commissioning runbooks
slug: prepare-isaac-port-and-real-hardware-commissioning-runbooks
status: done
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

- [x] Document future reset/observe/execute/close transport APIs, MJCF-to-Isaac asset correspondence, cameras, action scales, and parity tests.
- [x] Provide configuration templates and a staged hardware runbook: replay, calibration, bounded single-joint/hand checks, then supervised task execution.
- [x] Separate checks possible on the Mac from future machine/robot checks; link TASK-025 and TASK-026 for actual execution.

## Notes

- Milestone: m5.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

docs/ISAAC_PORT.md and docs/HARDWARE.md define future adapter APIs, parity checks, asset/camera/action correspondence, missing-value calibration templates and staged commissioning. Mac simulation/mock checks are clearly separated from future TASK-025 Isaac execution and TASK-026 physical commissioning.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-004]] [[TASK-009]] [[TASK-021]] %%
