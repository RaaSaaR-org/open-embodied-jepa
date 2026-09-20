---
id: TASK-026
aliases:
- TASK-026
title: 'Future: commission SDK2 on the physical G1 and Dex3'
slug: future-commission-sdk2-on-the-physical-g1-and-dex3
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
- "[[TASK-021]]"
- "[[TASK-022]]"
- "[[TASK-023]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Future: commission SDK2 on the physical G1 and Dex3

## Description

Execute the prepared commissioning runbook with actual hardware access and calibrated limits. It is not a prerequisite for the local MVP.

## Acceptance Criteria

- [ ] Verify exact robot/firmware/joint/camera/hand configuration and live timestamped observations.
- [ ] Complete staged bounded motion, stop, disconnect, IK, and grasp checks before supervised task execution.
- [ ] Archive real-hardware outcomes separately from simulator results and document transfer failures and required calibration.

## Notes

- Milestone: future_platform.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## 2026-09-20 full-task request audit

No physical G1/Dex3 is available. Commissioning requires actual hardware, calibration and supervised physical access. Prepared contracts/runbooks remain available; acceptance is not complete. Continue all locally executable apple-MVP work independently.
