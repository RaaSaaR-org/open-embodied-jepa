---
id: TASK-010
aliases:
- TASK-010
title: Collect and validate the first shared MuJoCo dataset
slug: collect-and-validate-the-first-shared-mujoco-dataset
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
- "[[TASK-007]]"
- "[[TASK-009]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Collect and validate the first shared MuJoCo dataset

## Description

Collect a local pilot using scripted behavior or teleoperation with varied successful/failed trajectories. Follow docs/DATA_PLAN.md and exclude reserved apple-to-plate combinations.

## Acceptance Criteria

- [ ] Publish data/source/license/action manifests and episode-grouped train/validation/test assignments with hashes.
- [ ] Report action and state coverage, failures, frame synchronization, storage, and sample-quality exclusions.
- [ ] The same canonical batch feeds both model adapters; record executed rather than merely requested actions and preserve raw data.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
