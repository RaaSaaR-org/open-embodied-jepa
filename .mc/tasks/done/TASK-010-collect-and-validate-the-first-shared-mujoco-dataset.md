---
id: TASK-010
aliases:
- TASK-010
title: Collect and validate the first shared MuJoCo dataset
slug: collect-and-validate-the-first-shared-mujoco-dataset
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

- [x] Publish data/source/license/action manifests and episode-grouped train/validation/test assignments with hashes.
- [x] Report action and state coverage, failures, frame synchronization, storage, and sample-quality exclusions.
- [x] The same canonical batch feeds both model adapters; record executed rather than merely requested actions and preserve raw data.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Versioned manifests publish source/license/action/split hashes for the 100-episode reach pilot and 184-episode assembled corpus. Reach coverage has 4,836 transitions; the expanded corpus has 42,127. Large payloads remain local by artifact policy. Canonical batches store accepted applied targets and raw physical targets, distinct from requested commands and measured realized motion.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-007]] [[TASK-009]] %%
