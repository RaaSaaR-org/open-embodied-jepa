---
id: TASK-021
aliases:
- TASK-021
title: Prepare SDK2 transport mapping with mock command validation
slug: prepare-sdk2-transport-mapping-with-mock-command-validation
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
- "[[TASK-008]]"
- "[[TASK-009]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Prepare SDK2 transport mapping with mock command validation

## Description

Implement or specify an isolated SDK2 transport seam and versioned G1/Dex3 message mappings. Validate conversion with local mocks; runtime SDK2/hardware integration remains later work.

## Acceptance Criteria

- [x] Record joint/channel mappings, camera/state timestamps, hardware configuration fields, and transport capability requirements.
- [x] Mock tests cover arm/hand command conversion, limits, disconnect/stale input, and explicit enable/stop behavior.
- [x] Mac simulation imports and runs without SDK2 installed; no hardware execution is claimed from mock results.

## Notes

- Milestone: m5.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Deliverable review evidence

Mock SDK2 mappings, pinned source references, channel semantics and mandatory calibration fields are documented in docs/HARDWARE.md. Full lowcmd mapping has 29 active and 6 disabled slots. Tests cover conversion, limits, stale/disconnected state, enable/stop and physical-execution refusal. Real networking, CRC, controller ownership and actuation remain outside the local MVP.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-008]] [[TASK-009]] %%
