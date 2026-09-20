---
id: TASK-021
aliases:
- TASK-021
title: Prepare SDK2 transport mapping with mock command validation
slug: prepare-sdk2-transport-mapping-with-mock-command-validation
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

- [ ] Record joint/channel mappings, camera/state timestamps, hardware configuration fields, and transport capability requirements.
- [ ] Mock tests cover arm/hand command conversion, limits, disconnect/stale input, and explicit enable/stop behavior.
- [ ] Mac simulation imports and runs without SDK2 installed; no hardware execution is claimed from mock results.

## Notes

- Milestone: m5.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
