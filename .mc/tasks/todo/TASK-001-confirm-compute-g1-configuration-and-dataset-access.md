---
id: TASK-001
aliases:
- TASK-001
title: Confirm compute, G1 configuration, and dataset access
slug: confirm-compute-g1-configuration-and-dataset-access
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- m0
- discovery
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---



# Confirm compute, G1 configuration, and dataset access

## Description

Record the confirmed Mac-only / MuJoCo-first constraint and inventory remaining local resources. Observed: arm64 macOS 26.5.1, 48 GiB memory, uv available. No robot or dataset access is assumed.

## Acceptance Criteria

- [x] Record exact chip, free disk, supported Python candidate, and tested CPU/MPS availability.
- [x] Inventory existing recordings, provenance, and intended G1 EDU4 / dual Dex3 configuration; explicitly mark missing resources.
- [x] Choose local data/output locations and a measured memory/storage budget; do not add a remote GPU prerequisite.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## Implementation evidence — 2026-09-20

See `docs/RESOURCES.md` and `scripts/resource_probe.py`. Python 3.12.13 / Torch 2.14.0 CPU and MPS forward/backward checks passed on Apple M5 Pro, 48 GiB, 536.6 GiB free. No recordings or hardware access supplied; simulation collection is selected. Local artifact paths and bounded resource budgets documented. Report: `outputs/feasibility/resources.json` (local, ignored).
