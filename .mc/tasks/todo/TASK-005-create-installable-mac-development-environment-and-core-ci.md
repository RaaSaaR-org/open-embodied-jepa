---
id: TASK-005
aliases:
- TASK-005
title: Create installable Mac development environment and core CI
slug: create-installable-mac-development-environment-and-core-ci
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m1
- mvp
sprint: ''
depends_on:
- "[[TASK-002]]"
- "[[TASK-003]]"
- "[[TASK-004]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Create installable Mac development environment and core CI

## Description

Create packaging, supported Python selection, reproducible locks, dependency notices, and CPU-first CI. The original-code Apache-2.0 license was added during repository bootstrap. Use evidence from the Mac feasibility spikes.

## Acceptance Criteria

- [ ] A clean macOS arm64 environment installs and imports the core; dependency versions and hashes are locked.
- [ ] MuJoCo, external models, Isaac, and SDK2 remain isolated extras or environments with lazy imports; no mandatory CUDA.
- [ ] Formatting, meaningful CPU contract checks, small fixture handling, and license inventory run in CI; document CPU/MPS validation commands.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
