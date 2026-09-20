---
id: TASK-005
aliases:
- TASK-005
title: Create installable Mac development environment and core CI
slug: create-installable-mac-development-environment-and-core-ci
status: review
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

- [x] A clean macOS arm64 environment installs and imports the core; dependency versions and hashes are locked.
- [x] MuJoCo, external models, Isaac, and SDK2 remain isolated extras or environments with lazy imports; no mandatory CUDA.
- [x] Formatting, meaningful CPU contract checks, small fixture handling, and license inventory run in CI; document CPU/MPS validation commands.

## Notes

- Milestone: m1.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence — 2026-09-20

Installable Python3.12 package and hashed `uv.lock`; optional learning/sim/lewm extras, lazy named registry, core import isolation. Clean `/tmp/open-embodied-jepa-core-check` environment installed via `UV_PROJECT_ENVIRONMENT=... uv sync --locked`, imports without torch/mujoco and full core tests passed. GitHub Actions added for macOS/Linux lint, format, tests and dependency metadata inventory; remote run pending PR. See `docs/SETUP.md`. MuJoCo agent independently reviewed coordinator foundation changes and reported no blockers; optional registry validation hardening applied/tested.
%% mc-links: [[TASK-002]] [[TASK-003]] [[TASK-004]] %%
