---
id: TASK-003
aliases:
- TASK-003
title: Validate MuJoCo rendering and G1 plus dual Dex3 assets on Mac
slug: validate-mujoco-rendering-and-g1-plus-dual-dex3-assets-on-mac
status: backlog
priority: 1
owner: ''
projects: []
customers: []
tags:
- m0
- mvp
sprint: ''
depends_on:
- "[[TASK-001]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Validate MuJoCo rendering and G1 plus dual Dex3 assets on Mac

## Description

Use native MuJoCo Python bindings locally. Audit candidate Unitree MJCF/meshes; do not assume the upstream DDS/Linux launcher works on macOS. Compose or convert authorized Dex3 assets if necessary.

## Acceptance Criteria

- [ ] Load the verified robot asset, step physics, reset deterministically, and render RGB on the local Mac; save commands and environment versions.
- [ ] Verify both hands, joint/actuator order, collision geometry, inertias, camera, asset terms, and EDU4 differences; estimate any missing model work.
- [ ] Measure step/render throughput and demonstrate the macOS viewer path using mjpython where required; no SDK2 or Isaac dependency.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
