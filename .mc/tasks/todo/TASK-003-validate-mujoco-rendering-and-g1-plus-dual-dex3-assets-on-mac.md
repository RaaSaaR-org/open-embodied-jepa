---
id: TASK-003
aliases:
- TASK-003
title: Validate MuJoCo rendering and G1 plus dual Dex3 assets on Mac
slug: validate-mujoco-rendering-and-g1-plus-dual-dex3-assets-on-mac
status: review
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

- [x] Load the verified robot asset, step physics, reset deterministically, and render RGB on the local Mac; save commands and environment versions.
- [x] Verify both hands, joint/actuator order, collision geometry, inertias, camera, asset terms, and EDU4 differences; estimate any missing model work.
- [x] Measure step/render throughput and demonstrate the macOS viewer path using mjpython where required; no SDK2 or Isaac dependency.

## Notes

- Milestone: m0.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence — 2026-09-20

Pinned official dual-Dex3 Unitree asset `ffa21a1e811a4ffd5d31d0318674c950f73eb62c`, verified SHA-256 and BSD-3-Clause source terms. Native MuJoCo 3.13.0 exact reset/step traces, RGB and bounded mjpython viewer passed. Both hands: seven articulated/limited/actuated joints with mass/inertia/collision geometry. See `assets/manifest.json`, `docs/MUJOCO_SPIKE.md`, `scripts/fetch_assets.py`, `scripts/spikes/mujoco_probe.py`. EDU4 serial-specific calibration and grasp fidelity remain explicit future validation.
%% mc-links: [[TASK-001]] %%
