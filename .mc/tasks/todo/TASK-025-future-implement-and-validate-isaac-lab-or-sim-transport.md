---
id: TASK-025
aliases:
- TASK-025
title: 'Future: implement and validate Isaac Lab or Sim transport'
slug: future-implement-and-validate-isaac-lab-or-sim-transport
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
- "[[TASK-022]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-28
---

# Future: implement and validate Isaac Lab or Sim transport

## Description

Execute this task only when suitable Isaac compute and assets are available. It is not a prerequisite for local MuJoCo MVP acceptance.

## Acceptance Criteria

- [ ] Pin a supported Isaac environment and corresponding G1/Dex3 assets, then implement the prepared transport interface.
- [ ] Run reset/observation/action/timing parity checks and a separately labeled cross-simulator benchmark.
- [ ] Reuse models, canonical data, CEM/MPC and evaluator interfaces without simulator-specific branches outside adapters.

## Notes

- Milestone: future_platform.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## 2026-09-20 full-task request audit

No supported Isaac compute/runtime is available on the current Mac; runtime parity and cross-simulator results require that external resource. Prepared contracts/runbooks remain available; acceptance is not complete. Continue all locally executable apple-MVP work independently.

## 2026-09-28 Isaac Sim bring-up spike (development, not gated)

The owner approved a development spike on the Linux PC (RTX 5080). It used the local `isaaclab_arena` image (Isaac Sim 6.0.0-rc.22, Isaac Lab 3.0.0, image `sha256:2588b526…`) and NVIDIA's G1 + Dex3 USD. The spike met all five of its criteria: headless GPU container; 43-joint articulation with the root fixed (0.0 m drift); named joint comparison against the MuJoCo model; one onboard-pose RGB frame; step time and VRAM. The 29 leg, waist and arm limits match. 10 of the 14 Dex3 limits are narrower in the USD, and the joint order differs from MJCF, so joints must be mapped by name. Physics-only stepping took a median 2.73 ms per 0.002 s step with one environment, and the Isaac process peaked at 4 666 MiB of GPU memory. Details, commands and caveats (unpinned staging-S3 asset, release-candidate simulator, no drive-model parity) are in `docs/ISAAC_BRINGUP_SPIKE.md`; scripts are in `scripts/isaac/`. This does not meet any acceptance criterion above: nothing is pinned by hash, no transport is implemented and no parity check has run. Status stays `backlog`.
