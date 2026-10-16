---
id: TASK-097
aliases:
- TASK-097
title: 'SO-101 S1: pinned SO-101 MJCF, tabletop scene with wrist and overview cameras, joint-control embodiment adapter and MuJoCo transport'
slug: so101-sim-embodiment
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- simulation
- embodiment
- foundation
sprint: ''
depends_on:
- "[[TASK-096]]"
due_date: ''
created: 2026-10-16
updated: 2026-10-16
---

# SO-101 S1: pinned SO-101 MJCF, tabletop scene with wrist and overview cameras, joint-control embodiment adapter and MuJoCo transport

## Description

Phase S1 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md) (DECISIONS 2026-10-16 (bp),
R28.5–R28.6). Upstream, checked through the GitHub API on 2026-10-16 and re-checked at fetch
time: MuJoCo Menagerie `robotstudio_so101/` (Apache-2.0; preferred), derived from TheRobotStudio
`SO-ARM100` `Simulation/SO101/so101_new_calib.xml` (Apache-2.0; fallback). Menagerie's
`trs_so_arm100` is the SO-100 and is not used.

Delivers `scripts/fetch_so101.py` (pinned revision, per-file SHA-256, into ignored `assets/`), a
tabletop scene (table, cube, target region, Menagerie's existing `wrist_cam` and a fixed
front/overview camera), `configs/so101_sim_action.json`, an `so101` embodiment adapter with IK-free
joint-position control and a MuJoCo transport in new modules, registered lazily as `"module:attr"`
strings. The G1 files pinned by frozen protocols (`simulation.py`, `embodiment.py`,
`constraints.py`, `configs/g1_sim_action.json`, see SO101_PLAN.md S0) stay byte-identical. Simulation only; real
execution stays disabled (R28.3). Software only; no experiment.

## Acceptance Criteria

- [x] The fetch is reproducible and hash-verified; the pinned revision, file hashes and licence
  are recorded in `docs/DEPENDENCIES.md` at that point.
- [x] The scene loads and steps under `MUJOCO_GL=egl`; both cameras render, and their frames are
  inspected (the cube is visible in the wrist view at grasp distance).
- [x] Action normalize/denormalize round-trip: arm channels exact, gripper within 2e-7 (its
  map is affine); joint-limit and rate guards are tested
  and recorded, never applied silently to stored actions; a zero action holds pose; a fixed
  seed is deterministic.
- [x] Every G1 hash-pinned runtime check still passes at HEAD (each `frozen_files_check`).
- [x] The core-import check, ruff and pytest stay green; graphics tests are behind
  `JEPA_TEST_RENDER=1`.
- [x] PR reviewed independently and merged.

## Notes

- 2026-10-16, created by R28.6 (branch `docs/so101-plan`).
- 2026-10-16, done in #233: `scripts/fetch_so101.py` and `assets/so101_manifest.json` (Menagerie
  `robotstudio_so101` at `3abca2e`, 26 files hash-verified), `so101_simulation.py`, `so101.py`
  (`SO101Embodiment`), `configs/so101_sim_action.json`, the development-only `so101_scripted.py`
  and `scripts/dev_so101_reach.py`, and `docs/SO101_SIMULATION.md`; `config.py` additive (an
  SO-101 config is refused at load until S2 registers tasks). Deviations, as reviewed: the
  checkout goes to ignored `third_party/` (the G1 convention) rather than `assets/`, with the
  manifest under `assets/`; a zero *arm* action holds pose, while a zero gripper command means
  half-open, so `hold_action()` holds the gripper. Every G1 hash-pinned file is byte-identical.
  Software only, simulation only; the scripted reach (7 of 8 development seeds lifted) is not
  gated and makes no claim.
