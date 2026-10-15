---
id: TASK-096
aliases:
- TASK-096
title: 'SO-101 S0: take the action dimension from the action schema and add joint_delta_so101_v0'
slug: so101-action-schema-generalisation
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- contracts
- foundation
sprint: ''
depends_on: []
due_date: ''
created: 2026-10-16
updated: 2026-10-16
---

# SO-101 S0: take the action dimension from the action schema and add joint_delta_so101_v0

## Description

Phase S0 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md) (DECISIONS 2026-10-16 (bp),
R28.1–R28.9). The core assumes the 14-D `ee_delta_grasp_v0` through the module constant
`ACTION_DIM` and literal 14s. This task makes the action dimension come from the action schema
and registers `joint_delta_so101_v0`: five arm joint-position deltas plus one gripper command, in
LeRobot `so_follower`'s order `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`.

Files in scope: `contracts.py`, `config.py` (lines 201–206 reject every schema but
`ee_delta_grasp_v0`), `planning.py`, `data.py`, `policy.py`, `models/native.py`, `models/lewm.py`
(`input_dim=14`), `models/frozen_encoder.py`, `models/latent_critic.py`. G1 task-line modules and
every hash-pinned runner stay unedited. Software only; no experiment.

## Acceptance Criteria

- [ ] `ee_delta_grasp_v0` behaviour is unchanged: every existing test passes without edits to its
  expectations, and a G1 config resolves to the same 14-D path.
- [ ] `joint_delta_so101_v0` is registered with fixed names, order and dimension 6; config
  validation, the data round-trip (including the `action_valid=False` final row), CEM and both
  backends' training step and prediction are tested with it on synthetic data.
- [ ] No G1 model implementation hash changes, or a reviewed compatibility manifest is committed
  (R28.4, as TASK-072 did).
- [ ] The core-import check passes (no `torch`/`mujoco` after `import embodied_jepa`); ruff check,
  ruff format --check and pytest are green.
- [ ] PR reviewed independently and merged.

## Notes

- 2026-10-16, created by R28.6 (branch `docs/so101-plan`).
