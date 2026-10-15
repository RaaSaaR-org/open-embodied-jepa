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

Code that assumes 14 dimensions, including but not limited to: `contracts.py` (`ACTION_NAMES`,
`ACTION_DIM`, the validators), `config.py` (lines 201–206 reject every schema but
`ee_delta_grasp_v0`), `planning.py`, `data.py` (lines 94 and 127), `policy.py`, `audit.py`,
`models/native.py`, `models/lewm.py` (`input_dim=14`), `models/frozen_encoder.py`,
`models/frozen_tokens.py` and `models/latent_critic.py`.

**Coupling with G1 hashes (R28.4).** Several of these files are byte-pinned. `models/base.py`,
`models/native.py`, `models/lewm.py`, `models/readout.py`, `readout_labels.py`,
`models/frozen_encoder.py`, `models/frozen_tokens.py` and `policy.py` feed model implementation
hashes, and `models/lewm.py` also feeds TASK-087's G hash (`grounded_wm_model.py`), which
`zero_shot_runtime.py` and `grasp_corpus_wm_runtime.py` check; a byte change invalidates every
native/LeWM checkpoint since TASK-072 (TASK-077/080/081/083 included) or the G checkpoints of the
paused grasp line. TASK-091–TASK-094's frozen modules pin `contracts.py`, `planning.py`,
`models/lewm.py`, `embodiment.py`, `simulation.py`, `constraints.py`, `devices.py`,
`run_tools.py` and `configs/g1_sim_action.json`, and `tests/test_grasp_*.py` assert those pins.

**Design (chosen; SO101_PLAN.md S0).** Those files stay byte-identical and `ee_delta_grasp_v0`
stays on the existing classes. The SO-101 path goes through new files: a schema module
registering `joint_delta_so101_v0`, schema-parametrised native and LeWM subclasses that override
only the action-input construction, and a schema-aware CEM wrapper; unpinned files (`config.py`,
`data.py`, `audit.py`) change additively only. Fallback, only if a pinned file cannot stay
byte-identical: a TASK-072-style compatibility manifest naming the invalidated checkpoint
families, the hash-checking runtimes and frozen protocols and the last compatible revision,
recorded by a ruling before merge; frozen modules' pins are never edited. G1 task-line modules
and every hash-pinned runner stay unedited. Software only; no experiment.

## Acceptance Criteria

- [ ] `ee_delta_grasp_v0` behaviour is unchanged: every existing test passes without edits to its
  expectations, and a G1 config resolves to the same 14-D path.
- [ ] `joint_delta_so101_v0` is registered with fixed names, order and dimension 6; config
  validation, the data round-trip (including the `action_valid=False` final row), CEM and both
  backends' training step and prediction are tested with it on synthetic data.
- [ ] Every G1 hash-pinned runtime check still passes at HEAD: each `frozen_files_check`, and the
  implementation hashes of the native, LeWM, frozen-encoder, frozen-token and policy models and
  of TASK-087's G are unchanged against `origin/main` (R28.4). Only under the fallback: a
  reviewed compatibility manifest and a ruling, as TASK-072 did.
- [ ] The core-import check passes (no `torch`/`mujoco` after `import embodied_jepa`); ruff check,
  ruff format --check and pytest are green.
- [ ] PR reviewed independently and merged.

## Notes

- 2026-10-16, created by R28.6 (branch `docs/so101-plan`).
