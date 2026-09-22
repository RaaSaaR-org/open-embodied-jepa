---
id: TASK-053
aliases:
- TASK-053
title: Remove the import-order dependence in the privileged rollout twin
slug: remove-the-import-order-dependence-in-the-privileged-rollout-twin
status: review
priority: 3
owner: ''
projects: []
customers: []
tags:
- hygiene
- testing
sprint: ''
depends_on:
- "[[TASK-051]]"
due_date: ''
created: 2026-09-23
updated: 2026-09-23
---

# Remove the import-order dependence in the privileged rollout twin

## Description
Follow-up recorded in TASK-051: `tests/test_apple_evaluation.py::test_demo_replay_is_open_loop_non_learned_and_exhausts`
failed deterministically when that module was run on its own and passed in the full suite.

`src/embodied_jepa/privileged_rollout.py` created the rollout twin with a module-level
`class _BlindSimulation(MuJoCoSimulation)` statement, over a name bound at import time. The
evaluator reaches that module through a lazy `from embodied_jepa.object_ceiling import
GUARD_REFUSALS` inside `project_open_loop`, and the test fixture has replaced
`embodied_jepa.simulation.MuJoCoSimulation` with a non-class factory by then. Whichever module
imported first therefore decided the base class: alone, the class statement ran under the
replacement and raised `TypeError: function() argument 'code' must be code, not str`, which the
attempt worker recorded as `runtime_error`; in the full suite an earlier test had already
imported the module, so the cached class kept the real base.

Hygiene only. No experimental behaviour, result, manifest or frozen value changes: in production
nothing replaces that attribute, so the twin is the same class built from the same base.

## Acceptance Criteria
- [x] The twin subclass is built at construction time from the `embodied_jepa.simulation` module
      attribute, so importing `privileged_rollout` (and `object_ceiling*`) never depends on
      import order.
- [x] `uv run --no-sync pytest tests/test_apple_evaluation.py` passes on its own and the full
      suite still passes.
- [x] A regression test fails on the pre-fix source and passes after it.
- [x] Every other test module passes on its own, checked in CI.
- [x] ruff check, ruff format --check, full pytest, `mc validate`.

## Notes
- Branch `fix/task-053-rollout-twin-import-order` from main `51c87ee` (TASK-051 closure).
%% mc-links: [[TASK-051]] %%
