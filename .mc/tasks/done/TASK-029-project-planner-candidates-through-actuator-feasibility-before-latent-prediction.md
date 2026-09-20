---
id: TASK-029
aliases:
- TASK-029
title: Project planner candidates through actuator feasibility before latent prediction
slug: project-planner-candidates-through-actuator-feasibility-before-latent-prediction
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- research-followup
- planner
sprint: ''
depends_on:
- "[[TASK-020]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---





# Project planner candidates through actuator feasibility before latent prediction

## Description
The frozen MVP comparison scores requested absolute grasp targets while the embodiment can rate-limit them. A measured open-hand request +1 executes approximately −0.8181818 in its first step. Add a model-independent feasibility path so the world model scores the commands expected to be accepted, using current robot state and manifest limits. The final model traces also identify right-arm joint-rate rejection in all 300 learned-policy attempts, so include pose/IK feasibility as well as grasp clipping. This is a separate research follow-up; preserve the original frozen comparison unchanged.

## Acceptance Criteria
- [x] Specify a shared candidate projection/constraint API with horizon-consistent grasp rate bounds and no privileged object truth or unavailable future state.
- [x] Verify predicted candidate actions agree with accepted targets in bounded deterministic robot tests, retain requested/applied traces, and run both model backends unchanged through the same projection.
- [x] Preregister a new matched comparison with fixed checkpoints/data/cohort/budget before execution; retain all original results and report whether guard stops and physical success improve.

## Notes

- Found by independent model/planner review before the final MVP evaluation; see `docs/reviews/runtime-review.md`.
- Source, dataset and model checkpoints remain immutable during the current six-run experiment. No post-result tuning or silent checkpoint-hash migration. Treat the inspected v0 cohort as development evidence in follow-up work; do not relabel reused outcomes as a newly sealed test.
### Implementation scope

- Isolated branch `fix/task-029-planner-feasibility`; user `CLAUDE.md` left untouched.
- Add opt-in shared candidate projection, robot-only snapshot/kinematics, common
  target-preparation checks, horizon-consistent grasp targets, feasibility masks,
  and sampled/projected/requested/applied trace separation. Models/checkpoints,
  action schema, physical limits, original frozen experiment artifacts unchanged.
- Independent contract review and independent behavior-test work delegated.
- New paired development comparison will use fixed seed-1 checkpoints, reused
  environment seeds 20000–20004, projection off/on, both backends, 100-step cap,
  and 600-second total budget; preregister before execution. This is a bounded
  control diagnostic, not a fresh sealed generalization or full-horizon result.

### Pre-experiment validation

- Full optional suite: `PYTHONPATH=src JEPA_TEST_RENDER=1
  LEROBOT_SOURCE=third_party/lerobot .venv/bin/python -m pytest -q` → **351 passed**.
- Ruff lint/format and `git diff --check` passed.
- Independent review found and verified fixes for legacy schema-v1 compatibility
  and stale MuJoCo derived kinematics; all 400 original episodes still validate.
- Independent randomized physics check: 40/40 projected first commands accepted
  exactly. See `docs/reviews/feasibility-review.md` and behavior regressions.

### Delivered evidence

- PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/4 (merge pending at
  this task update; deliverable-based acceptance complete).
- Preregistered code/protocol revision `680be1b8d3a422d3d90a566e2f40ebf93f405859`;
  models/checkpoints unchanged, exact source/input hashes in the versioned manifest.
- Executed fixed 600-second comparison: **15 completed, one interrupted, four
  unstarted**. Exit 2 records incompleteness; no retries/extensions or relabeling.
- Seven complete pairs: joint-rate stops **7/7 off → 0/7 on**. Native commands
  11.75→100 mean across four pairs; LeWM 7.33→88.67 across three pairs.
- **666/666** accepted projected commands exactly match scored actions. Native
  reach 1/4 on; no grasp/place/full success. Two LeWM deadline stops remain.
- Report: `docs/experiments/feasibility_results_v1.md`; compact evidence:
  `benchmarks/manifests/feasibility-results-v1.json`; raw logs/video:
  `outputs/feasibility-v1/`. Independent audit recomputed all completed results.
- Completion means API/behavior checks and the preregistered bounded diagnostic
  were delivered, not that all planned episodes completed or learned manipulation
  succeeded. TASK-030 owns latency optimization and a new complete comparison.
%% mc-links: [[TASK-020]] %%
