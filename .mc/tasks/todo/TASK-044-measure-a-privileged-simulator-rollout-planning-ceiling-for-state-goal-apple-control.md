---
id: TASK-044
aliases:
- TASK-044
title: Measure a privileged simulator-rollout planning ceiling for state-goal apple control
slug: measure-a-privileged-simulator-rollout-planning-ceiling-for-state-goal-apple-control
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- control
- diagnostics
sprint: ''
depends_on:
- "[[TASK-043]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---




# Measure a privileged simulator-rollout planning ceiling for state-goal apple control

## Question
TASK-043's learned state-goal MPC stalled on goal 0 on 4/4 development resets, and its predicted costs (median 0.044-0.069) were far below the measured distances (0.126-0.228). With perfect dynamics, can the SAME state-goal scaffold and CEM planner (same retrieved TRAIN goals, tolerances, 64-command stall bound, H16, 16 candidates, 2 iterations, commitment 1, max 1000 commands, resets 43000-43003) progress through the scorer's grasp stage? This separates model error from goal/planner design. The mode is a NON-LEARNED privileged diagnostic ceiling and is never a learned result.

## Acceptance Criteria
- [ ] Preregister `docs/experiments/apple_privileged_ceiling_v1.md` (label, privileged-truth justification, candidate scoring, budgets, gate, interpretation, frozen command) before any development attempt.
- [ ] Implement the ceiling in isolation (`src/embodied_jepa/privileged_rollout.py`, `evaluate_apple.py --modes privileged_rollout`): exact MuJoCo rollouts in a non-rendering twin, same state-goal metric; no change to sensor.py, base.py, the planner core or defaults, scoring, guards or learned-mode behaviour; not registered as a model.
- [ ] Tests: rollout/live parity, live simulator untouched, cost equals the metric, rejection ranking, acknowledgement and registry isolation, plan/report isolation from learned gates and the final stage; ruff and full pytest pass.
- [ ] Independent pre-run review; fix blockers and record revisions before running.
- [ ] Run the frozen command exactly once from a clean checkout into a new output directory; record every outcome, the ceiling gate and readings in a results doc and manifest.
- [ ] Deliver through PR; TASK-033/034 physical acceptance stays open (this is not a learned result).

## Scope and resources
CPU, H16/K16/2 rounds/commitment 1, max 1000 commands, 5 s per-command deadline, 840 s per attempt, 3600 s global, 4 attempts. Final cohort 44000-44019 and TEST untouched.
%% mc-links: [[TASK-043]] %%
