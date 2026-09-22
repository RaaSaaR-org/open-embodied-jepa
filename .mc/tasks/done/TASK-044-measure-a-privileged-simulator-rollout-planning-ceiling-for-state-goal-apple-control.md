---
id: TASK-044
aliases:
- TASK-044
title: Measure a privileged simulator-rollout planning ceiling for state-goal apple control
slug: measure-a-privileged-simulator-rollout-planning-ceiling-for-state-goal-apple-control
status: done
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
- [x] Preregister `docs/experiments/apple_privileged_ceiling_v1.md` (label, privileged-truth justification, candidate scoring, budgets, gate, interpretation, frozen command) before any development attempt.
- [x] Implement the ceiling in isolation (`src/embodied_jepa/privileged_rollout.py`, `evaluate_apple.py --modes privileged_rollout`): exact MuJoCo rollouts in a non-rendering twin, same state-goal metric; no change to sensor.py, base.py, the planner core or defaults, scoring, guards or learned-mode behaviour; not registered as a model.
- [x] Tests: rollout/live parity, live simulator untouched, cost equals the metric, rejection ranking, acknowledgement and registry isolation, plan/report isolation from learned gates and the final stage; ruff and full pytest pass.
- [x] Independent pre-run review; fix blockers and record revisions before running.
- [x] Run the frozen command exactly once from a clean checkout into a new output directory; record every outcome, the ceiling gate and readings in a results doc and manifest.
- [x] Deliver through PR; TASK-033/034 physical acceptance stays open (this is not a learned result).

## Scope and resources
CPU, H16/K16/2 rounds/commitment 1, max 1000 commands, 5 s per-command deadline, 840 s per attempt, 3600 s global, 4 attempts. Final cohort 44000-44019 and TEST untouched.
## Phase 1 record (implementation, no development attempt)
Branch `feat/task-044-privileged-ceiling` from main `cb81c22`. `src/embodied_jepa/privileged_rollout.py` (not registered in MODELS; explicit acknowledgement) restores a `mj_copyData` snapshot of the live simulator into a non-rendering twin per candidate, executes each step through the twin's `project_candidates` + unchanged `execute`, and scores with the frozen state-goal `observed_distance`. `evaluate_apple.py --modes privileged_rollout` runs alone, development only, writes `ceiling_gate` (never `state_gate`). Learned paths unchanged.
- TRAIN-reset 42000 smoke (scratch, hashes stubbed): first build without per-step re-projection had 279 joint-rate rejections in 40 commands; fixed to re-project per step (fidelity fix, disclosed). After fix: 0.96 s/command median, parity exact 63/63, goal 0 not reached in 64 commands (min 0.0093 vs tol 0.0052). No scaffold parameter changed.
- Pre-run review (fresh subagent, 7a5ceb5): no blockers; R1 applied (readings on counted attempts only with inconclusive fallback; close-goal reading `>`; parity counted per attempt and required by the gate). Checks: ruff clean, full pytest 596 passed / 13 skipped (graphics opt-in, timm).

## Phase 2 record (single frozen run)
Executed once from clean checkout `733bfdca972e001a83835a8f4b193807f980e4c6`, exact frozen command, new output `outputs/apple-privileged-ceiling-v1/`. Exit 0, report `completed`, provenance valid, 4/4 attempts counted, 371.964 s global, 62.6-119.0 s per attempt of 840 s.
- Outcome: **ceiling gate failed** (0/4 grasp, 0 stages). goal_stall on goal 0 (43000, 43001) and goal 1 (43002, 43003). Rollouts exact (0 mismatches / 364 checks), 0 rejected candidates. Conclusive reading `state_goal_tracking_inadequate` = true: planner/goal design must change before further model training.
- Diagnostic: exact H16 endpoints were inside tolerance in 20-81% of searches on the stalled goals. Measured closest approach was 1.0-4.2x tolerance, with wide oscillation (median 4.1-35x). This is consistent with, but does not prove, receding-horizon endpoint chasing under commitment 1.
- Post-run verification (fresh subagent) recomputed all numbers from raw outputs and re-hashed the inputs/artifacts. Corrections applied: closest distances now include the termination observation (43000 0.0183, 43001 0.0243); "within 0.0005"; oscillation wording; planning-only medians; softened endpoint-chasing claim.
- Evidence: `docs/experiments/apple_privileged_ceiling_results_v1.md`, `benchmarks/manifests/apple-privileged-ceiling-v1.json`. Not a learned result; TASK-033/034 stay open; final cohort and TEST untouched.
- Recommended next (not started): redesign the scaffold (time-indexed running cost over the horizon, or commitment/shrinking horizon) and gate it with the same privileged ceiling before pairing with learned dynamics.

- Delivery: PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/14. Execute/report/deliver acceptance is met; the ceiling gate is explicitly failed, and TASK-033/034 physical acceptance stays open.

%% mc-links: [[TASK-043]] %%
