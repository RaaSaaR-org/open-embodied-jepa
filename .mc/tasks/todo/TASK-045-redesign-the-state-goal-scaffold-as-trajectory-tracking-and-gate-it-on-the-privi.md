---
id: TASK-045
aliases:
- TASK-045
title: Redesign the state-goal scaffold as trajectory tracking and gate it on the privileged ceiling
slug: redesign-the-state-goal-scaffold-as-trajectory-tracking-and-gate-it-on-the-privi
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
- "[[TASK-044]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---


# Redesign the state-goal scaffold as trajectory tracking and gate it on the privileged ceiling

## Question
TASK-044 showed that the TASK-043 endpoint state-goal scaffold stalls at goals 0-1 on 4/4 development resets even with exact MuJoCo-rollout dynamics (suspected endpoint chasing). Does scoring the whole H16 rollout against the retrieved TRAIN demonstration's time-indexed right-arm+hand trajectory, advancing by measured progress, reach the scorer's grasp stage on >=2/4 of resets 43000-43003 with the same CEM budget and the same privileged exact-rollout forward model? If so, run one preregistered learned comparison (learned, dynamics_shuffle, persistence) on the same resets.

## Acceptance Criteria
- [ ] Preregister `docs/experiments/apple_trajectory_tracking_v1.md` (design, TRAIN-only justification, budgets, gates, readings, conditional learned stage, frozen commands) before any development attempt.
- [ ] Implement the tracking controller in isolation (`src/embodied_jepa/trajectory_tracking.py`, `evaluate_apple.py --goal-kind trajectory`) behind the generic model contract; no change to sensor.py, base.py, state_goal_sensor.py, privileged_rollout.py, waypoint_planning.py, scorer or guards; image/state plans and gates unchanged.
- [ ] Tests: cost/progress/stall/termination semantics, CEM sampling parity, TRAIN-only references, plan/gate isolation, worker wiring; ruff and full pytest pass.
- [ ] Independent pre-run review; fix blockers and record revisions before running.
- [ ] Run the stage-1 frozen command exactly once from a clean checkout; run stage 2 once only if stage 1 passes; record every outcome in a results doc and manifest.
- [ ] Independent post-run verification of numbers against raw outputs; deliver through PR. TASK-033/034 stay open unless a learned result meets their own acceptance.

## Scope and resources
CPU. Stage 1: 4 privileged attempts, 600 s/attempt, 3600 s global. Stage 2 (conditional): 12 attempts, 200 s/attempt, 3600 s global. Final cohort 44000-44019 and TEST untouched.

## Phase 1 record (implementation, no development attempt)
Branch `feat/task-045-trajectory-tracking` from main `b2e0789`.
- TRAIN-only design analysis (16 nominal demonstrations): ~178 of 502 frames per demo are dead time (adjacent 14-field distance < 0.1x the median positive adjacent distance; longest run 72-74 frames, spans 144-210 and 332-405, full 86-D state stationary). Keyframing at 0.1x keeps 338-342 rows; first close row 141-142; recorded-grasp row 195-199. 0.01x/0.5x keep 366-371/309-314 rows.
- TRAIN-reset 42000 smoke (scratch, hashes stubbed, max_steps 64): retrieval apple-42000; state library byte-identical to TASK-044; 0.985 s/command median planning, parity 63/63 exact, 0 rejections; reference index 40 after 64 commands. No parameter changed.

%% mc-links: [[TASK-044]] %%
