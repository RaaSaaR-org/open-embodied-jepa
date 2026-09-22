---
id: TASK-045
aliases:
- TASK-045
title: Redesign the state-goal scaffold as trajectory tracking and gate it on the privileged ceiling
slug: redesign-the-state-goal-scaffold-as-trajectory-tracking-and-gate-it-on-the-privi
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
- "[[TASK-044]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---



# Redesign the state-goal scaffold as trajectory tracking and gate it on the privileged ceiling

## Question
TASK-044 showed that the TASK-043 endpoint state-goal scaffold stalls at goals 0-1 on 4/4 development resets even with exact MuJoCo-rollout dynamics (suspected endpoint chasing). Does scoring the whole H16 rollout against the retrieved TRAIN demonstration's time-indexed right-arm+hand trajectory, advancing by measured progress, reach the scorer's grasp stage on >=2/4 of resets 43000-43003 with the same CEM budget and the same privileged exact-rollout forward model? If so, run one preregistered learned comparison (learned, dynamics_shuffle, persistence) on the same resets.

## Acceptance Criteria
- [x] Preregister `docs/experiments/apple_trajectory_tracking_v1.md` (design, TRAIN-only justification, budgets, gates, readings, conditional learned stage, frozen commands) before any development attempt.
- [x] Implement the tracking controller in isolation (`src/embodied_jepa/trajectory_tracking.py`, `evaluate_apple.py --goal-kind trajectory`) behind the generic model contract; no change to sensor.py, base.py, state_goal_sensor.py, privileged_rollout.py, waypoint_planning.py, scorer or guards; image/state plans and gates unchanged.
- [x] Tests: cost/progress/stall/termination semantics, CEM sampling parity, TRAIN-only references, plan/gate isolation, worker wiring; ruff and full pytest pass.
- [x] Independent pre-run review; fix blockers and record revisions before running.
- [x] Run the stage-1 frozen command exactly once from a clean checkout; run stage 2 once only if stage 1 passes; record every outcome in a results doc and manifest.
- [x] Independent post-run verification of numbers against raw outputs; deliver through PR. TASK-033/034 stay open unless a learned result meets their own acceptance.

## Scope and resources
CPU. Stage 1: 4 privileged attempts, 840 s/attempt (R1; was 600), 3600 s global. Stage 2 (conditional): 12 attempts, 200 s/attempt, 3600 s global. Final cohort 44000-44019 and TEST untouched.

## Phase 1 record (implementation, no development attempt)
Branch `feat/task-045-trajectory-tracking` from main `b2e0789`.
- TRAIN-only design analysis (16 nominal demonstrations): ~178 of 502 frames per demo are dead time (adjacent 14-field distance < 0.1x the median positive adjacent distance; longest run 72-74 frames, spans 144-210 and 332-405, full 86-D state stationary). Keyframing at 0.1x keeps 338-342 rows; first close row 141-142; recorded-grasp row 195-199. 0.01x/0.5x keep 366-371/309-314 rows.
- TRAIN-reset 42000 smoke (scratch, hashes stubbed, max_steps 64): retrieval apple-42000; state library byte-identical to TASK-044; 0.985 s/command median planning, parity 63/63 exact, 0 rejections; reference index 40 after 64 commands. No parameter changed.
- Pre-run review (fresh subagent, d85e52b): 1 blocker (600 s cap + timeout rule could erase a latched grasp) fixed by raising to 840 s; recommended fixes applied as revision R1 (final-row hold for trailing dead time, conclusive-only no_model_contribution, endpoint diagnostic label, unified reference hash, doc wording). No gate threshold, reset, mode or planner parameter changed.

## Phase 2 record (single frozen stage-1 run)
Executed once from clean tracked checkout `bae083815d61d0d9b38a2aaf02f834ab1925fca4` (only untracked CLAUDE.md), exact frozen stage-1 command, new output `outputs/apple-trajectory-tracking-ceiling-v1/`. Exit 0, report `completed`, provenance valid, 4/4 counted, 1730.7 s global, 303.5-497.9 s per attempt of 840 s.
- Outcome: **stage-1 ceiling gate failed** (0/4 grasp; 4 summed stages = reach on 4/4). Terminations: reference_stall 43000 (row 337/340), 43001 (row 145/339), 43003 (row 339/342); reference_complete 43002. Apple fell off the table without being lifted on 4/4 (within the close phase on 43001; after the demo grasp row on the others). Rollouts exact (0 mismatches / 1694 checks).
- Conclusive readings: arm_pose_tracking_insufficient_for_grasp = true (3/4 passed the demo-grasp row without grasp); trajectory_tracking_inadequate = false. Stage 2 (learned) NOT run per protocol.
- Diagnostics: pre-close tracking distance median 0.012-0.015; close-phase median 0.066-0.157; joint-velocity-guard rollout rejections on 3 resets from command 238-259; final-row hold was 4-5 commands (bae0838 text estimated ~9).
- Evidence: `docs/experiments/apple_trajectory_tracking_results_v1.md`, `benchmarks/manifests/apple-trajectory-tracking-v1.json`. Not a learned result; TASK-033/034 stay open; final cohort and TEST untouched.
- Post-run verification (fresh subagent) recomputed all numbers and re-hashed inputs/artifacts/source snapshot: no numeric discrepancies. Wording corrected: dropped timing (not all during closing), hand contact is instantaneous not latched, only 43002 completed the reference, distance rise precedes contact, guard rejections precede contact, mechanism claims labelled interpretation, TASK-044 comparison by scorer stages only.
- Recommended next (not started): privileged ceiling of hybrid phase control (tracking to the close row, then the demo's recorded closing actions open loop) or an object-aware close-phase cost.

- Delivery: PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/15. Acceptance met with the stage-1 gate explicitly failed and stage 2 not run; TASK-033/034 stay open.

## Resumed review on 2026-09-22
The user authorized continuation with subagents. Existing committed TASK-045 work
was recovered at d85e52b rather than duplicated. Coordinator owns Git, tracker and
execution; contracts agent owns independent scientific review; model agent owns
focused gate-reporting fixes/tests; simulation agent verifies inputs and outcomes.
All 623 optional/model/rendering tests passed in 11.89 seconds before review fixes;
ruff check/format (95 files), MC validation/index (45 tasks), and diff checks passed.
Independent verification matched 63 asset payloads and 566 encoded corpus files,
without decoding TEST. Neither development output directory existed and no run
was active. Protocol R1 narrows causal claims without changing numeric gates,
seeds, controller settings, budgets or commands. Final pre-run checks follow the
reporting fix before one stage-1 launch.

Final pre-run verification after reporting correction: 625 tests passed in 13.61 seconds; ruff check and format (95 files) passed. The two new regressions preserve numeric gate pass with missing attempts while requiring complete cohorts for descriptive conclusiveness. Historical gate functions are unchanged.
%% mc-links: [[TASK-044]] %%
