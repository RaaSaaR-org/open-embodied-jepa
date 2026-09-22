---
id: TASK-043
aliases:
- TASK-043
title: Test demonstration-state-goal learned MPC for apple control
slug: test-demonstration-state-goal-learned-mpc-for-apple-control
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- control
sprint: ''
depends_on:
- "[[TASK-042]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---



# Test demonstration-state-goal learned MPC for apple control

## Question
No learned run has passed orient/descend. `SensorWorldModel.distance` scores only predicted 24x24 RGB and discards pose forecasts, which ranked siblings better offline (TASK-041: pose 89.8% vs pixel 73.4%; TASK-042 hybrid failed its gate at +4.30 pp). Can the frozen H16 sensor_wm reach a scored grasp when MPC plans on its predicted right-arm+hand qpos against TRAIN demonstration state goals spaced one horizon (16 frames) apart, with proprioceptive progress? This is a new hypothesis ("demonstration-state-goal MPC"), not a retune, and TASK-041/042 VAL numbers informed it.

## Acceptance Criteria
- [x] Preregister protocol `docs/experiments/apple_state_goal_control_v1.md` (hypothesis, disclosure, frozen inputs, 16-attempt design, budgets, primary/secondary gates, falsification readings, frozen command).
- [x] Implement frozen `state_goal_sensor_wm_v1` (TRAIN-normalized predicted right-arm+hand qpos MSE; visual diagnostic weight 0), optional `StateWaypoint`/proprio progress/goal-stall bound in `waypoint_planning.py` with default image-path parity, and `evaluate_apple.py --goal-kind state` with initial-RGB TRAIN retrieval, stride16 goals, TRAIN-only tolerances, dwell 1, 64-command stall failure and non-learned `demo_replay`. No change to sensor.py, base.py, planner core, scoring or guards; no retraining.
- [x] Tests: conformance, TRAIN-only goal provenance, stall accounting, default-path parity, failure cases; ruff and full pytest pass.
- [x] Run the single frozen 16-attempt development comparison (resets 43000-43003 x learned/dynamics_shuffle/persistence/demo_replay) once, without retries, and report every outcome and the preregistered gate.
- [x] Review evidence and deliver through PR; TASK-033/034 physical acceptance stays open unless separately demonstrated.

## Scope and resources
CPU4, H16, K16, 2 CEM rounds, commitment 1, max 1000 commands, 200 s/attempt, 3600 s global. Final cohort 44000-44019 and TEST stay untouched.

## Phase 1 record (implementation, no physical experiment)
Branch `feat/task-043-state-goal-control` from main `5d73e56`. Frozen inputs: `data/apple-branches-v1` manifest `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`; checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt` `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`.
- Allowed pre-physics calibration check: the state-goal library builds in about 0.4 s after a 5 s episode load. It has 16 candidate nominal TRAIN demonstrations x 32 goals, 16-20/32 goals overlap a neighbour's tolerance, and self-retrieval is 16/16. In a single window, the demonstration's first 16 actions were predicted farther from goal 0 (0.039) than hold (0.017). This is a warning sign, not a gate.
- Recorded software smoke on TRAIN reset 42000 (not a development reset): scratch output only, max_steps 40, input-hash verification stubbed, learned + demo_replay. Both ran 40 commands without error (4.1 s / 0.8 s) and retrieved apple-42000. Learned observed distance to goal 0 grew from 0.11 to 0.20 against a 0.0052 tolerance. This is a runtime check only, not evidence.
- Checks: ruff check/format clean; full suite with `JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot` gave 588 passed, 7 skipped (timm-dependent JEPA-WMS tests; timm is not installed in this env).
- Pre-run revision R1 (independent review, before any development attempt; no gate/reset/mode/budget/tolerance change): `state_gate` now scores only cleanly completed, provenance-valid attempts ("missing or failed count as 0"), and falsification readings (close-goal stall, reach-without-grasp, scaffold) are computed into `report.json["state_gate"]["falsification"]` with `first_close_goal_index` recorded per demonstration/attempt. Offline TRAIN-only review check: cross-q90 sets 470/512 thresholds; goal-0 threshold 0.0048-0.018 vs segment 0.096-0.128; first close goal is 13 in all 16 demos; in 16/16 goal-0 windows hold is predicted closer (0.015-0.024) than the demo's own actions (0.034-0.044) while measured hold stays 0.10-0.13; demo<hold in 64.5% of 496 windows. Risks (a)-(d) judged declared, not blockers.

## Phase 2 record (single frozen development run)
Executed once, no retries, from clean checkout `a90bf5b48cb3326935d2222d4074607df1b0c33b` with the exact frozen command; output `outputs/apple-state-goal-control-v1/` (did not exist before). Exit 0, report `completed`, provenance valid, 16/16 attempts started and counted, global wall 129.985 s of 3600 s, each attempt 6.8-7.4 s of 200 s. Hashes: plan `fc515aa7a700153fd3c27d58933c96dde814e1165b067f08099051b318acb3d1`, resolved plan `b0cf4205c318df64924daba68aa255469c7e3cbfdb0bffc420069ba293b43faf`, report `42e91b611ce1499c18ead841ec09f5e18a317abdda1a34bcd533539f63ce55ee`, state_goals `839190fcf652c6849eba592d9cac5c63bb15a97166e201d71bff7fb4f654f88d`, state_calibration `d8dae05105a4031d4c00d676faeb4b45e86ca8e6bdbacd0af650b2b42ce38d75`; checkpoint/corpus/action/asset hashes unchanged and re-verified after the run.
- Outcome: **primary gate failed.** learned/dynamics_shuffle/persistence: 12/12 `goal_stall` on goal 0 after 64 commands, all scorer stages false (summed stages 0/0/0). Measured distance to goal 0 grew from 0.10-0.12 to 0.17-0.81 and never came within 3x of the 0.006-0.018 tolerances. Secondary: 0 learned successes.
- Non-learned `demo_replay` (open-loop retrieved TRAIN actions): success on 43000/43001/43003, reach+grasp+transport then `demo_exhausted` on 43002 (18 stages). Reference only, not a learned result.
- Falsification (`report.json["state_gate"]["falsification"]`): model/planner failure under state goals **true** (4/4 learned stalled before close goal 13); grasp-precision bottleneck false (0 learned reach); scaffold-explains false (learned scored 0 stages). Diagnostic: learned selected costs (median 0.044-0.069) were far below the measured distances produced (median 0.126-0.228), matching the R1 forecast-bias warning.
- Evidence: `docs/experiments/apple_state_goal_control_results_v1.md`, `benchmarks/manifests/apple-state-goal-control-v1.json` (per-attempt records, 70 artifact hashes). Final cohort 44000-44019 and TEST untouched. TASK-033/034 physical acceptance stays open.
- Recommended next step (not started): a privileged MuJoCo-rollout planning ceiling on the same scaffold/resets; only if it passes, grasp-phase data plus retraining with object-relative palm offsets under a new preregistration.
- Delivery: independent reviewer subagent recomputed all table values, gate/falsification readings, diagnostics and hashes from raw outputs; one factual error (close-goal frame 208 -> 224) and wording nits fixed in the results doc. Post-run checks: ruff check/format clean; full suite (render + LeRobot source) 590 passed, 7 skipped (timm); mc validate/index and diff check passed. Delivered in PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/13. Execute/report/deliver acceptance is met; the primary gate is explicitly failed, and TASK-033/034 physical acceptance stays open.
%% mc-links: [[TASK-042]] %%
