---
id: TASK-046
aliases:
- TASK-046
title: Test hybrid phase control (tracked approach + demo close) under the privileged ceiling
slug: test-hybrid-phase-control-tracked-approach-demo-close-under-the-privileged-ceiling
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
- "[[TASK-045]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---



# Test hybrid phase control (tracked approach + demo close) under the privileged ceiling

## Question
TASK-045's privileged exact-rollout trajectory tracker reached the scorer's reach stage on 4/4 development resets 43000-43003 but grasp on 0/4 (apple fell without lift on 4/4); tracking error rose after the close row. Open-loop `demo_replay` (TASK-043) grasped 4/4 on the same resets. Does privileged trajectory tracking up to the retrieved TRAIN demonstration's close row, followed by open-loop execution of that demonstration's own remaining actions from the matched frame, reach the scorer's grasp stage on >=2/4 resets? NON-LEARNED diagnostic only.

## Acceptance Criteria
- [ ] Preregister `docs/experiments/apple_hybrid_phase_v1.md` (design, TRAIN-only handoff justification, primary gate, secondary early-handoff arm and demo_replay same-code reference with their own readings, budgets <=840 s/attempt and <=3600 s global, interpretations, frozen command) before any development attempt.
- [ ] Implement in isolation (`src/embodied_jepa/hybrid_phase.py`, `evaluate_apple.py --goal-kind hybrid`); no change to sensor.py, base.py, trajectory_tracking.py, privileged_rollout.py, waypoint_planning.py, scorer or guards; image/state/trajectory plans and gates unchanged.
- [ ] Tests for handoff, replay, termination, gate and worker wiring; ruff and full pytest pass.
- [ ] Recorded software smoke on TRAIN reset 42000 only; independent pre-run review; fix blockers before running.
- [ ] Run the frozen command exactly once from a clean checkout into a new output directory; record every outcome (results doc, manifest, this task, TASK-033 line).
- [ ] Independent post-run verification of numbers against raw outputs; deliver through PR. TASK-033/034 stay open; final cohort 44000-44019 and TEST untouched.

## Scope and resources
CPU. One command, 12 attempts (4 primary `privileged_hybrid`, 4 `demo_replay`, 4 secondary `privileged_hybrid_early`), mode-major, 840 s/attempt, 3600 s global.

## Phase 1 record (implementation, no development attempt)
Branch `feat/task-046-hybrid-phase` from main `d83d8e3`. Implementation `eadb99c`, protocol `0c419e4`, smoke-informed revision R0 `323aa6d` (guard refusal during replay -> clean `replay_projection_rejected`; no gate/handoff/budget change).
- TRAIN-only handoff numbers (16 references): first closing frame 211 in all demos; close row 141/142 at frame 211; row close-16 at frame 136-137, before the blocked descent (~frames 144-210).
- TRAIN reset 42000 smoke (scratch, hashes stubbed): TRAIN artifacts byte-identical to TASK-045; primary handed off at row 143/cmd 208, guard refusal on replay action 233 (runtime_error before R0, replay_projection_rejected after R0, identical rerun); early arm handed off at row 126/cmd 205, replayed 364 actions, apple dropped; no grasp on either (runtime check, not evidence). Parity 207/207 and 204/204 exact.
