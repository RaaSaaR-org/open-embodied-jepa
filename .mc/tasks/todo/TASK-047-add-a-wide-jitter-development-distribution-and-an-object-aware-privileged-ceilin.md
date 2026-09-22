---
id: TASK-047
aliases:
- TASK-047
title: Add a wide-jitter development distribution and an object-aware privileged ceiling
slug: add-a-wide-jitter-development-distribution-and-an-object-aware-privileged-ceilin
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
- "[[TASK-046]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---



# Add a wide-jitter development distribution and an object-aware privileged ceiling

## Question
Roadmap T1. On the narrow +-6 mm development resets 43000-43003 open-loop `demo_replay` already grasps 4/4 (TASK-043/046), so the benchmark cannot separate a world model from replay; and every privileged ceiling so far (TASK-044/045/046) scored arm/hand joints only, never the apple. On a NEW wide-jitter development distribution (apple +-3 cm, plate +-2 cm, resets 45000-45007), does an object-aware exact-MuJoCo-rollout ceiling grasp+lift on >=6/8 while `demo_replay` grasps on <= ceiling-3? NON-LEARNED diagnostic only.

## Acceptance Criteria
- [ ] Wide-jitter development distribution as a new cohort (45000-45007, apple +-3 cm, plate +-2 cm) without changing the narrow development cohort, final cohort 44000-44019, TEST or TASK-034 criteria.
- [ ] Object-aware privileged ceiling (`src/embodied_jepa/object_ceiling.py`, `evaluate_apple.py --goal-kind object`) reusing the TASK-044 twin; simulator truth only in this labelled arm; tests for isolation, phase machine, costs, parity, plan/gate and worker wiring.
- [ ] Preregister protocol `docs/experiments/apple_wide_object_ceiling_v1.md` + pre-run manifest `benchmarks/manifests/apple-wide-object-ceiling-v1.json` (arms, gate, readings with next steps, frozen budget/command) before any 45000-range attempt.
- [ ] Software smoke on TRAIN reset 42000 only; fresh pre-run review; fix blockers.
- [ ] Run the frozen command exactly once from a clean checkout into a new output directory; record every outcome (results doc, manifest, this task).
- [ ] Fresh post-run verification of numbers/hashes; ruff, pytest, `mc validate`; deliver through PR.

## Scope and resources
CPU. One command, 24 attempts (8 `demo_replay`, 8 `scripted_oracle`, 8 `privileged_object`), mode-major, 960 s/attempt, 8400 s global.

## Phase 1 record (implementation, no 45000-range attempt)
Branch `feat/task-047-wide-jitter-object-ceiling` from main `fdfb67e`.
- Design probes (not evidence; disclosed in the protocol): TRAIN reset 42000 and 4 deterministic +-3/+-2 cm corner layouts (seed 0). Changes driven by probes: geodesic rotation error (small-angle cross product stalls at the initial 90 deg), blocked-descent rule (thumb meets the table ~12 cm above the apple centre, as in every collector demo), blocked transport/lower hand-over, rotation weight 0.2, H6/K24/std 0.3. Final probes: 42000 and corner (+3,-3|-2,+2) full success (387/338 commands, 0 full-state parity mismatches); scripted collector succeeded on 42000 and all 4 corners.
- TRAIN reset 42000 smoke through the full evaluator (scratch `outputs/task047-scratch/smoke-42000-a`): completed, 3/3 counted, 287 s; TRAIN artifacts byte-identical to TASK-045/046 (`839190fc...`, `d8dae051...`); demo_replay (apple-42000), scripted_oracle and privileged_object (388 commands, 387/387 exact robot and full-state parity) all succeeded. Runtime check only.

## Notes
%% mc-links: [[TASK-046]] %%
