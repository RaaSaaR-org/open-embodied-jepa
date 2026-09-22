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

## Phase 2 record (single frozen run)
Pre-run review (fresh subagent, `docs/reviews/apple_wide_object_ceiling_review.md`): 2 blocking findings (guard refusal in demo_replay/scripted was an uncounted runtime_error; gate passed without all ceiling attempts counted) fixed as R1 `9ec5938` with non-vacuous parity, feasible-only rollouts and tests; post-R1 TRAIN-42000 smoke identical; re-review cleared the run (`613ecf8`, docs-only on top of R1).
Executed once from clean tracked checkout `613ecf81f800e3ecf3d16653b19fd7e9d051a751` (only untracked CLAUDE.md), exact frozen command, new output `outputs/apple-wide-object-ceiling-v1/` (2026-09-22T14:05:44Z-14:43:46Z). Exit 0, report `completed`, provenance valid, 24/24 counted, 2282.0 s global, no attempt shortened or timed out.
- **Primary gate failed (conclusive):** `privileged_object` grasp 5/8 (>=6 needed); demo_replay grasp 2/8 (<= ceiling-3 held); rollouts exact (0/3314 robot and 0/3314 full-state mismatches, non-vacuous). Outcome `ceiling_inadequate_task_feasible` because `scripted_oracle` grasped and succeeded 8/8.
- Ceiling per reset: success 45001/45002/45005; grasp+transport but release 4.5-4.6 cm off plate on 45000/45007; descend stall on 45003/45004 (palm height-blocked at about +0.115 m, but xy error had drifted to 1.05-1.06/1.21-1.25 cm, just outside the 1.0 cm blocked window); lift stall on 45006 (palm drifted to 3.6 cm behind the apple during close; apple itself moved 0.3 mm). Post-verification corrections: see the results doc. demo_replay: success 45000/45006, guard_refused on 5/8, demo_exhausted 45005.
- Evidence: `docs/experiments/apple_wide_object_ceiling_results_v1.md`, `benchmarks/manifests/apple-wide-object-ceiling-v1.json`. NON-LEARNED; TASK-033/034 stay open; narrow dev, final cohort and TEST untouched.
- Preregistered next step: diagnose/redesign the ceiling's cost and transitions under a new preregistration; do not pair with a learned model; T2 may proceed only with the wide distribution validated by the scripted reference (8/8).

## Notes
%% mc-links: [[TASK-046]] %%
