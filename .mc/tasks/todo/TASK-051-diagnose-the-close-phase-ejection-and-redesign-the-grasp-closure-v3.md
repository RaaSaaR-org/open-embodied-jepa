---
id: TASK-051
aliases:
- TASK-051
title: Diagnose the close-phase ejection and redesign the grasp closure (ceiling v3)
slug: diagnose-the-close-phase-ejection-and-redesign-the-grasp-closure-v3
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
- "[[TASK-049]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---

# Diagnose the close-phase ejection and redesign the grasp closure (ceiling v3)

## Question
TASK-049's object-aware exact-rollout ceiling v2 failed its primary gate on the fresh wide-jitter development resets 45100-45107 (5/8 success, 5/8 grasp; needed 6 and 7) while the scripted collector succeeded 8/8, giving outcome `ceiling_inadequate_task_feasible`. All four failures were one mode: 8-9 commands into the close phase the closing fingers ejected the apple (0.8-3.1 cm of xy motion, peaking 2.7-3.5 cm). Entry geometry did not separate failures from successes and the cause was not isolated. What causes the close-phase ejection, and does a principled redesign of the grasp closure (ceiling v3), expressed only in quantities a learned head could predict, reach full success >=6/8 and grasp+lift >=7/8 on a FRESH wide-jitter development reset range under exact dynamics? NON-LEARNED diagnostic only.

## Acceptance Criteria
- [x] Written, evidence-backed diagnosis of the close-phase ejection from instrumented TRAIN-side runs (per-finger contact forces, apple impulse/displacement per command, palm pose, finger joint trajectories, grasp command schedule, ceiling close vs scripted close command-by-command). Exploratory, clearly labelled, not gate evidence.
- [x] Ceiling v3 added alongside v1/v2 (v1 and v2 behaviour and evidence unchanged), with the closure redesign following from the diagnosis. Tests.
- [x] Simulator physics/contact parameters unchanged, or, if changed, documented loudly as a task-realism change with the old config kept and the scripted baseline re-run.
- [x] Development tuning only on a documented TRAIN-side tuning range (49100-49131; not 45000-45007, not 45100-45107, not the gated range 45200-45207, not 44000-44019, not TEST).
- [x] Preregister `docs/experiments/apple_wide_grasp_closure_v3.md` + `benchmarks/manifests/apple-wide-grasp-closure-v3.json` (fresh dev resets, arms ceiling v3/scripted_oracle/demo_replay, primary gate, readings with next steps, frozen budget/command) before any attempt on the gated range.
- [x] Fresh pre-run review; fix blockers; TRAIN-only smoke.
- [x] Frozen command run once from a clean committed revision into a new output directory; every outcome recorded.
- [x] Fresh post-run verification; results doc; ruff, pytest, `mc validate`; PR, CI, merge.

## Phase 2 record (single frozen run)
Executed once from the clean tracked checkout `b673f3de50f04aa50281cf555112da05ac90d0c8`, exact frozen command, new output `outputs/apple-wide-grasp-closure-v3/` in the main checkout (2026-09-22T20:06:07Z-21:26:49Z). Exit 0, report `completed`, provenance valid, 72/72 counted, 4836.6 s of 32400 s, no attempt shortened, timed out or deadline-missed; every ceiling attempt terminated with `success`.
- **Primary gate PASSED (conclusive):** on the fresh resets 45200-45207 the v3 ceiling succeeded 8/8 (>=6 needed) and grasped 8/8 (>=7 needed); rollouts exact (0/2378 robot and 0/2378 full-state mismatches on the primary cohort, exactly executed steps - 1 checks per attempt; 6935+6935 across all 24 ceiling attempts) -> outcome `ceiling_adequate`. `demo_replay` grasped 2/8 (`replay_separated_diagnostic` true). `scripted_oracle` 8/8. First time the privileged object-aware ceiling has cleared its gate.
- **Close-phase ejection eliminated:** apple motion 0.49-1.35 cm on all 24 ceiling attempts against the 1.5 cm threshold; v2's gated failures were 2.70-19.10 cm on the same measure.
- **Secondary (non-gating) 45100-45107 and 45000-45007:** v3 16/16 grasp and 16/16 success, including every reset where v2 ejected the apple (v2: 5/8 and 7/8; v1: 3/8 on 45000-45007). Those resets shaped v1/v2 or produced the failures this redesign targets, so they are diagnostics, not independent evidence.
- Cost: 0.621-0.649 s per ceiling command, 265-342 commands per attempt, max per-command control time 1.074 s against the 10 s deadline.
- **Process failure recorded, not excused:** the run was launched from `b673f3d` while the pre-run reviewer was still verifying, because the author took the review file on disk for the verdict. The reviewer's final verdict was BLOCK on three prose corrections (B2/B3/B4); they were applied as R2 (`b2dc4d4`) before any gated outcome was inspected, derived entirely from TRAIN-side data, and confirmed cleared. R2's only `src/` change is the v3 module docstring (AST identical once docstrings are blanked), so the snapshot is byte-identical to `b673f3d` and differs from HEAD in exactly that file. The protocol now records that a gated run starts on the reviewer's reported verdict.
- Post-run verification (fresh subagent) recomputed the gate from the 72 attempt reports with its own code, confirmed provenance, the frozen command, every cell of both tables, the leakage sweep and that nothing frozen changed after the run. It corrected two v2-related numbers (the gated-failure range to 2.70-19.10 cm and the tuning range to 2.45-9.12 cm, and noted six rather than four of v2's gated closes crossed the 1.5 cm threshold); both corrections strengthen the contrast.
- Evidence: `docs/experiments/apple_wide_grasp_closure_results_v3.md`, `docs/experiments/apple_grasp_closure_diagnosis.md`, `benchmarks/manifests/apple-wide-grasp-closure-v3.json`, `docs/reviews/apple_wide_grasp_closure_v3_review.md`. NON-LEARNED; learned Apple->Plate remains at zero successes; TASK-033/034 stay open; narrow dev, final cohort 44000-44019 and TEST untouched.
- Preregistered next step: **T4** - pair this cost/phase design with a learned world model whose heads predict its terms, trained on the wide-jitter TRAIN corpus (TASK-048), under its own preregistration; keep this ceiling as the exact-dynamics upper reference. The close phase needs no prediction at all.

## Phase 1 record (implementation and preregistration; no 45200/45100/45000-range attempt)
Branch `feat/task-051-grasp-closure-v3` from main `6739427`.
- Diagnosis on TRAIN-side tuning resets (declared range 49100-49131; stepped 49100-49117, 49120, 49124-49131) with two uncommitted scratch harnesses under the main checkout's ignored `outputs/task051-scratch/`: per-command close-phase forensics and a paired experiment that snapshots the live `MjData` plus the controller at the first close command and replays 16 candidate closure designs (289 replays) from that identical state. Written up in `docs/experiments/apple_grasp_closure_diagnosis.md` (EXPLORATORY, not gate evidence).
- Cause: the v2 close cost is nearly flat in the palm command while the Dex3 synergy is shutting (eleven commands under the joint-rate limit), so CEM proposal noise sets the palm's motion; when it drives the palm down and sideways quickly, the still-open thumb strikes the apple first and sweeps it out. First hand-apple contact at close command 4-8 ejected on 4/4, at 10-11 held on 12/12; the collector contacts at 11 on all 16 resets. Simulator physics and contact parameters are unchanged.
- v3 in `src/embodied_jepa/object_ceiling_v3.py` (v2 subclass overriding `_bounds()` for the close phase only: lateral and rotational deltas pinned to zero, vertical delta in [-close_descent_bound, 0]); `evaluate_apple.py --object-ceiling-version 3` with the fresh cohort 45200-45207 and the non-gating secondary cohorts, the object gate parameterised by version (v2 output unchanged); tests in `tests/test_object_ceiling_v3.py` and `tests/test_apple_evaluation.py`.
- TRAIN reset 42000 smoke (`outputs/task051-scratch/smoke-42000-a`): completed, 3/3 success, v3 283 commands, parity 282/282 robot-state and 282/282 full-state exact, TRAIN artifacts byte-identical to TASK-045/046/047/049.
- Preregistration: `docs/experiments/apple_wide_grasp_closure_v3.md`, `benchmarks/manifests/apple-wide-grasp-closure-v3.json`.
- Pre-run review (fresh subagent, `docs/reviews/apple_wide_grasp_closure_v3_review.md`): no blocking finding, CLEAR TO RUN; nine non-blocking items applied as R1 with no change to any reset, seed, threshold, budget, parameter or command.

## Follow-up (not in this task's scope)
`tests/test_apple_evaluation.py::test_demo_replay_is_open_loop_non_learned_and_exhausts` fails when that module is run in isolation and passes in the full suite: a lazy `from embodied_jepa.object_ceiling import GUARD_REFUSALS` inside `project_open_loop` re-executes `class _BlindSimulation(MuJoCoSimulation)` against a monkeypatched `MuJoCoSimulation`. The code path and the test are identical to main, so this is inherited. Fix by pre-importing in the fixture.

## Scope and resources
CPU. One command, 72 attempts (24 each of `demo_replay`, `scripted_oracle`, `privileged_object` v3): the fresh primary cohort 45200-45207, then the non-gating secondary cohorts 45100-45107 and 45000-45007, mode-major, 1200 s/attempt, 32400 s global, 10 s/command deadline.

## Notes
%% mc-links: [[TASK-049]] %%
