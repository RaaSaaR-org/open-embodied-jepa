---
id: TASK-049
aliases:
- TASK-049
title: Redesign the object-aware ceiling (v2) and gate it on fresh wide-jitter development resets
slug: redesign-the-object-aware-ceiling-v2-and-gate-it-on-fresh-wide-jitter-resets
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
- "[[TASK-047]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---




# Redesign the object-aware ceiling (v2) and gate it on fresh wide-jitter development resets

## Question
TASK-047's object-aware exact-rollout ceiling grasped 5/8 on the wide-jitter development resets 45000-45007 (gate >=6) while the scripted collector succeeded 8/8: outcome `ceiling_inadequate_task_feasible`. Its post-hoc failures were geometric (descent cost almost blind to xy, fixed 1 cm blocked window, close phase pressing towards an unreachable point, release allowed up to the 4 cm plate radius). Does a principled v2 cost/phase design, expressed only in quantities a learned head could predict (palm-apple offset, apple height, apple-plate offset, grasp/contact state), reach full success >=6/8 and grasp+lift >=7/8 on FRESH wide-jitter development resets under exact dynamics? NON-LEARNED diagnostic only.

## Acceptance Criteria
- [x] Object ceiling v2 added alongside v1 (`src/embodied_jepa/object_ceiling_v2.py`, v1 behaviour and evidence unchanged): xy-weighted descent cost and xy-keyed blocked-descent detection; close phase guarding against palm drift from the achieved pose (v1 pressure + 1 cm dead band, see protocol); transport/lower gated on apple-over-plate-centre tolerance before release. Tests.
- [x] Development tuning only on a documented TRAIN-side tuning range (not 45000-45007, not the new dev range, not 44000-44019, not TEST).
- [x] Preregister `docs/experiments/apple_wide_object_ceiling_v2.md` + `benchmarks/manifests/apple-wide-object-ceiling-v2.json` (fresh dev resets, arms ceiling v2/scripted_oracle/demo_replay, primary gate, readings with next steps, frozen budget/command) before any attempt on the fresh range.
- [x] Fresh pre-run review; fix blockers; TRAIN-only smoke.
- [x] Frozen command run once from a clean committed revision into a new output directory; every outcome recorded.
- [x] Fresh post-run verification; results doc; ruff, pytest, `mc validate`; PR, CI, merge.

## Scope and resources
CPU. One command, 48 attempts (16 each of `demo_replay`, `scripted_oracle`, `privileged_object` v2; primary 45100-45107 then secondary 45000-45007), mode-major, 1200 s/attempt, 20400 s global, 10 s/command deadline.

## Phase 1 record (implementation, no 45100/45000-range attempt)
Branch `feat/task-049-object-ceiling-v2` from main `61f8ec1`.
- v2 in `src/embodied_jepa/object_ceiling_v2.py` (subclass of v1; v1 behaviour unchanged apart from a behaviour-preserving `_record_progress` hook); `evaluate_apple.py --object-ceiling-version 2` (v1 plan unchanged), gate `object_ceiling_v2_gate`; tests in `tests/test_object_ceiling_v2.py` and `tests/test_apple_evaluation.py`.
- Tuning range 49000-49015 (wide-jitter draws, used nowhere else); 10 probe rounds disclosed in the protocol; frozen v2g reached full success on 15/16 tuning resets.
- TRAIN reset 42000 smoke (`outputs/task049-scratch/smoke-42000-a`, worktree scratch): completed, 3/3 success, v2 278 commands, parity 277/277 exact, TRAIN artifacts byte-identical.
- Preregistration: `docs/experiments/apple_wide_object_ceiling_v2.md`, `benchmarks/manifests/apple-wide-object-ceiling-v2.json`.

## Phase 2 record (single frozen run)
Pre-run review (fresh subagent, `docs/reviews/apple_wide_object_ceiling_v2_review.md`): no blocking finding, CLEAR TO RUN; non-blocking items applied as R1 `7b94f06` (a probe-vs-live release exactness test plus protocol wording; no reset, seed, threshold, budget, parameter or command changed).
Executed once from the clean tracked checkout `7b94f06410f8453d4cf2e88d67bccc9682a9820c`, exact frozen command, new output `outputs/apple-wide-object-ceiling-v2/` in the main checkout (2026-09-22T16:49:44Z-17:43:21Z). Exit 0, report `completed`, provenance valid, 48/48 counted, 3216.8 s of 20400 s, no attempt shortened, timed out or deadline-missed.
- **Primary gate failed (conclusive):** on the fresh resets 45100-45107 the v2 ceiling succeeded 5/8 (>=6 needed) and grasped 5/8 (>=7 needed); rollouts exact (0/2169 robot and 0/2169 full-state mismatches, non-vacuous); `scripted_oracle` 8/8 -> outcome `ceiling_inadequate_task_feasible`. `demo_replay` grasped 4/8 (separation diagnostic false, non-gating).
- **Secondary (non-gating) 45000-45007:** v2 grasped 7/8 and succeeded 7/8 against v1's 5/8 and 3/8; the v1 descend stalls, lift failure and off-centre releases are gone. Those resets informed the redesign, so this is a diagnostic, not independent evidence.
- All four ceiling failures are one mode: the closing fingers eject the apple 8-9 commands into the close; entry geometry does not separate failures from successes; cause not isolated.
- Evidence: `docs/experiments/apple_wide_object_ceiling_results_v2.md`, `benchmarks/manifests/apple-wide-object-ceiling-v2.json`. NON-LEARNED; TASK-033/034 stay open; narrow dev, final cohort and TEST untouched.
- Post-run verification (fresh subagent) confirmed the gate, provenance, snapshot byte-equality and manifest; it corrected six diagnostic numbers and three wording items, all applied.
- Preregistered next step: diagnose the close-phase ejection and redesign the grasp closure under a new preregistration; do not pair the ceiling with a learned model; the placement predictor and descent/carry costs may carry over.

## Notes
%% mc-links: [[TASK-047]] %%
