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
- [ ] Object ceiling v2 added alongside v1 (`src/embodied_jepa/object_ceiling_v2.py`, v1 behaviour and evidence unchanged): xy-weighted descent cost and xy-keyed blocked-descent detection; close phase holding the achieved palm pose; transport/lower gated on apple-over-plate-centre tolerance before release. Tests.
- [ ] Development tuning only on a documented TRAIN-side tuning range (not 45000-45007, not the new dev range, not 44000-44019, not TEST).
- [ ] Preregister `docs/experiments/apple_wide_object_ceiling_v2.md` + `benchmarks/manifests/apple-wide-object-ceiling-v2.json` (fresh dev resets, arms ceiling v2/scripted_oracle/demo_replay, primary gate, readings with next steps, frozen budget/command) before any attempt on the fresh range.
- [ ] Fresh pre-run review; fix blockers; TRAIN-only smoke.
- [ ] Frozen command run once from a clean committed revision into a new output directory; every outcome recorded.
- [ ] Fresh post-run verification; results doc; ruff, pytest, `mc validate`; PR, CI, merge.

## Scope and resources
CPU. One command, 48 attempts (16 each of `demo_replay`, `scripted_oracle`, `privileged_object` v2; primary 45100-45107 then secondary 45000-45007), mode-major, 1200 s/attempt, 20400 s global, 10 s/command deadline.

## Phase 1 record (implementation, no 45100/45000-range attempt)
Branch `feat/task-049-object-ceiling-v2` from main `61f8ec1`.
- v2 in `src/embodied_jepa/object_ceiling_v2.py` (subclass of v1; v1 behaviour unchanged apart from a behaviour-preserving `_record_progress` hook); `evaluate_apple.py --object-ceiling-version 2` (v1 plan unchanged), gate `object_ceiling_v2_gate`; tests in `tests/test_object_ceiling_v2.py` and `tests/test_apple_evaluation.py`.
- Tuning range 49000-49015 (wide-jitter draws, used nowhere else); 10 probe rounds disclosed in the protocol; frozen v2g reached full success on 15/16 tuning resets.
- TRAIN reset 42000 smoke (`outputs/task049-scratch/smoke-42000-a`, worktree scratch): completed, 3/3 success, v2 278 commands, parity 277/277 exact, TRAIN artifacts byte-identical.
- Preregistration: `docs/experiments/apple_wide_object_ceiling_v2.md`, `benchmarks/manifests/apple-wide-object-ceiling-v2.json`.

## Notes
%% mc-links: [[TASK-047]] %%
