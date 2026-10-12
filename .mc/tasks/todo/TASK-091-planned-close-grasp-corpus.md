---
id: TASK-091
aliases:
- TASK-091
title: 'Planned close within the plan: TASK-090''s A′ on fresh roots with A2 redefined, then its closed-loop grasp-and-lift comparison on fresh resets'
slug: planned-close-grasp-corpus
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- data
- planning
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-090]]"
due_date: ''
created: 2026-10-13
updated: 2026-10-13
---

# Planned close within the plan: TASK-090's A′ on fresh roots with A2 redefined, then its closed-loop grasp-and-lift comparison on fresh resets

## Description

R27.43's recommendation after TASK-090 closed at **A′-FAIL** (R27.34; escalate, no clause; BD and
BS never ran). Its development diagnosis D-A2 read **D-TIMING** (R27.42; development, not gated,
one run, offline, simulation only, on 64 roots 905000–905063): the three A2-failing models plan
the close at step 2–4 of the 8-step plan, and all nine G models close by plan step ≤ 4 on 64/64
roots. TASK-091 takes the "corrected A2 definition" branch, not a planner change (R27.46).
TASK-089's step-averaged CEM planner (λ = 2.0552, receding horizon, first command executed) and
everything else in TASK-090 §2–§7 stay frozen, so the comparison of M-v2g with M-v1p stays
unconfounded. It reuses TASK-090's checkpoints and corpora unchanged (no retraining).

- **Stage A″** (a no-regression gate, offline) is TASK-090's A′ on 64 fresh roots (906000–906063),
  with A2's close bar redefined: "the A2-seed plan's grasp channel (dim 13) > 0.5 at some plan
  step k ≤ 4, on ≥ 48/64 roots". close@R1 is reported. lift@R2 ≥ 48/64 is kept. A1 and the A′-FAIL
  rule are unchanged. The k ≤ 4 threshold was chosen after seeing D-A2's development numbers.
- **Stage BD, GO and Stage BS** run exactly as TASK-090 §7 (arms, bar rule, n = 128, reset-level
  sign-flip tests, rows B-VOID…B-UNDER, the §9 clause in its scope), on fresh resets (BD
  90300–90331, BS 907000–907127), plus reported receding-horizon deferral measures.

This is not Apple→Plate. R7 does not change.

Protocol: [grasp_play_corpus_v1_planned_close.md](../../../docs/experiments/grasp_play_corpus_v1_planned_close.md)
(**DRAFT**, R27.45–R27.54; DECISIONS (as)).

## Acceptance criteria

- [ ] DRAFT preregistration reviewed independently and merged
- [ ] Stage 0: frozen code for the redefined A2 measure (the full 8-step plan at the A2 seed, its first command checked bitwise against `first_planned`), the new cohorts and the deferral measures; seed checks; checkpoint and feature-store hash checks; debug smoke and per-job timing; reviewed; protocol frozen at merge
- [ ] Stage A″ recorded (A″-FAIL escalates, no clause)
- [ ] Stage BD recorded, the bar set; independent GO posted before Stage BS
- [ ] Stage BS run; results document, row and DECISIONS rulings reviewed and merged
- [ ] Evidence in `~/develop/emai/evidence/task091-*/` with `SHA256SUMS`; two MP4s (one success if any, one failure)
- [ ] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds: A″ roots 906000–906063, BD 90300–90331, BS 907000–907127, Stage 0 debug 90700–90799.
  There is no new model seed: the checkpoints are TASK-090's and TASK-087's. The salts are reused
  (TASK-088's 8801–8804 and TASK-090's 9002, 9004 and 9006); there is no new salt.
- `worktrees/task087-run` must not be archived or removed until TASK-091 closes, because its
  feature store and projection are read (as in TASK-090 §17).
- Budget: A″ under 1 GPU h; BD about 2 h; BS about 15.5 GPU h (R27.25's projection). Every GPU job
  stays under the 7 200 s per-job cap and takes one `gpu_run.sh --wait` lock, and BOARD.md's
  Requests are checked before each job.
%% mc-links: [[TASK-090]] %%
