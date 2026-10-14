---
id: TASK-091
aliases:
- TASK-091
title: 'Planned close within the plan: TASK-090''s A′ on fresh roots with A2 redefined, then its closed-loop grasp-and-lift comparison on fresh resets'
slug: planned-close-grasp-corpus
status: done
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
one run, offline, simulation only, on 64 roots 905000–905063): the three A2-failing models mostly
plan the close at step 2–4 of the 8-step plan, and all nine G models close by plan step ≤ 4 on 64/64
roots. TASK-091 takes the "corrected A2 definition" branch, not a planner change (R27.46).
TASK-089's step-averaged CEM planner (λ = 2.0552, receding horizon, first command executed) and
everything else in TASK-090 §2–§7 stay frozen, so no planner change confounds the comparison of M-v2g with M-v1p (R27.50's deferral confound
remains). It reuses TASK-090's checkpoints and corpora unchanged (no retraining).

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

- [x] DRAFT preregistration reviewed independently and merged (#209, `940de4e`)
- [x] Stage 0: frozen code for the redefined A2 measure (the full 8-step plan at the A2 seed, its first command checked bitwise against `first_planned`), the new cohorts and the deferral measures; seed checks; checkpoint and feature-store hash checks; debug smoke and per-job timing; reviewed; protocol frozen at merge
- [x] Stage A″ recorded (A″-FAIL escalates, no clause)
- [x] Stage BD recorded, the bar set; independent GO posted before Stage BS
- [x] Stage BS run; results document, row and DECISIONS rulings reviewed and merged
- [x] Evidence in `~/develop/emai/evidence/task091-*/` with `SHA256SUMS`; two MP4s (one success if any, one failure)
- [x] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds: A″ roots 906000–906063, BD 90300–90331, BS 907000–907127, Stage 0 debug 90700–90799.
  There is no new model seed: the checkpoints are TASK-090's and TASK-087's. The salts are reused
  (TASK-088's 8801–8804 and TASK-090's 9002, 9004 and 9006); there is no new salt.
- `worktrees/task087-run` must not be archived or removed until TASK-091 closes, because its
  feature store and projection are read (as in TASK-090 §17).
- Budget: A″ under 1 GPU h; BD about 2 h; BS about 15.5 GPU h (R27.25's projection). Every GPU job
  stays under the 7 200 s per-job cap and takes one `gpu_run.sh --wait` lock, and BOARD.md's
  Requests are checked before each job.
- 2026-10-13, Stage 0 (branch `feat/task091-stage0`; R27.55–R27.60, DECISIONS (at); protocol §14):
  code `grasp_corpus_wm_planned_close{,_runtime,_run}.py`, `scripts/{run,summarize}_task091.py`,
  `scripts/run_task091_stage.sh`, `tests/test_grasp_corpus_planned_close.py`. Hash checks and
  the seed check pass; debug smoke on 90700–90799 (debug, not results): A″ first-command check
  bitwise on all plans, every closed-loop arm runs, split = unsplit per reset bitwise, the plan
  recorder leaves executed commands bitwise unchanged. Stage 0 changes nothing of §12 (processes,
  BS split 2 × 64, third deferral measure kept); worst 64-reset job projects to 4 814 s. About
  0.52 GPU h. Evidence `~/develop/emai/evidence/task091-stage0/` (`SHA256SUMS`). The protocol is
  frozen at the Stage 0 PR's merge; A″ runs next (`scripts/run_task091_stage.sh a`).
- 2026-10-13, Stage A″ (branch `docs/task091-a`; R27.61–R27.64, DECISIONS (au); protocol §15):
  ran once at `dbab1a5` (#210's merge) from a clean tree in `worktrees/task091-run`, after
  `verify` and the seed check passed. Row **A″-PASS** (summarizer: `fails: {'v2g': 0, 'v1p': 1}`).
  On 64 roots 906000–906063: close≤4@R1 64 / 64 for all nine G models; lift@R2 52–63 for the six
  new G models (G-v1-87101 34, reported only); close@R1 reported 0–64. G-v1p-90102 fails A1 at
  h = 8 (lift-vs-stay 59 < 61) and stays in M-v1p's pooled count. One run, offline, simulation
  only; a no-regression gate, not a closed-loop result. About 0.18 GPU h (68 s + 566 s). Evidence
  `~/develop/emai/evidence/task091-a/` (`SHA256SUMS`). Next: Stage BD (tell V2D first).
- 2026-10-13, Stage BD (branch `docs/task091-bd`; R27.65–R27.68, DECISIONS (av); protocol §16):
  ran once at `dbab1a5` from a clean tree in `worktrees/task091-run`, after `verify`, the seed
  check and the tests passed. Development, 32 resets 90300–90331, one run, simulation only, not
  gated. ik 26 / 32 sets the bar at 0.40 (BS pooled bar 156 / 384, per-seed floor 26 / 128).
  M-v2g-90100 5 / 32, M-v1p-90100 3 / 32, M-v1-87100 2 / 32 (reported), ik-nopress 24 / 32,
  hold 0 / 32, random 0 / 32. Deferral (reported only): M-v2g-90100 closed in the grasp phase on
  every episode (median command 10); about half its grasp-phase re-plans planned the close at
  step 1. Both BD world-model seeds are late planners (§9's confound). No cap hit; 8 jobs,
  5 873 s (about 1.63 GPU h). Evidence `~/develop/emai/evidence/task091-bd/` (`SHA256SUMS`).
  Next: an independent GO or NO-GO on this record; BS only after GO (tell V2D first).
- 2026-10-14, Stage BS (branch `docs/task091-bs`; R27.69–R27.73, DECISIONS (aw); protocol §17):
  ran once after the independent GO on #212, at `dbab1a5` from a clean tree in
  `worktrees/task091-run`, after `verify` and the seed check passed. 128 gated resets
  907000–907127, one run, simulation only. Row **B-FAIL** (summarizer `problems: []`): M-v2g
  22 / 384 (8, 8, 6 of 128) against M-v1p 26 / 384 (11, 7, 8); pooled bar 156 / 384 and per-seed
  floor 26 / 128 not met; "M-v2g beats M-v1p" p = 0.767; "beats random" p = 1.9e-6; bootstrap
  upper bound +0.023 < δ = +0.10. ik 82 / 128, M-v1-87100 1 / 128 (reported). Determinism re-run
  matched; no cap hit; 20 jobs, 53 260 s (about 14.8 GPU h). Per-seed deferral measures in §17
  (late-planning seeds did not score lower in this run; that timing played no part is not shown).
  C1 2.38 and C2 1.69 apply. The §7 clause fires (scope unchanged); TASK-091 closes with the
  record's merge (R27.72); R7 unchanged. Evidence `~/develop/emai/evidence/task091-bs/`
  (`SHA256SUMS`, two MP4s).
%% mc-links: [[TASK-090]] %%
