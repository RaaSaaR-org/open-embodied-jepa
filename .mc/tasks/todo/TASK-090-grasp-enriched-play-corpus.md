---
id: TASK-090
aliases:
- TASK-090
title: 'Grasp-enriched play corpus: retrain grounded LeWM on more scripted grasp-and-lift, against an equal-size control, then closed-loop grasp-and-lift'
slug: grasp-enriched-play-corpus
status: in-progress
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
- "[[TASK-089]]"
due_date: ''
created: 2026-10-12
updated: 2026-10-12
---

# Grasp-enriched play corpus: retrain grounded LeWM on more scripted grasp-and-lift, against an equal-size control, then closed-loop grasp-and-lift

## Description

The data option of DECISIONS R26.34 after TASK-089 closed A-PASS(step-mean) → B-FAIL (G-chg-87100
0/64, exact 95 % 0.000–0.056, bar 26/64; replication seeds 8/64 and 7/64, reported only; simulation
only, one model seed gated; "LeWM needed" not shown; the clause fired, R26.33). Two equal-size
additions to `play-v1` train: play-v2g (3 200 episodes from a new grasp-enriched collector,
`play_policy_grasp_v0`) and play-v1p (3 200 episodes from the unchanged `play_policy_v0`). TASK-087's
G recipe, latent, projection and budget, and TASK-089's step-averaged CEM planner, are held fixed.
A data gate (Stage C) before training, validity (E) and no-regression (A′) offline gates, then
Stage BD, an independent GO and the gated Stage BS: all three M-v2g seeds pooled must reach
3 × ⌈bar × n⌉ (bar min(0.40, 0.9 × ik's BD rate); n = 64, or 128 by Stage 0's power rule), each
seed ⌈⌈bar × n⌉ / 2⌉, and beat random and M-v1p (exact one-sided sign-flip tests over per-reset
differences, p < 0.01). Not Apple→Plate; R7 does not change.

Protocol: [grasp_play_corpus_v1.md](../../../docs/experiments/grasp_play_corpus_v1.md) (FROZEN
at Stage 0's merge, R27.1–R27.28; Stage 0's record is its §17; the collection and Stage C
record is its §18; featurisation, Stage T, E and A′ (ending A′-FAIL) are its §19).

## Acceptance criteria

- [x] DRAFT preregistration reviewed independently and merged (#202)
- [x] Stage 0: collector, label script, training driver (pinned `play-v1` moments; store mmap/location), planner loader, tests; 300 + 100 episode debug collection with the Stage C ratios disclosed; path and hash checks after the data-disk move; A-K0 (A-CEILING if h = 8 does not qualify); timed debug smoke; power estimate and BS size; reviewed; protocol frozen at merge (#203)
- [x] Collection (play-v1p, play-v2g) and Stage C recorded; C-FAIL stops before training (C-PASS, §18, R27.29–R27.31)
- [x] Featurisation and Stage T (8 jobs) recorded (F-DONE, T-DONE, §19, R27.32)
- [x] Stage E and Stage A′ recorded (E-PASS; **A′-FAIL**, §19, R27.33–R27.34: escalate, no clause; BD and BS not run)
- [ ] Stage BD recorded, the bar set; independent GO posted before Stage BS
- [ ] Stage BS run; results document, row and DECISIONS rulings reviewed and merged
- [ ] Evidence in `~/develop/emai/evidence/task090-*/` with `SHA256SUMS`; two MP4s (one success if any, one failure)
- [ ] CLAUDE.md entry points and the plan's Progress table updated

## Notes

- Seeds: play-v1p 910000–913199, play-v2g 920000–923199, model 90100–90102, A-K0 90000–90031,
  A′ 903000–903063, BD 90200–90231, BS 904000–904127 (enlarged by Stage 0's power rule, R27.22), debug
  90900–90999, Stage 0 debug collection 930000–930399; salts 9002–9006 (9001 skipped).
- Large artifacts on `/mnt/data/emai/projects/open-embodied-jepa/` through the repo's links.
- Stage 0 (R27.21–R27.28): Stage C's bars revised to C1 ≥ 2.0, C2 ≥ 1.5 (debug points 2.31 and
  1.64 below the declared 2.5 and 1.75); A-K0 passes at h = 8; Stage BS n = 128; Stage T in memory;
  M-v1's 87101 and 87102 dropped from Stage BS.
- If the clause fires (B-FAIL only), the declared next step is the model-first design (a
  grasp-outcome head), with its own preregistration.
- Collection and Stage C (R27.29–R27.31, 2026-10-12, at `53f0b72`): play-v2g 3 199 stored
  (train 2 879), play-v1p 3 197 (train 2 877); row **C-PASS**: C1 2.383 (bar 2.0; reported 95 %
  2.080–2.736), C2 1.693 (bar 1.5; 1.560–1.840), C3 1.045 (0.75–1.25; 0.893–1.228). Evidence
  `~/develop/emai/evidence/task090-c/`. Featurisation and Stage T wait until this record is reviewed
  and merged (R27.30).
- Featurisation, Stage T, E and A′ (R27.32–R27.35, 2026-10-12, at `53f0b72`): F-DONE, T-DONE,
  E-PASS, then **A′-FAIL**. A1 passed for every model; A2 close@R1 fell below 48 / 64 for
  G-v1p-90100 (1), G-v1p-90102 (20) and G-v2g-90100 (0), so M-v1p failed on two seeds. The task
  escalates with no clause; Stage BD and BS do not run; the escalation's ruling is pending. Evidence
  `~/develop/emai/evidence/task090-{f,t,e,a}/`; about 6.7 GPU hours through A′.
