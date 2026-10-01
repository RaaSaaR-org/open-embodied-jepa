---
id: TASK-074
aliases:
- TASK-074
title: 'LeWM planner v2: LeWM chooses the put-down spot after a far plate move'
slug: lewm-planner-v2-lewm-chooses-the-put-down-spot-after-a-far-plate-move
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- planner
- preregistration
- inconclusive
sprint: ''
depends_on:
- "[[TASK-066]]"
- "[[TASK-072]]"
- "[[TASK-073]]"
due_date: ''
created: 2026-09-30
updated: 2026-10-01
---


# LeWM planner v2: LeWM chooses the put-down spot after a far plate move

## Description

TASK-074 is the owner's D7 fallback after TASK-073 ended S-NO-CONDITION (`apple_lewm_planner_v2`,
`lewm-planner`).

**The formulation.** On `apple-to-plate-v2` the plate moves 9 or 12 cm to the robot's right (−y)
at step 300. This is a simulation-only diagnostic condition. P-3 does the pick. From step 405 a
LeWM predictor over frozen DINOv2 4 × 4 patch tokens (TASK-066's latent, h = 16) chooses where
e9's scripted place primitive puts the apple:
- the candidates are 74 put-down spots per decision (coarse, then fine);
- the cost is the predicted apple-minus-plate offset against e9's release offset.

**Credit** is only against:
- P-3 given the true plate (P-truth);
- the action-blind (L-N), scene-blind (L-shuf) and random (L-rand) choosers;
- and within 6/64 of a readout-only twin (H-twin).

P-far, a retrained BC policy, is a reported control.

**The owner's decisions of 2026-09-30, verbatim:**
- D1 "Beat P-3, tie readout (Recommended)";
- D2 "Put-down spots (Recommended)";
- D3+D4 "Right-side move, place only (Recommended)";
- D5 "Yes, add it (Recommended)".

**The owner's explicit delegation of 2026-09-30, verbatim:** "do the work without me - if you
have decidions, choose your recommandation. do the work in subagents, use this chat just for
updates. use subagents and workflows. goal is to continue working and try to get a real LeWM for
the Unitree G1 without my help".

Every other ruling is therefore delegated, and is labelled "decided by Claude under owner
delegation, 2026-09-30". This includes the gated authorisation:
- the gated stage may run on an authorisation record signed by the task owner, or by Claude under
  this delegation;
- either way, only after a pre-run reviewer's reported GO;
- the main session posts a notice in chat before each gated run.

**The design probes** used development seeds 54700–54999, which are now spent. The bars were set
after seeing them, and the protocol discloses this.

**Stages,** each a separate reviewed step:
1. the Stage 0 preregistration PR (this card's first PR);
2. K1, the condition gate (development);
3. the `apple-far-shift-v2` corpus;
4. W/N training, P-far, ranking and the offline decision (O1–O5);
5. the D3 development closed loop;
6. gated S/U, only on a separate owner or delegated authorisation record, after a reported
   reviewer GO, with a chat notice before each gated run;
7. the results PR.

Learned Apple→Plate on the frozen benchmark is still 0 successes. An L-plan success would be
"P-3's learned pick plus a LeWM planner choosing where a scripted place primitive puts the
apple", not LeWM driving the whole episode, and not the v1 benchmark. No control line is primary.

## Acceptance Criteria
- [x] Stage 0: protocol `docs/experiments/apple_lewm_planner_v2.md` and manifest
      `benchmarks/manifests/apple-lewm-planner-v2.json`, with the frozen block, the stored cohort
      values and digests, the corpus plan digest and the pins.
- [x] Stage 0 code: the modules, the runner and the tests. The #108 follow-ups are fixed first
      (#111) and used through `run_guards`.
- [x] Stage 0 smokes on 54650–54699 only, with the train stage's memory measured at full corpus
      scale.
- [x] Stage 0 merged on an independent reviewer's reported APPROVE.
- [x] K1 on a fresh pre-run reviewer's reported GO, with the coordinator told first.
- [ ] Corpus, training, P-far, ranking and the offline decision, each on its own reported GO.
      Corpus done (CORPUS-SEALED); training escalated twice (run-3 ESCALATE-BUDGET-LAST-TWO);
      P-far, ranking and the offline decision never ran (TASK-074 closed INCONCLUSIVE).
- [ ] D3, on its own reported GO. Not run (closed INCONCLUSIVE before it).
- [ ] Gated S/U only after the owner or delegated authorisation record is in the manifest,
      validated by `check_authorisation`, after a reported reviewer GO and a chat notice. Not run.
- [ ] The results PR, reviewed with every restated number checked.

## Notes
- 2026-09-30: #113 (preregistration) merged as `9d9b03c`. K1 ran on its GO: **K1-PASS** at
  9 cm (B-oracle-shift 32, H-handover 32, P-stale 0, P-truth 19), report `25701293…`. The
  corpus ran on its GO: **CORPUS-SEALED**, manifest `fe7ab915…`, report `16e6417e…`.
- 2026-09-30: train run-1 ended **V** on G-memory (12.04 > 12.00 GiB) after its boundary,
  report `78137de0…`. Fix and O2 encoded-readout finding in Addendum A1 (#114, `f52c905`).
- 2026-10-01: train run-2 ended **ESCALATE-BUDGET** (wanted 80 000 > cap 60 000), report
  `f5974cd2…`. Addendum A2 (#115, `5e53ef3`) raised the train cap to 80 000.
- 2026-10-01: train run-3 ended **ESCALATE-BUDGET-LAST-TWO** (N-7412 selected 80 000), report
  `25cb1bf3…`. Ruling (decided by Claude under owner delegation, 2026-10-01): no further raise
  or train run; TASK-074 closes **INCONCLUSIVE** ("close without the clause"). The §7 clause
  does not fire; its scope is not refuted and not closed. The binding constraint is the
  observation/readout (encoded readout 2.872 cm against O2's uncalibrated 1.0 cm bar).
  Results: `docs/experiments/apple_lewm_planner_v2_results.md`. No LeWM controller ran in
  closed loop. Learned Apple→Plate is still 0 successes. The next task is a measurement-first
  observation-ceiling study, to be preregistered separately.
