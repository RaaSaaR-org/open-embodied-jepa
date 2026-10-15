---
id: TASK-094
aliases:
- TASK-094
title: 'Grasp-closure locus diagnostic: does the partial closure the world-model arms command cause the grasp failures, and is it the goal and cost, the model or the search (measurement only, no controller)'
slug: grasp-closure-locus-diagnostic
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- diagnostic
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-093]]"
due_date: ''
created: 2026-10-15
updated: 2026-10-15
---


# Grasp-closure locus diagnostic: does the partial closure the world-model arms command cause the grasp failures, and is it the goal and cost, the model or the search (measurement only, no controller)

## Description

**DRAFT.** Chosen by R27.156 (DECISIONS (bh); design note
[grasp_next_step_v1.md](../../../docs/experiments/grasp_next_step_v1.md)). The preregistration is
not yet written.

**Why.** An offline read of the Stage BS reports of TASK-089, TASK-091 and TASK-093 (R27.152)
has these qualifiers: one run per stage, simulation only, pooled, an association, and read after
the results. It finds:

- **Contact without a hold.** The world-model arms touch the target in the close phase on
  982 / 1 856 episodes, but succeed on only 105 of those. ik succeeds on 220 / 302.
- **The closure they command.** They command a partial closure (median lift-phase mean 0.53–0.73
  per run, against ik's 1.00). It sits near the demonstrator's *measured*, object-blocked hand
  (median synergy about 0.68), not near its +1 command.
- **Success by commanded closure.** Given contact, successes are 21 / 739 below 0.7 and 17 / 23
  at 0.9 or above (commanded before the object rose).
- **The corpus.** Partial closures at contact are nearly absent from `play-v1` (97.1 % of contact
  commands are ≥ 0.8).

**The question.** Is that closure the physical cause of the failures? If it is, does the failure
sit in the goal and cost (b), in G (a) or in the search (c)?

**Parts.** Each runs once, on fresh seeds that the preregistration checks.

- **K, the physics dose (scripted, no model).**
  - TASK-088's IK follower with its closing command held at +1.0, 0.9, 0.8, 0.7 and the close
    subgoal's measured synergy.
  - +1.0 and the measured synergy are also run with a palm-target offset of about 1.2 cm.
- **M, the ranking (offline).**
  - At fresh roots at the end of the demonstrator's close, scripted branches follow the lift at
    each hold level and are simulated to their true outcome.
  - Each branch is scored by TASK-089's frozen step-averaged cost, both from G-87100–87102's
    predictions and from the branch's real frames and states (the ceiling).
- **S, the search (offline).** TASK-089's frozen planner at the same roots: its chosen hold level,
  and G's cost of that plan against the squeeze branch.

**Rows** (declared with bars and margins in the preregistration; routing only, no claim):
- K-NULL means the grasp line pauses (R27.157).
- K-CAUSAL is followed by one of:
  - L-COST (b);
  - L-MODEL (a);
  - L-SEARCH (c);
  - L-MIXED.

  Each routes to a ruling.

**Limits.**
- No world-model closed loop and no training.
- No planner, cost, horizon or action-parametrisation variant is preregistered for grasping, so
  R26.33, R27.71 and R27.149 are not engaged.
- The scripted probes are not learned results.
- Not Apple→Plate; R7 does not change.

**Budget:** about 0.5 GPU h (ceiling 1.0) plus CPU, with every GPU job under
`gpu_run.sh --wait --board`.

## Acceptance criteria

- [ ] DRAFT preregistration (`docs/experiments/grasp_closure_locus_v1.md`), reviewed
  independently and merged. It fixes the seeds (checked by the seed check), the hold levels, the
  palm offset, the roots, the bars and margins and the row order before anything runs.
- [ ] Stage 0: opt-in code with no frozen file edited, tests, the degeneracy checks of M's
  branches on development roots, and a debug smoke. The protocol is frozen at the Stage 0 merge.
- [ ] K, M and S recorded at the frozen revision from a clean tree, with evidence in
  `~/develop/emai/evidence/task094-*`.
- [ ] Row recorded with a next-step ruling (K-NULL pauses the grasp line); R7 unchanged.

## Notes

- 2026-10-15, created as DRAFT by R27.156 (branch `docs/grasp-next-step-ruling`). The offline read
  behind it is in `~/develop/emai/evidence/grasp-next-step/`; it used the CPU only.
