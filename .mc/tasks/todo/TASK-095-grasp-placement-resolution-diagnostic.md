---
id: TASK-095
aliases:
- TASK-095
title: 'Grasp placement-resolution diagnostic: which palm offsets cost the scripted follower its lifts, and do the frozen cost and G rank an exact placement above a costly offset (measurement only, no controller)'
slug: grasp-placement-resolution-diagnostic
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
- "[[TASK-094]]"
due_date: ''
created: 2026-10-15
updated: 2026-10-15
---


# Grasp placement-resolution diagnostic: which palm offsets cost the scripted follower its lifts, and do the frozen cost and G rank an exact placement above a costly offset (measurement only, no controller)

## Description

**DRAFT.** Chosen by R27.205 (DECISIONS (bn), R27.200–R27.208; design note
[grasp_line_next_v2.md](../../../docs/experiments/grasp_line_next_v2.md)), after TASK-094's
K-NULL paused the grasp line (R27.197). The pause stands for controllers. This task lifts it for
a measurement only (R27.206).

**Why.** The evidence has these qualifiers: one run per stage, simulation only, pooled for
orientation, associations read after the results.
- **Placement, not closure, decided the lift in K.** TASK-094's K found that, at the follower's
  exact pose, partial closure lost less than 0.15 of the lifts (K-NULL). A 1.2 cm +y palm offset
  cut the follower from 88 to 31 of 128.
- **The WM arms rarely establish a grasp.** A CPU read of the TASK-089, TASK-091 and TASK-093
  Stage BS reports (R27.200–R27.201) finds an established grasp at the end of the close on
  194 / 1 848 WM episodes, against 236 / 320 for ik.
- **Their held placement error is about twice the dose.** The WM palm sits a median 2.62 cm from
  the grasp pose while closing (ik 1.23 cm, which includes its press). That is about twice the
  dose that cost the follower 0.445 in +y.
- **The earlier 19 / 179 figure** conditioned on a transient minimum distance, not on a held
  placement.
- **The pose term is nearly blind to it.** The frozen cost's pose term changes by a median 0.0065
  for a +y 1.2 cm offset (R27.203; arithmetic, a best case for G). Whether the latent term
  resolves it is not measured.

**The question.** Can the frozen step-averaged cost resolve a placement error that costs lifts?
If it can, does G's prediction resolve it?

**Parts.** Each runs once, on fresh seeds that the preregistration checks.

- **K′, the placement dose (scripted, no model).**
  - TASK-094's K-1.0 follower on 128 fresh paired resets.
  - Rigid palm offsets {±x, ±y} × {0.5, 1.2, 2.5 cm} (pelvis frame, all four subgoals), and a
    yaw offset of ±0.17 rad.
  - It reports the tolerance envelope, and checks that the primary offset, +y 1.2 cm (fixed by
    R27.205), still costs at least 0.15.
- **R, the ranking (offline, frozen G, no training).**
  - Fresh roots at the start of the close (TASK-089's R1).
  - The follower's close-and-lift with the palm at the goal and at each offset, simulated to the
    true outcome.
  - Each branch is scored by TASK-089's frozen step-averaged cost from G-87100–87102's
    predictions (G-87102 `last_two_triggered`) and from real frames and measured states (the
    ceiling), over 8 commands (16 reported), with the latent and pose terms reported apart.
  - The read is on discordant roots: the share on which exact ranks first.
- **S′, the search (offline, reported).** TASK-089's frozen planner at the same roots: its miss
  against the exact branch, and G's predicted palm displacement for its plan.

**Rows** (declared with bars and margins in the preregistration; routing only, no claim):
- R-VOID;
- R-UNREAD;
- **R-COST**: the ceiling ranks exact first on fewer than 0.75 of the discordant roots. Routes to
  C′ (a Cartesian palm term, under R26.33) or W (a sensor or data change, open under R27.71);
- **R-MODEL**: the ceiling does, and G does not, on two of three seeds. Routes to W or a model
  change;
- **R-RESOLVED**: both do. Routes to F (full-embodiment projection) or a search variant, under
  R26.33.

Each routes to its own ruling. The preregistration states which rows are put forward as R26.33
evidence of a different kind. **Disclosed expectation:** R-COST or R-MODEL is the likelier row.

**Limits.**
- No world-model closed loop, no training and no controller.
- No planner, cost, horizon or action-parametrisation variant is preregistered for grasping, so
  R26.33, R27.71 and R27.149 are not engaged.
- The scripted probes are not learned results.
- Phase 4 is not admitted, and any Phase 4 precursor ruling stays the owner's.
- Not Apple→Plate; R7 does not change.

**Budget:** about 0.5 GPU h (ceiling 1.0) plus CPU, with every GPU job under
`gpu_run.sh --wait --board`.

## Acceptance criteria

- [ ] DRAFT preregistration (`docs/experiments/grasp_placement_resolution_v1.md`), reviewed
  independently and merged. It fixes the seeds (checked by the seed check), the offsets, the
  roots, the windows, the bars and margins and the row order before anything runs.
- [ ] Stage 0: opt-in code with no frozen file edited (TASK-094's runners and summarizer stay
  unedited; a new summarizer carries the determinism-report size fix of R27.195), tests,
  degeneracy checks on development roots and a debug smoke. The protocol is frozen at the Stage 0
  merge.
- [ ] K′, R and S′ recorded at the frozen revision from a clean tree, with evidence in
  `~/develop/emai/evidence/task095-*`.
- [ ] Row recorded with a next-step ruling; R7 unchanged.

## Notes

- 2026-10-15, created as DRAFT by R27.205 (branch `docs/grasp-next-direction`). The offline read
  behind it is in `~/develop/emai/evidence/grasp-next-v2/`; it used the CPU only (run reports,
  MuJoCo forward kinematics with no simulation step, and G-87100's stored state moments).
