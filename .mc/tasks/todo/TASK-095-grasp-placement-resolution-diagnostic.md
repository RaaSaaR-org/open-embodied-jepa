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

**Status: on hold — owner paused 2026-10-15; not confirmed.** The owner paused the project
instead of confirming this task (DECISIONS 2026-10-15 (bo), R27.209–R27.215). It stays in todo;
nothing for it has been written or run. Resume from [RESUME.md](../../../docs/RESUME.md); the
owner's confirmation is still the first acceptance criterion.

**DRAFT; recommended to the owner, not yet confirmed.** Recommended by R27.205 (DECISIONS (bn),
R27.200–R27.208; design note
[grasp_line_next_v2.md](../../../docs/experiments/grasp_line_next_v2.md)), after TASK-094's
K-NULL paused the grasp line (R27.197). The TASK-094 card reserves a pause ruling to the owner,
so this task **waits for the owner's confirmation before any preregistration or GPU work**
(R27.206). If confirmed, it lifts the pause for this measurement only; the pause stands for
controllers. On 2026-10-15 the owner was told in chat that the grasp line is paused and that D
and X are the owner's to decide.

**Why.** The evidence has these qualifiers: one run per stage, simulation only, pooled for
orientation, associations read after the results.
- **Placement, not closure, decided the lift in K.** TASK-094's K found that, at the follower's
  exact pose, partial closure lost less than 0.15 of the lifts (K-NULL). A 1.2 cm +y palm offset
  cut the follower from 88 to 31 of 128.
- **The WM arms rarely establish a grasp.** A CPU read of the TASK-089, TASK-091 and TASK-093
  Stage BS reports (R27.200–R27.201) finds an established grasp at the end of the close on
  194 / 1 848 WM episodes, against 236 / 320 for ik.
- **Their held placement error is about twice the dose.** The WM palm sits a median 2.62 cm from
  the grasp pose while closing (from the first touch; the whole close phase where there was none;
  ik 1.23 cm, which includes its press). That is about twice the size of the dose that cost the
  follower 0.445 in +y; its direction, including any vertical part, is not measured.
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
  - Rigid palm offsets {±x, ±y, ±z} × {0.5, 1.2, 2.5 cm} (pelvis frame, all four subgoals), and
    a yaw offset of ±0.17 rad: 21 arms. z is dosed because the WM arms' held distance may be
    partly vertical (note §2.2).
  - It reports the tolerance envelope, and checks that the primary offset, +y 1.2 cm (fixed by
    R27.205), still costs at least 0.15. That offset was chosen after K as its costliest, and it
    is the direction in which the pose term is least sensitive, so the choice favours discordant
    roots and R-COST; R's ranking is reported at every K′ offset beside the primary one.
- **R, the ranking (offline, frozen G, no training).**
  - Fresh roots at the start of the close (TASK-089's R1).
  - The follower's close-and-lift with the palm at the goal and at each offset, simulated to the
    true outcome.
  - Each branch is scored by TASK-089's frozen step-averaged cost from G-87100–87102's
    predictions (G-87102 `last_two_triggered`) and from real frames and measured states (the
    ceiling), over 8 commands (16 reported), with the latent and pose terms reported apart.
  - The read is on discordant roots (the exact branch lifts, the +y 1.2 cm branch does not): the
    share on which the exact branch's cost is below the +y 1.2 cm branch's.
- **S′, the search (offline, reported).** TASK-089's frozen planner at the same roots: its miss
  against the exact branch, and G's predicted palm displacement for its plan.
- **V, the command replay (CPU only, no model, no GPU; reported, diagnostic, not a row input).**
  - Replays the logged 14-D `applied` commands of the TASK-089, TASK-091 and TASK-093 Stage BS
    WM episodes open loop from their recorded reset seeds, and logs the palm's direction and
    orientation error against the grasp subgoal and the object's pose through the close.
  - Replay fidelity is checked first: each replayed episode must reproduce every logged per-step
    `distances`, `contact` and `rise` value exactly (and its success flag) at the run's recorded
    revision; non-reproducing episodes are excluded and counted, and below a share the
    preregistration fixes, V is unreadable.
  - It gives roots at the WM arms' own states (a reported second root type for R).

**Rows** (declared with bars and margins in the preregistration; routing only, no claim):
- R-VOID;
- R-UNREAD (fewer discordant roots than TASK-094 §14's n_d minimum of 16, or a bar the
  preregistration declares);
- **R-COST**: the ceiling ranks the exact branch above the primary offset branch on fewer than
  0.75 of the discordant roots. Routes to C′ (a Cartesian palm term, under R26.33) or W (a
  sensor or data change, open under R27.71);
- **R-MODEL**: the ceiling does, and G does not, on two of three seeds (G's per-seed bar declared
  in the preregistration). Routes to W or a model
  change;
- **R-RESOLVED**: both do. Routes to F (full-embodiment projection) or a search variant, under
  R26.33.

Each routes to its own ruling. The preregistration states which rows are put forward as R26.33
evidence of a different kind. **Disclosed expectation:** R-COST or R-MODEL is the likelier row.

**Limits.**
- No world-model closed loop, no training and no controller.
- No planner, cost, horizon or action-parametrisation variant is preregistered for grasping, so
  R26.33, R27.71 and R27.149 are not engaged.
- The scripted probes are not learned results. No K′ or R episode enters a training corpus
  (R27.71).
- At R1 an offset branch moves the palm by v during the close (not K′'s dose), and the 8-command
  window holds no lift and may cover only part of v; the preregistration states the displacement
  reached by window steps 4 and 8 and reports a window-end-only cost. R tests whether the cost
  resolves the placement against the goal, not whether it predicts the lift.
- Phase 4 is not admitted, and any Phase 4 precursor ruling stays the owner's.
- Not Apple→Plate; R7 does not change.

**Budget:** about 0.6 GPU h (K′'s 21 arms about 0.2 GPU h; ceiling 1.0) plus CPU (V runs on the
CPU only), with every GPU job under `gpu_run.sh --wait --board`.

## Acceptance criteria

- [ ] The owner confirms TASK-095 (lifting the pause for this measurement only) before any
  preregistration or GPU work.
- [ ] DRAFT preregistration (`docs/experiments/grasp_placement_resolution_v1.md`), reviewed
  independently and merged. It fixes the seeds (checked by the seed check), the offsets, the
  roots, the windows, the bars and margins and the row order before anything runs.
- [ ] Stage 0: opt-in code with no frozen file edited (TASK-094's runners and summarizer stay
  unedited; a new summarizer carries the determinism-report size fix of R27.195), tests,
  degeneracy checks on development roots and a debug smoke. The protocol is frozen at the Stage 0
  merge.
- [ ] K′, R, S′ and V recorded at the frozen revision from a clean tree, with evidence in
  `~/develop/emai/evidence/task095-*`.
- [ ] Row recorded with a next-step ruling; R7 unchanged.

## Notes

- 2026-10-15, created as DRAFT by R27.205 (branch `docs/grasp-next-direction`). The offline read
  behind it is in `~/develop/emai/evidence/grasp-next-v2/`; it used the CPU only (run reports,
  MuJoCo forward kinematics with no simulation step, and G-87100's stored state moments).
- 2026-10-15, review of #229: recommended to the owner rather than ruled (no citable owner
  request; the TASK-094 card reserves a pause ruling to the owner); ±z added to K′ (21 arms);
  the primary offset's post-K selection disclosed; R's roots and window limits stated; V, a
  CPU-only command replay, added as a reported part.
- 2026-10-15, on hold: the owner paused the project (DECISIONS (bo), R27.209–R27.215) without
  confirming this task. No preregistration, code or GPU work exists for it.
