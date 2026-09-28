---
id: TASK-069
aliases:
- TASK-069
title: 'Development-only feasibility scan: which minimal task or embodiment change lets a privileged scripted expert rest the apple on the plate (toward apple-to-plate-v2)'
slug: development-only-feasibility-scan-which-minimal-task-or-embodiment-change-lets-a
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- scripted-expert
- task-definition
sprint: ''
depends_on:
- "[[TASK-068]]"
due_date: ''
created: 2026-09-28
updated: 2026-09-28
---


# Development-only feasibility scan: which minimal task or embodiment change lets a privileged scripted expert rest the apple on the plate (toward apple-to-plate-v2)

## Description

Owner ruling R11 (verbatim in `docs/experiments/apple_resting_expert_v1.md` §8) closed TASK-068.
Under the frozen v1 task, no privileged scripted expert design tried rested the apple on the
plate: 0 at rest in 269 completed development attempts. The logs give two reasons:
- the place pose is beyond the fixed-pelvis arm's reach;
- the apple, a sphere with no rolling resistance, keeps rolling.

TASK-069 asks which minimal task or embodiment change makes a privileged scripted expert rest the
apple on the plate. **It is a development-only feasibility scan: no gate, nothing frozen.** Its
output is a recommendation for an `apple-to-plate-v2` task definition, for the owner to rule on.
The v1 task, its history and the 0/150 MVP benchmark stay untouched and separately named.

The cells:
- **(b3)** apple rolling and torsional friction (condim 6, or other physically plausible values),
  on its own;
- **(b2)** plates sampled within the reachable set-down region, measured with the reach tool, on
  its own;
- **b2 + b3** together;
- **(b1)** feasibility only, answered from code and models, with no expert work: can the waist
  joints be added to the IK while keeping the frozen 14-D `ee_delta_grasp_v0` schema? What does
  the MJCF actuate? Would the extra reach bring the v1 plate region within place distance?
- **(b4)** is excluded, because it needs a new action contract.

For each cell, the scan reports:
- the at-rest count (`apple_at_rest_v0`) and the latched count, at plate exact and at 1.0 cm plate
  error;
- the stops on the joint-velocity guard;
- the landing speed.

This is privileged scripted engineering, not a learned result. **Learned Apple→Plate is still 0
successes.**

## Seeds
- **Development: 50100–50199**, with plate-error directions from `default_rng(6840)`. The range
  is declared here, before any TASK-069 episode runs.
- It is disjoint from:
  - TASK-068's development range, 50000–50099;
  - every range in `first_policy.FORBIDDEN_RANGES`;
  - the 46000–46999 block, and cohorts C and D.
- The repository search of 2026-09-28 (TASK-068 protocol §2) found no other seed use in
  50000–59999.
- A preregistered v2 expert gate, if the owner rules one, gets fresh seeds of its own.

## Acceptance Criteria
- [x] Set-down reach region measured with the reach tool; b1 answered from code and models.
- [x] Cells b3, b2 and b2+b3 scanned on the development seeds, each with at-rest and latched
      counts at plate exact and at 1.0 cm, guard stops and landing speed.
- [x] A recommendation for `apple-to-plate-v2`, with what the logs show kept apart from what is
      inferred. Reviewed (PR #91). The owner rules on it (R12).

## Notes
- 2026-09-28: card opened under owner ruling R11 §4.
- 2026-09-28: scan run (`docs/experiments/apple_to_plate_v2_feasibility.md`). On development
  seeds, with TASK-068's d12 release: b3b (apple condim 6, rolling 0.003 m) rests 64/64 at plate
  exact and 64/64 at 1.0 cm; b3a (the scene's declared 0.001 m, enabled) 64/64 and 55/64; v1
  0/128. b2 alone 0 at rest, with 7–32 guard stops per cell; b2+b3 is worse than b3 alone. b1:
  waist pitch reaches the v1 plate region kinematically, near its limit, and moves the camera.
  Recommendation: v2 = v1 + b3b. Awaiting review, then the owner's ruling. Privileged scripted
  engineering; learned Apple→Plate is still 0 successes.
- 2026-09-28: PR #91 merged (rebase merge; `552ebd2`…`d545ee0` on `main`). Owner ruling R12 (verbatim
  in `docs/experiments/apple_to_plate_v2_expert.md` §0) set v2 = v1 + apple contact condim 6 with
  the scene's declared friction `1 / 0.01 / 0.001`. It did not take the recommended b3b values and
  rejected b2 and b1. TASK-069 is done; v2 continued in TASK-070. Learned Apple→Plate is still 0
  successes.

%% mc-links: [[TASK-068]] %%
