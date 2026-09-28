---
id: TASK-068
aliases:
- TASK-068
title: Scripted expert that rests the apple on the plate, with an at-rest success metric
slug: scripted-expert-that-rests-the-apple-on-the-plate-with-an-at-rest-success-metric
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- scripted-expert
- evaluation
sprint: ''
depends_on:
- "[[TASK-067]]"
due_date: ''
created: 2026-09-28
updated: 2026-09-28
---


# Scripted expert that rests the apple on the plate, with an at-rest success metric

## Description

TASK-067 stopped before training a policy because the privileged scripted expert (the
demonstration source) does not leave the apple on the plate: it drops the apple about 15.5 cm,
the apple rolls to the rim, and at rest only 4/32 (plate exact) and 1/32 (1.0 cm plate error)
ended inside the 4 cm radius. The default scorer latches transient crossings of the disc. The
one place-then-open redesign never got the palm low enough (lowest 0.113 m against 0.035 m),
for a reason TASK-067 did not identify. See `docs/DECISIONS.md` (2026-09-28, TASK-067).

TASK-068 fixes the expert first, and measures it honestly:
1. a separately named **at-rest** success check (the default `AppleToPlateTask` scorer, its 4 cm
   radius and past results are not changed);
2. a **descent diagnosis** (no gate): why the palm could not reach a low place pose;
3. **development** of a privileged scripted expert on a declared development seed range, with
   every design and its counts logged;
4. a **preregistration** PR, then a **gated evaluation** on fresh seeds, only on the pre-run
   reviewer's reported GO.

This is engineering on a privileged scripted controller, not a learned result. **Learned
Apple→Plate is still 0 successes.**

Protocol and log: `docs/experiments/apple_resting_expert_v1.md`.

## Frozen (owner brief)
- No change to the task, the plate geometry, the scorer radius or the 28/32 bar.
- No learned policy. The test split is not decoded.
- Conditions: plate exact, plate error 1.0 cm, and 1.5 cm reported only, with TASK-067's C0
  plate perturbation (`first_policy_runtime.perturbed_truth`).
- Gate (for the gated run): at rest ≥ 28/32 at plate exact AND ≥ 28/32 at 1.0 cm; latched-scorer
  counts reported beside.

## Seeds
- **Development: 50000–50099** (`resting_expert.DEV_SEEDS`), plate-error directions from
  `default_rng(6830)`. Fixed in code before the first logged development run; the repository
  search of 2026-09-28 found 50000–50999 unused (the suggested 47000–47099 is inside
  `apple_look_v1_corpus`, 47000–47199, so it is not free).
- **Gated:** not yet declared; it will be fresh, disjoint from the development range and from
  every spent or reserved range, and fixed in the preregistration PR.

## Acceptance Criteria
- [x] At-rest check implemented under its own name (`task.apple_at_rest`,
      `task.AppleAtRestCheck`, `AtRestThresholds` version `apple_at_rest_v0`), with tests; the
      default scorer is unchanged.
- [x] Descent diagnosis run and reported from logs and code only.
- [x] Development log of every design with its development counts.
- [ ] Preregistration PR (expert design and revision, metric and thresholds, conditions, seeds,
      budget) merged on an independent reviewer's reported APPROVE.
- [ ] Gated evaluation run only on the pre-run reviewer's reported GO; results PR merged on a
      reported APPROVE; failures reported plainly.

## Notes
- 2026-09-28: card opened by the TASK-068 agent on the coordinator's brief.
- 2026-09-28: at-rest check, expert code, development harness and descent diagnosis written;
  development designs d1–d12 and the collector baseline run. **Every design is 0 at rest on
  development seeds** (see the protocol §5). Blocker reported to the owner before any
  preregistration; see the protocol §6.

%% mc-links: [[TASK-067]] %%
