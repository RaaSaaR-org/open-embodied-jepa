---
id: TASK-059
aliases:
- TASK-059
title: Measure whether the reset frame carries the apple position and the expert's first command
slug: measure-whether-the-reset-frame-carries-the-apple-position-and-the-expert-s-firs
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- perception
- evaluation
sprint: ''
depends_on:
- "[[TASK-057]]"
due_date: ''
created: 2026-09-25
updated: 2026-09-25
---



# Measure whether the reset frame carries the apple position and the expert's first command

## Description

TASK-057 ended in Outcome X, so the abandonment clause of `apple_policy_v1.md` §7 fired. No new
control formulation may be preregistered on this corpus and this 112 px camera, and the next task
must be a perception or data task. **Learned Apple→Plate is still 0 successes.**

At reset the robot state is identical on every reset, so only the image can locate the apple.
Every behaviour-cloning arm commands a step-0 `right_dx` that does not depend on the reset. The
expert's step-0 `right_dx` changes sign with the apple's position.

This task is a preregistered **information-ceiling probe**. It asks whether a readout can recover
the apple position and the expert's step-0 command (above all the sign and size of dx) from the
**reset frame**. It compares five feature sources:
- the frozen E0 encoder (as arm A2 uses it);
- the A3 fine-tuned encoder;
- a random-init encoder;
- raw 112 px pixels;
- a 448 px native render.

It trains no controller, opens no cohort, and never decodes the test split.

Protocol: `docs/experiments/apple_info_ceiling_v1.md`.
Manifest: `benchmarks/manifests/apple-info-ceiling-v1.json`.

**Pre-freeze calibration findings.** None of these fits a readout.
- A 112 px re-render is byte-identical to the stored frame on 190/190 train+val roots.
- The apple is **occluded at reset on 111/190 roots at 112 px** (106/190 at 448 px). Where it is
  visible, it covers at most 21 px.
- A downsampled 448 px render does **not** reproduce the stored frame to any tolerance that tells
  one reset from another. So the native source is reported only; this is the owner's ruling of
  2026-09-25.

## Acceptance Criteria
- [ ] The preregistration and manifest merge on an independent reviewer's reported APPROVE, before
      any readout is fitted (PR 1).
- [ ] Probe code merges with tests that exercise every guard in both directions (split, hashes,
      render, expert, prior) and the resolution-equivalence checker (reported, not a guard), on a
      reported APPROVE (PR 2).
- [ ] The gated run starts on the pre-run reviewer's reported verdict, with a clean tree and all
      guards passing.
- [ ] The results document states the outcome row, the numbers with CIs against their baselines,
      and every negative result, and recommends (does not choose) the next task (PR 3).
- [ ] The test split, cohort D and cohort C are not touched.

## Notes
- PR 1 also closes TASK-057 (results PR #47, `e840968`).
%% mc-links: [[TASK-057]] %%
