---
id: TASK-059
aliases:
- TASK-059
title: Measure whether the reset frame carries the apple position and the expert's first command
slug: measure-whether-the-reset-frame-carries-the-apple-position-and-the-expert-s-firs
status: done
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
- [x] The preregistration and manifest merge on an independent reviewer's reported APPROVE, before
      any readout is fitted (PR 1).
- [x] Probe code merges with tests that exercise every guard in both directions (split, hashes,
      render, expert, prior) and the resolution-equivalence checker (reported, not a guard), on a
      reported APPROVE (PR 2).
- [x] The gated run starts on the pre-run reviewer's reported verdict, with a clean tree and all
      guards passing.
- [x] The results document states the outcome row, the numbers with CIs against their baselines,
      and every negative result, and recommends (does not choose) the next task (PR 3).
- [x] The test split, cohort D and cohort C are not touched.

## Notes
- PR 1 also closes TASK-057 (results PR #47, `e840968`).

## Results (2026-09-25)

**Outcome O-OCC-NONE.** No decisional source (E0, A3, random, raw-112) succeeds, either on all
190 roots or on the 79 visible ones. On the visible stratum no decisional 112 px source beats the
visibility-aware prior either:
- raw-112 reaches a ratio of 0.964 [0.751, 1.207];
- E0 reaches 2.067 cm, worse than the prior's 1.720 cm.

So occlusion explains the occluded roots (111/190, behind `right_wrist_yaw_link`) but not the
failure as a whole. E0 is worse than raw-112 (+0.296 cm [0.110, 0.475]) and better than the
random encoder (−0.212 cm [−0.383, −0.045]).

- **PRs.** Prereg #48 (`781664f`); probe #49 (`ccb8fd7`); run-1 void fix #52 (`b209740`). Each
  merged on an independent reviewer's reported APPROVE, with green CI.
- **Runs.** Run-1 was VOID: it crashed at serialisation and wrote no output. Run-2 was the
  single permitted repeat: clean tree at `b209740`, CPU, 619 s. Its report sha256 is `9bff6c46…aac3`.
- **Documents.** Results: `docs/experiments/apple_info_ceiling_v1_results.md`. Manifest:
  `benchmarks/manifests/apple-info-ceiling-v1-results.json`.
- **Recommended next task (the owner chooses).** A joint preregistered re-probe with this probe
  frozen. It would cross two axes: look-first pose or motion (the apple is visible by step 9 on
  the expert path), and apple pixel count (`overview` camera, higher-resolution or closer
  camera). The named risk is that look-first alone may not suffice.
%% mc-links: [[TASK-057]] %%
