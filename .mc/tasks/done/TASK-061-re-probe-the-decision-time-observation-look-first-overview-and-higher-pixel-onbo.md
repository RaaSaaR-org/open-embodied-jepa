---
id: TASK-061
aliases:
- TASK-061
title: 'Re-probe the decision-time observation: look-first, overview and higher-pixel onboard arms'
slug: re-probe-the-decision-time-observation-look-first-overview-and-higher-pixel-onbo
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
- "[[TASK-059]]"
due_date: ''
created: 2026-09-25
updated: 2026-09-26
---




# Re-probe the decision-time observation: look-first, overview and higher-pixel onboard arms

## Description

TASK-059 ended in **O-OCC-NONE**: the 112 px onboard reset frame does not give the apple
position or the expert's step-0 `right_dx` beyond a prior. The right wrist hides the apple on
111/190 resets, and even a visible apple (1–21 px) is not read beyond the visibility-aware prior.
**Learned Apple→Plate is still 0 successes.** The control line stays closed on this corpus and
camera (TASK-054 and TASK-057 clauses). This task preregisters no control formulation, trains no
policy and collects no corpus.

This task is a preregistered **observation re-probe**. It uses the TASK-059 probe unchanged:
the same readouts, CV, baselines, bars, bootstrap and strata. Four decisional hypotheses are
tested under Holm:

- **L-raw** and **L-E0**: a reset-independent 8-command look motion, then the 112 px onboard
  frame. The apple is visible on 190/190 roots after the look.
- **LH-raw**: the same look, with a 224 px onboard frame.
- **O-raw**: the `overview` camera at 112 px, at reset.

Reported only: onboard 224 px at reset, the 448↓ render path (unrounded and uint8), and the
TASK-059 anchor.

Protocol: `docs/experiments/apple_observation_reprobe_v1.md`.
Manifest: `benchmarks/manifests/apple-observation-reprobe-v1.json`.
Calibration: `scripts/calibrate_observation_reprobe.py`. It fits nothing.

## Acceptance Criteria
- [x] The preregistration, manifest and calibration script merge on an independent reviewer's
      reported APPROVE, with green CI, before any readout is fitted (PR 1).
- [x] The runner merges with tests (PR 2) that exercise, in both directions:
  - every guard, the render-path checks, Holm, the p-values and the spurious check;
  - the reset-independence of the look motion;
  - crash → `report.json` with outcome V and a `void_reason`;
  - non-finite → `null` plus `non_finite_fields`.

  It merges on a reported APPROVE with green CI.
- [x] The gated run starts only on the pre-run reviewer's reported verdict, with a clean tree.
- [x] The results document, the results manifest and the summarize script (PR 3) state:
  - the outcome row, and the numbers with CIs against their baselines;
  - every negative result;
  - a recommended (not chosen) next task.
- [x] The test split, cohort C and cohort D are untouched, and `exemption_spent` stays false.

## Notes
- Void rule: an early stop is V. One from-scratch repeat is allowed; a second void is
  INCONCLUSIVE.
## Results (2026-09-25)

**Outcome O-LOOK-RAW** (run-1, 44be6d6, 155 s CPU, all guards passed; report sha256 `289470f4…fa88`).
- L-raw, LH-raw and O-raw pass the unchanged TASK-059 bars, Holm and the apple-hidden check.
  L-raw reads 0.469 cm (ratio to B-occ 0.242 [0.211, 0.289]), with dx sign 180/190.
  On the 111 roots hidden at reset it reads 0.482 cm.
- L-E0 fails (1.273 cm, upper bound 0.779; 147/190) and is no better than a random-init encoder.
- The anchor reproduces TASK-059 bit-exactly. R-float differs by at most 5.7e-9: a cold-render
  artifact, with an owner ruling.
- Recommended next task (the owner chooses): encoder work on post-look frames, then a
  prereg-gated look-prefix corpus at 112 px.
- Learned Apple→Plate is still 0 successes. `exemption_spent` is false.
- PRs: #54 (prereg), #55 (runner), PR 3 (results).
- Results: `docs/experiments/apple_observation_reprobe_v1_results.md`.
%% mc-links: [[TASK-059]] %%
