---
id: TASK-075
aliases:
- TASK-075
title: 'Observation ceiling v2: measure each view''s readout ceiling against the place tolerance'
slug: observation-ceiling-v2-measure-each-views-readout-ceiling-against-the-place-tolerance
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- perception
- measurement
- preregistration
- lewm
- clause-fired
sprint: ''
depends_on:
- "[[TASK-074]]"
due_date: ''
created: 2026-10-01
updated: 2026-10-01
---


# Observation ceiling v2: measure each view's readout ceiling against the place tolerance

## Description

TASK-075 (`apple_obs_ceiling_v2`) is the measurement-first observation study that TASK-074's
results named as the next step (`apple_lewm_planner_v2_results.md` §7). TASK-074 closed
INCONCLUSIVE because a W-independent readout ceiling (R_off on pooled DINOv2 tokens, 2.872 cm on
O2's windows, against a clock prior's 3.123 cm) sat above an uncalibrated 1.0 cm bar.

**Chosen under the owner's delegation of 2026-09-30** (decided by Claude under owner delegation,
ruled 2026-10-01): candidate (d) of the design proposal. Views onboard 112 (reference), onboard
native 224, the palm-centred hand crop and overview 224; candidate (c)'s learning curve as one
factor; real-robot pretraining (b) deferred.

**Every bar is calibrated from measured quantities:**
- τ, the place tolerance: the largest planted target error at which P-3's pick plus e9's place
  still rests ≥ 28/32, under TASK-074's condition, measured on development seeds 55000–55031
  before the freeze;
- a view is admitted if c_V (the upper 95 % bound of R_off's median offset error) ≤ τ, it beats
  its random-init floor and a clock prior (ratio upper bound < 1.0), and the plate-hidden check
  is clean;
- rows OBS-ONBOARD, OBS-EXTRA, OBS-REPRESENTATION (escalate), OBS-NONE (clause);
- the next world-model task's O2-type bar must satisfy c_V ≤ bar ≤ τ.

**Corpus.** `apple-far-shift-v2`'s stored 112 px frames are the reference view; its 240 train and
30 val roots are re-simulated to render the new views (Stage 1, checked against the stored
arrays); its 30 test roots are excluded and never simulated, rendered or decoded.

This is a measurement study. Learned Apple→Plate on the frozen benchmark is still 0 successes;
no control claim is made and no control line is primary.

## Acceptance Criteria
- [x] Stage 0: protocol `docs/experiments/apple_obs_ceiling_v2.md`, manifest
      `benchmarks/manifests/apple-obs-ceiling-v2.json` (frozen block, τ record, pins), the
      modules, the runner and the tests.
- [x] Stage 0: τ measured on 55000–55031 before the freeze; the frozen-code smokes, including
      the full-scale readouts memory probe, on smoke seeds only.
- [x] Stage 0 merged on an independent reviewer's reported APPROVE.
- [x] Stage 1 (render) on its own reported GO: VIEWS-SEALED.
- [x] Stage 2 (readouts) on its own reported GO: a row (OBS-NONE).
- [ ] The results PR, reviewed with every restated number checked. Opened; review pending.

## Notes
- 2026-10-01: preregistration PR opened (Stage 0). `mc` was not available on the Linux PC, so
  `mc validate` and `mc index` were not run; the frontmatter follows TASK-074's card.
- 2026-10-01: #117 merged as `3434538`. Render ran once on its GO: **VIEWS-SEALED** (270/270
  roots' arrays equal, 0 frames outside the render rule), report `753e05f2…a58c`, views manifest
  `ea627a8f…4d77`. Readouts ran once on its GO: **OBS-NONE**, report `96ab76b2…7b3f`.
  G-repro-off reproduced TASK-074 exactly. No view is admitted at τ = 1.0 cm (R_off c_upper:
  hand crop 2.655, onboard 112 3.394, onboard 224 3.321, overview 224 3.438 cm); A3 (beat the
  clock prior) fails on every view; R_full and R_pix fail B (best: R_pix overview 224, c_upper
  2.022 cm). Nothing is admitted at τ = 0.5 or 1.5 cm. The learning curve is not still falling on
  any view, so the §7 clause fires and the next step is "a task or condition change".
  Reported only: the plate position is read to 0.49–0.68 cm, the apple to 2.2–3.1 cm. Results:
  `docs/experiments/apple_obs_ceiling_v2_results.md`. Learned Apple→Plate is still 0 successes.
  `mc` was not available on the Linux PC, so `mc validate` was not run.
