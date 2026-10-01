---
id: TASK-075
aliases:
- TASK-075
title: 'Observation ceiling v2: measure each view''s readout ceiling against the place tolerance'
slug: observation-ceiling-v2-measure-each-views-readout-ceiling-against-the-place-tolerance
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- perception
- measurement
- preregistration
- lewm
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
- [ ] Stage 0: protocol `docs/experiments/apple_obs_ceiling_v2.md`, manifest
      `benchmarks/manifests/apple-obs-ceiling-v2.json` (frozen block, τ record, pins), the
      modules, the runner and the tests.
- [ ] Stage 0: τ measured on 55000–55031 before the freeze; the frozen-code smokes, including
      the full-scale readouts memory probe, on smoke seeds only.
- [ ] Stage 0 merged on an independent reviewer's reported APPROVE.
- [ ] Stage 1 (render) on its own reported GO: VIEWS-SEALED.
- [ ] Stage 2 (readouts) on its own reported GO: a row.
- [ ] The results PR, reviewed with every restated number checked.

## Notes
- 2026-10-01: preregistration PR opened (Stage 0). `mc` was not available on the Linux PC, so
  `mc validate` and `mc index` were not run; the frontmatter follows TASK-074's card.
