---
id: TASK-042
aliases:
- TASK-042
title: Integrate visual and learned pose goal costs and test bounded apple control
slug: integrate-visual-and-learned-pose-goal-costs-and-test-bounded-apple-control
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- model
- control
sprint: ''
depends_on:
- "[[TASK-041]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---

# Integrate visual and learned pose goal costs and test bounded apple control

## Question
TASK-041 establishes useful image-only arm-goal estimates on three held-out parents. Can a fixed combination with visual evidence improve the existing world-model controller, with action-conditioned model controls retained?

## Acceptance Criteria
- [ ] Implement an isolated frozen composition backend with image-only goals/progress, unchanged sensor predictions, fixed visual/pose weighting, strict portable artifacts, and meaningful conformance/failure tests.
- [ ] Calibrate two cost scales using all 26 original TRAIN parents under the prospective stride28 median-of-medians rule, preserving zero pairs; no VAL/test/scorer input.
- [ ] Freeze and review source/protocol; run one 120-second bundle preparation and one 60-second saved-forecast hybrid gate, preserving failures.
- [ ] Independently verify combined H16 gain of at least five percentage points versus pixel and positive gain on each VAL parent with complete coverage before physics.
- [ ] If gate passes, run exactly reset 43000 with learned/persistence/shuffle, commitment 1, maximum 1,000 commands, 300 seconds/attempt within 1,000 global seconds; report every planned outcome without threshold or resource changes. If gate fails, preserve an explicit physical stop.
- [ ] Independently review evidence and deliver through PR; retain physical acceptance open unless separately demonstrated.

## Scope and ownership
Protocol `docs/experiments/apple_aligned_control_v1.md`. Coordinator owns Git, MC, protocol and architecture docs. Model agent owns new backend/config registration/tests. Simulation agent owns bundle/calibration/offline audit/tests. Contracts agent reviews semantics and scientific evidence. No changes to sensor.py, base.py, generic planner, task scoring or physical guards. No training, TEST decoding or final 20-reset cohort.

## Resources and uncertainty
Mac CPU4 only. Offline stages total 180 seconds allocation, conditional physical comparison 1,000 seconds. This n=1 development comparison is not final acceptance. Image-derived current pose can affect progress even when forecast pose is useful; visual term is retained but does not guarantee object-state recognition. PR #11 evidence is independently audited and awaiting CI/merge.
