---
id: TASK-041
aliases:
- TASK-041
title: Test image-only goal alignment against frozen predicted futures
slug: test-image-only-goal-alignment-against-frozen-predicted-futures
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- representation
- diagnostics
sprint: ''
depends_on:
- "[[TASK-040]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---

# Test image-only goal alignment against frozen predicted futures

## Question and scope
Can an image-only encoder estimate seven measured right-arm position fields on unseen parent sessions, and does its goal ranking remain useful with the frozen H16 sensor forecasts? Repeated physical controller changes have failed despite passing action-assignment diagnostics. This offline task tests a representation hypothesis before any controller integration.

## Acceptance Criteria
- [x] Freeze a reviewed protocol, exact position names, TRAIN-only fitting, parent-balanced evaluation, candidate/baseline interpretation, promotion gates and budgets before fitting.
- [ ] Implement one deterministic CPU4/seed0 experiment comparing TRAIN mean, TRAIN-only nearest-image retrieval and a small neural mean/uncertainty head. No TEST decoding, physical rollout or goal-state input at inference.
- [ ] Validate split isolation, grouping, predicted-future rather than measured-only ranking, ambiguity reporting, artifact provenance and bounded failure behavior.
- [ ] Run at most one preregistered attempt; record all candidates, failures, timing, parent-level metrics, uncertainty and limitations.
- [ ] Independently review saved evidence and deliver through a reviewed PR.

## Boundaries
No sensor checkpoint implementation changes or dynamics retraining. Image-to-position estimation is learned from paired TRAIN sensor measurements; VAL positions are evaluation labels only. Seven arm positions are an identifiability screen, not a complete apple/plate cost. Preserve visual object evidence in any future design. Distinguish failure of a neural head to beat nearest-image retrieval from failure of all goal-alignment candidates. No final acceptance cohort or deployment in this task. Exact protocol and budgets are under review; fitting has not started.

## Ownership
Coordinator owns Git/MC and protocol approval. Independent agents inspect implementation feasibility, data semantics and scientific acceptance. TASK-040 evidence is complete; its PR is awaiting CI and merge.

## Frozen prospective design
Protocol: `docs/experiments/apple_goal_alignment_v1.md`; planning-time data inspection: `docs/experiments/apple_goal_alignment_data_audit.md`. Exactly 1,392 TRAIN and 360 VAL selected frames; CPU4/seed0, 2,000 updates, B128, final checkpoint only. Independent NN/neural gates require at least 20% encoder-error reduction versus TRAIN mean and at least five percentage points better predicted H16 goal ranking versus pixel, with positive gain in each VAL parent. Three stages have hard 120/300/60-wall-second caps; TRAIN bank freezes before VAL decoding and neural fit loads TRAIN only. No fitting or learned inference has run. Implementation reviews precede execution.
