---
id: TASK-041
aliases:
- TASK-041
title: Test image-only goal alignment against frozen predicted futures
slug: test-image-only-goal-alignment-against-frozen-predicted-futures
status: review
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
- [x] Implement one deterministic CPU4/seed0 experiment comparing TRAIN mean, TRAIN-only nearest-image retrieval and a small neural mean/uncertainty head. No TEST decoding, physical rollout or goal-state input at inference.
- [x] Validate split isolation, grouping, predicted-future rather than measured-only ranking, ambiguity reporting, artifact provenance and bounded failure behavior.
- [x] Run at most one preregistered attempt; record all candidates, failures, timing, parent-level metrics, uncertainty and limitations.
- [x] Independently review saved evidence.
- [ ] Deliver implementation and evidence through a reviewed PR.

## Boundaries
No sensor checkpoint implementation changes or dynamics retraining. Image-to-position estimation is learned from paired TRAIN sensor measurements; VAL positions are evaluation labels only. Seven arm positions are an identifiability screen, not a complete apple/plate cost. Preserve visual object evidence in any future design. Distinguish failure of a neural head to beat nearest-image retrieval from failure of all goal-alignment candidates. No final acceptance cohort or deployment in this task. Exact protocol and budgets are under review; fitting has not started.

## Ownership
Coordinator owns Git/MC and protocol approval. Independent agents inspect implementation feasibility, data semantics and scientific acceptance. TASK-040 evidence is complete; its PR is awaiting CI and merge.

## Frozen prospective design
Protocol: `docs/experiments/apple_goal_alignment_v1.md`; planning-time data inspection: `docs/experiments/apple_goal_alignment_data_audit.md`. Exactly 1,392 TRAIN and 360 VAL selected frames; CPU4/seed0, 2,000 updates, B128, final checkpoint only. Independent NN/neural gates require at least 20% encoder-error reduction versus TRAIN mean and at least five percentage points better predicted H16 goal ranking versus pixel, with positive gain in each VAL parent. Three stages have hard 120/300/60-wall-second caps; TRAIN bank freezes before VAL decoding and neural fit loads TRAIN only. No fitting or learned inference has run. Implementation reviews precede execution.

## Implementation and pre-run review
`scripts/apple_goal_alignment.py` preserves the frozen sensor/planner implementation and exposes only offline prepare/fit/evaluate stages. Independent scientific and runtime reviews found no remaining blockers. Twenty-four focused tests passed; full optional/model/rendering suite **542 passed in 11.09 seconds**. Ruff checks/format for 85 files, diff and MC checks passed. Review resolved timestamp validation, actual NN spread, neural-failure isolation, producer seals, encoded-payload integrity and deadline finalization. Gradient max-norm 10 is recorded prospectively. Exact metadata sample hash `37aaa45ef11995240be8ee84c7779ad006d35c03b4519ac0c1c6f77c0d35f6f5`. Run remains pending until this source and protocol commit.

## Completed bounded experiment
Source `f4f99fa1bf5fdcec7f1647927e74d0b0de8228b0`, outputs `outputs/apple-goal-alignment-v1`. Prepare/fit/evaluate completed once in 6.514556 / 2.682128 / 2.011637 wall seconds, each within its cap. Neural fit finished exactly 2,000 updates. Both candidates passed the prospective screen: H16 predicted ranking neural 89.81618%, NN 89.70628%, frozen pixel 73.37780%, with positive gains in all three VAL parents. Independent audit recomputed 3,456 cost vectors and 16,128 pair records, parent/bootstrap summaries, encoder errors and ambiguity diagnostics from saved arrays without inference. Neural replacement edge over NN is only 0.10990 percentage points. Arm alignment alone does not identify grasp/placement.

Coordinator supplemental launch ledger was written after preparation began because a system-Python datetime.UTC preamble failed; each stage correctly registered immutable inputs before its work. No stage was repeated. The final result and manifest disclose this bookkeeping issue. Final acceptance stays untouched and TASK-033 remains open. Delivery pending.
%% mc-links: [[TASK-040]] %%
