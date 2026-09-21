---
id: TASK-040
aliases:
- TASK-040
title: Test measured-arrival feedback at the stalled apple waypoint
slug: test-measured-arrival-feedback-at-the-stalled-apple-waypoint
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- diagnostics
- control
sprint: ''
depends_on:
- "[[TASK-039]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---


# Test measured-arrival feedback at the stalled apple waypoint

## Question

At command60 of learned reset43000, a cached command carried the robot outside goal2 tolerance after one qualifying observation. Compare original continuation against a recurring measured-image arrival interrupt from that exact development state. This tests one feedback opportunity, not a general manipulation solution.

## Acceptance Criteria

- [x] Freeze root60, original trace/source/checkpoint/data, two 16-command siblings and 60-wall/40-CPU-second budget; preserve all physical guards and image-goal thresholds.
- [x] Independently review full state/RNG/cache restoration, exact original replay, recurring image-only intervention, completed-dwell accounting and durable failure reporting; pass focused tests.
- [x] Commit the protocol and run exactly one bounded attempt, preserving all 32 intended slots and any failure or unavailable primary outcome.
- [x] Independently audit saved evidence, report whether normal goal2 dwell completes, and distinguish this local result from physical success or population improvement.
- [ ] Deliver code, protocol, result manifest and review through a PR.

## Notes

Implementation prepared outside the active comparison tree. The intervention clears only an existing commitment when measured distance is within the current unchanged goal threshold, then invokes ordinary model-ranked planning. No forced hold, future state, scoring truth, training, TEST decoding or final-cohort execution. See `docs/experiments/apple_arrival_feedback_v1.md`. Existing two learned failures and four control timeouts remain preserved.

## Completed audit
Committed source `b667e19` completed the single audit in 5.493 wall / 5.165 CPU seconds, exit 0. Both siblings accepted 16 commands, remained at goal 2, and reached maximum dwell 1; no physical task stage occurred. Independent review verified all 39 input hashes, eight artifact hashes, 32 action acknowledgements, original continuation, and the one actual cache clear. See `docs/experiments/apple_arrival_feedback_results_v1.md`, compact manifest, and `docs/reviews/apple_arrival_feedback_review.md`. Twelve integrated tests and Ruff passed. Delivery pending; this negative research result does not satisfy physical acceptance.
%% mc-links: [[TASK-039]] %%
