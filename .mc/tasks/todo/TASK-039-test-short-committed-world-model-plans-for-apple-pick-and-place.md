---
id: TASK-039
aliases:
- TASK-039
title: Test short committed world-model plans for apple pick and place
slug: test-short-committed-world-model-plans-for-apple-pick-and-place
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- planning
- control
sprint: ''
depends_on:
- "[[TASK-038]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---

# Test short committed world-model plans for apple pick and place

## Description

Saved-state diagnostic TASK-038 found that selected H16 plans made useful progress when executed at two stalled states, while repeated first-action replanning did not. Test four-command commitment with fresh feasibility checks as a scoped controller hypothesis. The frozen model, image goals, action contract, physics and scorer stay unchanged. This is not a claim that replanning is the unique failure cause.

## Acceptance Criteria

- [ ] Add optional generic commitment length, default1 preserving previous behavior/RNG; four means the newly planned command plus three acknowledged cached commands.
- [ ] Observe image progress and dwell every command; discard commitment on goal advancement, reproject cached commands against current measurements, and fresh-search once if infeasible or altered. Actual execute rejection remains terminal. Preserve full-H16 warm state, acknowledgement, freshness and combined planning deadline.
- [ ] Trace original plan identity, committed offset and freshly validated action without representing reused scores as fresh ranking; verify meaningful controller/timeout failure cases and independent review.
- [ ] Preregister one unchanged-model development comparison: two development seeds, learned/persistence/shuffle, max1000 commands, four-command commitment, 90-wall-second attempt caps within600 global wall. Preserve all planned statuses and no final-cohort exposure.
- [ ] Run the bounded comparison once; report stage success, failures, latency, command acceptance and model ablations with provenance. No threshold relaxation or automatic retry.
- [ ] Deliver reviewed implementation/evidence through a PR; keep physical acceptance open unless the actual success criteria pass.

## Ownership

Coordinator owns MissionControl and Git. Implementation agent owns waypoint/evaluation changes and focused tests. Independent agents review controller/physical semantics and scientific evidence. The audit is merged separately; this task begins from its reviewed branch while that PR completes delivery.
