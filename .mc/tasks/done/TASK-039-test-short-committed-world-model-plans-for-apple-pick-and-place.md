---
id: TASK-039
aliases:
- TASK-039
title: Test short committed world-model plans for apple pick and place
slug: test-short-committed-world-model-plans-for-apple-pick-and-place
status: done
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

- [x] Add optional generic commitment length, default 1 preserving previous behavior/RNG; four means the newly planned command plus three acknowledged cached commands.
- [x] Observe image progress and dwell every command; discard commitment on goal advancement, reproject cached commands against current measurements, and fresh-search once if infeasible or altered. Actual execute rejection remains terminal. Preserve full-H16 warm state, acknowledgement, freshness and combined planning deadline.
- [x] Trace original plan identity, committed offset and freshly validated action without representing reused scores as fresh ranking; verify meaningful controller/timeout failure cases and independent review.
- [x] Preregister one unchanged-model development comparison: two development seeds, learned/persistence/shuffle, max 1,000 commands, four-command commitment, 90-wall-second attempt caps within 600 global wall. Preserve all planned statuses and no final-cohort exposure.
- [x] Run the bounded comparison once; report stage success, failures, latency, command acceptance and model ablations with provenance. No threshold relaxation or automatic retry.
- [x] Deliver reviewed implementation/evidence through a PR; keep physical acceptance open unless the actual success criteria pass.

## Ownership

Coordinator owns MissionControl and Git. Implementation agent owns waypoint/evaluation changes and focused tests. Independent agents review controller/physical semantics and scientific evidence. The audit is merged separately; this task begins from its reviewed branch while that PR completes delivery.

## Implementation review

Controller implementation passed independent review and 24 focused tests. A separate historical comparator matched default-one actions, costs, random state, waypoint progress and H16 warm arrays across 144 synthetic transitions (three modes/four seeds). The prospective protocol retains the frozen model/goals and explicitly discloses the new per-attempt allocation rule. Final controller/evaluator tests passed 58 cases in 0.13 seconds; the full optional/rendering suite passed 488 cases before the final evaluator-only correction. All lint/format and MC checks passed. Independent reviews found no remaining blocker. The bounded physical comparison remains pending until the reviewed source and protocol are committed.
## Completed development comparison

Source `acf5c67`: all six attempts started within 462.529 seconds. Both learned runs completed 1,000 commands; four controls reached their 90-second attempt allocation with soft timeouts. No attempt reached, grasped or placed the apple. Independent evidence audit verified 4,382 acknowledgments, 2,741 exact cached actions, eight tiny fresh-search roundoff clippings, all input identities and no pending execution uncertainty. See `docs/experiments/apple_control_commitment_results_v1.md` and its compact manifest. PR #9: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/9. Software checks pass; merge pending. Physical acceptance remains open in TASK-033.

## Verified delivery
PR #9 merged as `19ac6bbda8b7dd69d7ec0d3da50ad080daa9f11a` after all three current-head CI checks passed. Independent implementation and evidence reviews are recorded. Negative physical results remain unchanged; TASK-033 stays open.
%% mc-links: [[TASK-038]] %%
