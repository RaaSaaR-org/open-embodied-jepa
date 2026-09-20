---
id: TASK-030
aliases:
- TASK-030
title: Reduce candidate projection latency and complete the matched diagnostic
slug: reduce-candidate-projection-latency-and-complete-the-matched-diagnostic
status: in-progress
priority: 2
owner: ''
projects: []
customers: []
tags:
- research-followup
- planner
sprint: ''
depends_on:
- "[[TASK-029]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---



# Reduce candidate projection latency and complete the matched diagnostic

## Description
<!-- What needs to be done -->

## Acceptance Criteria
- [ ] Criterion 1

## Notes
## Scope and evidence

TASK-029 completed its bounded diagnostic with 15/20 completed episodes, one
interrupted, four unstarted. All 666 projected commands agreed exactly with
accepted targets, but IK/backtracking dominated planning cost. Two completed LeWM
runs exceeded the five-second control deadline; projection-on p95 planning was
4.222 seconds for LeWM and 0.330 seconds for native. Neither model grasped/placed.

## Acceptance criteria

- [ ] Profile a fixed bounded fixture (up to 60 CPU seconds) and identify repeated
  robot-kinematic/IK work; preserve current action, hand-rate and workspace checks.
- [ ] Optimize the preview with explicit numerical-agreement tolerances and
  regressions for tracking lag, stale transforms, invalid candidates and deadlines.
  Do not use object truth, relax guards or substitute raw requests for scored targets.
- [ ] Preregister fresh output paths, fixed models/cohort/candidate budgets and
  a total runtime cap before evaluating; retain TASK-029's incomplete artifacts.
- [ ] Report all intended attempts, control latency, command agreement, guard stops
  and physical stages. Aim to complete the 20-episode comparison within 600 seconds;
  stop and report a negative result if that remains infeasible. No retraining here.

## Next research gate

Stable runtime still does not imply useful learned manipulation. Isolate image-goal
reaching against scripted/hold/random controls before authorizing another larger
training or full-task experiment; preregister that separate work.
%% mc-links: [[TASK-029]] %%
