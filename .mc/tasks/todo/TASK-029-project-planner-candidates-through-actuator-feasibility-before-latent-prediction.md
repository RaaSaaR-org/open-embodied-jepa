---
id: TASK-029
aliases:
- TASK-029
title: Project planner candidates through actuator feasibility before latent prediction
slug: project-planner-candidates-through-actuator-feasibility-before-latent-prediction
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- research-followup
- planner
sprint: ''
depends_on:
- "[[TASK-020]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---


# Project planner candidates through actuator feasibility before latent prediction

## Description
The frozen MVP comparison scores requested absolute grasp targets while the embodiment can rate-limit them. A measured open-hand request +1 executes approximately −0.8181818 in its first step. Add a model-independent feasibility path so the world model scores the commands expected to be accepted, using current robot state and manifest limits. The final model traces also identify right-arm joint-rate rejection in all 300 learned-policy attempts, so include pose/IK feasibility as well as grasp clipping. This is a separate research follow-up; preserve the original frozen comparison unchanged.

## Acceptance Criteria
- [ ] Specify a shared candidate projection/constraint API with horizon-consistent grasp rate bounds and no privileged object truth or unavailable future state.
- [ ] Verify predicted candidate actions agree with accepted targets in bounded deterministic robot tests, retain requested/applied traces, and run both model backends unchanged through the same projection.
- [ ] Preregister a new matched comparison with fixed checkpoints/data/cohort/budget before execution; retain all original results and report whether guard stops and physical success improve.

## Notes

- Found by independent model/planner review before the final MVP evaluation; see `docs/reviews/runtime-review.md`.
- Source, dataset and model checkpoints remain immutable during the current six-run experiment. No post-result tuning or silent checkpoint-hash migration. Treat the inspected v0 cohort as development evidence in follow-up work; do not relabel reused outcomes as a newly sealed test.
%% mc-links: [[TASK-020]] %%
