---
id: TASK-034
aliases:
- TASK-034
title: Validate world-model apple pick-and-place on a fresh frozen cohort and publish replay
slug: validate-world-model-apple-pick-and-place-on-a-fresh-frozen-cohort-and-publish-r
status: backlog
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- evaluation
sprint: ''
depends_on:
- "[[TASK-033]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Measure working apple pick-and-place and deliver evidence

## Acceptance criteria

- Freeze a new unseen reset cohort, checkpoint/config hashes, task thresholds, action budget, run cap and controls before evaluation. These are new task-specific tests, not the old zero-shot claim.
- Working-MVP target: at least 16/20 complete Apple→Plate successes on the frozen cohort, with all failures/timeouts included; report Wilson uncertainty and full stage breakdown. This is an acceptance target, not a promised research outcome.
- Run matched hold/random and learned-dynamics ablation controls; demonstrate that learned dynamics influence useful action selection.
- Store full local records and compact public manifests, actual MuJoCo replay video plus provenance, and exact reproduction commands.
- Independent software/scientific review, appropriate tests, PR and verified merge/CI. If target fails, report failure and continue via a newly declared development task, preserving sealed results.

## Authorization and workflow

User requested completion toward a working world-model apple pick-and-place MVP on 2026-09-20, with subagents and end-to-end delivery. Coordinator owns Git and task state. Experiments are bounded and recorded before execution.
