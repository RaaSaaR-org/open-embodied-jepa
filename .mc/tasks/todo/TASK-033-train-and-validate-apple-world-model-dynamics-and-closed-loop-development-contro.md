---
id: TASK-033
aliases:
- TASK-033
title: Train and validate apple world-model dynamics and closed-loop development control
slug: train-and-validate-apple-world-model-dynamics-and-closed-loop-development-contro
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- training
sprint: ''
depends_on:
- "[[TASK-030]]"
- "[[TASK-031]]"
- "[[TASK-032]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---


# Train and pass dynamics and closed-loop development gates

## Acceptance criteria

- Preregister each bounded training attempt with immutable data/split/action/source hashes, seed, objective, validation selection, memory and runtime caps. Initial budget ceiling: 1800 wall seconds per attempt; no unlimited search.
- Use training/validation only; measure prediction versus persistence and shuffled actions, multi-step drift, collapse where relevant, and state/action sensitivity.
- Run frozen development controls and learned policy with all attempts counted. Require apple grasp/transport/release success with unchanged evaluator before final testing.
- Measure a dynamics-disabled or shuffled-dynamics ablation to establish the learned model contributes to control.
- Preserve negative attempts and declare subsequent hypothesis changes before execution; never retune the sealed final cohort.

## Authorization and workflow

User requested completion toward a working world-model apple pick-and-place MVP on 2026-09-20, with subagents and end-to-end delivery. Coordinator owns Git and task state. Experiments are bounded and recorded before execution.
%% mc-links: [[TASK-030]] [[TASK-031]] [[TASK-032]] %%

## Execution evidence

Initial training protocol and supervised runner committed before execution. Planned3000updates,B16,H8,seed0,CPU4,1800s totalwall. Data/source/protocol hashes mandatory. Physical learned ApplePlate remains unproven; all negative results retained.

First sensor model completed3000 updates and failed both learned development runs (0/6 comparison placements, including three errored controls). Matched-branch retraining improved causal assignment but missed the frozen primary H16 error-reduction gate. Preserve all artifacts and keep this task in progress. H16 balanced-intervention training subsequently passed the unchanged primary causal gate (89.40% assignment; 81.41% endpoint-error reduction). The first physical v2 attempt was externally paused after574 acknowledged commands. An unchanged preregistered resume comparison completed one learned failure at1000 commands (reach only), timed out persistence at846 commands, and left four attempts unstarted under its600-second total cap. No final-cohort evaluation has run. A prospective saved-state forecast audit will investigate the observed goal5 stall before any further controller change.

The separate four-command comparison (TASK-039) subsequently started all six attempts: two learned runs completed 1,000 commands, four controls timed out, and no physical stage was reached. Forecast audit and controller implementation tasks can be delivered independently of this unmet physical gate. A paired arrival-feedback diagnostic and a goal-metric research review follow; final evaluation remains untouched.
