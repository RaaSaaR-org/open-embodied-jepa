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

TASK-041 passed an offline image-goal screen; TASK-042 then missed the fixed combined-cost gate (+4.30pp versus required +5pp) and stopped before physics. These offline results do not meet this task's physical acceptance. The user requested pausing work after TASK-042 delivery; leave this task open and the final cohort untouched.

TASK-043 ran its single preregistered demonstration-state-goal comparison (source a90bf5b, resets 43000-43003, 16 attempts, 130 s): primary gate failed. All 12 learned/dynamics_shuffle/persistence attempts stalled on goal 0 with zero scorer stages; the non-learned open-loop demo_replay reference succeeded on 3/4 resets. Learned Apple->Plate is still unsuccessful; this task stays open and the final cohort untouched. Evidence: docs/experiments/apple_state_goal_control_results_v1.md, benchmarks/manifests/apple-state-goal-control-v1.json.

TASK-044 privileged MuJoCo-rollout ceiling (non-learned, source 733bfdc, resets 43000-43003): with exact dynamics the same state-goal scaffold also stalled (goal 0/1, 0/4 grasp, 0 stages), so the scaffold, not only the learned dynamics, blocks progress. Evidence: docs/experiments/apple_privileged_ceiling_results_v1.md.

TASK-045 privileged trajectory-tracking ceiling (non-learned, source bae0838, resets 43000-43003): tracking the demo's arm+hand trajectory reached reach/contact on 4/4 but grasp on 0/4 (apple dropped off the table 4/4); gate failed, learned stage not run. Evidence: docs/experiments/apple_trajectory_tracking_results_v1.md.

TASK-046 privileged hybrid phase ceiling (non-learned, source f8dac63, resets 43000-43003): tracked approach + open-loop demo close grasped 0/4 with handoff at the close row (primary gate failed) and 2/4 (full success) with handoff one horizon earlier; demo_replay reproduced 4/4 grasp. Evidence: docs/experiments/apple_hybrid_phase_results_v1.md.
