---
id: TASK-032
aliases:
- TASK-032
title: Implement an observable action-conditioned world model for staged apple planning
slug: implement-an-observable-action-conditioned-world-model-for-staged-apple-planning
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- world-model
sprint: ''
depends_on:
- "[[TASK-031]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-21
---




# Implement a world model whose predictions support apple control

## Acceptance criteria

- Record diagnosis and design before implementation: visual observability, proprioception, action conditioning, temporal horizon and goal cost.
- Preserve model/planner/embodiment boundaries. No simulator object truth, task scores, future measured state or oracle controller targets enter deployed model/planner inputs.
- A learned action-conditioned transition model must predict candidate outcomes and materially determine action selection. Script replay, imitation alone, or an analytic controller is not accepted as a working learned WM.
- If staged goals, engineered perception or demonstrations are required, document them explicitly as a new mode; retain the historical native/LeWM common benchmark.
- Test transition rollout, action perturbation, checkpoint reproducibility and inference input isolation.

## Authorization and workflow

User requested completion toward a working world-model apple pick-and-place MVP on 2026-09-20, with subagents and end-to-end delivery. Coordinator owns Git and task state. Experiments are bounded and recorded before execution.
## Execution evidence

sensor_wm implements learned visual+proprioceptive residual prediction with frozen training normalization, image-only goals, checkpoint provenance and action controls. Dense image-waypoint shooting MPC ranks projected candidates through opaque learned predictions. No simulator truth enters inference. Independent waypoint and training reviews fixed stale-acknowledgement handling and supervision gaps. Synthetic learning and actual generic sensor training integration pass. Physical learned-control evidence is TASK033/034, not claimed here.
%% mc-links: [[TASK-031]] %%

## Deliverable review

Implementation acceptance is satisfied: synthetic learned transitions pass persistence/shuffle controls, strict checkpoint/resume and input-isolation checks pass, and saved physical traces show learned costs materially chose actions. The first closed-loop diagnostic nevertheless failed all placements. Marking this implementation task done does not satisfy TASK033/034's physical acceptance gates. Reviewed code and full449-test local integration/graphics suite pass. PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/5; merge pending.
