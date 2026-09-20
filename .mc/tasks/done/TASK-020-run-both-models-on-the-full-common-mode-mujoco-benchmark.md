---
id: TASK-020
aliases:
- TASK-020
title: Run both models on the full common-mode MuJoCo benchmark
slug: run-both-models-on-the-full-common-mode-mujoco-benchmark
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m4
- mvp
sprint: ''
depends_on:
- "[[TASK-017]]"
- "[[TASK-018]]"
- "[[TASK-019]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---



# Run both models on the full common-mode MuJoCo benchmark

## Description

Run final Apple-to-Plate and stage evaluations locally under a shared measured Mac budget; retrain on the final shared corpus when expanded.

## Acceptance Criteria

- [x] Archive both checkpoints and matched common-mode configurations; run all planned seeds/episodes or explicitly document incomplete work.
- [x] Report full-task/stage success, confidence intervals, latency, memory, replans, controls, and zero-shot outcomes without omitting failures.
- [x] Produce a reproducible comparison report and rollout references; clearly separate engineering completeness from research success.

## Notes

- Milestone: m4.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence (ongoing)

Final protocol/configs and serial bounded orchestrator prepared before execution: 50 resets × 3 training seeds × 2 models, plus 50 hold and 50 random, 805 steps, shared CEM H4/64 candidates/3 iterations/8 elites. No final model outcome observed yet. Final dataset assembly and training follow expanded release collection. Incomplete/budgeted attempts must remain explicit.

Delivery PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 (draft; merge pending).

## Final measured evidence

Completed all six trained-model cohorts and both controls: 400/400 planned attempts in 595.082 seconds. Every run scored 0/50 with zero stage successes. Learned models each have a descriptive 0/150 across three repeated-reset cohorts; no pooled independence claim. All model/random episodes stopped on right joint rate limits; hold reached its step limit. Full reports, hashes and rollout references are in docs/experiments/mvp_results.md and benchmarks/manifests/mvp-results-v0.json. Negative results remain unchanged; TASK-029 tracks actuator-feasible planning.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-017]] [[TASK-018]] [[TASK-019]] %%
