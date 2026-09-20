---
id: TASK-017
aliases:
- TASK-017
title: Implement common benchmark runner and machine-readable results
slug: implement-common-benchmark-runner-and-machine-readable-results
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m3
- mvp
sprint: ''
depends_on:
- "[[TASK-016]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Implement common benchmark runner and machine-readable results

## Description

Implement the experiment manifest, per-episode JSONL, aggregates, and shared metric code described in docs/EVALUATION.md.

## Acceptance Criteria

- [x] Validate result schemas and archive code/data/split/action/checkpoint hashes, seeds, hardware, engine, budgets, and resolved configs.
- [x] Report physical success with intervals, model diagnostics, train time, device-appropriate memory, latency and planning throughput.
- [x] Record all failures/timeouts and null-with-reason missing measurements; compare matched cohorts and do not compare raw latent scales across models.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Final measured evidence

All 400 final episode records pass strict schema validation. The summary audit recomputes all eight summaries, verifies selected checkpoint and source hashes, and checks common data/split/action/goals/planner settings. Per-seed success intervals, timings, throughput, process RSS, replans and failures are published in benchmarks/manifests/mvp-results-v0.json. Missing collision classification remains null with a reason.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-016]] %%
