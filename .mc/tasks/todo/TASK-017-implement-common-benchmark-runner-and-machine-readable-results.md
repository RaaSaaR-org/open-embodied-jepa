---
id: TASK-017
aliases:
- TASK-017
title: Implement common benchmark runner and machine-readable results
slug: implement-common-benchmark-runner-and-machine-readable-results
status: backlog
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

- [ ] Validate result schemas and archive code/data/split/action/checkpoint hashes, seeds, hardware, engine, budgets, and resolved configs.
- [ ] Report physical success with intervals, model diagnostics, train time, device-appropriate memory, latency and planning throughput.
- [ ] Record all failures/timeouts and null-with-reason missing measurements; compare matched cohorts and do not compare raw latent scales across models.

## Notes

- Milestone: m3.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
