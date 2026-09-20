---
id: TASK-019
aliases:
- TASK-019
title: Freeze compositional generalization and test manifests
slug: freeze-compositional-generalization-and-test-manifests
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m4
- mvp
sprint: ''
depends_on:
- "[[TASK-010]]"
- "[[TASK-018]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---

# Freeze compositional generalization and test manifests

## Description

Reserve apple-to-plate combinations as described in the PRD and distinguish in-distribution evaluation from zero-shot composition tests.

## Acceptance Criteria

- [ ] Publish seen/unseen object, receptacle, appearance, and pairing tables with immutable episode/goal/reset manifests.
- [ ] Verify no session/frame/goal leakage and no task-specific training or normalization fitting on the held-out cohort.
- [ ] Freeze evaluation seeds, episode counts, thresholds and planned comparisons before unsealing final results.

## Notes

- Milestone: m4.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
