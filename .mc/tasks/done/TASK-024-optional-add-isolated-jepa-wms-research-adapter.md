---
id: TASK-024
aliases:
- TASK-024
title: 'Optional: add isolated JEPA-WMs research adapter'
slug: optional-add-isolated-jepa-wms-research-adapter
status: done
priority: 4
owner: ''
projects: []
customers: []
tags:
- post_mvp
- optional
sprint: ''
depends_on:
- "[[TASK-016]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-21
---



# Optional: add isolated JEPA-WMs research adapter

## Description

Integrate JEPA-WMs only as an optional research backend after source, checkpoint, encoder, and dependency terms are tracked.

## Acceptance Criteria

- [x] Required permissive core remains installable without restricted components.
- [x] Adapter passes shared conformance and the same dataset/action/planner path.
- [x] Results and documentation identify artifact terms, permitted intended use, and any benchmark deviations.

## Notes

- Milestone: post_mvp.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
%% mc-links: [[TASK-016]] %%

## Delivery evidence (2026-09-21)

The opt-in adapter loads unchanged pinned upstream encoder/predictor modules in a private namespace and uses this project's explicitly different training recipe. No upstream source or weights are bundled in the permissive wheel. CPU only; random-initialization compatibility does not establish apple manipulation or paper reproduction.

Coordinator review covered source integrity/import isolation, recursive action semantics, normalization/checkpoint identity, config registration, dependency/license isolation and the complete tests/docs diff. No blocking findings remain. This is code review by a separate coordinating agent, not independent human approval.

Validation: `PYTHONPATH=src:outputs/task024-deps/site JEPA_WMS_SOURCE=outputs/task024-source/jepa-wms JEPA_TEST_THREADS=1 .venv/bin/python -m pytest -q` — **355 passed, 6 skipped** (three upstream dataset compatibility and three graphics opt-in). Nine actual-upstream tests passed; source, dependencies and bounded probe evidence are in `docs/experiments/jepa_wms_spike.md` and `benchmarks/manifests/jepa-wms-spike.json`. Ruff, formatting, lockfile and wheel checks passed. Deliverable criteria are satisfied; remote PR merge is pending and will be verified separately.
