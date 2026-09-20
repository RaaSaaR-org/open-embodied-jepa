---
id: TASK-038
aliases:
- TASK-038
title: Audit saved-state forecasts behind the apple control stall
slug: audit-saved-state-forecasts-behind-the-apple-control-stall
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- diagnostics
- world-model
sprint: ''
depends_on:
- "[[TASK-037]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---

# Audit saved-state forecasts behind the apple control stall

## Description

The balanced H16 model passes the matched intervention prediction gate but stalls at goal5 during physical control. Distinguish optimistic plan forecasts from future action-projection mismatch and receding-horizon replanning by executing complete frozen candidate sequences from three exactly replayed development roots. This is a diagnostic, not a new controller or performance claim.

## Acceptance Criteria

- [ ] Preregister exactly three roots (119,300,550), winner/demonstration/hold siblings, original frozen trace/source/checkpoint, strict replay checks and 180-wall/120-CPU-second budget before execution.
- [ ] Independently review complete physics/controller/scorer state restoration, actual-applied action attribution, leakage boundaries and durable partial-result accounting; pass meaningful focused tests.
- [ ] Run the single bounded attempt and preserve all nine planned branches, replay failures/timeouts, actual sensor/action prefixes, provenance and H1/H16 forecast errors where available. No automatic retries or budget extensions.
- [ ] Interpret observed evidence and limits without inferring physical success or uniquely attributing model error; specify the next bounded hypothesis if supported.
- [ ] Deliver reviewed code, protocol, compact result manifest and MC evidence through a PR.

## Scope and execution

Original interrupted control-v2 artifacts are immutable. Neither final cohort nor TEST images may be opened. No core model/planner or physics changes. Scripts use the original captured runtime to reproduce the original controller. The implementation and pre-run review are prepared outside the active experiment tree; the coordinator integrates and commits before launching.
