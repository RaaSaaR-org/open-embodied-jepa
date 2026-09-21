---
id: TASK-038
aliases:
- TASK-038
title: Audit saved-state forecasts behind the apple control stall
slug: audit-saved-state-forecasts-behind-the-apple-control-stall
status: done
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

- [x] Preregister exactly three roots (119,300,550), winner/demonstration/hold siblings, original frozen trace/source/checkpoint, strict replay checks and 180-wall/120-CPU-second budget before execution.
- [x] Independently review complete physics/controller/scorer state restoration, actual-applied action attribution, leakage boundaries and durable partial-result accounting; pass meaningful focused tests.
- [x] Run the single bounded attempt and preserve all nine planned branches, replay failures/timeouts, actual sensor/action prefixes, provenance and H1/H16 forecast errors where available. No automatic retries or budget extensions.
- [x] Interpret observed evidence and limits without inferring physical success or uniquely attributing model error; specify the next bounded hypothesis if supported.
- [x] Deliver reviewed code, protocol, compact result manifest and MC evidence through a PR.

## Scope and execution

Original interrupted control-v2 artifacts are immutable. Neither final cohort nor TEST images may be opened. No core model/planner or physics changes. Scripts use the original captured runtime to reproduce the original controller. The implementation and pre-run review are prepared outside the active experiment tree; the coordinator integrates and commits before launching.

## Execution evidence

Single registered attempt at c86a076 completed in55.45 wall/54.80 CPU seconds. All550 original commands replayed; eight branches completed16 commands and root550 winner was rejected on command10 after9 accepted due to joint-rate limit. All137 accepted branch commands exactly match scored actions. Original inputs remained valid. Root119/300 complete winning plans made more actual progress than demonstration or hold despite closed-loop stalling; this supports testing bounded commitment as a new hypothesis, not a universal model-accuracy claim. Result report, compact manifest and independent evidence review are delivered.
## Review and validation

Independent saved-evidence audit recomputed all17 measured endpoint metric sets from stored predictions and sensors, verified37 input identities,322 artifacts and137 exact applied commands; no findings. Full optional/graphics suite:475 passed in13.38s. Ruff check/format and MC validate/index passed. See `docs/reviews/apple_control_forecast_review.md`. The next separately scoped experiment tests four-command commitment with fresh feasibility checks. PR #8 (https://github.com/RaaSaaR-org/open-embodied-jepa/pull/8) merged at `00239ae613e51dc290ea6e4d07267c7435c3bc69` after all Linux/macOS core and optional integration checks passed on head `02bb0db`.
%% mc-links: [[TASK-037]] %%
