---
id: TASK-060
aliases:
- TASK-060
title: Bring CLAUDE.md and AGENTS.md status wording into line with TASK-057
slug: bring-claude-md-and-agents-md-status-wording-into-line-with-task-057
status: review
priority: 2
owner: ''
projects: []
customers: []
tags:
- docs
- hygiene
sprint: ''
depends_on:
- "[[TASK-057]]"
due_date: ''
created: 2026-09-25
updated: 2026-09-25
---



# Bring CLAUDE.md and AGENTS.md status wording into line with TASK-057

## Description
`CLAUDE.md` and `AGENTS.md` still describe behaviour cloning with the world model as a critic as
the current control line. Since TASK-057 (Outcome X) the `apple_policy_v1.md` §7 abandonment
clause has also fired, so neither CEM over the world-model cost (TASK-054) nor behaviour cloning
(TASK-057) is the primary line, and the next task is a perception/data task (TASK-059). The
"0/150" figure and the "every task-specific apple attempt since has failed its declared gate"
sentence in `CLAUDE.md` are also imprecise.

Documentation only. No experiment is run; nothing under `docs/experiments/`,
`benchmarks/manifests/` or `docs/DECISIONS.md` changes. Learned Apple->Plate remains at
0 successes.

## Acceptance Criteria
- [x] Both abandonments (TASK-054 CEM, TASK-057 BC) and their consequence are stated as the
      merged `docs/DECISIONS.md` entry and `apple_policy_diagnostics_v1_results.md` word them;
      TASK-059 named as the perception/data task; LeWM backend, encoder and product goal kept
      as not abandoned.
- [x] The 0/150 figure is tied to its benchmark (TASK-020, `mvp_results.md`), cohort and models.
- [x] The "every attempt since" sentence is exactly true and scoped to learned control attempts
      through TASK-057, with non-learned and diagnostic successes excluded.
- [x] `planning.py` note no longer implies a primary control line exists.
- [ ] Independent review reports APPROVE; CI green; merged.

## Notes
- Branch `docs/agent-status-refresh` from origin/main `781664f`.
%% mc-links: [[TASK-057]] %%
