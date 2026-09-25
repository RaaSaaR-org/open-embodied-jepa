---
id: TASK-060
aliases:
- TASK-060
title: Bring CLAUDE.md and AGENTS.md status wording into line with TASK-057
slug: bring-claude-md-and-agents-md-status-wording-into-line-with-task-057
status: done
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
- [x] Independent review reports APPROVE; CI green; merged (PR #50, `464ca9b`).
- [x] Follow-up sweep: present-tense "behaviour cloning is the current line" wording corrected
      in `README.md` (status section and planner note), `docs/EVALUATION.md`,
      `docs/ARCHITECTURE.md` and `docs/MVP_PLAN.md`, keeping the dated history (follow-up PR).

## Notes
- Branch `docs/agent-status-refresh` from origin/main `781664f`.
- Delivery: PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/50, squash-merged as
  `464ca9b`. Independent reviewer (reported verdicts): round 1 APPROVE, round 2
  REQUEST_CHANGES (an unsourced claim that the task owner selected TASK-059; fixed), round 3
  APPROVE. CI (core on macOS and Ubuntu, macos-integration) green on the PR head.
- The reviewer found the same stale wording in `docs/EVALUATION.md`, `docs/ARCHITECTURE.md`,
  `docs/MVP_PLAN.md` and the README status section. It is fixed under this task in a follow-up
  PR on branch `docs/task-060-status-sweep`, which also closes this card. Documentation only;
  no DECISIONS entry, experiment doc, manifest or TASK-059 file changed.
%% mc-links: [[TASK-057]] %%
