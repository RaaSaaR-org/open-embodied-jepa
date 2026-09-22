---
id: TASK-051
aliases:
- TASK-051
title: Diagnose the close-phase ejection and redesign the grasp closure (ceiling v3)
slug: diagnose-the-close-phase-ejection-and-redesign-the-grasp-closure-v3
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- control
- diagnostics
sprint: ''
depends_on:
- "[[TASK-049]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---

# Diagnose the close-phase ejection and redesign the grasp closure (ceiling v3)

## Question
TASK-049's object-aware exact-rollout ceiling v2 failed its primary gate on the fresh wide-jitter development resets 45100-45107 (5/8 success, 5/8 grasp; needed 6 and 7) while the scripted collector succeeded 8/8, giving outcome `ceiling_inadequate_task_feasible`. All four failures were one mode: 8-9 commands into the close phase the closing fingers ejected the apple (0.8-3.1 cm of xy motion, peaking 2.7-3.5 cm). Entry geometry did not separate failures from successes and the cause was not isolated. What causes the close-phase ejection, and does a principled redesign of the grasp closure (ceiling v3), expressed only in quantities a learned head could predict, reach full success >=6/8 and grasp+lift >=7/8 on a FRESH wide-jitter development reset range under exact dynamics? NON-LEARNED diagnostic only.

## Acceptance Criteria
- [ ] Written, evidence-backed diagnosis of the close-phase ejection from instrumented TRAIN-side runs (per-finger contact forces, apple impulse/displacement per command, palm pose, finger joint trajectories, grasp command schedule, ceiling close vs scripted close command-by-command). Exploratory, clearly labelled, not gate evidence.
- [ ] Ceiling v3 added alongside v1/v2 (v1 and v2 behaviour and evidence unchanged), with the closure redesign following from the diagnosis. Tests.
- [ ] Simulator physics/contact parameters unchanged, or, if changed, documented loudly as a task-realism change with the old config kept and the scripted baseline re-run.
- [ ] Development tuning only on a documented TRAIN-side tuning range (not 45000-45007, not 45100-45107, not the gated range, not 44000-44019, not TEST).
- [ ] Preregister `docs/experiments/apple_wide_grasp_closure_v3.md` + `benchmarks/manifests/apple-wide-grasp-closure-v3.json` (fresh dev resets, arms ceiling v3/scripted_oracle/demo_replay, primary gate, readings with next steps, frozen budget/command) before any attempt on the gated range.
- [ ] Fresh pre-run review; fix blockers; TRAIN-only smoke.
- [ ] Frozen command run once from a clean committed revision into a new output directory; every outcome recorded.
- [ ] Fresh post-run verification; results doc; ruff, pytest, `mc validate`; PR, CI, merge.

## Scope and resources
CPU. One command, 48 attempts (16 each of `demo_replay`, `scripted_oracle`, `privileged_object` v3; primary fresh range, then the secondary 45100-45107 cohort), mode-major.

## Notes
%% mc-links: [[TASK-049]] %%
