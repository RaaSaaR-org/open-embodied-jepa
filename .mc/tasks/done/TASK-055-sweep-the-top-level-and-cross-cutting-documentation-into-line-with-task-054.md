---
id: TASK-055
aliases:
- TASK-055
title: Sweep the top-level and cross-cutting documentation into line with TASK-054
slug: sweep-the-top-level-and-cross-cutting-documentation-into-line-with-task-054
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
- "[[TASK-054]]"
due_date: ''
created: 2026-09-24
updated: 2026-09-24
---


# Sweep the top-level and cross-cutting documentation into line with TASK-054

## Description
The experimental record moved from the MVP benchmark through the apple world model v2, v3 and
v4 protocols, and the narrative documentation was not swept in that time. TASK-054 also changed
the control formulation: all four v4 arms failed the 14 preregistered gates, the primary gate
G2a (rollout readout error divided by the model's own persistence readout, threshold 0.8) came
in at 0.8763 / 0.8814 / 0.8635 / 0.9036, the untouched control passed more gates (10/14) than
every intervention (9/14), and the protocol's pre-declared Outcome B fired: CEM over this
world-model cost is abandoned as the primary control line in favour of behaviour cloning with
the world model as a critic.

Honest-status sweep only. No experiment was run, no protocol, result document, review or
manifest under `docs/experiments/`, `docs/reviews/` or `benchmarks/manifests/` was rewritten,
no frozen value changed, and `PRD.md` is preserved as the original snapshot per AGENTS.md.
Learned Apple->Plate remains at zero successes.

## Acceptance Criteria
- [x] A dated status section in `README.md` states where the project is: learned Apple->Plate
      at 0 successes, what is demonstrated (encoder, grasp AUROC with its control,
      backend-swap invariant, infrastructure), what has been ruled out with evidence, and that
      the control line moved from sampling-based planning to behaviour cloning with the world
      model as a critic. `docs/MVP_PLAN.md` cross-links it.
- [x] The pivot is recorded as a dated decision in `docs/DECISIONS.md` with the gate numbers,
      the five failed attempts on the rollout gap, and what is explicitly not abandoned (the
      LeWM backend, the encoder, the product goal, the planner implementation).
- [x] Every top-level and cross-cutting doc audited for claims that are now false, overstated
      or stale; overstated sentences deleted or corrected rather than replaced by invention;
      refuted hypotheses marked refuted with a pointer, never deleted.
- [x] `CLAUDE.md` tracked and corrected where the audit showed it stale.
- [x] ruff check, ruff format --check, `mc validate`, `mc index`.

## Notes
- Branch `docs/task-055-honest-status-sweep` from main `bc1fca6` (TASK-054 closure).
- Documentation only: no source, test, config, manifest or experiment artifact changed, so the
  research record is untouched and no result is revised.
- Delivery: PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/31 (squash-merged as
  `ac9625d`). Independent review found two wrong figures and both were fixed before merge: the
  best-ever G2a (the record is 0.831 at v3, not v4's 0.8635) and the world-model v2 wall-clock
  cap (6,000 s per run, not 10,800 s). Five non-blocking findings were also taken. ruff check,
  ruff format --check, `mc validate`, `mc index` and all three CI jobs were green on the merged
  head.
- Documentation only. Nothing under `docs/experiments/`, `docs/reviews/` or
  `benchmarks/manifests/` was rewritten, `PRD.md` is preserved as the original snapshot, no
  frozen value changed, and learned Apple->Plate remains at zero successes.
%% mc-links: [[TASK-054]] %%
