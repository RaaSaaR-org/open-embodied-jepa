---
id: TASK-092
aliases:
- TASK-092
title: 'Rate-aware grounded world model: train G on rate-equivalent grasp requests, gate it on rate fidelity, then the closed-loop grasp-and-lift comparison against the unchanged G'
slug: rate-aware-grasp-world-model
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- model
- planning
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-091]]"
due_date: ''
created: 2026-10-14
updated: 2026-10-14
---


# Rate-aware grounded world model: train G on rate-equivalent grasp requests, gate it on rate fidelity, then the closed-loop grasp-and-lift comparison against the unchanged G

## Description

R27.71's model change after TASK-091 closed at **B-FAIL** and its clause fired. The development
measurements behind it (protocol §1.2, §11; not gated, CPU only) are these:

- The right-grasp channel is rate-limited to 2 / 11 per command, and no `play-v1` training step
  exceeds that.
- In TASK-091 Stage BS (one run, simulation only), 28–45 % of the chosen plans' grasp steps per
  seed did exceed it.
- Under grasp requests the embodiment executes identically to the recorded ones, the G models
  predict the right-hand state about 7–11 times worse at h = 8 (826 val roots).

TASK-092 trains TASK-087's G with rate-equivalent grasp requests on half of the windows (**G-req**).
Everything else stays TASK-087's: the corpus, latent, recipe, budget and model seeds 87100–87102,
so G-req is exactly paired with the unchanged G. The stages are:

- **Stage T:** three training jobs.
- **Stage R** (offline, `play-v1` test, gates the closed loop):
  - R1, rate fidelity: the hand-state ratio's upper bound must be ≤ 1 + 0.25 × (the paired
    control's ratio − 1). The measured ceiling is 1.
  - R2, no regression: within δ measured from the control seeds.
  - E, validity.
- **Stage A‴:** TASK-091's A″ on fresh roots 908000–908063, as a no-regression gate.
- **Closed loop:** BD (90400–90431), an independent GO, then BS (n = 128, 909000–909127). M-req
  runs against M-v1 (TASK-087's G, unchanged), with TASK-089's step-averaged planner frozen. The
  bar is min(0.40, 0.9 r). Rows: B-VOID, B-PASS, B-GAIN, B-BAR, B-FAIL and B-UNDER. The clause
  fires on B-FAIL only.

This is not Apple→Plate. R7 does not change.

Protocol: [grasp_rate_aware_wm_v1.md](../../../docs/experiments/grasp_rate_aware_wm_v1.md)
(DRAFT R27.74–R27.88, DECISIONS (ax); FROZEN at the Stage 0 merge #215, Stage 0 record §15,
R27.89–R27.94, DECISIONS (ay); Stage T, R and A‴ record §16, R27.95–R27.99, DECISIONS (az); escalation ruling §17, R27.100–R27.106, DECISIONS (ba):
closed at A‴-FAIL).

## Acceptance criteria

- [x] DRAFT preregistration reviewed independently and merged (#214)
- [x] Stage 0, reviewed, with the protocol frozen at its merge (#215; record §15,
  evidence `~/develop/emai/evidence/task092-stage0/`). It covers:
  - opt-in modules for the augmentation, the loss subclass, the training driver, the guarded
    loader, Stage R, A‴ and the rows;
  - the pairing check (p = 0 gives TASK-087's weights bitwise on debug seed 87900);
  - the seed and hash checks;
  - the power recomputed with 20 000 simulations;
  - a timed debug smoke on 90500–90599.
- [x] Stage T recorded (T-VOID on a failed check or a cap hit): T-DONE (§16.1)
- [x] Stage R recorded (R-PREMISE, E-FAIL or R-FAIL escalate, with no clause): R-PASS (§16.2)
- [x] Stage A‴ recorded (A‴-FAIL escalates, with no clause): **A‴-FAIL** (§16.3); BD and BS do not run
- [x] ~~Stage BD recorded, the bar set, and an independent GO posted before Stage BS~~ — not run:
  A‴-FAIL is a stop row (R27.97); closed by R27.100
- [x] ~~Stage BS run; the record, row and DECISIONS rulings reviewed and merged~~ — not run (R27.97,
  R27.100)
- [x] Escalation ruled: closed at A‴-FAIL, no clause (§17, R27.100–R27.106, DECISIONS (ba))
- [x] Evidence in `~/develop/emai/evidence/task092-*/` with `SHA256SUMS` (the two MP4s belonged to
  BS, which did not run)
- [x] CLAUDE.md entry points (added at Stage 0, D-RATE's script added at §17) and the plan's
  Progress table updated

## Notes

- Seeds. Model seeds 87100–87102 (TASK-087's, reused for pairing). A‴ roots 908000–908063. BD
  90400–90431. BS 909000–909127, as two 64-reset jobs per world-model run. Debug 90500–90599. The
  pairing check uses 87900.
- Salts. New: 9008 (the augmentation) and 9009 (Stage R's random requests). Reused: TASK-087's
  8711 and 8713, TASK-088's 8801–8804 and TASK-090's 9002, 9004 and 9006.
- Budget, about 17 GPU h: Stage T 1.9, BD 1.2 and BS 12.8, the rest under 1. Every job runs under
  the 7 200 s cap and one `gpu_run.sh --wait` lock, and BOARD.md's Requests are checked first.
- `worktrees/task087-run` (TASK-087's feature store and projection) must not be archived or
  removed until TASK-092 closes.
- The development evidence behind the DRAFT is `~/develop/emai/evidence/task092-prototype/`
  (`SHA256SUMS` sha256 `fd972ef4…910c`, regenerated after the review of #214).
- 2026-10-14, Stage T, Stage R and Stage A‴ (branch `docs/task092-t-r-a`; R27.95–R27.99, DECISIONS
  (az); protocol §16). Each ran once at `64e87f0` from a clean tree, after `verify` and the seed
  check passed, with no `- v2d:` request on BOARD.md.
  - **T-DONE:** three jobs, 6 713 s, kept updates 45 000 / 51 000 / 54 000, `last_two_triggered`
    false.
  - **R-PASS** on all three seeds (one run, offline, simulation only, `play-v1` test). ρ_req was
    0.99–1.02 against ρ_ctrl 15.0–20.3, with no R2 regression and E passed.
  - **A‴-FAIL.** close≤4@R1 was 44 / 51 / 36 and lift@R2 55 / 35 / 24, against bars of 48; A1
    passed for every model.
  - A post-hoc reading (reported only) finds that, in executed terms over the first four steps
    at R1 and the first command at R2 (whose executed-grasp half cannot fail), the request-level
    bars do not separate the families; over eight steps at R1 two G-req seeds close at the
    maximum rate less often than G-v1 (54 / 38 / 23 against 44 / 56 / 53).
  - Escalated with no clause; BD and BS do not run; the escalation needs its own ruling. GPU
    about 2.0 h. Evidence `~/develop/emai/evidence/task092-{t,r,a}/` (`SHA256SUMS`).
- 2026-10-14, escalation ruling (branch `docs/task092-ruling`; R27.100–R27.106, DECISIONS (ba);
  protocol §17). **TASK-092 closes at A‴-FAIL as recorded; the clause does not fire; R7 is
  unchanged.** One development-only closed-loop reading, **D-RATE**, is authorised: M-req-87100
  against M-v1-87100 on 32 fresh resets **90810–90841** (cohort `dev-drate`), at `64e87f0` through
  `scripts/dev_task092_drate.py`; not gated, not Stage BD, no claim. Its rule, declared before it
  runs: D-VOID, D-MOVES (d ≥ +4), D-WORSE (d ≤ −4), D-FLAT (|d| ≤ 3); 32 resets cannot establish
  an effect. D-MOVES → a powered closed-loop prereg of G-req with a degeneracy-checked,
  development-calibrated executed-measure gate; D-FLAT or D-WORSE → a planner-side rate
  projection under R26.33, with its own ruling and prereg. It runs after this PR merges; its
  record is a second PR (§18, DECISIONS (bb)).
- 2026-10-14, D-RATE record (branch `docs/task092-drate-record`; R27.107–R27.110, DECISIONS (bb);
  protocol §18): **D-VOID** (development, not gated). The M-req run stopped at model load before
  any episode, because the driver ran outside the run tree and the frozen loader resolves
  `third_party/le-wm` relative to the working directory. M-v1 did not run, so there are no counts.
  Resets 90810–90841 are spent. Ruled (R27.109): **D-RATE′** on fresh resets 90850–90881, with
  everything else unchanged, and two tested guards (the run tree as working directory, and a
  preflight that loads both models before any ledger claim). It runs after this PR is merged. GPU 21 s. Evidence `~/develop/emai/evidence/task092-drate/`.
