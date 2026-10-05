---
id: TASK-077
aliases:
- TASK-077
title: 'LeWM committed aim under C1-M: LeWM chooses the single place aim from the encoded frame'
slug: lewm-committed-aim-under-c1m-lewm-chooses-the-single-place-aim-from-the-encoded-frame
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- preregistration
- draft
sprint: ''
depends_on:
- "[[TASK-076]]"
due_date: ''
created: 2026-10-05
updated: 2026-10-05
---

# LeWM committed aim under C1-M: LeWM chooses the single place aim from the encoded frame

## Description

The C1-M feasibility record (`docs/experiments/apple_lewm_next_v2_c1m_feasibility.md`, R16,
#141) ended **M-PROCEED**, which admits only the drafting of a preregistration. This task is that
preregistration and, once frozen, its run.

**The question.** Under C1-M (v2's reset, P-3's pick, a post-pick plate move at step 300 over a
4 cm disc, cell A's reactive-plate rule from 405, one aim committed at 405, e9's place), does a
LeWM token predictor (TASK-066's recipe on frozen DINOv2 tokens pooled to 8 × 8, rolled 60 steps)
choosing that aim from the encoded current frame reach R9.8's primary claim, "LeWM-driven
closed-loop success", on a fresh gated cohort? That needs a calibrated bar (56/64),
non-inferiority to the best non-world-model arm within δ = 8/64 (an allocation), and exact
McNemar wins over the trained action-blind twin N, the trained scene-blind twins L-shuf and
L-mean, and a random choice. "LeWM needed" is reported only and not expected.

**The protocol (DRAFT, not frozen):**
[`docs/experiments/apple_lewm_c1m_v2.md`](../../../docs/experiments/apple_lewm_c1m_v2.md).
Rulings R17.1–R17.18 (`docs/DECISIONS.md`, decision 2026-10-05 (b)), decided by Claude under
owner delegation (2026-09-30). The plan is `docs/PLAN.md`.

Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7. This task does not
change it unless an L-PASS is followed by its own reviewed ruling.

## Acceptance Criteria
- [ ] DRAFT preregistration reviewed independently (this card's first PR; docs plus a
      development cost probe on synthetic data only).
- [ ] Stage 0 PR: stage code, frozen block, manifest, tests (seed ranges, `check_budget`,
      `select_checkpoint`/`last_two_triggered`, the `"not evaluated"` sentinel, no runner imports,
      no privileged read in W or a twin), debug smokes and the scale probe.
- [ ] K0 on a reported GO; K0's values in the frozen block; the freeze merged on an independent
      reviewer's reported APPROVE.
- [ ] Stages C, O, T, G, D and S, each on its own reported GO, from a clean worktree of the merged
      revision, after a full `pytest` at that revision (G-tests).
- [ ] Results PR, with every restated number checked by an independent reviewer.

## Notes
- 2026-10-05: card opened under R17.1 on branch `docs/task077-prereg-draft`. DRAFT protocol
  written; no seed of the block 66000–68999 simulated. Development cost probe
  (`scripts/probe_task077_cost.py`, synthetic features, debug torch seeds 66990–66991, through
  `scripts/gpu_run.sh --wait`): `outputs/task077-cost-1` ran out of GPU memory at 8 × 8, T = 60,
  batch 64 (no report); `outputs/task077-cost-2/report.json` (sha256 `0669bf24…378b`, at
  `2dc6106`): 0.161 s per update at T = 60, batch 16, 5.48 GiB; `outputs/task077-cost-cpu-1/report.json`
  (sha256 `f760f32f…28e1`, at `5218857`): 7.43 s for 147 candidates × 60 steps on one CPU thread.
  Stage T is estimated at 22–58 h of GPU in eight per-job slots.
- 2026-10-05: independent review of #142 at `26894ca`: REQUEST CHANGES (two blocking: the clause's
  scope and trigger; the void rule's counting). Revised under R17.15–R17.18 (decided by Claude
  under owner delegation): R15.8's clause scope extended by the move, with one declared exclusion
  (4 × 4 on a larger corpus); L-NO-GAIN needs a failed McNemar test and a detectable shortfall,
  misses within noise are L-TWIN-NEAR; a second V counts within the same stage, and Stage T voids
  per job; the ten non-blocking items fixed; the five open points settled. Stage T is now a
  scenario band of about 8–33 h with the Stage-0 storage fix (up to about 58 h without).
- 2026-10-05: Stage 0 under R17.19 on branch `feat/task077-stage0` (record
  `docs/experiments/apple_lewm_c1m_v2_stage0.md`). New modules `lewm_c1m_v2{,_runtime,_offline,_train}.py`
  and `scripts/run_lewm_c1m_v2.py`, DRAFT manifest, 34 tests. Simulations (10 000 trials per
  configuration): G-NI's size at the margin 2.4–3.8 % (one comparator) and 0.9–3.5 % (better of two);
  clause false fire 2.3–3.1 % per twin at +7/64, 0.9–3.1 % for L-INFERIOR at δ. The storage fix's
  gather is 3.7 ms per batch, and Stage T's own code ran at 0.166–0.223 s per update on synthetic
  features at the real sizes, which puts Stage T at about 7.4–9.9 h to 32–43 h. Debug smokes
  (66900–66999) of every stage passed mechanically at `a98d893`. No K0: it needs a reviewer GO.
- 2026-10-05: independent review of #143 at `0dea22b`: REQUEST CHANGES (five blocking: a reset
  refused before 405 voided D or S; the 1e-6 m determinism tolerance against renderer noise; the
  artifact chain; Stage D's cap; CI timing). Fixed under R17.20–R17.24 (decided by Claude under
  owner delegation), with the eleven non-blocking items; debug chain re-run at `9e772e0`; no K0.
