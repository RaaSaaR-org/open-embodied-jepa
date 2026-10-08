---
id: TASK-080
aliases:
- TASK-080
title: 'LeWM committed aim under C1-M, read by a readout fitted on predicted latents'
slug: lewm-committed-aim-under-c1m-read-by-a-readout-fitted-on-predicted-latents
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- preregistration
- frozen
- l-near
sprint: ''
depends_on:
- "[[TASK-077]]"
due_date: ''
created: 2026-10-07
updated: 2026-10-08
---

# LeWM committed aim under C1-M, read by a readout fitted on predicted latents

## Description

TASK-077 closed **G-NO-BAR** (`docs/experiments/apple_lewm_c1m_v2_results.md`, R17.52): its
predictor passed G1–G4, but R8, a plate readout fitted on encoded frames, read W's predicted plate
at r at 1.58–1.87 cm against τ_commit = 1.0 cm. The decomposition record
(`docs/experiments/apple_lewm_c1m_v2_decomposition.md`, R17.53–R17.54, #152) ended **D-READOUT**:
a ridge refitted on W's own stand-in predictions reads the plate at 0.556–0.578 cm in val median,
an optimistic development reading. This task is the preregistration that row names.

**The question.** Under C1-M, does W's committed aim at 405, with the predicted plate read by
R-S (a dual ridge fitted on W's own predicted latents under the kinematic stand-in's chunks), reach
R9.8's primary claim, "LeWM-driven closed-loop success", on a fresh gated cohort? TASK-077's six
Stage T checkpoints are reused (no training). A fresh 500-root corpus (65300–65799) gives a gate
split whose aims are built from R-plate's reading, as the closed loop builds its grid, and a
true-plate contrast half for the corpus-aim confound. Stage R gates the readout on those fresh
roots and the offline aims before any closed loop; D and S carry TASK-077's bars and rows (56/64,
non-inferiority within δ = 8/64, McNemar wins over N, L-shuf, L-mean and L-rand). "LeWM needed" is
reported only.

**The protocol (FROZEN after K0′-PASS, R18.22):**
[`docs/experiments/apple_lewm_c1m_v2_pred_readout.md`](../../../docs/experiments/apple_lewm_c1m_v2_pred_readout.md).
Rulings R18.1–R18.14 (`docs/DECISIONS.md`, decision 2026-10-07), decided by Claude under owner
delegation (2026-09-30). The plan is `docs/PLAN.md`.

Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7. This task does not
change it unless an L-PASS is followed by its own reviewed ruling.

## Acceptance Criteria
- [x] DRAFT preregistration reviewed independently (this card's first PR; docs only, plus
      #151's five minor nits).
- [x] Stage 0 PR: stage code in new modules, frozen block, manifest, tests (seed ranges against
      every forbidden range including 66000–68999; the reused artifacts' sha256s; W reads R-S and
      N reads R-N, never R8; gate-P aims built from p̂; the `"not evaluated"` sentinel; no
      privileged read in W or a twin), debug smokes, scale probes and R18.13's development dry run.
- [x] K0′ on a reported GO; K0′'s values in the frozen block.
- [x] The freeze merged on an independent reviewer's reported APPROVE (#155, `514110d`).
- [x] Stages C′, R, D and S, each on its own reported GO, from a clean worktree of the merged
      revision, after a full `pytest` at that revision (G-tests).
- [x] Results PR, with every restated number checked by an independent reviewer.

## Notes
- 2026-10-07: card opened under R18.1 on branch `docs/task080-prereg-draft`. DRAFT protocol only;
  nothing run. Open questions for the freeze are the protocol's §15.
- 2026-10-07: Stage 0 (R18.15–R18.21; `docs/experiments/apple_lewm_c1m_v2_pred_readout_stage0.md`):
  new modules `lewm_pr_v2{,_runtime,_offline}.py` and `scripts/run_lewm_pr_v2.py`, DRAFT manifest,
  31 tests, debug smokes (65900–65999), caps, and R18.13's dry run at `e4babc1` (development, optimistic, no bar;
  seed 66800 flagged `last_two_triggered`; W 56.88/64 offline, N 25.94, W − N +30.9, predictions
  not closed-loop counts; report sha256 `e3becdfa…aed5`). §15 questions 1–5 ruled;
  6 recommended. Next: K0′ on a reviewer's GO, then the freeze.
- 2026-10-07: K0′ ran once on the reviewer's GO (#154, issuecomment-6047218319) at `931281a`:
  **K0′-PASS** (report `outputs/task080-k0-1/report.json`, sha256 `a0939e3e…4c16`; evidence copy
  `~/develop/emai/evidence/task080-k0/`). K′ 32/29/31/14/18/5 of 32 at 0/0.5/1/1.5/2/3 cm; pooled
  K ∪ K′ 64/58/60/32/29/7 of 64; pooled τ_commit = 1.0 cm (not tightened); ceiling 32/32;
  r_K′ = 459; no stop. The freeze PR (R18.22) writes these into the frozen block, sets STATUS
  FROZEN and pins the block, the files and the protocol document. Next: Stage C′ on its own GO.
- 2026-10-08: Stage C′ ran once on the reviewer's GO (#155, issuecomment-6048275092) at `514110d`:
  **CORPUS-SEALED** (R18.23; protocol §8.2). 499 of 500 roots kept (65688, gate-P, excluded as
  `no_decision`; 0.2 % overall, 0.4 % gate-P, 0 % contrast-T; bar 2 %); gate-P 249, contrast-T
  250. Sealed manifest sha256 `deebd83db6de53e23dbde0b921ae7f7c1cf79cb24c6dd27bfae066c2f5017c4e`;
  report sha256 `63b085df…f0ca`; evidence copy `~/develop/emai/evidence/task080-stagec/`. No
  Stage R readout fitted on or evaluated against a fresh root yet (R-plate read their 405 frames by
  design; wording corrected in R18.25). Stage R's plan is §8.3 (R18.24): `tests`, then
  `featurise` through `gpu_run.sh`, then `rgate`, on one reported GO at the record's merge
  commit. The record PR also
  fixes #155's review nits 1–3 (protocol §8.1 and §14 wording, a PLAN.md line); the protocol
  document's sha256 is re-pinned in the manifest.
- 2026-10-08: Stage R ran once on the reviewer's GO (#156, issuecomment-6048908259) at `33cea5c`
  (worktree `task080-stager`): TESTS-PASS, FEATURISED, then **R-PASS** (R18.25; protocol §8.4;
  rgate report sha256 `bedb8966…8ea7`; evidence copy `~/develop/emai/evidence/task080-stager/`).
  Offline only. Gate-P (249 roots), seeds 66800/66801/66802: median e_S 0.547/0.584/0.589 cm
  (upper bounds 0.610/0.657/0.643, R1 bar 1.0); e_S/e_N 0.311/0.371/0.385 (upper bounds < 0.44);
  e_L lower bounds 2.12–2.21 cm (R3 > 1.0; corrected from 2.13 in R18.27); ceiling 0.455 cm.
  Offline predicted counts (predictions, not closed-loop counts; seed 66800,
  `last_two_triggered`): W 56.92/64 (A1 ≥ 56: margin 0.92, under one reset), N 27.79, L-shuf
  19.67, L-mean 28.95, L-rand 11.85; H-rule 58.64, H-sysid 53.93. The
  record also takes #156's review nits (a)–(d). Stage D's plan is §8.5 (R18.26): one CPU
  `closed --cohort D` on 65100–65115 on its own GO at the record's merge commit.
- 2026-10-08: Stage D ran once on the reviewer's GO (#157, issuecomment-6050425196) at `db34adc`
  (worktree `task080-staged`), 01:42:36–01:49:55 UTC: **D-PASS** (R18.27; protocol §8.6; report
  sha256 `1602447a…0611`; evidence copy `~/develop/emai/evidence/task080-staged/`). Counted
  successes on the 16 development resets 65100–65115: W 16/16, N 8/16, L-shuf 5/16, L-mean 9/16,
  H-final(commit) 16/16 (privileged ceiling). W − max twin = +7 (bar +3). W's median aim error
  0.546 cm; no fallback or refusal. A non-gating development result (one run, one model seed,
  simulation only), not the gated one. It is the first LeWM-driven closed loop on v2 whose counts
  are read, so R18.28 updates R7's v2 LeWM clause as a named exception to R18.12 (CLAUDE.md,
  README and the status docs follow). The record also takes #157's review nits 2–4. Stage S's
  plan is §8.7 (R18.29): one CPU `closed --cohort S` on 65200–65263, every arm, with `--stage-d`,
  on its own GO at the record's merge commit.
- 2026-10-08: Stage S ran once on the reviewer's GO (#158, issuecomment-6051252854) at `bfae0fa`
  (worktree `task080-stages`), 02:58:59–03:25:20 UTC: **L-NEAR** (R18.30; report sha256
  `53cfdec1…393e`; evidence copy `~/develop/emai/evidence/task080-stages/`). Counted successes on
  the 64 gated resets 65200–65263 (one run, one model seed 66800 flagged `last_two_triggered`,
  simulation only): W 58/64, N 26, L-shuf 21, L-mean 30, L-rand 11, H-rule 63, H-sysid 62,
  H-final(commit) 62 (privileged); H-read 64 and H-now 1 (privileged, reported only). G-bar passes;
  the four McNemar tests pass (p ≤ 9.7 × 10⁻⁷); G-NI fails against H-rule (W − H-rule −5/64,
  95 % [−10, 0] against −8/64; not detectably inferior). Escalate, no clause, no claim. W's
  determinism re-run matched on all four resets (the log's "3/4" is its success count). R18.31
  updates R7 with these counts (a factual correction, no claim). Results document
  `docs/experiments/apple_lewm_c1m_v2_pred_readout_results.md`; **TASK-080 closes with outcome
  L-NEAR** (R18.32, which records a recommendation only: no repeat of S with this design; any next
  LeWM task on v2 drafted first as a design note). Card moved to done.
