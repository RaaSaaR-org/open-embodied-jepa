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
updated: 2026-10-06
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

**The protocol (FROZEN after K0-PASS, R17.25; in force once merged on an independent APPROVE):**
[`docs/experiments/apple_lewm_c1m_v2.md`](../../../docs/experiments/apple_lewm_c1m_v2.md).
Rulings R17.1–R17.48 (`docs/DECISIONS.md`, decision 2026-10-05 (b)), decided by Claude under
owner delegation (2026-09-30). The plan is `docs/PLAN.md`.

Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7. This task does not
change it unless an L-PASS is followed by its own reviewed ruling.

## Acceptance Criteria
- [x] DRAFT preregistration reviewed independently (this card's first PR; docs plus a
      development cost probe on synthetic data only).
- [x] Stage 0 PR: stage code, frozen block, manifest, tests (seed ranges, `check_budget`,
      `select_checkpoint`/`last_two_triggered`, the `"not evaluated"` sentinel, no runner imports,
      no privileged read in W or a twin), debug smokes and the scale probe.
- [x] K0 on a reported GO (K0-PASS at `306fbdc`); K0's values in the frozen block.
- [x] The freeze merged on an independent reviewer's reported APPROVE (#144, `862d63c`).
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
- 2026-10-05: K0 ran once at `306fbdc` on the #143 reviewer's GO and ended **K0-PASS**
  (`outputs/task077-k0-1/report.json`, sha256 `9be44fd9…f235`): τ_commit = 1.0 cm (32/29/29/18/
  11/2 of 32), ceiling 32/32, r_K = 460, palm speed 0.00227 cm per step; proxies H-now 0, N 13,
  shuf 10, mean 15 of 32. Freeze PR on branch `task077-freeze` under R17.25–R17.29: STATUS FROZEN,
  frozen sha `f28e5e2c…548d` pinned; feature-file check (R17.26); determinism re-run gates the
  reading and the success outcome, target reported (R17.27); Stage O readouts and Stage C scale
  probes on synthetic and debug data, caps kept (R17.28); `OPENBLAS_NUM_THREADS=16` is the
  declared G-threads environment, not a bug (R17.29). Thin margins disclosed: r_K 5 steps inside
  r; 0.5 and 1.0 cm each 1 above the 28/32 bar.
- 2026-10-05: Stage C ran once at `862d63c` (GO on #144) and ended **V on G-memory** (PSS
  12.22 GiB > 12.00, peak 12.896) in seed preparation, before any root was collected
  (`outputs/task077-corpus-1/report.json`, sha256 `734fa771…94a6c3`). Cause: the pinned
  `cohort_estimates` held all 2 000 seeds' float64 DINOv2 tokens plus two `cross_gram` copies
  (about 4.4 GiB). Erratum 2026-10-05 (R17.30–R17.34, branch `task077-memory-fix`): the runner
  streams the tokens (bit-identical estimates; seed preparation 7.75 GiB at 2 000 seeds),
  `featurise_corpus` re-maps its stores (same files), preflight checks the protocol sha
  directly, the load-average key is renamed, `--log` and `outputs/` creation; disclosures of
  `cprobe-3` and the 2.21 load; EGL takes no GPU lock. Frozen block unchanged. Stage C's one
  repeat needs the merged fix and a reported GO.
- 2026-10-05: Stage C's one repeat ran at `4f30fbb` (GO on #145, issuecomment-5990368000) and
  ended **CORPUS-SEALED** (R17.35): 1 995/2 000 roots, 5 train roots excluded as `no_decision`
  (67692, 67888, 68047, 68414, 68774); sealed manifest
  `outputs/task077-corpus-2/corpus/manifest.json` (worktree `task077-corpus2`), sha256
  `ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb`; report sha256
  `2f84515bbdd612c75f56439355eff580f0617f0b5a7e64652a3626653dd07e47`; peak PSS 8.38 GiB; 1 234 s;
  two render disagreements (67133, 68878: 4 px by at most 1 level, states equal). The void run 1
  report (`734fa771…94a6c3`) is kept.
- 2026-10-05: Erratum 2026-10-05 (b) (R17.36–R17.39, branch `task077-stageo-prep`): the
  bit-identity claim corrected (a one-row final chunk differs by about 1e-15; cannot occur here;
  now refused); the featurisation scale probe (`scripts/probe_task077_featurise.py`, synthetic
  1 995-root corpus, the runner's own `featurise` stage through `gpu_run.sh --wait`, at
  `ab5935f`): 212.7 s against the 3 600 s cap, peak PSS 1.76 GiB, GPU 0.90 GiB; the determinism
  re-run's commit target gated at 0.6 cm (R17.38, before Stage S's GO); G-anchor checks the
  frozen 256 frames (R17.39). Disclosed: `task077-fscale-1` was V on G-hash (docs edited in its
  worktree during the run). Frozen block unchanged. Stage O needs this erratum merged and its
  own reported GO.
- 2026-10-05: Stage O ran once at `8721516` (GO on #146, issuecomment-5991796440) from the clean
  worktree `task077-stageo` and ended **O-PASS** (R17.40): O1 c_plate 0.404 cm [0.385, 0.417] ≤ 1.0;
  O3 prior ratio 0.153 [0.145, 0.160]; O4 plate-hidden 4.579 [4.443, 4.673] > 1.0; learning curve
  still falling (reported). Reports (sha256): tests `80fc8344…84b4`, featurise `f187c7c8…e9ee`,
  readouts `452045d2…8b78`; fits (content): moments `5415eea4…`, R8 `62a5ea8a…`, R-plate
  `08bde901…`, mean latent `85da6336…`. Disclosed (R17.41): the featurise command lacked
  `--min-free-gib 8 --board` (lock held; runner's 8 GiB guard held; board not updated) and its
  report has no peak GPU memory.
- 2026-10-05: Stage T prep (R17.41–R17.44, branch `task077-staget-prep`): the runner records
  `gpu_memory` for GPU stages; finds Stage O's fits by name in `--fits` and job checkpoints beside
  their reports (Stage O's recorded paths were relative to its worktree); R17.38 reworded as an
  amendment; W's CPU roll-out measured bit-identical across 7 processes × 2 passes
  (`task077-rdet-1`, `5f94100b…6fde`; debug and random-init models, synthetic commands, 4 debug
  roots, no stand-in chunks; not the trained W); Stage T scale re-probed at 50 kept states
  (`task077-tscale-1`, `b409e339…2e9d`: 0.169 s per update, 13.09 GiB PSS, 5.48 GiB GPU);
  Stage T's eight job commands written out (protocol §7.6), about 7.6–44.5 h of GPU. Stage T needs
  this PR merged and its own reported GO.
- 2026-10-05: independent review of #147 at `60b6722`: REQUEST CHANGES (one blocking: §7.6's
  script did not stop on CAL-T-ESCALATE, which exits 0, nor check `tests`). R17.45: the commands
  are now `scripts/run_task077_staget.sh` (`set -euo pipefail`; TESTS-PASS, T-JOB-DONE and
  T-PLANNED required after each step), tested with a fake runner and `gpu_run.sh`; the
  determinism probe's limits added to PLAN.md and this log; `sysid.json` noted as existence-checked,
  not read.
- 2026-10-06: Stage T record and pause (R17.46–R17.48, protocol §7.7, branch
  `docs/task077-staget-pause`). Stage T started at `215fcce` on the GO
  (#147, issuecomment-5993481678) in worktree `task077-staget`. tests TESTS-PASS (`928277d3…e761`);
  cal-W T-JOB-DONE (kept 46 000, val 0.372733; `2339f5f1…0cb8`); cal-N T-JOB-DONE (kept 38 000,
  0.736775; `42ad6cf8…4423`); plan T-PLANNED (u_sat 46 000 / 38 000, U = 95 000, every 4 750; rank
  reference 0.254; `7c714760…7cdfb`); W-66800 T-JOB-DONE (95 000 updates, 18 405 s, kept 95 000,
  0.350399, `last_two_triggered` true; `42195ded…757b`). N-66800 was started automatically by the
  driver and stopped by SIGTERM at the owner's request to pause and free the GPU (22:20:41Z):
  **V**, report `bb2c442a…3355`, voiding that job only. Prevention (R17.47): the driver's pause
  file, `--stop-after`, resume mode with sha256 checks of kept jobs, `--repeat` into a new folder
  (`-2`), and a refusal after a second V; tested with a fake runner. Five jobs remain (about
  25.6 h). N-66800's repeat is its last; a second V ends TASK-077 INCONCLUSIVE. Nothing has been
  launched; the owner wants the GPU free. No result is claimed.
