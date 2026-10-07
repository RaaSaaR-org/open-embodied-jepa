# Apple→Plate LeWM committed aim under C1-M: results (TASK-077)

**Outcome: G-NO-BAR at Stage G, with Stage T's row T-DONE.** Escalate, no clause (§8.2 of the
protocol). **Stage D and Stage S did not run**, so no closed-loop arm of §5 has a count. **The
abandonment clause (§11) does not fire**, and nothing in its scope is closed. TASK-077 closes
here, at protocol step 10, with outcome G-NO-BAR (DECISIONS R17.52).

- **What passed (offline, gate split, h = 60, all three model seeds).** G1–G4: the 8 × 8 LeWM token
  predictor W's 60-step roll-out from the encoded 405 frame is not collapsed, is action-sensitive,
  and beats both copy-last and an equally trained no-action model N. This is the first passed
  gate of 8 × 8 dynamics at any horizon in this repository. It is an offline dynamics result only, not a
  control result.
- **What failed.** G5 (a), on all three seeds: R8, the dual-ridge plate readout fitted on encoded
  frames at r = 465, reads the plate from W's *predicted* latent at r with a median error of
  **1.581, 1.617 and 1.873 cm** (upper 95 % bounds 1.697, 1.794 and 2.076 cm), against
  τ_commit = 1.0 cm. The same readout on the *encoded* frame at r reads 0.411 cm on the same
  roots. G5 (b), W against N, passes on all three seeds.
- **The closed loop would have faced a larger error.** On the stand-in commands that W would rank
  in closed loop, the same statistic is 3.63, 3.09 and 3.16 cm, worse than N's predicted plate on
  every seed (reported only).
- **No LeWM-driven controller has run in closed loop on v2.** The offline aims' predicted counts
  (W 22.6, N 25.4 of 64; §4) are predictions from offline aim errors through K0's τ curve, not
  closed-loop counts, and they gate nothing.

**What this does not show.** Nothing about a closed-loop count of any arm; nothing about LeWM as
a controller; nothing about other readouts of the predicted latent, other horizons, commit steps
or grids (the clause did not fire, so all of them stay open, §11); nothing about v1, Arena or the
real G1. It is one run of each stage, on one corpus (`apple-c1m-v2`), one camera (onboard 112 px)
and one frozen encoder (DINOv2 ViT-S/14), in simulation.

**The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged and still true**
(R17.14, R17.52): v1 is still 0/150 per backend, P-3's M2 result is as stated, no LeWM-driven
controller has run in closed loop on v2, and privileged, scripted, oracle and GR00T successes
are not project-learned results.

Protocol: [`apple_lewm_c1m_v2.md`](apple_lewm_c1m_v2.md) (STATUS FROZEN; frozen block
`src/embodied_jepa/lewm_c1m_v2.py`, sha256
`f28e5e2cd23d110f40ff043c7308e0bb9b3b71f46a2a4b6940bc536cc5e3548d`, unchanged through every
stage). Stage 0 record: [`apple_lewm_c1m_v2_stage0.md`](apple_lewm_c1m_v2_stage0.md). Manifest:
`benchmarks/manifests/apple-lewm-c1m-v2.json`. Rulings: R17.1–R17.52 in
[DECISIONS.md](../DECISIONS.md), decision 2026-10-05 (b), each decided or recorded by Claude under
owner delegation (2026-09-30). The stage-by-stage records are protocol §7.1–§7.9; this document
restates their numbers and adds no measurement.

---

## 1. Stage-by-stage record

Every stage ran from a fresh, clean worktree of a merged revision, on an independent reviewer's
reported GO, after a full `pytest` at that revision (G-tests). Reports are git-ignored and stay
on the Linux PC (RTX 5080). Times are UTC.

| stage | revision | worktree (under `/home/huhn/develop/emai/worktrees/`) | report | report sha256 | outcome |
|---|---|---|---|---|---|
| K0 (before the freeze) | `306fbdc` | `task077-k0` | `outputs/task077-k0-1/report.json` | `9be44fd93996c03cacc6cde089225f0f986676020f2ceb2138f062131f1af235` | **K0-PASS** |
| C, run 1 | `862d63c` | `task077-corpus` | `outputs/task077-corpus-1/report.json` | `734fa771…94a6c3` | **V** (G-memory, before any root; protocol §7.2) |
| C, the one repeat | `4f30fbb` | `task077-corpus2` | `outputs/task077-corpus-2/report.json` | `2f84515bbdd612c75f56439355eff580f0617f0b5a7e64652a3626653dd07e47` | **CORPUS-SEALED** |
| O, G-tests | `8721516` | `task077-stageo` | `outputs/task077-o-tests-1/report.json` | `80fc8344b128e03f7604481364ead13bba1fbba2bef0d535a0b950df3e3784b4` | TESTS-PASS |
| O, featurisation (GPU) | `8721516` | `task077-stageo` | `outputs/task077-featurise-1/report.json` | `f187c7c8179ffe0d1883739be0f30ae960c6033e4c1d088d800075490ef3e9ee` | FEATURISED |
| O, readouts | `8721516` | `task077-stageo` | `outputs/task077-readouts-1/report.json` | `452045d22c21b067c8bfe78cb72f4e842fc61f3c44a9e17b6fae860c244f8b78` | **O-PASS** |
| T, run 1 (tests, cal-W, cal-N, plan, W-66800) | `215fcce` | `task077-staget` | protocol §7.7's table | §7.7 | TESTS-PASS (2014 passed), T-JOB-DONE ×3, **T-PLANNED** |
| T, N-66800 attempt 1 | `215fcce` | `task077-staget` | `outputs/task077-t-N-66800-1/report.json` | `bb2c442a3a610ea89c10ad22d089a58b3d27d9e6df22d2c02a16b9f3a7ec3355` | **V** (SIGTERM at the owner's pause; protocol §7.7) |
| T, resume (N-66800-2, W/N-66801, W/N-66802) | `2aa4f5b` | `task077-staget2` | protocol §7.8's table | §7.8 | TESTS-PASS (2035 passed), T-JOB-DONE ×5 |
| G | `1cb5f79` | `task077-stageg` | `outputs/task077-gates-1/report.json` | `dbd9e71414aeec4bd2710c17586bb4bcd59bc7e137c6fd3807da19cac2e15727` | **G-NO-BAR**; Stage T **T-DONE** |
| D, S | — | — | — | — | **not run** (G-NO-BAR) |

Two Vs happened, each in a different stage, each repeated once after a recorded cause and fix,
and neither repeated stage voided again (§10.1; Vs in different stages do not add up, R17.16).
No stage was repeated for its result.

### 1.1 K0: K0-PASS (protocol §7.1)

On K's 32 resets (66000–66031): **τ_commit = 1.0 cm** (H-final(commit)'s aim plus a planted
error of 0, 0.5, 1, 1.5, 2, 3 cm: 32, 29, 29, 18, 11, 2 of 32), ceiling N_K(0) = 32/32,
**r_K = 460** against the frozen r = 465, median palm speed at 405 0.00227 cm per step. The
privileged proxies (reported): H-now 0, N-proxy 13, shuf-proxy 10, mean-proxy 15 of 32.

### 1.2 Stage C: CORPUS-SEALED (protocol §7.2–§7.3)

1 995 of 2 000 roots kept (5 train roots excluded as `no_decision`, 0.25 % against the 2 % bar):
**1 495 train, 250 val, 250 gate**. Sealed manifest sha256
`ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb`. The corpus is the privileged
scripted collector's (the aim drawn over the box from the true plate and palm); its outcomes are
not learned results.

### 1.3 Stage O: O-PASS (protocol §7.5)

Cross-fitted over the 1 745 train + val roots (5 outer folds):

| gate | measured (cm, 95 % interval) | bar | result |
|---|---|---|---|
| O1: c_plate, the 8 × 8 readout at r | 0.404 [0.385, 0.417] | upper ≤ τ_commit = 1.0 | pass |
| O3: c_plate / constant prior (2.644 cm) | 0.153 [0.145, 0.160] | upper < 1.0 | pass |
| O4: plate-hidden renders at r | 4.579 [4.443, 4.673] | lower > 1.0 | pass |

Reported only: the learning curve was **still falling** (0.625, 0.515, 0.446, 0.404 cm at 1/4,
1/2, 3/4 and all of each fold's fit roots); the plate moves 8.43 cm from 405 to r. Two process
anomalies were disclosed and changed no number (R17.41: the featurise command lacked
`--min-free-gib 8 --board`; no peak GPU memory was recorded).

### 1.4 Stage T: all eight jobs T-JOB-DONE; row T-DONE (protocol §7.7–§7.8)

The plan (T-PLANNED): u_sat(W) = 46 000, u_sat(N) = 38 000, so **U = 95 000 updates**, a
selection every 4 750 (20 points), below the 100 000 cap; G1's bars B_rank 0.12 and B_std 0.38
(rank reference 0.254 from cal-W on val), not raised by the truncation controls (the rank-4
truncation missed the rank bar by 0.0013).

| seed | arm | kept update | kept val criterion | raw argmin (val) | `last_two_triggered` | checkpoint sha256 |
|---|---|---|---|---|---|---|
| 66800 | W | 95 000 | 0.350399 | 95 000 (0.350399) | **true** | `891d2664…76b8` |
| 66800 | N | 38 000 | 0.741328 | 38 000 | false | `51b51035…d2c2` |
| 66801 | W | 71 250 | 0.364904 | 95 000 (0.361504) | false | `27aeadab…c193` |
| 66801 | N | 80 750 | 0.729304 | 80 750 | false | `15f5dd41…2d37` |
| 66802 | W | 80 750 | 0.364660 | 95 000 (0.363314) | false | `ba2614ec…97b4` |
| 66802 | N | 38 000 | 0.746241 | 47 500 (0.742907) | false | `4f92a8fe…bfef` |

The val criterion is a selection criterion and a raw latent MSE, not a success measure, and W's
and N's values are not comparable as a success claim. Stage T's row **T-DONE** is recorded in
Stage G's report (`stage_t.row`, `decide_t`, R17.24).

## 2. Stage G: G-NO-BAR (protocol §7.9, §8.2)

One run at `1cb5f79`, 2 369 s against the 7 200 s cap, peak process-tree PSS 8.79 GiB of 12 GiB,
G-tests 2035 passed and 37 skipped on a clean tree. **Primary seed 66800** by §7.6's rule (the W
with the lowest kept val criterion). **W-66800's `last_two_triggered` is true** and is stated
beside every seed-66800 W-versus-N comparison below (marked †). No N is flagged
(`n_unsaturated` false on all three seeds).

### 2.1 The gates (gate split, 250 roots, window set E60, h = 60)

Bootstrap 95 % intervals over gate roots (salt 8106). "point" is the statistic on all 250 roots.

| gate (bar) | 66800 † | 66801 | 66802 | passes |
|---|---|---|---|---|
| G1 (i) collapsed fraction (≤ 0.05) | 0.0 | 0.0 | 0.0 | all |
| G1 (ii) effective-rank ratio (≥ 0.12) | 0.283 | 0.272 | 0.273 | all |
| G1 (ii) std ratio (≥ 0.38) | 0.776 | 0.754 | 0.756 | all |
| G1 (iii) rank ratio W − N: point; interval (lower > 0) | 0.140; [0.142, 0.167] | 0.101; [0.097, 0.124] | 0.158; [0.159, 0.189] | all |
| G2 MSE(W)/MSE(copy-last): point; upper (≤ 0.8) | 0.143; 0.148 | 0.146; 0.151 | 0.146; 0.151 | all |
| G3 MSE(W)/MSE(N): point; upper (< 1.0) | 0.359; 0.377 | 0.382; 0.404 | 0.384; 0.403 | all |
| G4 wrong/true: point; lower (≥ 1.10) | 3.84; 3.62 | 3.71; 3.49 | 3.71; 3.50 | all |
| G4 zero/true: point; lower (≥ 1.10) | 6.52; 6.30 | 16.07; 15.64 | 6.10; 5.93 | all |
| **G5 (a) R8 on W's predicted latent at r, median (upper ≤ 1.0 cm)** | **1.581 [1.448, 1.697]** | **1.617 [1.506, 1.794]** | **1.873 [1.693, 2.076]** | **none** |
| G5 (b) median e_W / median e_N: point; upper (< 1.0) | 0.652; 0.716 | 0.706; 0.804 | 0.726; 0.798 | all |

Effective ranks at h = 60: W 7.21, 6.94 and 6.96; N 3.63, 4.35 and 2.93; encoded 25.52. G5 fails
on part (a) alone, on every seed, so the row is G-NO-BAR (G1–G4 pass on all seeds, G5 fails on
at least one). `clause_fires` is false.

### 2.2 Reported only (not gated)

- **The encoded readout on the gate split** (R8 on the encoded frame at r): median 0.411 cm
  [0.360, 0.452], 87.5th percentile 0.77 cm. **W minus it:** +1.170 [1.034, 1.296], +1.206
  [1.096, 1.399] and +1.462 [1.278, 1.663] cm.
- **N's predicted plate at r** (e_N): 2.426 [2.205, 2.603] †, 2.289 and 2.581 cm.
- **G5 (a)'s 87.5th percentiles:** 3.28, 3.07 and 3.29 cm.
- **The scaled tolerance 1.5 τ_commit = 1.5 cm** (§8.2) is missed too (upper bounds 1.70, 1.79
  and 2.08 cm).
- **G5 on the stand-in chunks** (W rolled under the kinematic stand-in's 60 commands for each gate
  root's own committed aim, the commands W ranks in closed loop): 3.625 [3.247, 3.996] †, 3.085
  [2.855, 3.318] and 3.164 [2.852, 3.405] cm. **Stand-in minus executed:** +2.044 [1.729, 2.379],
  +1.469 [1.233, 1.668] and +1.291 [0.956, 1.548] cm. Under the stand-in, W's median
  predicted-plate error is above N's (2.43 †, 2.29, 2.58 cm) on every seed, and W's normalised
  latent MSE at h = 60 is too (1.569 †, 1.727, 1.899 against N's 1.047, 1.003, 0.995).
- **G2–G4 at h = 16 and 30** pass their bars at both horizons on every seed (G2 upper ≤ 0.204;
  G3 upper ≤ 0.634; G4 lower ≥ 1.647). W's normalised MSE on the true commands barely changes
  with the horizon (h = 16, 30, 60): 0.355, 0.350, 0.376 (66800); 0.376, 0.373, 0.383 (66801);
  0.374, 0.370, 0.383 (66802). N's grows from 0.62–0.67 to 1.00–1.05, copy-last's from 1.89 to
  2.62. No plate reading exists at h = 16 or 30: R8 is fitted at r only.
- **W-66801's zero-command ratio is about 2.5 × the others'** (G4 zero/true 16.07 against 6.52
  and 6.10; W's normalised MSE with zero commands 6.15 against 2.45 and 2.33). This is reported,
  with no reading of its cause; it passes G4 like the others.

## 3. Caveats

1. **W-66800's `last_two_triggered` is true (†).** Its kept point is the last of its 20 (95 000;
   the previous point read 0.362693 at 90 250), so by the saturation flag its curve may still have
   been falling at U. By §4.5 the flag changes no budget and no row, and §4.5's reporting duty is
   written for an N. It bears on every seed-66800 W-versus-N comparison: G1 (iii), G3, G5 (b),
   the stand-in comparison with N, and the offline aims (primary seed 66800). An under-trained W
   would, if anything, weaken W against N rather than favour it; that is a reading, not a
   measurement. None of the seed-66800 passes rests on it alone: G1 (iii), G3 and G5 (b) pass on
   66801 and 66802 too, whose W flags are false.
2. **All three W curves have their raw minimum at the last point, 95 000.** W-66801's and
   W-66802's flags are false only because the 1 % rule kept earlier points (71 250 and 80 750);
   their last points are lower by 0.0034 and 0.0013. So all three W models may have been slightly
   under-trained at U. No rule extends U (it was fixed by the plan from the calibration runs), and
   nothing here measures whether a longer budget would move G5.
3. **N's offline predicted count is above W's** (25.4 against 22.6 of 64, primary seed 66800 †;
   N's median aim error 2.113 cm against W's 2.346, intervals overlapping). This is reported
   only, from offline aim errors mapped through K0's τ curve, not from closed-loop attempts. One
   reading, not a measurement: N's prediction does not depend on the commands, so the stand-in's
   command mismatch costs W and not N (§2.2).
4. **G1 (iii)'s point estimate lies below its own bootstrap interval on two seeds** (66800: 0.140
   against [0.142, 0.167]; 66802: 0.158 against [0.159, 0.189]). This is a property of the
   percentile bootstrap applied to an effective rank, read from the code
   (`lewm_c1m_v2_offline.comparative_rank`), not a fault in the run:
   - each resample draws 250 gate roots with replacement, so it holds about 63 % distinct roots
     and duplicates the rest. Duplicated rows re-weight the centred Gram matrix, and the
     effective rank (the exponential of the spectral entropy) is a nonlinear function of its
     spectrum;
   - the three ranks (W, N, encoded) respond differently to that re-weighting, so the centre of
     the resampling distribution of (W − N)/encoded is shifted from the full-sample value, and the
     plain percentile interval does not correct for such a shift (no bias correction is declared
     in §8.2). Here the shift happens to lie above the point estimate;
   - **it does not change the gate.** G1 (iii) passes on the declared statistic, the lower bound,
     on all three seeds (0.142, 0.097, 0.159), and the point estimates (0.140, 0.101, 0.158) are
     also well above 0 on every seed. Nothing is re-computed or re-thresholded here.
5. **The gate split has now been read.** Its 250 roots are no longer fresh for these six models,
   R8 or any choice informed by §2. Any later gated test of a remedy needs fresh roots.
6. **The kept checkpoints were selected on val**, and G1's bars were calibrated on val. Any later
   development measurement on val with these models (R17.51) is not out of sample for selection.
7. **Thin margins carried from K0** (§7.1): r_K = 460 against r = 465 (5 steps); the 0.5 and 1.0 cm
   levels of the τ curve each sit one success above the 28/32 bar, so τ_commit = 1.0 cm is an
   upper reading of the tolerance on 32 resets. A τ_commit of 0.5 cm would have halved G5 (a)'s
   bar; G5 (a) fails at 1.0 cm already.
8. **One run, one corpus, one camera, one encoder, in simulation**, at the frozen budget. The
   readout R8 is the one Stage O fitted on encoded train frames; it was not refitted on predicted
   latents (that would be a change after seeing the numbers, so it is the subject of the
   decomposition record R17.51 recommends, on train and val only).

## 4. Offline aims (§7 step 7; reported only, not closed loop)

Primary seed 66800 († W-66800 flagged); the same controller code as the closed loop, on the 250
gate roots' logged 405 states, with the stand-in chunks; p̂ from R-plate on Stage O's CUDA full
tokens at 405; no fallbacks. The aim error is against the rule's fixed point
g\* = (p − κh)/(1 − κ). The predicted count maps each aim error through K0's τ curve (32, 29, 29,
18, 11, 2 of 32 at 0, 0.5, 1, 1.5, 2, 3 cm).

| arm | median aim error (cm) [95 %] | 87.5th pct | clip-binding fraction | predicted count of 64 |
|---|---|---|---|---|
| W | 2.346 [2.123, 2.570] | 3.31 | 0.216 | 22.6 |
| N | 2.113 [1.905, 2.238] | 3.21 | 0.104 | 25.4 |
| L-shuf | 2.756 [2.559, 2.928] | 3.57 | 0.260 | 19.1 |
| L-mean | 2.612 [2.420, 2.810] | 3.54 | 0.252 | 19.8 |
| L-rand | 6.002 [4.867, 7.141] | 13.64 | 0.0 | 9.6 |

Every arm's prediction is far below the 56/64 bar. The privileged arms (H-final(commit), H-read,
H-now) and the hand-written arms (H-rule, H-sysid) have no offline aim and no count here: they run
only in Stage D and S, which did not run. **No arm of §5 has a closed-loop count under this
protocol.**

## 5. Rows and what follows

| stage | row | consequence (protocol §7, §8) |
|---|---|---|
| K0 | K0-PASS | K0's values entered the frozen block; the freeze |
| C | V, then CORPUS-SEALED | the sealed corpus feeds every later stage |
| O | O-PASS | Stage T may run on its GO |
| T | T-DONE (decided in Stage G) | the six models enter Stage G |
| G | **G-NO-BAR** | **escalate, no clause**; Stage D and S do not run (the runner accepts only a G-PASS gates report for them) |
| D, S | not run | no closed-loop row (L-PASS, L-NO-GAIN, L-INFERIOR, …) exists |

**The clause does not fire.** §11 fires only on L-NO-GAIN or L-INFERIOR. Nothing in its scope is
closed: every readout of the predicted latent (trained readout heads included), every pooled grid,
other commit steps and horizons, history longer than one, `action_chunk` and
`predictor_step_embedding`, the full token grid, another encoder, and a declared task change all
stay open. Never closed by this task: the LeWM backend, DINOv2 as an encoder, v2, Arena and the
product goal.

**What follows** (R17.51, a DRAFT recommendation, and R17.52). §8.2 says only "escalate, no
clause". The binding constraint measured here is the plate's readability from W's prediction at
r after 60 recursive steps (0.41 cm encoded against 1.58–1.87 cm predicted), with a stand-in
command mismatch of similar size on top (+1.29 to +2.04 cm). R17.51 recommends an offline,
CPU-only decomposition record on the train and val splits with the six kept checkpoints, with a
decision rule declared before it runs, choosing among a readout on predicted latents, a
command-matching change and a declared task change. That record is outside TASK-077 and needs no
new training. TASK-079's precondition (TASK-077 L-PASS) is not met.

## 6. Process

- The DRAFT (#142) and Stage 0 (#143) were revised after independent REQUEST CHANGES reviews;
  K0 ran on #143's GO; the freeze (#144) merged on an independent APPROVE; Stages C, O, T and G
  each ran on their own reported GOs (#144–#149), each from a clean worktree of a merged revision.
- Stage C's V (G-memory) and N-66800's V (a stop signal at the owner's pause) were each followed
  by a recorded cause and fix and one repeat (Erratum 2026-10-05, R17.30–R17.34; R17.46–R17.48).
- After the freeze, the frozen block and its sha256 never changed. Erratum 2026-10-05 (b) added
  one post-freeze amendment (R17.38: the determinism re-run's commit target gated at 0.6 cm, a
  tightening for Stage S, which did not run).
- Nothing was re-thresholded, refitted, retrained or re-selected after any number was seen. No
  experiment was run for this document: every number is read from the reports above and the
  protocol's records (§7.1–§7.9).
- Evidence copies: Stage T's two worktrees and Stage G's report, log and stdout are in the SSD
  evidence store `~/develop/emai/evidence/` (`task077-staget/`, `task077-staget2/`,
  `task077-stageg/`, with sha256 manifests under `_checksums/`). The run worktrees stay in place
  as the originals.
