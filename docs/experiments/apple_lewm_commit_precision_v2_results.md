# Apple→Plate LeWM committed aim under C1-M, solved as the fixed point of a local affine fit: results (TASK-081)

**Outcome: L-PASS at Stage S.** "G-bar, G-NI and the four McNemar tests pass: the primary claim,
'LeWM-driven closed-loop success'" (protocol §7.2). **The abandonment clause (§9) does not fire.**
The secondary claim, **"LeWM needed", is not shown**: the non-learned rule controller H-rule, which
is given the simulator's plate law, scored higher than W, and measurably so (the upper bound of the
paired interval of W − H-rule is below 0). **The solver change's contribution is not shown either**
(W − W-frozen +1/128, reported only). TASK-081 closes here with outcome L-PASS (DECISIONS
R19.20–R19.23, decided by Claude under owner delegation).

On the gated cohort S (128 fresh resets, 70200–70327; one run, one model seed 66800, whose
`last_two_triggered` flag is set; in simulation only; under the declared simulation-only plate
condition C1-M), P-3's learned pick (behaviour cloning, not a world model) and e9's scripted place
primitive are the same in every arm; the arms differ only in the single place aim committed at step 405.
W, the LeWM 8 × 8 token predictor W-66800 on frozen DINOv2 features, read by R-S (a ridge
fitted on W's own predicted latents under the kinematic stand-in's commands), with its committed
aim solved by **affine_local** (the clipped fixed point of a least-squares affine fit to W's own
predictions over up to 25 grid candidates near the grid argmin), chose that aim:

- **What passed.** G-bar: **W 118/128** (exact 95 % interval 0.861–0.962) ≥ 112/128. G-NI: against
  the better of the two hand-written non-world-model arms, H-rule (125/128; H-sysid 121/128), the
  paired difference W − H-rule is **−7/128**, discordant 1 (W only) / 8 (H-rule only),
  reset-bootstrap 95 % interval **[−13, −2]/128**; its lower bound is above −δ = −16/128 (δ an
  allocation, 12.5 points carried as a fraction from TASK-080, not calibrated), so non-inferiority
  within δ holds. The four twin tests: W beat its action-blind twin N (73/128), its scene-blind
  twins L-shuf (39/128) and L-mean (59/128) and a random grid aim L-rand (20/128), each by an exact
  one-sided McNemar test at p < 0.01 (p from 8.0 × 10⁻²⁹ to 2.8 × 10⁻¹⁰). W's controller read no
  task truth at run time (0 reads on all 128 attempts).
- **What it does not show.** "LeWM needed": W was not better than H-rule (1 reset against 8,
  exact one-sided McNemar p = 0.998); **H-rule is better than W on this cohort** (the interval
  [−13, −2] excludes 0), and the privileged look-ahead ceiling H-final(commit) scored 126/128. That
  the new solver mattered: TASK-080's W unchanged (W-frozen, reported only) scored **117/128** on
  the same resets, which would itself have cleared the 112/128 bar; W − W-frozen = +1/128
  (discordant 6 / 5, 95 % [−5, +7], p = 0.5). Post hoc (computed for this record, not
  preregistered, gating nothing): W-frozen − H-rule = −8/128, 95 % [−14, −2], whose lower bound is
  also above −16/128 (§3.3).
- **The determinism re-run passed.** W's re-run on 70200–70203 matched on all four resets
  (reading and commit-target differences exactly 0.0, outcomes identical); the log's "4/4" is the
  re-run's success count.

**What this is.** The first gated "LeWM-driven closed-loop success" in the protocol's sense (R9.8)
on `apple-to-plate-v2`: under C1-M, a LeWM predictor's choice of the single committed place aim
reached the preregistered bar, was non-inferior within an allocated 12.5-point margin to the best
hand-written arm, and was used (it beat its action-blind, scene-blind and random twins).

**What this is not.** It is not a LeWM policy: LeWM chooses only the place aim, once, at step 405;
the pick is P-3's and the place primitive is e9's. It does not show that LeWM is needed or that it
beats hand-written control (H-rule is measurably better here). It does not show that the
affine_local solver caused the pass (§3.3). It says nothing about v1's 0/150, Arena, the real G1,
other conditions, other cameras, other model seeds, or whether DINOv2 pretraining matters (no
random-init floor ran for W). TASK-080's gated Stage S (58/64, **L-NEAR**) stands as recorded and is
not re-read: this is a different cohort, and the 128-reset size was chosen after TASK-080's L-NEAR.

**The canonical status sentence** (DECISIONS 2026-10-02, R7) is changed by **R19.21**, the reviewed
ruling an L-PASS needs (R18.12, R19.10): it now states this gated success with its qualifiers (§5).

Protocol: [`apple_lewm_commit_precision_v2.md`](apple_lewm_commit_precision_v2.md) (STATUS FROZEN;
frozen block `src/embodied_jepa/lewm_cp_v2.py`, sha256
`77ccc6364624c7da9dcca1d09827a0f6a25a670a00392d488e7b0216e090705a`, unchanged through every
stage; Stage S's result is its §6.7). Stage 0 record:
[`apple_lewm_commit_precision_v2_stage0.md`](apple_lewm_commit_precision_v2_stage0.md). Design
note: [`apple_lewm_commit_precision_v2_design.md`](apple_lewm_commit_precision_v2_design.md).
Manifest: `benchmarks/manifests/apple-lewm-cp-v2.json`. Rulings: R19.1–R19.23 in
[DECISIONS.md](../DECISIONS.md), each decided or recorded by Claude under owner delegation
(2026-09-30). TASK-080's results:
[`apple_lewm_c1m_v2_pred_readout_results.md`](apple_lewm_c1m_v2_pred_readout_results.md).

---

## 1. Stage-by-stage record

Each stage ran once, from a fresh, clean worktree of a merged revision, on an independent
reviewer's reported GO, after a full `pytest` at that revision (G-tests). No stage was voided or
repeated. No calibration stage, corpus or offline gate ran (R19.4). Reports are git-ignored and stay
on the Linux PC (RTX 5080), with copies in the SSD evidence store. Times are UTC.

| stage | revision | worktree (under `/home/huhn/develop/emai/worktrees/`) | report | report sha256 | outcome |
|---|---|---|---|---|---|
| D (development, 16 resets) | `2a6b633` | `task081-staged` | `outputs/task081-d-1/report.json` | `b58ae61f4751ada0eda3501e0971f2f5367bab2f8f798656172dfa540be1f920` | **D-PASS** (protocol §6.5, R19.17) |
| S (gated, 128 resets) | `e9c294f` | `task081-stages` | `outputs/task081-s-1/report.json` | `ded7e167dafac35a531f58042a4fe95c5dd8ccebf225dd91955b8734230cd097` | **L-PASS** (§6.7, R19.20) |

**Stage S's run.** GO: #164, issuecomment-6055011674. One `closed --cohort S` invocation on the CPU
with `--stage-d` set to D's report (`b58ae61f…e920`, checked by the runner and by the GO), 07:35:43–
08:31:01 UTC (3 318 s, cap 21 600; first outcome 07:39:17). In-run G-tests at `e9c294f`: 2123
passed, 37 skipped, exit 0, tracked tree clean at start and end. Peak process-tree PSS 9.76 GiB
(cap 12; RSS 11.18 GiB); disk free ≥ 62.9 GiB; 4 workers; G-quiet met without waiting. G-repro
(170 decoded train roots, test split not decoded): all eight checks true, **no** render or re-render
disagreement. Frozen-block pin, protocol-document check (`6b0f832b…3022`) and TASK-080's document
check (`bb1b2772…62bd`) matched; the solver file's sha256 `ba3c8d03…91f1` matched its pin. Evidence
copy: `~/develop/emai/evidence/task081-stages/` (report, log, empty stdout; manifest
`_checksums/task081-stages.sha256`, sha256 `0ec44911…e318`; verified against source and copy).

---

## 2. Stage S: every arm (n = 128)

Counted successes on 70200–70327 with exact (Clopper–Pearson) 95 % intervals, and each arm's paired
difference from W (W − arm, in resets of 128) with discordant counts and the reset-bootstrap 95 %
interval (10 000 resamples, salt 8302). One reset, **70302**, was refused before step 405 in every
arm (`guard_refusal` at step 223, before any grasp; the same refusal in all eleven arms) and counts
as a failure in each, as in TASK-080.

| arm | what it is | successes | exact 95 % | W − arm | W only / arm only | 95 % interval |
|---|---|---:|---|---:|---|---|
| **W** | LeWM W-66800 + R-S, affine_local | **118** | 0.861–0.962 | – | – | – |
| W-frozen | TASK-080's W unchanged (reported only) | 117 | 0.851–0.956 | +1 | 6 / 5 | [−5, +7] |
| N | action-blind twin (R-N), affine_local | 73 | 0.480–0.657 | +45 | 51 / 6 | [+32, +58] |
| L-shuf | scene-blind twin (shuffled), affine_local | 39 | 0.226–0.392 | +79 | 83 / 4 | [+66, +91] |
| L-mean | scene-blind twin (mean latent), affine_local | 59 | 0.372–0.551 | +59 | 65 / 6 | [+46, +72] |
| L-rand | uniform feasible grid aim, no solver | 20 | 0.098–0.231 | +98 | 99 / 1 | [+88, +107] |
| **H-rule** | hand-written rule, given the plate law; not learned | **125** | 0.933–0.995 | **−7** | **1 / 8** | **[−13, −2]** |
| H-sysid | hand-written, fitted plate law; not learned | 121 | 0.891–0.978 | −3 | 6 / 9 | [−11, +4] |
| H-final(commit) | privileged look-ahead ceiling | 126 | 0.945–0.998 | −8 | 1 / 9 | [−14, −2] |
| H-read | privileged (reported only) | 117 | 0.851–0.956 | +1 | 10 / 9 | [−8, +10] |
| H-now | privileged, plate now (reported only) | 6 | 0.017–0.099 | +112 | 112 / 0 | [+104, +119] |

### 2.1 The gated rows (§7.2), first match

| check | value | result |
|---|---|---|
| V (void rule) | no guard, crash, cap or stop; determinism re-run matched | not V |
| S-VOID-CEILING | H-final(commit) 126 ≥ 112 | does not hold |
| L-NO-GAIN | every twin test passes (below) | does not hold |
| L-INFERIOR | upper bound of W − C = −2 > −16 | does not hold |
| **L-PASS** | G-bar 118 ≥ 112; G-NI lower bound −13 > −16 (C = H-rule, 125 > 121); four McNemar tests at p < 0.01 | **holds** |

| twin test (exact one-sided McNemar) | W only / twin only | p | passes (p < 0.01) |
|---|---|---:|---|
| G-N (W > N) | 51 / 6 | 2.84 × 10⁻¹⁰ | yes |
| G-shuf (W > L-shuf) | 83 / 4 | 1.51 × 10⁻²⁰ | yes |
| G-mean (W > L-mean) | 65 / 6 | 6.66 × 10⁻¹⁴ | yes |
| G-rand (W > L-rand) | 99 / 1 | 7.97 × 10⁻²⁹ | yes |

**Secondary, "LeWM needed"** (reported only, never a gate): W > C by an exact one-sided McNemar test,
W only 1, H-rule only 8, **p = 0.998: not shown.** Beyond "not shown", the paired interval of
W − H-rule, [−13, −2], lies wholly below 0: on this cohort W is **measurably below** H-rule (by
about 2 to 13 resets of 128), while still inside the allocated non-inferiority margin of 16. H-rule
failed on 70247, 70280 and 70302; W failed on 70211, 70222, 70227, 70233, 70242, 70247, 70276,
70285, 70302 and 70312.

### 2.2 Determinism (G-determinism)

W's re-run on 70200–70203: all four committed, all four succeeded in both runs, reading difference
0.0 and commit-target difference 0.0 on each (tolerances 0.1 cm and 0.6 cm). The stage is not void.

---

## 3. Reported only (§7.3; no row, claim or clause depends on these)

### 3.1 Aim error, fixed-point error, landing miss and τ-curve prediction

Aim error is against H-final(commit)'s committed aim on the same reset (median with
reset-bootstrap 95 % interval, 87.5th percentile, maximum; n = 127, the refused reset excluded).
Fixed-point error and landing miss are TASK-080's carried measures (median, 87.5th percentile,
maximum). The τ-curve prediction maps each arm's aim errors through TASK-080's pooled τ curve: a
prediction, not a count.

| arm | aim error median [95 %] cm | p87.5 | max | fixed-point error median / p87.5 / max cm | landing miss median / p87.5 / max cm | τ-curve predicted count | counted |
|---|---|---:|---:|---|---|---:|---:|
| W | 0.295 [0.249, 0.333] | 0.567 | 1.200 | 0.567 / 0.800 / 1.651 | 0.415 / 0.893 / 1.813 | 117.23 | 118 |
| W-frozen | 0.347 [0.301, 0.404] | 0.680 | 1.501 | 0.597 / 0.975 / 1.785 | 0.533 / 1.019 / 2.331 | 116.31 | 117 |
| N | 1.848 [1.689, 2.134] | 2.674 | 3.867 | 2.122 / 3.022 / 4.162 | 2.729 / 4.102 / 5.681 | 52.80 | 73 |
| L-shuf | 2.710 [2.389, 2.993] | 3.996 | 14.849 | 2.801 / 4.128 / 14.356 | 4.025 / 6.018 / 18.424 | 42.53 | 39 |
| L-mean | 1.983 [1.778, 2.202] | 3.081 | 3.944 | 1.987 / 3.226 / 4.180 | 2.958 / 4.575 / 5.806 | 56.76 | 59 |
| L-rand | 6.204 [4.518, 7.333] | 13.524 | 20.317 | 6.010 / 13.151 / 19.959 | 8.338 / 16.423 / 23.570 | 22.28 | 20 |
| H-rule | 0.124 [0.111, 0.137] | 0.222 | 0.450 | 0.502 / 0.616 / 0.815 | 0.189 / 0.342 / 0.548 | 117.41 | 125 |
| H-sysid | 0.461 [0.398, 0.526] | 1.009 | 1.531 | 0.745 / 1.472 / 1.936 | 0.674 / 1.536 / 2.202 | 107.83 | 121 |
| H-final(commit) | – | – | – | 0.454 / 0.505 / 0.515 | 0.093 / 0.116 / 0.132 | 117.03 | 126 |
| H-read | – | – | – | 0.469 / 0.825 / 1.688 | 0.602 / 1.190 / 2.506 | 117.42 | 117 |
| H-now | – | – | – | 6.351 / 7.407 / 8.754 | 9.081 / 10.399 / 11.298 | 14.00 | 6 |

The τ curve under-predicted W's closed-loop count by about one reset again (117.23 against 118), and
it under-predicts H-rule (117.41 against 125) and H-sysid by more; it is not a calibrated predictor
of counts near the ceiling.

### 3.2 Final-distance distributions (§7.3 item 4)

The apple's final distance from the plate centre, in cm, over the 127 attempts that reached the
place (median with interval, 12.5th and 87.5th percentiles, maximum), and the number of those 127
that ended between 3.5 and 4.5 cm, within 0.5 cm of the scorer's 4 cm radius.

| arm | all: median [95 %] | p12.5 / p87.5 / max | successes: n, median | failures: n, median | 3.5–4.5 cm |
|---|---|---|---|---|---:|
| W | 3.479 [3.361, 3.535] | 3.036 / 3.832 / 4.506 | 118, 3.455 | 9, 4.268 | 59 |
| W-frozen | 3.389 [3.247, 3.471] | 2.834 / 3.852 / 4.562 | 117, 3.313 | 10, 4.214 | 51 |
| N | 3.852 [3.766, 3.985] | 3.401 / 4.531 / 113.967 | 73, 3.653 | 54, 4.431 | 83 |
| L-shuf | 4.281 [4.118, 4.381] | 3.331 / 29.543 / 179.115 | 39, 3.628 | 88, 4.474 | 67 |
| L-mean | 4.064 [3.823, 4.234] | 3.279 / 4.554 / 105.050 | 59, 3.572 | 68, 4.458 | 75 |
| L-rand | 22.909 [20.185, 25.632] | 3.810 / 141.148 / 206.646 | 20, 3.650 | 107, 25.575 | 37 |
| H-rule | 3.381 [3.303, 3.497] | 2.757 / 3.767 / 4.016 | 125, 3.381 | 2, 4.014 | 52 |
| H-sysid | 3.559 [3.422, 3.668] | 2.868 / 3.879 / 4.508 | 121, 3.529 | 6, 4.183 | 67 |
| H-final(commit) | 3.360 [3.275, 3.459] | 2.860 / 3.740 / 4.037 | 126, 3.356 | 1, 4.037 | 46 |
| H-read | 3.471 [3.315, 3.540] | 2.830 / 3.841 / 4.544 | 117, 3.398 | 10, 4.280 | 57 |
| H-now | 28.917 [27.345, 32.245] | 18.767 / 190.312 / 209.131 | 6, 3.306 | 121, 28.951 | 4 |

As in TASK-080, every arm whose aim is near the plate rests the apple about 3.4–3.6 cm from the
centre in median, so 46–67 of 127 attempts of each of the six arms whose aims are near the plate
(W, W-frozen, H-rule, H-sysid, H-final(commit), H-read) end within 0.5 cm of the 4 cm radius. W's
nine failures that reached the place ended 4.02–4.51 cm from the centre (median 4.27): they are
near misses at the scorer's boundary, not gross misses. Counts near the ceiling therefore depend
on more than the aim, which adds noise to every arm's count, the ceiling's included.

### 3.3 The solver's effect, W − W-frozen (§7.3 item 2)

| quantity | value |
|---|---|
| W − W-frozen | **+1/128**; W only 6, W-frozen only 5; 95 % [−5, +7]; exact one-sided McNemar p = 0.5 |
| W-frozen's capped (not converged) resets | 37 of 128 |
| on those 37 resets | W 33, W-frozen 32 |
| median aim error (vs H-final(commit)) | W 0.295 [0.249, 0.333] cm, W-frozen 0.347 [0.301, 0.404] cm |

**The solver's contribution to the count is not shown.** The aim errors' intervals overlap, and
the counts differ by one reset. TASK-080's solver, on these resets, scored 117/128, which itself
clears G-bar's 112. **Post hoc** (computed for this record from the report's per-reset outcomes
with the frozen estimator `lewm_cp_v2.paired_interval`, salt 8302; not preregistered; W-frozen is
read by no row): W-frozen − H-rule = −8/128 (1 / 9), 95 % [−14, −2], lower bound above −16; and
W-frozen beats the four twins of this run by exact one-sided McNemar at p ≤ 5.1 × 10⁻¹⁰ (50/6,
80/2, 61/3, 98/1). These twins carry affine_local, not TASK-080's refinement, so this is not
TASK-080's design re-run. The reading is plain: on this cohort, **TASK-080's W with its own solver
would also have met every gated bar**. The move from TASK-080's L-NEAR (n = 64, W − H-rule −5/64,
[−10, 0]) to this L-PASS is consistent with the larger cohort narrowing the interval (the margin is
the same 12.5 points), and it is not shown to come from the solver. The design note's development
gain (62/64 against 57/64 on seeds 70000–70063) did not reproduce on fresh seeds; it carried a
selection caveat (the best of three executed variants).

### 3.4 affine_local's diagnostics (§7.3 item 5)

| arm | decisions | fallbacks | clipped | points in the fit (median) | rank < 3 | complex eigenvalue pairs | residual at the committed aim, median cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 127 | 0 | 0 | 25 | 0 | 32 | 0.279 [0.238, 0.330] (max 1.026) |
| N | 127 | 0 | 21 | 25 | 0 | 18 | 0.000 |
| L-shuf | 127 | 0 | 33 | 25 | 0 | 32 | 0.252 |
| L-mean | 127 | 0 | 11 | 25 | 0 | 41 | 0.176 |

For W, the largest real part of J's eigenvalues had a median of −0.377 (max 0.145) and the
smallest a median of −0.533 (32 decisions had a complex pair); the clip never bound and no fallback
was used. N's residual is 0 because the action-blind predictor's predictions do not depend on the aim
(J = 0), so its fixed point is its constant prediction. No arm used a fallback (G-solver: every
record logged its arm's declared solver).

### 3.5 Seed 66800's flag

W-66800's `last_two_triggered` flag is true and every W training curve was lowest at its last point
(TASK-077 results §3, caveats 1–2): the primary model's checkpoint selection may not have
saturated. It stands beside every comparison above, which are all seed-66800 comparisons. No other
model seed ran in closed loop.

---

## 4. Caveats

1. **One run, one model seed.** One closed-loop run of one predictor (W-66800, flagged
   `last_two_triggered`), with one readout (R-S) and one solver; no other seed of W ran, so
   seed-to-seed variation is unmeasured.
2. **Simulation only, one declared condition.** `apple-to-plate-v2` in MuJoCo under C1-M, a
   declared simulation-only plate condition (a reactive plate rule, κ = −0.5, L = 2, s1 = 525, with
   a post-pick move of radius ≤ 4 cm); the scorer and the plate law are imposed simulator laws.
   Nothing here transfers as such to Arena or the real G1.
3. **LeWM makes one decision.** It chooses only the single place aim committed at step 405, between
   P-3's learned pick (behaviour cloning on a frozen DINOv2 readout, not a world model) and e9's
   scripted place primitive. It is not a LeWM policy for the task.
4. **Frozen DINOv2 features.** W predicts in a pooled 8 × 8 grid of frozen, externally pretrained
   DINOv2 patch tokens of the onboard 112 px frame; no random-init floor ran for W, so whether
   pretraining matters is not tested.
5. **"LeWM needed" is not shown, and H-rule is better.** H-rule, a non-learned controller given the
   simulator's plate law, scored 125/128 against W's 118/128; W − H-rule [−13, −2] excludes 0. The
   pass is non-inferiority within an **allocated, uncalibrated** 12.5-point margin (16/128).
6. **The solver effect is not shown** (§3.3): W − W-frozen +1/128 [−5, +7], p = 0.5, and W-frozen's
   117/128 would itself have cleared the bar (post hoc, it would also have met G-NI).
7. **TASK-080's gated Stage S stands as recorded** (58/64, L-NEAR, no claim). It is not re-read or
   pooled with this cohort.
8. **The 128-reset cohort was chosen after TASK-080's L-NEAR** (R18.35, by power at the development
   rate; the bar and margin keep TASK-080's fractions). A larger cohort makes non-inferiority
   within a fixed fractional margin easier to show for the same true rates.
9. **The scorer's margin is small** (§3.2): successes in all near-plate arms rest about 3.3–3.5 cm
   from the centre (median) against a 4 cm radius, and W's failures are near misses at that boundary.
10. **The development check that admitted the solver was selected** (the best of three executed
    variants on seeds 70000–70063), and Stage D's 16/16 has a wide interval (0.794–1.000).

---

## 5. R7 (R19.21)

R7 changes after an L-PASS only through its own reviewed ruling (R18.12, R19.10). R19.21 is that
ruling. The sentence in force from its merge, quoted verbatim and in full by the entry documents,
is in DECISIONS.md under R19.21. Its v2 LeWM part states this gated success with every qualifier of
§4 that bears on what was shown: one run, one model seed (flagged), simulation only, the declared
condition C1-M, LeWM choosing only the single place aim, frozen DINOv2 features, non-inferiority to
H-rule within the allocated margin while H-rule scored measurably higher, "LeWM needed" not shown,
the solver's effect not shown, TASK-080's L-NEAR standing, and the cohort size chosen after it.

---

## 6. Next step (R19.23; recorded only, not ruled)

TASK-081 closes. No repeat of Stage S, and no further gated run, is planned by this record.
Recommended directions, for the owner, in this order:

1. **R18.32 (a): a condition in which no hand-written arm is given the plate law**, declared as a
   task change, drafted first as a design note with its own preregistration and fresh seeds. This
   is the only direction here that can test "LeWM needed": under C1-M, H-rule is handed the
   simulator's law and is measurably better than W.
2. **TASK-079's precondition.** R19.10 proposed that a TASK-081 L-PASS count as TASK-079's
   "TASK-077 L-PASS (or its equivalent row)"; this record recommends adopting it by its own owner
   ruling. TASK-078's pass and an owner ruling on Arena as a gated benchmark stay required.
3. **Robustness of this result before any broader claim**: more model seeds of W (seed 66800 is
   flagged), broader conditions (other move radii, κ, commit steps) and a random-init floor for W,
   each preregistered.
4. **No further solver work on this condition**: the solver's effect was not shown on fresh seeds.
