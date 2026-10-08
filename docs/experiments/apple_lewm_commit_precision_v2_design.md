# W's commit precision under C1-M: the next LeWM task on apple-to-plate-v2 (TASK-081, DRAFT design note)

**Status: DRAFT design note. It is not a protocol and nothing in it is frozen.** No bar below is
gated. It carries one development feasibility check (§4), run on development seeds only
(70000–70063, plus a two-seed smoke on 70090–70091), CPU only, under TASK-080's condition with
TASK-080's artifacts unchanged. No seed of any range used or reserved by TASK-076, TASK-077 or
TASK-080 was simulated, and no model was trained.

Every choice in it is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R18.33–R18.35 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (e):

- **R18.33** picks R18.32's direction (b), W's commit precision, first, and keeps (a), a
  condition in which no hand-written arm is given the plate law (a task change), as the
  documented alternative;
- **R18.34** takes the seed block 70000–71999 and the salts 8301–8312 for TASK-081, with
  70000–70099 for development;
- **R18.35** records §4's development numbers and recommends a preregistration (§6).

Any task the note leads to needs, before a single cohort seed is simulated:
- its own preregistration;
- an independent review;
- fresh seeds.

The canonical status sentence (DECISIONS 2026-10-02, R7, as updated by R18.31) is unchanged by
this note. It is quoted in full in DECISIONS.md (R18.31). Its v2 LeWM part says that TASK-080's W
reached 58/64 on the gated cohort S but failed the preregistered non-inferiority test against
H-rule, the non-learned rule controller (63/64), and that the row is L-NEAR. **Nothing here is a
result.** Every development number below is one run, one model seed (66800, flagged
`last_two_triggered`), in simulation only. These are not gated counts and not project-learned
results.

---

## 1. Where we start

TASK-080 closed **L-NEAR** at Stage S
([results](apple_lewm_c1m_v2_pred_readout_results.md) §2; R18.30–R18.32). On the 64 gated resets
65200–65263:

- **W 58/64** (G-bar 56 passes);
- W beat N 26/64, L-shuf 21/64, L-mean 30/64 and L-rand 11/64 (exact one-sided McNemar,
  p < 0.01);
- **H-rule 63/64.** H-rule is the hand-written rule controller. It is given the simulator's plate
  law and reads the plate with R-plate. The paired difference W − H-rule is −5/64, with a
  reset-bootstrap 95 % interval of [−10, 0]/64. Its lower bound is below −δ = −8/64, so G-NI
  fails. Its upper bound is not below −8/64, so W is not detectably inferior beyond the margin;
- the privileged ceiling H-final(commit) 62/64.

The results document records one post-hoc observation (§2.6), stated there as a hypothesis, not a
finding: **all six of W's failures came on the 17 resets where W's refinement stopped at its
10-refinement cap; all 47 converged resets succeeded.** R18.32 offered two directions for the
owner; R18.33 picks (b) first.

**W's controller** (TASK-077 §5.1, `lewm_c1m_v2_runtime.choose_from_grid`, unchanged in TASK-080):
1. Build 147 candidates g on the box around p̂ (R-plate's reading at 405) and the palm h.
2. Roll each candidate's stand-in chunk out with W from the encoded 405 frame to r = 465, and read
   the predicted plate p̃(g) with R-S.
3. Take the candidate with the lowest |p̃(g) − g|.
4. Refine at most 10 times, g ← clip(p̃(g)), one roll-out per refinement. Stop as converged once
   the move |clip(p̃(g)) − g| is at most τ_commit/4 = 0.25 cm.
5. At the cap, commit the last move **unevaluated**.

The aim sought is the fixed point g = p̃(g) of W's own prediction. The rule's law is
plate(r) = p + κ(palm − h), with κ = −0.5. If W's predicted plate followed that law exactly, the
step would contract by |κ| = 0.5 per refinement, and the cap could not bind from the grid's
starting error.

## 2. Diagnosis: why the capped refinement fails

### 2.1 What the code allows

The loop stops in only three ways: convergence; an infeasible stand-in chunk for the next iterate
(logged `stopped: infeasible_refinement`); or the cap. The box clip can also change an iterate.

On S, **no W iterate was clipped** (W's clip-binding fraction is 0, and every attempt logs
`clipped` false). **No refinement stopped on an infeasible chunk** (no `stopped` entry in any of
the 64 logs). So every non-converged reset is a cap, reached after 10 roll-outs whose moves never
fell to 0.25 cm.

### 2.2 What the logs show (post hoc on S and D; diagnosis, not evidence)

Read from TASK-080's Stage S report (`outputs/task080-s-1/report.json`, sha256 `53cfdec1…393e`)
and Stage D report (`outputs/task080-d-1/report.json`, sha256 `1602447a…0611`); nothing was re-run:

1. **The cap binds often, on S and D alike.** On S, 17 of 64 resets were capped. On D, 3 of 16
   were capped (65104, 65105 and 65111), and all 16 resets succeeded.
2. **The iterates oscillate; they do not drift away.** Consecutive refinement steps point in
   nearly opposite directions. The median over resets of the mean cosine between consecutive
   steps is −0.84 on the capped resets and −0.85 on the converged ones. On capped resets the move
   size either stalls at 0.3–0.9 cm (65212, 65214 and 65225 are 2-cycles, cosine −1.00) or grows
   (65245: 0.26 → 2.18 cm; 65247: 0.61 → 1.61 cm; 65201 up to 3.77 cm).
3. **The coarse slope is not the cause.** Stage R's per-root least-squares slope of p̃(g) against
   g over the whole grid has a median matrix of about diag(−0.13, −0.41) and a trace/2 of −0.273
   on 249 gate-P roots. On the dry run's 250 val roots it is diag(−0.13, −0.42) and −0.275. That slope is negative and well inside
   (−1, 1), so the undamped step would contract at that scale. It is **at the 0.25–1 cm scale of
   the refinement** that the predicted plate stops behaving like a contraction. There it shows
   2-cycles and growing oscillations. That is a non-smooth or noisy p̃ (a different stand-in chunk
   and roll-out per iterate, read by a ridge), not a wrong κ.
4. **What the cap commits.** The reference here is H-final(commit)'s aim on the same reset. That
   is the privileged look-ahead's converged aim, and it landed within 0.131 cm of the plate on
   every S reset. It is a better reference than the formula g\* = (p − κh)/(1 − κ), which every
   arm misses by about 0.48 cm (H-final's own g\*-error is 0.478–0.517 cm) because the palm does
   not end exactly at the aim. Against it:

   | S resets | W's committed aim (median, cm) | midpoint of the last two iterates | evaluated point with the smallest residual | H-rule's aim |
   |---|---:|---:|---:|---:|
   | 17 capped | **0.54** | 0.27 | 0.23 | 0.14 |
   | 47 converged | 0.29 | 0.31 | 0.35 | 0.12 |

   The landing miss is about (1 − κ) = 1.5 times the aim error. W's median landing miss is
   0.79 cm on capped resets and 0.42 cm on converged ones.
5. **The scorer's margin is small.** Every arm's apple comes to rest about 3.4 cm from the plate
   centre in median (W's successes 3.43 cm, H-rule's 3.41, H-final's 3.32). The scorer's radius
   is 4 cm, and W's six failures ended 4.17–4.55 cm from the centre. A few millimetres of aim
   error in the wrong direction cross the boundary, which is also why the privileged ceiling
   failed twice (4.00 and 4.02 cm).
6. **Ruled out as the cause on S:**
   - the box edge (no clipping);
   - an infeasible chunk (no stop);
   - the box coordinate a (median 0.352 capped against 0.354 converged);
   - p̂'s error at 405, measured as the distance between g\* computed from p̂ and from the true
     plate: median 0.085 cm capped against 0.113 cm converged;
   - κ itself (§2.2 point 3).

**Diagnosis (a hypothesis for the development check).** The cap binds because the
single-evaluation fixed-point step g ← clip(p̃(g)) meets a predicted-plate map that is rough at
the refinement's scale. Its residual floor, about 0.3–0.5 cm, is above the 0.25 cm tolerance, and
the step oscillates instead of contracting. At the cap, the controller commits one extreme of
that oscillation. The fault is in the solver's step and its use of single noisy evaluations, not
in the step size relative to κ, the box, or the readout's bias near the edge.

Two caveats travel with it:
- converged resets are not as precise as H-rule either (0.29 against 0.12 cm on S), so a solver
  fix can close only part of the gap;
- every number in this section is post hoc on one gated run.

### 2.3 What the diagnosis predicts

- A **damped step** collapses 2-cycles but not noise.
- **Raising the cap** helps neither.
- **Committing the best evaluated point** helps only when the oscillation brackets the fixed
  point.
- **A fit over many of W's predictions** (W's own grid roll-outs) averages the per-evaluation
  noise. It should help converged resets too.

§4 tests these predictions on development seeds.

## 3. Candidate fixes (no retraining)

Every variant keeps W, R-S, the 147-candidate grid, the stand-in chunks and the grid argmin. It
changes only what is done after the grid, so the twins (N, L-shuf, L-mean) can carry the same
solver unchanged. None reads task truth or the plate law.

| variant | what it does after the grid | extra roll-outs | why it might help | risk |
|---|---|---:|---|---|
| **frozen** | TASK-077's step, cap 10 (the TASK-080 controller) | ≤ 10 | – | the failure mode above |
| **cap30** | the same step, cap 30 | ≤ 30 | if the step is slowly contracting | does nothing against 2-cycles or noise |
| **damped** | g ← g + ½(clip(p̃(g)) − g), cap 10, same stop rule | ≤ 10 | contracts for local slopes in (−3, 1); kills 2-cycles | noise floor unchanged |
| **bestres** | the frozen path; at the cap, the evaluated point with the smallest residual | 0 | no unevaluated commit | the residual itself is noisy |
| **affine_local** | least-squares p̃(g) ≈ c + J g over the feasible grid points within 2 a-steps and 2 cm in b of the argmin (up to 25), commit the clipped solution of g = c + J g | 0 (1 logged) | uses up to 25 of W's own predictions: averages the noise; J is estimated from W, not given | a locally curved p̃; a near-singular I − J (falls back to the argmin) |
| **affine_global** | the same over all feasible grid points | 0 (1 logged) | more averaging | curvature and bias over the whole 6 cm box |

Not tried, recorded for the preregistration's choice:
- **Polyak averaging** of the iterates. On development seeds the last-two midpoint was not
  better than the committed iterate on capped resets (median 0.47 against 0.47 cm), unlike on S.
- **Newton or secant steps** from finite differences. They difference two noisy evaluations, the
  wrong direction for a noise floor.
- **Several stand-in chunks per aim.** This changes the controller's inputs.
- **Refitting R-S** or **retraining W**. Not justified by §4: the solver alone closes most of the
  gap on development seeds, and a refit or retrain would reopen TASK-080's Stage R and T
  questions.

## 4. Development feasibility check (development seeds, CPU; not gated)

### 4.1 The run

- **Code.** `src/embodied_jepa/commit_precision_dev.py` holds the variants, each a pure function
  of the predictor, and a development worker. The worker is TASK-080's worker unchanged, with W's
  `choose` wrapped so that **every** variant's aim is computed at 405 from the same roll-outs and
  logged, and the attempt executes the variant its task names. The frozen variant is checked
  against `choose_from_grid`'s aim and convergence on every call. The runner is
  `scripts/dev_commit_precision.py`. It uses TASK-080's runner as a library: `sim_preflight`, the
  sha256 checks of every reused artifact through `old_chain`, Stage R's R-S at its recorded
  sha256, and G-repro's P-readout refit. Tests: `tests/test_commit_precision_dev.py`.
- **Fixed before the first run** (the commit `3a24abb`):
  - the six variants;
  - the damping ½ and the caps;
  - the local neighbourhood;
  - the three executed arms **W:frozen, W:damped and W:affine_local**;
  - the references H-final(commit) (privileged) and H-rule (the TASK-080 comparator);
  - the seeds.
- **Seeds.** 70000–70063 for development (smoke 70090–70091). A search of all 112 local and
  remote refs (`src`, `scripts`, `tests`, `configs`, `docs`, `benchmarks`, `.mc`) found no
  integer 70000–71999 used as a seed; the only hits were fixed-point decimals such as 0.70736.
  The working files' `src` and `scripts` of every worktree had none. The block lies outside
  every range in `lewm_pr_v2.FORBIDDEN_RANGES` and outside TASK-080's block 65000–65999. The
  condition is C1-M exactly as TASK-080 ran it, including the move salt 8201's draw per seed.
  W-66800, R-S (Stage R's fit), R-plate and P-3 are unchanged.
- **Execution.** At `3a24abb` (clean tracked tree), worktree
  `/home/huhn/develop/emai/worktrees/task081-design`, CPU, 4 workers, threads as TASK-080 pins,
  on 2026-10-08, about 06:05–06:21 CEST. Wall time 965 s. All 320 attempts (5 arms × 64 resets)
  ran to `policy_complete`, with no block, and the pool closed with all 4 workers joined.
  Every W and H-rule attempt passed the privileged-read check (0 task-truth reads). There were no
  render disagreements on the development frames. G-repro's re-render of the carried TASK-072
  seeds had two one-level, 4-pixel disagreements (51202 and 51388, equal states), resolved by the
  carried majority rule.
- **Report.** `outputs/task081-dev-1/report.json`, sha256
  `febe728098037cc56f957037a0881de836509afe27467f18335c14e5ae63fb7f`. Summary
  (`--analyse`, the code in this PR): `outputs/task081-dev-1/summary.json`, sha256
  `3eafbe9e…9422`.
- **Evidence.** A copy is in `~/develop/emai/evidence/task081-dev/` (manifest
  `_checksums/task081-dev.sha256`, sha256 `6cf58404…02b4`).
- **Consistency.** The three W runs logged identical variant aims at 405 on every reset
  (checked).

### 4.2 Numbers

**Aim error against H-final(commit)'s aim on the same reset** (cm; median with its
reset-bootstrap 95 % interval). The predicted count maps each reset's error through TASK-080's
pooled τ curve (`lewm_pr_v2.predicted_count`). It is a prediction, not a count. "Not converged"
counts the resets where the variant stopped at its cap.

| variant | median [95 %] | p87.5 | max | on the 19 frozen-capped resets | on the 45 converged | not converged | predicted of 64 |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen | 0.382 [0.322, 0.473] | 0.969 | 2.844 | 0.474 | 0.330 | **19** | 57.95 |
| cap30 | 0.362 [0.304, 0.471] | 0.721 | 2.175 | 0.406 | 0.330 | 14 | 59.12 |
| damped | 0.346 [0.278, 0.447] | 0.719 | 1.297 | 0.454 | 0.335 | 9 | 59.68 |
| bestres | 0.357 [0.271, 0.469] | 0.722 | 1.316 | 0.404 | 0.330 | (19) | 59.69 |
| **affine_local** | **0.243 [0.205, 0.322]** | **0.494** | 1.157 | **0.284** | **0.235** | – | **60.72** |
| affine_global | 0.412 [0.334, 0.465] | 0.668 | 1.186 | 0.410 | 0.414 | – | 59.69 |

**Closed loop** (counted success, one run, development; landing miss is the commit target's
distance from the plate at s1 = 525):

| arm | count | landing miss median / p87.5 / max (cm) | − H-rule [95 %] | W only / H-rule only |
|---|---:|---|---|---|
| W:frozen | 57/64 | 0.577 / 1.408 / 4.285 | −7 [−12, −3] | 0 / 7 |
| W:damped | 58/64 | 0.534 / 1.025 / 1.880 | −6 [−11, −2] | 0 / 6 |
| **W:affine_local** | **62/64** | 0.370 / 0.745 / 1.677 | **−2 [−5, 0]** | 0 / 2 |
| H-final(commit) (privileged, not learned) | 63/64 | 0.081 / 0.121 / 0.136 | – | – |
| H-rule (not learned; given the plate law) | 64/64 | 0.168 / 0.295 / 0.519 | – | – |

- **The cap binds as often as on S.** 19 of 64 resets were capped (S: 17 of 64). W:frozen failed
  on 7 resets: 5 of the 19 capped and 2 of the 45 converged. S's pattern of "every failure
  capped" does not hold exactly on fresh seeds.
- **Damping helps a little.** It halves the non-converged resets (9) and gains one closed-loop
  success (58). On 70023 and 70009 it stalls at a residual of about 0.3–0.5 cm: the noise floor of
  §2.2.
- **Raising the cap helps less** (14 still not converged). Best-residual helps the tail only.
- **affine_local helps everywhere.** It is better on converged resets too (0.235 against
  0.330 cm), as §2.3 predicted for a noise-averaging fit. Its aims never fell back to the argmin
  and never sat on the box edge. Its median residual |clip(p̃(g)) − g| at the committed aim is
  0.28 cm (p87.5 0.60). The local J has median eigenvalues −0.58 and −0.42 on capped resets and
  −0.51 and −0.38 on converged ones. Only 1 of 19 capped resets has an eigenvalue below −1. So at
  the 2 cm scale W's prediction is a contraction; the trouble is below it.
- **affine_global is worse** (0.412 cm). Fitting the whole 6 cm box costs more in curvature and
  bias than it gains in averaging.
- **Closed loop: W:affine_local 62/64 against W:frozen 57/64.** On the paired resets affine_local
  won 7 and lost 2 (exact one-sided McNemar p = 0.090). Against H-rule (64/64 here) it is −2/64
  with interval [−5, 0]. Its lower bound is above −8/64, so on these development resets the
  TASK-080 G-NI estimator would pass for W:affine_local. It would fail for W:frozen (−7,
  [−12, −3]) and W:damped (−6, [−11, −2]).

**Caveats.**
1. One run, one model seed, 64 development resets, simulation only.
2. affine_local is the best of three executed variants, chosen after seeing them. The 62/64 is
   therefore optimistic (selection), and a fresh-seed test is needed.
3. H-rule's 64/64 and H-final's 63/64 here are development counts, not S's.
4. The τ-curve predictions (60.72 against 57.95) and the closed-loop counts (62 against 57) agree
   in direction. Both are within a few resets of each other, as on S (§2.7 of the results).
5. Nothing here re-reads S. S's seeds were used only for §2's post-hoc diagnosis.

## 5. G-NI's power, and whether the margin or the cohort should change

**The estimator is TASK-080's, unchanged.** It is the 2.5th percentile of the reset-bootstrap sum
of W − C over n paired resets, and it must be > −δn. The bootstrap depends on the resets only
through (k+, k−), the resets won by W only and by C only. `commit_precision_dev.ni_pass_table`
computes pass or fail per (k+, k−) with 4 000 resamples (salt 8301), and `ni_power` sums the
multinomial probability of the passing cells. The couplings are as TASK-080 §10:
- "overlap": the outcomes as nested as the rates allow;
- "independent";
- "half": halfway between, in the discordant probabilities.

The check reproduces S: (k+, k−) = (1, 6) fails at δn = 8. The size at the margin (W = C − δ) is
3.4–4.0 % at n = 64, 2.6–3.6 % at 96 and 2.6–3.4 % at 128 (nominal 2.5 %; the percentile
bootstrap is slightly liberal at small n, as TASK-077's Stage-0 simulation also found).

**Power of G-NI** (overlap / half / independent). C's planning rate is 0.984 (S's 63/64) or
0.992 (S and development pooled, 127/128); δ = 8/64 of n:

| n (δn) | C | W = 0.938 | W = 0.953 | W = 0.969 |
|---|---|---|---|---|
| 64 (8) | 0.984 | 0.66 / 0.63 / 0.58 | 0.86 / 0.82 / 0.77 | 0.98 / 0.96 / 0.92 |
| 64 (8) | 0.992 | 0.54 / 0.53 / 0.52 | 0.76 / 0.74 / 0.72 | 0.94 / 0.92 / 0.90 |
| 96 (12) | 0.984 | 0.85 / 0.76 / 0.69 | 0.97 / 0.93 / 0.88 | 1.00 / 0.99 / 0.98 |
| 96 (12) | 0.992 | 0.74 / 0.69 / 0.65 | 0.92 / 0.89 / 0.85 | 0.99 / 0.99 / 0.97 |
| 128 (16) | 0.984 | 0.93 / 0.87 / 0.82 | 0.99 / 0.98 / 0.95 | 1.00 / 1.00 / 1.00 |
| 128 (16) | 0.992 | 0.85 / 0.80 / 0.77 | 0.97 / 0.95 / 0.93 | 1.00 / 1.00 / 0.99 |

At TASK-080's own S rates (W 0.906, C 0.984, n = 64) the same computation gives 0.24–0.26. At
W = C − 4/64 (0.922 against 0.984) it gives 0.39–0.43, inside TASK-080 §10's 0.14–0.43.

**What W's rate is, for planning.** The two development readings of affine_local are:
- the τ-curve prediction, 60.72/64 = 0.949;
- the closed-loop count, 62/64 = 0.969, which carries the selection caveat.

The planning rate is taken as **0.95**, the lower of the two. At W = 0.953 the power is 0.77–0.86
at n = 64, 0.85–0.97 at n = 96 and 0.93–0.99 at n = 128, against C = 0.984–0.992. G-bar (the
bar's fraction 0.875 of n) has power ≥ 0.997 at W = 0.953 for every n here.

**The margin stays δ = 8/64 (12.5 points).** It is an allocation, not a calibration (TASK-080
§10, R15.6's label). Changing it now, after S's −5/64 with lower bound −10, would be fitting the
margin to S, and no measured quantity supports a different value. The one calibrated candidate
points the other way. The pooled τ curve loses 6/64 from 0 to 0.5 cm of planted error (64 → 58),
so a "τ-calibrated" margin would be about 6/64, a **tighter** test. If the preregistration wants
a calibrated margin, that is the direction it would move, and it would need its own ruling.
Recommended: keep 8/64, so that the test stays comparable with TASK-080's.

**The cohort.** A larger cohort at the same fractional margin is justified by power at the
development rate, not by S's outcome. Recommended: **n = 128** gated resets (G-bar 112/128). This
gives G-NI power ≥ 0.93 at W = 0.953 against C up to 0.992. TASK-080's S took 1 581 s on the CPU
for 64 resets and ten arms, so 128 resets cost about 55 minutes. The alternative, n = 64, gives
0.72–0.86 at the same rates. It is the cheaper repeat of TASK-080's design, but it leaves a 14–28 %
chance that G-NI fails again at the planning rate.

## 6. Recommendation (R18.35)

**Preregister TASK-081**, with W's solver changed from the frozen refinement to **affine_local**.
Declare it as a change to W's controller form after the grid, not a new model:
- W-66800, R-S, the grid, the stand-in chunks, C1-M, P-3 and e9 are all unchanged;
- no retraining, no refit;
- fresh seeds in 70100–71999.

The twins N, L-shuf and L-mean carry the same solver, as they carried W's controller form in
TASK-080. L-rand, H-rule, H-sysid, H-final(commit), H-read and H-now are unchanged.

Proposed shape, for the preregistration to fix:
1. **A frozen-solver arm, W-frozen** (TASK-080's controller on the same resets), reported only.
   Its paired difference with W measures the solver's effect on fresh seeds. That makes the
   hypothesis from S and §4 a tested one.
2. **G-NI unchanged** (estimator, δ = 8/64 as a fraction of n, C the better of H-rule and
   H-sysid). G-bar at 0.875 n. The four twin tests unchanged (exact one-sided McNemar, p < 0.01).
   The rows as TASK-080 §9.5.
3. **n = 128 for the gated cohort** (§5). A development cohort D of 16 resets runs first, as in
   TASK-080. Both come from 70100–71999 under R18.34.
4. **A Stage-R-style offline gate before the closed loop is optional.** TASK-080's Stage R
   already passed for this W and R-S. The only change is the solver, and the closed loop is
   cheap. If one is kept, the preregistration says so.
5. **The abandonment clause** carries TASK-080 §12's scope with the solver change named. An
   L-INFERIOR or L-NO-GAIN would fire it as there.

**If the owner prefers (a)**, a condition in which no hand-written arm is given the plate law, it
remains open. It is a task change and needs its own design note and K0-style headroom check.
(b) does not block it, and §4's solver would carry over to it.

## 7. Open questions for the preregistration

1. **Local neighbourhood size.** It was fixed at 2 a-steps × 2 cm (up to 25 points) before the
   run. The preregistration should keep it; a sweep would be new development. Should
   affine_local be followed by damped refinements from its solution? Not tested here.
2. **Planning rate for C.** It is 0.984 (S) or 0.992 (S and development pooled). Recommended: plan
   with 0.992, the stricter.
3. **The scorer's boundary.** Every arm's apple rests about 3.4 cm from the centre against a 4 cm
   radius (§2.2 point 5), so outcomes near the boundary depend on more than the aim. This is a
   property of e9's place primitive and the scorer, both unchanged by design. The preregistration
   should report each arm's final-distance distribution beside its count.
4. **Seed 66800's `last_two_triggered` flag** carries over unchanged.
5. **R7.** A TASK-081 L-PASS would still need its own reviewed ruling before R7 states a gated
   LeWM success (R18.12's rule).
