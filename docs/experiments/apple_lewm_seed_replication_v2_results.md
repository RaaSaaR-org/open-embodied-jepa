# TASK-083 results: TASK-081's L-PASS replicated with W's two other model seeds under C1-M

**Outcome: REP-PASS** (R21.19, decided by Claude under owner delegation). Both of W's other model
seeds of TASK-077's training run, 66801 and 66802, each reached TASK-081's L-PASS on their own, on
128 fresh gated resets, in one run, in simulation only, under the declared simulation-only plate
condition C1-M. LeWM chooses only the single place aim committed at step 405, between P-3's learned
pick (behaviour cloning, not a world model) and e9's scripted place primitive. Protocol (FROZEN):
[apple_lewm_seed_replication_v2.md](apple_lewm_seed_replication_v2.md); Stage 0:
[record](apple_lewm_seed_replication_v2_stage0.md); rulings R21.1–R21.22 in
[DECISIONS.md](../DECISIONS.md), decisions 2026-10-08 (o)–(q) and 2026-10-09 (r).

## 1. The run

| | value |
|---|---|
| GO | the independent pre-launch review's "Stage S GO at fef0490…" (#175, issuecomment-6068743705), after a first NO-GO for main's integration run cancelled at its 20-min timeout (rerun green) |
| revision | `fef0490` (#175's merge), clean tracked tree at start and end, worktree `task083-stages` |
| command | protocol §6.4, once; CPU only, 4 workers, no GPU |
| time (UTC) | 2026-10-08 20:47:08–22:02:00, **4 491 s** (cap 21 600; Stage 0 scale about 4 740) |
| G-quiet | load 1.48 / 0.43 at the start (bar 2.0 / 2.0); MemAvailable 25.6 GiB |
| in-run G-tests | 2243 passed, 37 skipped |
| G-memory | peak process-tree PSS 9.48 GiB (cap 12) |
| guards | every pin (TASK-076, 077, 080, 081, this task's four and its protocol document), G-seed (each pool given its own seed's W, N, R-S and R-N, logged), G-solver, G-privileged, G-repro (no render disagreement); no attempt refused before 405 in any arm; W's determinism re-run matched on S's first four resets for both seeds |
| report | `outputs/task083-s-1/report.json`, sha256 **`b6f7c898cfb5a2f24f46ca6b8ff809a09f9b9e4c560e27a74e4b19949c56038e`**; log `646e9b3a…71bb`; stdout empty |
| evidence | `~/develop/emai/evidence/task083-stages/` with `SHA256SUMS` (sha256 `cdbebcd9…88f0`) |

## 2. Counts on cohort S (75200–75327; counted success: at rest on the plate after a latched grasp and place)

| arm | successes | exact 95 % |
|---|---:|---|
| **W[66801]** (LeWM, seed 66801, flag not set) | **124** | 0.922–0.991 |
| N[66801] (action-blind) | 51 | 0.313–0.489 |
| L-shuf[66801] (scene-blind) | 36 | 0.205–0.368 |
| L-mean[66801] (scene-blind) | 60 | 0.380–0.559 |
| **W[66802]** (LeWM, seed 66802, flag not set) | **120** | 0.881–0.973 |
| N[66802] | 62 | 0.395–0.574 |
| L-shuf[66802] | 42 | 0.248–0.417 |
| L-mean[66802] | 56 | 0.350–0.528 |
| L-rand (shared) | 17 | 0.079–0.204 |
| **H-rule** (not learned, given the simulator's plate law; the comparator C) | **123** | 0.911–0.987 |
| H-sysid (not learned) | 121 | 0.891–0.978 |
| H-final(commit) (privileged look-ahead ceiling, not learned) | 126 | 0.945–0.998 |
| W[66800] (TASK-081's W; **reported only**, read by no row) | 123 | 0.911–0.987 |

## 3. The per-seed rows (§7.2) and the combined row (§7.3)

Paired differences are reset-bootstrap 95 % intervals (salt 8501); "W only / arm only" are the
discordant resets; McNemar tests are exact and one-sided.

| | seed 66801 | seed 66802 |
|---|---|---|
| G-bar (≥ 112/128) | 124: pass | 120: pass |
| G-NI, W − C (C = H-rule, 123 > 121) | **+1/128** (5 / 4), [−5, +7]: lower bound above −16, pass | **−3/128** (5 / 8), [−10, +4]: pass |
| W − N | +73 (75 / 2), p 2.0 × 10⁻²⁰ | +58 (62 / 4), p 1.0 × 10⁻¹⁴ |
| W − L-shuf | +88 (89 / 1), p 7.4 × 10⁻²⁶ | +78 (81 / 3), p 5.1 × 10⁻²¹ |
| W − L-mean | +64 (66 / 2), p 8.0 × 10⁻¹⁸ | +64 (68 / 4), p 2.3 × 10⁻¹⁶ |
| W − L-rand | +107 (107 / 0), p 6.2 × 10⁻³³ | +103 (105 / 2), p 3.6 × 10⁻²⁹ |
| S-VOID-CEILING | no (126 ≥ 112) | no |
| **row** | **L-PASS** | **L-PASS** |
| secondary "LeWM needed" (W > C, p < 0.01) | **not shown** (p 0.50) | **not shown** (p 0.87) |
| H-rule measurably better (upper bound < 0)? | no (interval includes 0) | no (interval includes 0) |

**Combined row: REP-PASS** (both seeds L-PASS individually). No clause (§9). Neither seed is
detectably inferior; nothing is pooled.

## 4. Reported only (§7.4)

- **W[66800] on this cohort** (TASK-081's W unchanged): 123/128 (0.911–0.987); W[66800] − H-rule
  0 (5 / 5) [−6, +6]; − W[66801] −1 [−7, +5]; − W[66802] +3 [−4, +10]. TASK-081's S had it at
  118/128 against H-rule's 125/128; on these resets the two are level.
- **Between seeds:** W[66801] − W[66802] = +4/128 (8 / 4), [−2, +11], two-sided exact McNemar
  p 0.39: no detectable difference.
- **Aim error against H-final(commit)'s aim, median [95 %] / p87.5 / max (cm):** W[66801] 0.321
  [0.280, 0.352] / 0.59 / 1.58; W[66802] 0.318 [0.279, 0.372] / 0.64 / 1.57; W[66800] 0.265
  [0.229, 0.312] / 0.52 / 1.05; H-rule 0.130 / 0.21 / 0.36; H-sysid 0.503 / 1.08 / 1.62; the twins
  1.8–2.8 in median; L-rand 6.5. W-66800's 0.295 cm on TASK-081's S is between.
- **τ-curve predictions** (a prediction through TASK-080's pooled τ curve, not a count): W[66801]
  116.1, W[66802] 115.0, W[66800] 117.8, H-rule 117.1, H-final 117.0 of 128; the closed-loop counts
  sit at or above them.
- **Final distance from the plate centre:** median about 3.5 cm for every W and for H-rule and
  H-final (W[66801] 3.49, W[66802] 3.46, H-rule 3.50, H-final 3.45), with 56–63 of 128 attempts of
  each ending 3.5–4.5 cm from the centre, against the scorer's 4 cm radius; the W failures that
  reached the place ended about 4.2–4.3 cm out in median. Counts near that boundary depend on more
  than the aim (TASK-081 §12).
- **affine_local:** no fallback and no clipped aim for any W; the twins clipped 5–41 aims.
- Per-attempt medians 13.5 s (W), 6.8 s (N), 4.5 s (H-rule), 8.1 s (H-final).

## 5. What this shows and what it does not

**Shows:** under C1-M, on 128 further fresh gated resets, the W models of TASK-077's two other
seeds, each with its own action-blind twin and its own readout and with TASK-081's controller
unchanged, each reached TASK-081's primary claim, "LeWM-driven closed-loop success": the 112/128
bar, the four twin tests at p < 0.01, and non-inferiority to H-rule within the allocated −16/128.
So TASK-081's L-PASS is not specific to the one model seed it was run with, nor to that seed's
`last_two_triggered` flag. The design note's solver choice and TASK-081's cohort size, both made on
66800's runs, held on two seeds that played no part in them.

**Does not show:**
- **"LeWM needed"**: for neither seed is W detectably better than H-rule, which is given the
  simulator's plate law. On this cohort H-rule is also **not** measurably better than either W
  (both intervals include 0), unlike TASK-081's S (−7/128, [−13, −2]); that is a reading of one
  cohort, not a reversal of TASK-081's finding, and W-66800 is level with H-rule here too.
- **A replication of the pipeline**: the three seeds share TASK-077's corpus, recipe, budget,
  featurisation and DINOv2 encoder; all three W training curves were lowest at their last point.
  Nothing here varies the condition, the encoder, the view, the simulator or the task.
- **The solver's effect** (still not shown; no solver comparison ran).
- Anything about v1's 0/150, Arena, the real G1 or a LeWM policy: LeWM chooses only the single
  place aim, and the plate law is an imposed simulator condition.
- The comparator and ceiling are shared by both seeds on the same resets, so the two passes are
  not independent of the cohort's difficulty; W[66800]'s 123/128 suggests this cohort was at least
  as easy as TASK-081's S for LeWM.

## 6. R7

REP-PASS's wording (§10) is applied to R7 by its own ruling, R21.20 ([DECISIONS.md](../DECISIONS.md),
decision 2026-10-09 (r)), with every qualifier kept.
