# Apple→Plate plate-readout twin v2: results (TASK-076)

**Outcome: TWIN-PASS, the claim row of §6.2. K-pred's row is PRED-INFEASIBLE** (escalate, no
clause). Neither abandonment clause fires.

**The claim, verbatim from the protocol (§1):** "Under TASK-074's 9 cm condition, P-3's pick
followed by e9's scripted place aimed at a frozen-DINOv2 ridge readout of the plate in the
current onboard 112 px frame reaches the calibrated place bar on fresh gated resets and beats an
image-free prior." **It does not say** (§1, verbatim list):
- anything about LeWM or any world model;
- that the pick, or the whole episode, is learned end to end (the pick is P-3, the place is
  scripted);
- that pretrained features are needed (the random-init floor is reported, not gated);
- anything about v1, its 0/150, Arena or the real G1.

**The numbers behind it** (gated cohorts S, seeds 56200–56263, and U, seeds 56300–56331; report
`outputs/task076-SU-1/report.json`, sha256 `242655f0…74326`). Intervals on counts are exact
(Clopper–Pearson) 95 % intervals, computed for this document from the counts.

| arm | what the place aims at | learned? | S (of 64) | U (of 32) |
|---|---|---|---|---|
| **H-twin** | R-plate's reading of the current onboard 112 px frame | P-3's pick is learned; the place is e9's scripted primitive; the target is a learned ridge readout | **64** [0.944, 1.000] | **31** [0.838, 0.999] |
| H-handover | the true moved plate (privileged) | no: the privileged ceiling | 64 [0.944, 1.000] | 31 [0.838, 0.999] |
| H-clock | K0's fitted per-step median plate (reads no image) | no: an image-free prior fitted on privileged labels | 51 [0.678, 0.887] | — |
| H-stale | P-3's post-look plate estimate (before the move) | the condition's check | 0 [0.000, 0.056] | — |
| H-floor | R-plate refitted on the seed-0 random-init DINOv2 | as H-twin, random-init features | 64 [0.944, 1.000] | — |
| P-stale | P-3 unchanged | learned (P-3) | 0 [0.000, 0.056] | 30 [0.792, 0.992] |

- **G1** (H-twin(S) ≥ 56/64): 64/64, passes.
- **G2** (H-twin(S) > H-clock(S), exact one-sided McNemar p < 0.01): b = 13 resets where H-twin
  succeeds and H-clock fails, c = 0 the other way, p = 0.5¹³ = 1.22 × 10⁻⁴. Passes.
- **G3** (H-twin(U) ≥ P-stale(U) − 2/32): 31 ≥ 30 − 2 = 28. Passes.
- **No void or escalation row fires.** S-VOID-CONDITION: H-stale(S) = 0 ≤ 8/64 and
  H-handover(S) = 64 ≥ 56. U-VOID-CEILING: H-handover(U) = 31 ≥ 28.
- **The paired H-twin − H-handover difference** on S is 0, with interval [0, 0]. Both arms
  succeeded on all 64 resets, so there is no discordant pair and the bootstrap interval is
  degenerate. It shows no measurable gap to the ceiling on these 64 resets. It is not an
  interval on a rate.
- **The determinism re-run** of H-twin on 56200–56203 matched on all four seeds (largest command
  difference 0.0).

**What this is, and what it is not.**
- **No LeWM and no world model is in the loop.** No arm of TASK-076 runs a predictor. This is not
  a LeWM-driven success under §9's definition, and it must not be reported as progress in LeWM
  successes.
- **H-twin is a hybrid.** It is P-3's learned pick plus e9's *scripted* place, aimed at a learned
  ridge readout of the camera. It is not a learned end-to-end policy.
- **H-handover and H-clock are not learned.** H-handover reads the true plate. H-clock reads no
  image, but its targets come from privileged labels on K0's development resets.
- **Pretraining is not shown to matter.** The random-init floor H-floor also scored 64/64, and
  offline the floor read the plate *better* than pretrained DINOv2 (§2.2). The plate position is
  readable even from random-init features, as the random-init floors were in TASK-063/064, and as
  in Stage O.
- **One run, one camera (onboard 112 px), one encoder and one floor seed, simulation only**, on
  one condition (TASK-074's 9 cm move at step 300) and one reset distribution.

**The status sentence is unchanged.** R7's canonical sentence (DECISIONS 2026-10-02) stays as it
is. Every clause in it is still true, and this result is not a learned Apple→Plate policy or a
LeWM result (ruling R13.1, decided by Claude under owner delegation; §6). Verbatim:

> Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend
> (`native_jepa` and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an
> MLP on a frozen DINOv2 readout, trained on demonstrations from the privileged scripted expert
> e9; not a world model) scored 40/40 counted successes on the held-out cohort C against 39/40
> for its random-init encoder control R-3 (one run, one training seed per arm, 40 resets, one
> camera at 112 px onboard, a narrow reset distribution), so TASK-072 M2 is M2-FAIL on G3
> (encoder pretraining contributed nothing measurable), and cohort C is no longer held out. No
> LeWM-driven controller has run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate
> runs are on v1, with 0 successes. Scripted-expert, privileged-ceiling, oracle and GR00T
> successes are not project-learned results. (LeWM also ran in closed loop on
> the TASK-014 development reach pilot, a reach task, not Apple→Plate: its v2 target-space
> selector reached 1/5 goals, with intervals overlapping the 0/5 controls.)

**Next step, per the protocol.** For TWIN-PASS (§13.3): "the claim of §1; results PR; the next
LeWM task is chosen by K-pred's row". **K-pred's row is PRED-INFEASIBLE**: "escalate, no clause;
PLAN.md's Branch B" (§13.3). As §9 says: "If the row is PRED-NONE, PRED-NEAR, PRED-INFEASIBLE,
PRED-NO-BAR or PRED-NOT-RUN, Option 1 has not advanced the LeWM goal beyond calibration. PLAN.md's
Branch B applies." No TASK-077 is admitted under cell A. The follow-up is the Branch B design note
in [apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (#135, merged as `ff37d96`; a
DRAFT design note under R9.8 of DECISIONS 2026-10-04; §5).

Protocol: [`apple_plate_twin_v2.md`](apple_plate_twin_v2.md), FROZEN and merged as `702a7d9`
(#137) on an independent reviewer's reported APPROVE. Manifest:
`benchmarks/manifests/apple-plate-twin-v2.json` (frozen sha256
`011dff8ac65dde7341850ac4c4d9c569b399f1e3a8d2a1e978d3a3b9bbb7570b`, unchanged). Stage-0 record:
[`apple_plate_twin_v2_stage0.md`](apple_plate_twin_v2_stage0.md). The pre-run GOs are comments
on PR #134 (K0) and PR #137 (O, D with K-pred, S/U).

---

## 1. Stage-by-stage record

K0 ran from the main checkout at `2d0bdb7`, before the freeze (protocol §5.1). Every later stage
ran once, from the run worktree `/home/huhn/develop/emai/worktrees/task076-run` (made with
`scripts/new_worktree.sh --run`) at `702a7d9cc041efb4b2ab14772197210c03051e17`, with a clean
tracked tree, each on its own independent reviewer's reported GO. The reports are git-ignored and
stay on the Linux PC. Times are UTC, from each `report.json`.

| stage | revision | report (sha256) | started → ended (total s) | outcome |
|---|---|---|---|---|
| K0 (development, before the freeze) | `2d0bdb7` | main checkout `outputs/task076-k0-1/report.json` (`ef4b541067dc969ee5ebfc9ee78e2c719a4b84e4d3ed0d71a167feb6fcd0c876`) | 2026-10-04T17:52:07Z → 17:57:27Z (319.75) | **K0-PASS** |
| O (offline admission) | `702a7d9` | `task076-run/outputs/task076-O-1/report.json` (`77ff37581e65420199cb09525ad261c86ce1be13f2b571c9ad17d5fcd71b05fd`) | 18:37:22Z → 18:39:43Z (140.9) | **O-PASS** |
| D (development closed loop) | `702a7d9` | `task076-run/outputs/task076-D-1/report.json` (`a84c00e2e487134973a775bb553640e5ec512de45e6e8de2cdf18886d178e93f`) | 18:45:53Z → 18:47:07Z (74.5) | **D-PASS** |
| K-pred, cell M-a | `702a7d9` | `task076-run/outputs/task076-Kpred-Ma-1/report.json` (`d512a7126a3e87386fbee07b6f3769b9eb155493fd355079ca370d9006a7106e`) | 18:48:24Z → 18:51:00Z (156.0) | KPRED-DONE (reported only) |
| K-pred, cell M-b | `702a7d9` | `task076-run/outputs/task076-Kpred-Mb-1/report.json` (`0c976384246a8125195a77443b425712bc48cc2401b43114aa17b299831a9130`) | 18:53:46Z → 18:56:20Z (154.7) | KPRED-DONE (reported only) |
| S/U (gated) | `702a7d9` | `task076-run/outputs/task076-SU-1/report.json` (`242655f009aa132911bf526788819186964ae8aa6351878b0f5cb62627474326`) | 19:00:59Z → 19:08:00Z (421.76) | **TWIN-PASS** |

GO comments: K0
[#134 5982693642](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/134#issuecomment-5982693642);
freeze APPROVE
[#137 5983084117](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/137#issuecomment-5983084117);
Stage O
[#137 5983130529](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/137#issuecomment-5983130529);
D and K-pred
[#137 5983196642](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/137#issuecomment-5983196642);
S/U
[#137 5983322337](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/137#issuecomment-5983322337).
No stage voided, and none was repeated.

**Common to every post-freeze report.** `frozen_sha256` is `011dff8a…570b` and
`protocol_status` is FROZEN. `revision_at_end` equals `revision`. `tracked_tree_dirty` and
`smoke` are false, and there is no void record. All 84 pins at the end equal the pins at
preflight. `non_finite_fields` is empty. The encoder digests are pretrained `3a697b87…2af27` and
floor `546b9011…729d2`. The platform was the Linux PC: RTX 5080, MuJoCo 3.13.0,
`MUJOCO_GL=egl`. D, K-pred and S/U ran on the CPU with 6 workers and without the GPU lock (§8);
only Stage O's featurisation took the lock. Every simulating stage re-ran G-repro inside itself,
and all eight of TASK-072 run-1's reproduction checks were true in D, M-a, M-b and S/U.

| stage | start load (1 / 5 min) | peak tree PSS (GiB, ceiling 12) | first cohort render or outcome boundary |
|---|---|---|---|
| O | 1.13 / 1.16 | 4.17 | `first_outcome_utc` 18:37:52Z |
| D | 1.49 / 1.71 | 8.86 | 18:46:38Z |
| M-a | 1.12 / 1.90 | 8.85 | 18:49:09Z |
| M-b | 0.55 / 1.87 | 8.80 | 18:54:30Z |
| S/U | 0.20 / 1.36 | 9.36 | 19:01:44Z |

**The caps.** The 7 200 s invocation cap is checked only between arms (S/U GO, note a), so the
results check it here: every stage's `total_seconds` is far below it (largest: S/U, 421.76 s).
The largest single attempt over all closed-loop stages (K0, D, M-a, M-b, S/U) was 6.31 s (an H-twin
attempt in D, seed 56122; the largest in S/U was 6.24 s), against the 300 s per-attempt cap.

### 1.1 K0 (K0-PASS; recorded in protocol §5.1)

Cohort K, seeds 56000–56031 (development, now spent). The τ curve (H-handover with a planted
target error; counted successes of 32):

| planted error (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| counted successes / 32 | 32 | 31 | **28** | 23 | 17 | 18 | 16 | 8 | 5 |

τ_re = 1.0 cm. N_K(0) = 32/32 [0.891, 1.000]. H-clock(K) = 24/32 [0.566, 0.885], in-sample.
H-stale(K) = 0/32 [0.000, 0.109]. No stop fired. G2's predicted feasibility (reported only):
b = 8, c = 0, scaled point p = 1.53 × 10⁻⁵, predicted pass probability 0.9983. These values are
`K0_MEASURED` in the frozen block.

### 1.2 Stage O: O-PASS

- **Roots.** 253 read (225 train, 28 val), as the frozen `READ_ROOTS` requires, giving 1 518
  decision-frame rows and 1 518 plate-hidden rows. The outer folds hold 51, 51, 51, 50 and 50
  roots. `test_split_decoded` is false.
- **`decoded_episodes` is 270, against 253 read** (caveat, explained in the D/K-pred GO, reason
  2, from the code). `load_rows` opens every root of the read splits: 240 train + 30 val = 270.
  The reader counts a root as decoded before the band check, and the band check then drops the
  17 roots (15 train, 2 val) whose band ends at or before step 400. All 17 are train or val
  roots. The reader refuses the test split before it opens any file, so no test root was
  decoded.
- **Featurisation.** 3 036 frames in 7.26 s on the GPU, under the shared lock
  (`gpu_lock_held` true). CPU anchor check: largest |difference| 2.19 × 10⁻⁵ on 32 frames (bound
  10⁻³). GPU peak 0.48 GiB allocated, 0.57 GiB reserved (cap 3.0 GiB). The cross-fit took
  101.1 s.
- **Selected λ (relative).** R-plate 0.1 in folds 0, 1, 2 and 4, 0.01 in fold 3. R-plate-floor
  0.1 in every fold. R-plate-pool 0.01 in folds 0, 2 and 3, 0.1 in folds 1 and 4.
- **The closed-loop readouts**, fitted after the outcome boundary on all 1 518 rows (253 groups),
  both at λ_rel 0.1. Parameter digests: R-plate `e89e605b…c2bd`, R-plate-floor `0a59e9b9…aded`.
  Every D, K-pred and S/U worker checked them. The per-frame error file `errors.npz` has sha256
  `a90f8c11…bb35`.

## 2. Stage O in numbers (§6.1)

### 2.1 The gates

τ_re = 1.0 cm. Medians are over 1 518 frames on 253 roots. The intervals are root-clustered
bootstrap percentile intervals (10 000 resamples, 95 %, salt 7604).

| gate | condition | value | result |
|---|---|---|---|
| **O1** precision | upper bound of R-plate's median error ≤ τ_re | median 0.403 cm [0.378, **0.416**] | pass |
| **O3** clock prior | upper bound of median(e_R-plate) / median(e_clock) < 1.0 | 0.093 [0.085, **0.106**] | pass |
| **O4** plate hidden | lower bound of R-plate's median error on plate-hidden frames > τ_re | median 4.132 cm [**3.713**, 4.659] | pass |

Row, first match: no V, O4 holds (not TWIN-OFF-ARM), O1 and O3 hold (not TWIN-OFF-FAIL), so
**O-PASS**. The clause does not fire.

### 2.2 Reported only (they gate nothing)

| quantity | value |
|---|---|
| offline clock prior, median error | 4.320 cm [3.803, 4.626] |
| **R-plate-floor** (random-init, seed 0), median error | **0.278 cm [0.260, 0.297]** |
| R-plate / R-plate-floor | **1.446 [1.355, 1.526]** |
| **c_plate**: R-plate-pool (pooled 4 × 4 LeWM latent), median error | 0.626 cm [0.596, **0.662**]; c_plate = 0.662 cm |
| O2: R-plate's 87.5th percentile | 0.806 cm [0.738, 0.857] |
| O2: τ-curve-mapped predicted count (K0's curve) | 30.54 of 32 [30.34, 30.71], written before any closed loop |
| per-step median error, steps 405 / 421 / 437 / 453 / 469 / 485 | 0.452 / 0.432 / 0.403 / 0.404 / 0.372 / 0.345 cm |
| signed mean error, x | +0.014 cm [−0.029, +0.060] |
| signed mean error, y | +0.001 cm [−0.031, +0.033] |

**Caveat: the random-init floor beats R-plate.** R-plate-floor's median error is 0.278 cm,
against 0.403 cm for pretrained R-plate, and the ratio's interval [1.355, 1.526] excludes 1. The
protocol makes the floor reported only (§6.1: "the random-init floor tied P-3 in M2, and this
task does not ask whether pretraining helps"), so it enters no row. It means that the plate's
position is read here from features a randomly initialised ViT also provides. Pretraining is not
what makes the readout work. This matches the random-init floors of TASK-063/064, which met
those tasks' bars on their own, and M2's R-3 (39/40). The closed loop agrees (§3.3).

**c_plate.** K-P5 (c_plate ≤ τ_re) would hold: 0.662 ≤ 1.0. It decides no row here, because
cell A was removed (§4). It is recorded for any later LeWM task that reads the plate from the
pooled latent, with the TASK-074 lesson attached: c_plate ≤ B ≤ τ_re.

### 2.3 The O2 text in the protocol (corrected in this PR)

The Stage O GO's note b found that §6.1's O2 paragraph quoted TASK-075's
τ curve ("22/32 at 1.5 cm, 23/32 at 2 cm") instead of K0's (23/32 at 1.5 cm, 17/32 at 2 cm,
non-monotone from 2 to 2.5 cm, 17 → 18). The paragraph is reported-only text; the frozen block
and every bar are unaffected. This PR corrects the sentence in the protocol, labelled "Erratum
2026-10-04" with the original wording kept (the TASK-058 convention), and re-records the document's
sha256 in the manifest: `737280a1…7570` before, `c56292b8…41e5` after (ruling R13.2; §6).

## 3. Stage D and Stage S/U (closed loop)

### 3.1 Stage D: D-PASS

Cohort D, seeds 56120–56135 (development), with TASK-074's step-300 shift. Nothing was refitted
on D.

| arm | count (of 16) | exact 95 % interval | bar (TWIN-DEV-STOP below it) |
|---|---|---|---|
| H-twin | 15 | [0.698, 0.998] | 12 |
| H-handover (privileged, not learned) | 16 | [0.794, 1.000] | 14 |

H-twin failed only on seed 56127. There its readings were 1.88–2.93 cm from the moved plate, all
offset in +x (computed from the report's logged readings and the logged shift); H-handover
succeeded on the same seed. The S/U GO verified D-PASS by hand and pinned D's sha256 (GO note c
of the Stage O GO: `stage_gated` has no D-PASS guard; §7).

### 3.2 Stage S/U: TWIN-PASS

The table at the top gives the counts. Further detail from the S/U report:
- **H-clock's 13 failures** (seeds 56205, 56206, 56207, 56212, 56214, 56218, 56226, 56235,
  56246, 56250, 56255, 56261 and 56263) all latched a grasp and a place but did not leave the
  apple at rest on the plate; the final apple–plate distance was 4.21–4.60 cm. Its targets are
  the same at every decision step, plate xy = (0.48204, −0.18196) m, because the plate is static
  after the step-300 shift.
- **H-stale and P-stale fail on all 64 S resets.** The condition needs a reading after the move,
  as K0 found (H-stale(K) 0/32).
- **U.** Seed 56314 failed in all three U arms with `guard_refusal` at step 223, before any
  grasp. That is P-3's pick, which the three arms share, not the place. P-stale also failed on
  56322 (grasp and place latched, apple 4.06 cm from the plate, not at rest).
- **No privileged read** in a non-privileged arm: `privileged_ok` is true on every attempt of
  every arm (`check_privileged`; H-twin, H-clock, H-stale, H-floor and P-stale are the
  non-privileged arms).
- **Render.** `render_disagreements` and `rerender_disagreements` are empty in the S/U report.

### 3.3 The readout in the loop (descriptive; computed from the S/U report's logged readings)

The per-decision readings are logged with every attempt. Against the true moved plate (S) or the
reset plate (U), over six decisions per attempt:

| arm | readings | median error (cm) | 87.5th percentile (cm) | largest (cm) |
|---|---|---|---|---|
| H-twin, S | 384 | 0.273 | 0.622 | 1.149 |
| H-floor, S | 384 | 0.180 | 0.379 | 0.852 |
| H-twin, U | 186 | 0.282 | 0.558 | 1.500 |

These are not report fields and gate nothing; they are recomputed from the report for this
document. Every computed percentile in this document uses numpy's default (linear) method, the
repository's convention and the one behind Stage O's 87.5th percentile. The closed-loop error is below Stage O's cross-fitted offline error (0.403 cm), and the
random-init floor is again more precise than the pretrained readout.

## 4. Stage K-pred (reported only; its row is PRED-INFEASIBLE)

### 4.1 The row

Cell A, the only admitting cell, was removed at Stage 0 (R8.16: the plate moved a median of
0.060 cm after the last decision under H-final, against the 2 cm needed). With Stage O at O-PASS,
`decide_kpred(None, offline_row="O-PASS", c_plate_cm=0.6616, tau_re_cm=1.0)` returns:

> `{'row': 'PRED-INFEASIBLE', 'reason': 'cell A removed at Stage 0', 'consequence': "escalate, no clause; PLAN.md's Branch B"}`

**K-pred's row is PRED-INFEASIBLE: escalate, no clause.** K-pred's own clause fires only on
PRED-NONE, so it does not fire. K-P1–K-P4 and m2\* were never measured, because they are defined
on cell A. Only the constant-velocity cells M-a and M-b ran, and they never admit (§6.3).

### 4.2 The M cells (reported only, never bars)

No step-300 shift. The plate moves at constant velocity from s0 = 405 to s1 = 525 along a
direction drawn from the −y arc (salt 7606), 12 cm in M-a (4.0 cm still to go at the last
decision, 485) and 9 cm in M-b (3.0 cm to go). Counts of 32, exact 95 % intervals:

| arm | aimed at | learned? | M-a (56040–56071) | M-b (56080–56111) |
|---|---|---|---|---|
| H-final | the plate's scheduled position at s1 | no: privileged | 32 [0.891, 1.000] | 32 [0.891, 1.000] |
| H-now | the true plate at the decision step | no: privileged | 7 [0.093, 0.400] | 1 [0.001, 0.162] |
| H-twin | R-plate's reading of the current frame | the readout is learned; the place is scripted | 11 [0.186, 0.532] | 4 [0.035, 0.290] |
| H-cv | a least-squares constant-velocity line through H-twin's readings, extrapolated to s1 | no: hand-written, knows the M family's form and s1 | 24 [0.566, 0.885] | 24 [0.566, 0.885] |

Paired with H-final on the same resets (b = H-final succeeds and the other fails; c = the
reverse): H-now b = 25, c = 0 (M-a) and 31, 0 (M-b); H-twin 21, 0 and 28, 0; H-cv 8, 0 and 8, 0.

- **No move was refused** and no apple–plate contact happened before s1 in any arm of either cell
  (earliest contact at step 608). The executed 2-step palm term was at most 0.0104 cm.
- **H-twin's reading error per cell** (GO note c; computed from the logged readings against the
  scheduled plate position at each decision step): median 0.354 cm, 87.5th percentile 0.688 cm
  in M-a; 0.326 and 0.583 cm in M-b, over 192 readings each. That is slightly above the
  static-plate error of §3.3 (0.273 cm), so R-plate kept reading a moving plate.
- **Where H-cv loses its 8/32.** H-cv's last aim (at 485) missed the plate's position at s1 by a
  median of 1.280 cm (87.5th percentile 1.763 cm) in M-a and 0.817 cm (1.469 cm) in M-b, against
  τ_re = 1.0 cm (computed from the logged extrapolations). H-twin's and H-now's last aims missed
  by about the distance still to go (4.1 and 4.0 cm in M-a; 3.2 and 3.0 cm in M-b). So H-cv's
  shortfall comes from extrapolating noisy single-frame readings 40 steps ahead, not from
  anything the robot's action does.
- **Not explained, and not interpreted:** H-now scored 7/32 in M-a but 1/32 in M-b, although M-b
  has less distance to go; and H-twin scored above H-now in both cells (6 against 2 discordant
  pairs in M-a, 3 against 0 in M-b). The directions differ per reset, τ is anisotropic, and the
  cells are small.

### 4.3 What M-a and M-b suggest for the next task

**H-cv sits 8/32 below H-final in both cells**, exactly the size of the +8/32 headroom bar. The
protocol expected the opposite in advance ("H-cv, which knows the constant-velocity form and s1,
is expected to reach about H-final", §3.2), so that expectation did not hold. **But the M cells
are action-independent:** the plate's motion does not depend on the robot's action, so an
action-blind predictor is expected to tie an action-conditioned one there (§3.2). A world model
could at most do better than H-cv by extrapolating more precisely, which a better action-blind
filter could do too. That does not satisfy §9's requirement that the LeWM arm beat an
action-blind predictor. So M-a and M-b do not admit a LeWM task, as the protocol says, and they
do not change the row.

They suggest two things for the Branch B design note (#135), both descriptive:
- **Not predicting costs a lot when the plate is still moving** (H-now 7/32 and 1/32; H-twin
  11/32 and 4/32), and a hand-written constant-velocity extrapolator recovers most of it but not
  all (24/32). In a moving-plate condition, a predictor's job is bounded by perception noise
  amplified over the look-ahead, not only by the motion model.
- **The admitting cell must make the target depend on the robot's own action**, which neither M
  cell does. Cell A tried and could not (R8.16). The design note's candidate C1 is the next attempt at
  such a condition.

## 5. The rows and what follows

| stage | row | consequence (protocol §13.3) |
|---|---|---|
| K0 | K0-PASS | K0's values entered the frozen block |
| O | O-PASS | Stage D and Stage K-pred may run, each on a GO |
| D | D-PASS | Stage S/U may run on a GO |
| S/U | **TWIN-PASS** | "the claim of §1; results PR; the next LeWM task is chosen by K-pred's row" |
| K-pred | **PRED-INFEASIBLE** | "escalate, no clause; PLAN.md's Branch B" |

**The next-step line, per the protocol: the K-pred row decides the next LeWM task.** The row is
PRED-INFEASIBLE, so PLAN.md's Branch B applies: no TASK-077 under cell A, and Option 1 has not
advanced the LeWM goal beyond calibration (§9). Branch B's entry for PRED-INFEASIBLE (recommended
under R8) is a design note for a different action-dependent target, with its own K0-style
headroom check and no world model. That note is
[apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (#135, merged as `ff37d96`; a DRAFT
design note, not a protocol), under R9.8 of DECISIONS 2026-10-04, which splits a later claim into a
gated "LeWM-driven closed-loop success" and a reported-only "LeWM needed". Its next step is a
CPU-only feasibility record for its candidate C1, and any task from it needs its own
preregistration.

**Neither clause fires.** The twin clause fires only on TWIN-OFF-FAIL, TWIN-FAIL or TWIN-HARM;
K-pred's only on PRED-NONE. Nothing is closed by this task. Never closed by it (§7): the LeWM
backend, DINOv2 as an encoder, the v2 task, Arena and the product goal.

**What TASK-076 hands a later LeWM task** (§0, §9), measured here:
- τ_re = 1.0 cm (K0) and c_plate = 0.662 cm on the pooled 4 × 4 latent (Stage O), so a
  predicted-latent plate bar B with c_plate ≤ B ≤ τ_re is feasible on static frames of this
  corpus. Any condition with a moving or occluded plate needs its own ceiling.
- The perception baseline: H-twin 64/64 on S under TASK-074's condition. Any later LeWM place
  planner under that condition would be reported beside it, and could at best tie it.

## 6. Rulings (DECISIONS 2026-10-04, R13.1–R13.3; each decided by Claude under owner delegation)

The prefix R13 is new and unused elsewhere: R10 is taken by TASK-067's owner ruling and by
TASK-075's protocol, R11 by TASK-068's owner ruling, and R12 by the owner ruling that defines
`apple-to-plate-v2`.

- **R13.1 — R7's canonical sentence is unchanged.** Every clause in it is still true after
  TASK-076: v1 is 0/150 per backend; P-3's M2 result is as stated; no LeWM-driven controller has
  run in closed loop on v2; and privileged-ceiling successes are not project-learned results.
  TWIN-PASS is a hybrid (a learned pick, a scripted place and a learned perception readout), with
  no world model, and its random-init floor scored the same. Adding it to the sentence would
  invite reading it as learned or LeWM progress. Documents that cite it use this wording:
  "TASK-076 TWIN-PASS: P-3's learned pick plus e9's scripted place aimed at a frozen-DINOv2 ridge
  readout of the plate scored 64/64 on gated resets under TASK-074's 9 cm condition (random-init
  floor also 64/64; one run, simulation only); no world model; not a learned end-to-end policy."
- **R13.2 — the O2 sentence in the protocol is corrected after the freeze.** It quoted TASK-075's
  τ curve, not K0's (Stage O GO, note b). It is reported-only text, the frozen block and every
  bar are unchanged, and the correction is labelled "Erratum 2026-10-04" in place, keeping the
  original wording (the TASK-058 errata convention). The manifest's
  `protocol_document_sha256` is re-recorded (previously `737280a1…7570`).
- **R13.3 — next: Branch B, through the design note of #135.** K-pred's PRED-INFEASIBLE row
  decides the next LeWM task (§13.3). No TASK-077 is opened under cell A. The follow-up is
  [apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (#135, merged as `ff37d96`), a DRAFT
  design note, not a protocol; any candidate it recommends needs its own feasibility record and
  preregistration. No H-twin variant (a hand crop, more seeds, another encoder) follows
  in TASK-076 (R8.13).

## 7. Process notes from the GO reviews

- **The O2 sentence** (Stage O GO, note b): corrected in this PR (§2.3, R13.2).
- **G2's redundant condition** (S/U GO, note b): `decide_gated`'s G2 also requires
  n_twin > n_clock. Under the one-sided exact test p < 0.01 already implies it, so it is
  harmless. Here 64 > 51 and p = 1.22 × 10⁻⁴.
- **The invocation cap is checked only between arms** (S/U GO, note a): each `pool.map` gets the
  full 7 200 s, so a run that crossed the cap inside the last arm or the re-run would not be
  voided. It cannot matter here: S/U's `total_seconds` is 421.76 s (§1).
- **`stage_gated` has no D-PASS guard** (Stage O GO, note c; D/K-pred GO, reason 12): it checks
  only the O-PASS report, although its error message names D-PASS. A code guard would have been
  a post-freeze change to a pinned file, so the S/U GO reviewer checked D-PASS by hand instead,
  and pinned D's report sha256 `a84c00e2…e93f` in the GO.
- **`READ_ROOTS` is not asserted at run time** (Stage O GO, note a): the D/K-pred GO checked
  `roots.by_split` = 225 / 28 against it before D ran.
- **One re-render disagreement in M-a's G-repro.** On TASK-072 training root 51235, two of three
  re-renders agreed, and the third differed by one level in 4 pixels, with equal simulator
  states. The render-majority rule decides such a case by design (the third render, then the
  majority; no majority is a V), and all eight G-repro checks were true. It is not a V. D, M-b and
  S/U had no disagreement.
- **Keep the run worktree.** The workers loaded the closed-loop readouts from absolute paths
  under `task076-run/outputs/task076-O-1/`, and every post-freeze report lives in that worktree.
  It and the `--evidence` root `worktrees/task076-evidence` are kept as this task's evidence
  ([STORAGE.md](../STORAGE.md)).

## 8. Process

- #134 (Stage 0) and #137 (K0 values and the freeze) merged on independent reviews; K0 ran at
  `2d0bdb7` on its GO; O, D, K-pred (M-a, M-b) and S/U each ran once at `702a7d9` on their own
  reported GOs, in the order O, D, M-a, M-b, S/U. No stage voided, and none was repeated.
- Nothing was re-thresholded, refitted or re-selected after numbers were seen. The frozen block,
  its sha256 and the 84 pins are unchanged. The only protocol text change is the O2 correction
  (R13.2).
- No experiment was run for this document. Every count, gate value and row is read from the six
  reports above. The exact count intervals, the paired discordant counts of the M cells and the
  closed-loop reading errors are computed here from those reports, and are labelled as such.
