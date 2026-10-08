# Apple→Plate LeWM committed aim under C1-M: TASK-081's L-PASS replicated with W's two other model seeds (TASK-083)

**STATUS: DRAFT** (R21.1–R21.12, decided by Claude under owner delegation; not frozen, nothing
run). No seed of this task's block (75000–75999) has been simulated, nothing is trained and
nothing is fitted. The freeze needs Stage 0 (§6.1) and an independent reviewer's reported APPROVE;
the gated Stage S needs its own reported GO.

- **Admitted by:** R19.23 (3) (DECISIONS 2026-10-08 (j)), a recommendation recorded for the owner:
  "before any broader claim, robustness of this result: more model seeds of W (66800 is flagged)
  … each preregistered", and R19.25's note that "R19.23 (3)'s two extra W seeds under C1-M need no
  training (R-S exists for 66801 and 66802)". The owner has been told that this replication will
  run.
- **Rulings:** R21.1–R21.12 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (o). Each is
  **decided by Claude under owner delegation (2026-09-30)**.
- **Task card:** `.mc/tasks/todo/TASK-083-replicate-task-081-s-l-pass-with-w-s-other-two-model-seeds.md`.
- **Template:** TASK-081's frozen protocol,
  [apple_lewm_commit_precision_v2.md](apple_lewm_commit_precision_v2.md) (frozen block
  `77ccc636…705a`). Where this document says "carried", the TASK-081 section named is in force
  unchanged (and through it TASK-080's and TASK-077's); only the differences are written out.

The canonical status sentence (DECISIONS 2026-10-02, R7, as changed by R19.21) is not changed by
this draft. Its v2 LeWM part states TASK-081's L-PASS: one run, **one model seed** (66800, flagged
`last_two_triggered`), 118/128, under C1-M. This task asks whether that result holds for the two
other model seeds of the same training run.

---

## 0. What changes from TASK-081, stated first

TASK-083 keeps TASK-081's closed loop, solver, arms, gates, bar, margin and row ladder, and changes
only:

1. **The model seed.** W and its action-blind twin N are the checkpoints of TASK-077's seeds
   **66801** and **66802** (each W paired with its own seed's N), read by that seed's R-S and R-N
   from TASK-080's Stage R. TASK-081 ran seed 66800 only. Nothing is trained or fitted (§2.2).
2. **Two seeds on one cohort.** Both seeds' candidate arms (W, N, L-shuf, L-mean) run on the
   **same 128 fresh resets**, and the seed-independent arms (L-rand, H-rule, H-sysid,
   H-final(commit)) run **once** on them and serve both seeds' tests (§4).
3. **Each seed is tested on its own** with TASK-081's ladder (§7.2), and the replication claim
   requires **both** seeds to reach L-PASS (§7.3). There is no pooling across seeds.
4. **No abandonment clause** (§9): TASK-081's L-PASS is a gated result; a replication that fails
   narrows what R7 may say about it, and R7 states the outcome whatever it is (§10).
5. **Reported only, W-66800 on the same resets** (TASK-081's W unchanged), so that the cohort can
   be compared with TASK-081's S. It is read by no row. **Dropped**: W-frozen, H-read and H-now
   (reported only in TASK-081, read by no row; R19.23 (4) recommends no further solver work).
6. **No Stage D** (§6.3), **fresh seeds** 75000–75999 and **fresh salts** 8501–8512 (§5).
7. **A new thin runner** (`scripts/run_lewm_rep_v2.py`), because TASK-081's frozen runner takes the
   model seed from its frozen block (`PRIMARY_SEED = 66800`) and cannot be pointed at another seed
   without editing a hash-pinned file (§2.3).

LeWM still makes **one decision**: the place aim committed at step 405, between P-3's learned pick
(behaviour cloning, not a world model) and e9's scripted place primitive, under the declared
simulation-only plate condition C1-M. The claim, if any, is "LeWM-driven aim selection at 405",
replicated across model seeds of one training run; not "a LeWM policy".

## 1. The question and the claims

**Question.** Under C1-M, with TASK-081's controller unchanged (affine_local after the grid, R-S,
τ_commit 1.0 cm), do the W models of TASK-077's other two seeds, 66801 and 66802, each reach R9.8's
primary claim on 128 fresh gated resets?

**Per-seed primary claim** (R9.8, TASK-081 §1, for s ∈ {66801, 66802}, all on cohort S):
1. W[s] makes the decision from the encoded current frame with no privileged read at run time
   (`task_truth_in_controller` = 0).
2. **G-bar:** W[s] ≥ 112/128.
3. **G-NI:** W[s] non-inferior to the better of H-rule and H-sysid on S within δ = 16/128 (an
   allocation, not calibrated; carried from TASK-081).
4. **The twin tests:** W[s] beats N[s], L-shuf[s], L-mean[s] and L-rand, each by an exact one-sided
   McNemar test at p < 0.01.

A seed whose items 2–4 all hold is **L-PASS** for that seed.

**The replication claim, "TASK-081's L-PASS replicates across W's model seeds"**: both seeds are
L-PASS (§7.3, REP-PASS). It is a conjunction of the two per-seed claims: each is tested at its own
level, and the conjunction rejects only if both do, so its error rate is at most either test's and
no multiplicity adjustment is needed or applied (an intersection–union test). Any statement about
one seed alone is reported per seed and is not the replication claim.

**Secondary, "LeWM needed"** (per seed, reported only, never a gate, not expected): W[s] detectably
better than the better of H-rule and H-sysid (exact one-sided McNemar p < 0.01). H-rule scored
125/128 on TASK-081's S and is given the simulator's plate law.

**What REP-PASS would show:** that TASK-081's result is not specific to the one model seed it was
run with: two further W models trained by the same recipe on the same corpus, each with its own
action-blind twin and its own readout, each reach TASK-081's gated bar, beat their twins and are
non-inferior to H-rule within the same allocated margin, on 128 further fresh resets.

**What it would not show:** TASK-081 §1's list unchanged (that LeWM is needed; a LeWM policy;
anything about v1's 0/150, Arena or the real G1; that pretraining matters; other conditions, grids
or horizons); a replication across training corpora, recipes, encoders, conditions or simulators
(the three seeds share TASK-077's corpus, recipe, featurisation and DINOv2 encoder); that the
solver change of TASK-081 mattered.

## 2. What is carried, with its source

### 2.1 The condition, the controller and the arms' code (TASK-081 §2.1, §3, §4)

C1-M exactly as TASK-081 §2.1 records it (v2's `wide_reset_values`; P-3 with G-repro's post-look
estimates; the post-pick move at the observation of step 300, ρ\* = 4 cm, drawn per reset with
TASK-080's salt 8201; cell A's reactive-plate rule κ = −0.5, L = 2, s0 = 405, s1 = 525; one aim
committed at c = 405 and executed by e9's place primitive with no re-aim; the 147-candidate box;
r = 465, h = 60; TASK-080's scorer). The solver is TASK-081 §3.1's **affine_local**, for W and its
three model twins alike; L-rand draws a uniform feasible candidate.

Code: TASK-081's `lewm_cp_v2_runtime.py` (the worker: `run_task`, `worker_init`,
`choose_affine_local`, `choose_l_rand`) is **imported, not edited**; so are TASK-081's frozen block
`lewm_cp_v2.py`, TASK-080's and TASK-077's modules and `commit_precision_dev.py`. Their pins must
still match (G-frozen, §11). The worker already takes the model seed, checkpoint path and sha256
from its configuration (`lewm_c1m_v2_runtime.world_model`), so a seed is chosen by configuration,
with no code path changed.

### 2.2 Artifacts reused, each checked by sha256 before use (G-hash)

| artifact | source | sha256 (as recorded) |
|---|---|---|
| W-66801, N-66801, W-66802, N-66802 (and W-66800, N-66800 for W-66800 reported only) | TASK-077 Stage T, `task077-staget2` (66801, 66802) and `task077-staget` / `task077-staget2` (66800) | W-66801 `27aeadab…c193`, N-66801 `15f5dd41…2d37`, W-66802 `ba2614ec…97b4`, N-66802 `4f92a8fe…bfef` (W-66800 `891d2664…76b8`, N-66800 `51b51035…d2c2`); `lewm_pr_v2.REUSED["models"]` |
| R-S and R-N of seeds 66801 and 66802 | TASK-080 Stage R (R-PASS, R18.25), report `outputs/task080-rgate-1/report.json` in `task080-stager`, sha256 `bedb8966…48ea7` at revision `33cea5c` | content R-S[66801] `1cc85203…dc81`, R-N[66801] `206e664f…0243`, R-S[66802] `28abf388…f1a5`, R-N[66802] `33db51c9…d647`; files `r_s_66801.npz` `b171ca18…e98e`, `r_n_66801.npz` `3d232203…f524`, `r_s_66802.npz` `178d23a0…5744`, `r_n_66802.npz` `26f27542…abe2` |
| R-S and R-N of seed 66800 (W-66800, reported only) | the same report | content `6e05d223…1046`, `15405f3f…5271` (TASK-081's) |
| R-plate, L-mean's mean 405 latent, H-sysid's fit, the moments | TASK-077 Stage O, through TASK-080's `REUSED` | as TASK-081 §2.2 |
| τ_commit = 1.0 cm and the pooled τ curve | TASK-080 K0′, carried by TASK-081 §6.2 | 64, 58, 60, 32, 29, 7 of 64 at 0, 0.5, 1, 1.5, 2, 3 cm |
| TASK-081's frozen block and its six file pins | `benchmarks/manifests/apple-lewm-cp-v2.json` | frozen `77ccc636…705a` |

The full sha256s are in the Stage-0 frozen block. R-S and R-N of 66801 and 66802 were fitted in
TASK-080's Stage R by the same rule as 66800's (dual ridge on W's or N's predictions over the 1 995
old roots, λ selected by inner folds) and gated there on every seed (R0–R3 pass on all three
seeds; TASK-080 §8.4). They have never been used in a closed loop. The artifacts stay where TASK-077
and TASK-080 left them and are read only.

### 2.3 Why a new runner (R21.11)

TASK-081's runner `scripts/run_lewm_cp_v2.py` (pinned) builds the workers' configuration from
`lewm_pr_v2.PRIMARY_SEED` and reads only that seed's R-S and R-N from the Stage R report, and its
cohorts are its own seed ranges (`lewm_cp_v2.SEED_RANGES`); it has no argument for a model seed or a
cohort. Changing that means editing a hash-pinned file, which is not allowed. So Stage 0 adds a new
thin runner, `scripts/run_lewm_rep_v2.py`, and a new frozen block, `src/embodied_jepa/lewm_rep_v2.py`.
The runner copies the pieces of TASK-081's runner it needs **verbatim** (a test compares each
copy's source with TASK-081's, as TASK-081's own runner did with TASK-080's), and differs only in:
the model seed per worker pool (one pool per seed, run one after the other, each holding that
seed's W, N, R-S and R-N), the arm set of §4, this task's cohort and salts, and the per-seed and
combined ladders of §7. It loads no other script (`tests/test_no_runner_imports.py`).

## 3. The disclosures (R21.8)

- **Seed 66800 was selected earlier, and everything since was developed on it.** TASK-077's rule
  made 66800 the primary seed as the W with the lowest kept validation criterion (0.350399), before
  any closed loop (TASK-077 results §1.4). Every closed-loop run of LeWM on v2 since used only
  66800: TASK-080's Stage D (16/16) and Stage S (58/64, L-NEAR), the TASK-081 design note's
  development check, from which affine_local was chosen as the best of three executed variants
  after seeing them (62/64), and TASK-081's Stage D (16/16) and Stage S (118/128, L-PASS), whose
  128-reset cohort size was chosen after TASK-080's L-NEAR. Seeds 66801 and 66802 have never run in
  closed loop. So the solver, the cohort size and every development reading were chosen on 66800;
  a replication on 66801 and 66802 is the first test of them on seeds that played no part in those
  choices.
- **The training flags are what they are.** `last_two_triggered` is **false** for W-66801 (kept
  update 71 250, val criterion 0.364904) and W-66802 (80 750, 0.364660), and **true** for W-66800
  (95 000, 0.350399). All three W curves have their raw minimum at the last point (95 000); 66801's
  and 66802's flags are false only because the 1 % rule kept earlier points, whose last points are
  lower by 0.0034 and 0.0013 (TASK-077 results §3, caveat 2). So all three W models may be slightly
  under-trained at the fixed budget. No N is flagged (N-66801 kept 80 750, N-66802 38 000).
- **The other two seeds read slightly worse offline.** On TASK-080's gate-P (249 fresh roots), R-S
  on W's prediction read the plate at a median of 0.547 cm (66800), 0.584 (66801) and 0.589 cm
  (66802), with 95 % upper bounds 0.610, 0.657 and 0.643 cm against τ_commit = 1.0 cm; R-S's
  learning curve was flat at its end for 66800 and still falling for 66801 and 66802 (TASK-080
  §8.4). No offline aim (A1) was computed for 66801 or 66802; TASK-080's A1 (56.92 of 64) is
  66800's only.
- **W-66801's zero-command ratio is about 2.5 × the others'** (TASK-077's G4: 16.07 against 6.52
  and 6.10), unexplained and passing its gate.
- **The three seeds are not independent replications of the pipeline.** They share TASK-077's
  corpus, recipe, budget, featurisation and encoder; only the torch seed (initialisation and window
  sampling) differs.

## 4. Arms (R21.2)

| arm | model seed | what it is | readout | solver | read by |
|---|---|---|---|---|---|
| **W[66801]**, **W[66802]** | 66801, 66802 | that seed's W from the encoded 405 frame | that seed's **R-S** | affine_local | its seed's ladder |
| **N[s]** (action-blind) | s | that seed's N, zero commands | that seed's **R-N** | affine_local | its seed's ladder |
| **L-shuf[s]** (scene-blind) | s | W[s] from the logged 405 frame of **W[s]'s** attempt on the next reset in cohort order that reached 405, with this reset's candidate chunks (R17.20) | R-S[s] | affine_local | its seed's ladder |
| **L-mean[s]** (scene-blind) | s | W[s] from L-mean's mean 405 latent | R-S[s] | affine_local | its seed's ladder |
| **L-rand** | none | a feasible grid candidate drawn uniformly (TASK-081's salt 8303, carried, on the fresh resets) | none | none | both ladders |
| **H-rule** | none | `RuleCommit` (knows the plate law; not learned) | none | – | both ladders (comparator) |
| **H-sysid** | none | `SysidAim` with TASK-077 Stage O's fit (not learned) | none | – | both ladders (comparator) |
| **H-final(commit)** | none | the ceiling: `LookaheadAim` once at 405 in cloned state; **privileged, not learned** | — | – | both ladders (S-VOID-CEILING) |
| **W[66800]** (reported only) | 66800 | TASK-081's W unchanged (R-S[66800], affine_local) | R-S[66800] | affine_local | **no row** |

Every arm runs once per reset of S, paired. The seed-independent arms (L-rand and the three H arms)
do not load a world model, so one run of each serves both seeds; running them twice would add only
simulator noise to the comparator. The comparator C is the better of H-rule and H-sysid on S by
count (a tie to H-rule), the same C for both seeds. Infeasible candidates are dropped for every arm
alike. W[66800] has no twins here; its count is reported beside the cohort's other counts, so that
the cohort's difficulty can be read against TASK-081's S, and it gates nothing.

## 5. Seeds, cohorts and salts (R21.10)

**Block 75000–75999** (fresh). It lies outside every forbidden range of TASK-082's list
(`lewm_ul_v2.FORBIDDEN_RANGES`, which includes TASK-081's block 70000–71999, TASK-080's
65000–65999, TASK-077's 66000–68999 and every earlier cohort) and outside TASK-082's block
72000–74999; Stage 0's test re-checks this in code.

| seeds | use |
|---|---|
| 75200–75327 | **S**: the gated cohort (128 resets) |
| 75900–75999 | **debug**: runner mechanics only; nothing in it is read |
| everything else in 75000–75999 | reserved; any use needs its own ruling |

| salt | use |
|---|---|
| 8501 | every bootstrap of this task (10 000 resamples, reset-clustered): each seed's G-NI, every paired interval, every median interval |
| 8502 | Stage 0's power simulation |
| 8503–8512 | reserved; any use needs its own ruling |
| 8201 (TASK-080's, carried) | the move draw per reset, `default_rng(SeedSequence([8201, seed, k]))` |
| 8303 (TASK-081's, carried) | L-rand's draw per reset (TASK-081's worker draws with it; the resets are fresh, so the draws are) |

**The search** (2026-10-08, at `c0b2df6`). `git grep -E '\b75[0-9]{3}\b'` over `src`, `scripts`,
`tests`, `configs` and `benchmarks` on all 138 local and remote refs, and `grep` over every
worktree's `src` and `scripts` and over `docs/DECISIONS.md` and `.mc`, find no integer in
75000–75999. `git grep -E '\b85(0[1-9]|1[0-2])\b'` over `src`, `scripts`, `tests` and `configs`
on every ref and over every worktree's `src` finds none of 8501–8512. The union of every earlier
task's salts (`lewm_ul_v2`'s list) is 7701–7704, 7801–7802, 7901–7907, 8101–8112, 8201–8212,
8301–8312 and 8401–8412. No R21 label exists on any ref.

## 6. Stages

Nothing from S is simulated before its reported GO. Every stage runs from a clean worktree of the
merged revision (`scripts/new_worktree.sh <dir> --run --from <rev>`), after G-tests.

### 6.1 The sequence

1. **Stage 0 (after this draft's review).** New files only: `src/embodied_jepa/lewm_rep_v2.py`
   (the frozen-block candidate), `scripts/run_lewm_rep_v2.py`, `tests/test_lewm_rep_v2.py` and the
   DRAFT manifest `benchmarks/manifests/apple-lewm-rep-v2.json`. Tests: the seed ranges and salts
   against every forbidden range and earlier salt; the reused artifacts' sha256 checks (the four
   checkpoints, R-S and R-N of each seed by content and file sha256, TASK-081's frozen block and
   six file pins, the Stage R report); that each seed's pool loads that seed's W, N, R-S and R-N
   and no other (a wrong-seed configuration is refused); that W[s], L-shuf[s] and L-mean[s] read
   R-S[s] and N[s] reads R-N[s], never R8; that L-shuf[s]'s foreign frames are W[s]'s own; that the
   runner's copied functions equal TASK-081's verbatim; the per-seed ladder against TASK-081's
   `decide_s` (equal rows on synthetic outcomes when the salt is the same) and every row in both
   directions; the combined rows of §7.3 in both directions; G-solver; no task truth in a
   candidate arm; the `"not evaluated"` sentinel; `install_guards`; `assert_local_import`.
   **Debug smokes** on 75900–75999 only: one S-style run of every arm, both seeds and W[66800], on
   four debug resets, with the determinism re-run, which set §12's scale and caps. **The power
   simulation** of §8 (salt 8502). The frozen-block candidate's sha256 is recorded.
2. **The freeze.** STATUS FROZEN with the frozen-block sha pin, the manifest's file pins and this
   document's sha256, merged on an independent reviewer's reported APPROVE. No value is measured
   between Stage 0 and the freeze; the freeze PR may only set the status and pins, apply review
   fixes that change no bar, and write Stage S's plan.
3. **Stage S, gated (on a reported GO; CPU only).** Cohort S, every arm of §4 once per reset,
   then each seed's determinism re-run of W[s] on S's first four resets (TASK-081's check: reading
   at 0.1 cm, commit target at 0.6 cm, success identical; a difference is a G-determinism guard
   error). One invocation; the rows of §7.
4. **Results PR.** Every arm per seed, privileged and hand-written arms labelled as not learned,
   each seed's restated numbers with their qualifiers, checked by an independent reviewer; R7 is
   then changed by its own reviewed ruling (§10).

### 6.2 No K0, corpus or Stage R (carried from TASK-081 §6.2–6.3)

τ_commit = 1.0 cm is carried; the condition, P-3, e9, the scorer and the simulator are unchanged,
and the ceiling is re-measured on every reset (S-VOID-CEILING). Nothing is fitted: R-S and R-N of
66801 and 66802 were fitted and gated (R0–R3) in TASK-080's Stage R.

### 6.3 No Stage D (R21.7)

TASK-081's Stage D was a development stop before S (W < 12/16, ceiling < 14/16 or W − best twin
< +3/16). It is dropped here: (a) its role was to stop a new controller form before the gated
cohort, and the controller form is unchanged and has passed TASK-081's S; (b) the question here is
exactly whether other seeds pass the gated test, so a development stop on 16 resets could only
withhold that measurement, and a seed's failure is a result to report, not a reason to stop; (c)
the debug smokes of Stage 0 run both seeds' W, N and twins end to end, which tests the mechanics a
D would otherwise exercise. The cost of one Stage S is about 1.5 h of CPU (§12).

## 7. Gates, bars and rows

Intervals are reset-clustered bootstrap percentile intervals, 10 000 resamples, 95 %, salt 8501,
with TASK-080's estimators (`lewm_pr_v2.paired_interval`, `median_ci`; the same functions with this
task's salt). Every bar is carried from TASK-081 §7.2 unchanged.

### 7.1 The per-seed bars (n = 128, for each s ∈ {66801, 66802})

- **G-bar:** W[s](S) ≥ **112/128** (carried; 0.875 × 128).
- **G-NI:** the lower bound of the paired 95 % interval of W[s] − C > **−16/128**, C the better of
  H-rule and H-sysid on S (carried; δ an allocation).
- **G-N, G-shuf, G-mean, G-rand:** W[s] > N[s], L-shuf[s], L-mean[s], L-rand, each by an exact
  one-sided McNemar test at p < 0.01 on the paired resets (carried).
- **"Detectably"** as TASK-081 §7.2 (R17.15): a twin test fails with W detectably no better when
  the upper bound of W[s] − twin < +7 resets (a count); W[s] is detectably inferior when the upper
  bound of W[s] − C < −16/128.

### 7.2 The per-seed ladder (TASK-081's, first match)

For each seed separately, with its own W, N, L-shuf and L-mean and the shared L-rand, C and
H-final(commit):

| row | condition |
|---|---|
| **V** | the void rule (§11) |
| **S-VOID-CEILING** | H-final(commit)(S) < 112/128 (shared: if it holds, it holds for both seeds) |
| **L-NO-GAIN** | for at least one twin, the McNemar test fails **and** W[s] is detectably no better |
| **L-INFERIOR** | W[s] is detectably inferior beyond δ (upper bound of W[s] − C < −16/128) |
| **L-PASS** | G-bar, G-NI and the four McNemar tests all pass |
| **L-TWIN-NEAR** | at least one McNemar test fails, and every failed one is a miss within noise |
| **L-NEAR** | G-NI fails, but W[s] is not detectably inferior beyond δ |
| **L-BAR** | otherwise (G-bar fails, with G-NI and the tests passing) |

The ladder is `lewm_cp_v2.decide_s`'s order with n = 128, G-bar 112, δ 16 and this task's salt.
Stage 0 tests that it returns the same row as TASK-081's `decide_s` on the same outcomes when the
salt is the same. W[66800] is read by no row.

### 7.3 The combined row (R21.4; first match)

| combined row | condition | meaning |
|---|---|---|
| **V** | the void rule (§11) | one repeat after a recorded fix |
| **REP-VOID-CEILING** | S-VOID-CEILING (H-final(commit)(S) < 112/128) | no reading; R7 unchanged |
| **REP-PASS** | both seeds L-PASS | **the replication claim**: TASK-081's L-PASS replicates across W's model seeds of this training run |
| **REP-ONE** | exactly one seed L-PASS | **not replicated**: the result depends on the model seed |
| **REP-NONE** | neither seed L-PASS | **not replicated** with either seed |

Rules declared now:
- **No pooling to rescue a seed.** The two seeds' counts are never summed or averaged into a test;
  a seed that misses its bar is not rescued by the other's margin, by W[66800]'s count on this
  cohort, or by TASK-081's earlier pass. A pooled or averaged figure may be printed as a
  description only, labelled as such, and no row reads it.
- **No joint test is added.** The replication claim is the conjunction of the two per-seed claims
  (§1), which needs no multiplicity adjustment. No "at least one seed" statement is a claim; under
  REP-ONE the seed that passed is reported as passing its own ladder, with the other seed's row
  beside it, never alone.
- **Each seed is reported separately**, with its own counts, intervals, rows and "LeWM needed"
  test. Within REP-ONE and REP-NONE the per-seed rows (L-NO-GAIN, L-INFERIOR, L-TWIN-NEAR, L-NEAR,
  L-BAR) are stated; a seed that is L-NO-GAIN or L-INFERIOR is described as **detectably** failing
  to replicate (a twin detectably no worse than W, or W detectably inferior to the comparator beyond
  δ), the other non-pass rows as **not replicated, within noise of the bar or margin**.
- The two seeds share the comparator, L-rand and the ceiling on the same resets, so their rows are
  not independent: an easy or hard cohort, or a strong comparator draw, moves both. This is stated
  beside every combined reading.

### 7.4 Reported in every row (not gates)

1. Every arm's count with its exact binomial 95 % interval; for each seed, every paired difference
   from W[s] with its discordant counts and interval; each seed's "LeWM needed" test.
2. **W[66800]** on S: its count, interval and paired differences from H-rule, C and each W[s]
   (reported only).
3. **Between seeds** (reported only): W[66801] − W[66802], with discordant counts, interval and a
   two-sided exact McNemar p; it gates nothing.
4. **Aim error against H-final(commit)'s committed aim** on the same reset for every candidate arm,
   H-rule and H-sysid (median with interval, 87.5th percentile, maximum); the fixed-point error and
   landing miss; each arm's predicted count through the pooled τ curve (a prediction, not a count).
5. **The final-distance distribution per arm** (TASK-081 §7.3 item 4), with the number of attempts
   ending 3.5–4.5 cm from the plate centre.
6. **affine_local's diagnostics** for each candidate arm (fallbacks by cause, clip-binding, points
   in the fit, J's eigenvalues, the logged residual).
7. Each seed's training flags (§3) beside its rows.

## 8. Power (R21.9; recomputed in Stage 0 with salt 8502)

**Planning rates.** C (H-rule) at **0.977** (TASK-080's and TASK-081's S pooled: 63/64 and 125/128,
188/192) and **0.984** (the stricter used by TASK-081); H-sysid at 0.953 (62/64 and 121/128 pooled,
183/192). W at the grid 0.906 (TASK-080's S rate for W-66800), **0.922** (TASK-081's S rate for
W-66800, 118/128), 0.938 and 0.953 (TASK-081's planning rate). Nothing measured on 66801 or 66802
in closed loop exists to plan from.

**A draft computation** (for this draft only; scratch code, salt 8599, 4 000 trials per cell, the
reset-bootstrap G-NI estimator through `lewm_cp_v2.ni_passes`; H-rule drawn at p_C; each W[s]
coupled to H-rule by the coupling, conditionally independent of the other W given H-rule; H-sysid
coupled to H-rule halfway; C the better of the two by count; L-PASS taken as G-bar and G-NI, the
twin tests' power being above 0.999 at TASK-081's twin rates). Per-seed L-PASS rate / both seeds
L-PASS (REP-PASS), overlap · half · independent:

| p_C (H-rule) | W = 0.906 | W = 0.922 | W = 0.938 | W = 0.953 |
|---|---|---|---|---|
| 0.977 | 0.56/0.32 · 0.45/0.21 · 0.41/0.19 | 0.81/0.67 · 0.71/0.51 · 0.65/0.44 | 0.96/0.93 · 0.91/0.83 · 0.86/0.74 | 1.00/1.00 · 0.99/0.98 · 0.97/0.94 |
| 0.984 | 0.44/0.19 · 0.36/0.13 · 0.34/0.13 | 0.72/0.52 · 0.64/0.41 · 0.59/0.35 | 0.92/0.85 · 0.87/0.76 · 0.82/0.67 | 0.99/0.98 · 0.98/0.95 · 0.95/0.90 |

**Stated plainly:** if the other seeds' true rate equals W-66800's observed 118/128 (0.922), each
seed passes with probability about 0.6–0.8 and **both pass with probability only about 0.35–0.67**.
So REP-ONE or REP-NONE is a likely outcome even if the three seeds were equally good, and a
non-pass at these rates is weak evidence against TASK-081's result; REP-PASS is likely (≥ 0.67)
only if the true rate is about 0.938 or above. The cohort size is kept at 128 because the task is
to replicate TASK-081's design exactly (its bar, margin and n); a larger cohort (256 resets would
raise the per-seed rate at W = 0.922 to about 0.95) would be a different test, and is not chosen.
The size at the margin and the L-INFERIOR false-fire rate are TASK-081's (2.6–3.6 % and 1.4–1.7 %
per seed, Stage 0 record §4); Stage 0 recomputes them with this comparator.

## 9. No abandonment clause (R21.5)

**No clause fires in TASK-083, on any row.** TASK-081's clause (fires on L-NO-GAIN or L-INFERIOR,
closing TASK-077's scope) belonged to the first gated test of the controller; that test ended
L-PASS. A replication asks whether that pass is robust across model seeds, not whether the line is
viable, so its failure narrows what may be said about the pass (§10) rather than closing a
direction. A seed's L-NO-GAIN or L-INFERIOR is recorded as a **detectable** failure to replicate
and is stated in R7. Nothing in TASK-077's scope (R17.10) is closed or reopened by this task.

## 10. What each combined row means for R7 (R21.6)

R7's v2 LeWM part is changed only by its own reviewed ruling after Stage S (R18.12's rule,
carried). Declared now, so that the ruling has no choice of wording to make after the result:

- **REP-PASS.** R7 adds, after its TASK-081 sentence, that TASK-083 replicated the L-PASS with the
  two other model seeds of the same training run (66801 and 66802, whose `last_two_triggered`
  flags are not set), each reaching L-PASS on its own on 128 further fresh gated resets under the
  same condition, with each seed's count, exact interval and paired difference from the comparator
  and its interval, and whether "LeWM needed" is shown for either. The qualifiers stay: simulation
  only; C1-M; LeWM choosing only the single place aim between P-3's learned pick and e9's scripted
  place; frozen DINOv2 features; one training corpus and recipe; non-inferiority within an
  allocated margin. If H-rule is measurably better for a seed (the interval's upper bound below 0),
  R7 says so for that seed. "One model seed" in R7's TASK-081 sentence then reads as TASK-081's
  run, with the replication beside it; it does not become "three runs".
- **REP-ONE.** R7 adds that a preregistered replication with the two other model seeds reached
  L-PASS with one (its seed and count) and not with the other (its seed, row and count), so the
  result depends on the model seed; TASK-081's L-PASS stands as recorded.
- **REP-NONE.** R7 adds that the replication with the two other model seeds did not reach L-PASS
  with either (their rows and counts), so the gated success is shown for one model seed only and
  did not replicate across seeds; TASK-081's L-PASS stands as recorded.
- In REP-ONE and REP-NONE, a seed whose row is L-NO-GAIN or L-INFERIOR is stated with the word
  "detectably" (§7.3); the others as "not replicated within noise of the bar or margin".
- **In every one of these rows, R7 also states W-66800's count on cohort S**, labelled reported
  only, whatever it is, so that the cohort's difficulty is not selectively reported.
- **REP-VOID-CEILING and V:** R7 is unchanged; the record states the void.
- Whatever the row, the owner ruling on TASK-079's "equivalent row" (R19.22) stays unruled by this
  task.

## 11. Void rule, guards and memory (carried from TASK-081 §10)

- **Void and repeat:** one repeat of Stage S after a committed, pushed and recorded fix, on the
  same seeds in a new output directory; a second V ends TASK-083 INCONCLUSIVE.
- **Guards:** TASK-081 §10's list (G-tests in-run, G-sentinel, G-hash, G-frozen, G-repro, G-threads,
  G-quiet, G-privileged, G-finite, G-memory at 12 GiB process-tree PSS, G-disk at 10 GiB free,
  G-solver), with **G-frozen and G-hash extended** to TASK-081's frozen block and six file pins,
  and **G-seed (new)**: each pool's configuration names exactly one model seed, its W and N
  checkpoints and its R-S and R-N by content sha256, every candidate record carries that seed, and
  the runner refuses a record whose seed or readout is not its pool's. **G-split here:** nothing is
  fitted; the runner refuses any closed-loop seed outside S's range. No GPU is used.
- **Caps** (provisional; set at Stage 0 at ≥ 1.5 × the scaled worst case of the debug smokes):
  Stage S **21 600 s**, per attempt **300 s**. G-memory 12 GiB process-tree PSS (one pool at a
  time; TASK-081's S peaked at 9.76 GiB).

## 12. Compute estimate (R21.12)

From TASK-081's Stage D and Stage 0 per-attempt medians (W 13.5 s, N 6.8, L-shuf 13.7, L-mean
13.6, L-rand 6.5, H-rule 4.5, H-sysid 4.3, H-final 8.2): about 48 s per reset per seed's four
candidate arms, 13.5 s for W[66800] and 23.5 s for the four shared arms, so about 133 s per reset
in all; 128 resets on 4 workers ≈ 4 260 s, plus the two determinism re-runs, three pool start-ups,
G-tests (about 165 s) and G-repro (about 42 s): **about 75–90 min of CPU**. No training, no
featurisation and no GPU. Disk: reports only (a few MB).

## 13. Risks, stated now

- **Power** (§8): at W-66800's observed rate a replication passes on both seeds with probability
  about one half; a REP-ONE or REP-NONE is likely even if nothing differs between seeds.
- **H-rule is strong** (125/128 on TASK-081's S) and given the simulator's plate law.
- **The scorer's margin is small** (TASK-081: most near-plate attempts rest 3.3–3.6 cm from the
  centre against a 4 cm radius), which adds noise to every count.
- **The other seeds read the plate slightly worse offline** (§3), and W-66801 has an unexplained
  zero-command ratio.
- **Everything here is one condition imposed by the simulator (C1-M)**, one camera, one encoder,
  one corpus; nothing transfers as such to Arena or the real G1.

## 14. Stage 0's open items (to be settled there, changing no bar)

1. The caps from the debug smokes' scale.
2. The power table recomputed with salt 8502 and 20 000 trials per cell, including the size at the
   margin and L-INFERIOR's false-fire rate for this comparator.
3. The pools' order (proposed: 66801, 66802, then W[66800]'s; one pool alive at a time) and
   each pool's peak memory against G-memory.
