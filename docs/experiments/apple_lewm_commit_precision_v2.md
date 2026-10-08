# Apple→Plate LeWM committed aim under C1-M, solved as the fixed point of a local affine fit to W's own predictions (TASK-081)

**STATUS: FROZEN** (R19.15, decided by Claude under owner delegation; no calibration stage, so
the freeze follows Stage 0 directly). The frozen block is `src/embodied_jepa/lewm_cp_v2.py`; its
sha256 **`77ccc6364624c7da9dcca1d09827a0f6a25a670a00392d488e7b0216e090705a`** is pinned in
`tests/test_lewm_cp_v2.py` and in the manifest (`benchmarks/manifests/apple-lewm-cp-v2.json`), with
the manifest's six file pins and this document's sha256. **The freeze takes effect when it is
merged on an independent reviewer's reported APPROVE.** Stage D and Stage S each still need their
own reported GO (§6.1). Stage D's plan is §6.4, Stage D's result §6.5 and Stage S's plan §6.6.

History. The DRAFT (R19.1–R19.11, #161) was reviewed independently. Stage 0
([record](apple_lewm_commit_precision_v2_stage0.md), R19.12–R19.14, #162) added the code, debug
smokes on 71900–71999 only, the caps and the power simulation; its candidate block was
`857a3078…32ef`. The freeze (R19.15) set STATUS FROZEN and applied #162's review: per-arm exact
intervals and every arm's paired difference from W are computed by the runner (§7.3 item 1), and
Stage S refuses a D-PASS report run under another frozen block. No seed of D or S had been
simulated at the freeze. Stage D, the non-gating development closed loop on 16 fresh resets, then
ended **D-PASS** (§6.5, R19.17): W 16/16, W-frozen 15/16 (reported only), N 5/16, L-shuf 5/16,
L-mean 8/16, H-rule 16/16 (reported only) and the privileged H-final(commit) 16/16; a development
result, not the gated one. No seed of S has been simulated. Stage S's plan is §6.6.

- **Admitted by:** the TASK-081 design note
  ([apple_lewm_commit_precision_v2_design.md](apple_lewm_commit_precision_v2_design.md), DRAFT,
  R18.33–R18.35, #160), whose §6 recommends this preregistration.
- **Rulings:** R19.1–R19.11 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (f). Each is
  **decided by Claude under owner delegation (2026-09-30)**. They build on R9.8 and R9.9 (the claim
  split), R15 and R16 (C1-M), R17 (TASK-077) and R18 (TASK-080, whose frozen protocol is the
  template here).
- **Task card:** `.mc/tasks/todo/TASK-081-lewm-commit-precision-under-c1m-w-s-aim-solver.md`.
- **Template:** TASK-080's frozen protocol,
  [apple_lewm_c1m_v2_pred_readout.md](apple_lewm_c1m_v2_pred_readout.md). Where this document
  says "carried", the TASK-080 section named is in force unchanged (and through it TASK-077's);
  only the differences are written out here.

The canonical status sentence (DECISIONS 2026-10-02, R7, as updated by R18.28 and R18.31) was not
changed by this draft. (Stage D's record corrects its last clause, R19.18, §6.5 and §13; the
paragraph below is kept as frozen.) It is quoted verbatim and in full in DECISIONS.md (R18.31). Its v2 LeWM part
says that TASK-080's W reached 58/64 on the gated cohort S, failed the preregistered
non-inferiority test against the non-learned rule controller H-rule (63/64), and that the row is
L-NEAR. §13 records one factual issue with its last clause.

---

## 0. What changes from TASK-080, stated first

TASK-080 ended **L-NEAR** at Stage S
([results](apple_lewm_c1m_v2_pred_readout_results.md), R18.30–R18.32): on 64 gated resets
W 58/64 passed G-bar (56/64) and the four twin tests, and failed G-NI against H-rule 63/64
(W − H-rule −5/64, reset-bootstrap 95 % interval [−10, 0]/64 against −δ = −8/64; not detectably
inferior). The design note diagnosed, post hoc, that W's capped single-evaluation refinement
oscillates on a predicted-plate map that is rough below about 1 cm and commits one extreme of the
oscillation at the cap. Its development check (seeds 70000–70063, CPU, one run, not gated) found
W:frozen 57/64, W:damped 58/64 and **W:affine_local 62/64** against H-rule 64/64 and the privileged
H-final(commit) 63/64; affine_local was the best of three executed variants, chosen after seeing
them, so its 62/64 is optimistic (design note §4.2, caveat 2).

TASK-081 keeps **everything** of TASK-080's closed loop except:

1. **W's solver after the grid** (§3). TASK-077 §5.1's refinement (at most 10 single roll-outs
   g ← clip(p̃(g)), stop at a move ≤ τ_commit/4, commit the last move unevaluated at the cap) is
   replaced by **affine_local**: the clipped fixed point of a least-squares affine fit
   p̃(g) ≈ c + J g to W's own grid predictions near the grid argmin. W-66800, R-S, the 147-candidate
   grid, the stand-in chunks, the start latent and the grid argmin are unchanged. It is a change
   to W's controller form after the grid, not a new model.
2. **The twins carry the same solver.** N, L-shuf and L-mean commit with affine_local, as they
   carried W's controller form in TASK-080. L-rand is unchanged (a uniform feasible candidate, no
   solver).
3. **A new arm, W-frozen, reported only**: TASK-080's W unchanged (its controller, R-S, the
   primary seed) on the same resets, so that W − W-frozen measures the solver's effect on fresh
   seeds (§7.3). It gates nothing.
4. **A larger gated cohort**: S has **128** resets; G-bar is **112/128** and the non-inferiority
   margin **16/128**, the same fractions as TASK-080's 56/64 and 8/64 (§7).
5. **Fresh seeds** in 70100–71999 (R18.34) and salts 8302–8304 (§5).
6. **No Stage K0, C′ or R.** τ_commit = 1.0 cm is carried from TASK-080's pooled K ∪ K′ (§6.2);
   no corpus is collected and nothing is fitted (§6.3).

LeWM still makes **one decision**: the place aim committed at step 405. P-3 (behaviour cloning,
not a world model) picks before 405; e9's scripted place primitive executes the committed aim
after 405, with no re-aim. The claim, if any, is **"LeWM-driven aim selection at 405"** under C1-M,
not "a LeWM policy".

## 1. The question and the two claims (R9.8, R9.9; carried from TASK-080 §1)

**Question.** Under C1-M, with W-66800 and R-S unchanged and only W's solver after the grid
changed to affine_local, does W reach R9.8's primary claim on 128 fresh gated resets?

**Primary claim, "LeWM-driven closed-loop success"** (R9.8; all of the following, on cohort S):
1. W's predictor makes the decision from the encoded current frame, with no privileged read at run
   time (`task_truth_in_controller` = 0, as TASK-080).
2. **The bar:** W(S) ≥ 112/128 (G-bar, §7.2).
3. **Non-inferiority:** W is non-inferior to the better of H-rule and H-sysid on S within
   δ = 16/128 (G-NI). δ is **an allocation, not calibrated** (R15.6's label, R17.8, R18.35): the
   fraction 8/64 = 12.5 points is carried unchanged.
4. **W beats the action-blind twin N, both scene-blind twins (L-shuf, L-mean) and a random choice
   (L-rand)**, each by an exact one-sided McNemar test at p < 0.01.

Items 2 and 3 must both hold. Item 4 shows that the prediction is used.

**Secondary claim, "LeWM needed"**: W detectably better than the better of H-rule and H-sysid
(exact one-sided McNemar p < 0.01). **Reported only, never a gate, and not expected** (H-rule
scored 63/64 on TASK-080's S and 64/64 on the design note's development seeds). A primary pass
never supports a "LeWM needed" statement.

**What a pass would show:** TASK-080 §1's sentence with one addition: "… its predicted plate is
read by a ridge fitted on its own predictions under the kinematic stand-in's commands, **and the
committed aim is the clipped fixed point of a least-squares affine fit to its own predictions over
up to 25 grid candidates near the grid argmin**". That would be the first gated LeWM-driven
closed-loop success on v2. It would change R7's sentence only through its own reviewed ruling
(§13).

**What it would not show:** TASK-080 §1's list unchanged (that LeWM is needed; a LeWM policy;
anything about v1's 0/150, Arena or the real G1; that pretraining matters; anything about other
conditions, grids or horizons), plus: that the solver is the cause of any difference from
TASK-080's S (the two cohorts are different resets; W − W-frozen on S is the paired reading, §7.3,
and it is reported only).

## 2. What is carried, with its source

### 2.1 The condition and the code (TASK-080 §2.1, §6)

C1-M exactly as TASK-080 §2.1 records it: v2's `wide_reset_values`; P-3 (`7988162d…60be8`) with
G-repro's post-look estimates; the move at the observation of step 300, ρ\* = 4 cm (disc),
off-table re-draw and blocked move as R16.3, **drawn per seed with TASK-080's salt 8201**
(`lewm_pr_v2.move_offset`, unchanged; the seeds are fresh, so the draws are); cell A's
`CellMotion` (κ = −0.5, L = 2, s0 = 405, s1 = 525); one aim committed at c = 405 and executed by
e9's place primitive with no re-aim; the 147-candidate box g = p̂ + a·(h − p̂) + b·n,
a ∈ [−0.5, 0.5] in steps of 0.05, b ∈ [−3, +3] cm in steps of 1 cm; r = 465, h = 60. The scorer
is TASK-080's (counted success: the apple at rest on the plate under `apple_at_rest_v0` after a
latched grasp and a latched place, T71-R1/R2).

Code: TASK-080's modules (`lewm_pr_v2.py`, `_runtime`, `_offline`, `scripts/run_lewm_pr_v2.py`),
TASK-077's (`lewm_c1m_v2*`) and the design note's solver functions
(`commit_precision_dev.affine_fixed_point` and `local_mask`) are **imported, not edited**; their
pins must still match (G-frozen, §10). The Stage-0 PR adds new modules only (provisional names:
`lewm_cp_v2.py` with this task's frozen block, `lewm_cp_v2_runtime.py`, `scripts/run_lewm_cp_v2.py`,
`tests/test_lewm_cp_v2.py`, `benchmarks/manifests/apple-lewm-cp-v2.json`).

### 2.2 Artifacts reused, each checked by sha256 before use (G-hash)

| artifact | source | sha256 (as recorded) |
|---|---|---|
| W-66800 and N-66800 (the primary seed; the other four job reports are checked as TASK-080 checked them) | TASK-077 Stage T | W `891d2664…76b8`, N `51b51035…d2c2` |
| R-S and R-N of seed 66800 | TASK-080 Stage R (§8.4), report `bedb8966…48ea7` | content R-S `6e05d223…1046`, R-N `15405f3f…5271` |
| R-plate (the 405 reading that builds every grid) | TASK-077 Stage O | content `08bde901…9eb9` |
| L-mean's mean 405 latent | TASK-077 Stage O | content `85da6336…81bd` |
| H-sysid | TASK-077 Stage O | `sysid.json` file `671841cd…121fc` |
| R8 (**only** for the privileged H-read) | TASK-077 Stage O | content `62a5ea8a…3b4a` |
| τ_commit and the pooled τ curve | TASK-080 K0′ (§8.1), report `a0939e3e…4c16` | τ_commit 1.0 cm; 64, 58, 60, 32, 29, 7 of 64 at 0, 0.5, 1, 1.5, 2, 3 cm |
| the solver's code | the design note (#160), `src/embodied_jepa/commit_precision_dev.py` | file `ba3c8d03…91f1` |
| TASK-080's frozen block and its seven file pins | `benchmarks/manifests/apple-lewm-pr-v2.json` | frozen `0fc095dc…be064` |

The artifacts stay where TASK-077 and TASK-080 left them (the `task077-stageo`, `task077-staget`,
`task077-staget2` and `task080-stager` worktrees, with evidence copies under
`~/develop/emai/evidence/`). They are read only. TASK-080's fresh corpus is **not** read (§6.3).

**The primary seed stays 66800**, flagged `last_two_triggered` (W-66800 selected one of its last two
checkpoints, and every W curve was lowest at its last point; TASK-077 results §3, caveats 1–2). The
flag is stated beside every seed-66800 comparison, as in TASK-080 (design note §7, question 4).

## 3. The solver: affine_local (R19.2)

### 3.1 Definition

For an arm X ∈ {W, N, L-shuf, L-mean}, at step 405, with X's start latent and readout as in
TASK-080 §6 (W and the scene-blind twins read R-S, N reads R-N):

1. **The grid (unchanged).** Build the 147 candidates from p̂ and h, drop the infeasible ones
   (stand-in chunk infeasible), roll every feasible candidate's stand-in chunk out with X from the
   start latent to r = 465 in one batch, and read the predicted plate p̃(g). If every candidate is
   infeasible, X aims at p̂ (a counted attempt), as TASK-080.
2. **The grid argmin (unchanged).** The feasible candidate with the lowest |p̃(g) − g| (ties to the
   lowest row-major index); call it g₀ = (a₀, b₀) in box coordinates.
3. **The neighbourhood.** The feasible candidates with |a − a₀| ≤ 2 a-steps (0.10) and
   |b − b₀| ≤ 2 cm: up to 25 points, fewer at the box edge or where candidates are infeasible.
4. **The fit.** Ordinary least squares p̃(g) ≈ c + J g over those points (a 2 × 2 J and a 2-vector c,
   from the targets g in metres and their already computed predictions; no new roll-out).
5. **The solve.** g_fp = (I − J)⁻¹ c, then clipped to the box around p̂ and h exactly as TASK-077's
   refinement clips (`lewm_next_c1.clip_to_box` with a ∈ [−0.5, 0.5]).
6. **The fallbacks**, each to g₀ and each logged: fewer than 4 points in the neighbourhood;
   |det(I − J)| < 10⁻⁶; the clipped g_fp's stand-in chunk is infeasible.
7. **The commit.** The clipped g_fp (or g₀ on a fallback) is committed at 405. One further
   roll-out of the committed aim's chunk is made **for the log only** (its residual
   |clip(p̃(g)) − g|); it never changes the aim.

There is no refinement loop, no tolerance and no cap. τ_commit/4 is used only by W-frozen. Step 3
is `commit_precision_dev.local_mask`, and steps 4–5 with the first two fallbacks of step 6 are
`commit_precision_dev.affine_fixed_point`, both at their pinned sha256 (§2.2); the infeasible-chunk
fallback and the logged roll-out follow the affine_local branch of the design note's
`all_variants`. That is the code the design note's development check executed; the Stage-0
runtime wraps it, and its tests check equality with `all_variants(...)["affine_local"]` on
synthetic predictors, including every fallback. Stage 0 also logs the fit's design rank (a rank
below 3, a collinear neighbourhood, makes `lstsq` return its minimum-norm fit; reported only, the
solver is unchanged; #161 review, nit N8).

**For N** the predicted plate is the same for every candidate (zero commands), so the fit gives
J = 0 up to rounding and g_fp = clip(p̃): the same aim TASK-080's refinement reached for N after
one step. N's solver therefore changes nothing material; it is declared, not assumed.

### 3.2 Why this variant, and why no extra refinement (design note §7, question 1)

- **Frozen before fresh data.** The variant, its neighbourhood (2 a-steps × 2 cm, up to 25 points)
  and its fallback were fixed before the design note's development run (commit `3a24abb`), and the
  development check is the only evidence about it. Adding damped refinements after the affine
  solution would be a fourth, untested variant chosen after the development numbers. **Decided:
  no extra refinement.**
- **What the development check showed** (design note §4.2; development, one run, selection
  caveat): median aim error against H-final(commit)'s aim on the same reset 0.243 cm
  [0.205, 0.322] for affine_local against 0.382 [0.322, 0.473] for the frozen refinement; better
  on both the frozen-capped resets (0.284 against 0.474) and the converged ones (0.235 against
  0.330); no fallback to the argmin and no aim on the box edge in 64 resets; closed loop 62/64
  against 57/64, paired 7 won and 2 lost (exact one-sided McNemar p = 0.090).
- **Not chosen:** `damped` (58/64; stalls at the noise floor), `cap30` and `bestres` (help the tail
  only), `affine_global` (0.412 cm: curvature over the 6 cm box), Polyak averaging, Newton or
  secant steps, several stand-in chunks per aim, refitting R-S or retraining W (design note §3).

## 4. Arms (carried from TASK-080 §6, with the solver of §3)

| arm | what it is | readout | solver |
|---|---|---|---|
| **W** | the primary seed's W from the encoded 405 frame | **R-S** | **affine_local** |
| **W-frozen** (reported only) | TASK-080's W unchanged: TASK-077 §5.1's refinement, cap 10, stop at τ_commit/4 = 0.25 cm | R-S | frozen |
| **N** (action-blind) | the primary seed's N, zero commands | **R-N** | affine_local |
| **L-shuf** (scene-blind) | W from the logged 405 frame of **W's** attempt on the next reset in cohort order that reached 405, with this reset's candidate chunks (R17.20) | R-S | affine_local |
| **L-mean** (scene-blind) | W from L-mean's mean 405 latent | R-S | affine_local |
| **L-rand** | a feasible grid candidate drawn uniformly (salt **8303**) | none | none |
| **H-rule** | `RuleCommit`: the rule's fixed point on p̂ and h, single commit, clipped (knows the rule; not learned) | none | – |
| **H-sysid** | `SysidAim` with TASK-077 Stage O's fit, inverted with TASK-077's controller form, clipped (not learned) | none | – |
| **H-final(commit)** | the ceiling: `LookaheadAim` once at 405 in cloned state, unclipped; **privileged, not learned** | — | – |
| **H-read**, **H-now** | reported only, as TASK-080 §6; **privileged** | R8 (H-read) | – |

Infeasible candidates are dropped for every arm alike. Each arm's clip-binding fraction is reported.
H-sysid keeps TASK-080's solver (`lewm_next_c1.choose_aim`, its own grid and refinement on the
fitted law); only W and the three twins change solver (#161 review, nit N9).
The non-inferiority comparator C is the better of H-rule and H-sysid on S (a tie goes to H-rule).
W-frozen is **not** a comparator and not a twin: no row reads it.

## 5. Seeds, cohorts and salts (R19.6)

**Block 70000–71999** (R18.34). 70000–70099 is the design note's development range (70000–70063
and the smoke seeds 70090–70091 were simulated there) and is **not** used here. Every range below
lies in 70100–71999, outside every range of `lewm_pr_v2.FORBIDDEN_RANGES` and outside TASK-080's
block 65000–65999; Stage 0's test re-checks this in code.

| seeds | use |
|---|---|
| 70100–70115 | **D**: the development closed loop (16 resets) |
| 70200–70327 | **S**: the gated cohort (128 resets) |
| 71900–71999 | **debug**: runner mechanics only; nothing in it is read |
| 70000–70099 | the design note's development range; not used |
| everything else in 70100–71999 | reserved; any use needs its own ruling |

| salt | use |
|---|---|
| 8301 | used by the design note's development summary and power table; **not reused** |
| 8302 | every bootstrap of this task (10 000 resamples, reset-clustered): G-NI, every paired interval, every median interval |
| 8303 | L-rand's draw per reset |
| 8304 | Stage 0's power simulations |
| 8305–8312 | reserved; any use needs its own ruling |
| 8201 (TASK-080's, carried) | the move draw per reset, `default_rng(SeedSequence([8201, seed, k]))` |

**The search** (2026-10-08, at `4710b21`). R18.34's search covered the block (no integer
70000–71999 used as a seed on any of 112 local and remote refs or in any worktree's `src` and
`scripts`) and the salts 8301–8312 (not found on any ref in `src`, `scripts`, `tests` or
`configs`). Re-checked for this draft: `git grep -w` for 70100, 70115, 70200, 70327, 71900, 71999
and 8302–8305 in `src`, `scripts`, `tests`, `configs`, `benchmarks`, `docs` and `.mc` finds no
seed use; the hits are block declarations only (the task card's and R18.34's "70100–71999", the
design note, and `scripts/dev_commit_precision.py`'s `DEV_BLOCK = (70000, 71999)`; #161 review,
nit N3). No R19 label existed on any local or remote ref.

## 6. Stages and the staged GO flow

Nothing from D or S is simulated before the reported GO of its stage. Every stage runs from a clean
worktree of the merged revision (`scripts/new_worktree.sh <dir> --run --from <rev>`), after G-tests.

### 6.1 The sequence

1. **Stage 0 (after this draft's review).** New modules only (§2.1), the DRAFT manifest, the
   frozen-block candidate and tests: the seed ranges and salts against every forbidden range,
   TASK-080's block and the design note's development range; the reused artifacts' sha256 checks
   (including `commit_precision_dev.py`'s pin and TASK-080's frozen block and file pins); that W,
   L-shuf and L-mean read R-S and N reads R-N, never R8; that the affine_local runtime equals the
   design note's `all_variants(...)["affine_local"]` on synthetic predictors (every fallback
   included) and that W-frozen equals TASK-080's `choose`; the twins' solver; L-rand's salt 8303;
   no task truth in any candidate arm; the `"not evaluated"` sentinel; every row of §7 in both
   directions; `install_guards` and `assert_local_import`. **Debug smokes** on 71900–71999 only:
   one D-style and one S-style run of every arm on a few debug resets, which set the scale for
   §11's caps. **The power simulation** (salt 8304) of §8 with the better-of-two comparator. The
   frozen-block candidate's sha256 is recorded.
2. **The freeze.** STATUS FROZEN with the frozen-block sha pin, the manifest's file pins and this
   document's sha256, merged on an independent reviewer's reported APPROVE. No value is measured
   between Stage 0 and the freeze; the freeze PR may only (a) set the status and pins, (b) apply
   review fixes that change no bar, and (c) write Stage D's plan.
3. **Stage D, the development closed loop (on a GO; CPU).** W, W-frozen, N, L-shuf, L-mean,
   H-final(commit) and H-rule on D's 16 resets (W-frozen and H-rule reported only).
   **L-DEV-STOP** (escalate, no clause) if W < 12/16, or H-final(commit) < 14/16, or
   W − max(N, L-shuf, L-mean) < +3/16 (carried from TASK-080 §8 step 6); otherwise **D-PASS**.
   Nothing is fitted or tuned on D.
4. **Stage S, gated (on a reported GO after D-PASS; CPU).** Cohort S, every arm of §4 once per
   reset, paired, with TASK-077's determinism re-run of **W** (affine_local) on S's first four
   resets (reading gated at 0.1 cm, commit target at 0.6 cm, success identical; R17.21, R17.27,
   R17.38). Rows: §7.2.
5. **Results PR.** Every arm is reported, privileged and hand-written arms labelled as not learned,
   and an independent reviewer checks every restated number.

### 6.2 No new K0: τ_commit is carried (R19.4)

τ_commit = **1.0 cm** is TASK-080's pooled K ∪ K′ value (64 resets: 64, 58, 60, 32, 29 and 7 of 64
at 0, 0.5, 1, 1.5, 2 and 3 cm of planted error; TASK-080 §8.1, R18.22). It is carried, not
re-measured, because:
- **Nothing it measures has changed.** K0 measures the tolerance of e9's place primitive and the
  scorer to a planted aim error under C1-M, and the ceiling. The condition, P-3, e9, the scorer and
  the simulator are unchanged; only W's solver changes, and K0 does not involve W.
- **Little in TASK-081 depends on it.** G-bar's fraction 0.875 is τ_commit's bar fraction (28/32,
  TASK-076 G1's form), which a re-measured curve would not change. τ_commit/4 is W-frozen's stop
  rule (unchanged by construction). The τ-curve mapping of aim errors to predicted counts is
  reported only (§7.3). affine_local reads no tolerance.
- **The ceiling is re-measured anyway.** H-final(commit) runs on every D and S reset;
  S-VOID-CEILING (H-final(commit)(S) < 112/128) and D's ceiling stop (< 14/16) are the checks a new
  K0 would add.
- A new K0 would spend fresh seeds and a stage for no gate.

### 6.3 No offline gate (Stage R) and no corpus (R19.4; design note §7, question 5)

**Decided: no Stage R.** Reasons:
- **Nothing is fitted.** TASK-080's Stage R existed to test a new readout (R-S) on fresh roots
  before any closed loop. R-S, R-N, W and N are unchanged, and R0–R3 of their R-PASS do not
  depend on the solver. A1 and A2 were computed with the frozen solver and are **not re-tested**
  for affine_local (#161 review, nit N5); the closed loop measures the solver directly.
- **An offline check on fresh roots is not cheap enough to be worth its proxy.** TASK-080's fresh
  corpus has been read; a fresh one needs a new privileged collection (a Stage C with its own rows)
  plus every arm's offline aims, about an hour of CPU and a stage of its own. Its output would be a
  predicted count through the τ curve, which measures aims against the rule's idealised fixed point
  g\* (that every arm misses by about 0.48 cm, design note §2.2 point 4) and which has
  under-predicted the closed loop twice by about one reset (TASK-080: A1 56.92 against W's 58/64;
  design note: 60.72 against 62/64).
- **The closed loop is the direct and cheap measurement.** Stage D (16 resets, about 10 min) is a
  development stop before S, and S itself takes about an hour (§11). An offline gate would add a
  false-stop channel without protecting anything G-bar, G-NI and the twin tests do not.
- **The paired W-frozen arm** replaces the offline before-and-after comparison: it measures the
  solver's effect on the same fresh resets in closed loop (§7.3).

### 6.4 Stage D's plan (R19.16)

Stage D runs once, on its own reported GO at the freeze's merge commit, from a fresh clean worktree
of that commit (`scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task081-staged --run
--from <merge sha>`). It is one invocation, from the worktree root, on the CPU (no GPU lock; the
stage creates no CUDA context):

- **`closed --cohort D`**: the 16 development resets **70100–70115**, arms **W, W-frozen, N,
  L-shuf, L-mean, H-final(commit) and H-rule** (W-frozen and H-rule reported only), W and its
  twins reading R-S and N reading R-N of the primary seed 66800 from TASK-080's Stage R fits (the
  report checked at `bedb8966…48ea7`, R-S and R-N at their content sha256s), with TASK-077's six
  job reports, featurisation and Stage O fits and TASK-076's evidence root (G-repro). The runner
  runs its own G-quiet and in-run G-tests first. Nothing is fitted or tuned on D.
- **Rows (§7.1):** **L-DEV-STOP** (escalate, no clause) if W < 12/16, H-final(commit) < 14/16 or
  W − max(N, L-shuf, L-mean) < +3/16; otherwise **D-PASS**, after which Stage S may get its own GO.
- **Caps and resources (§10):** 7 200 s for the stage (scaled about 510 s) and 300 s per attempt;
  G-memory 12 GiB process-tree PSS (debug peak 9.49 GiB); MemAvailable ≥ 16 GiB at the start;
  G-quiet load ≤ 2.0 / 2.0; G-disk ≥ 10 GiB free.
- **What D is and is not.** A development cohort of 16 resets that gates only Stage S's GO; not a
  gated claim. Its record makes R7's factual correction (§13, R19.10, R19.14) as its own ruling,
  adding D's counts as development results.

If the invocation ends in anything other than a §7.1 row, nothing further is launched and the case
is ruled under §10 first. The exact command is in the GO.

### 6.5 Stage D's result: D-PASS (R19.17)

Stage D ran once, on the reviewer's reported GO (#163,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/163#issuecomment-6054000652>), at
`2a6b633`, the freeze's merge commit, on a clean tracked tree with STATUS FROZEN and the frozen
sha `77ccc636…705a`, in the worktree `/home/huhn/develop/emai/worktrees/task081-staged`, with the
command the GO named, once. **D is a non-gating development cohort of 16 fresh resets
(70100–70115), one run, one model seed (W-66800, flagged `last_two_triggered`), in simulation
only. It gates only the GO of Stage S and is not the gated result.** In every arm, P-3's learned
pick and e9's scripted place are the same; the arms differ only in the single place aim committed
at step 405, under the declared simulation-only condition C1-M.

| | UTC (2026-10-08) | seconds |
|---|---|---:|
| start; G-quiet passed (load 0.13 / 0.18, bar 2.0 / 2.0) | 06:38:15 | – |
| in-run G-tests: 2123 passed, 37 skipped, at `2a6b633`, clean | 06:38:15–06:41:00 | 164 |
| G-repro (8 of 8 checks; 170 train roots decoded, no test split) | – | 42 |
| `first_outcome_utc` | 06:41:45 | – |
| arms, in order (log lines, converted from local time): W, W-frozen, N, L-shuf, L-mean, H-final(commit), H-rule done at | 06:42:41, 06:43:38, 06:44:06, 06:45:00, 06:45:55, 06:46:28, 06:46:46 | – |
| end, report written: **D-PASS** | 06:46:47 | **512** (cap 7 200; scaled estimate about 510) |

- **Guards.** G-frozen: the frozen block at run time was `77ccc636…705a`, equal to the pin.
  G-hash checked this task's 6 own pins with the protocol document's pin `5d91a177…e7f5`,
  TASK-080's 7 pins and its protocol document (`bb1b2772…`), TASK-077's 13 pins and TASK-076's 84
  pins; the solver file `commit_precision_dev.py` matched `ba3c8d03…91f1`; the six checkpoints
  matched their sha256s; the Stage R report the GO named (`bedb8966…48ea7`) was accepted, and R-S
  and R-N of seed 66800 loaded against their content sha256s. The revision at the end was still
  `2a6b633` and the tracked tree stayed clean. No non-finite field. G-memory: peak process-tree PSS
  9.41 GiB (RSS 10.88 GiB, 9 processes; cap 12), MemAvailable 24.7 GiB at the start. G-disk: at
  least 62.5 GiB free throughout. Threads as pinned (MKL and OMP 6, OpenBLAS 16), 4 world-model
  workers; the pool closed cleanly (4 joined, none killed). No render disagreement between arms
  and no render retry. Every attempt ran its 725 steps to `policy_complete`; the post-pick plate
  move was applied on every attempt; no contact with the plate before s1 was recorded on any
  attempt; no attempt was refused before 405. `task_truth_in_controller` was 0 on every attempt
  and every attempt passed the privileged-read check (G-privileged). G-solver passed: every W, N,
  L-shuf and L-mean decision logged affine_local, and every W-frozen decision TASK-080's solver
  (`frozen`). No determinism
  re-run is part of D (`"not evaluated"`).
- **One G-repro re-render disagreement, resolved by its majority rule.** In G-repro's re-render
  of the carried TASK-072 run-1 seeds, seed 51344's three renders gave two identical frames and a
  third differing by one level in 4 pixels, with equal simulator states; the majority frame was
  used and all 8 reproduction checks passed. This is the carried `render_majority` mitigation, the
  same case TASK-080's Stage S recorded for seed 51533; it is not a V and it touches no cohort-D
  frame (`render_disagreements` for D is empty).

**Counts** (counted success: the apple at rest on the plate under `apple_at_rest_v0` after a
latched grasp and a latched place, T71-R1/R2). Exact binomial 95 % intervals on 16 resets are
wide. The paired difference's interval is the reset-bootstrap percentile interval (salt 8302):

| arm | count | 95 % (exact) | W − arm [95 %] | resets W only / arm only |
|---|---:|---|---|---|
| **W** (affine_local, R-S, seed 66800) | **16/16** | 0.794–1.000 | – | – |
| W-frozen (TASK-080's W unchanged; **reported only**) | 15/16 | 0.698–0.998 | +1 [0, +3] | 1 / 0 |
| N (action-blind; R-N) | 5/16 | 0.110–0.587 | +11 [+7, +14] | 11 / 0 |
| L-shuf (scene-blind; R-S) | 5/16 | 0.110–0.587 | +11 [+7, +14] | 11 / 0 |
| L-mean (scene-blind; R-S) | 8/16 | 0.247–0.753 | +8 [+4, +12] | 8 / 0 |
| H-final(commit) (**privileged** look-ahead ceiling, not learned) | 16/16 | 0.794–1.000 | 0 [0, 0] | 0 / 0 |
| H-rule (non-learned, given the plate law; **reported only** in D) | 16/16 | 0.794–1.000 | 0 [0, 0] | 0 / 0 |

**Rows (§6.1 step 3, §7.1).** W 16 ≥ 12; H-final(commit) 16 ≥ 14; W − max(N, L-shuf, L-mean) =
16 − 8 = **+8 ≥ +3**. No stop holds: **D-PASS**, `clause_fires` false. Every twin's success is
also a W success on the same reset. D runs no McNemar test; the discordant counts are reported,
not tested, and D's bars are not S's.

**The solver's effect (§7.3 item 2, reported only).** W − W-frozen = +1/16 (discordant 1 / 0;
interval [0, +3]; exact one-sided McNemar p = 0.5). W-frozen's refinement did not converge (was
capped) on 9 of 16 resets (70100, 70102, 70105, 70106, 70110, 70111, 70112, 70114, 70115); on
those, W scored 9/9 and W-frozen 8/9 (its one failure, 70111, ended with a latched place but 4.5 cm from
the centre, outside the scorer's 4 cm radius). On 16 resets this neither confirms nor refutes the
design note's development hypothesis; it gates nothing.

**The committed aims** (§7.3 item 3). Aim error is the commit target's distance from
H-final(commit)'s committed aim on the same reset; the fixed-point error is the distance from the
rule's fixed point g\*; the landing miss is the distance from the plate at s1 = 525. The τ-curve
count is a prediction through TASK-080's pooled τ curve, not a count:

| arm | aim error median [95 %] | p87.5 | max | fixed-point error median / max | landing miss median / p87.5 / max | τ-curve prediction | clip-binding | s per attempt (median / max) |
|---|---|---:|---:|---|---|---:|---:|---|
| W | 0.262 [0.154, 0.324] | 0.519 | 0.688 | 0.577 / 0.795 | 0.353 / 0.735 / 1.076 | 14.77 | 0 | 13.50 / 15.53 |
| W-frozen | 0.321 [0.186, 0.439] | 0.745 | 1.742 | 0.537 / 1.232 | 0.446 / 1.203 / 2.552 | 14.58 | 0 | 14.10 / 14.43 |
| N | 2.190 [1.574, 2.528] | 2.577 | 3.369 | 2.356 / 3.670 | 3.319 / 3.804 / 5.150 | 6.46 | 0.0625 | 6.75 / 7.01 |
| L-shuf | 2.763 [1.780, 3.630] | 4.077 | 5.206 | 2.705 / 4.773 | 4.281 / 6.011 / 7.788 | 5.51 | 0.1875 | 13.66 / 13.87 |
| L-mean | 1.718 [1.471, 2.259] | 3.002 | 3.441 | 1.957 / 3.774 | 2.541 / 4.518 / 5.310 | 6.88 | 0 | 13.61 / 13.78 |
| H-final(commit) | 0 (its own aim) | – | – | 0.477 / 0.513 | 0.103 / 0.119 / 0.128 | 14.60 | 0 | 8.23 / 8.43 |
| H-rule | 0.098 [0.075, 0.164] | 0.185 | 0.203 | 0.473 / 0.577 | 0.175 / 0.254 / 0.296 | 14.70 | 0.25 | 4.46 / 4.53 |

- **affine_local's diagnostics** (§7.3 item 5). W: no fallback, no clipped aim, a median of 25 points
  in the fit (all 147 grid candidates were feasible on every reset in W, N, L-shuf and L-mean), median
  residual at the committed aim 0.227 cm (max 1.334), real parts of J's eigenvalues with medians
  −0.535 and −0.323, two resets with a complex pair; W's decision took 9.3 s in median (148
  roll-outs). N, being action-blind, predicts the same plate for every aim, so its fit's J is 0
  and its fixed point is the constant prediction (residual 0), as §3.1 expected. L-shuf clipped 3
  aims, N 1, L-mean none; no arm fell back.
- **The final distance** (§7.3 item 4). The apple's median final distance from the plate centre
  over all 16 attempts was 3.38 cm for W, 3.48 for W-frozen, 3.34 for H-rule and 3.42 for
  H-final(commit) (maxima 3.87, 4.52, 3.84 and 3.97); 6–7 of 16 attempts of each of W, W-frozen, H-rule and H-final(commit) ended
  between 3.5 and 4.5 cm, within 0.5 cm of the scorer's 4 cm radius. So counts near the boundary
  depend on more than the aim, as §12 states. Three L-shuf attempts ended 0.25–1.48 m from the
  plate (70111, 70105, 70101); L-shuf on 70104 ended at rest 2.4 cm from the centre without a
  latched place, so it is not counted.

**Per reset** (S = counted success, – = not; the number is the aim error against H-final(commit),
in cm):

| reset | W | W-frozen | N | L-shuf | L-mean | H-final(commit) | H-rule |
|---|---|---|---|---|---|---|---|
| 70100 | S 0.08 | S 0.31 | – 1.79 | S 1.52 | – 1.49 | S | S 0.01 |
| 70101 | S 0.07 | S 0.08 | S 1.02 | – 3.16 | – 1.29 | S | S 0.16 |
| 70102 | S 0.26 | S 0.47 | S 1.94 | – 4.14 | S 3.00 | S | S 0.09 |
| 70103 | S 0.15 | S 0.37 | – 3.37 | – 1.78 | – 1.57 | S | S 0.20 |
| 70104 | S 0.39 | S 0.33 | – 2.56 | – 3.63 | – 1.85 | S | S 0.17 |
| 70105 | S 0.10 | S 0.16 | S 0.63 | – 3.84 | S 2.19 | S | S 0.07 |
| 70106 | S 0.26 | S 0.35 | S 0.68 | – 4.07 | – 3.03 | S | S 0.07 |
| 70107 | S 0.23 | S 0.04 | – 2.53 | S 2.76 | – 2.19 | S | S 0.05 |
| 70108 | S 0.65 | S 0.28 | – 2.31 | S 0.15 | S 1.50 | S | S 0.10 |
| 70109 | S 0.10 | S 0.20 | – 2.16 | – 1.89 | S 1.27 | S | S 0.14 |
| 70110 | S 0.50 | S 0.44 | – 2.13 | S 3.42 | – 2.05 | S | S 0.13 |
| 70111 | S 0.32 | – 1.74 | – 2.29 | – 5.21 | S 2.47 | S | S 0.06 |
| 70112 | S 0.69 | S 1.02 | S 0.66 | – 2.76 | – 3.44 | S | S 0.19 |
| 70113 | S 0.26 | S 0.10 | – 2.57 | – 1.25 | S 1.59 | S | S 0.10 |
| 70114 | S 0.23 | S 0.19 | – 2.22 | S 1.52 | S 0.91 | S | S 0.09 |
| 70115 | S 0.32 | S 0.71 | – 2.64 | – 2.19 | S 1.37 | S | S 0.18 |

**What D does and does not show.**
- It does not test G-bar, G-NI or any McNemar test: those are Stage S's, on 128 fresh resets.
  16/16 on 16 resets is compatible with any true rate from about 0.79 upward (the exact interval),
  so D neither confirms nor refutes the planning rate 0.95 (§8).
- H-rule (16/16) and the privileged ceiling (16/16) tie W here; D does not separate W from either.
  Whether W is non-inferior to the better of H-rule and H-sysid within 16/128 is S's G-NI; H-sysid
  and L-rand did not run in D.
- Everything is one model seed (66800, flagged `last_two_triggered`: W-66800 selected one of its
  last two checkpoints), one camera, one encoder, one condition imposed by the simulator (C1-M), in
  MuJoCo only.

**Beside R7** (§13). R7's clause "its only other closed-loop Apple→Plate runs whose counts are read
are on v1, with 0 successes" was no longer true after the design note's development check, and D
adds further read v2 counts. **R19.18** ([DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (i))
replaces it with a factual statement of TASK-081's development counts (the design note's check and
D), labelled as non-gating development runs; the statement that LeWM has no gated closed-loop
Apple→Plate success is kept.

**Row: D-PASS.** The clause does not fire, nothing is closed. **Stage S may get its GO** (§6.1
step 4, §6.6).

**Evidence.**
- Report (git-ignored, in the worktree): `outputs/task081-d-1/report.json`, sha256
  **`b58ae61f4751ada0eda3501e0971f2f5367bab2f8f798656172dfa540be1f920`**; log
  `outputs/task081-d-1.log` (sha256 `835ecd29…5c8c`; the seven arm lines and the outcome, nothing
  else); stdout capture `outputs/task081-d-1.stdout` (empty, 0 bytes, as `--log` redirects the
  process after it opens).
- A copy of the three is in `~/develop/emai/evidence/task081-staged/` with the manifest
  `_checksums/task081-staged.sha256` (sha256 `1954ee22…6f33`), verified against the source and the
  copy.
- The `task081-staged` worktree is not edited and must not be removed before TASK-081's results
  PR: Stage S reads its D-PASS report (`--stage-d`).

### 6.6 Stage S's plan (R19.19)

Stage S runs once, on its own reported GO at this record's merge commit, from a fresh clean
worktree of that commit (`scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task081-stages
--run --from <merge sha>`). It is one invocation, from the worktree root, on the CPU (no GPU lock;
the stage creates no CUDA context):

- **`closed --cohort S`**: the 128 gated resets **70200–70327**, every arm of §4 once per reset,
  paired: **W, W-frozen, N, L-shuf, L-mean, L-rand, H-rule, H-sysid, H-final(commit), H-read and
  H-now** (W-frozen, H-read and H-now reported only; H-final, H-read and H-now privileged), then
  TASK-077's determinism re-run of W on S's first four resets (reading 0.1 cm, commit target
  0.6 cm, success identical; a difference is a G-determinism guard error). The inputs are D's: the
  six checkpoint reports, TASK-077's featurisation and Stage O fits, TASK-076's evidence root and
  TASK-080's Stage R report, **plus `--stage-d` set to the D-PASS report above**; the runner refuses
  any `--stage-d` that is not this task's non-debug D-PASS report run under the same frozen block.
  The GO also checks that report's sha256 (`b58ae61f…e920`) before launch, which the runner does
  not.
- **Rows:** §7.2, first match (V, S-VOID-CEILING, L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR,
  L-NEAR, L-BAR). Only L-NO-GAIN and L-INFERIOR fire the clause (§9). L-PASS is the primary claim,
  "LeWM-driven closed-loop success", and only it, with its own reviewed ruling, lets R7 state a
  gated LeWM result (§13).
- **Caps and resources (§10):** Stage S 21 600 s and 300 s per attempt; G-memory 12 GiB
  process-tree PSS (D peaked at 9.41 GiB, the debug smokes at 9.49); MemAvailable at start
  ≥ 16 GiB; G-quiet load ≤ 2.0 / 2.0 at the start; G-disk ≥ 10 GiB free.
- **Expected time.** From D's per-attempt medians and Stage 0's for the four arms D did not run
  (L-rand 6.53, H-sysid 4.28, H-read 8.08 and H-now 4.16 s; §11), about 97 s per reset over the
  eleven arms; 128 resets on 4 workers take about 52 min, plus the determinism re-run and D's
  measured overhead before the first outcome (210 s: G-tests, G-repro, the P readout's refit):
  **about 55–65 min**, far under the cap.
- **Launch-time checks** (the GO lists them): the worktree's HEAD is the merge commit with a clean
  tracked tree and no `outputs/task081-s*` anywhere; the 1- and 5-minute load averages ≤ 2.0 and no
  other heavy CPU job running or about to start (`hz.Clock` starts before G-quiet's wait, so a long
  wait counts against the cap); MemAvailable ≥ 16 GiB; ≥ 10 GiB free; the Stage R report
  `bedb8966…48ea7` and its fits unchanged; the D report `b58ae61f…e920`; no seed of 70200–70327
  simulated before.

If the invocation ends in anything other than a §7.2 row, nothing further is launched and the case
is ruled under §10 first. The exact command is in the GO. After S, the results PR reports every arm
(§6.1 step 5).

## 7. Gates, bars and rows

Intervals are reset-clustered bootstrap percentile intervals, 10 000 resamples, 95 %, salt 8302,
with TASK-080's estimators (`lewm_pr_v2.paired_interval`, `median_ci`; the same functions with this
task's salt). Every bar is labelled **measured**, **definitional**, **carried** or **allocation**.

### 7.1 Stage D (16 resets)

**L-DEV-STOP** or **D-PASS** (§6.1 step 3; the bars 12/16, 14/16 and +3/16 are **carried** from
TASK-080, with TASK-077's false-stop probabilities). W-frozen and H-rule are reported beside them.

### 7.2 Stage S, the gated rows (n = 128)

- **G-bar:** W(S) ≥ **112/128**. **Derivation:** TASK-080's G-bar 56/64 is τ_commit's bar fraction
  0.875 (28/32, TASK-076 G1's form; **carried**) applied to n; 0.875 × 128 = 112. **S-VOID-CEILING**
  if H-final(commit)(S) < 112/128.
- **G-NI:** the lower bound of the paired 95 % interval of W − C > **−16/128**, C the better of
  H-rule and H-sysid on S; the estimator is TASK-080's unchanged (the 2.5th percentile of the
  reset-bootstrap sum of W − C over the n paired resets); δ = 16/128 = 8/64 of n, an
  **allocation** carried as a fraction (R18.35: no measured quantity supports another value; a
  τ-calibrated margin would be tighter, about 6/64, and would need its own ruling).
- **G-N, G-shuf, G-mean, G-rand:** W > arm, exact one-sided McNemar p < 0.01 on the paired resets
  (**carried**).
- **"Detectably"** as R17.15: a twin test passes at McNemar p < 0.01; W is detectably no better than
  a twin when that test fails **and** the upper bound of W − twin < **+7** resets; W is detectably
  inferior when the upper bound of W − C < −16/128. The +7 is **definitional and stays a count**:
  it is the smallest number of discordant pairs, all in W's favour, at which the one-sided exact
  test can reach p < 0.01 (0.5⁷ ≈ 0.0078), which does not depend on n. It is not scaled to 14/128.

| row (first match) | condition | consequence |
|---|---|---|
| **V** | the void rule (§10) | one repeat after a recorded fix |
| **S-VOID-CEILING** | H-final(commit)(S) < 112/128 | escalate, no clause, no claim |
| **L-NO-GAIN** | for at least one of N, L-shuf, L-mean, L-rand, the McNemar test fails **and** W is detectably no better | **the clause fires** (§9) |
| **L-INFERIOR** | W is detectably inferior beyond δ (upper bound of W − C < −16/128) | **the clause fires** (§9) |
| **L-PASS** | G-bar, G-NI and the four McNemar tests all pass | **the primary claim, "LeWM-driven closed-loop success"**; "LeWM needed" is then reported |
| **L-TWIN-NEAR** | at least one McNemar test fails, and every failed one is a miss within noise | escalate, no clause, no claim |
| **L-NEAR** | G-NI fails, but W is not detectably inferior beyond δ | escalate, no clause, no claim |
| **L-BAR** | otherwise (G-bar fails, with G-NI and the tests passing) | escalate, no clause, no claim |

The ladder is TASK-080 §9.5's (`lewm_pr_v2.decide_s`'s order) with n = 128, G-bar 112, δ 16 and
this task's bootstrap salt. **W-frozen is read by no row.** Any repeat after L-TWIN-NEAR, L-NEAR or
L-BAR needs fresh seeds and its own ruling.

### 7.3 Reported in every row (R19.7; not gates)

1. Every arm's count with its exact binomial 95 % interval; every paired difference with its
   discordant counts and interval; H-read's and H-now's counts; the secondary claim's test.
2. **The solver's effect, W − W-frozen**: the paired difference, its discordant counts
   (W only / W-frozen only), its 95 % interval and the exact one-sided McNemar p. This is the
   design note's hypothesis tested on fresh seeds; it is reported, and no row, claim or clause
   depends on it. W-frozen's number of capped (not converged) resets, and W's and W-frozen's
   counts on those resets.
3. **Aim error against H-final(commit)'s committed aim on the same reset** (the design note's
   measure) for W, W-frozen, N, L-shuf, L-mean, L-rand, H-rule and H-sysid: median with interval,
   87.5th percentile, maximum. Beside it, the carried fixed-point error (against g\*) and landing
   miss (against the plate at s1 = 525), and each arm's predicted count through the pooled τ curve
   (a prediction, not a count).
4. **The final-distance distribution per arm** (design note §7, question 3): the apple's final
   distance from the plate centre, in cm, over all attempts that reached the place, and separately
   over successes and failures: median, 12.5th and 87.5th percentiles, maximum, and the number of
   attempts that ended between 3.5 and 4.5 cm (within 0.5 cm of the scorer's 4 cm radius). Every
   arm's apple rested about 3.4 cm from the centre in TASK-080's S (design note §2.2 point 5), so
   counts near the boundary depend on more than the aim; the distributions make that visible.
5. **affine_local's diagnostics** for W and each twin: fallbacks by cause, clip-binding, points in
   the fit, the real parts of J's eigenvalues and the number of complex pairs, and the logged
   residual at the committed aim.
6. W-66800's `last_two_triggered` flag beside every seed-66800 comparison.

## 8. Power (R19.8; design note §5, recomputed in Stage 0)

**Planning rates** (design note §7, question 2). C at **0.992** (S and the development seeds
pooled, 127/128), the stricter of 0.984 (S alone) and 0.992. W at **0.95**, the lower of the design
note's two development readings of affine_local (the τ-curve prediction 60.72/64 = 0.949 and the
closed-loop 62/64 = 0.969, the latter carrying the selection caveat).

**G-bar** (exact binomial, P(X ≥ 112 of 128)): 0.57 at a true rate of 0.875, 0.76 at 0.89, 0.91 at
0.906, 0.98 at 0.922, 0.9975 at 0.938 and > 0.999 at 0.953 (computed for this draft; the 0.938
value corrected from "≥ 0.998", #161 review, nit N1).

**G-NI** (the design note's §5 table, one comparator C = H-rule, salt 8301, 4 000 resamples per
(k+, k−) cell; overlap / half / independent couplings), n = 128, δn = 16:

| C | W = 0.938 | W = 0.953 | W = 0.969 |
|---|---|---|---|
| 0.984 | 0.93 / 0.87 / 0.82 | 0.99 / 0.98 / 0.95 | 1.00 / 1.00 / 1.00 |
| 0.992 | 0.85 / 0.80 / 0.77 | 0.97 / 0.95 / 0.93 | 1.00 / 1.00 / 0.99 |

At n = 64 the same cells read 0.52–0.66 (W 0.938) and 0.72–0.86 (W 0.953): the larger cohort is
justified by power at the development rate, not by S's outcome (R18.35). The size at the margin
(W = C − δ) is 2.8–3.4 % at n = 128 (nominal 2.5 %) in the design note's computation; the
L-INFERIOR false-fire probability at the margin is a different tail, computed in Stage 0 (below;
#161 review, nit N2). G-NI's real comparator is the better of H-rule and H-sysid, which lowers
power slightly; **Stage 0 recomputes this table with salt 8304, 10 000 resamples per cell and the
better-of-two comparator** (H-sysid at its TASK-080 S rate, 62/64), and adds rows at W = 0.906
(TASK-080's S rate) and W = 0.922, where G-NI's power is expected to be low: a W that the solver did
not improve is unlikely to pass.

**Stage 0's recomputation** ([record](apple_lewm_commit_precision_v2_stage0.md) §4; salt 8304,
20 000 trials per cell, 10 000 resamples per (k+, k−) cell, report `5043c3ab…a56c`). With the
better-of-two comparator (H-sysid at 62/64), n = 128, δ = 16, overlap / half / independent:

| W | C (H-rule) = 0.984 | C (H-rule) = 0.992 |
|---|---|---|
| 0.906 | 0.43 / 0.36 / 0.32 | 0.33 / 0.28 / 0.26 |
| 0.922 | 0.71 / 0.62 / 0.57 | 0.59 / 0.54 / 0.50 |
| 0.938 | 0.93 / 0.86 / 0.81 | 0.85 / 0.80 / 0.76 |
| 0.953 | 0.99 / 0.98 / 0.95 | 0.97 / 0.95 / 0.93 |
| 0.969 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

The single-comparator values agree with the design note's table to within 0.01. At the planning
rates G-NI's power is **0.93–0.97**; at TASK-080's S rate for W (0.906) it is 0.26–0.43. The size
at the margin is **2.6–3.6 %**, and **the clause's false-fire rate at the margin (L-INFERIOR with
W = C − δ) is 1.4–1.7 %**. The twin tests' power is ≥ 0.9997 in every coupling at TASK-080's twin
rates and against a twin at 0.60.

**The twin tests.** At TASK-080's S rates (N 26/64, L-shuf 21/64, L-mean 30/64, L-rand 11/64) and
W ≥ 0.9, the exact one-sided McNemar test at p < 0.01 on 128 paired resets has power above 0.999 in
every coupling (TASK-080 §10's table at n = 64 already gave ≥ 0.95 at N 0.60); Stage 0 recomputes it
at n = 128. N's aim under affine_local is the same as under TASK-080's refinement (§3.1), so N's rate
is not expected to move. **The binding risk is G-NI**, as in TASK-080.

**L-PASS at the planning rates** is close to G-NI's power (G-bar and the twin tests are near 1):
about 0.93–0.97 at W = 0.953 against C = 0.992, and lower if W's true rate is below the development
readings.

## 9. The abandonment clause and its scope (R19.9)

**It fires on L-NO-GAIN or L-INFERIOR only**, as TASK-080 §12 and R17.15 define them. It does not
fire on L-DEV-STOP, S-VOID-CEILING, L-TWIN-NEAR, L-NEAR, L-BAR, V or INCONCLUSIVE.

**What closes: TASK-077's scope (R17.10) unchanged.** Neither TASK-077's nor TASK-080's clause
fired, so the scope is still open, and TASK-081 tests a member of it:

> "LeWM aim selection with a single aim committed at 405 under the declared reactive-plate rule
> (κ = −0.5, L = 2, s1 = 525) on v2, with the declared post-pick plate move of radius ρ\* = 4 cm
> (the disc, which also covers the −y half-disc, at radii ≤ 4 cm), from onboard 112 px frozen
> DINOv2 pooled tokens, with TASK-066-family predictors."

with TASK-077's one declared narrowing (a 4 × 4 pooled latent on a corpus larger than 1 024 roots
stays open). **If the clause fires, the results document states the restored scope plainly**: one
8 × 8 run, read by a dual ridge fitted on its own stand-in predictions **and committed by the
affine fixed point of §3**, closes under this condition every other pooled grid (except that 4 × 4
case), every readout of the predicted latent (TASK-080 §12's wording), **and every post-grid solver
for the committed aim from the grid's predictions**: the frozen refinement with any cap or damping,
best-residual, local or global affine or higher-order fits, Newton or secant steps and iterate
averaging, not only the solver that was run.

**What does not close:** TASK-080 §12's list unchanged (the full, unpooled token grid; history
longer than one, `action_chunk` and `predictor_step_embedding`; a fine-tuned or another encoder;
other views or resolutions; other κ, L, commit steps, move distributions or radii above 4 cm; C1
without the move; C2; TASK-076's results; the LeWM backend; v2; the product goal), plus R18.32's
direction (a), a condition in which no hand-written arm is given the plate law (a task change), and
several stand-in chunks per aim (a change to the controller's inputs, design note §3).

## 10. Void rule, guards and memory (carried from TASK-080 §11)

- **Void and repeat:** TASK-080 §11 (one repeat per stage after a committed, pushed and recorded
  fix, on the same seeds in a new output directory; a second V of the same stage ends TASK-081 as
  INCONCLUSIVE; Vs in different stages do not add up).
- **Guards:** TASK-080 §11's list (G-tests in-run, G-sentinel, G-hash, G-frozen, G-repro,
  G-threads with the same pinned environment, G-quiet, G-privileged, G-finite, G-memory at 12 GiB
  process-tree PSS, G-disk at 10 GiB free), with **G-frozen and G-hash extended** to TASK-080's
  frozen block and seven file pins, the TASK-080 Stage R report's sha256 and R-S's and R-N's content
  sha256s, and `commit_precision_dev.py`'s file sha256. **G-split here:** nothing is fitted;
  the runner refuses any closed-loop seed outside its range and any S run without a non-debug
  D-PASS report of this task. **G-solver (new):** every W, N, L-shuf and L-mean attempt logs its
  solver, and the runner refuses a record whose solver is not the arm's declared one (W-frozen's is
  TASK-080's `choose`; the others' affine_local). No GPU is used.
- **Caps** (set at Stage 0, R19.12; each ≥ 1.5 × its scaled worst case from the debug smokes):
  Stage D **7 200 s** (scaled about 510 s), Stage S **21 600 s** (scaled about 3 420 s), per
  closed-loop attempt **300 s** (slowest debug attempt 15.6 s). G-memory: debug peak 9.49 GiB
  process-tree PSS against 12 GiB.

## 11. Compute estimate (R19.11)

| stage | estimate | basis |
|---|---|---|
| Stage 0 | debug smokes a few minutes each; power simulation a few minutes | TASK-080's Stage 0 |
| D | about 10 min CPU | per-attempt medians from TASK-080's D and debug S (W 13.85 s, N 6.77, L-shuf 14.02, L-mean 13.92, H-final 7.74, H-rule 4.50; W-frozen as W): about 75 s per reset over the seven arms, 16 resets on 4 workers ≈ 300 s, plus G-tests (about 165 s) and G-repro (about 42 s) |
| S | about 1 h CPU | the same medians plus L-rand 6.53, H-sysid 4.28, H-read 8.08 and H-now 4.16 s: about 98 s per reset over the eleven arms, 128 resets on 4 workers ≈ 3 130 s, plus the determinism re-run, G-tests and G-repro; TASK-080's S took 1 581 s for 64 resets and ten arms |

No training, no featurisation and no GPU. Disk: reports only (TASK-080's S report was a few MB).

## 12. Risks, stated now

- **The development 62/64 is optimistic** (selection among three variants after seeing them). The
  planning rate is the lower τ-curve reading, 0.95; if W's true rate is near TASK-080's 0.906, G-NI
  is likely to fail (L-NEAR), and §7.3's W − W-frozen will show whether the solver helped at all.
- **H-rule is strong** (63/64 on TASK-080's S, 64/64 on development seeds): non-inferiority within
  12.5 points needs W at about 0.95 or above.
- **The scorer's margin is small** (§7.3 item 4): outcomes near the 4 cm boundary depend on more
  than the aim, which adds noise to every arm's count, the ceiling's included (H-final(commit) 62/64
  on TASK-080's S).
- **W-66800 is flagged** (`last_two_triggered`).
- **The scorer and condition are imposed simulator laws**, as TASK-077 §13 says; nothing here
  transfers as such to Arena or the real G1.

## 13. R7 (design note §7, question 6; R19.10)

- **An L-PASS would still need its own reviewed ruling** before R7 states a gated LeWM success
  (R18.12's rule, carried). It is proposed, not ruled, that a TASK-081 L-PASS count as TASK-079's
  "TASK-077 L-PASS (or its equivalent row)" precondition in PLAN.md, as R18.21 proposed for
  TASK-080; TASK-078's pass and an owner ruling on Arena stay required.
- **A factual issue with R7's last clause, recorded now.** R7's v2 LeWM part (as updated by
  R18.31) ends "LeWM therefore has no gated closed-loop Apple→Plate success, and its only other
  closed-loop Apple→Plate runs whose counts are read are on v1, with 0 successes." (R7 itself ends
  with the "Scripted-expert … not project-learned results" sentence; #161 review, nit N6.) The design note's
  development check (R18.35) ran W in closed loop on v2 on seeds 70000–70063 and its counts were
  read (W:frozen 57/64, W:damped 58/64, W:affine_local 62/64), so the "only other … on v1" clause is
  no longer literally true. It is a statement about development runs, not a success claim, and the
  first part ("no gated closed-loop Apple→Plate success") stays true. **Decided (R19.10):** the
  correction is made once, in Stage D's record, which adds TASK-081's own development counts in
  the same way R18.28 did for TASK-080; until then this section and R19.10 record the discrepancy.
  If Stage D does not run, the correction is made in TASK-081's closing PR. **Kept at Stage 0
  (R19.14; #161 review, nit N7, which suggested an earlier fix):** the sentence is quoted verbatim
  in the repository's instruction and entry files (AGENTS.md, CLAUDE.md, README.md and others,
  R18.31's list), so its correction is a cross-document edit made once, by the main session with
  Stage D's record, rather than twice; the discrepancy is recorded here, in R19.10 and in R19.14
  until then.
- **Corrected at Stage D's record (R19.18).** After D-PASS (§6.5), R7's last clause was replaced
  in every file that quotes R7, by a factual statement of TASK-081's development counts (the
  design note's check and D), labelled as non-gating development runs; "LeWM therefore has no
  gated closed-loop Apple→Plate success" is kept. The quote above is the clause as it stood at
  the freeze.

## 14. Open questions of the design note, resolved (R19.2–R19.10)

1. **Neighbourhood and an extra damped refinement:** the neighbourhood stays 2 a-steps × 2 cm (up
   to 25 points); **no** damped refinement after the affine solution (§3.2).
2. **Planning rate for C:** 0.992, the stricter (§8).
3. **The scorer's boundary:** every arm's final-distance distribution is reported beside its count
   (§7.3 item 4).
4. **Seed 66800's flag:** carried unchanged and stated beside every seed-66800 comparison (§2.2).
5. **A Stage-R-style offline gate:** not run (§6.3).
6. **R7:** an L-PASS needs its own reviewed ruling; the literal-truth issue is recorded and
   corrected at Stage D's record (§13).
