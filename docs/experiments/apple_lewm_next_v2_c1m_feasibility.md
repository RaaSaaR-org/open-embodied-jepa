# C1-M feasibility record: C1 plus a declared post-pick plate move (development)

**What this is.** The development feasibility record that the ruling R15
([apple_lewm_next_v2_direction.md](apple_lewm_next_v2_direction.md), §5.3–§5.5; merged in #140)
requires before any protocol for C1-M. It runs R15's checks M-F1 to M-F7 on the CPU, with **no
world model**: every arm below is a privileged calculation from simulator truth (H-final(commit),
H-read, the proxies, the reach and corpus collectors) or a hand-written non-world-model arm
(H-rule, H-sysid). Nothing here is a protocol, a gate of a learned arm or a learned result. The
template is the C1 record ([apple_lewm_next_v2_c1_feasibility.md](apple_lewm_next_v2_c1_feasibility.md),
R14); the design note is [apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (R9).

**Status: declarations only (§1), written and committed before any seed of the block below is
simulated.** §2 (results) is written after the record run.

Every choice below is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R16.1–R16.12 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-05, entered there **in the same
commit as this section, before the run** (the C1 review's lesson: R14.1–R14.8 existed only in the
record until after its run). R1–R15 are taken; a search of every local and remote ref (79) and
every worktree's `docs` on 2026-10-05 found no R16.

The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged by this record, verbatim:

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

## 1. Declarations (written before the run)

### 1.1 Seeds, salts and the debug range (R16.2)

R15.9's block 58000–58999 served R15's arithmetic only and is not used here. This record's block
is new.

| range | seeds | use |
|---|---|---|
| block | 63000–64999 | C1-M's feasibility record only |
| M1 | 63000–63031 | M-F1 (H-final(commit) at each radius, every radius and family on the same 32 resets in the same order) and M-F2 (the reach check at ρ\*) |
| F3 | 63100–63131 | M-F3's 32 fresh paired resets (every proxy and the fresh ceiling); also the first half of M-F5b's 64 and M-F6's comparators |
| R | 63132–63163 | the second half of M-F5b's 64 paired resets (H-final(commit) and H-read) |
| T | 63200–63231 | M-F4: τ_commit's 32 resets (every planted level on the same 32) |
| corpus | 63300–64323 | M-F5a: the 1 024 roots of the declared-aim corpus at ρ\* (and H-sysid's fit) |
| debug | 64900–64999 | mechanics debugging of the runner only; nothing in it is read |
| reserved | everything else in 63000–64999 | unused; any use needs its own ruling |

| salt | use |
|---|---|
| 7901 | the move draw per reset: `default_rng(SeedSequence([7901, seed, k]))`, k the re-draw index (§1.2) |
| 7902 | the corpus's uniform (a, b) draw per root: `default_rng(SeedSequence([7902, seed]))` |
| 7903 | the readouts' 5 outer folds by root |
| 7904 | λ selection (5-fold inner CV grouped by root) in every ridge |
| 7905 | every bootstrap interval (10 000 resamples) and the McNemar feasibility resampling |
| 7906 | τ_commit's planted-error direction per reset: `default_rng(SeedSequence([7906, seed]))` |
| 7907 | the learning curve's nested subsample: one permutation per outer fold, `default_rng(SeedSequence([7907, fold]))` |

**The search** (2026-10-05, at `0c64ec2`): `git grep -w` over all 79 local and remote refs (every
path), plus the working files (`src`, `scripts`, `tests`, `configs`, `docs`, `benchmarks`) of
every other worktree under `/home/huhn/develop/emai/worktrees/` and the two other checkouts. In
63000–64999 the only matches are `"global_wall_seconds": 64800` (TASK-062's manifest), a decimal
fraction's digits (`0.64898`, `apple_wide_grasp_closure_results_v3.md`), a byte count (`63617`)
in `uv.lock` and a latency (`64786`) in a non-repository worktree; none is a seed. 59000–59999
(TASK-068/069's declared ranges) and 60000–62999 (seed-like values in manifests and docs) were
not taken. 7901–7909 appear nowhere in `src`, `scripts`, `tests` or
`configs` on any ref or worktree (the only `79xx` match anywhere is a `7900` in a results
manifest). The block lies outside every forbidden range of TASK-076 and C1
(`lewm_next_c1.FORBIDDEN_RANGES`), outside TASK-076's block 56000–56999, C1's 57000–57999 and
R15's 58000–58999; the salts are distinct from every earlier task's and from R15's 7801–7802.
`lewm_next_c1m.check_seed_ranges` checks this, and the tests pin it.

**Resets.** `wm_critic_v2.wide_reset_values(seed)`: v2's own reset, unchanged (plate
(0.49, −0.09) ± 2 cm, apple (0.34, −0.18) ± 3 cm, uniform per axis), as in C1.

### 1.2 The condition: C1 plus one declared post-pick plate move (R16.3)

C1 unchanged (R14.2): P-3 picks with G-repro's post-look estimates; cell A's rule as TASK-076's
`CellMotion` implements it (κ = −0.5, L = 2, s0 = 405, s1 = 525; a refused plate move ends the
attempt as a counted failure); a single aim committed at 405 (`CommitController`); e9's place
primitive. **Plus one change (R15.5):**

- **When and how.** Just before the observation of post-look step 300, the hook writes the plate's
  `body_pos` by the reset's stored offset and calls `mj_forward` (`plate_shift.move_plate`, TASK-073's
  hook function, unchanged), outside every controller. The rule's base, plate(405), is then the
  moved plate: the plate is static from 300 to 405 and follows the rule from 405.
- **The draw.** One draw per reset seed, shared by every radius and both families:
  u, v = two uniforms from `default_rng(SeedSequence([7901, seed, k]))`, k = 0 first.
  Disc of radius ρ: offset = ρ·√u·(cos 2πv, sin 2πv). The −y half-disc of radius ρ: offset =
  ρ·√u·(cos(π + πv), sin(π + πv)) (its y component is ≤ 0). Both are uniform over their region.
- **Re-draw rule (off-table).** If the reset's plate plus the offset at the grid's largest radius
  (6 cm) in either family leaves `wm_critic_v2.TABLETOP` (x 0.18–0.65 m, y −0.32–0.32 m), k is
  increased by one and both uniforms are drawn again (at most 50 times; the stored k is reported).
  With v2's reset no draw can leave the tabletop (x ≤ 0.57 m, y ≥ −0.17 m), so k = 0 is expected
  everywhere; the rule is declared so that nothing is decided later.
- **A blocked move** (`move_plate` refuses because the moved plate would touch something other
  than the table: the message starts with `SHIFT_BLOCKED_PREFIX`) ends the attempt as a counted
  failure in every arm, recorded as blocked. Any other refusal of the move (non-finite, wrong
  height) is a guard (V). An attempt that ends before step 300 never has its plate moved and is a
  counted failure as it ends; an attempt that reaches the observation of step 300 without the move
  applied is a guard (V).
- **p̄ (the mean-proxy's plate)** is the move distribution's mean plate at the decision: v2's
  reset centre (0.49, −0.09) plus the family's mean offset, 0 for the disc and
  (0, −4ρ\*/(3π)) for the −y half-disc (the analytic centroid, as R15 §3.3).

### 1.3 The code (R16.1)

- **Reused unchanged:** TASK-076's worker, `CellMotion`, `AimController`, `LookaheadAim` (H-final),
  `RuleAim`'s reading, `PlateBrancher`, the harness's G-repro, render majority and pins; C1's
  `C1Motion`, `CommitController`, `GeometricAim`, `RuleCommit`, `SysidAim`, geometry and box.
  **No hash-pinned TASK-073–076 file and no C1 file is edited.** The runner checks TASK-076's
  manifest pins at the start and the end, and the sha256 of C1's three code files against
  `0c64ec2`.
- **New code** (behind tests, imports lazy): `src/embodied_jepa/lewm_next_c1m.py` (constants,
  draws, statistics with this record's salts, the check rules and the row ladder),
  `src/embodied_jepa/lewm_next_c1m_runtime.py` (the move hook, the attempt, H-read, the planted
  aim, H-rule-stale) and `scripts/run_c1m_feasibility.py` (one invocation, every stage in order,
  with the early stop).
- **ρ\*, r, a_lo and τ_commit are set by their rules in code**, inside the one invocation. No
  constant is set by hand between stages, and none is changed after a result.
- **GPU: none.** Every stage runs on the CPU, featurisation included; no GPU lock, no CUDA context.
- **The quiet machine.** The record run starts only when the 1- and 5-minute load averages are
  both ≤ 2.0 (`wm_critic_v2.QUIET_MACHINE`); the runner waits for that and records the wait. Six
  CPU workers, one torch thread each (TASK-076's settings).

### 1.4 M-F1: ρ\* and the ceiling (R16.4)

On M1's 32 resets, H-final(commit) (TASK-076's `LookaheadAim` called once at 405, unclipped,
tolerance τ_re/4 = 0.25 cm, at most 10 iterations; as C1) under the disc move at ρ = 3, 4, 5, 6 cm
in that order. A radius **passes** at ≥ 30/32 counted successes, refused and blocked moves counted
as failures. The loop **stops at the first radius that fails** (no larger radius can be ρ\* under
the monotone rule). **ρ\*** = the largest radius such that it and every smaller one pass. If 3 cm
fails in the disc family, the same loop runs **once** on the −y half-disc family; if 3 cm fails
there too, the record ends **M-INFEASIBLE**. M1's 32 resets serve only this choice and M-F2.

### 1.5 M-F2: motion, r and reach at ρ\* (R16.4)

- From M-F1's attempts at ρ\* (C1-F1's definitions, R14.3): the **remaining motion**, the median
  over unrefused attempts of |plate(525) − plate(405)|, bar ≥ 2 cm; **r**, the earliest step in
  405–525 at which ≥ 28/32 attempts are within 0.1 cm of plate(525) (refused never within),
  recorded; the palm speed at 405 and m2, recorded (> 0.5 cm per step would give W and N the last
  two commands in a protocol; not a failure).
- **Reach** (R14.4's rule, at ρ\* and its family, on M1's resets): e9's aim committed at 405 to
  g = p + a·(h − p) + b·n from the true p and h for a ∈ {−0.5, −0.4, −0.3, −0.2},
  b ∈ {−3, 0, +3} cm (12 × 32 attempts); complete = no fallback and ≥ 505 commands executed with
  no refusal; a reset is reachable at a level if all three b complete; a level is reachable at
  ≥ 31/32; a_lo is the most negative level reachable together with every level above it.
- **M-F2 fails** if the remaining motion is below 2 cm, or a_lo or r is undefined.

### 1.6 M-F3: the twins lose at ρ\* (R16.5)

On F3's 32 **fresh** resets, at ρ\* in its family, single commit at 405 from the true p and h
(p the moved plate), every proxy aim clipped to the box around the true plate (C1's
`clip_to_box`, a_lo from M-F2), as C1:

- **H-final(commit)**, the fresh ceiling (R15.7: M-F1's ceiling was selected for passing).
- **H-now:** the true plate at 405. **N-proxy:** g = p + a_N·(h − p), a_N = min(0.5·(1 − a_m), 0.5),
  a_m = (a_lo + 0.5)/2.
- **Shuf-proxy:** (p′ − κh)/(1 − κ), p′ the true plate at 405 of reset (i + 1) mod 32 in F3's order
  (its reset plate plus its own offset at ρ\*; the plate is static from 300 to 405).
- **Mean-proxy:** (p̄ − κh)/(1 − κ), p̄ as in §1.2.

**Bars (R15.7, the strict reading).** For each of the four proxies, the paired headroom
(ceiling − proxy on the same resets) must have the **lower bound of its paired, reset-clustered
95 % bootstrap interval ≥ +8/32** (salt 7905, C1's estimator). For the shuf- and mean-proxy, the
McNemar feasibility for the 64-reset test (R8.12's form: the fraction of 10 000 resamples of 64
pairs drawn with replacement from the 32, salt 7905, whose exact one-sided McNemar p is below 0.01)
must be ≥ 0.8; the point p on the doubled discordant counts is reported.

**The fresh ceiling (R16.5, the reviewer's note 2 on #140).** If H-final(commit) scores **below
30/32 on F3**, the row is **M-INFEASIBLE**, whatever the proxies show: the ceiling bar is the
admission precondition, and a twin's headroom measured against a ceiling that fails it cannot fire
a clause. Its proxies are still reported.

**Early stop (R15.7).** M-F1 to M-F3 run first. If M-F3 does not pass (or M-F1/M-F2 fail), the
record stops there with its row, and M-F4 to M-F7 are **not run** (reported as not run).
**R16.6 (the reviewer's note 3 on #140):** with ρ\* ≤ 4 cm in the disc family (≤ 5 cm in the −y
half-disc) M-TWINS-ESCALATE is the projected row (R15 §3.3: strict margins +4.0 and +7.7 at 3 and
4 cm, disc), and the projected-pass radii (5–6 cm, disc) are the ones most exposed to a ceiling
drop (TASK-074's design Probe A saw e9 fall to 21/32 at an 8 cm any-direction shift). **An early
stop at ρ\* ≤ 4 cm is a realistic outcome of this record, not an unlikely one.**

### 1.7 M-F4 to M-F7 (run only if M-F1 to M-F3 pass; R16.8)

- **M-F4, τ_commit** (on T's 32 resets at ρ\*): H-final(commit)'s converged aim (unclipped) plus a
  planted error of 0, 0.5, 1, 1.5, 2 or 3 cm in a per-reset uniform direction (angle 2π·u,
  u from salt 7906; one direction per reset, shared by every level). Every level runs on the same
  32 resets (6 × 32 attempts). τ_commit = the largest level such that every level up to it reaches
  ≥ 28/32 (TASK-076 K0's rule); **undefined if level 0 misses**.
- **M-F5a, the readout offline** (the corpus, 1 024 roots, 63300–64323, at ρ\*): P-3 picks, the move
  at 300, and at 405 e9's aim is committed to (a, b) uniform over the box (a ∈ [a_lo, 0.5],
  b ∈ [−3, +3] cm, salt 7902) from the true p and h (the note's privileged scripted collector). At
  405 and at r the onboard 112 px frame is kept, and at r also a plate-hidden render of the same
  state (TASK-075's renderer, as C1). A root that does not reach r is excluded and counted.
  - **Readouts at r** (pooled 4 × 4 and pooled 8 × 8 tokens of the frozen pretrained DINOv2
    ViT-S/14, `frozen_tokens.pool_tokens`): dual ridge to the true plate at r, cross-fitted over 5
    outer folds by root (salt 7903), λ by inner CV (salt 7904, TASK-076's λ grid). The same fold's
    readout reads the held-out plate-hidden renders at r. Medians with bootstrap intervals over
    roots (salt 7905), 87.5th percentiles.
  - **Bar (O4's form, against the measured tolerance):** the lower 95 % bound of the plate-hidden
    median error **> τ_commit**, on 4 × 4 (it gates the row M-ARM-KEYED); on 8 × 8 it is computed
    as well and admits 8 × 8 to M-F5b (if it fails there, 8 × 8 is not run in M-F5b and counts as
    a miss whose interval is undefined). If τ_commit is undefined the check has no bar and is
    reported.
  - **The learning curve** (R15.6): fractions 1/4, 1/2, 3/4 and 1 of each outer fold's fit roots,
    **nested** (the first ⌊f·m + 0.5⌋ roots of one permutation per fold, salt 7907) and **shared
    between grids** (the same subsets for 4 × 4 and 8 × 8), each predicting the same held-out roots.
    **Falling-curve guard (TASK-075 §7's form):** on a grid, if the lower 95 % bound of
    median(e at 3/4) − median(e at all), bootstrapped over roots (paired, salt 7905), is above 0,
    that grid's readout is still data-limited.
  - Also reported: R-plate (full tokens) at 405, cross-fitted on the same folds (the plate input
    of H-rule and H-sysid); a constant prior at r; the plate's motion from 405 to r.
- **M-F5b, H-read** (on F3 + R = 64 resets, 63100–63163, at ρ\*). H-read is H-final(commit)'s
  cloned-state look-ahead (TASK-076's `PlateBrancher` save and restore, the rule active in the
  clone) in which, at every iteration, the roll-out of the place primitive aimed at g_k runs **to
  r** (not s1), the onboard 112 px frame at r is **rendered** in the clone (every earlier step is
  blank, as in H-final), and **g_{k+1} = the pooled readout's reading of that frame** (CPU, batch
  1, the same featurisation as the corpus). g0 = the true plate at 405, as in H-final: a disclosed
  privileged start of the fixed-point iteration, whose converged point depends only on the readout.
  Stop at |g_{k+1} − g_k| ≤ 0.25 cm, at most 10 iterations (aim at the last iterate, logged as not
  converged); a branch that stops before r ends the iteration (aim at the last iterate). Its
  readout is fitted on **all** corpus rows (dual ridge, λ by inner CV, salt 7904); the corpus roots
  are disjoint from the 64 resets. H-final(commit) on F3 is M-F3's (the same deterministic arm);
  it runs on R's 32. The true plate at r in each clone is logged (privileged, reported only).
  - **Bar:** paired ceiling − H-read **≤ 4/64**, a point reading (R15.6; an allocation of the
    note's proposed δ = 8/64, not calibrated, inside the noise). 4 × 4 first; **8 × 8 only if
    4 × 4 misses** (and its plate-hidden check passes). The paired interval (salt 7905) is
    reported for every grid run.
- **M-F6, comparators** (on F3's 32 resets at ρ\*, reported, no bar), single commit at 405, each
  aim clipped to the box around its own plate input (C1's code):
  - **H-rule / H-sysid** on the reading: R-plate (full tokens, fitted on every corpus root's frame
    at 405; dual ridge, salt 7904) read from the onboard frame at 405; H-sysid's regression
    plate(r) ≈ c + A·p + B·h + C·g is fitted on the corpus with the out-of-fold readings at 405;
  - **H-rule-true / H-sysid-true:** the true plate at 405 (H-sysid-true fitted with the true plate);
  - **H-now-reaim:** TASK-076's H-now unchanged (re-aims at the current true plate, 405–485);
  - **H-rule-stale:** H-rule's fixed point fed **P-3's post-look plate estimate** (G-repro's
    reading of the post-look frame, before the move: the pre-move reading P-3 itself acts on),
    clipped to the box around that estimate.
  The better of H-rule and H-sysid on the reading is the protocol's non-inferiority comparator.
- **M-F7, cost:** the wall time of each H-final(commit) and H-read attempt; bar ≤ 60 s at the
  maximum, above which the per-attempt cap is reviewed (not a stop).

### 1.8 The rows, in order (R16.9)

`lewm_next_c1m.verdict`, R15.8's ladder plus R16.5:

1. M-F1 has no passing radius in either family, or M-F2 fails, or **M-F3's fresh ceiling is below
   30/32** → **M-INFEASIBLE**: escalate, no clause.
2. M-F3: a **scene-blind** proxy's paired upper bound < +8/32 (detectably below) →
   **M-TWINS-NONE: the clause fires** (R15 §5.5, scoped to the family in which ρ\* was set, at
   radii ≤ ρ\*). Any other M-F3 failure (a lower bound below +8 while the upper bound is not, an
   H-now or N-proxy failure, or a feasibility < 0.8) → **M-TWINS-ESCALATE**: escalate, no clause.
   The record stops (§1.6).
3. M-F4: τ_commit undefined → **M-NO-TAU**: escalate, no clause.
4. M-F5a: the 4 × 4 plate-hidden check fails → **M-ARM-KEYED**: escalate, no clause.
5. M-F5b: H-read misses on 4 × 4 and on 8 × 8 (or 8 × 8 is not admitted) → if on **both** grids the
   paired interval of ceiling − H-read lies entirely above 8/64 (lower bound > 8/64) **and** neither
   grid's learning curve is still falling → **M-READ-NONE: the clause fires** (R15 §5.5). If a
   curve is still falling → **M-NO-BAR-DATA**: escalate to a larger corpus, no clause. Otherwise
   → **M-NO-BAR**: escalate, no clause.
6. Otherwise → **M-PROCEED**: a preregistration may be drafted under R9.8 and R9.9 (R15.8, row 6),
   with its own K0, Stage O, a `check_budget`-passing training block and an independent review.

M-F6 has no bar, and M-F7 only reviews the cap. Nothing here closes the LeWM backend, v2, the
product goal, TASK-076's results or C2.

### 1.9 How intervals are stated (R16.7, the reviewer's note 1 on #140)

Every interval in this record is **computed from the observed pairs** with the paired,
reset-clustered bootstrap (salt 7905), never read off a quoted value. R15's quoted intervals hold
**with at most one reversed pair** (C1 had one): with C1's estimator, a difference of 4/64 gives
[+1, +8] with no reversed pair, [−1, +9] with one and [−1, +10] with two; a headroom of +10/32
with two reversed pairs gives [+3, +16] (a lower half-width of 7, not 6). The general rule: at a
fixed difference, each reversed pair adds two discordant pairs and widens the interval, so
R15.2's 6/32 half-width is a value for at most one reversed pair, not a bound.

### 1.10 Void and repeat (R16.10)

- A run that ends **V** (any guard: a tracked-tree change, a pin, a dead worker, a cap, memory, a
  signal) is void. Its report is kept, its sha256 recorded and the cause disclosed in §2.
- **At most one repeat**, and only **after a fix that is committed (and pushed) and recorded**
  here: the cause, the fix's commit and the void report's sha256. A cause outside the code (an
  operator action, a signal, the machine's load) still needs a committed record of the cause and
  of its prevention before the repeat. The repeat runs from scratch at the fix's revision, on the
  same seeds (the simulation is deterministic, so fresh seeds would not make it fairer).
- A second V ends the record **V-ESCALATE**: no row, escalate to the owner.
- A run that completes is the record. No run is repeated because of its result, and no tracked
  file is edited in a run's worktree while a run is going.

### 1.11 Debug runs (R16.11)

Debug runs happen **only after the commit that declares this section, and only on committed
code** (`tracked_tree_dirty: false` with every runner file tracked; the runner refuses a debug run
otherwise). They simulate only 64900–64999 at small sizes (M1 64900–64905, F3 64910–64915,
R 64920–64925, T 64930–64935, corpus 64940–64979), run every stage whatever a check shows, and use
fixed stand-ins where six resets cannot define a value (ρ\* = 3 cm disc, a_lo = −0.2, r = 485,
τ_commit = 1.0 cm). Nothing in them is read for the row; their reports are listed in §2.

### 1.12 What this record would and would not show

A pass (M-PROCEED) admits the drafting of a preregistration only; it is not a LeWM result.
M-INFEASIBLE, M-TWINS-ESCALATE, M-NO-TAU, M-ARM-KEYED, M-NO-BAR-DATA and M-NO-BAR escalate without
a clause. M-TWINS-NONE and M-READ-NONE fire R15 §5.5's clause with its declared scope. The
canonical status sentence is unchanged by any row of this record (R16.12).

## 2. Results (written after the run)

Not yet run.
