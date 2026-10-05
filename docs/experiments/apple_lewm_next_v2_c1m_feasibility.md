# C1-M feasibility record: C1 plus a declared post-pick plate move (development)

**What this is.** The development feasibility record that the ruling R15
([apple_lewm_next_v2_direction.md](apple_lewm_next_v2_direction.md), §5.3–§5.5; merged in #140)
requires before any protocol for C1-M. It runs R15's checks M-F1 to M-F7 on the CPU, with **no
world model**: every arm below is a privileged calculation from simulator truth (H-final(commit),
H-read, the proxies, the reach and corpus collectors) or a hand-written non-world-model arm
(H-rule, H-sysid). Nothing here is a protocol, a gate of a learned arm or a learned result. The
template is the C1 record ([apple_lewm_next_v2_c1_feasibility.md](apple_lewm_next_v2_c1_feasibility.md),
R14); the design note is [apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (R9).

**Outcome: M-PROCEED** (§2.9), on one run, CPU only, with no world model. At ρ\* = 4 cm (disc;
5 cm reached 29/32) the fresh ceiling scored 32/32 and every proxy cleared the strict +8/32 bar
(mean-proxy 12/32, headroom +20, interval [+15, +25]). τ_commit = 1.0 cm. The 4 × 4 oracle-dynamics
readout arm H-read missed its allowance (64 against 58 of 64, +6/64), and the 8 × 8 arm met it
exactly (64 against 60, +4/64, interval [+1, +8], a point reading inside the noise). **M-PROCEED
admits only the drafting of a preregistration**, and that preregistration is committed to an
8 × 8 latent whose dynamics no earlier task has gated. **No world model ran, and no arm here is a
learned or LeWM result.**

§1 (the declarations) was written and committed (`8601c1b`, pushed) before any seed of the block
was simulated, and is kept as written. The code followed in `d77f001`; one debug run on debug seeds
and then the one record run used that commit (§2.1).

Every choice below is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R16.1–R16.12 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-05, entered there **in the same
commit as this section, before the run** (the C1 review's lesson: R14.1–R14.8 existed only in the
record until after its run). R16.13–R16.14 follow the run. R1–R15 are taken; a search of every local and remote ref (79) and
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

Every number below is read from the record's `report.json` (§2.1). Intervals are the paired,
reset-clustered bootstrap or the bootstrap over roots, with salt 7905 (§1.9).

### 2.1 The run and its provenance

- **The record run:** `scripts/run_c1m_feasibility.py --output outputs/c1m-run-1 --evidence
  /home/huhn/develop/emai/worktrees/task076-evidence`, in the worktree
  `/home/huhn/develop/emai/worktrees/c1m-run` (made with `scripts/new_worktree.sh --run --from
  d77f001`).
  - Code revision `d77f001c2b776cd6c8d8628bb640d164da508ed5` at the start and at the end,
    `tracked_tree_dirty: false`; TASK-076's manifest pins matched at the start and the end; C1's
    three code files matched their `0c64ec2` blobs.
  - 2026-10-04 23:09:20Z to 23:48:37Z (2 357 s). The runner waited 180 s for a quiet machine (the
    debug run had just ended); load averages at the start 0.30, 1.95, 2.60 (1, 5, 15 min). Six CPU
    workers, one torch thread each; peak process-tree PSS 10.42 GiB (ceiling 12 GiB); no GPU, no
    GPU lock, no CUDA context.
  - G-repro: all 8 of TASK-072 run-1's reproduction checks passed. No render disagreement on any
    cohort.
  - Outcome: **M-PROCEED**. **One run, no void, no repeat** (R16.10 was not needed).
- **Files** (all in `outputs/c1m-run-1/`, git-ignored, on the Linux PC):

  | file | sha256 |
  |---|---|
  | `report.json` | `916d7c018597510cdc9f79b7d934c6e06ca9ebe210476b36313b7d0d841b5a02` |
  | `corpus.npz` (1 018 rows) | `fe3ff3eb3cd870e7ddef0b0cbdd817145d5b7cbfcb4e7c6e46ce3f3f43b5520d` |
  | `r_read_4.npz` (H-read's 4 × 4 readout) | `26c74689ddbd876435ac6b1593e72536e1b80fa4e2779625fe8b7ec122bad680` |
  | `r_read_8.npz` (H-read's 8 × 8 readout) | `79ee9dd11624989e7d4dd9fd697a8d7d3e9f43371fe6b03d00ba96ba0f82aa6e` |
  | `r_plate.npz` (R-plate at 405, full tokens) | `9e918adca0320fb5917e89e39a28986457bead16f469dce9c503f04357f60c40` |

- **Seeds simulated:** M1 63000–63031, F3 63100–63131, R 63132–63163, T 63200–63231, corpus
  63300–64323; the debug run below on 64900–64905, 64910–64915, 64920–64925, 64930–64935 and
  64940–64979. No other seed.
- **The debug run, disclosed (R16.11).** `outputs/c1m-debug-1` in the worktree
  `/home/huhn/develop/emai/worktrees/c1m-debug` (`--run --from d77f001`) ran 2026-10-04
  23:04:54Z–23:08:57Z, after the declaring commit `8601c1b` (22:58:01Z) and on the committed code
  `d77f001` (23:04:47Z), with a clean tracked tree; report sha256
  `e2c7a2a82a0543a106ce4129dc8ca1a279bc77d5dd723f83a9ac28ef29bc9e1d`, outcome `C1M-DEBUG`. It ran
  every stage with the declared stand-ins and was used only to see that every stage completes.
  Its log printed arm counts on its 6 resets; nothing in it was read for the row, and no code,
  bar or constant changed between it and the record run (both are `d77f001`).
- **One post-run code edit, no behaviour change.** After the run, the list of C1's code files
  that the runner hash-checks moved from `scripts/run_c1m_feasibility.py` into
  `lewm_next_c1m.py` (`C1_CODE_FILES`, `C1_REFERENCE`), because the repository's audit test
  (`tests/test_no_runner_imports.py`) reads a script's file name in a runner as a load of that
  script. The values are unchanged; the record is the run at `d77f001`, and a re-run at the later
  revision would check the same three files against the same `0c64ec2`.
- **A field to read with care.** The report's `early_verdict` (`M-NO-TAU`) is the ladder evaluated
  with M-F1 to M-F3 only, before τ_commit existed; the runner used it only to decide the early
  stop, whose rows (M-INFEASIBLE, M-TWINS-NONE, M-TWINS-ESCALATE) it did not reach. It is not a
  row of this record; `verdict` is.

### 2.2 M-F1: ρ\* and the ceiling: **ρ\* = 4 cm (disc)**

H-final(commit) on M1's 32 resets under the disc move:

| radius | count /32 | refused or blocked | failures | median landing miss (cm) | median offset (cm) |
|---|---|---|---|---|---|
| 3 cm | **30** | 0 | 63019 (latched, ended 4.0 cm from the plate, not at rest), 63031 | 0.092 | 1.89 |
| 4 cm | **30** | 0 | the same two | 0.103 | 2.52 |
| 5 cm | **29** (fails) | 0 | 63019, 63023 (ended 4.46 cm away, not at rest), 63031 | 0.087 | 3.15 |

- 6 cm was not run (the loop stops at the first failing radius, §1.4), and the −y half-disc was
  not needed. **ρ\* = 4 cm, disc**, by one reset: 5 cm missed the bar by 1/32.
- **Seed 63031 fails at every radius for a reason unrelated to the move:** a guard refusal at step
  238, during P-3's pick, before the step-300 move (its plate was never moved). So the M-F1
  ceilings are 30/31 and 29/31 among resets that reached the move.
- No draw needed a re-draw; no move was blocked; every attempt that reached step 300 had a latched
  grasp before it (31 of 32 at each radius). All look-aheads aimed within 0.14 cm of plate(525);
  committed aims at a = 0.347–0.370 (the fixed point is a = 1/3).

### 2.3 M-F2: motion, r and reach at 4 cm: **PASS**

| quantity | value | bar |
|---|---|---|
| median \|plate(525) − plate(405)\| (31 unrefused attempts) | **6.85 cm** (range 4.90–8.56) | ≥ 2 cm: passes |
| r | **465** (28/32 within 0.1 cm of plate(525) at 465; 24 at 460; 31 from 468) | recorded |
| horizon r − 405 | 60 steps (C1: 55) | recorded |
| median palm speed at 405 | 0.0031 cm per step | ≤ 0.5: no extra commands |
| m2 at 405 | median 0.0025 cm, max 0.019 cm | reported |
| a_lo | **−0.5**: at every level and every b, 31/32 transfers complete (63031 never reaches 405) | defined: passes |

### 2.4 M-F3: the twins lose at 4 cm: **PASS** (strict reading)

On F3's 32 fresh resets, paired against the fresh ceiling; a_N = 0.5; p̄ = (0.49, −0.09).

| arm | count /32 | headroom (ceiling − arm) | 95 % interval | discordant (ceiling only, arm only) | McNemar feasibility (≥ 0.8) | median landing miss (cm) | result |
|---|---|---|---|---|---|---|---|
| **H-final(commit)** (fresh ceiling) | **32** | — | — | — | — | 0.094 | ≥ 30: passes |
| H-now | 0 | **+32** | [+32, +32] | 32, 0 | — | 9.10 | passes |
| N-proxy | 8 | **+24** | [+19, +28] | 24, 0 | — | 4.17 | passes |
| shuf-proxy | 8 | **+24** | [+19, +28] | 24, 0 | **1.000** (p 3.6 × 10⁻¹⁵ doubled) | 4.46 | passes |
| **mean-proxy** | **12** | **+20** | **[+15, +25]** | 20, 0 | **1.000** (p 9.1 × 10⁻¹³ doubled) | 2.74 | passes |

- **Every lower bound is at or above +8**, so M-F3 passes under the strict reading (R15.7). No
  reversed pair occurred in any comparison, so R16.7's widening did not arise.
- **It passed where R15 projected an escalation.** R15 §3.3 projected the clipped mean twin at
  16.3/32 at 4 cm (strict margin +7.7, marginal) and the shuf twin at 12.1; the measured
  proxies scored 12 and 8, and the fresh ceiling 32 rather than the 30 the projection assumed. A
  likely contributor, *not tested here*: the projection used R15 §3.2's development tolerance curve
  (about 85 % up to 2 cm), and the τ_commit measured below is much sharper (15/32 at 1.5 cm). The
  offsets on F3 had a median of 3.11 cm (max 3.98 cm). **R16.6's realistic early stop did not
  happen**, by a clear margin on the twins and by one reset on ρ\*.
- The proxies are privileged calculations, not trained twins (as in C1).

### 2.5 M-F4: τ_commit = **1.0 cm**

H-final(commit)'s aim plus a planted error, on T's 32 resets at 4 cm (median offset 2.94 cm):

| planted error (cm) | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| count /32 | 32 | 29 | 30 | 15 | 13 | 9 |
| median landing miss (cm) | 0.08 | 0.76 | 1.50 | 2.25 | 2.99 | 4.48 |

τ_commit is the largest level up to which every level reaches 28/32: **1.0 cm** (the counts are not
monotone at 0.5–1.0, 29 then 30; 1.5 cm falls to 15/32). This is far sharper than R15 §3.2's
development curve, which pooled C1's arms without a move. The landing miss is larger than the
planted error because the plate follows the palm (κ = −0.5): a planted error of e moves the
landing by about 1.5·e.

### 2.6 M-F5a: the readout offline: plate-hidden **PASS**; both learning curves **still falling**

The corpus: 1 024 roots (63300–64323) at 4 cm, 1 018 rows. Six roots never reached 405 (guard
refusals during P-3's pick at steps 223–243: 63521, 63604, 63659, 63952, 64132, 64140); 0
fallbacks; 133/1 024 counted successes (aims uniform over the box; not read). The plate moved a
median of 8.55 cm between 405 and r ([8.44, 8.64]). Featurisation took 68 s on the CPU.

| readout at r = 465 (cross-fitted, 5 folds by root) | median error (cm) | 95 % interval | 87.5th pct | bar | result |
|---|---|---|---|---|---|
| pooled 4 × 4 | 0.650 | [0.622, 0.676] | 1.23 | reported | — |
| **pooled 4 × 4, plate-hidden renders** | 6.12 | **[5.82, 6.35]** | 10.07 | lower bound > τ_commit = 1.0 cm | **passes** |
| pooled 8 × 8 | 0.478 | [0.455, 0.511] | 0.96 | reported | — |
| pooled 8 × 8, plate-hidden renders | 4.31 | [4.12, 4.44] | 6.78 | lower bound > 1.0 cm (admits 8 × 8) | passes |
| R-plate (full tokens) at 405 (reported) | 0.205 | [0.196, 0.214] | 0.37 | — | — |
| constant prior at r (reported) | 2.66 | [2.55, 2.82] | 4.29 | — | — |
| H-sysid's plate(r), true / reading (reported) | 0.474 / 0.526 | [0.457, 0.499] / [0.504, 0.557] | 0.90 / 0.96 | — | — |

λ (relative) was 0.01 in every fold on both grids at r (the grid runs from 0.001), and 0.1 in four
folds and 0.01 in one for R-plate at 405.

**The learning curve** (nested, the same rows for both grids; fit roots per fold 204, 407–408,
611, 814–815):

| fraction of fit roots | 1/4 | 1/2 | 3/4 | all | guard: median(e_3/4) − median(e_all) | still falling? |
|---|---|---|---|---|---|---|
| 4 × 4 median (cm) | 1.027 | 0.785 | 0.717 | 0.650 | +0.067 [+0.037, +0.095] | **yes** |
| 8 × 8 median (cm) | 0.813 | 0.616 | 0.531 | 0.478 | +0.053 [+0.027, +0.074] | **yes** |

Both readouts are still data-limited at 1 024 roots (TASK-075 §7's guard). This does not change the
row (8 × 8 passed M-F5b), but it says a protocol's Stage O on a larger corpus may read better.

### 2.7 M-F5b: H-read: 4 × 4 **misses**, 8 × 8 **meets the allowance exactly**

On F3 + R (64 resets, 63100–63163), paired against H-final(commit), which scored **64/64** (32/32
on F3, M-F3's attempts, and 32/32 on R):

| latent | H-read count /64 | ceiling − H-read | 95 % interval | bar ≤ 4/64 | per-iteration reading error in the clone (median, cm) | converged (≤ 0.25 cm) | median landing miss (cm) |
|---|---|---|---|---|---|---|---|
| pooled 4 × 4 | 58 | **+6** | [+2, +11] | **misses** | 0.647 [0.610, 0.683] (581 readings) | 18/64 | 0.99 |
| **pooled 8 × 8** | **60** | **+4** | **[+1, +8]** | **passes (at the bar)** | 0.421 [0.391, 0.445] (573 readings) | 21/64 | 0.63 |

- **8 × 8 passes as a point reading exactly at the allowance**, with an interval of [+1, +8] that
  reaches the whole proposed δ (8/64). As R15.6 stated in advance, a pass this close to the bar is
  partly chance; one more failure would have made the row M-NO-BAR.
- 8 × 8 ran because 4 × 4 missed and its plate-hidden check passed (§1.7). 4 × 4's miss is not
  detectably beyond δ (lower bound +2 ≤ 8).
- **The fixed point rarely converged.** The per-iteration reading error (0.42–0.65 cm) is above the
  0.25 cm stopping tolerance, so most H-read attempts used all 10 iterations and aimed at the last
  iterate (8 × 8: 46 of 64 attempts used all 10 iterations, 43 of them without converging; 4 × 4: 47 and 46). H-read is therefore "the real readout with
  perfect dynamics, iterated at most 10 times", and its count includes that iteration noise. A
  protocol's W uses the note's controller form (grid, then refinement), not this loop.
- H-read is privileged (it holds the robot and rolls the clone) and is not W: it reads the
  encoded *true* frame at r. Whether a predicted 8 × 8 latent reads as well is the protocol's
  question.

### 2.8 M-F6 and M-F7 (reported)

On F3's 32 resets, paired against the fresh ceiling (32/32):

| arm | plate input | count /32 | median landing miss (cm) | ceiling − arm, interval |
|---|---|---|---|---|
| H-rule | R-plate reading at 405 | **30** | 0.23 | +2, [0, +5] |
| H-sysid | R-plate reading at 405 | **30** | 0.62 | +2, [0, +5] |
| H-rule-true | true plate | 31 | 0.08 | +1, [0, +3] |
| H-sysid-true | true plate | 31 | 0.65 | +1, [0, +3] |
| H-rule-stale | P-3's post-look (pre-move) estimate | 17 | 3.13 | +15, [+10, +21] |
| H-now-reaim (not a comparator; re-aims 405–485) | true plate | 32 | — | 0, [0, 0] |

- The closed-loop R-plate reading at 405 erred by a median 0.191 cm ([0.172, 0.258], 64 readings).
- **The non-inferiority comparator** is the better of H-rule and H-sysid on the reading: they tie
  at 30/32. Both sit within noise of the ceiling, so "LeWM needed" is not expected in a protocol
  (stated in advance by R15 §5.7).
- **H-rule-stale** (the rule with the pre-move plate) scored 17/32, against R15 §5.1's
  projection of about 18.9/32 at 4 cm (*estimate*): a twin that reads the plate before the move
  loses clearly.
- **M-F7, cost: PASS.** 224 H-final(commit) and H-read attempts (M-F1 at ρ\*, F3, R, both H-read
  grids): median 8.19 s, **9.72 s at most** (bar ≤ 60 s).

### 2.9 The row: **M-PROCEED**

| check | bar | result |
|---|---|---|
| M-F1 | largest radius with every smaller one ≥ 30/32 | **ρ\* = 4 cm** (disc: 30, 30, 29 at 3, 4, 5 cm) |
| M-F2 | remaining ≥ 2 cm; r and a_lo defined | 6.85 cm, r = 465, a_lo = −0.5: pass |
| M-F3 | fresh ceiling ≥ 30/32; each lower bound ≥ +8/32; feasibility ≥ 0.8 | ceiling 32/32; lower bounds +32, +19, +19, +15; feasibility 1.000 and 1.000: pass |
| M-F4 | τ_commit defined | **1.0 cm**: pass |
| M-F5a | 4 × 4 plate-hidden lower bound > τ_commit | 5.82 cm: pass (8 × 8: 4.12 cm) |
| M-F5b | ceiling − H-read ≤ 4/64 | 4 × 4 +6 (miss); **8 × 8 +4, [+1, +8]: pass at the bar** |
| M-F6 | none | H-rule 30, H-sysid 30 on the reading; H-rule-stale 17 |
| M-F7 | ≤ 60 s | 9.72 s: pass |

By R15.8's ladder (§1.8), **the row is M-PROCEED, and no clause fires.** What it admits, and only
that: a preregistration may be drafted under R9.8 and R9.9, with its own K0 (τ_commit and the
ceiling re-measured on fresh seeds), Stage O (c_plate at r **on the 8 × 8 grid**), a
`check_budget`-passing training block, the offline horizon and dynamics gates at h = r − 405 = 60
**on 8 × 8** (R15.6: choosing 8 × 8 commits the protocol to gating its own dynamics there, with
no reach-back over TASK-066's 4 × 4 gates), a 16-reset development closed loop with a stop rule,
and an independent review. No protocol is drafted here, no GPU is used, and no task number is
assigned (R15.1). The canonical status sentence is unchanged (R16.12).

### 2.10 Caveats

- **One run, development smokes.** 32 resets for each of M-F1, M-F3 and M-F4, 64 for M-F5b, 1 018
  corpus rows. ρ\* was set by one reset (5 cm: 29/32), and the 8 × 8 H-read passed exactly at its
  allowance; both are close calls.
- **The readout allowance is an allocation, not a calibrated bar** (R15.6): 4/64 is half of the
  design note's proposed δ = 8/64, which is itself not frozen; the 8 × 8 interval reaches all of δ.
- **The proxies are privileged calculations**, not trained twins; the real L-mean, L-shuf and N are
  measured only in a protocol. The ceiling stands in for W in the feasibility calculation
  (optimistic for the test).
- **H-read is not W.** It reads the encoded true frame at r from a perfect roll-out. W must keep
  the plate through a 60-step predicted roll-out of an 8 × 8 latent whose dynamics no task has
  gated (TASK-066 gated 4 × 4 at h = 8 and 16 on a static apple). The horizon remains the main
  risk (R15 §5.8), and an 8 × 8 latent has four times the tokens, with an unmeasured GPU cost
  (R15 §5.6).
- **H-read's fixed point rarely converged** (21/64 on 8 × 8), so its count includes iteration
  noise, and its g0 is the true plate (a disclosed privileged start).
- **Both learning curves were still falling** at 1 024 roots; a larger corpus may read better.
- **τ_commit = 1.0 cm with non-monotone counts** (29/32 at 0.5 cm, 30/32 at 1.0 cm) and a sharp
  drop at 1.5 cm (15/32); a protocol's K0 re-measures it.
- **R15's projection (Stage −1) underestimated the twins' loss at 4 cm** (projected mean twin
  16.3/32, measured mean-proxy 12/32), probably through its development tolerance curve; this is
  an interpretation, not tested here.
- **Seed 63031 and six corpus roots failed in P-3's pick** (guard refusals at steps 223–243),
  before the move; they count as failures (63031) or are excluded (corpus), as declared.
- No world model ran, and nothing here is a learned or LeWM result. Simulation only.
