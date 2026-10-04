# C1 feasibility record: a single aim committed at 405 under the reactive-plate rule (development)

**What this is.** The development feasibility record that the DRAFT design note
[apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) (§4.1, §7 step 2; R9.6) requires
before any protocol for its candidate C1. It runs the note's checks C1-F1 to C1-F6 on the CPU,
with **no world model**: every arm below is either a privileged calculation from simulator truth
(H-final, the proxies, the reach and corpus collectors) or a hand-written non-world-model arm
(H-rule, H-sysid). Nothing here is a protocol, a gate of a learned arm or a learned result.

**Outcome: C1-TWINS-ESCALATE** (§2). C1-F3 fails on the mean-proxy (L-mean's stand-in): its
headroom below the ceiling is +6/32 (bar +8/32; interval [+1, +11], so *near* under R8.14's
guard) and its predicted McNemar feasibility is 0.728 (bar 0.8). By the note's row C1 escalates
before any protocol, with no clause; none of R9.10's remedies is chosen here. C1-F5's readout at
r also misses its estimate (median 0.886 cm against 0.5 cm), which on its own would be NO-BAR.
C1-F1, C1-F2 and C1-F6 pass, and C1-F5's plate-hidden check passes. **No world model ran; no arm
here is a learned or LeWM result.**

§1 (the declarations) was written and committed (`2370ab3`, pushed) before the run, and is kept
as written. The run is `outputs/c1-run-2` at `2370ab3` (§2.1).

Every choice below is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R14.1–R14.10 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-04 (c) (R14.1–R14.8 before
the run, R14.9–R14.10 after it). R9 and R13 (and every
lower number) are taken; a search of every ref at `82da722` found no R14.

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

### 1.1 Seeds and salts (R14.1)

The design note names no block ("a newly declared smoke block that is checked against every range
on main"). The block is declared here.

| range | seeds | use |
|---|---|---|
| block | 57000–57999 | C1's feasibility record only |
| F | 57000–57031 | the 32 paired closed-loop resets of C1-F1, F2, F3, F4 and F6 (every arm runs on the same 32 resets in the same order) |
| corpus | 57100–57355 | the 256 roots of the declared-aim smoke corpus (C1-F5 and H-sysid) |
| debug | 57900–57999 | mechanics debugging of the runner only; nothing in it is read |
| reserved | everything else in 57000–57999 | unused; any use needs its own ruling |

| salt | use |
|---|---|
| 7701 | the corpus's uniform (a, b) draw per root: `default_rng(SeedSequence([7701, seed]))` |
| 7702 | C1-F5's 5 outer folds by root |
| 7703 | λ selection (5-fold inner CV grouped by root) in every ridge |
| 7704 | every bootstrap interval (10 000 resamples) and the McNemar feasibility resampling |

**The search** (2026-10-04, at `82da722`): `git grep` over all 68 local and remote refs (src,
scripts, tests, configs, benchmarks, docs, `.mc`, `PLAN.md`, and every other path), plus the
working files of every other worktree under `/home/huhn/develop/emai/worktrees/`. In 57000–57999
the only matches are `"global_wall_seconds": 57600.0` (TASK-066's manifests) and a byte count
(`57304`) in `uv.lock`; neither is a seed. 7701–7709 appear nowhere in src, scripts, tests or
configs on any ref. The block lies outside every forbidden range of TASK-076
(`plate_twin_v2.FORBIDDEN_RANGES`) and outside TASK-076's own block 56000–56999; the salts are
distinct from every earlier task's salt. `lewm_next_c1.check_seed_ranges` checks this, and the
tests pin it.

**Resets.** `wm_critic_v2.wide_reset_values(seed)`: v2's own reset (plate (0.49, −0.09) ± 2 cm,
apple (0.34, −0.18) ± 3 cm, uniform per axis), as the note requires ("v2 with its own reset,
unchanged"). Not TASK-074's `condition_reset`, whose re-draw rule exists for the step-300 shift
that C1 does not have. No jitter knob, no shift.

### 1.2 The condition and the code (R14.2)

- **The rule** is TASK-076's cell A as implemented in `plate_twin_v2_runtime.CellMotion`, unchanged:
  κ = −0.5, L = 2, s0 = 405, s1 = 525, plate(t) = plate(405) + κ·(palm(max(t − 2, 405)) −
  palm(405)), a refused plate move ends the attempt as a counted failure.
- **One controller change**, as the note declares: a single aim committed at 405
  (`lewm_next_c1_runtime.CommitController`, TASK-076's `AimController` with one decision). e9's
  primitive then runs the transfer, lower and open with no re-aim. P-3 picks before 405.
- **P-3's post-look estimates** come from G-repro (TASK-072 run-1's readout, refitted exactly,
  from the restored evidence root `/home/huhn/develop/emai/worktrees/task076-evidence`), not from
  the truth stand-in that TASK-076's Stage-0 smokes had to use.
- **TASK-076's code is reused unchanged**: the worker, the hook, `AimController`, `LookaheadAim`
  (H-final), `RuleAim`'s fixed point (H-rule), `TwinAim`'s reading, the harness's G-repro, render
  majority and pins. The runner checks TASK-076's manifest pins at the start and the end.
- **New code** (behind tests, imports lazy): `src/embodied_jepa/lewm_next_c1.py` (constants,
  geometry, statistics, the check rules), `src/embodied_jepa/lewm_next_c1_runtime.py` (the commit
  controller, the aims, the hook subclass that keeps the plate path and captures frames) and
  `scripts/run_c1_feasibility.py` (one invocation, every stage in order).
- **r and a_lo are set by their rules in code**, inside the one invocation, in the order the note
  needs them (ceiling, reach, proxies, corpus, offline, comparators). No constant is set by hand
  between stages, and none is changed after a result.
- **H-final(commit)** is TASK-076's `LookaheadAim` called once, at 405: g0 = the true plate;
  g_{k+1} = the plate at s1 after the place primitive aimed at g_k runs in cloned state with the
  rule active; stop at |g_{k+1} − g_k| ≤ τ_re/4 = 0.25 cm (τ_re = 1.0 cm, K0, R8.19), at most 10
  iterations. It is the privileged ceiling and is not clipped to the box.
- **GPU: none.** Every stage runs on the CPU, the DINOv2 featurisation included (the note: "the
  smokes need no GPU"; the closed-loop readouts encode on the CPU anyway). No GPU lock is taken and
  no CUDA context is created.
- **The debug run** (57900–57999) exercises every stage at small sizes; where its six resets
  cannot define r or a_lo it uses fixed stand-ins (485, −0.2). It is mechanics only and is not
  read.

### 1.3 The box, its coordinates and the clip (from the note)

At 405, with p the plate (the true plate for a privileged arm, the reading for a non-privileged
one) and h the proprioceptive right-palm xy: g = p + a·(h − p) + b·n, n the unit vector
perpendicular to h − p (h − p rotated by +90°). The box is a ∈ [a_lo, 0.5], b ∈ [−3, +3] cm; the
grid steps are 0.05 in a and 1 cm in b. The fixed point (p − κh)/(1 − κ) lies at a = 1/3, b = 0.

### 1.4 C1-F1: remaining motion, r and the palm speed (R14.3)

On H-final(commit) on F:
- **Remaining motion:** the median, over attempts not refused, of |plate(525) − plate(405)|.
  Bar: ≥ 2 cm.
- **r:** the earliest step t in 405–525 at which |plate(t) − plate(525)| ≤ 0.1 cm on at least
  87.5 % of the 32 attempts (28 of 32); a refused attempt is never within. Recorded, no bar.
- **Palm speed at 405:** ‖palm(405) − palm(404)‖ per step, median over attempts. Recorded; if it
  exceeds 0.5 cm per step, the protocol gives W and N the last 2 executed commands (the note's
  declared consequence, not a failure). m2 at 405 = |κ|·‖palm(405) − palm(403)‖ is reported.
- **Failure row:** C1 is infeasible; escalate, no clause.

### 1.5 C1-F2: the ceiling and reach (R14.4)

- **Ceiling:** H-final(commit)'s counted successes on F, a refused move counted as a failure.
  Bars: ≥ 30/32, and refusals ≤ 8/32.
- **Reach check, for a_lo.** For each a in {−0.5, −0.4, −0.3, −0.2} and each b in {−3, 0, +3} cm,
  e9's aim is committed at 405 to g = p + a·(h − p) + b·n from the true p and h, on F's 32 resets
  (12 × 32 attempts). A transfer is **complete** when the aim is accepted without falling back
  (its release pose is reachable) and at least 505 commands execute (the transfer 405–504 runs to
  its end) with no refusal of any kind. A reset is reachable at a level only if all three b
  complete; a level is reachable if at least 31/32 resets are. **a_lo** is the most negative level
  that is reachable together with every level above it (the τ_re rule's monotone form; R14.4
  makes "the most negative of … whose candidates are reachable" monotone). If −0.2 is not
  reachable, a_lo is undefined and C1-F2 fails.
- **Failure row:** C1 is infeasible; escalate, no clause.

### 1.6 C1-F3: the twins lose (R14.5)

Privileged proxies on F's 32 resets, each a single commit at 405 from the true p and h, every aim
clipped to the box (the note: every arm that chooses from candidates uses the same box and clip):
- **H-now:** the true plate at 405 (static, so H-cv is the same).
- **N-proxy:** g = p + a_N·(h − p), b = 0, with a_N = min(0.5·(1 − a_m), 0.5) and
  a_m = (a_lo + 0.5)/2.
- **Shuf-proxy:** the note's formula, (p′ − κh)/(1 − κ), with p′ the true plate of reset
  (i + 1) mod 32 in F's seed order and h this reset's palm.
- **Mean-proxy:** the note's formula, (p̄ − κh)/(1 − κ), with p̄ = (0.49, −0.09), the reset
  distribution's mean plate.

**The rule.** For each proxy, the paired difference ceiling − proxy (same resets) must be at
least +8/32 (K-P2's form). A failed one is classified by R8.14's guard: *detectably below* when
the upper bound of its paired, reset-clustered 95 % bootstrap interval (salt 7704) is below 8/32,
*near* otherwise; both escalate without a clause here, so the guard labels the failure and
changes no action. For the shuf- and mean-proxy, the McNemar feasibility for the 64-reset test
(R8.12's form: the fraction of 10 000 resamples of 64 pairs drawn with replacement from the 32,
salt 7704, whose exact one-sided McNemar p for ceiling against proxy is below 0.01) must be at
least 0.8. The point p on the doubled discordant counts is reported beside it. The ceiling
stands in for W here, which is optimistic for the test, as H-handover was in R8.12.

**Reported only, never a gate:** the offset-consistent variants of the two scene-blind proxies,
g = g_final + (p′ − p)/(1 − κ) and g = g_final + (p̄ − p)/(1 − κ), where g_final is
H-final(commit)'s own committed aim on the same reset. The note's formulae assume the palm ends
at the aim; a perfect W rolled out from a foreign or mean start would include e9's release offset
and land at exactly |Δp| or |p − p̄| from this reset's fixed point. If the formula proxies score
lower than these variants, the gating proxies are optimistic for C1 (they make the headroom
larger), and that is stated beside the result.

**Failure row:** escalate before any protocol, no clause; for L-mean the remedies of R9.10 are
the options, and none is chosen.

### 1.7 C1-F4: the best non-world-model arm (reported, no bar; R14.7)

On F's 32 resets, single commit at 405, each aim clipped to the box around its own plate input:
- **H-rule:** TASK-076's `RuleAim` fixed point, g = p̂ + κ·(palm_g(523) − palm(405)), palm_g
  from the kinematic stand-in's place-primitive path under g, iterated to 0.25 cm (at most 10),
  from the smoke R-plate's reading p̂ of the onboard frame at 405 (CPU, batch 1).
- **H-sysid:** a least-squares regression plate(r) ≈ c + A·p + B·h + C·g on the smoke corpus,
  inverted with W's controller form: score every grid candidate by |prediction − g|, take the
  best, then refine g ← clip(prediction(g)) at most 10 times, stopping at 0.25 cm. The reading
  version is fitted on the corpus's out-of-fold R-plate readings at 405 and fed the closed-loop
  reading; the true version is fitted and fed with the true plate.
- **H-rule-true, H-sysid-true:** their true-plate versions (their ceilings).
- **H-now-reaim:** TASK-076's H-now unchanged (re-aims at the current true plate at 405, 421, …,
  485); reported only, not a comparator.
- **The smoke R-plate** is TASK-076's R-plate form (dual ridge on the full pretrained DINOv2
  tokens, λ by inner CV, salt 7703), fitted on every corpus root's onboard frame at 405.

### 1.8 C1-F5: the readout at r (R14.6)

- **The smoke corpus.** 256 roots (57100–57355). P-3 picks; at 405 e9's aim is committed to
  (a, b) drawn uniformly over the box (a ∈ [a_lo, 0.5], b ∈ [−3, +3] cm, salt 7701) from the true
  p and h (the note's privileged scripted collector). At 405 and at r the onboard 112 px
  observation is kept, and at r also a plate-hidden render of the same state (TASK-075's 112 px
  renderer with the plate's geoms in a hidden group, as TASK-076's O4 frames). A root that does
  not reach r is excluded and counted.
- **R-plate-pool at r:** a dual ridge from the pooled 4 × 4 pretrained DINOv2 latent (6 144-d) of
  the frame at r to the true plate at r, cross-fitted over 5 outer folds by root (salt 7702), λ by
  inner CV (salt 7703). The same fold's readout reads the held-out plate-hidden renders at r.
- **Bars.** At r: the **point median** of the cross-fitted error ≤ τ_re/2 = 0.5 cm (the note
  writes "median error"; c_plate's upper-bound form belongs to the protocol's Stage O, and the
  upper bound is reported here). Plate-hidden at r: the **lower 95 % bound** of the median error
  (bootstrap over roots, salt 7704) > τ_re = 1.0 cm (O4's form).
- **Reported:** R-plate (full tokens) at 405, cross-fitted on the same folds (it bounds H-rule and
  H-sysid); a constant prior at r (the fit folds' mean plate at r); H-sysid's cross-fitted
  residuals; the plate's motion from 405 to r.
- **Failure rows:** plate-hidden check fails → the readout at r may be keyed on the arm; escalate,
  no clause. Readout at r misses its estimate → no B is feasible (NO-BAR); escalate, no clause.

### 1.9 C1-F6: cost

The wall time of each H-final(commit) attempt (≤ 10 look-ahead roll-outs at the single decision).
Bar: ≤ 60 s. Above it, the per-attempt cap is reviewed (TASK-076's rule); it is not a stop.

### 1.10 The verdict (R14.8)

The note's failure rows, in its order (`lewm_next_c1.verdict`):
1. C1-F1 or C1-F2 fails → **C1-INFEASIBLE**: escalate, no clause.
2. C1-F3 fails → **C1-TWINS-ESCALATE**: escalate before any protocol, no clause.
3. C1-F5's plate-hidden check fails → **C1-ARM-KEYED**: escalate, no clause.
4. C1-F5's readout at r misses → **C1-NO-BAR**: escalate, no clause.
5. Otherwise → **C1-PROCEED**: a preregistration may be drafted (R9.6), with its own K0
   (τ_commit and the ceiling), Stage O and an independent review.

No row fires a clause, and none closes the LeWM backend, v2 or the product goal. C1-F4 has no
bar, and C1-F6 only reviews the cap. Every check is computed and reported whatever an earlier one
shows, except where a_lo or r is undefined (then the stages that need them cannot run).

## 2. Results (written after the run)

### 2.1 The run and its provenance

- **The record run:** `scripts/run_c1_feasibility.py --output outputs/c1-run-2 --evidence
  /home/huhn/develop/emai/worktrees/task076-evidence`, in the worktree
  `/home/huhn/develop/emai/worktrees/c1-feasibility` (made with `scripts/new_worktree.sh --run`
  from `origin/main` at `82da722`, branch `dev/c1-feasibility`).
  - Code revision `2370ab37dcef35a644c6676b1dd7dd7dfae11d43`, `tracked_tree_dirty: false` at the
    start and the end; TASK-076's manifest pins matched at the start and the end.
  - 2026-10-04 20:39:10Z to 20:54:21Z (910 s); 6 CPU workers, 1 torch thread each; peak
    process-tree PSS 9.69 GiB (ceiling 12 GiB); no GPU, no GPU lock, no CUDA context.
  - G-repro: all 8 of TASK-072 run-1's reproduction checks passed (39 s).
  - Outcome: **C1-TWINS-ESCALATE**.
- **Files** (all in `outputs/c1-run-2/`, git-ignored, on the Linux PC):

  | file | sha256 |
  |---|---|
  | `report.json` | `7779709cf62ee61fac3e2acfbe16b2049479d02f8dc0af47fd809ccb993fcef6` |
  | `corpus.npz` (255 rows) | `93e96c3f82b476e2a5096d22d6eb09e04bfa0323ecfdd26b7a25885e3275cdb5` |
  | `errors.npz` (C1-F5's per-root errors) | `d81eb3e9bbe385405b0ffdb25c7ad9af8ccb13a84f41d230c72070f2e411ef1a` |
  | `r_plate.npz` (the smoke R-plate; content sha256 `893b52d5…f00c`) | `4279e232354915b58b3f9a1bf5223294dead3471962517cf8b964271ab52fcb9` |

- **The voided first run (R14.9).** `outputs/c1-run-1` (report sha256
  `f1b470eb…41aa8`), at the same revision, ran every stage and then ended **V**: "G-hash: the
  tracked tree changed during the run". The cause was the operator (Claude): during the run it edited a tracked
  file in the worktree (the TASK-076 card's merge sha, `.mc/tasks/done/`), not code. The edit was
  stashed and the run repeated once, from scratch, at the same revision on a clean tree
  (run-2). Every reported value of run-2 equals run-1's. The only differences are the renderer's
  documented one-level pixel differences: two plate-hidden renders (roots 79 and 114 of the
  corpus, 12 pixels, one level each; their errors moved by at most 0.00024 cm, no reported
  statistic changed) and run-1's G-repro re-render disagreement on seed 51235 (settled by the
  render majority; none in run-2). The npz files differ in their sha256 for that reason and
  because `np.savez` stamps the zip entries. Run-2 is the record.
- **Seeds simulated:** F 57000–57031 (every arm), corpus 57100–57355, and the debug run
  `outputs/c1-debug-1` on 57900–57905 and 57910–57939 (mechanics only, not read). No other seed.

### 2.2 C1-F1: remaining motion, r and the palm speed: **PASS**

On H-final(commit), 32 attempts, none refused:

| quantity | value | bar |
|---|---|---|
| median \|plate(525) − plate(405)\| | **6.72 cm** (range 5.49–8.23) | ≥ 2 cm: passes |
| r | **460** (28/32 within 0.1 cm of plate(525) at 460; 26/32 at 459; 32/32 from 465) | recorded |
| horizon r − 405 | 55 steps (the note's estimate: ≤ 80) | recorded |
| median palm speed at 405 | 0.0030 cm per step (max 0.016) | ≤ 0.5: W and N need no extra commands |
| m2 at 405 = \|κ\|·‖palm(405) − palm(403)‖ | median 0.0022 cm, max 0.013 cm | reported |

### 2.3 C1-F2: the ceiling and reach: **PASS**

- **Ceiling H-final(commit): 31/32**, refused 0/32 (bars ≥ 30 and ≤ 8). The one failure, seed
  57014, completed with a latched grasp and ended 4.14 cm from the plate, not at rest. All 32
  look-aheads converged (at most 7 iterations). H-final's committed aims lie at a = 0.348–0.359
  (the rule's fixed point is a = 1/3), and its aim landed within 0.124 cm of plate(525)
  (median 0.086 cm).
- **Reach: a_lo = −0.5.** At every level (−0.5, −0.4, −0.3, −0.2) and every b (−3, 0, +3 cm),
  32/32 transfers completed: no fallback, every attempt ran to e9's end (`policy_complete`). Those
  attempts' counted successes are not part of the check (they aim behind the moving plate and
  fail, as expected).

### 2.4 C1-F3: the twins lose: **FAIL** (the mean-proxy)

Paired against the ceiling's 31/32 on the same resets; intervals are the paired,
reset-clustered 95 % bootstrap (salt 7704); feasibility is R8.12's form for the 64-reset test.

| proxy | count /32 | headroom (ceiling − proxy) | 95 % interval | discordant (ceiling only, proxy only) | McNemar feasibility (≥ 0.8) | median landing miss (cm) | the note's estimate | result |
|---|---|---|---|---|---|---|---|---|
| H-now | 0 | **+31** | [+29, +32] | 31, 0 | — | 9.31 | 0–5/32; 8–12 cm | passes |
| N-proxy (a_N = 0.5) | 12 | **+19** | [+13, +25] | 20, 1 | — | 4.16 | 5–8/32; 4–6 cm | passes |
| shuf-proxy | 21 | **+10** | [+4, +16] | 11, 1 | **0.985** (p 1.8 × 10⁻⁵ doubled) | 1.90 | ~19.9/32; 2.05 cm | passes |
| **mean-proxy** | **25** | **+6** | **[+1, +11]** | 7, 1 | **0.728** (p 0.0021 doubled) | 1.65 | ~22.6/32; 1.60 cm | **fails both** |

- **The mean-proxy fails both of its bars.** Its headroom +6/32 is below +8/32, and its interval's
  upper bound (+11) is above 8, so under R8.14's guard it is *near*, not detectably below; its
  feasibility 0.728 is below 0.8. The note stated this risk in advance ("L-mean makes C1's
  feasibility marginal … about 7–9/32 … at or below C1-F3's +8/32 bar").
- **The reported-only offset-consistent variants make it worse, not better** (R14.7, declared in
  §1.6): mean-proxy-offset 27/32 (headroom +4, interval [+1, +8]) and shuf-proxy-offset 20/32
  (headroom +11, interval [+6, +16]). So the gating mean-proxy formula was, if anything,
  optimistic for C1: a twin that includes e9's release offset lands closer still.

### 2.5 C1-F4: the best non-world-model arm (reported, no bar)

On F's 32 resets, single commit at 405 (H-now-reaim re-aims 405–485):

| arm | plate input | count /32 | median landing miss (cm) | paired vs ceiling (difference, interval) |
|---|---|---|---|---|
| H-rule | smoke R-plate reading | **30** | 0.23 | +1, [−2, +5] |
| H-sysid | smoke R-plate reading | **30** | 0.67 | +1, [−2, +5] |
| H-rule-true | true plate | 32 | 0.07 | −1, [−3, 0] |
| H-sysid-true | true plate | 31 | 0.66 | 0, [−3, +3] |
| H-now-reaim (reported only, not a comparator) | true plate, re-aimed | 31 | — (re-aimed; the first aim's miss is not its landing) | 0, [−3, +3] |

- The smoke R-plate's reading error at 405 in these closed-loop attempts: median 0.223 cm
  (95 % interval [0.187, 0.255], 64 readings of H-rule and H-sysid).
- **The non-inferiority comparator** would be the better of H-rule and H-sysid on the reading:
  they tie at 30/32. Both sit within noise of the ceiling, as the note expected; "LeWM needed"
  would not be expected in a protocol under C1 (stated in advance by the note).
- H-sysid's regression: cross-fitted plate(r) residual median 0.469 cm on the reading (0.353 cm
  with the true plate) on the corpus.

### 2.6 C1-F5: the readout at r = 460: plate-hidden **PASS**, readout **FAIL**

On the smoke corpus: 256 roots simulated, 255 rows (root 57150 ended at step 235 on a guard
refusal during P-3's pick and never reached 405; 0 fallbacks; 23/256 counted successes, which
is expected with aims spread uniformly over the box and is not read). The plate moved a median of
8.29 cm between 405 and r (interval [8.14, 8.46]).

| readout (cross-fitted, 5 folds by root) | median error (cm) | 95 % interval | 87.5th pct | bar | result |
|---|---|---|---|---|---|
| **R-plate-pool at r** (pooled 4 × 4) | **0.886** | [0.778, 0.993] | 1.52 | point median ≤ 0.5 cm | **fails** |
| **R-plate-pool on plate-hidden renders at r** | 3.865 | **[3.460, 4.285]** | 7.26 | lower bound > 1.0 cm | **passes** |
| R-plate (full tokens) at 405 (reported) | 0.236 | [0.219, 0.264] | 0.48 | — | — |
| constant prior at r (reported) | 1.771 | [1.623, 1.964] | 2.93 | — | — |

- λ (relative): 0.01 in every fold for R-plate-pool at r; 0.01 in three folds and 0.1 in two for
  R-plate at 405; 0.01 for the closed-loop smoke R-plate.
- **What the miss means.** The pooled latent of the true frame at r reads the moved plate to
  0.886 cm in median, above the note's estimate τ_re/2 = 0.5 cm, and its whole interval lies
  above 0.5 cm. A predicted latent can only be read less precisely, so by the note's row no B is
  feasible on this readout (NO-BAR). It still beats the constant prior and is not keyed on the
  arm (the plate-hidden check passes). This is a smoke estimate on 255 roots, not c_plate; the
  protocol's Stage O would measure c_plate itself, but the row applies to this estimate.

### 2.7 C1-F6: cost: **PASS**

H-final(commit) attempts took 8.2 s in median and **8.4 s at most** (bar ≤ 60 s); the
per-attempt cap is not reviewed.

### 2.8 Verdict: **C1-TWINS-ESCALATE** (escalate, no clause)

| check | bar | result |
|---|---|---|
| C1-F1 | remaining ≥ 2 cm | 6.72 cm: pass (r = 460; palm speed 0.003 cm per step) |
| C1-F2 | ceiling ≥ 30/32, refused ≤ 8/32, a_lo defined | 31/32, 0, a_lo = −0.5: pass |
| C1-F3 | each proxy ≤ ceiling − 8/32; scene-blind feasibility ≥ 0.8 | H-now, N, shuf pass; **mean-proxy +6/32 (near) and 0.728: fail** |
| C1-F4 | none | H-rule 30/32, H-sysid 30/32 on the reading |
| C1-F5 | readout at r ≤ 0.5 cm; plate-hidden lower bound > 1.0 cm | **0.886 cm: fail**; 3.460 cm: pass |
| C1-F6 | ≤ 60 s | 8.4 s: pass |

By the note's rows, in its order (§1.10): C1-F1 and C1-F2 pass, so C1 is feasible as a
condition; **C1-F3 fails, so C1 escalates before any protocol, with no clause** (R14.10). For
L-mean the note's remedies are the options (R9.10): a reset distribution with more plate spread,
declared as a task change under R2 by a declared rule; a larger gated cohort; or a different
commit step or rule under its own feasibility record. **None is chosen here.** Independently of
F3, C1-F5's readout miss is a second escalation (NO-BAR): a remedy for F3 alone would still meet
it, so any remedy must also answer the readout at r.

**No protocol is drafted, no GPU is used, and no LeWM task is admitted.** Nothing closes: not
the LeWM backend, not v2, not the product goal, and not C1's condition (no clause is attached to
any row of this record). The canonical status sentence is unchanged.

### 2.9 Caveats

- Development smokes: one run, 32 paired resets for F1–F4 and F6, 255 corpus rows for F5; the
  ceiling stands in for W in F3's feasibility (optimistic for the test).
- The proxies are privileged calculations, not trained twins; the real L-mean, L-shuf and N would
  be measured only in a protocol. The offset-consistent variants suggest the real L-mean would be
  at least as strong as its proxy.
- C1-F5 reads the encoded *true* frame at r; W would read a *predicted* latent after a 55-step
  roll-out, which can only add error.
- The τ curve and τ_re = 1.0 cm are TASK-076's (re-aimed, under TASK-074's condition); τ_commit
  was not measured, as the note says it belongs to a protocol's K0.
