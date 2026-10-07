# TASK-080 Stage 0: code, frozen block, debug smokes, scale probes and R18.13's dry run

**Development only.** No seed of K′, D, S or F was simulated. Every simulated attempt used the
debug range 65900–65999. The one non-debug run is R18.13's dry run, which reads only TASK-077's
existing train, val and old-gate featurisation (no simulation, no fresh root) and sets no bar.
No K0′ was run; K0′ needs a separate reviewer GO. The protocol
([apple_lewm_c1m_v2_pred_readout.md](apple_lewm_c1m_v2_pred_readout.md)) stays **DRAFT**. Every
choice below was **decided by Claude under owner delegation (2026-09-30)** (R18.15–R18.21 in
[DECISIONS.md](../DECISIONS.md)).

The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged. Nothing here runs a
LeWM controller in closed loop on a counted cohort; the debug closed loops ran on four debug
resets and are not read.

## 1. What Stage 0 added (§8 step 1)

All of it is in new modules. TASK-077's modules (`lewm_c1m_v2*`, hash-pinned by its frozen
manifest) are imported, never edited: the runner checks TASK-077's frozen block against its pin
`f28e5e2c…548d` and its 13 pinned files (G-frozen), and TASK-076's 84 pins.

| file | what |
|---|---|
| `src/embodied_jepa/lewm_pr_v2.py` | the frozen block (DRAFT): seeds 65000–65999 and the forbidden ranges including TASK-077's whole block 66000–68999; salts 8201–8212; the reused artifacts' sha256 (§2.2); the condition (C1-M, carried); K0′'s pooled τ rule and stops; the fresh corpus's split and exclusion rule (overall and per half); Stage R's gates R0–R3, A1–A2 and its row ladder; Stage S's ladder with salt 8206; TASK-077's estimators with salt 8206; §10's power (exact G-bar, the McNemar twin table); the `"not evaluated"` sentinel |
| `src/embodied_jepa/lewm_pr_v2_runtime.py` | the workers: `PredReadoutAim` (TASK-077's `WorldModelAim` with each arm's own readout: **R-S** for W, L-shuf and L-mean, **R-N** for N; `readout_of` refuses any other arm, so no candidate arm can read R8); L-rand with salt 8209; `CollectF`, the fresh corpus's collector (gate-P's box from **p̂**, contrast-T's from the true plate, both logging p̂); the offline aims of all seven arms (H-rule's `RuleCommit` iteration and H-sysid's `choose_aim` from a logged 405 state) |
| `src/embodied_jepa/lewm_pr_v2_offline.py` | root loading, the roll-outs (S: W under the stand-in chunk; N: zero commands; L: W from the mean latent), the dual ridges with salt 8205 (`fit_readouts` refuses any fresh or TASK-080 root: G-fresh), the reported cross-fit (8204) and learning curve (8208), the offline aims' summary, the echo slope (§5.3 point 4), G1–G4 on fresh roots (reported; 8206, 8210), the fresh corpus store and its featurisation |
| `scripts/run_lewm_pr_v2.py` | one invocation per stage: `tests`, `k0`, `corpus`, `featurise`, `rgate`, `closed`, `simulate`, `dryrun`; G-tests, G-sentinel, G-hash (TASK-077's artifacts by sha256: featurise and readouts reports, moments, R8, R-plate, the mean latent, `sysid.json`, the six checkpoints), G-frozen, G-fresh, G-privileged, G-memory, G-disk, G-quiet, G-GPU |
| `benchmarks/manifests/apple-lewm-pr-v2.json` | DRAFT: the frozen block and its sha256; no pin |
| `tests/test_lewm_pr_v2.py` | 31 tests (§2) |

While DRAFT, the runner allows only `tests`, `simulate`, `k0` and `dryrun` (and `--debug` runs);
after the freeze `k0` and `dryrun` are refused (they run once, before it).

## 2. Tests

`tests/test_lewm_pr_v2.py` covers §8 step 1's list: the manifest equals the module and a DRAFT
carries no pin; TASK-077's frozen block and 13 pins are unchanged; the seed block, its ranges and
salts against every forbidden range including 66000–68999; G-fresh (no fit may include a fresh or
TASK-080 root); the move (salt 8201, not TASK-077's 8101), corpus aim, split and planted error;
the estimators equal TASK-077's bit for bit at salt 8106 and use 8206 otherwise; the power table
reproduces §10's within 0.02; W, L-shuf and L-mean read R-S and N reads R-N, through a fake
worker that records which readout each offline aim loads, and no call to R8 exists in the
runtime; gate-P's aim is built from p̂ and never from the true plate (contrast-T's from the true
plate); the controller form is TASK-077's except L-rand's salt; G-privileged statically and in
`run_arm`; the sentinel; every row ladder in both directions (K0′, C′, R, D, S); the runner loads
no other script, uses `install_guards`, `assert_local_import`, `gpu_guard` and the pins; DRAFT
refuses the gated stages; `closed_core` on a fake pool (refused resets, L-shuf's foreign frame,
the salt-8201 move); Stage R writes `first_outcome_utc` after the fits and before it loads
either fresh half; the closed-loop workers get R-S and R-N (and R8 only with H-read); `closed
--cohort S` refuses to run without Stage D's D-PASS report (added after the #154 review, with the
two tests before them; no dry-run code path changed). The full suite at `e4babc1`: 2 064 passed,
37 skipped.

## 3. Debug smokes (debug seeds only; nothing read)

At `e4babc1` (the dry run's code; `7b58c57` for the first dry-run smoke), clean tree, outputs in
`outputs/task080-smoke-*` of the `task080-stage0` worktree (git-ignored). Every row below is a
`-DEBUG` row of four debug resets or roots and is not read; several are non-pass rows because the
bars count 32 or 64 resets.

| stage | outcome | seconds | peak PSS (GiB) | what it exercised |
|---|---|---:|---:|---|
| `k0` (65900–65903) | CAL-ESCALATE-DEBUG | 92 | 8.0 | every planted level, r_K′ (463), the palm speed, the pooled-τ ladder |
| `corpus` (65940–65943) | CORPUS-SEALED-DEBUG | 46 | 8.8 | `CollectF` in both halves; checked from the roots: gate-P targets equal `from_box(a, b, p̂, h)` exactly, contrast-T targets `from_box(a, b, p, h)` |
| `featurise` (GPU, `gpu_run.sh --wait --min-free-gib 8 --board`) | FEATURISED-DEBUG | 7 | 1.5 | both halves; G-anchor 1.4e-4 on 256 frames (bound 1e-3) |
| `rgate` (36 fit roots) | R-COMMAND-KEYED-DEBUG | 77 | 8.1 | fits, cross-fit, gate-P and contrast-T readings, the confound, learning curve, G1–G4, every offline aim, the ladder |
| `closed --cohort D` (65910–65913) | L-DEV-STOP-DEBUG | 105 | 9.3 | W (R-S, 149 roll-outs, 9.8 s per decision), N, L-shuf, L-mean, H-final |
| `closed --cohort S` (65920–65923) | S-VOID-CEILING-DEBUG | 146 | 9.7 | all ten arms and the determinism re-run (passed) |
| `dryrun` (24 fit, 6 val roots) | DRYRUN-DONE-DEBUG | 104 | 9.1 | the dry run's whole path |

## 4. Scale probes and caps (§11)

Each cap is at least 1.5 × the measured or scaled worst case (§11); the draft's working values
are kept, and none is the binding constraint.

| stage | cap (s) | basis | worst case (s) | cap / worst |
|---|---:|---|---:|---:|
| K0′ | 7 200 | TASK-077 K0: 579 s for about 350 attempts (6 levels and 4 proxies × 32, 6 workers); K0′ runs 6 × 32 = 192; debug K0′ 92 s for 24 | ≤ 600 | ≥ 12 |
| C′ | 7 200 | TASK-077 Stage C: 1 234 s for 2 000 roots; debug C′ 46 s for 4 roots (setup about 40 s); gate-P adds one CPU encoding per root | ≤ 700 | ≥ 10 |
| R featurisation | 3 600 | TASK-077: 212.7 s for 1 995 roots; debug 7 s for 4 roots including G-anchor | ≤ 200 | ≥ 18 |
| R (CPU) | 14 400 | the dry run is Stage R's own code at 1 745 fit and 250 evaluated roots: 3 312 s including G-tests (fits and cross-fit 739 s, readings 101 s, offline aims 2 073 s summed over the arms); Stage R adds 250 fit roots (×1.14 on the fits), a second evaluated half (+101 s), the learning curve on gate-P and G1–G4 (about 3 000 roll-outs, about 170 s) | ≤ 4 200 | ≥ 3.4 |
| D | 7 200 | debug D: per-attempt medians W 15.5 s, N 7.0, L-shuf 13.7, L-mean 13.7, H-final 8.3 on 4 workers; 16 resets | ≤ 400 | ≥ 18 |
| S | 21 600 | debug S: the ten arms' per-attempt medians sum to 86.0 s; 64 resets on 4 workers plus the determinism re-run | ≤ 1 600 | ≥ 13 |
| per attempt | 300 | the slowest debug attempt 16.0 s (W) | 16 | 18.8 |
| dry run | 14 400 | measured | 3 312 | 4.3 |

Peak process-tree PSS: 9.96 GiB in the dry run (4 world-model workers; the main process frees its
models and fit predictions before the workers start), 9.7 GiB in debug S, against the 12 GiB
ceiling (G-memory). Disk: 72 GiB free at the dry run's start.

## 5. Power (§10)

`simulate` at `e4babc1` (report sha256 `f28b7082…98cc`, 200 000 trials per cell, salt 8206):
G-bar's exact power is 0.593, 0.732, 0.856, 0.940, 0.982 and 1.000 at true rates 0.875, 0.89,
0.906, 0.922, 0.9375 and 0.969, and the recomputed McNemar twin table reproduces §10's to within
0.01 in every cell (for example 0.946 / 0.608 / 0.458 at W 0.875, N 0.70; 0.889 / 0.543 / 0.414 at
W 0.906, N 0.75; nested / half / independent). At the dry run's predicted rates (§6) the twin
tests' power is 0.9997 or more (0.99976 the lowest) against every twin and coupling, and G-bar's is **0.72**.

G-bar's power hardly depends on the cohort size near the bar (exact, at the same 87.5 % fraction):

| W's true rate | 64 resets (56) | 96 resets (84) | 128 resets (112) |
|---|---:|---:|---:|
| 0.875 | 0.593 | 0.576 | 0.566 |
| 0.889 | 0.723 | 0.734 | 0.747 |
| 0.906 | 0.856 | 0.885 | 0.908 |
| 0.922 | 0.940 | 0.964 | 0.978 |

## 6. R18.13's dry run (development; sets no bar)

`dryrun` at `e4babc1`, clean tree, after G-tests (2 064 passed, 37 skipped); 2026-10-07
19:09:43Z–20:04:55Z (3 312 s, of which 210 s waiting for a quiet machine after the smokes);
report `outputs/task080-dryrun-1/report.json` in the `task080-stage0` worktree, sha256
`e3becdfa…aed5`; evidence copy (with the simulate report and the smoke reports) in
`~/develop/emai/evidence/task080-stage0/`, manifest `_checksums/task080-stage0.sha256` (`1a48d6bf…5ed4`), verified. TASK-077's artifacts
were checked by sha256 (§2.2) and all 21 train, val and old-gate feature files against the
featurise report. **Development only; it sets no bar and gates nothing (R18.15).** It is
optimistic twice over: val selected the checkpoints, and the readouts were fitted on roots whose
aims were built from the true plate (the corpus-aim confound, §5.1). The offline aims themselves
build each grid from p̂ (R-plate's reading of the stored 405 tokens; its val error 0.139 cm
median), as the closed loop does.

**Readouts.** R-S, R-N and R-L of each seed were fitted on the 1 745 train and old-gate roots
(0 infeasible stand-in chunks; λ_rel 0.001, 0.001 and 0.1 on every seed) and read on the 250 val
roots (medians in cm, 95 % intervals, salt 8206; τ_commit = 1.0 cm, K0's):

| seed | e_S (R-S on W's stand-in prediction) | e_N (R-N on N) | e_S / e_N | e_L (R-L, commands alone) | frozen R8 on W's stand-in prediction |
|---|---|---|---|---|---|
| 66800 (flagged) | 0.540 [0.504, 0.581] | 1.583 [1.467, 1.749] | 0.341 [0.305, 0.380] | 2.316 [2.092, 2.651] | 3.412 |
| 66801 | 0.529 [0.480, 0.571] | 1.421 [1.254, 1.628] | 0.372 [0.310, 0.430] | 2.437 [2.104, 2.625] | 2.861 |
| 66802 | 0.535 [0.480, 0.597] | 1.581 [1.347, 1.707] | 0.339 [0.298, 0.407] | 2.309 [2.093, 2.613] | 2.972 |

The encoded ceiling (R8 on the encoded frame at r, R0's form) reads 0.382 [0.343, 0.424] cm on
val; e_S sits 0.15–0.16 cm above it on every seed. In their forms (not gates), R0–R3 all hold on
val on every seed: R1's upper bounds 0.57–0.60 ≤ 1.0, R2's 0.38–0.43 < 1.0, R3's lower bounds
2.09–2.10 > 1.0. Within 1.5 τ: 96–97 % of val roots for e_S, 45–53 % for e_N, 23–26 % for e_L.

**Offline aims** (primary seed 66800, flagged `last_two_triggered`; the grid from p̂; the aim error
against the rule's fixed point; predicted counts of 64 through K0's curve; predictions, not
closed-loop counts):

| arm | predicted count of 64 | aim error median [95 %] (cm) | 87.5th pct (cm) | clip-binding |
|---|---:|---|---:|---:|
| **W** | **56.88** | 0.612 [0.559, 0.660] | 1.044 | 0.008 |
| N | 25.94 | 2.069 [1.886, 2.242] | 3.026 | 0.128 |
| L-shuf | 21.24 | 2.373 [2.156, 2.698] | 3.907 | 0.320 |
| L-mean | 25.54 | 1.990 [1.847, 2.106] | 3.176 | 0.108 |
| L-rand | 8.54 | 5.921 [5.225, 6.778] | 13.629 | 0 |
| H-rule (reported) | 58.50 | 0.492 [0.477, 0.498] | 0.590 | 0.284 |
| H-sysid (reported) | 54.22 | 0.759 [0.666, 0.814] | 1.352 | 0 |

W minus each twin: N **+30.9**, L-shuf +35.6, L-mean +31.3, L-rand +48.3 of 64. In their forms,
A1 (56.88 ≥ 56) and A2 (every difference ≥ +7) hold, A1 by **0.88 of 64**. W − H-rule is −1.6
of 64 (the offline reading of G-NI). No fallback in any arm.

**The echo slope** (§5.3 point 4, W's predicted plate against the aim over the grid, per root):
median matrix [[−0.125, 0.019], [−0.014, −0.416]], Frobenius distance 0.40 [0.39, 0.41] to κI
(κ = −0.5) and 1.82 [1.81, 1.83] to I. The readout does not echo the aim; its response is weaker
than the rule's along the first axis.

**Learning curves** (R-S fitted on nested 1/4, 1/2, 3/4 and all of the 1 745 fit roots, read on
val, medians in cm): 0.810, 0.706, 0.653, 0.540 (66800); 0.787, 0.688, 0.546, 0.529 (66801);
0.812, 0.602, 0.537, 0.535 (66802). From 3/4 to all of the roots they fall by 0.113, 0.017 and
0.002 cm: still falling on 66800, nearly flat on 66801 and flat on 66802. Cross-fitted over the fit roots (salt 8204),
R-S reads the old-gate roots (out of sample for W) at 0.529, 0.627, 0.592 cm and the train roots
(in sample) at 0.576, 0.571, 0.593 cm.

**Reading.** On development data (optimistic; primary seed 66800, flagged `last_two_triggered`) the
predicted-latent readout reads the plate well inside τ, N is far behind (W − N ≈ +31 of 64, three
times §10's +10 threshold), and the binding margin is W's
own count against the 56/64 bar: 56.88 predicted, so G-bar's power at that rate is about 0.72,
and the fresh-root gate A1 may fail if the fresh roots read only slightly worse.

## 7. §15's open questions (R18.16–R18.21)

The rulings are R18.16–R18.20 in [DECISIONS.md](../DECISIONS.md), each **decided by Claude under
owner delegation (2026-09-30)**; R18.21 records the recommendation on question 6 and the K0′
preparation. The protocol's §15 carries them; the protocol stays DRAFT.

1. **The primary seed (R18.16): the carried rule, 66800.** The dry run gives no reason to depart
   from a rule fixed before any number: the three seeds' e_S on val are within 0.011 cm of each
   other (0.529–0.540), so no seed rule on the readouts' error would change the choice in a way
   that matters, and choosing on it now would be a choice made after the numbers. The flag stays
   beside every seed-66800 number.
2. **The fit set (R18.17): all 1 995 old roots.** The R-S learning curves on val are still falling
   on one seed at 1 745 roots (from 3/4 to all of the roots: −0.113 cm on 66800, −0.017 on
   66801, −0.002 on 66802, so nearly flat on two), and none rises. R-S cross-fitted reads the
   old-gate roots, out of sample for W, at 0.529, 0.627 and 0.592 cm against 0.576, 0.571 and
   0.593 on train: better on 66800, worse by 0.056 cm on 66801, equal on 66802; the fresh gate
   measures the out-of-sample cost directly. Adding val's 250 roots is a change from the decomposition record's
   reading, disclosed.
3. **The cohort size and A2 (R18.18): 64 resets and +7/64, unchanged.** The protocol's trigger
   (§10: W − N below about +10/64) is not met: W − N is +30.9 of 64 offline (development,
   optimistic; primary seed 66800, flagged `last_two_triggered`), and every twin is +21 or more
   behind W, so the twin tests are not the binding risk (power 0.9997 or more at the
   dry run's rates). The binding risk is G-bar: W's predicted count is 56.88 against 56, and a
   larger cohort at the same fraction does not help near the bar (power 0.72 at 64 resets, 0.73
   at 96, 0.75 at 128, at a true rate of 0.889). The freeze PR states this risk beside §10.
4. **τ_commit (R18.19): the pooled K ∪ K′ rule (the draft).** The dry run's W aim errors
   (median 0.61 cm, 87.5th percentile 1.04 cm) sit between the 0.5 and 1.0 cm levels, where K's
   counts were each one success above the bar; 64 pooled resets measure that part of the curve
   better than 32. The dry run's per-root aim errors are kept in its report so the freeze PR maps
   them through the pooled curve (`simulate --dryrun … --k0-prime …`). If the pooled τ falls to
   0.5 cm, R1 and every bar tied to τ tighten (§8 step 2); on val, R1's upper bounds of 0.57–0.60
   would then not meet it.
5. **G1–G4 on fresh roots (R18.20): reported, not gated (the draft).** They passed on TASK-077's
   gate split with these checkpoints; the question here is the readout, which R0–R3 and A1–A2
   gate; G1's bars are TASK-077's.
6. **TASK-079's precondition (R18.21, recommendation only).** Recommended: a TASK-080 L-PASS
   counts as TASK-077's "L-PASS (or its equivalent row)" for TASK-079, because TASK-080 tests the
   same claim under the same condition with a refitted readout; TASK-078's pass and an owner
   ruling on Arena stay required. This is the owner's decision and is not ruled here.

## 8. Before K0′

K0′ is prepared, not run (R18.21). It needs an independent reviewer's reported GO at this PR's
merge revision.

- **Command** (from a clean worktree of the merge revision made with
  `scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task080-k0 --from <merge sha> --run`):
  `.venv/bin/python scripts/run_lewm_pr_v2.py k0 --output outputs/task080-k0-1 --log outputs/task080-k0-1.log --evidence /home/huhn/develop/emai/worktrees/task076-evidence`
- **Inputs:** K′ = 65000–65031 (32 resets; never simulated, checked by the seed search and
  G-fresh), the evidence root `task076-evidence` (G-repro), the pinned MuJoCo and encoder
  digests. CPU only, 6 workers, no GPU lock; cap 7 200 s; G-tests runs the full suite first.
- **A GO must check:** the merge revision and a clean tree; CI green on it; the protocol still
  DRAFT, the manifest without a pin, `K0_PRIME_MEASURED` None; TASK-077's frozen block and pins
  unchanged; the K′ range and salt 8207 (planted direction) and 8201 (move); the stops
  (K′ level 0 < 28/32, N_K′(0) < 30/32, r_K′ later than 465 or undefined, median palm speed at
  405 > 0.5 cm per step, pooled τ undefined: CAL-ESCALATE, no clause, nothing frozen); that the
  pooled rule is "every level up to τ reaches ≥ 56/64 over K and K′"; that nothing from D, S or F
  is simulated; at least 10 GiB disk free and a quiet machine.
- **After K0′:** its values go into `K0_PRIME_MEASURED`; the freeze PR re-maps the dry run through
  the pooled curve, restates §10 and is merged on an independent reviewer's APPROVE.
