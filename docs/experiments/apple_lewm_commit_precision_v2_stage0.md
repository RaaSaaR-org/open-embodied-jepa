# TASK-081 Stage 0: code, debug smokes, scale and power (record)

**Status: Stage 0 record; the protocol stays DRAFT.** Protocol:
[apple_lewm_commit_precision_v2.md](apple_lewm_commit_precision_v2.md) §6.1 step 1. Rulings
R19.12–R19.14 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (g), each **decided by
Claude under owner delegation (2026-09-30)**. Nothing here is a result: every closed loop below ran
on **debug seeds 71900–71999 only**, and nothing in them is read. No seed of D (70100–70115) or S
(70200–70327) has been simulated.

## 1. Code (new modules only)

| file | what |
|---|---|
| `src/embodied_jepa/lewm_cp_v2.py` | the frozen-block candidate: carried pins, seeds and salts, the solver's declaration, arms, the D and S ladders at n = 128, the solver effect (reported), the power simulation (salt 8304), the sentinel, Stage 0's record (`STAGE0`) |
| `src/embodied_jepa/lewm_cp_v2_runtime.py` | the worker: `choose_affine_local` (the grid and its argmin as TASK-077's `choose_from_grid` computes them, then the design note's `commit_precision_dev.local_mask` and `affine_fixed_point`, the fallbacks to the argmin and one logged roll-out), `choose_l_rand` (salt 8303), and `run_attempt_task`, which runs TASK-080's `run_attempt_task` with `lewm_pr_v2_runtime.choose` replaced for that one task and restored after it; W-frozen runs as TASK-080's W and is relabelled |
| `scripts/run_lewm_cp_v2.py` | the runner: `closed --cohort D|S` and `simulate`. New runners may not load other scripts (`tests/test_no_runner_imports.py`), so the pieces of TASK-080's runner it uses are **copied verbatim** (TASK-077's chain `old_chain`, `worker_config`, `sim_preflight`, the cohort estimates, `completed`, `attempt_tasks`, `run_arm`, `arm_summary`, `lean`, G-tests, G-quiet, G-disk); a test checks each copy's source against `scripts/run_lewm_pr_v2.py` |
| `tests/test_lewm_cp_v2.py` | 43 tests (below) |
| `benchmarks/manifests/apple-lewm-cp-v2.json` | the DRAFT manifest: the frozen-block candidate and its sha256, no pin, no file pins |

Nothing of TASK-076, TASK-077, TASK-080 or the design note is edited. Every stage checks TASK-076's
84 pins, TASK-077's 13 pins and frozen block, TASK-080's frozen block (`0fc095dc…be064`), its seven
file pins and its protocol document's pin, `commit_precision_dev.py`'s sha256
(`ba3c8d03…91f1`), TASK-080's Stage R report (`bedb8966…48ea7`) and R-S's and R-N's content
sha256s (`6e05d223…1046`, `15405f3f…5271`).

**What the tests check:** the manifest equals the module and a DRAFT carries no pin; the carried
pins (TASK-080, TASK-077, the solver file, τ_commit 1.0 cm and the pooled curve, the primary seed);
every seed range inside 70100–71999 and outside every forbidden range, TASK-080's block and the
design note's development range; the salts (8302–8304 in use, 8301 used, 8305–8312 reserved, the
move salt 8201 carried); the bars (112, 16, 7 as a count); **affine_local equal to the design
note's `all_variants(...)["affine_local"]`** on four slopes with and without noise, and on each
fallback (near-singular, fewer than 4 points, an infeasible chunk at the solution) and the
all-infeasible case; for N, the aim equal to TASK-080's refinement's; **W-frozen equal to
TASK-080's `choose`** on four slopes; L-rand's salt 8303 (different draws from TASK-080's 8209);
the arms' solvers and readouts (never R8); the per-task replacement of `choose` restored after
every task, W-frozen relabelled and a wrong logged solver refused (G-solver); no task truth in the
choosers; the S ladder in both directions at n = 128 (every row), the same rows as TASK-080's
`decide_s` on 40 random outcome sets when patched to n = 64; the D ladder; the solver effect;
the power pieces (TASK-080's (k+, k−) = (1, 6) fails G-NI at n = 64); the runner's carried copies,
its guards, its refusals (misused flags, `closed` while DRAFT, a `--stage-d` that is not this
task's D-PASS, a Stage R report that is not TASK-080's); `closed_core` with every arm and its
solver for D and S; and the module imports without torch or MuJoCo.

## 2. Debug smokes (71900–71999 only; nothing is read)

Inputs as Stage D will use them: TASK-076's evidence root, TASK-077's featurisation, Stage O fits
and six job reports, TASK-080's Stage R report. CPU, 4 world-model workers, no GPU.

| run | revision | outcome | seconds | peak PSS (GiB) | note |
|---|---|---|---:|---:|---|
| closedD-1 | `19e3e83` | V | 22 | – | `KeyError: 'stages'`: the report lacked the record TASK-076's harness writes into; fixed in `2faf9e5` |
| closedD-2 | `2faf9e5` | L-DEV-STOP-DEBUG | 121 | 9.38 | every D arm ran; on 4 debug resets the +3 margin stop is expected to bind |
| closedS-1 | `2faf9e5` | V | 162 | – | in-run G-tests passed (2120 passed, 37 skipped); then G-plan refused the `--stage-d` given (closedD-2, not D-PASS-DEBUG), as designed |
| closedS-2 | `2faf9e5` | S-VOID-CEILING-DEBUG | 160 | 9.40 | every S arm and the determinism re-run (ok); on 4 resets H-final < 112 always |
| closedD-3 | `9c82b97` | L-DEV-STOP-DEBUG | 121 | 9.49 | the final code (the fit's design rank logged); the same counts as closedD-2 |

`2faf9e5` and `5bb9579` have the same tree (the branch was rebased onto #161's merge). Every W,
N, L-shuf and L-mean decision logged `solver: affine_local`, W-frozen `frozen` and L-rand `none`;
no affine_local decision fell back; in closedD-3, where the fit's design rank is logged, no fit
had a rank below 3.

**Per-attempt medians** (debug S, seconds): W 15.56, W-frozen 13.97, N 7.01, L-shuf 13.91, L-mean
13.75, L-rand 6.65, H-rule 4.49, H-sysid 4.31, H-final 7.99, H-read 8.16, H-now 4.21; the slowest
attempt took 15.6 s.

## 3. Scale and caps (R19.12)

- **D:** about 76 s of attempts per reset over seven arms, 16 resets on 4 workers ≈ 305 s, plus
  G-tests (about 160 s) and G-repro (about 42 s): **about 510 s**. Cap **7 200 s** (14 ×).
- **S:** about 100 s per reset over eleven arms, 128 resets on 4 workers ≈ 3 200 s, plus the
  determinism re-run, G-tests and G-repro: **about 3 420 s (57 min)**. Cap **21 600 s** (6.3 ×).
- **Per attempt:** cap **300 s** against a slowest attempt of 15.6 s (19 ×).
- **Memory:** peak process-tree PSS 9.49 GiB against the 12 GiB ceiling (TASK-080's D and S peaked
  at 9.70 and 9.65 GiB).

The provisional caps of protocol §10 stand; each is ≥ 1.5 × its scaled worst case.

## 4. Power (R19.13; `simulate`, salt 8304)

`run_lewm_cp_v2.py simulate --output outputs/task081-simulate-1` at `2faf9e5` (DRAFT allowed;
in-run G-tests 2120 passed, 37 skipped), report sha256
`5043c3aba4a9a3c1646336c65b7ff7963f4f78f83393f6b509ea43e07186a56c`. 20 000 trials per cell; each
(k+, k−) cell's G-NI decision from 10 000 multinomial resamples (the reset bootstrap of the summed
difference depends on the resets only through (k+, k−)). H-sysid at 62/64 (its TASK-080 S rate),
coupled to W as H-rule is and conditionally independent of it; C the better of the two per trial.

**G-NI's power with the better-of-two comparator**, n = 128, δ = 16 (overlap / half / independent):

| W | C (H-rule) = 0.984 | C (H-rule) = 0.992 |
|---|---|---|
| 0.906 | 0.43 / 0.36 / 0.32 | 0.33 / 0.28 / 0.26 |
| 0.922 | 0.71 / 0.62 / 0.57 | 0.59 / 0.54 / 0.50 |
| 0.938 | 0.93 / 0.86 / 0.81 | 0.85 / 0.80 / 0.76 |
| 0.953 | 0.99 / 0.98 / 0.95 | 0.97 / 0.95 / 0.93 |
| 0.969 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

- The single-comparator values agree with the design note's §5 table (salt 8301, 4 000 resamples)
  to within 0.01; the better-of-two comparator lowers them by at most 0.03 here.
- **At the planning rates** (W 0.953, C 0.992): **0.93–0.97**. At TASK-080's S rate for W (0.906)
  the power is 0.26–0.43: a W that the solver did not improve is unlikely to pass G-NI.
- **Size at the margin** (G-NI passes with W = C − δ, one comparator): **2.6–3.6 %** (nominal
  2.5 %). **The clause's false-fire rate at the margin** (L-INFERIOR, upper bound < −16, with
  W = C − δ): **1.4–1.7 %**. These are different tails (#161 review, nit N2).
- **G-bar** (exact): 0.566 at 0.875, 0.759 at 0.89, 0.908 at 0.906, 0.978 at 0.922, **0.9975** at
  0.938 and 0.9999 at 0.953 (#161 review, nit N1: 0.938 is 0.9975, not "≥ 0.998").
- **The twin tests** (exact McNemar, p < 0.01, 128 pairs): ≥ 0.9997 in every coupling at W 0.906
  against N 26/64 and L-mean 30/64 (TASK-080's S rates), and against a twin at 0.60.

## 5. What Stage 0 does not do

It runs no D or S seed, fits nothing and measures nothing that sets a bar. The freeze (protocol
§6.1 step 2) only sets STATUS FROZEN, the pins and Stage D's plan.
