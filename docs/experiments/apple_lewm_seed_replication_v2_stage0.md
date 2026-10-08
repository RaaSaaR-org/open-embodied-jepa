# TASK-083 Stage 0: code, debug smokes, scale and power (record)

**Status: Stage 0 record; the protocol stays DRAFT.** Protocol:
[apple_lewm_seed_replication_v2.md](apple_lewm_seed_replication_v2.md) §6.1 step 1. Rulings
R21.13–R21.16 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (p), each **decided by
Claude under owner delegation (2026-09-30)**. Nothing here is a result: every closed loop below ran
on **debug seeds 75910–75913 only**, and nothing in them is read. No seed of S (75200–75327) has
been simulated.

## 1. Code (new files only)

| file | what |
|---|---|
| `src/embodied_jepa/lewm_rep_v2.py` | the frozen-block candidate: carried pins (TASK-081's frozen block `77ccc636…705a`, TASK-080's, TASK-077's), the six checkpoints and the twelve readout sha256s (R-S and R-N of 66800, 66801 and 66802, by content and by file), the seed flags, seeds and salts, the arms (per seed, shared, reference, dropped), the per-seed ladder (TASK-081's `decide_s` with salt 8501, no clause), the combined row (`decide_combined`), the reported statistics (exact intervals, pairs, W[66801] − W[66802] with a two-sided exact McNemar p), R7's wording per row, the power simulation (salt 8502), the sentinel and Stage 0's record (`STAGE0`) |
| `scripts/run_lewm_rep_v2.py` | the runner: `closed --cohort S` and `simulate`. It loads no other script; the pieces of TASK-081's runner it uses (the worker pool, `old_chain`, `worker_config`, `sim_preflight`, the cohort estimates, `attempt_tasks`, `run_arm` with G-solver, `arm_summary`, `lean`, G-tests, G-quiet, G-disk, TASK-077's and TASK-080's pin checks) are **copied verbatim**, and a test compares each copy's full source lines with `scripts/run_lewm_cp_v2.py`. New: `check_task081` (TASK-081's frozen block, six file pins and protocol document), `stage_r_seed_readouts` (R-S and R-N of every seed at their content and file sha256, refusing a swapped file), `seed_worker_config` and `check_seed_config` (G-seed), `check_seed_records` (each candidate decision's logged readout is its arm's), `SeedPools` (one pool per model seed in `POOL_ORDER`; the first refits the P readout, renders the cohort and runs the shared arms; G-look's reference is carried across pools; each pool is closed in a `finally`), `seed_arms`, `shared_arms`, `decide` and `reported` |
| `tests/test_lewm_rep_v2.py` | 28 tests (below) |
| `benchmarks/manifests/apple-lewm-rep-v2.json` | the DRAFT manifest: the frozen-block candidate and its sha256, no pin, no file pins |

The frozen-block candidate's sha256 at this record's commit is
**`4b73d1e5c7e83dbaf5306a929c05f067ba7c15bbc8f3a5c7e5a2f1603722d9ff`** (STATUS DRAFT, with
`STAGE0` filled in); the freeze sets STATUS FROZEN, which changes it, and pins the result.

The worker is TASK-081's `lewm_cp_v2_runtime`, imported unchanged: it takes the model seed, the
checkpoint path and sha256 and the readouts from its configuration, so a seed is chosen by
configuration only. Nothing of TASK-076, TASK-077, TASK-080 or TASK-081 is edited.

**What the tests check:** the manifest equals the module and a DRAFT carries no pin; TASK-081's
frozen block, six file pins and protocol document, TASK-080's and TASK-077's frozen blocks; the
reused checkpoints equal `lewm_pr_v2.REUSED`, 66800's readouts equal TASK-081's, and the twelve
readout sha256s are distinct; the seed flags; S = 75200–75327 (128) and debug 75910–75913 inside
75000–75999 and outside every forbidden range (TASK-081's list and block, TASK-082's block); salts
8501–8502 in use, 8503–8512 reserved, 8201 and 8303 carried, 8301–8312 and 8401–8412 earlier; a
model seed other than 66800–66802 refused; the move draw and reset equal TASK-080's on the fresh
seeds; the bars (112, 16, 7, p < 0.01); **the per-seed ladder equal to TASK-081's `decide_s`** on
40 random outcome sets when given TASK-081's salt (rows, G-NI intervals, twin tests), and every row
in both directions at n = 128 with `clause_fires` always false; the bootstrap salt 8501; **the
combined row in both directions** (REP-PASS, REP-ONE with the passing seed named, REP-NONE, the
detectably-not-replicated list, REP-VOID-CEILING, V; a mixed ceiling, a void seed, a missing seed
or a wrong seed refused); no pooling (a seed one reset short stays short beside a seed at 128/128);
the labels; the two-sided McNemar p and the between-seed difference; the exact intervals (118/128
gives 0.861–0.962); the power pieces; the runner's carried copies, its guards and that it loads no
other script; its refusals (misused flags, `closed` while DRAFT, a Stage R report that is not
TASK-080's, a swapped readout file or sha256 for a seed); `seed_worker_config` for 66800 equal to
TASK-081's `worker_config`, and for 66801 and 66802 that seed's W, N, R-S and R-N, a wrong pairing
refused; `check_seed_records` (a decision of an arm with a declared readout must log exactly that
readout; a missing one is refused); **`closed_core` with fake pools**: the pools opened and closed in
order 66801, 66802, 66800, the shared arms once in the first pool, each seed's W, N, L-shuf,
L-mean and determinism re-run in its own pool, L-shuf's foreign frame from that seed's own W on the
next reset that reached 405, W[66800] alone in the third pool, the combined row REP-PASS on a
designed outcome, and a failing pool closed before the error propagates; the sentinel; the modules
import without torch or MuJoCo.

## 2. Debug smokes (debug seeds 75910–75913; nothing is read)

From the worktree `task083-stage0`, CPU only, 4 workers, on a quiet machine. `closedS-1` and
`closedS-2` ran at `93dde1e` (kept on the branch `prov/task083-stage0-smokes`; after the rebase
onto the reviewed protocol its code files are byte-identical at `ee9dd91`); `closedS-3` ran at
`aca1f0e`, the code after the Stage 0 review's fixes (G-seed refuses a missing readout; L-shuf's
foreign frames computed only where L-shuf runs; the power's replicates):

| smoke | revision | G-tests | outcome | seconds | peak tree PSS | report sha256 |
|---|---|---|---|---:|---:|---|
| `closedS-1` | `93dde1e` | skipped (`--debug-skip-tests`) | REP-VOID-CEILING-DEBUG | 220 | 9.42 GiB | `fddf99e2…a9ef3` |
| `closedS-2` | `93dde1e` | 2241 passed, 37 skipped | REP-VOID-CEILING-DEBUG | 381 | 9.58 GiB | `cb4cdfa4…c23dc` |
| `closedS-3` | `aca1f0e` | 2241 passed, 37 skipped | REP-VOID-CEILING-DEBUG | 384 | 9.17 GiB | `4c37259f…8ba6` |

Every arm ran on every debug reset in every pool, in the declared order (L-rand, H-rule, H-sysid,
H-final, then W, N, L-shuf, L-mean and W's determinism re-run for 66801 and for 66802, then
W[66800]). All three smokes gave the same counts; W's determinism re-run matched for both seeds; every
candidate decision logged affine_local (L-rand none) and the arm's own readout (`r_s` or `r_n`);
each pool logged the W and N checkpoint sha256s and the R-S and R-N content and file sha256s it was
given, each equal to the frozen block; the three pools closed cleanly (4 joined, none killed, each);
no render disagreement. The combined row is REP-VOID-CEILING-DEBUG because four debug resets
cannot reach the 112 ceiling, as designed. The debug counts are not read.

## 3. Scale and caps (§11, §12)

Per-attempt medians in `closedS-2` (s): W[66801] 13.80, N[66801] 6.99, L-shuf[66801] 14.13,
L-mean[66801] 13.99, W[66802] 15.34, N[66802] 6.93, L-shuf[66802] 13.96, L-mean[66802] 13.79,
W[66800] 15.32, L-rand 8.21, H-rule 4.47, H-sysid 4.30, H-final 7.98; slowest attempt 15.41 s.
That is **139.2 s per reset**; 128 resets on 4 workers ≈ 4 455 s, plus the 210 s before the first
outcome (G-tests, G-repro, the P readout's refit, the cohort's frames), the two determinism re-runs
(about 30 s) and two further pool starts: **about 4 740 s (79 min)**. The provisional caps stand:
Stage S **21 600 s** (4.6 × the scaled estimate) and **300 s** per attempt (19 × the slowest).
G-memory: peak 9.58 GiB process-tree PSS (9.17 GiB in `closedS-3`) against 12 GiB, one pool alive
at a time.

## 4. Power (§8; salt 8502; report `0e138bb1…1e68`)

`simulate-2` at `aca1f0e` (in-run G-tests 2241 passed, 37 skipped; 20 000 trials per cell), with
each seed's W coupled to H-rule by the coupling and conditionally independent of the other W given
H-rule, H-sysid at 183/192 coupled halfway, C the better of the two by count. G-NI's pass for a
trial's (k+, k−) is its probability over the bootstrap: the share of 25 independent bootstraps
(10 000 resamples each, as the run makes them) whose 2.5th percentile lies above −16; the two
seeds' bootstraps are independent given the outcomes (the run shares one bootstrap index for both
seeds; the fully dependent bound moves the joint rate by at most 0.002, Stage 0 review). Per-seed L-PASS rate (G-bar and G-NI) / both
seeds, overlap · half · independent:

| p_C | W = 0.906 | W = 0.922 | W = 0.938 | W = 0.953 |
|---|---|---|---|---|
| 0.979 | 0.52/0.28 · 0.45/0.20 · 0.40/0.17 | 0.79/0.63 · 0.71/0.50 · 0.63/0.42 | 0.95/0.91 · 0.91/0.82 · 0.85/0.73 | 1.00/0.99 · 0.99/0.97 · 0.97/0.93 |
| 0.984 | 0.45/0.20 · 0.39/0.16 · 0.35/0.13 | 0.72/0.53 · 0.65/0.42 · 0.59/0.36 | 0.93/0.86 · 0.88/0.77 · 0.83/0.69 | 0.99/0.98 · 0.98/0.96 · 0.96/0.92 |
| 0.992 | 0.33/0.11 · 0.31/0.10 · 0.29/0.09 | 0.59/0.35 · 0.55/0.31 · 0.52/0.27 | 0.84/0.71 · 0.81/0.66 · 0.78/0.60 | 0.97/0.94 · 0.96/0.91 · 0.94/0.88 |

So at W-66800's observed 0.922 each seed passes with probability **0.52–0.79** and both with
**0.27–0.63**; at 0.938, 0.77–0.95 and 0.60–0.91; at 0.953, 0.94–1.00 and 0.88–0.99. **Size at the
margin** (W = C − δ): a seed's L-PASS rate (G-bar and G-NI, not G-NI alone as TASK-081 reported it)
2.6–3.5 %, both seeds 0.09–0.16 %; L-INFERIOR's false fire at the margin 1.2–1.6 % per seed. G-bar's
exact power: 0.908 at 0.906, 0.978 at 0.922, 0.998 at 0.938. The twin tests are not simulated
(TASK-081 Stage 0: ≥ 0.9997 at TASK-080's twin rates).

**Superseded, and why.** `simulate-1` (at `93dde1e`, report `b88b0c0d…ecf4`) and the draft's
scratch table (4 000 trials, salt 8599) each decided G-NI for a (k+, k−) cell with **one**
bootstrap draw, so a cell at the boundary passed or failed outright, and which cells flipped
depended on the salt (for example (1, 10) failed in the draft and passed in `simulate-1`; the run's
own estimator puts its lower bound at −16, a fail). Those cells carry much of the mass in the half
and independent couplings, so the two tables differ by up to 0.06 there, more than simulation noise
(Stage 0 review). Averaging over 25 bootstraps per cell estimates the pass probability the run
faces; `simulate-2` lies between the two and supersedes both.

## 5. Open items of the protocol's §14, settled

1. **Caps:** confirmed (§3).
2. **Power:** recomputed (§4) with G-NI averaged over bootstrap replicates; the protocol's §8
   quotes it and keeps the draft table only as superseded.
3. **The pools' order:** 66801, 66802, then W[66800]'s, one alive at a time; peak 9.58 GiB.

## 6. What is next

The freeze: STATUS FROZEN, the frozen-block pin, the manifest's file pins and the protocol
document's sha256, merged on an independent reviewer's APPROVE; then Stage S on its own reported
GO, from a clean worktree of the merge commit, CPU only, on a quiet machine with ≥ 16 GiB
available. Evidence copies of the smokes and the power run:
`~/develop/emai/evidence/task083-stage0/` with `SHA256SUMS`.
