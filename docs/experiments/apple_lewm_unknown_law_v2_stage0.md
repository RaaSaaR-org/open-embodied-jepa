# TASK-082 Stage 0: code, debug smokes, scale probes and the power simulation that fixes δ (record)

**Status: Stage 0 record; the protocol stays DRAFT.** Protocol:
[apple_lewm_unknown_law_v2.md](apple_lewm_unknown_law_v2.md) §7.1 step 2. Rulings R20.17–R20.22 in
[DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (m), each **decided by Claude under owner
delegation (2026-09-30)**. Nothing here is a result: every simulated attempt ran on **debug seeds
74800–74899 only**, every model was a debug model (300 updates) on the debug corpus, the Stage G
probe ran on synthetic latents, and nothing in any of them is read. No seed of K (72400–72463),
D (72500–72515), S (72600–72727) or the corpus (72800–74799) has been simulated, and no model has
been trained on the real corpus. The canonical status sentence (R7) is unchanged.

## 1. Code (new modules only)

| file | what |
|---|---|
| `src/embodied_jepa/lewm_ul_v2.py` | the frozen-block candidate: carried pins (TASK-077, TASK-080, TASK-081 frozen blocks; `plate_law_dev.py`, `plate_law_dev_runtime.py` and `commit_precision_dev.py` at their Stage-0 sha256), the carried R-plate and TASK-077 plan (U = 95 000, G1's bars), seeds and salts (8405–8412 with sub-keys), U-sat's parameters checked against `plate_law_dev.LAWS["sat"]`, H-rule's written law (κ = −0.5), every stage's ladder (K0, C, O, G, D, S), the statistics (salt 8411), the power simulation and δ's rule (salt 8412, sub-key 3), Stage 0's record (`STAGE0`) |
| `src/embodied_jepa/lewm_ul_v2_runtime.py` | the workers: `UlMotion` (TASK-077's `CorpusMotion` whose `target` is the design note's `LawMotion.target`, law `sat`), `UlAim` (TASK-080's `PredReadoutAim` form with this task's models and readouts, committed by TASK-081's `choose_affine_local`, imported unchanged), L-rand (8412/1), `CollectUl` (the corpus collector with the ceiling label: TASK-076's `LookaheadAim` once at 405 in cloned state), `KrrHatAim` (H-sysid-krr), `PAim`, `RuleFitCommit` (H-rule-fit), `ReadingAim` (K0's p̂ for the clip-binding check), the attempt, the corpus arrays and the offline aims |
| `src/embodied_jepa/lewm_ul_v2_offline.py` | the corpus store and seal, the featurisation (train, val, gate-P; the table adds p̂ and the label), Stage O's `readouts_core` (O1, O3, O4 cross-fitted over train + val; the design note's §4.2 table at full size; the train-only fits: R8, moments, mean latent, H-sysid on the logged CPU p̂, κ for H-rule-fit, H-sysid-krr, P-aim), Stage G's pieces (fits of R-S, R-N, R-L refusing any gate-P root, the dynamics gates with the carried G1 bars, the offline aims against the logged label, the echo slope beside U-sat's Jacobian) |
| `src/embodied_jepa/lewm_ul_v2_train.py` | TASK-077's training loop **copied verbatim** (a test compares the source), with `WindowSampler` bound to a subclass drawing from `SeedSequence([8409, seed])` and this task's model metadata |
| `scripts/run_lewm_ul_v2.py` | the runner: `tests`, `k0`, `corpus`, `labelcheck` (debug only), `featurise`, `readouts`, `train`, `gates`, `gscale`, `closed --cohort D|S`, `simulate`; TASK-077's helpers copied verbatim (a test compares 15 functions' source); G-tests, G-sentinel, G-hash/G-frozen on everything carried, G-law, G-privileged, G-solver, G-split, G-fresh, G-quiet, G-disk, G-memory, G-GPU |
| `tests/test_lewm_ul_v2.py` | 82 tests (below) |
| `benchmarks/manifests/apple-lewm-ul-v2.json` | the DRAFT manifest: the frozen-block candidate and its sha256, no pin, no file pins |

Nothing of TASK-076, TASK-077, TASK-080, TASK-081 or the design note is edited.

**What the tests check** (protocol §7.1 step 2's list): the manifest equals the module and a DRAFT
carries no pin; the carried frozen blocks, their file pins and documents, and the three pinned
development files; every seed range inside R19.24's free ranges and outside every forbidden range
(TASK-081's list and block, the design note's three ranges), the salts fresh, and `check_seeds`
refusing anything but the declared range; the draws (move 8405, corpus aim 8406, split 8407,
planted direction 8408, sampler 8409, folds 8410 with sub-keys, bootstrap 8411, L-rand and G4's
permutation 8412); **the runtime's law equals `sat_displacement` at A, D, Θ, L** (before 405, after
405 with the lag, held after s1, and not C1-M's rule); G-law refusing a record not under U-sat;
**H-rule's law is C1-M's κ = −0.5** (and H-rule-fit's offline aim gets the fitted κ); **gate-P's aim
is built from p̂ and train's from the true plate**; **the label never changes the committed aim**
(and the bit-exact trajectory check below); the readouts each arm reads (R-S for W, L-shuf, L-mean,
R-N for N, R-S of its seed for a spread arm, never R8); **affine_local equal to TASK-081's runtime**
(and to the design note's tested variant) for every candidate arm on four slopes; L-rand's salt;
**no task truth in any non-privileged arm** (an AST scan of the candidate, learned-tier and
fitted-rule code); the learned tier's fits (κ recovered, the kernel ridge's grids, the learned-tier
folds differ from the design note's); K0's τ, r_K and every stop in both directions (and the
τ < 1 cm ruling point); the clip-binding count; Stage O's `readouts_core` on a tiny synthetic
featurisation; Stage G refusing gate-P roots in a fit; every Stage G row in both directions
(H-GATE-FAIL before R-VOID-CEILING); the offline aims against the label; the echo slope's
Jacobian; **the training loop is TASK-077's verbatim with salt 8409**, and `select_checkpoint` and
`last_two_triggered` with no raw argmin; **`check_budget` on the carried budget block**; every
Stage S row in both directions, the secondary claim and the learned tier read by no row, the ladder
equal to TASK-081's at δ = 16 and salt 8302 on 40 random outcome sets; the D and O rows; salt
8411; the power pieces and δ's rule; the runner's carried code, its guards (`assert_local_import`,
`install_guards`, `MemoryWatch`, `check_pins`, `scale_probe`, `check_law`, …), no runner imports,
its refusals (misused flags, every post-freeze stage while DRAFT, `labelcheck` outside debug, K0
after the freeze, a D report that is not this task's D-PASS, a TASK-077 readouts report that is not
the pinned one); `closed_core` with every arm for D and S on a fake pool (refused resets, L-shuf's
foreign frame, each arm's fit); `run_arm`'s G-solver, G-privileged and G-law; the sentinel; the
stage cap excluding G-quiet's wait (R20.18); that the gscale probe calls Stage G's own functions;
and that the modules import without torch or MuJoCo.

## 2. Debug smokes (74800–74899 only; nothing is read)

Inputs: TASK-076's evidence root (`/home/huhn/develop/emai/worktrees/task076-evidence`, G-repro
passed in every simulated stage) and TASK-077's R-plate (`task077-stageo/outputs/
task077-readouts-1/fits`, readouts report `452045d2…8b78`, content `08bde901…9eb9`). Worktree
`task082-s0chain`; `outputs/task082-smoke-1/<stage>/report.json`. The GPU jobs ran through
`scripts/gpu_run.sh --wait --min-free-gib 8 --board`, each in its own slot, after a `tests` record
at their revision (TESTS-PASS, 2 210 passed, 37 skipped).

| stage | revision | outcome | seconds | peak PSS (GiB) | note |
|---|---|---|---:|---:|---|
| k0 (74800–74803) | `2c861a1` | CAL-ESCALATE-DEBUG | 118 | 8.97 | every level, H-rule, H-now; the clip-binding check read p̂ on the 3 level-0 resets that reached 405 (0 outside) |
| corpus (74830–74869) | `2c861a1` | CORPUS-SEALED-DEBUG | 99 | 9.53 | 40 roots, 0 excluded, 40 labels, all converged |
| labelcheck (74870–74875) | `2c861a1` | **LABELS-UNCHANGED-DEBUG** | 62 | 9.45 | each root collected with and without the label: frames, commands, plate, palm, hidden render, target, 405 state, p̂, executed steps, termination and outcome **bit-identical on 6 of 6** |
| featurise | `8232523` | FEATURISED-DEBUG | 11 | 1.48 | G-anchor 8.8e-5 (bound 1e-3) |
| readouts | `8232523` | O-NO-BAR-DEBUG | 0.4 | – | 35 fit roots |
| train × 6 (W, N of 74890–74892, 300 updates) | `8232523` | T-JOB-DONE-DEBUG | 54–56 | 2.6–2.7 | 0.167 s per update (median); 5.5 GiB GPU allocated |
| gates | `8232523` | H-GATE-FAIL-DEBUG | 132 | 9.73 | every phase: fits, gate-P readings, dynamics, the offline aims of 12 arms |
| closed D (74810–74813) | `8232523` | L-DEV-STOP-DEBUG | 136 | 9.36 | every D arm |
| closed S (74820–74823) | `8232523` | S-VOID-CEILING-DEBUG | 1 136 | **10.76** | every S arm (15) and the determinism re-run (ok); in-run G-tests 2 211 passed; about 13 minutes of the time were the simulator setup on a machine another session had loaded (load 19) |

Earlier smoke Vs, each fixed and committed before the chain above: the first `k0` and `corpus`
attempts stopped at G-memory's preflight while another session's Isaac job held 13 GiB (nothing
simulated; reports deleted); `featurise` at `2c861a1` refused the sealed corpus because the manifest
writes its splits in sorted order (`open_corpus` fixed in `8232523`, with a test).

Every candidate decision logged `solver: affine_local` (L-rand `none`); no affine_local decision
fell back; `task_truth_in_controller` was 0 for every non-privileged arm; every record carried the
U-sat law (G-law).

## 3. Scale and caps (§11)

| stage | measured | scaled worst case | cap | cap / worst |
|---|---|---|---:|---:|
| K0 | planted attempt ≤ 8.6 s, H-rule 4.6 s, H-now 4.2 s | 512 attempts on 6 workers ≈ 650 s, + setup and G-tests ≈ **910 s** | 7 200 | 7.9 |
| C | **per root with the label: 5.8 s median, 8.2 s max** (the look-ahead's branches 600 steps in median, 840 at most); 0.61 MB per root | 2 000 roots at 8.2 s on 6 workers ≈ 2 730 s, + estimates, G-repro, G-tests ≈ **3 100 s** | 14 400 | 4.6 |
| O featurisation | 40 roots in 11 s; TASK-077's probe 213 s for 1 995 | ≈ **500 s** | 3 600 | 7.2 |
| O readouts | TASK-077's probe 285 s; the learned tier's kernel ridges about 3.6 s per fit (1 500 rows, 16 grid points × 5 folds) | ≈ **330 s** | 7 200 | 22 |
| T (each job) | 0.167 s per update (debug); TASK-077's jobs 17 180–21 301 s | carried | 46 800 | ≥ 2.2 |
| G | **gscale** (below): fits 751 s, gate-P readings 101 s, dynamics 158 s; offline aims scaled from the debug Stage G 5 202 s; stand-in chunks ≈ 83 s; G-tests ≈ 200 s | ≈ **6 500 s** | 14 400 | 2.2 |
| D | 67 s of attempts per reset over seven arms | 16 resets on 4 workers ≈ 280 s, + setup, G-tests ≈ **600 s** | 7 200 | 12 |
| S | 127 s of attempts per reset over 15 arms (the slowest attempt 15.9 s) | 128 resets on 4 workers ≈ 4 070 s, + determinism, setup, G-tests ≈ **4 600 s** | 21 600 | 4.7 |
| per attempt | slowest 15.9 s | | 300 | 19 |

**The Stage G probe** (`run_lewm_ul_v2.py gscale`, report sha256 `199e2704…8557`, at `8232523`,
DRAFT-allowed, in-run G-tests): Stage G's own `g_fit_core`, `g_read_core` and `g_dynamics_core`
through `run_tools.scale_probe` on synthetic latents at the real sizes (1 750 fit roots, 250 gate-P
roots and a 250-root synthetic gate-P store, 65 frames each) with the three debug checkpoints (the
real architecture); peak probe PSS 4.05 GiB. The offline aims (per root and per worker) are scaled
linearly from the debug Stage G's wall seconds: W-like arms 3.6–4.1 s per root on 4 workers,
N 1.0 s, L-rand 0.9 s, the hand-written and learned-tier arms below 0.1 s.

**Memory.** The debug closed S peaked at **10.76 GiB** process-tree PSS against the 12 GiB ceiling
(TASK-081's S peaked at 9.65 GiB): the workers now also hold the two spread seeds' W. The margin is
thinner than TASK-081's; Stage S keeps 4 workers (R20.21).

**Every provisional cap of §11 stands** (each ≥ 1.5 × its scaled worst case).

## 4. Power and δ (§9.2, §10; R20.20)

`run_lewm_ul_v2.py simulate --output outputs/task082-simulate-2` at `148a2b8` (DRAFT-allowed,
in-run G-tests), report sha256 `dd1a22d8…5efb`. 20 000 trials per cell; each (k+, k−) cell's G-NI
decision from 10 000 multinomial resamples (salt 8412, sub-key 3). H-sysid at C and H-rule at
C − 0.10, each coupled to W, C the better of the two per trial. The first run, `simulate-1` at
`d9625c7`, was voided by its 3 600 s cap after a 73-minute G-quiet wait with its computation done;
its power tables equal `simulate-2`'s exactly (R20.18).

**δ's rule.** G-NI's power at W = C under the half coupling (C = 0.80 / 0.85 / 0.875):

| δ | power at W = C (half) | size at the margin, max over C and couplings | L-INFERIOR at the margin, max | qualifies |
|---|---|---|---|---|
| 8/128 | 0.387 / 0.462 / 0.526 | 3.97 % | 2.49 % | no (power) |
| 12/128 | 0.709 / 0.792 / 0.853 | 3.84 % | 2.22 % | no (power at C = 0.80, 0.85) |
| **16/128** | **0.917 / 0.960 / 0.978** | 3.51 % | 2.37 % | **yes** |

**δ = 16/128** (not the fallback; the rule could not tighten TASK-081's margin). The size reading
is the stricter of the single-comparator (H-sysid alone) and the better-of-two rates (declared in
R20.17); every size is at most 3.97 %, so the reading changes nothing.

**G-bar** (exact): 0.566 at W = 0.875, 0.908 at 0.906, 0.978 at 0.922, 0.9975 at 0.938, 0.9999 at
0.953.

**G-NI at δ = 16** with the better-of-two comparator (overlap / half / independent): ≥ 0.97 in every
cell with W ≥ 0.906 (≥ 0.99 with W ≥ 0.922); at W = 0.875, 1.00 / 0.98 / 0.83 for C = 0.875 and
≥ 0.93 otherwise.

**The secondary claim** (W > C, exact one-sided McNemar p < 0.01; overlap / half / independent):

| W | C = 0.80 | C = 0.85 | C = 0.875 |
|---|---|---|---|
| 0.875 | 0.86 / 0.30 / 0.19 | 0.04 / 0.03 / 0.02 | 0.00 / 0.00 / 0.01 |
| 0.906 | 0.99 / 0.65 / 0.45 | 0.58 / 0.19 / 0.12 | 0.11 / 0.05 / 0.04 |
| 0.922 | 1.00 / 0.81 / 0.63 | 0.83 / 0.36 / 0.24 | 0.39 / 0.14 / 0.10 |
| 0.938 | 1.00 / 0.92 / 0.80 | 0.94 / 0.58 / 0.41 | 0.70 / 0.30 / 0.21 |
| 0.953 | 1.00 / 0.97 / 0.91 | 0.98 / 0.77 / 0.61 | 0.88 / 0.51 / 0.38 |

As the protocol's reading said: low to moderate unless W sits well above C; at C = 0.875 (the
development check's H-sysid on the true plate) the secondary claim's power is at most 0.38 with
independent outcomes.

**The twin tests** (exact): ≥ 0.9993 in every coupling at TASK-081's twin rates (N 73/128, L-shuf
39/128, L-mean 59/128, L-rand 20/128) for every W in the grid; against a stronger N at 0.75: 0.997 /
0.737 / 0.532 at W = 0.875, 1.00 / 0.931 / 0.804 at 0.906, 1.00 / 0.975 / 0.904 at 0.922.

**L-PASS** (the product under the overlap coupling, at δ = 16): 0.566, 0.908, 0.978, 0.998 and
1.000 at W = 0.875 … 0.953 for every C in the grid; G-bar binds.

## 5. What Stage 0 does not do, and what blocks K0

It simulates no seed of K, D, S or the corpus, trains nothing on the real corpus and measures
nothing that sets a bar; δ is fixed by the pre-declared rule only. **Nothing known blocks K0**
except its own reported GO (R20.1): `run_lewm_ul_v2.py k0 --output <new> --evidence
/home/huhn/develop/emai/worktrees/task076-evidence --r-plate-fits
/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-readouts-1/fits`, from a clean
worktree of the merged revision (`scripts/new_worktree.sh <dir> --run --from <rev>`), CPU only,
with at least 16 GiB of memory available (G-memory: another session's Isaac jobs held 13 GiB during
this Stage 0) and a quiet machine (G-quiet). K0's values and δ = 16/128 enter the frozen block at
the freeze.
