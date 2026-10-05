# TASK-077 Stage 0: code, frozen block, simulations, storage fix and debug smokes

**Development only.** No seed of K, D, S or the corpus was simulated. Every simulated attempt
used the debug range 66900–66999, and every training run used synthetic features or that range's
smoke corpus. Nothing here is a result, and nothing is read from it. No K0 was run; K0 needs a
separate reviewer GO. The protocol ([apple_lewm_c1m_v2.md](apple_lewm_c1m_v2.md)) stays
**DRAFT**. Every choice below was **decided by Claude under owner delegation (2026-09-30)**
(R17.19 in [DECISIONS.md](../DECISIONS.md)). Revised after the independent review of #143
(REQUEST CHANGES at `0dea22b`) under R17.20–R17.24: refused resets, the determinism tolerance,
the artifact chain, Stage D's cap, CI, and eleven smaller items (§6).

The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged.

## 1. What Stage 0 added (§7 step 1)

All of it is in new modules. No hash-pinned TASK-073–076 file and no C1 or C1-M file is
edited. The runner checks C1's and C1-M's six code files byte for byte against `c764ac9`
(G-hash), and it checks TASK-076's 84 pins.

| file | what |
|---|---|
| `src/embodied_jepa/lewm_c1m_v2.py` | the frozen block as module constants: seeds, salts 8101–8112, the condition, the budget block, every stage's row ladder, the `"not evaluated"` sentinel (`NOT_EVALUATED`, `check_sentinel`; every ladder refuses a missing input), the statistics (salt 8106) and the Stage-0 simulations |
| `src/embodied_jepa/lewm_c1m_v2_runtime.py` | the workers: W's controller form (`choose_from_grid`, a pure function of the predictor), `WorldModelAim` for W, N, L-shuf, L-mean and L-rand, the corpus collector, the frame and executed-command logs, the stand-in chunks and the offline aims |
| `src/embodied_jepa/lewm_c1m_v2_offline.py` | the corpus store and its seal; Stage O (featurisation, admission, the train-only fits); Stage G (E60, G1–G5, Gram-matrix ranks, the truncation controls) |
| `src/embodied_jepa/lewm_c1m_v2_train.py` | the shared training code: the contiguous per-root store, the prefetching sampler, the training loop with `select_checkpoint` and `last_two_triggered` |
| `scripts/run_lewm_c1m_v2.py` | one invocation per stage (`tests`, `k0`, `corpus`, `featurise`, `readouts`, `train`, `plan`, `gates`, `closed`, `simulate`, `scale`), with G-tests, G-sentinel, G-hash, G-frozen, G-privileged, G-memory, G-disk, G-quiet and G-GPU |
| `benchmarks/manifests/apple-lewm-c1m-v2.json` | DRAFT: the frozen block and its sha256; no pin |
| `tests/test_lewm_c1m_v2.py` | 41 tests (§7 step 1's list and the review's fixes; listed below) |

**Guards new here.**
- **G-tests.** A CPU stage runs the full suite itself before it starts, and records the summary,
  exit status and timestamps. A GPU job verifies a `tests` stage report instead: the same
  revision as HEAD, exit 0, no failure, finished after HEAD's commit time.
- **G-sentinel.** Every stage report has a `fields` block whose entries start as
  `"not evaluated"`, and only the stage's own setter changes one. At the end, a row in a field
  that was never set is refused (the `early_verdict` bug).
- **G-frozen.** While the protocol is DRAFT, the runner refuses every non-debug stage except
  `tests`, `simulate`, `scale` and `k0`. Every stage also checks the manifest's pin directly:
  unset while DRAFT, equal to the frozen block's sha256 after the freeze.
- **G-split's artifact chain (R17.22).** Every stage after C requires the sealed corpus manifest's
  sha256. The featurise report (FEATURISED) and the readouts report (O-PASS; any O row in a debug
  run) must come from that corpus, and a debug report never feeds a real stage. The moments, R8,
  R-plate and L-mean's mean latent are checked against Stage O's recorded sha256s, and every model
  against the corpus and the moments it was trained with.

**The tests** cover:
- the seed ranges against every forbidden range, and the salts;
- `check_budget` on the block (`consistent`, `max_request` 100 000 = `min_cap`), and that the
  budget rule never escalates;
- `select_checkpoint` and `last_two_triggered` in use, with no raw argmin;
- the sentinel and missing-input refusals;
- every row ladder in both directions, including that no clause row fires when all four McNemar
  tests and G-NI pass;
- McNemar's minimum separation of 7;
- that the vectorised bootstrap is the loop estimator;
- W's controller form on fake predictors: contraction to the rule's fixed point, ties,
  infeasible candidates, L-rand and the all-infeasible fallback;
- that W's aim never reads the hook, the simulator or the truth;
- the G-privileged refusal;
- the store, the prefetcher, the streamed moments and the gather;
- the runner's guards, and that it loads no other script;
- that the scale probe calls the train stage's own function;
- the DRAFT refusals and the misused flags;
- the G-tests record checks;
- lazy imports;
- a CPU training run with a tiny encoder and a reduced 2 × 2 latent (the same loop), whose kept
  checkpoint is `select_checkpoint`'s;
- Stage D and Stage S's own code (`closed_core`) driven by a fake pool, with two resets refused
  before 405: no V, the refused resets counted as misses for every arm, L-shuf's frame from the
  next reset that reached 405, and the determinism re-run passing them as identically refused;
- the determinism tolerance; the artifact chain (every tamper and every wrong report refused);
  Stage D's and Stage S's caps; the direct pin check; `decide_g`'s debug flag; T-DONE.

**Three fixes from the #142 approval**, applied to the protocol:
- §12: at U = 60 000, the figure is 22–38 h.
- §13's cost risk now says "and per job in Stage T (R17.16)".
- §11: if the clause fires, the results document must state the restored scope plainly: every
  other pooled grid and every readout of the predicted latent, including trained readout heads.

## 2. The Stage-0 simulations (§8.4, R17.15, R17.17 point 4)

`run_lewm_c1m_v2.py simulate`: report `outputs/task077-stage0-sim-1/report.json`, sha256
`85adbdee…557075f`, revision `f4b6b48` (the simulation code is unchanged since), clean tree,
full suite passed first (1983 passed).

**Setup.** 10 000 trials per configuration, of 64 paired resets each. The estimator is the real
one: `paired_interval`'s percentile bootstrap with the salt-8106 resample matrix applied to every
trial (M · d), and the exact one-sided McNemar test.

**The comparators.** H-rule and H-sysid each have true rate p_C. Their joint distribution with W
is either "independent" or "nested" (no W-only reset). Between the two comparators, a reset is
shared with probability 0, 0.5 (the reviewer's case) or 1 (one comparator). C is the better of
the two on the trial; a tie goes to H-rule.

**Non-inferiority (G-NI: lower bound of W − C > −8/64).** These are G-NI pass rates.

| p_C | dependence | true W − C = 0 | −2/64 | −4/64 | **−8/64 (the size)** |
|---|---|---|---|---|---|
| 30/32 | independent, share 0 / 0.5 / 1 | 0.69 / 0.72 / 0.77 | 0.35 / 0.39 / 0.45 | 0.14 / 0.17 / 0.21 | **1.2 / 1.3 / 2.4 %** |
| 30/32 | nested, share 0 / 0.5 / 1 | 1.00 | 0.76 / 0.80 / 0.86 | 0.26 / 0.32 / 0.43 | **0.9 / 1.8 / 3.2 %** |
| 31/32 | independent | 0.91 / 0.92 / 0.93 | 0.54 / 0.57 / 0.61 | 0.22 / 0.24 / 0.28 | **1.6 / 1.9 / 2.7 %** |
| 31/32 | nested | 1.00 | 0.77 / 0.80 / 0.86 | 0.29 / 0.34 / 0.43 | **1.6 / 2.0 / 3.5 %** |
| 32/32 | nested (the only case) | 1.00 | 0.86 | 0.42–0.43 | **3.3–3.5 %** |

- **The test's size at the margin.** With one comparator (share 1, or the "single" columns of
  the report) it is **2.4–3.8 %** against the nominal one-sided 2.5 %. That is slightly
  anti-conservative, as the review found (3.3 %). With the better of two chosen after S it is
  **0.9–3.5 %**.
- **The max-of-two comparator lowers power.** At W = C with independent outcomes and C at 30/32,
  power falls from 0.77 to 0.69.
- **L-INFERIOR's false fire at δ.** When the true W − C is exactly −8/64, the upper bound falls
  below −8/64 in **0.9–2.0 %** of trials with one comparator and **0.9–3.1 %** with the better of
  two.

**Twins (L-NO-GAIN: failed McNemar and upper bound of W − arm < +7/64).**

| W | true advantage | reversed pairs per 64 | McNemar passes | L-NO-GAIN per twin | ≥ 1 of 4 twins fires |
|---|---|---|---|---|---|
| 56/64 | +7/64 | 0 / 1 / 2 | 0.55 / 0.39 / 0.29 | **2.4 / 3.0 / 2.6 %** | 9.0 / 11.7 / 10.9 % |
| 60/64 | +7/64 | 0 / 1 / 2 | 0.56 / 0.38 / 0.29 | **2.3 / 3.1 / 3.0 %** | 9.2 / 12.1 / 10.6 % |
| 56/64 | +3/64 | 0 / 1 / 2 | 0.03–0.04 | 31–42 % | 75–88 % |
| 56/64 | 0 | 0 / 1 / 2 | 0.00 | 87–100 % | ≥ 99.9 % |

- **The clause's false-fire rate at a true advantage of exactly +7/64** is **2.3–3.1 % per
  twin**, about R8.14's 2.2–3.0 %. It is 9–12 % if all four twins sat exactly at the boundary at
  once. That is a worst case: the C1-M proxies sat 20–24/32 below the ceiling.
- **Power is low at +7/64.** At exactly +7/64 the McNemar test passes only 29–56 % of the time
  per twin, so the row would most often be L-TWIN-NEAR (an escalation).

**The protocol asked for a re-run "with K0's counts".** K0 runs no comparator arm, so its only
count that bears on C is the ceiling N_K(0). K0-PASS needs N_K(0) ≥ 30/32, so a re-run at
min(30/32, N_K(0)) would always be 30/32, a configuration already in the grid above. The first
version of this PR re-ran it in K0 anyway; that vacuous re-run is dropped (R17.24). The interval
stays the percentile interval. No exact or score interval was declared.

## 3. The storage fix and its measured speed (§12, R17.18)

**What changed.** Each split is stored as one array `[roots, 65, 24 576]` float32. A training
window is one slice, `features[root, k : k + 61]`, and a batch is 16 slice copies into a
preallocated ring buffer. A prefetch thread fills the next batch while the GPU trains. The
sampler's draws happen in the thread in a fixed order.

**How it was measured.** `run_lewm_c1m_v2.py scale`, through `scripts/gpu_run.sh --wait
--min-free-gib 8 --board`. It runs Stage T's own `train_job_core` (via `run_tools.scale_probe`)
on synthetic features at the real sizes:
- 1 500 train roots, 9.6 GB in RAM, and 250 val roots;
- batch 16, T = 60, 600 updates, two val selections;
- model seed 66992 (debug).

The synthetic store was written to the scratchpad and removed afterwards.

| | run 1 (`f4b6b48`, `09628d90…2b52701`) | run 2 (`a98d893`, `27e4ada4…2ec301a`) | run 3 (`9e772e0`, `a505e621…abb2`) |
|---|---|---|---|
| gather alone, one batch of 16 windows (96 MB) | **3.6 ms** | **3.7 ms** | **3.6 ms** |
| per update, median (incl. the wait for the batch) | **0.223 s** | **0.166 s** | **0.219 s** |
| per update, 95th percentile | 0.277 s | 0.232 s | 0.262 s |
| time the training loop waited for a batch, 600 updates | 0.15 s | 0.14 s | — |
| one val selection (250 val roots) | not measured | not measured | 14.3–17.4 s |
| 1-minute load at start / end | not recorded | not recorded | 0.97 / 1.35 |
| peak process-tree PSS (ceiling 18 GiB, margin 2) | 12.64 GiB | 12.74 GiB | 12.69 GiB |
| the same N job twice (200 updates, seed 66993) | bit-identical | bit-identical | bit-identical |

**Reading.**
- **The gather itself is no longer a cost.** The draft's fancy-indexed gather took about 0.14 s
  for batch 16. The slice gather takes 3.6–3.7 ms in the prefetch thread, and the loop waits about
  0.2 ms per update.
- **The end-to-end gain is modest** (the #143 review). Per update is 0.166–0.223 s (median) and
  0.232–0.277 s (95th percentile), against the probe's 0.161 s for compute alone on a reused batch
  and the draft's estimate of about 0.30 s without the fix. The 95th percentile is close to that
  unfixed estimate.
- **Part of the update time is unexplained.** Run 3, on a quiet machine (load 0.97 at start),
  matched run 1, so run 2's faster 0.166 s is the outlier and the load does not explain it. The
  likely remainder over 0.161 s is the model's own per-update host copy and transfer of the batch
  (`train_step_features` copies it; no model file was changed); it is not measured separately.
  Every stage now records the load average at its start and end.
- **Revised Stage T band** (protocol §12, replacing "31–33 h with the fix"):
  - At the rule's minimum: 160 000 updates × 0.166–0.223 s is about **7.4–9.9 h**.
  - At the cap: 700 000 updates is about **32–43 h** (54 h at the 95th percentile).
- **The cap holds.** One job's worst case at the cap is 100 000 × 0.277 s = 27 700 s, plus 20
  selections of at most 17.4 s (348 s; measured in run 3, the first draft's
  `estimate_with_selection_s` was mislabelled and is replaced by this measured figure). The
  Stage T cap of 46 800 s is 1.67 × the sum, so it is kept (≥ 1.5 ×).

## 4. Debug smokes (debug seeds only; nothing read)

**The chain at `9e772e0` (after the #143 review's fixes).** Every stage ran in order on a clean
tree, as `outputs/task077-smoke-9e772e0/<stage>/report.json`. The `tests` stage and K0 ran the
full suite (1990 passed, 37 skipped); the later CPU stages ran with `--debug-skip-tests`
(recorded), since the first chain had already run G-tests in every CPU stage. The GPU jobs
verified the `tests` record, each in its own `gpu_run.sh` slot. Every stage after C checked the
artifact chain: the same corpus manifest (`2af6335d…`), the featurise and readouts reports, the
moments, R8, R-plate, the mean latent and both models' corpus and moments.

| stage | row | seconds | peak PSS (GiB) | sha256 |
|---|---|---|---|---|
| tests | TESTS-PASS | 143 | 3.03 | `c7a67816…1d3b` |
| scale | SCALE-PROBED | 349 | 12.80 (probe 12.69) | `a505e621…abb2` |
| k0 (K 66900–66903) | CAL-ESCALATE (4 resets) | 254 | 8.20 | `bfbe2b2c…b160` |
| corpus (40 roots) | CORPUS-SEALED (0 excluded) | 63 | 8.60 | `4ba7b0b4…761e` |
| featurise | FEATURISED | 7 | 1.41 | `9f82f564…94c8` |
| readouts | O-NO-BAR | 0.4 | — | `d736a452…48ea` |
| train cal-W / cal-N | T-JOB-DONE | 71 / 71 | 2.81 / 2.82 | `655c42c8…79f2` / `9dfd17da…d176` |
| plan | T-PLANNED | 2 | 0.67 | `d24e01e6…f794` |
| train W / N, 66992 | T-JOB-DONE | 37 / 37 | 2.74 / 2.82 | `d353c67a…436d` / `882a46ae…d4ee` |
| gates (and Stage T's row) | H-GATE-FAIL; `stage_t` T-DONE | 69 | 8.19 | `7542bcb2…b5f6` |
| closed D | L-DEV-STOP | 103 | 9.54 | `0fbfb7d5…e32f` |
| closed S | S-VOID-CEILING | 145 | 9.26 | `6b2de4ca…6577` |

No reset was refused before 405 in either chain (the refused-reset path is covered by the fake-pool
tests). The determinism re-run reproduced W's R-plate readings and commit targets exactly (0 m,
tolerance 0.1 cm), and no reading differed between arms on a reset in this chain.

**The first chain, at `a98d893` (before the review's fixes), kept beside it:** the same stages,
every CPU stage running the full suite first (1983 passed), sha256s in the record's first version
and in `outputs/task077-smoke-a98d893`. Its rows were the same.

**Renderer differences disclosed (R17.21).** In the first chain, the R-plate reading of the 405
frame differed between arms on the same reset twice, although the post-look frame (G-repro) and
the palm agreed:
- **seed 66920 (closed S):** N read (0.502269, −0.057836), every other arm (0.502208, −0.057844):
  6.1e-5 m; N's start latent sha differed from W's (`1a924562` against `63f2d121`);
- **seed 66910 (closed D):** L-mean read (0.517201, −0.101965), W, N and L-shuf (0.517185,
  −0.101966): 1.6e-5 m.

The probable cause is the EGL renderer's history in a worker (the effect TASK-073's render retry
handles for the post-look frame); the 405 frame is not retried. §5.2's premise is corrected: the
405 frame matches across arms only up to this nondeterminism. The determinism re-run's tolerance
is declared at 0.1 cm (R17.21), more than 15 × these differences and 10 × below τ_commit; the first
draft's 1e-6 m would have voided an S run on such a flake.

**Earlier smoke Vs (first chain), each fixed and committed before it:**
- K0 ran without EGL (`64b138e`).
- The report lacked the harness's `stages` section (`177aa04`).
- The corpus kept the plate only from 405 (`6a99778`).
- The runner read earlier stage reports without their debug suffix (`aebf5ec`).
- Debug D and S tripped G-memory with 6 workers: about 1.7 GiB PSS per worker, which loads
  DINOv2 and a world model. Stages G, D and S now use 4 workers (`WM_WORKERS`; TASK-073/074's
  `H_WORKERS`) (`bc0e4c6`).

**What the smokes show.**
- Every arm runs: W, N, L-shuf, L-mean, L-rand, H-rule, H-sysid, H-final(commit), H-read and
  H-now, plus K0's planted levels and proxies.
- `task_truth_in_controller` is 0 for every non-privileged arm.
- W's decision rolls 147 candidates plus up to 10 refinements on one CPU thread in about 9.7 s.
  An attempt takes at most 15.6 s (cap 300 s).
- The corpus root store has the declared shapes: 65 frames, 64 commands in [−1, 1], plate and
  palm at every kept step, and the plate-hidden render at r.
- The featurised store is the per-root contiguous layout that Stage T reads.

**Caps (§10.3).** The provisional caps are kept; each is at least 1.5 × the measured or scaled
worst case:
- K0: about 320 attempts of at most 8.3 s on 6 workers, plus about 6 min of preflight.
- D (its own cap, 7 200 s; R17.23): about 10 min.
- S (21 600 s): about 35 min.
- G (7 200 s): about 45 min, most of it the offline aims.
- Stage T: as in §3.
- **Not measured at scale:** Stage O's readouts on 1 750 roots (cap 7 200 s, CPU) and Stage C's
  2 000 roots (cap 14 400 s; C1-M did 1 024 roots in 39 min). Stage O's train moments are
  streamed from the file unmapped, so the 9.6 GB store does not enter the stage's PSS.

## 5. Open before K0, and before Stage O

- **A reviewer's GO for K0** on this PR, at its head revision. Nothing else is known to block it.
- **The evidence root.** K0 needs TASK-072 run-1's evidence at
  `/home/huhn/develop/emai/worktrees/task076-evidence`. It passed G-repro in every smoke.
- **K0 runs from a clean worktree of this PR's head** (`scripts/new_worktree.sh --run --from
  <rev>`). The command is `run_lewm_c1m_v2.py k0 --output <new> --evidence <root>`. Before
  starting, the main session posts a notice in chat.
- **Required before Stage O's GO (R17.24):** a scale probe of Stage O's readouts and of Stage C's
  per-root cost, on synthetic or debug data, that sets their caps to at least 1.5 × the measured
  worst case. *(Done: R17.28, §7 below. The featurisation's own scale probe, the #144
  approval's note 5, is done too: R17.37, protocol §7.4, 212.7 s for 1 995 synthetic roots
  against the 3 600 s cap.)*

## 6. The #143 review's changes (R17.20–R17.24)

| finding | change |
|---|---|
| 1, a reset refused before 405 | a concordant fail-fail pair for every arm, in the denominator, never a V; L-shuf's frame from the next reset in cohort order that reached 405; the determinism re-run checks such a reset for an identical refusal; refusals counted per arm in K0, D and S (`closed_core`, `foreign_reached`, `determinism_check`; fake-pool tests for D and S) |
| 2, the determinism tolerance | 0.1 cm on the R-plate reading and the commit target (R17.21); §5.2 corrected; the two differences disclosed (§4) |
| 3, the artifact chain | the corpus sha required in every later stage; the featurise and readouts reports read by row and debug flag; the moments, R8, R-plate, mean latent and models checked by sha256 (`artifact_chain`, `load_moments`, `load_readout`, `check_mean_latent`, `check_job`; tests for each) |
| 4, Stage D's cap | its own 7 200 s (`stage_cap`) |
| 5, CI | the training test runs on a reduced 2 × 2 latent (18 s to 5 s locally for the file) |
| non-blocking 1 | §10.3 now gives the measured 27 700 s; §13 the 54 h at the 95th percentile |
| 2 | the selection time is measured (14.3–17.4 s) and the estimate relabelled |
| 3 | the vacuous K0 re-run is dropped and the reason stated |
| 4 | `decide_t` emits T-DONE in Stage G's report; the per-job and plan rows are documented in §7 |
| 5 | `decide_g(debug=)`: the frozen ladder refuses the debug seed |
| 6 | Stage O no longer loads the gate split's table |
| 7 | a direct frozen-pin check in every stage's preflight |
| 8 | `check_code` also compares the recorded blobs with `c764ac9` when that commit is present |
| 9 | the static G-privileged scan covers `WorldModelAim` and every helper it calls |
| 10 | the storage-gain wording (§3) and the load average in every stage |
| 11 | the Stage O and Stage C scale probe is required before Stage O's GO (§5; protocol §7 step 5) |

## 7. Before the freeze: K0, the #143 approval's notes and the scale probes (R17.25–R17.29)

**K0 ran once** at `306fbdc` on the reviewer's reported GO and ended **K0-PASS**. Its values and
thin margins are in the protocol's §7.1 and in the frozen block's `K0_MEASURED`. §5's K0 items
are therefore done. The remaining items, all development work, ran at `86985a2` on a clean tree
on the Linux PC:

- **The feature-file check (the approval's note 1, R17.26).** `verify_feature_files` hashes
  every file of each split that a stage reads, and compares it with the featurise report's
  `files_sha256` before the stage reads it. The splits are: readouts and train, train + val;
  plan, val; gates, gate. Tests cover a tampered file, a missing file, an unrecorded split and
  the per-stage splits through `artifact_chain`.
- **The determinism rule (note 2, R17.27).** The re-run gates the R-plate reading at 0.1 cm, the
  commit or refusal, and the success outcome. The commit target's difference is reported only.
  - This existing smoke data does not characterise a target tolerance: two debug S re-runs of 4
    resets each (`a98d893` and `9e772e0`), both with zero difference.
  - That is why the rule is chosen rather than a measured tolerance.
- **The thread environment (R17.29).** `OPENBLAS_NUM_THREADS=16` with OMP and MKL at 6 is the
  declared G-threads environment. It comes from the owner ruling of 2026-09-29 (TASK-073), and
  TASK-076's K0 ran with it too. It is not a bug, so nothing is changed.
- **The scale probe of Stage O's readouts (R17.28).**
  - What ran: `run_lewm_c1m_v2.py oscale --output outputs/task077-oscale-1 --scratch <scratch>`,
    in the worktree `task077-freeze` (git-ignored), report sha256
    `ea09d23b649c20f760c408f02a0f827991c4adbf44f2cc5b0b3260dd2a3a048c`. It ran `readouts_core`
    itself through `run_tools.scale_probe` on synthetic features at the real sizes (1 500 train
    and 250 val roots, 12.1 GB), after the feature-file check. The run included in-run G-tests
    (1994 passed, 37 skipped).
  - Results: the feature-file check took 7.9 s and `readouts_core` 98.9 s, with a probe peak PSS
    of 2.96 GiB. The whole stage, including writing the synthetic store and G-tests, took 285 s.
    The cap stays at 7 200 s.
  - Its admission row on random features (O-NO-BAR) means nothing.
- **The Stage C probe (R17.28).** Two runs on the debug range only, in the worktree
  `task077-cprobe` at `86985a2`:
  - **The debug corpus stage:** 40 roots, 66940–66979, `--debug --debug-skip-tests`, 66 s,
    CORPUS-SEALED-DEBUG, report sha256 `e2cce1815ec310c22dfa50f9111dbb25248faf4ad8f62b9c1545f57d4599f988`.
  - **`scripts/probe_task077_stage_c.py`** on 100 roots, 66900–66999, report sha256
    `3ae3486ff4fead4d3d8c73d923501e7d3879ac596453244c02fcff4623487d28`. It uses the runner's
    own setup, cohort estimates, `collect` attempts and `write_root`.
    - It ran from an uncommitted copy whose code is the committed file before formatting. That
      copy's first launch had no `__main__` guard. Its spawned workers re-ran it and failed at
      start, so the load spiked to about 20 and nothing was simulated. The copy was fixed.
    - **Erratum 2026-10-05 (R17.33; the #144 approval's item 2).** A further launch,
      `task077-cprobe-3`, crashed in `sim_setup` → `hz.refit_p_readout` with
      `KeyError: 'stages'`, because the copy's report lacked `"stages"`. It had re-rendered
      TASK-072 evidence frames for G-repro, rendered no TASK-077 seed and read nothing. The copy
      was fixed again, and `task077-cprobe-4` is the probe cited here. This record first said
      the probe "ran after the load fell back under 2.0". In fact the load at its start was
      1.97 / 2.21 (see below): the 5-minute value was above 2.0.
  - **Results:**
    - Setup: 41 s, including G-repro.
    - Cohort estimates: 3.6 s per 100 seeds.
    - Collection: 0.48 s of wall time per root on 6 workers; attempts took 2.6 s median and
      3.24 s at most.
    - Size: about 0.61 MB per root.
    - Peak PSS: 8.46 GiB. There were no exclusions and no render disagreements.
    - Load at the start of the run: 1.97 / 2.21. The 5-minute value was just above the 2.0 rule,
      which binds gated stages, not this probe.
  - **The worst case for 2 000 roots** is about 1 400 s: every root at 3.24 s on 6 workers, plus
    writing, estimates, setup and G-tests. The 14 400 s cap is about 10 × that and stays. The
    estimates map's 1 800 s cap compares with about 72 s for 2 000 seeds. The corpus needs about
    1.2 GB of disk.
