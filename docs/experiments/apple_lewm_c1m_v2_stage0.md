# TASK-077 Stage 0: code, frozen block, simulations, storage fix and debug smokes

**Development only.** No seed of K, D, S or the corpus was simulated. Every simulated attempt
used the debug range 66900–66999, and every training run used synthetic features or that range's
smoke corpus. Nothing here is a result, and nothing is read from it. No K0 was run; K0 needs a
separate reviewer GO. The protocol ([apple_lewm_c1m_v2.md](apple_lewm_c1m_v2.md)) stays
**DRAFT**. Every choice below was **decided by Claude under owner delegation (2026-09-30)**
(R17.19 in [DECISIONS.md](../DECISIONS.md)).

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
| `tests/test_lewm_c1m_v2.py` | 34 tests (§7 step 1's list; listed below) |

**Guards new here.**
- **G-tests.** A CPU stage runs the full suite itself before it starts, and records the summary,
  exit status and timestamps. A GPU job verifies a `tests` stage report instead: the same
  revision as HEAD, exit 0, no failure, finished after HEAD's commit time.
- **G-sentinel.** Every stage report has a `fields` block whose entries start as
  `"not evaluated"`, and only the stage's own setter changes one. At the end, a row in a field
  that was never set is refused (the `early_verdict` bug).
- **G-frozen.** While the protocol is DRAFT, the runner refuses every non-debug stage except
  `tests`, `simulate`, `scale` and `k0`.

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
- a CPU training run with a tiny encoder, whose kept checkpoint is `select_checkpoint`'s.

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
count that bears on C is the ceiling N_K(0).
- K0's runner re-runs the non-inferiority simulation at a comparator rate of min(30/32, N_K(0))
  and reports it beside the proxies.
- The interval stays the percentile interval. No exact or score interval was declared.

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

| | run 1 (`f4b6b48`, sha256 `09628d90…2b52701`) | run 2 (`a98d893`, sha256 `27e4ada4…2ec301a`) |
|---|---|---|
| gather alone, one batch of 16 windows (96 MB) | **3.6 ms (26.5 GB/s)** | **3.7 ms (26.0 GB/s)** |
| per update, median (incl. the wait for the batch) | **0.223 s** | **0.166 s** |
| per update, 95th percentile | 0.277 s | 0.232 s |
| time the training loop waited for a batch, 600 updates | 0.15 s | 0.14 s |
| peak process-tree PSS (ceiling 18 GiB, margin 2) | 12.64 GiB | 12.74 GiB |
| the same N job twice (200 updates, seed 66993) | bit-identical | bit-identical |

**Reading.**
- **The gather is no longer a cost.** The draft's fancy-indexed gather took about 0.14 s for
  batch 16. The slice gather takes 3.7 ms and runs in the prefetch thread, so the loop waits
  about 0.2 ms per update.
- **Per update is 0.166–0.223 s**, against the probe's 0.161 s for compute alone on a reused
  batch.
  - The two runs differ by a third, and the cause is not established (the GPU stages do not
    record the CPU load). The cap below is checked against run 1's slower figures.
  - The likely remainder over 0.161 s is the model's own per-update host copy and transfer of
    the batch (`train_step_features` copies it; no model file was changed). This is not
    measured separately.
- **Revised Stage T band** (protocol §12, replacing "31–33 h with the fix"):
  - At the rule's minimum: 160 000 updates × 0.166–0.223 s is about **7.4–9.9 h**.
  - At the cap: 700 000 updates is about **32–43 h** (54 h at the 95th percentile).
- **The cap holds.** One job's worst case at the cap is 100 000 × 0.277 s = 27 700 s. The Stage
  T cap of 46 800 s is 1.69 × that, so it is kept (≥ 1.5 ×).

## 4. Debug smokes (debug seeds only; nothing read)

**The chain.** Every stage ran in order at revision `a98d893`, clean tree, as
`outputs/task077-smoke-a98d893/<stage>/report.json`:
- K0, the corpus and every later CPU stage each ran the full suite first (1983 passed, 37
  skips).
- The GPU jobs verified the `tests` record of that revision, each in its own `gpu_run.sh` slot.

**Earlier smoke runs that ended V.** The fixes, each committed before the chain:
- K0 ran without EGL (`64b138e`).
- The report lacked the harness's `stages` section (`177aa04`).
- The corpus kept the plate only from 405 (`6a99778`).
- The runner read earlier stage reports without their debug suffix (`aebf5ec`).
- Debug D and S tripped G-memory with 6 workers: about 1.7 GiB PSS per worker, which loads
  DINOv2 and a world model. Stages G, D and S now use 4 workers (`WM_WORKERS`; TASK-073/074's
  `H_WORKERS`) (`bc0e4c6`).

| stage | debug size | row | seconds | peak PSS (GiB) | sha256 |
|---|---|---|---|---|---|
| tests | — | TESTS-PASS | 156 | — | `8fa168c5…99bc8` |
| scale | synthetic | SCALE-PROBED | 256 | 12.74 | `27e4ada4…2ec301a` |
| k0 | K 66900–66903, 6 levels + 4 proxies | CAL-ESCALATE (4 resets) | 275 | 8.24 | `5e616d03…65c793` |
| corpus | 40 roots 66940–66979 (30/5/5) | CORPUS-SEALED (0 excluded) | 219 | 8.27 | `bec3248a…ecf9` |
| featurise | 2 640 frames, CUDA | FEATURISED (G-anchor 1.1e-4 ≤ 1e-3) | 7 | 1.38 | `1e1540ef…94f` |
| readouts | 35 roots, 5 folds | O-NO-BAR | 156 | 7.75 | `4538d902…8fde` |
| train cal-W / cal-N | 400 updates | T-JOB-DONE | 71 / 71 | 2.81 / 2.79 | `43220321…80d` / `44d715c2…af72` |
| plan | 5 val roots | T-PLANNED | 158 | 7.77 | `0fc47226…02db` |
| train W / N, 66992 | 200 updates | T-JOB-DONE | 37 / 38 | 2.79 / 2.82 | `4885cb3e…5ec4` / `69aa7345…13ae` |
| gates | 5 gate roots, 1 seed | H-GATE-FAIL | 226 | 8.64 | `e88f2e2a…24c3` |
| closed D | 4 resets × 5 arms | L-DEV-STOP | 259 | 9.77 | `db714126…8068` |
| closed S | 4 resets × 10 arms + determinism re-run | S-VOID-CEILING | 302 | 9.78 | `a825a300…652a` |

**What the smokes exercise.** The rows above are mechanics on 4–40 debug items and
200–400-update models; they are not read.

**What they show:**
- Every arm runs: W, N, L-shuf, L-mean, L-rand, H-rule, H-sysid, H-final(commit), H-read and
  H-now, plus K0's planted levels and proxies.
- `task_truth_in_controller` is 0 for every non-privileged arm.
- W's decision rolls 147 candidates plus up to 10 refinements on one CPU thread in about 9.7 s.
  An attempt takes at most 15.6 s (cap 300 s).
- The determinism re-run of W reproduced its aims exactly (0 m).
- The corpus root store has the declared shapes: 65 frames, 64 commands in [−1, 1], plate and
  palm at every kept step, and the plate-hidden render at r.
- The featurised store is the per-root contiguous layout that Stage T reads.

**Caps (§10.3).** The provisional caps are kept; each is at least 1.5 × the measured or scaled
worst case:
- K0: about 320 attempts of at most 8.3 s on 6 workers, plus about 6 min of preflight.
- D: about 10 min.
- S: about 35 min (cap 21 600 s).
- G: about 45 min, most of it the offline aims (cap 7 200 s).
- Stage T: as in §3.
- **Not measured at scale:** Stage O's readouts on 1 750 roots (cap 7 200 s, CPU) and Stage C's
  2 000 roots (cap 14 400 s; C1-M did 1 024 roots in 39 min). Stage O's train moments are now
  streamed from the file unmapped, so the 9.6 GB store does not enter the stage's PSS. A scale
  probe of the readouts on synthetic features is recommended before Stage O's GO; it is not a K0
  prerequisite.

## 5. Open before K0

- **A reviewer's GO for K0** on this PR, at its head revision.
- **The evidence root.** K0 needs TASK-072 run-1's evidence at
  `/home/huhn/develop/emai/worktrees/task076-evidence`. It passed G-repro in every smoke.
- **K0 runs from a clean worktree of this PR's head** (`scripts/new_worktree.sh --run --from
  <rev>`). The command is `run_lewm_c1m_v2.py k0 --output <new> --evidence <root>`. Before
  starting, the main session posts a notice in chat.
