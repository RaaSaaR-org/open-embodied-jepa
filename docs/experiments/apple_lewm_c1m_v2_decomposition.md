# TASK-077 decomposition record: where W's predicted-plate error comes from (development)

**What this is.** The development record that R17.51 recommends after TASK-077's Stage G ended
G-NO-BAR ([results](apple_lewm_c1m_v2_results.md), R17.50–R17.52). It decomposes G5 (a)'s miss
into a readout term, a horizon term and a command-mismatch term, using **the six kept Stage T
checkpoints as they are**, on the **train and val splits of `apple-c1m-v2` only**, on the CPU.
There is **no new training, no new collection, no closed loop and no gate-split read**. It gates
no claim and changes no row of TASK-077. Its only output is the choice of the next direction, by
a rule declared in §1.5 before any number below was computed.

Every choice here is **decided by Claude under owner delegation (2026-09-30)**. The declaration is
R17.53 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-05 (b), entered **in the same commit
as this section, before the run**. The run's ruling, which applies the rule, is R17.54.

The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged by this record. Nothing
here runs a controller.

## 1. Declarations (written and pushed before the run)

### 1.1 Inputs (read only; every one checked by sha256)

| input | where (under `/home/huhn/develop/emai/worktrees/`) | check |
|---|---|---|
| featurisation, **train and val only** | `task077-stageo/outputs/task077-featurise-1/features/` (`features8_{train,val}.npy`, `commands_*`, `roots_*`, `table_*`) | every file of the two splits against the featurise report's `files_sha256` (`off.verify_feature_files`, splits train and val); the report's corpus sha256 is `ad8974b2…43fb` |
| normalisation moments and R8 | `task077-stageo/outputs/task077-readouts-1/fits/` | content sha256 against the readouts report (`5415eea4…`, `62a5ea8a…`) |
| the six models | `task077-staget/outputs/task077-t-W-66800-1/`, `task077-staget2/outputs/task077-t-{N-66800-2,W-66801-1,N-66801-1,W-66802-1,N-66802-1}/` | each report T-JOB-DONE; each checkpoint against its report's sha256 (`tr.load_model`) |
| P-3's checkpoint (the stand-in workers' robot setup only) | `task076-evidence` | `hz.check_evidence` |

**The gate split is never opened.** The script refuses any path that names it, and the record
lists every file it opened.

### 1.2 What is computed

The 1 745 train + val roots (1 495 train, 250 val), in the featurisation's order (train, then val).
For each root, frame 405 (index 2) is the start; h ∈ {16, 30, 60}; the target is the true plate at
step 405 + h from the corpus table. Everything runs on the CPU in float32 with one torch thread per
process, as Stage G did.

1. **Stand-in chunks.** For each root, the kinematic stand-in's 60 commands for that root's own
   committed aim (`target`), from its logged 405 state (`lewm_c1m_v2_runtime.run_chunks_task`,
   `place_planner.primitive_chunks`, horizon 60), exactly as Stage G computed them for the gate
   roots. A root whose chunk is marked infeasible is counted and kept, as in Stage G.
2. **Roll-outs**, for each seed s ∈ {66800, 66801, 66802}: W^s from the encoded 405 latent under
   (a) the root's executed commands 405–464 and (b) its stand-in chunk; N^s under zero commands.
   The predicted latents at h = 16, 30 and 60 are kept (raw feature space, as R8 reads them).
3. **Readouts**, each a dual ridge of Stage O's form (`lewm_c1m_v2_offline.ridge`: the relative λ
   grid, 5 inner folds grouped by root, salt 8105), **cross-fitted over the 1 745 roots in Stage
   O's 5 outer folds** (`lm.outer_folds(1745)`, salt 8104; the same assignment Stage O used),
   target the plate at 405 + h:
   - **E_h**: the encoded 8 × 8 latent at 405 + h (the ceiling at that frame);
   - **X_h^s**: fitted and evaluated on W^s's predictions under the executed commands;
   - **S_h^s**: fitted and evaluated on W^s's predictions under the stand-in chunks;
   - **N_h^s**: fitted and evaluated on N^s's predictions;
   - reported only: **XS_h^s**, fitted on the executed-command predictions of the fold's fit roots
     and evaluated on the held-out roots' stand-in predictions (the stand-in's cost with the
     readout fixed); and the frozen R8 (Stage O's fit on the encoded train frames at r) on W^s's
     predictions at h = 60 under both command sets (the G5 (a) statistic, on train and val).
4. **Statistics.** Median plate error in cm with a root-bootstrap 95 % interval
   (`lm.median_ci`, 10 000 resamples, salt 8106); paired ratios and differences of medians
   (`lm.median_ratio_ci`, `lm.median_difference_ci`). **The rule reads the 250 val roots' held-out
   errors only.** The train roots' errors are reported separately: W was trained on train
   windows, so its predictions there are in sample.
5. **The stand-in's command mismatch** (measurement 3; reported only). Per root, m = the RMS over
   the 60 × 14 normalised commands of (stand-in − executed), and the per-dimension RMS. W's cost
   per root is d = e_S − e_X (with the refit readouts, and with the frozen R8). Reported: the
   Spearman correlation of m with d (root-bootstrap interval, salt 8106), and the medians of e_X,
   e_S and d by quartile of m, on val and on all roots.
6. **Reported only: G1 (iii)'s bootstrap on val.** The results document (§3, caveat 4) explains
   from the code why G1 (iii)'s point estimate fell below its own percentile interval on the gate
   split. This record recomputes the same statistic (`comparative_rank`'s resampling, 2 000
   resamples, salt 8106) on the val split's h = 60 predictions and reports the point estimate,
   the mean of the resamples and the interval for each seed, to show the shift on data that is
   not the gate split. It decides nothing.

### 1.3 Bars, tied to τ_commit

τ_commit = **1.0 cm** (K0, measured; protocol §7.1), the bar G5 (a) used. The ratio bar against N
is G5 (b)'s definitional < 1.0. No other bar is used. The 1.5 τ_commit scaled tolerance (§8.2) is
reported beside each statistic, not used by the rule (as in G5).

### 1.4 What the measurement can and cannot say (stated before the run)

- **The val roots are not out of sample for selection.** The kept checkpoints were selected on
  val by their latent MSE, and G1's bars were calibrated on val. The val roots' predictions were
  never trained on.
- **The refit readouts are mostly fitted on train-root predictions**, which are in sample for W
  and may be cleaner than predictions on fresh roots. So a pass here is an optimistic reading;
  it chooses a direction, it is not evidence that a remedy works. Any remedy needs its own
  preregistration and fresh gated roots (the gate split has been read).
- **The stand-in chunks here are those of each root's own committed aim**, as in Stage G. In the
  closed loop W ranks 147 candidates' chunks; this measures the readability of the predicted
  plate under the stand-in, not the ranking.
- One run, one corpus, one camera, one encoder; development only.

### 1.5 The decision rule (declared before any number; first match)

All three seeds must satisfy a condition for it to hold. "ub" is the upper 95 % bound on the val
roots.

| row | condition | the direction it names |
|---|---|---|
| **D-VOID-CEILING** | ub of median E_60 > τ_commit on val (the encoded readout itself misses, so nothing below is informative) | escalate to the owner; no direction is chosen |
| **D-READOUT** | for every seed: ub of median S_60^s ≤ τ_commit **and** ub of median S_60^s / median N_60^s < 1.0 | **a readout on predicted latents**: a new preregistration whose controller reads the plate with a readout fitted on W's predicted latents under the stand-in at r, on fresh gated roots |
| **D-COMMAND** | not D-READOUT; for every seed: ub of median X_60^s ≤ τ_commit **and** ub of median X_60^s / median N_60^s < 1.0 | **a command-matching change**: a design note that removes the stand-in's mismatch (for example W evaluated on commands closer to the executed ones, or trained on stand-in commands), with its own measurement before any preregistration |
| **D-TASK** | otherwise | **a declared task change**, through a design note before any preregistration |

**D-TASK's sub-reading** (part of the rule; it names which task change the recommendation
describes, and no other row depends on it):
- **D-TASK/H**: for some h ∈ {16, 30}, on every seed, ub of median X_h^s ≤ τ_commit and ub of
  median X_h^s / median N_h^s < 1.0. The predicted plate survives a shorter roll-out, so the task
  change to describe is a later commit step (a shorter horizon), which the protocol counts as a
  task change (§11).
- **D-TASK/R**: no such h. The plate is not readable from W's prediction within τ_commit even at
  16 steps, so the task change to describe is one whose decision does not need a sub-τ_commit
  reading of the predicted plate.

The rule never reopens a TASK-077 row, never fires or narrows its clause, and never changes R7.

### 1.6 Cost, guards and evidence

- CPU only, no GPU lock. Estimated 20–40 min: about 15 700 roll-outs (0.055 s each on one
  thread, protocol §15), about 135 ridge fits, the stand-in chunks on 4 worker processes.
- The run starts from a clean tracked tree at a pushed commit; it records the revision, the
  script's sha256, every input's sha256, the seeds and salts, the thread environment (the
  protocol's G-threads: `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=16`,
  `MKL_DYNAMIC=FALSE`, one torch thread), the load at the start, the peak process-tree PSS
  (ceiling 12 GiB, as Stage G) and the wall time. A failed check stops the run with no row.
- The report goes to `outputs/task077-decomp-1/report.json` in the run worktree (git-ignored),
  and is copied with its log to `~/develop/emai/evidence/task077-decomp/` with a sha256
  manifest in `_checksums/`.
- The run is done once. If it fails before writing its row, the cause is recorded and it is
  repeated once in a new output folder.

## 2. Result

*(Filled in after the run, by R17.54.)*
