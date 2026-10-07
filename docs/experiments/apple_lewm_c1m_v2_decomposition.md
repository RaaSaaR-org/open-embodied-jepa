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

**Outcome: D-READOUT** (§2): a readout refitted on W's predicted latents under the stand-in
reads the plate at r within τ_commit on every seed (val medians 0.556–0.578 cm, upper bounds
0.612–0.636 cm), while the frozen R8 reads the same predictions at 2.86–3.41 cm. The direction it
names is a readout on predicted latents, through a new preregistration on fresh roots.

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

## 2. Result (R17.54)

**Row: D-READOUT** (first match of §1.5; one run). On the val roots, a dual ridge refitted on W's
own predicted latents at r reads the plate at **0.578, 0.564 and 0.556 cm** in median under the
stand-in chunks (upper bounds 0.636, 0.625 and 0.612 cm, against τ_commit = 1.0 cm), and at
0.34–0.38 of N's refitted error (upper bounds 0.40–0.45) on every seed. The frozen R8, fitted on encoded frames, reads the
same predictions at 2.86–3.41 cm. **The plate information survives W's 60-step roll-out; what
missed G5 (a) was the readout's distribution shift**, not the prediction. The direction the rule
names is **a readout on predicted latents** (§2.6). This is a development reading on the train
and val splits with checkpoints selected on val; it gates nothing and changes no TASK-077 row,
clause or R7.

§1 was committed in `ab6d73f` (2026-10-07 16:19:38Z) and pushed before any number was computed;
the run used that commit, unchanged.

### 2.1 The run

| item | value |
|---|---|
| revision | `ab6d73f0932579f0672c139cc01589c8cb145dc3`, clean tracked tree, unchanged to the end |
| script | `scripts/decompose_task077.py`, sha256 `6010a498bee81805a79415cdf2457291e214455d7184a4fb7e6fafa01f68c8b9` |
| report | `outputs/task077-decomp-1/report.json` in `/home/huhn/develop/emai/worktrees/task077-decomp`, sha256 `8a7742609462ce161029860af102022cda169843f8ca159422c8624ae5dd9c5a`; log sha256 `5119bc50…9c5a` |
| span | 2026-10-07 16:30:19Z–16:43:51Z, 812 s; load 0.11 / 0.15 at the start; peak process-tree PSS 5.88 GiB (ceiling 12) |
| inputs | the 14 train and val feature files (verified against the featurise report `f187c7c8…e9ee`), corpus `ad8974b2…43fb`; moments `5415eea4…`, R8 `62a5ea8a…`; the six checkpoints at their Stage T sha256s (§7.8); evidence root `task076-evidence` checked |
| roots | 1 495 train + 250 val; Stage O's outer folds, 349 roots each |
| stand-in chunks | 1 745 roots in 8.2 s on 4 workers; **0 infeasible** |
| threads | `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=16`, `MKL_DYNAMIC=FALSE`; one torch thread |

A 40-root smoke (`outputs/task077-decomp-smoke-1`, 20 roots per split, the same commit) ran
first to check the mechanics; its row (D-VOID-CEILING on 40 roots) is not read. The log holds
MuJoCo EGL `__del__` tracebacks from the stand-in workers' interpreter teardown, as Stage G's did.
Evidence: `~/develop/emai/evidence/task077-decomp/` with `_checksums/task077-decomp.sha256`
(`5e2a9e88…7f29`), verified against source and copy.

### 2.2 The rule's inputs (val roots, h = 60; medians in cm, 95 % intervals)

| statistic | 66800 † | 66801 | 66802 | bar |
|---|---|---|---|---|
| E_60, encoded (the ceiling) | 0.410 [0.379, 0.440] | (same) | (same) | upper ≤ 1.0 (D-VOID-CEILING) |
| **S_60**, refit on W's stand-in predictions | **0.578 [0.535, 0.636]** | **0.564 [0.529, 0.625]** | **0.556 [0.503, 0.612]** | upper ≤ 1.0 |
| **S_60 / N_60** | **0.344 [0.313, 0.413]** | **0.381 [0.343, 0.451]** | **0.343 [0.303, 0.398]** | upper < 1.0 |
| X_60, refit on W's executed-command predictions | 0.530 [0.470, 0.564] | 0.473 [0.435, 0.512] | 0.565 [0.512, 0.622] | (D-COMMAND) |
| X_60 / N_60 | 0.316 [0.278, 0.380] | 0.320 [0.285, 0.372] | 0.349 [0.313, 0.404] | (D-COMMAND) |
| N_60, refit on N's predictions | 1.680 [1.409, 1.785] | 1.481 [1.304, 1.617] | 1.620 [1.437, 1.723] | — |

† W-66800's `last_two_triggered` is true (results §3, caveat 1).

Every check of the rule holds on every seed (S_60, X_60, X_16 and X_30 all meet both bars), so the
first match is D-READOUT; D-TASK's sub-reading is not reached.

### 2.3 Reported beside the rule (val unless stated)

- **The frozen R8 on the same predictions** (the G5 (a) statistic on train and val): executed
  1.561 [1.393, 1.674], 1.587 [1.481, 1.722] and 1.774 [1.643, 1.943] cm; stand-in 3.412, 2.861
  and 2.972 cm. On train: 1.547, 1.617, 1.785 and 3.457, 2.945, 3.061 cm. These are close to the
  gate split's 1.58–1.87 and 3.09–3.63 cm, so val and train do not look easier than the gate
  split for this statistic.
- **Refit minus encoded at r (X_60 − E_60):** +0.120 [0.053, 0.168], +0.063 [0.019, 0.109] and
  +0.155 [0.098, 0.211] cm. **Stand-in minus executed, both refit (S_60 − X_60):** +0.048
  [−0.008, 0.125], +0.091 [0.041, 0.162] and −0.009 [−0.086, 0.063] cm. With refit readouts the
  stand-in costs at most about 0.1 cm in median, against +1.29 to +2.04 cm with the frozen R8 on
  the gate split.
- **XS: a readout fitted on executed-command predictions, read on stand-in predictions:** 5.45,
  9.47 and 4.62 cm at h = 60 (1.27–1.93 cm at h = 16 and 30). **The refit readout does not
  transfer across command distributions**: it must be fitted on predictions made under the same
  kind of commands it will read.
- **Train roots** (in sample for W): X_60 0.547, 0.504, 0.598; S_60 0.603, 0.590, 0.607; N_60
  1.255, 0.853, 1.243 cm. W's train and val readings are close; N's refit reads train roots
  better than val roots.

### 2.4 Per horizon (measurement 2; val; medians in cm)

| h | E_h (encoded) | X_h: 66800 †, 66801, 66802 | S_h | N_h |
|---|---|---|---|---|
| 16 | 0.249 [0.225, 0.276] | 0.235, 0.232, 0.223 | 0.296, 0.326, 0.307 | 0.429, 0.360, 0.370 |
| 30 | 0.338 [0.306, 0.395] | 0.289, 0.281, 0.286 | 0.389, 0.425, 0.364 | 0.676, 0.580, 0.609 |
| 60 | 0.410 [0.379, 0.440] | 0.530, 0.473, 0.565 | 0.578, 0.564, 0.556 | 1.680, 1.481, 1.620 |

- X_h's upper bounds are at most 0.622 cm at every horizon; every X_h / N_h upper bound is below
  0.73. At h = 16 the S_h / N_h upper bound for 66801 is 1.004 (not part of the rule, which reads
  S at h = 60 only).
- **At h = 16 and 30 the refit predicted latent reads the plate no worse than the encoded frame**
  (X − E: −0.015, −0.018, −0.027 cm at h = 16, intervals including 0; −0.050, −0.057, −0.052 cm
  at h = 30, upper bounds −0.007, −0.014, −0.004). One reading, not a measurement: the prediction
  is a function of the 405 frame, where R-plate reads the plate to 0.15 cm, and of the commands,
  so it may carry less frame-specific nuisance than an encoded later frame. At h = 60 W is
  0.06–0.16 cm worse than the encoded frame.
- The error grows with the horizon for every source; W's grows from about 0.23 to 0.47–0.57 cm,
  N's from 0.36–0.43 to 1.48–1.68 cm.

### 2.5 The stand-in's command mismatch (measurement 3; reported only)

- **Size.** The per-root RMS of (stand-in − executed) over the 60 × 14 normalised commands has
  median 0.112 [0.109, 0.114] on val (0.113 on train). It sits in the six commanded dimensions
  6–11 (RMS 0.231, 0.218, 0.230, 0.042, 0.102, 0.034; the other eight are 0 in both). In dimensions
  9–11 the stand-in barely varies (std 0.006, 0.013, 0.005) where the executed commands do (0.030,
  0.020, 0.025); in 6–8 the stand-in's std is 0.173, 0.168, 0.136 against 0.118, 0.179, 0.123.
- **Cost against size.** The per-root mismatch spans a narrow range (val quartile edges 0.098,
  0.112, 0.117) and does not predict W's per-root cost. Spearman ρ of m with d = e_S − e_X on val:
  −0.035, −0.016, −0.021 with refit readouts and +0.106, +0.077, −0.070 with the frozen R8 (every
  interval includes 0; on all 1 745 roots |ρ| ≤ 0.163). With the frozen R8 the cost is about
  +1.0 to +2.0 cm in every quartile of m; with refit readouts it is −0.04 to +0.25 cm.
- **Reading** (not a measurement): the stand-in's cost under the frozen R8 is a shift common to
  all roots, which moves the predicted latents away from where R8 was fitted. A readout fitted on
  stand-in predictions absorbs it (§2.3, S − X). No command-matching change is indicated by this
  measurement, which is consistent with the rule's row.

### 2.6 G1 (iii)'s bootstrap on val (reported only; results §3, caveat 4)

| seed | point | percentile interval | resample mean |
|---|---|---|---|
| 66800 † | 0.164 | [0.168, 0.193] | 0.180 |
| 66801 | 0.109 | [0.104, 0.135] | 0.119 |
| 66802 | 0.176 | [0.179, 0.209] | 0.194 |

The encoded latents' effective rank falls from 25.77 on the 250 val roots to 22.22 on average
over the resamples (−14 %), W's from 7.51 to 7.25 and N's from 3.29 to 3.25 (66800; the others
alike). The ratio difference (W − N)/encoded therefore rises under resampling, and the point
estimate sits below the interval on two of three seeds, as on the gate split. This measures the
shift the results document explained from the code; it decides nothing.

### 2.7 What the row names, and the recommended next task

**D-READOUT: a readout on predicted latents.** The recommended next task is **a new DRAFT
preregistration (the next free number, TASK-080; TASK-078 and TASK-079 are reserved in PLAN.md)**,
under the same C1-M condition, in which W's controller reads the plate with a dual ridge fitted on
W's own predicted latents at r **under stand-in chunks**, not with R8. It needs its own review,
freeze and GOs. Points this record hands to that draft, from the numbers above:
1. **Fit the readout on stand-in predictions** across the box (as the corpus's uniform aims are),
   because a readout fitted on executed-command predictions reads stand-in predictions at
   4.6–9.5 cm (§2.3). The refinement's chunks must come from the same stand-in.
2. **Fresh gated roots.** The gate split has been read; this record used train and val, and the
   kept checkpoints were selected on val. Whether the six checkpoints are reused or retrained is
   that draft's choice; reusing them needs a fresh offline gate split from new corpus roots.
3. **A ranking check before any closed loop.** This record measures the readability of the
   committed aim's predicted plate, not whether p̃(g) ranks 147 candidates correctly. The draft
   should re-gate G5 with the new readout on fresh roots and repeat the offline aims (§7 step 7)
   before Stage D.
4. **The corpus's aims are built from the true plate**; the closed loop builds its grid from
   R-plate's reading (0.15 cm at 405). A readout fitted on corpus predictions may use the aim's
   relation to the true plate, which the closed loop carries only through p̂. The draft should
   state this and measure it offline.
5. **N stays a real twin.** N's refit reads 1.48–1.68 cm at r; W's is 0.32–0.38 of N's.
   The twin tests remain meaningful with refit readouts for both.

D-COMMAND and D-TASK are not chosen. The per-horizon numbers (§2.4) would have met D-TASK/H's
bars as well, which is recorded but not used.

### 2.8 What this does not show

- Not a closed-loop result, not a gate, and not evidence that the remedy works: the readouts are
  mostly fitted on train-root predictions (in sample for W), and the checkpoints were selected on
  val. R7 is unchanged, and no LeWM-driven controller has run in closed loop on v2.
- Nothing about the ranking of candidates, the offline aims or the predicted counts under the new
  readout (point 3 above).
- Nothing reopens TASK-077: its row stays G-NO-BAR, its clause did not fire, and changing its
  readout after seeing the gate numbers is exactly why a new preregistration on fresh roots is
  needed.
- One run, one corpus, one camera, one encoder, in simulation.
