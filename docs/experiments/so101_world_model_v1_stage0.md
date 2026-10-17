# TASK-100 Stage 0: code, D0 calibration, debug smoke and determinism (record)

**Status: Stage 0 record; the protocol is still DRAFT and freezes when this PR merges.**
Protocol: [so101_world_model_v1.md](so101_world_model_v1.md) §10. Rulings R28.27–R28.33 in
[DECISIONS.md](../DECISIONS.md), decision 2026-10-17 (br), each **decided by Claude under owner
delegation**. **Nothing here is a result.** Every calibration read **D0 only** (train episodes of
shards 00–02: D0-fit = shards 00–01, 180 episodes; D0-eval = shard 02, 90 episodes). The smoke
read val episodes as stand-ins for val and test to exercise the code path; its numbers set
nothing and are not reported as results. **The test split was never opened**, decoded or
featurised. No gated model has been trained. R7 does not change (R28.9).

**Stage 0 row: S0-DONE — no Stage 0 rule escalates.** The grid rule chose g4; G1's bars came from
calibration L and survived the truncation check unraised; calibration L's G4 lower bounds are
above 1.10 and its R-pred medians are inside both G5 bars at h = 8 and h = 16, so no owner
ruling on G4 or G5 is needed before the freeze; neither moving subset is halved; the CUDA
determinism check is bit-identical; Stage 0 used **1.98 GPU h** of its 4 GPU h cap; the
projection of the gated stages fits the 20 GPU h hard cap (§6 below).

## 1. Code (new files only)

| file | what |
|---|---|
| `src/embodied_jepa/so101_wm.py` | NumPy: constants, the fit sample, D0 and the smoke views, roots, windows, the wrong and candidate tables, the cluster bootstrap (sums, weighted medians), G1's statistics (Gram spectrum, truncation, G1 (iii)'s projected per-episode moments), the ridge readouts (primal/dual, grouped CV), the subset, grid and G5-bar rules, G6, the budget rule, the gates, the rows, the frozen Stage 0 choices (`CHOSEN_GRID`, `BARS`, `MOVING`) and the composed files' pins |
| `src/embodied_jepa/so101_wm_data.py` | corpus reading (both cameras; labels from the sidecar truth and `observation.state`), the frozen DINOv2 featuriser at 224 px (no resize), g8's projection, the feature stores, the flat-latent standardisation, the window sampler |
| `src/embodied_jepa/so101_wm_runtime.py` | runner helpers: report header, clean-tree and frozen-manifest guards, model building and loading, chunked roll-outs, §5's val criterion *(a module not named in §10.1; disclosed)* |
| `src/embodied_jepa/models/schema_frozen_tokens.py` | §4: TASK-066's frozen-token composition in front of S0's `SchemaLeWM` / `SchemaNativeJEPA`, action width 6, 32 × 384 (g4) or 128 × 96 (g8) tokens, cached standardised features only |
| `scripts/featurise_task100.py` | stages `stage0`, `trainval` (F), `test` (F′, refuses without the val report's sha256) and `fit` (CPU: standardisation and G1 (iii)'s basis) |
| `scripts/train_task100.py` | one job per call: `job`, `probe` (§8.1 and its budget), `calibration`, `debug`; `--resume … --extend-to` is G6's one extension |
| `scripts/evaluate_task100.py` | passes `fitpred` (GPU), `readouts` (CPU), `evalpred` (GPU), `score` (CPU); purposes `grid`, `calibrate`, `smoke`, `val`, `test` |
| `scripts/summarize_task100.py` | Markdown tables from a score report *(a script not named in §10.1; disclosed)* |
| `scripts/run_task100_stage.sh` | every Stage 0 and gated stage, one `gpu_run.sh --wait --board` lock per GPU job |
| `tests/test_so101_wm.py` | 36 tests on tiny synthetic data, CPU (below) |
| `benchmarks/manifests/so101-world-model-v1.json` | the freeze: SHA-256 pins of the files above, of the composed existing files, the Stage 0 choices and the Stage 0 evidence hashes |

No existing file changed: `git diff --name-status origin/main` lists additions only (and the
documents), `config.py` is untouched, and the composed files (`models/frozen_tokens.py`,
`models/frozen_encoder.py`, `models/base.py`, `models/lewm.py`, `models/native.py`,
`models/schema_backends.py`, `models/readout.py`, `readout_labels.py`, `pretrained_encoder.py`,
`contracts.py`, `action_schemas.py`, `schema_data.py`, `run_tools.py`, `devices.py`,
`so101_play_corpus.py`, `configs/so101_sim_action.json`) match their `origin/main` bytes (pinned in
`so101_wm.COMPOSED_FILES_SHA256`; a test checks them). Every gated stage refuses to start unless the
tracked tree is clean and every pinned file matches the frozen manifest (G-frozen).

**Tests** (`tests/test_so101_wm.py`): windows stay inside episodes; the root rule (37 roots per
episode, start + 32 ≤ 400); the wrong and candidate tables use other episodes, have the declared
classes and are fixed by their salts; ties count against; the gripper rate clip; hold commands;
the cluster bootstrap on a known case (unit weights reproduce NumPy medians; identical episodes
give a degenerate interval; the draws are fixed by salt 10014); the full-width G1 statistics and
truncation; **G1 (iii)'s per-episode moments reproduce the direct projected rank ratios**, also
under a resample's counts; G1's bars and rounding; the ridge's primal and dual forms are one
estimator, folds are episode-grouped; the label units; the moving-subset halving rule; the grid
rule in both directions; the G5 bars; G6 and its single extension; the budget rule lowering U and
escalating; **every gate and every row in both directions** (VOID, INCONCLUSIVE, WM-PASS with S5's
model, WM-PARTIAL by G6 or G5, WM-FAIL); the feature store, its views (D0-fit, D0-eval, the fit
sample is train seeds divisible by 4) and **the test-split guard**; the sampler's declared stream,
its resume and calibration N_L's shared stream; g8's projection against a direct fit; **a 6-D
schema through the frozen-token class for both backends** (`train_step_features`, prediction
shapes, a 14-D action refused); the backend swap is one key; **checkpoints refuse another
implementation hash** and other provenance; the composed files match their pins; only additions
against `origin/main` under `src/` and `configs/`; `import embodied_jepa` stays torch- and
MuJoCo-free; seeds and salts as declared.

## 2. Featurisation probe (§10.2)

Both encoders on D0 (270 episodes) and the smoke's 40 val episodes, 401 frames × 2 cameras each.

| | pretrained (g4 and g8, with g8's pass A) | random-init (g4) |
|---|---|---|
| images | 248 620 (pass B) + 144 360 (pass A) | 248 620 |
| encoder throughput in the model, images/s | 532 | 932 |
| wall-clock images/s (pass B, decoding included) | 294 | 847 |
| GPU against CPU on the first 64 frames, max \|Δ\| (relative to the token std) | g4 4.4e-5 (2.2e-5), g8 1.7e-4 (7.8e-5) | g4 1.3e-6 (1.5e-6), g8 2.8e-6 (3.0e-6) |
| one episode featurised twice on the GPU | bit-identical (g4, g8) | bit-identical (g4, g8) |
| lock time | 1 509 s | 456 s |

**The pretrained run's throughput is not a clean measurement**: a CPU dry-run of the pipeline
(mine) ran beside it and halved the decoders' share of the CPU. The random-init run had the
machine to itself; the encoder is the same architecture, so **F is projected from its rate**:
0.948 s an episode, 2 850 train and val episodes per encoder at the chosen grid g4 (one pass, no
projection) plus about 150 s of corpus verification, **0.79 GPU h per encoder, 1.58 GPU h for
both**.

**g8's projection** (fitted on D0-fit, 72 180 frames, 128 slots per frame): k = 96 of 384 channels
keep **0.801** of the standardised channel variance. Encoder digests: pretrained `3a697b87…`,
random-init `546b9011…`. The flat-latent standardisation (fitted on every D0-fit frame) has no
dimension at the 1e-6 floor for either encoder. G1 (iii)'s 256-direction basis (fitted on every
D0-fit frame, pretrained) keeps 0.839 of the standardised latent's variance at g4 (0.753 at g8).

## 3. The grid rule (§10.3)

R-enc fitted on every fifth D0-fit frame (14 436 frames), applied to every D0-eval frame
(36 090):

| grid | median cube error (cm) | median jaw error (rad) | penalties (cube, jaw) |
|---|---|---|---|
| g4 | **1.598** | 0.1111 | 10³, 10³ |
| g8 | **1.624** | 0.1186 | 10³, 10⁴ |

g8 / g4 = 1.016 > 0.85, so **g4 is chosen** (TASK-066's grid, 32 tokens × 384). Report
`outputs/task100-stage0/grid_report.json` (`8bd2a484…`).

## 4. G1's bars (§10.4)

Calibration L (`L-100900`) and calibration N_L (`N_L-100911`, initialised from 100911, drawing
calibration L's windows from the stream `[10011, 100900]`), 20 000 updates each on D0-fit at g4,
§5's recipe. **Stage 0 does not open val, so neither calibration model is selected: the final
state at 20 000 is kept** (disclosed; §10.4 does not say how a calibration checkpoint is chosen,
and §5's selection reads val). Their D0-eval criterion curves (every 1 000 updates) are recorded:
L fell from 0.583 to 0.434 and was still falling slowly at 20 000 (best last quarter 0.4337
against 0.4398 up to 15 000, a 1.4 % drop: G6's 2 % rule would pass); N_L fell from 0.666 to its
best 0.5709 at 17 000 (0.6 %). Median seconds per update 0.0904 (L) and 0.0930 (N_L); peak GPU
memory 3.2 GB.

On the D0-eval roots (90 episodes × 37 = 3 330 roots), at h = 8 / 16:

| calibration L | h = 8 | h = 16 |
|---|---|---|
| collapsed fraction | 0.000 | 0.000 |
| effective rank, predicted / encoded | 46.0 / 178.4 = **0.258** | 41.9 / 176.5 = **0.237** |
| std ratio | **0.759** | **0.758** |

**B_rank = max(0.10, ½ × min(0.258, 0.237)) = 0.11** (0.1187 rounded down) and
**B_std = max(0.25, ½ × min(0.759, 0.758)) = 0.37** (0.3791 rounded down).

**Truncation check** (the combined G1 including (iii) against calibration N_L): every rank-k
truncation of calibration L's D0-eval predictions **fails**, at both horizons, so **B_rank is not
raised**:

| truncation | h = 8: rank ratio / std ratio / (iii) 97.5 % interval | h = 16 |
|---|---|---|
| k = 1 | 0.006 / 0.236 / [−0.319, −0.305] | 0.006 / 0.240 / [−0.238, −0.222] |
| k = 2 | 0.011 / 0.350 / [−0.306, −0.291] | 0.011 / 0.355 / [−0.225, −0.208] |
| k = 4 | 0.021 / 0.452 / [−0.280, −0.266] | 0.021 / 0.459 / [−0.199, −0.183] |

(each fails (ii)'s rank bar and (iii); k = 4 passes the std bar alone).

**Calibration L's other gate values on D0-eval** (disclosed; they move no bar; 97.5 % cluster
intervals over the 90 D0-eval episodes):

| quantity | h = 8 | h = 16 |
|---|---|---|
| G1 (iii): rank ratio L − N_L, projected (full width) | 0.099 [0.094, 0.107] (0.066) | 0.156 [0.150, 0.167] (0.103) |
| G2: acc(L) / acc(copy-last) | 0.605 [0.591, 0.620] | 0.470 [0.461, 0.479] |
| G3: acc(L) / acc(N_L) | 0.754 [0.746, 0.763] | 0.657 [0.648, 0.666] |
| G4: sens (wrong / true) | **1.802 [1.769, 1.835]** | **2.127 [2.083, 2.174]** |
| zero / true, hold / true (reported) | 1.61, 1.43 | 1.94, 1.79 |

**G4:** the lower bounds (1.769, 2.083) are above 1.10 at both horizons, so no G4 ruling is needed.

## 5. G5's bars (§10.5) and the moving subsets (§7.4)

R-enc (g4, fitted on every fifth D0-fit frame) on every D0-eval frame: cube **1.598 cm**, jaw
**0.1111 rad**, so **B_cube = 1.5 × 1.598 = 2.397 cm** and **B_jaw = 1.5 × 0.1111 = 0.1666 rad**
(6.4° × 1.5 = 9.5°).

Calibration L's R-pred (fitted on its own predictions from the 6 660 D0-fit roots) on the D0-eval
roots, with the copy-last readout and calibration N_L's R-pred on the moving subsets:

| | h = 8 | h = 16 |
|---|---|---|
| cube median (cm), all roots | 1.892 [1.772, 2.049] (bar 2.397) | 2.048 [1.869, 2.192] (bar 2.397) |
| cube, M_cube: L / copy-last; L / N_L | 0.567 [0.503, 0.622]; 0.585 [0.532, 0.644] (664 roots) | 0.465 [0.420, 0.511]; 0.470 [0.419, 0.519] (867) |
| jaw median (rad), all roots | 0.0612 [0.0573, 0.0648] (bar 0.1666) | 0.0547 [0.0509, 0.0590] |
| jaw, M_jaw: L / copy-last; L / N_L | 0.200 [0.178, 0.225]; 0.215 [0.189, 0.244] (807) | 0.131 [0.118, 0.146]; 0.147 [0.130, 0.166] (1 226) |
| R-enc applied to L's predictions (the shift diagnostic): cube, jaw | 2.69 cm, 0.175 rad | 2.88 cm, 0.184 rad |

Both medians are inside both bars at both horizons, so no G5 ruling is needed. Two disclosures:

- **R-pred reads the jaw better than R-enc reads the encoded frame** (0.061 against 0.111 rad):
  the predictor is given the absolute gripper command, which names the jaw's target (the review
  note folded into §6), so the predicted latent carries command information an image does not.
  B_jaw is relative to the image readout; G5 (a) on the jaw is therefore a weak test, and G5 (c)
  on the jaw is partly met by the schema.
- R-enc on the predictions (2.69 cm at h = 8) reads 1.42 × worse than R-pred (1.89 cm):
  TASK-077's distribution shift appears here too, which is why G5 is gated on R-pred.
- Penalties: R-enc 10³ for both labels; R-pred 10³ (cube) and 10² (jaw); the copy-last readout's jaw
  penalty is at the grid's upper edge (10⁵) at h = 16 and 32. Recorded, not changed.

**Moving subsets** (h = 8, D0-eval): M_cube (≥ 1.0 cm) 664 of 3 330 roots (19.9 %, 1 107 projected
of 5 550); M_jaw (≥ 0.2 rad) 807 (24.2 %, 1 345 projected). Both are above 300, so **neither
threshold is halved**: 1.0 cm and 0.2 rad stand.

Reported only (§9, calibration L at h = 8): the executed sequence ranks strictly best among 16
candidates on 17.4 % of roots by latent error (chance 6.25 %; normalised rank 0.18) and on 8.5 % by
the physical goal cost (normalised rank 0.27); the ranking ceiling: some candidate equals the
executed sequence over 8 steps on 0.2 % of roots, and no root has all 16 candidates at the same
jaw target. Share of roots with R-pred's cube error below 0.5 cm: 3.5 %; below 3 cm: 75.9 %.

Report `outputs/task100-stage0/calibration_report_4710557.json` (`a2def141…`).

## 6. Debug smoke and determinism (§10.6)

Every arm ran once with debug seed 100901 (L, J, N_L, N_J and R; R on the random-init features)
for 400 updates on D0-fit, selecting every 100 updates on the smoke's val stand-in (`smoke-val`:
the val episodes of shards 00–03, 20 episodes, 740 roots). Then the whole evaluation path ran:
`fitpred` on D0-fit, `readouts` (R-enc and copy-last for both encoders, R-pred for the five
models at h ∈ {1, 4, 8, 16, 32}), `evalpred` and `score` with `smoke-val` standing in for val and
`smoke-test` (the val episodes of shards 04–07, 20 episodes, 740 roots) standing in for test.
G6 was evaluated on each job's curve (it fails for every 400-update job, as it should), and the
row logic ran on the smoke's single seed (`smoke_report_smoke-test_row.json`, at `042f6dd`): it
returned WM-PARTIAL for L (G5 failing) — **an exercise of the code path, not a row**; 400-update
models on 20 episodes say nothing about the gated models. Parameter counts: L, N_L and R
2 842 310; J and N_J 177 536. Median seconds per update (measured while another process loaded
the CPU, so upper bounds): L-type 0.106–0.108, J-type 0.079. `evalpred` took 10–13 s per model on
740 roots and `fitpred` about 17 s per model on 6 660 roots.

**CUDA determinism:** `L-100902` (debug seed) trained twice for 2 000 updates on D0-fit with the
deterministic set-up: final state SHA-256 `64b9b4c2…` both times, and the kept checkpoint file
`5f123406…` both times (weights, optimiser and RNG): **bit-identical**.

## 7. GPU hours and the projection against the cap (§8.2)

Every GPU job went through `scripts/gpu_run.sh --wait --board` (the lock's own log, copied to
the evidence folder as `gpu_run_ledger.log`):

| job | lock time (s) |
|---|---|
| featurise D0, pretrained, g4 and g8 (pass A on D0-fit, pass B, probe) | 1 509 |
| featurise D0, random-init, g4 | 456 |
| calibration L (20 000 updates) | 1 897 |
| calibration N_L (20 000 updates) | 1 945 |
| calibration `fitpred` (2 models, 6 660 D0-fit roots) | 36 |
| calibration `evalpred` (2 models, 3 330 D0-eval roots) | 92 |
| determinism, `L-100902` twice (2 000 updates each) | 342 + 392 |
| smoke, five 400-update jobs | 55 + 41 + 53 + 42 + 66 |
| smoke `fitpred` (5 models) | 88 |
| smoke `evalpred` (val and test stand-ins) | 58 + 53 |
| **total** | **7 125 s = 1.98 GPU h** (Stage 0 cap 4 GPU h) |

CPU-only passes (fits, readouts, scores) took no lock. The ~10 s micro-benchmark outside the lock
(§8) is not in the table.

**Projection of the gated stages** from the measured rates (§8.2). Measured inputs: F's clean
featurisation rate 0.948 s an episode (random-init pass B; same architecture as the pretrained
encoder) plus about 150 s of corpus verification per encoder; calibration L's clean 0.0904 s an
update and N_L's 0.0930 s (R is L's architecture); J-type 0.079 s (a contended upper bound); a
val criterion about 7 s per selection on 5 550 roots (4 s measured on 3 330); `fitpred` about 70 s
and `evalpred` about 80 s per model at the gated sizes (scaled from 18 s per 6 660 roots and 45 s
per 3 330 roots).

| stage | U = 20 000 | U = 25 000 | U = 30 000 |
|---|---|---|---|
| Stage 0 (used) | 1.98 | 1.98 | 1.98 |
| F: train + val, both encoders, g4 | 1.58 | 1.58 | 1.58 |
| B: probe, 60 000 updates + 30 selections | 1.57 | 1.57 | 1.57 |
| T: 15 jobs (9 L-type, 6 J-type) | 7.74 | 9.53 | 11.32 |
| T′: a full extension, 12 jobs × 0.5 U (only if G6 triggers) | 3.06 | 3.82 | 4.58 |
| V, F′, E (`fitpred` and `evalpred` on val and test, featurise test) | 1.2 | 1.2 | 1.2 |
| **total with a full T′** | **17.1** | **19.7** | **22.2** |
| total without T′ | 14.1 | 15.9 | 17.6 |

So the gated stages are projected at **12.1–17.7 GPU h** (U = 20 000 to 25 000, with or without a
full T′) against the **18.0 GPU h** left of the 20 GPU h hard cap. U itself comes from Stage B's
probe (§8.1); if the rule gives U = 30 000, the cap forces it down: §8.2's own projection
(`so101_wm.budget_from_probe`, which charges every job at the probe's rate, about 0.09 s) lowers U
in steps of 5 000 until T + a full T′ + 1.5 GPU h for V–E fits the remainder, which at these rates
is U = 25 000 (14.7 of the 14.9 GPU h left after Stage 0, F and B). It escalates only if
U = 20 000 does not fit (projected 11.9 GPU h, so it fits). The margin at U = 25 000 is thin; the
1.5 × overrun rule of §8.2 stops a stage that runs over its projection.

## 8. Disclosures, deviations and implementation choices

- **Calibration checkpoints** are the final states (Stage 0 does not open val; §4 above).
- **Two runner files not named in §10.1** were added: `so101_wm_runtime.py` (shared runner
  helpers) and `summarize_task100.py` (tables from a report). The evaluation is split into four
  passes (`fitpred`, `readouts`, `evalpred`, `score`) so that the GPU lock is never held for CPU
  work (§8.2: CPU stages take no lock). R-pred is fitted once on train (at Stage V) and reused at E.
- **Standardisation lives in the data layer** (floor 1e-6, §3); the model's frozen-feature
  normalisation buffers are the identity (TASK-065's buffers would floor at 1e-3).
- **Ridge readouts:** one ridge per label with an unpenalised intercept, the penalty chosen per
  label by the summed squared error over the label's columns in 5-fold episode-grouped CV (ties:
  the smaller penalty), cube in cm and jaw in rad, float64 on the CPU; the normal equations
  (N ≥ D) or the dual form (N < D), one estimator.
- **R-enc for the grid rule and B_ℓ** is applied to every D0-eval frame; for the gated ceiling
  (reported) to the frames t + h of the test roots and to every test frame.
- **Weighted medians** in the cluster bootstrap: the average of the lower and upper weighted
  medians (NumPy's median under unit weights).
- **Hold commands:** arm zero, the gripper at the root's first executed command, every step.
  **The jaw ranking ceiling** simulates the gripper rate clip (0.25 rad a step) from the sidecar's
  gripper target at the root.
- **Truncation check** at both gate horizons (§10.4 names no horizon).
- **Parameter counts at the adapters' defaults**: L, N_L and R 2 842 310; J and N_J 177 536 (the
  native residual token step's MLP at `hidden_dim` 128). No size was tuned (§4).
- **The prediction files are float32** on the HDD (`outputs/task100-*`), hashed in their reports.
- **A micro-benchmark outside the lock.** While writing the model class I ran one 20-update timing
  of each backend on the GPU (about 10 s) without `gpu_run.sh`; nothing else held the GPU then. It
  is not part of any record.
- **A duplicate process.** From 03:08 UTC a process launched by another agent on this machine
  (OpenClaw, outside this session) re-ran the calibration score command in this worktree and
  restarted it twice; it competed for the CPU and slowed every CPU pass after it (the smoke's
  readouts took 1 390 s against 241 s for the calibration's). It ran the same code at an
  earlier revision; its output, if any, is not used. This record's calibration report is my run
  at `4710557`, written under a new name.
- **Review notes folded into the protocol** before the freeze (marked "Clarified at Stage 0"):
  the samplers' seeds (§5), R-pred fitted on in-sample predictions (§7.3, §13), the jaw command
  leak in G5 (c) (§6, §13), g8's two passes (§3).

## 9. The freeze

`benchmarks/manifests/so101-world-model-v1.json` records status FROZEN (effective at this PR's
merge), the SHA-256 of every file in §1 and of the composed files, the chosen grid, the bars
(B_rank 0.11, B_std 0.37, B_cube 2.397 cm, B_jaw 0.1666 rad), the subset thresholds (1.0 cm,
0.2 rad) and the Stage 0 evidence hashes; `so101_wm.CHOSEN_GRID`, `BARS` and `MOVING` hold the same
values, and a test checks that the manifest's pins match the tree. The gated stages run from a
clean run tree at the merge revision (`scripts/new_worktree.sh <dir>-run --from <sha> --run`).

## 10. Paths and hashes

- Stage 0 outputs (HDD): `outputs/task100-stage0/` (grid, calibration and smoke reports and
  passes), `outputs/task100-stage0-features-{pretrained,random_init}/` (D0 and smoke feature
  stores, about 7.9 GB), `checkpoints/task100-stage0/` (calibration, determinism and smoke jobs).
- Evidence (SSD): `~/develop/emai/evidence/task100-stage0/` (every report, the logs and the GPU
  ledger; 40 files), `SHA256SUMS` SHA-256
  `7bee61cf1e021809a6a4afd28dff765cdf486dff5f69faebff49a012908369c2`.

| record | SHA-256 |
|---|---|
| pretrained D0 features `report.json` (top) / `g4/report.json` / `g4/report_fit.json` | `09011ac7…` / `49104dca…` / `f8fdce29…` |
| pretrained `g8/report.json` / `g8/report_fit.json` | `268bd0d6…` / `9588782a…` |
| random-init D0 features `report.json` / `g4/report.json` / `g4/report_fit.json` | `c1914ebe…` / `34642639…` / `369a4911…` |
| grid report | `8bd2a4849e8b8093fe433ed874d75d41fc997bddfc33d3846f2a33b4d6b57859` |
| calibration report (`calibration_report_4710557.json`) | `a2def1416d139a74795ad255c1a7bf4f6575c9e42ccfba72b1a141c7d92c254d` |
| smoke reports: val stand-in / test stand-in / test stand-in with the row exercised | `f9d9bee7…` / `a293257d…` / `24dfae21…` |
| calibration L / N_L kept checkpoint (= final state) | `93c15296…` / `f0d9b491…` |
| determinism a = b: checkpoint file, final state | `5f123406…`, `64b9b4c2…` |
| implementation SHA-256 (LeWM token class / native token class) | `e5cda6ee1a89…` / `78ba9dbe00eb…` |

Revisions (all on this PR's branch; every run's tracked tree was clean): pretrained featurisation
at `d7f931b`; random-init featurisation, both pretrained fits and the grid rule at `6dbcd89`; the
random-init fit, calibration L and N_L, and the calibration `fitpred`, `readouts` and `evalpred`
at `1025700`; the determinism check at `6c0a428`; the calibration score, the smoke's training,
`fitpred` and `readouts` at `4710557`; the smoke's `evalpred` and scores at `9c0a2cf`; the row
exercise at `042f6dd`. Between `d7f931b` and the freeze the code changed only by a batched
matmul and single-threaded small eigensolves in the bootstrap (speed), the frozen-manifest guard,
the calibration pair's twin matching (the first calibration score stopped on it), the frozen
constants and the smoke's row exercise; the featurisation, training and model code that produced
the numbers above is byte-identical to the frozen code (`models/schema_frozen_tokens.py`,
`so101_wm_data.py`, `featurise_task100.py` and `train_task100.py` are unchanged since `d7f931b`;
the implementation hashes are unchanged).
