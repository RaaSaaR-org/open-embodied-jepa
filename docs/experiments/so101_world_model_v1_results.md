# TASK-100 results: SO-101 S4, an offline world-model study on `so101-play-v1` (row WM-PASS; J, `native_jepa`, is S5's model)

**Status: gated record.** Protocol [so101_world_model_v1.md](so101_world_model_v1.md) (FROZEN at
`f0da5bf`, the Stage 0 merge, #237); Stage 0 record
[so101_world_model_v1_stage0.md](so101_world_model_v1_stage0.md). Rulings R28.34–R28.40 in
[DECISIONS.md](../DECISIONS.md), decision 2026-10-19 (bs), each **decided by Claude under owner
delegation**. **This is an offline world-model result in simulation. It is not a control result,
no planner or closed loop ran, and it is not an Apple→Plate experiment.** SO-101 results do not
enter R7; R7's canonical status sentence does not change (R28.9, R28.26).

## Row

**WM-PASS, with arm J only** (`{"row": "WM-PASS", "arms": ["J"], "s5_model": "J"}`, §11, first
match).

- **J**, the `native_jepa` residual token step over frozen DINOv2 ViT-S/14 tokens of both cameras,
  passed **G1–G6 on all three model seeds (100100, 100101, 100102) at h = 8 and at h = 16** on the
  test split (150 episodes, 5 550 roots).
- **L**, the LeWM token predictor (the product-goal backend), passed **G1–G5 on all three seeds at
  both horizons** but **failed G6, training saturation, on seeds 100100 and 100102** after G6's one
  extension (T′, 20 000 → 30 000 updates): its val criterion still fell by 2.82 % and 2.60 % in
  the last quarter of training (bar 2 %). L is therefore not a passing arm, and **by §11 J, not L,
  is S5's model** (R28.35).
- The row says the J recipe's predictions on this corpus do not collapse, beat copy-last and an
  equally trained no-action twin, respond to the commands, and keep the cube position and the jaw
  angle readable from the predicted latent within the declared bars, with a budget that
  saturated. **It does not say that LeWM failed to learn dynamics** (L passed every quality gate,
  and on several reported measures it is ahead of J, see below), **and it does not say anything
  about control.**

The row was recomputed on the CPU from the stored test report with the frozen code at `f0da5bf`
(R28.34): `summarize_task100.py --check-sha256` reproduces `summary.md` byte for byte, and
`so101_wm.gates_for`, `so101_wm.saturation` (on every job's stored val curve) and
`so101_wm.decide` give the same gates, the same G6 flags and the same row as the report.

## What ran

All stages ran at `f0da5bf` from the clean run tree `worktrees/task100-run` (tracked tree clean
at every stage, frozen manifest SHA-256 `e35929a3…2503` checked by every job), every GPU job
under `scripts/gpu_run.sh --wait --board`; every one of the 29 GPU jobs ended with status 0.

| stage | what | UTC | GPU h (lock) |
|---|---|---|---|
| F | featurise train + val, both encoders, g4; then (CPU) the fit-sample standardisation and G1 (iii)'s 256-direction basis; hashed SSD working copy | 2026-10-17 07:50–09:52 | 1.638 |
| B | budget probe L-100920, 60 000 updates | 09:52–11:32 | 1.675 |
| T | 15 jobs (L, J, N_L, N_J, R × 3 seeds) at U = 20 000 | 11:33–19:43 | 8.174 |
| T′ | G6's one extension: L and N_L × 3 seeds resumed to 30 000 | 19:44–21:33 | 1.820 |
| V | `fitpred` (GPU), readouts (CPU), `evalpred` on val (GPU), score; val report written, SHA-256 recorded | 21:34–2026-10-18 05:31 | 1.129 |
| F′ | featurise test, both encoders, refusing without the val report's hash | 05:31–05:46 | 0.256 |
| E | `evalpred` on test (GPU), score (CPU), the row | 05:47–06:38 | 0.323 |
| **gated total** | | | **15.015** |
| Stage 0 (#237) | | 2026-10-17 | 1.979 |
| **all of TASK-100** | | | **16.99 of the 20 GPU h cap** |

- **Budget rule (§8.1–§8.2).** The probe's val criterion reached within 1 % of its minimum at
  u\* = 50 000, so the rule gave U = max(20 000, min(60 000, ⌈1.5 × 50 000 / 5 000⌉ × 5 000)) =
  60 000; the cap then lowered U in steps of 5 000 to **U = 20 000** (T projected 7.77 h, a full
  T′ 3.11 h), with no escalation. So every gated model was trained at **one third of the probe
  rule's U**; this matters for L's G6 (below).
- **G6 after T.** J and N_J saturated on all seeds. L-100101 (2.57 %) and L-100102 (2.02 %) failed,
  so all six L and N_L models extended (§6). R failed G6 on all three seeds and gets no extension
  (§6); its G6 is reported.
- **Val report** `outputs/task100-run/val_report.json`, SHA-256
  `b3c3eddef04da3e29a1fad952ce619af6f65714574a0e749f8f8f4c6f633aa31`, recorded in
  `runlog.txt` at 05:31:05Z and in the task card by commit `eb202e9` (05:31:18Z), before F′
  started (05:31:28Z). F′'s feature reports carry the same hash.
- **Test report** `outputs/task100-run/test_report.json`, SHA-256
  `9ca7144ef2ba943cfcca1057ed237cf35812e96ca605f05b7d16d249bef20c77`, outcome E-DONE; it pins the
  15 evaluated checkpoints (each matches the `model.pt` hash in the evidence) and the `evalpred`
  and readouts reports it read.

## Gates

Bars (frozen at Stage 0): B_rank 0.11, B_std 0.37, B_cube 2.397 cm, B_jaw 0.1666 rad; G2, G3 and
G5 (b)/(c) need ub < 1.0; G4 needs lb ≥ 1.10; G1 (i) collapsed fraction ≤ 0.05; G1 (iii) lb > 0.
Intervals are two-sided 97.5 % cluster-bootstrap intervals over the 150 test episodes (10 000
resamples, salt 10014). Moving subsets: M_cube 1 133 and 1 444 roots, M_jaw 1 426 and 2 124
roots at h = 8 and 16 (all above 100, so no subset VOID).

| arm | seed | h | G1 | G2 | G3 | G4 | G5 cube a/b/c | G5 jaw a/b/c | G6 |
|---|---|---|---|---|---|---|---|---|---|
| J | 100100 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| J | 100100 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| J | 100101 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| J | 100101 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| J | 100102 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| J | 100102 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| L | 100100 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | **FAIL** |
| L | 100100 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | **FAIL** |
| L | 100101 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| L | 100101 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | pass |
| L | 100102 | 8 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | **FAIL** |
| L | 100102 | 16 | pass | pass | pass | pass | pass/pass/pass | pass/pass/pass | **FAIL** |

### Gate quantities at the gated horizons (97.5 % intervals; L after T′)

| X | h | G1 coll. | rank ratio (≥ 0.11) | std ratio (≥ 0.37) | G1 (iii) lb > 0 | G2 copy ratio ub < 1 | G3 X/N_X ub < 1 | G4 sens lb ≥ 1.10 | G5a cube median ub ≤ 2.397 cm | G5b cube/copy (M) | G5c cube/N_X (M) | G5a jaw median ub ≤ 0.1666 rad | G5b jaw/copy (M) | G5c jaw/N_X (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J-100100 | 8 | 0.000 | 0.710 | 0.783 | 0.017 [0.013, 0.020] | 0.741 [0.737, 0.746] | 0.930 [0.927, 0.933] | 1.183 [1.175, 1.191] | 1.622 [1.547, 1.704] | 0.706 [0.665, 0.748] | 0.716 [0.676, 0.759] | 0.0503 [0.0482, 0.0526] | 0.170 [0.154, 0.188] | 0.172 [0.155, 0.188] |
| J-100100 | 16 | 0.000 | 0.712 | 0.719 | 0.039 [0.035, 0.042] | 0.662 [0.658, 0.667] | 0.903 [0.899, 0.906] | 1.253 [1.243, 1.264] | 1.796 [1.721, 1.877] | 0.619 [0.577, 0.654] | 0.611 [0.573, 0.646] | 0.0473 [0.0448, 0.0498] | 0.116 [0.106, 0.127] | 0.120 [0.108, 0.130] |
| J-100101 | 8 | 0.000 | 0.713 | 0.785 | 0.018 [0.013, 0.021] | 0.740 [0.735, 0.745] | 0.927 [0.924, 0.931] | 1.190 [1.181, 1.199] | 1.608 [1.540, 1.687] | 0.693 [0.646, 0.735] | 0.694 [0.656, 0.733] | 0.0495 [0.0469, 0.0523] | 0.173 [0.159, 0.193] | 0.173 [0.160, 0.190] |
| J-100101 | 16 | 0.000 | 0.719 | 0.720 | 0.043 [0.037, 0.047] | 0.659 [0.655, 0.664] | 0.899 [0.895, 0.904] | 1.264 [1.252, 1.275] | 1.744 [1.673, 1.827] | 0.595 [0.571, 0.627] | 0.597 [0.564, 0.630] | 0.0500 [0.0473, 0.0524] | 0.114 [0.106, 0.125] | 0.117 [0.108, 0.129] |
| J-100102 | 8 | 0.000 | 0.704 | 0.784 | 0.019 [0.015, 0.022] | 0.744 [0.739, 0.748] | 0.934 [0.931, 0.936] | 1.166 [1.159, 1.174] | 1.611 [1.536, 1.687] | 0.711 [0.667, 0.753] | 0.714 [0.675, 0.761] | 0.0508 [0.0484, 0.0533] | 0.181 [0.165, 0.201] | 0.182 [0.166, 0.200] |
| J-100102 | 16 | 0.000 | 0.705 | 0.718 | 0.043 [0.038, 0.047] | 0.666 [0.661, 0.670] | 0.909 [0.905, 0.912] | 1.236 [1.226, 1.247] | 1.772 [1.704, 1.853] | 0.610 [0.583, 0.646] | 0.617 [0.581, 0.651] | 0.0511 [0.0488, 0.0537] | 0.128 [0.118, 0.138] | 0.132 [0.120, 0.142] |
| L-100100-ext | 8 | 0.000 | 0.229 | 0.740 | 0.108 [0.104, 0.112] | 0.571 [0.561, 0.581] | 0.742 [0.735, 0.750] | 1.822 [1.794, 1.850] | 1.668 [1.572, 1.765] | 0.551 [0.512, 0.594] | 0.560 [0.530, 0.599] | 0.0528 [0.0498, 0.0559] | 0.147 [0.135, 0.162] | 0.157 [0.142, 0.175] |
| L-100100-ext | 16 | 0.000 | 0.202 | 0.738 | 0.172 [0.170, 0.178] | 0.436 [0.429, 0.444] | 0.667 [0.659, 0.676] | 2.207 [2.166, 2.249] | 1.764 [1.636, 1.882] | 0.409 [0.384, 0.439] | 0.410 [0.381, 0.442] | 0.0471 [0.0447, 0.0503] | 0.105 [0.096, 0.115] | 0.110 [0.101, 0.121] |
| L-100101-ext | 8 | 0.000 | 0.228 | 0.741 | 0.102 [0.099, 0.106] | 0.575 [0.565, 0.586] | 0.750 [0.743, 0.757] | 1.818 [1.789, 1.847] | 1.663 [1.565, 1.767] | 0.534 [0.495, 0.573] | 0.558 [0.517, 0.595] | 0.0524 [0.0498, 0.0554] | 0.146 [0.133, 0.161] | 0.151 [0.136, 0.167] |
| L-100101-ext | 16 | 0.000 | 0.204 | 0.739 | 0.166 [0.164, 0.171] | 0.442 [0.434, 0.450] | 0.675 [0.667, 0.684] | 2.176 [2.136, 2.216] | 1.705 [1.600, 1.829] | 0.414 [0.384, 0.440] | 0.420 [0.388, 0.448] | 0.0460 [0.0436, 0.0488] | 0.104 [0.096, 0.113] | 0.108 [0.099, 0.117] |
| L-100102-ext | 8 | 0.000 | 0.234 | 0.749 | 0.117 [0.114, 0.122] | 0.566 [0.556, 0.576] | 0.740 [0.733, 0.748] | 1.854 [1.824, 1.884] | 1.636 [1.535, 1.736] | 0.534 [0.497, 0.575] | 0.538 [0.507, 0.577] | 0.0531 [0.0506, 0.0561] | 0.146 [0.135, 0.158] | 0.153 [0.140, 0.167] |
| L-100102-ext | 16 | 0.000 | 0.209 | 0.748 | 0.179 [0.176, 0.185] | 0.433 [0.425, 0.441] | 0.666 [0.658, 0.675] | 2.245 [2.202, 2.289] | 1.694 [1.571, 1.830] | 0.399 [0.373, 0.430] | 0.408 [0.379, 0.442] | 0.0461 [0.0437, 0.0490] | 0.106 [0.097, 0.115] | 0.109 [0.100, 0.118] |

### G6 per model (2 % rule; L and N_L after T′)

| model | updates | kept update | last_two_triggered | relative drop at T (20 000) | relative drop after T′ (30 000) | G6 |
|---|---|---|---|---|---|---|
| J-100100 | 20000 | 15000 | False | 0.94 % (pass) | — | pass |
| J-100101 | 20000 | 16000 | False | 1.10 % (pass) | — | pass |
| J-100102 | 20000 | 14000 | False | 0.81 % (pass) | — | pass |
| N_J-100100 | 20000 | 4000 | False | 0.12 % (pass) | — | pass |
| N_J-100101 | 20000 | 4000 | False | 0.10 % (pass) | — | pass |
| N_J-100102 | 20000 | 4000 | False | 0.16 % (pass) | — | pass |
| L-100100 | 30000 | 26000 | False | 1.69 % (pass) | 2.82 % | FAIL |
| L-100101 | 30000 | 26000 | False | 2.57 % (fail) | 1.63 % | pass |
| L-100102 | 30000 | 30000 | True | 2.02 % (fail) | 2.60 % | FAIL |
| N_L-100100 | 30000 | 21000 | False | 0.82 % (pass) | 0.88 % | pass |
| N_L-100101 | 30000 | 19000 | False | 1.63 % (pass) | 0.90 % | pass |
| N_L-100102 | 30000 | 21000 | False | 0.32 % (pass) | 0.72 % | pass |
| R-100100 | 20000 | 20000 | True | 5.60 % (fail) | — | FAIL |
| R-100101 | 20000 | 18000 | False | 2.32 % (fail) | — | FAIL |
| R-100102 | 20000 | 20000 | True | 4.47 % (fail) | — | FAIL |

Reading G6:

- **J's budget saturated with room to spare**: its kept checkpoints are at 14 000–16 000 of 20 000
  updates and its last-quarter drops are 0.81–1.10 %; N_J kept 4 000 (0.10–0.16 %).
- **L's did not.** L-100100 passed at 20 000 (1.69 %) and failed after the extension (2.82 %),
  L-100102 failed both times (2.02 %, then 2.60 %, kept at 30 000 with `last_two_triggered` set),
  and L-100101 failed at 20 000 (2.57 %) and passed after it (1.63 %). The probe had shown the same
  shape: L-100920's criterion still fell between 40 000 and 58 000 updates, and its rule asked for
  U = 60 000 before the cap lowered it to 20 000. **L's G6 failure is a budget finding under this
  cap, not a quality finding**: it says the LeWM token predictor was still improving when its
  training stopped, and nothing about what it would reach at saturation.
- R (the random-init floor, L's architecture) also failed G6 on every seed (2.32–5.60 %, two with
  `last_two_triggered`), consistent with the same architecture needing more updates.

## Reported only (no gate)

### Gate quantities at h ∈ {1, 4, 32} (J and L)

| X | h | rank ratio | std ratio | copy ratio | X/N_X | sens | cube median (cm) | jaw median (rad) |
|---|---|---|---|---|---|---|---|---|
| J-100100 | 1 | 0.900 | 0.944 | 0.937 [0.935, 0.939] | 0.998 [0.997, 0.999] | 1.024 [1.023, 1.026] | 1.442 [1.382, 1.515] | 0.0830 [0.0794, 0.0883] |
| J-100100 | 4 | 0.761 | 0.848 | 0.821 [0.817, 0.826] | 0.961 [0.958, 0.963] | 1.123 [1.116, 1.130] | 1.524 [1.457, 1.599] | 0.0538 [0.0514, 0.0561] |
| J-100100 | 32 | 0.805 | 0.668 | 0.614 [0.609, 0.618] | 0.895 [0.891, 0.899] | 1.289 [1.278, 1.300] | 2.132 [2.051, 2.242] | 0.0514 [0.0487, 0.0543] |
| J-100101 | 1 | 0.900 | 0.944 | 0.938 [0.936, 0.940] | 0.999 [0.998, 1.000] | 1.024 [1.023, 1.025] | 1.439 [1.373, 1.516] | 0.0807 [0.0767, 0.0848] |
| J-100101 | 4 | 0.763 | 0.850 | 0.820 [0.816, 0.825] | 0.959 [0.957, 0.961] | 1.125 [1.119, 1.132] | 1.527 [1.444, 1.603] | 0.0513 [0.0490, 0.0536] |
| J-100101 | 32 | 0.829 | 0.663 | 0.610 [0.606, 0.615] | 0.892 [0.888, 0.896] | 1.298 [1.286, 1.311] | 2.117 [2.047, 2.201] | 0.0543 [0.0523, 0.0566] |
| J-100102 | 1 | 0.898 | 0.944 | 0.938 [0.936, 0.940] | 0.999 [0.999, 1.000] | 1.020 [1.019, 1.021] | 1.434 [1.370, 1.519] | 0.0829 [0.0792, 0.0874] |
| J-100102 | 4 | 0.756 | 0.849 | 0.823 [0.819, 0.827] | 0.963 [0.961, 0.966] | 1.104 [1.099, 1.109] | 1.515 [1.446, 1.594] | 0.0527 [0.0501, 0.0558] |
| J-100102 | 32 | 0.795 | 0.662 | 0.619 [0.615, 0.623] | 0.906 [0.902, 0.910] | 1.267 [1.256, 1.279] | 2.169 [2.079, 2.272] | 0.0584 [0.0558, 0.0613] |
| L-100100-ext | 1 | 0.461 | 0.830 | 1.338 [1.314, 1.364] | 0.964 [0.962, 0.966] | 1.099 [1.094, 1.104] | 1.630 [1.556, 1.709] | 0.1041 [0.0993, 0.1094] |
| L-100100-ext | 4 | 0.284 | 0.757 | 0.759 [0.747, 0.774] | 0.834 [0.828, 0.839] | 1.497 [1.477, 1.516] | 1.635 [1.545, 1.733] | 0.0636 [0.0607, 0.0673] |
| L-100100-ext | 32 | 0.193 | 0.741 | 0.372 [0.366, 0.380] | 0.629 [0.619, 0.639] | 2.408 [2.357, 2.459] | 1.969 [1.825, 2.116] | 0.0469 [0.0438, 0.0504] |
| L-100101-ext | 1 | 0.461 | 0.828 | 1.338 [1.314, 1.364] | 0.966 [0.965, 0.968] | 1.099 [1.094, 1.104] | 1.652 [1.583, 1.743] | 0.1061 [0.1004, 0.1121] |
| L-100101-ext | 4 | 0.282 | 0.756 | 0.763 [0.750, 0.778] | 0.839 [0.834, 0.845] | 1.505 [1.484, 1.525] | 1.670 [1.580, 1.770] | 0.0635 [0.0608, 0.0668] |
| L-100101-ext | 32 | 0.199 | 0.735 | 0.383 [0.376, 0.390] | 0.642 [0.633, 0.652] | 2.320 [2.272, 2.368] | 1.876 [1.741, 1.999] | 0.0478 [0.0447, 0.0507] |
| L-100102-ext | 1 | 0.466 | 0.834 | 1.321 [1.298, 1.347] | 0.960 [0.958, 0.962] | 1.109 [1.104, 1.115] | 1.638 [1.546, 1.712] | 0.1037 [0.0991, 0.1083] |
| L-100102-ext | 4 | 0.289 | 0.765 | 0.751 [0.739, 0.765] | 0.828 [0.823, 0.834] | 1.521 [1.500, 1.542] | 1.628 [1.524, 1.729] | 0.0661 [0.0634, 0.0691] |
| L-100102-ext | 32 | 0.204 | 0.748 | 0.370 [0.363, 0.377] | 0.625 [0.615, 0.636] | 2.445 [2.390, 2.500] | 1.895 [1.767, 2.045] | 0.0484 [0.0452, 0.0515] |

Gate-level (97.5 %) intervals, not gated at these horizons. At **h = 1 L is worse than copy-last** (copy ratio 1.32–1.34, upper bounds above 1) while J is just below it (0.94); from h = 4 on L's copy ratio is below J's at every horizon (0.37–0.38 against 0.61–0.62 at h = 32). L's predicted latent keeps less effective rank than J's (0.19–0.47 against 0.70–0.90 of the encoded rank), above B_rank = 0.11 at every horizon.

### Readouts, the distribution-shift diagnostic and the absolute references (h = 8 and 16)

R-enc on encoded test frames (the ceiling; 95 %): pretrained cube 1.381 cm [1.323, 1.433] at
h = 8 and 1.399 cm [1.349, 1.451] at h = 16, jaw 0.0982 and 0.0960 rad; all 60 150 test frames
1.391 cm and 0.0969 rad. Random-init: cube 2.267 and 2.254 cm, jaw 0.1557 and 0.1543 rad. The
copy-last readout (start frame only; pretrained) reads the cube at t + h to 1.752 cm (h = 8) and
2.044 cm (h = 16), the jaw to 0.187 and 0.332 rad.

How much of G5 (a)'s 1.5 × the pass used (R-pred's test median over the gated R-enc's test
median at the same h): **cube 1.16–1.28 × for J and 1.19–1.26 × for L**; **jaw 0.49–0.53 × for J
and 0.48–0.54 × for L**. The jaw ratio below 1 is the schema at work, as §6 and §13 disclose:
X receives the absolute gripper command, which names the jaw's target, so the prediction can
read the jaw better than the start frame's encoder; it does not show that the model learned the
jaw's motion from images. B_ℓ came from a readout fitted on 180 D0 episodes and is looser than
1.5 × the gated ceiling (§10.5): against the gated ceiling, 1.5 × would be 2.07 cm (h = 8) and
2.10 cm (h = 16), and every J and L R-pred cube upper bound (1.687–1.882 cm) is also below those.

Per model, 95 % intervals (the copy ratio here is the 95 % reported interval; G2's gated one is above):

| model | h | copy ratio | G2 alloc. ≤ 0.8 | zero/true | hold/true | R-pred cube (cm) | R-enc on pred cube (cm) | R-pred jaw (rad) | R-enc on pred jaw (rad) | share cube < 0.5 cm | share cube < 3 cm | G1 (iii) full width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J-100100 | 8 | 0.741 [0.738, 0.745] | pass | 1.119 [1.115, 1.123] | 1.107 [1.103, 1.111] | 1.622 [1.547, 1.704] | 1.941 [1.866, 2.010] | 0.0503 [0.0482, 0.0526] | 0.1784 [0.1705, 0.1868] | 0.047 | 0.851 | 0.048 |
| J-100100 | 16 | 0.662 [0.658, 0.666] | pass | 1.193 [1.187, 1.199] | 1.191 [1.185, 1.198] | 1.796 [1.721, 1.877] | 2.434 [2.353, 2.528] | 0.0473 [0.0448, 0.0498] | 0.2651 [0.2557, 0.2773] | 0.034 | 0.803 | 0.075 |
| J-100101 | 8 | 0.740 [0.736, 0.744] | pass | 1.119 [1.115, 1.123] | 1.110 [1.106, 1.115] | 1.608 [1.540, 1.687] | 1.934 [1.864, 1.996] | 0.0495 [0.0469, 0.0523] | 0.1758 [0.1669, 0.1859] | 0.042 | 0.857 | 0.046 |
| J-100101 | 16 | 0.659 [0.655, 0.664] | pass | 1.195 [1.189, 1.201] | 1.196 [1.190, 1.202] | 1.744 [1.673, 1.827] | 2.427 [2.339, 2.518] | 0.0500 [0.0473, 0.0524] | 0.2604 [0.2460, 0.2759] | 0.040 | 0.815 | 0.075 |
| J-100102 | 8 | 0.744 [0.740, 0.748] | pass | 1.109 [1.106, 1.113] | 1.102 [1.098, 1.106] | 1.611 [1.536, 1.687] | 1.904 [1.846, 1.979] | 0.0508 [0.0484, 0.0533] | 0.1830 [0.1719, 0.1939] | 0.045 | 0.849 | 0.041 |
| J-100102 | 16 | 0.666 [0.662, 0.670] | pass | 1.180 [1.174, 1.185] | 1.181 [1.175, 1.186] | 1.772 [1.704, 1.853] | 2.425 [2.332, 2.498] | 0.0511 [0.0488, 0.0537] | 0.2687 [0.2553, 0.2824] | 0.035 | 0.805 | 0.065 |
| N_J-100100 | 8 | 0.797 [0.794, 0.801] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.706 [1.644, 1.786] | 1.897 [1.834, 1.965] | 0.1893 [0.1758, 0.2044] | 0.2222 [0.2093, 0.2337] | 0.037 | 0.817 | — |
| N_J-100100 | 16 | 0.734 [0.730, 0.737] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.972 [1.890, 2.056] | 2.344 [2.262, 2.446] | 0.3247 [0.3047, 0.3475] | 0.3467 [0.3295, 0.3647] | 0.024 | 0.727 | — |
| N_J-100101 | 8 | 0.798 [0.794, 0.801] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.686 [1.626, 1.767] | 1.883 [1.822, 1.953] | 0.1885 [0.1736, 0.2020] | 0.2202 [0.2100, 0.2317] | 0.041 | 0.815 | — |
| N_J-100101 | 16 | 0.733 [0.730, 0.737] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.940 [1.874, 2.025] | 2.377 [2.286, 2.494] | 0.3235 [0.3022, 0.3454] | 0.3536 [0.3397, 0.3698] | 0.023 | 0.731 | — |
| N_J-100102 | 8 | 0.797 [0.793, 0.800] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.707 [1.630, 1.786] | 1.910 [1.850, 1.984] | 0.1883 [0.1745, 0.2023] | 0.2242 [0.2120, 0.2362] | 0.035 | 0.818 | — |
| N_J-100102 | 16 | 0.732 [0.729, 0.736] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.955 [1.890, 2.032] | 2.429 [2.334, 2.522] | 0.3244 [0.2991, 0.3472] | 0.3523 [0.3391, 0.3695] | 0.023 | 0.731 | — |
| L-100100-ext | 8 | 0.571 [0.562, 0.580] | pass | 1.648 [1.629, 1.668] | 1.468 [1.452, 1.484] | 1.668 [1.572, 1.765] | 2.684 [2.537, 2.823] | 0.0528 [0.0498, 0.0559] | 0.1794 [0.1709, 0.1883] | 0.051 | 0.812 | 0.068 |
| L-100100-ext | 16 | 0.436 [0.430, 0.443] | pass | 2.003 [1.975, 2.031] | 1.860 [1.834, 1.885] | 1.764 [1.636, 1.882] | 2.904 [2.731, 3.053] | 0.0471 [0.0447, 0.0503] | 0.1714 [0.1632, 0.1807] | 0.047 | 0.768 | 0.105 |
| L-100101-ext | 8 | 0.575 [0.566, 0.584] | pass | 1.646 [1.627, 1.667] | 1.466 [1.450, 1.483] | 1.663 [1.565, 1.767] | 2.536 [2.413, 2.697] | 0.0524 [0.0498, 0.0554] | 0.1711 [0.1620, 0.1800] | 0.050 | 0.814 | 0.065 |
| L-100101-ext | 16 | 0.442 [0.435, 0.449] | pass | 1.992 [1.964, 2.020] | 1.860 [1.834, 1.886] | 1.705 [1.600, 1.829] | 2.732 [2.584, 2.915] | 0.0460 [0.0436, 0.0488] | 0.1733 [0.1648, 0.1816] | 0.050 | 0.783 | 0.103 |
| L-100102-ext | 8 | 0.566 [0.557, 0.574] | pass | 1.680 [1.659, 1.701] | 1.495 [1.478, 1.513] | 1.636 [1.535, 1.736] | 2.595 [2.452, 2.761] | 0.0531 [0.0506, 0.0561] | 0.1685 [0.1601, 0.1776] | 0.052 | 0.819 | 0.073 |
| L-100102-ext | 16 | 0.433 [0.426, 0.440] | pass | 2.068 [2.037, 2.098] | 1.918 [1.889, 1.946] | 1.694 [1.571, 1.830] | 2.761 [2.611, 2.903] | 0.0461 [0.0437, 0.0490] | 0.1649 [0.1565, 0.1733] | 0.054 | 0.780 | 0.109 |
| N_L-100100-ext | 8 | 0.769 [0.761, 0.777] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.737 [1.654, 1.826] | 2.930 [2.791, 3.068] | 0.1895 [0.1759, 0.2038] | 0.3383 [0.3228, 0.3531] | 0.040 | 0.803 | — |
| N_L-100100-ext | 16 | 0.654 [0.649, 0.660] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.906 [1.809, 2.028] | 3.561 [3.387, 3.743] | 0.3178 [0.2932, 0.3466] | 0.4300 [0.4106, 0.4509] | 0.040 | 0.727 | — |
| N_L-100101-ext | 8 | 0.767 [0.760, 0.775] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.782 [1.705, 1.866] | 2.823 [2.683, 2.963] | 0.1902 [0.1751, 0.2034] | 0.3193 [0.3066, 0.3388] | 0.040 | 0.800 | — |
| N_L-100101-ext | 16 | 0.654 [0.649, 0.660] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.966 [1.866, 2.072] | 3.188 [3.011, 3.384] | 0.3177 [0.2960, 0.3438] | 0.4047 [0.3815, 0.4272] | 0.034 | 0.720 | — |
| N_L-100102-ext | 8 | 0.764 [0.757, 0.771] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.818 [1.741, 1.903] | 2.713 [2.565, 2.839] | 0.1869 [0.1715, 0.2034] | 0.3318 [0.3194, 0.3448] | 0.033 | 0.792 | — |
| N_L-100102-ext | 16 | 0.650 [0.645, 0.655] | — | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.987 [1.887, 2.100] | 3.049 [2.891, 3.194] | 0.3246 [0.3004, 0.3529] | 0.4145 [0.3973, 0.4327] | 0.031 | 0.719 | — |
| R-100100 | 8 | 0.406 [0.396, 0.416] | — | 2.475 [2.413, 2.539] | 2.196 [2.147, 2.248] | 2.036 [1.885, 2.202] | 2.859 [2.738, 2.993] | 0.0768 [0.0721, 0.0814] | 0.2784 [0.2649, 0.2915] | 0.037 | 0.697 | — |
| R-100100 | 16 | 0.315 [0.306, 0.324] | — | 3.085 [2.993, 3.178] | 2.886 [2.806, 2.966] | 2.099 [1.966, 2.267] | 3.019 [2.858, 3.170] | 0.0755 [0.0711, 0.0799] | 0.2714 [0.2589, 0.2830] | 0.039 | 0.673 | — |
| R-100101 | 8 | 0.409 [0.399, 0.419] | — | 2.479 [2.413, 2.546] | 2.165 [2.117, 2.215] | 2.107 [1.984, 2.286] | 2.819 [2.695, 3.002] | 0.0723 [0.0683, 0.0770] | 0.2563 [0.2449, 0.2660] | 0.038 | 0.689 | — |
| R-100101 | 16 | 0.316 [0.308, 0.325] | — | 3.042 [2.953, 3.131] | 2.816 [2.741, 2.889] | 2.160 [2.010, 2.331] | 2.997 [2.855, 3.185] | 0.0676 [0.0637, 0.0717] | 0.2467 [0.2353, 0.2583] | 0.035 | 0.665 | — |
| R-100102 | 8 | 0.405 [0.395, 0.415] | — | 2.573 [2.505, 2.640] | 2.273 [2.218, 2.330] | 2.015 [1.899, 2.164] | 2.832 [2.702, 2.955] | 0.0724 [0.0681, 0.0773] | 0.2885 [0.2763, 0.2980] | 0.038 | 0.709 | — |
| R-100102 | 16 | 0.313 [0.305, 0.322] | — | 3.266 [3.172, 3.356] | 3.042 [2.957, 3.126] | 2.077 [1.952, 2.241] | 3.015 [2.873, 3.164] | 0.0676 [0.0637, 0.0716] | 0.2982 [0.2879, 0.3083] | 0.039 | 0.683 | — |

- **R-enc on predictions** (the readout fitted on encoded frames, applied to predicted latents)
  reads the cube at 1.90–2.43 cm for J and 2.54–2.90 cm for L, and the jaw at 0.16–0.27 rad,
  against R-pred's 1.61–1.80 cm and 0.046–0.053 rad: TASK-077's distribution shift is present for
  both backends, larger for L, and R-pred (the gated readout) removes most of it.
- **Against S2's envelope** (§7.3, §9): R-pred's median cube error is 1.6–1.8 cm, about 3 × S2's
  0.5 cm grasp-offset tolerance; only **3.4–5.4 % of test roots** read the cube within 0.5 cm,
  and 77–86 % within S2's 3 cm place-aim tolerance. **G5 passing does not mean the predicted cube
  is precise enough to grasp**; that is S5's question, and these numbers say a controller that
  takes the grasp point from this readout alone would very likely miss.
- **The encoder floor (§9's declared reading).** L's R-pred cube median is below R's on all three
  seeds at h = 8 and h = 16: paired L − R −0.37 [−0.47, −0.28], −0.44 [−0.56, −0.37] and −0.38
  [−0.48, −0.30] cm at h = 8, and −0.33 [−0.43, −0.25], −0.45 [−0.56, −0.36] and −0.38 [−0.47,
  −0.31] cm at h = 16 (95 %; every upper bound < 0). By the declared reading, **pretrained
  features read the predicted cube better than the random-init floor** here. R failed G6 on all
  seeds, so its budget did not saturate either; the reading is about these trained models, and
  the floor shares L's recipe (§13).

### §9's planning measure (offline ranking against the true future frame)

16 candidates per root (§7.2); chance top-1 1/16 = 0.0625, chance normalised rank 0.5; 95 % intervals.

| model | h | latent top-1 | latent norm. rank | physical top-1 | physical norm. rank |
|---|---|---|---|---|---|
| J-100100 | 4 | 0.076 [0.067, 0.085] | 0.275 [0.268, 0.281] | 0.084 [0.076, 0.093] | 0.264 [0.259, 0.270] |
| J-100100 | 8 | 0.130 [0.118, 0.142] | 0.259 [0.252, 0.266] | 0.116 [0.106, 0.125] | 0.252 [0.247, 0.258] |
| J-100100 | 16 | 0.150 [0.138, 0.162] | 0.248 [0.240, 0.256] | 0.123 [0.113, 0.133] | 0.240 [0.235, 0.246] |
| J-100101 | 4 | 0.062 [0.054, 0.070] | 0.280 [0.274, 0.286] | 0.096 [0.088, 0.104] | 0.259 [0.253, 0.264] |
| J-100101 | 8 | 0.120 [0.110, 0.130] | 0.258 [0.251, 0.265] | 0.113 [0.103, 0.123] | 0.251 [0.245, 0.256] |
| J-100101 | 16 | 0.145 [0.132, 0.157] | 0.242 [0.234, 0.251] | 0.112 [0.103, 0.122] | 0.241 [0.235, 0.246] |
| J-100102 | 4 | 0.062 [0.055, 0.070] | 0.283 [0.276, 0.290] | 0.091 [0.084, 0.099] | 0.261 [0.255, 0.266] |
| J-100102 | 8 | 0.097 [0.088, 0.107] | 0.269 [0.263, 0.276] | 0.115 [0.105, 0.126] | 0.252 [0.246, 0.258] |
| J-100102 | 16 | 0.118 [0.108, 0.129] | 0.254 [0.247, 0.262] | 0.100 [0.091, 0.110] | 0.248 [0.242, 0.254] |
| L-100100-ext | 4 | 0.219 [0.205, 0.232] | 0.142 [0.137, 0.147] | 0.065 [0.058, 0.071] | 0.280 [0.275, 0.286] |
| L-100100-ext | 8 | 0.225 [0.211, 0.240] | 0.146 [0.140, 0.152] | 0.084 [0.076, 0.092] | 0.255 [0.249, 0.261] |
| L-100100-ext | 16 | 0.194 [0.179, 0.209] | 0.161 [0.154, 0.167] | 0.102 [0.093, 0.112] | 0.234 [0.227, 0.242] |
| L-100101-ext | 4 | 0.211 [0.198, 0.224] | 0.147 [0.142, 0.152] | 0.066 [0.059, 0.073] | 0.284 [0.278, 0.290] |
| L-100101-ext | 8 | 0.217 [0.205, 0.231] | 0.150 [0.144, 0.156] | 0.094 [0.086, 0.103] | 0.250 [0.244, 0.257] |
| L-100101-ext | 16 | 0.163 [0.152, 0.175] | 0.171 [0.165, 0.177] | 0.109 [0.100, 0.118] | 0.227 [0.220, 0.233] |
| L-100102-ext | 4 | 0.232 [0.219, 0.247] | 0.139 [0.134, 0.145] | 0.063 [0.056, 0.070] | 0.285 [0.280, 0.291] |
| L-100102-ext | 8 | 0.253 [0.237, 0.269] | 0.140 [0.134, 0.146] | 0.088 [0.081, 0.096] | 0.256 [0.250, 0.262] |
| L-100102-ext | 16 | 0.207 [0.193, 0.222] | 0.156 [0.150, 0.163] | 0.094 [0.085, 0.103] | 0.233 [0.225, 0.240] |
| N_J-100100 | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100100 | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100100 | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100101 | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100101 | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100101 | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100102 | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100102 | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_J-100102 | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100100-ext | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100100-ext | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100100-ext | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100101-ext | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100101-ext | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100101-ext | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100102-ext | 4 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100102-ext | 8 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| N_L-100102-ext | 16 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| R-100100 | 4 | 0.207 [0.192, 0.222] | 0.146 [0.139, 0.153] | 0.063 [0.057, 0.070] | 0.290 [0.284, 0.297] |
| R-100100 | 8 | 0.198 [0.183, 0.213] | 0.155 [0.148, 0.162] | 0.080 [0.073, 0.088] | 0.273 [0.266, 0.281] |
| R-100100 | 16 | 0.163 [0.148, 0.177] | 0.172 [0.165, 0.181] | 0.089 [0.080, 0.099] | 0.258 [0.250, 0.266] |
| R-100101 | 4 | 0.216 [0.202, 0.231] | 0.143 [0.136, 0.150] | 0.057 [0.051, 0.063] | 0.289 [0.282, 0.295] |
| R-100101 | 8 | 0.220 [0.204, 0.237] | 0.151 [0.144, 0.158] | 0.076 [0.069, 0.084] | 0.271 [0.264, 0.278] |
| R-100101 | 16 | 0.186 [0.170, 0.203] | 0.168 [0.160, 0.176] | 0.096 [0.086, 0.107] | 0.251 [0.242, 0.259] |
| R-100102 | 4 | 0.201 [0.188, 0.215] | 0.144 [0.138, 0.150] | 0.059 [0.052, 0.066] | 0.292 [0.285, 0.299] |
| R-100102 | 8 | 0.193 [0.178, 0.208] | 0.157 [0.150, 0.164] | 0.075 [0.068, 0.082] | 0.267 [0.260, 0.274] |
| R-100102 | 16 | 0.156 [0.142, 0.170] | 0.179 [0.172, 0.187] | 0.094 [0.086, 0.103] | 0.248 [0.241, 0.256] |

by class (h = 8, share of roots where the executed sequence beats every candidate of the class):

| model | σ 0.1 latent / physical | σ 0.3 | gripper-only | other episode |
|---|---|---|---|---|
| J-100100 | 0.240 / 0.227 | 0.545 / 0.391 | 0.409 / 0.584 | 0.824 / 0.738 |
| J-100101 | 0.226 / 0.224 | 0.542 / 0.387 | 0.405 / 0.583 | 0.833 / 0.739 |
| J-100102 | 0.215 / 0.228 | 0.518 / 0.393 | 0.374 / 0.574 | 0.821 / 0.738 |
| L-100100-ext | 0.405 / 0.181 | 0.812 / 0.403 | 0.525 / 0.554 | 0.959 / 0.746 |
| L-100101-ext | 0.397 / 0.192 | 0.810 / 0.417 | 0.508 / 0.561 | 0.961 / 0.743 |
| L-100102-ext | 0.423 / 0.180 | 0.809 / 0.403 | 0.546 / 0.559 | 0.957 / 0.738 |


Reading (reported only; not a planner, which has no goal frame):

- **Latent cost:** every J and L model ranks the executed command sequence above chance; **L ranks
  it markedly better than J** (top-1 0.16–0.25 against 0.06–0.15; normalised rank 0.14–0.17
  against 0.24–0.28), consistent with L's larger action sensitivity (sens 1.82–2.45 against
  1.17–1.30 at h = 8 to 32) and its lower copy ratio. R ranks about as well as L in its own
  latent.
- **Physical cost** (R-pred's cube and jaw against the truth at t + h): J and L are similar
  (top-1 0.06–0.12, normalised rank 0.23–0.29). Both separate other-episode and σ 0.3 candidates
  much better than σ 0.1 ones, so fine command differences are not resolved by the physical cost.
- N arms receive zero commands for every candidate, so every candidate ties and ties count
  against (top-1 0, rank 1 by construction); copy-last ties everywhere. The ranking ceiling is
  not binding (one minus the sequence-tie share 0.969, 0.999 and 1.000 at h = 4, 8, 16; no root
  where all candidates reach the same jaw target).

### Compute

| arm | parameters | s per update (median) | peak CUDA allocated | implementation SHA-256 |
|---|---|---|---|---|
| J, N_J | 177 536 | 0.074 | 2.35 GiB | `78ba9dbe00eb…` |
| L, N_L, R | 2 842 310 | 0.094 | 3.01 GiB | `e5cda6ee1a89…` |

J's predictor has about 1/16 of L's parameters and trains about 22 % faster per update.

## Deviations, incidents and disclosures

- **The run agent was interrupted after Stage E.** The session running the stages was ended by a
  session restart after Stage E had finished; it had no effect on any stage. Evidence: every one of
  the 29 GPU jobs in `gpu_run_ledger.log` ends with status 0; `stageE.console` ends with
  "E-DONE … sha256 9ca7144e…" and `exit 0`, and its `summary.md` is written at 06:38Z; the GPU
  board shows the lock released by the last job at 06:06:29Z and no holder since; no TASK-100
  process was running on 2026-10-19. The only trace is that `runlog.txt` has no closing "stage E
  done" line (its last line is "stage F′ done; Stage E start"); E's completion is recorded by its
  console, the test report and `summary.md`. Housekeeping (this record, the evidence hashes, the
  SSD deletion) was done afterwards on the CPU in a new session (R28.34).
- **Per-stage 1.5 × rule (§8.2).** Measured against the projections made before each stage: F
  1.638 h against 1.58 (1.04 ×), B 1.675 against 1.57 (1.07 ×), T 8.174 against 7.77 (1.05 ×), T′
  1.820 against half of a full T′'s 3.11 (six of twelve jobs; 1.17 ×); V, F′ and E together
  1.708 h against the Stage 0 projection's single row of 1.2 GPU h (**1.42 ×**). No stage
  exceeded 1.5 ×, so no VOID. Disclosed: within V, the `fitpred` pass took 2 837 s (about 189 s a
  model on 25 123 fit-sample roots) against the Stage 0 record's per-model estimate of about 70 s,
  about 2.7 ×; the rule was read, as the projection tables state it, per stage row (V, F′ and E
  are one row in both §8.2 and the Stage 0 projection), and that row stayed under 1.5 ×
  (R28.36).
- **U was set by the cap, not by the probe.** §8.2 allows it and it is what the frozen code did;
  it is the proximate cause of L's G6 failure (above).
- **G6's extension changed one seed's G6 the other way.** L-100100 passed G6 at 20 000 and failed
  it at 30 000, because the extension's last quarter (22 500–30 000) still improved by more than 2 %.
  §6 decides on the post-T′ curve; this is the declared rule, applied as written.
- **R-pred's in-sample fit, the jaw's command leak in G5 (c), G1 (iii)'s projected basis, and G4's
  carried bar** are as disclosed in §6, §7.3 and §13; none changed.
- **SSD working copy deleted (§8.4).** After both HDD stores
  (`outputs/task100-features-{pretrained,random_init}` on `/mnt/data`) were re-hashed in full on
  2026-10-19 and every file matched its feature report and the SSD copy's own manifest, the
  working copy `~/develop/emai/ssd-work/task100-features-pretrained` (28 166 073 721 bytes, 20
  files) and `…-random_init` (28 140 907 423 bytes, 19 files) were deleted at
  2026-10-19T11:50:57Z; `/` went from 51.3 GB to 107.6 GB free. Record:
  `ssd-working-copy-deletion.txt` in the evidence. The HDD hash lists' modification times
  (11:50:49Z, 66 s after the checkpoint hash list) record when the lists were written, not when
  hashing began; the start time of the HDD re-hash (about 56 GB) was not recorded (it
  presumably ran before or alongside the checkpoint hashing). The independent review re-hashed both
  `latents_train.npy` files and every HDD file under 50 MB and found them matching.
- Nothing was re-thresholded, retrained, re-selected or re-chosen after the test numbers were seen.

## Evidence

`~/develop/emai/evidence/task100-run/` (37 MB), **`SHA256SUMS` SHA-256
`4078ad1263abeb58824d4f76820f9d54dee98074ae2a8eba090d2bca135a4093`** (207 files; `sha256sum -c`
passes):

- `summary.md`, `val_report.json` and `VAL_REPORT_SHA256.txt`, `gpu_run_ledger.log`,
  `gpu_hours.txt` (per stage, from the ledger);
- `run/`: the whole small part of `outputs/task100-run/` (test and val reports, every stage console
  and log, `runlog.txt`, the SSD-copy manifests, the `fitpred`, `evalpred` and readouts reports,
  and all 88 readout files);
- `checkpoint-reports/` (every job's `report.json` and `started.json`), `features/` (both stores'
  feature, fit and test reports and episode tables);
- `hashes/`: SHA-256 of all 110 files under `checkpoints/task100-run/` (5.4 GB, kept on the HDD),
  and of every file of both HDD feature stores (29 and 28 files, about 28 GB each, kept);
- `recompute/`: the CPU re-checks of this record (summarizer re-run, gates, G6 and row
  recomputed, the tables);
- `ssd-working-copy-deletion.txt`.

Large artifacts stay where the run wrote them and are not copied: `checkpoints/task100-run/`,
`outputs/task100-features-*`, `outputs/task100-run/{fitpred,evalpred-val,evalpred-test}` (about
127 GB of stored predictions). Stage 0's evidence is `~/develop/emai/evidence/task100-stage0/`
(SHA256SUMS `7bee61cf…`).

## Caveats (each one is required when this result is restated)

- **Offline only, in simulation.** No planner, controller or closed loop ran; a ranking against
  the true future frame is not a planner. WM-PASS admits S5's preregistration; it is not evidence
  that any controller lifts the cube.
- **One run; three model seeds that share one corpus, one recipe, one encoder** (frozen DINOv2
  ViT-S/14, g4) and one floor seed. A pass is evidence for this recipe on this data.
- **A privileged scripted corpus:** `so101-play-v1`, collected by `so101_play_policy_v0`, one red
  cube on a tan table under fixed lighting, partial joint coverage and 26 % clipped commands; val
  and test are random episodes of the same distribution, not held-out conditions.
- **`native_jepa`, not LeWM, passed.** L failed G6 on two seeds at a budget the GPU cap set to a
  third of what the probe rule asked for; whether LeWM's token predictor would pass at saturation
  is **not shown either way**.
- **G5's bars are relative** to the encoded-frame readout (no absolute anchor; set from a 180-episode
  fit and looser than 1.5 × the gated ceiling), and the predicted cube is read to about 1.6–1.8 cm
  in median, with under 6 % of roots within S2's 0.5 cm grasp tolerance; **G5 (c) on the jaw is
  partly met by the schema** (the absolute gripper command).
- **G1 (iii) is a 256-direction projection** of the latent; **R-pred is fitted on in-sample
  predictions**; **G4's 1.10 is a carried allocation**, not calibrated on this corpus.
- **The test split of `so101-play-v1` has now been read.** Any later study that is gated on it is
  no longer held out on this corpus.

## What this cannot show (§13, restated)

That a controller built on J, or on L, succeeds at `so101_cube_lift_v0` (S5's question, with its
own scripted reference arm and controls); that LeWM would or would not pass at a saturating
budget; that the result holds for another corpus, scene, lighting, collector, encoder, grid or
floor seed; that the jaw's motion is learned from images rather than read from the command; that
the predicted cube is precise enough to grasp.
