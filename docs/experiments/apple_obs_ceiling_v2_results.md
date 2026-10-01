# Apple→Plate observation ceiling v2: results (TASK-075)

**Outcome: OBS-NONE. The §7 abandonment clause fires.** Under the frozen readout family, no view
reads the apple-minus-plate offset within τ = 1.0 cm. The learning curve is not still falling on
any view, so the protocol's next step is "a task or condition change" (§7).

- **What was measured.** The frozen family is cross-fitted ridge readouts on three kinds of
  features: frozen DINOv2 pooled 4 × 4 tokens (R_off, the admission readout), full tokens (R_full,
  kernel ridge, linear or RBF) and raw pixels (R_pix, kernel ridge). Four views were read: the
  onboard camera at 112 px (the reference) and at 224 px, the palm-centred hand crop, and the
  overview camera at 224 px. The readouts were measured on 915 moving windows from 218 roots of
  `apple-far-shift-v2`'s 270 train + val roots (253 read).
- **No view is admitted with R_off.** The upper 95 % bound of the median offset error (c_V) is
  2.655 cm for the hand crop, 3.394 cm for onboard 112, 3.321 cm for onboard 224 and 3.438 cm for
  overview 224. τ is 1.0 cm. A1 and A3 fail on every view.
- **R_full and R_pix fail as well.** No view passes B. The best value is R_pix on overview 224:
  c_upper 2.022 cm, with an upper bound of 0.867 on its ratio to the clock prior.
- **Every view fails A3**: none of them beats a clock prior that reads no image (median 2.350 cm).
  A3 does not depend on τ, so, for the frozen R_off as fitted, no value of τ would admit any view
  under the frozen admission rule.
  The sensitivity row admits nothing at 0.5 or at 1.5 cm.
- **A reported-only decomposition** (§4.1's plate and apple target readouts, which gate nothing)
  points to where the error sits. The same pooled-token ridge reads the **plate** position to a
  median of 0.493–0.682 cm. It reads the **apple** position to 2.211–3.122 cm. The offset error
  follows the apple term, not the plate term (§3.4).
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). This is a measurement study. It trained no world model and no policy, ran no
  controller in closed loop, and makes no control claim. No control line is primary.

Protocol: [`apple_obs_ceiling_v2.md`](apple_obs_ceiling_v2.md), preregistered in #117 (merged as
`3434538`). Manifest: `benchmarks/manifests/apple-obs-ceiling-v2.json` (frozen sha
`a64b58331cebac08a95bf62ede6b9b834be188f7e30574887bf4ec0bc1c9ee24`, unchanged; protocol document
sha256 `496149153eee538bc85bfdc087aae29a940d8a23dc807e3deb2db7c783ab9e90`, unchanged). The pre-run
GOs and the operator records are comments on PR #117.

---

## 1. Stage-by-stage record

Each stage ran once, from the detached run worktree
`/home/huhn/develop/emai/worktrees/task075-run` at `34345382cbf25269b1450e22a8b7c0c5359d0c60`
with a clean tracked tree. Each ran on an independent pre-run reviewer's reported GO, posted on
PR #117. The reports are git-ignored and stay on the Linux PC under the run worktree. Times are
UTC, taken from each `report.json`.

| stage / attempt | revision | report (sha256) | started → ended (s) | outcome |
|---|---|---|---|---|
| τ (Stage 0, development) | `26c64d9` | `task075-prereg/outputs/task075-tau/run-1/report.json` (`08a152f14e63902e7e045f8890a000c0eedd0d7ac751af6bd879e6e512c519d3`) | 2026-10-01T12:35:57Z → 12:40:22Z (264.9) | **τ = 1.0 cm** |
| render (Stage 1) | `3434538` | `outputs/task075-render/run-1/report.json` (`753e05f2ed4895069b5e8537b6df6a088a0e50bc63093b20147b3a3e49f6a58c`) | 2026-10-01T14:46:12Z → 14:49:32Z (200.7) | **VIEWS-SEALED** |
| readouts (Stage 2) | `3434538` | `outputs/task075-readouts/run-1/report.json` (`96ab76b2ac7a821868955d77da165034e218fbdc02cd2fc1525ca3423f087b3f`) | 2026-10-01T14:55:31Z → 15:21:25Z (1 553.6) | **OBS-NONE** |

GO comments on PR #117: render
[5933844704](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/117#issuecomment-5933844704),
readouts
[5934015019](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/117#issuecomment-5934015019).
Operator records: render
[5933937501](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/117#issuecomment-5933937501),
readouts
[5934526719](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/117#issuecomment-5934526719).
Neither stage voided, and neither was repeated.

**Common to both stages.** In both reports the frozen sha256 is `a64b5833…ee24`. The revision at
the end equals the revision at the start. `tracked_tree_dirty` and `test_split_decoded` are false.
All 77 pins at the end equal the pins at preflight and the manifest's `hashes`. The encoder
digests (pretrained `3a697b87…2af27`, floor `546b9011…729d2`) match the manifest. The platform
was the Linux PC: RTX 5080, MuJoCo 3.13.0, torch 2.14.0+cu130, `MUJOCO_GL=egl`, strict CUDA
determinism. The resident GR00T service (pid 14247, 6 626 MiB) was on the GPU at both starts and
was not touched.

### 1.1 τ (Stage 0, before the freeze; recorded in protocol §2)

τ = the largest planted target error at which P-3's pick followed by e9's place still leaves the
apple at rest on the plate on ≥ 28/32 resets, with every smaller level passing. The condition was
TASK-074's 9 cm move at step 300, on development seeds 55000–55031, which are now spent.

| planted error \|e\| (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| counted successes (of 32) | 31 | 31 | **28** | 22 | 23 | 21 | 15 | 11 | 8 |
| at rest (of 32) | 31 | 31 | 28 | 22 | 23 | 21 | 15 | 11 | 13 |
| ≥ 28? | yes | yes | yes | **no** | no | no | no | no | no |

- **τ = 1.0 cm**, and it sits exactly on the bar. The Wilson 95 % interval at 1.0 cm is
  [0.72, 0.95].
- 10 of the 32 seeds are non-monotone across the levels.
- τ is anisotropic: errors towards the robot's right (−y) fail earlier.
- G-repro passed, and G-planted was clean (largest difference 0.0 m). Start load was 0.05 / 1.43,
  and the peak tree PSS was 7.94 GiB.

### 1.2 Render (Stage 1): VIEWS-SEALED

The stage re-simulated 270 train + val roots (240 + 30) from their stored metadata and rendered
the three new views at the 30 rendered steps per root. It also rendered every view's apple-hidden
and plate-hidden frames at the six eval steps.

- **Boundary** (`cohort_first_render_utc`): 2026-10-01T14:46:14Z.
- **G-repro-render.** Every array was equal on **270/270** roots, with no failed roots. Of the
  7 590 stored reference frames compared, **7 576 were identical**, 14 were within the render
  rule and **0 were outside it**. The 14 came from 14 roots, one frame each. Each differed by
  at most one level, in at most 10 pixels (the rule allows 16).
- **Test-split exclusion.** 30 test roots were excluded, never simulated, rendered or decoded.
  The digest of their ids is `4fa7512c…57bf`.
- **Views store** `apple-far-shift-v2-views`: manifest sha256
  `ea627a8fb6b8d7d4057a1a02b1d9c698549fe0a1719808c49a3e6d8a86554d77`. It holds 444 014 634 B,
  at most 1 867 036 B per root. The median was 4.53 s per root on 6 workers.
- **Source.** The corpus manifest is `fe7ab9150216…b134bd`, TASK-074's sealed
  `apple-far-shift-v2`.
- **Resources.** Load at start was 0.41 / 1.09 (1 / 5 min) and MemAvailable 25.25 GiB. Disk was
  32.05 GiB free, against 10.70 GiB needed; the minimum during the run was 31.64 GiB. The peak
  tree PSS was 7.98 GiB (ceiling 12), across 8 processes. The pool closed with 6 joined and 0
  killed.

### 1.3 Readouts (Stage 2): OBS-NONE

- **G-repro-off** ran first and reproduced TASK-074 exactly on all 8 fields: encoded
  2.8722625765232763 cm, persistence 4.350821813169937 cm, clock prior 3.1225649505050876 cm,
  90 windows, 22 roots, R_off readout sha256 `260af3f7…803157`, 5 175 rows and 225 groups. It
  used 44 781 band frames and took 113.7 s.
- **Read roots** (TASK-074's rule: the band reaches past step 400): **253** of the 270, of which
  225 are train and 28 val. They fall into outer folds of 51, 51, 51, 50 and 50 roots.
- **Boundary** (`first_outcome_utc`): 2026-10-01T14:57:40Z.
- **Per view.** 10 626 frames were featurised, made up of 7 590 visible frames and 1 518 each of
  apple-hidden and plate-hidden. In the CPU anchor check, the largest |difference| was
  2.86e-5 to 5.34e-5, within the 1e-3 bound. Featurisation plus the full-token Gram took 76–92 s,
  the pixel Gram 8.5–55 s and the cross-fit 221–228 s.
- **Windows.** O2's moving cohort gave **915 windows on 218 roots**, the same for every view.
- **Per-window error files** (`errors_<view>.npz`, sha256): onboard112
  `b4308c099b5b3d32fd382d8825a5d1501fb9c75eb29e203c2f308fa28a0ffdf0`, onboard224
  `d7b4e7f5fdcbc83cb32e41ea4070e774eaa3c40adf9778de7c1d3e5213312717`, handcrop
  `bc6b877c61fbaf08ea11a0c69b3f3dccf77af61e03c0c40efdc2d23442af947d`, overview224
  `ee4060fadec6e32069cd295c070fa989457b386f9a7a864bef0ac5b8a54a9b0b`.
- **Resources.** Load at start was 0.22 / 1.17 and MemAvailable 25.02 GiB. The GPU had
  8.51 GiB free at the start; it peaked at 0.85 GiB allocated and 0.96 GiB reserved, against a
  3.0 GiB cap. The peak tree PSS was 8.50 GiB (ceiling 12), in 1 process. `non_finite_fields`
  is empty, and the log has no traceback.
- **The decision**, verbatim from the report: `row` OBS-NONE, `admitted` [], `chosen_view` null,
  `abandonment_clause_fires` true, `learning_curve_still_falling` false for all four views,
  `onboard_representation_passes` null, `downstream_bar_interval` null, `sensitivity_admitted`
  {0.5: [], 1.5: []}, `tau_cm` 1.0.

## 2. Admission (§6.1, R_off) and representation (§6.2, R_full and R_pix)

### 2.1 Admission: no view is admitted

All four conditions must hold, with τ = 1.0 cm. The "upper" and "lower" columns are the 95 %
bounds the frozen checks read.

| view | A1: c_V (≤ 1.0) | A2: R_off / R_floor upper (< 1.0) | A3: R_off / clock upper (< 1.0) | A4: plate-hidden lower (> 1.0) | admitted |
|---|---|---|---|---|---|
| onboard112 | 3.394 **fail** | 1.236 **fail** | 1.486 **fail** | 6.722 pass | no |
| onboard224 | 3.321 **fail** | 1.280 **fail** | 1.444 **fail** | 6.817 pass | no |
| handcrop | 2.655 **fail** | 0.912 pass | 1.145 **fail** | 3.618 pass | no |
| overview224 | 3.438 **fail** | 1.359 **fail** | 1.486 **fail** | 7.383 pass | no |

### 2.2 Representation (B = A1, A3, A4 with R_full or R_pix): nothing passes

| view | R_full c_upper | R_full / clock upper | R_full plate-hidden lower | R_pix c_upper | R_pix / clock upper | R_pix plate-hidden lower | passes |
|---|---|---|---|---|---|---|---|
| onboard112 | 2.699 | 1.167 | 6.794 | 2.426 | 1.036 | 4.120 | none |
| onboard224 | 2.608 | 1.122 | 5.327 | 2.353 | 1.002 | 4.007 | none |
| handcrop | 2.450 | 1.058 | 3.237 | 2.941 | 1.282 | 4.959 | none |
| overview224 | 2.727 | 1.174 | 5.001 | **2.022** | **0.867** | 3.591 | none (B1 fails) |

Only one readout beats the clock prior with 95 % confidence: R_pix on overview 224 (B3 passes).
B1 fails for every readout on every view. The inner CV picked RBF or linear kernels fold by
fold (`views.<view>.selections`), so the kernel family included a nonlinear option on every view.

### 2.3 Every readout's median error (cm), with 95 % root-clustered intervals

915 windows on 218 roots per view. The clock prior, o\* and R_proprio do not read the image, so
they are the same for every view.

| readout | onboard112 | onboard224 | handcrop | overview224 |
|---|---|---|---|---|
| **R_off** | 3.232 [3.003, 3.394] | 3.084 [2.846, 3.321] | 2.482 [2.326, 2.655] | 3.232 [3.001, 3.438] |
| R_floor (random-init) | 2.782 [2.606, 3.034] | 2.612 [2.467, 2.788] | 2.945 [2.772, 3.085] | 2.567 [2.394, 2.729] |
| clock prior | 2.350 [2.195, 2.550] | = | = | = |
| o\* (TASK-074's prior) | 2.591 [2.366, 2.810] | = | = | = |
| R_proprio (robot state only) | 2.482 [2.290, 2.637] | = | = | = |
| persistence (R_off on the frame at t) | 4.859 [4.654, 5.073] | 4.915 [4.605, 5.238] | 4.165 [4.010, 4.392] | 4.907 [4.613, 5.170] |
| R_full | 2.517 [2.339, 2.699] | 2.381 [2.173, 2.608] | 2.302 [2.155, 2.450] | 2.509 [2.260, 2.727] |
| R_pix | 2.189 [2.005, 2.426] | 2.091 [1.967, 2.353] | 2.784 [2.592, 2.941] | 1.840 [1.663, 2.022] |
| R_off, plate hidden | 7.143 [6.722, 7.562] | 7.226 [6.817, 7.597] | 3.812 [3.618, 4.069] | 7.856 [7.383, 8.272] |
| R_off, apple hidden | 5.681 [5.311, 6.008] | 4.792 [4.611, 5.188] | 4.541 [4.324, 4.813] | 8.082 [7.422, 8.555] |
| R_full, plate hidden | 7.055 [6.794, 7.347] | 5.648 [5.327, 6.021] | 3.514 [3.237, 3.717] | 5.282 [5.001, 5.501] |
| R_full, apple hidden | 4.308 [4.073, 4.578] | 3.437 [3.150, 3.655] | 4.042 [3.762, 4.284] | 5.996 [5.524, 6.557] |
| R_pix, plate hidden | 4.350 [4.120, 4.586] | 4.185 [4.007, 4.448] | 5.170 [4.959, 5.481] | 3.751 [3.591, 3.942] |
| R_pix, apple hidden | 2.218 [2.058, 2.420] | 2.141 [1.995, 2.388] | 2.915 [2.748, 3.088] | 1.956 [1.805, 2.158] |
| plate target (reported) | 0.493 [0.455, 0.511] | 0.510 [0.471, 0.538] | 0.682 [0.640, 0.721] | 0.550 [0.511, 0.591] |
| apple target (reported) | 3.122 [2.900, 3.278] | 3.040 [2.786, 3.208] | 2.211 [2.057, 2.365] | 3.074 [2.836, 3.244] |

### 2.4 Paired ratios (cluster-median ratio, 95 % interval)

| ratio | onboard112 | onboard224 | handcrop | overview224 |
|---|---|---|---|---|
| R_off / R_floor | 1.162 [1.061, 1.236] | 1.181 [1.075, 1.280] | 0.843 [0.785, 0.912] | 1.259 [1.165, 1.359] |
| R_off / clock | 1.375 [1.247, 1.486] | 1.312 [1.177, 1.444] | 1.056 [0.962, 1.145] | 1.375 [1.247, 1.486] |
| R_off / o\* | 1.248 [1.138, 1.373] | 1.191 [1.073, 1.324] | 0.958 [0.876, 1.058] | 1.247 [1.134, 1.370] |
| R_off / R_proprio | 1.302 [1.191, 1.420] | 1.243 [1.129, 1.371] | 1.000 [0.923, 1.092] | 1.302 [1.186, 1.425] |
| R_full / clock | 1.071 [0.974, 1.167] | 1.013 [0.911, 1.122] | 0.980 [0.895, 1.058] | 1.068 [0.949, 1.174] |
| R_pix / clock | 0.931 [0.844, 1.036] | 0.890 [0.815, 1.002] | 1.185 [1.078, 1.282] | 0.783 [0.703, 0.867] |

## 3. Reported-only statistics (§6.4; they gate nothing)

### 3.1 The 87.5th percentile, the τ-curve prediction and the signed bias of R_off

| view | 87.5th percentile (cm) | predicted successes of 32 (τ curve) | signed mean x (cm) | signed mean y (cm) |
|---|---|---|---|---|
| onboard112 | 6.100 [5.745, 6.452] | 16.09 [15.57, 16.62] | −0.179 [−1.362, +0.658] | −0.152 [−0.600, +0.583] |
| onboard224 | 6.323 [5.776, 6.969] | 16.43 [15.82, 17.04] | −0.519 [−1.709, +0.300] | −0.018 [−0.472, +0.723] |
| handcrop | 4.647 [4.339, 4.881] | 18.56 [17.99, 19.11] | −0.359 [−1.710, +0.504] | −0.144 [−0.613, +0.631] |
| overview224 | 6.633 [6.222, 7.077] | 16.22 [15.68, 16.76] | −0.043 [−1.279, +0.865] | −0.278 [−0.761, +0.492] |

- The τ-curve prediction maps each window's error to the measured success fraction. It is
  optimistic above 5 cm, which maps to 8/32. It gives **16.1–18.6 of 32** across the views,
  against the 28/32 bar.
- No pooled signed mean is distinguishable from zero: every interval contains 0. So the error is
  not a single constant offset across windows. A zero pooled mean does not rule out a bias that
  depends on the condition (for example on the move's direction; τ is anisotropic in y), which
  was not measured.

### 3.2 The sensitivity row

With A1–A4 unchanged, the views admitted would be: at τ = 0.5 cm, **none**; at τ = 1.5 cm,
**none**. A3 fails on every view and does not depend on τ, so no τ would admit a view under the
frozen rule. This is a property of the frozen R_off as fitted and of the rule, derived here; it is
not a separate computation.

### 3.3 Subsets: shifted and unshifted roots (reported, not gated)

| view | subset | windows / roots | R_off | R_floor | clock prior |
|---|---|---|---|---|---|
| onboard112 | shifted | 652 / 160 | 2.992 [2.851, 3.234] | 2.705 [2.542, 2.932] | 2.228 [2.036, 2.427] |
| onboard112 | unshifted | 263 / 58 | 3.940 [3.333, 4.378] | 3.110 [2.713, 3.514] | 3.021 [2.535, 3.442] |
| onboard224 | shifted | 652 / 160 | 2.853 [2.662, 3.102] | 2.514 [2.345, 2.697] | 2.228 [2.036, 2.427] |
| onboard224 | unshifted | 263 / 58 | 3.566 [3.117, 4.177] | 2.895 [2.639, 3.144] | 3.021 [2.535, 3.442] |
| handcrop | shifted | 652 / 160 | 2.309 [2.165, 2.496] | 2.812 [2.623, 2.982] | 2.228 [2.036, 2.427] |
| handcrop | unshifted | 263 / 58 | 2.970 [2.734, 3.227] | 3.314 [2.962, 3.789] | 3.021 [2.535, 3.442] |
| overview224 | shifted | 652 / 160 | 2.931 [2.700, 3.176] | 2.233 [2.064, 2.442] | 2.228 [2.036, 2.427] |
| overview224 | unshifted | 263 / 58 | 3.969 [3.554, 4.358] | 3.454 [3.123, 3.857] | 3.021 [2.535, 3.442] |

On the shifted roots, which are τ's kind of condition, R_off is still at 2.3–3.0 cm in median, and no view's
median is below the clock prior's median of 2.23 cm. On the hand crop the intervals overlap
(2.309 [2.165, 2.496] against 2.228 [2.036, 2.427]).

### 3.4 Representation checks and what the reported readouts suggest (descriptive, not gates)

- **The plate is in view, and the readouts use it.** Hiding the plate raises every readout's
  error by a large amount. For R_off the rise is 3.2 to 7.1 cm on onboard 112, and 2.5 to 3.8 cm
  on the hand crop. A4 passes on every view. A4 is necessary, not sufficient (§9b).
- **The apple contributes little to the pixel readout.** Hiding the apple barely moves R_pix
  (onboard 112: 2.189 to 2.218 cm; overview 224: 1.840 to 1.956 cm). It does move R_off and
  R_full a lot. This supports that R_pix's offset reading does not come from seeing the
  apple. That it comes from the plate and, likely, the arm is inferred, not measured: the
  plate-hidden rise is consistent with it, but A4 is necessary, not sufficient (§9b), and the
  arm's contribution was not isolated.
- **The offset error follows the apple term.** The same pooled-token ridge, fitted to the plate
  position or to the apple position instead of the offset, reads them as follows:
  - the **plate** to a median of 0.493, 0.510, 0.682 and 0.550 cm (upper bounds 0.511–0.721);
  - the **apple** to 3.122, 3.040, 2.211 and 3.074 cm,

  for onboard 112, onboard 224, the hand crop and overview 224 in that order. The offset is
  apple minus plate, and its error (2.48–3.23 cm) sits at the apple term's level. In the carry
  phase the apple is in the hand and partly occluded by it. **Caveats.** These are §4.1's
  reported plate and apple target readouts. No floor, clock-prior, plate-hidden or τ check was
  defined for them, so they are not admissions. Errors of the two components do not add
  linearly. In this corpus e9's arm moves towards the (possibly mis-aimed) plate, so part of the
  plate reading may come from the arm's pose.
- **Pretrained features do not beat the random-init floor except on the hand crop.** R_off /
  R_floor is above 1 with 95 % confidence on onboard 112, onboard 224 and overview 224
  (1.16–1.26). On the hand crop it is 0.843 [0.785, 0.912], the only A2 pass.
- **The hand crop is the best R_off view, but it is no better than reading the robot state.**
  R_off / R_proprio is 1.000 [0.923, 1.092] on the hand crop. R_proprio reads no image (2.482 cm).

### 3.5 The learning curve (R11)

| view | median(e_50) − median(e_100) (cm) | median(e_25) − median(e_50) (cm) | still falling (frozen rule: lower bound of the first > 0) |
|---|---|---|---|
| onboard112 | −0.782 [−1.033, −0.488] | 0.127 [−0.147, 0.378] | **false** |
| onboard224 | −0.349 [−0.586, −0.126] | −0.178 [−0.529, 0.107] | **false** |
| handcrop | −0.181 [−0.361, 0.061] | 0.609 [0.331, 0.832] | **false** |
| overview224 | −0.731 [−0.991, −0.455] | −0.148 [−0.360, 0.098] | **false** |

- The frozen rule's "still falling" is false on every view.
- **Stated precisely, the curve is not still falling, and on three views it is reversed.** On
  onboard 112, onboard 224 and overview 224, R_off fitted on 50 % of each fold's training roots
  has a *lower* median error than the one fitted on 100 %, and the interval excludes zero. On the
  hand crop, 25 → 50 % fell (0.609 [0.331, 0.832]) but 50 → 100 % did not detectably fall.
- This reversal is an anomaly, recorded without an explanation. It hints that R_off's λ
  selection may not track the reported error: λ is chosen by inner-CV MSE over all fit rows
  (every 8th band frame from 384 to 560), not by the moving-window median error (untested). On
  the hand crop the 50 % fit's median, 2.300 cm, is even below the clock prior's 2.350 cm.
- **The frozen rule does not select candidate (c), a larger corpus.** The reversal makes the curve
  uninformative about data scaling; it is not evidence against more data. It does not change the
  row: every learning-curve median is ≥ 2.30 cm, far from τ = 1.0 cm.

### 3.6 Comparison with TASK-074's reference numbers (descriptive)

G-repro-off reproduced TASK-074's own numbers byte for byte (§1.3). On TASK-074's 90 val windows,
the reference readout read 2.872 cm against a clock prior of 3.123 cm. On this study's 915
cross-fitted windows, the same view reads 3.232 cm against 2.350 cm. The window set differs
(218 roots instead of 22), as do the fit (the dual form on four of five folds, about 202 roots,
instead of the primal form on 225 train roots) and the clock prior's fit set. On the larger
window set, the clock prior that reads no image is **better in median** than every R_off. It is
detectably better (ratio lower bound > 1) on onboard 112, onboard 224 and overview 224; on the
hand crop the ratio R_off / clock is 1.056 [0.962, 1.145], whose interval includes 1. TASK-074's small
margin over the clock prior (0.25 cm) does not hold on this cohort.

## 4. What OBS-NONE shows, stated precisely

**Shown.** Under the frozen readout family, no view reads the apple-minus-plate offset at t + 16
on O2's moving windows within τ = 1.0 cm. The family is cross-fitted ridge on frozen DINOv2
pooled 4 × 4 tokens, kernel ridge (linear or RBF) on full tokens, and kernel ridge on raw pixels.
The views are onboard 112, onboard 224, the hand crop (112 px from a 320 px render) and
overview 224. The data are `apple-far-shift-v2`'s 270 train + val roots (253 read, 218 with
moving windows), under TASK-074's plan. More specifically:
- the best upper bound with the admission readout is 2.655 cm (the hand crop), and the best
  with any readout is 2.022 cm (R_pix, overview 224);
- no R_off beats the clock prior, and only one readout of any kind does (R_pix, overview 224);
- the learning curve is not still falling on any view.

**Not shown.** The protocol's scope (§9b, §6.2) and the frozen family limit what this result
covers:
- **Only linear and kernel-ridge readouts on frozen features were tested.** No trained nonlinear
  detector (an MLP or attention head over tokens, a CNN on pixels, a keypoint or segmentation
  model) was tested. No fine-tuned or task-trained features were tested.
- **No temporal aggregation.** Each readout reads one frame. Aggregating over frames or decisions,
  or filtering, was not tested.
- **No other sensors.** Depth, stereo, wrist cameras (the scene has none, R10) and tactile or
  contact signals were not read. Only `overview` and `onboard_rgb` exist in the scene.
- **One encoder, one input size per view, one floor seed, one crop size**, and one corpus. That
  corpus is TASK-074's privileged scripted e9 plan with mis-aims and noise, not P-3's own carry.
- **The quantity is the offset at t + 16 on O2's windows.** Other phases and other quantities
  are not admitted or refuted. That includes the plate position alone, whose reported readout is
  0.49–0.68 cm (§3.4).
- **τ is one run at exactly its bar**, anisotropic, and models a constant target error. A
  re-measured τ could move by one level, but no τ admits a view here, because A3 fails (§3.2).
- Nothing here is about a world model's prediction or about any controller.

## 5. The clause and what it closes

**The row is OBS-NONE, and the abandonment clause fires** (`abandonment_clause_fires: true`).

The clause's scope, verbatim (protocol §7, `CLAUSE_SCOPE`):

> "a LeWM world-model task (planner or critic) for the place phase of `apple-to-plate-v2` under
> TASK-074's condition, built on any of these four views with frozen DINOv2 features: none is
> preregistered without new evidence of a different kind (a new view, a new readout family or a
> task or condition change)". The LeWM backend, the v2 task and the product goal stay open.

The protocol's next-step entries for OBS-NONE, verbatim (§7, "Next steps by row"):

> | OBS-NONE, learning curve still falling on any view (the lower bound of median(e_50) − median(e_100) > 0) | candidate (c): a larger corpus, preregistered with its own memory plan |
> | OBS-NONE, otherwise | a task or condition change |

The learning curve is not still falling on any view (§3.5), so the entry that applies is
**"OBS-NONE, otherwise: a task or condition change"**. The report's consequence, verbatim:
"clause: no view reads the offset within tau, even raw pixels; if the learning curve is still
falling the next step is a larger corpus (candidate (c)), otherwise a task or condition change".

**What the clause closes:** preregistering another LeWM planner or critic for the place phase
under TASK-074's condition on these four views with frozen DINOv2 features, without new evidence
of a different kind.

**What stays open:**
- the LeWM backend, the v2 task and the product goal;
- new views, new readout families, and task or condition changes, as kinds of new evidence;
- candidate (b), real-robot pretraining, which stays deferred (§7).

## 6. Context

- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (0/150 per model). τ's
  counts are privileged and scripted (P-3's pick plus e9's place aimed at a planted target), not
  learned results.
- TASK-074 closed INCONCLUSIVE because a W-independent readout ceiling (2.872 cm) sat above an
  uncalibrated 1.0 cm bar. **This study confirms the observation as the limit at the frozen
  readout family, with a calibrated bar.** Measured against τ, the offset is not read within the
  place's tolerance from any of the four views, even with full tokens or raw pixels.
- It also refines where the limit sits. Per the reported plate and apple target readouts (§3.4),
  the offset error tracks the apple term (2.2–3.1 cm), while the plate is read to about 0.5 cm.
  This agrees with TASK-074's design Probe C (`apple_lewm_planner_v2.md`:172–179): a
  kernel-ridge *plate* readout trained with far frames read 0.55–0.69 cm, and e9's place aimed at
  it scored 31/32 (9 cm) and 30/32 (12 cm) in closed loop. That was a development probe on P-truth
  frames, not a gated result.

## 7. Next-step options (for the owner, or Claude under the owner's delegation, to decide)

The applicable entry is "a task or condition change" (§5). The options below are concrete
candidates. For each, it says whether the clause permits it, how it moves towards a LeWM-driven
G1 policy, and what its first decisive experiment would cost. Costs are estimates from this
task's measured stage times (τ: 288 attempts in 265 s on 6 workers; readouts: about 26 min).

### Option 1 (recommended): the place reads the plate, not the offset (a task-interface change)

- **What.** Change what the place consumes. The target becomes the plate position read from the
  image, as in TASK-074's perception twin H-twin. The apple stays where e9's primitive already
  puts it, by kinematics. The apple-minus-plate offset is no longer read from the image.
- **Why.** The reported plate readout is 0.49–0.68 cm, under τ, against 2.2–3.1 cm for the apple
  term (§3.4). τ itself is defined as an error in the *aimed plate target* (§1.1). Probe C's
  readout twin scored 31/32 and 30/32 in development.
- **Does the clause permit it?** The perception-twin closed loop is not a LeWM world-model task,
  so the clause does not close it. It runs no world model. A later *LeWM* task built on a plate
  target would change the task interface: the quantity changes from the offset to the plate. The
  owner, or Claude under the owner's delegation of 2026-09-30, must rule explicitly whether that
  counts as "a task change" under §7. It is none of the clause's other two kinds (a new view, a new readout family), and it rests partly on a
  reported-only number, so it must not be presented as a silent reopening.
- **How it moves towards LeWM.** It would give the first gated, image-in-the-loop place on v2
  with a frozen encoder: the perception baseline (TASK-074's T bar) that any LeWM place planner
  must tie. It would also give a calibrated readout bar c ≤ B ≤ τ that is feasible on the plate
  (upper bounds 0.51–0.72 cm against τ = 1.0 cm). **Limit, stated plainly:** with a static,
  visible plate, a world model is not needed for the place (TASK-074's protocol §9b, `apple_lewm_planner_v2.md`:713–714: "A pass
  would not show that a world model is needed"). LeWM earns its place only in a follow-up
  condition where the target must be *predicted*, not read. Examples: the plate keeps moving
  during the place, or it leaves the view during the carry. That follow-up is the LeWM task this
  option leads to.
- **First decisive experiment.** Two parts, under a new preregistration:
  1. *Offline:* the plate readout's own floor, clock-prior and plate-hidden checks, with the
     frozen τ, on the sealed views store (no new render). About 30 min of machine time.
  2. *Closed loop:* H-twin, which is P-3's pick plus e9's place aimed at the frozen-DINOv2 plate
     readout of the onboard 112 px frame, run on 32 fresh development resets under TASK-074's
     9 cm condition. The bar is 28/32 (the τ family). It runs beside H-handover (true plate,
     the ceiling) and an image-free clock-prior arm (the control). About 96 attempts, under
     10 min of machine time.

  The code largely exists (`place_planner` mode `twin`, TASK-074's R-plate). Expected effort is
  about a day of protocol and review work. **Decisive either way.** A pass locates the limit in
  the offset quantity, not the camera. A fail despite a 0.5 cm median plate reading points at
  per-decision variance, tails or τ's anisotropy, which leads to Option 2.

### Option 2: closed-loop correction at the place (a new place primitive)

- **What.** Give e9's place primitive its own local feedback, so the upstream target only needs
  to land within the correction's capture radius. The feedback could be contact-triggered release
  plus a short local visual servo on the hand crop, or on contact.
- **Does the clause permit it?** Developing and measuring the corrected place primitive (with
  its own τ curve) is not a LeWM world-model task, so the clause does not close it. The protocol
  does not classify a new place primitive as "a task or condition change": `apple-to-plate-v2`
  and TASK-074's condition stay the same, and the primitive is part of the solution (changing it
  changes τ). Whether a later LeWM task built on it counts as "a task change" under §7 needs the
  same explicit ruling as Option 1, by the owner, or Claude under the owner's delegation of
  2026-09-30. It must not be presented as a silent reopening.
- **How it moves towards LeWM.** It separates a coarse target from fine precision, which the
  local loop supplies. If a later LeWM task is ruled admissible, a LeWM planner could supply the
  coarse target, with its bar set by the capture radius instead of 1 cm.
  **Caveat:** the clock prior already reads 2.35 cm with no image. If the capture radius exceeds about 2.5 cm, an image-free prior would suffice, and the LeWM part
  would not be tested. The condition then needs targets that a prior cannot guess.
- **First decisive experiment.** Re-measure the τ curve (levels 0–5 cm, 32 development resets
  per level) with the corrected primitive. It passes if τ_servo exceeds every R_off c_upper,
  that is, τ_servo ≥ 3.5 cm. The servo must read only non-privileged signals; a privileged
  servo is a ceiling only. Cost: several days to develop the servo, then about 5 min of machine
  time per τ curve.

### Option 3: a condition change that raises τ (a larger plate, or a different move)

- **What.** Enlarge the plate, or change the move magnitude or direction (τ is anisotropic: +y
  tolerated about 2 cm), so that τ exceeds the readout ceiling.
- **Does the clause permit it?** A different move magnitude or direction is plainly a condition
  change. A larger plate is a scene or task variant of v2, which needs its own review as a
  separate benchmark.
- **How it moves towards LeWM.** Weakly. A3 fails on every view independent of τ (§3.2), so a
  larger τ alone admits nothing under the frozen admission. A condition with τ ≳ 2.4 cm would
  also let the image-free clock prior succeed, so vision, and with it LeWM, would not be
  exercised. It would only help if paired with a condition that makes the clock prior worse,
  such as wider plate-position variance.
- **First decisive experiment.** A τ curve under the scaled plate on fresh development seeds,
  about 5 min of machine time, plus the clock prior's error under that condition. Not
  recommended.

### Option 4: a new readout family on the offset (a trained nonlinear detector or temporal aggregation)

- **What.** An MLP or attention head over the full tokens, or over stacked frames from the six
  decisions, trained on the same 253 roots with the same cross-fitting and gates.
- **Does the clause permit it?** The scope names "a new readout family" as new evidence of a
  different kind. It is not, however, the next step that §7's table names for a flat curve.
  The owner, or Claude under the owner's delegation of 2026-09-30, must rule. Fine-tuning the encoder on this corpus would come close to TASK-062's
  closed in-corpus encoder-training line.
- **How it moves towards LeWM.** If it passed, the TASK-074 formulation (the offset cost) would
  become feasible again on an onboard view. **Evidence against:** RBF kernel ridge is already
  nonlinear and reached only 2.30–2.52 cm (R_full). The error sits in the apple term (2.2–3.1 cm),
  where the apple is in the hand and partly occluded. The best view (the hand crop) is no better
  than reading the robot state.
- **First decisive experiment.** An offline cross-fitted trained head on the sealed views store,
  with A1–A4 unchanged and τ unchanged. About 1–2 h on the GPU. Not recommended first.

### Recommendation

**Option 1.** It is the cheapest decisive test (under 1 h of machine time after a day of
protocol work) and needs no new camera or scene. It uses evidence this study already produced:
the plate readout is under τ while the apple term is not. It also tells us which of Options 2 and
4 is worth paying for. Stated plainly, the LeWM step comes after it: Option 1 only sets the
perception baseline and the quantity. The LeWM task it leads to needs a condition in which the
place target must be predicted, and that condition is itself a task or condition change that
needs its own preregistration.

## 8. Process

- #117 (protocol, τ, code, smokes) merged as `3434538` on an independent APPROVE, re-reviewed
  after fixes.
- The render and readouts stages each ran once on their own reported GO. Neither voided, so
  there was no repeat.
- Nothing was re-thresholded, refitted or re-selected after numbers were seen. No frozen block,
  pin or protocol text was changed in this results PR. The frozen sha256 and all 77 pins are
  unchanged, which the protocol's test checks.
- No experiment was run for this document. Every number above is read from the two stage
  reports, the τ record in the manifest, or the protocol text cited.
