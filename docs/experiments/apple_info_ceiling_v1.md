# Apple→Plate information ceiling v1: can the reset frame say where the apple is? (TASK-059)

**Status: preregistration. No readout has been fitted on any image or encoder feature.** The only
numbers below come from a pre-freeze calibration. It renders frames and evaluates prior-only
predictors that read no observation (`scripts/calibrate_info_ceiling.py`, committed with this
document). **The test split is never decoded. The development cohort D (45000–45007,
45100–45107) and cohort C (45300–45339) are never simulated.**

This is a **perception/data task**, as `apple_policy_v1.md` §7 requires now that its abandonment
clause has fired (TASK-057, Outcome X). It trains no controller. It preregisters no control
formulation, and it does not present behaviour cloning or sampling-based planning as the
project's control approach. **Learned Apple→Plate is still 0 successes.**

Predecessor: [`apple_policy_diagnostics_v1.md`](apple_policy_diagnostics_v1.md) /
[`_results.md`](apple_policy_diagnostics_v1_results.md). Manifest:
`benchmarks/manifests/apple-info-ceiling-v1.json`.

---

## 1. The question

TASK-057 established three facts, all on the development cohort:

- **Proprioception carries no reset information.** At reset the robot state is the same on every
  reset, so only the image can locate the apple.
- **The arms get the first command wrong.** At step 0 every behaviour-cloning arm commands a
  `right_dx` that does not depend on the reset. The scripted expert's `right_dx` changes sign with
  the apple's position. The arms miss it by 3.8–5.3× the D1 threshold (closed loop), and they miss it on their
  own training frames too (§13.4 there).
- **The observation pipeline is not the cause.** The observation the loop receives is
  byte-identical to the training observation (D1-pipeline).

Once motion has started, proprioception alone locates the apple to about 1 cm (claim audit S5-04),
so a behaviour-cloning shortcut is available *after* step 0 but not *at* it.

**Question:** can a readout recover the apple position, and the expert's step-0 command (above
all the sign and size of `right_dx`), from the **reset frame**? And if it can, which feature
source carries that information?

## 2. Facts established before freezing (calibration; fits nothing)

Command: `uv run --no-sync python scripts/calibrate_info_ceiling.py --output
outputs/task059-info-ceiling/calibration-v2.json` (v1 of the artifact predates review fixes that
added fields; every shared value except elapsed time is identical). Scope: the 190 train + val roots of
`data/apple-wide-v1` (170 train, 20 val), including aim-offset roots (§3). Elapsed about 20 s on
CPU. Every figure below is in the manifest under `calibration`.

### 2.1 The 112 px pipeline reproduces exactly

- **Re-rendering matches the stored frames.** Rendering each reset at 112 px from its recorded
  coordinates gives a frame **byte-identical to the stored training frame on 190/190 roots**.
- **Reset proprioception does not vary.** Its maximum per-dimension standard deviation across the
  190 resets is **0.0**.
- **The apple label is the reset truth.** It agrees with the reset's simulator truth to 1.5e-8 m.

### 2.2 On most resets the apple cannot be seen at reset

We counted the apple's pixels at reset with a segmentation render of the `onboard_rgb` camera,
from the reset pose. **This count is privileged and used for analysis only. It is never a readout
input.**

| | 112 px | 448 px |
|---|---|---|
| resets with **0** apple pixels (occluded) | **111 / 190** (val: 12 / 20) | **106 / 190** |
| visible pixels: p25 / p50 / p75 / p90 / max | 0 / 0 / 4 / 9 / **21** | 0 / 0 / 57.75 / 141.1 / 334 |

- **Occlusion depends on the apple's y, not its x.** The mean apple y is −0.193 m on occluded
  resets and −0.166 m on visible ones; the mean x is 0.341 m on both. The apple is hidden behind
  the right hand: on all 111 occluded resets the body drawn at the apple centre's projection
  (448 px) is `right_wrist_yaw_link` (`body_at_apple_centre_on_occluded_448`).
- **So occlusion is roughly independent of the dx sign.** 57.3 % of dx > 0 resets are occluded,
  and 60.3 % of dx < 0 resets.
- **Where the apple is visible, it is tiny.** At 112 px the image has 12 544 pixels, and the apple
  covers at most 21 of them.

### 2.3 Stored frames of different resets are almost identical

The robot pose is the same at every reset, so two stored reset frames differ only where the apple,
the plate and their shadows fall:

- the nearest other reset's frame differs from a given frame by as little as **0.038 levels** of
  mean absolute difference;
- the median over resets of that nearest-frame difference is 0.125 levels.

**We state the consequence here, before any readout exists.** On an occluded reset the frame
differs from other resets only through the plate, which is placed independently of the apple, and
through any apple shadow the camera can see. **Unless such a cue exists, the best any readout can
do on occluded resets is the prior.** §10 tests any readout that beats the prior there before
anyone believes it.

The step-0 dx sign then has a ceiling:

- A readout that is perfect on visible resets and at the majority prior on occluded ones scores
  **0.768** on all 190 (`ceiling_if_visible_perfect_and_occluded_at_majority`).
- That is below the 0.85 dx-sign threshold of §9.
- So an overall pass for a source **requires** information on occluded resets. §9 and §13 are
  written with that in view.

### 2.4 The native-resolution render does NOT reproduce the stored frame (source (v) withdrawn from decisions)

"Native" here means **448 px**, the largest integer multiple of 112 that the scene's offscreen
framebuffer allows (640 × 480). We box-downsampled the 448 px render by 4 × 4 and compared it with
the stored 112 px frame:

| check | result |
|---|---|
| byte-for-byte | **fails** |
| per-frame mean absolute difference | 2.54–2.62 levels; maximum pixel 136.2; 11.3 % of pixels differ by > 8 levels; 98.7 % of those lie at edges (stored-frame level differs by > 8 from a 4-neighbour) |
| does the downsample pick out its own reset among the 190? | **162 / 190** |
| same, after subtracting a fixed bias field estimated on train roots | residual ≤ 0.21 levels per frame, but only **169 / 190** |
| emulating the 4× MSAA sample pattern from a non-MSAA 448 px render | mean ≥ 1.75 levels on 6 train roots — **unverified**: an uncommitted scratch probe, not reproducible from committed code and used for no decision |

- **Why it fails.** The 112 px frame is rendered with MuJoCo's 4× MSAA, and a box filter does not
  reproduce that sampling.
- **Why no tolerance rescues it.** The mismatch (0.130–0.212 levels per frame even after bias correction) exceeds
  the smallest gap between two different resets' frames (0.038 levels). **So no tolerance can be
  stated under which the downsampled native render reproduces its own stored frame rather than
  another reset's.**
- **Consequence.** The task owner ruled on this on 2026-09-25, before this document froze:
  - source (v) is **run and reported, but gates nothing**;
  - its "the camera resolution is the problem" reading is **withdrawn**. A difference between
    (iv) and (v) confounds resolution with the rendering path.
  - Source (v-b) (§5) is reported to show how large that confound is.

### 2.5 The targets

We recomputed the expert's step-0 command at each reset (`scripted.apple_collector_policy`, the
shadow expert that TASK-057 Amendment 1 verified):

- it equals the recorded `collector__base_action[0]` **exactly (max |Δ| = 0.0) on all 152 non-aim
  roots**;
- on the 38 aim-offset roots it differs by up to 0.8, by design.

At reset, only `right_dx` and `right_dy` vary:

| component | at reset |
|---|---|
| `right_dz` | always +0.4 |
| `right_droll` | always −0.5 |
| `right_dpitch` | always −0.5 |
| `right_dyaw` | always +0.5 |
| `right_grasp` | always −1 |
| `right_dy` | within [−0.400, −0.125], mean −0.393 |
| `right_dx` | **positive on 117/190 (61.6 %)**, never 0; 77.4 % at the ±0.4 clip; changes sign at apple x ≈ 0.3330 m |

### 2.6 Prior-only baselines (10-fold CV of §3, and train → val)

None of these reads an image or a feature. Each one is fit on the training folds only.

| baseline | all 190 | occluded (111) | visible (79) | val (20), fit on train |
|---|---|---|---|---|
| predict the mean: median apple-xy error | **2.369 cm** | 2.233 cm | 2.559 cm | 2.384 cm |
| occlusion-conditional mean* (median) | **1.939 cm** | 2.006 cm | 1.720 cm | not used |
| majority dx sign: accuracy | **0.616** | 0.604 | 0.633 | 0.600 |
| y-conditional dx-sign prior*: accuracy | 0.616 | **0.604** | 0.633 | not used |
| constant (median) dx: MAE | **0.314** | 0.316 | 0.310 | 0.343 |

**Stratum rule.** Every stratum value is the all-roots out-of-fold prediction restricted to the
stratum, except the occlusion-conditional mean, which is fitted within the stratum by definition.
The run recomputes every value in this table by the same rule, and G-prior compares them all.

A constant +0.4 dx (the clip) scores MAE 0.306, marginally below the median constant; both are far
above the T3 bar (§9), so the choice of constant does not affect any threshold.

\* Privileged baselines. They are fed the true occlusion flag, or the true apple y in 5
quantile bins, and are used **only** to set a harder bar. The occlusion-conditional mean is what a
readout could reach if it only learned the one thing the image obviously shows: whether the apple
is visible. The y-conditional prior is what it could reach on dx sign if it learned "how hidden"
and nothing about x.

The clock-only and proprioception-only baselines **coincide with the constant baseline** at step
0. The clock reads 0 and the state is identical on every reset (§2.1).

## 3. Cohort and split

- **Roots.** All 190 train + val roots of the frozen collection plan
  (`collect_apple_wide.make_plan`), one reset frame each: 170 train and 20 val. The one episode
  per reset is the grouping unit, so no reset can appear on both sides of a split.
  - **Aim-offset roots are included.** Their offset changes only the script's aim, not the reset
    frame or the apple.
  - For the same reason their dx target is the **recomputed** expert (§2.5), not the recorded
    label.
  - **Branch episodes are excluded.** They start at the pre-grasp state, so they have no reset
    frame of their own.
- **Primary estimate: grouped 10-fold cross-validation over the 190 roots.**
  - Fold rule: sort the roots by seed, draw `np.random.default_rng(59).permutation(190)`, and assign
    fold = position mod 10. That gives 19 roots per fold.
  - The fold assignment's sha256 is `1dcc9086…c9b172a` (manifest `split.fold_assignment_sha256`).
  - Every choice inside a fold (normalization, bandwidth, λ, family) is made on that fold's
    training part only.
  - **Why CV is primary.** The readout families and their capacity are fixed a priori, and tuning is
    nested inside the folds, so no fold's held-out roots influence its own model. The frozen val
    split has only 20 roots: a Wilson interval on 19/20 still reaches down to 0.764. That cannot
    carry a decision.
- **Secondary confirmation: the frozen split, train (170) → val (20).** Same pipeline, fit on
  train with 5-fold inner CV (inner seed 5910), scored once on val against baselines fitted on
  train. It is reported with CIs and **decides nothing**.
  A disagreement with the primary estimate is reported as a caveat.
- **Never touched:**
  - the test split, which is never decoded: no test frame, action or label is read. (Loading the
    encoders through `build_policy` constructs `DatasetStore`, whose `verify()` reads every
    episode file's bytes only to check its sha256; that is the only contact.)
  - cohort D and cohort C, and every seed outside 48000–48199.
- **Disclosure.** TASK-057's D1-pipeline re-rendered 15 of the 20 val roots. They are members of
  the val split and were never a gating cohort; this task uses them as val roots like any other.

## 4. Targets

| ID | target | source of truth |
|---|---|---|
| **T1** | apple xy at reset (world frame, m) | `privileged__apple_position_world[0][:2]` (equals reset truth, §2.1) |
| **T2** | sign of the expert's step-0 `right_dx` | recomputed expert (§2.5) |
| **T3** | the expert's step-0 `right_dx` value, in [−0.4, 0.4] | recomputed expert |
| T4 (reported) | step-0 `right_dy`; apple x alone | same |

## 5. Feature sources

All sources read the same reset frame of each root. Sources (i)–(iv) read the **stored** 112 px
training frame, which §2.1 shows is identical to a fresh render.

| ID | source | role |
|---|---|---|
| **(i) E0** | `image_features` of the frozen E0 encoder, exactly as arm A2 uses it (`checkpoints/task056-policy-v1/a2.pt`, E0 = `checkpoints/task054-wm-v4/leworldmodel_baseline.pt`; encoder weights sha256 `6590d51c…52de95`), 128-d | **decisional** |
| **(ii) A3** | `image_features` of arm A3's fine-tuned encoder (`a3.pt`, weights `f6316863…74f609`), 128-d | **decisional** |
| **(iii) random** | `image_features` of arm A1's frozen random-init encoder: the same architecture and seed (`a1.pt`, weights `cae0ad88…59437b`), 128-d | **decisional; the floor** |
| **(iv) raw-112** | the stored 112 × 112 × 3 frame, flattened, /255 | **decisional** |
| (v) raw-448 | a 448 px native render of the recorded reset, flattened, /255 | **reported only** (§2.4) |
| (v-b) raw-448↓ | (v) box-downsampled to 112 px, flattened, /255 | reported only: the size of the rendering-path confound |

The encoders are rebuilt with `measure_policy_offline_conditionals.build_policy`, the same loader
that TASK-057's runner used, and run on CPU in `inference=True` mode. `policy.py` is loaded
unchanged and is never edited.

**Not a source, by design:** proprioception. It is identical on every reset (§2.1), so it equals
the constant baseline.

## 6. Readout families and capacity: identical for every source

Every readout is fitted in the **dual (kernel) form** on n ≤ 171 training rows. The procedure and
the λ grid are identical for every source, and capacity is bounded by n and λ for all of them. It
is not literally equal: a linear kernel on a 128-d feature has rank ≤ 128, while the pixel kernel
can reach rank ≈ n.

- **Preprocessing (same for every source).**
  - Cast to float64.
  - Centre by the training-part mean.
  - Do not scale per dimension. Per-pixel standardization would amplify near-constant pixels, and
    applying one rule to every source keeps the comparison fair.
- **F1 — linear ridge.** Kernel K = X Xᵀ on the centred features.
- **F2 — RBF kernel ridge.** K = exp(−‖x − x′‖² / (2σ²)), where σ² is the median squared pairwise
  distance among the training-part rows.
- **λ grid.** λ ∈ {10⁻⁴, 10⁻³, …, 10⁴} × mean(diag K_train), 9 values. Multiplying by the diagonal
  makes the grid scale-free.
- **Targets.** Each target is centred by its training-part mean, which acts as the intercept.
  - T1 is fitted as one 2-output problem with a shared λ.
  - T2 and T3 share one scalar regression on the dx value.
- **Nested selection.**
  - For every (source, target group, outer fold), choose the family and λ (18 configurations) by
    5-fold inner CV on the outer training part, minimizing mean squared error: xy for T1, dx for
    T2/T3.
  - Inner folds: `default_rng(5900 + outer_fold).permutation(m)`, where m is the size of the outer
    training part and its rows keep their seed order; inner fold = position mod 5.
  - Centring, σ² and the λ scale are recomputed on each inner fit part, never on the rows it
    predicts.
  - Refit on the whole outer training part, then predict the held-out fold once.
- **Predictions.**
  - T2 is the sign of the unclipped dx prediction; a prediction of exactly 0 counts as wrong.
  - T3 is the prediction clipped to [−0.4, 0.4].
  - A derived dx sign, computed from the T1 prediction through the expert's formula, is reported
    only.
- **Declared not done.** A small CNN trained from scratch. With 190 frames it would add a capacity
  axis that cannot be matched across sources. The brief allowed a linear/ridge readout on pixels
  instead.

## 7. Metrics and uncertainty

All metrics use the 190 out-of-fold predictions, one per root.

- **T1.** The median Euclidean apple-xy error in cm is primary; the mean is reported.
  - 95 % percentile bootstrap CI over roots: 10 000 resamples. For a set of n roots the index
    matrix is `default_rng(5901).integers(0, n, (10000, n))`, drawn over that set only (a stratum
    is resampled within itself). The same matrix serves every source and every baseline on that
    set, so all ratios and differences are paired.
  - Ratio to a baseline: the **paired** bootstrap of median(source) / median(baseline), with the
    same resample indices for both.
- **T2.** Accuracy with a Wilson 95 % interval. Declared caveat: out-of-fold predictions come
  from ten different models, so the Wilson interval's independence assumption holds only
  approximately.
- **T3.** MAE with a bootstrap CI, and a paired-bootstrap ratio to the constant-median-dx MAE.
- **Comparing two sources.** A paired bootstrap of the difference in median T1 error. A McNemar
  exact test on the discordant T2 pairs is reported.
- **Strata.** Everything is reported on all roots, on occluded roots (0 apple pixels at 112 px) and
  on visible roots (≥ 1 pixel). The stratum baselines follow §2.6's stratum rule.
- **No multiplicity correction.** Four decisional sources times two strata are read. The thresholds
  below are absolute and conservative, and this is declared rather than corrected.

## 8. Baselines

Every threshold is set against these baselines:

- **B-mean**, predict the mean;
- **B-occ**, the occlusion-conditional mean (privileged; the harder bar for T1);
- **B-maj**, the majority dx sign (equal to clock-only and proprio-only at step 0);
- **B-y**, the y-conditional dx-sign prior (privileged; on occluded resets);
- **B-const**, the constant median dx;
- **B-rand**, source (iii), the random-encoder floor, measured in the run.

The values that B-mean, B-occ, B-maj, B-y and B-const take on this cohort are fixed above (§2.6)
and recomputed identically in the run. The run refuses to continue if its recomputed prior
baselines differ from the manifest by more than 1e-9.

## 9. Thresholds, each justified against a blind or prior-only baseline

A source **passes a target** on a set of roots (all, or a stratum) when every condition in its row
holds.

| target | pass requires | justification |
|---|---|---|
| **T1** | median error **≤ 1.5 cm**, **and** the upper bound of the paired-bootstrap 95 % CI of median(source) / median(B-occ) **≤ 0.6** | 1.5 cm is the grasp tolerance that the task turns on (`apple_policy_v1.md` §2.2). B-mean scores 2.369 cm, and B-occ, which already knows visibility, scores 1.939 cm overall (1.720 visible, 2.006 occluded). **Neither prior passes:** both exceed 1.5 cm and have a ratio of 1.0 or more. The ratio condition turns 1.5 cm into an effective ≈ 1.16 cm overall, and ≈ 1.03 cm on visible roots. |
| **T2** | accuracy **≥ 0.85**, **and** Wilson 95 % lower bound **> B-maj** on the same roots (0.616 overall, 0.633 visible, 0.604 occluded) | The majority prior (the same as clock-only and proprio-only) scores 0.616. A score of 0.85 removes ≥ 60 % of the prior's errors. At n = 190, 162/190 (the first count ≥ 0.85) has a Wilson lower bound of 0.795. At n = 79 (visible), 68/79 has 0.768. Both are clear of the prior. |
| **T3** | the upper bound of the paired-bootstrap 95 % CI of MAE(source) / MAE(B-const) **≤ 0.6** | B-const scores 0.314 (ratio 1.0 by construction). A sign-perfect readout that outputs ±0.4 scores MAE 0.040 (0.13×) on these targets, since 77 % sit at the clip. |

**A source succeeds on a set of roots when it passes T1, T2 and T3 there.** On any stratum,
**it beats the prior on that stratum** when either of the following holds:

- the Wilson lower bound of its T2 accuracy exceeds max(B-maj, B-y) on the stratum; or
- the upper bound of its T1 ratio to B-occ is below 1.0.

**Random-encoder qualifier (B-rand).** Suppose E0 or A3 succeeds on a set of roots. It is also
said to **beat the random floor** there only if the paired-bootstrap 95 % CI of median T1 error
(source − random) lies entirely below 0 **and** the source's T2 accuracy exceeds random's. If not,
the reading adds "not attributable to pretraining". The floor is a measured qualifier, not a pass
condition, because a random convolutional/ViT feature can localize a salient object. That is
exactly what (iii) exists to reveal.

## 10. Occluded resets: a readout that beats the prior needs an explanation before it is believed

§2.3 predicts that no readout beats the prior on occluded resets. Suppose some source does
(§9, "beats the prior") on the occluded stratum. Then the run applies **the same out-of-fold fitted
readouts, unchanged**, to two re-renders of every occluded reset:

1. **apple-hidden:** the apple geom is moved to a hidden geom group. It is not drawn and casts no
   shadow; everything else is unchanged.
2. **shadow-off:** shadows are disabled for the whole scene, and the apple is present.

These re-renders are 112 px, from the same recorded reset. The encoders receive them through the
same `image_features` path. Only decisional sources are re-rendered.

The rule:

- **Still above the prior on apple-hidden frames** (by the same criterion). The cue is **not the
  apple**; it is a spurious regularity such as the plate or rendering noise. The source's
  occluded-stratum result is recorded as **spurious**, and the source is treated as **not
  succeeding overall**, whatever its numbers.
- **Drops to the prior on apple-hidden frames.** The cue is apple-caused. The shadow-off result
  says whether it is the shadow (it no longer beats the prior, by the same criterion) or residual
   apple pixels below segmentation
  resolution (it does not). The result is **believed**, with that explanation attached.

If no source beats the prior on occluded resets, the check is not run and that is recorded.

## 11. Descriptive measurements (non-decisional)

1. **Visibility over time, train roots only (170).**
   - From each train reset, the run executes the **nominal** expert (`apple_collector_policy`, no
     perturbation). It follows the collector's own step: clip to the collection bounds, project,
     execute. It runs orient + descend, 210 commands, or until a guard stop, which is recorded.
   - After every command it counts apple pixels at 112 px (onboard, segmentation).
   - It reports the first step t₁ with ≥ 1 pixel and the first step t₁₀ with ≥ 10 pixels, plus the
     share of resets never visible within 210 steps.
   - **Is the reset-dependent command still needed after t₁?** At t₁ it reports:
     - the remaining x offset to the orient target, |target_x − palm_x| (base frame);
     - whether that offset still exceeds the 6 mm band in which dx leaves the ±0.4 clip. Above
       6 mm, the expert is still issuing a saturated, reset-dependent command;
     - whether sign(dx(t₁)) = sign(dx(0)).
   - It also reports the step at which the expert's x offset first falls inside the 6 mm band. If
     t₁ usually comes before that step, a look-first reset has time to act. If not, the command
     has already been decided before the apple can be seen.
2. **Other cameras at reset (190 roots).** The scene defines `onboard_rgb` (the model input) and
   `overview`, a fixed world camera. The corpus also records `hand_crop_rgb`, a 112 px window of a
   320 px onboard render placed by kinematics. The run reports apple pixels at reset for:
   - `overview`, at 112 and 448 px;
   - the `hand_crop_rgb` window.
3. **Frame near-identity.** It reports the mean absolute difference between stored frames for
   pairs of **occluded** resets (min, median), and the number of pixels differing by more than 8
   levels.
4. **Native equivalence (§2.4)** is re-measured in the run by the committed checker and reported,
   whatever it says.

## 12. Guards: any failure voids the run (no outcome is read)

| guard | condition |
|---|---|
| **G-hash** | Every hashed input equals the manifest's `hashes`: dataset manifest, the three policy checkpoints, the E0 checkpoint, `configs/apple_policy_v1.yaml` and its parents `configs/apple_wm_v4_lewm.yaml` and `configs/apple_wm_v4.yaml`, `collect_apple_wide.py`, `calibrate_info_ceiling.py`, `measure_policy_offline_conditionals.py`, `scripted.py`, `simulation.py`, `embodiment.py`, `policy.py`, `readout_labels.py`, `assets/manifest.json`, `configs/g1_sim_action.json`. Every episode file read is checked against the dataset manifest's sha256 before it is decoded. Each encoder's weights sha256 equals the manifest's. |
| **G-split** | The roots are exactly the plan's 190 train + val seeds, and each root's split agrees with the dataset manifest's frozen splits. The runner refuses any test-split episode id, any seed in cohort C (45300–45339) or cohort D (45000–45007, 45100–45107), and any seed outside 48000–48199. The fold assignment hash equals the manifest's. |
| **G-render** | The 112 px re-render is byte-identical to the stored frame on **190/190**, and the maximum std of reset proprioception is ≤ 1e-6. |
| **G-expert** | The recomputed expert equals the recorded step-0 label to within 1e-6 on all 152 non-aim roots, and the apple label equals the reset truth to within 1e-6 m. |
| **G-prior** | The recomputed prior baselines equal the manifest's to within 1e-9 (§8). |

The native-equivalence check (§2.4) is **not** a voiding guard. Its failure is already known, and
it is why source (v) gates nothing. PR 2's tests must exercise every guard, and the equivalence
checker, in both directions.

## 13. Pre-declared outcomes

The outcomes are read on the **primary (CV)** estimate, over the decisional sources (i)–(iv), after
§10. They are evaluated in order, and the first row that matches is the outcome.

| row | condition | reading | recommended next direction (the owner chooses) |
|---|---|---|---|
| **VOID** | any §12 guard fails | nothing is read | fix, and rerun as a disclosed new version |
| **O-BC** | **E0 succeeds on all roots** | The reset frame carries the apple and the step-0 command, and E0 exposes them to a readout. **The BC objective or data is the problem**, not perception. | corpus design (e.g. demonstrations that start with motion, on-policy relabelling) |
| **O-ENC** | E0 does not succeed on all roots, but A3, raw-112 or random does | The information is in the frame, and **E0's frozen encoder discards it**. A3 succeeding means fine-tuning recovers it. | encoder / representation work |
| **O-OCC-BC** | no decisional source succeeds on all roots, **and E0 succeeds on the visible stratum** | The frame carries the apple only when it is visible. **Occlusion at reset is binding on the occluded resets (camera placement or reset pose); on visible resets, BC is the problem.** | camera placement or reset pose (a look-first reset), informed by §11.1–11.2 |
| **O-OCC-ENC** | no decisional source succeeds on all roots, E0 does **not** succeed on the visible stratum, but raw-112, A3 or random does | Occlusion is binding, **and** E0 discards the visible apple as well | camera or reset pose, plus encoder |
| **O-OCC-NONE** | no decisional source succeeds on all roots, and no decisional source succeeds on the visible stratum | Even a visible apple at ≤ 21 px is not read out at 112 px. **The camera is the problem**, and this protocol cannot separate resolution from placement. (v)'s numbers are reported beside it as suggestive only (§2.4). | camera change (placement and/or resolution), with an equivalence-validated render path |

**Why no "native only" row.** The brief asked for a row reading "it succeeds only at native
resolution: the camera is the problem". §2.4 withdraws it: without a validated render path, a
difference between (iv) and (v) cannot be attributed to resolution. That reading survives only as
a **qualifier on O-OCC-NONE**, labelled unvalidated.

**Stated in advance (§2.3).** An overall success needs information on occluded resets, which the
frame should not carry. **The expected outcome is therefore one of the O-OCC rows.** An O-BC or
O-ENC outcome is believed only after §10.

**In every outcome:**

- every source's numbers, CIs, baselines and strata are published, whatever they say;
- the secondary train → val estimate is published beside the primary one;
- nothing is refitted or re-thresholded after the numbers are seen;
- the executing agent **recommends** a next task and does not choose one.

## 14. Budget and recording

- **Device:** CPU. **Caps:** 2 h wall-clock for the whole run; about 15 min is expected.
- **Seeds:** folds 59; inner folds 5900 + k; bootstrap 5901.
- **Recorded:**
  - the code revision, with a clean tree required;
  - every hash in §12;
  - the encoder weight digests;
  - the fold hash;
  - per-root out-of-fold predictions;
  - selected family and λ per fold;
  - elapsed time and peak RSS.
- **Output:** `outputs/task059-info-ceiling/run-1/`. The runner refuses to overwrite any existing
  path. Nothing prior under `data/`, `checkpoints/` or `outputs/` is modified.

## 15. Not done in this task (declared)

- No controller is trained or evaluated, and no closed-loop attempt is run.
- No cohort is opened, and the test split is not decoded.
- No new camera, resolution or reset pose is collected. Those are candidates for the task this one
  recommends.
- No change is made to `policy.py`, `models/`, or any TASK-056 checkpoint.
- No small CNN is trained (§6).

## 16. Process

1. **PR 1** contains this document, the manifest and the calibration script. It merges on an
   independent reviewer's **reported** APPROVE.
2. **PR 2** contains the probe code and tests that exercise every guard in both directions. It
   merges on a reported APPROVE.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3** contains the results, including every negative result.
5. **A protocol defect found after this freezes** is escalated to the task owner and fixed only
   through a disclosed amendment, never silently.
