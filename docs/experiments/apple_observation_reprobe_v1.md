# Apple→Plate observation re-probe v1: which decision-time observation shows the apple? (TASK-061)

**Status: preregistration. No readout has been fitted on any frame or encoder feature of any arm
below.** The only numbers here come from a pre-freeze calibration that renders frames, counts
apple pixels, validates render paths and evaluates prior-only predictors that read no
observation (`scripts/calibrate_observation_reprobe.py`, committed with this document). **The test
split is never decoded. Cohort D (45000–45007, 45100–45107) and cohort C (45300–45339) are never
simulated. `exemption_spent` (`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays
`false`.**

This is a **perception/observation task**. It trains no controller, collects no corpus and
preregisters no control formulation: the TASK-054 and TASK-057 abandonment clauses
(`apple_policy_v1.md` §7, `docs/DECISIONS.md`) still hold on `apple-wide-v1` and the 112 px
onboard camera. Nothing here presents behaviour cloning or sampling-based planning as the
project's control approach. **Learned Apple→Plate is still 0 successes.**

Predecessor: [`apple_info_ceiling_v1.md`](apple_info_ceiling_v1.md) /
[`_results.md`](apple_info_ceiling_v1_results.md) (TASK-059, outcome O-OCC-NONE). Manifest:
`benchmarks/manifests/apple-observation-reprobe-v1.json`.

---

## 1. The question

TASK-059 found that the 112 px onboard reset frame does not give the apple position or the
expert's step-0 `right_dx` beyond a prior. Two things fail at once:

- the right wrist (`right_wrist_yaw_link`) hides the apple on 111 of the 190 train + val resets;
- on the 79 visible resets no 112 px source beats the visibility-aware prior either. A visible
  apple covers 1–21 px (median 5).

Non-decisional hints from that run: the `overview` camera sees the apple on 190/190 resets
(≈ 12 px at 112 px), and along the expert's own path the apple becomes visible within 9 steps on
every train root.

**Question:** which change to the robot's observation at decision time makes the apple position
and the expert's first reset-dependent command readable beyond the prior, with the TASK-059 probe
**unchanged**: the same readouts, cross-validation scheme, baselines, bars, bootstrap and
stratification?

The arms vary the observation along two axes: **what is visible** (a reset-independent look
motion that clears the wrist) and **how many apple pixels there are** (resolution and camera).

## 2. What the probe reuses unchanged

Everything below is TASK-059's protocol (`apple_info_ceiling_v1.md` §3, §6–§9), executed by the
same code: `src/embodied_jepa/info_ceiling.py` and the frame, render and encoder helpers of
`scripts/probe_info_ceiling.py`. Both files are hashed in the manifest (G-hash), so the run cannot
use an edited copy.

- **Roots.** The 190 train + val roots of the frozen collection plan (170 train, 20 val), one
  decision frame each; aim-offset roots included; branch episodes excluded.
- **Folds.** Grouped 10-fold CV, `default_rng(59).permutation(190)`, fold = position mod 10. The
  fold hash is TASK-059's `1dcc9086…c9b172a` (recomputed in the calibration).
- **Readouts.** Linear and RBF kernel ridge in dual form, λ ∈ {10⁻⁴ … 10⁴} × mean diag K, family
  and λ by nested 5-fold inner CV (inner seed 5900 + k), centring by the training-part mean,
  no per-dimension scaling. Raw pixels are flattened and divided by 255.
- **Targets, metrics, bootstrap.** T1 apple xy (median cm), T2 sign of the expert's `right_dx`,
  T3 its value clipped to [−0.4, 0.4]; 10 000 paired bootstrap resamples, `default_rng(5901)`;
  Wilson intervals for T2.
- **Baselines.** B-mean, B-occ, B-maj, B-y, B-const, as in TASK-059 §8. **B-occ keeps TASK-059's
  flag**: the apple's visibility in the 112 px onboard frame **at reset** (111 occluded). §6
  explains why this is the harder bar for every arm.
- **Bars (§9 there), per source, on a set of roots.**
  - T1: median ≤ 1.5 cm **and** upper 95 % bound of the paired ratio to B-occ ≤ 0.6;
  - T2: accuracy ≥ 0.85 **and** Wilson 95 % lower bound > B-maj;
  - T3: upper 95 % bound of the paired MAE ratio to B-const ≤ 0.6.
  A source *succeeds* when it passes all three. "Beats the prior" keeps TASK-059's definition.
- **Secondary estimate.** Train (170) → val (20), reported, decides nothing.

**What changes:** only the decision frame (§3), the dx target where the decision state changes
(§4), and a multiplicity rule across arms (§7). The bars themselves are not changed.

## 3. Arms

### 3.1 The look motion (chosen by visibility only, before the freeze)

**Definition.** Eight identical commands, fixed in the manifest and executed from the reset before
the decision frame is taken:

| field | value |
|---|---|
| left arm (0–5) | 0 |
| `right_dx` | **0** |
| `right_dy` | −0.4 |
| `right_dz` | +0.4 |
| `right_droll`, `right_dpitch`, `right_dyaw` | −0.5, −0.5, +0.5 |
| left grasp, right grasp | −1, −1 (open) |

This is the expert's own step-0 command with its reset-dependent parts fixed: `right_dx` set to 0,
and `right_dy` set to its mode (−0.4; it lies in [−0.400, −0.125] at reset, mean −0.393). Every
other component already equals the expert's constant at reset (TASK-059 §2.5). The sequence
sha256 over its float32 bytes is `17bfb070…15be587c`.

Each command goes through the collector's own step: clip to the collection bounds, project,
execute. **The motion reads nothing about the reset.** The sequence is a constant, and the code
that executes it receives only the robot (PR 2 tests this, §9).

**How it was chosen: from rendering facts only.** The selection used apple visibility and pixel
counts from the 112 px onboard segmentation render. It never used a readout, a command target,
the expert's command or any label. The calibration ran five constant right-arm candidates for
1–10 steps on all 190 roots. For each, it counted the roots with ≥ 1 apple pixel in the 112 px
onboard frame:

| candidate (dx, dy, dz, droll, dpitch, dyaw) | visible roots after k = 0 / 2 / 4 / 6 / 8 / 10 steps |
|---|---|
| **expert without dx** (0, −0.4, 0.4, −0.5, −0.5, 0.5) | 79 / 99 / 132 / 167 / **190** / 190 |
| raise (0, 0, 0.4, 0, 0, 0) | 79 / 77 / 92 / 98 / 103 / 123 |
| raise and out (0, −0.4, 0.4, 0, 0, 0) | 79 / 85 / 122 / 148 / 174 / 189 |
| raise and back (−0.4, 0, 0.4, 0, 0, 0) | 79 / 80 / 82 / 104 / 117 / 126 |
| out (0, −0.4, 0, 0, 0, 0) | 79 / 89 / 110 / 134 / 155 / 173 |

The rule in `choose_look` takes, per candidate, the smallest k at which every root is visible,
then the candidate with the smallest such k (ties by listed order). It selects **expert without
dx, k = 8**. **Disclosure:** the five candidates were first explored in an uncommitted scratch
script that computed the same visibility table; the rule was written after seeing it. The
selection uses visibility only. It reads no target and fits nothing, and the committed script
reproduces the table.

**Facts about the chosen look (calibration, 190 roots):**

- The applied (post-projection) commands are **identical on every root** and equal the requested
  ones (max |Δ| = 0.0). No projection was infeasible and no command was rejected.
- **The robot state after the look is identical on every root** (maximum absolute spread 0.0).
  So proprioception still carries no reset information, and a clock reads 8 on every root.
- **The look does not move the apple or the plate.** Apple xy moves by at most 1.5e-15 m. The
  plate does not move. There is no hand–apple contact after the look. (The calibration checks
  contact only at the end; the run checks it after every look command, §9.) The apple settles 3.37 mm in z under gravity on
  every root. A hold control (the same 8 steps with a zero arm command) moves the apple
  identically (look − hold = 0.0 m on every root), so the settling is not the look's doing. T1
  therefore stays the reset apple xy.
- **The apple is visible on 190/190 roots after the look** at 112 px: min / p25 / median / p75 /
  max = 1 / 9 / 12 / 19 / 45 px. It is visible on all 111 roots it was hidden on at reset.

### 3.2 Arms and their decisional sources

| arm | decision frame | decisional source(s) | reported-only sources |
|---|---|---|---|
| **L** look-first | 112 px `onboard_rgb` after the look | **raw pixels**; **E0** | A3, random (the B-rand floor for E0) |
| **LH** look-first, higher resolution | 224 px `onboard_rgb` after the look | **raw pixels** | — |
| **O** overview | 112 px `overview` at reset | **raw pixels** | E0 |
| H higher resolution at reset | 224 px `onboard_rgb` at reset | — (§3.3) | raw pixels |
| R rendering path | 448 px `onboard_rgb` at reset, 4×4 box-downsampled | — | unrounded float (TASK-059's 448↓), rounded to uint8 |
| anchor | the stored 112 px reset frame | — | raw pixels (TASK-059's raw-112, re-run) |

That is **four decisional hypotheses**: L-raw, L-E0, LH-raw and O-raw.

**Encoders.** E0 is `checkpoints/task054-wm-v4/leworldmodel_baseline.pt` as arm A2 uses it (weights
`6590d51c…52de95`), loaded exactly as in TASK-059. Its input contract is a 112 × 112 RGB frame
from the `onboard_rgb` camera.
- E0 is decisional only in **L**, the one arm whose frame meets that contract.
- On the `overview` frame E0 is shape-compatible, but the camera is one it was never trained on,
  so that result is reported only.
- At 224 px it is not applicable.
- **No encoder is fine-tuned or trained.** At n = 190 a new encoder would add a capacity axis that
  cannot be matched across arms (TASK-059 §6, "declared not done").

### 3.3 Why H and R are reported only

- **H (224 px at reset).** The wrist still hides the apple on **109/190** roots at 224 px (106 at
  448 px). TASK-059 §2.3 applies: with the occluded roots at the majority prior, T2 cannot reach
  0.85 overall (the ceiling is about 0.77, TASK-059's 0.768 recomputed for 109 occluded roots). Spending a Holm step on an arm that cannot pass would only
  cost the others power. Its visible-stratum numbers are reported as the resolution-only
  comparison with TASK-059.
- **R (448↓).** Neither variant is a render path that a robot observation would use.
  - R-u8 rounds R-float to uint8, as a stored frame would be. The pair tests TASK-059 §6.3's
    quantisation confound.
  - Declared reading: if the paired bootstrap CI of the median T1 difference R-u8 − R-float on
    the reset-visible stratum (`compare_sources`) contains 0, quantisation does not explain the
    448↓ result. **It gates nothing.**
- **Anchor and R-float repeat TASK-059 sources.** The anchor repeats raw-112 and R-float repeats
  (v-b), on the same roots, targets, folds and code. They are expected to reproduce TASK-059's
  results-manifest numbers exactly on the same machine.
  - Both are reported only.
  - A mismatch changes no row. It is reported as a caveat and escalated to the task owner before
    the results document is written, because it would mean the reuse is not what §2 claims.

### 3.4 Not tested, declared

- **A narrower or closer onboard camera.** It needs a scene change in `simulation.py`, a hashed
  file. The pixel axis is covered by LH (median 44 apple px) and O.
- **Look-first × overview.** The fixed `overview` camera sees the apple on 190/190 resets
  already.
- **448 px as a decisional arm.** At reset the wrist still hides the apple on 106/190 roots at
  448 px, and TASK-059's native raw-448 read worse than 448↓ on visible roots (1.482 cm against
  0.946 cm). LH already tests the look with 4× the pixel area of L.

## 4. Targets

| ID | target | L, LH (post-look) | O, H, R, anchor (at reset) |
|---|---|---|---|
| **T1** | apple xy (world, m) | the reset apple xy (the look does not move it, §3.1) | the reset apple xy |
| **T2** | sign of the expert's `right_dx` | **at the post-look state** | at reset (TASK-059's target) |
| **T3** | the expert's `right_dx` in [−0.4, 0.4] | at the post-look state | at reset |
| T4 (reported) | `right_dy`; apple x alone; dx sign derived from T1 | same state as T2 | same |

**"The expert's command at the post-look state"** is `apple_collector_policy(reset truth)`
constructed fresh, in the orient phase at step 0, and evaluated at the post-look robot pose. This
is the same function TASK-059 used, and G-expert checks it against the recorded labels at reset.
Its `right_dx` = clip((apple_x − 0.015 − palm_x) / 0.015, ±0.4). The post-look palm x (0.3176 m)
is the same on every root.

Post-look target facts (calibration):
- `right_dx` > 0 on **118/190** (never 0), 77.9 % at the clip;
- its sign equals the reset sign on 189/190;
- the sign changes at apple x ≈ 0.332–0.333 m;
- `right_dz`, `right_droll`, `right_dpitch` and `right_dyaw` are constant (+0.4, −0.5, −0.5, −0.5);
- `right_dy` now varies across [−0.4, 0.4] (T4, reported).

The 224 px instance computes the identical post-look command on 190/190 roots.

**T4's derived dx sign** at the post-look state is the expert's command at the post-look pose
(palm x 0.31755 m) for an apple at the *predicted* xy; at reset it is TASK-059's.

## 5. Render-path validation (not a guard: a failed arm becomes non-decisional)

TASK-059 validated its 112 px path by showing that a re-render reproduces the stored training
frame byte for byte. The post-look, 224 px and overview frames have no stored counterpart. So
this protocol validates that **the probe's frame is the frame the robot's own observation path
produces**, and that it is reproducible.

**What the checks can and cannot show.**
- The frames come from the stored-frame pipeline by construction: `G1Embodiment.observe()` →
  `MuJoCoSimulation.render()` on one `mujoco.Renderer`, at another size or camera.
- P5 checks that the instances differ in nothing else.
- P1, P2 and P4 can fail only through nondeterminism or dependence on render history. They are
  determinism checks.

**An unforeseen rendering fact, found while PR 2 was being written and before this document froze.**
A freshly created MuJoCo renderer's **first** render can differ from its later renders of the
same state. In the first scratch finding the difference was 1 pixel, by 1 level. From the second
render on, the output is stable. The task owner ruled on the handling (2026-09-25).

The calibration's first-render check quantifies it. On each of the 190 reset states it creates a
fresh renderer, renders twice, and compares the two renders:

| size / camera | roots where the first render differs from the second |
|---|---|
| 112 px `onboard_rgb` | **0 / 190** |
| 112 px `overview` | **0 / 190** |
| 224 px `onboard_rgb` | **190 / 190** |
| 224 px `overview` | 8 / 190 (not used by any arm) |

- **112 px is unaffected**, so neither TASK-059 nor the stored `apple-wide-v1` corpus is
  affected. The claim rests on two things:
  - this first-render check, which finds 0/190 differences for both 112 px cameras;
  - TASK-059's G-render, which compared fresh re-renders with the stored frames and found them
    byte-identical on 190/190. Its first root's frame was a first render.
- **A replica comparison (P1) cannot see this effect.** Both instances' first renders shift
  alike.
- **Rule, applied uniformly to every arm, including the 112 px and overview arms:** every
  renderer renders once and discards the result before any frame is kept. That covers the
  simulation's own, the segmentation, the native 448 px and the ablation renderers.
- **P4 checks every root.** Re-rendering each arm's decision state must reproduce its frame.
- **Failure consequence.** A P4 or P5 failure demotes the arm (below); it does not void the run.

The checks:

| check | what | calibration |
|---|---|---|
| P0 | 112 px onboard re-render at reset equals the stored frame (TASK-059 G-render) | 190/190 |
| P1 | each arm's frame equals the frame from an independent replica simulation that performed the same reset and look, byte for byte. Onboard frames come from `G1Embodiment.observe()` of a `MuJoCoSimulation` at that size; `overview` comes from `MuJoCoSimulation.render(camera="overview")` | L 190/190, LH 190/190, O 190/190, H 190/190, R-u8 190/190 |
| P2 | the frame survives a PNG encode/decode unchanged. This is the same PIL PNG encoding as `data.py` uses (re-implemented, not called) | 190/190 |
| P3 | the 224 px instance's joint state equals the 112 px instance's exactly, at reset and after the look (resolution does not touch physics) | 190/190 |
| P4 | an immediate re-render of the arm's decision state, through the same path, equals the kept frame (independent of render history) | L 190/190, LH 190/190, O 190/190 |
| P5 | the 112 and 224 px instances are built from the same MJCF, with the same offscreen buffer (640 × 480), anti-aliasing samples (4) and shadow map size; only the render size differs | holds |

**Consequence, fixed now.** An arm is decisional only if every applicable check holds on 190/190
in the run:
- L: P0, P1, P2 and P4;
- LH: P1, P2, P3, P4 and P5;
- O: P1, P2 and P4.

If one fails, that arm's hypotheses are recorded as **non-decisional (render path not
validated)** and cannot pass.
- **P0 is also G-render.** A P0 failure is therefore a G-render failure, and the run is V
  (§9, which comes first).
- The Holm family is **not** shrunk. A failed validation costs power; it never buys it.
- R-u8 is not byte-identical to the stored frame (0/190, as TASK-059 found). That is why R is
  reported only.

## 6. Baselines and strata

**B-occ uses TASK-059's reset visibility flag in every arm.**
- In L, LH and O the apple is visible on 190/190 roots (§3.1, and 0 `overview` occlusions). So an
  arm-own visibility-conditional mean equals B-mean: 2.369 cm, weaker than B-occ's 1.939 cm.
  TASK-059's flag is therefore the harder bar.
- The arm-own conditional mean is computed and reported. For H (109 occluded at 224 px) it is
  1.968 cm.
- A training fold that lacks the held root's flag falls back to the training-fold mean. This
  rule is needed only if an arm has very few occluded roots.

**Prior-only baselines** (calibration; they read no observation):

| baseline | reset targets (O, H, R, anchor) = TASK-059, all 190 | post-look targets (L, LH), all 190 | post-look, reset-occluded 111 / reset-visible 79 |
|---|---|---|---|
| B-mean median xy error | 2.369 cm | 2.369 cm | 2.233 / 2.559 cm |
| **B-occ** median xy error | **1.939 cm** | **1.939 cm** | 2.006 / 1.720 cm |
| **B-maj** dx-sign accuracy | **0.616** | **0.621** (118/190) | 0.613 / 0.633 |
| B-y dx-sign accuracy | 0.616 | 0.621 | 0.613 / 0.633 |
| **B-const** dx MAE | **0.314** | **0.304** | 0.308 / 0.298 |

The reset-target baselines equal TASK-059's manifest exactly (max |Δ| = 0.0). The run recomputes
all of them and voids if any differs by more than 1e-9 (G-prior).

**No prior passes a bar.**
- T1: B-mean and B-occ exceed 1.5 cm, and their ratio to B-occ is ≥ 1.
- T2: 0.621 < 0.85.
- T3: the ratio is 1 by construction.

A sign-perfect ±0.4 readout would score MAE 0.041 on the post-look targets (0.13× B-const).

**Strata, reported for every source:**
- all roots (decisional);
- the TASK-059 strata: reset-occluded (111) and reset-visible (79);
- the arm's own occluded and visible strata, where non-empty.

The reset-occluded stratum is where the look motion is supposed to add information.

## 7. Multiplicity: Holm across the four decisional hypotheses

The four hypotheses are L-raw, L-E0, LH-raw and O-raw. A hypothesis **passes** only if all four of
the following hold:

1. **Render path validated** in the run (§5).
2. **TASK-059 success on all 190 roots** with the bars unchanged (`info_ceiling.evaluate`, `succeeds`).
3. **Holm rejection** at a family-wise one-sided α = 0.025. That level is the one-sided size of
   each 95 % bar.
   - Each hypothesis gets a p-value p_h = max(p_T1, p_T2, p_T3), the intersection-union test of its
     three bars:
     - p_T1 = the share of the 10 000 paired resamples whose median ratio to B-occ exceeds 0.6;
     - p_T3 = the share whose MAE ratio to B-const exceeds 0.6;
     - p_T2 = 1 − Φ(z), with z = (acc − p₀) / √(p₀(1 − p₀)/n) and p₀ = B-maj on the same roots.
       This is the score test that the Wilson interval inverts. If p₀ is 0 or 1, p_T2 = 1 (never
       passes).
   - p_T1 and p_T3 use **the same draws as the percentile bars**: `ic.bootstrap_indices(190)`
     (seed 5901) and the same median/mean statistics. A resample whose ratio is non-finite (a
     zero baseline) counts as exceeding 0.6, matching `paired_ratio`'s largest-float rule.
   - If the point conditions fail (median > 1.5 cm or accuracy < 0.85), p_h = 1.
   - Sort p_h ascending; ties go to the listed order L-raw, L-E0, LH-raw, O-raw. The j-th
     hypothesis is rejected if every p up to it is ≤ 0.025 / (4 − j + 1). The thresholds are
     0.00625, 0.00833, 0.0125 and 0.025.
   - Condition 2 is kept as well, so that no hypothesis passes on a bootstrap share that the
     unadjusted percentile bar would reject.
4. **Not spurious.** Apply the same out-of-fold readouts, unchanged, to an **apple-hidden**
   re-render of every root's decision frame. The apple geom is moved to a hidden group, so it is
   neither drawn nor casts a shadow. Each root is predicted by the readout whose fold held it out.
   - If the hidden frames still **beat the prior on all roots** (TASK-059's definition), the cue
     is not the apple and the hypothesis does **not** pass.
   - The ablation renderer must reproduce the arm's un-ablated frame byte for byte on every root.
     If it does not, the check cannot be made and the hypothesis does not pass.
   - The check is always run and reported for all four hypotheses, whatever their result.
   - The scene-wide shadow-off re-render is dropped: TASK-059's ruling found it uninformative.

"Beats the prior" (TASK-059 §9) is reported per stratum at the unadjusted 95 % level. It is
descriptive and decides nothing here.

## 8. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§9), or the run stops before writing a complete report (crash, wall cap) | nothing is read | one repeat (§10) |
| **O-LOOK-E0** | L-E0 passes | A reset-independent look at the existing 112 px onboard camera makes the apple and the first command readable, and the existing E0 encoder exposes them. | A **corpus collection** whose episodes start with this look prefix, at the 112 px onboard camera. **It is still prereg-gated**, and no control formulation is preregistered until its own probe passes. |
| **O-LOOK-RAW** | L-raw passes, L-E0 does not | The look makes the information present at 112 px, but frozen E0 does not expose it. | The look-prefix corpus at 112 px, prereg-gated, **plus** encoder/representation work, preregistered separately. |
| **O-LOOK-224** | neither L hypothesis passes; LH-raw passes | Clearing the wrist is necessary but not sufficient at 112 px. With more pixels per apple it suffices. | A look-prefix corpus at a **224 px onboard camera** (a camera-resolution change), prereg-gated. It needs an encoder at the new input size, because E0's contract is 112 px. |
| **O-OVERVIEW** | no look hypothesis passes; O-raw passes | A fixed scene camera reads the apple where the onboard views tested here do not. | A corpus with the `overview` camera, prereg-gated. **On the real robot this is a hardware/workspace design change** (the G1 has no scene camera), stated as such. |
| **O-NONE** | no hypothesis passes | No tested observation change makes the apple position and the first command readable with the frozen readouts at n = 190, not even with the wrist cleared and 44 median apple pixels. | **A hardware or camera design change**, stated as such: camera placement or intrinsics beyond those tested (e.g. a wrist or dedicated table camera), or a perception approach this protocol does not test. **No corpus collection on any tested observation.** |

**Always reported, whatever the row:**
- every hypothesis's numbers, p-values, Holm steps and spurious check;
- every source on every stratum, the secondary estimate, and every reported-only source;
- the rows that also match further down the order. Example: under O-LOOK-E0, whether O-raw
  passed too.

**Qualifiers, not rows:**
- a hypothesis that meets the unadjusted bars but not Holm is reported as "unadjusted only; not a
  pass";
- a hypothesis that beats the prior on all roots without passing is reported as "partial
  information".

**Stated in advance.**
- TASK-059's visible stratum is the named risk: at 112 px a visible apple of median 5 px was not
  read beyond the prior, even by raw pixels. L's post-look apple has a median of 12 px, O's a
  median of 12 px, and LH's a median of 44 px.
- **An O-NONE outcome is a live possibility and would be reported as such.**
- The order of the rows follows the size of the change to the robot:
  - a reset prefix only (L at 112 px, with the existing encoder, then without it);
  - then an onboard camera setting (LH);
  - then an external camera (O).

**In every outcome:** nothing is refitted or re-thresholded after the numbers are seen. Learned
Apple→Plate stays at 0 successes. `exemption_spent` stays `false`. The executing agent recommends
a next task and does not choose it.

## 9. Guards: any failure voids the run

| guard | condition |
|---|---|
| **G-hash** | Every file in the manifest's `hashes` matches. This is TASK-059's list plus `info_ceiling.py`, `probe_info_ceiling.py`, `calibrate_observation_reprobe.py` and TASK-059's manifest. Every episode file is checked against the dataset manifest's sha256 before it is decoded. Encoder weight digests match. The tree is clean. |
| **G-split** | Exactly the plan's 190 train + val roots, with splits that agree with the dataset manifest. Test ids, cohort C, cohort D and seeds outside 48000–48199 are refused. The fold hash matches. |
| **G-render** | The 112 px re-render at reset is byte-identical to the stored frame on 190/190, and the maximum std of reset proprioception is ≤ 1e-6. |
| **G-expert** | The recomputed reset expert equals the recorded step-0 label within 1e-6 on the 152 non-aim roots, and the apple label equals the reset truth within 1e-6 m. |
| **G-look** | (a) The executed sequence's sha256 equals the manifest's. (b) On every root, in every simulation instance, the applied commands equal the requested ones (max abs 0) and are identical across roots; no projection is infeasible and no command is rejected. (c) The post-look joint state's maximum absolute spread across roots is ≤ 1e-6. (d) **After every look command**, on the 112 px instance: apple xy displacement from reset ≤ 1e-6 m, plate displacement ≤ 1e-6 m, and no hand–apple contact. |
| **G-prior** | The reset-target baselines equal TASK-059's manifest, and the post-look baselines equal this manifest's calibration, each within 1e-9 on every key. |

Render-path validation (§5) is **not** a void guard; it demotes an arm.

**PR 2's tests** must exercise, in both directions:
- every guard;
- the render-path checks, including P4 and P5: a planted 1-pixel difference fails the check and
  demotes the arm, and a clean render passes (owner ruling);
- the Holm step-down;
- the p-values against the Wilson and percentile bars;
- the spurious check;
- a proof that the look motion is reset-independent. The sequence is a constant that takes no
  argument, and executing it against two different resets issues identical command streams;
- a crash that still writes `report.json` with outcome `V` and a `void_reason`;
- non-finite values written as `null` and listed in `non_finite_fields`.

## 10. Void rule

- A run that stops early for any reason (a guard, a crash, the wall cap) is **V**. Nothing in it
  is read.
- **Exactly one from-scratch repeat** is allowed, with the same seeds, caps and device and a
  clean tree, into a new directory (`run-2`).
- **A second void closes TASK-061 as INCONCLUSIVE.**
- A fix between the two runs is limited to runner mechanics. It goes through a reviewed PR and a
  fresh pre-run GO, and it is disclosed in the results.

## 11. Budget and recording

- **Device:** CPU. **Wall cap:** 3 h for the whole run; expected 30–60 min. (The calibration,
  with four simulation instances, takes about 4.5 min. TASK-059's run took 619 s for six sources;
  this one has eleven feature sources, counting the reported-only ones.)
- **Seeds:** folds 59; inner folds 5900 + k; bootstrap 5901 (all as TASK-059).
- **Recorded:**
  - the code revision, with a clean tree required;
  - every hash;
  - encoder digests;
  - the fold hash;
  - the look sequence hash and the applied commands;
  - per-root out-of-fold predictions for every source;
  - selections per fold;
  - p-values and Holm steps;
  - spurious checks;
  - render-path validation counts;
  - apple pixel counts per arm;
  - elapsed time and peak RSS.
- **Output:** `outputs/task061-observation-reprobe/run-1/`. The runner refuses to overwrite.
  Nothing prior under `data/`, `checkpoints/` or `outputs/` is modified.

## 12. Pre-freeze calibration (fits nothing)

`uv run --no-sync python scripts/calibrate_observation_reprobe.py --output
outputs/task061-observation-reprobe/calibration-v5.json` (sha256 in the manifest; about 400 s on
CPU).

- **What it reads.** It opens only the 190 train + val episode files, each after its sha256
  check, and constructs no `DatasetStore`.
- **What it computes.** Visibility, pixel counts, render-path checks, the look facts, the targets
  and the prior-only baselines. It fits no readout.
- **Superseded artifacts.** Earlier versions of the script wrote `calibration-draft-1.json`,
  `calibration.json`, `calibration-v2.json`, `calibration-v3.json` and `calibration-v4.json`. All
  are kept, and their hashes are in the manifest.
  - The draft reported apple displacement as one 3-D number (3.37 mm). It was then split into xy
    (1.5e-15 m) and z-settling.
  - v1 lacked the hold control of §3.1.
  - v2 lacked the renderer warm-up and P4 (§5). v2 was the version under the first review.
  - v3 lacked P5.
  - v4 lacked the first-render check (§5), which the owner's ruling asked for.
  - Every value the versions share is identical, apart from elapsed time.
- **Disclosure: the prior baselines.** The brief limited pre-freeze calibration to rendering facts.
  The prior-only baselines also need the expert's post-look command, a simulation fact. They read
  no observation and fit no readout, exactly as TASK-059's calibration did, and they are needed to
  check the bars against a prior-only baseline.

## 13. Not done (declared)

- No controller is trained or evaluated, and no closed-loop attempt is run.
- No corpus is collected, and no dataset is written.
- No cohort is opened. The test split is not decoded, and cohorts C and D are never simulated.
  - `DatasetStore.verify`, reached through the encoder loader, reads episode bytes only to check
    their sha256, as in TASK-059.
- No encoder is trained or fine-tuned. No change is made to `policy.py`, `models/`, `simulation.py`,
  `info_ceiling.py`, `probe_info_ceiling.py`, or any checkpoint.
- The carry failure after a grasp (TASK-057 §5) stays out of scope.

## 14. Process

1. **PR 1:** this document, the manifest, the calibration script and the task card. It merges on
   an independent reviewer's **reported** APPROVE and green CI.
2. **PR 2:** the runner and the tests of §9. It merges on a reported APPROVE and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk. The same rule applies to merging: only on the
   reviewer's reported APPROVE.
4. **PR 3:** the results document, the results manifest, a summarize script and the recommended
   next task. The card goes to done.
5. **One task, one agent.** No closeout or follow-up agent is launched while the original task
   agent can still resume.
6. **A protocol defect found after the freeze** is escalated to the task owner and fixed only
   through a disclosed amendment.
