# Apple→Plate observation ceiling v2: how precisely each view reads the place, against how precisely the place must be aimed (TASK-075)

**Status: preregistration (Stage 0).** Only these have run:
- the place-tolerance calibration τ (§2), on development seeds 55000–55031, which are now spent;
- the smokes of §10, on smoke seeds 55050–55099, which are mechanics only.

No root of the source corpus has been re-rendered, and no readout of a new view has been fitted.
The reference view's TASK-074 numbers (2.872 cm and its baselines) were known before this
protocol was written; they are disclosed in §1.

**Status lines**
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). Nothing in this task changes that number.
- **This is a measurement study.** It trains no world model and no policy, runs no controller in
  closed loop except the privileged τ calibration, and makes no control claim. Nothing here makes
  LeWM, sampling-based planning or behaviour cloning the project's control approach. No control
  line is primary.
- **What the best row, OBS-ONBOARD, would show:** that a frozen-DINOv2 readout of one onboard
  view reads the apple-minus-plate offset, during the carry and place of TASK-074's condition,
  with the upper 95 % bound of its *median* error at or below τ, the largest *constant* target
  error the place tolerates; that it beats its random-init floor and a prior that reads no image;
  and that it does not do so without the plate in view. A1 is necessary, not sufficient: up to
  half the windows may err by more than τ (§6.1, the reported statistics of §6.4). It would license the next
  LeWM task on that view with a calibrated readout bar. It would not show that a world model
  predicts that offset.

Manifest: `benchmarks/manifests/apple-obs-ceiling-v2.json` (the frozen block, the τ calibration
record, the source corpus facts, the pins, the smoke record).
- **Design:** `src/embodied_jepa/obs_ceiling_v2.py`.
- **Workers:** `src/embodied_jepa/obs_ceiling_v2_runtime.py` (the τ attempt, the multi-view
  re-render).
- **Offline:** `src/embodied_jepa/obs_ceiling_v2_offline.py` (features, Gram matrices, the
  cross-fitted readouts, the statistics).
- **Runner:** `scripts/run_obs_ceiling_v2.py`, modes `preflight`, `tau`, `source` (smoke only),
  `render` and `readouts`.
- **Tests:** `tests/test_obs_ceiling_v2.py`.
- **Pinned:** the frozen block, every code file of this task, its test file, `hand_crop.py`, and
  every TASK-074 pin, unchanged (a test compares all of them).
- **Not pinned:** this document and the manifest; the smoke record is added after the smokes, so
  that the smoked code stays the frozen code. The manifest records this document's sha256
  (`protocol_document_sha256`), and a test checks it.
- **No existing file is modified.** TASK-074's runner is imported, unmodified, for its guards,
  its featurisation and its corpus seal.

---

## 0. The delegation and the rulings made under it

**The owner's delegation of 2026-09-30, verbatim** (`OWNER_DECISIONS["owner_delegation_verbatim"]`,
carried from TASK-074):

> "do the work without me - if you have decidions, choose your recommandation. do the work in
> subagents, use this chat just for updates. use subagents and workflows. goal is to continue
> working and try to get a real LeWM for the Unitree G1 without my help"

Every ruling below picks the option the design proposal marked (recommended), or, where it gave
none, the option this protocol recommends. Each is labelled **"decided by Claude under owner
delegation (2026-09-30), ruled 2026-10-01"** (`obs_ceiling_v2.DELEGATED`,
`DELEGATED_CHOICES`).

| # | ruling | what it fixes |
|---|---|---|
| R1 | the next task is candidate (d), measurement first (the proposal's recommendation) | views onboard 112 (reference), onboard native 224, the hand crop, overview 224; (c)'s learning curve as a factor; real-robot pretraining (b) deferred |
| R2 | τ under TASK-074's condition with K1's H-handover arm and a planted error | §2 |
| R3 | τ's levels 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5 cm on 32 resets; the bar 28/32; "every smaller level passes" | §2 |
| R4 | reuse `apple-far-shift-v2`: its stored 112 px frames are the reference view; the new views come from re-simulating its train and val roots; its test roots are excluded | §3 |
| R5 | cross-fitting over all 270 read roots | §4.3 |
| R6 | "beats the floor / the clock prior" = the upper 95 % bound of the paired median-error ratio < 1.0 | §6.1, deviation X-2 |
| R7 | the clock prior is cross-fitted on the outer-training roots; TASK-074's o* prior is reported | §4.2 |
| R8 | the plate-hidden check gates; the apple-hidden check is reported | §6.1 |
| R9 | OBS-REPRESENTATION needs R_full or R_pix to pass while no view passes with R_off | §6.2 |
| R10 | view order by the size of the change to the robot; no wrist view | §3.1 |
| R11 | the learning curve informs only the next step under OBS-NONE | §7 |

## 1. The question, and why now

**Question.** Under TASK-074's condition (`apple-to-plate-v2`, the plate moved 9 cm to the
robot's right at step 300), for each of four views: is a frozen-DINOv2 readout of the
apple-minus-plate offset, at the frames a place planner would read, precise enough for the place
to succeed, and is that precision caused by seeing the plate?

**Why.** TASK-074 closed INCONCLUSIVE (`apple_lewm_planner_v2_results.md`:3–7). Its binding
finding was on the observation side:
- R_off on encoded pooled 4 × 4 DINOv2 tokens at t + 16 has a median error of **2.872 cm** on
  O2's moving windows (90 windows, 22 val roots), against 4.351 cm for persistence and
  **3.123 cm** for a clock prior that reads no image (`apple_lewm_planner_v2_results.md`:14–20,
  151–152).
- O2 required ≤ 1.0 cm (`apple_lewm_planner_v2.md`:487). That bar was carried from TASK-073 and
  never calibrated (`apple_lewm_planner_v2_results.md`:309–320).
- The lesson (`apple_lewm_planner_v2_results.md`:329–334): a bar on a readout must be calibrated
  from that readout's measured ceiling on representative data before the freeze.

**Motivation only (descriptive, not a gate, and not used by any bar here).** TASK-074's
descriptive diagnostic on run-3's models found the action-blind N about as good as W on O2's
windows: N's ratio against persistence had upper bounds 0.831, 0.756 and 0.699 (0.70–0.83)
against W's 0.826, 0.800 and 0.757 (`apple_lewm_planner_v2_results.md`:291–293, 302–306). On
those windows the readout, not action conditioning, set the floor.

**What is already known about the views (different quantities, phases and readouts; none is
the quantity measured here):**
- TASK-061 read the post-look apple from raw 112 px onboard pixels to 0.469 cm and from 224 px
  to 0.386 cm, and the reset apple from overview 112 px pixels to 0.183 cm
  (`apple_observation_reprobe_v1_results.md`:92–95); resolution added 0.084 cm once the wrist was
  clear (`apple_observation_reprobe_v1_results.md`:163–164).
- The hand-crop window shows about 8× the apple pixels of onboard 112 px at reset (p75 31 against
  4; `apple_info_ceiling_v1_results.md`:245–247). It was never read with frozen DINOv2.
- TASK-074's Probe C read the far plate from full tokens with kernel ridge to 0.55–0.69 cm when
  its training covered far frames, against 5.53 cm when it did not (`apple_lewm_planner_v2.md`:
  172–177).
- What makes R_off read 2.87 cm is unknown: pooling, the linear readout, coverage or occlusion.

**What this study does not ask.** Whether a world model predicts the offset; whether any
controller succeeds. Those belong to the next task, which this study's row licenses or not.

## 2. Stage 0: the place tolerance τ (development; measured before the freeze)

**Definition** (`TAU`, frozen). τ is the largest planted target error at which P-3's pick
followed by e9's place still leaves the apple at rest on the plate on ≥ 28 of 32 resets.
- **Arm.** K1's H-handover (`lewm_planner_v2_runtime.run_attempt_task`, arm H-handover; P-3's
  pick, then `place_planner.PlannerController` in mode "truth", which aims e9's place primitive
  at the true moved plate at every decision step 405, 421, …, 485). Here the harness's plate
  schedule is displaced by a planted error e (`obs_ceiling_v2_runtime.run_planted_task`): only
  the aimed target moves; the plate, the shift and P-3's pick are untouched. A guard
  (`planted_check`, G-planted) checks that every non-fallback decision aimed at plate + e to
  1e-9 m.
- **Condition.** TASK-074's: the plate moves 9 cm to the robot's right (−y) at step 300, by
  `lewm_planner_v2.shift_vector`, on `condition_reset` resets (K1-PASS at 9 cm, H-handover 32/32,
  `apple_lewm_planner_v2_results.md`:47, 68–80).
- **Levels.** |e| ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5} cm. One direction per reset, uniform on
  [0, 2π) from `SeedSequence([7501, seed])`, the same at every level.
- **Count.** The counted success (T71-R1/R2: at rest after a latched grasp and place), the
  TASK-070/K1 bar family (`apple_to_plate_v2_expert.md`:3–5).
- **Rule.** τ = the largest level L such that every level ≤ L reaches ≥ 28/32. If level 0 fails,
  the row is TAU-NONE: the place itself is the constraint, and nothing is rendered.
- **Seeds.** 55000–55031, development, now spent (`SEED_RANGES["tau"]`; a repository search of
  2026-10-01 found no seed use in 55000–55999).

**Why this quantity.** The planner's put-down spot is where e9's place primitive is aimed. A
readout error in the plate (or the offset) becomes a target error of the same size in cm. A
constant planted error is the simplest model of that: the target is frozen at 505 from the last
decision's reading, so a readout error at the last decision acts as a constant target error.
A per-frame readout error that varies between decisions is not modelled (§9b).

**Result.** The calibration ran once, as `run_obs_ceiling_v2.py tau`, at `26c64d9` on a clean
tree (frozen sha `de3218c7…` before the record below was added), on the Linux PC:
- report `outputs/task075-tau/run-1/report.json` in the `task075-prereg` worktree, sha256
  `08a152f14e63902e7e045f8890a000c0eedd0d7ac751af6bd879e6e512c519d3`;
- 2026-10-01T12:35:57Z → 12:40:22Z (265 s), start load 0.05 / 1.43 (1 / 5 min), MemAvailable
  checked, peak tree PSS 7.94 GiB (8 processes);
- G-repro: every TASK-072 run-1 reproduction check passed; no render disagreement;
- G-planted: every decision aimed at the true plate plus e (largest difference 0.0 m), with no
  fallback and no blocked move at any level.

| planted error \|e\| (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| counted successes (of 32) | 31 | 31 | **28** | 22 | 23 | 21 | 15 | 11 | 8 |
| at rest (of 32) | 31 | 31 | 28 | 22 | 23 | 21 | 15 | 11 | 13 |
| ≥ 28? | yes | yes | yes | **no** | no | no | no | no | no |

**τ = 1.0 cm** (`TAU_MEASURED`, frozen; the seeds are spent).
- The one level-0 failure is seed 55028, where P-3's grasp failed (no latched grasp).
- **τ sits exactly on the bar** (28/32 at 1.0 cm). One run per level. Wilson 95 % intervals:
  31/32 at 0.5 cm [0.84, 0.99]; **28/32 at 1.0 cm [0.72, 0.95]**; 22/32 at 1.5 cm [0.51, 0.82].
  The curve is noisy: 22 at 1.5 cm against 23 at 2.0 cm, and **10 of the 32 seeds are
  non-monotone** across the levels (a seed fails at one level and succeeds at a larger one).
  A confidence-bounded rule could not define τ at n = 32 (even 31/32 has a lower bound of 0.84
  < 0.875), so the point rule fixed before the run stays; §6.3 pre-commits a re-measurement.
- **τ = 1.0 cm coincides numerically with TASK-074's uncalibrated O2 bar of 1.0 cm.** It was
  measured here under the rule fixed before the run (at `26c64d9`), not carried from TASK-074.
- **τ depends on the error's direction.** Excluding seed 55028 (its grasp fails at every
  level), errors pointing to the robot's right (−y half) fail earlier: 16/19 against 12/12 (+y
  half) at 1.0 cm, 11/19 against 11/12 at 1.5 cm, 12/19 against 11/12 at 2.0 cm. The uniform draw
  put 20 of the 32 directions in the −y half. See §9b.
- For comparison only (other conditions): TASK-070's e9 rested 30/32 at 1.0 cm plate error on its
  gated seeds and 59/64 (1.0 cm) and 57/64 (1.5 cm) on development seeds
  (`apple_to_plate_v2_expert.md`:3–5, 150–158); TASK-074's Probe A gave 6/32 for e9 aimed at a
  readout with a 3.5 cm median plate error (`apple_lewm_planner_v2.md`:154–156).
- **What τ = 1.0 cm implies for the reference view, before anything is run:** TASK-074's
  reference readout reads 2.872 cm on its val windows (§1). A1 needs the *upper bound* of the
  cross-fitted median at or below 1.0 cm, a factor of about 2.9 below that; the reference view is
  very unlikely to be admitted. This was visible when τ
  was frozen; τ was not chosen, it was measured, and the rule (§2) was fixed before the run.

## 3. The views and the corpus

### 3.1 The four views (`VIEWS`, `VIEW_SPECS`)

Ordered by the size of the change to the robot (R10):

| view | what it is | change to the robot |
|---|---|---|
| `onboard112` (reference) | `apple-far-shift-v2`'s stored onboard frames, 112 px; DINOv2 sees them resized to 224 by bicubic interpolation (`pretrained_encoder.py`:127–143) | none |
| `onboard224` | the same head camera (`onboard_rgb`, `simulation.py`:96–104) rendered natively at 224 px; DINOv2 sees it without a resize | resolution |
| `handcrop` | a 112 px window of a 320 px onboard render, centred on the projected right palm from robot kinematics only (`hand_crop.py`:1–12, 28–71; TASK-048) | resolution and a kinematic crop of the same camera |
| `overview224` | the scene's external `overview` camera (`simulation.py`:93–95) at 224 px | an external camera, against PRD.md:208 ("Do not require additional cameras for MVP") |

There is no wrist view: the scene defines only `overview` and `onboard_rgb`
(`simulation.py`:93–104), and a scene change needs its own review (R10).

**Deployability (unverified).** Whether the G1's head camera delivers ≥ 224 px frames is
**[unverified]**; the repository holds no measured camera manifest (docs/HARDWARE.md:37, 61–62
list it as a commissioning item).

### 3.2 The corpus: reuse, re-render, exclusion (R4)

- **Source.** `apple-far-shift-v2` (TASK-074's corpus; manifest sha256 `fe7ab915…b134bd`, plan
  digest `41c50571…34c0b0`, collected at `9d9b03c`; `apple_lewm_planner_v2_results.md`:47–48,
  88–93). 300 roots: 240 train, 30 val, 30 test. It was collected by the privileged scripted
  collector e9, under the TASK-074 plan (25 % unshifted, 75 % moved by 3–12 cm at step 300, 50 %
  mis-aimed, noise levels 0–3; `apple_lewm_planner_v2.md`:392–398).
- **The reference view is reused as stored.** Its frames are read from the sealed corpus.
- **The new views are a stage of this protocol, not development work.** Stage 1 (`render`)
  re-simulates each of the 270 train and val roots from its own stored metadata (seed, reset,
  shift, mis-aim, noise) with TASK-073's collector loop, and captures the three new views at the
  rendered steps. A re-simulated root must equal the stored one: every array (states, base,
  requested, applied, phase, apple, plate, dropped, hand contact, latched) exactly, and its
  112 px frames at the rendered steps by the render rule (at most one level in at most 16
  pixels; `lewm_planner_v2.RENDER`). Anything else is a V (G-repro-render).
- **Test-split exclusion, recorded.** The 30 test roots are never simulated, rendered or
  decoded: `CorpusReader` refuses them (Q-split), the render stage lists only train and val, and
  the sorted test ids' digest (`4fa7512c…7bf`) is frozen and checked.
- **Hidden renders.** At the six eval steps 421, 437, …, 501, every view is also rendered with
  the apple hidden and with the plate hidden (every geom of the body moved to geom group 5 and
  that group turned off; the groups are restored at once; physics is untouched, which the
  array check above verifies).
- **Warm-up.** Every extra renderer renders once, discarded, at the start of every root
  (`apple_observation_reprobe_v1_results.md`:271–272).
- **The views store** `apple-far-shift-v2-views` holds, per root, the three new views at the 30
  rendered steps and every view's hidden renders; it is sealed with every file's sha256.

### 3.3 The steps (`TRAIN_STEPS`, `DECISION_STEPS`, `EVAL_STEPS`, `RENDER_STEPS`)

- Fit rows: every 8th band frame from 384 to 560 (23), TASK-074's R_off rows
  (`apple_lewm_planner_v2.md`:1041–1045).
- Windows: the decision steps t = 405, 421, …, 485; the readout reads the frame at t + 16
  (421, …, 501). Persistence reads the frame at t.
- Rendered: the union, 30 steps per root. A root is read if its band reaches past 400
  (TASK-074's featurisation rule, `run_lewm_planner_v2.py`:612–666).

## 4. The readouts (`READOUTS`)

### 4.1 The families, per view

| readout | features | estimator | role |
|---|---|---|---|
| **R_off** | pooled 4 × 4 pretrained DINOv2 patch tokens, 6 144-d | `RidgeReadout` (standardised on the fit rows, λ from {1e-3 … 1e3} by grouped inner CV), solved in the dual form, which gives the same readout as the primal (`latent_critic.py`:43–45; tested) | **admission** |
| R_floor | the same, from the seed-0 random-init DINOv2 (`pretrained_encoder.random_init`) | as R_off | floor (A2) |
| R_full | full pretrained tokens, 98 304-d | `info_ceiling`'s kernel ridge: linear or RBF on the Gram matrix, centred on the fit rows, λ ∈ {1e-4 … 1e4} (`info_ceiling.py`:35–36, 285–326), family and λ by inner CV grouped by root | representation (B) |
| R_pix | the view's raw uint8 pixels | as R_full | the information ceiling (B), TASK-061's family |
| R_proprio | the stored robot state | `RidgeReadout`, primal | reported: a kinematic baseline |
| plate / apple target | pooled pretrained tokens | as R_off, targets plate xy and apple xy | reported |

DINOv2 runs on CUDA under strict determinism, batch 64, with a CPU anchor check per view
(max |difference| of the pooled latent ≤ 1e-3). The full-token Gram is accumulated over two
column halves of the token dimensions, so only half of the full tokens is held at a time.

### 4.2 The baselines

- **Clock prior** (R7, gating in A3): the per-step median offset over the outer-training roots
  at t + 16. It reads no image and is fitted on the same data as the readout.
- TASK-074's o*(t + 16) prior (`lewm_planner_v2.O_STAR_CM`): reported.
- Persistence (R_off on the frame at t): reported.

### 4.3 Cross-fitting, folds, learning curve (R5)

- **Outer folds:** 5, by root, over the read roots (a seeded permutation in plan order,
  position mod 5; salt 7502). Every root's windows are read by readouts fitted on the other four
  folds only; no window's own root enters its fit. Tests check this (a spy on every fit's
  groups; poisoned held-out labels for the kernel ridge).
- **Inner folds:** 5, grouped by root (salt 7503).
- **Windows:** O2's moving cohort: every read root's decision-step windows whose true offset
  moves by ≥ 1 cm over the 16 commands (`apple_lewm_planner_v2.md`:485–486). The quantity is
  O2's: the Euclidean error (cm) of the offset readout at t + 16.
- **Learning curve** (candidate (c), R11): R_off fitted on 25 and 50 % of each outer fold's
  training roots (salt 7504), beside 100 %.
- **Intervals:** root-clustered bootstrap percentile intervals, 10 000 resamples, 95 %
  (salt 7505).

### 4.4 G-repro-off: the reference reproduces TASK-074 first

Before any new number, the readouts stage runs TASK-074's own R_off path
(`run_lewm_planner_v2.featurise_sources`, `lewm_planner_v2_offline.fit_r_off`,
`o2_statistics`) on the stored reference frames. It must reproduce TASK-074's train-stage numbers
exactly: encoded 2.8722625765232763 cm, persistence 4.350821813169937 cm, clock prior
3.1225649505050876 cm, 90 windows, 22 roots, R_off's readout sha256 `260af3f7…803157`, 5 175 rows,
225 groups (`TASK074_REFERENCE`; these were byte-identical in TASK-074's runs 1–3,
`apple_lewm_planner_v2_results.md`:151–152, 323–324). Anything else is a V (G-repro-off).

## 5. Stages and stop rules

Nothing of Stage 1 or 2 runs before this PR is merged **and** a pre-run reviewer's GO is reported
for that stage. The coordinator is told before each stage starts.

1. **Stage 0 (this PR).** The protocol, the code, the tests, the τ calibration (§2) and the
   smokes (§10).
2. **Stage 1, render** (`run_obs_ceiling_v2.py render`). The 270 roots, 6 workers; G-repro-render
   (§3.2); outcome VIEWS-SEALED. Boundary: `cohort_first_render_utc`.
3. **Stage 2, readouts** (`readouts`). G-repro-off (§4.4); the read roots and folds; then
   `first_outcome_utc` (the boundary: before the first view is featurised or fitted); then per
   view: featurisation, the Gram matrices, the cross-fit, the statistics; then the row (§6.2).
   The errors of every readout on every window are saved per view.
4. **Stage 3, results PR.** Every view's numbers are reported, with the reference view's TASK-074
   numbers beside them. An independent reviewer checks every restated number.

## 6. Gates and rows

### 6.1 Admission of a view (`ADMISSION`, R_off; all four must hold)

| | condition | why |
|---|---|---|
| **A1** precision | c_V := the upper 95 % bound of R_off(V)'s median error ≤ τ | it compares a median upper bound with a constant-error tolerance: necessary for the place, not sufficient (half the windows may exceed the median; the reported 87.5th percentile and τ-curve prediction of §6.4 show by how much) |
| **A2** floor | the upper bound of median(e_R_off) / median(e_R_floor) < 1.0 | pretrained features carry the precision (TASK-063's lesson; in M2 the random-init floor tied P-3) |
| **A3** clock prior | the upper bound of median(e_R_off) / median(e_clock) < 1.0 | the image adds over a prior that reads none (TASK-074: 2.872 against 3.123 cm) |
| **A4** plate hidden | the lower bound of R_off(V)'s median error on the plate-hidden t + 16 frames > τ | without the plate in view, the readout does not reach the place precision; otherwise the precision comes from something else, for example e9's arm pose, which is aimed at the plate in this corpus |

The apple-hidden error is reported, not gating (R8): in the carry phase the apple sits in the
hand, whose pose is a legitimate kinematic cue. The plate is the quantity only vision provides
after the move.

**Every bar is a measured quantity or a definitional 1.0.** τ is measured (§2); c_V is measured;
the ratio bars ask for strict superiority with 95 % confidence and carry no tuned margin (R6,
X-2); A4 reuses τ.

### 6.2 The rows (`decide`, first match)

| row | condition | consequence (`ROW_CONSEQUENCES`) |
|---|---|---|
| **V** | the void rule (§7) | — |
| **TAU-NONE** | τ undefined (level 0 < 28/32) | escalate; found in Stage 0, so the protocol is not frozen |
| **OBS-ONBOARD** | an onboard view (onboard 112, onboard 224, hand crop, in that order) is admitted; the first is chosen | **close.** The next task preregisters the LeWM place planner or critic on that view |
| **OBS-EXTRA** | no onboard view is admitted; overview 224 is | **escalate:** a hardware change against PRD.md:208 needs the owner |
| **OBS-REPRESENTATION** | no view is admitted with R_off; some view passes B1, B3 and B4 with R_full or R_pix | **escalate:** the pooled latent or its linear readout binds, not the data; a representation change is a design question for the owner, not a silent reopening of a closed line |
| **OBS-NONE** | otherwise | **the clause fires** (§7) |

**B (`REPRESENTATION_PASS`)** is A1, A3 and A4 with R_full or R_pix in place of R_off. There is no
floor for raw pixels, and R_full's floor is not fitted (§9b).

**Under OBS-EXTRA** the onboard views that pass B with R_full or R_pix are listed beside the row
(`decide`'s `onboard_representation_passes`), so the owner's hardware ruling sees the
representation alternative too. The row order is unchanged.

### 6.3 The downstream rule (for the next world-model task)

If the row is OBS-ONBOARD with view V*, the next task's O2-type encoded-readout bar B must satisfy
**c_V\* ≤ B ≤ τ** (`downstream_bar_interval`, `bar_is_valid`), and its predicted-latent bar must
be ≤ τ, with any allowance above c_V\* calibrated on development data before its freeze (not
carried; TASK-066's 0.5 cm G5 margin was uncalibrated,
`apple_token_dynamics_v1_results.md`:234). TASK-074's 1.0 cm bar against a 2.872 cm readout is
exactly what `bar_is_valid` refuses (tested). If c_V > τ for
every view, no world-model task is preregistered on these views.

**Pre-committed re-measurement of τ** (`TAU_REMEASURE`, fixed now, before any view number exists). On an
OBS-ONBOARD or OBS-EXTRA outcome, τ is re-measured on fresh development seeds, with the same arm,
condition, levels, direction rule and rule, before the next world-model task freezes; its bar B
uses the re-measured τ (c_V\* ≤ B ≤ τ_re). If the re-measured τ falls below c_V\*, no bar is
feasible and that task is not frozen. This is because τ = 1.0 cm rests on one run at exactly the
bar (§2, §9b).

### 6.4 Reported-only statistics (`REPORTED`; they gate nothing)

Defined here, in the frozen block, before any view number exists:
- **The 87.5th percentile** of R_off(V)'s window errors, with a root-clustered interval.
- **The τ-curve prediction.** Each window's R_off(V) error e (cm) is mapped to the measured
  counted-success fraction (`TAU_MEASURED["counts"]` / 32) by linear interpolation between the
  levels 0, 0.5, …, 5 cm (`numpy.interp`, `tau_curve_fraction`); e > 5 cm takes the 5 cm
  fraction 8/32, which is optimistic there. The predicted successes out of 32 are 32 × the mean of
  the mapped values over the windows, with a root-clustered interval. It treats each window's
  error as a constant target error in a random direction, which is what τ measured. For scale: a
  window error of 0.75 cm maps to 29.5/32, 1.0 cm to 28/32, 1.25 cm to 25/32 (tested).
- **The signed mean error** (predicted minus true offset) along x and along y, in cm, with
  root-clustered intervals: τ's tolerance depends on the direction (§9b).
- **A sensitivity row:** the views that would be admitted, with A1–A4 unchanged, at τ = 0.5 cm
  and at τ = 1.5 cm, the neighbouring levels (`decision.sensitivity_admitted`). Never a row.

## 7. Void rule, abandonment clause, next steps

**Void rule** (frozen, `VOID_RULE`). A guard, a crash, a cap, a CUDA allocation failure or a
failed reproduction (G-repro-render, G-repro-off) makes the stage V, and nothing in it is read.
Batches are never shrunk to fit.
- A V before a stage's boundary is not a spent attempt; it may be repeated as-is, recorded.
- After the boundary, a stage may be repeated once from scratch after a reviewed fix; a second V
  closes TASK-075 as INCONCLUSIVE.
- Nothing is re-thresholded, refitted or re-selected after its numbers are seen.

**Abandonment clause** (fixed now, `CLAUSE_ROWS`, `CLAUSE_SCOPE`). It fires on **OBS-NONE** only.
It closes: "a LeWM world-model task (planner or critic) for the place phase of
`apple-to-plate-v2` under TASK-074's condition, built on any of these four views with frozen
DINOv2 features: none is preregistered without new evidence of a different kind (a new view, a
new readout family or a task or condition change)". The LeWM backend, the v2 task and the
product goal stay open.

**Next steps by row:**

| row | next step |
|---|---|
| OBS-ONBOARD (V*) | re-measure τ on fresh seeds (§6.3, `TAU_REMEASURE`), then preregister the LeWM place planner or critic on V*, with c_V\* ≤ B ≤ τ_re; a new corpus on V* is that task's own stage |
| OBS-EXTRA | the owner rules on an external camera (PRD.md:208), seeing any onboard B passes; τ is re-measured (§6.3) before any world-model task on the overview freezes; no onboard world-model task is preregistered meanwhile |
| OBS-REPRESENTATION | the owner rules on a representation change (for example a full-token latent for the world model); the study reports which readout and view passed |
| OBS-NONE, learning curve still falling on any view (the lower bound of median(e_50) − median(e_100) > 0) | candidate (c): a larger corpus, preregistered with its own memory plan |
| OBS-NONE, otherwise | a task or condition change |
| TAU-NONE | the place itself is re-examined before any perception work |

Candidate (b) (real-robot pretraining) stays deferred under every row: it changes only the
world model, and none of these rows is about the world model.

## 8. Platform, guards, resources

- **Platform** as TASK-074 (`lewm_planner_v2.PLATFORM`): the Linux PC, `MUJOCO_GL=egl`, MuJoCo
  3.13.0, the main process in strict CUDA determinism; simulation on the CPU with 6 workers;
  DINOv2 on CUDA; the fits on the CPU (OpenBLAS 16 threads, `THREAD_ENV`).
- **Quiet machine** (G-quiet, carried): a stage starts only with the 1- and 5-minute load
  averages ≤ 2.0.
- **Memory** (G-memory, carried): summed PSS of the process tree ≤ 12 GiB, sampled every 0.5 s
  (`run_guards.process_tree_memory`), and MemAvailable ≥ 16 GiB at the start. **The full-scale
  probe** (`readouts --smoke --scale`) runs the readouts stage's own function, `readouts_core`,
  on the smoke corpus presented at the real sizes (240 train + 30 val slots, every slot's source
  episode and views file decoded afresh), so the per-root terms scale as in the real stage
  (TASK-074 addendum A1's lesson). Its peak must be ≥ 2.0 GiB below the ceiling
  (G-memory-margin).
- **GPU** (G-gpu, new): the GPU is shared with a resident service of about 6.6 GB that is never
  touched. The readouts stage starts only if free GPU memory ≥ 4.0 GiB + 3.0 GiB, and caps its
  own process at 3.0 GiB (`torch.cuda.set_per_process_memory_fraction`), so ≥ 4 GiB stays free.
  An allocation failure is a V.
- **Disk** (G-disk, new): the render stage starts only with ≥ 10 GiB free plus 1.5 × the
  projected store (the smoke's largest file × 270), and a watch voids it below 10 GiB.
- **Caps:** 43 200 s per invocation, 300 s per attempt.

## 9. Why this is different from the closed lines

- **TASK-054, TASK-057, TASK-062, TASK-065** (`CLAUDE.md`, research-evidence rules): nothing is
  trained here; no control formulation is proposed; no encoder is trained; the latent is not
  CLS.
- **TASK-074's clause** (`apple_lewm_planner_v2.md`:580–588) did not fire (INCONCLUSIVE); its
  scope is untouched. This study is the "data or hardware change" its consequence names, made
  as a measurement before any planner variant. It re-renders `apple-far-shift-v2`'s states in
  new views; it preregisters no grid, cost, horizon or proposal variant.

## 9b. Limitations (stated plainly; not redesigned)

- **τ is one run of 32 resets per level, at exactly the bar.** Wilson 95 % intervals: 28/32
  [0.72, 0.95] at 1.0 cm, 31/32 [0.84, 0.99] at 0.5 cm, 22/32 [0.51, 0.82] at 1.5 cm; 10/32 seeds
  are non-monotone across the levels. τ can move by one level between runs; §6.3 pre-commits a
  re-measurement on fresh seeds before any world-model task freezes, and §6.4 reports the views
  admitted at 0.5 and 1.5 cm.
- **τ is anisotropic.** From the τ report's `per_reset` and `directions_rad` (seed 55028
  excluded): at 1.0 cm, 16/19 in the −y half against 12/12 in the +y half; at 1.5 cm, 11/19
  against 11/12; at 2.0 cm, 12/19 against 11/12. Errors towards the robot's right fail early,
  errors towards +y are tolerated to about 2 cm. A readout whose error is biased along y faces a
  tolerance that differs from τ; each view's signed mean error is reported (§6.4), not gated.
- **τ models a constant target error.** A readout whose error varies between the six decisions
  is not modelled; the last decision (485) sets the frozen target.
- **The views share one rollout.** They are paired (good) but not independent corpora; the
  corpus is TASK-074's plan (privileged e9, mis-aims, noise), not P-3's own carry.
- **c_V is the offset at t + 16 on O2's windows.** Other phases (the pick, the post-look frame)
  are not measured.
- **A4 is necessary, not sufficient.** A readout fitted on frames with the plate may fail on
  plate-hidden frames merely because they are out of its training distribution, so a clean A4
  does not prove the precision is plate-caused. R_proprio (a readout of the robot state alone) is
  reported beside it.
- **τ's condition and c_V's windows differ.** τ is measured under a 9 cm move; c_V pools all read
  roots (unshifted, and moves of 3–12 cm). c_V on the shifted and unshifted roots' windows is
  reported beside it (`subsets`), not gated.
- **The expert aims at the plate.** In this corpus e9's arm moves towards the (mis-aimed) plate,
  so the arm's pose carries plate information. A4 and R_proprio expose this; they do not remove
  it.
- **The ratio bars have no margin.** A view can pass A2 or A3 by a hair.
- **R_full has no floor, R_pix none by construction.** OBS-REPRESENTATION is an escalation,
  not an admission.
- **Four views, one encoder, one input size per view, one floor seed.**
- **The hand crop is 112 px from a 320 px render** (TASK-048's setting); other crop sizes are
  untested.

## 10. Stage-0 smokes (smoke seeds 55050–55099; nothing in them is read)

All smokes ran on the Linux PC, in the `task075-prereg` worktree's git-ignored `outputs/`. **The
rows the smokes printed are meaningless and are not read** (the readouts smoke's row uses a
placeholder τ of 2.0 cm on 13 smoke roots). Only mechanics, memory, time and disk are recorded.

### 10.1 The frozen-code smokes (the record for the GO)

After the #117 review's fixes (`79172c7`), the four smokes ran again at `79172c7c`, frozen sha
`a64b5833…ee24`, on a clean tree (`tracked_tree_dirty: false`), on smoke seeds only, each started
after the 1- and 5-minute load averages were ≤ 1.8, in `outputs/task075-smoke-3`. After them only
this document and the manifest changed; neither is pinned.

| smoke (report sha256) | start load (1, 5 min) | result |
|---|---|---|
| `tau --smoke` (`271cf0a1…`) | 0.720, 1.194 | 4 smoke seeds × levels 0 and 3 cm (4/4, 3/4, as before); G-repro passed; G-planted clean; 50 s; peak tree PSS 8.22 GiB |
| `source --smoke` (`153c8b40…`; corpus manifest `84df9423…`) | 1.614, 1.620 | TASK-074's collector on 16 smoke roots (10 / 4 / 2), sealed with TASK-074's seal; 22 s; peak PSS 7.47 GiB |
| `render --smoke` (`235c84c3…`) | 1.455, 1.623 | 14 train + val roots re-simulated (the 2 test roots excluded); **every array equal on 14/14 roots; 389/390 reference frames identical, 1 within the render rule** (one level in 8 pixels), 0 outside; 4.5 s per root (median, 6 workers); **1.85 MB per root** (largest); 17 s; peak PSS 7.91 GiB |
| `readouts --smoke --scale` (`93d561b1…`) | 1.604, 1.668 | the smoke stage (13 read roots, every view, every readout, the reported statistics of §6.4 and the sensitivity row, the row code path), then **the full-scale probe**: 240 + 30 slots decoded afresh, 251 read roots, G-repro-off's band featurisation at 270 slots; **peak tree PSS 8.45 GiB** (ceiling 12, margin 2.0: passes); GPU peak reserved 0.95 GiB (cap 3.0), 8.51 GiB free at start; the probe took 1 478 s, the smoke 1 539 s |

The previous record (§10.4) ran at `63c55086` under frozen sha `f6ed707c…`; the review's fixes
changed the frozen block (the reported statistics, `TAU_REMEASURE`) and the readouts code, so the
chain was repeated.

**Margins, stated plainly.**
- The readouts peak (8.45 GiB; 8.51 GiB in the earlier chain) is about 1.5 GiB inside the 10 GiB
  margin line. It was measured on
  cycled smoke episodes (786 frames each, like the real roots); the real stage reads about the
  same number of roots (TASK-074's train-stage rule kept 225 of 240 train roots).
- The scale probe's start load was 1.75 (5 minutes), under the 2.0 rule.
- One reference frame differed from the stored one by one level in 8 pixels in the record's
  render smoke (and in a development render smoke); the earlier frozen-code chain saw none. The
  render rule tolerates it; it is the renderer's known rare one-level effect.

### 10.2 Development smokes (historical; not the record)

While the code was written (`26c64d9` and its working tree), the same four smokes ran under
`outputs/task075-scratch/` and a first chain at `ad667d9` under `outputs/task075-smoke/` with a
dirty tracked tree (this document had been edited). The first render smoke at the working tree V'd
on an empty index array for a root that ended before step 384; the fix (an integer index) is in
the frozen code. A second V'd on G-hash, because the fix changed a pinned file before the
manifest's pins were refreshed. The `ad667d9` chain was stopped by its operator before its readouts smoke, and the
whole chain was repeated on a clean tree (§10.1). The first full-scale probe (`readouts-scale-1`,
working tree before `26c64d9`) peaked at 8.47 GiB and took 1 396 s.

### 10.3 A development check of G-repro-off on the real reference frames (before the freeze)

G-repro-off is exact equality, so a numerical drift in it would void the readouts stage before its
boundary on every repeat. It was therefore checked before the freeze, on the sealed
`apple-far-shift-v2` frames, through the runner's own `repro_off` under the stage's GPU cap
(`gpu_guard`, 3.0 GiB) and strict determinism. This reads only what TASK-074 already read and
published (§1, §4.4); it fits no readout of any new view.
- The script, as it ran (not pinned, not part of the runner):
  [`apple_obs_ceiling_v2_dev/repro_check.py`](apple_obs_ceiling_v2_dev/repro_check.py), sha256
  `2775a384b804992428add5b641b978cbfed2bfd8a6a40552bafb112af9d5b9fc`; run at `9e44657` (clean tracked tree), 2026-10-01 15:26 CEST, start load
  0.16 / 1.77.
- **Every TASK-074 number is reproduced exactly:** encoded 2.8722625765232763 cm, persistence
  4.350821813169937 cm, clock prior 3.1225649505050876 cm, 90 windows, 22 roots, R_off readout
  sha256 `260af3f7…803157`, 5 175 rows, 225 groups.
- **Stored output** (review of #117): the script's JSON output, verbatim as it ran, is committed
  unpinned as [`apple_obs_ceiling_v2_dev/repro_check_output.json`](apple_obs_ceiling_v2_dev/repro_check_output.json),
  sha256 `c1f2eead5509d8e8868c24bc9c9d90c9950d0d14f6b1e679bb1d7a4e3d2cff78`; its `matches` are
  all true. The script calls `repro_off(..., smoke=True)`, which reports a mismatch rather than
  raising, so the evidence is that recorded `matches` block, not an exit code. The independent
  reviewer of #117 re-ran it and reproduced every number.
- 270 train and val roots decoded, no test root; 44 781 band frames in 111 s; GPU peak reserved
  0.95 GiB (8.51 GiB free at the start).

### 10.4 Superseded frozen-code smokes (at `63c55086`, frozen sha `f6ed707c…d8a5`)

Before the #117 review, the same four smokes ran at `63c55086` on a clean tree
(`outputs/task075-smoke-2`): tau (`56765ab4…`, load 1.162 / 1.685, 50 s, 7.95 GiB), source
(`49f74f91…`, corpus `aafd09bb…`), render (`d3d7c0ac…`: 14/14 arrays equal, 390/390 frames
identical, 1.85 MB per root) and readouts with the scale probe (`3c353747…`: peak tree PSS
8.51 GiB, GPU peak 0.95 GiB, probe 1 414 s). They are superseded by §10.1 and kept as history.

## 11. Compute estimate (measured where marked)

| stage | measured or estimated | basis |
|---|---|---|
| Stage 0, τ | **265 s** (measured) | 288 attempts + the G-repro refit on 6 workers |
| Stage 1, render | about 4–6 min wall; store about 0.5 GB | 4.5 s per root (smoke median) × 270 / 6 workers; 1.85 MB per root |
| Stage 2, readouts | about 25–30 min wall | the full-scale probe: 1 478 s (1 414 s before the review's additions) at 251 read roots, 4 views (per view: featurisation and the full-token Gram 63–86 s, the pixel Gram 8–39 s, the cross-fit 209–214 s) |
| memory | peak tree PSS 8.45 GiB (readouts), 7.91 GiB (render) | §10.1 |
| GPU | peak reserved 0.95 GiB; cap 3.0 GiB; ≥ 4 GiB stays free | §10.1; the resident service (about 6.6–6.8 GB) is never touched |
| disk | about 33 GB free at the smokes; the stage needs ≥ 10 GiB + 0.75 GB | `render_smoke.bytes_per_root_max` × 270 × 1.5 |
| **total** | **under 1 h of machine time** after the GOs | no world-model training |

## 12. Deviations from the design proposal

| # | proposal | here | why |
|---|---|---|---|
| X-1 | a fresh corpus on an unused seed range | reuse `apple-far-shift-v2`; re-render its train and val roots in the new views | the reference view exists; reuse pairs every view with the stored reference and with TASK-074's measured numbers (R4) |
| X-2 | the floor/prior margin "calibrated from the reference view's development spread" | strict superiority (upper bound < 1.0), no margin | a development readout would have shown views' numbers before the freeze (R6) |
| X-3 | readouts on held-back val roots | cross-fitted over 270 roots | c_V's interval on 22 val roots would be too wide to admit anything (R5) |
| X-4 | "a clean apple- and plate-hidden spurious check" | the plate-hidden check gates, the apple-hidden check is reported | the apple is in the hand during the carry (R8) |
| X-5 | a wrist view "only if its scene change passes review" | no wrist view | no scene change is made in this task (R10) |
| X-6 | every band frame rendered | 30 steps per root | the readouts read only these; storage about 2 MB per root instead of about 20 GB for the band |
