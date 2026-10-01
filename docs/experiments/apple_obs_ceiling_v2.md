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
  within the precision the place primitive tolerates, beats its random-init floor and a prior
  that reads no image, and does not do so without the plate in view. It would license the next
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
- TASK-061 read the post-look apple at reset from raw 112 px onboard pixels to 0.469 cm, from
  224 px to 0.386 cm, and from overview pixels to 0.183 cm
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
- **Seeds.** 55000–55031, development, now spent (§4 of the design, `SEED_RANGES["tau"]`).

**Why this quantity.** The planner's put-down spot is where e9's place primitive is aimed. A
readout error in the plate (or the offset) becomes a target error of the same size in cm. A
constant planted error is the simplest model of that: the target is frozen at 505 from the last
decision's reading, so a readout error at the last decision acts as a constant target error.
A per-frame readout error that varies between decisions is not modelled (§9b).

**Result** (run `tau` at ⟨TBD-tau-revision⟩, report sha256 ⟨TBD-tau-sha⟩; start load ⟨TBD⟩):

⟨TBD-tau-table⟩

**τ = ⟨TBD-tau⟩ cm** (`TAU_MEASURED`, frozen).

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
  folds only; no window's own root enters its fit. A test checks this with poisoned labels.
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
| **A1** precision | c_V := the upper 95 % bound of R_off(V)'s median error ≤ τ | the readout is at least as precise as the place needs, with its uncertainty counted against it |
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

### 6.3 The downstream rule (for the next world-model task)

If the row is OBS-ONBOARD with view V*, the next task's O2-type encoded-readout bar B must satisfy
**c_V\* ≤ B ≤ τ** (`downstream_bar_interval`, `bar_is_valid`), and its predicted-latent bar must
be ≤ τ, with any allowance above c_V\* calibrated on development data before its freeze (not
carried; TASK-066's 0.5 cm G5 margin was uncalibrated, `apple_token_dynamics_v1_results.md`:234). TASK-074's 1.0 cm
bar against a 2.872 cm readout is exactly what `bar_is_valid` refuses (tested). If c_V > τ for
every view, no world-model task is preregistered on these views.

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
| OBS-ONBOARD (V*) | preregister the LeWM place planner or critic on V*, with c_V\* ≤ B ≤ τ (§6.3); a new corpus on V* is that task's own stage |
| OBS-EXTRA | the owner rules on an external camera (PRD.md:208); no onboard world-model task is preregistered meanwhile |
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

- **τ is one run of 32 resets per level, with no interval.** A level's count is binomial; τ can
  move by one level between runs.
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

⟨TBD-smokes⟩

## 11. Compute estimate (measured where marked)

⟨TBD-compute⟩

## 12. Deviations from the design proposal

| # | proposal | here | why |
|---|---|---|---|
| X-1 | a fresh corpus on an unused seed range | reuse `apple-far-shift-v2`; re-render its train and val roots in the new views | the reference view exists; reuse pairs every view with the stored reference and with TASK-074's measured numbers (R4) |
| X-2 | the floor/prior margin "calibrated from the reference view's development spread" | strict superiority (upper bound < 1.0), no margin | a development readout would have shown views' numbers before the freeze (R6) |
| X-3 | readouts on held-back val roots | cross-fitted over 270 roots | c_V's interval on 22 val roots would be too wide to admit anything (R5) |
| X-4 | "a clean apple- and plate-hidden spurious check" | the plate-hidden check gates, the apple-hidden check is reported | the apple is in the hand during the carry (R8) |
| X-5 | a wrist view "only if its scene change passes review" | no wrist view | no scene change is made in this task (R10) |
| X-6 | every band frame rendered | 30 steps per root | the readouts read only these; storage about 2 MB per root instead of about 20 GB for the band |
