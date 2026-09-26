# Apple→Plate look-prefix corpus v1: a 112 px corpus whose episodes start with the look, with the frozen DINOv2 encoder as candidate (TASK-064)

**Status: preregistration. No seed of the corpus range (47000–47199) has been simulated.** The
only simulation so far is a disclosed pilot on the separate pilot range 47900–47931 (§13), with
its readability targets replaced by seeded noise, so it measured nothing about readability.
**The test split is never decoded. Cohort D (45000–45007, 45100–45107) and cohort C
(45300–45339) are never simulated. `exemption_spent`
(`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays `false`.**

This is a **data task**. It builds one corpus and validates it: that the collection went as
designed, and that the pinned frozen encoder reads the apple from the corpus's own post-look
frames on fresh resets. It trains nothing, fits no world model and preregisters no control
formulation. CEM over the world-model cost was abandoned as the primary control line at TASK-054,
and at TASK-057 the behaviour-cloning line stopped on `apple-wide-v1` and the 112 px onboard
camera, with no third control formulation preregistered on them (`docs/DECISIONS.md`). Both
clauses still hold, and nothing here implies a third formulation on this corpus either.
**Learned Apple→Plate is still 0 successes.** The corpus comes from the **privileged scripted
collector**, which reads simulator truth at reset. Its successes are scripted, not learned.

Predecessors:
- [`apple_pretrained_encoder_v1.md`](apple_pretrained_encoder_v1.md) /
  [`_results.md`](apple_pretrained_encoder_v1_results.md) (TASK-063, outcome **O-PT-POOLED**);
- [`apple_observation_reprobe_v1.md`](apple_observation_reprobe_v1.md) /
  [`_results.md`](apple_observation_reprobe_v1_results.md) (TASK-061, outcome **O-LOOK-RAW**;
  it defined the look);
- [`apple_wide_collection_v1.md`](apple_wide_collection_v1.md) /
  [`_results_v1.md`](apple_wide_collection_results_v1.md) (TASK-048, the `apple-wide-v1`
  corpus; this corpus reuses its collector mechanics).

Manifest: `benchmarks/manifests/apple-look-corpus-v1.json`. Code that fixes the design:
`src/embodied_jepa/look_corpus.py` (seeds, plan, splits, look, checks, gate rule, rows), committed
with this document and pinned by hash.

---

## 1. What this task is, and the claim it can make

TASK-063's O-PT-POOLED row pre-declared the next task: **preregister the look-prefix 112 px
corpus, with the pinned frozen DINOv2 ViT-S/14 as the candidate image encoder.** This is that
task.

**Question.** Can the scripted collector produce a 200-reset corpus at the 112 px onboard camera,
every episode starting with TASK-061's constant look, that (a) passes the TASK-048 acceptance
checks carried over unchanged, and (b) on its own train + val post-look frames, from **fresh
resets the look and the encoder were never chosen or validated on**, is readable by the frozen
DINOv2 CLS feature under TASK-063's unchanged probe, beyond the same architecture's random init?

**Claim scope, fixed now.** A pass (C-ACCEPT, §10) says two things:
- *this* corpus was collected as designed;
- *this* pinned encoder's frozen CLS feature makes the post-look apple position and the expert's
  first post-look command readable by kernel ridge on this corpus's 190 train + val roots, beyond
  its random init.

It does **not** say that a world model trained on the corpus predicts the apple, that any
controller works, or anything about closed-loop behaviour. Nothing here chooses a world-model or
control design.

**Why fresh resets.** TASK-061 chose the look on the 190 train + val roots of `apple-wide-v1`
(by visibility, disclosed there), and TASK-063 validated the encoder on the same 190 roots. A
corpus on the same resets would reproduce TASK-063's frames exactly and would test nothing new.
This corpus uses a new reset range (§4.1). Its readability gate is therefore a **replication of
TASK-063's P-cls result on 190 new roots** as well as a check of the corpus. It can fail, and a
failure has a pre-declared consequence (§10).

## 2. What is reused unchanged

| piece | source | how |
|---|---|---|
| reset rule | TASK-047 `evaluate_apple.wide_reset`: apple ±3 cm, plate ±2 cm | re-implemented in `look_corpus.wide_reset`; a test pins equality |
| collector policy | `scripted.apple_collector_policy` (TASK-048's `scripted_oracle`) | unchanged; built from the **reset** truth |
| episode mechanics | `scripts/collect_apple_wide.py`: `Perturber`, `Controller`, `step`, `snapshot`, `restore`, `outcome` | loaded unchanged; its sha256 is pinned (G-hash) |
| perturbations, aim offsets, branches | TASK-048 §"Episodes": OU noise levels 0–3, bursts, every fifth root aim-offset, three branches per root at the pre-grasp state, five branch kinds | the same values; a test pins equality |
| collection bounds | `LOWER`/`UPPER` of TASK-048 | unchanged |
| the look | TASK-061 §3.1: eight identical commands, sequence sha256 `17bfb070…15be587c` | `observation_reprobe.look_sequence()`, unchanged |
| renderer warm-up | TASK-061 §5 (owner ruling 2026-09-25) | every renderer renders once and discards the result before any frame is kept |
| encoder | TASK-063 §4: `facebook/dinov2-small` @ `ed25f3a3`, pinned files and digests | `pretrained_encoder`, unchanged |
| probe | TASK-061's L probe as TASK-063 ran it: readouts, folds, bars, floor rule, p-values, apple-hidden check | `info_ceiling`, `observation_reprobe`, `encoder_study.arm_pvalues`, `probe_info_ceiling.evaluate_source`, unchanged |
| storage | `data.py` (LeRobot v3 profile, PNG-in-Parquet), label sidecars (`training_labels`) | unchanged |

**What changes relative to `apple-wide-v1`, and only this:**
1. every root episode starts with the 8-command look; the collector policy starts at the
   post-look pose;
2. a new reset range and split seed (§4.1, §6);
3. one camera: `onboard_rgb` at 112 px; the hand crop is not stored (§4.4);
4. the renderer warm-up applies to the collector;
5. decoding-based QA reads train + val only (§6);
6. the readability gate (§7).

## 3. Context carried forward (reported, not decisional)

### 3.1 TASK-063's result

Outcome **O-PT-POOLED** (run-1, 190 train + val roots of `apple-wide-v1`, TASK-061's post-look
112 px frame):

| source | T1 median (cm) | T2 dx sign | meets bars |
|---|---|---|---|
| **P-cls** (frozen DINOv2 CLS, 384-d) | 0.538 [0.487, 0.611] | 183/190 | yes, beats R-cls, Holm-rejected |
| P-tok (frozen tokens, 98 304-d) | 0.394 [0.363, 0.447] | 187/190 | yes, beats R-tok, Holm-rejected |
| R-cls (seed-0 random init, CLS) | 0.853 [0.771, 0.990] | 169/190 | **yes** |
| R-tok (seed-0 random init, tokens) | 0.500 [0.457, 0.572] | 181/190 | **yes** |
| L-raw (raw pixels) | 0.469 [0.419, 0.528] | 180/190 | yes |

P-cls − R-cls: −0.316 cm [−0.456, −0.214], T2 183 against 169.

### 3.2 Caveats carried forward, unchanged

They all apply to this task and to anything built on its corpus.
- **P-cls is not detectably different from raw pixels.** P-cls − L-raw was +0.068 cm
  [−0.010, 0.153]. No equivalence margin was preregistered, so this is not a claim of
  equivalence either way.
- **Both random-init floors met the bars on their own**, the pooled R-cls included. The pass is
  "pretraining makes the pooled feature read the apple better than this architecture does
  untrained", not "only a pretrained feature reads it".
- **One encoder, one input size, one floor seed.** Floor-seed variance is not measured. This task
  keeps the same single floor seed (0), so it does not measure it either.
- **The cause is not identified.** TASK-063 changed the pretraining data, the width (384 against
  128) and the input size (224 against 112 px) at once, and does not show which mattered.
- **Readability is not prediction.** Nothing shows that a world model on this encoder keeps the
  apple through prediction, or that a controller could use it.

## 4. Corpus design (fixed now)

### 4.1 Seeds, roots and cohorts

| | seeds | role |
|---|---|---|
| **This corpus (frozen)** | **47000–47199** (200 resets) | the corpus, with its own train/val/test |
| Pilot (disclosed, §13) | 47900–47931 (32 resets) | runtime and rate check only; never part of the corpus |
| `apple-wide-v1` and its pilots | 48000–48199, 48900–48931 | unchanged; never simulated here |
| Cohort D (wide development) | 45000–45007, 45100–45107 | never simulated |
| Cohort C | 45300–45339 | never simulated |
| Final cohort | 44000–44019 | never simulated |
| TASK-051 wide v3 cohort | 45200–45207 | never simulated |
| Earlier ranges | 41000–41101, 42000–42031, 43000–43004, 49000–49199, 20000–20049 | never simulated here |

- **No collision.** No seed in 47000–47199 or 47900–47931 is used anywhere else in the
  repository (checked by search). `look_corpus.check_seeds` refuses every range in the table
  above (`RESERVED_RANGES`), and `tests/test_look_corpus.py` pins the refusal.
- **Reset rule.** TASK-047's, per seed: `rng = default_rng(seed)`, apple
  `(0.34, −0.18) + U(±0.03)²`, then plate `(0.49, −0.09) + U(±0.02)²`.
- **The corpus's own val/test** are for future model selection and held-out diagnostics only.
  Closed-loop cohorts, if any task ever needs one, remain the reserved cohorts above, and none is
  opened here.

### 4.2 Episodes

**Root episodes (200, one per reset).**
1. Reset; the frame at index 0 is the reset frame.
2. **The look prefix.** TASK-061's eight constant commands (left arm 0; right arm
   `(0, −0.4, 0.4, −0.5, −0.5, 0.5)`; both grasps −1), **unperturbed on every root**, each through
   the collector's own step (clip, project, execute, record). They are stored as ordinary
   transitions, with `collector__phase_index` = −1. **Frame index 8 is the post-look decision
   frame** on every root episode.
3. **The collector policy** is built from the **reset** truth (as TASK-061's post-look expert
   is) and runs from the post-look pose for its full 745-command budget, stopping at success. The
   look does not count against the budget.
4. **Perturbation** (TASK-048, unchanged, applied only after the look): noise level cycles 0, 1,
   2, 3 over the seed order (50 roots each); level ℓ ≥ 1 adds Ornstein–Uhlenbeck noise
   (θ = 0.85, stationary σ = ℓ × 0.04 on the right-arm translation, ℓ × 0.025 on the rotation,
   ℓ × 0.1 on the right grasp) and bursts (probability 0.01ℓ per command, 4 commands of uniform
   ±0.5 right-arm deltas).
5. **Aim offset.** Every fifth root (40) shifts the orient/descend/close/lift targets by
   1.5–3.0 cm in xy.

**Branches (3 per root, 600 planned).** As TASK-048: snapshot at the first command of the
policy's `close` phase (the pre-grasp state); each branch restores it and runs `close` + `lift`
with its own perturbation stream (level max(1, root level)) and one modification. The five kinds
(`noise_only`, `shift_close`, `weak_close`, `early_lift`, `open_during_lift`) are assigned by the
slot `(3·index + b + index // 5) mod 5`. Branches start after the look, so they carry no look
prefix; their first frame is the root's pre-grasp frame. Restoration exactness is checked as in
TASK-048.

**Stored.** Every episode with at least 8 transitions (TASK-048's rule). At most 800 episodes.

**Per-root draws** (aim offset, branch parameters, noise seeds): `default_rng(SeedSequence([seed,
64]))`, in TASK-048's order.

### 4.3 What the look is, and what it is not

- It is the same constant on every reset; `look_sequence()` takes no argument. The collector's
  `LookController` returns only rows of that constant; PR 2's tests must prove that two different
  resets issue identical look streams (§9).
- TASK-061 established, on the `apple-wide-v1` roots, that after the look the robot state is
  identical across roots, the apple and plate do not move, and the hand does not touch the apple.
  This corpus **re-checks all of it on every one of its 200 roots at collection time** (check L1,
  §6.2), and the readability gate re-checks the frames (§8).
- The look is an **observation prefix**, not a control component. Nothing here says a controller
  should or will issue it.

### 4.4 Observations (the model-input streams)

| stream | shape | source |
|---|---|---|
| `onboard_rgb` | 112 × 112 × 3 uint8 | the simulator's own render of the torso camera (`MuJoCoSimulation(width=112, height=112)`) |
| `observation.state` | 86 floats | the 43 joint positions and 43 velocities (unchanged schema) |
| `action` | 14 floats | the executed (applied) normalised command, `ee_delta_grasp_v0` |

- **112 px** is the resolution of TASK-061's L arm and TASK-063's probe. No other resolution is
  stored.
- **No hand crop.** `apple-wide-v1` also stored a 112 px window of a 320 px render. It is dropped
  here: the candidate encoder's validated input is the onboard frame, a second camera made readout
  precision worse at world model v3, and a 320 px render is subject to the first-render artifact that TASK-061
  found at 224 px. A future hand-crop corpus would need its own preregistration.
- **Renderer warm-up.** The simulation's renderer is created and renders once, discarded, before
  the reset (TASK-061's `make_robot` rule). TASK-061 found no first-render artifact at 112 px
  onboard (0/190), so this is a precaution; the readability gate verifies the stored frames
  against fresh re-renders (§8).
- **Simulator truth goes only into label sidecars** (TASK-048's `training_labels` groups: no
  `robot__` key, since there is no crop window; `collector__` and `privileged__` as before). The
  privileged-label boundary of TASK-048 applies unchanged: `privileged_outcome_labels`,
  `termination` and LeRobot `next.terminated` must never be model inputs, loss weights or
  selection filters.

### 4.5 Episode counts, expected

From the pilot (§13) scaled to 200 roots: about 130 root successes, about 790 stored episodes,
about 210 000 transitions and about 2.2 GB. These expectations are not thresholds.

## 5. The candidate image encoder, and how it applies to 112 px frames

**The encoder is a candidate, not a component of this corpus.** The corpus stores **pixels**,
never features. Any encoder can later be applied to it, and choosing this one as a world model's
image encoder needs its own preregistration (§10).

**Input handling, unchanged from TASK-063 §5.1** (`pretrained_encoder.preprocess`, pinned by
hash):
1. the stored uint8 112 × 112 RGB frame, decoded from the corpus's PNG;
2. scaled to [0, 1];
3. resized to **224 × 224 by bicubic interpolation** (`torch.nn.functional.interpolate`,
   `align_corners=False`, `antialias=False`), **no crop**;
4. normalised with the ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

Patch 14 gives a 16 × 16 token grid. The position embeddings are interpolated from the 37 × 37
training grid by transformers' own bicubic rule. The HF image processor is not used, because its
centre crop cuts the borders. CPU, float32, eager attention, `eval()`, `no_grad`, batches of 16;
features cast to float64 for the readouts.

**Read-out point: the CLS token after the final LayerNorm (`pooler_output`, 384-d)**, P-cls,
the arm the O-PT-POOLED row named. The 256 final patch tokens (P-tok) and their mean (P-mean) are
reported only.

**Consistency with TASK-063.** The frames entering the encoder are the corpus's stored frame 8,
which the gate checks is byte-identical to a fresh re-render of the post-look state through the
simulation's own 112 px path, the path TASK-061's `post_112` frames came from. The encoder
module, its preprocessing and its read-out points are the same bytes as TASK-063's (G-hash). The
only differences are the resets and the path the frame takes: the corpus's PNG, which is lossless
(TASK-061's P2 check held on 190/190).

**No fine-tuning, no training, no feature cache.** The weights are frozen and never updated.

## 6. Splits, and what may be decoded

### 6.1 The partition

- **Whole-reset (session) partition, fixed in the plan before collection:**
  `default_rng(64).shuffle(seeds)`, the first 20 val, the next 10 test, the remaining 170 train.
  Session id `look-reset-<seed>`; branches share their root's session and split.
- **Val sessions:** 47027, 47030, 47041, 47058, 47059, 47060, 47068, 47076, 47082, 47083, 47086,
  47102, 47103, 47117, 47146, 47166, 47171, 47182, 47186, 47195.
- **Test sessions:** 47007, 47009, 47042, 47062, 47088, 47100, 47124, 47162, 47181, 47184.
- **Sealing.** `freeze_split_assignments`, no held-out pair; normalisation is fitted on train
  only.
- **Frozen plan.** `make_plan(FROZEN_SEEDS)` writes a plan whose sha256 is
  `99dd7ffc305cceab4d4e1eb1438a47b8f5a469f1fab0ca3493a4e689af32a71b` on macOS arm64. The
  platform-robust hash (every float rounded to 1e-9, TASK-048's rule, because libm's sin/cos
  differ in the last ulp across platforms) is
  `37ad7a31e05efac99e3225fe1ed5a69fdd57d790cb2d458cddb1e896a1240f30`. The recorded `plan.json`
  of the run is authoritative.

### 6.2 The test split is never decoded

"Decoded" means reading any frame, state, action, timestamp or label-sidecar row of an episode.
- **Test episodes are written once, at assembly, and never decoded afterwards.** Their files are
  checked by sha256 only (`DatasetStore.verify`, the sidecar hash against the episode metadata).
- **Every fact the checks need on all splits is recorded by the collector at collection time**,
  in memory, before storage: the look records (L1), frame intervals, frame shapes, terminations
  and outcome labels.
- **Manifest metadata is read for counts on all splits** (lengths, terminations, the privileged
  outcome flags). That is not decoding, and it is how TASK-048 counted.
- **Decoding-based QA reads train + val only:** A6–A8, the A10 audit, the A11 sidecar loads and
  the readability gate. A reader that refuses any other episode id enforces it, and PR 2's tests
  must pin the refusal (§9).
- **For any later use of the corpus:** train on train, select on val, never decode test during
  training or selection.

## 7. Acceptance checks (fixed now; `collection_report.json["verdict"]`)

The thresholds are **TASK-048's frozen values, unchanged**. The pilot (§13) found rates close to
TASK-048's pilot-c, so there was no reason to move them; they were not tuned on it.

| check | condition | TASK-048 pilot-c | this pilot (32 roots) |
|---|---|---|---|
| A1 | stored root episodes ≥ 190 (of 200) | 32/32 | 32/32 |
| A2 | stored branch episodes ≥ 450 (of 600) | 96/96 | 95/96 |
| A3 | root full-task successes ≥ 80 | 20/32 | 21/32 |
| A4 | grasp-phase failure fraction ∈ [0.20, 0.80] | 0.516 | 0.488 |
| A5 | grasp-phase successes ≥ 250 | 62/128 | 65/127 |
| A6 | each right-arm action dim 6–11: ≥ 3 % of transitions > +0.1 **and** ≥ 3 % < −0.1 (train + val) | min 0.060 | min 0.072 / 0.089 |
| A7 | off-script fraction (max \|applied − base\| > 0.02) ≥ 0.50 (train + val) | 0.876 | 0.865 |
| A8 | sibling-branch pairs (same root, both ≥ 16 commands) with onboard RGB RMS ≥ 1/255 at step 16: ≥ 90 % (train + val) | 78/78 | 78/78 |
| A9 | no split leakage: no session spans splits, every session in its planned split, normalisation episodes = train, train/val/test nonempty | pass | pass |
| A10 | audit: every train + val episode decodes, every decoded frame interval is 0.05 s, and every stored episode's collection-time intervals are 0.05 s | pass | pass |
| A11 | train + val sidecars load with matching hash and row counts, and their first 8 phase labels are the look; test sidecars match their recorded sha256; privileged access without acknowledgement is refused | pass | pass |
| A12 | `onboard_rgb` stored at 112 × 112 × 3, and no other camera | pass | pass |
| A13 | run integrity: all 200 roots completed without a runtime error; every branched root restored exactly | pass | pass |
| **L1** | **the look, on every root, at collection time:** all 8 commands executed; applied = requested exactly (max \|Δ\| = 0); post-look joint state spread across roots ≤ 1e-6; apple xy move ≤ 1e-6 m and plate move ≤ 1e-6 m after every look command; no hand–apple contact during the look | — | pass (spread 0.0; apple xy 1.5e-15 m) |

- `data_all_passed` is true only when every check holds.
- A1–A5 "pilot" values are counts on 32 roots; the thresholds are for 200.
- **What A7 and A8 show** (TASK-048's reading, unchanged): the perturbations were injected and
  are visible. They are not evidence of a model's action sensitivity.
- **Descriptive only, never gating:** outcome tables by noise level, branch kind, split and
  failure stage; left-arm coverage (never excited, by design).

## 8. The readability gate (fixed now)

On the **190 train + val roots** (170 train, 20 val) of the sealed corpus. **The test split is
not read.**

**Frames and targets.**
- **The decision frame** is the stored frame 8 of each root episode, decoded from the corpus.
- **Fresh re-render.** Each root is re-simulated from its reset through the look in a fresh
  simulation (with warm-up). This gives:
  - the render-path guard (Q-render, §9);
  - the reset visibility flag for B-occ (the apple's 112 px segmentation pixels at reset,
    TASK-059's flag);
  - the apple-hidden twin of the post-look frame for the spurious check;
  - the targets.
- **Targets (TASK-061 §4, unchanged):**
  - T1 is the reset apple xy (simulator truth at reset);
  - T2 and T3 are the sign and value of the expert's `right_dx` at the post-look state:
    `apple_collector_policy(reset truth)` built fresh and evaluated at the post-look pose.
- **Q-expert (§9)** checks that this recomputed command equals the recorded first policy base
  action (`collector__base_action[8]`) on every root without an aim offset.

**Readouts, folds, statistics, bars (unchanged).**
- Linear and RBF kernel ridge, nested inner CV (`info_ceiling`, λ grid 1e-4…1e4).
- Grouped 10-fold CV, `default_rng(59)`, over the 190 roots sorted by seed. The fold hash is
  `da6b5b5a9d0a621db316b94b4e7c246137da96caa94e1ff9994dd0fce4d5486b` (pinned; Q-folds).
- 10 000 paired bootstrap resamples (seed 5901), Wilson intervals.
- In-run baselines B-occ, B-maj and B-const on these roots. They are computed fresh; this is a
  new cohort, so nothing is compared with an earlier baseline.
- The TASK-059 bars, on all 190 roots:
  - T1: median ≤ 1.5 cm **and** the upper 95 % bound of the ratio to B-occ ≤ 0.6;
  - T2: accuracy ≥ 0.85 **and** the Wilson lower bound > B-maj;
  - T3: the upper bound of the MAE ratio to B-const ≤ 0.6.

**The gate passes only if P-cls meets all of these:**
- **R1** it succeeds on all 190 roots under the bars (`info_ceiling.evaluate`);
- **R2** it beats its floor R-cls, the seed-0 random init of the same architecture read at the
  same point (`info_ceiling.beats_random_floor`: the paired 95 % CI of the median T1 difference
  lies below 0, **and** the T2 accuracy is higher);
- **R3** p_arm ≤ 0.025, one-sided. p_arm is TASK-062's `encoder_study.arm_pvalues`:
  max(p_T1, p_T2, p_T3, p_F), or 1 if a point condition fails. There is one decisional
  hypothesis, so there is no Holm family;
- **R4** it is not spurious: the same out-of-fold readouts, applied to the apple-hidden frames
  featurised by the same encoder, do not beat the prior on all roots, and the ablation renderer
  reproduces the un-ablated frame on 190/190.

**Reported only** (none decides anything):
- P-tok, R-tok, P-mean, R-mean and L-raw (raw pixels) on every stratum;
- the secondary train → val estimate;
- paired comparisons P-cls − R-cls, P-cls − L-raw, P-tok − R-tok and P-tok − L-raw;
- visibility facts (reset-occluded roots, post-look apple pixels);
- the prior summary;
- TASK-063's P-cls, side by side.

**Stated in advance.**
- **R2's T2 condition is strict.** R-cls reached 169/190 on TASK-063's roots. If it does better
  here, P-cls needs at least one root more.
- **A failure is a live possibility.** TASK-063's roots were the roots the look was chosen on,
  and these are not.
- A pass would not remove any caveat of §3.2.

## 9. Guards: any failure voids the run (outcome V)

| guard | condition |
|---|---|
| **G-hash** | Every file in the manifest's `hashes` matches before collection starts, and the tracked tree is clean. The collector re-checks every `src/**/*.py` hash and its tracked inputs at finalize; a change during the run is V. |
| **G-plan** | The recorded `plan.json` hashes to the pinned value (exact on macOS arm64; rounded everywhere) and is unchanged at finalize. |
| **G-stop** | The collection supervisor's wall cap is not reached, every worker exits 0, and no root is left unstarted by the 4500 s worker cap (an unstarted root is an early stop). |
| **Q-split** | The readability gate reads exactly the planned 170 train + 20 val roots, each in its planned split; any other episode id is refused. |
| **Q-folds** | The fold hash equals the pinned value. |
| **Q-render** | On every read root, the stored frame 0 equals a fresh re-render of the reset, the stored frame 8 equals a fresh re-render after the look, and an immediate second render of the post-look state equals the first. |
| **Q-expert** | The recomputed post-look expert command equals `collector__base_action[8]` within 1e-6 on every non-aim read root; every read root's first 8 phase labels are the look; the stored apple label at frame 0 equals the reset truth within 1e-6 m. |
| **G-weights** | TASK-063's pinned DINOv2 files and both weight digests (pretrained `3a697b87…2af27`, seed-0 floor `3d305f9c…7c9db`). |
| **G-repro** | A second forward pass of the pretrained encoder on the post-look frames is bit-identical. |
| **G-finite** | No non-finite feature (a mechanical failure, not a readability result). |
| **G-wall** | The whole run (collection, assembly, checks, gate) stays under the global wall cap (§11). |

**What a failed root does, fixed now.**
- **A planned train or val root that is not stored** (a root-level runtime error, a look stopped
  before 8 transitions, or a root never started) leaves the gate without its preregistered 190
  roots. Q-split fails and **the run is V**. A never-started root is already V under G-stop.
- **A root-level runtime error on a test root** does not stop the run (Q-split never reads test
  roots): A13 fails, and A1 too if the root is not stored, so the row is C-DATA-FAIL.
- **A failed look on a stored root** (for example contact, or a moved apple) fails L1. That is a
  data outcome, not V. Because a failed look also confounds the readability gate, C-READ-FAIL
  requires L1 to hold (§10).
- Any other exception is a crash, and a crash is V.

**PR 2's tests** must exercise, in both directions: every guard; every acceptance check and L1 on
planted records; the readability rule R1–R4 through `readability_passes`; every outcome row; that
the look is reset-independent (two resets issue identical look streams); that the QA reader refuses
a test episode; that a crash writes `collection_report.json` with outcome V and a `void_reason`;
non-finite values written as `null` and listed in `non_finite_fields`.

## 10. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§9), or the run stops before writing a complete report (crash, a cap) | nothing is read | one from-scratch repeat (§12) |
| **C-ACCEPT** | every acceptance check (A1–A13, L1) passes **and** the readability gate passes | The look-prefix 112 px corpus was collected as designed, and on its own train + val post-look frames from fresh resets, the frozen DINOv2 CLS reads the apple beyond its random init. The corpus is accepted as `apple-look-v1`. | A **separately preregistered** world-model task on `apple-look-v1` with this frozen encoder as a candidate image encoder. It needs its own hypothesis, controls, collapse and action-sensitivity diagnostics, and its own gates. **No control formulation is implied.** |
| **C-READ-FAIL** | the readability gate fails **and** L1 holds (whatever the other acceptance checks say) | TASK-063's P-cls result does not replicate on this corpus's fresh resets under the unchanged probe. | **The abandonment clause fires** (below). |
| **C-DATA-FAIL** | otherwise: at least one acceptance check fails (with the gate passing, or with the gate failing while L1 fails) | The corpus is not as designed. If the gate failed too, that failure is reported as **confounded by the failed look** and is not read as a readability result. | The corpus is kept as evidence and **not accepted**, and it is not silently repaired. Any re-collection uses a **new seed range and a new version**, preregistered separately. The abandonment clause does not fire. |

**INCONCLUSIVE (task-level, not a row of a run).** A second V closes TASK-064 as INCONCLUSIVE
(§12). Nothing is read, the abandonment clause does not fire, and the owner decides what follows.

**Abandonment clause (C-READ-FAIL).**
- **What closes.** The frozen-DINOv2 route to a look-prefix corpus: no world model is
  preregistered on this corpus with this encoder, and no further pretrained encoder, read-out
  point, input size or look variant is preregistered **on this corpus** without new evidence of a
  different kind.
- **The corpus is kept, sealed, as evidence.** It is not deleted or repaired, and the owner
  decides whether it has any other use.
- **What does not close:** the LeWM backend, the encoder as a component, and the product goal.
- **Context, recorded here and not preregistered as a follow-up.** The remaining route on record
  is the one TASK-061 and TASK-063 recorded: the fixed `overview` camera (TASK-061 O-raw:
  0.183 cm [0.168, 0.207], 185/190). That is a **hardware/workspace change on the robot**, not a
  model change. It is the owner's decision and would need its own preregistration.

**Always reported, whatever the row:**
- every acceptance quantity and check;
- every readability number, with its CIs, floor comparison, p-values and spurious check;
- every reported-only source and paired comparison;
- the rows that also match further down (a C-READ-FAIL with a failed acceptance check also lists
  C-DATA-FAIL).

**In every outcome:**
- nothing is re-thresholded, re-collected, refitted or re-chosen after the numbers are seen;
- learned Apple→Plate stays at 0 successes, and `exemption_spent` stays `false`;
- the executing agent recommends a next task and does not choose it.

## 11. Budget, device, seeds, recording

- **Device: CPU for everything.** Collection is MuJoCo on CPU. The encoder's forward pass runs on
  CPU in float32, as in TASK-063, so that features are bit-reproducible (G-repro). Nothing is
  trained, so MPS is not used.
- **Workers:** 12 collection workers, one BLAS thread each.
- **Caps:**
  - collection supervisor: **5400 s**; each worker stops starting new roots at **4500 s**
    (TASK-048's values);
  - **global wall cap: 10 800 s** for the whole command (collection, assembly, checks, gate).
- **Expected** (pilot, §13): collection about 15–20 min, assembly and checks about 15 min, gate
  about 5 min. Storage about 2.2 GB for the corpus plus about 2 GB of shards.
- **Seeds:** resets 47000–47199; split 64; per-root draws `SeedSequence([seed, 64])`; folds 59;
  inner 5900 + k; bootstrap 5901; floor `torch.manual_seed(0)`. Nothing else is random.
- **Recorded** (`<work>/collection_report.json` and the dataset's provenance):
  - the code revision, with a clean tracked tree required;
  - every pinned hash, every `src/**/*.py` hash, the plan hash, the dataset manifest hash;
  - runtime versions and the `uv.lock` hash;
  - per-root statuses, times and look records;
  - every acceptance quantity;
  - the readability gate's per-root out-of-fold predictions, selections, p-values, floor
    comparison and spurious check;
  - the weight digests;
  - elapsed times and peak RSS.
- **Output:** dataset `data/apple-look-v1/`, work `data/apple-look-v1-work/`. The collector refuses
  to overwrite either. No prior file under `data/`, `checkpoints/` or `outputs/` is written or
  modified. **The data is never committed**; the results PR commits the manifest and hashes only.

**Frozen run command** (from a clean checkout of the reviewed commit, after the pre-run GO):

```sh
uv run --no-sync python scripts/collect_apple_look.py run --seeds frozen \
    --output data/apple-look-v1 --work data/apple-look-v1-work
```

## 12. Void rule

- A run that stops early for any reason (a guard, a crash, a cap) is **V**, with a
  `void_reason`. Nothing in it is read.
- **Exactly one from-scratch repeat** is allowed, into `data/apple-look-v1-run2` and
  `data/apple-look-v1-run2-work`, with the same seeds, caps, workers and device, and a clean tree.
  A void run's directories are kept as evidence.
- `finalize` (re-assembly from existing shards) is **not** a recovery path for the gated run.
  After a V the repeat re-simulates from scratch.
- **A second void closes TASK-064 as INCONCLUSIVE.**
- A fix between the runs is limited to runner mechanics. It goes through a reviewed PR and a fresh
  pre-run GO, and it is disclosed in the results.
- Any owner ruling is recorded with a UTC timestamp before the run it affects.

## 13. Pilot (disclosed; not evidence, not in the corpus)

Before this freeze, a draft of the PR 2 collector ran once on the pilot seeds 47900–47931 (32
roots, 12 workers), into `outputs/task064-scratch/pilot-a/` (plan sha256 `53bcdf42…2767c`, report
sha256 in the manifest).
- **Its purpose:** check that the collector policy still succeeds from the post-look pose, and
  measure time and storage.
- **The readability gate ran in its smoke mode**, with the targets replaced by seeded noise, so it
  revealed nothing about readability. It checked only the wiring: Q-render, Q-expert (24 non-aim
  roots, max |Δ| 0.0), the ablation renderer (30/30) and G-repro.
- **Facts.**
  - 21/32 root successes; 24/32 root grasps; 127 stored episodes (95 of 96 branches).
  - Grasp-phase failure fraction 0.488.
  - Terminations: 21 success, 78 branch_complete, 19 guard_refused, 9 policy_complete.
  - Look on every root: applied = requested, post-look state spread 0.0, apple xy move
    1.5e-15 m, no contact.
  - Post-look apple visible on 30/30 read roots (6–40 px); 15/30 were hidden at reset.
  - Collection 146 s; whole command 189 s; mean 48 s per root under 12-way contention;
    347 MB for 127 episodes.
- **Thresholds were not changed** in response. The rates are close to TASK-048's pilot-c, so its
  frozen thresholds were kept.
- **The pilot report's `outcome` field reads "C-READ-FAIL".** That value comes from the noise
  targets and the draft code, and it means nothing. PR 2's collector records a pilot or smoke run
  as `"smoke (not a row)"` instead.
- An earlier 4-root smoke (`smoke-a`, readability skipped) checked the pipeline only.
- The pilot draft differs from the collector that PR 2 commits only as PR 2 discloses.

## 14. Not done (declared)

- No encoder is trained or fine-tuned; no world model is trained; no controller is trained or
  evaluated; no closed-loop attempt is run.
- No cohort is opened. The test split is not decoded, and cohorts C and D are never simulated.
- No change is made to `models/`, `policy.py`, `simulation.py`, `scripted.py`, `info_ceiling.py`,
  `observation_reprobe.py`, `encoder_study.py`, `pretrained_encoder.py`, `collect_apple_wide.py`,
  or any earlier corpus or checkpoint.
- No second encoder, input size, read-out point or look is tested. Floor-seed variance is not
  measured.
- The hand crop and the overview camera are not collected.

## 15. Process

1. **PR 1:** this document, the manifest, `look_corpus.py`, its tests and the task card. It merges
   on an independent reviewer's **reported** APPROVE and green CI.
2. **PR 2:** the collector (`scripts/collect_apple_look.py`), the readability gate
   (`scripts/read_apple_look.py`) and the guard tests of §9, and the manifest's pins of both
   scripts. It merges on a reported APPROVE and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3:** the results document, the results manifest (dataset manifest hash and every
   restated number), a summarize script and the recommended next task. A reviewer checks every
   restated number against `collection_report.json` with a script. The card goes to done.
5. **One task, one agent.** A protocol defect found after the freeze is escalated to the task owner
   and fixed only through a disclosed amendment.

## 16. Changes made in PR 2 (disclosed; no seed of 47000–47199 simulated)

PR 2 commits the collector and the readability gate. The protocol's rows, gates, thresholds,
seeds and caps are unchanged. The following changes were made after PR 1 merged, all before any
corpus seed was simulated:

- **Review nits from PR 1, applied:**
  - `look_corpus.decide` refuses an inconsistent input, where L1 failed but every acceptance
    check passed;
  - two wording fixes in §9 (a runtime error on a test root) and §4.1 ("never simulated here").
- **What the committed collector adds to the pilot draft:**
  - the G-hash / G-plan preflight;
  - the G-stop for unstarted roots;
  - `look_ok` passed to the decision;
  - the L1 check against the planned seeds;
  - a non-finite feature is V;
  - pilot and smoke runs record `"smoke (not a row)"`;
  - the QA reader was renamed to `TrainValReader`.
- **A final smoke of the committed code, `pilot-c`, on 47900–47931.** Readability ran in smoke
  mode, with noise targets. Report sha256 `28ae2770…0e57`.
  - The facts match pilot-a: 21/32 root successes, 127 episodes, L1 ok, Q-expert 24 roots with
    max |Δ| 0.0, ablation 30/30, G-repro bit-identical.
  - The outcome is recorded as "smoke (not a row)". The whole command took 189 s.
- **A rendering fact found while comparing the three pilot runs**
  (pilot-a, pilot-b, pilot-c: 32 roots, 33 878 frames each):
  - **Physics, states, actions and timestamps are bit-identical across the runs.**
  - Three stored frames mid-episode differ between some pair of runs:
    - `look-47910` frame 142: 2 px, 1 level;
    - `look-47926` frame 117: 1 px, 1 level;
    - `look-47929-b1-early_lift` frame 35: 1 px, 2 levels.
  - No frame 0 or frame 8 differed.
  - The 12 workers render concurrently, so the offscreen renderer is not strictly
    deterministic at the pixel level, at a rate of a few frames in 10⁵.
  - **Consequences, stated before the run:**
    1. A from-scratch repeat is not guaranteed to reproduce the dataset's bytes or its manifest
       hash. It reproduces physics.
    2. Q-render compares 190 × 2 stored frames with single-process re-renders by exact equality.
       At the observed rate, a spurious one-pixel mismatch has a probability of order 1 %. Such a
       mismatch voids the run (V), and the one repeat applies. **Q-render is not relaxed.** The
       exact-equality rule was preregistered, and TASK-061 and TASK-063 met it on 190/190.
    3. The corpus frames are the frames the simulator rendered. A 1–2-level difference on one pixel
       is not a change of scene content.
