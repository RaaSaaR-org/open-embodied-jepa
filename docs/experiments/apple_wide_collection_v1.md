# Apple wide-jitter TRAIN corpus v1: preregistration (TASK-048)

This is the prospective protocol for roadmap step T2. It is committed, and
reviewed, before any seed in the frozen range 48000–48199 is simulated.

**What it is.** A training corpus for world model v2 (the LeWM and native
backends, TASK-050). It is produced by the **privileged scripted collector**
(`scripted.apple_collector_policy`, which reads simulator truth at reset),
with seeded action perturbations and grasp-phase branches. Nothing in this
corpus, and no number in its results, is a learned-control result. Learned
Apple→Plate stays at zero successes.

## Why

- The earlier model was a 24×24 RGB MLP trained on 32 narrow-jitter (±6 mm)
  episodes. At that resolution the grasp is not visible, and at that jitter
  replay already solves the development resets (TASK-043/046).
- TASK-047 established the wide distribution (apple ±3 cm, plate ±2 cm): the
  scripted collector succeeds 8/8 there, open-loop replay 2/8. It is feasible,
  and it separates replay from object-aware control.
- A learned model therefore needs (a) many wide-distribution resets,
  (b) images in which the hand and apple are resolvable, and (c) actions that
  vary around the grasp, including grasps that fail, so that the action
  actually changes the predicted outcome (action sensitivity).

## Seeds, resets and cohorts

| | Seeds | Role |
|---|---|---|
| **This corpus (frozen)** | **48000–48199** (200 resets) | TRAIN corpus with its own train/val/test partition |
| Pilot design probes (this protocol) | 48900–48931 | tuning only, never part of the corpus |
| Earlier TRAIN/VAL/TEST collection | 42000–42031 | unchanged |
| Narrow development | 43000–43004 | unchanged |
| Final cohort | 44000–44019 | unchanged, never simulated here |
| Wide development | 45000–45007 | unchanged, never simulated here |
| Mechanics probes, MVP, pilots | 41000–41101, 20000-range, 300/1000 | unchanged |

- **Reset rule.** The TASK-047 rule, `evaluate_apple.wide_reset(seed)`:
  `rng = default_rng(seed)`, then apple `(0.34, −0.18) + U(±0.03)²`, then plate
  `(0.49, −0.09) + U(±0.02)²`. The collector re-implements it, and a test pins
  equality with `evaluate_apple.wide_reset`.
- **Collision.** No seed in 48000–48199 or 48900–48931 appears anywhere else
  in the repository (checked by search and pinned by a test against the ranges
  above).
- **The corpus's own val/test** are for model selection and held-out
  prediction diagnostics only. The closed-loop cohorts stay 45000–45007
  (development) and 44000–44019 (final).

## Observations (model inputs)

| Stream | Shape | Source |
|---|---|---|
| `onboard_rgb` | 112×112×3 uint8 | the simulator's own render of the torso camera (`MuJoCoSimulation(width=112, height=112)`) |
| `hand_crop_rgb` | 112×112×3 uint8 | a 112-px window cut from a 320-px render of the same camera, centred on the projected right palm |
| `observation.state` | 86 floats | the 43 joint positions and 43 velocities (the unchanged state schema) |
| `action` | 14 floats | the executed (applied) normalised command, `ee_delta_grasp_v0` |

- **Why 112 px.** The LeWM adapter builds a ViT with
  `image_size % patch_size == 0`; with the default patch 14, 112 px gives 8×8
  patches. The native backend interpolates to its own `image_size`. 112 px is
  4.7× the previous 24 px in each direction, and the storage cost is within
  budget (pilot: 530 MB per 128 episodes).
- **Hand crop (`src/embodied_jepa/hand_crop.py`).** The window is placed around
  the pinhole projection of the `right_ee` palm site through the torso camera
  (fovy 75°), then shifted to stay inside the 320-px image. It is therefore not
  always centred: in pilot-c the window was clamped at the right image edge in
  37% of frames (about 64% during lift), although the projected palm always
  stayed inside the crop (columns 192–316). Both are rigidly attached to robot links, so the window depends
  only on the robot's own kinematics: no object, contact or task quantity is
  read. It is an allowed model input.
  - In the 320-px render the window spans about 35% of the image width
    (about 27 cm at the table), and the apple is roughly 20 px across.
  - The window reads the simulator's link-pose buffers (`site_xpos`,
    `cam_xpos`), the same buffers the renderer draws. After `mj_step` these lag
    `qpos` by one substep, so the window is consistent with the pixels but is
    not an exact function of the stored `observation.state`. The per-frame
    window is stored as the `robot__hand_crop_window` label.
  - At evaluation time, the crop must be computed by the same
    `HandCrop.window`/`capture` (TASK-050's responsibility), and `onboard_rgb`
    must be rendered at 112 px, not the evaluator's current 96 px.
  - **Model compatibility.** Both backends read a single `config["camera"]`
    and bilinearly resize it to `config["image_size"]` (`models/base.py`). The
    defaults are 56 (LeWM) and 64 (base), so TASK-050 must set
    `image_size: 112`, or the resolution gain is lost; 112 % 14 == 0 is valid
    for LeWM. Using `hand_crop_rgb` together with `onboard_rgb` needs a
    second-camera path that `VisualModel` does not yet have.

## Collector and perturbations (`scripts/collect_apple_wide.py`)

Every command goes through the evaluator's path: the right-arm bounds
(±0.5; left arm 0; left grasp −1), then the mandatory `project_candidates`,
then `execute`.

- A velocity-guard refusal during projection is a clean stop (`guard_refused`),
  as in TASK-047.
- The applied action is what is stored.
- The collector's base action, the requested (perturbed) action and the
  projected command are stored as collector labels.

**Root episodes (200, one per reset).**

- The unchanged collector policy (the TASK-047 `scripted_oracle`) runs its
  745-command phase budget, stopping at success.
- **Noise level** cycles 0, 1, 2, 3 over the seed order (50 roots each).
  Level 0 is the unperturbed script.
- **Level ℓ ≥ 1 adds** Ornstein–Uhlenbeck noise (θ = 0.85, stationary σ = ℓ ×
  0.04 on the right-arm translation, ℓ × 0.025 on the rotation, ℓ × 0.1 on the
  right grasp). It also adds random bursts: with probability 0.01ℓ per
  command, 4 commands of uniform ±0.5 right-arm deltas.
- **Aim offset.** Every fifth root (40) shifts the orient/descend/close/lift
  targets by a random 1.5–3.0 cm in xy, which produces misaligned grasps.

**Branches (3 per root, 600 planned).**

- Each root is snapshotted at the first command of the policy's `close`
  phase, the pre-grasp state (command 210).
- Each branch restores that snapshot and runs `close` + `lift`, then stops.
- Every branch draws its own perturbation stream at level max(1, root level),
  and adds one modification. The kinds cycle, 120 each. Slot
  `(3·index + b + index // 5) mod 5` assigns them, so the 40 aim-offset roots
  receive every kind equally (24 each; review R1, B2):

| Kind | Modification |
|---|---|
| `noise_only` | none |
| `shift_close` | close/lift targets shifted 1.0–2.5 cm in xy |
| `weak_close` | right grasp target during close/lift ∈ U(−0.2, 0.8) instead of 1 |
| `early_lift` | close shortened to 3–30 commands (of 45) |
| `open_during_lift` | hand opens after 2–40 lift commands |

- **Restoration exactness.** Before branching, the worker checks two things:
  the restored state reproduces the root's sensors at the branch frame, and
  re-executing the root's own next command reproduces the root's next frame
  and applied action. Each branch's first frame must equal the root's.

## Outcome labels and the privileged-label boundary

**Outcome labels.** For every stored episode the collector records:

- the unchanged `AppleToPlateTask` stage flags (reach, grasp = contact lift of
  at least 5 cm, transport, place, success);
- `failure_stage` (the first stage not reached);
- `grasp_phase_attempted` (the episode reached the pre-grasp state, where the
  first `close` command was due; every branch does);
- `grasp_phase_failure` (attempted, but the grasp stage was not reached);
- `dropped_after_grasp`;
- the termination.

These go into the episode metadata under `privileged_outcome_labels`: they are
scorer outputs derived from simulator truth.

The same holds for `termination`, and for the LeRobot `next.terminated` flag,
which is true only for successful roots. These fields are **not** covered by
the sidecar gate. They are privileged: they must not be used as model inputs,
as loss weights, or to filter episodes for model selection. The canonical
`SequenceBatch.terminated` marks only episode boundaries, so current training
code does not see them.

**Label sidecars (`src/embodied_jepa/training_labels.py`).** Per-step labels
live in `<dataset>/labels/<episode_id>.npz`. They are not part of
`Episode`/`SequenceBatch`, and each file's SHA-256 is recorded in the
episode's metadata, which the dataset manifest hashes. Keys are grouped by
prefix:

| Group | Keys | Who may read |
|---|---|---|
| `robot__` | `hand_crop_window` [T+1, 2] | model-side metadata (robot kinematics only) |
| `collector__` | `phase_index`, `base_action`, `requested_action`, `projected_action`, `perturbation` [T] | training/analysis only (the collector read truth at reset) |
| `privileged__` | `apple_position_world`, `apple_velocity_world`, `plate_position_world`, `palm_position_world`, `palm_minus_apple_world`, `hand_contact`, `apple_dropped`, `stages` [T+1] | training-time auxiliary targets or analysis only |

**Enforcement.**

- `training_labels.load` refuses the `collector` and `privileged` groups unless
  it is called with `acknowledge_privileged_training_labels=True`.
- It rejects unprefixed keys, a hash mismatch and path escapes.
- A test asserts that no package module other than `training_labels.py`, and
  none of the evaluation and training scripts, imports it.

Privileged labels must never be a model input, a planner input or a planning
cost.

## Splits and normalisation

- **Partition.** The partition is whole-reset (session `wide-reset-<seed>`)
  and fixed in the plan before collection:
  `default_rng(48).shuffle(seeds)`, then the first 20 are val, the next 10
  test and the remaining 170 train.
- **Branches.** A branch shares its root's session, and therefore its split.
- **Sealing.** The corpus is sealed with `freeze_split_assignments` and no
  held-out pair (every episode is apple/plate). Normalisation is fitted on
  train only.
- **Val sessions:** 48005, 48030, 48039, 48040, 48052, 48054, 48071, 48072,
  48074, 48086, 48096, 48103, 48113, 48117, 48119, 48137, 48168, 48169, 48180,
  48193.
- **Test sessions:** 48007, 48041, 48066, 48067, 48073, 48082, 48104, 48127,
  48154, 48155.
- **Frozen plan.** `collect_apple_wide.py plan --seeds frozen` writes a
  byte-identical plan with SHA-256
  `15ed1a99e45114a5cec6013d345804ec561fad859dc3f0dd89dd93ec1e33062c`.

## Acceptance checks (decided now; `collection_report.json["verdict"]`)

The thresholds were set from pilot rates (below) with margin; they are frozen.

**Threshold history (review R1, B3).** pilot-c was scored with provisional
values: A3 ≥ 60, A5 ≥ 200, A7 ≥ 0.40, and A4 ≥ 0.20 with no upper bound. After
pilot-c they were tightened to the frozen values below.

- The tightening follows the pilot rates scaled to 200 roots and about 800
  grasp-phase episodes.
- The expected values are A3 ≈ 125 (62.5% of 200) and A5 ≈ 390 (62/128 × 800).
- The frozen thresholds therefore keep roughly a 35% margin.
- A4 gained an upper bound, so that a corpus of nearly all failures also fails
  the check.
"Grasp-phase attempts" are the stored episodes with `grasp_phase_attempted`.

| Check | Condition | Pilot-c value |
|---|---|---|
| A1 | stored root episodes ≥ 190 (of 200) | 32/32 |
| A2 | stored branch episodes ≥ 450 (of 600) | 96/96 |
| A3 | root full-task successes ≥ 80 | 20/32 (62.5%) |
| A4 | grasp-phase failure fraction ∈ [0.20, 0.80] | 0.516 |
| A5 | grasp-phase successes (grasp stage reached) ≥ 250 | 62/128 |
| A6 | each right-arm action dim 6–11: ≥ 3% of transitions > +0.1 **and** ≥ 3% < −0.1 | min 0.060 |
| A7 | off-script fraction (max \|applied − base\| > 0.02) ≥ 0.50 | 0.876 |
| A8 | sibling-branch pairs (same root, both ≥ 16 commands) with onboard RGB RMS ≥ 1/255 at step 16: ≥ 90% | 78/78 |
| A9 | no split leakage: no session spans splits, every session is in its planned split, normalisation episodes = train, and train/val/test are nonempty | pass |
| A10 | `embodied_jepa.audit.audit_dataset` succeeds on the sealed corpus, covers every episode, and every frame interval is 0.05 s | pass |
| A11 | every sidecar loads with a matching hash and correct row counts; privileged access without acknowledgement is refused | pass |
| A12 | both cameras stored at 112×112×3 | pass |
| A13 | run integrity: all 200 roots completed without runtime error, every branched root restored exactly, no supervisor timeout, all workers exit 0 | pass |

**Verdict.**

- **Pass.** `all_passed` is true only when every check holds. The corpus is
  then accepted as the TASK-050 training corpus.
- **Fail.** A failed check is recorded as a failure in the results. The corpus
  is kept as evidence and is not silently repaired; any re-collection uses a
  new seed range and a new version.
- **What A7 and A8 show.** Both are close to true by construction. A7 is
  essentially the fraction of transitions at noise level ≥ 1: in pilot-c,
  0.02 at level 0 and 0.99–1.0 at levels 1–3. A8 compares siblings with
  independent noise seeds. They verify that the perturbations were injected
  and are visible. They are **not** evidence of a model's action sensitivity:
  the collapse and action-sensitivity diagnostics belong to TASK-050.
- **Strictness of A13.** A single runtime error, for example a MuJoCo
  instability in any of the 800 episodes, fails A13 and hence the corpus
  verdict (pilot-c had 0 of 128). Episodes that such a root had already saved
  are still assembled and reported.
- **Descriptive only.** The per-kind and per-noise-level outcome tables, and
  the action coverage of the left arm (by design, never excited, because the
  planner fixes it), are reported but never gate.

## Pilot design probes (disclosed; not evidence, not in the corpus)

All pilots used seeds 48900–48915 or 48900–48931, 12 or fewer workers, and
scratch outputs under `outputs/task048-scratch/`.

- **smoke-a (2 roots).** Pipeline check only.
  - Split sealing correctly refused 2 sessions, because test and val took both.
  - Worker outputs were valid.
- **pilot-a (16 roots, 2 branches, σ = 0.06/0.04/0.15 per level, bursts 0.02ℓ).**
  - 8/16 root successes; grasp-phase failure fraction 0.63; 11 guard stops.
  - One sibling pair was RGB-identical at step 16. Branches then shared the
    root's noise stream, and `early_lift` versus `open_during_lift` had
    identical 16-command prefixes.
  - **Fix:** every branch gets its own noise seed (`reseed_noise` became
    `noise_only`).
- **pilot-b (16 roots, 3 branches, the same noise).**
  - 5/16 root successes; grasp-phase failure fraction 0.70.
  - Levels 2–3 grasped 1/8 roots.
  - `weak_close` and `early_lift` never grasped (0/19). The failures were too
    one-sided to show a success/failure boundary.
  - **Changes:** σ reduced to 0.04/0.025/0.1 per level; bursts to 0.01ℓ;
    `weak_close` range from U(−0.4, 0.4) to U(−0.2, 0.8); `early_lift` from
    3–20 to 3–30 commands.
- **pilot-c (32 roots, final design).**
  - Root successes 20/32. By level: 0 → 6/8, 1 → 6/8, 2 → 5/8, 3 → 3/8.
  - Branch grasps: `noise_only` 14/20, `shift_close` 8/19, `weak_close` 2/19,
    `early_lift` 6/19, `open_during_lift` 9/19.
  - Grasp-phase failure fraction 0.52; 27/128 guard stops; 78/78 sibling
    pairs distinct.
  - About 95 s per root with 12 workers; 372 s total for 32 roots including
    assembly and audit; 530 MB.
  - All checks except the count thresholds (A1–A3, A5, which need 200 roots)
    passed.

Only the acceptance thresholds and the integrity check (A13) changed after
pilot-c.

## Budget (frozen)

- **Parallelism.** 12 worker processes on CPU MuJoCo (one thread each), with
  roots assigned round-robin. Each worker is the single writer of its own
  shard directory (`<work>/episodes/<id>/`). The supervisor then assembles all
  shards, in plan order, into one `DatasetStore` as its single writer.
- **Limits.** A worker starts no new root after 4,500 s. The supervisor kills
  the workers at 5,400 s. Any root not started or killed is recorded, and it
  fails A13.
- **Expected time.** About 200 × 95 s / 12 ≈ 27 min of collection, plus about
  10 min for assembly, audit and acceptance. Storage is about 3.5 GB for the
  corpus plus about 3 GB of shards.

## Frozen run command

Execute **once**, from a clean tracked checkout of the reviewed commit, into
directories that do not yet exist, under the main checkout's ignored `data/`:

```sh
.venv/bin/python scripts/collect_apple_wide.py run --seeds frozen \
  --workers 12 --max-seconds 5400 --worker-seconds 4500 \
  --output /Users/sebastian/develop/emai/experiments/JEPA/open-embodied-jepa/data/apple-wide-v1 \
  --work /Users/sebastian/develop/emai/experiments/JEPA/open-embodied-jepa/data/apple-wide-v1-work
```

- **Refusals.** The script refuses a dirty tracked tree and an existing
  output. A change to the source or the tracked inputs during collection fails
  A13.
- **Recovery.** If assembly or scoring raises after collection,
  `collect_apple_wide.py finalize --work <work> --output <new dataset dir>`
  reruns assembly, sealing, audit and acceptance from the existing shards. It
  never simulates, so it does not count as a second run of the frozen seeds.
  A partial `collection_report.json` is always written.
- **Runtime versions.** Provenance also records the Python, platform, NumPy
  and MuJoCo versions and the `uv.lock` SHA-256. It records the git revision, every
  `src/**/*.py` hash, the collector, protocol, action-manifest and
  asset-manifest hashes, and the plan hash.
- **Recording.** Every outcome goes into `apple_wide_collection_results_v1.md`
  and `benchmarks/manifests/apple-wide-collection-v1.json`, including
  failures.

## Limitations (declared now)

- **Oracle actions.** The actions come from an oracle that knew the initial
  object poses. A model trained on them learns dynamics around oracle
  behaviour and its perturbations, not an unbiased action distribution.
- **Unexcited dimensions.** The left arm and left grasp are never excited, so
  the model cannot learn their effect. The planner fixes them.
- **Occlusion.** The torso camera looks down on the back of the right hand, so
  the apple is partly occluded during the grasp. The hand crop increases
  resolution, not the viewpoint.
- **Outcome labels are stage flags** of the frozen scorer. A `grasp` is a
  contact lift of at least 5 cm, not a force-closure measurement.
- **Branch length.** Branches stop after `lift`, so they carry grasp-phase
  outcomes only; transport and place come from root episodes.

## Pre-run review revision R1

The fresh pre-run review found three blocking items. All were fixed before
any 48000-range seed was simulated.

- **B1: the final run path had never executed.** Added the `finalize`
  subcommand and an always-written report. A smoke on pilot seeds (below) then
  ran the exact code to be frozen.
- **B2: branch kinds were confounded with aim-offset roots.** All 40 roots
  received only `weak_close`, `early_lift` and `open_during_lift`. The kind
  slot is now decoupled; a test pins 24 of each kind on aim-offset roots. As a
  result, the frozen plan hash changed from `0046e14c…ba24` to the value
  above; the seeds and splits are unchanged.
- **B3: the pilot-c provisional thresholds were not disclosed.** They are now
  disclosed, with the rationale.

The non-blocking recommendations were also applied:

- runtime versions in provenance;
- tracked-input re-verification;
- accurate hand-crop wording (clamping, substep lag);
- the privileged status of outcome and termination fields;
- the interpretation of A7 and A8;
- the model image-size note;
- a test that pins the plan hash;
- the A13 strictness note;
- `--worker-seconds` default 4500.
