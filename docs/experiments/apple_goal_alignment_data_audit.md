# TASK041 data/goal identifiability audit — read-only

No model fitting, prediction, simulation, source edit, Git/MC change, or TEST image
decode occurred. Read metadata for the sealed corpus; decoded canonical RGB only
for one TRAIN original, one VAL original and one TRAIN branch to check access
cost. Read all TRAIN/VAL state/mask columns without decoding images. Loaded the
checkpoint with weights_only=True solely to inspect configuration/buffer shapes.

## Exact schema and sensor mapping

Corpus: data/apple-branches-v1, manifest SHA256
6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331.
Checkpoint SHA256
0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5.
Canonical g1_dex3_proprio_v0 state has 86 required float32 fields: 43 positions
(rad), followed by 43 velocities (rad/s). All 557 TRAIN/VAL episode state tables
are finite; no state-mask value is false. Stored onboard_rgb is uint8 RGB,
96×96×3, at 20 Hz. Robot state, state mask and float64 observation timestamps
are available at each frame, including terminal observation frames.

| Canonical position index (zero-based) | Exact field | Sensor latent index |
|---:|---|---:|
|22|right_shoulder_pitch_joint.position|1750|
|23|right_shoulder_roll_joint.position|1751|
|24|right_shoulder_yaw_joint.position|1752|
|25|right_elbow_joint.position|1753|
|26|right_wrist_roll_joint.position|1754|
|27|right_wrist_pitch_joint.position|1755|
|28|right_wrist_yaw_joint.position|1756|

Their velocity counterparts are canonical indices 65–71, latent indices
1793–1799. Resolve names against StateSchema rather than hard-coding slices in
new code, and assert exact names/units. The actual checkpoint config has
image_size=24, latent_dim=1814, max_horizon=64: 1,728 RGB-grid features followed
by 86 proprioceptive values. RGB flattening is channel-first after resize;
these are normalized sensor features, not arbitrary learned latent coordinates.
sensor_mean, sensor_scale and delta_scale each have shape[1814]. Position targets
or prediction errors must use the matching frozen mean/scale, not refit VAL
statistics. Raw qpos can be recovered with z*sensor_scale+sensor_mean. The image
only encode_goal path receives no joint positions.

## Split, parent and phase counts

| Split | Original episodes | Branch episodes | Total episodes | Frames | Transitions |
|---|---:|---:|---:|---:|---:|
|TRAIN|26|384|410|19679|19269|
|VAL|3|144|147|3953|3806|

TEST contains three originals and no branches; none was decoded in this audit.

TRAIN branch parents are apple-42000,42001,42004,42005,42008,42009,42012,42013
(all nominal). VAL parents are apple-42007,42019,42031 (all closure_burst).
Each branch parent has six roots × eight alternatives = 48 branches.
TRAIN has 48 roots and VAL 18; every branch has 17 observations/16 actions.
Alternatives are recorded, hold, half_motion, reverse_translation,
reverse_rotation, up, open, closed. Parent episode/session identity must be used
for grouping; branches are not independent reset episodes. TRAIN/VAL parent
identities are disjoint. VAL's closure-burst-only parents make it a variant-shift
evaluation, not a representative sample of all nominal resets.

All original TRAIN and VAL episodes contain orient, descend, close, lift,
transfer and release_high, with at least four distinct pre-action frames in each.
Original phase lengths: orient130, descend80, close45, lift150, transfer60;
release_high ranges36–100 in TRAIN and32–38 in VAL. TRAIN apple-42003 is the
single failed original (closure_burst,596 observations); it also has30
lower_open actions. Retain this failed original. Do not silently filter successes
or resample its phases; record that lower_open is outside the fixed six-phase
sampling design. All three VAL originals succeeded during their collection.

TRAIN roots: orient57, descend162, close225, lift322, transfer427,
release_high475, each with64 branches (eight parents × eight alternatives).
VAL has the same first five roots, each24 branches; release_high is473 for one
parent and476 for two, totaling24 branches. Use metadata.root_frame/root_id,
not guessed offsets. root frame0 is byte-identical across siblings; do not count
that duplication as independent identification evidence.

## Deterministic bounded sampling proposal

For each of the six named phases in each original, take four observed-frame
indices at offsets floor(linspace(0,n_phase_actions-1,4)) within that phase's
pre-action labels, preserving release_high's name. Exclude the terminal frame
from phase sampling unless explicitly labeled; do not silently assign a future
phase. Include every declared original, including failed TRAIN42003.
For each branch take observation frames8 and16 (H8/H16 endpoints), assigned its
registered root phase. Freeze indices and parent identities before metric work.

This gives exactly **1392 TRAIN samples** (26×24 +384×2) and **360 VAL samples**
(3×24 +144×2), before any explicitly reported exact-content deduplication. Do not
silently deduplicate after examining scores. Within a branch parent there are120
samples (24 original +96 endpoints); TRAIN parents without branches have24.
A pooled average therefore overweights the eight branch-rich TRAIN parents.
Average within declared phase/endpoint strata and then equally across parents;
report original-frame and branch-endpoint strata separately. For TRAIN retrieval
error exclude the entire query parent/session from the reference bank, not just
its exact row. For VAL retrieval use only the TRAIN bank.

The proposed VAL arbitrary-goal check is feasible from this metadata: each of18
VAL roots has eight H16 endpoints and four original same-phase goals from each
of the OTHER two VAL parents: 8 goals/root, 144 root-goal comparisons, and1152
candidate-goal distances. VAL goal q is an evaluation label only; it must not
enter the RGB goal encoder, model latent, optimizer or inferred image→q target.
Keep all parents/roots, including noninformative/tied comparisons with explicit
counts. No TEST goal or image is needed.

Canonical DatasetStore.read_episode verifies the episode payload/schema and
fully decodes that chosen episode. Measured CPU-only access: opening and corpus
verification0.263s; TRAIN502-frame episode0.245s on first read; VAL504-frame
original0.080s;17-frame TRAIN branch0.004s. Reading just state/mask columns for
all557 TRAIN/VAL files took1.121s. These are access probes, not a full sampled
prep timing guarantee. Stream one episode at a time, retain only chosen frames,
and discard the rest. The1752 selected raw96×96 RGB frames occupy48,439,296 bytes
(~46.2MiB), much less than materializing all corpus images.

A <=120-second preparation supervisor should readily accommodate this scale on
the measured host, but must stop and report unavailable samples if exceeded.
Do not extend or silently drop slow parents. A <=60-second offline evaluation
is a plausible preregistered cap for144 H16 forecasts and this small reference
bank; actual model timing was not measured here. Reuse each root's forecast for
its eight goals. Chunk nearest-neighbor matrix products/queries instead of
allocating a1392×360×1728 broadcast difference tensor (~3.46GB float32).
A1392×360 distance matrix itself is only~2MB. Freeze tie handling, preprocessing,
normalization and sample indices before looking at ranking outcomes.

## Identifiability and leakage risks

1. Same-parent/time-neighbor retrieval can make image→joint estimates appear
   accurate through demonstration replay. Exclude full parent/session for TRAIN
   self-evaluation; VAL bank is TRAIN only. Phase may select audit strata/goals,
   but must not select the runtime nearest-neighbor answer using unavailable
   phase metadata. Record whether goals were fixed before outcomes.
2. Nearly identical right-arm q does not identify object or hand state. This is
   present in actual saved endpoints: VAL apple-42031-release_high hold vs closed
   has maximum seven-arm-joint difference0.00793284rad, but apple heights differ
   by0.14859885m and apple/plate distances by0.02196738m. TRAIN
   apple-42004-release_high hold vs closed differs by at most0.00826323rad while
   apple height differs0.14721305m. Across saved sibling endpoint pairs,18 pairs
   meet this audit's descriptive screen(max arm delta<.01rad and object height
   or plate-distance difference>.01m). Thresholds describe this read-only screen;
   they are not a prospective acceptance gate or tuned controller limit.
3. The above arm-q ambiguities are not exact-state collisions: hand q/qvel and RGB
   may distinguish them. Arm-pose error can evaluate alignment only; it cannot
   certify object placement, contact/grasp, release or overall task success.
   Preserve image error and physical task evaluation as separate evidence.
4. Camera pixels depend on posture, reset jitter, occlusion and object motion.
   Low-resolution RGB can alias configurations. A fixed TRAIN bank is a data
   baseline, not automatically an identifiable inverse sensor model; report
   multi-neighbor q dispersion and ties without using hidden object truth to
   choose a neighbor. Changing nearest-neighbor targets into a runtime privileged
   q goal would violate the RGB-only goal interface.

Supporting read-only state checks are saved separately at
/tmp/apple_goal_alignment_state_checks.json. No project files were modified.

## Planning-time semantic inspection provenance

This inspection preceded the prospective TASK041 model-evaluation freeze. It is
not a blinded model result. The exact executed state-table and screening code is
preserved at `/tmp/apple_goal_alignment_state_checks.py`: the same Python body
as the read-only shell invocation, saved afterward without rerunning it.

The screen read only `observation.state` and `observation.state_mask` Parquet
columns for all 410 TRAIN and 147 VAL episodes. It compared final branch
observation arm positions at indices 22–28 within each of 66 root IDs, examining
66 × C(8, 2) = 1,848 sibling endpoint pairs. Object height and object/plate
distance came exclusively from previously saved evaluator outputs in
`episode.metadata.final_score`. They were not recomputed with physics or
inferred from images. The exact descriptive criteria were:

```python
max(abs(q_a - q_b)) < 0.01
max(abs(height_a - height_b), abs(distance_a - distance_b)) > 0.01
```

All 18 matching examples, rather than only the illustrations above, remain in
`/tmp/apple_goal_alignment_state_checks.json`. TRAIN/VAL membership came from
the sealed manifest; the screening code opened no TEST files. No scoring metadata
was passed to an encoder, predictor, retrieval feature, candidate selector or
controller. These examples support retaining a separate visual/object-state
requirement; they do not set or validate any proposed arm-error acceptance
threshold. Publishing this note changes documentation only.
