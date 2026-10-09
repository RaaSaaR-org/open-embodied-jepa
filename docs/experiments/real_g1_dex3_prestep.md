# TASK-086 — Phase 1 real-data pre-step: license check and conversion of open real G1 + Dex3 data

**Status: DRAFT** (R23.10–R23.15, decided by Claude under owner delegation). It is frozen when the
converter PR (§7) merges; after that, nothing in §2–§6 changes without its own reviewed ruling.
This is the pre-step that [JEPA_ZERO_SHOT_PLAN.md](../JEPA_ZERO_SHOT_PLAN.md#open-real-g1-data)
puts in front of Phase 2's "sim play + open real G1/Dex3 data" arm; the play corpus itself is
TASK-085 ([play_corpus_v1.md](play_corpus_v1.md)).

**This is not an Apple→Plate experiment and not a learned result.** R7's canonical status sentence
does not change.

## 1. Question

Do the openly licensed real G1 + Dex3-1 teleoperation sets convert to our action schema
`ee_delta_grasp_v0` with most steps inside [−1, 1] and with a grasp that fits our one-scalar Dex3
synergy? If not, the real-data arm is dropped and the reason recorded (the plan's stop rule).

## 2. Sources and license decision (Hugging Face, read 2026-10-09)

| Source | Revision | License (repo tag) | Use |
| --- | --- | --- | --- |
| `unitreerobotics/G1_Dex3_*_Dataset`, 13 sets (BlockStacking, CameraPackaging, GraspSquare, ObjectPlacement, Pouring, ToastedBread, PickApple, PickBottle, PickCharger, PickGum, PickSnack, PickTissue, PickDoll) | recorded per set at download | Apache-2.0 | training-eligible; GraspSquare is excluded if its data files are byte-identical to BlockStacking's (the plan's suspected duplicate) |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` (4 folders: apple, grapes, pear, starfruit) | `0d7bdd06e6…` | CC-BY-4.0 | training-eligible, attribution recorded in `docs/DEPENDENCIES.md` |
| `nvidia/GR00T-N1.7-AppleToPlate` | `d89c126a71…` | CC-BY-4.0 | **held-out real-image test set only**: converted and measured, every episode in `holdout`, never training data |
| `USC-PSI-Lab/Humanoid-Everyday-G1` | — | no tag on the repo; README (MIT) and the full set's card (Apache-2.0) disagree | **not used**: the license is unresolved and no reply from the authors is in hand |
| `mncai/G1_Dex3_Trash_LocoManipulation` | — | Apache-2.0 | not used (whole-body walking; the plan rates it a poor fit) |

A dataset license tag is taken as the license; a source-code license is not. The full revisions,
file hashes and the attribution text are recorded at download.

## 3. Download budget

Data (states and actions, Parquet) and metadata of every row in §2 marked training-eligible or
held-out: about 0.75 GB. Video only for `GR00T-Teleop-G1` (≈ 0.46 GB), `GR00T-N1.7-AppleToPlate`
(≈ 0.93 GB) and one camera of Unitree PickApple (`cam_left_high`, ≈ 1.5 GB) — our task family,
within the disk budget left by TASK-085 (≥ 10 GB stay free). The other Unitree videos (≈ 65 GB)
are not downloaded here; Phase 2 may add them by its own ruling when disk allows. Raw downloads
are deleted after conversion; their hashes and revisions stay recorded, so they can be fetched
again.

## 4. Conversion (`src/embodied_jepa/real_g1_convert.py`)

- **Joints by name.** Unitree's `kRightShoulderPitch`-style names map to our MJCF names by a fixed
  table in the module (hand joints by name: Unitree orders the left hand thumb, middle, index and
  the right hand thumb, index, middle); NVIDIA uses our names. Waist joints are 0 for Unitree's
  28-D sets (recorded) and taken from the data for NVIDIA's 43-D sets; legs are ignored (our pelvis
  is fixed).
- **20 Hz.** 30 Hz sources take the source frame nearest to t = k / 20 (frame round(1.5 k)), for
  joints and images alike; 20 Hz sources are used as they are.
- **Invalid frames.** A frame whose mapped joint lies more than 0.1 rad outside its MJCF range
  (for example the two impossible PickApple rows) is invalid.
- **Arm action** from the **measured state** (`observation.state`): forward kinematics with our G1
  MJCF to the `left_ee` / `right_ee` sites in the pelvis frame; Δp = p(k+1) − p(k) divided by 1.5
  cm; the rotation part is the roll-pitch-yaw r with R(k+1) = `rotation_delta`(r) · R(k), divided
  by 0.06 rad. (Our simulator stores a commanded delta; a realised delta is the closest real-data
  equivalent. The same conversion from the recorded `action` targets is reported.)
- **Grasp** per hand: the least-squares position t of the 7 named Dex3 joints at k + 1 on the line
  from our open to our closed synergy (`configs/g1_sim_action.json`), grasp = 2 t − 1; the RMS
  residual per joint (rad) is kept.
- **Out of range.** A step with any |a_i| > 1, or touching an invalid frame, is flagged; flagged
  steps are never clipped. Each source episode is cut at flagged steps; the remaining runs of
  ≥ 20 steps are stored as episodes of one session (the source episode).
- **State:** positions of our 43 actuated joints, velocities by finite difference at 20 Hz; joints
  the source lacks (legs; waist for Unitree) are masked invalid.
- **Images** (where downloaded): the head or ego camera (`cam_left_high`, `ego_view`), centre crop
  480 × 480, resized to 112 × 112 (Lanczos), stored under its own key `real_head_rgb` (a second
  visual domain, never mixed with `onboard_rgb`).
- **Storage:** one `DatasetStore` per source set under `data/real-g1-v1/<set>/`, provenance with
  repository, revision, license and file hashes. Sets without downloaded video are measured (§5)
  but not stored as image corpora.
- **Splits** by source episode, per set: rank by sha256 of `"8603:<repo>:<episode>"`, 5 % val,
  5 % test, the rest train; AppleToPlate entirely `holdout`.

## 5. Stop rule and rows

Measured on every converted step of the training-eligible sets of §2 (pooled), and per set:

- **R-RANGE:** the fraction of steps with any |a_i| > 1 (the 14 dimensions). The arm is dropped if
  the pooled fraction is **> 0.5** ("most converted steps"); a single set above 0.5 is excluded.
- **R-GRASP:** per hand, the median RMS residual over frames, and the fraction of frames with t
  outside [−0.25, 1.25]. The grasp fit is **poor** if, pooled, either hand has a median residual
  **> 0.15 rad** (about an eighth of the synergy's largest joint travel, 1.2 rad) or more than
  **25 %** of its frames outside [−0.25, 1.25].

Rows (first match): **R-VOID** if a license is not confirmed as in §2 or a conversion error
prevents the measurement; **R-DROP-RANGE** if R-RANGE fails; **R-DROP-GRASP** if the grasp fit is
poor; **R-KEEP** otherwise. R-KEEP keeps the real-data arm for Phase 2 (its own preregistration);
a drop row drops the arm and records why; neither is a clause on anything else.

**Reported only:** per set and pooled, the step fractions above per dimension, the 50th / 95th /
99th percentiles of |a_i|, steps and hours kept, episodes cut and dropped, invalid frames, the
`action`-target variant, and the same numbers for AppleToPlate (held out, outside the decision).

## 6. What this does not show

Converting is not evidence that real data helps a world model tested in simulation (the plan's
Phase 3 decides that), and real teleoperation is task demonstration data, not play.

## 7. Runtime and evidence

The converter and its tests come in one PR, frozen at merge; the run follows on the merged
revision on CPU. Evidence (download manifest with revisions and hashes, license record, conversion
report, `SHA256SUMS`) goes to `~/develop/emai/evidence/task086-*/`.
