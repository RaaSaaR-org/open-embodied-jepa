# TASK-086 results — license check and conversion of open real G1 + Dex3 data

Protocol: [real_g1_dex3_prestep.md](real_g1_dex3_prestep.md) (FROZEN at `f3f71b7`, #184;
R23.10–R23.15, R23.20–R23.22). Rulings on this record: DECISIONS 2026-10-09 (y), R23.27–R23.30,
decided by Claude under owner delegation.

**This is not an Apple→Plate experiment and not a learned result. R7's canonical status sentence
does not change.**

## 1. Row: R-DROP-GRASP — the real-data arm is dropped

| Rule (pooled over the 13 training-eligible sets) | Bar | Measured | Fails |
| --- | --- | --- | --- |
| R-RANGE: steps with any arm \|a\| > 1 | > 0.5 drops | **5.5 %** (90 406 of 1 650 821 valid steps) | no |
| R-GRASP left: median RMS residual | > 0.15 rad drops | **0.487 rad** | **yes** |
| R-GRASP right: median RMS residual | > 0.15 rad drops | **0.461 rad** | **yes** |
| R-GRASP: frames with t outside [−0.25, 1.25] | > 25 % drops | 0.8 % left, 0.3 % right | no |

First match (protocol §5): no void reason (every set downloaded at a pinned revision with the
expected license tag), R-RANGE passes, R-GRASP fails on both hands, so the row is
**R-DROP-GRASP**: the plan's "sim play + open real G1/Dex3 data" arm is dropped for Phase 2.
`conversion_report.json` sha256 `264956e7…dd48`.

**Why.** The arm motion converts well: 94.5 % of valid steps are inside the arm range (the
99th percentile of |a| per arm dimension is 0.41–1.36). The grasp does not: our one-scalar Dex3
synergy (open → closed line in `configs/g1_sim_action.json`) leaves a median residual of
0.33–0.57 rad on every Unitree set, two to four times the bar. NVIDIA's GR00T-Teleop-G1 alone
fits (0.095 left, 0.109 right), but the pooled rule is dominated by Unitree's 1.53 M of 1.66 M
steps. The likely cause, not tested here: Unitree's hand joints do not follow our synergy's line
(converter development, protocol §6, saw PickApple's left thumb_0 held near 0.52 rad, where the
synergy holds 0), whether by teleoperation style or by a joint-zero or sign convention we did not
check.

## 2. Run

- Downloads 2026-10-09 at `f3f71b7`, each repository pinned to the revision it resolved (NVIDIA's
  two at the protocol's revisions); the Hub rate-limited the first attempt (HTTP 429), the retry
  downloaded every set. Large files checked against the Hub's sha256. ≈ 3.5 GB raw (3.3 GiB by the recorded file sizes), deleted after
  conversion (protocol §3); revisions, file hashes, licenses (`LICENCE` for AppleToPlate; no
  NOTICE files in the Unitree sets) in `downloads.json`.
- Conversion on CPU in the run worktree, both passes completed, exit 0.
- Evidence: `~/develop/emai/evidence/task086-run/` (downloads.json, conversion_report.json and
  its pass-1 copy, logs, the sets' README and license files), `SHA256SUMS` sha256
  `4f1aef32…e83c`.

## 3. Reported only

| Set | Episodes | Arm out of range | Median residual L / R (rad) | Hours kept |
| --- | --- | --- | --- | --- |
| Unitree BlockStacking | 301 | 8.4 % | 0.548 / 0.529 | 2.22 |
| Unitree CameraPackaging | 201 | 5.5 % | 0.550 / 0.528 | 2.11 |
| Unitree ObjectPlacement | 210 | 21.2 % | 0.564 / 0.567 | 0.44 |
| Unitree Pouring | 311 | 7.7 % | 0.567 / 0.565 | 0.98 |
| Unitree ToastedBread | 418 | 4.9 % | 0.480 / 0.476 | 2.45 |
| Unitree PickApple | 201 | 1.9 % | 0.459 / 0.439 | 0.81 |
| Unitree PickBottle | 202 | 3.1 % | 0.424 / 0.438 | 1.03 |
| Unitree PickCharger | 200 | 6.3 % | 0.541 / 0.433 | 0.92 |
| Unitree PickGum | 199 | 6.0 % | 0.507 / 0.454 | 0.87 |
| Unitree PickSnack | 200 | 2.0 % | 0.332 / 0.411 | 0.10 |
| Unitree PickTissue | 205 | 2.2 % | 0.456 / 0.488 | 1.03 |
| Unitree PickDoll | 203 | 1.1 % | 0.464 / 0.401 | 0.76 |
| NVIDIA GR00T-Teleop-G1 | 1 095 | 9.9 % | 0.095 / 0.109 | 0.86 |
| pooled (training-eligible) | 3 946 | 5.5 % | 0.487 / 0.461 | 14.59 of 23.0 converted |

- "Hours kept" counts runs of ≥ 20 unflagged steps; a step is flagged when an arm dimension
  leaves [−1, 1], a grasp leaves [−1, 1] (26 % of pooled valid steps; the synergy position's 95th
  percentile is 1.16–1.18, so mostly just beyond the closed end) or a frame is invalid. No set exceeded 0.5 on its own arm range, so
  none was excluded.
- The `action`-target variant (recorded targets instead of measured states): arm out of range
  5.9 % pooled.
- **GraspSquare is a duplicate of BlockStacking:** 301 episodes each, decoded states and actions
  identical. It was excluded as declared.
- **Label problem:** all 201 Unitree PickApple episodes carry the task string "Pick up the red cup
  on the table."
- **Stores written** (protocol §4; kept on disk, not for use now that the arm is dropped):
  GR00T-Teleop-G1 1 773 segments from 913 source episodes (823 / 45 / 45 source episodes), 1.2
  GB; Unitree PickApple 408 segments from 142 source episodes (128 / 7 / 7), 0.9 GB; manifest
  hashes in `data/real-g1-v1/corpus.json`.
- **The held-out AppleToPlate set did not convert:** under the assumed joint layout (§9), 34 646
  of 114 424 resampled frames (30 %) were invalid (a joint more than 0.1 rad outside its range) and
  every valid step had a grasp outside [−1, 1], so no segment was kept and its store is empty. The
  layout assumption is the likely cause; not diagnosed. It was outside the decision.

## 4. Caveats

- The bars (0.15 rad, 25 %, 0.5) were declared choices, not calibrated (protocol §6); a looser
  grasp bar would not have changed the row (the pooled residual is about three times the bar).
- One converter, our MJCF's kinematics and our synergy endpoints; the realised palm deltas come
  from measured states.
- License: Apache-2.0 and CC-BY-4.0 by the repositories' tags; Humanoid Everyday G1 was not used
  (license unresolved, its authors not asked).

## 5. What changes

- TASK-086 closes with **R-DROP-GRASP** (R23.28): Phase 2 is planned on sim play only; the
  converted stores stay on disk unused.
- Not decided here, a possible later ruling: a GR00T-Teleop-G1-only real arm (its grasp fits;
  0.86 h kept), or a per-provider hand mapping for Unitree's sets, each with its own
  preregistration.
- **R7 does not change** (R23.29).
