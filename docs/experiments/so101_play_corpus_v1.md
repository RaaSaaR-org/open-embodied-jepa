# SO-101 play corpus `so101-play-v1`: collector and measurement before training (S3, TASK-099)

**Development data. No preregistration, no gate, no claim.** The corpus was collected by a
**privileged scripted** collector: it reads the cube pose and contacts from simulator truth to
choose where to reach and whether to regrasp. Its grasps, lifts and placements are scripted
outcomes, never learned results, and none is an Apple→Plate result. R7 does not change (R28.9).
Every number below is a development measurement of this one corpus (one collection run per
corpus, simulation only), not a bar.

Plan: [SO101_PLAN.md](../SO101_PLAN.md) S3; rulings: [DECISIONS](../DECISIONS.md) 2026-10-16
(bp), R28.1–R28.9; scene and embodiment: [SO101_SIMULATION.md](../SO101_SIMULATION.md) (S1);
tasks and scorers: [so101_cube_tasks_dev.md](so101_cube_tasks_dev.md) (S2).

## What was built

- `src/embodied_jepa/so101_play_corpus.py`: the collector `so101_play_policy_v0`, per-episode
  facts and storage. Modelled on TASK-085's `play_corpus.py` and TASK-090's
  `play_corpus_grasp.py`. Neither file was edited.
- `scripts/collect_so101_play.py`: runs the collector in parallel shards. Each shard is a
  `SchemaDatasetStore` (S0's `schema_data.py`, the LeRobot v3 PNG-in-Parquet profile sized by
  `joint_delta_so101_v0`) with one sidecar `.npz` per episode. Each shard is sealed with its
  split, and a `corpus.json` records the revision, the code hashes and each shard's manifest
  hash. The script never overwrites: a complete shard is skipped and an incomplete one is
  refused. Pilot and corpus runs need a clean tracked tree, `MUJOCO_GL=egl` and the GPU lock
  (`scripts/gpu_run.sh`).
- `scripts/measure_so101_play.py`: the measures below. It writes a JSON report and a contact
  sheet into a new directory.
- `tests/test_so101_play_corpus.py`: the split rule, the resets, the window counting and the
  facts on synthetic trajectories. It also checks episode determinism and that the stored
  actions are the executed ones, and collects, seals, re-reads and measures a tiny corpus on the
  CPU (black frames).
- `benchmarks/manifests/so101-play-v1.json`: the hashes of the corpus and of the pilot.
- Every G1 hash-pinned file is byte-identical. The diff against `origin/main` adds files only,
  and `config.py` is untouched.

## The collector (`so101_play_policy_v0`)

One episode is a reset, 5 settle commands (not stored), then 400 commands at 20 Hz (20 s). Its
random stream is `numpy.random.default_rng([9901, seed])`, so an episode is a deterministic
function of its seed. Each command is computed as follows:

- An IK goal (S1's damped least-squares `solve_ik` on a scratch model, top-down approach) is
  tracked with `clip(gain · (q_goal − q_target) / scale, ±vmax)`.
- Ornstein–Uhlenbeck noise is added on the five arm channels (θ = 0.85).
- The result is clipped to [−1, 1].
- The gripper channel is an absolute command.

| Item | Declared |
|---|---|
| Reset | Cube x ∈ [0.16, 0.25] m, y ∈ [−0.12, 0.12] m, yaw ±45°; target marker x ∈ [0.16, 0.22], y ∈ [−0.14, 0.14], ≥ 8 cm from the cube; arm joints at home ± 0.06 rad, gripper U(0.2, 1.2) rad. These boxes contain S2's task reset boxes (a test checks this) |
| Modes (per episode) | scripted 0.45 (noise σ 0.05), perturbed 0.40 (σ 0.15, bursts of random joint deltas: chance 1/50 per command, 5–14 commands, N(0, 0.6) per joint, gripper noise σ 0.1), random 0.15 (σ 0.25, gripper noise σ 0.2, wander-heavy) |
| Skills (per segment) | pick_place 0.40, pick_lift 0.20, push 0.12, poke 0.08, wander 0.20; a cube outside the reachable band forces wander |
| Pick | Pre-grasp 5–8 cm above, descend, close, lift 6–12 cm; S2's grasp geometry (closing axis on a cube face, fixed jaw 5 mm off the face); 2 mm aim noise; in 15 % of picks the wrist is mis-turned by 0.3–0.8 rad |
| Place | Destination: the marker (60 %, ± 6 mm) or a random table point; optional hold 0–19 commands; transport, lower to 2–20 mm above the table, open to a random 0–0.6, back off, retreat |
| pick_lift | Hold 10–39 commands, then drop from the lift height (50 %) or lower and release |
| Injected pick failures | none 0.55, miss (aim 2–4 cm off) 0.08, offset (a rigid grasp-aim error beyond S2's measured envelope: 8–12 mm toward the fixed jaw, or 12–20 mm along the face) 0.10, early_close 0.07, no_close 0.05, drop (open during transport) 0.08, abort 0.07 |
| Regrasp | After each lift, a privileged check: is the cube 2 cm up with both jaws in contact? If not, with probability 0.7 and at most 2 times, open, rise and pick again |
| Other perturbations | Random detours of 3–8 cm on reach and transport (probability 0.1 / 0.4 / 0.4 by mode); random tracking gain U(0.5, 1) and speed cap U(0.4, 1) per segment |
| Stored action | Exactly `result.applied_action` from `SO101Embodiment.execute`, so a joint-limit or gripper-rate clip is already in it; `action.physical` is its denormalization. The requested action, a clip flag, the joint targets, the phase label, the skill index and the cube, contact and gripper-site truth go to the sidecar (labels only, never model inputs) |
| Early end | A `stopped` execution ends the episode. Its last observed frame is dropped, so every stored action was executed. This never happened in this corpus |

The parameters were set when the collector was written. Before the pilot they were checked
once, on development seeds 990000–990027, for errors, clip sources and grasp rates, without
rendering. Nothing was changed after that check, and those seeds are never stored.

**Image resolution: 224 × 224 for both cameras.** That is 16 × 16 patches for DINOv2 ViT-S/14,
the reference encoder planned for S4, and the size at which S1 and S2 inspected the frames. A
112 px frame gives only 8 × 8 patches, and the G1 line's binding constraint was centimetre
perception at contact (TASK-075). Storage was cheap: in a probe of one place episode, PNG frames
averaged 6.6 KB (wrist) and 15.9 KB (overview) at 224 px, against 2.9 and 6.3 KB at 112 px. A
consumer can always downsample.

**Splits.** Episodes are grouped by seed into shards of consecutive seeds. Within a shard,
seeds are ranked by `sha256("9902:<seed>")`: the lowest 5 go to val, the next 5 to test and the
rest to train. Each episode is its own session. Val and test come from the same distribution as
train. They are not held-out conditions.

## Runs

| Corpus | Seeds | Episodes | Code revision | Collection |
|---|---|---|---|---|
| Pilot `data/so101-play-pilot-v1` | 990100–990163 (2 shards of 32; 2 val + 2 test each) | 64 (56 / 4 / 4) | `bf12d37` | 66.1 s wall, 2 workers |
| `data/so101-play-v1` | 991000–993999 (30 shards of 100; 5 val + 5 test each) | 3 000 (2 700 / 150 / 150) | `bf12d37` | 866.1 s wall (14.4 min), 12 workers |

Both runs were made on the RTX 5080
(`NVIDIA GeForce RTX 5080/PCIe/SSE2 / 4.6.0 NVIDIA 595.99.02`, EGL) from a clean tree, under
`scripts/gpu_run.sh --wait --board`. No episode was discarded, raised an error or stopped early.
`data/` is the HDD (`/mnt/data/emai/projects/open-embodied-jepa/data`).

**Projection from the pilot, against the result.** The pilot spent 66.1 s × 2 workers / 64 =
2.07 worker-seconds per episode. That projected 3 000 / 12 × 2.07 ≈ 516 s for the corpus with 12
workers. The corpus took 866 s, 1.7 times the projection: the mean time per episode (simulation
and rendering, without writing) rose from 0.74 s to 1.17 s with 12 concurrent workers on 16 CPU
threads. The projected 9.0 MiB per episode (26 GiB in all) held: the corpus used 26.5 GiB of the
1.7 TB free on `/mnt/data`.

## Measures (`so101-play-v1`, all 3 000 episodes, 1 200 000 commands)

Report: `outputs/so101-play-v1-measure/report.json` (SHA-256 `8a5a5e09…`; copied to the
evidence folder). The measurement took 321 s on the CPU and decoded every frame. The pilot's
report (`outputs/so101-play-pilot-v1-measure/report.json`) agrees with every figure below to
within a few percentage points.

### Integrity and split leakage

- No problems were found. All 30 shards open with a full hash check, and every manifest hash
  matches `corpus.json`. There is one revision and one pair of salts. Every seed lies inside its
  shard with no duplicates, every frozen split equals the rule, and every holdout is empty.
- Leakage is 0 for every pair of train, val and test, on each of: episode ids, session ids,
  seeds, reset cube xy, and the SHA-256 of the first PNG frame of each camera.

### Action-range coverage per channel (normalized commands)

| Channel | 1 % | 5 % | 50 % | 95 % | 99 % | std | at −1 | at +1 | \|a\| < 0.05 | 20 bins covered (≥ 0.1 %) |
|---|---|---|---|---|---|---|---|---|---|---|
| shoulder_pan | −0.975 | −0.653 | 0.000 | 0.655 | 0.976 | 0.326 | 0.8 % | 0.8 % | 47.9 % | 20 (train 20) |
| shoulder_lift | −0.988 | −0.692 | 0.007 | 0.776 | 1.000 | 0.380 | 0.9 % | 1.2 % | 38.9 % | 20 (20) |
| elbow_flex | −1.000 | −0.868 | −0.005 | 0.808 | 1.000 | 0.470 | 1.9 % | 1.4 % | 28.9 % | 20 (20) |
| wrist_flex | −0.989 | −0.721 | 0.000 | 0.792 | 1.000 | 0.396 | 0.9 % | 1.2 % | 43.6 % | 20 (20) |
| wrist_roll | −0.969 | −0.649 | 0.000 | 0.649 | 0.968 | 0.323 | 0.8 % | 0.8 % | 48.4 % | 20 (20) |
| gripper | −1.000 | −1.000 | 0.082 | 1.000 | 1.000 | 0.814 | 24.8 % | 7.5 % | 2.1 % | 20 (20) |

Every bin of every channel holds at least 13 582 commands. **Clipping:** 26.4 % of commands
were applied differently from the request. By channel: wrist_flex 15.1 % (joint limit),
gripper 12.2 % (rate limit), wrist_roll 0.6 %, elbow_flex 0.01 %. The wrist_flex clips arise
because the top-down IK goals for lifted and transport poses sit at its upper limit (1.658 rad),
so the positive half of the noise is clipped there. The stored action is the clipped, executed
one.

### Joint-position coverage (measured state, against the joint limits)

| Joint | Limits (rad) | 1 % – 99 % | Range spanned (1–99 %) | 20 bins covered |
|---|---|---|---|---|
| shoulder_pan | ±1.920 | −0.875 – 0.883 | 46 % | 12 |
| shoulder_lift | ±1.745 | −1.444 – 0.626 | 59 % | 16 |
| elbow_flex | ±1.690 | −1.024 – 1.140 | 64 % | 17 |
| wrist_flex | ±1.658 | 0.256 – 1.658 | 42 % | 11 |
| wrist_roll | −2.744 – 2.841 | −2.573 – 2.559 | 92 % | 20 |
| gripper | −0.175 – 1.745 | −0.176 – 1.748 | 100 % | 20 |

The corpus covers the workspace in front of the base. It does not cover the whole joint space:
wrist_flex stays on the positive side (top-down approaches), and shoulder_pan stays within
about ±0.88 rad except in the wander segments.

### Gripper balance

- Commands in the closing half (< 0) are 48.1 % and in the opening half 51.9 %. Commands at −1
  are 24.8 % and at +1 7.5 %.
- Measured jaw angle: quantiles 5/25/50/75/95 % are −0.175 / 0.159 / 0.863 / 1.533 / 1.745 rad.
  The jaw is below 0.3 rad (closed) on 39.9 % of frames and above 1.0 rad (open) on 47.1 %.
- Both jaws touch the cube on 22.0 % of frames.

### Lift and place windows (S2's scorers, `so101_tasks.py`, on the sidecar truth)

| Split | Episodes | Lift windows (cube ≥ 5 cm up with a gripper contact for ≥ 20 steps) | Episodes with `so101_cube_lift_v0` success | Steps held (≥ 5 cm, contact) | Episodes picked (place-task pick) | Episodes with `so101_cube_place_v0` success at the marker | Placements anywhere (after a pick, at rest and free for ≥ 20 steps) |
|---|---|---|---|---|---|---|---|
| train | 2 700 | 2 939 | 2 160 | 146 310 | 2 254 | 914 | 3 240 |
| val | 150 | 162 | 114 | 8 141 | 119 | 46 | 171 |
| test | 150 | 161 | 116 | 8 193 | 120 | 42 | 181 |
| all | 3 000 | 3 262 | 2 390 (79.7 %) | 162 644 | 2 493 | 1 002 (33.4 %) | 3 592 |

The cube was moved at least 2 cm in 2 716 of 3 000 episodes, and no cube was lost off the table.
The scorers are S2's held-window scorers, applied offline to whole episodes, so a passing
crossing never counts. A place success here means the scorer's condition held at some point in
a play episode. It does not mean a task episode succeeded.

### Grasps (per pick attempt)

An attempt counts as *grasped* when, at the end of its lift, the cube is at least 2 cm up with
both jaws in contact. Of 6 414 attempts, 5 171 reached the end of their lift; the others were
aborts or were cut off by the end of the episode. Of those 5 171, 3 380 were grasped (65.4 %).

| Attempt | Grasped / attempts reaching the lift end |
|---|---|
| No injected failure | 2 267 / 2 622 (86.5 %) |
| drop (grasp first, open later) | 222 / 253 |
| offset (beyond S2's envelope) | 119 / 475 (25.1 %) |
| miss (2–4 cm off) | 33 / 390 |
| early_close | 11 / 306 |
| no_close | 0 / 243 |
| First regrasp | 680 / 814 (83.5 %) |
| Second regrasp | 48 / 68 |

Skills in the corpus: pick_place 3 705, wander 3 312, pick_lift 1 933, regrasp 1 247, push
1 173, poke 771. Modes: scripted 1 370, perturbed 1 193, random 437. Command share by phase:
reach 22.5 %, wander 20.4 %, descend 10.7 %, lift 9.3 %, retreat 7.9 %, close 6.6 %, transport
6.3 %, hold 4.6 %, lower 4.0 %, push 3.4 %, release 2.2 %, poke 0.8 %, drop 0.8 %, regrasp
opening 0.8 %.

### Camera frames (every frame decoded, 1 203 000 per camera)

| Camera | Shape | Blank frames (pixel std < 2) | Identical consecutive frames | Cube visible (red pixels > 0.02 % of the frame) | Mean PNG size |
|---|---|---|---|---|---|
| wrist_rgb | 224 × 224 × 3 | 0 | 0 | 78.2 % | 7.9 KB |
| overview_rgb | 224 × 224 × 3 | 0 | 0 | 99.1 % | 15.5 KB |

The contact sheets (the corpus's at seeds 991000, 992111, 993220 and 993994; the pilot's also)
were inspected by eye. In the wrist view the cube is large between the jaws at grasp and while
held, the blue marker is in view during places, and some wander poses show the robot's own
links. The overview shows the whole arm, the cube and the marker in every contact-sheet frame. The
cube-visible fraction is a colour threshold, not a perception measure.

### Storage

The corpus takes 26.54 GiB: 9.06 MiB per episode, or 23.1 KiB per stored frame for both
cameras with state and actions.

## Hashes and paths

- Corpus: `data/so101-play-v1/` (`corpus.json` SHA-256 `c81571de2bebe76704129d2814f39959d341ec8cba169fc310f7c22d8dd45b69`);
  per-shard manifest hashes and the code hashes are in
  [`benchmarks/manifests/so101-play-v1.json`](../../benchmarks/manifests/so101-play-v1.json).
- Pilot: `data/so101-play-pilot-v1/` (`corpus.json` SHA-256 `f341c8a17a00763297c63eac4686ad4056224fa9f9083c6e783435a0c8b942ee`).
- Measure reports: `outputs/so101-play-v1-measure/` (`report.json` `8a5a5e09223bf2ce4c82b8e38ee3b0dfed8a654f8341ea8493ba97d1859f7afe`)
  and `outputs/so101-play-pilot-v1-measure/` (`552e6cbb8bdfdee2b502973a9099e45bad3654c7cb7567076ab67655d3acbaf2`).
- Evidence copy: `~/develop/emai/evidence/task099-so101-play/` (both `corpus.json`, all shard
  reports, both measure reports and contact sheets). Its `SHA256SUMS` has SHA-256
  `5be4f58c9f2b151e1cc1ab028e83155f5b34f5f3d839be56fa577bebe90fabc6`.
- Code revision of both collections: `bf12d37`. The measure script and the docs were committed
  afterwards; the collection code did not change.

## Caveats

- **Privileged and scripted.** The collector reads simulator truth. Its success counts describe
  the data, not any controller that could be learned from it.
- **One collection run, one scene, one cube, one target marker.** The scene has fixed lighting
  and colours (a red cube on a tan table), so a model trained on it may lean on colour.
- **The joint coverage is partial** (see the table above). The actions sit on IK goals for
  top-down grasps plus noise and wander. Nearly half of the arm commands are near zero (holds
  and settled tracking).
- **Clipped commands are frequent** (26 %, mostly the wrist_flex limit and the gripper rate).
  The stored action is the executed one, so action and state agree. But at the wrist_flex limit,
  a positive command and a zero command have the same effect.
- **`gripper_contact`** follows S2: any one of the fixed jaw, the moving jaw or the camera mount
  touching the cube. The per-attempt grasp fact requires both jaws.
- **Place successes at the marker** include incidental ones, because 40 % of placements aim at
  a random table point.
- **Val and test are random episodes** of the same distribution. A closed-loop evaluation (S5)
  needs its own fresh reset seeds from the S2 task streams, which this corpus did not use.
- **The timing projection was optimistic** by a factor of 1.7 at 12 workers (see Runs).

## Not done here

No model was trained and no gate was set. S4 (LeWM and `native_jepa` with offline gates) needs
its own preregistration, and its bars must be calibrated from measured ceilings (lesson 1 of
the plan). The encoded (DINOv2-feature) fork of this corpus is S4's: `fork_unsealed` is not yet
available for schema-sized stores (`schema_data.py`).
