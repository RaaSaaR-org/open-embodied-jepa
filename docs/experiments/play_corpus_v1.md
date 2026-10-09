# TASK-085 — Phase 1: the MuJoCo play corpus `play-v1` (one arm, one Dex3 hand)

**Status: FROZEN when the Stage 0 PR merges** (R23.1–R23.9 and R23.16–R23.19, decided by Claude
under owner delegation). After that merge nothing in §2–§10 changes without its own reviewed
ruling. Stage 0's record is §14.
This is Phase 1 of the proposal [JEPA_ZERO_SHOT_PLAN.md](../JEPA_ZERO_SHOT_PLAN.md) ("Play
corpus"). Its real-data pre-step is a separate task, TASK-086
([real_g1_dex3_prestep.md](real_g1_dex3_prestep.md)).

**This is not an Apple→Plate experiment and not a learned result.** The play policy is a
privileged scripted collector (it reads simulator object truth to choose where to reach); grasps,
lifts or placements in this corpus are scripted-collector events. R7's canonical status sentence
does not change, whatever row this task ends in.

## 1. Question

Can we collect, on this machine, 10–50 simulated hours of task-agnostic play by G1's right arm and
right Dex3 hand, with random, perturbed and deliberately failed behaviour and the full robot state
logged, that covers the Phase 3 test workspace (§9) and moves an object in at least 20 % of
episodes (the plan's Phase 1 gate)?

## 2. Scene (`play_scene_v0`, `src/embodied_jepa/play_corpus.py`)

- The v1/v2 robot and table unchanged: fixed-pelvis G1 29-DoF with both Dex3 hands
  (`g1_29dof_with_hand.xml`, pinned assets), pelvis at world (0, 0, 0.793), table top z 0.74.
- **Camera:** the onboard torso camera only (`onboard_rgb`, 75° vertical field of view), rendered
  at **112 × 112** px. No wrist camera: it would double the storage (§5) on a fixed disk budget,
  the Phase 2 arms are defined on one camera, and earlier tasks found no camera that the
  readouts needed beyond this one (TASK-075 tested four views); a wrist camera stays a possible
  later corpus, not part of this one.
- **Objects (movable, free joints).** apple: sphere r 2.7 cm, 80 g, red (the v1 apple body);
  cube: 4.6 cm box, 60 g, green; banana: capsule r 1.8 cm, half-length 3.5 cm, lying, 50 g,
  yellow; can: cylinder r 2.4 cm, height 6 cm, 70 g, purple. **Plate:** v1's static blue plate
  (r 7 cm, rim), position randomised per episode. All movable objects use v2's contact
  (`apple_to_plate_v2`: condim 6, friction 1 / 0.01 / 0.001); the apple is switched by
  `apply_v2_scene`, which checks it is the v1 apple.
- **Layout per episode.** Plate xy uniform in x ∈ [0.30, 0.52], y ∈ [−0.30, −0.02] (world m).
  k ∈ {1, 2, 3, 4} objects uniformly, a uniform random subset of the four, each at a uniform xy in
  the **reset region** x ∈ [0.26, 0.54], y ∈ [−0.34, 0.00] (world) with ≥ 1 cm clearance between
  collision radii (plate included), uniform yaw. Objects not on the table are parked on the floor behind the robot, out of
  the onboard view.

## 3. Robot, actions, control rate

- Action schema `ee_delta_grasp_v0` (14-D) and the unchanged manifest `configs/g1_sim_action.json`
  (1.5 cm and 0.06 rad per unit, **20 Hz**, control dt 0.05 s, joint rate limit 2 rad/s, measured
  joint velocity stop 5 rad/s).
- **One arm, one hand:** the right arm (dims 6–11) and right grasp (dim 13) are played; the left
  arm commands are 0 and the left grasp −1 (open) in every step. The schema stays 14-D, so the
  corpus loads with every existing reader; Phase 2 may drop the constant dimensions.
- Every request goes through `G1Embodiment.project_candidates` (the planner's own backtracking of
  the arm delta through 1, 1/2, …, 1/64, 0) and is then executed. **The stored action is the
  executed (applied) one;** the request is kept in the sidecar (§6).

## 4. Play policy (`play_policy_v0`)

Per episode a mode is drawn: **scripted** 0.40, **perturbed** 0.35, **random** 0.25. In scripted
and perturbed episodes the policy chains skills until the episode ends: **pick_place** 0.55,
**push** 0.25, **poke** 0.10, **wander** 0.10 (wander only when no object is on the table). Random
episodes are wander only.

- **pick_place** (e9's pick recipe): pick a random on-table object; palm target = object centre +
  (−1.5, 0, 5.2) cm (e9's palm offset) + aim noise N(0, 0.8 cm) in x, y (0.4 cm in z); hover 8 cm
  above for 50–79 commands; descend 40–59; close to a closure U(0.85, 1.0) for 25–39; lift by
  U(10, 20) cm in 40–69; carry to a destination: the plate centre (0.4) or a uniform point of the
  reset region (0.6), plus e9's palm offset (−1.5 cm in x) and N(0, 1 cm) noise in x and y, then
  clamped into e9's right-arm reach sphere (r 0.485 m, `RestingPlaceExpert.release_pose`, which
  also sets the lowest reachable palm height there); lower, open, retreat. **Deliberate failures** per pick: none 0.65, *miss* (aim moved 4–8 cm in a random
  direction) 0.10, *early_close* (close at hover) 0.07, *drop* (open mid-carry) 0.08, *no_close*
  (descend and carry with the hand open) 0.05, *abort* (leave for a random point after the
  descent) 0.05.
- **push:** approach 9 cm behind the object at 2–5 cm above its centre, sweep through it 5–15 cm
  in a random horizontal direction, hand opening U(−1, 0.3).
- **poke:** hover above an object, touch down (−1 to +2 cm about the palm target) with the hand
  U(−1, 0) closed, retreat.
- **wander:** 2–5 random palm points, uniform in x ∈ [0.20, 0.52], y ∈ [−0.36, −0.02],
  z ∈ [0.00, 0.28] (pelvis frame), 15–44 commands each, palm yaw U(±0.5 rad) per point, grasp −1
  or U(−1, 1) with equal probability.
- **Tracking law.** Scripted skills use e9's law: arm command = (target − palm) / 1.5 cm, clipped
  at ±0.4; wander uses half gain, clipped at ±0.6. Orientation is driven to e9's palm-down
  orientation turned by a yaw (U(±0.2 rad) per pick, push or poke; U(±0.5 rad) per wander point;
  in random mode instead an orientation drifting as an AR(1) process, clipped at ±0.6 rad per
  axis), commands clipped at ±0.8. The requested grasp
  moves at most 0.2 per command.
- **Noise.** AR(1) action noise on the six arm dimensions (coefficient 0.85, stationary standard
  deviation 0.10 / 0.25 / 0.45 normalized units in scripted / perturbed / random; rotation
  dimensions at half that). Perturbed episodes add bursts: started with probability 1/60 per
  command, 8–19 commands long, a constant N(0, 0.7) offset (rotation × 0.4). The arm command is
  clipped to [−1, 1] before projection.
- Randomness: `numpy.random.default_rng([8701, seed])` per episode (layout, mode, skills, noise).
- Parameters this section does not list (poke and push phase lengths and heights, the hold and
  move lengths of *early_close*, *drop* and *abort*, the place, carry and retreat heights, the
  random-mode orientation drift, the noise state starting at 0) are those of `play_corpus.py` at
  the frozen revision. Wander's 0.10 weight applies whether or not objects are on the table; with
  no object on the table every skill is wander.

## 5. Episodes and size

- Reset to the layout, open both hands, then 10 hold commands (not stored) so the objects settle;
  then **up to 400 commands (20 s), 401 frames**.
- An episode ends early, *truncated*, when the projection is infeasible or raises (the 5 rad/s
  measured-velocity stop), or when a command is rejected; the frame observed before the failed
  command is replaced by a fresh observation of the same state, so every stored action was
  executed and T stored actions have T + 1 frames. It is stored if it holds ≥ 50 commands,
  else discarded and counted. An exception from MuJoCo discards the episode and is counted; the
  shard continues. No episode is terminated (no task end).
- **3 200 episode seeds**, 850000–853199, in **32 shards of 100** (shard s holds 850000 + 100 s …
  850000 + 100 s + 99). Expected ≈ 15.8 h and ≈ 11.5 GB (development: mean 356 stored commands,
  9.8 KB per PNG frame; §14 updates this to ≈ 12.2 GB). 15.8 h sits inside the plan's 10–50 h; the size is set by the disk: on 39
  GB free, with ≥ 10 GB to keep free and room for TASK-086, ≈ 12 GB is the budget. **Disk rule:**
  a shard does not start while `/` has < 12 GiB free; skipped shards are recorded and the corpus
  is what was written (G-SIZE then decides).

## 6. Storage and logging

- `data/play-v1/shard-SS/`: one `DatasetStore` per shard (`jepa_lerobot_v3_png_v0`, fps 20,
  camera `onboard_rgb` 112 × 112 PNG), task `play_v0`, object_id `play`, container_id `plate`;
  every episode is its own session. Shards keep `DatasetStore`'s per-write hash check affordable.
- **Robot state (model inputs):** `observation.state` = positions and velocities of all 43
  actuated joints (`g1_dex3_proprio_v0`), plus the physical action (`action.physical`).
- **Sidecar per episode** `play/<episode_id>.npz` (its sha256 in the episode metadata): requested
  actions and whether the executed action differs from the request (projection or execution
  clipping); both palm poses (`*_ee` site, pelvis frame,
  position and rotation) and the commanded joint targets at every frame; per object, world
  position, quaternion, linear velocity, and robot, hand and grasp contact flags (grasp = thumb
  plus index or middle in contact); plate position; the skill index per frame. **Everything in the
  sidecar is labels only and never a model input** (the requested actions, too, are computed from
  privileged object truth).
- Episode metadata: seed, layout, mode, skill and failure log, stop reason, per-episode facts
  (§10). A corpus manifest `data/play-v1/corpus.json` lists shards with their manifest hashes,
  seeds, salts, counts, code revision and the renderer string.
- Evidence (logs, corpus manifest, coverage report, `SHA256SUMS`) goes to
  `~/develop/emai/evidence/task085-*/`.

## 7. Splits

By episode, frozen inside each shard with `freeze_split_assignments` (`heldout_combinations=()`):
rank the shard's stored episodes by sha256 of `"8702:<seed>"`; the lowest 5 are **val**, the next
5 **test**, the rest **train** (2 880 / 160 / 160 if nothing is lost). A corpus shard with fewer
than 11 stored episodes is left unsealed, recorded and excluded. Debug shards (§12) use 1 val and
1 test. Phase 2 trains on train and selects on val; test is not opened until a later
preregistration uses it.

play-v1 contains apple-to-plate carries (the apple is one of the four objects and 40 % of carries
go to the plate), so it cannot back any later claim that relies on the v1 held-out (apple, plate)
pairing.

## 8. Seeds and salts

Development 85000–85999 (used before this draft: 85000–85191 and 85100–85147, §11, with policy
salt 8501, which TASK-083 had already used; the corpus uses fresh salts); Stage 0 debug
86000–86099; corpus 850000–853199; policy salt **8701**; split salt **8702**. Later tasks do not
reuse these seeds or salts.

## 9. The Phase 3 test workspace (for the coverage gate)

Phase 3 tests reach and grasp-and-lift by the right hand. Its preregistration draws object resets
and hand goals from inside this workspace, or reports this corpus's coverage of whatever it uses
instead.

- **Object region** (world xy where test objects are placed): x ∈ [0.30, 0.46], y ∈ [−0.28,
  −0.04]; **4 × 6 cells** of 4 cm. It contains the v1/v2 apple and plate reset areas; its far edge
  stops at x 0.46 because a palm-down grasp did not reach lower than about 10 cm above the table
  beyond palm x ≈ 0.44 in development.
- **Hand region** (right palm, `right_ee` site, pelvis frame): x ∈ [0.24, 0.44], y ∈ [−0.30,
  −0.02], z ∈ [0.06, 0.26] (world z 0.853–1.053; a palm-down grasp holds the palm at about
  0.06–0.10); **5 × 7 × 5 = 175 cells** of 4 cm.

## 10. Gate (on the train split) and rows

- **G-SIZE:** stored commands (all splits) × 0.05 s ≥ **10.0 h**.
- **G-HAND:** ≥ **95 %** of the 175 hand cells hold the palm in ≥ **200** train frames.
- **G-OBJ:** each of the 24 object cells holds the first-frame position of an on-table object in
  ≥ **20** train episodes, and ≥ **90 %** of the cells have ≥ **10** train episodes in which the
  robot contacts an object that started in the cell.
- **G-MOVE (the plan's gate):** ≥ **20 %** of train episodes move an object: an on-table object's
  xy is ≥ 2 cm from its first-frame xy at some frame, and the robot contacts that object at some
  frame of the episode.

Rows (first match): **P1-VOID** if a seed, hash or revision check fails or a crash prevents the
measurement; **P1-PASS** if all four gates pass; **P1-FAIL** otherwise, naming the failing gates.
P1-PASS lets Phase 2 preregister on train and val. P1-FAIL escalates: one top-up or policy change
needs its own ruling; there is no abandonment clause (nothing is closed).

These bars are not demanding: G-HAND and G-OBJ were set after the development coverage (§11),
and the hand region's floor sits where this collector's palm-down grasp bottoms out, so they are
close to pass-by-construction, and G-MOVE's 20 % (the plan's) is well below the development 57 %.
They check that the corpus covers the declared workspace, not that it is good play data; the
results document says so.

**Reported only:** lifted episodes (an object ≥ 3 cm above its start with grasp contact), grasp
contact, per mode / skill / failure type; truncation reasons; per-dimension action mean and
spread, the share of saturated (|a| ≥ 0.99) and projection-reduced commands; per-object facts;
coverage maps; bytes and wall time; the same gate quantities on val and test.

## 11. Disclosed development before this draft

On development seeds 85000–85191 (and 85100–85147 for the pick sweeps), the policy was tuned
before any corpus seed was run: faster palm-down orientation (rotation clip 0.8; with e9's 0.5
the first hover ended before the palm was down and picks almost never lifted), e9's grasp height,
a grasp ramp, smaller aim noise, and the reach-sphere clamp. With the final settings, 192
development episodes gave: an object moved in 57 %, a lift in 9.4 %, grasp contact in 20 %;
22 % ended early, every one on the 5 rad/s velocity stop of a right-hand joint (mostly the thumb
snapping free after pressing on the table while closing); 356 stored commands on average; 9.8 KB
per frame; about 1 200 commands per second with 12 processes. The hand and object regions of §9
and the G-HAND / G-OBJ counts were fixed after seeing the development coverage (at 192 episodes,
175 of 175 hand cells were visited, the sparsest by 38 frames); the 20 % move bar is the plan's.
These are development readings, not results.

## 12. Stage 0 and freeze

Stage 0 (its own PR): the module, the runner `scripts/collect_play_corpus.py` (shards in parallel
processes; one GPU lock for the job, §13), the coverage and gate script
`scripts/measure_play_corpus.py`, tests, and a debug smoke on 86000–86015 (two shards of 8)
through storage, splits (1 val, 1 test per debug shard) and the gate script. Stage 0 may change only the episode count and the
disk floor of §5, each by a recorded rule; anything else needs a ruling. The protocol is frozen at
the Stage 0 merge.

## 13. Runtime

The Linux PC; MuJoCo on CPU, frames rendered with NVIDIA EGL on the RTX 5080 (as every earlier v2
corpus), the job under `scripts/gpu_run.sh --wait --min-free-gib 8 --board`; the GL renderer
string is recorded. Phase 3 renders with the same renderer or declares the difference. Frames are
not bit-identical across renderers.

## 14. Stage 0 record (R23.16–R23.19)

At `cfb2364` (this PR's code commit, clean tree), NVIDIA EGL (`NVIDIA Corporation / NVIDIA GeForce
RTX 5080/PCIe/SSE2 / 4.6.0 NVIDIA 595.91.07`), under `gpu_run.sh`. Code:
`src/embodied_jepa/play_corpus.py`, `scripts/collect_play_corpus.py`,
`scripts/measure_play_corpus.py`, `tests/test_play_corpus.py` (layouts, split rule, policy
bounds, gate arithmetic and rows, the core-import check; graphics opt-in: a storage round trip,
same seed twice giving identical actions, states and object positions, the objects' resting
heights and the banana lying flat, and T + 1 frames after an early end). Evidence:
`~/develop/emai/evidence/task085-stage0/` (`SHA256SUMS` sha256 `112a746d…8ca666`).

| Debug run | Seeds | Stored episodes | Commands (mean) | Ended early | Bytes per command | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| smoke, 2 shards × 8, 2 workers | 86000–86015 | 16 / 16 | 6 342 (396) | 2 | 10 614 | 24.5 s |
| scale, 10 shards × 8, 10 workers | 86016–86095 | 80 / 80 | 28 494 (356) | 21 | 10 695 | 28.7 s |

Both ran the whole path (storage, sidecars, splits 1 / 1 / rest, the gate script with its seed,
split and salt checks; no void reason, no error, no discard); their rows (P1-FAIL on G-SIZE,
G-HAND, G-OBJ) are what a debug-sized corpus gives and are not evidence. Every early end was a
right-hand joint passing the 5 rad/s velocity stop (thumb_1 in 21 of 23). Reported only, the 60
scale train episodes: an object moved in 68 %, a lift in 18 %, grasp contact in 32 %.

**Found in review and fixed before this record** (the superseded first runs are kept in
`~/develop/emai/evidence/task085-stage0-superseded-upright-banana/`): the banana's body
quaternion repeated its geom's 90° turn, so it stood upright (and the §11 development episodes
had it upright too); *abort* closed the hand before leaving, against §4. Rendering note: NVIDIA
EGL is not bit-reproducible here; the same seed twice gave identical physics, and one frame of 61
differed by one level in 6 pixels.

**Projection and the two Stage 0 knobs (R23.17).** At 356 commands and 10.7 KB per command, 3 200
episodes give ≈ 15.8 h and ≈ 12.2 GB (this replaces §5's ≈ 11.5 GB) in ≈ 15 min with 12 workers;
with 39 GB free and TASK-086's ≈ 7 GB, ≥ 12 GiB stays free. The episode count stays **3 200** and
the disk floor **12 GiB**.
