# TASK-085 results — the MuJoCo play corpus `play-v1` (Phase 1)

Protocol: [play_corpus_v1.md](play_corpus_v1.md) (FROZEN at `3ee2557`, #183; R23.1–R23.9,
R23.16–R23.19). Rulings on this record: DECISIONS 2026-10-09 (x), R23.23–R23.26, decided by
Claude under owner delegation.

**This is not an Apple→Plate experiment and not a learned result.** The corpus was produced by a
privileged scripted play collector; its grasps, lifts and placements are scripted-collector
events. **R7's canonical status sentence does not change.**

## 1. Row: P1-PASS

| Gate (train split) | Bar | Measured | Pass |
| --- | --- | --- | --- |
| G-SIZE (all splits) | ≥ 10.0 h | **16.06 h** (1 156 589 commands) | yes |
| G-HAND | ≥ 95 % of 175 cells with ≥ 200 frames | **97.7 %** (171 of 175) | yes |
| G-OBJ starts | every one of 24 cells ≥ 20 episodes | min **61** (max 128) | yes |
| G-OBJ contacts | ≥ 90 % of cells with ≥ 10 contact episodes | **100 %** (min 28) | yes |
| G-MOVE | ≥ 20 % of episodes move an object | **55.9 %** (1 607 of 2 874) | yes |

No void reason (hashes, revision, salts, seeds per shard, frozen splits against the rule all
checked by the gate script). `gate_report.json` sha256 `fd9c6953…0fec`.

These bars are not demanding (protocol §10): G-HAND and G-OBJ were set after development coverage,
and the hand region's floor sits where this collector's palm-down grasp bottoms out, so they were
close to pass-by-construction; G-MOVE's 20 % is the plan's bar, against 68 % in the Stage 0 scale
probe. P1-PASS says the corpus covers the declared test workspace and contains object
interaction; it does not say the corpus is good play data for a world model. That is Phase 2's
question.

## 2. Run

- At `3ee2557` (clean tree), worktree `task085-run`, NVIDIA EGL on the RTX 5080
  (`NVIDIA Corporation / NVIDIA GeForce RTX 5080/PCIe/SSE2 / 4.6.0 NVIDIA 595.91.07`) under one
  `gpu_run.sh` lock, 12 workers, 2026-10-09, 1 025 s wall time. All 32 shards complete, none
  skipped or excluded; no re-run.
- 3 200 seeds (850000–853199): **3 194 stored**, 6 discarded as shorter than 50 commands, 0
  errors. Splits 2 874 train / 160 val / 160 test (5 / 5 per shard).
- `data/play-v1/` (git-ignored, on the SSD): 12.3 GB. Free disk after the run: 27 GB.
- Evidence: `~/develop/emai/evidence/task085-run/` (collector log, per-shard reports,
  `corpus.json`, `gate_report.json`), `SHA256SUMS` sha256 `5950abf6…12bd`.

## 3. Reported only

| Train episodes | n | Object moved | Lift | Grasp contact | Ended early | Mean commands |
| --- | --- | --- | --- | --- | --- | --- |
| scripted | 1 128 | 59.2 % | 17.5 % | 29.7 % | 18.5 % | 364 |
| perturbed | 1 042 | 60.1 % | 9.8 % | 24.1 % | 26.5 % | 349 |
| random | 704 | 44.5 % | 0 % | 4.0 % | 11.2 % | 379 |
| all | 2 874 | 55.9 % | 10.4 % | 21.4 % | 19.6 % | 362 |

- Lift = an object ≥ 3 cm above its start with grasp contact (thumb plus index or middle). Lifts
  per object (train): cube 119, can 111, apple 65, banana 4; moves: apple 614, can 478, cube 447,
  banana 420. The banana, lying flat, was almost never lifted.
- Skills (train): wander 3 490, pick_place 2 762 (of which 980 with a deliberate failure: miss
  286, drop 214, early_close 213, abort 136, no_close 131), push 1 272, poke 521.
- Early ends (all splits, 628): a right-hand joint passing the 5 rad/s velocity stop in 611
  (thumb_1 523, index 47, middle 37, thumb_2 4), an infeasible projection in 17. Episodes that
  end early are shorter, so post-grasp phases are somewhat under-represented.
- The four hand cells under 200 frames are all at x 0.24–0.32, y −0.10 to −0.02, z 0.18–0.26
  (high and close to the torso): 148, 171, 49 and 167 frames.
- Commands: the right-arm dimensions hit ±1 in 1.4–2.4 % of steps; the left arm is constant (0)
  and the left grasp constant (−1), as declared.
- Val and test (160 each): object moved 56.3 % / 54.4 %, lift 13.1 % / 10.0 %.

## 4. Caveats

- One collector design, one seed range, one renderer (NVIDIA EGL is not bit-reproducible: in
  Stage 0 one frame of 61 differed by one level in 6 pixels; physics was identical), one camera
  (onboard 112 px), right arm only, simulation only.
- The play policy is privileged and scripted, built around e9's pick; "random" means noisy
  scripted skills and random targets, not uniformly random commands.
- The corpus contains apple-to-plate carries, so it cannot back a claim that relies on the v1
  held-out (apple, plate) pairing.
- About a fifth of episodes end early on the embodiment's velocity stop; this is recorded, not
  hidden, and their stored frames are valid.

## 5. What changes

- TASK-085 closes with **P1-PASS** (R23.24). Phase 2 may preregister on `play-v1` train and val;
  test stays unopened until a later preregistration uses it.
- **R7 does not change** (R23.25).
