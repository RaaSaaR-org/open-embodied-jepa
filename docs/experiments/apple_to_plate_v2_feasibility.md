# Apple→Plate v2 feasibility scan (TASK-069): which minimal change lets a scripted expert rest the apple on the plate?

**Status (2026-09-28): development-only feasibility scan, no gate, nothing frozen.** It was run
under owner ruling R11 §4, which is recorded verbatim in
[`apple_resting_expert_v1.md`](apple_resting_expert_v1.md) §8. The output is a recommendation
for an `apple-to-plate-v2` task definition (§6). The owner rules on that definition. Only after
that ruling would a preregistered v2 expert gate follow, on fresh seeds.

**Scope.**
- This is privileged scripted engineering: the expert reads simulator truth. It is not a learned
  result, and no learned policy was trained or run.
- No corpus was read. The test split was not decoded.
- **Learned Apple→Plate is still 0 successes.**
- The v1 task, its scorer, its history and the 0/150 MVP benchmark are untouched.
  `simulation.py`, `task.py` and `scripted.py` are byte-identical to `main`. Every scene change
  below is applied at run time to one simulator instance's compiled model.

## 1. Question, seeds and measures

**Question (R11 §4).** Which minimal task or embodiment change makes a privileged scripted expert
rest the apple on the plate?

**Seeds.** Development seeds 50100–50199 (`v2_feasibility.DEV_SEEDS`). The plate-error directions
come from `default_rng(6840)`.
- Both were declared on the TASK-069 card before any TASK-069 episode.
- `check_seeds` refuses TASK-068's range (50000–50099), every range in
  `first_policy.FORBIDDEN_RANGES`, and the 46000–46999 block. The PR #90 reviewer re-checked this
  disjointness.
- run-1 used seeds 50100–50131. run-2 used 50132–50163, as a replicate of three cells.

**Conditions.** Plate exact, and a 1.0 cm plate error. The plate error is TASK-067's C0
perturbation, `first_policy_runtime.perturbed_truth`: the expert believes the plate is 1.0 cm
away, in a fixed direction per seed.

**The attempt.** The TASK-068 harness (`resting_expert.run_attempt`) runs, in order:
1. the reset;
2. the look;
3. the expert (at most 740 commands, or 745 for the collector, whose attempts therefore run 5
   steps past v1's 800-step cap);
4. a 60-step settle.

**Measures reported per cell.**
- **At rest** (`apple_at_rest_v0`, TASK-068 §1). On each of the last 20 settle steps, the apple is
  within 4 cm of the plate centre, supported, moving at ≤ 0.001 m/s, and not touching the hand.
- **Latched.** The v1 scorer's per-step success at any step of the attempt.
- **Guard stops.** Attempts ended by the embodiment's "measured joint velocity limit exceeded"
  guard. They count as not at rest.
- **Landing speed.** The apple's horizontal speed at its first contact with the plate base.

## 2. The candidate changes as run

| code | change | how it is applied |
|---|---|---|
| **b3a** | apple contact **condim 6**, keeping the scene's own declared friction `1 .01 .001`: torsional 0.01 m, rolling 0.001 m | `v2_feasibility.apply_apple_friction`, run-time edit of the apple geom. The v1 scene declares these coefficients on the apple and plate base but leaves `condim` at MuJoCo's default 3, where only sliding friction acts. MuJoCo uses the larger condim of the two geoms in a contact. |
| **b3b** | condim 6 with torsional 0.02 m and rolling **0.003 m** | as b3a |
| **b2** | plate centre uniform in x 0.30–0.37 m, y −0.25 to −0.04 m (base frame), redrawn until it clears the apple by the reset contract's 0.103 m. The apple draw is v1's. | `v2_feasibility.setdown_reset`. The box is chosen from the reach map (§3): on the map's 2 cm grid, every palm set-down point for it is reachable. |
| **b1** | the waist in the IK | feasibility from code and models only (§4), as R11 asks |
| b4 | per-finger hand control | excluded by R11 (new action contract) |

**The experts:**
- **d12:** TASK-068's `RestingPlaceExpert` with the hand pitched +0.45 rad. It was TASK-068's best
  release, 0/64 at rest under v1.
- **d1:** `RestingPlaceExpert` defaults.
- **collector:** TASK-067's `apple_collector_policy`.
- **setdown:** `RestingPlaceExpert` with the palm floor at 0.0354 m. That puts the carried apple
  about 0.5 cm above its resting height, over a b2 plate. `setdown-pitch` is the same with the
  +0.45 rad pitch.

**Nothing was tuned for b2.** Its experts are the TASK-068 expert with a lower floor.

## 3. The set-down reach region (for b2)

`scripts/measure_v2_reach.py`, run-1 at `a0f922b` (clean tree, 31 s, CPU;
`outputs/task069-reach/run-1/report.json`, sha256 `75905732…`).
- **Method:** the embodiment's own IK (`solve_ik`, 7 arm joints, the collector's palm-down
  rotation), with 20 fixed restarts of 300 iterations each.
- **Grid:** palm x 0.26–0.50 and y −0.30 to +0.04, in 2 cm steps.
- **Set-down height:** the resting apple's centre plus the carried palm-over-apple offset
  (4.44 cm) plus a clearance of 0, 0.5 or 1.0 cm. That is palm z 0.030, 0.035 or 0.040 m.

**What the map shows.** At palm z 0.035 m (0.5 cm clearance), the reachable set-down palm
positions are:
- x 0.26–0.32 for y −0.30 to −0.04;
- x 0.34–0.36 for y −0.26 (−0.28 at x 0.34) to −0.04;
- x 0.38 for y −0.22 to −0.06.

Nothing is reachable at x ≥ 0.40 at that height, or at y ≥ −0.02. At 1.0 cm clearance, x 0.40
adds y −0.16 to −0.12.

**The b2 box.** It puts the palm at x 0.285–0.355 and y −0.25 to −0.04. Every 2 cm grid point in
that range is reachable (checked on the grid only). The v1 palm set-down points (palm x ≥ 0.455
for plate x ≥ 0.47) are **about 7.5 cm beyond** the last reachable grid column (palm x 0.38); with
the 2 cm grid, the gap is 5.5–7.5 cm.

## 4. (b1) The waist: feasibility from code and models only

**What the MJCF actuates, from the model and `simulation.py`.**
- 43 actuators: both legs (12), the waist (yaw, roll, pitch), both arms (14), and both Dex3
  hands (14).
- The pelvis's free joint is removed (`simulation._scene`), so the base is fixed.
- The transport sends targets for all 43. `G1Embodiment._prepare_side` writes only arm and hand
  targets. The waist therefore stays at its reset targets under PD control, with gravity
  compensation.
- Waist ranges: yaw ±2.618 rad, roll ±0.52 rad, pitch ±0.52 rad.

**Could the IK use the waist while keeping `ee_delta_grasp_v0`?** Code reading; this was not
implemented or run.
- The 14-D action is a pelvis-frame palm delta (6 per side) plus two grasp scalars, so the
  schema does not name joints.
- Adding the waist to the right palm's IK would change only `G1Embodiment`: `arm_ids["right"]`,
  the IK Jacobian columns, the rate limits, and writing waist targets.
- **Consequences, from the code:**
  - The waist carries the torso, so it moves both shoulders. The left palm then moves unless its
    IK compensates. Today each side is solved independently from the same snapshot.
  - The onboard camera is mounted on `torso_link`, so every image would move with the waist.
    That changes the observation distribution the look and every corpus were built on.
  - The embodiment and its "fixed upright pelvis scene" assumption (`OracleManipulationPolicy`)
    would need a new version, even with the action schema unchanged.

**Would the extra reach cover the v1 plate region? Kinematics only.** This is `measure_v2_reach.py`
run-3 at `13dcef8` (clean tree, 45 s; sha256 `f7d2862e…`). It used 17 palm targets at z 0.0354 m:
TASK-068's eight place targets, plus the v1 plate region's corners and centre, less 1.5 cm in x.

| joints in the IK | palm-down pose residual, position | rotation | left palm moved from reset |
|---|---|---|---|
| 7 right-arm joints (today) | **4.65–8.13 cm** | 0.016–0.110 rad | 0 |
| + waist pitch only | **0.0** on all 17 (pitch 0.286–0.52 rad, several at the 0.52 limit) | 0.0 | 9.2–16.6 cm |
| + all three waist joints | 0.0 on all 17 | 0.0 | 0.9–61.4 cm |

- **With the waist pitch, the v1 plate region comes within set-down reach kinematically**, but
  with little margin: 6 of 17 solutions put the pitch at 0.513–0.52 rad, 5 of them at the 0.52
  limit.
- **Not modelled:** balance (the pelvis is fixed in the scene), waist torque (50 N·m limit),
  collisions of the leaning torso, and the camera's changed view.
- **Position only**, with any palm orientation: the arm alone misses by 2.88–8.14 cm.

## 5. Results

**Provenance.**
- **run-1:** `scripts/scan_v2_feasibility.py --cells all --count 32` on seeds 50100–50131, 2263 s,
  CPU, 8 workers. Report `outputs/task069-scan/run-1/report.json`, sha256 `7bf82da7…`.
  - Its `revision` field says `13dcef8`. The script read HEAD at the end, and two commits landed
    during the run. Both touched only `scripts/measure_v2_reach.py`, which the scan does not
    import (`git diff --stat a0f922b 13dcef8`). The run's output directory was created after
    `a0f922b` was committed. So the code the scan imports is identical at `a0f922b` and
    `13dcef8`, and the tracked tree was clean at the end. Uncommitted edits made and reverted
    during the run cannot be excluded, because the script then read `dirty` only at the end.
  - The script now records the revision at the start (`1fbd43d`).
- **run-2:** `--cells v1-d12,b3a-d12,b3b-d12 --start 32 --count 32` on seeds 50132–50163, at
  `1fbd43d` (clean tree, 602 s). sha256 `5929edd4…`.

Each cell is N = 32 at each level. "Guard" is guard stops. "Inside" means within 4 cm on all 20
window steps, at any speed. "Land v" is the q10 / q50 / q90 horizontal speed at first plate-base
contact, in m/s.

| cell | seeds | plate | **at rest** | latched | guard | inside | land v |
|---|---|---|---|---|---|---|---|
| v1-d12 (v1 control) | run-1 | exact / 1.0 cm | **0 / 0** | 32 / 31 | 0 / 0 | 1 / 3 | 0.071 / 0.091 / 0.102 ; 0.052 / 0.089 / 0.099 |
| v1-d12 | run-2 | exact / 1.0 cm | **0 / 0** | 32 / 26 | 0 / 0 | 3 / 1 | 0.071 / 0.089 / 0.106 ; 0.060 / 0.088 / 0.102 |
| v1-collector (v1 control) | run-1 | exact / 1.0 cm | **0 / 0** | 32 / 18 | 0 / 0 | 3 / 3 | — (the collector has no `open` phase) |
| **b3a-d12** | run-1 | exact / 1.0 cm | **32 / 26** | 32 / 32 | 0 / 0 | 32 / 26 | 0.056 / 0.076 / 0.083 ; 0.049 / 0.076 / 0.088 |
| **b3a-d12** | run-2 | exact / 1.0 cm | **32 / 29** | 32 / 32 | 0 / 0 | 32 / 29 | 0.048 / 0.074 / 0.086 ; 0.052 / 0.069 / 0.083 |
| b3a-d1 | run-1 | exact / 1.0 cm | **15 / 13** | 28 / 22 | 0 / 0 | 18 / 13 | 0.018 / 0.161 / 0.289 ; 0.054 / 0.181 / 0.309 |
| b3a-collector | run-1 | exact / 1.0 cm | **8 / 8** | 32 / 29 | 0 / 0 | 8 / 8 | — |
| **b3b-d12** | run-1 | exact / 1.0 cm | **32 / 32** | 32 / 32 | 0 / 0 | 32 / 32 | 0.043 / 0.049 / 0.058 ; 0.041 / 0.049 / 0.063 |
| **b3b-d12** | run-2 | exact / 1.0 cm | **32 / 32** | 32 / 32 | 0 / 0 | 32 / 32 | 0.034 / 0.047 / 0.061 ; 0.034 / 0.047 / 0.060 |
| b2-setdown | run-1 | exact / 1.0 cm | **0 / 0** | 27 / 22 | 7 / 14 | 12 / 13 | 0.127 / 0.283 / 0.311 ; 0.096 / 0.158 / 0.292 |
| b2-setdown-pitch | run-1 | exact / 1.0 cm | **0 / 0** | 32 / 24 | **32 / 29** | 0 / 0 | 0.009 / 0.021 / 0.038 ; 0.007 / 0.038 / 0.074 |
| b2b3a-setdown | run-1 | exact / 1.0 cm | **4 / 14** | 27 / 21 | 7 / 14 | 4 / 14 | 0.128 / 0.290 / 0.315 ; 0.045 / 0.124 / 0.277 |
| b2b3a-setdown-pitch | run-1 | exact / 1.0 cm | **0 / 2** | 32 / 24 | 31 / 28 | 0 / 2 | 0.006 / 0.021 / 0.037 ; 0.008 / 0.036 / 0.067 |
| b2b3b-setdown | run-1 | exact / 1.0 cm | **23 / 4** | 27 / 16 | 7 / 15 | 23 / 4 | 0.078 / 0.277 / 0.298 ; 0.032 / 0.102 / 0.265 |
| b2b3b-setdown-pitch | run-1 | exact / 1.0 cm | **0 / 3** | 32 / 25 | 32 / 29 | 0 / 3 | 0.007 / 0.017 / 0.035 ; 0.008 / 0.025 / 0.051 |

- There were no errors in either run.
- Every early stop was the joint-velocity guard, and every guard stop was in a b2 cell.
- Some cells have fewer landing-speed measurements than attempts: 27 of 32 in the b2 cells at
  1.0 cm, and 31 of 32 in b3a-d1 at 1.0 cm. In the b2 cells the missing attempts stopped on the
  guard before the apple touched the plate base. In b3a-d1 (seed 50126) the attempt completed,
  but the apple never touched the plate base after the opening and ended 115 cm from the plate.
  The quantiles are over the measured attempts.

**What the logs show directly.**
1. **b3 alone, with the d12 release, gives at-rest counts at or near 32/32.** The v1 plates, v1
   expert pick and v1 scorer radius are unchanged, and there were no guard stops.
   - **b3b-d12:** 64/64 at plate exact and 64/64 at 1.0 cm, over two blocks of 32 seeds.
   - **b3a-d12:** 64/64 at plate exact, and 55/64 at 1.0 cm (26 and 29).
   - **The rolling persists without b3.** The same expert and seeds without b3 (v1-d12) rest
     0/128. The median landing speed falls from 0.088–0.091 m/s under v1 to 0.069–0.076 m/s
     with b3a and 0.047–0.049 m/s with b3b, about half. It stays about 50× above the at-rest
     bar. The verdict flips because, with b3, the apple then stops rolling: its highest speed in
     the final window is ≤ 6.5e-4 m/s under b3a (cell medians 2.9–3.5e-4 m/s) and about 3e-11
     m/s under b3b.
   - The 9 b3a attempts that are not at rest all fail the 4 cm radius, not the speed bar.
   - condim 6 also changes the apple's contacts with the fingers during the release. The scan
     does not separate that from the rolling resistance after landing.
2. **b3 does not rescue the other experts.** The collector gets 8/32 and 8/32; d1 gets 15/32 and
   13/32. Both release higher or with more speed. d1's landing-speed median is 0.16–0.18 m/s.
   The collector drops the apple about 15.5 cm (measured in TASK-067's landing diagnosis, not in
   this scan; the collector has no `open` phase for this scan's release fields).
3. **b2 alone gives 0 at rest.** With the set-down expert, guard stops end 7–32 of 32 attempts
   per cell. Without the pitch, the landing speeds are high (a median 0.16–0.28 m/s), even though
   the apple is low when the hand opens.
   - **The guard is not examined** beyond one exploratory trace on seed 50100 (b2-setdown, not
     committed). There, during the lowering and the opening, the index, middle and thumb links
     touched the plate base and rim.
   - **Inference, not measured:** with the palm at set-down height the open fingers reach the
     plate, and that contact drives the finger joints past the 5 rad/s guard, or pushes the apple.
4. **b2 + b3 is worse than b3 alone** in every configuration run: 0–23 at plate exact and 2–14 at
   1.0 cm. The b2 guard stops and fast landings carry over.

**What is not shown.**
- Whether the b3 values are physically right for an apple. This is inference, not measured.
  MuJoCo's rolling coefficient is a lever arm in metres. 0.001–0.003 m on a 2.7 cm radius sphere
  corresponds to a dimensionless rolling-resistance coefficient of about 0.04–0.11. Whether that
  suits an apple on a plate is an unsourced assumption, not a measurement, and a real apple is
  not a sphere.
- **The torsional value.** b3b doubles the torsional coefficient to 0.02 m, as large as a 2 cm
  contact patch on a 2.7 cm-radius sphere. That doubling is not justified, and the scan did not
  separate its effect from the rolling increase.
- **b3b may be more forgiving than a real apple.** Under b3b the apple is effectively stuck once
  it lands: about 3e-11 m/s in the final window, in every attempt.
- Robustness to other seeds, levels (1.5 cm was not run) or experts, beyond the 64 attempts per
  level above.
- Anything learned. A scripted expert that rests the apple is a prerequisite for demonstrations,
  not evidence that a policy can learn them.

## 6. Recommendation for `apple-to-plate-v2` (for the owner's ruling)

**Recommended v2 definition: v1 plus b3, and nothing else.**
- **The scene change:** the apple geom's contact dimensionality becomes 6, with rolling and
  torsional friction.
- **Unchanged:** the robot, the fixed pelvis, the action schema `ee_delta_grasp_v0`, the plate
  geometry and distribution, the 4 cm radius, and the look.

**Which values** (the owner's choice):
- **b3b** (rolling 0.003 m, torsional 0.02 m): 128/128 at rest over both levels on development
  seeds with d12.
- **b3a** (the scene's own declared 0.001 / 0.01, only enabled): 119/128. At 1.0 cm it gave 55/64,
  which is below a 28/32 per-block bar in one of the two blocks (26/32).
- **Recommendation: b3b.** Declare it as a new named scene constant (for example
  `apple_contact_v2 = condim 6, friction 1 / 0.02 / 0.003`), not an edit of the v1 scene.
- Its physical plausibility is inferred, not measured (§5).

**Scorer.** v2 would score with `apple_at_rest_v0` as its success. The latched v1 scorer would be
reported beside it.

**Not recommended:**
- **b2 (moved plates):** 0 at rest alone, heavy guard stops, and it would move the task's
  geometry.
- **b1 (waist in the IK):** kinematically sufficient only with the waist pitch at or near its
  limit. It changes the embodiment and moves the onboard camera, so every image changes. Balance
  and torque are unmodelled.

**What would follow, if the owner rules v2 = v1 + b3b:**
- a v2 scene variant in code (a new, separately named scene option; the v1 scene is unchanged);
- then a preregistered expert gate on fresh seeds, with ≥ 28/32 at rest at plate exact and at
  1.0 cm, with d12 frozen as the expert.

**Learned Apple→Plate is still 0 successes.**

## 7. Reproduction

```sh
uv run --no-sync pytest tests/test_v2_feasibility.py
uv run --no-sync python scripts/measure_v2_reach.py --output outputs/task069-reach/<new>
uv run --no-sync python scripts/scan_v2_feasibility.py --cells all --count 32 \
    --output outputs/task069-scan/<new>
uv run --no-sync python scripts/scan_v2_feasibility.py --cells v1-d12,b3a-d12,b3b-d12 \
    --start 32 --count 32 --output outputs/task069-scan/<new>
```

Both scripts refuse to overwrite an output directory. The artifacts are git-ignored.
- `measure_v2_reach.py` run-1 (`a0f922b`) and run-2 (`d9920f1`) are earlier versions of the same
  tool, without the palm-down and the pitch-only parts. Their set-down grid is the one this
  document cites (sha256 `75905732…` and `483a606c…`).
- **Commit SHAs in this document are branch SHAs.** A rebase merge rewrites them. The pre-merge
  head will be tagged before the merge.
