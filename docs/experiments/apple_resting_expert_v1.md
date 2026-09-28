# Apple→Plate resting expert v1: at-rest metric, descent diagnosis, development log (TASK-068)

**Status (2026-09-28): development stopped on a blocker, before any preregistration.** No
design reached a single at-rest success on the development seeds (§5). This document is **not**
a preregistration. No gated seeds are declared or spent, and no gated run was started. The
owner's ruling is needed (§6).

This is engineering on a **privileged scripted controller**. No learned policy was trained or
run, no corpus was read, and the test split was not decoded. Scripted-expert results are not
learned results. **Learned Apple→Plate is still 0 successes.**

Card: `.mc/tasks/todo/TASK-068-…md`. Background: `docs/DECISIONS.md` (2026-09-28, TASK-067), and
[`apple_first_policy_v1_landing_diagnosis.md`](apple_first_policy_v1_landing_diagnosis.md).

## 1. The at-rest check (`apple_at_rest_v0`)

**Code.** `at_rest.apple_at_rest` (a pure function over per-step arrays),
`at_rest.AppleAtRestCheck` (records simulator truth once per executed step) and
`at_rest.AtRestThresholds`, in a new module `src/embodied_jepa/at_rest.py`. It is a separate,
end-of-episode check. `AppleToPlateTask`, its thresholds (`tabletop_proxy_v0`), its 4 cm radius
and every past result are unchanged; `task.py` and `scripted.py` are byte-identical to `main`
(earlier manifests pin their hashes). Tests: `tests/test_at_rest.py`.

**Definition.** The attempt runs the expert's commands, then a settle of **60 steps** (3.0 s at
20 Hz) with the arm commanded still and both hands open. At rest means that on **every one of
the final 20 steps** (1.0 s, all inside the settle):

| condition | bar | source |
|---|---|---|
| apple centre to plate centre, world xy | ≤ 0.04 m | the default scorer's radius |
| apple height minus resting height (`container_surface_z + object_support_height`) | ≤ 0.012 m | the default scorer's support tolerance |
| apple linear speed | ≤ **0.001 m/s** | new; see below |
| apple–hand contact | none | as the default scorer |

**Why the speed bar is 0.001 m/s, not the scorer's 0.1 m/s.** The apple is a sphere with
MuJoCo's default condim-3 contact, which has no rolling resistance. A rolling apple does not
slow down on the plate. The descent diagnosis measured this directly (§4, part 3):
- an apple set on the plate with no velocity stays below 1e-10 m/s;
- one rolling at 0.002 m/s still rolls at 0.002 m/s 200 steps (10 s) later;
- ones rolling at 0.005 and 0.01 m/s roll unchanged until they reach the rim.

So a 0.1 m/s bar, or even 0.01 m/s, accepts an apple that is still rolling toward the rim. At
0.001 m/s an apple can drift at most 1 mm over the 1 s window.

**It is not a latch.** The window is the last 20 steps only, and every step must pass. A
rolling apple that crosses the 4 cm disc fails (tested).

**Reported beside it:** the latched scorer, i.e. `AppleToPlateTask`'s per-step `success` at any
step of the attempt, the settle included. This is how C0 and the TASK-067 probes counted, since
they stopped at the first success. An attempt here is at most 800 steps, C0's cap; the collector
baseline's is 805.

## 2. Seeds

- **Development: 50000–50099** (`resting_expert.DEV_SEEDS`), with plate-error directions from
  `default_rng(6830)`. The range was fixed in code before the first logged development run.
  Every wide-jitter reset in TASK-068 so far used a seed in this range, including the
  single-seed exploration. The reach map and rolling check (§4) use a fixed scene, not a reset
  seed.
- **The search, 2026-09-28.** Every 5-digit number from 40000 to 60000 in `docs/`,
  `benchmarks/manifests/`, `scripts/`, `src/`, `configs/`, `tests/` and `.mc/tasks/` was listed,
  along with every `range(...)` seed literal in code and the `data/` manifests.
  - Numbers from 50000 to 59999 occur only as non-seed values: step counts, CPU seconds, a PID,
    and digits of decimals.
  - `check_seeds()` also refuses every TASK-067 range and cohort, and the whole 46000–46999
    block (tests: `tests/test_resting_expert.py`).
- **The suggested gated range 47000–47099 is not free.** It lies inside `apple_look_v1_corpus`
  (47000–47199; `first_policy.FORBIDDEN_RANGES`, `look_corpus.FROZEN_SEEDS`).
- **No gated range is declared.** One would be declared, fresh and checked, in a preregistration
  PR (§6).

## 3. The harness

`resting_expert.run_attempt` performs one attempt:
1. the TASK-047 wide-jitter reset (`scripts/evaluate_apple.py:wide_reset`);
2. the TASK-064 look (`first_policy_runtime.reset_and_look`);
3. the expert, built from the reset truth with the plate xy shifted by a fixed offset. This is
   TASK-067's C0 plate perturbation, `first_policy_runtime.perturbed_truth`, reused unchanged:
   the size is the level, and the direction is fixed per seed;
4. the expert's commands, clipped to the configured bounds and projected by the embodiment, as
   in C0;
5. the 60-step settle.

**Budget.** The expert gets at most **740 commands** after the look. With the settle this is
800 steps, TASK-067's `MAX_POLICY_STEPS`. The one exception is the collector baseline below,
which keeps its own 745 commands.

**Guards.** A guard refusal or an infeasible command ends the attempt. It is recorded as
**incomplete** and counts as not at rest.

**Every step records:**
- the phase, the command and the applied action;
- the palm pose;
- the apple's position and its linear and angular velocity;
- contact counts by pair: apple–hand, apple–plate base, apple–plate rim, hand–plate,
  apple–table;
- the scorer's success and distance.

**Device.** CPU, 8 spawned worker processes (`scripts/develop_resting_expert.py`).

## 4. Descent diagnosis (no gate)

**Question.** Why did TASK-067's place-then-open redesign not lower the palm? Its lowest point
was 0.113 m against a 0.035 m target, and it was still descending at about 0.44 mm/step with
the command saturated.

**What was run.** `scripts/diagnose_resting_descent.py`, **run-3** at `a8a579c` (clean tree,
205 s, CPU, one process). Report: `outputs/task068-descent-diagnosis/run-3/report.json`, sha256
`1f346d0f…f5a5`. Its replay, reach and rolling sections are identical to run-2's (`5345ef9`,
sha256 `31477e40…95ef`), which ran the same script before the code move below.
- **Part 1.** TASK-067's `PlaceThenOpen`, unchanged, on development seeds 50000–50007, plate
  exact. During its 100-command `lower_closed` phase, each step re-runs the embodiment's own
  acceptance check (`G1Embodiment._prepare_side`) on the full command, and the embodiment's IK
  (`solve_ik`) on the accepted step.
- **Part 2.** A palm-down reach map: the lowest height the embodiment's IK reaches with the
  collector's palm-down rotation, over a grid of palm xy, with 20 fixed random restarts of 300
  iterations each. Then, for each replayed place target, the smallest residual that a
  position-only IK reaches with any palm orientation, using the 7 right-arm joints within their
  limits.
- **Part 3.** The rolling check (§1).
- **run-1 is superseded, not deleted.** run-1 (`2764656`, sha256 `1916e93c…`) compared float32
  joint targets with float64 measured joints at 1e-9. Its hold count (0) was therefore wrong by
  construction. Its other fields match run-2 and run-3.

**What the logs and code show (8/8 attempts):**

| quantity | value |
|---|---|
| place target's distance from the right shoulder (base frame, shoulder at (0.000, −0.100, 0.292) m) | 0.522–0.554 m |
| smallest position-only IK residual to the place target, any palm orientation | **3.1–6.2 cm** |
| lowest palm-down height reachable at palm xy (0.47–0.49, −0.09) m | 0.16–0.22 m (map: 0.16 at x 0.47, 0.18 at 0.48, 0.22 at 0.49) |
| steps (of 100) where the full `lower_closed` command was refused, and why | 85–96 "right IK failed"; 1–6 "right joint rate limit"; the rest accepted (0–14, all early in the phase) |
| projection's accepted factor from step 11–21 of the phase onward | 1/8 of the command, on every remaining step (79–89 per attempt) |
| accepted steps whose IK returned the seed (the measured joints) unchanged | **every step from step 11–21 onward** (79–89 per attempt); each such step is ≤ 1.30 mm, inside the IK's 1.5 mm position tolerance. Every accepted step that moved the joints was ≥ 1.72 mm |
| palm descent over the last 20 steps | 0.38–0.44 mm/step (lowest palm z 0.084–0.119 m) |
| smallest right-arm joint margin to a joint limit | 0.69–0.74 rad |
| hand contacts other than with the apple, during `lower_closed` | 0 steps |

**Mechanism, as far as the logs and code show it:**

1. **The place pose is out of the arm's reach.** No right-arm configuration within the joint
   limits puts the palm within 3.1 cm of any of the eight place targets, whatever the palm
   orientation. With the palm down, the lowest reachable palm over the plate's centre region is
   0.16–0.22 m.
   - The pelvis is fixed: `simulation._scene` removes its free joint.
   - The IK uses the 7 arm joints only (`G1Embodiment.arm_ids`).
   - The waist and leg joints are actuated but not part of the `ee_delta_grasp_v0` action; they
     stay at their reset targets.
2. **Near the reach boundary the IK refuses the full step; the backtracking accepts a no-op.**
   - From step 11–21 of the phase onward, the full step toward the unreachable target is
     refused (almost always "right IK failed"). The projection (`project_candidates`) then
     backtracks through 1, 1/2, 1/4 and 1/8.
   - A 1/8 step is smaller than the IK's 1.5 mm position tolerance. So the IK "converges" at its
     seed and returns the measured joints unchanged: a hold, not a descent.
3. **The observed 0.4 mm/step descent happens during these holds.** The joint targets are
   re-set to the measured joints every step, yet the palm keeps sinking.
   - **Why it sinks is not identified by these logs.** The logs show that it sinks, and that the
     commanded joints equal the measured ones.
   - Inference, not measured: a load the PD's gravity compensation does not include, such as the
     held apple. Re-targeting to the measured joints each step would then ratchet the arm down.

**Candidates ruled out by the logs or code:**
- **Joint limits:** the smallest margin is 0.69 rad.
- **Collision:** no hand contact other than with the apple.
- **The workspace clamp:** z ≥ −0.12 m and x ≤ 0.55 m in `configs/g1_sim_action.json`; the
  target lies inside both.
- **The per-step command clamp:** the full command is refused by the IK, not by the clamp.
- **The command budget:** more commands would not reach a pose that the arm cannot reach.

**The reach limit also applies to the gated conditions.** The plate centre ranges over x
0.47–0.51 m, with a further ±1.5 cm of plate error. The held apple sits about 4.4 cm below the
palm (TASK-067's logs) and about 1.1 cm ahead of it (d1's logs, at the opening). So the apple
cannot be set down on the plate from any reachable palm-down pose. It must be released from at
least about 8 cm above its resting height: 8.0–23.2 cm in the development runs (§5).

## 5. Development log (every design, every count)

`scripts/develop_resting_expert.py`, designs in its `DESIGNS` dict. The expert is
`resting_expert.RestingPlaceExpert`. Its tests are in `tests/test_resting_expert.py`.
- **Pick:** the collector's own four pick phases.
- **Transfer:** to a release pose over the believed plate centre, with the palm 1.5 cm behind so
  the held apple is over the centre.
- **Release pose height:** the lowest point on a declared reach sphere about the shoulder
  (radius 0.485 m, chosen from the reach map), within 0.10–0.26 m, and pulled back in x if the
  xy is beyond reach.
- **Then:** lower (50 commands), steady (30), open on a ramp (50), clear (30), and retreat
  8 cm up (60), for 725 commands in total.

**Provenance.**
- **The table shows run-3**, at `a8a579c` on a clean tree: the code as it is in this PR.
- **run-2** was at `2764656`, a clean tree, before the at-rest check and the expert moved out
  of `task.py` and `scripted.py` (a move with no behaviour change).
- **run-1** used the same parameters on the uncommitted working tree during development.
- **Every run-3 count equals its run-2 and run-1 counts.**
- **Report sha256** (`outputs/task068-dev/<design>-run-3/report.json`):

  | design | sha256 |
  |---|---|
  | d1 | `bbe8e77a…` |
  | d2 | `e510da6b…` |
  | d3 | `ce864e00…` |
  | d4 | `3d7a6079…` |
  | d5 | `d519ee2a…` |
  | d6 | `b2c98d9e…` |
  | d7 | `08c532e5…` |
  | d8 | `d669923f…` |
  | d9 | `f46ecbce…` |
  | d10 | `b532a193…` |
  | d11 | `0bb0460b…` |
  | d12 | `0075e227…` |
  | collector | `0b118f85…` |

Columns: seeds are the first N development seeds; "incompl." is a guard stop before the settle
ended; "inside" means within 4 cm on all 20 window steps, whatever the speed; "window v" is the
highest apple speed in the window; "land v" is the apple's horizontal speed at its first contact
with the plate base; "drop" is the held apple's height above its resting height when the hand
starts to open.

| design | change from d1 | plate | N | incompl. | **at rest** | latched | inside | window v, min / median (m/s) | land v, median (m/s) | drop (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| d1 | (defaults: opening ramp 0.04) | 0 | 16 | 0 | **0** | 13 | 3 | 0.0020 / 0.0128 | 0.140 | 9.3–22.7 |
| d2 | ramp 1.0 (rate-limited by the embodiment) | 0 | 16 | 0 | **0** | 13 | 2 | 0.0049 / 0.0125 | 0.191 | 9.3–22.7 |
| d3 | ramp 0.2 (identical to d2: same rate limit) | 0 | 16 | 0 | **0** | 13 | 2 | 0.0049 / 0.0125 | 0.191 | 9.3–22.7 |
| d4 | ramp 1.0, hand pitched −0.3 rad | 0 | 16 | 0 | **0** | 13 | 1 | 0.0047 / 0.0135 | 0.151 | 9.2–23.2 |
| d5 | ramp 1.0, pitch +0.3 rad | 0 | 16 | 0 | **0** | 16 | 0 | 0.0139 / 0.0217 | 0.241 | 9.2–22.6 |
| d6 | ramp 1.0, hand commanded 6 cm back while opening | 0 | 16 | 0 | **0** | 16 | 0 | 0.0082 / 0.0157 | 0.252 | 9.3–22.8 |
| d7 | ramp 1.0, hand commanded 6 cm forward while opening | 0 | 16 | 0 | **0** | 7 | 0 | 0.0029 / 0.0186 | 0.161 | 9.3–22.7 |
| d8 | ramp 0.04, hand commanded 6 cm back, slowly, while opening | 0 | 16 | 0 | **0** | 15 | 2 | 0.0043 / 0.0143 | 0.215 | 9.3–22.8 |
| d9 | ramp 1.0, grasp closure 0.5 | 0 | 16 | **16** | **0** | 0 | 0 | — | — | — |
| d10 | ramp 1.0, palm 1.5 cm further back at the pick | 0 | 16 | **13** | **0** | 0 | 0 | 0.297 (3 complete, apple not grasped) | — | — |
| d11 | ramp 1.0, palm 1.5 cm further forward at the pick | 0 | 16 | 6 | **0** | 9 | 1 | 0.0055 / 0.0122 | 0.231 | 9.3–22.7 |
| d12 | pitch +0.45 rad (best in the scan below) | 0 | 32 | 0 | **0** | **32** | 2 | 0.0027 / 0.0092 | 0.089 | 9.3–22.7 |
| d12 | same | 1.0 cm | 32 | 0 | **0** | **29** | 3 | 0.0029 / 0.0111 | 0.087 | 8.0–22.7 |
| collector | baseline: `apple_collector_policy`, 745 commands | 0 | 32 | 0 | **0** | 32 | 3 | 0.0037 / 0.0096 | — | — |
| collector | same | 1.0 cm | 32 | 0 | **0** | 13 | 1 | 0.0030 / 0.0177 | — | — |

- Every incomplete attempt ended on the guard "measured joint velocity limit exceeded".
- **d12 illustrates the latch.** On development seeds its latched counts, 32/32 and 29/32, would
  meet a 28/32 bar counted with the latch. At rest it is 0/32 at both levels.
- **The baseline is not the TASK-067 measurement.** These are development seeds and directions,
  with this check. They do not re-measure TASK-067's 4/32, which used the scorer's per-step
  success at the final step, not this check.

**Single-seed release scan** (`scripts/scan_resting_release.py`, seed 50003, plate exact, at
`a8a579c`, one attempt per entry; `outputs/task068-dev/scan-50003-run-2/scan.jsonl`, sha256
`4e7de986…`; run-1 at `415d89d`, `a82620b1…`, gave the same values).
- **What varied:** the hand orientation during lower, steady, open and clear (a rotation about
  the base axes before the palm-down rotation), and the opening ramp.
- **What it shows:** how the apple leaves the hand. In the table, *v* is its horizontal velocity
  just after its last hand contact, *ω_y* its spin about y, and "land v" its horizontal speed
  when it first touches the plate base.

| entry | v (m/s) | ω_y (rad/s) | landing, from plate centre (cm) | land v (m/s) |
|---|---|---|---|---|
| ramp 0.04 (d1) | (+0.190, −0.006) | +13.0 | (+4.9, −0.1) | 0.235 |
| ramp 0.01 | (+0.146, +0.003) | +12.6 | (+4.4, 0.0) | 0.220 |
| pitch −0.9 / −0.6 | (+0.22, −0.01) / (+0.23, 0.00) | +12.3 / +12.9 | (+7.8, −0.7) / (+7.9, −0.4), off the plate | 0.465 / 0.460 |
| pitch +0.45 | (+0.020, +0.018) | +8.2 | (−0.3, +0.3) | 0.095 |
| pitch +0.6 | (−0.027, −0.017) | −4.9 | (−2.9, −0.1) | 0.076 |
| pitch +0.75 / +0.9 / +1.1 | (−0.11, −0.01) / (−0.06, 0.00) / (−0.12, 0.00) | −17.3 / −11.3 / −17.9 | (−4.9) / (−4.2) / (−5.7) in x | 0.199 / 0.218 / 0.024 |
| ramp 0.01, pitch +0.75 | (−0.106, −0.008) | −15.7 | (−5.8, −0.3) | 0.114 |
| ramp 1.0, pitch +0.6 / +0.9 | (+0.119, 0.0) / (+0.081, −0.004) | +13.5 / +11.9 | (−2.0, +1.0) / (−4.5, −0.7) | 0.126 / 0.130 |
| roll +0.5 / −0.5 | (+0.03, −0.11) / (+0.04, +0.10) | +10.5 / +9.6 | (+2.1, +0.1) / (+2.7, −0.2) | 0.159 / 0.177 |
| yaw +1.57 / −1.57 | (+0.14, +0.07) / (+0.12, −0.05) | +13.7 / +11.1 | (+1.9, +2.2) / (+2.0, −2.7) | 0.195 / 0.201 |

Every scan entry ended not at rest.

**What the development logs show directly.**
1. **The held apple rests on the thumb tip.** In the d1 grasp at seed 50001 (an exploratory
   contact print, not a committed script), the thumb tip
   (thumb_2) touches the apple 2.5 cm below and 0.6 cm behind its centre. Index and middle touch
   it 1.4 cm ahead, 2 cm to each side and 1.1 cm above; the wrist link is on top.
2. **Opening rolls the apple off the thumb.** In the traces examined (d1 and d2 at seed 50001;
   the scan's ramp 0.04 and 0.01 at seed 50003), it leaves the hand at 0.15–0.24 m/s forward,
   with a forward spin of 13–18 rad/s. Across d1–d8 the median horizontal speed at landing is
   0.14–0.25 m/s.
   - Pitching the hand shifts that velocity from forward to backward. Between +0.45 and
     +0.6 rad both the velocity and the spin change sign, and in the scan the two entries
     either side of those zero crossings (+0.45 and +0.6 rad) still land at 0.08–0.10 m/s.
     Nothing shows both reaching zero at one pitch, and d12 (+0.45 rad, 64 attempts) landed
     at a median 0.09 m/s.
3. **The apple never stops inside 4 cm.** It rolls without slowing (§1), reaches the rim, and
   rebounds. In every design, every complete attempt still moved at ≥ 0.002 m/s in the final
   window. No design cell had more than 3 attempts (of 16 or 32) inside 4 cm on all 20 window
   steps, at any speed.

**What-if, not a task change** (same scan script, seed 50003, `a8a579c`;
`outputs/task068-dev/scan-50003-whatif-run-2/scan.jsonl`, sha256 `4ee8734a…`; run-1 at
`627c030`, `0cb333fe…`, gave the same values).
- **Question:** would setting the apple down help, if the release point were within reach?
- **Setup:** the expert believes the plate is at (0.36, −0.10), a reachable table spot. The real
  plate does not move. The palm's floor is varied from 0.026 to 0.10 m.
- **Palm floor 0.026–0.055 m** (apple 1.5–2.3 cm above its resting height when the hand
  opens):
  - the apple leaves the hand at under 0.02 m/s;
  - after its last hand contact it moves 0.1 cm in three of them and 2.5 cm in one (not
    measured in one);
  - **every one of these attempts ended on the joint-velocity guard after the opening
    started.** The cause of the guard stop was not examined.
- **Palm floor 0.07 and 0.10 m** (apple 3.8 and 6.8 cm above its resting height): the attempts
  completed, and the apple rolled away (20 cm and 193 cm).
- **Scope:** one seed, and over the table, not the plate. It suggests that a low set-down can
  give a clean release, and that this hand at this height meets the guard. It establishes
  neither.

## 6. Blocker, and what the owner could rule

**Blocker.** Within the frozen task, plate geometry, embodiment and action contract, no design
tried produced an at-rest success on development seeds: **0 of 269 complete attempts** (304
attempts, 35 ended early on the guard) **across 15 design-by-level cells**, plus the scan. The
logs point to three facts that compound:
1. **Reach.** The fixed-pelvis right arm cannot bring the palm within 3 cm of a place pose over
   the plate, so the apple is released 8–23 cm above its resting height.
2. **Release.** The Dex3 synergy rolls the apple off the thumb as it opens, at about
   0.15–0.25 m/s with forward spin. Orientation changes moved the velocity and the spin but
   did not bring any development cell's median landing speed below 0.08 m/s.
3. **Physics.** The apple is a condim-3 sphere, so a rolling apple does not slow down on the
   plate. Most attempts end rolling at or along the rim, 4.5–4.6 cm from the centre; a few
   leave the plate (d4 and d7).

**Why a preregistration is not opened.** A 28/32 at-rest gate on a design that scores 0/32 in
development would be a foregone failure. A preregistration needs a frozen design, and none is
viable.

**Options for the owner.** None is started.
- **(a) Record the failure formally.** Preregister d12 (or d1) as frozen, on a fresh gated range
  (for example 50500–50531, to be checked in the PR), and run the gated evaluation to put the
  failure on the record.
- **(b) Close TASK-068 on the development finding.** Open a follow-up that changes one frozen
  element, each needing its own preregistration. None of these is tested; the what-if in §5
  suggests a low set-down can release cleanly but met the joint-velocity guard.
  - **(b1) Embodiment.** Let the expert, and the action contract, use the waist to lean forward,
    which might bring a set-down pose over the plate within reach. That needs a new
    action-schema version.
  - **(b2) Task distribution.** Plates within the arm's set-down reach.
  - **(b3) Scene physics.** Rolling and torsional friction on the apple (condim 6), so a rolling
    apple slows down as a real one would. This is a task change.
  - **(b4) Hand.** Per-finger control, or a release that does not roll the apple off the thumb.
    That is an action-contract change.
- **(c) Loosen the at-rest speed bar. The data say this would not rescue the gate.** Even
  ignoring speed, no cell had more than 3 attempts inside 4 cm on all 20 window steps (d12: 2
  and 3 of 32; the collector: 3 and 1 of 32).

Learned Apple→Plate is still 0 successes. No learned policy was trained or evaluated in
TASK-068.

## 7. Reproduction

```sh
uv run --no-sync pytest tests/test_at_rest.py tests/test_resting_expert.py
uv run --no-sync python scripts/diagnose_resting_descent.py \
    --output outputs/task068-descent-diagnosis/<new>
uv run --no-sync python scripts/develop_resting_expert.py --design d12 --count 32 --levels 0,1.0 \
    --output outputs/task068-dev/<new>
uv run --no-sync python scripts/scan_resting_release.py --seed 50003 --grid '<json list>'
```

Each script refuses to overwrite an existing output directory. The attempts are deterministic:
every run-3 count equals its run-2 and run-1 counts, and both scans reproduced their run-1
values, which in turn matched the exploratory values.
