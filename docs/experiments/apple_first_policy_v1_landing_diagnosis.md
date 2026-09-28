# Apple→Plate first learned policy v1: landing diagnosis and one redesign (TASK-067, owner ruling R9)

**This is diagnostic work with no gate, apart from the redesign's pass bar in §B.** It is scripted
and privileged only: no learned policy, no corpus read, no test-split decoding. Owner ruling R9 is
recorded verbatim in [`apple_first_policy_v1_release_probe.md`](apple_first_policy_v1_release_probe.md)
§6. **Learned Apple→Plate is still 0 successes.**

## A. Diagnosis (R9 step 2)

**What was run.** `scripts/diagnose_first_policy_landing.py` at `b664ff6`, on a clean tree:
- the **current** collector (`apple_collector_policy`, `transfer_x_shift` −0.035);
- the R8 probe's seeds 46800–46831. Those seeds are spent, and R9 allows reusing them for
  diagnosis only;
- the plate exact, and a 1.0 cm plate error in the probe's `default_rng(6810)` directions;
- after the look, the collector's full 745 commands, then 60 settle steps (hands open, no
  motion). The attempt does **not** stop at the scorer's success, so the whole post-grasp
  trajectory is logged.

That is 64 attempts, 0 errors, 209 s on 8 CPU workers. The report is
`outputs/task067-landing-diagnosis/run-1/report.json` (sha256 `d9966669…1f46`), with per-step
logs in `attempts/*.npz`. Each step records the phase and command, the palm pose, the apple's
pose and velocities, the plate's pose, every contact by pair type, and the scorer's reading.

**What the logs show directly, the same way at both levels (32/32 attempts each):**

| quantity | plate exact | 1.0 cm plate error |
|---|---|---|
| apple drop from the release pose to its rest height | 15.4–15.5 cm (every attempt) | 15.4–15.5 cm |
| apple speed around first plate contact, median | 1.28 m/s | 1.28 m/s |
| attempts with apple–rim contact after release | 32/32; first rim contact a median 16 steps (0.8 s) after release begins | 32/32; median 16 steps |
| apple–hand contact after the apple first touches the plate base | 0 steps (all attempts) | 0 steps |
| hand–plate contact | 0 steps | 0 steps |
| apple–table contact after release | 0 attempts | 0 attempts |
| plate body motion (position and orientation, maximum) | 0.0 (the plate is a fixed body) | 0.0 |
| **scorer success at any step** (what the probe and C0 counted, since they stop at the first success) | **32/32** | **21/32** |
| scorer success still true at the end of the settle | **4/32** | **1/32** |
| attempts whose success was transient (true at some step, false at the end) | 28 | 20 |
| final distance from the plate centre, attempts not in success at the end | 4.52–4.60 cm | 4.53–4.60 cm |
| apple speed at the end of the settle, maximum | 0.012 m/s | 0.018 m/s |

- **The two "scorer success at any step" counts reproduce the R8 probe's counts** (32 and 21) on
  the same seeds and directions.
- **Representative trace, seed 46801.** The release begins at step 465, and the apple is at z
  0.933 m. It hits the plate base by step 478 at 1.27 m/s and touches the rim at step 481. It then
  crosses the plate, back and forth, at 0.08–0.17 m/s. While passing within 4 cm at under
  0.1 m/s for 0.15 s it latches the scorer's success (step 502), and it leaves the disc again.
  It ends against the rim at 4.5–4.6 cm, still rolling slowly.

**The ring is the rim, as a matter of geometry.** The plate's rim is 16 capsules on a 6.7 cm
radius, with a capsule radius of 0.4 cm, centred 0.9 cm above the plate body. An apple of radius
2.7 cm resting on the base and touching the rim has its centre 4.61 cm from the plate centre at a
chord midpoint, and 4.74 cm at a segment end. The observed resting ring (4.52–4.60 cm here, 4.53–4.62 cm in the R8 probe) matches
the chord-midpoint contact. The logs also record apple–rim contacts at the end on some attempts.
The scorer's 4 cm radius lies inside that ring, so **an apple resting against the rim always
fails.**

**Mechanism: identified from the logs.**
- The collector opens the hand at the transfer height (`release_high`). The apple falls about
  15.5 cm onto the plate, bounces to the rim within about 0.8 s, and rolls around the plate.
- It comes to rest against the rim, 4.5–4.6 cm from the centre, outside the scorer's 4 cm radius.
- The scorer's "success" is usually a **transient** crossing of the 4 cm disc while the apple
  rolls slowly. That crossing latches the success in the probe and in C0, both of which stop at the
  first success.

**What the logs rule out.**
- The hand dragging the apple: there is no apple–hand contact after the apple first touches the
  plate.
- The plate being displaced: the plate is a fixed body and did not move.
- A scorer-frame problem: the plate position is fixed, and the distance is computed in world xy.
- The "released 3.5 cm short" explanation, which was already contradicted by the R8 probe.

**Why a 1.0 cm plate error lowers the count (inference).** The error changes where the rolling
apple crosses the disc and how slowly, so it changes whether a transient 0.15 s window inside 4 cm
occurs. The logs do not isolate this further.

**A consequence for the record, not a finding about learning.** Scripted-collector "successes"
under this scorer include transient crossings. Inference, not measured: the same may apply to this
corpus's collection (103/200 root successes in `apple-look-v1`).

> **Correction (2026-09-28, after independent review of PR #87; §A text above is kept as declared
> at `9088236`).** Three statements in §A overstate what the logs show:
> 1. **"Bounces to the rim" is wrong; the apple is carried and rolls to the rim.** After first
>    contact with the plate base the apple rises at most 0.18–0.32 cm above its rest height. It
>    reaches the rim about 3 steps later on horizontal velocity (median 0.20 m/s at first base
>    contact, range 0.20–0.25). That velocity is acquired while the hand opens: with apple–hand
>    contact still present the apple moves 2.7–2.8 cm in xy and leaves the hand at about
>    0.12 m/s horizontally (seed 46801, steps 470–476).
> 2. **Seed 46801 latches at step 500, not 502** (`first_success_step` in `report.json`).
> 3. **"Comes to rest against the rim" should read "ends slowly rolling along the rim, with
>    intermittent rim contact".** The failures end 0.1–0.9 mm inside the 4.61 cm contact ring at
>    about 0.01 m/s. Rim contact is present at the final step on 1/28 and 3/31 non-successes, and
>    within the last 60 steps on 28/28 and 31/31. The geometric statement that an apple resting
>    against the rim always fails is unchanged.

## B. The one redesign (R9 step 3), declared before it runs

**Change: place-then-open at the plate centre.** The target is the drop, bounce and roll above.
The expert is `apple_collector_policy` with `transfer_x_shift` 0.0, which puts the held apple over
the plate centre. Its last three phases are replaced, keeping the collector's own 745-command
budget:

| phase | commands | grasp | palm target |
|---|---|---|---|
| lower_closed | 100 | closed (+1) | the transfer xy, at `rest_z + 0.0444 + 0.005` (base frame) |
| open | 100 | opens on the collector's ramp (0.08 per command) | the same pose |
| retreat | 80 | open | the transfer pose |

- `rest_z` is the apple's resting centre height on the plate, from the scene constants
  (`container_surface_z + object_support_height`).
- **0.0444 m is the palm-over-apple height while the apple is carried.** It was read from the
  diagnosis logs on the spent seeds, and it is geometry, not an outcome. So the apple should be
  about 0.5 cm above its resting height when the hand opens.
- **Known risk, stated before running.** During `descend` the palm stalled about 6 cm above its
  target while the hand touched the table. If the hand meets the plate before reaching the place
  pose, the apple will be released higher and may still roll. That is the redesign's failure mode,
  and it is not patched.
- **Placing at the centre is part of this one change:** a placed apple rests where it is set, so
  the placing target is the centre. Nothing is tuned on any outcome of this probe.

**Seeds.** 46864–46895 (32). They are fresh, and disjoint from:
- 46000–46415 (spent);
- 46800–46831 (probe and diagnosis);
- 46832–46863 (reserved for v2 calibration);
- cohorts C and D;
- the smoke range;
- every range in `first_policy.FORBIDDEN_RANGES`.

`check_seeds()` asserts this.

**Plate errors.** 0, 1.0 and 1.5 cm, in a fixed direction per seed from `default_rng(6820)`.

**Pass bar (R9).** The redesign passes if it reaches **≥ 28/32 at plate exact and ≥ 28/32 at
1.0 cm**, counted with the scorer's success latch, as in C0 and the gated protocol. The 1.5 cm
level is reported only. So is the state after the 60-step settle: whether the apple is still
inside at rest, and its final distance.

**Consequences, as R9 sets them.**
- **If it passes:** a v2 amendment PR, as in R8 §3–4, with the collector change as code with
  tests. Then the pre-run reviewer's GO, then the gated run.
- **If it fails:** stop and report. There is no second design without a new ruling. Closing
  TASK-067 is the fallback.

**Script.** `scripts/probe_first_policy_place.py`. It runs the diagnosis loop with the redesigned
expert in command. Output goes to `outputs/task067-place-probe/run-1/report.json`.

## C. Redesign result

*§A and §B were committed at `9088236` (draft PR #87) before this run.*

**Row: FAIL.** The redesign reaches **25/32** with the plate exact and **20/32** at 1.0 cm. Both
are below the 28/32 bar. **Under R9 the work stops and is reported back.** There is no second
design without a new ruling. Closing TASK-067 is the fallback.

**Provenance.**
- run-1 at `9088236`, clean tree, 312 s, 8 CPU workers. 96 attempts, 0 errors, every attempt
  complete (no guard stop).
- `outputs/task067-place-probe/run-1/report.json`, sha256 `a5af156c…acb9`.

| plate error | successes (scorer latch; gated) | successes at rest after the settle (reported) | final distance, cm (q10 / q50 / q90) |
|---|---|---|---|
| 0 | **25** / 32 | 2 / 32 | 4.41 / 4.55 / 4.60 |
| 1.0 cm | **20** / 32 | 0 / 32 | 4.53 / 4.55 / 4.60 |
| 1.5 cm (reported only) | 17 / 32 | 0 / 32 | 4.53 / 4.55 / 4.57 |

**What the logs show (medians over the 32 attempts at each level).**
- **The palm did not reach the place pose within the 100-command budget.** During
  `lower_closed` its lowest point was 0.113 m (plate exact) and 0.111 m (1.0 cm) in the base
  frame, against a target of 0.035 m. It had not stalled: the commanded z stayed saturated at
  −0.4 (6 mm per step requested) for all 100 steps, and over the last 20 steps the palm was still
  descending at a median 0.44 mm/step (range 0.38–0.47) when the budget ran out.
- **The pre-declared risk (the hand meets the plate) did not occur** at 0 or 1.0 cm. During
  `lower_closed` there was zero hand–plate contact and zero other hand contact on all 64
  attempts; the only contact was apple–hand.
- **The held apple's lowest point was 8.1 cm above its resting height** (z 0.860 and 0.858 m,
  against 0.779 m). The hand therefore still let the apple fall about 8 cm, down from about
  15.5 cm with the current collector.
- The apple's top speed from the opening onward was 0.82 m/s (range 0.37–0.84 with the plate
  exact, 0.35–0.84 at 1.0 cm). Its horizontal speed at landing had a median of 0.22 m/s, and it
  reached the rim a median 3–4 steps later. It touched the rim on 32/32 attempts at both levels.
- Hand–plate contact occurred on 5 (plate exact) and 8 (1.0 cm) attempts, all during the opening
  (phase 6), after `lower_closed` had ended.
- The apple still ended rolling slowly along the rim, at 4.4–4.6 cm, outside the 4 cm radius.
- The report's `*_after_release` fields take "release" as the first phase ≥ 5, which for this
  policy is the start of `lower_closed`, not the opening. The numbers in this section were
  computed from the opening.

**Reading (interpretation).**
- Halving the drop did not stop the apple being carried and rolled to the rim. The redesign's
  intended place was never tested: within the fixed 100-command budget the hand was still about
  8 cm above setting the apple down.
- Contact is ruled out as the cause at the two gated levels. What remains is a slow descent
  (about 0.44 mm/step against 6 mm/step commanded). Why the rate is that low (IK, joint limits or
  the embodiment's step limits) is not identified by these logs.
- The 25/32 at plate exact is confounded by the centre shift, which is part of this one change:
  in the R8 probe, releasing at the centre alone scored 15/32 at plate exact, against 32/32 for
  the current collector on the latch.
- Success in this probe is counted over 805 steps, including the settle, while C0 caps at 800
  policy steps. No first success fell at step 745 or later in either run (first successes at
  steps 499–525 and 580–608), so no count changes.

**Not done, per R9:** no second design, no change to the task, the plate geometry, the scorer
radius or the 28/32 bar, and no v2 amendment. Learned Apple→Plate is still 0 successes.
