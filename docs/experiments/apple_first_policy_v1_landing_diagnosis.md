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
under this scorer include transient crossings. That applies to this corpus's collection too
(103/200 root successes in `apple-look-v1`). It was not measured there; this is an inference.

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

*Filled in after the run. Nothing below §B is written before it.*
