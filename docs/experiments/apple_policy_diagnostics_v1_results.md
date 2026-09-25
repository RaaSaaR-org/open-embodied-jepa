# Apple→Plate policy diagnostics v1: results (TASK-057)

**Outcome X. The abandonment clause fires.**

- **The run is valid:** all three voiding controls passed.
- **Clause (a) does not hold**, within D1-pipeline's stated scope (§2). The observation the loop
  receives is the training observation.
- **G-SUB fails:** no single substituted command channel reaches its threshold, and the best
  candidate scores 1/16.

Under §5.1 and Amendment 1, the *consequence* of `apple_policy_v1.md` §7's clause applies as
written (§6 quotes it verbatim). **This behaviour-cloning control line stops on this corpus and
this camera.** No third control formulation is preregistered on them. The next task is a
perception/data task: resolution, camera placement or corpus design. Cohort C remains unconsumed.
`exemption_spent` stays `false`, because the exemption was not claimed.

**Learned Apple→Plate is still 0 successes**: 0/64 attempts across the four arms, with nothing
substituted. This run, like the TASK-056 runs before it, was on the development cohort, which
never gates.

Protocol: [`apple_policy_diagnostics_v1.md`](apple_policy_diagnostics_v1.md), §1–§12 frozen, §13
Amendment 1, §14 Amendment 2. Run manifest:
`benchmarks/manifests/apple-policy-diagnostics-v1-results.json`. This document keeps the numbers
(§2–§4) separate from their reading (§5–§6).

---

## 1. Provenance

| item | value |
|---|---|
| command | `uv run --no-sync python scripts/diagnose_policy.py --output outputs/task057-diagnostics/run-1` |
| revision | `e9f46703831c765ce5539e14d7f9cf471cf76a0c` (main after PR #45), clean tree (`dirty: false`) |
| python source sha256 | `a3bbfa08…0bd8` |
| device | CPU (Amendment 1: the B2 reference ran on CPU) |
| config | `configs/apple_policy_v1.yaml` |
| checkpoints | `checkpoints/task056-policy-v1/{a0,a1,a2,a3}.pt`; sha256 verified against `amendment_1.hashes` before the run started |
| `policy.py` | sha256 `d682078b…489f`, unchanged |
| seeds | development cohort 45000–45007 and 45100–45107; the 15 val-root resets for D1-pipeline. **Cohort C was not touched, and neither was the test split.** |
| budget | caps of 6 h global, 300 s per attempt and 1 000 steps per attempt. Elapsed **2 792 s (46.5 min)**. No cap was reached. |
| report | `outputs/task057-diagnostics/run-1/report.json`, sha256 `4f0cc8a4…41c7` |
| per-attempt files | 256 files under `outputs/task057-diagnostics/run-1/attempts/`, each hashed in the run manifest |
| summary | `outputs/task057-diagnostics/run-1-summary.json`, produced by `scripts/summarize_policy_diagnostics.py` (committed). It supplies the per-attempt table, the grasp-attempt table and the D3 counts. The per-reset approach numbers, the representative trace, the like-for-like \|·\| statistics after the grasp latch, the D3 `dropped` counts and the failing-contrast distances come from `scripts/summarize_policy_approach.py` (committed, read-only). |
| pre-run review | independent reviewer's **reported** verdict APPROVE, no blocking findings, at `e9f4670` |

**B2 ruling, logged as instructed.** The task owner's ruling reached the executing agent at
**2026-09-25T11:07:35Z**.

- **The timing, stated exactly** (from the artifact timestamps):
  - B3, D1-pipeline and B1 were complete. The last B1 file was written at 11:07:33.7Z.
  - B2's first attempt (A2 with nothing substituted, seed 45000) had begun about 1 s earlier,
    at about 11:07:33.9Z. It completed at 11:07:47.7Z, 12 s after the ruling.
  - **No B2 result existed when the ruling was made.** Every B2 file postdates it.

The ruling:

- B2 is judged exactly as the frozen protocol states.
- A mismatch is a failed control, whatever its cause.
- A host-stall explanation may be given as context only.
- The sole exception is a stop that meets Amendment 2's definition, as recorded by the runner.

The ruling did not need to be applied, because B2 reproduced exactly (§2).

## 2. Controls: all passed, so the run is valid

| control | requirement | result |
|---|---|---|
| **B3** (run first): `full` substitution, A2 in command | ≥ 14/16 grasp resets | **16/16 grasp, 16/16 success**, every attempt ending in `success` |
| **D1-pipeline**, 15 val-root resets, zero commands executed | see §13.4 | **passes for all four arms.** 15/15 images byte-identical (A1–A3), state difference **0.0**, masks identical. Maximum \|`act` − clipped offline\| = 2.4e-7 (A0), 8.3e-7 (A1), **1.61e-6 (A2)**, 4.2e-7 (A3), against a tolerance of 1e-4 |
| **B1** `scripted_oracle` | ≥ 14/16 grasp, ≥ 12/16 success | **16/16 grasp, 16/16 success** |
| **B1** `hold`, `random` | 0/16 grasp and 0/16 success | **0/0** and **0/0**, every attempt ending in `step_limit` at 1 000 |
| **B2**: A2 with nothing substituted must reproduce `outputs/task056-cohort-d/a2.json` | same grasp seed, 16 termination reasons and 16 step counts | **reproduced exactly**: grasp seed 45100; all 16 `step_limit` at 1 000; no mismatches |

A0, A1 and A3 with nothing substituted also reproduced their TASK-056 reports attempt by attempt, although no control
requires this. A3 had `guard_refusal` on 45001 (157 steps) and 45003 (143 steps), and grasped
on 45006.

**Scope of D1-pipeline, stated so a pass is not over-read.** The live and offline predictions
come from the same rebuilt policy object. So D1-pipeline certifies three things:

- the observation the loop receives equals the training observation, byte for byte;
- the state and mask are identical;
- single-sample `act()` equals the batched path.

It cannot detect a construction defect that affects both paths. The construction itself is
covered by §9's re-derivation, which matches the training-time record to 7e-7.

**Correction to Amendment 1.** §13.4 and the manifest's `sound_pipeline_demonstration_on_train_roots`
say that on 8 train roots `act()` equals the clipped offline prediction "to 0.0". The committed
runner measures up to about 1.3e-6 on those roots (pre-run reviewer), and up to 1.61e-6 on the
val roots here. The margin to the 1e-4 tolerance is about 60× or more, so no gate is affected.
The "0.0" should not be repeated.

## 3. Gates

### D1-pipeline and D1-grasp: clause (a) does not hold

| arm | D1-pipeline | D1-grasp count (step-zero `right_grasp` > −0.5) | signed step-zero `right_grasp`, min … max over the 16 seeds |
|---|---|---|---|
| A0 | pass | 0/16 | −1.000 … −1.000 |
| A1 | pass | 0/16 | −1.000 … −0.999 |
| A2 | pass | 0/16 | −1.000 … −0.978 |
| A3 | pass | 0/16 | −1.000 … −0.998 |

The full signed 4 × 16 table is in `report.json` under `gates.D1_grasp_signed_table`. Every arm
commands the hand open at step zero, which is correct. Clause (a) does **not** hold by either
disjunct: D1-pipeline fails for 0 arms, and D1-grasp fires for 0 arms.

### G-SUB: fails for every candidate

A2 was in command, with the named channels taken from the shadow expert (`apple_collector_policy`)
on the policy's clock. **These are privileged-substitution diagnostics, not learned results.**

| candidate | A2 + substitution: grasp resets | threshold τ_c | clock-only blind complement (pre-run calibration) | prior-only constant complement |
|---|---|---|---|---|
| `all_translation` | **1/16** | 10 | 5/16 | 0/16 |
| `dx` | **0/16** | 10 | 5/16 | 0/16 |
| `dy` | **0/16** | 12 | 9/16 | 0/16 |
| `dz` | **0/16** | 8 | 1/16 | 0/16 |
| `droll` | **0/16** | 9 | 3/16 | 0/16 |
| `dpitch` | **1/16** | 9 | 2/16 | 0/16 |
| `dyaw` | **0/16** | 9 | 3/16 | 0/16 |
| `grasp` | **1/16** | 9 | 2/16 | 0/16 |

**G-SUB fails.** The best candidates reach 1/16, against thresholds of 8–12.

### Decision

The report's decision block reads:

- `outcome: X`
- `clause_a: false`
- `g_sub_passed: false`
- `claims_the_exemption: false`

These map to exactly one pre-declared outcome: **Outcome X, and the abandonment clause fires.**

## 4. Reported, not decisional

### D1-contrast: step-zero command against the shadow expert, as declared before the run

Ratio of the median over the 16 seeds of \|step-zero policy − shadow expert\| to 3 × Table A:

| arm | dx | dy | dz | droll | dpitch | dyaw |
|---|---|---|---|---|---|---|
| A0 | **4.84** | 0.20 | 0.15 | 0.25 | **1.08** | 0.13 |
| A1 | **5.33** | 0.24 | 0.32 | 0.13 | 0.30 | 0.43 |
| A2 | **4.49** | 0.24 | 0.12 | 0.11 | 0.20 | 0.68 |
| A3 | **3.81** | 0.12 | 0.26 | 0.20 | 0.91 | 0.33 |

This is what §13.7 declared from the offline calibration: no arm commands a reset-appropriate
`right_dx` at step zero. The closed-loop values sit close to the offline ones (val 4.84, 5.37,
4.55, 3.76). Under §13.4 this is **not** a pipeline finding.

### D2: first departure (persistence 5, τ_X = 3 × the median of the arm's Table A cells)

| arm | median `first_departure_step` | reading | per-seed values |
|---|---|---|---|
| A0 | 28 | inconclusive | 28 on 15 seeds, 117 on one |
| A1 | 116 | gradual | 110–120 on 15 seeds, 405 on one |
| A2 | 14 | inconclusive | 10–67 |
| A3 | 9.5 | inconclusive | 4–12 |

The limit declared in §13.5 applies. Open loop, sound arms already depart at 116–205. So A1's
*gradual* reading cannot separate compounding error from ordinary open-loop error. The other
three arms depart much earlier in closed loop than they do open loop on their own recorded
trajectories (A0 116, A2 205, A3 205). A departure by step 10–28 is well before `orient` ends at
step 130.

### Per-step truth in the traces (new in this run; claim audit S6-07)

With nothing substituted:

| arm | resets where the palm never came ≥ 1 cm closer to the apple than at reset | reach | hand contact | grasp | median palm–apple distance: at reset / closest |
|---|---|---|---|---|---|
| A0 | **16/16** | 0 | 0 | 0 | 0.099 / 0.099 m |
| A1 | **16/16** | 0 | 0 | 0 | 0.099 / 0.099 m |
| A2 | **15/16** | 1 | 1 | 1 | 0.099 / 0.098 m |
| A3 | **14/16** | 4 | 4 | 1 | 0.099 / 0.097 m |

The two attempts that grasped (§1.3 of the protocol, reproduced here) now have per-step records:

| | A2 / 45100 | A3 / 45006 |
|---|---|---|
| first contact, step | 139 | 145 |
| `grasp` latched, step | 216 | 196 |
| hand contact on every step after the latch | **yes, 784/784** | **yes, 804/804** |
| apple height after the latch (rest 0.7666 m) | 0.820 … 0.935 m | 0.821 … 0.936 m |
| controller's mean (dx, dy, dz) after the latch | (+0.001, −0.001, +0.133) | (+0.034, −0.019, +0.154) |
| controller's median \|dx\|, \|dy\| after the latch | 0.010, 0.007 | 0.031, 0.014 |
| shadow expert's mean (dx, dy) over its defined steps after the latch | (+0.114, +0.114), 529 steps | (+0.221, +0.334), 549 steps |
| final apple–plate distance | 0.218 m | 0.193 m |
| `transport` | never | never |

**Sustained contact is now measured, not inferred.** On both resets the hand held the apple on
every one of the ~800 steps after the grasp latched. The apple was lifted about 17 cm and never
dropped.

Named failing contrasts: A2/45000 and A3/45000 start at a palm–apple distance of 0.0996 m, which
is also their minimum.

- The distance grows during the orient-like rise, to 0.134 m (A2) and 0.142 m (A3) by step 130.
- It stays there to step 1 000; the maxima are 0.135 m and 0.143 m.
- Neither makes contact. The apple rises by 0.001 m.

### D3 detail

| candidate | reach | contact | grasp | success | apple `dropped` at the end (> 1 m from the plate) | terminations |
|---|---|---|---|---|---|---|
| `all_translation` | 2 | 2 | 1 | **1 (45003)** | 1 (1) | exhausted; one `success` |
| `dx` | 16 | 16 | 0 | 0 | 2 (2) | exhausted |
| `dy` | 12 | 12 | 0 | 0 | 1 (1) | **14/16 `guard_refusal`** at steps 428–527; the rest exhausted |
| `dz` | 4 | 4 | 0 | 0 | 0 | exhausted |
| `droll` | 0 | 0 | 0 | 0 | 0 | exhausted |
| `dpitch` | 1 | 1 | 1 | 0 | 0 | exhausted |
| `dyaw` | 0 | 0 | 0 | 0 | 0 | exhausted |
| `grasp` | 16 | 16 | 1 | 0 | **12 (11)** | exhausted; `guard_refusal` on 45003 |

On `grasp` and `dx`, reach and contact of 16/16 are not progress toward grasp. Under `grasp`, the
expert's closing hand knocks the apple off the table on 12 of 16 resets.

§5.3 requires the fraction of commands issued after the expert's `close` phase to be reported,
because it matters for any pass. The mean per candidate is 0.47 (`dy`) to 0.66. No candidate
passed, so the requirement bears on no gate. The single `all_translation` success is a
privileged-substitution result, **not a learned one**.

## 5. Reading: where grasp → carry → place → release breaks

This section is interpretation, not measurement. Each claim carries its qualifier.

1. **The dominant failure is before `reach`: the approach.**
   - On 61 of 64 unsubstituted attempts the palm never gets 1 cm closer to the apple than at
     reset.
   - **Context for that statistic, so it is not over-read.**
     - The expert's own `orient` target is 0.13 m above the apple (`scripted.py`: `obj + [−0.03,
       0, 0.13]` at `scripted.py:51`, plus the collector's +0.015 m palm offset). In B3 the expert's palm–apple
       distance at step 130 is 0.134 m.
     - The arms make the same early rise. Over steps 0–19 every arm's per-reset mean command
       lies in dy −0.17 to −0.44 and dz +0.21 to +0.25.
     - Palm–apple distance at step 130 exceeds its reset value on every attempt, by +0.014 to
       +0.049 m. So "never 1 cm closer than at reset" partly reflects this shared first phase.
   - **What follows on most resets is no descent.**
     - After step 100, the per-reset mean commanded translation is at most 0.07 in any component
       on every reset of A0 and A1.
     - It is at most 0.10 on 15/16 resets of A2 (including the grasp reset 45100) and on 12/16
       of A3.
     - A3's grasp reset 45006 is lifting, with mean dz +0.13.
   - **The exceptions are the resets that descend** (per-reset mean dz after step 100 below −0.15):
     - A2/45003: mean dz −0.40, and it ends 0.119 m from the apple without contact.
     - A3/45100: mean dz −0.23, and it gets to within 0.046 m of the apple.
     - A3/45001 and A3/45003: mean dz −0.19 and −0.22; both end on the joint-velocity guard, at
       steps 157 and 143.
     - A3/45006 (the grasp) and A2/45100 (the grasp) are the other close approaches.
   - Across resets the late command is nearly identical for A0 and A1 (standard deviation
     ≤ 0.006). It varies more for A2 and A3 (dz standard deviation 0.08–0.10), mostly because of
     these exceptions.
   - **Representative trace A2/45001**, from step 130 to 1 000:
     - The palm–apple distance is constant at 0.1385 m.
     - The policy commands a constant (+0.040, −0.028, −0.028), with standard deviation ≤ 0.002.
     - The shadow expert, on its own clock, commands bang-bang ±0.4 in each translation
       component: mean \|·\| 0.4 each, and mean (+0.40, +0.04, +0.11) over its 615 defined
       steps.
   - **On the non-descending resets, and on A2/45001 exactly,** the loop behaves as a fixed
     point: a small constant command, no motion, an unchanged state, and the same command
     again. This is not claimed for the descending resets above.
   - This is more consistent with H-shift (§1.5) than with H-pipeline. D1-pipeline shows the
     observation is the training observation, within its stated scope.
   - **Unmeasured: why** the policy's command collapses instead of continuing the approach. The
     candidate declared in §13.7 is that a policy which reads motion from joint velocities
     continues motion but cannot start it. It fits the traces and is not tested here.
2. **On the two resets where the approach happened, grasp and lift work and carry fails.**
   - Both grasps held for the rest of the episode with continuous contact.
   - After the grasp latch, the arm barely translates in the plane. Mean \|dx\|, \|dy\| are
     0.015 and 0.012 for A2, and 0.034 and 0.019 for A3.
   - The shadow expert's mean \|dx\|, \|dy\| over its defined steps in the same span is 0.40
     and 0.40 (A2) and 0.33 and 0.38 (A3).
   - Caveat: the expert runs on the policy's clock, not its own. At the latch it is in `close`
     (A2) and `descend` (A3), not in its carry phase. So this is a contrast in how much the arm
     moves, not a like-for-like comparison with the expert's carry.
   - The apple ends 0.19–0.22 m from the plate, and `transport` is never latched.
   - So on the only evidence available, **grasp → carry is where those attempts break.**
   - `place` and `release` are never reached, so nothing is known about them from the learned
     arms.
   - Caveat: this is n = 2 attempts, on resets that were already known before the run.
3. **No single command channel carries the failure.** G-SUB reached at most 1/16.
   - On every candidate, A2 plus substitution scored below the same substitution with a clock-only
     complement, by 1–9 attempts of 16.
     - On `dz`, `dpitch` and `grasp` the gap is a single attempt (0 against 1, 1 against 2, 1
       against 2).
     - The gap is intermediate on `dx` (0 against 5) and `all_translation` (1 against 5), and
       largest on `dy` (0 against 9). There 14/16 attempts end on the joint-velocity
       guard.
   - Against A2's own unsubstituted 1/16, three candidates are equal and five are one attempt
     lower.
   - The clock-only controller alone, with nothing substituted, already reaches 2/16 against A2's
     1/16 (§13.4).
   - The honest reading: substituting any one channel does not help this policy. Only on `dy` is
     there evidence that the learned complement makes a substituted channel markedly worse, with
     n = 16 per configuration.
4. **What this does not show.**
   - It does not show that a learned controller cannot do this task.
   - It does not show that the encoder is at fault. B5/B6 stand: A1 (random encoder) and A0 (no
     image) behave like A2 on the approach on most resets.
   - A substitution rescue would not have shown attainability (§5.3), and there was no rescue.

## 6. Consequence and recommendation

**The abandonment clause fires as written.** Amendment 1 defines "as written" as the consequence
of `apple_policy_v1.md` §7's clause, quoted here in full:

> **Abandonment clause. If Outcome D holds, this line stops. No third control formulation is
> preregistered on this corpus and this camera.** The next task is a perception/data task
> (resolution, camera placement, corpus design), and if that does not move the closed-loop number
> either, the honest published conclusion is that a 112 px onboard camera plus a single-mode
> scripted-collector corpus does not support learned Apple→Plate on this platform, and the
> product goal needs a data or hardware change — not another model.
>
> — `apple_policy_v1.md` §7, quoted verbatim

`apple_policy_diagnostics_v1.md` §5.1 adds three points:

- no further loss, head or output-parameterisation variant is preregistered on this line;
- cohort C remains unconsumed;
- the executing agent does not select the follow-up.

`exemption_spent` stays `false`. The task owner chooses the perception/data task.

**Recommendation for the owner, not a selection.** The evidence points to one question that a
perception/data task could settle cheaply before anything else is built:

> *Does the reset-frame observation carry the information that decides the approach?*

- At reset every arm commands the wrong `right_dx` for the reset, and it does so even
  **in-sample on its own training frames** (§13.4). Proprioception is identical across resets,
  so only the image can carry the apple's position at step zero.
- The claim audit (S5-04) shows that after the first commands the arm's own pose reads out the
  apple position to about 1 cm. That makes a shortcut available to BC that uses proprioception
  after the start. The traces are consistent with it, but it is **not tested**.

The proposed task is the **information-ceiling probe** that `apple_policy_v1.md` Outcome E names:

- a single-frame regression from the reset image to the apple position and to the expert's
  step-zero command;
- at 112 px and at the native render resolution;
- with the existing TASK-056 checkpoints left untouched.

Its result separates "the 112 px onboard camera cannot resolve the approach" (a camera or
resolution change) from "the information is there but BC never learned to use it" (corpus design,
e.g. demonstrations that start with motion, or on-policy relabelling, which the protocol recorded
as a candidate and did not preregister).

The carry failure after a successful grasp (§5, item 2) is a second, separate question. It should
be preregistered only after the approach question has an answer.

## 7. Failures, deviations and caveats kept in the record

- **Protocol defects found before the run and corrected openly.**
  - Amendment 1: the wrong shadow expert; a D1 that a sound pipeline fails; a D1-grasp cut that a
    blind predictor passes; a G-SUB threshold that a clock-only controller passes on `dy`; an
    undefined `first_departure_step`.
  - Amendment 2: an undefined outcome for a stopped run.
  - PR #45's review caught an unclipped D1 comparison in the runner that would have failed a sound
    pipeline.
- **One superseded pre-freeze calibration count** read the label sidecars of the test split. §13.3
  discloses it.
- **The development cohort was used before** (TASK-047 to TASK-056) and never gates. n = 16 per
  configuration, with deterministic simulation. No interval is attached to any count, and none
  is read as a population rate.
- **B1 harness.** `hold` and `random` run through the TASK-056 loop (configured-bounds clip plus
  projection), whereas `benchmark._control` executes them directly. The command definitions are
  identical.
- **The global cap** is checked between attempts, not inside one; it was not approached (46.5 min
  against 6 h).
- **The D1-pipeline scope limit** is stated in §2.
- **Corrected at review of this document:**
  - The failing-contrast distances were first misstated as constant.
  - The "no descent" and "at most 0.1" statements were first given without their per-reset
    exceptions.
  - The B2-ruling timing was first stated as "before B2 started".
  - The `dy`-gap reading was first stated as "actively degrades" on every candidate.
  - The §7 consequence was first paraphrased rather than quoted.
