# Apple wide-jitter object-aware ceiling v1: results

**Every arm here is a NON-LEARNED diagnostic.**

- `privileged_object` plans with exact MuJoCo rollouts, and its cost and phase
  transitions read simulator object state.
- `demo_replay` replays a retrieved TRAIN demonstration open loop.
- `scripted_oracle` is the privileged scripted collector.

No world model contributes to any outcome. Learned Apple→Plate remains at zero
successes.

**The preregistered primary gate failed, and the failure is conclusive.**

- `privileged_object` reached the scorer's grasp stage (reach, a ≥ 5 cm lift
  and hand contact) on **5/8** wide-jitter development resets; **≥ 6/8** was
  required.
- `demo_replay` grasped on **2/8**. The second condition,
  2 ≤ ceiling − 3, would have held.
- All 16 ceiling and replay attempts counted, provenance was valid and rollouts
  were exact.
- The computed outcome is **`ceiling_inadequate_task_feasible`**, because the
  scripted collector reference grasped and succeeded on **8/8**.

The preregistered reading is: "the scripted collector grasps where the
object-aware exact-dynamics ceiling does not: the phase cost/planner design,
not the distribution, is inadequate".

## Frozen execution

The command ran once, exactly as frozen in the
[protocol](apple_wide_object_ceiling_v1.md), from source revision
`613ecf81f800e3ecf3d16653b19fd7e9d051a751`.

- That revision is the commit that adds the R1 re-review. It differs from the
  R1 code commit `9ec5938` only in `docs/reviews/`.
- The tracked tree was clean. The only untracked file was `CLAUDE.md`, which is
  outside the snapshot.
- Output went to the new directory `outputs/apple-wide-object-ceiling-v1/`.
  The run started at 2026-09-22T14:05:44Z and ended at 14:43:46Z.
- There were no retries, reruns, replacement resets or parameter changes. The
  narrow development cohort, the final cohort 44000–44019 and TEST were not
  run.

Run outcome:

- The evaluator exited 0; the report status was `completed` with valid
  provenance, and all 24/24 attempts were counted.
- Every supervisor exited with code 0.
- Global wall time was **2,282.0 s** of 8,400 s.
- Every attempt was allocated its full 960 s; none was shortened or timed out.
- Arm totals: `demo_replay` 43.8 s, `scripted_oracle` 61.4 s,
  `privileged_object` 2,155.1 s.

Frozen inputs are the same as TASK-043–046:

- Checkpoint `0192b99a…a5a5`. It was used for retrieval only.
- Corpus manifest `6e9a5bcb…0331`.
- Action manifest `f247effe…38b1`.
- Asset manifest `2421e194…e993`.

The regenerated TRAIN artifacts were byte-identical to TASK-045/046:
`state_goals.npz` `839190fc…f88d` and `state_calibration.json` `d8dae051…8d75`.

The [compact manifest](../../benchmarks/manifests/apple-wide-object-ceiling-v1.json)
records full hashes, the gate, per-attempt records and the SHA-256 of all 126
raw artifacts.

## Every attempt

In this table:

- "Stages" is the number of ordered scorer stages.
- **G** means grasp: reach, a ≥ 5 cm lift and hand contact.
- Ceiling phase starts are 0-based command indices, and "(b)" marks a
  preceding phase that ended *blocked*.
- The parity column gives robot-state and full-state checks, and mismatches.

| Reset | Apple offset (cm) | Demo retrieved | `demo_replay` | `scripted_oracle` | `privileged_object` | Ceiling phases (start command) | Parity (checks / mism.) | Ceiling wall (s) |
|---|---|---|---|---|---|---|---|---:|
| 45000 | −0.04, +0.65 | apple-42025 | **G**, success (5) | **G**, success (5) | **G**, 3 stages, `phase_stall` in retreat | approach 0, descend 109, close 124 (b), lift 169, transport 209, lower 280 (b), release 352 (b), retreat 392 | 591 / 0 | 382.7 |
| 45001 | +2.41, −2.34 | apple-42012 | reach, `guard_refused` @ 221 | **G**, success (5) | **G**, **success** (5) | approach 0, descend 111, close 127 (b), lift 172, transport 207, lower 255, release 315 (b), retreat 355 | 363 / 0 | 241.7 |
| 45002 | −1.31, −2.51 | apple-42012 | reach, `guard_refused` @ 226 | **G**, success (5) | **G**, **success** (5) | approach 0, descend 111, close 138 (b), lift 183, transport 217, lower 273, release 377 (b), retreat 417 | 425 / 0 | 290.9 |
| 45003 | −2.76, +2.67 | apple-42020 | reach, `guard_refused` @ 261 | **G**, success (5) | 0 stages, `phase_stall` in **descend** | approach 0, descend 108 | 307 / 0 | 177.0 |
| 45004 | −0.94, +2.96 | apple-42017 | reach, `guard_refused` @ 223 | **G**, success (5) | 0 stages, `phase_stall` in **descend** | approach 0, descend 114 | 313 / 0 | 183.8 |
| 45005 | +1.22, −1.76 | apple-42017 | reach, `demo_exhausted` | **G**, success (5) | **G**, **success** (5) | approach 0, descend 110, close 150 (b), lift 195, transport 225, lower 270, release 307 (b), retreat 347 | 362 / 0 | 241.0 |
| 45006 | −1.24, +0.30 | apple-42020 | **G**, success (5) | **G**, success (5) | reach only (1), `phase_stall` in **lift** | approach 0, descend 110, close 126 (b), lift 171 | 370 / 0 | 241.9 |
| 45007 | +1.29, −0.77 | apple-42025 | reach, `guard_refused` @ 222 | **G**, success (5) | **G**, 3 stages, `phase_stall` in retreat | approach 0, descend 114, close 130 (b), lift 175, transport 214, lower 272 (b), release 344 (b), retreat 384 | 583 / 0 | 396.1 |

The parity counts are identical for the robot-state and full-state checks in
every attempt.

- **`demo_replay`.** The apple was lifted and the task succeeded only on the
  two resets whose apple offset from the centre is smallest: 45000 (0.65 cm)
  and 45006 (1.27 cm). The other six offsets are 1.50–3.84 cm.
  - On 5/8 resets the unchanged joint-velocity guard refused a replayed
    command after 221–261 commands. That is a counted `guard_refused` stop
    (review revision R1), which happened after `reach`.
  - 45005 replayed all 501 actions without a grasp.
- **`scripted_oracle`.** It succeeded on all 8 resets in 501–502 commands.
  This indicates (n = 8) that the wide distribution lies within the current
  grasp mechanics.
- **`privileged_object`.**
  - Full success on 3/8 (45001, 45002, 45005).
  - Grasp and transport, but no placement, on 45000 and 45007. The apple was
    released 4.5–4.6 cm from the plate centre, outside the 4 cm radius, and the
    retreat then stalled.
  - The descent never ended on 45003 and 45004.
  - The grasp failed on 45006.

## Gate and readings

From `report.json["object_ceiling_gate"]`:

- **Primary gate: failed.**
  - `privileged_object` grasped on 5 resets (≥ 6 needed).
  - `demo_replay` grasped on 2, which is ≤ 5 − 3, so the replay condition held.
  - Rollouts were exact: 0 mismatches in 3,314 robot-state checks and 3,314
    full-state checks. Each attempt made executed commands − 1 checks, so the
    checks were non-vacuous.
  - No ceiling attempt ended with `guard_refused`.
  - Every ceiling and replay attempt counted.
- **`conclusive`: true.**
- **`outcome`: `ceiling_inadequate_task_feasible`.** Scripted grasped on 8/8,
  which is ≥ 6.
- **Failure histogram of the ceiling (`failed_reset_furthest_phase`):** descend
  2, lift 1.
- **Other arm totals.**
  - Ceiling: 22 summed ordered stages, reach on 6/8.
  - `demo_replay`: 16 stages, 2 full successes.
  - `scripted_oracle`: 40 stages, 8 full successes.

## Diagnostics (post-hoc; not gate evidence, no retuning)

These are read from the traces after the run. They describe the frozen
configuration's failures and change nothing about the result.

- **45003 and 45004: the descent stalled just outside the blocked-descent xy
  window.** On every reset the palm reached its blocked height, +0.114–0.116 m
  above the apple centre, within 5–7 descend commands. That is the
  thumb–table contact.
  - When the height was first reached the palm's xy error to the grasp point
    was still inside the window (0.43 cm on 45003, 0.89 cm on 45004), but the
    blocked rule needs more than 10 commands of height history. By then the
    error had drifted past 1 cm (from command 12 on 45003 and command 9 on
    45004), and it stayed at **1.05–1.06 cm (45003) and 1.21–1.25 cm (45004)**
    for the rest of the phase. The preregistered blocked-descent window is
    1.0 cm, so the descent was never declared blocked, and after 200 commands
    the phase stalled.
  - The cost is consistent with this. The descent cost is the 3-D distance to the
    unreachable 5.2 cm grasp point, and that distance is dominated by the
    blocked ~6.4 cm height. A 1 cm xy error changes it by less than 1 mm, so
    the planner had almost no incentive to re-centre in xy.
  - On 45000, 45001, 45006 and 45007 the xy error in the last descend
    observation was 0.33–0.62 cm.
  - On 45002 and 45005 the descent drifted to the edge of the window: the error
    was 1.00–1.01 cm in the last descend observation, and the descent ended
    only after 27 and 40 commands. So the fixed 1 cm window was marginal on 4/8
    resets.
- **45006: the grasp closed off-centre.** At the start of lift, the palm was
  3.6 cm behind the apple in x (target 1.5 cm), with hand contact.
  - For 200 lift commands the apple did not rise, and the palm–apple offset
    stayed constant.
  - The offset arose during the close phase, and the apple did not cause it:
    the apple moved only 0.3 mm in xy during close, while the palm's x offset
    behind the apple grew from 1.15 cm to 3.62 cm. On the five grasping
    resets the offset at the start of lift was 1.3–1.7 cm.
  - *Interpretation, not isolated:* the palm was pushed or drifted backwards
    while the fingers closed. The close cost holds the palm on the
    unreachable grasp point and penalises apple xy displacement, but it did
    not prevent this.
- **45000 and 45007: the grasp and transport succeeded but the place failed.**
  - Transport and lower both ended *blocked* on both resets.
  - The release left the apple 4.5–4.6 cm from the plate centre. The scorer's
    place radius is 4 cm.
  - The release distance alone does not separate them from the successes. On
    45001, 45002 and 45005 the apple was also 4.3–4.7 cm from the centre at
    the end of release, and then settled to 3.7–4.0 cm during the retreat. On
    45000 and 45007 it stayed at about 4.6 cm. Why it settled is not isolated.
  - The retreat phase then ran to its 200-command stall. That is why these two
    attempts took about 390 s.
- **Common pattern (post-hoc; the 45006 cause is not isolated).** Every
  ceiling failure is associated with a phase transition or cost term that is
  geometric: a fixed window or a distance to an unreachable
  target. None comes from the exact dynamics, since parity was exact
  throughout.

## Interpretation and limits

This is n = 8 development evidence on new resets from a distribution that the
design probes spanned (see the protocol's design disclosure). It is not a
held-out test, and nothing here is a learned result.

- **Primary hypothesis: rejected at this n.** Under exact dynamics and perfect
  object state, this object-aware phase cost and planner grasped on 5/8. The
  6/8 bar was missed by one reset.
- **The distribution does what T1 needed it to do. This is a reading of the
  reference arms and not a gate.**
  - Open-loop `demo_replay` grasped on only 2/8, down from 4/4 on the narrow
    resets.
  - The scripted object-aware collector grasped on 8/8.
  - On these counts the wide distribution separates open-loop replay from
    the scripted object-aware reference. Because n = 8, this is a count, not a significance
    claim: 8 vs 2 gives Fisher p ≈ 0.007 two-sided, not preregistered.
  - It is within the current grasp mechanics.
- **Preregistered next step for this outcome.**
  1. Diagnose the ceiling's failure phases and redesign its cost/transitions
     under a new preregistration.
  2. Do not pair the ceiling with a learned model.
  3. T2 may proceed only with the wide distribution validated by the scripted
     reference.
- *Post-hoc candidates for that redesign, which need their own preregistration
  and were not tried here:*
  - an xy-weighted descent cost, or a blocked window keyed to the xy error
    rather than a fixed 1 cm;
  - a close-phase cost that holds the achieved palm pose instead of pressing
    toward the unreachable point;
  - a release placed on apple-over-plate xy tolerance.
- TASK-033/TASK-034 stay open. The final cohort and TEST stay untouched.

## Obligations from the pre-run review

- **R1 guard handling.** R1 mattered: 5/8 `demo_replay` attempts ended with a
  guard refusal. Before R1 these would have been uncounted runtime errors,
  which would have made the whole run inconclusive.
  - The refusal string is not repeated in those trace events. The only
    permitted message is `measured joint velocity limit exceeded`.
  - The ceiling arm had no `guard_refused` stop.
- **Revision cited.** The results cite the run's own revision `613ecf8`, not
  the smoke's.
- **Separation wording.** The separation between arms is reported as counts
  under a decision rule, not as a significance test.

## Post-run verification

A fresh verifier subagent, independent of the author, reproduced these results
from the raw outputs with its own code. It checked:

- every hash;
- the plan rebuilt from the frozen command;
- the snapshot's byte-equality with `613ecf8`;
- that `9ec5938..613ecf8` touches only `docs/reviews/`;
- the gate and outcome;
- all per-reset numbers;
- the Fisher p-values;
- all 126 manifest artifact hashes.

It confirmed the result. It corrected these items in this document:

- a missing blocked marker (45000 lower);
- the 45003/45004 xy-error timeline;
- the 45006 mechanism: the palm drifted, the apple did not move;
- the release-distance caveat;
- wording on feasibility and separation.

It found no discrepancy in the gate, the outcome or the manifest.
