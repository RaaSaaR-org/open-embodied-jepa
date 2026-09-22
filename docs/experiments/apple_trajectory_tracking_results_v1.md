# Apple demonstration trajectory tracking v1: negative result

**Stage 1 is a NON-LEARNED privileged diagnostic ceiling, not a learned result.**
It used exact MuJoCo rollouts as the planner's forward model. That is allowed only
because it measures a ceiling.

**The preregistered stage-1 gate failed.** The trajectory-tracking ceiling reached
the scorer's grasp stage on **0/4** development resets, where ≥ 2/4 was required.
Per the protocol, **stage 2 (the learned comparison) was not run.**

The redesign did fix the TASK-044 failure mode. Unlike the endpoint scaffold,
which stalled at goals 0–1 with 0 stages, trajectory tracking followed the
demonstration through orient and descend on 4/4 resets:

- It latched the scorer's **reach** stage, with hand contact, on 4/4 resets
  (4 summed ordered stages).
- On 3/4 resets it tracked to the end of the reference.

It never grasped. On 4/4 resets, the apple was pushed off the table
(`dropped`) during the closing phase. The conclusive reading
`arm_pose_tracking_insufficient_for_grasp` is **true**: 3/4 resets passed the
demonstration's recorded-grasp row without a grasp. `trajectory_tracking_inadequate`
is false.

## Frozen execution

The stage-1 command ran once, exactly as frozen in the
[protocol](apple_trajectory_tracking_v1.md) (including revision R1), from source
revision `bae083815d61d0d9b38a2aaf02f834ab1925fca4`:

- The tracked tree was clean. The only untracked file was `CLAUDE.md`, which is
  outside the snapshot.
- Output went to the new directory `outputs/apple-trajectory-tracking-ceiling-v1/`.
- There were no retries, reruns or parameter changes. The final cohort
  (44000–44019) and TEST were not run.

Run outcome:

- All 4 attempts completed, and the evaluator exited 0.
- Report status `completed`, provenance valid.
- Global wall time was **1,730.7 s** of 3,600 s.
- Each attempt was allocated its full 840 s and used 303.5–497.9 s. None timed
  out.

Frozen inputs are the same as TASK-043/044:

- Checkpoint `0192b99a…a5a5`. It was used only for retrieval, TRAIN
  normalization and the metric; its dynamics were not used.
- Corpus manifest `6e9a5bcb…0331`.
- Action manifest `f247effe…38b1`.
- Asset manifest `2421e194…e993`.

The regenerated retrieval library was byte-identical to TASK-043/044, as
predicted:

- `state_goals.npz` `839190fc…f88d`
- `state_calibration.json` `d8dae051…8d75`

New TRAIN tracking artifacts:

- `tracking_references.npz` `48418ea3…18ca`
- `tracking_calibration.json` `68b408d0…aabf`

The [compact manifest](../../benchmarks/manifests/apple-trajectory-tracking-v1.json)
records full hashes, the gate, per-attempt records and SHA-256 of the core
artifacts.

## Every attempt

Reference rows are indices into the keyframed TRAIN reference. "Close row" is
the first row at or after the demonstration's first closing frame; "grasp row"
is the demonstration's recorded-grasp row. Command numbers are 0-based.

| Reset | Retrieved demo | Rows | Close / grasp row | Commands | Last row | Termination | Reach (cmd, row) | Dropped (cmd) | Stages | Parity checks / mismatches | Wall (s) |
|---|---|---:|---|---:|---:|---|---|---:|---:|---|---:|
| 43000 | apple-42029 | 340 | 141 / 195 | 481 | 337 | reference_stall | 267, 150 | 309 | 1 | 480 / 0 | 497.9 |
| 43001 | apple-42025 | 339 | 141 / 195 | 284 | 145 | reference_stall | 231, 144 | 265 | 1 | 283 / 0 | 303.5 |
| 43002 | apple-42008 | 340 | 142 / 199 | 455 | 339 | reference_complete | 260, 149 | 294 | 1 | 454 / 0 | 452.0 |
| 43003 | apple-42021 | 342 | 142 / 198 | 478 | 339 | reference_stall | 280, 154 | 378 | 1 | 477 / 0 | 467.3 |

On each reset, hand contact first latched at the same command as reach.

At termination, every apple had fallen to about 0.027 m height (it started at
0.765 m on the table). It was 0.64–3.26 m from the plate. `grasp`, `transport`,
`place`, `release` and `success` were false on every reset.

## Gate and readings

From `report.json["tracking_ceiling_gate"]`, with 4/4 attempts counted:

- **Primary gate: failed.** 0 grasp resets (≥ 2 needed), 4 summed ordered
  stages (reach only), 0 full successes. `rollouts_exact` was true, with 0
  mismatches in 1,694 checks.
- **`trajectory_tracking_inadequate`: false.** 0 resets stalled before the close
  row.
- **`arm_pose_tracking_insufficient_for_grasp`: true.** On 3/4 resets (43000,
  43002, 43003), the last reference index reached the demonstration's grasp row
  without a scored grasp. 43001 stalled at row 145, between its close row (141)
  and its grasp row (195), after the apple had already dropped.
- **Interpretation (computed):** "trajectory-tracking scaffold fails under
  perfect dynamics; do not pair it with learned dynamics".
- `learned_stage_authorized` is false, so **stage 2 was not run.**

## Diagnostics (not gate evidence)

**Tracking worked until contact.**

- Before the close row, the measured tracking distance had a median of
  0.012–0.015. That is about the size of the TASK-043/044 goal tolerances, which
  the endpoint scaffold never reached.
- The tracker needed 222–249 commands to reach the close row (141–142 rows,
  about 1.6 commands per row).
- From the close row to the grasp row, the median distance rose to 0.066–0.157,
  and the maximum over the attempt was 0.20–0.53. Once the fingers touched the
  apple, the measured joints could no longer follow the demonstration's.

**The apple left the table during closing, on every reset.**

- Reach and contact latched at commands 231–280.
- `dropped` latched 34–98 commands later.
- That is consistent with the hand pushing the apple instead of enclosing it.
  The protocol's declared risk, "arm pose is not object pose", is the likely
  cause, but the traces do not isolate the mechanism.
- The reset jitter is ±6 mm, and the demonstration's joint trajectory carries no
  apple position.

**The joint-velocity guard fired in contact-phase rollouts.**

- On 43000, 43002 and 43003, the twin rejected 541, 213 and 356 candidates. The
  reason was always `measured joint velocity limit exceeded`, starting at
  commands 238–259.
- 43001 had none.
- Rejected candidates rank last, so this only narrowed the choice.

**Timing.**

- Median planning time was 0.97–1.04 s per command, with a maximum observe +
  plan time of 1.32 s, under the 5 s deadline.
- The 840 s cap never bound. The longest attempt, 497.9 s, would also have fit
  the original 600 s cap.

**Final-row hold.**

- Revision R1 held the final row for `frames[-1] − frames[-2]` commands. That
  value is 4–5 for these references, not "about 9" as the revision text
  estimated.
- Only 43002 terminated through it (`reference_complete`, 5 final-row
  commands). This does not affect the grasp gate.

## Interpretation and limits

This is n = 4 development evidence under one frozen configuration.

- **What was fixed.** Scoring the whole horizon against a progress-indexed
  demonstration trajectory removed the TASK-044 endpoint-chasing stall. Under
  exact dynamics, the planner now follows the demonstration's arm/hand
  trajectory through orient and descend to contact on every reset.
- **What still blocks grasp.** The object interaction, not the planner. A
  proprioceptive arm+hand reference contains no apple information. Tracking it
  after contact pushed the apple off the table on 4/4 resets, even though the
  forward model was exact.
- **Open-loop replay did better.** In TASK-043, open-loop `demo_replay`
  succeeded on 3/4 of the same resets. The measured-progress reference timing,
  the dead-time removal, or closed-loop replanning during contact may disturb
  the grasp the demonstration's own timing achieves. This run does not isolate
  which.
- **What follows.** Learned dynamics should **not** be paired with this scaffold
  yet (preregistered reading). TASK-033/TASK-034 stay open, and learned
  Apple→Plate remains at zero successes.

## Recommended next step (not started)

The goal/cost must carry object information, or the contact phase must be
handled differently. Candidates, each for its own preregistered privileged
ceiling on the same resets:

1. **Hybrid phase control.** Use trajectory tracking up to the close row, then
   execute the demonstration's own recorded closing-phase actions open loop from
   the matched row (a declared non-learned segment), and score grasp. This tests
   whether closed-loop replanning during contact is what breaks the grasp that
   `demo_replay` achieves.
2. **Object-aware cost under the ceiling.** Add an image- or keypoint-based
   apple-relative term during the close phase. The ceiling can use exact
   rollouts of rendered images. This tests whether object information in the
   cost is sufficient before any learned model is asked to predict it.
