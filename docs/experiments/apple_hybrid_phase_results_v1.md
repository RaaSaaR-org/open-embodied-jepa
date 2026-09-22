# Apple hybrid phase control v1: results

**Every arm here is a NON-LEARNED diagnostic.** Before the handoff, commands come
from CEM over exact MuJoCo rollouts (privileged simulator truth). After the
handoff, and in `demo_replay`, they are recorded TRAIN demonstration actions
replayed open loop. Any stage that latched after a handoff is attributed to the
**demonstration's actions**, not to any model. Learned Apple→Plate remains at
zero successes.

**The preregistered primary gate failed.** `privileged_hybrid`, with the handoff
at the demonstration's close row, reached the scorer's grasp stage on **0/4**
development resets; ≥ 2/4 was required. The failure is conclusive: all 4
attempts counted, provenance was valid and rollouts were exact. The reading
`replay_from_tracked_state_insufficient_for_grasp` is **true**: all 4 attempts
handed off, replayed, and did not grasp.

**The secondary arm grasped on 2/4.** `privileged_hybrid_early`, with the
handoff 16 rows (one horizon) before the close row, reached grasp **and full
success** on **2/4** resets (43001, 43003). By the preregistered attribution
rule, these successes come from replayed demonstration actions. This arm never
affects the primary gate. The preregistered comparison reading is "only the
earlier handoff reaches grasp on ≥2/4". That is an association over n = 4; it
does not identify a mechanism.

**`demo_replay` reproduced TASK-043 exactly.** Under the current code it
grasped 4/4 and succeeded 3/4. Every reset matched TASK-043's stage flags and
termination.

## Frozen execution

The command ran once, exactly as frozen in the
[protocol](apple_hybrid_phase_v1.md), from source revision
`f8dac630bc689d305ca5a8cc561e8ddb8ec9bd79`.

- That revision is the reviewed commit. It includes pre-run revisions R0
  (`323aa6d`, from the TRAIN smoke) and R1 (`f8dac63`, from the review).
- The tracked tree was clean. The only untracked file was `CLAUDE.md`, which is
  outside the snapshot.
- Output went to the new directory `outputs/apple-hybrid-phase-ceiling-v1/`.
- There were no retries, reruns or parameter changes. The final cohort
  (44000–44019) and TEST were not run.

Run outcome:

- The evaluator exited 0; the report status was `completed` with valid
  provenance, and all 12/12 attempts were counted.
- Global wall time was **1,899.1 s** of 3,600 s.
- Every attempt was allocated its full 840 s; none was shortened or timed out.
- Arm totals: primary 943.2 s, `demo_replay` 30.4 s, secondary 911.8 s.

Frozen inputs are the same as TASK-043/044/045:

- Checkpoint `0192b99a…a5a5`. It was used only for retrieval, TRAIN
  normalization and the tracking metric.
- Corpus manifest `6e9a5bcb…0331`.
- Action manifest `f247effe…38b1`.
- Asset manifest `2421e194…e993`.

All four regenerated TRAIN artifacts were byte-identical to TASK-045:
`839190fc…`, `d8dae051…`, `48418ea3…` and `68b408d0…`.

The [compact manifest](../../benchmarks/manifests/apple-hybrid-phase-v1.json)
records full hashes, the gate, per-attempt records and the SHA-256 of all 56 raw
artifacts.

## Every attempt

Rows are indices into the keyframed TRAIN reference. "Handoff" gives the
measured row, its original demonstration frame (which is where replay starts)
and the 0-based command. Stage commands are 0-based.

| Reset | Demo | Arm | Close / handoff row | Handoff (row, frame, cmd) | Replayed | Termination | Stage commands | Stages | Parity (checks / mism.) | Wall (s) |
|---|---|---|---|---|---:|---|---|---:|---|---:|
| 43000 | apple-42029 | primary | 141 / 141 | 142, 212, 230 | 289 | demo_exhausted | reach 233; dropped 408 | 1 | 229 / 0 | 230.3 |
| 43001 | apple-42025 | primary | 141 / 141 | 142, 212, 222 | 289 | demo_exhausted | reach 226 | 1 | 221 / 0 | 225.2 |
| 43002 | apple-42008 | primary | 142 / 142 | 142, 211, 234 | 290 | demo_exhausted | reach 237 | 1 | 233 / 0 | 238.6 |
| 43003 | apple-42021 | primary | 142 / 142 | 143, 212, 249 | 58 | replay_projection_rejected (action 270) | reach 259 | 1 | 248 / 0 | 249.1 |
| 43000 | apple-42029 | demo_replay | — | — | 501 | success | reach 221, grasp 269, success 500 | 5 | — | 7.6 |
| 43001 | apple-42025 | demo_replay | — | — | 501 | success | reach 221, grasp 269, success 500 | 5 | — | 7.6 |
| 43002 | apple-42008 | demo_replay | — | — | 501 | demo_exhausted | reach 221, grasp 268, transport 469 | 3 | — | 7.6 |
| 43003 | apple-42021 | demo_replay | — | — | 501 | success | reach 220, grasp 272, success 500 | 5 | — | 7.6 |
| 43000 | apple-42029 | early | 141 / 125 | 125, 136, 211 | 365 | demo_exhausted | reach 295; dropped 420 | 1 | 210 / 0 | 214.0 |
| 43001 | apple-42025 | early | 141 / 125 | 126, 137, 211 | 364 | **success** | reach 294, grasp 340, success 574 | 5 | 210 / 0 | 213.1 |
| 43002 | apple-42008 | early | 142 / 126 | 126, 137, 230 | 364 | demo_exhausted | reach 313; dropped 373 | 1 | 229 / 0 | 233.3 |
| 43003 | apple-42021 | early | 142 / 126 | 126, 137, 246 | 364 | **success** | reach 328, grasp 375, success 609 | 5 | 245 / 0 | 251.4 |

Retrieval chose the same demonstrations as TASK-043/044/045. In every hybrid
attempt, reach latched only after the handoff, 3–10 commands after it in the
primary arm and 82–84 in the secondary arm. So every scored stage in both hybrid
arms latched during replay. On 43003 the primary replay stopped because the
unchanged joint-velocity guard refused to project demonstration action 270
(`measured joint velocity limit exceeded`), the R0 clean stop.

End states:

- **Primary.** The apple dropped off the table on 43000 (height 0.027 m). It
  stayed on the table on 43001–43003 (height 0.767 m, plate distance
  0.13–0.48 m). The largest rise of the apple above its start was 1.3–9.5 mm.
- **Early.** On 43001 and 43003 the apple was lifted about 0.16 m and placed
  (plate distance 0.024–0.026 m). It dropped on 43000 and 43002.
- **`demo_replay`.** The apple was lifted about 0.17 m on every reset.

## Gate and readings

From `report.json["hybrid_ceiling_gate"]`:

- **Primary gate: failed.**
  - 0 grasp resets (≥ 2 needed), 4 summed ordered stages (reach only), 0 full
    successes.
  - Rollouts were exact: 0 mismatches in 931 checks.
  - `conclusive` was true.
- **`replay_from_tracked_state_insufficient_for_grasp`: true.** 4/4 resets
  handed off without a grasp.
  - `demo_exhausted` on 43000–43002.
  - `replay_projection_rejected` on 43003, at action 270, after 58 replayed
    commands.
- **`tracking_failed_before_handoff`: false.** 0/4 resets ended before the
  handoff.
- **Interpretation (computed):** "tracked approach + open-loop demonstration
  close does not reach grasp under exact dynamics".
- **Secondary arm.**
  - Conclusive, and `secondary_early_handoff_grasp_on_two` was true.
  - 2 grasp resets, 2 full successes, 12 summed ordered stages.
  - Rollouts were exact: 0 mismatches in 894 checks.
  - `handoff_comparison`: "only the earlier handoff reaches grasp on >=2/4:
    tracking over the last pre-close horizon (or skipping the dead-time
    descent) is associated with losing the grasp".
- **`demo_replay`.** Conclusive: 4 grasp resets, 3 full successes, 18 summed
  ordered stages.

## Diagnostics (not gate evidence)

- **Parity with TASK-045 before the handoff.** In all 8 hybrid attempts, every
  executed command before the handoff was bit-identical to TASK-045's command
  with the same index on the same reset. That is 230, 222, 234 and 249
  commands in the primary arm and 211, 211, 230 and 246 in the early arm.
  - The wrapper therefore did not perturb the tracker.
  - Both arms share the exact same tracked prefix up to the early arm's handoff.
- **Tracking at the handoff.**
  - The measured tracking distance was 0.024–0.031 at the primary handoff and
    0.023–0.028 at the early handoff.
  - No rollout candidate was rejected before either handoff.
  - Median planning time was 0.97–0.98 s, and the maximum observe + plan time
    was 1.07 s.
- **Where the handoff landed.**
  - The primary arm handed off at the close row or one past it (frames
    211–212).
  - The early arm handed off exactly at row close−16 on 3/4 resets, and one row
    past it on 43001 (row 126 against 125). The frame was 136–137.
  - The measured index never overshot the handoff row by more than one row.
- **How much the arms differ.** The early and primary handoffs were only
  3–19 commands apart (43000: 19, 43001: 11, 43002: 4, 43003: 3). Yet the early
  arm replays demonstration frames 136/137 → 211/212 (about 75 actions, the
  blocked descent and the start of closing), and the primary arm does not; the
  tracker crossed those same rows in those few commands.
  - On 43002 and 43003 the two arms' executed commands are identical up to
    command 230 and 246; the primary then tracked 4 and 3 more commands.
  - On 43003 the early arm succeeded and the primary did not.
  - *Interpretation, not isolated:* on 43003 the difference between the arms
    is therefore almost entirely the replayed descent segment versus 3 tracked
    commands through it. This is consistent with the keyframed tracker
    skipping the demonstration's blocked descent (the dead time that
    keyframing removed) being what loses the grasp. n = 1–2 resets; not a
    demonstrated mechanism.
- **The early arm is not reliable either.** On 43000 and 43002, replaying the
  whole descent and close from the tracked state still dropped the apple.
  `demo_replay` from reset grasped on all four. The tracked approach state
  therefore still matters, and arm-pose tracking carries no apple information.

## Interpretation and limits

This is n = 4 development evidence on resets already used by TASK-043/044/045,
under one frozen configuration. It is not a held-out test, and nothing here is
a learned result.

- **Primary hypothesis: rejected at this n.** Tracking up to the close row and
  then replaying the demonstration's remaining actions did not reproduce the
  demonstration's grasp on any reset.
  - Closed-loop replanning *during contact* is therefore not the whole story.
    The primary arm never replanned after the handoff, and reach latched only
    during replay.
  - The preregistered reading holds: open-loop demonstration closing from the
    close-row tracked state is insufficient.
- **Secondary arm: 2/4 grasp and full success.** Handing off one horizon earlier
  reproduced the demonstration's full pick-and-place on 43001 and 43003.
  - These successes belong to the **replayed demonstration actions**. The
    privileged tracker only brought the arm to the demonstration's pre-descent
    pose.
  - The comparison is an association over n = 4 and is not preregistered as a
    gate.
  - The early arm's 2/4 exactly meets the threshold used for primary gates.
    The ceiling for this configuration is therefore marginal, not comfortable.
- **What a learned component would replace next.** The only configuration
  where a non-learned scaffold reached grasp from a tracked state is:
  1. tracked approach to row close−16;
  2. open-loop demonstration actions from that frame.

  The next learned replacement is the **forward model of the approach phase
  only**. That means learned `state_goal_sensor_wm_v1` dynamics tracking to the
  same close−16 handoff, with the demonstration replay unchanged and
  `dynamics_shuffle`/`persistence` controls. Its gate must be judged against
  this privileged 2/4 ceiling. The replayed close segment remains
  demonstration memory, not a model; replacing it with an object-aware learned
  component is a later step.
- TASK-033/TASK-034 stay open. The final cohort and TEST stay untouched.

## Obligations from the pre-run review

- **Finding 3.** The early arm's actual replay starts were frames 136–137 on
  all four resets. The handoff never overshot the row by more than one.
- **Finding 4.** Guard stops are split out above: 1 primary
  `replay_projection_rejected` (43003, action 270) and 3 `demo_exhausted`.
- **Finding 5.** R0 also makes an infeasible replay projection clean. None
  occurred. A guard refusal while tracking, or in `demo_replay`, would still
  have been an uncounted runtime error. None occurred.
- **Finding 6.** Provenance stayed valid across the whole run.
- **Finding 7.** The run commit `f8dac63` is the review commit. It differs from
  the reviewed `e230884` by the review document plus the R1 fixes the review
  recommended (narrowed guard catch, dropped per-reset fields and protocol
  wording). No handoff row, gate, budget, seed or command changed.
