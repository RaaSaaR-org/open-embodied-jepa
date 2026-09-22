# Apple wide-jitter object-aware ceiling v2: results

**Every arm here is a NON-LEARNED diagnostic.** `privileged_object` (v2) plans
with exact MuJoCo rollouts and an exact release probe, and its cost and phase
transitions read simulator object state; `demo_replay` replays a retrieved TRAIN
demonstration open loop; `scripted_oracle` is the privileged scripted collector.
No world model contributes to any outcome. Learned Apple→Plate remains at zero
successes.

**The preregistered primary gate failed, and the failure is conclusive.**

- On the fresh development resets 45100–45107 the v2 ceiling reached full task
  success on **5/8** (≥ 6 required) and the scorer's grasp stage on **5/8**
  (≥ 7 required).
- All 8 primary ceiling attempts counted, provenance was valid, and rollouts were
  exact (0 mismatches in 2,169 robot-state and 2,169 full-state checks, each
  attempt checking executed commands − 1).
- The computed outcome is **`ceiling_inadequate_task_feasible`**, because the
  scripted collector succeeded on 8/8 primary resets.

The preregistered reading is: "the scripted collector succeeds where the v2
exact-dynamics ceiling does not grasp: the cost/phase design is still
inadequate". The preregistered next step is to diagnose the failure phases and
redesign under a new preregistration, and not to pair the ceiling with a learned
model.

**Secondary (non-gating) cohort.** On the TASK-047 resets 45000–45007 the same v2
ceiling grasped 7/8 and succeeded **7/8**, against v1's 5/8 grasp and 3/8 success
(TASK-047). Both v1's descend stalls (45003, 45004), its lift failure (45006) and
its two off-centre releases (45000, 45007) are gone; the one remaining failure
(45000) is the new dominant mode described below. This is a comparison on resets
whose v1 traces informed the redesign, so it is a diagnostic, not independent
evidence.

## Frozen execution

The command ran once, exactly as frozen in the
[protocol](apple_wide_object_ceiling_v2.md), from source revision
`7b94f06410f8453d4cf2e88d67bccc9682a9820c` (the pre-run review revision R1; the
tracked tree was clean, the only untracked files being the local `CLAUDE.md`,
`.claude/`, the scratch `outputs/` and the ignored `data`/`checkpoints`/
`third_party` symlinks of the worktree).

- Output directory (new): `outputs/apple-wide-object-ceiling-v2/` in the main
  checkout. Start 2026-09-22T16:49:44Z, end 17:43:21Z.
- Exit 0; report `completed`; provenance valid; 48/48 attempts counted;
  **3,216.8 s** of the 20,400 s budget. No attempt was shortened or timed out; no
  `deadline_miss` occurred.
- Arm totals: `demo_replay` 98.5 s, `scripted_oracle` 124.4 s,
  `privileged_object` 2,954.8 s (130–260 s per attempt, 0.60–0.69 s per command).
- There were no retries, reruns, replacement resets or parameter changes. The
  narrow development cohort, the final cohort 44000–44019 and TEST were not run.

Frozen inputs, as in TASK-043–047: checkpoint `0192b99a…a5a5` (retrieval only),
corpus manifest `6e9a5bcb…0331`, action manifest `f247effe…38b1`, asset manifest
`2421e194…e993`. The regenerated TRAIN artifacts were byte-identical to
TASK-045/046/047: `state_goals.npz` `839190fc…f88d`, `state_calibration.json`
`d8dae051…8d75`.

`planning_dynamics` in the plan still reads `privileged_mujoco_rollout_object_v1`:
the twin is v1's, unchanged. The v2 identity is carried by
`object_ceiling_version`, `result_label` and each attempt's `ceiling_version`.

The [compact manifest](../../benchmarks/manifests/apple-wide-object-ceiling-v2.json)
records the hashes, the gate and per-attempt records.

## Every attempt

"Stages" counts ordered scorer stages; **S** marks full success. Ceiling phase
starts are 0-based command indices; "b" marks a phase that ended *blocked*, "R" a
hand-over on a predicted placement (`release_predicted`). Parity is checks /
mismatches (robot-state and full-state counts were identical in every attempt).

### Primary: fresh resets 45100–45107 (the gate)

| Reset | `demo_replay` | `scripted_oracle` | `privileged_object` v2 | Ceiling phases | Parity | Probes | Ceiling wall (s) |
|---|---|---|---|---|---|---|---:|
| 45100 | 3 stages, `demo_exhausted` | **S** (5) | 1 stage, **`object_dropped`** | ap0 de112 cl129b li174 | 216 / 0 | 0 | 130 |
| 45101 | **S** (5) | **S** (5) | **S** (5) | ap0 de112 cl129b li174 tr199 re242R | 276 / 0 | 43 | 175 |
| 45102 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de110 cl126b li171 tr204 re245R | 279 / 0 | 41 | 177 |
| 45103 | 1 stage, `guard_refused` | **S** (5) | 1 stage, **`object_dropped`** | ap0 de112 cl131b li176 | 226 / 0 | 0 | 136 |
| 45104 | **S** (5) | **S** (5) | **S** (5) | ap0 de115 cl132b li177 tr232 re281R | 316 / 0 | 49 | 202 |
| 45105 | **S** (5) | **S** (5) | 1 stage, **`object_dropped`** | ap0 de109 cl126b li171 | 230 / 0 | 0 | 141 |
| 45106 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de109 cl124b li169 tr262 re293R | 326 / 0 | 31 | 210 |
| 45107 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de112 cl135b li180 tr226 re266R | 300 / 0 | 40 | 203 |

Primary totals: ceiling grasp 5/8 and success 5/8; `demo_replay` grasp 4/8 and
success 3/8; `scripted_oracle` grasp 8/8 and success 8/8.

### Secondary: TASK-047 resets 45000–45007 (never gating)

| Reset | `demo_replay` | `scripted_oracle` | `privileged_object` v2 | Ceiling phases | Parity | Ceiling wall (s) |
|---|---|---|---|---|---|---:|
| 45000 | **S** (5) | **S** (5) | 1 stage, **`object_dropped`** | ap0 de109 cl126b li171 | 257 / 0 | 168 |
| 45001 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111 cl128b li173 tr228 re271R | 289 / 0 | 199 |
| 45002 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111 cl127b li172 tr201 re247R | 268 / 0 | 178 |
| 45003 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de108 cl125b li170 tr208 re261R | 282 / 0 | 192 |
| 45004 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de114 cl134b li179 tr225 re264R | 300 / 0 | 205 |
| 45005 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de110 cl128b li173 tr207 re243R | 277 / 0 | 185 |
| 45006 | **S** (5) | **S** (5) | **S** (5) | ap0 de110 cl131b li176 tr216 re263R | 296 / 0 | 193 |
| 45007 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de114 cl132b li177 tr298 re339R | 374 / 0 | 260 |

Secondary totals: ceiling grasp 7/8 and success 7/8 (v1: 5/8 and 3/8);
`demo_replay` grasp 2/8 and success 2/8, reproducing TASK-047 exactly;
`scripted_oracle` 8/8.

## Gate and readings

From `report.json["object_ceiling_v2_gate"]`:

- **Primary gate: failed.** 5 successes (≥ 6 needed) and 5 grasp resets (≥ 7
  needed) on the primary cohort.
- Rollouts exact: 0 mismatches in 2,169 robot-state and 2,169 full-state checks
  on the primary cohort, each counted attempt making executed commands − 1 checks
  of each kind (non-vacuous). All 8 primary ceiling attempts counted; provenance
  valid; `conclusive: true`.
- **`outcome`: `ceiling_inadequate_task_feasible`** (`scripted_oracle` succeeded
  on 8/8 ≥ 6).
- Failure histogram (`failed_reset_furthest_phase`): `lift` 3 — all three primary
  failures stopped in the lift phase, with the apple already out of the hand.
- **`replay_separated_diagnostic`: false** (non-gating). On the fresh resets
  `demo_replay` grasped 4/8 (45100, 45101, 45104, 45105), which is not ≤ 5 − 3. A
  small apple offset does not explain which ones: three of the four grasps are on
  apples 1.35–1.80 cm from the TRAIN centre, but 45100 is 3.20 cm off (the second
  largest primary offset) and 45106, at 1.65 cm, ended `guard_refused`. The
  pattern is unexplained here. On the secondary resets replay grasped 2/8, exactly
  as in TASK-047.
- No ceiling attempt ended in `guard_refused`, `deadline_miss` or a timeout.

## Diagnostics (post-hoc; not gate evidence, no retuning)

- **One failure mode, in the close phase.** On all four failing ceiling attempts
  (45100, 45103, 45105 primary; 45000 secondary) the fingers pushed the apple out
  of the hand: 8–9 commands into the close the apple had moved 0.8–3.1 cm in xy,
  peaking at 2.7–3.5 cm one or two commands later, and it then rolled off the
  table, so the lift phase stalled with the apple gone (`object_dropped` at apple
  height 0.64–0.68 m). The descent, the close-phase entry geometry and the
  blocked-descent hand-over all behaved as designed.
- **The entry geometry does not separate the failures.** At the first close
  command the palm–apple offset was (−1.44 to −1.52 cm, |y| ≤ 0.7 mm, +11.59 to
  +11.67 cm) on the four failures and (−1.44 to −1.55 cm, |y| ≤ 0.9 mm, +11.49 to
  +11.64 cm) on the twelve successes: the two sets overlap. The maximum apple xy
  motion during a successful close was 0.48–0.91 cm on the five primary successes
  and 0.26–1.92 cm over all twelve (45002 and 45005 reached 1.9 and 1.6 cm and
  still held the apple), so the contrast with the failures is a tendency, not a
  clean separation. The difference therefore lies in the contact interaction
  during closure, not in where the hand was placed. The cause is not isolated;
  the exact dynamics were verified bit-exact throughout.
- **The v2 changes that were tested did what they were designed to do.** Every
  descent started at command 108–115 and ended *blocked* 15–23 commands later with
  the palm centred (no descend stall anywhere, against 2/8 in v1); every ceiling
  attempt that kept the apple reached the plate and released on a predicted
  placement (`release_predicted`), and every such release placed the apple: 5/5 on
  the primary cohort and 12/12 pooled, against 3/5 in v1. No transport stall
  occurred, and the 200-command retreat stalls that cost v1 about 390 s per failed
  place are gone — trivially so, since no v2 attempt ever entered `lower` or
  `retreat`: all twelve successes ended during `release`.
- **Release predictor cost.** 31–53 probes per successful attempt (one per
  transport command). It is nearly free: 0.598–0.694 s per command against v1's
  0.575–0.683 s, i.e. roughly 0.01–0.02 s per command, and the maximum observed
  control time was 1.12 s against the 10 s deadline.

## Interpretation and limits

- **Primary hypothesis: rejected at this n.** Under exact dynamics and perfect
  object state, v2 succeeded on 5/8 fresh resets; the gate needed 6 successes and
  7 grasps.
- **What the run does show** (diagnostics, not the gate): the v1 failure
  mechanisms this redesign targeted — the xy-blind descent, the drifting close,
  the off-centre release — did not recur, and the placement predictor converted
  every grasp into a success (5/5 primary, 12/12 pooled with the non-independent
  secondary cohort). The remaining gap is a *grasp-closure* problem: the hand
  ejects the apple on 3/8 primary draws (4/16 pooled, and 1/16 on the tuning
  resets, so the tuning set under-sampled this mode).
- The scripted collector, whose close phase presses towards the same grasp point
  with its own fixed command schedule, did not eject the apple on any of the 16
  resets. Whether the difference is the CEM's per-command jitter during closure,
  the phase-scheduled grasp ramp or the exact contact sequence is **not isolated**
  and needs its own preregistered diagnosis.
- n = 8 primary resets: the thresholds were preregistered decision rules, not
  significance tests, and every count has wide uncertainty.
- Nothing here is a learned result. TASK-033/TASK-034 stay open, the final cohort
  and TEST stay untouched, and the wide-jitter distribution remains validated by
  the scripted reference (16/16 successes across both cohorts).

## Preregistered next step for this outcome

1. Diagnose the close-phase ejection and redesign the grasp closure under a new
   preregistration (for example: an explicit closure schedule during the close,
   a contact-aware close cost, or a cost term on the apple's motion relative to
   the palm while the fingers move).
2. Do not pair this ceiling with a learned model yet.
3. The placement predictor and the descent/carry costs may be carried into that
   redesign unchanged; they are the parts this run exercised successfully.

## Obligations from the pre-run review

- **`planning_dynamics`** (item 5) is recorded above: it stays v1's, the twin
  being unchanged.
- **The smoke ran from the uncommitted tree** (item 7); the frozen run itself ran
  from `7b94f06`, and the post-run verifier confirmed the output snapshot is
  byte-identical to that commit.
- **Timeout margin** (item 9): the maximum observed per-command control time was
  1.12 s against the 10 s deadline, and the longest attempt took 260 s of its
  1,200 s cap. No `deadline_miss` or timeout occurred.
- **"v1 plan unchanged"** (item 11): the reviewer compared a freshly generated v1
  plan with the recorded TASK-047 `plan.json`; only run-time provenance keys
  differ.
- Unlike the v1 manifest, the v2 manifest records the report, plan, frozen-input
  and TRAIN-artifact hashes but not a SHA-256 of every raw artifact.

## Post-run verification

A fresh verifier subagent, independent of the author, reproduced this result from
the raw outputs with its own code. It confirmed: the output snapshot is
byte-identical to commit `7b94f06` (57 files, all `source_hashes`); the frozen
input and TRAIN-artifact hashes; that the executed plan equals the frozen command
and that the recorded ceiling config equals `ObjectCeilingV2Config()`; that all 16
resets regenerate bit-for-bit from the stated RNG rule; the gate recomputed from
the 48 attempt reports (5 successes, 5 grasps, required 6 and 7, failed;
2,169 + 2,169 parity checks with 0 mismatches and exactly `executed steps − 1`
checks per attempt; all attempts counted; `conclusive`, outcome
`ceiling_inadequate_task_feasible`); every cell of both per-attempt tables; the
arm totals, wall times and probe counts; that the secondary `demo_replay` arm
reproduces TASK-047 exactly; the v1 reference numbers; that the manifest's
post-run edit changed no seed, threshold, budget, config or gate text; and that no
45xxx reset was ever stepped by the tuning harness.

It corrected six numbers in this document's diagnostics — the close-phase onset
range, the successful-close motion range (two secondary successes exceeded the
stated bound), the palm–apple entry bounds and a cm/mm slip, the v1 per-command
comparison (v1 was 0.58–0.68 s, so v2 is marginally slower, not cheaper), the
descent-length statement, and the unsupported apple-offset explanation of the
replay counts — and asked for the vacuous `lower`/`retreat` claim, the
primary-vs-pooled counts and the missing pre-run-review obligations to be made
explicit. All of these are applied above. It found no discrepancy in the gate, the
outcome, the provenance or the manifest.
