# Apple wide-jitter grasp closure v3: results

**Every arm here is a NON-LEARNED diagnostic.** `privileged_object` (v3) plans with
exact MuJoCo rollouts and an exact release probe, and its cost and phase transitions read
simulator object state; `demo_replay` replays a retrieved TRAIN demonstration open loop;
`scripted_oracle` is the privileged scripted collector. No world model contributes to any
outcome. **Learned Apple→Plate remains at zero successes**, and TASK-033/TASK-034 stay
open.

**The preregistered primary gate passed.**

- On the fresh development resets 45200–45207 the v3 ceiling reached full task success on
  **8/8** (≥ 6 required) and the scorer's grasp stage on **8/8** (≥ 7 required).
- All 8 primary ceiling attempts counted, provenance was valid, and rollouts were exact
  (0 mismatches in 2,378 robot-state and 2,378 full-state checks on the primary cohort,
  each attempt checking executed commands − 1 of each kind).
- The computed outcome is **`ceiling_adequate`**, `conclusive: true`.

The preregistered reading is: "under exact dynamics and perfect object state the v3
object-aware cost and phase design grasps, lifts, transports and places on fresh
wide-jitter resets: the cost design is adequate as the target for a learned controller".
The preregistered next step is **T4**: pair this design with a learned world model whose
heads predict its terms, under its own preregistration, keeping this ceiling as the
exact-dynamics upper reference.

**The close-phase ejection that failed TASK-049 did not occur once.** Over all 24 ceiling
attempts the apple moved **0.49–1.35 cm** horizontally during the close; the threshold
this project used to call an ejection is 1.5 cm, and v2's four gated failures were
**2.70–19.10 cm** on the same measure (corrected by the post-run verifier; the
2.45–9.28 cm originally printed here were TRAIN-side tuning figures, not TASK-049's
gated failures). Every one of the 24 attempts terminated with `success`.

## Secondary cohorts (never gating): v3 succeeds where v2 failed

| Cohort | v1 | v2 | **v3** | `scripted_oracle` |
|---|---|---|---|---|
| 45200–45207 (primary, fresh) | — | — | **8/8 grasp, 8/8 success** | 8/8 |
| 45100–45107 (TASK-049 primary) | — | 5/8 grasp, 5/8 success | **8/8, 8/8** | 8/8 |
| 45000–45007 (TASK-047 primary) | 5/8, 3/8 | 7/8, 7/8 | **8/8, 8/8** | 8/8 |

v3 succeeded on **24/24** resets across all three cohorts, including every reset on which
v2 ejected the apple (45100, 45103, 45105 primary-for-v2, and 45000). The two secondary
cohorts are *not* independent evidence — 45000–45007 shaped v2's design and 45100–45107
produced the failures this redesign targets — so they are reported as diagnostics.

## Frozen execution

The command ran once, exactly as frozen in the
[protocol](apple_wide_grasp_closure_v3.md), from source revision
`b673f3de50f04aa50281cf555112da05ac90d0c8` with a clean tracked tree (the only untracked
files being the ignored `outputs/` scratch and the `data`/`checkpoints`/`third_party`
symlinks of the worktree).

- Output directory (new): `outputs/apple-wide-grasp-closure-v3/` in the main checkout.
  Start 2026-09-22T20:06:07Z, end 2026-09-22T21:26:49Z.
- Exit 0; report `completed`; provenance valid; 72/72 attempts counted; **4,836.6 s** of
  the 32,400 s budget. No attempt was shortened or timed out; no `deadline_miss` occurred;
  every ceiling attempt ended in `success`.
- Arm totals: `demo_replay` 147.9 s (4.2–7.8 s per attempt), `scripted_oracle` 185.8 s
  (7.7–8.0 s), `privileged_object` 4,443.5 s (166.8–222.0 s per attempt, 265–342 commands,
  **0.621–0.649 s per command**). Maximum observed per-command control time 1.074 s
  against the 10 s deadline.
- There were no retries, reruns, replacement resets or parameter changes. The narrow
  development cohort, the final cohort 44000–44019 and TEST were not run.

Frozen inputs, as in TASK-043–049: checkpoint `0192b99a…a5a5` (retrieval only), corpus
manifest `6e9a5bcb…0331`, action manifest `f247effe…38b1`, asset manifest `2421e194…e993`.
The regenerated TRAIN artifacts were byte-identical to TASK-045/046/047/049:
`state_goals.npz` `839190fc…f88d`, `state_calibration.json` `d8dae051…8d75`.

`planning_dynamics` in the plan reads `privileged_mujoco_rollout_object_v1`: the twin is
v1's, unchanged, and `object_ceiling_phases` is the unchanged phase list. The v3 identity
is carried by `object_ceiling_version`, `result_label` and each attempt's
`ceiling_version`.

The [compact manifest](../../benchmarks/manifests/apple-wide-grasp-closure-v3.json)
records the hashes, the gate and the per-attempt records.

### Process failure: the run was launched on an interim review file

The frozen run was launched from `b673f3d` while the pre-run reviewer was still
verifying. The author took the review *file on disk* for the verdict; it was an interim
draft reading "CLEAR TO RUN". The reviewer's final verdict, delivered after the launch,
was **BLOCK** on three factual corrections to the record (B2, B3, B4 — see
[the review](../reviews/apple_wide_grasp_closure_v3_review.md)). This is recorded as a
process failure, not excused. What follows from it:

- All three blocking items are statements in prose. None is a seed, threshold, cohort,
  budget, parameter, config or command, and none can change what the evaluator executed or
  how the gate was computed.
- They were applied as **R2 (`b2dc4d4`) before any gated outcome was inspected**, and are
  derived entirely from TRAIN-side data. The reviewer confirmed at `b2dc4d4` that the
  block is cleared and that the run does not need repeating.
- R2's only change under `src/` is the v3 module docstring. The AST of
  `object_ceiling_v3.py` is identical at `b673f3d` and `b2dc4d4` once docstrings are
  blanked, and every line after the module docstring is byte-identical, so the executed
  code is unchanged.
- **Consequence for verification:** the output snapshot is byte-identical to `b673f3d`,
  not to HEAD. A verifier comparing the snapshot against HEAD will find exactly one
  differing file — `src/embodied_jepa/object_ceiling_v3.py`, docstring only. That is
  expected and accounted for here.
- The protocol now records the rule this violated: a gated run starts on the reviewer's
  *reported* verdict, never on an artifact in the working tree.

## Every attempt

"Stages" counts ordered scorer stages; **S** marks full success. Ceiling phase starts are
0-based command indices; a mark is attached to **the phase that ended** — "b" for a phase
that ended *blocked*, "R" for a hand-over on a predicted placement
(`release_predicted`). (TASK-049's table put the mark on the following phase instead; the
convention here matches the legend.) Parity is checks / mismatches (robot-state and
full-state counts were identical in every attempt). "Close apple xy" is the largest
horizontal apple displacement during the 45-command close, recomputed from the trace.

### Primary: fresh resets 45200–45207 (the gate)

| Reset | `demo_replay` | `scripted_oracle` | `privileged_object` v3 | Ceiling phases | Parity | Probes | Close apple xy (cm) | Ceiling wall (s) |
|---|---|---|---|---|---|---:|---:|---:|
| 45200 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111b cl129 li174 tr217R re266 | 300 / 0 | 49 | 0.60 | 193 |
| 45201 | **S** (5) | **S** (5) | **S** (5) | ap0 de114b cl131 li176 tr220R re269 | 303 / 0 | 49 | 0.60 | 196 |
| 45202 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de113b cl130 li175 tr270R re307 | 341 / 0 | 37 | 1.24 | 222 |
| 45203 | 3 stages, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de117b cl136 li181 tr212R re254 | 288 / 0 | 42 | 0.60 | 185 |
| 45204 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111b cl129 li174 tr205R re247 | 282 / 0 | 42 | 0.61 | 181 |
| 45205 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111b cl128 li173 tr234R re276 | 295 / 0 | 42 | 0.67 | 191 |
| 45206 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de110b cl127 li172 tr204R re240 | 275 / 0 | 36 | 0.52 | 176 |
| 45207 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de114b cl134 li179 tr226R re258 | 294 / 0 | 32 | 1.35 | 188 |

Primary totals: ceiling grasp 8/8 and success 8/8; `demo_replay` grasp 2/8 and success
1/8; `scripted_oracle` grasp 8/8 and success 8/8.

### Secondary: TASK-049 resets 45100–45107 and TASK-047 resets 45000–45007 (never gating)

| Reset | `demo_replay` | `scripted_oracle` | `privileged_object` v3 | Ceiling phases | Parity | Probes | Close apple xy (cm) | Ceiling wall (s) |
|---|---|---|---|---|---|---:|---:|---:|
| 45100 | 3 stages, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de112b cl129 li174 tr208R re265 | 300 / 0 | 57 | 1.34 | 194 |
| 45101 | **S** (5) | **S** (5) | **S** (5) | ap0 de112b cl129 li174 tr205R re250 | 283 / 0 | 45 | 0.63 | 182 |
| 45102 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de110b cl126 li171 tr201R re245 | 279 / 0 | 44 | 0.58 | 180 |
| 45103 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de112b cl131 li176 tr207R re242 | 278 / 0 | 35 | 0.49 | 178 |
| 45104 | **S** (5) | **S** (5) | **S** (5) | ap0 de115b cl132 li177 tr208R re260 | 294 / 0 | 52 | 0.65 | 190 |
| 45105 | **S** (5) | **S** (5) | **S** (5) | ap0 de109b cl126 li171 tr201R re244 | 278 / 0 | 43 | 0.87 | 178 |
| 45106 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de109b cl124 li169 tr199R re231 | 265 / 0 | 32 | 0.61 | 169 |
| 45107 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de112b cl135 li180 tr212R re255 | 289 / 0 | 43 | 0.63 | 186 |
| 45000 | **S** (5) | **S** (5) | **S** (5) | ap0 de109b cl126 li171 tr199R re252 | 272 / 0 | 53 | 0.69 | 176 |
| 45001 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111b cl128 li173 tr266R re309 | 327 / 0 | 43 | 0.96 | 209 |
| 45002 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de111b cl127 li172 tr202R re248 | 281 / 0 | 46 | 0.59 | 177 |
| 45003 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de108b cl125 li170 tr198R re251 | 285 / 0 | 53 | 0.67 | 179 |
| 45004 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de114b cl134 li179 tr210R re249 | 284 / 0 | 39 | 0.61 | 177 |
| 45005 | 1 stage, `demo_exhausted` | **S** (5) | **S** (5) | ap0 de110b cl128 li173 tr203R re244 | 264 / 0 | 41 | 0.49 | 167 |
| 45006 | **S** (5) | **S** (5) | **S** (5) | ap0 de110b cl131 li176 tr206R re253 | 287 / 0 | 47 | 0.63 | 184 |
| 45007 | 1 stage, `guard_refused` | **S** (5) | **S** (5) | ap0 de114b cl132 li177 tr214R re257 | 291 / 0 | 43 | 0.51 | 187 |

Secondary totals: ceiling grasp 16/16 and success 16/16; `demo_replay` grasp 6/16 and
success 5/16; `scripted_oracle` 16/16. On 45000–45007 `demo_replay` reproduces TASK-047
and TASK-049 exactly (grasp 2/8, success 2/8).

## Gate and readings

From `report.json["object_ceiling_v3_gate"]`:

- **Primary gate: passed.** 8 successes (≥ 6 needed) and 8 grasp resets (≥ 7 needed) on
  the primary cohort.
- Rollouts exact: 0 mismatches in 2,378 robot-state and 2,378 full-state checks on the
  primary cohort, each counted attempt making at least executed commands − 1 checks of
  each kind (non-vacuous; 6,935 + 6,935 checks across all 24 ceiling attempts, 0
  mismatches). All 8 primary ceiling attempts counted; provenance valid; `conclusive:
  true`.
- **`outcome`: `ceiling_adequate`.**
- **`replay_separated_diagnostic`: true** (non-gating). On the fresh resets `demo_replay`
  grasped 2/8 against the ceiling's 8/8, which is ≤ 8 − 3.
- `failed_reset_furthest_phase` is empty: no ceiling attempt failed.
- No ceiling attempt ended in `guard_refused`, `phase_stall`, `deadline_miss` or a
  timeout. The declared guard-refusal risk (2/72 ≈ 2.8% in tuning) did not materialise
  in 24 attempts.

## Diagnostics (post-hoc; not gate evidence)

- **The redesign did what the diagnosis said it would.** Across all 24 ceiling attempts
  the close-phase apple displacement was 0.49–1.35 cm, with zero attempts at or above the
  1.5 cm ejection threshold. On the same measure v2 produced four ejections of
  **2.45–9.12 cm** in the 16 forensic tuning closes (49100–49115), and in its own 16 gated
  attempts **six** closes reached 1.5 cm — 2.70, 10.90, 11.04 and 19.10 cm on the four that
  dropped the apple (45103, 45105, 45000, 45100), plus 1.61 cm on 45005 and 1.92 cm on
  45002, both of which still succeeded. (Corrected by the post-run verifier; this bullet
  previously read "four ejections of 2.45–9.28 cm in 16 tuning closes and four in its 16
  gated attempts". 9.28 cm is the paired experiment's `v2` maximum over 53 replays, not one
  of the sixteen forensic closes, whose maximum is 9.12 cm; and six, not four, of v2's
  gated closes crossed the 1.5 cm threshold.)
- **Descent, carry and placement carried over unchanged, as predicted.** Every descent
  ended *blocked* 15–23 commands after it began (108–117), every close ran its 45
  commands, every lift reached the transport phase, and **every transport handed over on
  a predicted placement (`release_predicted`), 24/24, and every such release placed the
  apple**. No transport stall, no `lower` phase and no retreat occurred; all 24 successes
  ended during `release`.
- **Cost.** 0.621–0.649 s per ceiling command, 265–342 commands per attempt, 32–57 release
  probes per attempt. The whole run took 4,836.6 s of a 32,400 s budget.
- **The `demo_replay` reference is unchanged.** Its 2/8 grasp on the fresh cohort and 6/16
  on the secondary cohorts reproduce the TASK-047/049 pattern, so the separation between
  object-aware control and open-loop replay is intact on new draws.

## Interpretation and limits

- **The claim this run supports, exactly:** given *exact dynamics* and *perfect object
  state*, the v3 cost/phase design — the v2 descent and carry costs, the v2 release
  predictor, and a close phase that pins the lateral and rotational palm command and lets
  the planner only sink — reaches full Apple→Plate success on 8/8 fresh wide-jitter
  development resets. It says **nothing** about whether a learned model can supply those
  terms.
- **n = 8 primary resets.** The thresholds were preregistered decision rules, not
  significance tests. 8/8 is consistent with a true per-reset success probability anywhere
  above roughly 0.69 (one-sided 95%); the 24/24 pooled count is not independent, because
  16 of those resets shaped the v1/v2 designs or produced the failures this one targets.
- **Credit cannot be split between the two halves of the closure change.** The paired
  TRAIN experiment supports the lateral/rotational pin; the descent-only restriction rests
  on a two-replay comparison and is a guard-rail. This run cannot separate them.
- **The tuning distribution is the gate distribution.** The gate uses fresh draws, but
  from the same wide-jitter rule the 27 tuning resets spanned, and the protocol discloses
  that some gate draws sit close to tuned draws (nearest apple-xy neighbour 45206 ↔ 49129
  at 1.8 mm).
- **Nothing here is a learned result.** TASK-033/TASK-034 stay open, the final cohort
  44000–44019 and TEST stay untouched, and the wide-jitter distribution remains validated
  by the scripted reference (24/24 successes).

## Preregistered next step for this outcome

1. **T4**: pair this cost/phase design with a learned world model whose heads predict its
   terms, trained on the wide-jitter TRAIN corpus (TASK-048), under its own
   preregistration; keep this ceiling as the exact-dynamics upper reference.
2. The terms that head must predict are tabulated in the
   [protocol](apple_wide_grasp_closure_v3.md#cost-terms-a-learned-model-must-predict-for-t4).
   The close phase needs **no** prediction: v3's closure is a fixed schedule in the
   commanded palm delta and the grasp command, so the learned controller has to decide
   *when* to start the close and whether it succeeded, not how to steer during it. The
   hardest remaining head is the release-outcome predictor, which here runs the true
   simulator.

## Obligations from the pre-run review

- **`planning_dynamics`** is recorded above: it stays v1's, the twin being unchanged.
- **The run executed from `b673f3d`, not HEAD**, and the snapshot therefore differs from
  HEAD in exactly one file (the v3 module docstring). Recorded above under the process
  failure.
- **The in-repo "v1/v2 plans unchanged" assertions are self-referential**; the
  independent comparison against `origin/main` in the review is what establishes that
  claim (v1 plan identical, v2 plan identical, v2 gate identical over 50 randomized
  record sets).
- **A pre-existing, order-dependent test failure** (`test_demo_replay_is_open_loop_non_learned_and_exhausts`
  fails when its module runs alone, passes in the full suite) is inherited from `main`
  and recorded as a follow-up on the MC task, not fixed here.
- The smoke ran from the uncommitted tree; the frozen run itself ran from `b673f3d`.

## Post-run verification

Independent post-run verifier, fresh context, read-only against the raw outputs in the
main checkout, working from `93aff53`. Every figure below was recomputed with the
verifier's own code from `outputs/apple-wide-grasp-closure-v3/` (the 72 per-attempt
`report.json` files and their `trace.jsonl`), **not** read from
`report.json["object_ceiling_v3_gate"]`. No evaluation was re-run and no 45xxx reset was
stepped.

**Verdict: the recorded gate result is sound.** The primary gate passes on a fully
independent recomputation, the provenance chain is intact, and the run matches the frozen
command exactly. Two numbers in this document were wrong and have been corrected (both
concerned *v2*, not this run; both understated how badly v2 failed). Nothing else changed.

### Confirmed

1. **Provenance — both halves confirmed.** All 37 files in `plan.json["source_hashes"]`
   match the snapshot under `outputs/apple-wide-grasp-closure-v3/source/` byte for byte
   (0 mismatches, 0 missing on either side). Every snapshot file that exists in git at
   `b673f3d` is byte-identical to it. (The snapshot holds 61 files: the 37 hashed sources
   plus 21 `__pycache__/*.pyc` and three JSON files. `assets/manifest.json` and
   `configs/g1_sim_action.json` hash to the plan's `asset_manifest_sha256` and
   `action_manifest_sha256` respectively; the third, `models/jepa_wms_source.json`, is a
   tracked file outside `source_hashes` and is byte-identical to `b673f3d`, as are the
   other two.) Against HEAD
   `93aff53` exactly one file differs — `src/embodied_jepa/object_ceiling_v3.py` — and the
   difference is confined to the module docstring: the parsed AST is identical once
   docstrings are blanked, and every line after the module docstring (line 53 at
   `b673f3d`, 58 at HEAD) is byte-identical. HEAD's copy equals `b2dc4d4`'s.
2. **Executed command = frozen command.** `plan.json` reproduces every frozen argument:
   `stage=development`, `goal_kind=object`, `object_ceiling_version=3`,
   `demonstration_proposals=false`, horizon 6, stride 16, dwell 1, candidates 24,
   iterations 2, commitment 1, `max_steps` 1000, `attempt_max_seconds` 1200,
   `max_seconds` 32400, `control_timeout_seconds` 10 — identical to the manifest's
   `frozen_command` and `budget`. Seeds, cohorts and order match: mode-major
   `demo_replay → scripted_oracle → privileged_object`, primary 45200–45207 first inside
   each mode, so the eight gated ceiling attempts are 49–56 of 72 as the protocol states.
   `resolved_plan.json` differs from `plan.json` only by adding the two TRAIN-artifact
   hashes and the retrieval candidate list; the 72 attempt entries are identical.
   The recorded `object_ceiling` block equals `ObjectCeilingV3Config()` field for field
   (44 fields; only `seed` is not serialised, and it is recorded as `controller.seed = 0`),
   including `close_descent_bound = 0.5`. All 24 resets regenerate **bit-for-bit** from
   `rng = numpy.random.default_rng(seed)` with the declared centres and jitter, and the
   protocol's eight-row offset table reproduces to the printed 2 dp.
3. **The gate, recomputed from the 72 attempt reports.** Primary 45200–45207,
   `privileged_object`: **8/8 full successes** (≥ 6) and **8/8 grasp resets** (≥ 7), all
   8 attempts counted under TASK-047's rule (`status == completed`, termination not in
   `runtime_error`/`deadline_miss`/`attempt_timeout`, provenance valid). Parity: **2,378
   robot-state and 2,378 full-state checks, 0 mismatches**, and every attempt's check count
   equals exactly `executed_steps − 1`, so the non-vacuity rule holds with no slack. Across
   all 24 ceiling attempts: 6,935 + 6,935 checks, 0 mismatches. No attempt anywhere in the
   run has `provenance_valid: false`. `primary_gate_passed = true`, and under the
   preregistered precedence the outcome is **`ceiling_adequate`**, conclusive.
   `replay_separated_diagnostic` holds (`demo_replay` grasp 2 ≤ 8 − 3).
4. **Both per-attempt tables: every cell reproduces, with zero discrepancies.** 24 rows ×
   8 columns. `demo_replay` terminations and stage counts, `scripted_oracle` 24/24, the
   ceiling's ordered stages, the phase strings under the declared "mark on the phase that
   ended" convention (every `b` sits on `descend`, every `R` on `transport`), parity counts,
   release probes, close-phase apple xy — independently recomputed from each trace as the
   largest horizontal apple displacement from the apple's position at the first close
   command — and wall times to the printed precision. Every close ran exactly 45 commands;
   all 24 traces show the same phase sequence
   `approach → descend → close → lift → transport → release`, with no `lower` and no
   `retreat`, ending in `release`.
5. **The other numbers.** Arm totals 147.9 / 185.8 / 4,443.5 s and the per-attempt ranges
   4.2–7.8, 7.7–8.0, 166.8–222.0 s; 265–342 ceiling commands; 0.621–0.649 s per command
   (0.62146–0.64898 exactly); 32–57 release probes; maximum per-command control time
   **1.074 s** (45007, step 241) with **zero** commands over the 10 s deadline; 4,836.6 s of
   32,400 s (14.9%); start 20:06:07Z / end 21:26:49Z consistent with that wall time; 54/72
   attempt successes (24 + 24 + 6). The TRAIN artifacts hash to `839190fc…f88d` and
   `d8dae051…8d75` on recomputation from the files in the output directory, and the four
   frozen-input hashes match the protocol. `plan.json`, `resolved_plan.json` and
   `report.json` hash to the values the manifest records. Descents ended blocked 15–23
   commands after starting at 108–117; 24/24 transports handed over on `release_predicted`
   and every such release placed the apple. Zero of the 24 close-phase displacements reach
   1.5 cm (range 0.49–1.35 cm). The v1/v2 rows of the secondary-cohort table match
   `apple_wide_object_ceiling_results_v2.md` (v2: 5/8 and 5/8 on 45100–45107, 7/8 and 7/8
   on 45000–45007; v1: 5/8 and 3/8).
6. **Leakage — clean.** A structural sweep of every run directory under `outputs/`
   (extracting real seed fields and attempt-directory names, not digit substrings) finds
   **no** 45200–45207 attempt anywhere except this run, and **no** 44000–44019 attempt
   anywhere at all. `outputs/task051-scratch/` contains only 42000 (the smoke) and the
   tuning seeds 49100–49117, 49120, 49124–49131 — 27 seeds, exactly as declared, with
   49117 appearing in `scan-b-0.log` although it wrote no result file. The declared tuning
   range (49100–49131) and the gate range are disjoint. All 16 retrieval candidates are
   TRAIN-split episodes; no VAL, TEST or holdout episode was decoded.
7. **No post-run change to anything frozen.** `scripts/evaluate_apple.py` — which carries
   the cohorts, the gate rule and the thresholds — was last modified at `36025a9`, *before*
   the preregistration commit `dc44f42`, and is untouched by `b673f3d`, `b2dc4d4`, `f59b8df`
   and `93aff53`. Comparing the manifest leaf by leaf across `dc44f42 → b673f3d → HEAD`:
   **zero** leaf values changed except `status` ("preregistered" →
   "run_complete_primary_gate_passed"); `b673f3d` added only `planning_dynamics`,
   `pre_run_review` and `tuning_seeds_stepped`, and the post-run commit added only
   `run`, `result`, `review` and `attempts_detail`. No seed, threshold, budget, config
   value, gate-rule string or outcome-list entry was edited after the run.
8. **Statistics and caveats.** The one-sided 95% lower bound for 8/8 is
   0.05^(1/8) = 0.6877, so "above roughly 0.69" is right. The non-independence caveats on
   the secondary cohorts, the tuning-distribution caveat and the credit-splitting caveat
   are accurate and are not walked back anywhere. The nearest-neighbour disclosures
   reproduce exactly: 45206 ↔ 49129 at 1.8 mm in apple xy with plates 3.3 cm apart, minimum
   4-D L∞ 0.80 cm (45204 ↔ 49115), against 1.09 cm for gate-vs-45000s, gate-vs-45100s and
   within the gate cohort. The process-failure section matches the review: the final
   verdict was BLOCK on B2/B3/B4, all prose, applied at `b2dc4d4` and confirmed cleared.
   The diagnosis itself was re-derived from the scratch data and holds: the 16 forensic
   closes give four ejections at 2.45/3.12/7.79/9.12 cm against twelve holds at
   0.19–0.78 cm and the collector at 0.71–0.74 cm with no ejection, and every row of the
   paired-experiment table reproduces from the scan logs (`cage` 72 runs / 24 seeds /
   70 grasped / 0.33–5.61 cm, 0.33–1.45 cm when grasped; `v2` 53 / 19 / 49 /
   0.18–9.28 cm). The 2/72 guard-refusal risk is the true `cage` rate.
9. `pytest -q`: **676 passed, 13 skipped** (3 `LEROBOT_SOURCE`, 3 graphics opt-in, 7 no
   `timm`), exit 0. `ruff check` clean; `ruff format --check` clean (107 files);
   `mc validate` passes.

### Corrected by this verification

Two numbers about **v2** were wrong. Both understated v2's failures, so correcting them
does not weaken this run's result — it strengthens the contrast.

- **Line 28 (headline):** "v2's four failures were 2.45–9.28 cm" → **2.70–19.10 cm**.
  Recomputing the document's own measure on the v2 gated traces
  (`outputs/apple-wide-object-ceiling-v2/`), the four TASK-049 failures are 45103 at
  2.70 cm, 45105 at 10.90 cm, 45000 at 11.04 cm and 45100 at 19.10 cm. The 2.45–9.28 cm
  figures are TRAIN-side tuning numbers imported from the diagnosis and do not describe the
  gated failures.
- **Diagnostics bullet 1:** "four ejections of 2.45–9.28 cm in 16 tuning closes and four in
  its 16 gated attempts" → the 16 forensic tuning closes give four ejections of
  **2.45–9.12 cm** (9.28 cm is the paired experiment's `v2` maximum over 53 replays, not
  one of those sixteen), and on the ≥ 1.5 cm measure **six**, not four, of v2's 16 gated
  closes ejected — 45002 at 1.92 cm and 45005 at 1.61 cm crossed the threshold and still
  succeeded, as the diagnosis's own caveat about 49103 anticipates.

### Limits of this verification

- Provenance is verified by hash agreement between the plan, the snapshot and git; the
  verifier did not re-execute the simulator, so the traces are taken as recorded. The
  runtime parity checks are self-reported by the evaluator, and their *integrity* rests on
  code the verifier read rather than on an independent rollout.
- The protocol's claim that 45200–45207 were reset-only during design leaves no artifact
  and remains unfalsifiable from disk, as the pre-run review also noted. What is
  verifiable — that no attempt was ever *stepped* on those seeds outside this run — holds.
- This remains a NON-LEARNED privileged diagnostic. Nothing here bears on whether a learned
  model can supply the cost terms; learned Apple→Plate is still 0 successes and
  TASK-033/TASK-034 stay open.
