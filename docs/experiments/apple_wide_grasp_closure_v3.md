# Apple wide-jitter grasp closure v3: preregistered protocol (TASK-051)

**Frozen before any attempt on the gate cohort 45200–45207.** Every arm here is a
NON-LEARNED diagnostic. Nothing in this protocol is a learned result; learned
Apple→Plate remains at zero successes, and TASK-033/TASK-034 stay open.

## Question

TASK-049's object-aware exact-rollout ceiling v2 failed its primary gate on the fresh
wide-jitter development resets 45100–45107: 5/8 full successes and 5/8 grasps against
the required 6 and 7, while the scripted collector succeeded 8/8
(`ceiling_inadequate_task_feasible`). All four failures were one mode — the closing
fingers ejected the apple during the `close` phase.

TASK-051's TRAIN-side diagnosis
([apple_grasp_closure_diagnosis.md](apple_grasp_closure_diagnosis.md)) isolated the
mechanism: the v2 close cost is nearly flat in the palm command while the Dex3 synergy
is shutting, so the CEM's proposal noise sets the palm's motion over those eleven
commands; when the noise drives the palm down and sideways quickly, the still-open thumb
strikes the apple first and sweeps it out of the hand.

**Does a ceiling whose close phase cages the apple — lateral and rotational palm command
frozen, descent-only vertical command, unchanged synergy rate — reach full success ≥ 6/8
and grasp+lift ≥ 7/8 on FRESH wide-jitter development resets under exact dynamics?**

## What changes from v2, and only that

`src/embodied_jepa/object_ceiling_v3.py` subclasses the v2 controller and overrides one
method, `_bounds()`, for the `close` phase only:

| v2 close | v3 close |
|---|---|
| CEM plans all six right-arm deltas in ±`arm_bound` (0.5) | lateral (`action[6:8]`) and rotational (`action[9:12]`) deltas pinned to **zero** |
| vertical delta in ±0.5, so the palm may rise | vertical delta in **[−`close_descent_bound`, 0]**: the palm may only sink |
| grasp command stepped to +1 (rate-limited to 11 commands) | unchanged |
| close cost, phase length, transitions | unchanged |

Everything else is inherited from v2 unchanged: the xy-weighted descent and its
blocked-descent rule, the close cost (pressure to the grasp point, the palm-drift guard,
the apple-disturbance term), the lift/transport/lower carry costs, the release predictor,
the exactly-probed release, the phase order, the twin, and both runtime rollout-parity
checks. The CEM still runs on every close command (with one free degree of freedom), so
the parity checks stay non-vacuous. **v1 and v2 behaviour, plans, gates and evidence are
untouched; tests pin this.**

Simulator physics and contact parameters are **unchanged**. The diagnosis found nothing
unphysical in them and the scripted collector reaches 16/16 on the same resets under the
same physics.

**Credit cannot be split between the two halves of the change.** The paired experiment
supports the lateral/rotational pin: it is what separates `cage` (70/72 grasps, apple held
within 1.45 cm) from `v2`. The descent-only restriction is a conservative addition whose
direct evidence is the two-replay `zonly` row (frozen lateral, symmetric ±0.5 in z: 2/2,
0.56–0.60 cm), which is indistinguishable from `cage` at that n. It is kept because it is
what was tuned and it cannot hurt, but if the gate passes, the pass is not attributable to
it separately.

## Development-tuning disclosure (TRAIN-side tuning resets only)

v3 was designed and tuned on **tuning resets 49100–49131**, drawn with the same
wide-jitter rule (`evaluate_apple.wide_reset`). That range is used nowhere else in the
repository and is disjoint from the TRAIN/VAL/TEST collection 42000–42031, the narrow
development cohort 43000–43004, the final cohort 44000–44019, TASK-047's 45000–45007,
TASK-049's 45100–45107, TASK-049's own tuning range 49000–49015 and this protocol's
gate cohort 45200–45207.

**No control step was simulated on any 45000-, 45100- or 45200-range reset during the
design.** 45200–45207 were only *reset* (no command executed) to run the simulator's
reset-envelope check and to compute the offset table below.

The design probes ran in-process through an uncommitted scratch harness (same
embodiment, twin, controller and unchanged scorer, no rendering of results into the
repository) into the main checkout's ignored `outputs/task051-scratch/`. They are
disclosed in full in the diagnosis document: the forensics on 49100–49115 and the paired
closure experiment (sixteen closure designs replayed from identical close-entry states,
289 replays in all). 49100–49131 is the *declared* tuning range; the seeds actually
stepped are 49100–49117, 49120 and 49124–49131 (27 of the 32), the rest having been
reached by no probe. These are design runs, not evidence. Because of them, a pass shows adequacy
on new draws from the distribution the tuning resets spanned.

## Fresh development resets (new; nothing else changes)

Same distribution and rule as TASK-047/049 (apple ±3 cm, plate ±2 cm around the
unchanged centres (0.34, −0.18) and (0.49, −0.09); `rng = numpy.random.default_rng(seed)`),
new seeds **45200–45207**. All eight pass the simulator's reset-envelope check. The
offsets (cm) are computed from the rule, not from outcomes:

| Reset | Apple dx, dy | Plate dx, dy | Apple offset from centre |
|---|---|---|---:|
| 45200 | −1.70, +2.91 | +1.10, −1.26 | 3.37 |
| 45201 | −1.55, +1.40 | +0.57, +0.88 | 2.09 |
| 45202 | +1.55, +0.17 | −0.57, +0.97 | 1.56 |
| 45203 | +0.07, −0.88 | +0.21, −1.54 | 0.88 |
| 45204 | +1.40, −0.98 | +0.43, +1.44 | 1.71 |
| 45205 | +2.11, +0.80 | +0.00, −0.12 | 2.26 |
| 45206 | +2.31, +2.25 | −0.47, +1.93 | 3.23 |
| 45207 | +1.64, −2.80 | −1.93, −1.04 | 3.25 |

The narrow development cohort, the final cohort 44000–44019, TEST and every earlier plan
are untouched (tests pin this).

## Arms (one command, 72 attempts, mode-major)

| Order | Mode | Resets | Role |
|---|---|---|---|
| 1 | `demo_replay` | 45200–45207, then 45100–45107, then 45000–45007 | reported; never gates |
| 2 | `scripted_oracle` | same | feasibility reference; never gates |
| 3 | `privileged_object` (v3) | same | **gate** on 45200–45207 only |

`demo_replay` retrieval, `scripted_oracle` and the guarded open-loop projection are
exactly TASK-047's and TASK-049's.

## Frozen inputs (unchanged from TASK-043–049)

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5` (retrieval only).
- Action manifest `configs/g1_sim_action.json`; pinned G1/Dex3 assets.
- Regenerated TRAIN artifacts must be byte-identical to TASK-045/046/047/049:
  `state_goals.npz` `839190fc…f88d`, `state_calibration.json` `d8dae051…8d75`.

## Budget (frozen)

- 72 attempts, no retries, replacement seeds or reruns.
- ≤ 1,000 commands; **1,200 s per attempt**; **32,400 s global** including preparation
  and finalization; `min(attempt cap, global remaining)` allocation; CPU, Torch ≤ 4
  threads.
- **Per-command observe+plan deadline 10 s**, as TASK-049. The machine is shared with a
  parallel job; this is not a real-time claim, and a deadline miss would make the
  attempt uncounted.
- Planning case (not a worst case): 48 × ~20 s + 24 × 1,200 s ≈ 29,760 s < 32,400 s. The
  1,200 s cap applies to every attempt, so the arithmetic worst case is 86,400 s; the
  margin that matters is that the 48 non-gating attempts would have to average > 475 s
  each (~38× the smoke's 12.4–12.6 s) before the global cap could reach the eight primary
  ceiling attempts. The order is mode-major, so all 48 non-gating attempts run before any
  ceiling attempt; within `privileged_object` the primary seeds run first (attempts 49–56
  of 72), so budget pressure reaches the secondary cohorts before the gate.
- Expected: 0.6–1.0 s per ceiling command (the TRAIN smoke measured 0.91 s on a loaded
  machine), 250–350 commands per attempt, about 2–3 h in total.

## Primary gate (`report.json["object_ceiling_v3_gate"]`)

`primary_gate_passed` is true only when, on the primary resets 45200–45207:

- `privileged_object` (v3) **full successes ≥ 6 AND grasp resets ≥ 7**;
- zero robot-state and zero full-state rollout parity mismatches in counted ceiling
  attempts, over non-vacuous checks (≥ executed commands − 1 per attempt);
- all 8 primary ceiling attempts are counted;
- provenance is valid.

Counted attempts, clean terminations and zero-stage handling are TASK-047's, unchanged.
`demo_replay`, `scripted_oracle` and the secondary resets never change the gate through
their outcomes. Provenance is a whole-run property: invalid provenance on any attempt
(primary or secondary) invalidates the run and fails the gate.

## Readings and next steps (fixed now; `readings.outcome`)

| Outcome | Condition | Reading | Next step |
|---|---|---|---|
| `ceiling_adequate` | gate passed | Under exact dynamics and perfect object state the v3 design grasps, lifts, transports and places on fresh draws: the design is adequate as the target for a learned controller | **T4**: pair this design with a learned world model whose heads predict its terms (below), trained on the wide-jitter TRAIN corpus (TASK-048), under its own preregistration; keep this ceiling as the exact-dynamics upper reference |
| `grasp_adequate_place_inadequate` | grasp ≥ 7, success < 6 | The closure redesign fixed the grasp; transport/release does not carry it to full success | Redesign only transport/release under a new preregistration; T4 may target the grasp+lift sub-goal meanwhile |
| `ceiling_inadequate_task_feasible` | grasp < 7, scripted successes ≥ 6 | The scripted collector still succeeds where the v3 ceiling does not grasp: the closure redesign is insufficient | Diagnose the remaining failure phases and redesign under a new preregistration; do not pair the ceiling with a learned model |
| `ceiling_inadequate_scripted_also_fails` | grasp < 7, scripted successes < 6 | These draws may exceed the grasp mechanics | Audit the fresh resets and the distribution before any further cost redesign |
| `ceiling_inadequate_feasibility_reference_inconclusive` | grasp < 7, scripted arm incomplete | Cause unattributed | Repair the reference arm; repeat into a new output directory |
| `inconclusive` | a primary ceiling attempt uncounted, provenance invalid or rollouts inexact | — | Fix the defect; rerun into a new versioned output; keep this run as a recorded failure |

Non-gating diagnostics reported with the result:

- `replay_separated_diagnostic`: `demo_replay` grasp resets ≤ ceiling grasp resets − 3 on
  the primary resets (TASK-047's decision rule, as a count only).
- **The secondary cohorts, which are the direct v2-vs-v3 comparison on the same resets**:
  45100–45107 (v2 got 5/8 grasp and 5/8 success) and 45000–45007 (v2 got 7/8 and 7/8;
  v1 got 5/8 and 3/8). Those resets are not independent of the v2 design, and 45100–45107
  are not independent of this diagnosis, so they are reported as diagnostics only.
- Per reset and arm: termination, ordered stages, phase log with reached / blocked /
  `release_predicted` ends, release probes, parity counts, wall time.
- The close-phase ejection rate: per ceiling attempt, whether the apple left the hand
  during the close. Reported, never gating.

With n = 8 the thresholds are preregistered decision rules, not significance tests.

## Cost terms a learned model must predict for T4

Unchanged from TASK-049, minus one: the close phase no longer needs any prediction at
all, because v3's closure is a fixed schedule in proprioception. A learned controller has
to decide *when* to start the close and whether it succeeded, not how to steer during it.

| Term | Used in | Learned-head quantity |
|---|---|---|
| Palm–apple offset (3-D), vs the grasp offset | approach, descend, blocked-descent rule | predicted palm–apple relative position |
| Palm pose (position, top-down angle) | every planned phase | proprioception / forward kinematics (known) |
| Apple displacement from phase start | descend | predicted apple xy change |
| Apple rise above its start | lift, transport carry, lift→transport | predicted apple height |
| Palm–apple offset change (slip) | lift, transport, lower | predicted relative offset change |
| Hand–apple contact | lift/transport/lower, lift→transport, close→lift | predicted contact / grasp-state head |
| Apple–plate xy offset | transport, lower | predicted apple–plate relative position |
| Current apple, plate and contact state | every phase transition (read live here) | state estimation from the observation, not only forward prediction |
| Placement after the scripted gradual release | transport/lower → release gate | predicted "apple comes to rest on the plate within 3 cm" after the release sequence |
| **Nothing** | **close** | the closure is an open-loop proprioceptive primitive |

## Recorded TRAIN smoke (reset 42000 only)

The smoke runs `evaluate_apple.run` with the frozen v3 arguments and `WIDE_V3_COHORT`
patched to `(42000,)`, `WIDE_V3_SECONDARY` to `()` and `wide_reset` to the TRAIN
collector's reset rule, through the complete supervisor, snapshot, preparation and worker
path (scratch output `outputs/task051-scratch/smoke-42000-a` in the main checkout,
uncommitted working tree of this protocol's code). It is a runtime check, not evidence.

- Report `completed`, provenance valid, 3/3 attempts counted, 296.4 s global.
- TRAIN artifacts byte-identical to TASK-045/046/047/049: `state_goals.npz`
  `839190fc…f88d`, `state_calibration.json` `d8dae051…8d75`.
- `demo_replay` succeeded (501 commands, 12.6 s); `scripted_oracle` succeeded
  (501 commands, 12.4 s).
- `privileged_object` v3 succeeded in 283 commands (257.6 s, 0.91 s/command under a
  loaded machine): approach 0, descend 110, close 126 (descent blocked), lift 171,
  transport 203, release 247 (`release_predicted`); 44 release probes; 282/282
  robot-state and 282/282 full-state parity checks exact, i.e. the CEM machinery stays
  non-vacuous through the caged close.
- The one-reset gate object computed `ceiling_adequate` (gate code exercised only).

## Recorded for the results (pre-run review obligations)

- `planning_dynamics` in the plan stays `privileged_mujoco_rollout_object_v1`: the twin is
  v2's and v3 does not override it. `object_ceiling_phases` is likewise the unchanged
  phase list. The v3 identity is carried by `object_ceiling_version`, `result_label` and
  each attempt's `ceiling_version`.
- The "v1 and v2 plans unchanged" assertions inside `tests/test_apple_evaluation.py` are
  self-referential; the independent comparison against `origin/main` in the pre-run review
  is what establishes that claim.
- `tests/test_apple_evaluation.py::test_demo_replay_is_open_loop_non_learned_and_exhausts`
  fails when that module is run in isolation and passes in the full suite. The review
  traced it to a lazy import interacting with a monkeypatched `MuJoCoSimulation`; the code
  path and the test are identical to `origin/main`, so it is inherited, not introduced
  here. Recorded as a follow-up, not fixed in this task.

## Frozen run command

Execute **once**, from a clean checkout of the reviewed commit, into a new directory
under the main checkout's `outputs/`:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output <main checkout>/outputs/apple-wide-grasp-closure-v3 \
  --stage development --goal-kind object --object-ceiling-version 3 --no-proposals \
  --horizon 6 --stride 16 --dwell 1 --candidates 24 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 1200 \
  --max-seconds 32400 --control-timeout 10
```

Seeds default to 45200–45207, then 45100–45107, then 45000–45007; modes to
`demo_replay scripted_oracle privileged_object`. Every outcome, including failures,
stalls and timeouts, goes into `apple_wide_grasp_closure_results_v3.md` and
`benchmarks/manifests/apple-wide-grasp-closure-v3.json`.

## Known risks declared before outcomes

- **Privileged and non-learned.** A pass says the design suffices *given exact dynamics
  and perfect object state*; nothing about whether a learned model can supply the terms.
- **The one observed v3 failure mode in tuning was a guard refusal.** In the paired
  closure experiment one of 72 `cage` replays stopped on the embodiment's measured
  joint-velocity guard (5 rad/s) during the close with the apple ejected, and one more
  reached the replay budget: 2/72 ≈ 2.8% per attempt. Such an attempt is *counted* with
  zero stages. **The gate tolerates exactly one grasp failure in eight**, so at that rate
  the ≥ 7/8 grasp threshold clears with probability ≈ 0.98; a 7/8 or 6/8 outcome would not
  be a surprise, and would not be an excuse.
- **The close is 45 commands while the synergy finishes in eleven.** For the remaining ~34
  the planner may keep commanding descent into the closed hand; that is v2's behaviour,
  deliberately unchanged, and it is the mechanism behind the guard refusal above.
- **v3 forbids rising but does not require sinking.** The close cost is nearly flat in z
  while the palm is blocked, candidate 0 is always the all-zero hold and ties break to the
  lowest index, so nothing forces a descent command early in the close. The paired
  experiment's 70/72 is the empirical evidence that the planner does choose enough
  descent under this bound; there is no analytic guarantee.
- **Seed-integer disjointness does less work than it looks.** The tuning range is drawn
  from the same distribution and is four times larger, so some gate draws sit close to
  tuned draws: the nearest neighbour in apple xy is 45206 ↔ 49129 at 1.8 mm (their plates
  are 3.3 cm apart), and the minimum L∞ over the 4-D reset vector is 0.80 cm
  (45204 ↔ 49115) — tighter than gate-vs-45000 or gate-vs-45100 (1.09 cm each) and than
  the gate cohort's own internal separation (1.09 cm). The resets are distinct and a pass
  is framed as adequacy on new draws from the distribution the tuning resets spanned, not
  as independence from them.
- **Tuning.** 32 tuning resets from the same distribution shaped v3; the gate uses fresh
  draws but the same distribution.
- **Secondary cohorts are not independent.** 45100–45107 produced the failures this
  redesign targets, and 45000–45007 shaped v2. Neither gates.
- **Shared machine.** A parallel job may slow commands; the 10 s deadline and the
  1,200 s cap are sized for that, but a deadline miss or timeout would make the reading
  inconclusive.
- **Short run.** n = 8 primary resets; every count has wide uncertainty.
