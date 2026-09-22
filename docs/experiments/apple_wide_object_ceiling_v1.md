# Apple wide-jitter object-aware ceiling v1: preregistration

This is the prospective TASK-047 protocol (roadmap step T1). It is committed
before any attempt on the new development resets. **Every arm is a
NON-LEARNED diagnostic:**

- `privileged_object` is an exact MuJoCo-rollout planner whose cost and phase
  transitions read simulator object state.
- `demo_replay` is open-loop replay of a retrieved TRAIN demonstration.
- `scripted_oracle` is the privileged scripted collector.

No outcome here is a learned result, and none counts toward TASK-033/TASK-034.
Learned Apple→Plate stays at zero successes.

## Why

TASK-043–046 ran on development resets 43000–43003, whose reset jitter is
±6 mm. Two problems made those resets unable to answer the project's question.

1. **Replay already succeeds.** Open-loop `demo_replay` grasped 4/4 and
   succeeded 3/4 (TASK-043, reproduced in TASK-046). At this jitter the
   benchmark cannot tell a world model from memorised actions.
2. **No planner cost has been object-aware.** Every privileged ceiling so far
   scored only the robot's own arm and hand joints: the endpoint ceiling in
   TASK-044 (0/4 stages), trajectory tracking in TASK-045 (grasp 0/4) and hybrid
   control in TASK-046 (primary grasp 0/4). None of them knew where the apple
   was.

T1 therefore asks two questions on a new, wider development distribution:

- Does the distribution separate object-aware control from replay?
- Is an object-aware phase cost adequate when the dynamics are exact?

## Question and hypothesis

**Hypothesis (primary gate).** On the 8 new wide-jitter development resets:

- the object-aware privileged ceiling reaches the scorer's `grasp` stage on
  **≥ 6/8** resets; **and**
- `demo_replay` reaches `grasp` on **≤ (ceiling grasp resets − 3)**.

In the unchanged `AppleToPlateTask`, `grasp` means grasp plus lift: `reach`,
then an apple rise of at least 5 cm while the hand is in contact.

## New development distribution (the existing ones are not changed)

| | Narrow development (unchanged) | Final (unchanged) | **Wide development (new)** |
|---|---|---|---|
| Resets | 43000–43004 | 44000–44019 | **45000–45007** |
| Apple xy jitter | ±6 mm | ±6 mm | **±3 cm** |
| Plate xy jitter | ±6 mm | ±6 mm | **±2 cm** |

- **Centres.** The centres are unchanged: apple (0.34, −0.18) and plate
  (0.49, −0.09), world metres.
- **Rule.** The rule is `rng = default_rng(seed)`, then
  `object_xy = centre + rng.uniform(-0.03, 0.03, 2)`, then
  `plate_xy = centre + rng.uniform(-0.02, 0.02, 2)`. It is implemented as
  `evaluate_apple.wide_reset`.
- **Seed range.** The range 45000–45007 appears nowhere else in the repository.
  - TRAIN, VAL and TEST episodes are the 42000–42031 collection (and branches
    derived from those episodes).
  - The mechanics probes used 41000–41101, the MVP 20000-range and the
    manipulation pilots 300/1000.
- **Protected cohorts.** The narrow development cohort, the final cohort
  44000–44019, TEST and TASK-034's acceptance criteria are untouched. The code
  path for every existing goal kind is unchanged; a test pins this.
- **Validity.** All 8 resets satisfy the simulator's collision-envelope check;
  the smallest apple–plate clearance is 5.4 cm (45004). No outcome was run.
- **The 8 resets (offset from centre, cm).** These numbers are computed from the
  rule; they are not outcomes.

| Reset | Apple dx, dy | Plate dx, dy |
|---|---|---|
| 45000 | −0.04, +0.65 | +1.95, −0.88 |
| 45001 | +2.41, −2.34 | −1.24, +1.99 |
| 45002 | −1.31, −2.51 | −0.64, +1.61 |
| 45003 | −2.76, +2.67 | +0.73, +1.85 |
| 45004 | −0.94, +2.96 | −1.68, −0.79 |
| 45005 | +1.22, −1.76 | −0.45, −0.35 |
| 45006 | −1.24, +0.30 | +0.66, +0.59 |
| 45007 | +1.29, −0.77 | +0.78, −0.26 |

*Declared before outcomes:* on 45000 the apple lies within the narrow ±6.5 mm,
so `demo_replay` may well succeed there. The draw is kept; seeds are never
re-drawn.

## Arms (one command, 24 attempts, mode-major)

| Order | Mode | What it is | Role |
|---|---|---|---|
| 1 | `demo_replay` | The TASK-043 retrieved TRAIN demonstration, replayed open loop from reset | **gate** (upper bound on its grasps) |
| 2 | `scripted_oracle` | The TRAIN collector's scripted initial-truth policy (`scripted.apple_collector_policy`, identical to `scripts/collect_apple.py:make_policy(-0.035)`) | feasibility reference; never gates |
| 3 | `privileged_object` | The object-aware exact-rollout CEM ceiling (below) | **gate** (≥ 6/8) |

- **Order.** The cheap arms run first (TRAIN smoke: about 8 s each), so the
  expensive ceiling arm cannot starve them. The ceiling runs last and has
  ample budget (see Budget).
- **Retrieval.** `demo_replay` retrieval is unchanged. The 16 successful
  NOMINAL apple/plate TRAIN episodes are the candidates, and the one whose
  frame-0 RGB is nearest (under the frozen sensor image distance) to the live
  initial RGB is chosen. No state, object pose or score is used.
- **Execution.** `demo_replay` and `scripted_oracle` pass every command through
  the same bounds clip and mandatory `project_candidates`.

## The object-aware ceiling (`src/embodied_jepa/object_ceiling.py`)

### Isolation

- **Privileged by construction.** It is not in `MODELS`, and both
  `ObjectRolloutModel` and `ObjectCeilingController` require
  `acknowledge_privileged_ceiling=True`.
- **Where it may run.** The evaluator builds it only for the `privileged_object`
  mode of a declared `goal_kind: object` development plan. Object-plan modes
  are refused by every other goal kind.
- **Where simulator truth goes.** Simulator truth (apple pose, plate pose, hand
  contact, palm pose) enters only this arm's rollout features, costs and phase
  transitions. It never enters any model input. The frozen sensor checkpoint
  serves only `demo_replay` retrieval.

### Dynamics

The ceiling reuses the TASK-044 `PrivilegedRolloutModel` twin unchanged:

1. restore a copy of the live MjData;
2. project each candidate step and `execute` it through the unchanged
   embodiment (IK, rate limits, guards, PD transport);
3. mark a step invalid if it is rejected.

For each executed step it records these privileged features:

- palm position (world frame);
- palm orientation error;
- apple position;
- hand–apple contact;
- whether the apple has dropped.

The TASK-044 runtime parity check still compares the measured robot state with
the previous search's simulated first steps. It is extended to the complete
MuJoCo `qpos`/`qvel`, which includes the apple.

### Planner

A CEM-style search over the 14-D normalised action. It keeps a single elite:
each round re-centres on the best sequence so far, as in the TASK-045 tracker.
There are no demonstration proposals and no action seeding.

- Horizon 6, 24 candidates, 2 rounds, proposal std 0.3 halving to a 0.05
  floor, commitment 1.
- Candidates 0 and 1 are the hold and warm-start sequences. The warm start is
  the shifted previous plan, reset at every phase change.
- The right-arm deltas are bounded to ±0.5, as in every earlier plan. The left
  arm is 0 and the left grasp −1.
- **The right grasp is scheduled by phase through the bounds:** −1 (open) in
  approach, descend, release and retreat; +1 (closed) in close, lift, transport
  and lower. The embodiment's rate limit ramps it.
- **Candidate cost.** Each candidate is scored by the mean over the horizon of
  the per-step phase cost below. A rejected step costs 1,000 plus the cost of
  the last state that candidate reached (the TASK-044 convention).
- **Selection.** Only feasible candidates (after the live projection) are
  rolled out. The lowest cost wins; ties go to the lowest index. The RNG is
  seeded with the reset seed.

### Phase costs

In the table:

- p is the palm and a the apple.
- The grasp offset is `g(h) = a + (−0.015, 0, h)`, the collector's own offsets.
- R adds `0.2 · angle(palm, top-down)` to every phase.
- C adds a hand-contact penalty of 0.05 when the hand has no contact.
- S adds `0.25 · |(a − p) − (a − p)_phase start|`, a slip term.

| Phase | Per-step cost (+ R) | Ends when (live simulator truth) |
|---|---|---|
| approach | ‖p − g(0.13)‖ | ‖p − g(0.13)‖ < 1 cm and angle < 0.1 rad |
| descend | ‖p − g(0.052)‖ + 1.0 · ‖a_xy − a_xy,start‖ | ‖p − g(0.052)‖ < 8 mm, or **blocked** within 1 cm xy of g |
| close | ‖p − g(0.052)_start‖ + 1.0 · apple xy displacement | 45 commands (collector's count) |
| lift | max(0, a_z,0 + 0.15 − a_z) + ‖p_xy − p_xy,start‖ + S + C | rise ≥ 10 cm with contact |
| transport | ‖a_xy − plate_xy‖ + max(0, a_z,0 + 0.12 − a_z) + S + C | ‖a_xy − plate_xy‖ < 1.5 cm, or **blocked** within the 4 cm plate radius |
| lower | 2 · ‖a_xy − plate_xy‖ + \|a_z − (rest_z + 1.5 cm)\| + S + C | within 1 cm of that height, or **blocked** |
| release | ‖p − p_start‖ | 40 commands |
| retreat | ‖p − (p_start + 8 cm up)‖ | (runs until success or a limit) |

- **Blocked.** A phase is blocked when its progress measure improved by less
  than 2 mm over the last 10 commands. The measures are palm height (descend),
  apple–plate xy distance (transport) and apple height (lower).
- **Why the descent is always blocked.** The collector's thumb meets the table
  about 12 cm above the apple centre, before the 5.2 cm grasp height, in every
  collector demonstration. In the TRAIN-reset scripted rollout, the thumb–table
  contact was seen at palm height +0.117 m.
- **Stops.** Clean stops are counted, with the stages reached so far:
  - `phase_stall`: 200 commands in one phase;
  - `object_dropped`: apple height below 0.70 m;
  - `guard_refused`: the unchanged joint-velocity guard refuses a projection;
  - `step_limit`: 1,000 commands.

### Design disclosure (what was tuned, where)

The ceiling was developed on **TRAIN reset 42000** and on **4 deterministic
corner positions** (apple ±3 cm, plate ∓2 cm, seed 0). These are design probes
outside the cohort; no 45000–45007 reset was simulated.

- **Changes the probes drove:**
  - The orientation error was changed to the geodesic angle. The collector's
    small-angle cross product has a stationary point at 90°, which is exactly
    where the palm starts, and the planner stalled there.
  - A blocked descent ends the descent phase, after a probe stalled at the
    unreachable 5.2 cm grasp height.
  - Blocked transport and lower phases hand over to the next phase, after
    probes stalled 3.6 cm from the plate centre and above the plate.
  - The rotation weight went from 0.05 to 0.2, and horizon, candidates and std
    were changed for approach speed.
- **Last probe versions:** 42000 and corner (+3, −3 | −2, +2) cm both reached
  full success in 387 and 338 commands (about 270 s and 230 s), with 0/387 and
  0/338 full-state parity mismatches.
- **Earlier version** (before the blocked transport/lower rule): 42000 and
  three corners reached grasp and transport, then stalled in transport or
  lower. The corners were (+3, +3 | −2, −2), (−3, −3 | +2, +2) and
  (+3, −3 | −2, +2).
- **Earliest versions:** these stalled in approach (the 90° plateau) or failed
  with a cost-code bug before grasping.
- **The scripted collector** succeeded on 42000 and on all four corners (the
  three above plus (−3, +3 | +2, −2)).

These are design runs, not evidence, and are not reported as results. Because
of them, a ceiling pass shows adequacy on new draws from the distribution the
probes spanned, not on an independent held-out distribution.

## Frozen inputs (unchanged from TASK-043–046)

- **Corpus.** `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- **Checkpoint.** `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. It serves
  retrieval only.
- **Configuration and assets.** Action manifest `configs/g1_sim_action.json`
  and the pinned G1/Dex3 assets.
- **Regenerated TRAIN artifacts.** Preparation regenerates `state_goals.npz`
  and `state_calibration.json` (stride 16, dwell 1). They must be
  byte-identical to TASK-045/046 (`839190fc…f88d`, `d8dae051…8d75`). The TRAIN
  smoke confirmed this.

## Budget (frozen)

- **Attempts.** One command runs 24 attempts: `demo_replay` on 45000–45007,
  then `scripted_oracle`, then `privileged_object`. There are no retries,
  replacement seeds or reruns.
- **Limits.**
  - at most 1,000 commands;
  - a 5 s per-command observe+plan deadline;
  - **960 s per attempt**;
  - **8,400 s global**, including preparation and finalization;
  - the existing `min(attempt cap, global remaining)` allocation;
  - CPU, with Torch capped at 4 threads.
- **Worst case fits.** With every ceiling attempt at its cap, the total is
  16 × about 15 s + 8 × 960 s ≈ 7,920 s < 8,400 s. So no ceiling attempt can be
  shortened by the global cap unless preparation exceeds about 480 s (TRAIN
  smoke: about 10 s).
- **Expected time.** About 0.7–0.8 s per ceiling command and 350–450 commands
  per attempt: about 250–350 s per attempt and about 45 minutes in total.
- **Declared risk.** A 1,000-command ceiling attempt takes about 700–800 s
  (0.7–0.8 s per command), below the effective 955 s cap.
  - An attempt that times out, is not started or raises counts as 0 stages;
    stages it latched are reported as uncounted.
  - Any such attempt in the ceiling or `demo_replay` arm makes the reading
    inconclusive.

## Primary gate (`report.json["object_ceiling_gate"]`)

`primary_gate_passed` is true only when all of the following hold:

- `privileged_object` grasp resets ≥ 6 (of 8);
- AND `demo_replay` grasp resets ≤ `privileged_object` grasp resets − 3;
- AND zero robot-state and zero full-state rollout parity mismatches occur in
  counted ceiling attempts;
- AND those checks are non-vacuous: each counted ceiling attempt made at least
  (executed commands − 1) checks of each kind;
- AND all 8 `privileged_object` and all 8 `demo_replay` attempts are counted
  (R1);
- AND provenance is valid.

Which attempts count:

- **Counted:** status `completed`, a termination other than
  `runtime_error`/`deadline_miss`/`attempt_timeout`, and valid provenance.
- **Clean terminations whose stages count:** `success`, `step_limit`,
  `phase_stall`, `object_dropped`, `guard_refused`, `execution_rejected`,
  `demo_exhausted` and `policy_complete`.
- **Missing or failed attempts** count as 0.
- **Guard stops (R1).** In this object plan, `guard_refused` is a clean,
  counted stop in every arm. It is recorded when the unchanged joint-velocity
  guard refuses to project a command: a `demo_replay` or `scripted_oracle`
  command, or ceiling candidates. Any other projection error stays an uncounted
  `runtime_error`. Earlier plans are unchanged.

`scripted_oracle` never changes the gate.

## Readings and next steps (fixed now; `readings.outcome`)

A reading is conclusive only when every ceiling and `demo_replay` attempt is
counted, provenance is valid and rollouts are exact.

| Outcome | Condition | Reading | Next step |
|---|---|---|---|
| `separated` | gate passed | The distribution separates object-aware control from replay, and the object-aware phase cost is adequate under exact dynamics | **T2**: preregister and collect the wide-jitter TRAIN data (about 200 episodes, ≥ 96 px, dense grasp-phase branches) |
| `ceiling_adequate_replay_not_separated` | ceiling ≥ 6, replay > ceiling − 3 | The cost is adequate, but this jitter does not separate a model from replay | Preregister a wider distribution (larger apple and/or plate jitter) and repeat this two-arm test before T2 |
| `ceiling_inadequate_task_feasible` | ceiling < 6, scripted ≥ 6 | The scripted collector grasps where the exact-dynamics ceiling does not, so the cost/planner design is inadequate | Diagnose the failure phases and redesign under a new preregistration; do not pair the ceiling with a learned model. T2 data collection may proceed only as justified by the scripted reference |
| `ceiling_inadequate_scripted_also_fails` | ceiling < 6, scripted < 6 | The distribution exceeds the current grasp mechanics | Preregister a narrower wide distribution (for example apple ±2 cm) and repeat |
| `ceiling_inadequate_feasibility_reference_inconclusive` | ceiling < 6, scripted arm incomplete | The ceiling is inadequate, and the cause is unattributed | Repair the reference arm and repeat into a new output directory |
| `inconclusive` | any ceiling/replay attempt uncounted, provenance invalid or rollouts inexact | — | Fix the defect and rerun into a new versioned output; keep this run as a recorded failure |

Diagnostics that are not gate evidence:

- Per reset and arm:
  - the retrieved demonstration;
  - the termination;
  - the ordered stages;
  - reach, grasp and success;
  - the ceiling's furthest phase and phase log, with reached/blocked ends;
  - parity counts;
  - wall time.
- The furthest-phase histogram of failed ceiling resets.
- The scripted arm's final phase.

The results will also report full-task stages. Transport, place and success
are secondary; the gate uses grasp only.

## Recorded TRAIN smoke (reset 42000 only)

The smoke runs `evaluate_apple.run` with `WIDE_COHORT` patched to `(42000,)`
and `wide_reset` patched to the narrow TRAIN reset, through the complete
supervisor, snapshot, preparation and worker path. It is a runtime check, not
evidence. It ran into the scratch directory `outputs/task047-scratch/smoke-42000-a`
from the uncommitted working tree of this protocol's code:

- The report was `completed` with valid provenance. All 3 attempts were
  counted, in 287 s global.
- The TRAIN artifacts were byte-identical to TASK-045/046: `state_goals.npz`
  `839190fc…f88d` and `state_calibration.json` `d8dae051…8d75`.
- `demo_replay` retrieved `apple-42000` and succeeded (501 commands, 7.9 s).
- `scripted_oracle` succeeded (501 commands, 7.7 s, in phase `release_high`).
- `privileged_object` succeeded (388 commands, 263.8 s).
  - Its phase log was approach 0, descend 110, close 126 (the descent was
    blocked), lift 171, transport 209, lower 260 (transport reached), release
    339 (lower blocked) and retreat 379.
  - Parity: 387/387 robot-state checks and 387/387 full-state checks were
    exact.
  - Its commands were identical to the last design probe on 42000 (same seed).
- The one-reset gate object computed `ceiling_adequate_replay_not_separated`.
  That is the expected reading on a TRAIN reset, where replay succeeds, and it
  exercises the gate code only.

## Frozen run command

Execute **once**, from a clean checkout of the reviewed commit, into a directory
that does not exist yet:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-wide-object-ceiling-v1 \
  --stage development --goal-kind object --no-proposals \
  --horizon 6 --stride 16 --dwell 1 --candidates 24 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 960 \
  --max-seconds 8400 --control-timeout 5
```

- The seeds default to 45000–45007 and the modes to
  `demo_replay scripted_oracle privileged_object`.
- `--stride 16 --dwell 1` only rebuild the identical retrieval library.
- Every outcome will be recorded in `apple_wide_object_ceiling_results_v1.md`
  and `benchmarks/manifests/apple-wide-object-ceiling-v1.json`. That includes
  failures, stalls and timeouts.

## Pre-run review revision R1

The fresh pre-run review is in `docs/reviews/apple_wide_object_ceiling_review.md`.
It found two blocking issues, both fixed before any 45000-range attempt:

1. **Unequal guard handling.** A velocity-guard refusal in `demo_replay` or
   `scripted_oracle` was an uncounted `runtime_error`, so it made the whole run
   inconclusive, while the same event in the ceiling was a counted stop. For
   object plans the guard refusal is now a counted `guard_refused` in those
   arms too, via `evaluate_apple.project_open_loop`. Other errors propagate,
   and earlier plans still raise.
2. **Gate and readings could disagree.** `primary_gate_passed` did not require
   every ceiling attempt to be counted, while `conclusive` did. It now
   requires it.

The non-blocking recommendations were also applied:

- Parity must be non-vacuous (see the gate).
- Infeasible candidates are no longer rolled out.
- Tests were added for guard handling, grasp bounds, the warm-start reset and
  acknowledgement refusals.
- The CEM wording, the time estimates and the design-disclosure corner list
  were corrected.

No reset, seed, threshold, budget, phase parameter or command changed.

**Post-R1 TRAIN smoke.** The smoke was repeated into
`outputs/task047-scratch/smoke-42000-b`:

- The report was `completed`, in 285.3 s, with identical TRAIN artifact hashes.
- All three arms succeeded. `privileged_object` took 388 commands, with
  387/387 exact robot and full-state checks and `rollouts_exact` true.
- Every executed ceiling command was identical to the pre-R1 smoke.

**Statistical note (fixed now).** With n = 8 per arm, the 3-reset margin is a
preregistered *decision rule*, not a significance test. For example, 8 vs 5
grasps gives Fisher p ≈ 0.2. A `separated` outcome will be reported as "the
decision rule was met", with the counts. It will not be reported as a measured
significant separation.

## Known risks declared before outcomes

- **Privileged and non-learned.** A ceiling pass says the object-aware cost
  and phase machine suffice *given exact dynamics and perfect object state*. It
  says nothing about whether a learned model can supply either.
- **Design probes.** The design probes spanned the same ±3/±2 cm box (see the
  design disclosure above).
- **Scheduled gripper.** The right-grasp command is phase-scheduled, not
  planned. The planner chooses only the arm motion in each phase.
- **Replay may generalise.** Open-loop replay of the nearest TRAIN
  demonstration may still grasp when the apple lies near the TRAIN centre.
  45000 is such a reset, and the ±3 cm draws are uniform, so several resets can
  be within about 1–1.5 cm of the centre.
- **Short run.** There are 8 development resets per arm, so every count has
  wide uncertainty.
