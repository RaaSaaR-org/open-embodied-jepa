# Candidate-feasibility diagnostic results

**Projection fixed the observed scoring/execution mismatch and removed joint-rate
stops in all seven completed matched pairs. Learned pick-and-place still failed.**
The 600-second budget ended with **15/20 episodes completed, one interrupted and
four unstarted**. This is an incomplete development comparison, not a replacement
for the original frozen MVP benchmark or an independent generalization test.

## Matched outcomes

Only completed off/on pairs enter this table; all other planned outcomes remain
in the [versioned result manifest](../../benchmarks/manifests/feasibility-results-v1.json).

| Backend | Completed pairs / planned | Mean executed commands, off → on | Joint-rate stops, off → on | Deadline stops, off → on | Reach, off → on | Full success, off → on |
|---|---:|---:|---:|---:|---:|---:|
| Native JEPA | 4/5 | 11.75 → 100 | 4/4 → 0/4 | 0/4 → 0/4 | 0/4 → 1/4 | 0/4 → 0/4 |
| LeWM | 3/5 | 7.33 → 88.67 | 3/3 → 0/3 | 0/3 → 2/3 | 0/3 → 0/3 | 0/3 → 0/3 |

Native projected runs at seeds 20000–20003 all reached the 100-command cap; seed
20003 registered the task's reach stage. LeWM projected runs at 20000–20002
executed 94, 100 and 72 commands, respectively. The first and third stopped at the
five-second control deadline. None of the **15 completed episodes** achieved
grasp, transport, place, release or full success. No success/failure or command
count is imputed for the five episodes without completed records.

LeWM projection-off at seed 20003 completed with 31 commands and a joint-rate
stop; its projection-on counterpart was killed at the total budget, so that pair
is excluded above. All four conditions at seed 20004 were unstarted. Deterministic
run ordering means missing outcomes are not a random sample.

## Command agreement and runtime

All **666 executed projected commands** (400 native, 266 LeWM) matched both the
model-scored projected action and the accepted target **exactly**. This checks
command semantics, not realized end-effector displacement or collision safety.
All original checkpoint files and 400 historical result records remain unchanged.

| Projection-on runtime, completed episodes | Native JEPA | LeWM |
|---|---:|---:|
| Planning p50 / p95 | 0.239 / 0.330 s | 0.285 / 4.222 s |
| Whole control p50 / p95 | 0.250 / 0.342 s | 0.296 / 4.244 s |
| Mean projection time | 0.260 s | 1.454 s |
| Mean total planning time | 0.262 s | 1.464 s |

Latency pools include completed episodes' final deadline-missed plans, where
present. Even the typical projected plan exceeds the simulation's 50 ms control
interval: these are offline simulation rollouts, not demonstrated real-time
20 Hz control. Repeated IK/backtracking dominates cost; future work must reduce
it without weakening guards or changing candidate scoring semantics. More model
training would not remove this measured solver bottleneck.

## Reproduction and provenance

The [preregistered protocol](feasibility_v1.md) and executable input hashes were
committed and pushed **before execution** at
`680be1b8d3a422d3d90a566e2f40ebf93f405859`. The clean Python implementation hash was
`d8d20a50279c7ee112c206a67f69ab42f9c28df88f376e7a9daf233afed32941`.
Both unchanged training-seed-1 checkpoints used the same data, reused goal images,
resets, action limits, H4 / 64 candidates / 3 iterations / 8 elites, CPU four
threads, 100-command cap and five-second deadline. Projection off/on share the
new synchronized-kinematics execution fix; historical trajectories are not
claimed to reproduce under this new runtime.

Executed command:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_feasibility.py \
  --source-sha256 d8d20a50279c7ee112c206a67f69ab42f9c28df88f376e7a9daf233afed32941 \
  --output outputs/feasibility-v1
```

The runner returned exit code **2**, honestly indicating an incomplete comparison.
Wall-budget accounting ended at **600.065 seconds**, including 0.065 seconds of
supervisor/termination overhead. There was no retry, budget extension, model
update, threshold change or replacement of negative outcomes. Local raw evidence
is in `outputs/feasibility-v1/`; the original `comparison.json` SHA-256 is
`9d46b62c88d6b00e38d2696ac879fc25a766badb2ca95bc549e8fcb0f0463508`.
Raw records retain the original execution worktree paths when archived; locate
individual runs beneath the canonical artifact directory by their `run_id`.
The compact versioned manifest preserves every planned status, paired differences,
metrics, input/source hashes and completed episode/run hashes.

An independent agent revalidated all 15 completed episodes against the result
schema, recomputed every metric and paired summary, verified input and goal
lineage, and checked all 666 accepted actions. Pre-run validation passed all
**351 tests**, Ruff and MC checks; details are in the
[review record](../reviews/feasibility-review.md).

A local 960×720 replay video,
`outputs/feasibility-v1/lewm-projected-seed20000-quarter-speed.mp4`, shows the first
preregistered LeWM reset, not a selected best run. It replays the 94 recorded
commands at quarter simulation speed without new model inference. Its provenance
JSON records command/status/timestamp verification. Playback omits wall-clock
planning pauses and ends at the original planning-deadline termination.

## Interpretation and follow-up

The evidence supports better command feasibility for the completed development
pairs. It does not establish reliable reaching, grasping, generalization, future
contact feasibility or hardware readiness. One native reach event is a small
positive observation, not a validated learned reaching capability.

TASK-030 records the next runtime step: reduce projection latency while preserving
acceptance agreement, then preregister and complete a new bounded comparison.
After runtime control is stable, isolate image-goal reaching with a scripted
positive control before adding more manipulation training. Preserve this
incomplete result and the original negative MVP comparison.
