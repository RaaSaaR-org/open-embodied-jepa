# Independent review: image-waypoint MPC

Reviewed the new `src/embodied_jepa/waypoint_planning.py`, its focused tests and
`docs/experiments/waypoint_design.md` in the `jepa-apple-mvp` worktree after
`8ee7e45`. The module was uncommitted during review. This reviewer did not author
the waypoint controller; the reviewer authored its sensor-model counterpart and
earlier contracts. This is an independent agent review of the controller, not
external maintainer approval or independent authorship of every integration.

## Finding resolved

The original controller required a new observation to be later than the previous
observation but did not compare it with the intervening execution acknowledgement.
An observation at 0.025 s was accepted after an observation at 0 s and completed
execution at 0.05 s. This was reproduced using the opaque toy model.

The author added `last_execution_timestamp`: the next observation must be at
least as recent as the acknowledgement and strictly later than the prior
observation. The regression rejects 0.025 s and accepts the valid 0.05 s snapshot.
Review confirmed the fix and test. MuJoCo projection also checks current simulator
time, but that did not excuse the generic controller's incomplete clock check.

## Verified boundaries

- Progress receives only observed/goal RGB; ordering advances at most one goal
  per observation after consecutive image-distance dwell. Time or scorer stages
  cannot directly advance goals.
- Every candidate, including demonstration proposals, is projected before model
  prediction. Infeasible candidates cannot win. Opposite toy dynamics select
  opposite actions from identical proposals; demonstration selection is not
  hardcoded.
- Predictions and goal latents remain opaque to the controller. Goals contain
  no target joint state. Mandatory acknowledgements match the actual projected
  request, record transport changes and terminate after rejection.
- Persistence assigns equal observed-image cost and never calls dynamics;
  shuffled dynamics permute feasible action/cost associations explicitly.
  Seeded random ties avoid a deterministic preference for demonstrations.
- The implementation is bounded shooting/search with a best-candidate mean and
  scheduled variance. It is not elite-fitted CEM; reports must preserve that
  distinction. Changed controls legitimately produce different future states
  and warm starts.

Validation: `PYTHONPATH=src .venv/bin/python -m pytest -q
tests/test_waypoint_planning.py tests/test_training.py` passed 25 tests in 2.78 s,
including all 11 waypoint tests. The earlier 10-test version passed in 0.04 s
before the missing acknowledgement-clock case was added.

No remaining blocking finding in this scope. Runner-level train-membership
verification, calibrated thresholds, deadline handling, safe stopping and final
task scoring remain integration responsibilities. Completing image waypoints
must not be reported as physical task success. Real contact dynamics, meaningful
model dependence and the fresh task cohort remain unverified by these unit tests.

## Evaluation-runner follow-up review

Independently reviewed `scripts/evaluate_apple.py` and its tests. The initial
review requested three corrections before execution: an explicit per-control
deadline, retention of failed/terminated worker outcomes even when a report had
already been written, and post-attempt frozen-input verification. The revised
runner checks its registered five-second deadline before execution, records
supervisor return codes and overrides failed child completion, and checks the
snapshot sources/checkpoint/dataset/action/asset/generated-waypoint hashes both
before and after attempts. Integrity failure invalidates the comparison rather
than letting the last completed report silently certify changed inputs.

The initial single-demo calibration could make an immutable plate/reset mismatch
look like a controllable goal error. Before evaluation, the coordinator declared
a training-only tolerance extension: include 1.1 times the 90th percentile of
same-frame distances from successful nominal training episodes. Review verified
that reference IDs/frame indices/distances/counts are stored, missing frames are
skipped rather than clamped to a future endpoint, and non-training IDs are
rejected. Adjacent-waypoint distances and threshold overlaps are also reported.
Broad thresholds can make multiple successive goals visually indistinguishable;
that remains an empirical limitation, and the dynamics ablations are essential.
This task-specific temporal scaffold is not an independent image-goal or
zero-shot-generalization result.

After these changes, `PYTHONPATH=src .venv/bin/python -m pytest -q
tests/test_apple_evaluation.py tests/test_waypoint_planning.py` passed **27 tests
in 0.08 seconds** (16 runner, 11 controller). No physics evaluation was run by
this reviewer. The additional minor trace correction is verified: acknowledged
result rows now retain `control_seconds`, which replacing the temporary trace
with the acknowledgement had discarded. A focused regression also confirms
that a six-second plan cannot actuate under the five-second deadline.

No remaining blocking review finding in the revised controller/runner scope.
The validation and success limitations above remain; software review is not an
apple-manipulation result.
