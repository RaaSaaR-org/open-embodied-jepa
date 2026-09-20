# Apple task-specific delivery review

Reviewer: simulator/projection agent, 2026-09-21. Scope: assembled apple worktree
against `origin/main` at `e96c38908e7a20820b6d390b13f872e2d128ae06`, through branch HEAD
`11c8f3421959f97106f91234581331086847acb2`, including the newly committed README,
branch collection/training/diagnostic results and manifests, plus the pending
matched-diagnostic civil-wall guard fix and its regression test.
This is a local review, not GitHub maintainer approval or a claim of merge.

## Independence and coverage

I authored the projection optimization/joint-limit repair, waypoint controller,
apple evaluation runner, corpus audit and several result reports in this change.
My inspection of those portions is an author review. Their separate cross-agent
reviews are recorded in `apple_waypoint_review.md`, `apple_training_review.md` and
`../experiments/apple_branch_review.md`. This review independently concentrates
on `models/sensor.py`, generic normalization/training integration, the bounded
training supervisor, matched-action diagnostic, new training/diagnostic evidence,
and assembled README claims, which I did not author.

The sensor model uses measured current RGB/proprioception and recursively learned
future state. Image goals contain no goal joints, task scorer or simulator truth.
Normalization is fitted once on declared TRAIN episodes; validation selection
and the later causal diagnostic are separate. The diagnostic compares actual
sibling actions from identical root observations, uses denormalized visual errors
for cross-checkpoint interpretation, retains ties/noninformative pair counts,
and reports equal-root and parent-session uncertainty with its small-session
limitation. H8 results are correctly secondary and do not replace the failed H16
gate. No completed Apple→Plate learned-policy or final-cohort success is claimed.

The branch-corpus byte/split/action audit was independently executed earlier:
all 32 inherited payloads, including three TEST episodes, remain byte-identical;
all branches retain their parent's split/session, and no TEST image was decoded.
The updated diagnostic also restricts decoded episode IDs to TRAIN/VAL. The
historical unseen-pair benchmark and newer task-specific Apple training are
explicitly distinguished; the old 0/150-per-model result remains unchanged.

## Findings

One resource-supervision finding was corrected during this review.
`scripts/diagnose_apple_branches.py` initially used a single
`process.wait(timeout=115-elapsed())`. The local POSIX Python implementation uses
a monotonic deadline inside that call; it does not repeatedly recheck the runner's
civil-wall clock after a Mac host suspension. This weakens the declared hard
true-wall bound. The coordinator replaced the single wait with `wait_for_worker`, polling the
existing `elapsed()` maximum of wall/monotonic clocks every 0.1 seconds while
retaining the five-second finalization reserve. Independent re-review confirmed
the change and a clock-jump fixture (114 to 180 seconds) kills and waits for the
worker immediately after the next wakeup. The finding is **resolved**. The actual diagnostic completed in 8.323 seconds and therefore
is not invalidated by this forward-execution defect. No evidence of a suspension
or budget overrun exists in that run.

No remaining blocking correctness, lineage, leakage or inflated-acceptance finding was
identified in the reviewed scope. Green implementation tests and stronger action
assignment on the new data remain intermediate results: the new model's primary
VAL H16 reduction is only 6.2593%, below the frozen 10% threshold, despite 73.8669%
ranking accuracy. The next H16 experiment belongs to a separately preregistered
change; the fresh final control cohort remains unexecuted.

## Checks performed

- Rehashed all nine branch-training artifacts and four matched-diagnostic artifacts
  referenced by the new compact manifests, including both saved new checkpoints.
- Verified compact diagnostic summaries exactly equal the full saved report;
  recomputed the reported gate interpretation from its saved primary values.
- Verified all 528 diagnostic decoded episode IDs belong to TRAIN/VAL and none to
  TEST. No model inference, physics rollout or scientific rerun was performed.
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_sensor_model.py
  tests/test_apple_branch_diagnostics.py tests/test_apple_training.py
  tests/test_training.py -q`: **51 passed in 2.96 seconds**.
- After the supervision fix, independently reran
  `tests/test_apple_branch_diagnostics.py`: **15 passed in 0.04 seconds**.
- Coordinator separately reports **449 full optional/graphics tests passed**;
  I did not rerun that suite as part of this review.

Large datasets and checkpoints are local artifacts, not bundled into the public
repository. Public manifests pin their hashes, protocols and commands; they do
not imply that a clean remote checkout already contains those artifacts.
