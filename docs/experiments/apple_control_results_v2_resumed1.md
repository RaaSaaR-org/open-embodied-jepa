# H16 apple control v2: incomplete negative development result

The resumed comparison recorded **zero successful placements across six planned
attempts**, but only **one attempt completed**. One timed out and four never
started. This is not a completed six-episode performance estimate, and it does
not establish a working manipulation MVP or a causal control advantage.

The fixed supervisor stopped after **595.237 seconds**, reserving finalization
time within the 600-second cap; the command returned exit code 2. There was no
retry, budget extension, threshold change, or final-cohort execution. The
[original protocol](apple_control_development_v2.md) and
[prelaunch resume addendum](apple_control_development_v2_resume.md) govern this
run. The earlier user-interrupted run remains separately preserved in the
[interruption audit](apple_control_interruption_v2.md).

| Reset | Mode | Acknowledged commands | Last image goal | Ordered stages reached | Outcome |
|---|---|---:|---|---|---|
| 43000 | Learned | 1000 | 5 / frame 140 | Reach only | Completed: command limit |
| 43000 | Persistence | 846 | 1 / frame 28 | None | Hard wall timeout |
| 43000 | Shuffled dynamics | 0 | — | Unobserved | Not started: budget |
| 43001 | Learned | 0 | — | Unobserved | Not started: budget |
| 43001 | Persistence | 0 | — | Unobserved | Not started: budget |
| 43001 | Shuffled dynamics | 0 | — | Unobserved | Not started: budget |

The learned attempt spent 881 commands at frame 140. Its final observed image
distance was **1.38281**, against threshold **0.063722**. It moved the apple away
from the plate without establishing a grasp: the final object/plate distance was
0.46615 m. No grasp, transport, placement, or release occurred in either started
attempt. The learned attempt took 166.27 seconds; median observe/plan time was
90.13 ms. Persistence spent 844 commands at frame 28, ending at image distance
1.00096 against threshold 0.091176. Its median observe/plan time was 503.84 ms.
Maximum recorded per-command time was 966.13 ms, below the 5-second deadline.
The timeout occurred between acknowledged commands, with no uncertain pending
execution. Realized projection cost consumed the budget; persistence does not
invoke a learned predictor, so its higher cost is not model inference latency.

## Calibration, ranking, and execution evidence

All 19 image goals came from the preregistered nominal TRAIN demonstration
`apple-42000`; every tolerance used 16 successful nominal TRAIN references.
The audit verified reference membership from metadata without decoding TEST
images. **10/19 goals** had an adjacent image inside their threshold, with
12/36 directed adjacent overlaps. Three-observation dwell provides a temporal
scaffold in those regions. The first goal did not overlap its neighbor, and both
started attempts advanced beyond it; reset-image acceptance was not the stall.

Learned costs varied in all 2,000 search rounds (median cost spread 0.074242).
Selected actions came from 553 perturbations, 278 warm/search-best proposals,
106 demonstration proposals, and 63 holds. All 1,692 persistence rounds had
identical candidate costs; its seeded tie selection chose 704 demonstration
proposals, 54 perturbations, 50 warm/search-best proposals, and 38 holds.
These traces show that learned ranking affected action selection. They do not
show that ranking produced successful manipulation. Neither shuffled-dynamics
attempt started, so this run provides **no matched shuffled-control comparison**.

All 1,846 flushed pending commands have acknowledgments; there are no recorded
execution errors. All 1,000 learned actions and 843/846 persistence actions match
their projected/scored action exactly. Three persistence actions were marked
`clipped: right workspace`: maximum normalized-coordinate differences were
5.82e-11, 4.66e-10, and 7.45e-9, at steps 180, 204, and 252. These small numerical
changes are retained in the audit; exact equality is not claimed for them.
No joint-limit execution rejection recurred in the observed prefix.

The resumed learned run's first 574 commands exactly match the interrupted run's
projected/applied actions, image-goal indices, observed distances, and task scores.
They are a repeated prefix, not additional independent success evidence. The
interrupted artifacts retain their original hashes.

## Provenance and interpretation

Source revision: `b14eb5a64d00f732a929aa6acedaaeecf05f7811`.
The H16 checkpoint was
`0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`;
the branch dataset manifest was
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
All 566 dataset payloads, checkpoint, live and captured source, action/asset
manifests, upstream asset payloads, and generated goal/calibration hashes passed
post-run checks. No frozen input was changed during execution.

The [compact manifest](../../benchmarks/manifests/apple-control-development-v2-resumed1.json)
records the exact argv, hashes, all six statuses, and audit references. Raw local
artifacts are under `outputs/apple-control-development-v2-resumed1/`; its
`audit_results.py` regenerates the separate `audit.json` without inference or
physics. Run it from this worktree with `.venv/bin/python`.

The preceding H16 causal prediction gate passed, but that evidence has not
translated into closed-loop grasping. Horizon, waypoint spacing, and proposal
coverage changed together from v1, and this run is heavily budget-censored.
Neither cross-version improvement nor reliability can be inferred from these
results. The current configuration failed its bounded development comparison;
the final 20-reset cohort remains untouched.
