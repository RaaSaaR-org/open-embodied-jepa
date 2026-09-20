# Apple image-waypoint control v1: negative development result

The frozen six-attempt diagnostic achieved **0/6 successful apple placements**.
Both learned runs exhausted 1,000 commands without reaching or grasping the apple.
Three control runs encountered execution errors, so the runner correctly returned
exit code 2 and marked the comparison incomplete. All six planned attempts ran;
none was retried, replaced, or dropped. This does not meet the working-MVP gate.
The fresh final cohort remains unexecuted.

Protocol: [apple_control_development_v1.md](apple_control_development_v1.md).
Source revision: `b94090ee4662cb3021e85d83427a2371d0452b86`.
The run took **381.6696 true-wall seconds**, within the declared 600-second cap,
on CPU with four Torch threads. All frozen-input checks passed. The reproducible
command, exact source/input hashes, environment, per-attempt records and artifact
hashes are in
[the result manifest](../../benchmarks/manifests/apple-control-development-v1.json).
Raw local artifacts are under `outputs/apple-control-development-v1/`; its
`audit_results.py` derives `audit.json` without model inference or physics.

| Reset | Mode | Acknowledged commands | Last goal index / 51 | Ordered stages reached | Stop |
|---|---|---:|---:|---|---|
| 43000 | Learned | 1000 | 2 | None | Command limit |
| 43000 | Persistence | 1000 | 1 | None | Command limit |
| 43000 | Shuffled dynamics | 266 | 3 | None | Joint-target execution error |
| 43001 | Learned | 1000 | 5 | None | Command limit |
| 43001 | Persistence | 302 | 3 | Reach only | Joint-target execution error |
| 43001 | Shuffled dynamics | 635 | 3 | None | Joint-target execution error |

No attempt reached grasp, transport, placement or release. The first learned run
spent 972 commands at TRAIN frame 20; the second spent 956 at frame 50. Their last
image distances were 2.2104 and 1.0129, against thresholds 0.1008 and 0.0650.
The observed early sequence stalls are direct evidence against useful control
from this frozen configuration, despite favorable aggregate prediction loss.

## Calibration and split audit

The selected demonstration was `apple-42000`, the lexicographically first
successful nominal TRAIN episode. All 16 reference episodes were successful,
nominal and in TRAIN; all 52 goal frames had 16 valid corresponding-frame
references. No TEST images were used. Goals and tolerances were frozen before
physics; there was no outcome-driven recalibration.

**39/52 goals (75%)** had at least one adjacent goal image inside their tolerance;
65/102 directed adjacent comparisons overlapped. Ordered three-observation dwell
therefore supplies a substantial temporal scaffold in those regions, and must
not be mistaken for learned long-horizon manipulation. The first waypoint did
not overlap its neighbor. Initial reset-image distances 0.01417 and 0.00979 were
below its 0.35916 threshold, and every run advanced beyond it. Initial reset
appearance was therefore not the blocking gate in this experiment. Full-frame
RGB distances alone do not establish apple/contact visibility or localization.

## Action-ranking and execution audit

All **4,203 acknowledged applied actions exactly equaled their projected/scored
first actions**. There were also three flushed, unacknowledged pending commands
with execution uncertainty; these are preserved and are not counted as confirmed
successful actions. Each errored in `execute` with
`ContractError: joint targets exceed MJCF limits`, despite being projected feasible.
The pending steps were 266 (43000 shuffled), 302 (43001 persistence), and 635
(43001 shuffled). This is a separate projection/runtime consistency defect; the
unchanged v1 records remain negative while a new bounded investigation fixes it.

Learned candidate costs were nonconstant: median round spreads were 0.003730
and 0.001616 for the two resets. Learned mode selected direct demonstration
proposals only 7/1000 and 6/1000 times, versus perturbations 874/1000 and 968/1000.
These traces confirm that learned ranking influenced selection; they do not show
that its ranking was correct. Persistence costs were tied in every round, and
all 1,802 acknowledged shuffled-mode rounds used nonidentity score permutations.
The same frozen scaffold, proposals, projection and candidate budget applied to
all modes, though realized candidate trajectories differ after actions diverge.

The learned runs did not outperform controls on physical task success. Small
samples and three unequal error-truncated controls also prevent a clean
reliability or causal improvement estimate. The earlier weak phase-wise
shuffled-action diagnostic remains consistent with this negative control result.

The 43000 persistence run took 253.68 seconds versus 38.56/33.45 seconds for the
learned runs. Its median control computation was 252.46 ms, compared with
29.90/29.24 ms for learned, reflecting substantial realized projection cost;
this is not evidence that persistence inference itself is expensive. The maximum
recorded per-command observe/plan time was 376.03 ms, below the 5-second deadline.
No deadline miss, supervisor timeout, budget extension or final-test run occurred.
