# Four-command apple control v1: negative, time-limited development result

**All six planned attempts started; none reached or grasped the apple.** The two
learned attempts completed 1,000 commands each. Four controls stopped at their
per-attempt soft cutoffs. The comparison finished in **462.529 seconds**, inside
the fixed 600-second global budget, and returned exit code 2. There were zero
placements, but this is not a completed six-episode performance estimate: four
attempts are time-limited. The working manipulation gate remains unmet.

The [frozen protocol](apple_control_commitment_v1.md) used the same H16 checkpoint,
TRAIN demonstration/calibration, goals and reset pairs as control v2, with
four-command commitments and a new 90-second per-attempt allocation. All attempts
received the full 90-second allocation; none was shortened by the global budget.
Each soft cutoff reserved finalization time. No retry, extension, model fitting,
threshold change, or final-cohort execution occurred.

| Reset | Mode | Applied commands | Last goal / TRAIN frame | Worker wall (s) | Outcome |
|---|---|---:|---|---:|---|
| 43000 | learned | 1000 | 2 / 56 | 71.686 | step_limit |
| 43000 | persistence | 595 | 1 / 28 | 85.267 | attempt_timeout |
| 43000 | dynamics_shuffle | 467 | 1 / 28 | 85.333 | attempt_timeout |
| 43001 | learned | 1000 | 1 / 28 | 40.021 | step_limit |
| 43001 | persistence | 864 | 4 / 112 | 85.458 | attempt_timeout |
| 43001 | dynamics_shuffle | 456 | 1 / 28 | 85.276 | attempt_timeout |

Every scorer stage—reach, grasp, transport, place and release—remained false.
The first learned attempt ended at image distance 1.25431 versus threshold
0.060909; the second ended at 0.96982 versus 0.091176. Advancing through a few
image goals did not translate into physical task progress. The two learned runs
are direct negative observations under this frozen configuration. Unequal
command counts and the four censored controls prevent a complete matched-horizon
performance estimate or a reliability claim.

## Commitment and action audit

There were **4,382 acknowledged commands**: 1,641 followed fresh full searches,
and 2,741 used cached plan commands. Four additional full searches completed at
the soft cutoff but their commands were never sent, giving **1,645 total full
searches / 52,640 candidate evaluations**. Each cached decision evaluated zero
model candidates. Pending/result pairs are complete; no unmatched pending
command, runtime execution error, or uncertain actuation remains.

| Reset / mode | Applied search / cached | Applied commands per originating plan: 1 / 2 / 3 / 4 | Cache aborts before acknowledged commands |
|---|---:|---|---|
| 43000-learned | 253 / 747 | 2 / 3 / 0 / 248 | changed_cached_command: 3, waypoint_advanced: 1 |
| 43000-persistence | 344 / 251 | 251 / 11 / 6 / 76 | applied_action_changed: 3, changed_cached_command: 261, infeasible_cached_command: 2, waypoint_advanced: 1 |
| 43000-dynamics_shuffle | 276 / 191 | 205 / 9 / 4 / 58 | applied_action_changed: 1, changed_cached_command: 215, waypoint_advanced: 1 |
| 43001-learned | 256 / 744 | 3 / 6 / 3 / 244 | changed_cached_command: 10, waypoint_advanced: 1 |
| 43001-persistence | 218 / 646 | 1 / 2 / 1 / 214 | waypoint_advanced: 4 |
| 43001-dynamics_shuffle | 294 / 162 | 236 / 4 / 4 / 50 | applied_action_changed: 4, changed_cached_command: 238, waypoint_advanced: 1 |

The histogram counts actual accepted commands per originating search, including
short final prefixes; it does not label every short prefix an abort. The four
unsent timeout decisions add three further `changed_cached_command` aborts.
These cases correctly discarded the cache and searched again rather than
executing altered cached commands.

Every one of the **2,741 cached commands** matched its original scored sequence
offset, its freshly projected H1 command, and its actual applied action exactly.
The originating goal/image hash, observation timestamp and inherited cost also
matched, and cached traces carried no new candidate evaluations. No stale cost
was represented as a new prediction for the current observation.

Across all commands, 4,374/4,382 projected/applied vectors matched bitwise. Eight
fresh-search commands received workspace roundoff clipping, with a maximum
normalized-coordinate difference of **1.49e-8**. Their acknowledgments recorded
`applied_action_changed` and cleared the commitment; all are retained in the
trace. There was no measured-rate or joint-limit execution rejection.

## Costs, timing and frozen inputs

Learned candidate costs were nonconstant (median round spreads 0.01735 and
0.47943); persistence costs were tied. Shuffled mode retains its recorded
nonidentity cost permutations. These checks establish the intended controls,
not useful manipulation. Median observe/plan times ranged from 11.07 to 146.22 ms
across attempts; the maximum was 742.82 ms, below the five-second control deadline.
The evaluator did not record aggregate worker CPU seconds, so wall timing must
not be interpreted as CPU usage. Different realized projection work affects
runtime even when candidate budgets match.

Post-run checks passed for the checkpoint, all 566 source-corpus payloads, live
and captured runtime source, action/asset manifests and upstream asset payloads,
and frozen waypoint/calibration artifacts. All calibration references were
successful nominal TRAIN episodes. The unchanged 19 goals still contain ten
adjacent-overlap goals; image dwell remains a temporal aid, not task evidence.
No TEST image was used by this evaluation.

Source revision: `acf5c6781466ca4155dfbe4f94cfae6121f59576`.
The [compact manifest](../../benchmarks/manifests/apple-control-commitment-v1.json)
records the exact launch, hashes, six allocations/statuses, action/cache counts,
commitment lengths, aborts and timing. Local artifacts are in
`outputs/apple-control-commitment-v1/`; `audit_results.py` reconstructs the
separate trace audit without inference or physics, while `artifact_hashes.json`
binds the original results and derived audit. Source hashes remain in the plan.

TASK038 showed useful local ranking at two selected development roots. This
follow-up demonstrates that committing four guarded commands did not turn that
local evidence into a working apple pick-and-place controller. Changes in both
execution frequency and resource allocation, and the limited sample, preclude
attributing a cross-version effect to commitment length alone. The final
20-reset cohort remains untouched.
