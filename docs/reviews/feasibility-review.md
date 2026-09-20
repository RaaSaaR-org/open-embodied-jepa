# TASK-029 feasibility review

Reviewed the topic diff from `1fd138e` (PR #3 main) through the TASK-029
implementation, configuration, result validation, diagnostic runner and tests.
The root coordinator performed author review; a separate contract-review agent
reviewed without authoring this patch, and another agent independently wrote
behavior tests. This is independent agent review, not maintainer approval.

## Findings resolved before the registered experiment

1. Adding a planner option originally made schema-v1 validation reject historical
   results. The optional missing `project_candidates` field now means false;
   other required fields and unknown fields still fail. All **400 unchanged v0
   episodes** validated, with an explicit backward-compatibility regression test.
2. Live MuJoCo site transforms can lag the integrated `qpos` after `mj_step`.
   Seed 4 with RNG 341 produced a feasible projected action rejected by execution
   on its second step. Preview and execution now derive kinematics from the same
   full-precision robot-only observation snapshot. The exact case is a regression
   test. An independent bounded check over **8 resets × 5 steps** applied **40/40**
   projected commands with exact accepted-action equality and measured tracking
   lag up to **0.151659 rad**. No physics stepping or object truth enters preview.

## Acceptance checks and limits

The projection API, shared target preparation, raw/projected elite association,
invalid masks, no-feasible failure, requested/projected/applied traces, deadline
accounting, immutable inputs and paired development design were reviewed.
**351 tests passed**, including local rendering, models and pinned dataset reader;
Ruff lint/format and diff checks passed before freezing the implementation.

First-step agreement is tested, not a guarantee of future contact dynamics.
Horizon continuation assumes perfect joint-target tracking and is reprojected
following each observation. Arm backtracking can exclude a sequence even when
another unsampled feasible action exists. Execution guards remain authoritative.
The shared synchronized-kinematics correction also applies with projection off,
so the causal comparison is off/on under the same revised runtime, not a promise
of bit-identical historical trajectories. Physical robot and Isaac previews are
not implemented. Experiment outcomes are reported separately after execution.


## Post-experiment independent audit

The independent reviewer validated all 15 completed records, recomputed episode
metrics and paired summaries, checked pinned inputs and clean source identity,
and verified every derived goal's original pixel/reset lineage. All 666 executed
projected actions matched projected/requested/accepted values exactly. The audit
confirmed four native and three LeWM completed pairs, the native reach at 20003,
two LeWM deadline stops, and zero grasp/place successes. One interrupted and four
unstarted episodes retain unknown outcomes and remain visible in the manifest.
