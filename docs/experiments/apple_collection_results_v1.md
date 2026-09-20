# Apple task collection v1 results and artifact audit

The preregistered collection completed **32/32 attempts**, retaining all 32 episodes
and **16,132 applied transitions**. The privileged scripted controller succeeded
on **31/32** resets. This establishes useful task-specific demonstration data;
it is **not learned world-model control** or a broad manipulation reliability claim.

Supervisor time was **213.757 seconds**, within the 600-second cap. The worker
recorded 213.509 true-wall seconds and 196.781 aggregate process CPU seconds.
No retry, budget extension, split reshuffle, physics change or scorer relaxation
occurred. The source snapshot was created at commit `813d4bc`.

| Controller variant | Attempts | Successes |
|---|---:|---:|
| Nominal forward grasp / centered release | 16 | 16 |
| Position/rotation perturbation bursts | 8 | 8 |
| Grasp-closure perturbation bursts | 8 | 7 |

The failed closure-burst attempt, seed **42003**, reached contact lift and
transport but landed 0.04561m from the plate center, outside the unchanged 0.04m
threshold. It subsequently stopped with `right IK failed` during `lower_open`
after **595 applied commands**. The complete valid prefix remains a training
episode; the rejected final request is metadata, not an applied transition.

## Frozen partitions and coverage

This new task-specific corpus explicitly permits Apple→Plate in training. It does
not alter the original MVP held-out corpus or its conclusions. Whole-session
splits use seed42 and the declared 0.8/0.1/0.1 ratios, with no held-out pairing in
this corpus. All sessions and episode IDs are disjoint across partitions.

| Partition | Episodes | Successful episodes |
|---|---:|---:|
| Train | 26 | 25 |
| Validation | 3 | 3 |
| Test | 3 | 3 |
| Pair holdout | 0 | 0 |

All 32 stored episodes reached grasp and transport. Place/release succeeded for
the 31 complete successes. The phase distribution is unequal and must not be
described as balanced training windows:

| Applied-action phase | Train | Validation | Test metadata |
|---|---:|---:|---:|
| Orient | 3,380 | 390 | 390 |
| Descend | 2,080 | 240 | 240 |
| Close | 1,170 | 135 | 135 |
| Lift | 3,900 | 450 | 450 |
| Transfer | 1,560 | 180 | 180 |
| Release high | 1,005 | 107 | 110 |
| Lower open | 30 | 0 | 0 |

Later learned-model diagnostics should separate moving/contact/release windows
from long settling phases; uniform window sampling is dominated by orient/lift.
Only one explicit unsuccessful trajectory is available, so large-failure recovery
coverage remains limited despite the alternating perturbation/recovery commands.

## Read-only audit

The collector author performed a separate read-only artifact audit, including
independent recomputation of normalization statistics. This is not an independent
maintainer review of the collector implementation.

- Verified every DatasetStore payload hash and every frozen runtime snapshot file.
- Verified T+1 measured states/masks/timestamps for T applied actions, 86 measured
  proprioceptive entries, finite normalized/raw actions and 50ms timestamp spacing.
- Decoded all train/validation RGB and verified 96×96×3 T+1 frames. **No test images
  were decoded.** Test checks used encoded image presence/counts, Parquet sensor
  and action columns, and full-file hashes; stage/success counts came from existing
  collector metadata. No test data entered fitting or checkpoint selection.
- Recomputed state means/standard deviations and action means using exactly the
  26 training episodes, matching recorded normalization and valid-state counts.
- Verified raw actions against manifest conversion of applied normalized actions.
  Requested/applied actions differ on **320/16,132 steps**, principally while the
  hand target is rate-limited. Mean absolute 14D difference is **0.0014141**, maximum
  **1.8181818**; requested commands must not substitute for executed training labels.
- Right translation coordinates range approximately **[−0.440, +0.440]**, right
  rotation coordinates **[−0.520, +0.500]**, right grasp **[−1, +1]**. Left pose
  commands remain zero and left grasp remains −1. Exact per-channel extrema are
  recorded in the machine-readable audit.
- Verified **1,300 pixel-exact dense goal images** from all **25 successful training
  episodes**, with episode/frame/timestamp provenance and matching image hashes.
  No validation/test waypoint is exported, and goal entries contain no robot state
  or simulator truth. Oracle phase/score metadata is excluded from canonical
  SequenceBatch model inputs.

The collector's `training_ready=true` is supported: the corpus is sealed, goal
export is complete, and both train and validation contain successful trajectories.
Small validation/test cohorts and a narrow reset/appearance distribution remain
limitations; future physical evaluation must use its own predeclared resets.

## Reproducibility

Protocol: [apple_collection_v1.md](apple_collection_v1.md). Mechanics selection:
[apple_mechanics_probe.md](apple_mechanics_probe.md). Machine-readable audit:
[apple-collection-v1.json](../../benchmarks/manifests/apple-collection-v1.json).

- Dataset manifest SHA-256:
  `d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df`.
- Collection report SHA-256:
  `9f7d2e88e550a0db0e55ada74007102ca9b98e47a57abfce2b38b471c9812171`.
- Frozen runtime manifest SHA-256:
  `d8a8445763420db8b47796fdedc5125a753e38cc60225f461ac5d00a595a9787`.

Local full artifacts remain in `data/apple-task-v1` and
`data/apple-task-v1-runtime`; the audit script/log are in
`outputs/apple_collection_audit.py` and `outputs/apple_collection_audit.log`.
Large generated payloads are ignored by Git. The compact audit records the actual
counts, action ranges, split coverage, resource clocks and verification scope.
