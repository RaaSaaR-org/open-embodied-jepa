# TASK042 calibrated hybrid-cost results

The fixed hybrid cost **failed the registered offline primary gate**. H16 arm-ranking accuracy was **77.680% hybrid versus 73.378% pixel**, an improvement of **4.302 percentage points**, below the required **5 points**. All three VAL parents improved, but the aggregate threshold still failed. **No physical comparison was attempted.** This is not a zero-success result over three robot attempts; the conditional control stage never started. Final physical acceptance and the sealed final cohort remain unrun.

## Frozen experiment

One attempt used source `9942e1e34bf76bb6b15911149bf32d6dd66cc7aa` and the [prospective protocol](apple_aligned_control_v1.md). The existing sensor dynamics and TASK041 neural image-to-arm head were kept unchanged. The new bundle combines two group-mean errors with fixed weights:

`cost = 0.5 * visual_MSE / 0.08636228494017421 + 0.5 * pose_MSE / 0.15918705911991243`.

The scales are medians of parent medians over **447 complete, nonoverlapping stride-28 pairs from all 26 original TRAIN episodes**, including the failed demonstration. Branch, VAL, and TEST episodes did not enter calibration. Zero-distance pairs were retained by the recipe; no floor, positive-pair filter, alternate statistic, or coefficient search was applied. The actual ledger contains 0 zero visual pairs and 0 zero pose pairs.

The bundle SHA is `e32b47f7e498ab07b8b0dd1e3b759ec626aa2ff8035b075a53f68caab6233246`. Its sensor checkpoint is byte-identical to the existing H16 checkpoint and its neural head is byte-identical to TASK041's completed 2,000-update head. Relative members include calibration, pair distances, dataset and head-training provenance. Packaging code/protocol/current source identities are distinct from the recorded historical head-training source. The [compact manifest](../../benchmarks/manifests/apple-aligned-cost-v1.json) records all identities, member hashes, raw outputs, per-parent results, and resource measurements.

## Complete offline outcomes

Both stages completed successfully as software executions; the scientific gate returned **failed**. The audit reused TASK041's saved forecasts and image-goal means, with no new dynamics inference or fitting. It preserved all 18 roots, eight candidate futures per root, 144 cross-parent root-goal cases per horizon, and the same eligibility and parent-balanced aggregation rules. This reused VAL cohort is development/model selection, not independent confirmation.

| Comparison | H16 predicted ranking | H16 measured-endpoint ranking |
|---|---:|---:|
| Pixel | 73.378% | 82.249% |
| Frozen neural pose only | 89.816% | 98.910% |
| Fixed calibrated hybrid | 77.680% | 90.828% |

The hybrid lost **12.137 pp** relative to the neural-pose-only cost. That comparison does not justify discarding visual information: the arm-only representation cannot identify all object or hand outcomes. Measured-endpoint hybrid cost uses measured endpoint visual features and joints against the inferred goal joints; the runtime arrival metric instead infers both current and goal joints from RGB. These are deliberately different diagnostics.

| VAL parent | Pixel predicted | Hybrid predicted | Improvement |
|---|---:|---:|---:|
| apple-42007 | 73.242% | 77.383% | +4.142 pp |
| apple-42019 | 72.554% | 76.631% | +4.078 pp |
| apple-42031 | 74.338% | 79.024% | +4.686 pp |

The parent-cluster bootstrap 95% interval is **76.631–79.024%** for hybrid ranking and **+4.078–4.686 pp** for improvement. Only three parent clusters are available. H16 includes **2,888 of 4,032** possible pairs; the other 1,144 have insufficient arm separation. H8 includes 2,606 pairs and yields **80.064% hybrid versus 76.384% pixel**, a secondary improvement of 3.680 pp. All six roots per parent are informative. H8 and measured-state results do not override the H16 primary threshold.

The saved-array formula matched the actual runtime `hybrid_cost` helper within the frozen `rtol=1e-5, atol=1e-7`. Maximum absolute difference was **1.9073486e-6**; this satisfies the combined relative-plus-absolute tolerance, not an absolute-only 1e-7 bound. Per-root component costs and direct-formula values remain saved for verification. No candidate, difficult goal, or failed gate was dropped.

## Resources and provenance audit

| Stage | Wall allocation | Supervisor wall seconds | Worker CPU seconds | Peak RSS bytes | Execution |
|---|---:|---:|---:|---:|---|
| prepare | 120 | 4.744393 | 3.772367 | 692486144 | Completed |
| audit | 60 | 2.261272 | 1.398091 | 351846400 | Completed |

Stage elapsed times sum to **7.005666 seconds**, and worker CPU sums to **5.170458 seconds**, under four-thread CPU limits. These are separately supervised sums, not a continuously timed coordinator interval; worker CPU excludes the supervisor. No timeout or retry occurred.

An initial coordinator ledger-construction command referenced an incorrect manifest path and failed **before any experiment launch or output write**. The corrected launch ledger was successfully recorded before the single run; its SHA is `c3e612980d30935001e7917d3ba98b277836b8f6e2672f07b7b63ccb21147b8b`. This differs from TASK041's previously disclosed late supplemental ledger and is not a run retry.

Saved-evidence checks verified **64 sealed stage artifacts**, all 10 bundle members, all **566 encoded corpus payload hashes**, current registered input identities, and the prepare-to-audit identity chain. The exact 26 decoded episode IDs are original TRAIN members, with no VAL/TEST decoding. The pre-decode pair-ledger hash matches its final bytes. Recomputing all **447 pair distances** from the 26 saved normalized feature/joint caches reproduced every distance, parent median, and global scale **exactly**, without corpus decoding or learned inference.

Software validation: the coordinator reported **578 optional/model/rendering tests passed in 11.06 seconds**, with Ruff checks/formatting across 89 files, MissionControl validation, and diff checks passing. The builder-focused 19 tests also passed before execution. These software checks do not turn the failed research gate into a pass.

This resource/provenance check was performed by the builder author and is not presented as independent code approval. The [completed independent scientific review](../reviews/apple_aligned_sensor_review.md#completed-task042-saved-evidence-audit) reproduced the calibration, 1,728 cost vectors, 8,064 pair records, hierarchical rankings, and bootstrap results, and confirmed the failed gate. Historical raw outputs are preserved under `outputs/apple-aligned-cost-v1/`; no weights, scales, threshold, or goal cohort were changed after seeing the result. The authorized stopping rule ends this version before physical control, and the user-requested pause follows TASK042 delivery.
