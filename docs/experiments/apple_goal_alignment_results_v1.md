# TASK041 goal-alignment screen results

Both image-only encoders passed the registered offline arm-alignment screen. H16 predicted ranking was **89.816% neural**, **89.706% nearest-neighbor (NN)**, versus **73.378% pixel cost**. Both improved every VAL parent. Neural met the declared replacement rule, but its ranking advantage over NN was only **0.110 percentage points**; this is not evidence of meaningful control superiority. No physics, closed-loop manipulation, TEST decoding, or final-cohort evaluation was performed.

## Registered run and resources

One attempt used source `f4f99fa1bf5fdcec7f1647927e74d0b0de8228b0`, the [prospective protocol](apple_goal_alignment_v1.md), frozen `apple-branches-v1` corpus and H16 sensor checkpoint. Script SHA is `005123e9fc450c7e1485abdcf63d68a10a22f9ed5f45dcd7d21004a22e17ab7d`. The [compact manifest](../../benchmarks/manifests/apple-goal-alignment-v1.json) records complete source/protocol/checkpoint/data/head/cache/report/seal identities, per-parent metrics, and raw artifact hashes. Raw outputs remain under `outputs/apple-goal-alignment-v1/`.

| Stage | Allocation seconds | Supervisor wall seconds | Worker CPU seconds | Peak RSS bytes | Outcome |
|---|---:|---:|---:|---:|---|
| prepare | 120 | 6.514556 | 5.723556 | 632307712 | Completed |
| fit | 300 | 2.682128 | 2.363342 | 360267776 | Completed |
| evaluate | 60 | 2.011637 | 1.347775 | 518373376 | Completed |

The sum of stage supervisor times was **11.208321 seconds** and worker CPU totals **9.434673 seconds**, using four CPU threads. These are sums of separately measured stages, not a continuously timed coordinator interval; worker CPU excludes the supervisor. All three stages completed without retries, timeouts, or incomplete roots. Fitting completed exactly 2,000 updates; final TRAIN NLL was -2.925329, which is not a generalization metric.

The coordinator's supplemental launch-ledger preamble initially failed because system Python lacked `datetime.UTC`. Preparation nevertheless launched once and registered its immutable inputs before data work. `outputs/apple-goal-alignment-v1-launch.json` was written after preparation began and explicitly discloses that timing. It is not represented as a prelaunch coordinator artifact, and no stage was repeated.

## Coverage and screen outcomes

Preparation selected all **1,392 TRAIN and 360 VAL records**, covering 26 TRAIN and three VAL original parents. All 557 distinct allowed episodes were decoded, with no missing IDs and zero TEST IDs. The failed TRAIN original remains included; its extra `lower_open` phase is explicitly recorded as outside the six-phase screen. The TRAIN NN bank was frozen before VAL decoding.

All **18 roots, 144 candidate futures, and 144 independent root-goal cases** completed. Each horizon has 4,032 possible pair comparisons. H16 retains **2,888**, excluding 1,144 for insufficient arm separation. H8 retains 2,606, excluding 1,424 for arm separation and two label ties. Every parent has six informative roots. H16 eligible counts by phase are close 528, descend 432, lift 576, orient 576, release_high 176, and transfer 600. Rankings aggregate pairs within goal, goals within root, roots within parent, and then three parents equally; 2,888 pairs are not treated as independent samples.

| Method | VAL normalized MSE | Reduction versus TRAIN mean | H16 predicted ranking | Improvement versus pixel |
|---|---:|---:|---:|---:|
| Constant TRAIN mean | 0.999249619 | — | — | — |
| NN | 0.003869052 | 99.613% | 89.706% | +16.328 pp |
| Neural | 0.001119128 | 99.888% | 89.816% | +16.438 pp |
| Frozen pixel cost | — | — | 73.378% | — |

Both exceed the frozen ≥20% error-reduction and ≥5 pp ranking-improvement thresholds, with strictly positive improvement on each parent. Neural also has lower error and slightly higher H16 ranking than NN, satisfying the predeclared preference rule.

| VAL parent | Pixel | NN | Neural |
|---|---:|---:|---:|
| apple-42007 | 73.242% | 90.811% | 90.646% |
| apple-42019 | 72.554% | 88.288% | 88.444% |
| apple-42031 | 74.338% | 90.020% | 90.358% |

Parent-cluster bootstrap 95% ranking intervals are **88.288–90.811% NN** and **88.444–90.646% neural**; improvement intervals are **+15.682–17.569 pp NN** and **+15.891–17.405 pp neural**. There are only three independent parent clusters, so these intervals do not establish broad generalization or neural superiority.

## Secondary diagnostics and interpretation

Using measured future arm positions gives H16 ranking **98.830% NN / 98.910% neural**, versus their approximately 89.8% predicted-state rankings. This gap is consistent with remaining dynamics error affecting candidate ranking; it does not isolate a unique cause. H8 predicted ranking is **91.854% NN / 91.276% neural / 76.384% pixel**. H8 and matched-endpoint results remain secondary and do not determine the primary decision.

Neural per-joint RMSE ranges from 0.00343 to 0.01220 rad; NN ranges from 0.00402 to 0.02104 rad. Neural nominal 90% marginal intervals cover 90.45–99.83% across the seven fields, often conservatively. NN uncertainty is a neighbor-spread heuristic, not a calibrated probability. Risk curves retain 50%/75% of records within each parent; because source groups have different weights, these fractions do not equal retained weighted mass, which is recorded separately.

Zero sibling pairs met the registered large-arm-separation/near-identical-RGB ambiguity criterion. This does not resolve the complementary limitation: nearly identical arm configurations can accompany very different object heights or grasp states. The planning-time audit documented one VAL release pair with only 0.007933 rad maximum arm difference but 0.1486 m apple-height difference. Arm alignment alone therefore cannot certify object placement, grasp, or release.

## Evidence verification

A separate read-only audit checked all 38 sealed stage artifacts, all 566 manifest-listed encoded corpus payload hashes, the stage input chain, and current registered source/checkpoint/protocol identities. All matched. TEST files were hash-checked as encoded bytes only. Preparation's pre-VAL bank hash `56d74691e7729e215d7269d089324842afcd45753e2e478e32059788d0d5def1` equals the final bank hash. The saved access ledger contains exactly TRAIN/VAL IDs; source inspection establishes write ordering, while saved files are not a continuous independent access monitor.

Software evidence before launch: 24 focused tests passed in the independent runtime review; the coordinator reported 542 full-suite tests passed in 11.09 seconds. This report audits saved outcomes and does not rerun fitting, inference, or simulation. Independent scientific recomputation of the saved metrics is recorded separately by the contracts reviewer. The result supports further goal-representation development, not a completed Apple-to-Plate MVP.
