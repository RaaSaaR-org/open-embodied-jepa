# Matched branch diagnostic results v1

The branch-trained sensor model improves action-conditioned prediction, but **fails the preregistered primary VAL H16 gate**. Correct-action visual endpoint error is 6.26% lower than wrong-sibling-action error, below the required 10%. Ranking accuracy 73.87% exceeds the 70% threshold. The positive 95% bootstrap interval does not remove the unmet effect-size requirement. No controller or manipulation success is established.

The unchanged diagnostic completed 66/66 planned roots and decoded 528 TRAIN/VAL branch episodes, with no TEST images decoded, no timeout and no identity drift. Parent elapsed time was 8.323 seconds; worker CPU time 7.804 seconds. Both models retained their own original TRAIN-only normalization; no fitting or checkpoint selection occurred. The cohort comprises 48 TRAIN roots from 8 parent sessions and 18 VAL roots from 3 parent sessions, with 8 siblings per root.

| Model | Split | Horizon | Root-mean ranking | Correct-action visual error reduction | Numeric thresholds met |
|---|---|---:|---:|---:|---|
| Original sensor-v1 | TRAIN | 8 |70.71%|1.48%|No|
| Original sensor-v1 | TRAIN |16|72.79%|0.99%|No|
| Original sensor-v1 | VAL |8|73.77%|1.50%|No|
| Original sensor-v1 | **VAL** |**16**|**68.45%**|**1.18%**|**Primary fails**|
| Branches-v1 | TRAIN |8|78.80%|11.57%|Yes, secondary|
| Branches-v1 | TRAIN |16|76.35%|7.73%|No|
| Branches-v1 | VAL |8|78.81%|10.62%|Yes, secondary|
| Branches-v1 | **VAL** |**16**|**73.87%**|**6.26%**|**Primary fails**|

For the new model's primary outcome, root-bootstrap 95% intervals are 69.06–78.53% for ranking and 4.80–8.81% for error reduction. Parent-session cluster intervals are 72.29–75.11% and 5.99–6.65%, respectively; only three parent sessions support this sensitivity analysis. VAL H16 contains 493 informative pairs of 504 total, with no numerical ranking ties. The 11 other pairs remain reported rather than discarded from cohort accounting. VAL H8 contains 492 informative pairs. Persistence ranks 0.5 by construction; its measured visual endpoint errors and all per-pair visual/proprio/control contrasts remain in the full report.

Phase diagnostics identify limited transfer and release ranking. For branches-v1 VAL H16, transfer ranks 65.70% with 4.30% error reduction; release-high ranks 64.29% with 1.39% reduction. Closure improves to 82.91% and 17.89%, and orientation to 87.27% and 13.18%. These descriptive results use only three roots per phase and cannot establish a reliable phase-specific success threshold.

The improvement is evidence of a stronger learned response to sustained actual action alternatives in this development distribution. It does not establish adequate long-horizon prediction, apple/contact visibility, zero-shot generalization or causal model dependence of a successful controller. Secondary H8 success cannot retrospectively replace H16. A new training-horizon, representation or controller hypothesis requires a separate prospective protocol; this report leaves the frozen failed gate unchanged.

## Frozen evidence

- Source commit: `5138e808751cf8164c9b64a2dad27113d76b6f8e`.
- Branch corpus manifest: `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- New selected checkpoint: `6c93b950cdc938df3fb121537c53ce62d585fe9df07db7265415ca01749fd3cb`.
- Original selected checkpoint: `3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee`.
- [Preregistered protocol](apple_branch_diagnostics_v1.md), [compact evidence manifest](../../benchmarks/manifests/apple-branch-diagnostics-v1.json).
- Local full artifacts: `outputs/apple-branches-diagnostics-v1/{registration,report,supervisor}.json` and `child.log`; hashes are recorded in the compact manifest. No new model execution is required to inspect or recompute the saved summaries.
