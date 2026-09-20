# Balanced H16 branch training: matched diagnostic results

The H16 model **passes the unchanged primary VAL H16 causal-prediction gate**: action-ranking accuracy is **89.40%**, and correct-action visual endpoint error is **81.41% lower** than wrong-sibling-action error. The thresholds were 70%, 10%, and a strictly positive root-bootstrap lower bound for error reduction. Coverage also passes: 18 informative roots from three parent sessions, with 493 informative pairs out of 504 and no numerical ranking ties.

The trained model meets the operational action-assignment and endpoint-error criteria on this development cohort. It is **not evidence of successful closed-loop pick-and-place**. A controlled planner experiment is the next separate test. Both horizon and original/intervention sampling weights changed, so their individual contributions cannot be separated from this experiment.

| Model | VAL H16 ranking | Correct-action error reduction | Primary gate |
|---|---:|---:|---|
| Original sensor-v1 | 68.45% | 1.18% | Fail |
| Uniform-window branches-v1, H8 training | 73.87% | 6.26% | Fail |
| Balanced branches-h16-v1 | **89.40%** | **81.41%** | **Pass** |

The new model's root-bootstrap 95% intervals are **86.57–92.56%** for ranking and **76.70–85.32%** for error reduction. Parent-session cluster intervals are 86.73–90.81% and 74.78–84.74%. Only three VAL parents support that sensitivity analysis; the many correlated pairs are not independent trials. Mean root-weighted own-action raw visual-grid MSE is 0.00117094, versus 0.00629980 with sibling actions. Prediction error itself also decreases from the prior branch model's 0.00382851, so the larger contrast does not consist solely of worsening wrong-action error.

Secondary VAL H8 ranking is 84.05%, with 85.28% error reduction. TRAIN H16 gives 91.03% and 84.12%. All six VAL H16 phases improve relative to the prior branch model, although phase summaries have only three roots each:

| Phase | Ranking | Error reduction |
|---|---:|---:|
| Orient | 93.96% | 72.53% |
| Descend | 99.40% | 91.02% |
| Close | 87.28% | 79.13% |
| Lift | 87.04% | 82.52% |
| Transfer | 86.00% | 81.68% |
| Release-high | 82.74% | 36.65% |

The diagnostic completed all 66 roots and decoded all 528 TRAIN/VAL branch episodes in **8.328 seconds true wall / 7.733 seconds worker CPU**. No TEST images were decoded, no fitting or checkpoint reselection occurred, and no retries, timeouts or identity drift occurred. The original baseline's complete summaries reproduce exactly, and the ordered decoded cohort exactly matches the earlier H8-training diagnostic. Each checkpoint retained normalization from its own source corpus.

The model completed the frozen 3,000 updates, with step 3,000 selected by normalized visual prediction MSE averaged over the H16 rollout. Each training and validation batch used eight full-demonstration windows and eight intervention windows. The unchanged architecture and optimizer were trained under the prospective [H16 training protocol](apple_branch_training_h16_v1.md); the same [matched diagnostic definitions](apple_branch_diagnostics_v1.md) and thresholds were retained.

## Evidence

- Diagnostic source: `1b10983` (full revision and source digest in the manifest).
- Checkpoint SHA-256: `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`.
- Corpus manifest SHA-256: `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- [Compact evidence and historical comparison](../../benchmarks/manifests/apple-branch-h16-diagnostics-v1.json), including sampling-group hashes, per-phase results and full artifact hashes.
- Full local artifacts: `outputs/apple-branches-h16-diagnostics-v1/{registration,report,supervisor}.json` and `child.log`.

The previous failed gates and negative control rollouts remain unchanged. This diagnostic does not validate contact visibility, robustness beyond these parent sessions, the final held-out control cohort, real hardware, or a world-model contribution to a successful controller.
