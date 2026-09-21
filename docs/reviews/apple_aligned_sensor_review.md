# TASK042 aligned sensor scientific review

## Prospective design review

Reviewed `/tmp/apple_aligned_backend_design.md` on 2026-09-21 before implementation validation, scale calibration, additional inference or physics. This review applies the project `jepa-review` workflow. No blocking scientific objection was found to the proposed bounded experiment. Implementation correctness, numerical parity, artifact construction and experimental outcomes remain unverified at this stage.

The proposed cost preserves the image-only goal interface: the learned goal head receives RGB and estimates seven normalized arm positions, while the frozen action-conditioned model supplies predicted positions and visual features. Goal measurements and simulator scoring truth are not runtime inputs. Separate group means and fixed 0.5 weights make the combination explicit; they do not establish equal task importance. Retaining a visual term preserves sensitivity to visual differences in principle, but cannot guarantee that object or hand state receives enough weight.

The scale rule is prospectively specified and uses only the 26 original TRAIN parents, including the failure: nonoverlapping complete stride-28 pairs, median within each parent, then median over parents. This gives each parent equal influence and avoids letting intervention branches dominate calibration. Retaining zero pairs and stopping on nonpositive final scales prevents a silent post-result recipe change. The scale describes demonstrated change over 28 commands; it is neither a unit of physical correctness nor evidence that H16 prediction errors have the same distribution.

The offline gate evaluates the actual combined cost using saved forecasts, a fixed cohort and the original pair eligibility and parent-balanced aggregation. It is a legitimate development gate, but reuses VAL data already examined in TASK041 and must not be called a new held-out confirmation. Its labels rank arm alignment only. Passing does not establish object placement, grasp or release, and does not resolve the known similar-arm/different-object limitation.

Two details were requested before protocol freeze:

1. Declare numerical parity tolerance between adapter costs and the saved-array formula; suggested `rtol=1e-5, atol=1e-7` with the original float32 operands.
2. Define the offline measured-state hybrid explicitly as measured endpoint visual features and measured endpoint arm positions compared with goal visual features and inferred goal positions. This attribution diagnostic differs from the runtime image-only `observed_distance`, which infers both current and goal arm positions. Head error can therefore affect arrival even when forecast-cost parity holds.

The conditional MuJoCo comparison preserves the same checkpoint, blended metric, TRAIN-calibrated goal thresholds, goals, proposals, physics and guards across learned/persistence/shuffle modes. Commitment one removes the prior four-command variant. The planned three attempts use one development reset; resource caps and all incomplete attempts must remain visible. Different mode runtimes can censor trajectories at different command counts, so a timeout cannot be equated with failure after all 1,000 commands. A learned success with failed controls would support another bounded investigation, not final statistical manipulation acceptance or real-hardware validation.

Bundle validation must bind the original world-model normalization and schemas, completed head/config/TRAIN ledger, TASK041 producer evidence, calibration recipe and all member hashes. Paths must resolve within the declared bundle; incompatible or missing members must fail without retraining, inferred defaults or asset downloads. Historical checkpoints and source remain unchanged. Model-owned composition is consistent with the generic planner boundary; no backend-specific control policy is warranted.

The reviewer previously authored the sensor backend and contributed to TASK041's protocol and evidence audit, but is not implementing the new composition or builder. This is an independent design/implementation review within the project, not external replication or maintainer approval. Only this review document was authored; no calibration, fitting, inference, physics, Git or MissionControl operations were performed.
