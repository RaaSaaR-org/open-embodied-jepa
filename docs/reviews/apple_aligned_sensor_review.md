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

## Implementation review in progress

Protocol commit `a1d5df5` incorporates both requested definitions and explicitly labels reused VAL as development/model selection. Initial read-through of the builder's completed calibration sections confirms complete stride-28 endpoint selection and zero-retaining, equally weighted parent medians. The immutable corpus identity makes the declared 26-parent cohort explicit. No data execution was performed.

Early findings sent to the builder author: align its protocol filename and numerical tolerance with the committed protocol; record and verify the pair-ledger digest before decoding; validate the TASK041 producer chain consistently across preparation, head fitting and evaluation, rather than matching only corpus/checkpoint names. Bundle construction, saved-array comparison and backend code were still being implemented at this review boundary, so they remain unverified. These findings are not a final approval of an incomplete implementation.

### Completed builder and backend read-through

The builder now uses the committed protocol path and exact `rtol=1e-5, atol=1e-7` tolerance. It records the pair-ledger digest before decoding and checks it again before packaging. TASK041's three producer stages must have identical historical base identities, and their references to earlier preparation/fitting artifacts are checked against actual bytes. The historical metric-helper implementation must still match its registered digest. These resolve the early findings above.

The complete calibration path decodes only the selected original TRAIN episodes, reuses the frozen child normalization, checks the seven selected masks, preserves zero-distance pairs, and takes the declared two levels of median. The backend independently reconstructs expected pair coverage and calibration medians from bundled provenance. Its goal and progress methods receive images only; predicted cost reads only the selected seven positions from child predictions. The original child prediction implementation is delegated unchanged. Successful loading changes latent ownership only after all member/config/provenance checks; failed validation does not commit a new child.

The saved-array audit preserves the original cross-parent goal roster, exact eligibility labels and pixel rankings. It computes both measured and predicted blended costs and applies the H16 gate through the original parent-balanced aggregation. Helper-level parity uses float32 saved operands and the prescribed tolerance. This helper check must be complemented by a test of the full loaded wrapper's `distance` path, head equivalence and delegated predictions; those tests were still being completed at this review boundary.

`PYTHONPATH=src .venv/bin/pytest -q tests/test_apple_aligned_bundle.py` passed 17 synthetic tests in 0.04 seconds. They cover failure retention in parent selection, complete pair boundaries, split/session rejection, zero-retaining medians, degenerate-scale failure, each-parent gate enforcement, nested artifact corruption and incomplete/deadline result handling. No real corpus calibration, encoder inference or physics was run. A preliminary concern about a top-level camera config requirement was checked and withdrawn: the generic evaluator does not access `model.config['camera']`, and the builder's camera access is to the original sensor child.

No remaining scientific blocker was found in the completed code read-through. Final execution readiness still depends on the adapter integration tests and the coordinator's runtime/evaluator review. This statement does not mark the offline gate or physical comparison as passed.

### Final synthetic integration review

The completed adapter tests now exercise actual loaded `distance` outputs against an independent formula, exact delegated child predictions with candidate chunking, image-only progress/goal signatures, visual-only cost differences, load ownership invalidation, failure preservation, portable save/load, strict bundle corruption/provenance rejection, and generic evaluator loading plus TRAIN-image waypoint calibration. Packaging now binds source/protocol/revision and the historical goal-training source hash. The focused adapter/builder/config suite independently passed 55 tests in 1.33 seconds.

One test-quality issue was routed for correction: the initial head-equivalence check used an all-zero head fixture, which cannot distinguish incorrect activation/layer behavior. A deterministic nonzero-head comparison was requested; this is a verification gap, not a demonstrated defect in the reviewed head construction. A named-position mapping test with reordered and unrelated state fields was also suggested. The scientific implementation review otherwise found no remaining blocker; real-data calibration and offline/physical outcomes are still unverified.

A separate reviewer-run synthetic check subsequently used deterministic nonzero head weights (seed 19) and three nonzero 1,728-feature inputs. The adapter head and historical TASK041 factory produced bit-exact outputs. The seven named position indices also resolved correctly with reversed field order and unrelated position/velocity fields. This closes the substantive equivalence uncertainty independently of the pending persistent regression. Added per-parent saved normalized signal caches preserve the exact calibration operands for later evidence verification without repeated decoding or inference. There is no remaining scientific implementation blocker to the registered bounded attempt; experimental gates and physical outcomes remain unverified until their saved evidence exists.

### Final readiness conclusion

The author added persistent regressions for both requested cases. The nonzero-head test packages a deterministic TASK041 head, evaluates five varied synthetic RGB images, verifies all 35 means are nonzero and every output channel varies, then checks exact equality through the loaded wrapper. The named-field test uses reversed right-arm positions plus unrelated velocity fields with large nuisance values and distinct per-joint errors; the full wrapper cost matches the independently calculated seven-field result. Both tests were independently read and run.

The final adapter/builder/config suite passed **57 tests in 1.30 seconds**. All review findings and verification gaps above are resolved. The scientific/code review has **no remaining blocker to the registered TASK042 attempt**, subject to the separate completed runtime review. Calibration, offline-gate success and physical manipulation are not claimed by these synthetic checks. No work beyond TASK042 is authorized by this review, and the user's requested pause after TASK042 delivery remains in force.

## Completed TASK042 saved-evidence audit

The registered attempt at source `9942e1e34bf76bb6b15911149bf32d6dd66cc7aa` completed preparation and offline audit. **The fixed hybrid gate failed.** Primary H16 ranking was 77.67960358%, compared with 73.37780083% for pixel cost: a gain of 4.30180275 percentage points, below the required 5 points. Every parent improved, but that does not waive the aggregate threshold. The conditional physical comparison must not run. This completes the registered negative experiment, not the positive numerical acceptance criterion or the user's working-manipulation objective.

Independent verification used saved JSON, NumPy signal/forecast arrays and safe checkpoint metadata only. No encoder or world-model inference, image decoding, fitting or physics was repeated. The independent recomputation script is `/tmp/audit_aligned_saved.py`; its result is `/tmp/apple_aligned_saved_audit.json`.

All **64 sealed artifact hashes** match (43 preparation, 21 audit), as do every registered source/input identity and all 10 portable bundle member digests. The child sensor checkpoint remains byte-identical to SHA256 `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`; the completed 2,000-update head is `61e1897f80a688fea170a20b73870bac6762d3f6aa0b6c33bbcf449c4819b7d0`. Wrapper normalization metadata exactly matches the child and its complete frozen TRAIN membership. Both stages report completed status and post-run integrity verification without timeout.

The calibration audit independently reconstructs all **447 stride-28 pairs across 26 original TRAIN parents**, including the failed demonstration, directly from the saved normalized operands. Parent/session membership and every frame pair match the metadata-only rule; the pre-decode pair-ledger hash matches final bytes. All pair distances and both levels of medians match exactly. No pair was exactly zero in this attempt; the implementation still retains zeros as specified. Scales are:

- Visual: `0.08636228494017421`.
- Pose: `0.15918705911991243`.

The ranking audit independently reconstructs **1,728 cost vectors and 8,064 pair records** over both horizons, including measured and predicted components. It confirms the frozen cross-parent goal roster, unchanged TASK041 eligibility, tie handling, aggregation and 2,000-resample parent bootstrap. All 18 roots completed, with six informative roots per parent. H16 retains 2,888 of 4,032 pair comparisons; H8 retains 2,606. All runtime-helper costs satisfy the declared float32 formula tolerance. The recorded maximum absolute discrepancy, `1.9073486328125e-06`, is permitted by the combined relative and absolute tolerance; it is not evidence of exceeding an absolute-only bound.

| H16 method | Predicted ranking | Measured-state ranking |
|---|---:|---:|
| Pixel | 73.37780083% | 82.24926347% |
| Frozen neural pose-only comparator | 89.81618266% | 98.90993266% |
| Fixed visual/pose hybrid | 77.67960358% | 90.82773669% |

Hybrid gains by parent are +4.14151936, +4.07754630 and +4.68634259 percentage points. The parent-cluster 95% intervals are 76.63111772–79.02430556% for ranking and +4.07754630–4.68634259 points for gain. These intervals use only three parent clusters and reused development VAL data; they do not establish broad generalization. H8 hybrid ranking is 80.06442549%, a secondary result that cannot rescue the failed H16 gate.

The fixed mixture improves arm-ranking over pixels but loses substantial ranking accuracy relative to the already frozen pose-only comparator on this cohort. That describes this recipe's result; it does not establish that all visual/pose mixtures fail, that object information is unnecessary, or that a different controller would succeed. No new mixture, scale statistic, threshold or experiment is proposed or run as part of this audit. Physical success is **unassessed for this version**, not zero successes out of three attempted runs: the three modes were conditional and were not launched.

Preparation used 4.744393 seconds of supervisor wall time and 3.772367 worker CPU seconds; audit used 2.261272 wall and 1.398091 CPU seconds. Both remain within their separate 120/60-second allocations. The sum is a sum of separately supervised stages, not a continuous coordinator duration. The user's requested pause after TASK042 delivery remains in force; this review authorizes no follow-on work.

Final delivery cross-check: `docs/experiments/apple_aligned_control_results_v1.md` and `benchmarks/manifests/apple-aligned-cost-v1.json` agree with this independent audit. All 68 manifest-listed artifact hashes match, including stage seals and logs beyond the 64 sealed members. The corrected composition provenance exactly matches the checkpoint envelope's metadata. The report correctly preserves the failed numerical criterion and distinguishes the unattempted conditional physical stage from a zero-success trial. No remaining reporting finding was identified; no further experiment or follow-on task was performed.
