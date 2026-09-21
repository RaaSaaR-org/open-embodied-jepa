# TASK041 scientific implementation review

Reviewed 2026-09-21 before any authorized TASK041 fitting, world-model inference or physics. Scope: `scripts/apple_goal_alignment.py`, its focused tests, and `docs/experiments/apple_goal_alignment_v1.md`. No remaining blocking scientific defect was found in the reviewed implementation. Runtime supervision has a separate review; neither review constitutes a successful experimental result or external maintainer approval.

## Sampling and information boundaries

Metadata-only inspection of the actual frozen corpus reproduces 1,392 TRAIN and 360 VAL records, with no missing declared phase samples. The failed TRAIN parent `apple-42003` remains included; its extra `lower_open` phase is explicitly reported as unsampled. The deterministic sample-ledger SHA256 is `37aaa45ef11995240be8ee84c7779ad006d35c03b4519ac0c1c6f77c0d35f6f5`. Parent/session split validation, lexicographic sample ordering and parent→source-group→record weights match the protocol. Each parent's total weight was independently checked against the actual metadata. This check decoded no images or TEST payloads.

Preparation seals the TRAIN feature/label bank before VAL decoding. Fitting reads the TRAIN cache and uses the frozen world-model normalization; it does not choose a checkpoint from VAL. NN queries use only the TRAIN bank. Seven position fields resolve by exact schema names and units. Goal RGB is the sole input to either goal encoder. Measured goal joints enter offline error/ranking labels only. Frozen future predictions use the measured current observation and recorded applied actions. Sibling roots check starting RGB/state/masks and timestamp agreement. No simulated object pose or scoring state is an encoder input.

## Metrics and candidate decisions

The primary cross-parent H16 comparison uses identical eligible candidate pairs for every method. Eligibility checks measured endpoint arm separation and measured goal-distance separation separately. Cost ties receive half credit. Aggregation follows pairs within goals, goals within roots, roots within parents, and equal parent weighting. Measured-state and predicted-state costs are reported separately; H8 and matched-endpoint results cannot rescue the H16 gate.

NN and neural route gates are independent: 20% encoder-error reduction against the weighted TRAIN mean, at least five percentage points of predicted-ranking improvement against pixel cost, and strictly positive improvement on each VAL parent. Coverage must include all declared samples and three parents with at least two informative roots each. A missing or failed neural head does not invalidate a complete NN assessment. Neural replacement requires its own pass plus strictly better error and ranking than NN.

Uncertainty diagnostics do not alter primary eligibility or candidate costs. NN top-eight spread is exact, including zero, and is explicitly uncalibrated. Retained-coverage reports include counts and achieved weighted mass. Parent-cluster bootstrap uses paired parent resampling; a zero-baseline draw makes the reduction interval unavailable with an explicit invalid-draw count, rather than emitting nonfinite JSON or silently discarding draws. Saved full frozen prediction arrays permit later metric verification without another inference run.

## Review fixes and checks

Review feedback resolved explicit lexicographic ordering, artificial NN variance flooring, zero-denominator bootstrap handling, weighted retained-coverage disclosure, sibling timestamp validation and saved forecast evidence. The gradient-norm cap of 10 is now declared prospectively in the protocol and checkpoint configuration.

Independent checks completed:

- `PYTHONPATH=src .venv/bin/pytest -q tests/test_apple_goal_alignment.py`: 24 passed in 0.39 seconds.
- Additional synthetic checks of unequal pair-count aggregation, parent/source weighting, an aggregate gain with one non-improving parent, and NN eligibility without a neural result.
- Metadata-only actual-corpus sample counts, missing/extra phase reporting, deterministic ledger digest and equal total parent weights.
- Read-through of ranking array indexing, frozen-state normalization, measured-versus-predicted goal costs and independent candidate failure handling.

## Limits

These checks establish implementation agreement with the prospective screen, not its empirical outcome. There are only three VAL parent clusters. Seven arm positions cannot distinguish all hand/object configurations; the documented same-arm/different-object example remains an explicit limitation. Passing this screen would support an image-only arm-alignment representation, not apple manipulation success, an arm-only deployment policy or permission to change physical scoring.

The reviewer did not author this script or its training head implementation, but previously authored the sensor backend and parts of earlier diagnostic tooling and helped specify this protocol. This is an independent implementation review within the project team, not independent replication of the research result. No fitting, world-model inference, physics, source edits, Git operations or MissionControl changes were performed for this review; the only authored repository artifact is this review document.

## Completed saved-evidence audit

Completed 2026-09-21, after all three stages ended. This audit uses saved NumPy arrays and JSON only; no fitting, encoder inference, world-model inference, image decoding or physics was repeated.

Source revision: `f4f99fa1bf5fdcec7f1647927e74d0b0de8228b0`. Script SHA256: `005123e9fc450c7e1485abdcf63d68a10a22f9ed5f45dcd7d21004a22e17ab7d`. Protocol SHA256: `4b132f4641541a47951a151a9d1d675ce04676a7595aa43e9098ad36c9f0d658`. Frozen world-model checkpoint SHA256: `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. Corpus SHA256: `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.

All 31 registered source/input file hashes match current bytes. All 38 sealed artifact hashes match: prepare 9, fit 4, evaluate 25. Common source identities agree across stages. Every stage completed with integrity verified, no timeout and supervisor return code 0. Supervisor elapsed seconds: prepare 6.514556, fit 2.682128, evaluate 2.011637; worker CPU seconds 5.723556, 2.363342, 1.347775 respectively. Fit records all 2,000 prescribed updates.

Preparation records exactly 557 unique decoded episode IDs, equal to the TRAIN/VAL union and disjoint from TEST; all TRAIN IDs precede VAL IDs. The bank digest matches its recorded pre-VAL digest. TRAIN and VAL cache dimensions are 1392×1728 and 360×1728 with seven labels per record; 18 roots contain 8×16×14 applied-action sequences. The saved bank weights and weighted TRAIN mean were independently recomputed. NN saved estimates equal the referenced TRAIN-bank labels.

An independent NumPy audit reconstructed 3,456 cost vectors and 16,128 unordered-pair records across H8/H16 primary and matched-secondary cohorts from saved estimates, measured features/labels and frozen forecasts. It verified eligibility, labels, ties, scores, hierarchical aggregation, parent summaries and 2,000-draw parent bootstrap intervals. Encoder errors, radians RMSE, uncertainty coverage/widths, retained-risk weighted coverage and 504 ambiguity records were also recomputed. The audit script is `/tmp/audit_goal_alignment_saved.py`; its compact console result is `/tmp/apple_goal_alignment_saved_audit.json`. No original evaluation implementation was imported for these calculations.

Primary H16 has 144 cross-parent root-goal cases and 2,888 eligible comparisons of 4,032; 1,144 have low arm separation. All 18 roots are informative, six per parent. No measured image/arm ambiguity passes the protocol's narrow threshold among 504 pairs; this does not establish general visual identifiability or resolve arm/object ambiguity.

| Candidate | Encoder error reduction | Predicted H16 ranking | Gain vs pixel | Decision |
|---|---:|---:|---:|---|
| Pixel comparator | n/a |73.37780083%|n/a|comparator|
| NN |99.61280421%|89.70627906%|16.32847823 percentage points|passed|
| Neural |99.88800312%|89.81618266%|16.43838183 percentage points|passed|

Both candidates improve on pixel ranking on each parent. The neural replacement rule passes, but its aggregate ranking advantage over NN is only 0.10990360 percentage points; this is a prospective rule outcome, not proof of a meaningful or statistically established difference between encoders. Neural measured-state ranking 98.90993266% exceeds its predicted-state ranking 89.81618266%; the remaining gap is relevant before control but does not uniquely identify dynamics error. H8 remains secondary and favors NN slightly. All confidence intervals use only three parent clusters and must retain that limitation. Neither arm-only result establishes hand/object state, successful manipulation or physical deployment readiness.

Provenance exception: the supplemental coordinator launch ledger was written after preparation began because its system-Python preamble referenced unavailable `datetime.UTC`. The stage supervisor had correctly written its own immutable registration before data work. The supplemental ledger explicitly discloses its late creation; it must not be presented as a prelaunch artifact. There was one preparation attempt and no experimental retry according to the coordinator record and saved stage accounting. Saved files alone cannot independently prove the absence of every unrecorded external attempt.
