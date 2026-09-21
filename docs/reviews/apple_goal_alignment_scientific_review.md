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
