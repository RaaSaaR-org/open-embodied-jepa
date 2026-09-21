# TASK041 goal-alignment runtime review

Independent review of `scripts/apple_goal_alignment.py` and its synthetic tests against the prospective TASK041 protocol at `c02bd687d64e72084be592138c8af8e06de09a0d`. The reviewed implementation was uncommitted at that base; its SHA-256 is `005123e9fc450c7e1485abdcf63d68a10a22f9ed5f45dcd7d21004a22e17ab7d`. This reviewer authored the separate planning-time data audit, but did not author or edit this implementation. Scientific metric review is additionally assigned to the contracts reviewer.

No remaining blocking implementation finding was identified. This is code-review readiness for a separately authorized bounded offline experiment, not evidence that either representation passes its screen or that manipulation works. No corpus fitting, learned-model inference, simulation, TEST decoding, Git mutation, or tracker change was performed during this review.

## Input boundaries and provenance

The seven right-arm fields are resolved by exact StateSchema names, checked for radian units, and mapped through the frozen sensor normalization. RGB features alone enter goal inference; measured goal joints enter offline labels only. Current measured root proprioception is a legitimate world-model input, while future measured state never enters its prediction call. Candidate costs use inferred goal means, with no uncertainty weighting or scoring-truth inputs.

Sampling covers TRAIN and VAL only, includes failed originals, preserves parent/session membership, and rejects incomplete declared coverage. The TRAIN bank is serialized and hashed before VAL decoding. Sibling root RGB, state, masks, and complete timestamp sequences are compared; canonical RobotState construction uses the saved root timestamp. Applied episode actions supply the eight candidate futures. The frozen manifest and checkpoint hashes are enforced, source/protocol identities are checked across stages, producer seals verify cache artifacts, and all manifest-listed encoded payload hashes are checked before and after each stage. TEST files participate only in encoded byte verification, never image/state decoding.

## Failure and resource behavior

The supervisor measures the maximum of civil and monotonic elapsed time from entry, including worker imports. Stage allocations remain 120/300/60 seconds, with five seconds reserved for finalization. A late zero exit is a timeout. Post-supervision identity checks, payload verification, report writing, and sealing count toward completion eligibility; a deadline crossed during finalization is explicitly downgraded. Failure recording may itself finish after the deadline, but cannot be represented as a completed bounded experiment. CPU thread environment limits include OMP, MKL, OpenBLAS, and macOS VecLib at four threads.

Malformed or missing worker reports produce incomplete outcomes and planned candidate/root ledgers where the prepared ledger is available. Producer identity or cache failures prevent use of invalid prepared data. An absent, failed, malformed, or otherwise ineligible neural fit remains separate from the valid NN route; neural loading uses `weights_only=True`, requires exactly 2,000 updates, and catches neural-only inference failures. Incomplete evaluation cannot promote either candidate. Partial reports, caches, logs, and per-root prediction artifacts remain available; partial computation is not silently retried.

## Findings resolved during review

- Added sibling timestamp validation and explicit canonical root timestamps.
- Added strict expected sample coverage and missing-phase rejection.
- Isolated neural loading/inference failure from nearest-neighbor evaluation.
- Added sealed stage provenance, encoded payload checks after execution, and malformed-report handling.
- Included finalization in deadline eligibility and limited Mac BLAS threads explicitly.
- Guarded zero-denominator bootstrap draws, reporting unavailable intervals without nonfinite JSON.

## Validation and limits

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_apple_goal_alignment.py -q` passed **24 tests in 0.40 seconds** during final review. `git diff --check` passed. The tests exercise split/session leakage, exact field mapping, balanced weights, deterministic retrieval, comparison eligibility, parent aggregation, candidate separation, producer seals and malformed optional-fit isolation, zero-denominator bootstrap handling, and supervisor timeout behavior. A zeroed synthetic neural head checks output variance bounds; no trained checkpoint inference was run.

Two finalization tests initially omitted the dataset argument and could therefore pass through an incidental integrity exception. The test owner corrected these fixtures and added healthy completion versus post-seal deadline crossing, including final report/seal hash agreement. The final focused run passed these regressions. Actual decoding, stage throughput, fit convergence, and empirical representation quality remain unverified until the single authorized run. Three VAL parents limit statistical interpretation, and right-arm alignment alone does not identify object placement or grasp state.
