# TASK039 evaluator runtime checks

Scope: `scripts/evaluate_apple.py`, `tests/test_apple_evaluation.py`, and the
committed-command protocol's evaluator interfaces. This began as independent
review, then the coordinator transferred evaluator implementation ownership to
this agent. The following is therefore an **author check**, not independent
approval. The controller and final evaluator review belong to other agents.
No inference, physics, training, or control experiment was executed here.

The evaluator freezes optional `--attempt-max-seconds` (default absent) and
`--commitment-steps` (default 1). Each attempted worker receives the smaller of
its requested cap and the remaining global budget; setup and integrity checking
consume that global budget. There is no separate 55-second preparation limit
and no guarantee that all six attempts receive full 90-second allocations.
Requested/allocated/remaining time and global shortening are recorded. All
planned attempts remain in the report, including budget-prevented starts.

The child measures its optional soft cutoff from script entry, before runtime
imports, reserving `min(5, allocated_seconds/10)` for finalization. It checks
before observing and before execution. Observe, cached projection and any
fallback full search share the existing five-second control deadline. A pending
record still precedes execution; acknowledgment, actual action, checkpoint hash,
control time and evaluator-only task score are retained per command. Cached
controller cost/provenance fields pass through without being rewritten as a
newly evaluated forecast. Actual execution rejection remains terminal.

Issues corrected during the author check:

- A clean child soft timeout under a globally shortened allocation previously
  remained labeled as an attempt timeout. Both soft and hard cutoffs now record
  `timeout_scope` and distinguish attempt exhaustion from global exhaustion.
- A child that exited with code zero after a civil-clock jump could bypass the
  supervisor deadline check. Completion now checks elapsed wall/monotonic time
  before accepting that exit.
- Preparation previously became ready solely on a zero return code and existing
  output. It now also requires supervisor status `exited`, so a late preparation
  result cannot launch attempts.
- Invalid private worker budgets are rejected before output or simulation setup.

Validation: `PYTHONPATH=src .venv/bin/python -m pytest
 tests/test_apple_evaluation.py -q` passed **34 tests in 0.22 seconds**. Fixtures
cover full/global-shortened soft and hard timeouts, six-record accounting,
unchanged default behavior, explicit CLI forwarding, invalid limits, script-entry
startup time, both child cutoff locations, suspended civil clocks with a late
zero exit, late preparation, deadline prevention of actuation, and preservation
of partial/error/provenance evidence. Ruff formatting/checks and focused
`git diff --check` passed. These synthetic checks establish software behavior,
not robot manipulation success or a measured throughput claim.
