# Apple control forecast audit: independent pre-run review

Reviewer: contracts agent, 2026-09-21. Scope: prospective TASK038 script, synthetic tests and protocol in `/tmp/jepa-forecast-audit-work`, plus read-only inspection of the original interrupted control trace and frozen runtime interfaces. This reviewer did not author this audit implementation; they authored the sensor backend and earlier matched branch diagnostic. This is not external maintainer approval.

**No blocking findings remain in the reviewed draft.** The audit reconstructs planner state by replaying the original controller once, preserving RNG consumption and pending acknowledgements. Capturing both real projector results recovers full scored candidate sequences without substituting logged first actions. Siblings restore full MuJoCo data, actuator targets, grasp and scorer state; controller restoration includes RNG, warm sequence, progress counters and model-owned goal caches. Fresh observation refreshes derived projection state. Open-loop branches do not advance the controller, and the original pending command is executed once after restoring the root.

Replay must match discrete goal identity, advancement, candidate ordering, action values, timestamps and ordered scorer output before using a root. The original projected-sequence prediction and actual-applied-prefix reforecast use the same initial measured latent and frozen image goal. No future robot state is supplied to the model, no normalization is refit, and no TEST image decoding is required. Attribution remains diagnostic: the reforecast does not uniquely separate model bias, hidden state and servo tracking error.

Review findings resolved before execution:

- Replay comparison now checks observation/acknowledgement time, goal image hash and advancement, in addition to actions, candidate costs and selection.
- Interruption recovery now counts only contiguous validated durable frame/action files and reports possible extra execution only beyond that prefix or for malformed evidence. A stale pending marker does not erase an already acknowledged frame.
- H1/H16 unavailable metrics carry explicit reasons.
- The simulation reviewer identified scorer `TaskThresholds` dataclass serialization failure. Explicit dataclass conversion fixes persisted JSON while retaining the in-memory type used for restoration.

Independent command: `.venv/bin/python -m pytest -q /tmp/jepa-forecast-audit-work/test_apple_forecast_audit.py` from `jepa-apple-horizon`: **12 passed in 0.04 seconds**. These tests cover candidate recovery, replay mismatch rejection, RNG/warm/pending/cache restoration, mocked full-state restoration, exact unpadded actual-action reforecast, nullable score checks, durable acknowledgement versus interrupted execution, gaps and dataclass serialization. No actual model inference or physics was run during this review.

Original trace inspection confirmed 574 consecutive acknowledged commands and the frozen trace hash. Roots 119, 300 and 550 all use the stalled goal at demonstration frame 140; they are correlated development states. Nine planned branches remain in failure reports. The parent reserves finalization time within the 180-second wall budget, and the child installs its 120-second CPU limit before importing Torch. Actual replay determinism, real runtime interfaces and completion within those bounds remain unverified until the single authorized experiment. A failed or partial attempt must be retained without a retry or tolerance relaxation.
