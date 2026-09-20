# Task-specific apple world-model MVP

The 2026-09-20 user goal now requires demonstrated learned Apple→Plate control. Historical zero-shot results remain unchanged. New targeted apple training is allowed and must be labeled task-specific.

Execution graph: TASK-030 (runtime) and TASK-031 (mechanics/data) → TASK-032 (observable learned dynamics) → TASK-033 (training/development gates) → TASK-034 (fresh frozen evaluation and delivery). TASK-032 design can proceed alongside mechanics; dataset-dependent implementation waits for evidence.

Working target is 16/20 complete successes on fresh frozen resets with existing task thresholds, not merely passing software tests or generating a successful script. Learned action-conditioned predictions must participate in selecting actions; matched model ablations assess contribution. Protocols fix seeds, hashes, budgets and controls before each run. Negative results remain recorded.

Initial bounded probes: projection profiling ≤60 CPU seconds; apple mechanics ≤180 CPU seconds; model diagnosis ≤60 CPU seconds. Training attempts later have explicit budgets ≤1800 wall seconds each. The coordinator schedules expensive experiments to avoid timing/resource interference.

Isaac runtime and physical commissioning remain externally blocked by unavailable compute/hardware. Optional JEPA-WMs integration is reviewed separately and is not a substitute for improving the working apple model.
