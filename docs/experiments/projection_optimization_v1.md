# Scratch kinematics projection optimization

TASK-030 hypothesis: full forward dynamics are unnecessary inside inverse kinematics and candidate preview, which consume body/site transforms and joint Jacobians. The baseline profile spent 39 of 69 ms in 1,099 `mj_forward` calls for a 64-candidate, four-step fixture.

The optimization replaces full forward dynamics only on scratch states with `mj_kinematics` followed by `mj_comPos`. The latter updates joint axes needed by `mj_jacSite`. Live torque control, integration, reset, limits, numerical tolerances, backtracking and acceptance checks remain unchanged. Projection still uses the observed robot joints and command history, without object state or future sensors.

The fixed profile compares the optimized path against the same implementation with its scratch update replaced by the previous `mj_forward` call. It uses 64 candidates, horizon four, seed341, robot reset41, three timed repetitions, and two states: initial reset and after six physical torque-control commands. The left arm is inactive, matching the common planner configuration. RGB is explicitly synthetic; joint dynamics are real MuJoCo G1/Dex3 physics. This fixture measures projection latency, not rendering or learned task success.

| State | Reference median | Optimized median | Speedup | Outputs |
|---|---:|---:|---:|---|
| Reset | 69.90 ms | 32.66 ms | 2.14× | Bit-identical actions and feasibility |
| After six commands | 70.93 ms | 33.29 ms | 2.13× | Bit-identical actions and feasibility |

All 64 sequences were feasible in each profile state. The complete comparison used 1.32 process CPU seconds and 1.05 wall seconds, below the declared 60 CPU-second limit. This is a small fixed fixture on the local Apple Silicon Mac; it does not guarantee the same speedup near every joint limit or establish a 20 Hz full planning loop.

Reproduce with a fresh output path:

```sh
PYTHONPATH=src .venv/bin/python scripts/spikes/projection_profile.py --output outputs/projection_profile_v2
PYTHONPATH=src .venv/bin/python -m pytest tests/test_projection.py tests/test_embodiment.py -q
```

The profile writes its plan before execution, including exact source hashes, host platform, revision and random seeds, plus input candidates, reference/optimized cProfile reports, and measured results. Artifacts: `outputs/projection_profile_v2/`. Initial untouched-source reference arrays and source snapshot are retained separately in `outputs/projection_profile_v1/`.

Validation: 23 focused tests pass in 0.90 seconds. Two new tests compare the full-forward reference and optimized projections bit-for-bit using broad both-arm requests at reset and after actual dynamic motion. Existing tests retain the stale-transform two-step regression, preview/execution equality, purity, object-state independence, hand target rates, infeasible-zero handling, and CEM/MPC integration. Ruff and diff checks pass.

Optimized embodiment SHA-256: `34c47f7a11e8f2ba1a82c3f981a0a092a7d651a1ade0730b975b320ce8181083`.
