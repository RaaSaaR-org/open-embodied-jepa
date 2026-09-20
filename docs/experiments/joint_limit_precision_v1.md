# Joint-target precision investigation v1 — preregistration

Before reproduction, select exactly these three failed v1 development traces:

| Case | Failed pending step | Trace SHA-256 |
|---|---:|---|
| 43000-dynamics_shuffle | 266 | `fe7432a52f902c3bb6fd325a358c14a2341d09471743f028348082d05d24887a` |
| 43001-persistence | 302 | `13910de230c1ca849480f409bd9b4d507b32ad4f84e276dab7f0f4d9143b912a` |
| 43001-dynamics_shuffle | 635 | `c82c76487a65550214aafdcffc045d18bc2e0dc8a6a64762bba527b6204e6af6` |

Read `outputs/apple-control-development-v1/resolved_plan.json` for the unchanged
apple/plate reset coordinates. Replay each trace's acknowledged applied actions
through real headless MuJoCo, with fresh robot sensor snapshots. Do not rerun
world-model inference or sample new actions. At the failed pending action record
shared preparation's float64 target, its float32 transport representation, exact
MJCF bounds, violation magnitude, and robot state/last targets. Test the same
pending action's projection feasibility. Preserve output under the new
`outputs/joint-limit-precision-v1/` directory, refusing existing output.

Hypothesis: a float64 target on an MJCF boundary rounds outside its allowed interval
when cast to float32 for transport. An alternative is soft-limit physics allowing
a measured state beyond a hard command bound and immediate IK convergence returning
that measured value. Distinguish these with exact pre/post-conversion values.

Budget: one fixed replay of each of the three cases, **60 process CPU seconds**
and **120 true-wall seconds** total, supervised before all imports. No retries or
extensions for the reproduction. Report incomplete cases if a bound is exhausted.
No rendering, model fitting, scoring or final-cohort execution. The original v1
results remain immutable and negative. Any repair must preserve physical joint,
velocity, workspace and timing guards, and receive deterministic boundary and
late-snapshot regressions before a separately authorized rerun.

## Reproduction result and repair

All three fixed prefixes reproduced the original error in **1.8781 wall seconds /
2.1769 process CPU seconds**, with no world-model inference. Every pending action
was projected feasible and unchanged, then rejected by transport. Contrary to the
primary rounding hypothesis, the float64 IK results were already outside the
right wrist-roll lower bound of −1.97222 radians:

| Case | Float64 target | Violation before float32 | Violation after float32 |
|---|---:|---:|---:|
| 43000-dynamics_shuffle | −1.9722486538427482 | 0.0000286538427481 | 0.0000286734390258 |
| 43001-persistence | −1.9725240071079697 | 0.0003040071079696 | 0.0003040468978881 |
| 43001-dynamics_shuffle | −1.9722444779128485 | 0.0000244779128484 | 0.0000245011138915 |

MuJoCo's soft limit allowed measured wrist penetration. The scratch IK seed copied
that measured state and returned immediately when its EE residual met tolerance,
before its iterative clipping path ran. `_prepare_side` checked target rate but
not returned joint position limits, so projection declared those commands feasible.

The repair clamps only the scratch IK seed into command bounds before testing
convergence. The shared preparation path independently validates every returned
arm solution against the exact physical bounds, including early returns, and
checks rates after float32 command quantization. It also validates hand targets.
Physical endpoints are rounded inward by at most one float32 representable step
when nearest rounding would leave the interval. Invalid physical solver outputs
are rejected, not silently clipped. No physical limit, rate threshold, solver
residual tolerance, controller action scale or transport guard was relaxed;
no live qpos was changed by projection or preparation.

Each recorded late snapshot was restored once for regression validation. All three
unchanged projected actions now received `applied`, advanced actual PD physics by
0.05 seconds and commanded wrist roll −1.9722199440002441, inside the original
bound. This regression check took 0.1866 seconds. It is not a control rerun or a
success claim. Neither the historical traces nor their negative scores changed.

`tests/test_projection.py` embeds the exact first late snapshot without depending
on ignored artifacts. Additional tests cover both endpoint rounding directions,
strict rejection even one float64 step outside a bound, and a deliberately invalid
solver output failing projection and execution before actuation. Existing stale
snapshot, tracking lag, rate, object-independence, rotation and simulator tests
remain passing.

Validation command:

```sh
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_projection.py tests/test_embodiment.py tests/test_simulation.py -q
```

Result: **40 passed, 1 skipped** (graphics opt-in), 1.41 seconds. Ruff and diff checks
passed. Local reproducible evidence is under `outputs/joint-limit-precision-v1/`:
`reproduce.py`, `supervisor.json`, `report.json`, the three exact snapshots,
`validate_snapshots.py`, and `fixed-snapshot-validation.json`. Fixed embodiment
SHA-256: `2632259a8f17db4bd1543f8d6f5155ef47a3891e3712784a42bf294c43506ce9`.

Independent review by the contracts agent found no blocking findings: scratch-only repair, exact pre-quantization validation, inward transport rounding, post-quantization rate checks and shared projection/execution semantics preserve the physical guards. Coordinator review and focused tests also passed. PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/5.
