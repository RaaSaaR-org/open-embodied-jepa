# Apple mechanics probe — preregistered exploratory control

This is privileged scripted simulation control for mechanics diagnosis and future
data collection, **not learned world-model control**. Existing release-v1 data has
six recorded apple→bowl trajectories and none achieved the scorer's contact lift.
The nine successful release-v1 episodes therefore do not demonstrate apple grasp.

Hypothesis: the fixed grasp pose is mismatched to the spherical apple. Changing
the commanded palm offset may produce a stable contact lift and early release
without changing physics, finger geometry, gains, velocity guards, or scoring.

Before execution, freeze these twelve attempts. Reset seeds 41000 and 41001 use
independent uniform ±0.004m XY jitter around apple (0.34, −0.18) and plate
(0.49, −0.09), in that RNG order. Within each seed, execute these six candidates:

1. Original OracleManipulationPolicy, unchanged (805-command maximum).
2. EarlyReleaseOracleManipulationPolicy, opening ramp0.08, unchanged (745 commands).
3. Early release with descend/close palm height0.037m above initial object center.
4. Early release with descend/close palm height0.067m above initial object center.
5. Early release with palm X offset−0.045m relative to object/container center.
6. Early release with palm X offset−0.015m relative to object/container center.

Unchanged early-release offsets are X−0.03m and descend/close height0.052m.
X changes apply consistently to every phase, including transfer/release. Height
changes affect only descend and close. No contact-dependent tuning occurs during
or between attempts. Initialization truth is allowed solely for this labeled
oracle. All actuation goes through observe→execute, with actual collisions and
the unchanged ordered AppleToPlateTask default thresholds. Stop an attempt after
observed scorer success, rejected command, policy completion or resource cap.

The shared limit is **180 process CPU seconds and 180 true-wall seconds** (whichever
comes first), including initialization, recording and finalization. Stop new
commands at170s to reserve10s for outputs. A wall watchdog limits the subprocess
to180s; a process CPU resource limit applies where supported. Record every planned
attempt, including failures and unstarted/budget-truncated trials. No retries or
budget extensions. A few successes establish feasibility at these layouts, not
reliability over a broad reset distribution.

Save an immutable pre-execution plan and source snapshots, exact requested/applied
actions, 20Hz RGB/state/mask/timestamp transitions, phase labels, task scores,
object pose/contact truth for mechanics analysis, maximum joint velocity and its
joint name, failure reason, phase-end images, source hashes and resource clocks.
Rejected requests stay in the trace but never become applied transitions. Original
datasets and evaluation artifacts remain untouched; these Apple→Plate development
probes are not appended to the original held-out corpus.

To permit concurrent independent software work, the runner copies the complete
current Python source tree, script, action configuration and asset manifest into
an immutable adjacent `-runtime` directory before launching the worker. Upstream
robot assets remain a read-only shared path. The worker imports only that snapshot;
its full Python-source hashes are recorded, so subsequent worktree edits cannot
change an ongoing trial's implementation.

```sh
PYTHONPATH=src .venv/bin/python scripts/apple_mechanics_probe.py \
  --output outputs/apple-mechanics-v0
```

No trial has run under this protocol at registration. Results and recommendations
will be appended below, preserving the registered plan in the output directory.

## V0 observed results

All 12 attempts completed once in **53.17 true-wall seconds / 49.13 process CPU
seconds**. Ten failed and two achieved the unchanged ordered task success rule.
No physics, scoring threshold, gain, or safety limit changed. No learned model
participated. Complete plan/report/source snapshots are in
`outputs/apple-mechanics-v0`, with the immutable imported runtime alongside it in
`outputs/apple-mechanics-v0-runtime`.

| Candidate | Seed 41000 | Seed 41001 |
|---|---|---|
| Original | Velocity stop during close, 252 steps | Same, 252 steps |
| Early release | Velocity stop during close, 252 steps | Same, 252 steps |
| Lower grasp | Velocity stop during close, 252 steps | Same, 252 steps |
| Higher grasp | Velocity stop during close, 252 steps | Same, 252 steps |
| Rear grasp X−0.045m | Apple ejected; IK stop at593 steps | Apple ejected; IK stop at594 steps |
| Forward grasp X−0.015m | **Success at512 steps** | **Success at518 steps** |

The two successful attempts achieved contact lift, transport, supported release
and 0.15s rest during `release_high`. Final object/plate XY distances were
**0.03723m / 0.03973m**, close to the fixed 0.04m limit; broad reliability is not
established by two development resets. The selected forward offset shifts every
phase by +0.015m in base X relative to the existing early-release policy, retaining
its orientation, closure, phase durations and opening ramp0.08.

The height variants produced **identical requested-action arrays** to the original
controller before failure because the fixed-duration motions remained saturated.
Thus this probe did not effectively test a reached change in grasp height. Target
parameter changes alone are insufficient evidence that the executed grasp pose
changed. Rear-offset episodes never satisfied contact-lift scoring despite moving
the object; these are failures, not demonstrations.

Validated all **4,233 executed transitions**: T+1 RGB/state/mask/timestamps for T
requested/applied/raw actions, 86 measured state entries, 14D normalized actions,
50ms spacing, finite signals and unchanged runtime hashes. Rejected commands
remain only in trace records. Plan SHA-256:
`7303cc2866f0ee700d9852531059903de2566f550106f67dd6388ef5fafb5d3d`.
Report SHA-256:
`c51a6536f553fca994f026cb1f29e2f67378abe3fff92122fa2f7bfdb1a0823f`.

For a new explicitly task-specific corpus, start with this successful grasp
offset, retain nominal and perturbed-action attempts including failures, preserve
whole-session splits, and extract image waypoints only from successful training
episodes. Existing Apple→Plate holdout data and historical claims stay unchanged.
The small landing margin motivates a separately declared release-target probe.

## V1 landing refinement — preregistered, not yet executed

The two successful V0 apples land about +0.036m/+0.039m in X from plate center.
Keep the successful forward grasp exactly unchanged and compare additional base-X
shifts **0, −0.02, −0.035m** on transfer/release/lower/retreat targets only. Test all
three settings, in that order, on each new reset seed **41100 and 41101**, using the
same ±0.004m jitter rule. This is six fixed attempts, no retries or physics/scorer
changes. Resource cap: **60 true-wall and 60 process CPU seconds**, command cutoff
55s with5s reserved for finalization; retain incomplete attempts.

Select a setting for later collection by most observed successes, then greatest
mean final plate-radius margin among its successful attempts; an exact tie retains
the earlier declared setting. Require at least one successful attempt; no successes
means no selected setting. This selection is development-only and is not a new
reliability estimate. Future collection/evaluation seeds are not used here.

Execute only after the coordinator's independent feasibility comparison finishes:

```sh
PYTHONPATH=src .venv/bin/python scripts/apple_mechanics_probe.py \
  --refine-landing --output outputs/apple-mechanics-v1
```

## V1 landing results

All six attempts succeeded under the unchanged ordered scorer in **37.52
true-wall seconds / 34.50 process CPU seconds**, with no retries or limit changes.

| Additional transfer X shift | Seed 41100 steps / final XY error | Seed 41101 steps / final XY error | Mean success margin |
|---|---|---|---|
| 0 m | 512 / 0.03605 m | 513 / 0.03702 m | 0.00347 m |
| −0.02 m | 488 / 0.03165 m | 488 / 0.03176 m | 0.00829 m |
| **−0.035 m** | **501 / 0.02883 m** | **501 / 0.02885 m** | **0.01116 m** |

The frozen selection rule therefore chooses **−0.035 m**. Keep palm grasp X offset
−0.015 m; the selected transfer/release/lower/retreat targets have net X offset
−0.050 m relative to the container. This is an empirically selected development
controller, with only two resets per setting. It justifies a prospective data
collection attempt, not a reliability claim or a learned manipulation result.

Machine-readable report: `outputs/apple-mechanics-v1/report.json`, SHA-256
`4463e9557fd3ca5c1083534b6792a9675aebe5f0d29e9b69562881a841130b8f`.
Original V0 outputs and immutable runtime snapshots remain unchanged.
