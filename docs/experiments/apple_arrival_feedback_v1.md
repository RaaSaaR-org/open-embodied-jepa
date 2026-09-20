# Apple arrival-feedback audit v1 (prospective, TASK040)

Hypothesis: a four-command commitment can continue moving after measured arrival
at an image waypoint, losing the three-observation dwell needed for progression.
The sealed TASK039 learned reset 43000 entered goal2's threshold once, before
command60: distance 0.0583199933, threshold 0.0609087203, dwell1, cached offset2 of
plan58. After that command, distance rose to 0.0622814149 and dwell reset. This is
one observed feedback opportunity, not evidence of a general cure or causal proof.

This diagnostic is specified before execution. No runtime, model, task threshold,
training corpus, proposal distribution or final TEST cohort changes are allowed.
There is no forced hold, scripted motion, privileged-state inference or retraining.
All observations and actions are development data.

## Inputs and exact comparison

Use the sealed `outputs/apple-control-commitment-v1/attempts/43000-learned` trace:
SHA256 `f59e7eacfd22cfb1ff51cd0e85a127e6e91525dd5fe39e8303f5358506c2f90c`,
exactly 1,000 acknowledged commands. Use its immutable `source/` runtime and resolved
plan/waypoints, checkpoint SHA256
`0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`, dataset SHA256
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
The parent registers source/runtime/helper/script/protocol/input hashes and
Python/package/platform versions before spawning; recheck them after execution.
Do not decode TEST images. Existing TRAIN-derived image goals remain unchanged.

Replay accepted commands 0–59 once, rerunning the frozen controller and checking
observations' timestamps, image-goal identity/distance/dwell, candidate ranking,
selected sampled/projected/applied actions, commitment provenance and all ordered
score fields against the saved trace. Root60's original decision is also checked
without execution, then its controller state restored. Mismatches abort rather
than adjust tolerances (existing audit tolerance 1e-6 absolute /1e-5 relative,
timestamps 1e-9 absolute).

Capture full MuJoCo integration/contact/warm-start data, actuator targets, grasp
state, stop reason, scorer stage/dwell state, and controller RNG/warm/commitment/
pending/image-goal state. Restore these identically before each sibling and
require identical measured RGB, state/mask and observation timestamp. Reuse the
reviewed forecast-audit helpers rather than approximate warm state from actions.

Exactly two siblings, in this frozen order, each with 16 ordinary command attempts:

1. **Original**: unmodified frozen four-command controller. Verify its continuation
   also matches saved steps60–75, including task scores.
2. **Arrival interrupt**: before EVERY ordinary `step()`, measure image distance to
   the CURRENT goal. If distance is at or below its unchanged threshold and an
   existing commitment is present, clear that commitment and record its originating
   plan/offset. Then call normal `step()` once. It alone updates dwell, selects
   actions and consumes planner RNG. Apply this same rule throughout all 16 calls,
   including after any goal advance; this is a feedback-rule test, not merely one
   root cache clear. A normal search may still choose motion or hold. No action is
   forced. RNG divergence after the additional search is part of the intervention.

Use all existing projection/execute guards and the same five-second control
planning deadline, including the additional image measurement. Rejects terminate
a sibling. Preserve its accepted prefix, rejected command and missing slots.
A budget/error interruption preserves both planned sibling records and all 32
intended command slots, with incomplete/unavailable outcomes explicit. No retries.

## Outcomes and resource bounds

Primary outcome: normal image dwell completes and the controller advances past
goal2 during the 16 decision calls. Report first advance command, maximum goal2
dwell, per-command distance/threshold, all cache clears and selected origins,
requested/projected/applied actions, all physical stage flags and task success.
Do not substitute task reach, a single threshold crossing or image distance for
dwell completion. The last action's later effect is not silently treated as a
17th decision; endpoint measured distance may be reported separately if available.
An interrupted prefix with no observed advance is unavailable, not a completed
negative outcome. Stage truth is scoring-only and never supplied to control.

Hard cap: 60 true-wall seconds from script entry, 40 worker CPU seconds, four Torch
threads. Parent kills at57 seconds to reserve finalization; child stops new work
at53 wall or37 CPU seconds. Imports, source checks, replay, both siblings and report
writing count. Use maximum wall-clock and monotonic elapsed time so suspension
counts. No budget extensions. Report measured worker CPU and supervisor wall time.

Results describe one correlated development state, with the original continuation
as an exact replay check. Even a favorable result does not establish robust
closed-loop manipulation, a population success improvement or a final-MVP pass.
Write results separately without altering this preregistration.

After review, commit and authorization only:

```sh
.venv/bin/python scripts/audit_apple_arrival_feedback.py \
  --repository . \
  --original outputs/apple-control-commitment-v1 \
  --output outputs/apple-arrival-feedback-v1
```
