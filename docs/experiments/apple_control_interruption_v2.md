# Control v2: preserved user interruption

The first registered control-v2 comparison was stopped at the user's explicit
pause request. It is **incomplete**, with one partial attempt and five unstarted
attempts. It is not a completed 0/6 performance estimate or a successful control
evaluation. The original six-attempt plan, source snapshot, calibration and trace
remain unchanged under `outputs/apple-control-development-v2/`.

Read-only audit: `outputs/apple-control-development-v2-interruption-audit.json`,
SHA-256 `10d1728bf14096129e94f3c1e9c20a4b7bb7f5f0214dbf445a3d938a7d373f7b`.
Source revision was `1b109833ee1518c67a08276e8590daae39ccf6a0`; selected checkpoint
SHA-256 was `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`.
The launch record had confirmed the unchanged primary causal gate before this
run. Passing that gate did not establish manipulation success.

| Planned attempt | Status | Confirmed applied commands | Last recorded ordered stages |
|---|---|---:|---|
| 43000 learned | Interrupted by user pause | 574 | Reach only |
| 43000 persistence | Not started | 0 | Unobserved |
| 43000 dynamics shuffle | Not started | 0 | Unobserved |
| 43001 learned | Not started | 0 | Unobserved |
| 43001 persistence | Not started | 0 | Unobserved |
| 43001 dynamics shuffle | Not started | 0 | Unobserved |

The trace contains exactly 1,148 complete JSON lines: 574 `command_pending`
events and 574 matching `result` acknowledgements, with contiguous command indices
0–573. No malformed final line, recorded error or unmatched pending command exists.
Every acknowledged applied action exactly equals its projected/scored first action.
The last pending command is step 573 and has a matching `applied` acknowledgement
at simulation time 28.699999999994972 seconds. No additional command is inferred.
The trace SHA-256 is
`921eacf54ca7cb131ef5fe1abb7f99e32af11bb5423dc56fae459664286df4c2`.

Reach first became true at step 550. Grasp, transport, place, release and success
never became true in the saved prefix. The last active image goal was index 5,
`apple-42000/frame-140`; 455 commands were spent at that goal. The last image
distance was 0.4431644 versus threshold 0.06372191. Proposal origins were 314
perturbations, 141 warm/search-best, 94 demonstration proposals and 25 holds.
These are descriptive partial-run observations; they do not justify tuning this
frozen configuration or estimating causal improvement over unrun controls.

Calibration preserved 19 goals, with 10 goals having an adjacent-image overlap.
Recorded observe/plan durations sum to 50.2375 seconds; the maximum is 0.2309
seconds. These are not total experiment wall time. Neither an attempt completion
report nor top-level supervisor report was written before termination, so exact
total wall duration and final evaluator integrity status are unavailable.

## Termination and provenance evidence

The pause termination tool **did execute successfully**. Its saved result reports
SIGTERM requested for evaluator supervisor PID 61247 (parent 61238) and worker
PID 61270 (parent 61247); neither remained present at its subsequent check, so no
SIGKILL was needed. On the resumed audit turn, polling the original exec session
95262 returned exit 241, consistent with the wrapper propagating child return
−SIGTERM. A fresh process listing found no matching evaluator process.

A separate usage-limit error was mentioned by the coordinator, but this agent's
available record does not include its timestamp. The completed termination tool
result establishes that termination ran; exact timing relative to that separate
error cannot be independently reconstructed here. No uncertain pending command
is present in the durable trace.

The retrospective read-only integrity audit found no mismatch between launch and
registered checkpoint/corpus/revision, stored source snapshot hashes, calibration/
waypoint hashes or any of the 566 registered corpus payload hashes. This confirms
the artifacts available at audit time; it does not fabricate the missing final
supervisor report. Raw files were only read. No physics, inference, training,
retry or continuation occurred during the interruption audit.

Any comparison after the user's resume request requires a separate prospective
launch record and new output directory. It must preserve this interrupted run,
its partial evidence and all six originally planned attempts. The fresh final
control cohort remains untouched.
