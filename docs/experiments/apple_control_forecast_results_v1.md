# Apple control forecast audit v1: local progress with prediction and feasibility limits

The single preregistered audit completed in **55.455 wall seconds / 54.800 child
CPU seconds**, within its 180/120-second limits, with exit code 0. It matched
**550 original command executions and 551 planning steps** against the preserved
trace. All nine declared branches ran: eight completed H16; the root-550 winner
was rejected at command 10 by the unchanged right joint-rate guard after nine
accepted commands. This is a completed diagnostic with one rejected sequence,
not nine successful robot tasks.

The [protocol](apple_control_forecast_audit_v1.md) used roots 119, 300, and 550 of
the original interrupted learned run. Every branch targeted TRAIN frame 140
(goal index 5, tolerance 0.063722). Each root's full MuJoCo/scorer/controller state
was restored before its winning, best demonstration, and hold sequence. Restored
RGB, proprioception, masks, and timestamps matched exactly. Frozen input checks
passed after execution; no TEST images were decoded, fitting occurred, or final
cohort was run. No retries or changes to the declared audit were made.

## Predicted and measured image distance

Distances use the frozen normalized image metric; lower is better. H1 is 0.05 s;
H16 is 0.8 s. Each cell reports **predicted / measured**. Missing H16 remains
unavailable rather than being padded with the accepted prefix.

| Root | Candidate | Initial | H1 predicted / measured | H16 predicted / measured | Accepted / status |
|---|---|---:|---:|---:|---|
| 119 | winner | 0.30559 | 0.30908 / 0.28170 | 0.24092 / 0.09823 | 16 / completed |
| 119 | demonstration | 0.30559 | 0.30911 / 0.28558 | 0.24123 / 0.10592 | 16 / completed |
| 119 | hold | 0.30559 | 0.30853 / 0.28558 | 0.29535 / 0.28376 | 16 / completed |
| 300 | winner | 0.25590 | 0.23710 / 0.29472 | 0.15524 / 0.11486 | 16 / completed |
| 300 | demonstration | 0.25590 | 0.23982 / 0.28819 | 0.17424 / 0.20727 | 16 / completed |
| 300 | hold | 0.25590 | 0.23576 / 0.29405 | 0.17883 / 0.30067 | 16 / completed |
| 550 | winner | 0.32483 | 0.31165 / 0.31371 | Unavailable | 9 / command_rejected |
| 550 | demonstration | 0.32483 | 0.32010 / 0.31514 | 0.27165 / 0.27904 | 16 / completed |
| 550 | hold | 0.32483 | 0.31624 / 0.38738 | 0.28502 / 0.40072 | 16 / completed |

At roots 119 and 300, the model-selected winner had the lowest measured H16
image cost among these three candidates; the demonstration was second and hold
last. Neither winner reached the image-goal tolerance. This evidence does not
support a blanket claim that the model ranks every local action incorrectly.
It also does not establish that repeated replanning will complete the task.

Hold forecasts at roots 300 and 550 predicted improvement, while measured
image distance worsened (0.25590→0.30067 and 0.32483→0.40072). At root 550,
the selected winner's H16 surrogate feasibility did not survive open-loop
execution: command 10 hit the joint-rate guard. First-action projection
acceptance is therefore not a guarantee for all later commands under actual
tracking. No guard was relaxed and no object was teleported or welded.

All **137 accepted commands exactly equaled the scored projected actions**.
Forecasting their recorded applied sequence produced identical H1/H16 metrics to
forecasting the scored sequence wherever an endpoint existed. Action relabeling
therefore does not explain the measured forecast discrepancies in these accepted
prefixes. This does not separate model bias from servo behavior or unobserved
state, and it says nothing about the rejected winner's unobserved H16 endpoint.

## Sensor prediction errors

Each cell lists **RGB-grid MSE / qpos MSE (rad²) / qvel MSE (rad²/s²)**.
RGB uses the model's resized grid on the [0,1] scale, undoing normalization
without clipping predictions. The raw image error and normalized goal distance
are different metrics. Exact values and both action-label forecasts remain in
the saved report; the compact manifest preserves every available H1/H16 metric.

| Root | Candidate | H1 RGB / qpos / qvel | H16 RGB / qpos / qvel |
|---|---|---:|---:|
| 119 | winner | 5.822e-05 / 1.951e-06 / 0.009949 | 0.002041 / 0.0008109 / 0.007785 |
| 119 | demonstration | 4.408e-05 / 2.637e-06 / 0.007295 | 0.003021 / 0.0006904 / 0.009403 |
| 119 | hold | 4.351e-05 / 2.591e-06 / 0.007298 | 0.000228 / 0.0002557 / 0.006476 |
| 300 | winner | 6.55e-05 / 2.827e-05 / 0.01841 | 0.001118 / 0.01253 / 0.03754 |
| 300 | demonstration | 5.954e-05 / 4.054e-05 / 0.008895 | 0.0004742 / 0.01087 / 0.08803 |
| 300 | hold | 6.933e-05 / 4.612e-05 / 0.01081 | 0.0003739 / 0.01061 / 0.06032 |
| 550 | winner | 2.558e-05 / 7.177e-05 / 0.01238 | Unavailable |
| 550 | demonstration | 2.123e-05 / 6.35e-05 / 0.01175 | 0.0007965 / 0.03722 / 0.01099 |
| 550 | hold | 6.352e-05 / 7.244e-05 / 0.0121 | 0.0008788 / 0.03633 / 0.007718 |

No branch reached grasp, transport, placement, or release. All three root-550
branches reached the scorer's reach stage; roots 119 and 300 reached none.
These are short branches from three correlated development states, so zero
placements is not a full-task reliability estimate.

## Reproduction and limits

Audit source revision: `c86a076a81f0fe97c550f421ad4d2d453f52debd`.
Execution imported the original captured runtime under
`outputs/apple-control-development-v2/source`, including its original action
manifest, calibration, and checkpoint. The previous control run and all source
corpus payloads remained unchanged. Per-step replay checks covered goal identity,
dwell, timestamps, candidate costs/selection, requested/projected/applied actions,
and ordered task scoring at their preregistered tolerances. Post-run identities
and payload verification passed. Maximum replay planning time was 0.20941 s.

The [compact manifest](../../benchmarks/manifests/apple-control-forecast-audit-v1.json)
binds configuration, environment, protocol/script/checkpoint/data hashes,
all nine statuses, and endpoint metrics. Local artifacts are in
`outputs/apple-control-forecast-audit-v1/`: original report/plan, root integration
and controller snapshots, projected/applied sequences, predictions, measured
frames, scores, and a separate `artifact_hashes.json` inventory.

This audit narrows the stall diagnosis to a controller/model/physical-response
mismatch despite some useful local ranking. It does not identify a unique cause,
authorize automatic tuning, establish closed-loop improvement, or satisfy the
working apple pick-and-place gate.
