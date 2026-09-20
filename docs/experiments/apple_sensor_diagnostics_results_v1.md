# Apple sensor phase diagnostic results

The selected model predicts images better than persistence in all **24**
phase/horizon/split groups, but **does not show consistent benefit from the
correct actions**. True actions beat the within-phase shuffle in only **10/24**
groups, including **5/12 validation groups**. This is a negative result for the
action-ranking evidence needed by a world-model planner; persistence improvement
alone does not close that gap.

All groups used32 fixed windows, without replacement within each group. Different
groups may overlap frames; these are not768 independent trajectories. No model
fit, checkpoint substitution, test-image decoding or scoring-threshold change
occurred. The outer run completed in **4.206 seconds**, with worker time3.433
true-wall /3.568 process CPU seconds, inside the60 CPU/120 wall limits.

## Validation by phase

Prediction/persistence and shuffled advantage are computed within this fixed
sensor representation. Shuffled advantage is100×(shuffled MSE−prediction MSE) /
shuffled MSE; positive favors the actual actions. Tiny signs are descriptive,
not significance tests.

| Phase | H4 pred./persist. | H4 shuffle advantage | H8 pred./persist. | H8 shuffle advantage |
|---|---:|---:|---:|---:|
| orient | 0.171 | +0.153% | 0.089 | -0.484% |
| descend | 0.409 | -0.004% | 0.296 | -0.058% |
| close | 0.378 | -0.070% | 0.289 | -0.750% |
| lift | 0.147 | -0.908% | 0.218 | -0.107% |
| transfer | 0.150 | +0.369% | 0.095 | +0.854% |
| release_high | 0.699 | +0.097% | 0.581 | +0.016% |

Close and lift fail to benefit from correct actions at either validation horizon.
Even release, where the shuffle changes actions with normalized RMS0.117 atH4
and0.081 atH8, improves by only0.097% and0.016%. Most smaller motion contrasts
have RMS0.016–0.059. These interventions are limited to actions observed within
the same phase, so they cannot establish ranking of substantially different
counterfactual commands from the same starting state.

The full-frame consecutive RGB-motion proxy is largest during transfer
(validation RMS0.049–0.054 on[0,1] pixels) and smaller during descend/release
(about0.011–0.016). This does not isolate apple pixels or show whether contact is
visible: arm motion, occlusion and background dominate a whole-image statistic.
A nominal-trajectory shortcut remains plausible; the diagnostic does not prove
its internal mechanism.

## Provenance and interpretation

Protocol: [apple_sensor_diagnostics_v1.md](apple_sensor_diagnostics_v1.md).
Compact metrics for all24 groups, including raw MSE, proprioceptive error,
zero-action controls, cohort hashes and action contrasts:
[apple-sensor-phase-v1.json](../../benchmarks/manifests/apple-sensor-phase-v1.json).

Source revision `abed724a645c4bd35d7417853d30de2d3d3c9159`; selected checkpoint
step2100, SHA-256 `3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee`.
Dataset SHA-256 `d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df`.
Report SHA-256 `576febf4222de3e45a693bd08c51cbc4c569258644f0bb48f5c1897fa731d0c4`.

Decoded exactly26 TRAIN and3 VAL episodes; no TEST episode was decoded.
Post-run checks confirmed payload, source revision/Python, model implementation,
checkpoint, script and protocol hashes remained unchanged. Full cohort indices
and reports remain in local `outputs/apple-sensor-phase-v1/`.

The coordinator authorized a separate six-episode closed-loop **development
diagnostic** despite this weak action gate. That run cannot be called a passed
phase gate or a sealed final test. Keep this negative phase result regardless of
its physical outcome. The next data hypothesis should vary actions from matched
starting states rather than merely extending training on the same demonstrations.
