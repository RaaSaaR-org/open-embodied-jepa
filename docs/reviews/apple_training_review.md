# Independent review: sensor-model training integration

Reviewed `src/embodied_jepa/training.py`, `tests/test_training.py` and the initial
apple sensor training protocol. Scope includes the normalization integration in
`3fc4288` and subsequent working changes after `8ee7e45` for failure recording and
metric-definition propagation. The reviewer authored `sensor.py`, but did not
author these runner changes. This review does not independently certify the
sensor model's scientific effectiveness or constitute external maintainer approval.

## Result

No blocking defect found in the reviewed integration. The optional normalization
hook runs before initial validation, receives only sealed training episode IDs,
and streams each training transition once in chunks bounded by 64 and model
horizon. Train/validation/test session isolation remains enforced by EpisodeCache.
Adjacent chunks repeat their boundary observation in the feature moments; this is
documented in the model and is not test leakage. There is no claimed equal
weighting of every unique frame.

An unfitted sensor model refuses checkpoint saving. The added finalization guard
now records that secondary checkpoint error while preserving the original
normalization failure and writing the run report. It does not fabricate a latest
checkpoint. Selected sensor metrics retain definition version 3 in validation and
selection records; the historical JEPA collapse selector still rejects unsupported
metric definitions instead of applying learned-latent collapse thresholds to
fixed RGB features.

The raw-MSE selector can retain an untrained baseline. That behavior is disclosed;
passing a completed training run is not equivalent to passing the later
persistence, shuffled-action or physical-control gates. Prediction and control
failure must remain explicit even if all requested updates finish.

## Checks and remaining integration evidence

`PYTHONPATH=src .venv/bin/python -m pytest -q
tests/test_waypoint_planning.py tests/test_training.py` passed 25 tests in 2.78 s:
14 runner tests and 11 waypoint tests. Runner checks cover split exclusion,
normalization transition coverage, validation cohort determinism, budget failures,
original-error preservation and best/latest records.

The normalization test currently uses a NativeJEPA stub hook. A real `sensor_wm`
runner smoke remains necessary before the long attempt; the coordinator is
preparing that integration check. Unit tests on the actual sensor backend already
exercise train-only fitting, learning, save/load and MPS separately.

One protocol wording correction was sent to the coordinator: uniform admissible
window sampling does not give every transition equal multiplicity. Near-boundary
transitions occur in fewer windows. The protocol should describe the implemented
window sampling and retain phase-coverage limitations, without implying a phase
balancing sampler exists.

The proposed 1,800-second cap, fixed 3,000 updates/H8/batch16/CPU4, training-only
normalization, immutable corpus and fresh later evaluation are a reviewable first
attempt. This review has not run that attempt or established successful apple
control. Any later architecture, corpus or selector change needs a new declared
development experiment rather than rewriting this attempt's result.
