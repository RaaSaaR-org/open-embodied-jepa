"""Encoder/rollout split of the palm-apple gate error (NumPy only; no torch, no corpus).

The decomposition's job is to attribute a failed G1 between the encoder and the
prediction step. Two properties make it trustworthy, and both are asserted here on the
same fake models the v2/v3 metric tests use:

1. its ``rollout`` term must equal the gate's own ``palm_apple_moving_median_m`` on the
   identical cohort -- otherwise the split is not a split of the published number;
2. a model whose dynamics are exact must leave no rollout excess, so any excess it
   reports is genuinely the prediction step and not an artefact of the cohort.

None of this is a learned result.
"""

import importlib.util
from pathlib import Path

import numpy as np
from test_world_model_v2 import SCHEMA, ActionBlindFakeModel, ExactFakeModel, fixture_arrays

import embodied_jepa.world_model_v2 as wm

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "decompose_wm_v3_readout", ROOT / "scripts/decompose_wm_v3_readout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cohort():
    arrays = fixture_arrays()
    return arrays, wm.window_starts(arrays, max(wm.HORIZONS), stride=4)


def test_rollout_term_reproduces_the_published_gate_value():
    """The split must be a split of G1 itself, not of a differently built cohort."""
    module = _load()
    arrays, windows = _cohort()
    model = ExactFakeModel()
    gate = wm.window_metrics(model, arrays, windows, SCHEMA)[str(wm.GATE_HORIZON)]
    split = module.decompose(model, arrays, windows, SCHEMA)
    assert split["moving_windows"] == gate["palm_apple_moving_windows"]
    assert split["valid_windows"] == gate["valid_windows"]
    assert split["rollout_median_m"] == gate["palm_apple_moving_median_m"]
    assert split["persistence_median_m"] == gate["palm_apple_moving_persistence_median_m"]
    assert (
        split["true_displacement_median_m"]
        == (gate["palm_apple_moving_true_displacement_median_m"])
    )


def test_exact_dynamics_leave_no_rollout_excess():
    """With exact dynamics the whole error is the encoder's, so the excess vanishes.

    The tolerance is float accumulation over an 8-step rollout, not a modelling error:
    the residual is nanometres against a gate in centimetres.
    """
    module = _load()
    arrays, windows = _cohort()
    split = module.decompose(ExactFakeModel(), arrays, windows, SCHEMA)
    assert split["encoded_target_median_m"] == 0.0
    assert split["rollout_median_m"] < 1e-6
    assert split["rollout_excess_median_m"] < 1e-6


def test_action_blind_dynamics_move_the_error_into_the_rollout():
    """An action-blind predictor keeps a perfect encoder but a poor rollout."""
    module = _load()
    arrays, windows = _cohort()
    split = module.decompose(ActionBlindFakeModel(), arrays, windows, SCHEMA)
    # The encoder is still exact: a directly encoded observation reads out perfectly.
    assert split["encoded_start_median_m"] == 0.0
    assert split["encoded_target_median_m"] == 0.0
    # All of the error is therefore the prediction step's.
    assert split["rollout_median_m"] > 0.0
    assert np.isclose(split["rollout_excess_median_m"], split["rollout_median_m"])
    assert split["encoder_share_of_rollout"] == 0.0
