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
import pytest
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


@pytest.mark.parametrize("factory", [ExactFakeModel, ActionBlindFakeModel])
def test_rollout_term_reproduces_the_published_gate_value(factory):
    """The split must be a split of G1 itself, not of a differently built cohort.

    Run against an action-blind model as well as an exact one: with exact dynamics both
    sides of the rollout comparison are ~0, so only the action-blind case puts a
    non-trivial number on both sides of the equality.
    """
    module = _load()
    arrays, windows = _cohort()
    model = factory()
    gate = wm.window_metrics(model, arrays, windows, SCHEMA)[str(wm.GATE_HORIZON)]
    split = module.decompose(model, arrays, windows, SCHEMA)
    assert split["moving_windows"] == gate["palm_apple_moving_windows"]
    assert split["valid_windows"] == gate["valid_windows"]
    assert split["rollout_median_m"] == gate["palm_apple_moving_median_m"]
    assert split["persistence_median_m"] == gate["palm_apple_moving_persistence_median_m"]
    assert (
        split["true_displacement_median_m"] == gate["palm_apple_moving_true_displacement_median_m"]
    )


def test_windows_with_a_dropped_apple_at_the_target_leave_the_cohort():
    """The target-frame dropped-apple exclusion is the evaluator's, and must be kept.

    On the real corpus it removes 846 of 4,807 windows, so a split that forgot it would
    score a different cohort from the gate while still looking plausible.
    """
    module = _load()
    arrays, windows = _cohort()
    before = module.decompose(ExactFakeModel(), arrays, windows, SCHEMA)
    starts = arrays.offsets[windows[:, 0]] + windows[:, 1]
    dropped = arrays.targets["apple_dropped"].copy()
    dropped[starts[: len(starts) // 2] + wm.GATE_HORIZON] = 1.0
    arrays.targets["apple_dropped"] = dropped
    after = module.decompose(ExactFakeModel(), arrays, windows, SCHEMA)
    assert after["valid_windows"] < before["valid_windows"]
    assert after["moving_windows"] < before["moving_windows"]
    # And the evaluator agrees on the reduced cohort, which is the property that matters.
    gate = wm.window_metrics(ExactFakeModel(), arrays, windows, SCHEMA)[str(wm.GATE_HORIZON)]
    assert after["valid_windows"] == gate["valid_windows"]
    assert after["moving_windows"] == gate["palm_apple_moving_windows"]


def test_exact_dynamics_leave_no_rollout_excess():
    """With exact dynamics the whole error is the encoder's, so the excess vanishes.

    The tolerance is float accumulation over an 8-step rollout, not a modelling error:
    the residual is nanometres against a gate in centimetres.
    """
    module = _load()
    arrays, windows = _cohort()
    split = module.decompose(ExactFakeModel(), arrays, windows, SCHEMA)
    assert split["encoded_target_median_m"] == 0.0
    assert abs(split["rollout_median_m"]) < 1e-6
    assert abs(split["rollout_excess_median_m"]) < 1e-6


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
