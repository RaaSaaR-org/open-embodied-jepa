"""World model v3 candidate-ranking metric and gate set (NumPy only; no torch, no corpus).

The same exact/action-blind fake models as the v2 tests drive the new metric: a model
whose dynamics are exact must order sibling candidates perfectly and lose no true cost
by taking its own best, while an action-blind model must not. None of this is a learned
result.
"""

import numpy as np
import pytest
from test_world_model_v2 import (
    SCHEMA,
    ActionBlindFakeModel,
    ExactFakeModel,
    fixture_arrays,
)

import embodied_jepa.world_model_v2 as wm
from embodied_jepa import world_model_v3 as v3

GATES = {
    "G1_palm_apple_moving_h8_median_m": 0.015,
    "G2_max_ratio_to_control": 0.8,
    "G3_apple_plate_h8_median_m": 0.02,
    "G4_apple_height_grasp_h8_median_abs_m": 0.01,
    "G5_held_auroc_min": 0.85,
    "G5_minimum_per_class": 20,
    "G6_minimum_ranked_groups": 1,
    "G6a_within_state_spearman_min": 0.5,
    "G6b_top1_regret_max_m": 0.004,
    "G7_minimum_pairs": 1,
    "G7a_own_beats_swapped_min": 0.7,
    "G7b_divergence_spearman_min": 0.5,
    "G8_max_collapsed_fraction": 0.05,
    "G8_min_effective_rank": 4.0,
    "G8_min_latent_std_mean": 0.1,
}


def test_normalized_ranks_span_the_unit_interval_and_average_ties():
    np.testing.assert_allclose(v3._normalized_ranks([5.0, 1.0, 3.0]), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(v3._normalized_ranks([2.0, 2.0, 9.0]), [0.25, 0.25, 1.0])
    np.testing.assert_allclose(v3._normalized_ranks([7.0, 7.0]), [0.5, 0.5])


def test_exact_dynamics_rank_sibling_candidates_and_lose_no_true_cost():
    arrays = fixture_arrays()
    ranking = v3.candidate_ranking_metrics(ExactFakeModel(), arrays, SCHEMA)
    gate = ranking[str(v3.RANKING_GATE_HORIZON)]
    assert gate["groups_ranked"] >= 1
    assert gate["candidates_median"] >= v3.RANKING_MIN_CANDIDATES
    assert gate["within_state_spearman_pooled"] == pytest.approx(1.0)
    assert gate["within_state_spearman_mean"] == pytest.approx(1.0)
    assert gate["top1_regret_median_m"] == pytest.approx(0.0, abs=1e-9)
    assert gate["top1_regret_normalized_median"] == pytest.approx(0.0, abs=1e-9)
    # The label-only random-choice baseline is a property of the cohort, not the model,
    # so a perfect model must still beat it.
    assert gate["random_choice_regret_median_m"] > gate["top1_regret_median_m"]
    # Judging a candidate by a sibling's actions must not order it as well.
    assert gate["shuffled_within_state_spearman_pooled"] < gate["within_state_spearman_pooled"]
    assert gate["shuffled_top1_regret_median_m"] >= gate["top1_regret_median_m"]


def test_action_blind_model_cannot_rank_candidates_from_one_state():
    """Every candidate leaves the same state, so an action-blind model predicts the same
    cost for all of them: no ranking exists and the gate cannot pass."""
    arrays = fixture_arrays()
    ranking = v3.candidate_ranking_metrics(ActionBlindFakeModel(), arrays, SCHEMA)
    gate = ranking[str(v3.RANKING_GATE_HORIZON)]
    assert gate["groups_ranked"] >= 1
    assert gate["within_state_spearman_pooled"] is None  # constant predictions: undefined
    assert gate["top1_regret_median_m"] > 0.0


def test_v3_gates_apply_the_frozen_thresholds_and_replace_the_calibration_gate():
    arrays = fixture_arrays()
    model = ExactFakeModel()
    windows = wm.window_starts(arrays, 16, stride=4)
    metrics = {
        "windows": wm.window_metrics(model, arrays, windows, SCHEMA),
        "siblings": wm.sibling_metrics(model, arrays, SCHEMA),
        "collapse": model.latent_statistics(None),
        **v3.ranking_metrics(model, arrays, SCHEMA),
    }
    result = v3.evaluate_gates(metrics, GATES)
    assert "G6_approach_cost_calibration_h8" not in result
    assert result["G6a_within_state_ranking_spearman_h16"]["passed"]
    assert result["G6b_top1_regret_h16"]["passed"]
    assert result["G1_palm_apple_moving_h8"]["passed"]
    # v2's absolute calibration number survives as a descriptive measurement.
    assert metrics["windows"]["8"]["approach_cost_median_abs_log_ratio"] is not None

    blind = ActionBlindFakeModel()
    blind_metrics = {
        "windows": wm.window_metrics(blind, arrays, windows, SCHEMA),
        "siblings": wm.sibling_metrics(blind, arrays, SCHEMA),
        "collapse": blind.latent_statistics(None),
        **v3.ranking_metrics(blind, arrays, SCHEMA),
    }
    blind_result = v3.evaluate_gates(blind_metrics, GATES)
    assert not blind_result["all_passed"]
    assert not blind_result["G6a_within_state_ranking_spearman_h16"]["passed"]


def test_too_few_ranked_groups_fail_the_ranking_gates():
    arrays = fixture_arrays()
    model = ExactFakeModel()
    metrics = v3.ranking_metrics(model, arrays, SCHEMA)
    metrics["ranking"][str(v3.RANKING_GATE_HORIZON)]["groups_ranked"] = 0
    windows = wm.window_starts(arrays, 16, stride=4)
    metrics |= {
        "windows": wm.window_metrics(model, arrays, windows, SCHEMA),
        "siblings": wm.sibling_metrics(model, arrays, SCHEMA),
        "collapse": model.latent_statistics(None),
    }
    result = v3.evaluate_gates(metrics, GATES | {"G6_minimum_ranked_groups": 1})
    assert not result["G6a_within_state_ranking_spearman_h16"]["passed"]
    assert not result["G6b_top1_regret_h16"]["passed"]
