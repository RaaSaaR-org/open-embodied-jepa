"""World model v4 encoder/rollout decomposition and the G9 gate (NumPy only).

The same exact and action-blind fake models as the v2/v3 tests drive the new
measurement. None of this is a learned result: the fakes have no weights.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_world_model_v2 import (
    SCHEMA,
    ActionBlindFakeModel,
    ExactFakeModel,
    fixture_arrays,
)
from test_world_model_v3 import GATES as V3_GATES

import embodied_jepa.world_model_v2 as wm
from embodied_jepa import world_model_v4 as v4
from embodied_jepa.contracts import ContractError

GATES = V3_GATES | {"G9_rollout_excess_h8_median_m": 0.0053}
ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_wm_v4_contrasts", ROOT / "scripts/bootstrap_wm_v4_contrasts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _windows(arrays):
    return wm.window_starts(arrays, max(wm.HORIZONS), stride=4)


def test_exact_dynamics_leave_no_rollout_excess():
    arrays = fixture_arrays()
    windows = _windows(arrays)
    result = v4.decompose(ExactFakeModel(), arrays, windows, SCHEMA)
    assert result["moving_windows"] > 0
    # float32 accumulation over the fixture's cumulative-sum dynamics, not model error.
    assert result["encoded_target_median_m"] == pytest.approx(0.0, abs=1e-7)
    assert result["rollout_median_m"] == pytest.approx(0.0, abs=1e-7)
    assert result["rollout_excess_median_m"] == pytest.approx(0.0, abs=1e-7)


def test_an_action_blind_rollout_pays_the_whole_displacement_as_excess():
    """A do-nothing predictor does not get a small excess: it gets the persistence gap."""
    arrays = fixture_arrays()
    windows = _windows(arrays)
    result = v4.decompose(ActionBlindFakeModel(), arrays, windows, SCHEMA)
    # The fake encoder is exact, so encoded_target is 0 and the excess is the whole
    # rollout error, which for a frozen prediction is the persistence error.
    assert result["encoded_target_median_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["rollout_median_m"] == pytest.approx(result["persistence_median_m"])
    assert result["rollout_excess_median_m"] == pytest.approx(result["rollout_median_m"])
    assert result["rollout_excess_median_m"] > wm.MOVING_THRESHOLD_M


def test_the_decomposition_runs_on_the_gate_cohort_and_reproduces_its_numbers():
    arrays = fixture_arrays()
    windows = _windows(arrays)
    model = ActionBlindFakeModel()
    gated = wm.window_metrics(model, arrays, windows, SCHEMA)[str(wm.GATE_HORIZON)]
    split = v4.decompose(model, arrays, windows, SCHEMA)
    assert split["rollout_median_m"] == gated["palm_apple_moving_median_m"]
    assert split["persistence_median_m"] == gated["palm_apple_moving_persistence_median_m"]
    assert split["moving_windows"] == gated["palm_apple_moving_windows"]


def _metrics(model, arrays):
    windows = _windows(arrays)
    return {
        "windows": wm.window_metrics(model, arrays, windows, SCHEMA),
        "siblings": wm.sibling_metrics(model, arrays, SCHEMA),
        "collapse": wm.collapse_metrics(model, arrays, SCHEMA),
    } | v4.extra_metrics()(model, arrays, SCHEMA)


def test_g9_is_reported_alongside_the_carried_over_v3_gates():
    arrays = fixture_arrays()
    exact = v4.evaluate_gates(_metrics(ExactFakeModel(), arrays), GATES)
    blind = v4.evaluate_gates(_metrics(ActionBlindFakeModel(), arrays), GATES)
    # Every v3 gate name survives verbatim, with its threshold untouched.
    for name, entry in exact.items():
        if name == "all_passed" or name == "G9_rollout_excess_h8":
            continue
        assert entry["threshold"] == blind[name]["threshold"]
    assert exact["G9_rollout_excess_h8"]["threshold"] == GATES["G9_rollout_excess_h8_median_m"]
    assert exact["G9_rollout_excess_h8"]["passed"] is True
    # The action-blind model fails G9 *and* the action-sensitivity gates it is read with.
    assert blind["G9_rollout_excess_h8"]["passed"] is False
    assert blind["G2a_vs_persistence_h8"]["passed"] is False
    assert blind["all_passed"] is False


def test_a_decomposition_from_another_cohort_is_refused():
    arrays = fixture_arrays()
    metrics = _metrics(ExactFakeModel(), arrays)
    metrics["decomposition"] = dict(metrics["decomposition"], rollout_median_m=0.123)
    with pytest.raises(ContractError, match="same cohort"):
        v4.evaluate_gates(metrics, GATES)


def test_the_bootstrap_pairs_arms_and_refuses_mismatched_cohorts():
    paired_intervals = _bootstrap_module().paired_intervals

    rng = np.random.default_rng(7)
    moving = np.zeros(40, bool)
    moving[::2] = True
    episode = np.repeat(np.arange(8), 5)
    base = {
        "moving": moving,
        "episode": episode,
        "rollout": rng.uniform(0.02, 0.05, 40),
        "encoded_target": rng.uniform(0.01, 0.03, 40),
    }
    errors = {arm: dict(base) for arm in ("E0", "E1", "E2", "E3")}
    errors["E1"] = dict(base, rollout=base["rollout"] - 0.01)
    count, clusters, contrasts = paired_intervals(errors, seed=1, resamples=200)
    assert count == 20 and clusters == 8
    entry = contrasts["E1_minus_E0_action_chunk_rollout"]
    assert entry["point_m"] == pytest.approx(-0.01)
    # Both designs are reported and both bracket the point estimate.
    for design in ("iid", "cluster"):
        low, high = entry[design]["ci95_m"]
        assert low <= entry["point_m"] <= high

    errors["E2"] = dict(base, moving=~moving)
    with pytest.raises(ContractError, match="different moving cohort"):
        paired_intervals(errors, seed=1, resamples=10)
