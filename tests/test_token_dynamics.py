"""TASK-066 design code (``token_dynamics``): rules, gates and rows on synthetic inputs.

Nothing here reads the corpus or trains a model; nothing is a learned-dynamics or control claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from embodied_jepa import latent_dynamics as ld
from embodied_jepa import token_dynamics as td
from embodied_jepa.contracts import ContractError


# ----- the latent --------------------------------------------------------------------------------
def test_the_latent_is_the_4x4_pooled_token_grid():
    assert td.TOKEN_GRID == 4 and td.TOKENS == 16 and td.LATENT_DIM == 6144
    assert td.MODEL_CONFIG["latent_dim"] == td.LATENT_DIM
    assert td.MODEL_CONFIG["token_grid"] == td.TOKEN_GRID
    assert td.MODEL_CONFIG["sigreg_weight"] == 0.0
    assert td.BACKEND == "leworldmodel"


def test_data_evaluation_and_sampler_are_task065s():
    assert td.DATASET_MANIFEST_SHA256 == ld.DATASET_MANIFEST_SHA256
    assert td.READ_SPLITS == ("train", "val")
    assert td.sampler_seed is ld.sampler_seed
    assert (td.EVAL_STRIDE, td.GATED_HORIZONS) == (ld.EVAL_STRIDE, ld.GATED_HORIZONS)
    for key in ld.THRESHOLDS:
        if not key.startswith("G1") or key == "G1_max_collapsed_fraction":
            assert td.THRESHOLDS[key] == ld.THRESHOLDS[key]


# ----- streamed collapse statistics ---------------------------------------------------------------
@pytest.mark.parametrize("n, d", [(200, 12), (9, 12)])
def test_moments_equal_task065s_collapse_statistics(n, d):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(d, d))
    pred = rng.normal(size=(n, d)) @ base * 0.3 + 50.0  # a large offset: the shift matters
    enc = rng.normal(size=(n, d)) @ base + 50.0
    want = ld.collapse_statistics(pred, enc)
    shift = np.full(d, 50.0)
    mp, me = td.Moments(shift), td.Moments(shift)
    for lo in range(0, n, 7):  # streamed in uneven chunks
        mp.add(pred[lo : lo + 7])
        me.add(enc[lo : lo + 7])
    got = td.collapse_statistics(mp, me)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def test_moments_merge_pools_two_halves_exactly():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(30, 5)), rng.normal(size=(40, 5)) + 2
    shift = np.zeros(5)
    ma, mb, mab = td.Moments(shift), td.Moments(shift), td.Moments(shift)
    ma.add(a)
    mb.add(b)
    mab.add(np.concatenate([a, b]))
    merged = ma.merge(mb)
    assert merged.n == 70
    np.testing.assert_allclose(merged.centered(), mab.centered(), rtol=1e-12)
    assert merged.effective_rank() == pytest.approx(ld.effective_rank(np.concatenate([a, b])))
    with pytest.raises(ContractError, match="same shift"):
        ma.merge(td.Moments(np.ones(5)))
    with pytest.raises(ContractError, match="width"):
        ma.add(np.zeros((2, 4)))
    with pytest.raises(ContractError, match="two rows"):
        td.Moments(shift).effective_rank()
    with pytest.raises(ContractError, match="same windows"):
        td.collapse_statistics(ma, mb)


def test_rank_cache_is_invalidated_by_new_rows():
    m = td.Moments(np.zeros(3))
    m.add(np.array([[1.0, 0, 0], [-1.0, 0, 0]]))
    assert m.effective_rank() == pytest.approx(1.0)
    m.add(np.array([[0.0, 1, 0], [0.0, -1, 0]]))
    assert m.effective_rank() == pytest.approx(2.0)


def test_a_collapsed_prediction_has_a_low_rank_ratio():
    rng = np.random.default_rng(2)
    enc = rng.normal(size=(500, 20))
    collapsed = np.outer(rng.normal(size=500), rng.normal(size=20))  # rank one
    me, mc = td.Moments(np.zeros(20)), td.Moments(np.zeros(20))
    me.add(enc)
    mc.add(collapsed)
    assert td.collapse_statistics(mc, me)["effective_rank_ratio"] < 0.1


# ----- calibration rules --------------------------------------------------------------------------
def test_saturation_update_is_the_first_point_within_one_percent():
    curve = [[500, 1.0], [1000, 0.6], [1500, 0.505], [2000, 0.5], [2500, 0.52]]
    assert td.saturation_update(curve) == 1500  # 0.505 <= 0.5 * 1.01
    assert td.saturation_update(curve, tolerance=0.0) == 2000
    assert td.saturation_update([[500, 0.3]]) == 500
    with pytest.raises(ContractError, match="finite"):
        td.saturation_update([[500, float("nan")]])
    with pytest.raises(ContractError, match="increasing"):
        td.saturation_update([[1000, 1.0], [500, 0.9]])
    with pytest.raises(ContractError):
        td.saturation_update([])


@pytest.mark.parametrize(
    "sats, updates, escalate",
    [
        ([1000, 2000], 10_000, False),  # floor
        ([4500, 6000], 15_000, False),  # 2 x 6000 = 12 000 -> 15 000
        ([10_000], 20_000, False),
        ([15_000], 30_000, False),  # exactly the ceiling
        ([15_500], 30_000, True),  # beyond it: escalate, never silently clipped into a freeze
    ],
)
def test_budget_rule(sats, updates, escalate):
    rule = td.budget_rule(sats)
    assert rule["updates"] == updates and rule["escalate"] is escalate
    assert rule["updates"] % td.BUDGET_STEP == 0


def test_budget_rule_refuses_nonsense():
    with pytest.raises(ContractError):
        td.budget_rule([])
    with pytest.raises(ContractError):
        td.budget_rule([0])


def test_bar_rule_is_half_the_smallest_pilot_ratio_rounded_down_never_below_its_floor():
    assert td.bar_rule([0.62, 0.58, 0.60], 0.10)["bar"] == 0.29
    assert td.bar_rule([0.40], 0.10)["bar"] == 0.20
    assert td.bar_rule([0.999], 0.10)["bar"] == 0.49
    floored = td.bar_rule([0.15], td.RANK_FLOOR)
    assert floored["relative"] == 0.07 and floored["bar"] == td.RANK_FLOOR == 0.10
    assert td.bar_rule([0.4], td.STD_FLOOR)["bar"] == td.STD_FLOOR == 0.25
    low = td.bar_rule([0.09, 0.5], 0.10)
    assert low["escalate"] and low["reference"] == 0.09
    assert not td.bar_rule([0.1], 0.10)["escalate"]
    with pytest.raises(ContractError):
        td.bar_rule([], 0.10)
    with pytest.raises(ContractError):
        td.bar_rule([math.inf], 0.10)


# ----- gates --------------------------------------------------------------------------------------
BARS = td.THRESHOLDS | {"G1_min_effective_rank_ratio": 0.2, "G1_min_std_ratio": 0.4}


def _collapse(rank=0.3, std=0.8, collapsed=0.0):
    return {
        "effective_rank_ratio": rank,
        "std_ratio": std,
        "predicted_collapsed_fraction": collapsed,
    }


def _over_n(lo=0.02, undefined=0):
    return {"ci95": [lo, lo + 0.1], "undefined_resamples": undefined}


def test_g1_uses_the_calibrated_bars_in_both_directions():
    assert td.g1_passes(_collapse(), _over_n(), BARS)
    assert td.g1_passes(_collapse(rank=0.2, std=0.4), _over_n(), BARS)  # at the bar passes
    assert not td.g1_passes(_collapse(rank=0.19), _over_n(), BARS)
    assert not td.g1_passes(_collapse(std=0.39), _over_n(), BARS)
    assert not td.g1_passes(_collapse(collapsed=0.06), _over_n(), BARS)


def test_g1_comparative_needs_a_lower_bound_strictly_above_zero():
    assert not td.g1_passes(_collapse(), _over_n(lo=0.0), BARS)
    assert not td.g1_passes(_collapse(), _over_n(lo=-0.01), BARS)
    assert not td.g1_passes(_collapse(), _over_n(undefined=1), BARS)
    parts = td.g1_parts(_collapse(), _over_n(lo=-0.01), BARS)
    assert parts == {"collapsed_fraction": True, "rank": True, "std": True, "rank_over_N": False}


def test_g1_refuses_unset_or_sub_floor_bars():
    unset = td.THRESHOLDS | {"G1_min_effective_rank_ratio": None}
    with pytest.raises(ContractError, match="calibrated"):
        td.g1_passes(_collapse(), _over_n(), unset)
    low = BARS | {"G1_min_effective_rank_ratio": 0.05}
    with pytest.raises(ContractError, match="floor"):
        td.g1_passes(_collapse(), _over_n(), low)
    low = BARS | {"G1_min_std_ratio": 0.2}
    with pytest.raises(ContractError, match="floor"):
        td.g1_passes(_collapse(), _over_n(), low)


# ----- G1 (iii): the projected comparative ------------------------------------------------------
def _session_moments(rows, sessions, shift, basis):
    m = td.SessionMoments(shift, basis)
    m.add(rows, sessions)
    return m


def test_projection_basis_is_the_top_principal_directions():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(400, 6)) * np.array([5.0, 3, 2, 1, 0.5, 0.1])
    m = td.Moments(np.zeros(6))
    m.add(x)
    basis = td.projection_basis(m, 2)
    assert basis.shape == (6, 2)
    assert abs(basis[0, 0]) > 0.99 and abs(basis[1, 1]) > 0.99


def test_comparative_rank_point_equals_the_direct_projected_ratio():
    rng = np.random.default_rng(11)
    d, k = 8, 5
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :k]
    shift = np.zeros(d)
    sessions = np.repeat(np.arange(12), 20)
    enc = rng.normal(size=(240, d))
    w = enc * 0.8 + rng.normal(size=(240, d)) * 0.1  # keeps most directions
    n = np.outer(rng.normal(size=240), rng.normal(size=d))  # rank one
    out = td.comparative_rank(
        _session_moments(w, sessions, shift, basis),
        _session_moments(n, sessions, shift, basis),
        _session_moments(enc, sessions, shift, basis),
        resamples=300,
    )
    direct = (ld.effective_rank(w @ basis) - ld.effective_rank(n @ basis)) / ld.effective_rank(
        enc @ basis
    )
    assert out["difference"] == pytest.approx(direct, rel=1e-9)
    assert out["ci95"][0] > 0 and out["undefined_resamples"] == 0
    swapped = td.comparative_rank(
        _session_moments(n, sessions, shift, basis),
        _session_moments(w, sessions, shift, basis),
        _session_moments(enc, sessions, shift, basis),
        resamples=300,
    )
    assert swapped["ci95"][1] < 0  # the other direction: a collapsed W fails (iii)


def test_session_moments_merge_and_refusals():
    rng = np.random.default_rng(12)
    basis = np.eye(3)
    a = _session_moments(rng.normal(size=(10, 3)), np.array([0] * 5 + [1] * 5), 0, basis)
    b = _session_moments(rng.normal(size=(6, 3)), np.array([1] * 3 + [2] * 3), 0, basis)
    merged = a.merge(b)
    assert sorted(merged.sessions) == [0, 1, 2] and merged.sessions[1][0] == 8
    with pytest.raises(ContractError, match="one session"):
        a.add(np.zeros((2, 3)), np.array([0]))
    with pytest.raises(ContractError, match="same sessions"):
        td.comparative_rank(a, merged, merged, resamples=10)


def test_truncation_controls_score_low_rank():
    rng = np.random.default_rng(13)
    enc_rows = rng.normal(size=(500, 40))
    pred_rows = enc_rows * 0.7
    enc, pred = td.Moments(np.zeros(40)), td.Moments(np.zeros(40))
    enc.add(enc_rows)
    pred.add(pred_rows)
    full = td.collapse_statistics(pred, enc)
    for k in (1, 2, 4):
        control = td.truncated_spectrum_statistics(pred, enc, k)
        assert control["predicted_effective_rank"] <= k + 1e-9
        assert control["effective_rank_ratio"] < full["effective_rank_ratio"]
    one = td.truncated_spectrum_statistics(pred, enc, 1)
    assert one["predicted_effective_rank"] == pytest.approx(1.0)
    everything = td.truncated_spectrum_statistics(pred, enc, 40)
    assert everything["effective_rank_ratio"] == pytest.approx(full["effective_rank_ratio"])
    assert everything["std_ratio"] == pytest.approx(full["std_ratio"])


def _readability(median=1.0, ratio_hi=0.5):
    return {"median_cm": median, "ratio_to_B_occ": {"ci95": [0.1, ratio_hi]}}


def test_ceiling_is_the_t1_bar_on_the_encoded_grid():
    assert td.ceiling_passes(_readability())
    assert td.ceiling_passes(_readability(1.5, 0.6))
    assert not td.ceiling_passes(_readability(1.51))
    assert not td.ceiling_passes(_readability(ratio_hi=0.61))


def _seed(g1=True, g2=True, g3=True, g4=True, g5=True):
    flags = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}
    return td.seed_gates({h: dict(flags) for h in td.GATED_HORIZONS})


def test_seed_gates_add_the_latent_gates():
    assert _seed()["latent"] and _seed()["passes"]
    only_g1 = _seed(g1=False)
    assert only_g1["latent"] and not only_g1["dynamics"] and not only_g1["passes"]
    assert not _seed(g3=False)["latent"]
    mixed = {8: {g: True for g in td.GATES}, 16: {g: g != "G2" for g in td.GATES}}
    assert not td.seed_gates(mixed)["gates"]["G2"]


# ----- decision -----------------------------------------------------------------------------------
CEIL_OK = {8: True, 16: True}
CEIL_BAD = {8: True, 16: False}


@pytest.mark.parametrize(
    "seeds, ceiling, row",
    [
        ([_seed()] * 3, CEIL_OK, "WM-TOK-DYNAMICS"),
        ([_seed(), _seed(g5=False), _seed(g5=False)], CEIL_OK, "WM-TOK-UNSTABLE"),
        ([_seed(), _seed(), _seed(g2=False)], CEIL_OK, "WM-TOK-UNSTABLE"),
        ([_seed(g5=False)] * 3, CEIL_BAD, "WM-TOK-CEILING"),
        ([_seed(g5=False)] * 3, CEIL_OK, "WM-TOK-APPLE-LOST"),
        ([_seed(g5=False), _seed(g5=False), _seed(g1=False)], CEIL_OK, "WM-TOK-APPLE-LOST"),
        ([_seed(g1=False)] * 3, CEIL_OK, "WM-TOK-COLLAPSE"),
        ([_seed(g1=False)] * 3, CEIL_BAD, "WM-TOK-COLLAPSE"),  # the ceiling cannot rescue it
        ([_seed(g1=False), _seed(g1=False, g5=False), _seed(g3=False)], CEIL_OK, "WM-TOK-COLLAPSE"),
        ([_seed(g2=False)] * 3, CEIL_OK, "WM-TOK-NO-DYNAMICS"),
        ([_seed(g4=False), _seed(g3=False), _seed(g1=False)], CEIL_BAD, "WM-TOK-NO-DYNAMICS"),
    ],
)
def test_decide_takes_the_first_matching_row(seeds, ceiling, row):
    decision = td.decide(
        void=False, seeds=dict(zip(td.MODEL_SEEDS, seeds, strict=True)), ceiling=ceiling
    )
    assert decision["outcome"] == row
    assert decision["abandonment_clause_fires"] is (row in td.CLAUSE_ROWS)


def test_void_and_refusals():
    assert td.decide(void=True) == {"outcome": "V", "abandonment_clause_fires": False}
    seeds = dict.fromkeys(td.MODEL_SEEDS, _seed())
    with pytest.raises(ContractError, match="every model seed"):
        td.decide(void=False, seeds={0: _seed()}, ceiling=CEIL_OK)
    with pytest.raises(ContractError, match="ceiling"):
        td.decide(void=False, seeds=seeds, ceiling={8: True})
    with pytest.raises(ContractError, match="unknown outcome"):
        td.abandonment_fires("WM-DYNAMICS")  # TASK-065's row name is not this task's


def test_the_clause_fires_on_exactly_three_rows():
    fired = {row for row in td.ROWS if td.abandonment_fires(row)}
    assert fired == {"WM-TOK-APPLE-LOST", "WM-TOK-COLLAPSE", "WM-TOK-NO-DYNAMICS"}


def test_a_gated_decision_needs_the_frozen_values():
    if any(v is None for v in td.THRESHOLDS.values()) or td.UPDATES is None:
        with pytest.raises(ContractError, match="not frozen"):
            td.require_frozen()
    else:
        td.require_frozen()
