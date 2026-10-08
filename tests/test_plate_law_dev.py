"""TASK-082 design note's development plate laws (``plate_law_dev``); NumPy only."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import plate_law_dev as pl
from embodied_jepa.contracts import ContractError


def _straight_palm(step_m: float, direction=(1.0, 0.0)):
    d = np.asarray(direction, np.float64)
    return {k: step_m * (k - pl.S0) * d for k in range(pl.S0, pl.S1 + 1)}


def test_seed_ranges_are_fresh_and_disjoint():
    pl.check_seed_ranges()
    for low, high in pl.SEED_RANGES.values():
        for name, (o_low, o_high) in pl.FORBIDDEN_RANGES.items():
            assert high < o_low or low > o_high, name
    with pytest.raises(ContractError):
        pl.check_seeds([70200])
    pl.check_seeds([72000, 72355])


def test_salts_are_new():
    assert set(pl.SALTS.values()) == {8401, 8402, 8403, 8404}


def test_c1m_law_matches_cell_a():
    palm = _straight_palm(0.001)
    got = pl.displacement("c1m", palm, 500)
    want = c1.RULE["kappa"] * (palm[500 - pl.LAG] - palm[pl.S0])
    np.testing.assert_allclose(got, want)
    assert not pl.displacement("c1m", palm, pl.S0).any()
    np.testing.assert_allclose(pl.displacement("c1m", palm, 600), pl.displacement("c1m", palm, 525))


def test_sat_saturates_and_turns():
    p = pl.LAWS["sat"]
    small = pl.sat_displacement([0.01, 0.0], **p)
    large = pl.sat_displacement([1.0, 0.0], **p)
    assert np.linalg.norm(large) == pytest.approx(p["amplitude_m"], rel=1e-3)
    assert small[0] < 0  # toward the hand
    angle = np.arctan2(-large[1], -large[0])  # the turn of -d's direction
    assert angle == pytest.approx(p["swirl_rad"] * 1.0 / p["scale_m"], rel=1e-6)
    assert not pl.sat_displacement([0.0, 0.0], **p).any()


def test_play_has_a_dead_band_and_memory():
    p = pl.LAWS["play"]
    # inside the band: no motion
    assert not pl.displacement("play", _straight_palm(0.0001), 525).any()
    # out and back: the plate keeps part of the excursion (hysteresis)
    palm = {}
    for k in range(pl.S0, pl.S1 + 1):
        t = k - pl.S0
        x = 0.002 * t if t <= 60 else 0.002 * (120 - t)
        palm[k] = np.array([x, 0.0])
    end = pl.displacement("play", palm, 525)
    z = p["kappa"] * palm[525 - pl.LAG][0]  # the drive near the start again
    # the plate trails the drive by the band's width, on the side it came from
    assert end[0] == pytest.approx(z - p["width_m"])
    assert abs(end[0]) > abs(z)


def test_krr_recovers_a_smooth_law():
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 0.02, (200, 2))
    h = rng.normal(0.0, 0.02, (200, 2)) + [0.0, 0.25]
    g = p + rng.uniform(-0.5, 0.5, (200, 1)) * (h - p)
    y = p + np.stack(
        [pl.sat_displacement(gi - hi, **pl.LAWS["sat"]) for gi, hi in zip(g, h, strict=True)]
    )
    model = pl.krr_fit(p[:150], h[:150], g[:150], y[:150])
    err = np.linalg.norm(pl.krr_plate(model, p[150:], h[150:], g[150:]) - y[150:], axis=1)
    assert np.median(err) < 0.005


def test_paired_interval_counts():
    out = pl.paired_interval([True, True, False, True], [True, False, True, False])
    assert out["difference"] == 1 and out["only_first"] == 2 and out["only_second"] == 1
