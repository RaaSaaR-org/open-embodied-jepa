"""The C1 feasibility record's rules (development only; no world model)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


def test_seed_block_and_salts_are_fresh():
    c1.check_seed_ranges()
    assert c1.SEED_RANGES == {"F": (57000, 57031), "corpus": (57100, 57355)}
    assert c1.DEBUG_SEEDS == (57900, 57999)
    assert c1.SALTS == {
        "corpus_aim": 7701,
        "outer_folds": 7702,
        "inner_folds": 7703,
        "bootstrap": 7704,
    }
    for low, high in (*c1.SEED_RANGES.values(), c1.DEBUG_SEEDS):
        for f_low, f_high in c1.FORBIDDEN_RANGES.values():
            assert high < f_low or low > f_high
    assert len(c1.seeds_of("F")) == c1.F_RESETS
    assert len(c1.seeds_of("corpus")) == c1.CORPUS_ROOTS


def test_seed_guard():
    assert c1.check_seeds("F", c1.seeds_of("F")) == c1.seeds_of("F")
    with pytest.raises(pt.GuardError):
        c1.check_seeds("F", c1.seeds_of("F")[:-1])
    with pytest.raises(pt.GuardError):
        c1.check_seeds("F", (56000,), debug=True)  # TASK-076's block
    with pytest.raises(pt.GuardError):
        c1.check_seeds("F", (57000,), debug=True)  # not a debug seed
    assert c1.check_seeds("F", (57900, 57901), debug=True) == (57900, 57901)


def test_reset_is_v2s_own():
    r = c1.reset_of(57000)
    assert abs(r["plate_xy"][0] - 0.49) <= 0.02 and abs(r["plate_xy"][1] + 0.09) <= 0.02
    assert abs(r["object_xy"][0] - 0.34) <= 0.03 and abs(r["object_xy"][1] + 0.18) <= 0.03
    assert c1.PLATE_MEAN == (0.49, -0.09)


def test_box_geometry():
    p, h = np.array([0.49, -0.09]), np.array([0.30, -0.20])
    for a, b in ((0.0, 0.0), (1 / 3, 0.0), (-0.5, 0.03), (0.5, -0.03)):
        g = c1.from_box(a, b, p, h)
        assert np.allclose(c1.to_box(g, p, h), (a, b))
    assert np.allclose(c1.to_box(c1.fixed_point(p, h), p, h), (1 / 3, 0.0))
    far = c1.from_box(-0.9, 0.1, p, h)
    assert np.allclose(c1.to_box(c1.clip_to_box(far, p, h, -0.3), p, h), (-0.3, 0.03))
    cells = c1.grid(-0.2)
    assert len(cells) == 15 * 7 and cells[0] == (-0.2, -0.03) and cells[-1] == (0.5, 0.03)
    assert len(c1.grid(-0.5)) == 21 * 7


def test_n_proxy_matches_the_note():
    # design note §4.1's table: a_lo -> a_N
    for a_lo, a_n in ((-0.5, 0.5), (-0.4, 0.475), (-0.3, 0.45), (-0.2, 0.425)):
        assert c1.n_proxy_a(a_lo) == pytest.approx(a_n)


def test_proxy_aims():
    p, h = np.array([0.49, -0.09]), np.array([0.30, -0.20])
    assert np.allclose(c1.proxy_aim("H-now", p, h, -0.3), p)
    assert np.allclose(c1.to_box(c1.proxy_aim("N-proxy", p, h, -0.2), p, h), (0.425, 0.0))
    foreign = p + np.array([0.01, -0.015])
    g = c1.proxy_aim("shuf-proxy", p, h, -0.3, foreign_p=foreign)
    assert np.allclose(g, c1.fixed_point(foreign, h))  # inside the box: no clip
    assert np.allclose(c1.proxy_aim("mean-proxy", p, h, -0.3), c1.fixed_point(p, h))
    with pytest.raises(ContractError):
        c1.proxy_aim("shuf-proxy", p, h, -0.3)


def test_corpus_aim_is_uniform_in_the_box_and_seeded():
    draws = [c1.corpus_aim(s, -0.3) for s in c1.seeds_of("corpus")]
    a, b = np.array(draws).T
    assert a.min() >= -0.3 and a.max() <= 0.5 and np.abs(b).max() <= 0.03
    assert c1.corpus_aim(57100, -0.3) == c1.corpus_aim(57100, -0.3)


def test_read_step():
    def path(settle):
        return {t: (0.0 if t >= settle else 5.0) for t in range(405, 526)}

    distances = [path(470)] * 28 + [path(500)] * 4
    assert c1.read_step(distances) == 470
    assert c1.read_step([path(470)] * 27 + [path(500)] * 5) == 500
    assert c1.read_step([path(470)] * 27 + [None] * 5) is None


def test_a_lo_rule_is_monotone():
    assert c1.decide_a_lo({-0.5: 32, -0.4: 32, -0.3: 31, -0.2: 32}) == -0.5
    assert c1.decide_a_lo({-0.5: 30, -0.4: 32, -0.3: 31, -0.2: 32}) == -0.4
    assert c1.decide_a_lo({-0.5: 32, -0.4: 32, -0.3: 30, -0.2: 32}) == -0.2
    assert c1.decide_a_lo({-0.5: 32, -0.4: 32, -0.3: 32, -0.2: 30}) is None


def test_f3_rule_and_guard():
    ceiling = np.ones(32, bool)
    far = np.r_[np.ones(16, bool), np.zeros(16, bool)]
    near = np.r_[np.ones(25, bool), np.zeros(7, bool)]
    out = c1.decide_f3(ceiling, {"H-now": far, "shuf-proxy": far, "mean-proxy": near})
    assert out["proxies"]["H-now"]["passes"]
    assert out["proxies"]["shuf-proxy"]["feasibility"]["predicted_pass_probability"] > 0.99
    mean = out["proxies"]["mean-proxy"]
    assert not mean["headroom_passes"] and mean["near"] and not mean["detectably_below"]
    assert not out["passes"]
    tied = c1.decide_f3(ceiling, {"N-proxy": ceiling})["proxies"]["N-proxy"]
    assert tied["detectably_below"]


def test_sysid_and_the_controller_form():
    rng = np.random.default_rng(0)
    p = rng.normal([0.49, -0.09], 0.01, (50, 2))
    h = rng.normal([0.30, -0.20], 0.02, (50, 2))
    g = rng.normal([0.40, -0.13], 0.03, (50, 2))
    y = p - 0.5 * (g - h)  # the rule with the palm ending at the aim
    coef = c1.sysid_fit(p, h, g, y)
    assert np.allclose(c1.sysid_predict(coef, p, h, g), y)
    p0, h0 = p[0], h[0]
    aim, log = c1.choose_aim(lambda x: c1.sysid_predict(coef, p0, h0, x)[0], p0, h0, -0.3)
    assert log["converged"]
    assert np.allclose(c1.to_box(aim, p0, h0), (1 / 3, 0.0), atol=0.02)


def test_statistics_use_c1s_salt():
    a, b = np.ones(32, bool), np.r_[np.ones(20, bool), np.zeros(12, bool)]
    assert c1.paired_interval(a, b) == c1.paired_interval(a, b)
    assert c1.paired_interval(a, b)["difference"] == 12
    ci = c1.median_ci(np.arange(100.0))
    assert ci["ci95"][0] < 49.5 < ci["ci95"][1]
    folds = c1.outer_folds(256)
    assert sorted(np.bincount(folds)) == [51, 51, 51, 51, 52]


def test_verdict_order():
    ok1, ok2 = {"passes": True}, {"passes": True}
    f3_ok, f3_bad = {"passes": True}, {"passes": False}
    f5 = {"hidden_ok": True, "readout_ok": True}
    assert c1.verdict({"passes": False}, ok2, f3_ok, f5)["row"] == "C1-INFEASIBLE"
    assert c1.verdict(ok1, {"passes": False}, f3_ok, f5)["row"] == "C1-INFEASIBLE"
    assert c1.verdict(ok1, ok2, f3_bad, f5)["row"] == "C1-TWINS-ESCALATE"
    assert c1.verdict(ok1, ok2, f3_ok, f5 | {"hidden_ok": False})["row"] == "C1-ARM-KEYED"
    assert c1.verdict(ok1, ok2, f3_ok, f5 | {"readout_ok": False})["row"] == "C1-NO-BAR"
    assert c1.verdict(ok1, ok2, f3_ok, f5)["row"] == "C1-PROCEED"
    assert not c1.verdict(ok1, ok2, f3_bad, f5)["clause_fires"]


def test_bars_are_the_notes():
    assert c1.F1_REMAINING_MIN_CM == 2.0
    assert (c1.F2_CEILING_MIN, c1.F2_REFUSED_MAX, c1.REACH_MIN) == (30, 8, 31)
    assert (c1.HEADROOM_BAR, c1.FEASIBILITY_BAR, c1.FEASIBILITY_PAIRS) == (8, 0.8, 64)
    assert (c1.F5_READOUT_MAX_CM, c1.F5_HIDDEN_MIN_CM, c1.F6_SECONDS_MAX) == (0.5, 1.0, 60.0)
    assert c1.RULE == {"name": "A", "kappa": -0.5, "L": 2, "s0": 405, "s1": 525}
    assert c1.LOOKAHEAD_TOLERANCE_M == pytest.approx(0.0025)
    assert (c1.R_TOLERANCE_CM, c1.R_FRACTION, c1.PALM_SPEED_LIMIT_CM) == (0.1, 0.875, 0.5)


def test_imports_stay_lazy():
    code = (
        "import sys; import embodied_jepa.lewm_next_c1, embodied_jepa.lewm_next_c1_runtime; "
        "bad = [m for m in ('torch', 'mujoco') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
