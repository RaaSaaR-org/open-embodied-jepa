"""The C1-M feasibility record's rules (development only; no world model)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


def test_seed_block_and_salts_are_the_declared_ones():
    c1m.check_seed_ranges()
    assert c1m.SEED_BLOCK == (63000, 64999)
    assert c1m.SEED_RANGES == {
        "M1": (63000, 63031),
        "F3": (63100, 63131),
        "R": (63132, 63163),
        "T": (63200, 63231),
        "corpus": (63300, 64323),
    }
    assert c1m.DEBUG_SEEDS == (64900, 64999)
    assert c1m.SALTS == {
        "move": 7901,
        "corpus_aim": 7902,
        "outer_folds": 7903,
        "inner_folds": 7904,
        "bootstrap": 7905,
        "tau_direction": 7906,
        "learning_curve": 7907,
    }
    for low, high in (*c1m.SEED_RANGES.values(), c1m.DEBUG_SEEDS):
        for f_low, f_high in c1m.FORBIDDEN_RANGES.values():
            assert high < f_low or low > f_high
    assert {"c1_block", "r15_block"} <= set(c1m.FORBIDDEN_RANGES)
    assert not set(c1m.SALTS.values()) & {*c1.SALTS.values(), 7801, 7802}
    for role in ("M1", "F3", "R", "T"):
        assert len(c1m.seeds_of(role)) == c1m.F_RESETS
    assert len(c1m.seeds_of("corpus")) == c1m.CORPUS_ROOTS
    assert len(c1m.seeds_of("F3") + c1m.seeds_of("R")) == c1m.READ_RESETS


def test_seed_guard():
    assert c1m.check_seeds("F3", c1m.seeds_of("F3")) == c1m.seeds_of("F3")
    with pytest.raises(pt.GuardError):
        c1m.check_seeds("F3", c1m.seeds_of("F3")[:-1])
    with pytest.raises(pt.GuardError):
        c1m.check_seeds("F3", (57000,), debug=True)  # C1's block
    with pytest.raises(pt.GuardError):
        c1m.check_seeds("F3", (58000,), debug=True)  # R15's block
    with pytest.raises(pt.GuardError):
        c1m.check_seeds("F3", (63000,), debug=True)  # not a debug seed
    debug = c1m.seeds_of("M1", debug=True)
    assert debug == tuple(range(64900, 64906))
    assert c1m.check_seeds("M1", debug, debug=True) == debug


def test_move_draw_is_uniform_nested_and_on_the_table():
    seeds = range(63000, 63400)
    disc = np.array([c1m.move_offset(s, "disc", 5) for s in seeds])
    half = np.array([c1m.move_offset(s, "half_disc", 5) for s in seeds])
    assert np.linalg.norm(disc, axis=1).max() <= 0.05 + 1e-12
    assert np.linalg.norm(half, axis=1).max() <= 0.05 + 1e-12
    assert half[:, 1].max() <= 1e-12  # the -y half
    # uniform over the disc: P(|offset| <= rho / 2) = 1/4
    assert 0.18 < np.mean(np.linalg.norm(disc, axis=1) <= 0.025) < 0.32
    # one draw per reset, shared by every radius: offsets scale with rho
    for s in (63000, 63017):
        assert np.allclose(c1m.move_offset(s, "disc", 6), 2.0 * c1m.move_offset(s, "disc", 3))
    assert all(c1m.move_uniforms(s)["k"] == 0 for s in seeds)  # no re-draw on v2's reset
    plate = np.asarray(c1m.reset_of(63000)["plate_xy"])
    assert np.allclose(c1m.moved_plate(63000, "disc", 4), plate + c1m.move_offset(63000, "disc", 4))
    with pytest.raises(ContractError):
        c1m.move_offset(63000, "ring", 4)


def test_plate_mean_is_the_family_centroid():
    assert np.allclose(c1m.plate_mean("disc", 5), (0.49, -0.09))
    assert np.allclose(c1m.plate_mean("half_disc", 3), (0.49, -0.09 - 0.04 / math.pi))
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=200_000), rng.uniform(size=200_000)
    y = 0.03 * np.sqrt(u) * np.sin(math.pi + math.pi * v)
    assert y.mean() == pytest.approx(-0.04 / math.pi, abs=2e-4)


def test_planted_error_and_corpus_aim():
    for level in c1m.TAU_LEVELS_CM:
        e = np.asarray(c1m.planted_error_m(63200, level))
        assert np.linalg.norm(e) == pytest.approx(level / 100.0)
    one, two = np.asarray(c1m.planted_error_m(63200, 1.0)), c1m.planted_error_m(63200, 2.0)
    assert np.allclose(2.0 * one, two)  # one direction per reset
    draws = np.array([c1m.corpus_aim(s, -0.5) for s in c1m.seeds_of("corpus")])
    assert draws[:, 0].min() >= -0.5 and draws[:, 0].max() <= 0.5
    assert np.abs(draws[:, 1]).max() <= 0.03
    assert c1m.corpus_aim(63300, -0.5) != c1.corpus_aim(63300, -0.5)  # its own salt


def test_rho_star_is_monotone():
    assert c1m.rho_star({3: 31, 4: 30, 5: 29}) == 4
    assert c1m.rho_star({3: 32, 4: 32, 5: 32, 6: 30}) == 6
    assert c1m.rho_star({3: 29}) is None
    assert c1m.rho_star({3: 30, 5: 32}) == 3  # 4 not run: nothing above it counts


def test_f3_strict_reading_and_fresh_ceiling():
    ceiling = np.ones(32, bool)
    far = np.r_[np.ones(10, bool), np.zeros(22, bool)]  # +22
    near = np.r_[np.ones(22, bool), np.zeros(10, bool)]  # +10: interval reaches below +8
    out = c1m.decide_f3(ceiling, {"H-now": far, "shuf-proxy": far, "mean-proxy": near})
    assert out["proxies"]["H-now"]["passes"] and out["proxies"]["shuf-proxy"]["passes"]
    mean = out["proxies"]["mean-proxy"]
    assert mean["difference"] == 10 and mean["ci95"][0] < 8 and not mean["headroom_passes"]
    assert not mean["detectably_below"] and not out["twins_none"] and not out["passes"]
    close = np.r_[np.ones(29, bool), np.zeros(3, bool)]  # +3: detectably below +8
    out = c1m.decide_f3(ceiling, {"mean-proxy": close})
    assert out["twins_none"]
    low = np.r_[np.ones(29, bool), np.zeros(3, bool)]
    weak = c1m.decide_f3(low, {"H-now": np.zeros(32, bool)})
    assert not weak["ceiling_ok"] and not weak["passes"]


def test_tau_rule():
    counts = {0.0: 32, 0.5: 31, 1.0: 28, 1.5: 27, 2.0: 30, 3.0: 20}
    assert c1m.decide_tau(counts)["tau_commit_cm"] == 1.0
    assert c1m.decide_tau(counts | {0.0: 27})["tau_commit_cm"] is None
    with pytest.raises(ContractError):
        c1m.decide_tau({0.0: 32})


def test_nested_subsets_and_guard():
    fit = np.arange(200)
    subsets = c1m.nested_subsets(fit, 0)
    sizes = [len(subsets[f]) for f in c1m.FRACTIONS]
    assert sizes == [50, 100, 150, 200]
    for small, large in zip(c1m.FRACTIONS, c1m.FRACTIONS[1:], strict=False):
        assert set(subsets[small]) <= set(subsets[large])
    assert np.array_equal(subsets[1.0], fit)
    assert np.array_equal(c1m.nested_subsets(fit, 0)[0.5], subsets[0.5])
    assert not np.array_equal(c1m.nested_subsets(fit, 1)[0.5], subsets[0.5])
    rng = np.random.default_rng(1)
    e_all = rng.uniform(0.5, 1.0, 300)
    falling = c1m.median_difference_ci(e_all + 0.2, e_all)
    assert c1m.still_falling(falling)
    flat = c1m.median_difference_ci(e_all, e_all)
    assert not c1m.still_falling(flat)


def test_read_rule_and_hidden_check():
    ceiling = np.ones(64, bool)
    ok = c1m.decide_read(ceiling, np.r_[np.ones(60, bool), np.zeros(4, bool)])
    assert ok["passes"] and ok["difference"] == 4
    miss = c1m.decide_read(ceiling, np.r_[np.ones(40, bool), np.zeros(24, bool)])
    assert not miss["passes"] and miss["detectably_beyond_delta"]
    hidden = {"ci95": [1.4, 2.0]}
    assert c1m.hidden_ok(hidden, 1.0) and not c1m.hidden_ok(hidden, 1.5)
    assert c1m.hidden_ok(hidden, None) is None


def test_verdict_ladder():
    f2 = {"passes": True}
    f3 = {"ceiling_ok": True, "twins_none": False, "passes": True}
    tau = {"tau_commit_cm": 1.0}
    passed, missed = {"passes": True}, {"passes": False, "detectably_beyond_delta": True}
    near = {"passes": False, "detectably_beyond_delta": False}
    v = c1m.verdict
    assert v(rho=None, f2=None, f3=None)["row"] == "M-INFEASIBLE"
    assert v(rho=5, f2={"passes": False}, f3=None)["row"] == "M-INFEASIBLE"
    assert v(rho=5, f2=f2, f3=f3 | {"ceiling_ok": False, "twins_none": True})["row"] == (
        "M-INFEASIBLE"
    )
    twins = v(rho=5, f2=f2, f3=f3 | {"twins_none": True, "passes": False})
    assert twins["row"] == "M-TWINS-NONE" and twins["clause_fires"]
    esc = v(rho=4, f2=f2, f3=f3 | {"passes": False})
    assert esc["row"] == "M-TWINS-ESCALATE" and not esc["clause_fires"]
    assert v(rho=5, f2=f2, f3=f3, tau={"tau_commit_cm": None})["row"] == "M-NO-TAU"
    assert v(rho=5, f2=f2, f3=f3, tau=tau, hidden4_ok=False)["row"] == "M-ARM-KEYED"
    common = {"rho": 5, "f2": f2, "f3": f3, "tau": tau, "hidden4_ok": True}
    assert v(**common, reads={4: passed})["row"] == "M-PROCEED"
    assert v(**common, reads={4: missed, 8: passed})["row"] == "M-PROCEED"
    none = v(**common, reads={4: missed, 8: missed}, falling={4: False, 8: False})
    assert none["row"] == "M-READ-NONE" and none["clause_fires"]
    data = v(**common, reads={4: missed, 8: missed}, falling={4: False, 8: True})
    assert data["row"] == "M-NO-BAR-DATA" and not data["clause_fires"]
    assert v(**common, reads={4: missed, 8: near})["row"] == "M-NO-BAR"
    assert v(**common, reads={4: missed})["row"] == "M-NO-BAR"  # 8 x 8 not admitted


def test_bars_are_r15s():
    assert (c1m.CEILING_MIN, c1m.HEADROOM_BAR, c1m.FEASIBILITY_BAR) == (30, 8, 0.8)
    assert (c1m.READ_RESETS, c1m.READ_ALLOWANCE, c1m.READ_DELTA) == (64, 4, 8)
    assert c1m.TAU_LEVELS_CM == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0) and c1m.TAU_BAR == 28
    assert c1m.RHO_GRID_CM == (3, 4, 5, 6) and c1m.FAMILIES == ("disc", "half_disc")
    assert c1m.MOVE_STEP == 300 and c1m.RULE == c1.RULE
    assert c1m.GRIDS == (4, 8) and c1m.FRACTIONS == (0.25, 0.5, 0.75, 1.0)
    assert (c1m.COST_SECONDS_MAX, c1m.REMAINING_MIN_CM) == (60.0, 2.0)


def test_imports_stay_lazy():
    code = (
        "import sys; import embodied_jepa.lewm_next_c1m, embodied_jepa.lewm_next_c1m_runtime; "
        "bad = [m for m in ('torch', 'mujoco') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
