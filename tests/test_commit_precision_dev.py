"""The development solver variants of the TASK-081 design note (``commit_precision_dev``)."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_jepa import commit_precision_dev as cp
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_next_c1 as c1

P_HAT = np.array([0.47, -0.08])
H = np.array([0.30, -0.16])
TOL = 0.0025


def _setup(slope, fixed=None):
    """A synthetic predictor p~(g) = g* + S (g - g*); a chunk encodes its aim in two entries."""
    fixed = c1.from_box(0.31, 0.004, P_HAT, H) if fixed is None else fixed
    slope = np.asarray(slope, np.float64)
    targets = np.asarray([c1.from_box(a, b, P_HAT, H) for a, b in lm.GRID])

    def encode(g):
        c = np.zeros((lm.HORIZON, 14), np.float32)
        c[0, :2] = np.asarray(g, np.float32)
        return c

    def predict(commands):
        g = np.asarray(commands, np.float64)[:, 0, :2]
        return fixed + (g - fixed) @ slope.T

    def chunk_of(g):
        return encode(g), True

    chunks = np.stack([encode(g) for g in targets])
    feasible = np.ones(len(targets), bool)
    return fixed, targets, chunks, feasible, predict, chunk_of


def _variants(slope):
    fixed, targets, chunks, feasible, predict, chunk_of = _setup(slope)
    grid = predict(chunks)

    def single(g):
        return predict(chunk_of(g)[0][None])[0]

    variants = cp.all_variants(P_HAT, H, targets, feasible, grid, single, lambda g: True,
                               known=(), tolerance_m=TOL, a_lo=lm.A_LO)  # fmt: skip
    return fixed, variants, (targets, chunks, feasible, predict, chunk_of)


def test_alpha_one_is_the_frozen_controller():
    for slope in (np.diag([-0.13, -0.42]), -1.2 * np.eye(2), [[-0.9, 0.3], [0.2, -1.05]]):
        _fixed, variants, (targets, chunks, feasible, predict, chunk_of) = _variants(slope)
        g, log = wrt.choose_from_grid("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of,
                                      tolerance_m=TOL, seed=0)  # fmt: skip
        assert np.array_equal(variants["frozen"]["g"], g)
        assert variants["frozen"]["converged"] == log["converged"]


def test_damping_converges_where_the_frozen_step_oscillates():
    fixed, variants, _ = _variants(-1.2 * np.eye(2))  # locally steeper than -1: no contraction
    assert not variants["frozen"]["converged"]
    assert not variants["cap30"]["converged"]
    assert variants["damped"]["converged"]  # 1 - 0.5 (1 + 1.2) = -0.1
    assert np.linalg.norm(variants["damped"]["g"] - fixed) < 0.003


def test_affine_fixed_point_recovers_a_linear_map():
    fixed, variants, _ = _variants([[-0.3, 0.1], [0.05, -0.6]])
    assert np.linalg.norm(variants["affine_global"]["g"] - fixed) < 1e-9
    assert np.linalg.norm(variants["affine_local"]["g"] - fixed) < 1e-9
    assert variants["affine_local"]["points"] == 25


def test_best_residual_picks_the_smallest():
    ev = [(np.zeros(2), np.array([0.01, 0.0])), (np.ones(2), np.array([0.001, 0.0]))]
    g, res, i = cp.best_residual(ev)
    assert i == 1 and np.array_equal(g, np.ones(2)) and res == pytest.approx(0.001)


def test_only_declared_variants_execute():
    assert set(cp.EXECUTABLE) <= set(cp.VARIANTS)
    with pytest.raises(ValueError):
        cp.run_task({"kind": "attempt", "variant": "cap30"})


def test_ni_pass_table_matches_task080_stage_s():
    # TASK-080 Stage S: W only 1, H-rule only 6, interval [-10, 0] -> G-NI fails at margin 8
    table = cp.ni_pass_table(64, 8, k_max=12)
    assert not table[1, 6]
    assert table[0, 0] and table[3, 3]


def test_ni_power_bounds():
    table = cp.ni_pass_table(64, 8, k_max=30)
    assert cp.ni_power(0.98, 0.98, "overlap", n=64, margin=8, table=table) == pytest.approx(1.0)
    assert cp.ni_power(0.80, 0.98, "overlap", n=64, margin=8, table=table) < 0.2
