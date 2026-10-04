"""R15's development script: the scene-blind miss identity it relies on, and its curve binning."""

import importlib.util
from pathlib import Path

import numpy as np

from embodied_jepa import lewm_next_c1 as c1

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("r15", ROOT / "scripts/r15_direction_dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_blind_miss_is_the_plate_error_for_any_kappa():
    """R15.2: aiming at the fixed point of a wrong plate p_bar misses by exactly p_bar - p, whatever
    kappa (the palm ends at the aim; plate(s1) = p + kappa (g - h))."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        kappa = rng.uniform(-0.95, -0.05)
        p, p_bar, h = rng.normal(size=(3, 2))
        g = c1.fixed_point(p_bar, h, kappa)
        landed = p + kappa * (g - h)
        assert np.allclose(g - landed, p_bar - p)


def test_curve_binning_and_lookup():
    r15 = load()
    miss = np.array([0.1, 0.2, 0.3, 1.2, 1.4, 12.0])
    ok = np.array([True, False, True, True, True, False])
    curve = r15.tolerance_curve(miss, ok)
    assert [c["n"] for c in curve][:5] == [2, 1, 0, 2, 0]
    assert curve[0]["rate"] == 0.5 and curve[-1]["n"] == 1
    assert np.allclose(r15.expected_rate(curve, [0.0, 0.25, 1.49, 99.0]), [0.5, 1.0, 1.0, 0.0])
