"""TASK-082 design note, development only: candidate plate laws that no hand-written arm is given.

The design note ``docs/experiments/apple_lewm_unknown_law_v2_design.md`` asks for a condition in
which the plate's response to the robot's action is not handed to any non-privileged arm, so that
"LeWM needed" can be tested against learned, not hand-written, baselines (R19.23 (1), R19.24).
This module holds, as pure NumPy functions (no simulator, no torch):

* the **candidate laws**, each a map from the executed palm path after the commit step to the
  plate's displacement from its position at the commit step: ``c1m`` (TASK-077's cell-A rule,
  kappa = -0.5, L = 2, the within-run reference), ``sat`` (a saturating, swirling static law) and
  ``play`` (a hysteretic play operator on a linear drive, history-dependent). Their parameters were
  fixed here from TASK-077's recorded plate motion (about 8.4 cm from 405 to r, so a palm travel of
  about 17 cm at kappa = -0.5) **before any seed of the development block was simulated**;
* the **seed ranges and salts** of the development check (block 72000-74999, R19.24) and the
  move draw (C1-M's disc of radius 4 cm at step 300, with a fresh salt);
* the **system-identification baselines**: TASK-080's linear form (``lewm_next_c1.sysid_fit``,
  reused unchanged) and an RBF kernel ridge on the same inputs (a richer, learned, non-world-model
  class), with grouped cross-fitting;
* the bootstrap estimators of the record (salt 8403).

Nothing here is a gated arm, nothing is frozen, and no earlier task's module is edited.
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import lewm_cp_v2 as cp
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa.contracts import ContractError

DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
NO_WORLD_MODEL = True

# ----- seeds and salts (R19.24) -------------------------------------------------------------------
SEED_BLOCK = (72000, 74999)  # TASK-082's block; this record uses only the ranges below
SEED_RANGES = {
    "F": (72000, 72031),  # the check cohort: every arm under every law, same 32 resets
    "corpus": (72100, 72355),  # 256 roots of uniform box aims, the same roots under every law
}
DEBUG_SEEDS = (74900, 74999)  # mechanics only; nothing in them is read
DEBUG_RANGES = {"F": (74900, 74903), "corpus": (74910, 74925)}
RESERVED = (72356, 74899)  # everything else in the block: left for a protocol, any use is ruled
# every earlier forbidden range (TASK-081's list, which carries all of them) plus TASK-081's block
FORBIDDEN_RANGES = dict(cp.FORBIDDEN_RANGES) | {"task081_block": cp.SEED_BLOCK}
SALTS = {
    "move": 8401,  # default_rng(SeedSequence([8401, seed, k])), k the re-draw index
    "corpus_aim": 8402,  # the corpus's uniform (a, b) per root
    "bootstrap": 8403,  # every bootstrap interval (10 000 resamples)
    "folds": 8404,  # the sysid cross-fit's 5 folds and the kernel ridge's inner selection
}
MOVE_RHO_CM = 4  # C1-M's rho*, disc family (R16.3), unchanged
MOVE_FAMILY = "disc"
A_LO = -0.5  # C1-M's a_lo (M-F2), unchanged; the box and grid are lewm_next_c1's
BOOTSTRAP_RESAMPLES = 10_000
FOLDS = 5


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in the block, are disjoint, and avoid every forbidden range."""
    spans = dict(SEED_RANGES) | {"debug": DEBUG_SEEDS}
    for name, (low, high) in spans.items():
        if not SEED_BLOCK[0] <= low <= high <= SEED_BLOCK[1]:
            raise ContractError(f"G-seeds: {name} leaves the block {SEED_BLOCK}")
        for other, (o_low, o_high) in FORBIDDEN_RANGES.items():
            if low <= o_high and o_low <= high:
                raise ContractError(f"G-seeds: {name} overlaps {other}")
    ordered = sorted(spans.values())
    for (_l1, h1), (l2, _h2) in zip(ordered, ordered[1:], strict=False):
        if l2 <= h1:
            raise ContractError("G-seeds: two ranges overlap")
    for role, (low, high) in DEBUG_RANGES.items():
        if not DEBUG_SEEDS[0] <= low <= high <= DEBUG_SEEDS[1]:
            raise ContractError(f"G-seeds: debug {role} leaves the debug range")


def check_seeds(seeds, *, debug: bool = False) -> None:
    allowed = DEBUG_RANGES if debug else SEED_RANGES
    for s in seeds:
        if not any(low <= int(s) <= high for low, high in allowed.values()):
            raise ContractError(f"G-seeds: seed {s} lies in no declared range")


def reset_of(seed: int) -> dict:
    return c1m.reset_of(int(seed))


def move_offset(seed: int) -> np.ndarray:
    """C1-M's disc move at rho = 4 cm (R16.3's draw and re-draw rule) with salt 8401."""
    plate = np.asarray(reset_of(seed)["plate_xy"], np.float64)
    largest = max(c1m.RHO_GRID_CM) / 100.0
    for k in range(c1m.REDRAW_LIMIT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["move"], int(seed), k]))
        u, v = float(rng.uniform()), float(rng.uniform())
        if all(c1m._on_table(plate + c1m._offset(u, v, f, largest)) for f in c1m.FAMILIES):
            return c1m._offset(u, v, MOVE_FAMILY, MOVE_RHO_CM / 100.0)
    raise ContractError(f"G-move: seed {seed} found no on-table draw")


def corpus_aim(seed: int) -> tuple[float, float]:
    """(a, b) uniform over the box [a_lo, a_hi] x [-3, 3] cm, salt 8402."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    a = float(rng.uniform(A_LO, c1.A_HI))
    b = float(rng.uniform(-c1.B_HALF_M, c1.B_HALF_M))
    return a, b


# ----- the candidate laws -------------------------------------------------------------------------
S0, S1, LAG = c1.RULE["s0"], c1.RULE["s1"], c1.RULE["L"]  # 405, 525, 2: the window is C1-M's
LAWS = {
    "c1m": {"kappa": -0.5},  # TASK-077's cell A (the reference; H-rule is given exactly this)
    "sat": {"amplitude_m": 0.08, "scale_m": 0.12, "swirl_rad": 0.25},
    "play": {"kappa": -0.6, "width_m": 0.03},
}


def _rotate(v, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def sat_displacement(d, *, amplitude_m: float, scale_m: float, swirl_rad: float) -> np.ndarray:
    """-A tanh(|d|/D) R(theta |d|/D) d/|d|: toward the hand, saturating at A, and turned by an
    angle that grows with the palm's travel (memoryless, nonlinear)."""
    d = np.asarray(d, np.float64).reshape(2)
    n = float(np.linalg.norm(d))
    if n == 0.0:
        return np.zeros(2)
    x = n / scale_m
    return -amplitude_m * math.tanh(x) * _rotate(d / n, swirl_rad * x)


def play_step(y, z, width_m: float) -> np.ndarray:
    """One step of the 2-D play (backlash) operator: y stays until |z - y| exceeds the width, then
    is dragged so that |z - y| = width."""
    y, z = np.asarray(y, np.float64), np.asarray(z, np.float64)
    gap = z - y
    n = float(np.linalg.norm(gap))
    if n <= width_m:
        return y.copy()
    return z - width_m * gap / n


def displacement(law: str, palm: dict, step: int) -> np.ndarray:
    """The plate's displacement from its position at s0 at ``step``, from the executed palm xy
    history ``palm`` (step -> xy). Before s0 it is 0; after s1 it is held."""
    t = min(int(step), S1)
    if t <= S0:
        return np.zeros(2)
    u = max(t - LAG, S0)
    ref = np.asarray(palm[S0], np.float64)
    p = LAWS[law]
    if law == "c1m":
        return p["kappa"] * (np.asarray(palm[u], np.float64) - ref)
    if law == "sat":
        return sat_displacement(np.asarray(palm[u], np.float64) - ref, **p)
    if law == "play":
        y = np.zeros(2)
        for k in range(S0 + 1, u + 1):
            y = play_step(y, p["kappa"] * (np.asarray(palm[k], np.float64) - ref), p["width_m"])
        return y
    raise ContractError(f"unknown plate law {law!r}")


# ----- system identification ----------------------------------------------------------------
def krr_features(p, h, g) -> np.ndarray:
    p, h, g = (np.asarray(v, np.float64).reshape(-1, 2) for v in (p, h, g))
    return np.concatenate([p, h, g], axis=1)


KRR_LENGTHS = (0.5, 1.0, 2.0, 4.0)  # in standardised units
KRR_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


def _kernel(a, b, length: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / (length * length))


def krr_fit_fixed(x, y, length: float, lam: float) -> dict:
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-12
    xs = (x - mean) / std
    k = _kernel(xs, xs, length)
    alpha = np.linalg.solve(k + lam * len(xs) * np.eye(len(xs)), y)
    return {"x": xs, "alpha": alpha, "mean": mean, "std": std, "length": length, "lambda": lam}


def krr_predict(model: dict, x) -> np.ndarray:
    xs = (np.asarray(x, np.float64) - np.asarray(model["mean"])) / np.asarray(model["std"])
    k = _kernel(xs, np.asarray(model["x"]), float(model["length"]))
    return k @ np.asarray(model["alpha"])


def fold_of(n: int, salt_key: int = 0) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["folds"], salt_key, n]))
    return rng.permutation(np.arange(n) % FOLDS)


def krr_fit(p, h, g, plate_final) -> dict:
    """RBF kernel ridge of the displacement plate(s1) - p on standardised (p, h, g); length and
    lambda by 5-fold CV (salt 8404) on the median error."""
    x = krr_features(p, h, g)
    y = np.asarray(plate_final, np.float64) - np.asarray(p, np.float64)
    fold = fold_of(len(x), 1)
    best = None
    for length in KRR_LENGTHS:
        for lam in KRR_LAMBDAS:
            err = np.empty(len(x))
            for k in range(FOLDS):
                fit, held = fold != k, fold == k
                m = krr_fit_fixed(x[fit], y[fit], length, lam)
                err[held] = np.linalg.norm(krr_predict(m, x[held]) - y[held], axis=1)
            score = float(np.median(err))
            if best is None or score < best[0]:
                best = (score, length, lam)
    model = krr_fit_fixed(x, y, best[1], best[2])
    model["cv_median_m"] = best[0]
    return model


def krr_plate(model: dict, p, h, g) -> np.ndarray:
    return np.asarray(p, np.float64).reshape(-1, 2) + krr_predict(model, krr_features(p, h, g))


def model_to_json(model: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}


def cross_fitted_errors(p, h, g, plate_final) -> dict:
    """Out-of-fold |prediction - plate(s1)| (cm) for the linear sysid, the kernel ridge and the
    C1-M rule's open form p + kappa (g - h) (the palm assumed to end at the aim)."""
    p, h, g, y = (np.asarray(v, np.float64).reshape(-1, 2) for v in (p, h, g, plate_final))
    fold = fold_of(len(p), 2)
    out = {k: np.full(len(p), np.nan) for k in ("linear", "krr")}
    for k in range(FOLDS):
        fit, held = fold != k, fold == k
        coef = c1.sysid_fit(p[fit], h[fit], g[fit], y[fit])
        out["linear"][held] = 100 * np.linalg.norm(
            c1.sysid_predict(coef, p[held], h[held], g[held]) - y[held], axis=1
        )
        m = krr_fit(p[fit], h[fit], g[fit], y[fit])
        out["krr"][held] = 100 * np.linalg.norm(krr_plate(m, p[held], h[held], g[held]) - y[held],
                                                axis=1)  # fmt: skip
    out["c1m_rule_open"] = 100 * np.linalg.norm(p + LAWS["c1m"]["kappa"] * (g - h) - y, axis=1)
    return {k: v.tolist() for k, v in out.items()}


# ----- statistics (salt 8403) ---------------------------------------------------------------
def median_ci(values) -> dict:
    v = np.asarray(values, np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["bootstrap"], 1, len(v)]))
    boot = np.median(v[rng.integers(0, len(v), (BOOTSTRAP_RESAMPLES, len(v)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "median": float(np.median(v)),
        "ci95": [float(lo), float(hi)],
        "p87_5": float(np.percentile(v, 87.5)),
        "max": float(v.max()),
        "n": int(len(v)),
    }


def paired_interval(first, second) -> dict:
    """first - second in resets, with the reset-bootstrap 95 % interval and discordant counts."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ContractError("paired outcomes need the same resets")
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["bootstrap"], 2, len(d)]))
    boot = d[rng.integers(0, len(d), (BOOTSTRAP_RESAMPLES, len(d)))].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "difference": int(d.sum()),
        "ci95": [float(lo), float(hi)],
        "only_first": int(((a) & (~b)).sum()),
        "only_second": int(((~a) & (b)).sum()),
        "n": int(len(d)),
    }
