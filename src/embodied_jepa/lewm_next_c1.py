"""The C1 feasibility record (development only, before any protocol): constants, geometry,
statistics and the declared check rules.

Design note ``docs/experiments/apple_lewm_next_v2_design.md`` §4.1 (DRAFT, R9); record
``docs/experiments/apple_lewm_next_v2_c1_feasibility.md``; rulings R14.1-R14.n in
``docs/DECISIONS.md`` (decision 2026-10-04 (c)), each decided by Claude under owner delegation
(2026-09-30). Every value below was declared before any seed of the block was simulated.

C1: a single aim committed at 405 under cell A's reactive-plate rule (kappa = -0.5, L = 2,
s0 = 405, s1 = 525) on ``apple-to-plate-v2``'s own reset (TASK-047's wide jitter, no step-300
shift). The checks C1-F1 to C1-F6 run with no world model; every privileged proxy is a
calculation from simulator truth, not a learned arm. Nothing here is a protocol or a gate of a
learned arm.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

RECORD = "apple_lewm_next_v2_c1_feasibility"
DESIGN_NOTE = "apple_lewm_next_v2_design"
GuardError = pt.GuardError
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
NO_WORLD_MODEL = True

# ----- seeds and salts (declared before any run; R14.1) -----------------------------------------
# Searched on 2026-10-04 at 82da722 over every ref (68 local and remote branches), src, scripts,
# tests, configs, benchmarks, docs and .mc, and the other worktrees: no seed in 57000-57999 (the
# only matches are a wall-cap constant 57600.0 and a byte count in uv.lock), and no use of
# 7701-7709 in src, scripts, tests or configs.
SEED_BLOCK = (57000, 57999)
SEED_RANGES = {
    "F": (57000, 57031),  # C1-F1, F2, F3, F4, F6: the 32 paired closed-loop resets
    "corpus": (57100, 57355),  # C1-F5 and H-sysid: 256 roots of the declared-aim smoke corpus
}
DEBUG_SEEDS = (57900, 57999)  # mechanics debugging only; nothing in them is read
FORBIDDEN_RANGES = dict(pt.FORBIDDEN_RANGES) | {"task076_block": pt.TASK_BLOCK}
SALTS = {
    "corpus_aim": 7701,  # the corpus's uniform (a, b) draw per root
    "outer_folds": 7702,  # C1-F5's cross-fit by root
    "inner_folds": 7703,  # lambda selection in every ridge (R-plate, R-plate-pool)
    "bootstrap": 7704,  # every bootstrap interval and the McNemar feasibility resampling
}
RESET = (
    "wm_critic_v2.wide_reset_values(seed): v2's own reset (plate (0.49, -0.09) +- 2 cm, apple "
    "(0.34, -0.18) +- 3 cm, uniform per axis); no step-300 shift, no re-draw rule, no jitter knob"
)


def seeds_of(role: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    spans = dict(SEED_RANGES) | {"debug": DEBUG_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two C1 ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < SEED_BLOCK[0] or high > SEED_BLOCK[1]:
            raise GuardError(f"G-seeds: the range {name} leaves the block {SEED_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"G-seeds: the range {name} overlaps the forbidden {other}")
    used = {
        *pt.SALTS.values(),
        *pt.RESERVED_SALTS.values(),
        *pt.CARRIED_SALTS.values(),
        *pt.lp.SEEDS.values(),
        *pt.oc.SEEDS.values(),
        *pt.lp.MODEL_SEEDS,
        pt.lp.DESIGN_PROBE_SALT,
    }
    if len(set(SALTS.values())) != len(SALTS) or set(SALTS.values()) & used:
        raise GuardError("G-seeds: a C1 salt repeats or is an earlier task's salt")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """Plain ints; exactly the role's seeds in order, or distinct debug seeds in a debug run."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    if debug:
        low, high = DEBUG_SEEDS
        if any(not low <= s <= high for s in seeds) or len(set(seeds)) != len(seeds):
            raise GuardError(f"G-seeds: a debug run simulates {low}-{high} only")
        return seeds
    if seeds != seeds_of(role):
        raise GuardError(f"G-seeds: {role} simulates exactly {SEED_RANGES[role]}, once, in order")
    return seeds


def reset_of(seed: int) -> dict:
    r = wc.wide_reset_values(int(seed))
    return {
        "object_xy": [float(v) for v in r["object_xy"]],
        "plate_xy": [float(v) for v in r["plate_xy"]],
    }


# ----- the condition (design note §4.1) ---------------------------------------------------------
RULE = {"name": "A", "kappa": pt.CELL_A["kappa"], "L": pt.CELL_A["L"], "s0": 405, "s1": 525}
COMMIT_STEP = pt.TRANSFER_START  # 405: the single committed aim
TAU_RE_CM = float(pt.K0_MEASURED["tau_re_cm"])  # 1.0 (TASK-076 K0, R8.19)
LOOKAHEAD_TOLERANCE_M = pt.LOOKAHEAD["tolerance_fraction_of_tau"] * TAU_RE_CM / 100.0  # 0.25 cm
MAX_ITERATIONS = pt.LOOKAHEAD["max_iterations"]  # 10
PLATE_MEAN = tuple(float(v) for v in wc.RESET_CENTERS["plate_xy"])  # p-bar: the reset's mean

# ----- the candidate box and grid (design note §4.1, declared before any smoke) ------------------
A_HI = 0.5
A_LEVELS = (-0.5, -0.4, -0.3, -0.2)  # a_lo candidates, most negative first
B_HALF_M = 0.03
A_STEP = 0.05
B_STEP_M = 0.01
REACH_B_M = (-0.03, 0.0, 0.03)  # R14.4: a reset is reachable at a level if all three complete
REACH_MIN = 31  # of 32: at least 31/32 transfers complete
TRANSFER_END = pt.LOWER_START  # 505: a transfer is complete when commands 405-504 executed

# ----- the check bars (design note §4.1 table; none is tuned after a result) --------------------
F1_REMAINING_MIN_CM = 2.0  # TASK-076's bar
R_TOLERANCE_CM = 0.1  # r: within 0.1 cm of plate(s1) ...
R_FRACTION = 0.875  # ... on at least 87.5 % of the smoke attempts
PALM_SPEED_LIMIT_CM = 0.5  # per step: above it the protocol gives W and N the last 2 commands
F2_CEILING_MIN = 30  # of 32
F2_REFUSED_MAX = 8  # of 32 (TASK-076's removal rule: a quarter)
F_RESETS = 32
HEADROOM_BAR = 8  # of 32: each proxy <= ceiling - 8/32
FEASIBILITY_BAR = 0.8
FEASIBILITY_PAIRS = 64  # the protocol's gated cohort
FEASIBILITY_RESAMPLES = 10_000
MCNEMAR_P = 0.01
F5_READOUT_MAX_CM = TAU_RE_CM / 2.0  # 0.5: the median error at r (point estimate; R14.6)
F5_HIDDEN_MIN_CM = TAU_RE_CM  # 1.0: the lower bound of the plate-hidden median error
F6_SECONDS_MAX = 60.0  # TASK-076's cap-review rule
BOOTSTRAP_RESAMPLES = 10_000
OUTER_FOLDS = 5
INNER_FOLDS = 5
LAMBDA_GRID_RELATIVE = pt.LAMBDA_GRID_RELATIVE
CORPUS_ROOTS = 256

# ----- arms ---------------------------------------------------------------------------------------
PROXY_ARMS = ("H-now", "N-proxy", "shuf-proxy", "mean-proxy")  # C1-F3
SCENE_BLIND_PROXIES = ("shuf-proxy", "mean-proxy")  # each also needs the McNemar feasibility
COMPARATOR_ARMS = ("H-rule", "H-rule-true", "H-sysid", "H-sysid-true", "H-now-reaim")  # C1-F4
PRIVILEGED_ARMS = frozenset(
    {"H-final", "H-now", "N-proxy", "shuf-proxy", "mean-proxy", "reach", "collect"}
    | {"H-rule-true", "H-sysid-true", "H-now-reaim"}
)
ROWS = {
    "C1-INFEASIBLE": "C1-F1 or C1-F2 failed: C1 is infeasible; escalate, no clause",
    "C1-TWINS-ESCALATE": "C1-F3 failed (a proxy not below its bar, or a scene-blind feasibility "
    "below 0.8): escalate before any protocol, no clause; for L-mean the R9.10 remedies are the "
    "options and none is chosen",
    "C1-ARM-KEYED": "C1-F5's plate-hidden check failed: the readout at r may be keyed on the arm; "
    "escalate, no clause",
    "C1-NO-BAR": "C1-F5's readout at r missed its estimate: no B is feasible (NO-BAR); "
    "escalate, no clause",
    "C1-PROCEED": "C1-F1, F2, F3 and F5 passed: a preregistration may be drafted (R9.6), with its "
    "own K0, Stage O and independent review",
}


# ----- geometry -----------------------------------------------------------------------------------
def _frame(p, h):
    p, h = np.asarray(p, np.float64).reshape(2), np.asarray(h, np.float64).reshape(2)
    u = h - p
    norm = float(np.linalg.norm(u))
    if norm < 1e-6:
        raise ContractError("the palm sits on the plate: the box is undefined")
    e = u / norm
    return p, u, np.array([-e[1], e[0]])


def from_box(a: float, b_m: float, p, h) -> np.ndarray:
    """g = p + a (h - p) + b n, n the unit vector perpendicular to h - p (rotated +90 deg)."""
    p, u, n = _frame(p, h)
    return p + float(a) * u + float(b_m) * n


def to_box(g, p, h) -> tuple[float, float]:
    p, u, n = _frame(p, h)
    d = np.asarray(g, np.float64).reshape(2) - p
    return float(d @ u / (u @ u)), float(d @ n)


def clip_to_box(g, p, h, a_lo: float) -> np.ndarray:
    a, b = to_box(g, p, h)
    return from_box(min(max(a, a_lo), A_HI), min(max(b, -B_HALF_M), B_HALF_M), p, h)


def grid(a_lo: float) -> list[tuple[float, float]]:
    """The controller's grid: a from a_lo to 0.5 in 0.05, b from -3 to +3 cm in 1 cm."""
    n_a = int(round((A_HI - a_lo) / A_STEP)) + 1
    a_values = [round(a_lo + i * A_STEP, 10) for i in range(n_a)]
    b_values = [
        round(-B_HALF_M + j * B_STEP_M, 10) for j in range(int(round(2 * B_HALF_M / B_STEP_M)) + 1)
    ]
    return [(a, b) for a in a_values for b in b_values]


def fixed_point(p, h, kappa: float = RULE["kappa"]) -> np.ndarray:
    """(p - kappa h) / (1 - kappa): the landing point if the palm ends at the aim."""
    p, h = np.asarray(p, np.float64).reshape(2), np.asarray(h, np.float64).reshape(2)
    return (p - kappa * h) / (1.0 - kappa)


def n_proxy_a(a_lo: float) -> float:
    """a_N = min(0.5 (1 - a_m), a_hi), a_m = (a_lo + a_hi) / 2 (the corpus-mean outcome)."""
    a_m = (float(a_lo) + A_HI) / 2.0
    return min(0.5 * (1.0 - a_m), A_HI)


def proxy_aim(arm: str, p, h, a_lo: float, *, foreign_p=None) -> np.ndarray:
    """C1-F3's privileged proxies (true p and h at 405), clipped to the box as every arm that
    chooses from candidates is (R14.5)."""
    if arm == "H-now":
        return np.asarray(p, np.float64).reshape(2).copy()
    if arm == "N-proxy":
        return from_box(n_proxy_a(a_lo), 0.0, p, h)
    if arm == "shuf-proxy":
        if foreign_p is None:
            raise ContractError("the shuf-proxy needs the foreign reset's plate")
        return clip_to_box(fixed_point(foreign_p, h), p, h, a_lo)
    if arm == "mean-proxy":
        return clip_to_box(fixed_point(PLATE_MEAN, h), p, h, a_lo)
    raise ContractError(f"unknown proxy {arm!r}")


def corpus_aim(seed: int, a_lo: float) -> tuple[float, float]:
    """The corpus's (a, b): uniform over the box, from default_rng(SeedSequence([7701, seed]))."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    a = float(rng.uniform(a_lo, A_HI))
    b = float(rng.uniform(-B_HALF_M, B_HALF_M))
    return a, b


def foreign_index(i: int, n: int) -> int:
    return (int(i) + 1) % int(n)


# ----- H-sysid (a linear regression; not a world model) -------------------------------------------
def sysid_design(p, h, g) -> np.ndarray:
    p, h, g = (np.asarray(v, np.float64).reshape(-1, 2) for v in (p, h, g))
    return np.concatenate([np.ones((len(p), 1)), p, h, g], axis=1)


def sysid_fit(p, h, g, y) -> np.ndarray:
    """plate(r) ~ 1 + p + h + g, least squares: coefficients [7, 2]."""
    coef, *_ = np.linalg.lstsq(
        sysid_design(p, h, g), np.asarray(y, np.float64).reshape(-1, 2), rcond=None
    )
    return coef


def sysid_predict(coef, p, h, g) -> np.ndarray:
    return (sysid_design(p, h, g) @ np.asarray(coef, np.float64)).reshape(-1, 2)


def choose_aim(
    predict, p, h, a_lo: float, *, tolerance_m=LOOKAHEAD_TOLERANCE_M, max_iter=MAX_ITERATIONS
):
    """W's controller form (design note §4.1), for any plate predictor ``predict(g) -> xy``: score
    every grid candidate by |predict(g) - g|, take the best, then refine g <- clip(predict(g)) at
    most ``max_iter`` times, stopping at ``tolerance_m``."""
    candidates = [from_box(a, b, p, h) for a, b in grid(a_lo)]
    scores = [float(np.linalg.norm(predict(g) - g)) for g in candidates]
    best = int(np.argmin(scores))
    g = candidates[best]
    steps, converged = [], False
    for _ in range(int(max_iter)):
        nxt = clip_to_box(predict(g), p, h, a_lo)
        gap = float(np.linalg.norm(nxt - g))
        steps.append({"g": nxt.tolist(), "gap_cm": 100.0 * gap})
        g = nxt
        if gap <= tolerance_m:
            converged = True
            break
    log = {
        "grid_best": list(grid(a_lo)[best]),
        "grid_best_score_cm": 100.0 * scores[best],
        "iterations": steps,
        "converged": converged,
    }
    return g, log


# ----- statistics (C1's own salt) -----------------------------------------------------------------
def paired_interval(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """pt.paired_interval's estimator with C1's bootstrap salt (7704)."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ContractError("paired outcomes need the same resets")
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(SALTS["bootstrap"])
    boot = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    only_first, only_second = pt.discordant(a, b)
    return {
        "difference": int(d.sum()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(d)),
        "only_first": only_first,
        "only_second": only_second,
    }


def mcnemar_feasibility(ceiling, proxy, *, resamples: int = FEASIBILITY_RESAMPLES) -> dict:
    """R8.12's form: the exact one-sided McNemar p on the doubled discordant counts, and the
    fraction of resamples of 64 pairs (with replacement from the 32, salt 7704) with p < 0.01."""
    a, c_ = np.asarray(ceiling, bool), np.asarray(proxy, bool)
    b, c = pt.discordant(a, c_)
    rng = np.random.default_rng(SALTS["bootstrap"])
    passes = 0
    for _ in range(resamples):
        i = rng.integers(0, len(a), FEASIBILITY_PAIRS)
        bb, cc = pt.discordant(a[i], c_[i])
        passes += pt.mcnemar_one_sided(bb, cc) < MCNEMAR_P
    return {
        "b": b,
        "c": c,
        "point_p_doubled": pt.mcnemar_one_sided(2 * b, 2 * c),
        "predicted_pass_probability": passes / resamples,
        "resamples": int(resamples),
    }


def median_ci(values, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The median with its bootstrap 95 % percentile interval over i.i.d. rows (one row per root,
    so the root-clustered bootstrap is the plain one), salt 7704."""
    v = np.asarray(values, np.float64)
    rng = np.random.default_rng(SALTS["bootstrap"])
    boot = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median": float(np.median(v)), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def outer_folds(n: int) -> np.ndarray:
    """5 outer folds by root: a seeded permutation (salt 7702), position mod 5."""
    order = np.random.default_rng(SALTS["outer_folds"]).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


# ----- the check rules ----------------------------------------------------------------------------
def read_step(distances: list[dict | None], s0: int = RULE["s0"], s1: int = RULE["s1"]):
    """r: the earliest step t in [s0, s1] at which |plate(t) - plate(s1)| <= 0.1 cm on at least
    87.5 % of the attempts (R14.3: every attempt counts; a refused attempt, ``None``, is never
    within)."""
    need = math.ceil(R_FRACTION * len(distances) - 1e-12)
    for t in range(int(s0), int(s1) + 1):
        within = sum(1 for d in distances if d is not None and d.get(t, math.inf) <= R_TOLERANCE_CM)
        if within >= need:
            return t
    return None


def decide_a_lo(complete_by_level: dict) -> float | None:
    """R14.4: a_lo is the most negative level L such that every level >= L (of A_LEVELS) has at
    least 31/32 complete transfers; None if -0.2 itself fails."""
    a_lo = None
    for level in sorted(A_LEVELS, reverse=True):  # -0.2, -0.3, -0.4, -0.5
        if int(complete_by_level[level]) >= REACH_MIN:
            a_lo = level
        else:
            break
    return a_lo


def decide_f1(remaining_cm: list[float], r, palm_speed_cm: list[float]) -> dict:
    median = float(np.median(remaining_cm)) if remaining_cm else None
    speed = float(np.median(palm_speed_cm)) if palm_speed_cm else None
    return {
        "median_remaining_cm": median,
        "passes": median is not None and median >= F1_REMAINING_MIN_CM,
        "r": r,
        "horizon_steps": None if r is None else int(r) - COMMIT_STEP,
        "median_palm_speed_405_cm_per_step": speed,
        "last_two_commands_required": speed is not None and speed > PALM_SPEED_LIMIT_CM,
    }


def decide_f2(count: int, refused: int, a_lo) -> dict:
    return {
        "ceiling": int(count),
        "refused": int(refused),
        "a_lo": a_lo,
        "ceiling_ok": int(count) >= F2_CEILING_MIN,
        "refused_ok": int(refused) <= F2_REFUSED_MAX,
        "a_lo_defined": a_lo is not None,
        "passes": int(count) >= F2_CEILING_MIN
        and int(refused) <= F2_REFUSED_MAX
        and a_lo is not None,
    }


def decide_f3(ceiling, proxies: dict) -> dict:
    """Each proxy: the paired headroom (ceiling - proxy) >= +8/32 passes (K-P2's form); a failed
    one is 'detectably below' when its interval's upper bound is below 8/32 and 'near' otherwise
    (R8.14's guard, R14.5). The scene-blind proxies also need a feasibility >= 0.8."""
    out = {}
    for arm, outcomes in proxies.items():
        interval = paired_interval(ceiling, outcomes)
        passes = interval["difference"] >= HEADROOM_BAR
        item = interval | {
            "count": int(np.sum(outcomes)),
            "headroom_passes": passes,
            "detectably_below": (not passes) and interval["ci95"][1] < HEADROOM_BAR,
            "near": (not passes) and interval["ci95"][1] >= HEADROOM_BAR,
        }
        if arm in SCENE_BLIND_PROXIES:
            item["feasibility"] = mcnemar_feasibility(ceiling, outcomes)
            item["feasibility_passes"] = (
                item["feasibility"]["predicted_pass_probability"] >= FEASIBILITY_BAR
            )
        item["passes"] = passes and item.get("feasibility_passes", True)
        out[arm] = item
    return {"proxies": out, "passes": all(v["passes"] for v in out.values())}


def decide_f5(visible: dict, hidden: dict) -> dict:
    readout_ok = visible["median"] <= F5_READOUT_MAX_CM
    hidden_ok = hidden["ci95"][0] > F5_HIDDEN_MIN_CM
    return {
        "readout_at_r": visible,
        "plate_hidden_at_r": hidden,
        "readout_ok": bool(readout_ok),
        "hidden_ok": bool(hidden_ok),
        "passes": bool(readout_ok and hidden_ok),
    }


def decide_f6(seconds: list[float]) -> dict:
    worst = float(np.max(seconds))
    return {
        "max_seconds": worst,
        "median_seconds": float(np.median(seconds)),
        "passes": worst <= F6_SECONDS_MAX,
        "consequence": None if worst <= F6_SECONDS_MAX else "the per-attempt cap is reviewed",
    }


def verdict(f1: dict, f2: dict, f3: dict | None, f5: dict | None) -> dict:
    """The note's failure rows, in its order: F1/F2, then F3, then F5's plate-hidden check, then
    F5's readout. C1-F4 has no bar and C1-F6 only reviews the cap."""
    if not (f1["passes"] and f2["passes"]):
        row = "C1-INFEASIBLE"
    elif f3 is None or not f3["passes"]:
        row = "C1-TWINS-ESCALATE"
    elif f5 is None or not f5["hidden_ok"]:
        row = "C1-ARM-KEYED"
    elif not f5["readout_ok"]:
        row = "C1-NO-BAR"
    else:
        row = "C1-PROCEED"
    return {"row": row, "meaning": ROWS[row], "clause_fires": False}
