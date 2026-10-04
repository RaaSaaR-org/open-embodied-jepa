"""The C1-M feasibility record (development only, before any protocol): constants, the move draw,
statistics with this record's salts, the check rules and the row ladder.

Ruling R15 (``docs/experiments/apple_lewm_next_v2_direction.md`` §5.3-§5.5, merged in #140);
record ``docs/experiments/apple_lewm_next_v2_c1m_feasibility.md``; declarations R16.1-R16.12 in
``docs/DECISIONS.md`` (decision 2026-10-05), each decided by Claude under owner delegation
(2026-09-30). Every value below was declared (commit ``8601c1b``) before any seed of the block was
simulated.

C1-M: C1 (a single aim committed at 405 under cell A's reactive-plate rule, kappa = -0.5, L = 2,
s0 = 405, s1 = 525, on v2's own reset) plus one declared change: a post-pick plate move at step 300,
an offset uniform over a disc (or the -y half-disc) of radius rho. The checks M-F1 to M-F7 run with
no world model; every privileged arm is a calculation from simulator truth, not a learned arm.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

RECORD = "apple_lewm_next_v2_c1m_feasibility"
RULING = "apple_lewm_next_v2_direction"
DECLARING_COMMIT = "8601c1b"
GuardError = pt.GuardError
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
NO_WORLD_MODEL = True

# ----- seeds and salts (R16.2) --------------------------------------------------------------------
SEED_BLOCK = (63000, 64999)
SEED_RANGES = {
    "M1": (63000, 63031),  # M-F1 at every radius and M-F2's reach at rho*
    "F3": (63100, 63131),  # M-F3 (fresh), M-F6, the first half of M-F5b
    "R": (63132, 63163),  # the second half of M-F5b
    "T": (63200, 63231),  # M-F4: tau_commit
    "corpus": (63300, 64323),  # M-F5a: 1 024 roots
}
DEBUG_SEEDS = (64900, 64999)  # mechanics only; nothing in them is read
DEBUG_RANGES = {
    "M1": (64900, 64905),
    "F3": (64910, 64915),
    "R": (64920, 64925),
    "T": (64930, 64935),
    "corpus": (64940, 64979),
}
DEBUG_STANDINS = {"rho_cm": 3, "family": "disc", "a_lo": -0.2, "r": 485, "tau_commit_cm": 1.0}
R15_BLOCK = (58000, 58999)  # R15.9: reset values only, but not reused
FORBIDDEN_RANGES = dict(c1.FORBIDDEN_RANGES) | {"c1_block": c1.SEED_BLOCK, "r15_block": R15_BLOCK}
SALTS = {
    "move": 7901,  # the move draw per reset (u, v; k the re-draw index)
    "corpus_aim": 7902,  # the corpus's uniform (a, b) draw per root
    "outer_folds": 7903,  # the readouts' cross-fit by root
    "inner_folds": 7904,  # lambda selection in every ridge
    "bootstrap": 7905,  # every bootstrap interval and the McNemar feasibility resampling
    "tau_direction": 7906,  # M-F4's planted-error direction per reset
    "learning_curve": 7907,  # the nested subsample, one permutation per outer fold
}
EARLIER_SALTS = {**c1.SALTS, "r15_learning_curve": 7801, "r15_move_draw": 7802}


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    spans = dict(SEED_RANGES) | {"debug": DEBUG_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two C1-M ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < SEED_BLOCK[0] or high > SEED_BLOCK[1]:
            raise GuardError(f"G-seeds: the range {name} leaves the block {SEED_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"G-seeds: the range {name} overlaps the forbidden {other}")
    for low, high in DEBUG_RANGES.values():
        if low < DEBUG_SEEDS[0] or high > DEBUG_SEEDS[1]:
            raise GuardError("G-seeds: a debug range leaves the debug block")
    used = {
        *EARLIER_SALTS.values(),
        *pt.SALTS.values(),
        *pt.RESERVED_SALTS.values(),
        *pt.CARRIED_SALTS.values(),
        *pt.lp.SEEDS.values(),
        *pt.oc.SEEDS.values(),
        *pt.lp.MODEL_SEEDS,
        pt.lp.DESIGN_PROBE_SALT,
    }
    if len(set(SALTS.values())) != len(SALTS) or set(SALTS.values()) & used:
        raise GuardError("G-seeds: a C1-M salt repeats or is an earlier task's salt")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """Plain ints; exactly the role's seeds in order (the debug role's in a debug run)."""
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


reset_of = c1.reset_of

# ----- the condition (R16.3) ---------------------------------------------------------------------
RULE = c1.RULE
COMMIT_STEP = c1.COMMIT_STEP
MOVE_STEP = 300
RHO_GRID_CM = (3, 4, 5, 6)  # TASK-073's |d| grid (wm_critic_v2.SHIFT_GRID_CM), used as radii
FAMILIES = ("disc", "half_disc")  # the disc first; the -y half-disc once if 3 cm fails
TABLETOP = wc.TABLETOP
REDRAW_LIMIT = 50
RESET_PLATE_CENTRE = tuple(float(v) for v in wc.RESET_CENTERS["plate_xy"])  # (0.49, -0.09)
if tuple(RHO_GRID_CM) != tuple(wc.SHIFT_GRID_CM):
    raise ContractError("the radius grid is TASK-073's |d| grid")


def _offset(u: float, v: float, family: str, rho_m: float) -> np.ndarray:
    if family == "disc":
        theta = 2.0 * math.pi * v
    elif family == "half_disc":
        theta = math.pi + math.pi * v  # sin(theta) <= 0: the -y half
    else:
        raise ContractError(f"unknown move family {family!r}")
    r = float(rho_m) * math.sqrt(u)
    return np.array([r * math.cos(theta), r * math.sin(theta)])


def _on_table(xy) -> bool:
    return bool(
        TABLETOP["x"][0] <= xy[0] <= TABLETOP["x"][1]
        and TABLETOP["y"][0] <= xy[1] <= TABLETOP["y"][1]
    )


def move_uniforms(seed: int) -> dict:
    """u, v from ``default_rng(SeedSequence([7901, seed, k]))``, k = 0 first; k grows while the
    reset's plate plus the offset at 6 cm in either family leaves the tabletop (R16.3)."""
    plate = np.asarray(reset_of(seed)["plate_xy"], np.float64)
    largest = max(RHO_GRID_CM) / 100.0
    for k in range(REDRAW_LIMIT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["move"], int(seed), k]))
        u, v = float(rng.uniform()), float(rng.uniform())
        if all(_on_table(plate + _offset(u, v, f, largest)) for f in FAMILIES):
            return {"u": u, "v": v, "k": k}
    raise GuardError(f"G-move: seed {seed} found no on-table draw in {REDRAW_LIMIT} re-draws")


def move_offset(seed: int, family: str, rho_cm: float) -> np.ndarray:
    """The reset's stored offset (world xy, m) at radius ``rho_cm`` in ``family``."""
    d = move_uniforms(seed)
    return _offset(d["u"], d["v"], family, float(rho_cm) / 100.0)


def moved_plate(seed: int, family: str, rho_cm: float) -> np.ndarray:
    """The plate at the decision (405): the reset's plate plus its offset (static 300-405)."""
    return np.asarray(reset_of(seed)["plate_xy"], np.float64) + move_offset(seed, family, rho_cm)


def plate_mean(family: str, rho_cm: float) -> np.ndarray:
    """p-bar: v2's reset centre plus the family's mean offset (0, or the half-disc's centroid
    4 rho / (3 pi) towards -y)."""
    centre = np.asarray(RESET_PLATE_CENTRE, np.float64)
    if family == "disc":
        return centre
    if family == "half_disc":
        return centre + np.array([0.0, -4.0 * (float(rho_cm) / 100.0) / (3.0 * math.pi)])
    raise ContractError(f"unknown move family {family!r}")


# ----- M-F4 and the corpus ------------------------------------------------------------------------
TAU_LEVELS_CM = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)  # R15 §5.3 (a subset of TASK-076 K0's levels)
TAU_BAR = 28  # of 32 (TASK-076 K0's rule)
if not set(TAU_LEVELS_CM) <= set(pt.TAU["levels_cm"]) or TAU_BAR != pt.TAU["bar"]:
    raise ContractError("tau_commit uses TASK-076 K0's rule on R15's levels")


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    """The planted error at ``level_cm`` in the reset's direction, angle 2 pi u (salt 7906)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["tau_direction"], int(seed)]))
    angle = float(2.0 * np.pi * rng.uniform())
    r = float(level_cm) / 100.0
    return [r * math.cos(angle), r * math.sin(angle)]


def corpus_aim(seed: int, a_lo: float) -> tuple[float, float]:
    """The corpus's (a, b): uniform over the box, from default_rng(SeedSequence([7902, seed]))."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    return float(rng.uniform(a_lo, c1.A_HI)), float(rng.uniform(-c1.B_HALF_M, c1.B_HALF_M))


# ----- the check bars (R15 §5.3; none is tuned after a result) ------------------------------------
F_RESETS = 32
CEILING_MIN = 30  # of 32 (R9.6's ceiling bar; M-F1 and the fresh ceiling of M-F3)
REMAINING_MIN_CM = c1.F1_REMAINING_MIN_CM  # 2.0
HEADROOM_BAR = 8  # of 32: each proxy's paired interval lower bound >= +8 (the strict reading)
FEASIBILITY_BAR = 0.8
FEASIBILITY_PAIRS = 64
FEASIBILITY_RESAMPLES = 10_000
MCNEMAR_P = 0.01
READ_RESETS = 64
READ_ALLOWANCE = 4  # of 64: ceiling - H-read <= 4/64 (half of the note's proposed delta, 8/64)
READ_DELTA = 8  # of 64: the note's proposed delta (M-READ-NONE needs lower bounds above it)
GRIDS = (4, 8)  # 4 x 4 first; 8 x 8 only if 4 x 4 misses
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
COST_SECONDS_MAX = 60.0
BOOTSTRAP_RESAMPLES = 10_000
OUTER_FOLDS = 5
INNER_FOLDS = 5
LAMBDA_GRID_RELATIVE = pt.LAMBDA_GRID_RELATIVE
CORPUS_ROOTS = 1024

PROXY_ARMS = c1.PROXY_ARMS  # H-now, N-proxy, shuf-proxy, mean-proxy
SCENE_BLIND_PROXIES = c1.SCENE_BLIND_PROXIES
COMPARATOR_ARMS = (
    "H-rule",
    "H-rule-true",
    "H-sysid",
    "H-sysid-true",
    "H-now-reaim",
    "H-rule-stale",
)
PRIVILEGED_ARMS = frozenset(
    c1.PRIVILEGED_ARMS | {"H-final-planted", "H-read", "H-rule-stale"}
)  # H-read and H-final-planted hold the robot (privileged look-ahead); H-rule-stale reads no image
ROWS = {
    "M-INFEASIBLE": "M-F1 has no passing radius in either family, M-F2 failed, or M-F3's fresh "
    "ceiling is below 30/32: escalate, no clause",
    "M-TWINS-NONE": "a scene-blind proxy's headroom is detectably below +8/32 (paired upper bound "
    "< +8): the clause fires (R15 §5.5)",
    "M-TWINS-ESCALATE": "M-F3 failed otherwise (a lower bound below +8, or a feasibility below "
    "0.8): escalate, no clause; the record stops",
    "M-NO-TAU": "tau_commit is undefined (level 0 below 28/32): escalate, no clause",
    "M-ARM-KEYED": "the 4 x 4 plate-hidden check failed: escalate, no clause",
    "M-READ-NONE": "H-read missed on both grids, both lower bounds above 8/64, and neither "
    "learning curve still falling: the clause fires (R15 §5.5)",
    "M-NO-BAR-DATA": "H-read missed on both grids and a learning curve is still falling: "
    "escalate to a larger corpus, no clause",
    "M-NO-BAR": "H-read missed on both grids otherwise: escalate, no clause",
    "M-PROCEED": "a preregistration may be drafted under R9.8 and R9.9",
}
CLAUSE_ROWS = frozenset({"M-TWINS-NONE", "M-READ-NONE"})


# ----- statistics (this record's salts) -----------------------------------------------------------
def paired_interval(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """C1's estimator (``pt.paired_interval``'s form) with salt 7905."""
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
    """R8.12's form with salt 7905."""
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
    v = np.asarray(values, np.float64)
    rng = np.random.default_rng(SALTS["bootstrap"])
    boot = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median": float(np.median(v)), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def median_difference_ci(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """median(first) - median(second), paired by root, with its bootstrap interval (salt 7905)."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.shape != b.shape:
        raise ContractError("a paired median difference needs the same roots")
    rng = np.random.default_rng(SALTS["bootstrap"])
    boot = np.empty(resamples)
    for i in range(resamples):
        j = rng.integers(0, len(a), len(a))
        boot[i] = np.median(a[j]) - np.median(b[j])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(np.median(a) - np.median(b)), "ci95": [float(lo), float(hi)]}


def outer_folds(n: int) -> np.ndarray:
    order = np.random.default_rng(SALTS["outer_folds"]).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


def nested_subsets(fit_index, fold: int) -> dict[float, np.ndarray]:
    """Per fraction, the first floor(f m + 0.5) of one permutation of the fold's fit rows (salt
    7907, per fold): nested across fractions, and the same rows for every grid."""
    fit_index = np.asarray(fit_index, np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["learning_curve"], int(fold)]))
    order = rng.permutation(fit_index)
    m = len(order)
    return {f: np.sort(order[: int(math.floor(f * m + 0.5))]) for f in FRACTIONS}


# ----- the check rules ----------------------------------------------------------------------------
def rho_star(counts: dict) -> int | None:
    """The largest radius such that it and every smaller one reach >= 30/32. ``counts``: radius
    (cm) -> count, the radii run in order (a radius after the first failure is never run)."""
    star = None
    for rho in RHO_GRID_CM:
        if rho not in counts or int(counts[rho]) < CEILING_MIN:
            break
        star = rho
    return star


def decide_f2(remaining_cm: list[float], r, palm_speed_cm: list[float], a_lo) -> dict:
    f1 = c1.decide_f1(remaining_cm, r, palm_speed_cm)
    return f1 | {
        "a_lo": a_lo,
        "a_lo_defined": a_lo is not None,
        "r_defined": r is not None,
        "passes": bool(f1["passes"] and a_lo is not None and r is not None),
    }


def decide_f3(ceiling, proxies: dict) -> dict:
    """The strict reading (R15.7): each proxy's paired interval lower bound >= +8/32; each
    scene-blind proxy's feasibility >= 0.8; a scene-blind proxy whose upper bound is below +8 is
    detectably below (M-TWINS-NONE). The fresh ceiling must reach 30/32 (R16.5)."""
    out = {}
    for arm, outcomes in proxies.items():
        interval = paired_interval(ceiling, outcomes)
        lo, hi = interval["ci95"]
        item = interval | {
            "count": int(np.sum(outcomes)),
            "headroom_passes": bool(lo >= HEADROOM_BAR),
            "detectably_below": bool(hi < HEADROOM_BAR),
        }
        if arm in SCENE_BLIND_PROXIES:
            item["feasibility"] = mcnemar_feasibility(ceiling, outcomes)
            item["feasibility_passes"] = bool(
                item["feasibility"]["predicted_pass_probability"] >= FEASIBILITY_BAR
            )
        item["passes"] = bool(item["headroom_passes"] and item.get("feasibility_passes", True))
        out[arm] = item
    ceiling_count = int(np.sum(ceiling))
    return {
        "ceiling": ceiling_count,
        "ceiling_ok": ceiling_count >= CEILING_MIN,
        "proxies": out,
        "twins_none": any(out[a]["detectably_below"] for a in SCENE_BLIND_PROXIES if a in out),
        "passes": ceiling_count >= CEILING_MIN and all(v["passes"] for v in out.values()),
    }


def decide_tau(counts: dict) -> dict:
    """tau_commit: the largest level such that every level up to it reaches 28/32 (None if level 0
    misses)."""
    by_level = {float(k): int(v) for k, v in counts.items()}
    if sorted(by_level) != sorted(TAU_LEVELS_CM):
        raise ContractError("tau_commit needs every declared level")
    tau = None
    for level in sorted(by_level):
        if by_level[level] >= TAU_BAR:
            tau = level
        else:
            break
    return {"counts": {str(k): v for k, v in sorted(by_level.items())}, "tau_commit_cm": tau}


def hidden_ok(hidden: dict, tau_commit_cm) -> bool | None:
    """O4's form against the measured tolerance; None when tau_commit is undefined."""
    if tau_commit_cm is None:
        return None
    return bool(hidden["ci95"][0] > float(tau_commit_cm))


def still_falling(guard: dict) -> bool:
    """TASK-075 §7's form: the lower bound of median(e_3/4) - median(e_all) is above 0."""
    return bool(guard["ci95"][0] > 0.0)


def decide_read(ceiling, read) -> dict:
    """M-F5b on one grid: the paired ceiling - H-read <= 4/64 (a point reading)."""
    interval = paired_interval(ceiling, read)
    return interval | {
        "count": int(np.sum(read)),
        "ceiling": int(np.sum(ceiling)),
        "passes": interval["difference"] <= READ_ALLOWANCE,
        "detectably_beyond_delta": interval["ci95"][0] > READ_DELTA,
    }


def verdict(
    *,
    rho: int | None,
    f2: dict | None,
    f3: dict | None,
    tau: dict | None = None,
    hidden4_ok: bool | None = None,
    reads: dict | None = None,
    falling: dict | None = None,
) -> dict:
    """R15.8's ladder plus R16.5. ``reads``: grid -> decide_read output (a grid not run is absent);
    ``falling``: grid -> bool."""
    reads, falling = reads or {}, falling or {}
    if rho is None or f2 is None or not f2["passes"] or f3 is None or not f3["ceiling_ok"]:
        row = "M-INFEASIBLE"
    elif f3["twins_none"]:
        row = "M-TWINS-NONE"
    elif not f3["passes"]:
        row = "M-TWINS-ESCALATE"
    elif tau is None or tau["tau_commit_cm"] is None:
        row = "M-NO-TAU"
    elif not hidden4_ok:
        row = "M-ARM-KEYED"
    elif any(reads.get(g, {}).get("passes", False) for g in GRIDS):
        row = "M-PROCEED"
    elif all(g in reads and reads[g]["detectably_beyond_delta"] for g in GRIDS) and not any(
        falling.get(g, False) for g in GRIDS
    ):
        row = "M-READ-NONE"
    elif any(falling.get(g, False) for g in GRIDS):
        row = "M-NO-BAR-DATA"
    else:
        row = "M-NO-BAR"
    return {"row": row, "meaning": ROWS[row], "clause_fires": row in CLAUSE_ROWS}
