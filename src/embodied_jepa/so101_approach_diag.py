"""TASK-102 Δa (``docs/experiments/so101_approach_diagnostic_v1.md``): the development
decomposition of S0-APPROACH -- cohort, seeds and salts, the line states, the candidate sets, the
planner arms, the statistics (Spearman ρ, band ordering, d_flat), the per-step projection chain,
the oracle pose cost and the first-match rows.

NumPy only at import. TASK-101's and TASK-102's modules are imported, never edited. Nothing here
is registered or imported by ``embodied_jepa``. Simulator truth enters only the goal
demonstration, the placed line states, the privileged oracle arms and the reported logs.
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid as hy
from embodied_jepa.contracts import ContractError

PROTOCOL = "docs/experiments/so101_approach_diagnostic_v1.md"
TASK = "TASK-102"
STUDY = "Δa"
GuardError = cl.GuardError
Cohort = cl.Cohort

# ----- cohorts, seeds and salts (§1) --
COHORTS = {
    "L": Cohort("L", 104000, 16, (104016, 104031)),
    "debug": Cohort("debug", 104900, 10, None),  # plumbing and timing only; never counted
}
L2_SLOTS = 8  # Λ2 runs on L's slots 0-7
MIN_KEPT = 12
SALTS = {"cem": 10116, "candidates": 10117}


def cohort(name: str) -> Cohort:
    if name not in COHORTS:
        raise ContractError(f"unknown cohort {name!r}")
    return COHORTS[name]


def check_seed(name: str, seed) -> int:
    if type(seed) is not int and not isinstance(seed, np.integer):
        raise GuardError(f"G-seed: {seed!r} is not an integer seed")
    seed = int(seed)
    if not cohort(name).contains(seed):
        raise GuardError(f"G-seed: {seed} is outside cohort {name}")
    return seed


def keep_cohort(name: str, demonstrate) -> dict:
    """TASK-101 §4's replacement rule on this study's cohorts: ``demonstrate(seed) -> (ok,
    reason)`` over the main seeds in order; a rejected seed's slot takes the next reserve seed."""
    c = cohort(name)
    reserve = list(c.reserve_seeds)
    kept, rejected = [], []
    for seed in c.seeds:
        candidate = seed
        while True:
            ok, reason = demonstrate(candidate)
            if ok:
                kept.append(candidate)
                break
            rejected.append({"seed": candidate, "slot": len(kept), "reason": reason})
            if not reserve:
                raise GuardError(f"G-reserve: cohort {name}'s reserve is exhausted")
            candidate = reserve.pop(0)
    return {"kept": kept, "rejected": rejected, "drawn": len(kept) + len(rejected)}


def planner_seed(reset_seed: int) -> int:
    """§1: 10116 * 10^6 + reset seed, the same for every Λ2 arm on a reset."""
    return SALTS["cem"] * 10**6 + int(reset_seed)


def candidate_rng(seed: int, state_index: int) -> np.random.Generator:
    return np.random.default_rng([SALTS["candidates"], int(seed), int(state_index)])


# ----- Λ1's states (§3.1) --
DEMO_STATES = (0, 4, 8, 12, 16, 20, 24, 32, 40)
LINE_DISTANCES = (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.40, 0.48, 0.64)
RESET = "d0"  # the reset pose itself, the line's far end
SETTLE_COMMANDS = 10
CUBE_MOVED_M = 0.002
BANDS = ((0.08, 0.12), (0.12, 0.16), (0.16, 0.24), (0.24, 0.32), (0.32, 0.40), (0.40, 0.48),
         (0.48, 0.64))  # fmt: skip
LOW_BANDS = ((0.0, 0.02), (0.02, 0.04), (0.04, 0.08))
RESOLVED = 0.75
CANDIDATE_STATES = (0.08, 0.16, 0.32, 0.48, RESET)
FAR_STATES = (0.16, 0.32, 0.48, RESET)
CANDIDATES = 64
CANDIDATE_H = 16
FORMS = ("last", "mean")
HORIZONS = (8, 16)
PARTS = ("full", "wrist", "overview")  # Λ3: wrist tokens 0-15, overview tokens 16-31
HALF = cl.LATENT_DIM // 2


def line_distances(d0: float) -> tuple:
    """The nominal distances below d0, then the reset itself."""
    if not (math.isfinite(d0) and d0 > 0):
        raise ContractError("d0 must be finite and positive")
    return tuple(d for d in LINE_DISTANCES if d < d0) + (RESET,)


def line_point(q0, q1, d, d0: float) -> np.ndarray:
    """q(d) = q1 + (d / d0)(q0 - q1) for all six joints; ``RESET`` is q0 itself."""
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    if d == RESET:
        return q0.copy()
    if not 0 <= d < d0:
        raise ContractError(f"line distance {d} must lie in [0, d0)")
    return q1 + (float(d) / float(d0)) * (q0 - q1)


def part_errors(pred, goal) -> np.ndarray:
    """float64 [..., 3]: the latent error (mean squared difference, float64) over the full
    latent, the wrist half and the overview half."""
    p = np.asarray(pred, np.float64)
    g = np.asarray(goal, np.float64).reshape(-1)
    if p.shape[-1] != cl.LATENT_DIM or g.shape != (cl.LATENT_DIM,):
        raise ContractError("latents must be [..., 12288]")
    d = (p - g) ** 2
    return np.stack((d.mean(-1), d[..., :HALF].mean(-1), d[..., HALF:].mean(-1)), axis=-1).astype(
        np.float64
    )


def form_cost(step_errors, form: str, h: int) -> np.ndarray:
    """``last``: the error after command h; ``mean``: the running mean over commands 1..h (the
    planner's two cost forms). ``step_errors`` [..., T] with T >= h (time on the last axis)."""
    e = np.asarray(step_errors, np.float64)
    if form not in FORMS:
        raise ContractError(f"form must be one of {FORMS}")
    if e.shape[-1] < h:
        raise ContractError("not enough steps for this horizon")
    e = e[..., :h]
    return e[..., -1] if form == "last" else e.mean(-1)


# ----- the projection chain (§3.3, §4) --
def project_chain(requested, targets, params: cl.PreviewParams):
    """TASK-101's chained projector, one step at a time, keeping every step's joint targets.

    Returns (applied float32 [..., T, 6], targets float64 [..., T, 6]). Each step is
    ``cl.project_sequences`` on that one step from the previous step's targets, so the applied
    actions equal the chained call's bit for bit (a test checks it), and the targets are exactly
    what ``SO101Embodiment.execute`` sends for each requested command in turn."""
    a = np.asarray(requested)
    if a.dtype != np.float32 or a.ndim < 2 or a.shape[-1] != cl.JOINTS:
        raise ContractError("requested candidates must be float32 [..., T, 6]")
    lead = a.shape[:-2]
    old = np.broadcast_to(np.asarray(targets, np.float64), (*lead, cl.JOINTS)).copy()
    applied = np.empty_like(a)
    chain = np.empty((*a.shape[:-1], cl.JOINTS), np.float64)
    for t in range(a.shape[-2]):
        step, old = cl.project_sequences(np.ascontiguousarray(a[..., t : t + 1, :]), old, params)
        applied[..., t, :] = step[..., 0, :]
        chain[..., t, :] = old
    return applied, chain


def approach_sequence(targets, goal_q, params: cl.PreviewParams, horizon: int) -> np.ndarray:
    """The pose-law candidate: TASK-101's H-pose law toward ``goal_q``, chained over its own
    projected targets (requested, float32 [horizon, 6])."""
    t = np.asarray(targets, np.float64).copy()
    out = np.empty((int(horizon), cl.JOINTS), np.float32)
    for i in range(int(horizon)):
        out[i] = cl.pose_action(t, goal_q, params.scales)
        _, t = cl.project_sequences(out[i : i + 1], t, params)
    return out


def hold_sequence(targets, params: cl.PreviewParams, horizon: int) -> np.ndarray:
    """Zero arm deltas and the gripper command of the current gripper target
    (``SO101Embodiment.hold_action``'s arithmetic)."""
    width = params.gripper_open - params.gripper_closed
    g = np.clip(2.0 * (float(np.asarray(targets)[cl.JAW]) - params.gripper_closed) / width - 1.0,
                -1.0, 1.0)  # fmt: skip
    one = np.array([0, 0, 0, 0, 0, g], np.float32)
    return np.repeat(one[None], int(horizon), 0)


def candidate_set(seed, state_index, targets, goal_q, params) -> np.ndarray:
    """§3.3: approach, hold and 62 uniform random sequences (requested, float32 [64, 16, 6])."""
    rand = candidate_rng(seed, state_index).uniform(
        -1.0, 1.0, (CANDIDATES - 2, CANDIDATE_H, cl.JOINTS)
    )
    return np.concatenate(
        (
            approach_sequence(targets, goal_q, params, CANDIDATE_H)[None],
            hold_sequence(targets, params, CANDIDATE_H)[None],
            rand.astype(np.float32),
        )
    )


# ----- Λ2's arms (§4) --
TAU_H = 0.08
JAW_TOL = hy.HANDOVER_JAW_RAD
ARMS = {
    "J-ref": {"cost": "J", "samples": 256, "iterations": 6, "elites": 32, "horizon": 16},
    "J-big16": {"cost": "J", "samples": 1024, "iterations": 12, "elites": 128, "horizon": 16},
    "J-big32": {"cost": "J", "samples": 1024, "iterations": 12, "elites": 128, "horizon": 32},
    "O-img": {"cost": "img", "samples": 256, "iterations": 6, "elites": 32, "horizon": 16},
    "O-pose": {"cost": "pose", "samples": 256, "iterations": 6, "elites": 32, "horizon": 16},
}
J_ARMS = ("J-ref", "J-big16", "J-big32")
GPU_ARMS = ("J-ref", "J-big16", "J-big32", "O-img")


def check_arm(arm: str) -> dict:
    if arm not in ARMS:
        raise ContractError(f"unknown Λ2 arm {arm!r}")
    return ARMS[arm]


def pose_cost(q_end, goal_q) -> np.ndarray:
    """O-pose: (arm RMS to q1 / 0.08)^2 + (jaw error / 0.15)^2 after the last command."""
    q = np.asarray(q_end, np.float64)
    g = np.asarray(goal_q, np.float64)
    d = q[..., : cl.ARM_JOINTS] - g[: cl.ARM_JOINTS]
    rms2 = np.mean(d * d, axis=-1)
    jaw = q[..., cl.JAW] - g[cl.JAW]
    return rms2 / TAU_H**2 + (jaw / JAW_TOL) ** 2


def arm_rms(q, goal_q) -> np.ndarray:
    d = np.asarray(q, np.float64)[..., : cl.ARM_JOINTS] - np.asarray(goal_q)[: cl.ARM_JOINTS]
    return np.sqrt(np.mean(d * d, axis=-1))


# ----- statistics (§3.4) --
def ranks(x) -> np.ndarray:
    """Average ranks (1-based) with ties sharing their mean rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(1, len(x) + 1)
    values, inverse = np.unique(x, return_inverse=True)
    if len(values) < len(x):
        sums = np.bincount(inverse, weights=r)
        counts = np.bincount(inverse)
        r = (sums / counts)[inverse]
    return r


def spearman(x, y) -> float:
    """Spearman's ρ; NaN when either side is constant or has fewer than 3 values."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ContractError("spearman needs two equal 1-D arrays")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return math.nan
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def band_ordering(per_seed: dict, bands=BANDS) -> dict:
    """``per_seed``: ``{seed: {nominal d: cost}}`` (excluded states absent). For each band, the
    fraction of seeds with c(upper) > c(lower) among seeds with both states, its count, the
    median increment and the increments' IQR, and whether it is resolved (>= 0.75)."""
    out = {}
    for lo, hi in bands:
        inc = [
            float(v[hi]) - float(v[lo])
            for v in per_seed.values()
            if lo in v and hi in v and math.isfinite(v[lo]) and math.isfinite(v[hi])
        ]
        n = len(inc)
        up = sum(1 for x in inc if x > 0)
        acc = up / n if n else math.nan
        q1, q3 = np.percentile(inc, [25, 75]) if n else (math.nan, math.nan)
        med = float(np.median(inc)) if n else math.nan
        iqr = float(q3 - q1)
        out[f"{lo:.2f}-{hi:.2f}"] = {
            "lower": lo,
            "upper": hi,
            "seeds": n,
            "increasing": up,
            "accuracy": acc,
            "resolved": bool(n and acc >= RESOLVED),
            "median_increment": med,
            "increment_iqr": iqr,
            "gradient_to_noise": med / iqr if n and iqr > 0 else math.nan,
        }
    return out


def d_flat(bands: dict) -> float | str:
    """The lower edge of the first unresolved band going outward from 0.08 rad, or "> 0.64"."""
    for key in sorted(bands, key=lambda k: bands[k]["lower"]):
        b = bands[key]
        if b["lower"] < BANDS[0][0]:
            continue
        if not b["resolved"]:
            return b["lower"]
    return f"> {BANDS[-1][1]:.2f}"


def median(values) -> float:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else math.nan


# ----- rows (§6) --
ROWS = (
    "Δa-VOID",
    "Δa-NOT-REPRODUCED",
    "Δa-PLANNER-BOUND",
    "Δa-GEOMETRY",
    "Δa-PLANNER",
    "Δa-MODEL",
    "Δa-MIXED",
)
REACH = 6  # of 8
NOT_REPRODUCED = 4
J_FAIL = 2
DELTA_RHO = 0.20


def row(*, void: bool, h: dict, delta_rho: float) -> dict:
    """First match over §6's rows. ``h``: hand-overs by tolerance of 8 for every Λ2 arm."""
    if void:
        return {"row": ROWS[0], "fired": [ROWS[0]]}
    if sorted(h) != sorted(ARMS):
        raise ContractError(f"h needs every arm {sorted(ARMS)}")
    for v in h.values():
        if not 0 <= int(v) <= L2_SLOTS:
            raise ContractError("counts must lie in 0..8")
    checks = [
        (ROWS[1], h["J-ref"] >= NOT_REPRODUCED),
        (ROWS[2], h["O-pose"] < REACH),
        (ROWS[3], h["O-img"] < REACH),
        (ROWS[4], max(h["J-big16"], h["J-big32"]) >= REACH),
        (
            ROWS[5],
            max(h[a] for a in J_ARMS) <= J_FAIL
            and math.isfinite(delta_rho)
            and delta_rho >= DELTA_RHO,
        ),
    ]
    fired = [name for name, ok in checks if ok]
    return {"row": fired[0] if fired else ROWS[-1], "fired": fired or [ROWS[-1]]}
