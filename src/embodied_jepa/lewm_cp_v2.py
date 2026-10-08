"""TASK-081 (``apple_lewm_commit_precision_v2``): LeWM's committed aim under C1-M, solved as the
clipped fixed point of a local affine fit to W's own grid predictions, on 128 fresh gated resets.

Protocol ``docs/experiments/apple_lewm_commit_precision_v2.md`` (STATUS DRAFT; rulings R19 in
``docs/DECISIONS.md``, decision 2026-10-08 (f), each decided by Claude under owner delegation,
2026-09-30). This module is the protocol's **frozen block** as module constants (a candidate
until the freeze pins its sha256 in ``tests/test_lewm_cp_v2.py`` and the manifest), with its
statistics (salt 8302), the D and S row ladders at n = 128 and the Stage-0 power simulation
(salt 8304).

What changes from TASK-080 (protocol §0): W's step after the grid is **affine_local** (the design
note's tested variant, ``commit_precision_dev.affine_fixed_point`` and ``local_mask``, imported at
their pinned file sha256); N, L-shuf and L-mean carry the same solver; **W-frozen** (TASK-080's W
unchanged) is reported only; S has 128 resets with G-bar 112/128 and delta 16/128; nothing is
fitted, trained or collected, and tau_commit is carried from TASK-080's pooled K and K′.

TASK-080's and TASK-077's modules and the design note's solver module are imported, never
edited; G-frozen checks their pins. **G-sentinel**: every report field a stage has not reached
holds :data:`NOT_EVALUATED`; every ladder refuses a missing input.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import commit_precision_dev as cpd
from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_commit_precision_v2"
TASK = "TASK-081"
STATUS = "DRAFT"
DOCUMENT = "docs/experiments/apple_lewm_commit_precision_v2.md"
MANIFEST = "benchmarks/manifests/apple-lewm-cp-v2.json"
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
RULINGS = "DECISIONS.md decision 2026-10-08 (f), R19.1-R19.11"
GuardError = pt.GuardError
NOT_EVALUATED = lm.NOT_EVALUATED

# ----- carried and checked (§2, G-frozen, G-hash) ------------------------------------------------
TASK080_MANIFEST = pr.MANIFEST  # its seven file pins, its frozen pin and its document pin
TASK080_FROZEN_SHA256 = "0fc095dc947f0ac74ebf6d1c541098a1e6fec1897592256f97ac8962b97be064"
TASK077_FROZEN_SHA256 = pr.TASK077_FROZEN_SHA256
SOLVER_FILE = "src/embodied_jepa/commit_precision_dev.py"
SOLVER_FILE_SHA256 = "ba3c8d03f98ddaeb2d26fd13c4363995cb11492f02603eb8406ca42ef71f91f1"
OWN_FILES = (
    "src/embodied_jepa/lewm_cp_v2.py",
    "src/embodied_jepa/lewm_cp_v2_runtime.py",
    "scripts/run_lewm_cp_v2.py",
)
REUSED = {
    # TASK-080 Stage R (R-PASS, R18.25): the report and the primary seed's two readouts
    "stage_r_report_sha256": "bedb896692a9718ac598fa93c5d7dcc199090bf4cfa13d14fa0bd0fc53848ea7",
    "stage_r_revision": "33cea5cf9c45d83e4f4bb81a1d6b9fa28390a38c",
    "r_s": "6e05d223d0fbd0dda8875e9285c320be4f1d7bbc85e0ec36d1e0089e413c1046",
    "r_n": "15405f3f518c17734ad1198b10349ebd2dea97a79952fd8277ec04df9d365271",
    # TASK-077's artifacts, checked by TASK-080's ``old_chain`` against ``lewm_pr_v2.REUSED``
    "task077": "lewm_pr_v2.REUSED",
    # TASK-080's K0′ (R18.22): tau_commit and the pooled curve, carried (§6.2)
    "k0_prime_report_sha256": pr.K0_PRIME_MEASURED["report_sha256"],
}
PRIMARY_SEED = pr.PRIMARY_SEED  # 66800, flagged
PRIMARY_SEED_FLAG = pr.PRIMARY_SEED_FLAG
TAU_COMMIT_CM = float(pr.K0_PRIME_MEASURED["tau_commit_cm"])  # 1.0, carried (§6.2)
TAU_CURVE = dict(pr.K0_PRIME_MEASURED["pooled"]["counts_pooled"])  # 64/58/60/32/29/7 of 64
TAU_CURVE_RESETS = pr.POOLED_RESETS

# ----- seeds and salts (§5, R19.6) ---------------------------------------------------------------
SEED_BLOCK = (70000, 71999)  # R18.34
DEVELOPMENT_RANGE = (70000, 70099)  # the design note's development range; not used here
SEED_RANGES = {
    "D": (70100, 70115),  # the development closed loop
    "S": (70200, 70327),  # the gated cohort
}
DEBUG_SEEDS = (71900, 71999)  # runner mechanics only; nothing in them is read
DEBUG_RANGES = {"D": (71900, 71903), "S": (71910, 71913)}
FORBIDDEN_RANGES = dict(pr.FORBIDDEN_RANGES) | {
    "task080_block": pr.SEED_BLOCK,
    "task081_development": DEVELOPMENT_RANGE,
}
SALTS = {
    "bootstrap": 8302,  # every bootstrap of this task (10 000 resamples)
    "l_rand": 8303,  # L-rand's draw per reset
    "power": 8304,  # Stage 0's power simulations
}
USED_SALTS = {"design_note_development": cpd.POWER_SALT}  # 8301: not reused
RESERVED_SALTS = {f"reserved_{k}": k for k in range(8305, 8313)}  # any use needs its own ruling
CARRIED_SALTS = {"move": pr.SALTS["move"]}  # 8201, TASK-080's move draw on the fresh seeds
EARLIER_SALTS = {**pr.EARLIER_SALTS, **{f"task080_{k}": v for k, v in pr.SALTS.items()}}
EARLIER_SALTS |= {f"task080_{k}": v for k, v in pr.RESERVED_SALTS.items()}


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in 70100-71999, are disjoint, avoid every forbidden range (TASK-080's block
    and the design note's development range included); the salts are fresh."""
    spans = dict(SEED_RANGES) | {"debug": DEBUG_SEEDS}
    for name, (low, high) in spans.items():
        if not (SEED_BLOCK[0] <= low <= high <= SEED_BLOCK[1]):
            raise ContractError(f"G-seeds: {name} leaves the block {SEED_BLOCK}")
        for other, (o_low, o_high) in FORBIDDEN_RANGES.items():
            if low <= o_high and o_low <= high:
                raise ContractError(f"G-seeds: {name} overlaps {other}")
    items = sorted(spans.values())
    for (_l1, h1), (l2, _h2) in zip(items, items[1:], strict=False):
        if l2 <= h1:
            raise ContractError("G-seeds: two ranges overlap")
    for role, (low, high) in DEBUG_RANGES.items():
        if not (DEBUG_SEEDS[0] <= low <= high <= DEBUG_SEEDS[1]):
            raise ContractError(f"G-seeds: debug {role} leaves the debug range")
    if (SEED_RANGES["D"][1] - SEED_RANGES["D"][0] + 1) != D_RESETS or (
        SEED_RANGES["S"][1] - SEED_RANGES["S"][0] + 1
    ) != S_RESETS:
        raise ContractError("G-seeds: D has 16 resets and S 128")
    ours = [*SALTS.values(), *RESERVED_SALTS.values(), *USED_SALTS.values()]
    if sorted(ours) != list(range(8301, 8313)) or set(ours) & set(EARLIER_SALTS.values()):
        raise ContractError("G-salts: 8301-8312 are this task's and fresh")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """A cohort's seeds are exactly its declared range, in order (G-split)."""
    seeds = tuple(int(s) for s in seeds)
    if seeds != seeds_of(role, debug=debug):
        raise GuardError(f"G-split: {role}'s seeds are not its declared range")
    return seeds


reset_of = pr.reset_of
move_offset = pr.move_offset  # salt 8201, carried


def l_rand_index(seed: int, feasible) -> int:
    """L-rand: a uniform draw over the feasible grid candidates (salt 8303); -1 if none."""
    feasible = np.flatnonzero(np.asarray(feasible, bool))
    if len(feasible) == 0:
        return -1
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["l_rand"], int(seed)]))
    return int(feasible[int(rng.integers(0, len(feasible)))])


# ----- the condition (carried, §2.1) -------------------------------------------------------------
RULE = pr.RULE
COMMIT_STEP = pr.COMMIT_STEP
GRID = pr.GRID
HORIZON = pr.HORIZON
READ_STEP = pr.READ_STEP
A_LO = pr.A_LO
REFINE_MAX = pr.REFINE_MAX  # W-frozen only
REFINE_FRACTION_OF_TAU = pr.REFINE_FRACTION_OF_TAU  # W-frozen only

# ----- the solver (§3, R19.2) --------------------------------------------------------------------
SOLVER = {
    "name": "affine_local",
    "code": f"{SOLVER_FILE}: affine_fixed_point, local_mask (sha256 {SOLVER_FILE_SHA256})",
    "local_a_steps": cpd.LOCAL_A_STEPS,  # 2 a-steps of 0.05
    "local_b_m": cpd.LOCAL_B_M,  # 2 cm
    "max_points": (2 * cpd.LOCAL_A_STEPS + 1) * 5,  # 25
    "min_points": 4,
    "det_min": 1e-6,
    "fallback": "the grid argmin, on fewer than 4 points, |det(I - J)| < 1e-6 or an infeasible "
    "chunk at the clipped fixed point",
    "logged_rollouts": 1,  # the committed aim's residual, for the log only
    "extra_refinement": False,  # R19.2: no damped refinement after the affine solution
}
SOLVER_OF = {  # arm -> its solver after the grid (G-solver)
    "W": "affine_local",
    "W-frozen": "frozen",
    "N": "affine_local",
    "L-shuf": "affine_local",
    "L-mean": "affine_local",
    "L-rand": "none",
}
MODEL_ARM = {"W-frozen": "W"}  # W-frozen runs TASK-080's W (its model, start and readout)
READOUTS = {"W": "r_s", "W-frozen": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}

# ----- Stages D and S (§6, §7, R19.3, R19.5) ----------------------------------------------------
D_RESETS = 16
D_ARMS = ("W", "W-frozen", "N", "L-shuf", "L-mean", "H-final", "H-rule")
D_REPORTED = ("W-frozen", "H-rule")
D_BARS = dict(pr.D_BARS)  # W >= 12/16, H-final >= 14/16, W - max(N, L-shuf, L-mean) >= +3/16
S_RESETS = 128
S_ARMS = (
    "W",
    "W-frozen",
    "N",
    "L-shuf",
    "L-mean",
    "L-rand",
    "H-rule",
    "H-sysid",
    "H-final",
    "H-read",
    "H-now",
)
TWINS = pr.TWINS  # N, L-shuf, L-mean, L-rand
COMPARATORS = pr.COMPARATORS  # H-rule, H-sysid: the better on S; a tie goes to H-rule
REPORTED_ONLY_ARMS = ("W-frozen", "H-read", "H-now")
CANDIDATE_ARMS = frozenset({"W", "W-frozen", "N", "L-shuf", "L-mean", "L-rand"})
PRIVILEGED_ARMS = frozenset(pr.PRIVILEGED_ARMS)
BAR_FRACTION = pr.G_BAR / pr.S_RESETS  # 0.875: tau_commit's bar fraction (28/32), carried
DELTA_FRACTION = pr.DELTA / pr.S_RESETS  # 0.125: an allocation, carried as a fraction (R18.35)
G_BAR = round(BAR_FRACTION * S_RESETS)  # 112 of 128
DELTA = round(DELTA_FRACTION * S_RESETS)  # 16 of 128
MCNEMAR_P = pr.MCNEMAR_P  # 0.01
MIN_SEPARATION = lm.MIN_SEPARATION  # 7: a count (the McNemar test's minimum), independent of n
DETERMINISM_RESETS = pr.DETERMINISM_RESETS  # W's re-run on S's first four resets
S_ROWS = {
    "V": "one repeat of the stage after a recorded fix",
    "S-VOID-CEILING": "H-final(commit)(S) < 112/128: escalate, no clause, no claim",
    "L-NO-GAIN": "for a twin, the McNemar test fails and W is detectably no better (upper bound of "
    "W - arm < +7 resets): the clause fires",
    "L-INFERIOR": "W is detectably inferior beyond delta (upper bound of W - C < -16/128): the "
    "clause fires",
    "L-PASS": "G-bar, G-NI and the four McNemar tests pass: the primary claim, 'LeWM-driven "
    "closed-loop success'",
    "L-TWIN-NEAR": "a McNemar test fails and every failed one is a miss within noise: escalate, "
    "no clause, no claim",
    "L-NEAR": "G-NI fails, W not detectably inferior: escalate, no clause, no claim",
    "L-BAR": "G-bar fails with G-NI and the tests passing: escalate, no clause, no claim",
}
D_ROWS = ("L-DEV-STOP", "D-PASS")
CLAUSE_ROWS = frozenset({"L-NO-GAIN", "L-INFERIOR"})
NO_CLAUSE_ROWS = ("L-DEV-STOP", "S-VOID-CEILING", "L-TWIN-NEAR", "L-NEAR", "L-BAR", "V",
                  "INCONCLUSIVE")  # fmt: skip
ARMS = {
    "W": {
        "what": "the primary seed's W from the encoded 405 frame, read by R-S, committed by "
        "affine_local",
        "privileged": False,
    },  # fmt: skip
    "W-frozen": {
        "what": "TASK-080's W unchanged (TASK-077 §5.1's refinement, cap 10, stop at "
        "tau_commit/4), read by R-S; reported only",
        "privileged": False,
    },  # fmt: skip
    "N": {
        "what": "the primary seed's N, zero commands, read by R-N, committed by affine_local",
        "privileged": False,
    },  # fmt: skip
    "L-shuf": {
        "what": "W from the logged 405 frame of W's attempt on the next reset that reached "
        "405, this reset's candidate chunks, read by R-S, committed by affine_local",
        "privileged": False,
    },  # fmt: skip
    "L-mean": {
        "what": "W from L-mean's mean 405 latent, read by R-S, committed by affine_local",
        "privileged": False,
    },  # fmt: skip
    "L-rand": {
        "what": "a uniform feasible grid candidate (salt 8303), no solver",
        "privileged": False,
    },  # fmt: skip
    "H-rule": dict(pr.ARMS["H-rule"]),
    "H-sysid": dict(pr.ARMS["H-sysid"]),
    "H-final": dict(pr.ARMS["H-final"]),
    "H-read": dict(pr.ARMS["H-read"]),
    "H-now": dict(pr.ARMS["H-now"]),
}

# ----- statistics (salt 8302; TASK-080's estimators) ---------------------------------------------
BOOTSTRAP_RESAMPLES = pr.BOOTSTRAP_RESAMPLES  # 10 000


def paired_interval(first, second) -> dict:
    """TASK-080's paired reset-bootstrap interval of the summed difference, salt 8302."""
    return pr.paired_interval(first, second, salt=SALTS["bootstrap"])


def median_ci(values) -> dict:
    return pr.median_ci(values, salt=SALTS["bootstrap"])


def comparator(outcomes: dict) -> str:
    return lm.comparator(outcomes)


def twin_test(w, arm) -> dict:
    b, c = pt.discordant(w, arm)
    p = pt.mcnemar_one_sided(b, c)
    interval = paired_interval(w, arm)
    passes = bool(p < MCNEMAR_P)
    hi = interval["ci95"][1]
    return interval | {
        "b": b,
        "c": c,
        "p": p,
        "passes": passes,
        "detectably_no_better": bool(not passes and hi < MIN_SEPARATION),
        "miss_within_noise": bool(not passes and hi >= MIN_SEPARATION),
    }


def _require(**inputs) -> None:
    missing = sorted(k for k, v in inputs.items() if lm.is_missing(v))
    if missing:
        raise ContractError(f"G-sentinel: a row ladder was given missing inputs {missing}")


def decide_d(counts: dict) -> dict:
    """§7.1: TASK-080's D stops on W, N, L-shuf, L-mean and H-final(commit); W-frozen and H-rule
    are reported beside them and read by no stop."""
    _require(counts=counts)
    missing = [a for a in D_ARMS if not isinstance(counts.get(a), int)]
    if missing:
        raise ContractError(f"Stage D needs the counts of {missing}")
    out = pr.decide_d({a: counts[a] for a in lm.D_ARMS})
    return out | {"reported_only": {a: counts[a] for a in D_REPORTED}}


def solver_effect(w, w_frozen) -> dict:
    """§7.3 item 2, reported only: W - W-frozen with its discordant counts, interval and exact
    one-sided McNemar p (W better)."""
    b, c = pt.discordant(w, w_frozen)
    return paired_interval(w, w_frozen) | {"b": b, "c": c, "p": pt.mcnemar_one_sided(b, c)}


def decide_s(outcomes: dict | None, *, void: bool = False, debug: bool = False) -> dict:
    """§7.2's ladder, first match (TASK-080 §9.5's order at n = 128): V, S-VOID-CEILING,
    L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR, L-NEAR, L-BAR. W-frozen is read by no row."""
    if void:
        return {"row": "V", "meaning": S_ROWS["V"], "clause_fires": False}
    _require(outcomes=outcomes)
    needed = ("W", *TWINS, *COMPARATORS, "H-final")
    missing = [a for a in needed if a not in outcomes or outcomes[a] is None]
    if missing:
        raise ContractError(f"Stage S's ladder needs {missing}")
    w = np.asarray(outcomes["W"], bool)
    if w.shape != (S_RESETS,) and not debug:
        raise ContractError(f"Stage S is {S_RESETS} paired resets")
    if any(np.asarray(outcomes[a], bool).shape != w.shape for a in needed):
        raise ContractError("every arm is paired on the same resets")
    c_arm = comparator(outcomes)
    ni = paired_interval(w, outcomes[c_arm])
    twins = {a: twin_test(w, outcomes[a]) for a in TWINS}
    g_bar = int(w.sum()) >= G_BAR
    g_ni = bool(ni["ci95"][0] > -DELTA)
    inferior = bool(ni["ci95"][1] < -DELTA)
    ceiling = int(np.sum(outcomes["H-final"]))
    sb, sc = pt.discordant(w, outcomes[c_arm])
    secondary = {"b": sb, "c": sc, "p": pt.mcnemar_one_sided(sb, sc)}
    secondary["W_better_detectably"] = bool(secondary["p"] < MCNEMAR_P)
    if ceiling < G_BAR:
        row = "S-VOID-CEILING"
    elif any(t["detectably_no_better"] for t in twins.values()):
        row = "L-NO-GAIN"
    elif inferior:
        row = "L-INFERIOR"
    elif g_bar and g_ni and all(t["passes"] for t in twins.values()):
        row = "L-PASS"
    elif not all(t["passes"] for t in twins.values()):
        row = "L-TWIN-NEAR"
    elif not g_ni:
        row = "L-NEAR"
    else:
        row = "L-BAR"
    out = {
        "row": row,
        "meaning": S_ROWS[row],
        "clause_fires": row in CLAUSE_ROWS,
        "counts": {a: int(np.sum(v)) for a, v in outcomes.items() if v is not None},
        "comparator": c_arm,
        "G-bar": g_bar,
        "G-NI": ni | {"passes": g_ni, "detectably_inferior": inferior},
        "twins": twins,
        "ceiling": ceiling,
        "secondary_reported_only": secondary,
    }
    if outcomes.get("W-frozen") is not None:
        out["solver_effect_reported_only"] = solver_effect(w, outcomes["W-frozen"])
    return out


determinism_check = pr.determinism_check

# ----- the clause (§9, R19.9) --------------------------------------------------------------------
CLAUSE_SCOPE = lm.CLAUSE_SCOPE  # TASK-077's R17.10 unchanged
CLAUSE_NARROWING = lm.CLAUSE_NARROWING
CLAUSE_RESULTS_WORDING = (
    "if the clause fires, the results document states plainly that one 8 x 8 run, read by a dual "
    "ridge fitted on its own stand-in predictions and committed by the affine fixed point, closes "
    "under this condition every other pooled grid (except 4 x 4 on a corpus larger than 1 024 "
    "roots), every readout of the predicted latent (TASK-080 §12's wording) and every post-grid "
    "solver for the committed aim from the grid's predictions (the frozen refinement with any cap "
    "or damping, best-residual, local or global affine or higher-order fits, Newton or secant "
    "steps, iterate averaging), not only the solver that was run"
)
NEVER_CLOSED = (
    *pr.NEVER_CLOSED,
    "R18.32's direction (a): a condition in which no hand-written arm is given the plate law",
    "several stand-in chunks per aim",
)

# ----- caps, memory, guards (§10) ----------------------------------------------------------------
CAPS_SECONDS = {"D": 7_200.0, "S": 21_600.0, "per_attempt": 300.0, "simulate": 3_600.0}
CAP_FACTOR_MIN = 1.5
MEMORY = dict(pr.MEMORY)  # 12 GiB process-tree PSS
DISK_MIN_GIB = pr.DISK_MIN_GIB  # 10 GiB
WM_WORKERS = pr.WM_WORKERS  # 4
WORKER_TORCH_THREADS = pr.WORKER_TORCH_THREADS
THREAD_ENV = pr.THREAD_ENV
QUIET_MACHINE = pr.QUIET_MACHINE
VOID_RULE = {
    "void": pr.VOID_RULE["void"],
    "repeat": "at most one repeat per stage after a committed, pushed and recorded fix, on the "
    "same seeds in a new output directory; a second V of the same stage ends TASK-081 "
    "INCONCLUSIVE; Vs in different stages do not add up",
    "debug": "debug runs only on committed code, only on 71900-71999; nothing in them is read",
}
STAGES = ("D", "S")

# ----- power (§8, R19.8; salt 8304) --------------------------------------------------------------
PLANNING = {"C": 0.992, "W": 0.95}
POWER_COUPLINGS = cpd.COUPLINGS  # overlap, half, independent
POWER_RESAMPLES = BOOTSTRAP_RESAMPLES
POWER_TRIALS = 20_000
POWER_GRID = {
    "p_w": (0.906, 0.922, 0.938, 0.953, 0.969),
    "p_c": (0.984, 0.992),
    "p_sysid": 62 / 64,  # H-sysid's rate on TASK-080's S
}
POWER_DRAFT = {  # §8's table as the draft quotes it from the design note (one comparator)
    (0.984, 0.938): (0.93, 0.87, 0.82),
    (0.984, 0.953): (0.99, 0.98, 0.95),
    (0.984, 0.969): (1.00, 1.00, 1.00),
    (0.992, 0.938): (0.85, 0.80, 0.77),
    (0.992, 0.953): (0.97, 0.95, 0.93),
    (0.992, 0.969): (1.00, 1.00, 0.99),
}
TWIN_POWER_CELLS = ((0.906, 26 / 64), (0.906, 30 / 64), (0.906, 0.60), (0.953, 0.60))


def g_bar_power(rate: float, *, n: int = S_RESETS, bar: int = G_BAR) -> float:
    return pr.g_bar_power(rate, n=n, bar=bar)


_NI_CACHE: dict = {}


def ni_passes(kp: int, km: int, *, n: int = S_RESETS, margin: int = DELTA,
              resamples: int = POWER_RESAMPLES) -> bool:  # fmt: skip
    """Whether TASK-080's G-NI estimator passes for (k+, k-) resets won by W only and by C only:
    the reset-bootstrap sum depends on the resets only through (k+, k-), so it is a multinomial
    draw (salt 8304, one stream per cell)."""
    key = (int(kp), int(km), int(n), int(margin), int(resamples))
    if key not in _NI_CACHE:
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], *key]))
        draws = rng.multinomial(n, [kp / n, km / n, (n - kp - km) / n], size=resamples)
        _NI_CACHE[key] = bool(np.percentile(draws[:, 0] - draws[:, 1], 2.5) > -margin)
    return _NI_CACHE[key]


def ni_power(p_w: float, p_c: float, coupling: str, *, p_sysid: float | None = None,
             n: int = S_RESETS, margin: int = DELTA, trials: int = POWER_TRIALS,
             index: int = 0) -> dict:  # fmt: skip
    """G-NI's pass rate and L-INFERIOR's rate by simulation (salt 8304): W at ``p_w``; H-rule at
    ``p_c`` coupled to W by ``coupling``; with ``p_sysid``, H-sysid coupled to W the same way and
    conditionally independent of H-rule given W, C the better of the two on each trial (a tie to
    H-rule). The single-comparator rate is reported beside it."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], 1, int(index)]))
    w = rng.uniform(size=(trials, n)) < p_w
    q = cpd.discordance(p_w, p_c, coupling)
    rule = lm._draw_relative(rng, w, p_w, *q)
    comps = {"single": rule}
    if p_sysid is not None:
        sysid = lm._draw_relative(rng, w, p_w, *cpd.discordance(p_w, p_sysid, coupling))
        comps["better_of_two"] = np.where((rule.sum(1) >= sysid.sum(1))[:, None], rule, sysid)
    out = {"p_w": p_w, "p_c": p_c, "p_sysid": p_sysid, "coupling": coupling, "n": n,
           "trials": trials}  # fmt: skip
    for name, c in comps.items():
        kp, km = (w & ~c).sum(1), (~w & c).sum(1)
        cells = {}
        for a, b in zip(kp.tolist(), km.tolist(), strict=True):
            cells[(a, b)] = cells.get((a, b), 0) + 1
        g_ni = sum(k for (a, b), k in cells.items() if ni_passes(a, b, n=n, margin=margin))
        inferior = 0
        for (a, b), k in cells.items():  # upper bound < -margin, from the same multinomial form
            if b - a > margin and _ni_upper_below(a, b, n=n, margin=margin):
                inferior += k
        out[name] = {"g_ni_pass_rate": g_ni / trials, "l_inferior_rate": inferior / trials}
    return out


_UP_CACHE: dict = {}


def _ni_upper_below(kp: int, km: int, *, n: int, margin: int) -> bool:
    key = (int(kp), int(km), int(n), int(margin))
    if key not in _UP_CACHE:
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], 2, *key]))
        draws = rng.multinomial(n, [kp / n, km / n, (n - kp - km) / n], size=POWER_RESAMPLES)
        _UP_CACHE[key] = bool(np.percentile(draws[:, 0] - draws[:, 1], 97.5) < -margin)
    return _UP_CACHE[key]


def twin_power(p_w: float, p_t: float, coupling: str, *, n: int = S_RESETS) -> float:
    """The exact one-sided McNemar test's pass probability at p < 0.01, exactly (no simulation):
    the multinomial over (W only, twin only) at the coupling's discordance."""
    q_w, q_t = pr.discordant_probabilities(p_w, p_t, {"overlap": "nested"}.get(coupling, coupling))
    table = pr.mcnemar_pass_table(n)
    total = 0.0
    q_0 = max(1.0 - q_w - q_t, 0.0)
    for b in range(n + 1):
        for c in range(n + 1 - b):
            if not table[b, c]:
                continue
            if (q_w == 0 and b) or (q_t == 0 and c) or (q_0 == 0 and b + c < n):
                continue
            logp = math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(c + 1)
            logp -= math.lgamma(n - b - c + 1)
            logp += (b * math.log(q_w) if b else 0.0) + (c * math.log(q_t) if c else 0.0)
            logp += (n - b - c) * math.log(q_0) if n - b - c else 0.0
            total += math.exp(logp)
    return float(total)


def power_tables(*, trials: int = POWER_TRIALS, grid=None) -> dict:
    """§8 recomputed: G-bar exactly; G-NI and L-INFERIOR by simulation with one and with the
    better-of-two comparator; the twin tests exactly; the size at the margin."""
    grid = POWER_GRID if grid is None else grid
    ni, k = [], 0
    for p_c in grid["p_c"]:
        for p_w in grid["p_w"]:
            for coupling in POWER_COUPLINGS:
                ni.append(ni_power(p_w, p_c, coupling, p_sysid=grid["p_sysid"], trials=trials,
                                   index=k))  # fmt: skip
                k += 1
    size = []
    for p_c in grid["p_c"]:
        for coupling in POWER_COUPLINGS:
            size.append(ni_power(p_c - DELTA / S_RESETS, p_c, coupling, trials=trials, index=k))
            k += 1
    return {
        "g_bar": {str(r): g_bar_power(r) for r in (0.875, 0.89, 0.906, 0.922, 0.938, 0.953)},
        "g_ni": ni,
        "size_at_margin": size,
        "twins": [
            {"p_w": p_w, "p_t": p_t, **{c: twin_power(p_w, p_t, c) for c in POWER_COUPLINGS}}
            for p_w, p_t in TWIN_POWER_CELLS
        ],
        "trials": trials,
        "resamples_per_cell": POWER_RESAMPLES,
    }


# ----- helpers ------------------------------------------------------------------------------------
def is_missing(value) -> bool:
    return lm.is_missing(value)


def sentinel_fields(names) -> dict:
    return lm.sentinel_fields(names)


ROW_NAMES = frozenset({*S_ROWS, *D_ROWS})


def check_sentinel(report: dict, evaluated: set[str], fields) -> list[str]:
    """Every unevaluated field holds the sentinel; a row-like value there is refused."""
    bad = []
    for name in fields:
        value = report.get(name, NOT_EVALUATED)
        if name not in evaluated and value != NOT_EVALUATED:
            bad.append(name)
        if name not in evaluated and isinstance(value, dict) and value.get("row") in ROW_NAMES:
            bad.append(name)
    if bad:
        raise GuardError(f"G-sentinel: fields {sorted(set(bad))} hold values never evaluated")
    return bad


# ----- Stage 0 (filled by the Stage-0 PR; development only) ---------------------------------------
STAGE0: dict = {  # the record: docs/experiments/apple_lewm_commit_precision_v2_stage0.md
    "revision": "2faf9e5",  # the smokes and the power run; tree-identical to 5bb9579 (rebased)
    "smoke_root": "outputs/task081-smoke-* in the task081-stage0 worktree (git-ignored); debug "
    "seeds 71900-71999 only; nothing in them is read",
    "smokes": {
        "closedD-1": {
            "outcome": "V",
            "revision": "19e3e83",
            "reason": "KeyError 'stages' (the "
            "report lacked the record the harness writes; fixed in 2faf9e5)",
        },
        "closedD-2": {"outcome": "L-DEV-STOP-DEBUG", "seconds": 121, "peak_tree_pss_gib": 9.38},
        "closedS-1": {
            "outcome": "V",
            "reason": "G-plan refused a --stage-d report that was not "
            "D-PASS-DEBUG (the guard, as designed); its in-run G-tests passed: 2120 "
            "passed, 37 skipped",
        },
        "closedS-2": {
            "outcome": "S-VOID-CEILING-DEBUG",
            "seconds": 160,
            "peak_tree_pss_gib": 9.40,
            "determinism_ok": True,
        },
        "closedD-3": {
            "outcome": "L-DEV-STOP-DEBUG",
            "revision": "9c82b97",
            "seconds": 121,
            "peak_tree_pss_gib": 9.49,
            "note": "the final code (the design rank logged); the same counts as closedD-2",
        },
    },
    "seconds_per_attempt_median": {  # debug S (4 resets, 4 workers)
        "W": 15.56,
        "W-frozen": 13.97,
        "N": 7.01,
        "L-shuf": 13.91,
        "L-mean": 13.75,
        "L-rand": 6.65,
        "H-rule": 4.49,
        "H-sysid": 4.31,
        "H-final": 7.99,
        "H-read": 8.16,
        "H-now": 4.21,
    },
    "seconds_per_attempt_max": 15.6,
    "scaled_seconds": {"D": 510, "S": 3420},  # attempts on 4 workers plus G-tests and G-repro
    "simulate": {
        "report_sha256": "5043c3aba4a9a3c1646336c65b7ff7963f4f78f83393f6b509ea43e07186a56c",
        "g_ni_better_of_two_at_c_0992": {  # overlap / half / independent, 20 000 trials
            "0.906": [0.326, 0.283, 0.263],
            "0.922": [0.592, 0.537, 0.496],
            "0.938": [0.848, 0.797, 0.763],
            "0.953": [0.973, 0.952, 0.934],
            "0.969": [0.999, 0.998, 0.995],
        },
        "size_at_margin_single": [0.026, 0.036],  # G-NI passes at W = C - delta
        "l_inferior_at_margin_single": [0.014, 0.017],  # the clause's false fire at W = C - delta
        "twins_min": 0.9997,  # at TASK-080's twin rates and N at 0.60, every coupling
    },
}


# ----- the frozen block ---------------------------------------------------------------------------
def frozen_block() -> dict:
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "status": STATUS,
            "delegated": DELEGATED,
            "rulings": RULINGS,
            "not_evaluated": NOT_EVALUATED,
            "task080_frozen_sha256": TASK080_FROZEN_SHA256,
            "task080_manifest": TASK080_MANIFEST,
            "task077_frozen_sha256": TASK077_FROZEN_SHA256,
            "solver_file": SOLVER_FILE,
            "solver_file_sha256": SOLVER_FILE_SHA256,
            "own_files": OWN_FILES,
            "reused": REUSED,
            "primary_seed": PRIMARY_SEED,
            "primary_seed_flag": PRIMARY_SEED_FLAG,
            "tau_commit_cm": TAU_COMMIT_CM,
            "tau_curve": TAU_CURVE,
            "tau_curve_resets": TAU_CURVE_RESETS,
            "seed_block": SEED_BLOCK,
            "development_range": DEVELOPMENT_RANGE,
            "seed_ranges": SEED_RANGES,
            "debug_seeds": DEBUG_SEEDS,
            "debug_ranges": DEBUG_RANGES,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "used_salts": USED_SALTS,
            "reserved_salts": RESERVED_SALTS,
            "carried_salts": CARRIED_SALTS,
            "condition": {
                "rule": RULE,
                "commit_step": COMMIT_STEP,
                "grid": GRID,
                "horizon": HORIZON,
                "read_step": READ_STEP,
                "a_lo": A_LO,
                "refine_max_w_frozen": REFINE_MAX,
                "refine_fraction_of_tau_w_frozen": REFINE_FRACTION_OF_TAU,
            },
            "solver": SOLVER,
            "solver_of": SOLVER_OF,
            "model_arm": MODEL_ARM,
            "readouts": READOUTS,
            "d": {"resets": D_RESETS, "arms": D_ARMS, "reported": D_REPORTED, "bars": D_BARS},
            "s": {
                "resets": S_RESETS,
                "arms": S_ARMS,
                "twins": TWINS,
                "comparators": COMPARATORS,
                "reported_only": REPORTED_ONLY_ARMS,
                "bar_fraction": BAR_FRACTION,
                "delta_fraction": DELTA_FRACTION,
                "g_bar": G_BAR,
                "delta": DELTA,
                "mcnemar_p": MCNEMAR_P,
                "min_separation": MIN_SEPARATION,
                "determinism_resets": DETERMINISM_RESETS,
                "rows": S_ROWS,
            },
            "arms": ARMS,
            "clause_rows": CLAUSE_ROWS,
            "no_clause_rows": NO_CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "clause_narrowing": CLAUSE_NARROWING,
            "clause_results_wording": CLAUSE_RESULTS_WORDING,
            "never_closed": NEVER_CLOSED,
            "caps_seconds": CAPS_SECONDS,
            "cap_factor_min": CAP_FACTOR_MIN,
            "memory": MEMORY,
            "disk_min_gib": DISK_MIN_GIB,
            "wm_workers": WM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "void_rule": VOID_RULE,
            "stages": STAGES,
            "power": {
                "planning": PLANNING,
                "trials": POWER_TRIALS,
                "grid": POWER_GRID,
                "resamples_per_cell": POWER_RESAMPLES,
            },  # fmt: skip
            "stage0": STAGE0,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()


check_seed_ranges()
if G_BAR != 112 or DELTA != 16 or MIN_SEPARATION != 7:
    raise ContractError("S: G-bar 112/128, delta 16/128 and the McNemar minimum 7 (R19.5)")
if pr.frozen_sha256() != TASK080_FROZEN_SHA256:
    raise ContractError("G-frozen: TASK-080's frozen block is not its pin")
