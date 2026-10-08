"""TASK-083 (``apple_lewm_seed_replication_v2``): TASK-081's L-PASS replicated with W's two other
model seeds, 66801 and 66802, under C1-M, on one fresh cohort of 128 gated resets.

Protocol ``docs/experiments/apple_lewm_seed_replication_v2.md`` (rulings R21 in
``docs/DECISIONS.md``, decision 2026-10-08 (o), each decided by Claude under owner delegation,
2026-09-30). This module is the protocol's **frozen block** as module constants (a candidate
until the freeze pins its sha256 in ``tests/test_lewm_rep_v2.py`` and the manifest), with its
statistics (salt 8501), the per-seed ladder (TASK-081's, n = 128), the combined replication row and
the Stage-0 power simulation (salt 8502).

What changes from TASK-081 (protocol §0): the model seed (66801 and 66802, each W with its own N,
R-S and R-N from TASK-080's Stage R); both seeds' candidate arms on one cohort, with L-rand,
H-rule, H-sysid and H-final(commit) run once and shared; REP-PASS only if both seeds reach L-PASS;
no clause; W-66800 reported only; W-frozen, H-read and H-now dropped; no Stage D.

TASK-081's frozen block and worker (``lewm_cp_v2``, ``lewm_cp_v2_runtime``) and TASK-080's and
TASK-077's modules are imported, never edited; G-frozen checks their pins. **G-sentinel**: every
report field a stage has not reached holds :data:`NOT_EVALUATED`; every ladder refuses a missing
input.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_cp_v2 as cpv
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_seed_replication_v2"
TASK = "TASK-083"
STATUS = "DRAFT"
DOCUMENT = "docs/experiments/apple_lewm_seed_replication_v2.md"
MANIFEST = "benchmarks/manifests/apple-lewm-rep-v2.json"
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
RULINGS = "DECISIONS.md decision 2026-10-08 (o), R21.1-R21.12"
GuardError = pt.GuardError
NOT_EVALUATED = lm.NOT_EVALUATED

# ----- carried and checked (§2, G-frozen, G-hash) ------------------------------------------------
TASK081_MANIFEST = cpv.MANIFEST  # its six file pins, its frozen pin and its document pin
TASK081_FROZEN_SHA256 = "77ccc6364624c7da9dcca1d09827a0f6a25a670a00392d488e7b0216e090705a"
TASK080_FROZEN_SHA256 = cpv.TASK080_FROZEN_SHA256
TASK077_FROZEN_SHA256 = cpv.TASK077_FROZEN_SHA256
OWN_FILES = (
    "src/embodied_jepa/lewm_rep_v2.py",
    "scripts/run_lewm_rep_v2.py",
)
MODEL_SEEDS = (66801, 66802)  # the replication's seeds (R21.1)
REFERENCE_SEED = 66800  # TASK-081's W, reported only (R21.2)
POOL_ORDER = (66801, 66802, 66800)  # one worker pool per model seed, one after another
SEED_FLAGS = {  # TASK-077 results §1.4 (W's last_two_triggered; kept update; val criterion)
    66800: {"last_two_triggered": True, "kept_update": 95_000, "val_criterion": 0.350399},
    66801: {"last_two_triggered": False, "kept_update": 71_250, "val_criterion": 0.364904},
    66802: {"last_two_triggered": False, "kept_update": 80_750, "val_criterion": 0.364660},
}
REUSED = {
    # TASK-080 Stage R (R-PASS, R18.25): the report, and R-S and R-N of every seed by content and
    # by file sha256 (the files are ``fits/r_{s,n}_<seed>.npz`` beside the report)
    "stage_r_report_sha256": cpv.REUSED["stage_r_report_sha256"],
    "stage_r_revision": cpv.REUSED["stage_r_revision"],
    "readouts": {
        66800: {
            "r_s": "6e05d223d0fbd0dda8875e9285c320be4f1d7bbc85e0ec36d1e0089e413c1046",
            "r_n": "15405f3f518c17734ad1198b10349ebd2dea97a79952fd8277ec04df9d365271",
            "r_s_file": "ef83f6ede32daecf56d773430aaf0b376de0db4a5120e9a1e6718f4b312e3929",
            "r_n_file": "fe3d7a76d27710e1b5443f697175414cbd6cce465f77db7341b5502789bc2853",
        },
        66801: {
            "r_s": "1cc8520304539b7f80afbfdad7af7abb4d0334841cf7fa2cbf1fcbb77a0fdc81",
            "r_n": "206e664f70972cb9799409a3b9bccdcaa5d26fe8c59aaf28e6153804a2b40243",
            "r_s_file": "b171ca18a2245143d24040ddb87a473bb5ea678d4b531c14ff8e0b065723a98e",
            "r_n_file": "3d23220393fe0b025c711ce4c124de23997eb8567aae58cdc581ddddf8c5f524",
        },
        66802: {
            "r_s": "28abf388f2dbb2592d6ac974fa40c86bbe865567d1dcbc0039d4fa333a9cf1a5",
            "r_n": "33db51c93b618f69caa39f6afd41059ff2593619774c486b82206e599823d647",
            "r_s_file": "178d23a0532414f2e183ca8f1ac5b820c35f1c811a688a6c774a3624c7365744",
            "r_n_file": "26f275427fc13e0ec1b3985e192d53d7f776a146d6f264ee75868f412377abe2",
        },
    },
    # TASK-077's six checkpoints, checked by TASK-080's ``old_chain`` against this record
    "models": dict(pr.REUSED["models"]),
}
TAU_COMMIT_CM = cpv.TAU_COMMIT_CM  # 1.0, carried (§6.2)
TAU_CURVE = dict(cpv.TAU_CURVE)
TAU_CURVE_RESETS = cpv.TAU_CURVE_RESETS

# ----- seeds and salts (§5, R21.10) --------------------------------------------------------------
SEED_BLOCK = (75000, 75999)
SEED_RANGES = {"S": (75200, 75327)}  # the gated cohort, 128 resets
DEBUG_SEEDS = (75900, 75999)  # runner mechanics only; nothing in them is read
DEBUG_RANGES = {"S": (75910, 75913)}
TASK082_BLOCK = (72000, 74999)  # R19.24
FORBIDDEN_RANGES = dict(cpv.FORBIDDEN_RANGES) | {
    "task081_block": cpv.SEED_BLOCK,
    "task082_block": TASK082_BLOCK,
}
SALTS = {
    "bootstrap": 8501,  # every bootstrap of this task (10 000 resamples, reset-clustered)
    "power": 8502,  # Stage 0's power simulation
}
RESERVED_SALTS = {f"reserved_{k}": k for k in range(8503, 8513)}  # any use needs its own ruling
CARRIED_SALTS = {
    "move": pr.SALTS["move"],  # 8201, TASK-080's move draw on the fresh seeds
    "l_rand": cpv.SALTS["l_rand"],  # 8303, TASK-081's worker draws L-rand with it
}
TASK082_SALTS = tuple(range(8401, 8413))  # R19.24 (design note 8401-8404) and R20.11 (8405-8412)
EARLIER_SALTS = {
    **cpv.EARLIER_SALTS,
    **{f"task081_{k}": v for k, v in cpv.SALTS.items()},
    **{f"task081_{k}": v for k, v in cpv.USED_SALTS.items()},
    **{f"task081_{k}": v for k, v in cpv.RESERVED_SALTS.items()},
    **{f"task082_{v}": v for v in TASK082_SALTS},
}


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in 75000-75999, are disjoint, avoid every forbidden range (TASK-081's list,
    its block and TASK-082's block); S has 128 resets; the salts 8501-8512 are fresh."""
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
    if SEED_RANGES["S"][1] - SEED_RANGES["S"][0] + 1 != S_RESETS:
        raise ContractError("G-seeds: S has 128 resets")
    ours = [*SALTS.values(), *RESERVED_SALTS.values()]
    if sorted(ours) != list(range(8501, 8513)) or set(ours) & set(EARLIER_SALTS.values()):
        raise ContractError("G-salts: 8501-8512 are this task's and fresh")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """A cohort's seeds are exactly its declared range, in order (G-split)."""
    seeds = tuple(int(s) for s in seeds)
    if seeds != seeds_of(role, debug=debug):
        raise GuardError(f"G-split: {role}'s seeds are not its declared range")
    return seeds


def check_model_seed(seed) -> int:
    """G-seed: a pool's model seed is one of the replication's seeds or the reference seed."""
    if type(seed) is not int or seed not in POOL_ORDER:
        raise GuardError(f"G-seed: model seed {seed!r} is not one of {POOL_ORDER}")
    return seed


reset_of = cpv.reset_of
move_offset = cpv.move_offset  # salt 8201, carried

# ----- arms (§4, R21.2) ---------------------------------------------------------------------------
PER_SEED_ARMS = ("W", "N", "L-shuf", "L-mean")  # each model seed's candidate arms
SHARED_ARMS = ("L-rand", "H-rule", "H-sysid", "H-final")  # run once, read by both seeds' ladders
REFERENCE_ARMS = ("W",)  # W-66800 on the same resets, reported only
DROPPED_ARMS = ("W-frozen", "H-read", "H-now")  # TASK-081's reported-only arms, read by no row
SOLVER_OF = {a: cpv.SOLVER_OF[a] for a in (*PER_SEED_ARMS, "L-rand")}  # affine_local; L-rand none
READOUTS = {a: cpv.READOUTS[a] for a in PER_SEED_ARMS}  # W, L-shuf, L-mean: R-S; N: R-N
TWINS = cpv.TWINS  # N, L-shuf, L-mean, L-rand
COMPARATORS = cpv.COMPARATORS  # H-rule, H-sysid: the better on S; a tie goes to H-rule
PRIVILEGED_ARMS = frozenset({"H-final"})
CANDIDATE_ARMS = frozenset({*PER_SEED_ARMS, "L-rand"})


def label(arm: str, seed: int | None) -> str:
    """A record's key in the report: ``W[66801]`` for a per-seed arm, the arm name if shared."""
    return arm if seed is None else f"{arm}[{seed}]"


def seed_outcomes(outcomes: dict, seed: int) -> dict:
    """One seed's ladder input from the labelled outcomes: its four arms and the shared four."""
    return {a: outcomes[label(a, seed)] for a in PER_SEED_ARMS} | {
        a: outcomes[a] for a in SHARED_ARMS
    }


# ----- gates (§7, R21.3; TASK-081's at n = 128) --------------------------------------------------
S_RESETS = cpv.S_RESETS  # 128
G_BAR = cpv.G_BAR  # 112
DELTA = cpv.DELTA  # 16
MCNEMAR_P = cpv.MCNEMAR_P  # 0.01
MIN_SEPARATION = cpv.MIN_SEPARATION  # 7, a count
DETERMINISM_RESETS = cpv.DETERMINISM_RESETS  # each seed's W re-run on S's first four resets
S_ROWS = {k: v for k, v in cpv.S_ROWS.items()}  # per-seed rows, TASK-081's meanings
S_ROWS["L-NO-GAIN"] = S_ROWS["L-NO-GAIN"].replace("the clause fires", "detectably not replicated")
S_ROWS["L-INFERIOR"] = S_ROWS["L-INFERIOR"].replace("the clause fires", "detectably not replicated")
COMBINED_ROWS = {
    "V": "one repeat of Stage S after a recorded fix",
    "REP-VOID-CEILING": "H-final(commit)(S) < 112/128: no reading; R7 unchanged",
    "REP-PASS": "both seeds L-PASS: TASK-081's L-PASS replicates across W's model seeds of this "
    "training run",
    "REP-ONE": "exactly one seed L-PASS: not replicated; stated factually (which seed passed, "
    "which did not, rows and counts); at the declared power this does not show whether seed or "
    "chance caused the difference unless the other seed is L-NO-GAIN or L-INFERIOR",
    "REP-NONE": "neither seed L-PASS: not replicated with either seed; at the declared power a "
    "non-pass is weak evidence against TASK-081's result unless a seed is L-NO-GAIN or "
    "L-INFERIOR",
}
DETECTABLE_FAIL_ROWS = frozenset({"L-NO-GAIN", "L-INFERIOR"})
CLAUSE_ROWS: frozenset = frozenset()  # R21.5: no row of TASK-083 fires a clause
COMBINATION = {
    "claim": "REP-PASS: both seeds reach L-PASS individually (an intersection-union test; no "
    "multiplicity adjustment needed or applied; no joint test)",
    "pooling": "never: the seeds' counts are not summed or averaged into a test, and a seed is "
    "not rescued by the other seed, by W-66800 on this cohort or by TASK-081's earlier pass",
    "at_least_one": "no 'at least one seed' statement is a claim",
    "shared": "the seeds share the comparator, L-rand and the ceiling on the same resets, so their "
    "rows are not independent",
}

# ----- statistics (salt 8501; TASK-080's estimators) ---------------------------------------------
BOOTSTRAP_RESAMPLES = cpv.BOOTSTRAP_RESAMPLES  # 10 000


def paired_interval(first, second) -> dict:
    """TASK-080's paired reset-bootstrap interval of the summed difference, salt 8501."""
    return pr.paired_interval(first, second, salt=SALTS["bootstrap"])


def median_ci(values) -> dict:
    return pr.median_ci(values, salt=SALTS["bootstrap"])


def comparator(outcomes: dict) -> str:
    return lm.comparator(outcomes)


def twin_test(w, arm) -> dict:
    """TASK-081's ``twin_test`` with this task's bootstrap salt."""
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


def decide_seed(outcomes: dict | None, *, void: bool = False, debug: bool = False) -> dict:
    """One seed's ladder: TASK-081's ``decide_s`` (first match: V, S-VOID-CEILING, L-NO-GAIN,
    L-INFERIOR, L-PASS, L-TWIN-NEAR, L-NEAR, L-BAR) with this task's salt; no clause."""
    if void:
        return {"row": "V", "meaning": S_ROWS["V"], "clause_fires": False}
    _require(outcomes=outcomes)
    needed = ("W", *TWINS, *COMPARATORS, "H-final")
    missing = [a for a in needed if a not in outcomes or outcomes[a] is None]
    if missing:
        raise ContractError(f"a seed's ladder needs {missing}")
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
    return {
        "row": row,
        "meaning": S_ROWS[row],
        "clause_fires": False,
        "detectable_failure": row in DETECTABLE_FAIL_ROWS,
        "counts": {a: int(np.sum(outcomes[a])) for a in needed},
        "comparator": c_arm,
        "G-bar": g_bar,
        "G-NI": ni
        | {
            "passes": g_ni,
            "detectably_inferior": inferior,
            "upper_below_zero": bool(ni["ci95"][1] < 0),
        },  # fmt: skip
        "twins": twins,
        "ceiling": ceiling,
        "secondary_reported_only": secondary,
    }


def decide_combined(rows: dict | None, *, void: bool = False) -> dict:
    """§7.3, first match: V, REP-VOID-CEILING, REP-PASS, REP-ONE, REP-NONE (R21.4)."""
    if void:
        return {"row": "V", "meaning": COMBINED_ROWS["V"], "clause_fires": False}
    _require(rows=rows)
    if sorted(rows) != sorted(MODEL_SEEDS):
        raise ContractError(f"the combined row needs both seeds' rows, got {sorted(rows)}")
    per = {s: rows[s] for s in MODEL_SEEDS}
    if any(r == "V" for r in per.values()):
        raise ContractError("a void seed makes the stage void (decide with void=True)")
    ceiling = [r == "S-VOID-CEILING" for r in per.values()]
    if any(ceiling) and not all(ceiling):
        raise ContractError("S-VOID-CEILING is shared: it holds for both seeds or neither")
    passed = [s for s in MODEL_SEEDS if per[s] == "L-PASS"]
    if all(ceiling):
        row = "REP-VOID-CEILING"
    elif len(passed) == len(MODEL_SEEDS):
        row = "REP-PASS"
    elif len(passed) == 1:
        row = "REP-ONE"
    else:
        row = "REP-NONE"
    return {
        "row": row,
        "meaning": COMBINED_ROWS[row],
        "clause_fires": False,
        "per_seed": {str(s): per[s] for s in MODEL_SEEDS},
        "passed": passed,
        "detectably_not_replicated": [s for s in MODEL_SEEDS if per[s] in DETECTABLE_FAIL_ROWS],
    }


def _binom_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1)))


exact_interval = cpv.exact_interval


def mcnemar_two_sided(b: int, c: int) -> float:
    """The exact two-sided McNemar p (twice the smaller tail, capped at 1); reported only."""
    b, c = int(b), int(c)
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2.0 * _binom_cdf(min(b, c), b + c, 0.5)))


def counts_and_pairs(outcomes: dict, reference: str) -> dict:
    """Reported only: every labelled arm's count with its exact 95 % interval, and every arm's
    paired difference from ``reference`` with its discordant counts and interval (salt 8501)."""
    w = np.asarray(outcomes[reference], bool)
    counts = {
        a: {"count": int(np.sum(v)), "n": len(v), "ci95": exact_interval(int(np.sum(v)), len(v))}
        for a, v in outcomes.items()
    }
    pairs = {}
    for a, v in outcomes.items():
        if a == reference:
            continue
        b, c = pt.discordant(w, v)
        pairs[a] = paired_interval(w, v) | {"b": b, "c": c}
    return {"counts": counts, "minus_arm": pairs}


def between_seeds(first, second) -> dict:
    """§7.4 item 3, reported only: W[66801] - W[66802] with a two-sided exact McNemar p."""
    b, c = pt.discordant(first, second)
    return paired_interval(first, second) | {"b": b, "c": c, "p_two_sided": mcnemar_two_sided(b, c)}


determinism_check = cpv.determinism_check

# ----- the clause and R7 (§9, §10, R21.5, R21.6) -------------------------------------------------
CLAUSE = (
    "none: no row of TASK-083 fires a clause; nothing in TASK-077's scope is closed or reopened"
)
R7_BY_ROW = {
    "REP-PASS": "R7 adds the replication: both other seeds L-PASS, each seed's count, interval and "
    "comparator difference, every qualifier kept (the flags with the last-point caveat, the "
    "solver's effect not shown); H-rule measurably better stated per seed",
    "REP-ONE": "R7 adds, factually, which seed passed and which did not (rows and counts), and "
    "that at the declared power this does not show whether seed or chance caused the difference "
    "unless the other seed is L-NO-GAIN or L-INFERIOR",
    "REP-NONE": "R7 adds that the result did not replicate with either other seed (rows, counts), "
    "with the same power caveat",
    "every_reading_row": "a seed at L-NO-GAIN or L-INFERIOR is stated with 'detectably'; W-66800's "
    "count on this cohort is stated, reported only, whatever it is; TASK-081's L-PASS stands",
    "REP-VOID-CEILING": "R7 unchanged",
    "V": "R7 unchanged",
}

# ----- caps, memory, guards (§11) ----------------------------------------------------------------
CAPS_SECONDS = {"S": 21_600.0, "per_attempt": 300.0, "simulate": 3_600.0}
CAP_FACTOR_MIN = 1.5
MEMORY = dict(cpv.MEMORY)  # 12 GiB process-tree PSS
DISK_MIN_GIB = cpv.DISK_MIN_GIB  # 10 GiB
WM_WORKERS = cpv.WM_WORKERS  # 4
WORKER_TORCH_THREADS = cpv.WORKER_TORCH_THREADS
THREAD_ENV = cpv.THREAD_ENV
QUIET_MACHINE = cpv.QUIET_MACHINE
VOID_RULE = {
    "void": pr.VOID_RULE["void"],
    "repeat": "a V in any pool voids the whole of Stage S; at most one repeat, after a committed, "
    "pushed and recorded fix, re-running every pool and arm on the same seeds in a new output "
    "directory; a second V ends TASK-083 INCONCLUSIVE; any repeat after a non-pass row needs fresh "
    "seeds and its own ruling",
    "debug": "debug runs only on committed code, only on 75900-75999; nothing in them is read",
}
STAGES = ("S",)

# ----- power (§8, R21.9; salt 8502) --------------------------------------------------------------
PLANNING = {"C": (0.979, 0.984, 0.992), "sysid": 183 / 192, "W": (0.906, 0.922, 0.938, 0.953)}
POWER_COUPLINGS = ("overlap", "half", "independent")
POWER_RESAMPLES = BOOTSTRAP_RESAMPLES
POWER_TRIALS = 20_000
_NI_CACHE: dict = {}
_UP_CACHE: dict = {}


def _ni_bounds(kp: int, km: int, *, n: int = S_RESETS) -> tuple[float, float]:
    """The reset-bootstrap 2.5th and 97.5th percentiles of W - C for (k+, k-): a multinomial draw
    (salt 8502, one stream per cell)."""
    key = (int(kp), int(km), int(n))
    if key not in _NI_CACHE:
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], 1, *key]))
        draws = rng.multinomial(n, [kp / n, km / n, (n - kp - km) / n], size=POWER_RESAMPLES)
        d = draws[:, 0] - draws[:, 1]
        _NI_CACHE[key] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return _NI_CACHE[key]


def rep_power(p_w: float, p_c: float, coupling: str, *, p_sysid: float = PLANNING["sysid"],
              n: int = S_RESETS, trials: int = POWER_TRIALS, index: int = 0) -> dict:  # fmt: skip
    """Per-seed and joint pass rates by simulation (salt 8502): H-rule at ``p_c``; each seed's W
    at ``p_w``, coupled to H-rule by ``coupling`` and conditionally independent of the other W
    given H-rule; H-sysid coupled to H-rule halfway; C the better of the two by count (a tie to
    H-rule). A seed passes when G-bar and G-NI pass (the twin tests' power is above 0.999 at
    TASK-081's twin rates and is not simulated). Also each seed's L-INFERIOR rate."""
    from embodied_jepa import commit_precision_dev as cpd

    rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], 2, int(index)]))
    rule = rng.uniform(size=(trials, n)) < p_c
    sysid = lm._draw_relative(rng, rule, p_c, *cpd.discordance(p_c, p_sysid, "half"))
    ws = [lm._draw_relative(rng, rule, p_c, *cpd.discordance(p_c, p_w, coupling))
          for _ in MODEL_SEEDS]  # fmt: skip
    c = np.where((rule.sum(1) >= sysid.sum(1))[:, None], rule, sysid)
    passes, inferior = [], []
    for w in ws:
        kp, km = (w & ~c).sum(1), (~w & c).sum(1)
        bounds = np.array([_ni_bounds(a, b, n=n) for a, b in zip(kp, km, strict=True)])
        passes.append((bounds[:, 0] > -DELTA) & (w.sum(1) >= G_BAR))
        inferior.append(bounds[:, 1] < -DELTA)
    both = passes[0] & passes[1]
    return {
        "p_w": p_w, "p_c": p_c, "p_sysid": p_sysid, "coupling": coupling, "n": n,
        "trials": trials,
        "per_seed_pass": [float(p.mean()) for p in passes],
        "both_pass": float(both.mean()),
        "exactly_one": float((passes[0] ^ passes[1]).mean()),
        "neither": float((~passes[0] & ~passes[1]).mean()),
        "per_seed_l_inferior": [float(i.mean()) for i in inferior],
    }  # fmt: skip


def power_tables(*, trials: int = POWER_TRIALS) -> dict:
    """§8 recomputed: every planning cell, and the size at the margin (W = C - delta)."""
    cells, k = [], 0
    for p_c in PLANNING["C"]:
        for p_w in PLANNING["W"]:
            for coupling in POWER_COUPLINGS:
                cells.append(rep_power(p_w, p_c, coupling, trials=trials, index=k))
                k += 1
    size = []
    for p_c in PLANNING["C"]:
        for coupling in POWER_COUPLINGS:
            size.append(rep_power(p_c - DELTA / S_RESETS, p_c, coupling, trials=trials, index=k))
            k += 1
    return {
        "cells": cells,
        "size_at_margin": size,
        "g_bar": {str(r): cpv.g_bar_power(r) for r in PLANNING["W"]},
        "trials": trials,
        "resamples_per_cell": POWER_RESAMPLES,
    }


# ----- helpers ------------------------------------------------------------------------------------
def is_missing(value) -> bool:
    return lm.is_missing(value)


def sentinel_fields(names) -> dict:
    return lm.sentinel_fields(names)


ROW_NAMES = frozenset({*S_ROWS, *COMBINED_ROWS})


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
STAGE0: dict = {}


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
            "task081_frozen_sha256": TASK081_FROZEN_SHA256,
            "task081_manifest": TASK081_MANIFEST,
            "task080_frozen_sha256": TASK080_FROZEN_SHA256,
            "task077_frozen_sha256": TASK077_FROZEN_SHA256,
            "own_files": OWN_FILES,
            "model_seeds": MODEL_SEEDS,
            "reference_seed": REFERENCE_SEED,
            "pool_order": POOL_ORDER,
            "seed_flags": {str(k): v for k, v in SEED_FLAGS.items()},
            "reused": REUSED | {"readouts": {str(k): v for k, v in REUSED["readouts"].items()}},
            "tau_commit_cm": TAU_COMMIT_CM,
            "tau_curve": TAU_CURVE,
            "tau_curve_resets": TAU_CURVE_RESETS,
            "seed_block": SEED_BLOCK,
            "seed_ranges": SEED_RANGES,
            "debug_seeds": DEBUG_SEEDS,
            "debug_ranges": DEBUG_RANGES,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "reserved_salts": RESERVED_SALTS,
            "carried_salts": CARRIED_SALTS,
            "condition": "TASK-081's (lewm_cp_v2 frozen block, carried unchanged)",
            "solver": cpv.SOLVER,
            "arms": {
                "per_seed": PER_SEED_ARMS,
                "shared": SHARED_ARMS,
                "reference": REFERENCE_ARMS,
                "dropped": DROPPED_ARMS,
                "solver_of": SOLVER_OF,
                "readouts": READOUTS,
                "twins": TWINS,
                "comparators": COMPARATORS,
                "privileged": PRIVILEGED_ARMS,
            },
            "s": {
                "resets": S_RESETS,
                "g_bar": G_BAR,
                "delta": DELTA,
                "mcnemar_p": MCNEMAR_P,
                "min_separation": MIN_SEPARATION,
                "determinism_resets": DETERMINISM_RESETS,
                "rows": S_ROWS,
            },
            "combined_rows": COMBINED_ROWS,
            "combination": COMBINATION,
            "detectable_fail_rows": DETECTABLE_FAIL_ROWS,
            "clause_rows": CLAUSE_ROWS,
            "clause": CLAUSE,
            "r7_by_row": R7_BY_ROW,
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
                "couplings": POWER_COUPLINGS,
                "trials": POWER_TRIALS,
                "resamples_per_cell": POWER_RESAMPLES,
            },
            "stage0": STAGE0,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()


check_seed_ranges()
if G_BAR != 112 or DELTA != 16 or MIN_SEPARATION != 7 or S_RESETS != 128:
    raise ContractError("S: G-bar 112/128, delta 16/128 and the McNemar minimum 7 (R21.3)")
if cpv.frozen_sha256() != TASK081_FROZEN_SHA256:
    raise ContractError("G-frozen: TASK-081's frozen block is not its pin")
