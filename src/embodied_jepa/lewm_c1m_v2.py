"""TASK-077 (``apple_lewm_c1m_v2``): LeWM chooses the single place aim committed at 405 under C1-M.

Protocol ``docs/experiments/apple_lewm_c1m_v2.md`` (STATUS DRAFT; rulings R17.1-R17.18 in
``docs/DECISIONS.md``, decision 2026-10-05 (b), each decided by Claude under owner delegation,
2026-09-30). This module is the protocol's **frozen block** as module constants, with its
statistics, the Stage-0 simulations and every stage's row ladder. It is DRAFT until the freeze:
K0's measured values are ``None`` here until a reported K0 enters them, and the sha pin of
:func:`frozen_sha256` is set only at the freeze (``tests/test_lewm_c1m_v2.py``).

LeWM makes one decision: W, a TASK-066-family LeWM token predictor on frozen DINOv2 tokens pooled
to 8 x 8, encodes the onboard frame at 405, rolls every candidate aim's 60 stand-in commands to
r = 465, reads the predicted plate with R8 and picks the aim whose predicted plate is nearest to
it. Before 405 P-3 (behaviour cloning, not a world model) picks; after 405 e9's scripted place
primitive executes the committed aim. Every other arm is a trained twin (N, L-shuf, L-mean), a
random choice (L-rand), a hand-written arm (H-rule, H-sysid) or a privileged reference
(H-final(commit), H-read, H-now, K0's proxies).

**G-sentinel** (#141 review note 2): every report field for a check or row a stage has not reached
holds :data:`NOT_EVALUATED`, never a row name; every ladder below refuses a missing input.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import token_dynamics as td
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_c1m_v2"
TASK = "TASK-077"
STATUS = "DRAFT"
DOCUMENT = "docs/experiments/apple_lewm_c1m_v2.md"
MANIFEST = "benchmarks/manifests/apple-lewm-c1m-v2.json"
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
RULINGS = "DECISIONS.md decision 2026-10-05 (b), R17.1-R17.18 (and R17.19 for Stage 0)"
GuardError = pt.GuardError
NOT_EVALUATED = "not evaluated"  # G-sentinel: a field a stage has not reached

# ----- code carried unchanged (§2.2): C1's and C1-M's code files, byte-identical to main at the
# draft's merge (c764ac9), and TASK-076's 84 pins through its manifest
CARRIED_CODE_REFERENCE = "c764ac9540033894c640e9ef5ca71ecba47339a0"
CARRIED_CODE_FILES = (
    *c1m.C1_CODE_FILES,
    "src/embodied_jepa/lewm_next_c1m.py",
    "src/embodied_jepa/lewm_next_c1m_runtime.py",
    "scripts/run_c1m_feasibility.py",
)
CARRIED_CODE_BLOBS = {  # git blob ids at CARRIED_CODE_REFERENCE (checked without history)
    "src/embodied_jepa/lewm_next_c1.py": "ea988ad859184de021fd0841bb7333eebd1837e3",
    "src/embodied_jepa/lewm_next_c1_runtime.py": "fb230410659391fe7ae1ee72054e42fc95df2eb6",
    "scripts/run_c1_feasibility.py": "2084b13f8fc10552c5856b21c9d30931513e70f0",
    "src/embodied_jepa/lewm_next_c1m.py": "11efc4bc184078e8eb726d2186f6c57d889ccedb",
    "src/embodied_jepa/lewm_next_c1m_runtime.py": "0adf72dacdb2696f8a1457d32ab12feb40e75904",
    "scripts/run_c1m_feasibility.py": "3d4d9f5ebfbd81c034191745fda11f00041c0335",
}
if tuple(CARRIED_CODE_BLOBS) != CARRIED_CODE_FILES:
    raise ContractError("every carried code file has its recorded blob")
TASK076_MANIFEST = "benchmarks/manifests/apple-plate-twin-v2.json"
OWN_FILES = (
    "src/embodied_jepa/lewm_c1m_v2.py",
    "src/embodied_jepa/lewm_c1m_v2_runtime.py",
    "src/embodied_jepa/lewm_c1m_v2_offline.py",
    "src/embodied_jepa/lewm_c1m_v2_train.py",
    "scripts/run_lewm_c1m_v2.py",
)

# ----- seeds and salts (§6, R17.11) --------------------------------------------------------------
SEED_BLOCK = (66000, 68999)
SEED_RANGES = {
    "K": (66000, 66031),  # K0, before the freeze
    "D": (66100, 66115),  # the development closed loop
    "S": (66200, 66263),  # the gated cohort
    "corpus": (67000, 68999),  # 2 000 roots
}
MODEL_SEEDS = (66800, 66801, 66802)  # W and N (torch; not resets)
CALIBRATION_SEED = 66810  # the calibration W and N (torch; not a reset)
DEBUG_SEEDS = (66900, 66999)  # runner mechanics only; nothing in them is read
DEBUG_RANGES = {
    "K": (66900, 66903),
    "D": (66910, 66913),
    "S": (66920, 66923),
    "corpus": (66940, 66979),
}
DEBUG_MODEL_SEEDS = (66992, 66993)  # debug torch seeds (66990-66991: the cost probe, §15)
DEBUG_CALIBRATION_SEED = 66994
PROBE_TORCH_SEEDS = (66990, 66991)  # the draft's cost probe (§15), synthetic data
FORBIDDEN_RANGES = dict(c1m.FORBIDDEN_RANGES) | {"c1m_block": c1m.SEED_BLOCK}
SALTS = {
    "move": 8101,  # default_rng(SeedSequence([8101, seed, k])), k the re-draw index
    "corpus_aim": 8102,  # the corpus's uniform (a, b) per root
    "corpus_split": 8103,  # train / val / gate, one permutation, fixed before collection
    "outer_folds": 8104,  # Stage O's outer folds by root
    "inner_folds": 8105,  # lambda selection in every ridge
    "bootstrap": 8106,  # every bootstrap and every feasibility resampling (Stage-0 sims too)
    "tau_direction": 8107,  # K0's planted-error direction per reset
    "learning_curve": 8108,  # Stage O's nested learning-curve subsample
    "sampler": 8109,  # the training window sampler, SeedSequence([8109, model seed])
    "wrong_commands": 8110,  # G4's wrong-command permutation across gate roots
    "l_rand": 8111,  # L-rand's draw
}
RESERVED_SALTS = {"reserved": 8112}  # any use needs its own ruling
EARLIER_SALTS = {
    **c1m.EARLIER_SALTS,
    **{f"c1m_{k}": v for k, v in c1m.SALTS.items()},
}


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in the block, are disjoint, avoid every forbidden range (TASK-076, C1, C1-M,
    R15's 58000-58999); the salts are fresh (R17.11)."""
    spans = (
        dict(SEED_RANGES)
        | {"debug": DEBUG_SEEDS}
        | {f"model_{s}": (s, s) for s in MODEL_SEEDS}
        | {"calibration": (CALIBRATION_SEED, CALIBRATION_SEED)}
    )
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two TASK-077 ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < SEED_BLOCK[0] or high > SEED_BLOCK[1]:
            raise GuardError(f"G-seeds: the range {name} leaves the block {SEED_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"G-seeds: the range {name} overlaps the forbidden {other}")
    for low, high in DEBUG_RANGES.values():
        if low < DEBUG_SEEDS[0] or high > DEBUG_SEEDS[1]:
            raise GuardError("G-seeds: a debug range leaves the debug block")
    for s in (*DEBUG_MODEL_SEEDS, DEBUG_CALIBRATION_SEED, *PROBE_TORCH_SEEDS):
        if not DEBUG_SEEDS[0] <= s <= DEBUG_SEEDS[1]:
            raise GuardError("G-seeds: a debug torch seed leaves the debug block")
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
    mine = [*SALTS.values(), *RESERVED_SALTS.values()]
    if len(set(mine)) != len(mine) or set(mine) & used:
        raise GuardError("G-seeds: a TASK-077 salt repeats or is an earlier task's salt")
    if sorted(mine) != list(range(8101, 8113)):
        raise GuardError("G-seeds: the salts are 8101-8112")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """Plain ints; exactly the role's seeds in order (distinct debug seeds in a debug run)."""
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


def check_model_seed(seed: int, *, debug: bool = False) -> int:
    allowed = (
        (*DEBUG_MODEL_SEEDS, DEBUG_CALIBRATION_SEED) if debug else (*MODEL_SEEDS, CALIBRATION_SEED)
    )
    if type(seed) is not int or seed not in allowed:
        raise GuardError(f"G-seeds: model seed {seed!r} is not one of {allowed}")
    return seed


reset_of = c1.reset_of

# ----- the condition (§2.1; C1-M as recorded, R17.2) --------------------------------------------
RULE = c1.RULE  # cell A: kappa = -0.5, L = 2, s0 = 405, s1 = 525
COMMIT_STEP = c1.COMMIT_STEP  # 405
MOVE_STEP = c1m.MOVE_STEP  # 300
RHO_CM = 4  # rho* (M-F1), the disc
FAMILY = "disc"
A_LO = -0.5  # M-F2
A_HI = c1.A_HI
B_HALF_M = c1.B_HALF_M
READ_STEP = 465  # r (M-F2)
HORIZON = READ_STEP - COMMIT_STEP  # 60
GRID = tuple(c1.grid(A_LO))  # row-major over (a, b): 21 x 7 = 147 candidates
TAU_COMMIT_RECORD_CM = 1.0  # the C1-M record's M-F4; K0 re-measures it, K0's value is used
LOOKAHEAD_MAX_ITER = c1.MAX_ITERATIONS  # 10
REFINE_MAX = 10  # W's refinement: at most 10 single-candidate roll-outs
REFINE_FRACTION_OF_TAU = 0.25  # stop when the move is <= tau_commit / 4
P3_CHECKPOINT_SHA256 = pt.P3_CHECKPOINT_SHA256
if len(GRID) != 147 or HORIZON != 60:
    raise ContractError("the box grid has 147 candidates and the horizon is 60 steps")
if RULE["kappa"] != -0.5 or RULE["L"] != 2 or RULE["s0"] != 405 or RULE["s1"] != 525:
    raise ContractError("the rule is cell A's")

# the move (R16.3's rule, this task's salt): u, v with k = 0 first; k grows while the reset's plate
# plus the offset at C1-M's largest radius (6 cm) in either family leaves the tabletop
MOVE_REDRAW_RADIUS_CM = max(c1m.RHO_GRID_CM)  # 6, as R16.3
MOVE_REDRAW_LIMIT = c1m.REDRAW_LIMIT


def move_uniforms(seed: int) -> dict:
    plate = np.asarray(reset_of(seed)["plate_xy"], np.float64)
    largest = MOVE_REDRAW_RADIUS_CM / 100.0
    for k in range(MOVE_REDRAW_LIMIT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["move"], int(seed), k]))
        u, v = float(rng.uniform()), float(rng.uniform())
        if all(c1m._on_table(plate + c1m._offset(u, v, f, largest)) for f in c1m.FAMILIES):
            return {"u": u, "v": v, "k": k}
    raise GuardError(f"G-move: seed {seed} found no on-table draw in {MOVE_REDRAW_LIMIT} re-draws")


def move_offset(seed: int) -> np.ndarray:
    """The reset's offset (world xy, m): rho* sqrt(u) (cos 2 pi v, sin 2 pi v), rho* = 4 cm."""
    d = move_uniforms(seed)
    return c1m._offset(d["u"], d["v"], FAMILY, RHO_CM / 100.0)


def moved_plate(seed: int) -> np.ndarray:
    return np.asarray(reset_of(seed)["plate_xy"], np.float64) + move_offset(seed)


PLATE_MEAN = tuple(
    float(v) for v in c1m.plate_mean(FAMILY, RHO_CM)
)  # the disc's mean: (0.49, -0.09)


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    """K0's planted error at ``level_cm`` in the reset's direction (salt 8107)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["tau_direction"], int(seed)]))
    angle = float(2.0 * np.pi * rng.uniform())
    r = float(level_cm) / 100.0
    return [r * math.cos(angle), r * math.sin(angle)]


def corpus_aim(seed: int) -> tuple[float, float]:
    """The corpus's (a, b), uniform over the box (salt 8102)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    return float(rng.uniform(A_LO, A_HI)), float(rng.uniform(-B_HALF_M, B_HALF_M))


def l_rand_index(seed: int, feasible) -> int:
    """L-rand: a uniform draw over the feasible grid candidates (salt 8111); -1 if none."""
    feasible = np.flatnonzero(np.asarray(feasible, bool))
    if len(feasible) == 0:
        return -1
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["l_rand"], int(seed)]))
    return int(feasible[int(rng.integers(0, len(feasible)))])


def foreign_index(i: int, n: int) -> int:
    """L-shuf's foreign reset when every reset reached 405: (i + 1) mod n in the cohort's order."""
    return (int(i) + 1) % int(n)


def foreign_reached(i: int, reached) -> int | None:
    """L-shuf's foreign reset (R17.20): the next reset after ``i`` in cohort order, cyclically,
    whose W attempt reached 405 (logged its 405 frame). A reset refused before 405 fails in P-3's
    shared pick for every arm and needs no frame; ``None`` only if no other reset reached 405."""
    reached = [bool(r) for r in reached]
    n = len(reached)
    for k in range(1, n):
        j = (int(i) + k) % n
        if reached[j]:
            return j
    return None


def reached_405(record: dict) -> bool:
    """Whether an attempt reached step 405 (it logged the 405 frame)."""
    return record.get("frame405") is not None


def determinism_check(first: list[dict], again: list[dict], tol_m: float | None = None) -> dict:
    """W's re-run on S's first resets (R17.21): a reset refused before 405 must be refused again
    identically (termination reason and executed steps); a committed reset must commit again,
    with its R-plate reading and its commit target each within the declared tolerance (0.1 cm).
    ``ok`` is False on any mismatch (the stage is V)."""
    tol = DETERMINISM_TOLERANCE_M if tol_m is None else float(tol_m)
    items, ok = [], True
    for a, b in zip(first, again, strict=True):
        if a["seed"] != b["seed"]:
            raise ContractError("the re-run must cover the same resets in order")
        ca, cb = bool(a.get("commit")), bool(b.get("commit"))
        item = {"seed": a["seed"], "committed": [ca, cb]}
        if ca != cb:
            item["ok"] = False
        elif not ca:
            same = (a.get("termination_reason"), a.get("executed_steps")) == (
                b.get("termination_reason"),
                b.get("executed_steps"),
            )
            item |= {"refused_identically": same, "ok": same}
        else:
            ra = np.asarray(a["decisions"][0].get("reading", a["commit"]["target"]), np.float64)
            rb = np.asarray(b["decisions"][0].get("reading", b["commit"]["target"]), np.float64)
            dt = float(np.linalg.norm(np.asarray(a["commit"]["target"]) - b["commit"]["target"]))
            dr = float(np.linalg.norm(ra - rb))
            item |= {"target_diff_m": dt, "reading_diff_m": dr, "ok": dt <= tol and dr <= tol}
        ok = ok and item["ok"]
        items.append(item)
    return {"tolerance_m": tol, "resets": items, "ok": bool(ok)}


# ----- the corpus (§4.3) -------------------------------------------------------------------------
CORPUS_ROOTS = 2000
SPLIT_SIZES = {"train": 1500, "val": 250, "gate": 250}
DEBUG_SPLIT_SIZES = {"train": 30, "val": 5, "gate": 5}
FRAME_STEPS = (403, 467)  # kept onboard 112 px frames, inclusive (65)
COMMAND_STEPS = (403, 466)  # executed normalised 14-D commands, inclusive (64)
STOP_STEP = 467  # a root runs to step 467 and stops (the observation of 467 is its last frame)
WINDOW_START_STEPS = (403, 407)  # a training window starts uniformly in 403-407
N_FRAMES = FRAME_STEPS[1] - FRAME_STEPS[0] + 1
N_COMMANDS = COMMAND_STEPS[1] - COMMAND_STEPS[0] + 1
CORPUS_EXCLUDED_MAX_FRACTION = 0.02  # above it: CORPUS-ESCALATE
if N_FRAMES != 65 or N_COMMANDS != 64 or sum(SPLIT_SIZES.values()) != CORPUS_ROOTS:
    raise ContractError("the corpus keeps 65 frames and 64 commands of 2 000 roots")
if WINDOW_START_STEPS[1] + HORIZON > FRAME_STEPS[1]:
    raise ContractError("a training window must end inside the kept frames")


def corpus_split(seeds, *, debug: bool = False) -> dict[str, list[int]]:
    """One permutation of the corpus seeds (salt 8103): the first 1 500 train, then 250 val, then
    250 gate (debug: 30, 5, 5). Fixed before collection; split by root."""
    seeds = [int(s) for s in seeds]
    sizes = DEBUG_SPLIT_SIZES if debug else SPLIT_SIZES
    if len(seeds) != sum(sizes.values()) or len(set(seeds)) != len(seeds):
        raise ContractError("the split covers every corpus root once")
    order = np.random.default_rng(SALTS["corpus_split"]).permutation(len(seeds))
    out, lo = {}, 0
    for name, size in sizes.items():
        out[name] = sorted(seeds[i] for i in order[lo : lo + size])
        lo += size
    return out


def decide_corpus(roots: int, excluded: int) -> dict:
    _require(roots=roots, excluded=excluded)
    fraction = int(excluded) / int(roots)
    row = "CORPUS-ESCALATE" if fraction > CORPUS_EXCLUDED_MAX_FRACTION else "CORPUS-SEALED"
    return {"row": row, "excluded": int(excluded), "fraction": fraction, "clause_fires": False}


# ----- the model and training (§4) ---------------------------------------------------------------
BACKEND = td.BACKEND  # "leworldmodel": the one-line swap
TOKEN_GRID = 8
LATENT_DIM = TOKEN_GRID * TOKEN_GRID * td.TOKEN_WIDTH  # 24 576
MODEL_CONFIG = dict(td.MODEL_CONFIG) | {"token_grid": TOKEN_GRID, "latent_dim": LATENT_DIM}
TRAIN_HORIZON = HORIZON  # T = 60 (departure 1)
BATCH_SIZE = 16  # departure 2 (BatchNorm's batch statistics: departure 3)
ARMS_TRAINED = ("W", "N")
METRIC_FLOOR_STD = td.METRIC_FLOOR_STD  # 1e-3
DEVICE_TRAIN = "cuda"  # strict determinism, devices.configure_determinism("cuda", strict=True)
DEVICE_EVAL = "cpu"  # every prediction a gate or the closed loop reads: CPU, float32, 1 thread
if MODEL_CONFIG["max_horizon"] < HORIZON:
    raise ContractError("max_horizon must cover the 60-step roll-out")


def sampler_seed(model_seed: int) -> np.random.SeedSequence:
    """W and N of a seed draw the same windows in the same order (salt 8109)."""
    return np.random.SeedSequence([SALTS["sampler"], int(model_seed)])


BUDGET = {
    "calibration_runs": (("W", CALIBRATION_SEED), ("N", CALIBRATION_SEED)),
    "calibration_updates": 50_000,
    "calibration_select_every": 1_000,
    "saturation_tolerance": 0.01,
    "factor": 2.0,
    "step": 5_000,
    "min": 10_000,
    "cap": 100_000,
    "selection_points": 20,
}
SELECTION_TOLERANCE = BUDGET["saturation_tolerance"]


def budget_rule(u_sat_w: int, u_sat_n: int) -> dict:
    """U = clamp(5 000 ceil(2 max(u_sat) / 5 000), 10 000, 100 000); never escalates (§4.5)."""
    _require(u_sat_w=u_sat_w, u_sat_n=u_sat_n)
    values = [int(u_sat_w), int(u_sat_n)]
    if min(values) < 1 or max(values) > BUDGET["calibration_updates"]:
        raise ContractError("a saturation update lies inside the calibration run")
    wanted = BUDGET["factor"] * max(values)
    updates = int(BUDGET["step"] * math.ceil(wanted / BUDGET["step"]))
    updates = min(max(updates, BUDGET["min"]), BUDGET["cap"])
    if updates % BUDGET["selection_points"]:
        raise ContractError("U must divide into the selection points")
    return {
        "u_sat": {"W": values[0], "N": values[1]},
        "wanted": wanted,
        "updates": updates,
        "select_every": updates // BUDGET["selection_points"],
        "escalates": False,
    }


# ----- K0 (§7 step 2) ----------------------------------------------------------------------------
K_RESETS = 32
TAU_LEVELS_CM = c1m.TAU_LEVELS_CM  # 0, 0.5, 1, 1.5, 2, 3
TAU_BAR = 28
K0_CEILING_MIN = 30
R_TOLERANCE_CM = c1.R_TOLERANCE_CM  # 0.1
R_FRACTION_MIN = 28  # of 32 (R14.3's 87.5 %)
R_LATEST = READ_STEP  # r_K later than 465: CAL-ESCALATE
PALM_SPEED_LIMIT_CM = 0.5  # per step at 405 (the history check)
K0_PROXIES = ("H-now", "N-proxy", "shuf-proxy", "mean-proxy")
K0_SCENE_BLIND = ("shuf-proxy", "mean-proxy")
FEASIBILITY_PAIRS = 64
FEASIBILITY_BAR = 0.8  # below it: disclosed as a known risk, not a stop
K0_STOPS = {
    "level0_below_bar": "level 0 < 28/32 (tau_commit undefined)",
    "ceiling_below_30": "N_K(0) < 30/32",
    "r_late": "r_K later than 465 or undefined (the frozen h = 60 would read the plate before it "
    "settles)",
    "palm_fast": "the median palm speed at 405 > 0.5 cm per step (a history-two predictor would be "
    "a different model)",
}
K0_MEASURED = None  # filled from K0's report on a reported K0-PASS, before the freeze


def decide_tau(counts: dict) -> dict:
    return c1m.decide_tau(counts)


def read_step_k(distances) -> int | None:
    """r_K: the earliest step in [405, 525] at which >= 28/32 level-0 attempts are within 0.1 cm
    of plate(525) (R14.3; a refused attempt is never within)."""
    need = math.ceil(R_FRACTION_MIN / K_RESETS * len(distances) - 1e-12)  # 28 of K's 32
    for t in range(RULE["s0"], RULE["s1"] + 1):
        within = sum(1 for d in distances if d is not None and d.get(t, math.inf) <= R_TOLERANCE_CM)
        if within >= need:
            return t
    return None


def decide_k0(*, tau: dict, ceiling: int, r_k, palm_speed_median_cm) -> dict:
    """K0's stops, all CAL-ESCALATE (escalate, no clause, nothing frozen), else K0-PASS."""
    _require(tau=tau, ceiling=ceiling, palm_speed_median_cm=palm_speed_median_cm)
    stops = {
        "level0_below_bar": tau["tau_commit_cm"] is None,
        "ceiling_below_30": int(ceiling) < K0_CEILING_MIN,
        "r_late": r_k is None or int(r_k) > R_LATEST,
        "palm_fast": float(palm_speed_median_cm) > PALM_SPEED_LIMIT_CM,
    }
    row = "CAL-ESCALATE" if any(stops.values()) else "K0-PASS"
    return {"row": row, "stops": stops, "clause_fires": False}


# ----- Stage O (§8.1) ----------------------------------------------------------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 5
LAMBDA_GRID_RELATIVE = c1m.LAMBDA_GRID_RELATIVE
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_RESAMPLES = 10_000
REPORTED_GRIDS = (4,)  # 4 x 4 at 405 and r: reported only (caveat 2)
O_ROWS = {
    "O-ARM-KEYED": "O4 fails: escalate, no clause",
    "O-NO-BAR": "O1 or O3 fails: escalate, no clause (a still-falling curve says so)",
    "O-PASS": "Stage O admits the stages that follow",
}


def decide_o(*, c_plate: dict, prior_ratio: dict, hidden: dict, tau_commit_cm, falling) -> dict:
    """O1: upper bound of the cross-fitted 8 x 8 readout's median at r <= tau_commit; O3: upper
    bound of median(e) / median(prior) < 1.0; O4: lower bound of the plate-hidden median >
    tau_commit."""
    _require(c_plate=c_plate, prior_ratio=prior_ratio, hidden=hidden, tau_commit_cm=tau_commit_cm)
    o1 = bool(c_plate["ci95"][1] <= float(tau_commit_cm))
    o3 = bool(prior_ratio["ci95"][1] < 1.0)
    o4 = bool(hidden["ci95"][0] > float(tau_commit_cm))
    if not o4:
        row = "O-ARM-KEYED"
    elif not (o1 and o3):
        row = "O-NO-BAR"
    else:
        row = "O-PASS"
    return {
        "row": row,
        "O1": o1,
        "O3": o3,
        "O4": o4,
        "still_falling": falling,
        "meaning": O_ROWS[row],
        "clause_fires": False,
    }


# ----- Stage T (§4.4-§4.5) -----------------------------------------------------------------------
G1_RANK_FLOOR = td.RANK_FLOOR  # 0.10
G1_STD_FLOOR = td.STD_FLOOR  # 0.25
G1_BAR_FACTOR = td.BAR_FACTOR  # 0.5
CAL_T_ESCALATE_BELOW = td.BAR_ESCALATE_BELOW  # 0.10
TRUNCATION_CONTROLS = (1, 2, 4)  # the combined G1 must fail each (TASK-066's owner condition)


def g1_bars(rank_reference, std_reference) -> dict:
    """max(floor, floor_to_0.01(0.5 x the calibration W's ratio on val at h = 60))."""
    _require(rank_reference=rank_reference, std_reference=std_reference)
    rank, std = td.bar_rule([rank_reference], G1_RANK_FLOOR), td.bar_rule([std_reference], 0.0)
    std["bar"] = max(G1_STD_FLOOR, std["relative"])
    std["floor"] = G1_STD_FLOOR
    return {"rank": rank, "std": std}


def decide_t(*, rank_reference, jobs_complete: bool) -> dict:
    _require(rank_reference=rank_reference)
    if not jobs_complete:
        raise ContractError("Stage T's row needs every job complete")
    row = "CAL-T-ESCALATE" if float(rank_reference) < CAL_T_ESCALATE_BELOW else "T-DONE"
    return {"row": row, "clause_fires": False}


# ----- Stage G (§8.2) ----------------------------------------------------------------------------
G_THRESHOLDS = {
    "G1_max_collapsed_fraction": 0.05,  # carried (TASK-065)
    "G2_max_ratio_upper": 0.8,  # carried, an allocation
    "G3_max_ratio_upper_exclusive": 1.0,  # definitional
    "G4_min_ratio_lower": 1.10,  # carried, an allocation
    "G5_ratio_upper_exclusive": 1.0,  # definitional; G5 (a)'s bar is tau_commit (measured)
}
REPORTED_HORIZONS = (16, 30)
COMPARATIVE_RESAMPLES = 2_000
COLLAPSE_STD = 0.01  # a dimension with std < 0.01 of the metric scale is collapsed
G_ROWS = {
    "H-GATE-FAIL": "any of G1-G4 fails on any seed: escalate, no clause",
    "G-NO-BAR": "G1-G4 pass on all seeds and G5 fails on any: escalate, no clause",
    "G-PASS": "the offline gates pass on all three seeds",
}


def g1_passes(stats: dict, bars: dict, comparative: dict) -> dict:
    _require(stats=stats, bars=bars, comparative=comparative)
    parts = {
        "collapsed": stats["predicted_collapsed_fraction"]
        <= G_THRESHOLDS["G1_max_collapsed_fraction"],
        "rank": stats["effective_rank_ratio"] >= bars["rank"]["bar"],
        "std": stats["std_ratio"] >= bars["std"]["bar"],
        "rank_over_N": comparative["undefined_resamples"] == 0 and comparative["ci95"][0] > 0.0,
    }
    return {k: bool(v) for k, v in parts.items()} | {"passes": bool(all(parts.values()))}


def decide_g(per_seed: dict, *, debug: bool = False) -> dict:
    """``per_seed[seed] = {"G1": bool, ..., "G5": bool}`` for every model seed (a debug run:
    the one debug model seed, never in a real run)."""
    _require(per_seed=per_seed)
    expected = DEBUG_MODEL_SEEDS[:1] if debug else MODEL_SEEDS
    if sorted(per_seed) != sorted(expected):
        raise ContractError(f"Stage G needs every model seed {expected}")
    for seed, gates in per_seed.items():
        missing = [g for g in ("G1", "G2", "G3", "G4", "G5") if not isinstance(gates.get(g), bool)]
        if missing:
            raise ContractError(f"seed {seed}: gates {missing} not evaluated")
    if not all(all(g[k] for k in ("G1", "G2", "G3", "G4")) for g in per_seed.values()):
        row = "H-GATE-FAIL"
    elif not all(g["G5"] for g in per_seed.values()):
        row = "G-NO-BAR"
    else:
        row = "G-PASS"
    return {"row": row, "meaning": G_ROWS[row], "clause_fires": False}


# ----- Stage D (§7 step 8) -----------------------------------------------------------------------
D_RESETS = 16
D_ARMS = ("W", "N", "L-shuf", "L-mean", "H-final")
D_BARS = {"W_min": 12, "ceiling_min": 14, "margin_min": 3}  # of 16


def decide_d(counts: dict) -> dict:
    _require(counts=counts)
    missing = [a for a in D_ARMS if not isinstance(counts.get(a), int)]
    if missing:
        raise ContractError(f"Stage D needs the counts of {missing}")
    twins = max(counts["N"], counts["L-shuf"], counts["L-mean"])
    stops = {
        "W_below_12": counts["W"] < D_BARS["W_min"],
        "ceiling_below_14": counts["H-final"] < D_BARS["ceiling_min"],
        "margin_below_3": counts["W"] - twins < D_BARS["margin_min"],
    }
    row = "L-DEV-STOP" if any(stops.values()) else "D-PASS"
    return {"row": row, "stops": stops, "clause_fires": False}


# ----- Stage S (§8.4) ----------------------------------------------------------------------------
S_RESETS = 64
S_ARMS = ("W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid", "H-final", "H-read", "H-now")
TWINS = ("N", "L-shuf", "L-mean", "L-rand")
COMPARATORS = ("H-rule", "H-sysid")  # the better on S; a tie goes to H-rule
G_BAR = 56  # of 64 (tau_commit's bar fraction 28/32)
DELTA = 8  # of 64: an allocation (the design note's proposal, R15.6's label), not calibrated
MCNEMAR_P = 0.01
MIN_SEPARATION = 7  # of 64: the smallest b - c at which the one-sided McNemar test can pass
DETERMINISM_RESETS = 4  # W's re-run on S's first four resets
DETERMINISM_TOLERANCE_M = 0.001  # 0.1 cm on the R-plate reading and the commit target (R17.21):
# > 15 x the renderer noise seen in Stage 0 (about 6e-5 m in the 405 reading) and 10 x below tau
S_ROWS = {
    "V": "one repeat of the stage after a recorded fix",
    "S-VOID-CEILING": "H-final(commit)(S) < 56/64: escalate, no clause, no claim",
    "L-NO-GAIN": "for a twin, the McNemar test fails and W is detectably no better (upper bound of "
    "W - arm < +7/64): the clause fires",
    "L-INFERIOR": "W is detectably inferior beyond delta (upper bound of W - C < -8/64): the "
    "clause fires",
    "L-PASS": "G-bar, G-NI and the four McNemar tests pass: the primary claim, 'LeWM-driven "
    "closed-loop success'",
    "L-TWIN-NEAR": "a McNemar test fails and every failed one is a miss within noise: escalate, "
    "no clause, no claim",
    "L-NEAR": "G-NI fails, W not detectably inferior: escalate, no clause, no claim",
    "L-BAR": "G-bar fails with G-NI and the tests passing: escalate, no clause, no claim",
}
CLAUSE_ROWS = frozenset({"L-NO-GAIN", "L-INFERIOR"})
NO_CLAUSE_ROWS = (
    "CAL-ESCALATE",
    "CORPUS-ESCALATE",
    "O-ARM-KEYED",
    "O-NO-BAR",
    "CAL-T-ESCALATE",
    "H-GATE-FAIL",
    "G-NO-BAR",
    "L-DEV-STOP",
    "S-VOID-CEILING",
    "L-TWIN-NEAR",
    "L-NEAR",
    "L-BAR",
    "V",
    "INCONCLUSIVE",
)
if math.comb(MIN_SEPARATION, MIN_SEPARATION) / 2.0**MIN_SEPARATION >= MCNEMAR_P or (
    pt.mcnemar_one_sided(MIN_SEPARATION - 1, 0) < MCNEMAR_P
):
    raise ContractError("7 discordant pairs, all W's, is the McNemar test's minimum separation")


def comparator(outcomes: dict) -> str:
    """The better of H-rule and H-sysid on S (the larger count; a tie goes to H-rule)."""
    rule, sysid = int(np.sum(outcomes["H-rule"])), int(np.sum(outcomes["H-sysid"]))
    return "H-rule" if rule >= sysid else "H-sysid"


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


def decide_s(outcomes: dict | None, *, void: bool = False) -> dict:
    """§8.4's ladder, first match: V, S-VOID-CEILING, L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR,
    L-NEAR, L-BAR. ``outcomes``: arm -> 64 paired booleans for W, the twins, both comparators and
    H-final."""
    if void:
        return {"row": "V", "meaning": S_ROWS["V"], "clause_fires": False}
    _require(outcomes=outcomes)
    needed = ("W", *TWINS, *COMPARATORS, "H-final")
    missing = [a for a in needed if a not in outcomes or outcomes[a] is None]
    if missing:
        raise ContractError(f"Stage S's ladder needs {missing}")
    w = np.asarray(outcomes["W"], bool)
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
        "clause_fires": row in CLAUSE_ROWS,
        "counts": {a: int(np.sum(outcomes[a])) for a in outcomes if outcomes[a] is not None},
        "comparator": c_arm,
        "G-bar": g_bar,
        "G-NI": ni | {"passes": g_ni, "detectably_inferior": inferior},
        "twins": twins,
        "ceiling": ceiling,
        "secondary_reported_only": secondary,
    }


# ----- the clause (§11, R17.10) ------------------------------------------------------------------
CLAUSE_SCOPE = (
    "LeWM aim selection with a single aim committed at 405 under the declared reactive-plate rule "
    "(kappa = -0.5, L = 2, s1 = 525) on v2, with the declared post-pick plate move of radius "
    "rho* = 4 cm (the disc, which also covers the -y half-disc, at radii <= 4 cm), from onboard "
    "112 px frozen DINOv2 pooled tokens, with TASK-066-family predictors"
)
CLAUSE_NARROWING = "a 4 x 4 pooled latent on a corpus larger than C1-M's 1 024 roots is not covered"
CLAUSE_RESULTS_WORDING = (
    "if the clause fires, the results document states plainly that, with the design note's scope "
    "restored (R15.8), one 8 x 8 dual-ridge run closes under this condition every other pooled "
    "grid (except 4 x 4 on a corpus larger than 1 024 roots) and every readout of the predicted "
    "latent, including trained readout heads (#142 approval, note 3)"
)
NEVER_CLOSED = (
    "a 4 x 4 latent on a larger corpus",
    "the full, unpooled token grid",
    "history longer than one, action_chunk or predictor_step_embedding",
    "a fine-tuned or another encoder",
    "other views or resolutions",
    "other kappa, L, commit steps, move distributions or radii above 4 cm",
    "C1 without the move",
    "C2",
    "TASK-076's results",
    "the LeWM backend",
    "v2",
    "the product goal",
)

# ----- arms (§5) ---------------------------------------------------------------------------------
ARMS = {
    "W": {
        "what": "the trained W of the primary seed, from the encoded 405 frame",
        "privileged": False,
    },
    "N": {
        "what": "the trained action-blind N of the primary seed, zero commands",
        "privileged": False,
    },
    "L-shuf": {
        "what": "W from the logged 405 frame of W's attempt on reset (i + 1) mod n, this reset's "
        "candidate commands",
        "privileged": False,
    },
    "L-mean": {
        "what": "W from the train split's mean encoded 405 latent (raw feature space, Stage O)",
        "privileged": False,
    },
    "L-rand": {
        "what": "a uniform feasible grid candidate (salt 8111), no refinement",
        "privileged": False,
    },
    "H-rule": {"what": "RuleCommit on p-hat and h, clipped", "privileged": False},
    "H-sysid": {
        "what": "SysidAim fitted on the train split, out-of-fold readings",
        "privileged": False,
    },
    "H-final": {
        "what": "H-final(commit): LookaheadAim at 405 in cloned state, unclipped",
        "privileged": True,
    },
    "H-read": {
        "what": "H-final(commit)'s look-ahead, plate(r) read by R8 from the clone's frame at r",
        "privileged": True,
    },
    "H-now": {"what": "the true plate at 405", "privileged": True},
}
WORLD_MODEL_ARMS = ("W", "N", "L-shuf", "L-mean")
CANDIDATE_ARMS = ("W", "N", "L-shuf", "L-mean", "L-rand")  # share W's grid from R-plate's p-hat
NON_PRIVILEGED_ARMS = frozenset(a for a, s in ARMS.items() if not s["privileged"])
PRIVILEGED_ARMS = frozenset(
    {a for a, s in ARMS.items() if s["privileged"]}
    | {"H-final-planted", "N-proxy", "shuf-proxy", "mean-proxy", "collect"}
)

# ----- caps, memory, guards (§10) ----------------------------------------------------------------
CAPS_SECONDS = {  # provisional (§10.3); re-set from Stage 0's scale probe before the freeze
    "K0": 7_200.0,
    "C": 14_400.0,
    "O_featurisation": 3_600.0,
    "O_readouts": 7_200.0,
    "T_job": 46_800.0,
    "G": 7_200.0,
    "D": 7_200.0,
    "S": 21_600.0,
    "per_attempt": 300.0,
}
CAP_FACTOR_MIN = 1.5  # every cap >= 1.5 x the measured worst case
MEMORY = {
    "ceiling_gib": 12.0,  # process-tree PSS, CPU stages
    "train_ceiling_gib": 18.0,  # one Stage T job (its train-split cache is about 9.6 GB)
    "headroom_gib": pt.MEMORY["headroom_gib"],
    "sample_seconds": 0.5,
    "measure": "pss",
}
DISK_MIN_GIB = 10.0
DISK_MIN_START_GIB = {"C": 25.0, "O": 25.0}
GPU = {"min_free_gib": 8.0, "who": "oej:task077-<stage>", "require_lock": True}
SIM_WORKERS = pt.SIM_WORKERS  # 6
WM_WORKERS = 4  # stages whose workers load a world model (G, D, S): about 1.7 GiB PSS each
# (Stage 0's debug D and S with 6 workers tripped the 12 GiB ceiling; TASK-073/074's H_WORKERS)
WORKER_TORCH_THREADS = 1
THREAD_ENV = pt.THREAD_ENV
QUIET_MACHINE = pt.QUIET_MACHINE
FEATURE_ANCHOR = pt.FEATURE_ANCHOR
VOID_RULE = {
    "void": "any guard, crash, cap, stop signal or CUDA allocation failure makes a stage V; its "
    "report is kept with its sha256 and the cause disclosed",
    "repeat": "at most one repeat per stage after a committed, pushed and recorded fix, on the "
    "same seeds in a new output directory; a second V of the same stage ends TASK-077 "
    "INCONCLUSIVE; Vs in different stages do not add up (R17.16)",
    "stage_t": "each of Stage T's eight jobs is voided and repeated on its own; completed jobs "
    "with intact reports, checkpoints and sha256s are kept; a second V of the same job ends "
    "TASK-077 INCONCLUSIVE; the budget rule runs only after both calibration jobs complete",
    "debug": "debug runs only on committed code, only on 66900-66999; nothing in them is read",
}
STAGES = ("K0", "C", "O", "T", "G", "D", "S")
CPU_STAGES = ("K0", "C", "O-readouts", "G", "D", "S")
GPU_STAGES = ("O-featurise", "T")
G_TESTS = {
    "cpu": "the runner runs the full pytest suite itself at HEAD before its first simulation and "
    "records the summary line, exit status and timestamps; any failure is V",
    "gpu": "the runner verifies pre_run_tests.json: revision == HEAD, exit status 0, no failure "
    "or error in the summary, finished after HEAD's commit time",
}

# ----- Stage 0 (filled by the Stage-0 PR; development only, debug seeds and synthetic data) -------
STAGE0 = {  # the record: docs/experiments/apple_lewm_c1m_v2_stage0.md (development only)
    "smoke_revision": "a98d893",
    "smoke_root": "outputs/task077-smoke-a98d893 in the task077-stage0 worktree (git-ignored)",
    "simulations": {
        "revision": "f4b6b48",
        "report_sha256": "85adbdeea88a58e8f2def13caead1c29fe47f128d4f02df8d505a6eee557075f",
        "trials_per_configuration": 10_000,
        "ni_size_at_margin": "G-NI passes at a true W - C of exactly -8/64 in 2.4-3.8 % of trials "
        "with one comparator and 0.9-3.5 % with the better of two chosen after S (C at 30-32/32; "
        "nominal one-sided 2.5 %)",
        "ni_power_max_of_two": "at W = C: 0.69-0.77 (independent outcomes, C 30/32), 0.91-0.93 "
        "(31/32), 1.00 (nested); at -2/64: 0.35-0.86; at -4/64: 0.14-0.43",
        "l_inferior_false_fire_at_delta": "0.9-2.0 % with one comparator, 0.9-3.1 % with the "
        "better of two",
        "l_no_gain_false_fire_at_plus_7": "2.3-3.1 % per twin; 9.0-12.1 % that at least one of "
        "four twins all sitting at exactly +7/64 fires",
    },
    "scale_probe": {
        "revisions": ["f4b6b48", "a98d893"],
        "report_sha256": [
            "09628d9027d6fd6afe43823e6050eb795cbe7eca4b424d72d80ea14885b52701",
            "27e4ada4d680183aa6bbee082386130c597f8adc82850eb5c76ed111f2ec301a",
        ],
        "gather_seconds_per_batch": [0.0036, 0.0037],
        "per_update_median_seconds": [0.223, 0.166],
        "per_update_p95_seconds": [0.277, 0.232],
        "peak_tree_pss_gib": [12.64, 12.74],
        "bit_identical_rerun": True,
        "t_job_worst_case_seconds": 27_704,
        "t_job_cap_over_worst": 1.69,
    },
}


# ----- helpers ------------------------------------------------------------------------------------
def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value == NOT_EVALUATED)


def _require(**inputs) -> None:
    """G-sentinel: a ladder never runs with a missing input."""
    missing = sorted(k for k, v in inputs.items() if is_missing(v))
    if missing:
        raise ContractError(f"G-sentinel: a row ladder was given missing inputs {missing}")


def sentinel_fields(names) -> dict:
    return {str(n): NOT_EVALUATED for n in names}


def check_sentinel(report: dict, evaluated: set[str], fields) -> list[str]:
    """Every field in ``fields`` that a stage has not evaluated holds the sentinel; a row-like
    value in an unevaluated field is refused (the ``early_verdict`` bug)."""
    rows = {
        *S_ROWS,
        *O_ROWS,
        *G_ROWS,
        "CAL-ESCALATE",
        "K0-PASS",
        "CORPUS-SEALED",
        "CORPUS-ESCALATE",
        "CAL-T-ESCALATE",
        "T-DONE",
        "L-DEV-STOP",
        "D-PASS",
    }
    bad = []
    for name in fields:
        value = report.get(name, NOT_EVALUATED)
        if name not in evaluated and value != NOT_EVALUATED:
            bad.append(name)
        if name not in evaluated and isinstance(value, dict) and value.get("row") in rows:
            bad.append(name)
    if bad:
        raise GuardError(f"G-sentinel: fields {sorted(set(bad))} hold values never evaluated")
    return bad


# ----- statistics (salt 8106) --------------------------------------------------------------------
def bootstrap_index(n: int, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    """The resample matrix ``[resamples, n]`` every paired interval of n resets uses: drawn as
    ``resamples`` successive ``integers(0, n, n)`` calls of ``default_rng(8106)``."""
    rng = np.random.default_rng(SALTS["bootstrap"])
    return rng.integers(0, int(n), size=(int(resamples), int(n)))


def paired_interval(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The paired difference in successes (first - second), its reset-clustered percentile 95 %
    interval (salt 8106) and the discordant counts (``pt.paired_interval``'s estimator)."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ContractError("paired outcomes need the same resets")
    d = a.astype(np.int64) - b.astype(np.int64)
    boot = d[bootstrap_index(len(d), resamples)].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    only_first, only_second = pt.discordant(a, b)
    return {
        "difference": int(d.sum()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(d)),
        "only_first": only_first,
        "only_second": only_second,
    }


def median_ci(values, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    v = np.asarray(values, np.float64)
    idx = bootstrap_index(len(v), resamples)
    boot = np.median(v[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "median": float(np.median(v)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
        "p87_5": float(np.percentile(v, 87.5)),
    }


def median_ratio_ci(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """median(first) / median(second), paired by root (salt 8106)."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.shape != b.shape:
        raise ContractError("a paired median ratio needs the same roots")
    idx = bootstrap_index(len(a), resamples)
    boot = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(np.median(a) / np.median(b)), "ci95": [float(lo), float(hi)]}


def median_difference_ci(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.shape != b.shape:
        raise ContractError("a paired median difference needs the same roots")
    idx = bootstrap_index(len(a), resamples)
    boot = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(np.median(a) - np.median(b)), "ci95": [float(lo), float(hi)]}


def ratio_of_sums_ci(numerator, denominator, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """sum(num) / sum(den) over roots (one window per root), root-bootstrapped (salt 8106)."""
    num, den = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    if num.shape != den.shape or num.ndim != 1:
        raise ContractError("a ratio of sums needs one value per root on both sides")
    idx = bootstrap_index(len(num), resamples)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = num[idx].sum(1) / den[idx].sum(1)
    finite = boot[np.isfinite(boot)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "ratio": float(num.sum() / den.sum()),
        "ci95": [float(lo), float(hi)],
        "undefined_resamples": int(len(boot) - len(finite)),
    }


def mcnemar_feasibility(reference, arm, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """R8.12's form (K0, reported): the fraction of resamples of 64 pairs drawn with replacement
    from K's 32 pairs (salt 8106) whose exact one-sided McNemar p of reference > arm is < 0.01."""
    a, c_ = np.asarray(reference, bool), np.asarray(arm, bool)
    b, c = pt.discordant(a, c_)
    rng = np.random.default_rng(SALTS["bootstrap"])
    passes = 0
    for _ in range(int(resamples)):
        i = rng.integers(0, len(a), FEASIBILITY_PAIRS)
        bb, cc = pt.discordant(a[i], c_[i])
        passes += pt.mcnemar_one_sided(bb, cc) < MCNEMAR_P
    return {
        "b": b,
        "c": c,
        "point_p_doubled": pt.mcnemar_one_sided(2 * b, 2 * c),
        "predicted_pass_probability": passes / int(resamples),
        "resamples": int(resamples),
    }


def outer_folds(n: int) -> np.ndarray:
    order = np.random.default_rng(SALTS["outer_folds"]).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


def nested_subsets(fit_index, fold: int) -> dict[float, np.ndarray]:
    fit_index = np.asarray(fit_index, np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["learning_curve"], int(fold)]))
    order = rng.permutation(fit_index)
    m = len(order)
    return {f: np.sort(order[: int(math.floor(f * m + 0.5))]) for f in FRACTIONS}


def wrong_permutation(n: int) -> np.ndarray:
    """G4's wrong commands: a derangement of the gate roots (salt 8110; a fixed point is swapped
    with its neighbour)."""
    perm = np.random.default_rng(SALTS["wrong_commands"]).permutation(int(n))
    for i in range(int(n)):
        if perm[i] == i:
            j = (i + 1) % int(n)
            perm[i], perm[j] = perm[j], perm[i]
    if int(n) > 1 and np.any(perm == np.arange(int(n))):
        raise ContractError("the wrong-command permutation must move every root")
    return perm


def tau_curve_probability(aim_error_cm, tau_counts: dict, resets: int = K_RESETS) -> np.ndarray:
    """K0's tau curve as a success probability of an aim error: linear between the planted
    levels, the last level's count beyond it (reported only, §7 step 7)."""
    levels = np.asarray(sorted(float(k) for k in tau_counts), np.float64)
    counts = np.asarray([tau_counts[k] for k in sorted(tau_counts, key=float)], np.float64)
    return np.interp(np.asarray(aim_error_cm, np.float64), levels, counts / resets)


# ----- the Stage-0 simulations (§8.4; R17.15, R17.17 point 4) ----------------------------------
SIM_TRIALS = 10_000
SIM_SALT_STREAM = 1  # trials: default_rng(SeedSequence([8106, 1, config index]))


def _counts_matrix(n: int, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    """M[r, i] = how often reset i appears in resample r of :func:`bootstrap_index` (so that the
    bootstrap sums of every simulated trial are M @ d: the real estimator, exactly)."""
    idx = bootstrap_index(n, resamples)
    m = np.zeros((int(resamples), int(n)), np.float64)
    np.add.at(m, (np.arange(len(idx))[:, None], idx), 1.0)
    return m


def _intervals(d: np.ndarray, m: np.ndarray, chunk: int = 1_000) -> np.ndarray:
    """``d``: [trials, n] paired differences -> [trials, 2] percentile 95 % bounds."""
    out = np.empty((len(d), 2))
    for lo in range(0, len(d), chunk):
        boot = m @ d[lo : lo + chunk].T.astype(np.float64)
        out[lo : lo + chunk] = np.percentile(boot, [2.5, 97.5], axis=0).T
    return out


def _mcnemar_vec(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.array([pt.mcnemar_one_sided(int(x), int(y)) for x, y in zip(b, c, strict=True)])


def _draw_relative(rng, w, p_w: float, q_w_only: float, q_c_only: float) -> np.ndarray:
    """An arm with P(arm = 1 | W = 1) = (p_w - q_w_only) / p_w and P(arm = 1 | W = 0) =
    q_c_only / (1 - p_w), per reset."""
    p1 = (p_w - q_w_only) / p_w if p_w > 0 else 0.0
    p0 = q_c_only / (1.0 - p_w) if p_w < 1 else 0.0
    if not (0 <= p1 <= 1 and 0 <= p0 <= 1):
        raise ContractError("an impossible joint distribution")
    u = rng.uniform(size=w.shape)
    return np.where(w, u < p1, u < p0)


def simulate_non_inferiority(
    *,
    p_c: float,
    gap: int,
    dependence: str,
    share: float,
    n: int = S_RESETS,
    trials: int = SIM_TRIALS,
    config_index: int = 0,
    m: np.ndarray | None = None,
) -> dict:
    """G-NI and L-INFERIOR under the max-of-two comparator.

    W's true rate is p_c - gap/64; H-rule and H-sysid each have true rate p_c and the same joint
    distribution with W: ``dependence`` "nested" (no W-only reset: the comparator succeeds
    wherever W does) or "independent". Between the two comparators, each reset is shared with
    probability ``share`` (1: one comparator; the reviewer's case is 0.5); otherwise drawn
    afresh from the same conditional on W. C is the better on the trial (a tie goes to H-rule).
    Returns the G-NI pass rate (lower bound > -8/64; at gap 8 the test's size), the L-INFERIOR
    rate (upper bound < -8/64; at gap 8 the clause's false-fire rate) and the single-comparator
    rates beside them."""
    p_w = p_c - gap / n
    if not 0 < p_w <= 1:
        raise ContractError("W's rate leaves (0, 1]")
    if dependence == "nested":
        q_w_only, q_c_only = 0.0, p_c - p_w
    elif dependence == "independent":
        q_w_only, q_c_only = p_w * (1 - p_c), p_c * (1 - p_w)
    else:
        raise ContractError(f"unknown dependence {dependence!r}")
    rng = np.random.default_rng(
        np.random.SeedSequence([SALTS["bootstrap"], SIM_SALT_STREAM, int(config_index)])
    )
    w = rng.uniform(size=(trials, n)) < p_w
    a = _draw_relative(rng, w, p_w, q_w_only, q_c_only)
    fresh = _draw_relative(rng, w, p_w, q_w_only, q_c_only)
    shared = rng.uniform(size=w.shape) < share
    b = np.where(shared, a, fresh)
    c = np.where((a.sum(1) >= b.sum(1))[:, None], a, b)
    m = _counts_matrix(n) if m is None else m
    out = {}
    for name, comp in (("max_of_two", c), ("single", a)):
        bounds = _intervals(w.astype(np.int64) - comp.astype(np.int64), m)
        out[name] = {
            "g_ni_pass_rate": float(np.mean(bounds[:, 0] > -DELTA)),
            "l_inferior_rate": float(np.mean(bounds[:, 1] < -DELTA)),
        }
    return {
        "p_c": p_c,
        "gap_of_64": gap,
        "p_w": p_w,
        "dependence": dependence,
        "share": share,
        "trials": trials,
        **out,
    }


def simulate_twin(
    *,
    p_w: float,
    advantage: int,
    reversed_pairs: float,
    n: int = S_RESETS,
    trials: int = SIM_TRIALS,
    config_index: int = 0,
    twins: int = 1,
    m: np.ndarray | None = None,
) -> dict:
    """L-NO-GAIN for a twin whose true advantage is ``advantage``/64, with ``reversed_pairs``
    expected twin-only resets per 64 (R8.14's form). Returns the McNemar pass rate, the
    L-NO-GAIN rate per twin (failed test and upper bound < +7/64: at +7/64 the false-fire rate)
    and, with ``twins`` > 1 twins drawn conditionally independent given W, the rate at which at
    least one of them fires."""
    q_c_only = reversed_pairs / n
    q_w_only = q_c_only + advantage / n
    rng = np.random.default_rng(
        np.random.SeedSequence([SALTS["bootstrap"], SIM_SALT_STREAM + 1, int(config_index)])
    )
    w = rng.uniform(size=(trials, n)) < p_w
    m = _counts_matrix(n) if m is None else m
    fires, passes = [], []
    for _ in range(int(twins)):
        t = _draw_relative(rng, w, p_w, q_w_only, q_c_only)
        b = (w & ~t).sum(1)
        c = (~w & t).sum(1)
        p = _mcnemar_vec(b, c)
        bounds = _intervals(w.astype(np.int64) - t.astype(np.int64), m)
        ok = p < MCNEMAR_P
        passes.append(ok)
        fires.append(~ok & (bounds[:, 1] < MIN_SEPARATION))
    fires, passes = np.array(fires), np.array(passes)
    return {
        "p_w": p_w,
        "advantage_of_64": advantage,
        "reversed_pairs_per_64": reversed_pairs,
        "trials": trials,
        "twins": int(twins),
        "mcnemar_pass_rate": float(passes[0].mean()),
        "l_no_gain_rate": float(fires[0].mean()),
        "any_twin_fires_rate": float(fires.any(0).mean()),
        "all_pass_rate": float(passes.all(0).mean()),
    }


SIM_GRID = {
    "ni": {
        "p_c": (30 / 32, 31 / 32, 32 / 32),
        "gap_of_64": (0, 2, 4, 8),
        "dependence": ("independent", "nested"),
        "share": (0.0, 0.5, 1.0),
    },
    "twin": {
        "p_w": (56 / 64, 60 / 64),
        "advantage_of_64": (0, 3, 7),
        "reversed_pairs_per_64": (0.0, 1.0, 2.0),
    },
}


def stage0_simulations(*, trials: int = SIM_TRIALS, ni_grid=None, twin_grid=None) -> dict:
    """Every Stage-0 simulation on its declared grid (§8.4): the non-inferiority test's size at
    the margin and its power, with the max-of-two comparator and the single one beside it, and
    the clause's false-fire rates at +7/64 (L-NO-GAIN) and at delta (L-INFERIOR)."""
    ni_grid = SIM_GRID["ni"] if ni_grid is None else ni_grid
    twin_grid = SIM_GRID["twin"] if twin_grid is None else twin_grid
    m = _counts_matrix(S_RESETS)
    ni, k = [], 0
    for p_c in ni_grid["p_c"]:
        for dependence in ni_grid["dependence"]:
            if p_c >= 1.0 and dependence == "independent":
                continue  # a comparator that always succeeds has no W-only reset: nested only
            for share in ni_grid["share"]:
                for gap in ni_grid["gap_of_64"]:
                    ni.append(
                        simulate_non_inferiority(
                            p_c=p_c,
                            gap=gap,
                            dependence=dependence,
                            share=share,
                            trials=trials,
                            config_index=k,
                            m=m,
                        )
                    )
                    k += 1
    twin, k = [], 0
    for p_w in twin_grid["p_w"]:
        for advantage in twin_grid["advantage_of_64"]:
            for reversed_pairs in twin_grid["reversed_pairs_per_64"]:
                twin.append(
                    simulate_twin(
                        p_w=p_w,
                        advantage=advantage,
                        reversed_pairs=reversed_pairs,
                        trials=trials,
                        config_index=k,
                        twins=4,
                        m=m,
                    )
                )
                k += 1
    return {
        "trials": trials,
        "resamples": BOOTSTRAP_RESAMPLES,
        "estimator": "paired_interval's percentile bootstrap with the real salt-8106 resample "
        "matrix (M @ d per trial), exact one-sided McNemar",
        "non_inferiority": ni,
        "twins": twin,
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
            "seed_block": SEED_BLOCK,
            "seed_ranges": SEED_RANGES,
            "model_seeds": MODEL_SEEDS,
            "calibration_seed": CALIBRATION_SEED,
            "debug_seeds": DEBUG_SEEDS,
            "debug_ranges": DEBUG_RANGES,
            "debug_model_seeds": DEBUG_MODEL_SEEDS,
            "debug_calibration_seed": DEBUG_CALIBRATION_SEED,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "reserved_salts": RESERVED_SALTS,
            "condition": {
                "rule": RULE,
                "commit_step": COMMIT_STEP,
                "move_step": MOVE_STEP,
                "rho_cm": RHO_CM,
                "family": FAMILY,
                "redraw_radius_cm": MOVE_REDRAW_RADIUS_CM,
                "a_lo": A_LO,
                "a_hi": A_HI,
                "b_half_m": B_HALF_M,
                "read_step": READ_STEP,
                "horizon": HORIZON,
                "grid": GRID,
                "plate_mean": PLATE_MEAN,
                "tau_commit_record_cm": TAU_COMMIT_RECORD_CM,
                "refine_max": REFINE_MAX,
                "refine_fraction_of_tau": REFINE_FRACTION_OF_TAU,
                "lookahead_max_iter": LOOKAHEAD_MAX_ITER,
            },
            "p3_checkpoint_sha256": P3_CHECKPOINT_SHA256,
            "corpus": {
                "roots": CORPUS_ROOTS,
                "split_sizes": SPLIT_SIZES,
                "debug_split_sizes": DEBUG_SPLIT_SIZES,
                "frame_steps": FRAME_STEPS,
                "command_steps": COMMAND_STEPS,
                "stop_step": STOP_STEP,
                "window_start_steps": WINDOW_START_STEPS,
                "excluded_max_fraction": CORPUS_EXCLUDED_MAX_FRACTION,
            },
            "model": {
                "backend": BACKEND,
                "config": MODEL_CONFIG,
                "train_horizon": TRAIN_HORIZON,
                "batch_size": BATCH_SIZE,
                "device_train": DEVICE_TRAIN,
                "device_eval": DEVICE_EVAL,
            },
            "budget": BUDGET,
            "k0": {
                "resets": K_RESETS,
                "tau_levels_cm": TAU_LEVELS_CM,
                "tau_bar": TAU_BAR,
                "ceiling_min": K0_CEILING_MIN,
                "r_fraction_min": R_FRACTION_MIN,
                "r_latest": R_LATEST,
                "palm_speed_limit_cm": PALM_SPEED_LIMIT_CM,
                "proxies": K0_PROXIES,
                "stops": K0_STOPS,
                "feasibility_bar": FEASIBILITY_BAR,
            },
            "k0_measured": K0_MEASURED,
            "stage_o": {
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "lambda_grid_relative": LAMBDA_GRID_RELATIVE,
                "fractions": FRACTIONS,
                "rows": O_ROWS,
            },
            "g1": {
                "rank_floor": G1_RANK_FLOOR,
                "std_floor": G1_STD_FLOOR,
                "bar_factor": G1_BAR_FACTOR,
                "cal_t_escalate_below": CAL_T_ESCALATE_BELOW,
                "truncation_controls": TRUNCATION_CONTROLS,
            },
            "g_thresholds": G_THRESHOLDS,
            "g_rows": G_ROWS,
            "reported_horizons": REPORTED_HORIZONS,
            "comparative_resamples": COMPARATIVE_RESAMPLES,
            "d": {"resets": D_RESETS, "arms": D_ARMS, "bars": D_BARS},
            "s": {
                "resets": S_RESETS,
                "arms": S_ARMS,
                "twins": TWINS,
                "comparators": COMPARATORS,
                "g_bar": G_BAR,
                "delta": DELTA,
                "mcnemar_p": MCNEMAR_P,
                "min_separation": MIN_SEPARATION,
                "determinism": [DETERMINISM_RESETS, DETERMINISM_TOLERANCE_M],
                "rows": S_ROWS,
            },
            "clause_rows": CLAUSE_ROWS,
            "no_clause_rows": NO_CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "clause_narrowing": CLAUSE_NARROWING,
            "clause_results_wording": CLAUSE_RESULTS_WORDING,
            "never_closed": NEVER_CLOSED,
            "arms": ARMS,
            "caps_seconds": CAPS_SECONDS,
            "cap_factor_min": CAP_FACTOR_MIN,
            "memory": MEMORY,
            "disk_min_gib": DISK_MIN_GIB,
            "disk_min_start_gib": DISK_MIN_START_GIB,
            "gpu": GPU,
            "sim_workers": SIM_WORKERS,
            "wm_workers": WM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "feature_anchor": FEATURE_ANCHOR,
            "void_rule": VOID_RULE,
            "g_tests": G_TESTS,
            "stages": STAGES,
            "sim_trials": SIM_TRIALS,
            "sim_grid": SIM_GRID,
            "stage0": STAGE0,
            "carried_code_reference": CARRIED_CODE_REFERENCE,
            "carried_code_files": CARRIED_CODE_FILES,
            "carried_code_blobs": CARRIED_CODE_BLOBS,
            "task076_manifest": TASK076_MANIFEST,
            "own_files": OWN_FILES,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()


# the condition's sanity, checked at import
check_seed_ranges()
if wc.RESET_CENTERS["plate_xy"] != (0.49, -0.09):
    raise ContractError("p-bar is v2's reset centre")
