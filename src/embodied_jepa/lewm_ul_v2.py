"""TASK-082 (``apple_lewm_unknown_law_v2``): LeWM's committed aim under U-sat, a plate law no
hand-written arm is given, with TASK-077's recipe retrained on a corpus under that law.

Protocol ``docs/experiments/apple_lewm_unknown_law_v2.md`` (STATUS DRAFT; rulings R20 in
``docs/DECISIONS.md``, decision 2026-10-08 (l), Stage 0 (m), each decided by Claude under owner
delegation, 2026-09-30). This module is the protocol's **frozen block** as module constants (a
candidate until the freeze pins its sha256 in ``tests/test_lewm_ul_v2.py`` and the manifest), with
its statistics (salt 8411), every stage's row ladder and the Stage-0 power simulation that fixes
delta by R20.10's rule (salt 8412, sub-key 3).

What changes from TASK-081 (protocol §0): the plate's law after 405 is **U-sat**
(``plate_law_dev``'s ``sat`` at A = 8 cm, D = 12 cm, Theta = 0.25 rad, L = 2; a task change); the
model is **retrained** with TASK-077's recipe on a new 2 000-root corpus under U-sat (three seeds
per arm, the budget U = 95 000 and G1's bars carried); the readouts on predicted latents are
refitted (R-S, R-N, R-L); K0 is re-run on 64 resets; H-rule keeps **C1-M's written law**
(kappa = -0.5), a declared prior that is wrong under U-sat; a reported tier of learned non-LeWM
baselines (H-sysid-krr, P-aim) is added; delta is fixed from {8, 12, 16}/128 by a declared rule.

TASK-077's, TASK-080's and TASK-081's modules, ``commit_precision_dev.py`` and the design note's
``plate_law_dev.py`` / ``plate_law_dev_runtime.py`` are imported, never edited; G-frozen and
G-hash check their pins. **G-sentinel**: every report field a stage has not reached holds
:data:`NOT_EVALUATED`; every ladder refuses a missing input.

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
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import plate_law_dev as pl
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_unknown_law_v2"
TASK = "TASK-082"
STATUS = "DRAFT"
DOCUMENT = "docs/experiments/apple_lewm_unknown_law_v2.md"
MANIFEST = "benchmarks/manifests/apple-lewm-ul-v2.json"
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
RULINGS = "DECISIONS.md decision 2026-10-08 (l), R20.1-R20.16 (Stage 0: (m), R20.17-)"
GuardError = pt.GuardError
NOT_EVALUATED = lm.NOT_EVALUATED

# ----- carried and checked (§2.2, §2.3; G-frozen, G-hash) ----------------------------------------
TASK081_MANIFEST = cpv.MANIFEST
TASK081_FROZEN_SHA256 = "77ccc6364624c7da9dcca1d09827a0f6a25a670a00392d488e7b0216e090705a"
TASK080_MANIFEST = cpv.TASK080_MANIFEST
TASK080_FROZEN_SHA256 = cpv.TASK080_FROZEN_SHA256
TASK077_MANIFEST = lm.MANIFEST
TASK077_FROZEN_SHA256 = cpv.TASK077_FROZEN_SHA256
SOLVER_FILE = cpv.SOLVER_FILE
SOLVER_FILE_SHA256 = cpv.SOLVER_FILE_SHA256
PINNED_DEV_FILES = {  # the design note's development code, frozen inputs from Stage 0 (§2.2)
    "src/embodied_jepa/plate_law_dev.py": (
        "0ad8e530e5e1fe5c77c2410af0a99e26281b0243b16a2af506e7815a964c49b2"
    ),
    "src/embodied_jepa/plate_law_dev_runtime.py": (
        "dad06107d1262bf62fdbb31e7976e34fdfd29ba7f4dc93ad9e5f2399315a361e"
    ),
    SOLVER_FILE: SOLVER_FILE_SHA256,
}
OWN_FILES = (
    "src/embodied_jepa/lewm_ul_v2.py",
    "src/embodied_jepa/lewm_ul_v2_runtime.py",
    "src/embodied_jepa/lewm_ul_v2_offline.py",
    "src/embodied_jepa/lewm_ul_v2_train.py",
    "scripts/run_lewm_ul_v2.py",
)
CARRIED = {
    # R-plate (TASK-077 Stage O): the 405 plate reading that builds every grid (§2.3), by content
    # sha256, from TASK-077's O-PASS readouts report (file sha256)
    "r_plate": pr.REUSED["r_plate"],
    "task077_readouts_report_sha256": pr.REUSED["readouts_report_sha256"],
    "task077_corpus_sha256": pr.OLD_CORPUS_SHA256,
    # TASK-077's Stage T plan (§4.4, R20.5): the budget and G1's calibrated bars, carried
    "task077_plan_report_sha256": (
        "7c7147601c156a793c238e34d00ee74e9855f29d19e4ee0d653882335d67cdfb"
    ),
}
CARRIED_PLAN = {  # copied from TASK-077's plan report (sha256 above); not re-measured (R20.5)
    "updates": 95_000,
    "select_every": 4_750,
    "u_sat": {"W": 46_000, "N": 38_000},
    "wanted": 92_000.0,
}
G1_BARS = {  # TASK-077's calibrated G1 bars (calibration W on C1-M's val roots), carried
    "rank": {"bar": 0.12, "floor": 0.1, "relative": 0.12, "reference": 0.2539868780820061,
             "escalate": False},
    "std": {"bar": 0.38, "floor": 0.25, "relative": 0.38, "reference": 0.7738450290344542,
            "escalate": False},
}  # fmt: skip

# ----- seeds and salts (§8, R20.11) --------------------------------------------------------------
SEED_BLOCK = (72000, 74999)  # R19.24
SEED_RANGES = {
    "K": (72400, 72463),  # K0, 64 resets
    "D": (72500, 72515),  # the development closed loop
    "S": (72600, 72727),  # the gated cohort, 128 resets
    "corpus": (72800, 74799),  # apple-ul-v2: 2 000 roots
}
MODEL_SEEDS = (72360, 72361, 72362)  # W and N (torch; not resets)
RESERVED_RANGES = (
    (72032, 72099),
    (72356, 72359),
    (72363, 72399),
    (72464, 72499),
    (72516, 72599),
    (72728, 72799),
)
DEBUG_SEEDS = (74800, 74899)  # runner mechanics only; nothing in them is read
DEBUG_RANGES = {
    "K": (74800, 74803),
    "D": (74810, 74813),
    "S": (74820, 74823),
    "corpus": (74830, 74869),  # 40 roots: train 30, val 5, gate-P 5
    "labelcheck": (74870, 74875),  # Stage 0: the labelling pass leaves a root unchanged
}
DEBUG_MODEL_SEEDS = (74890, 74891, 74892)  # debug torch seeds (three, as Stage G needs)
DESIGN_NOTE_RANGES = {  # the design note's ranges, not used here (§8)
    "design_check_F": tuple(pl.SEED_RANGES["F"]),
    "design_corpus": tuple(pl.SEED_RANGES["corpus"]),
    "design_debug": tuple(pl.DEBUG_SEEDS),
}
FORBIDDEN_RANGES = (
    dict(cpv.FORBIDDEN_RANGES) | {"task081_block": cpv.SEED_BLOCK} | dict(DESIGN_NOTE_RANGES)
)
SALTS = {
    "move": 8405,  # default_rng(SeedSequence([8405, seed, k])), k the re-draw index
    "corpus_aim": 8406,  # the corpus's uniform (a, b) per root
    "corpus_split": 8407,  # train / val / gate-P, one permutation, fixed before collection
    "tau_direction": 8408,  # K0's planted-error direction per reset
    "sampler": 8409,  # the training window sampler: SeedSequence([8409, model seed])
    "folds": 8410,  # every fold assignment, with sub-keys (FOLD_KEYS)
    "bootstrap": 8411,  # every bootstrap (10 000 resamples, root- or reset-clustered)
    "misc": 8412,  # with sub-keys (MISC_KEYS)
}
FOLD_KEYS = {"outer": 1, "inner": 2, "learned_tier": 3, "learning_curve": 4}
MISC_KEYS = {"l_rand": 1, "wrong_commands": 2, "power": 3}
USED_SALTS = {f"design_note_{k}": v for k, v in pl.SALTS.items()}  # 8401-8404: not reused
EARLIER_SALTS = {
    **cpv.EARLIER_SALTS,
    **{f"task081_{k}": v for k, v in cpv.SALTS.items()},
    **{f"task081_{k}": v for k, v in cpv.USED_SALTS.items()},
    **{f"task081_{k}": v for k, v in cpv.RESERVED_SALTS.items()},
    **USED_SALTS,
}


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in 72000-74999, only where R19.24 left room (72032-72099, 72356-74899), are
    disjoint, avoid every forbidden range (TASK-081's list and block, the design note's three
    ranges); the salts 8405-8412 are fresh."""
    spans = (
        dict(SEED_RANGES)
        | {"debug": DEBUG_SEEDS}
        | {f"model_{s}": (s, s) for s in MODEL_SEEDS}
        | {f"reserved_{i}": r for i, r in enumerate(RESERVED_RANGES)}
    )
    free = ((72032, 72099), (72356, 74899))
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two TASK-082 ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < SEED_BLOCK[0] or high > SEED_BLOCK[1]:
            raise GuardError(f"G-seeds: the range {name} leaves the block {SEED_BLOCK}")
        if not any(f_low <= low and high <= f_high for f_low, f_high in free):
            raise GuardError(f"G-seeds: the range {name} is not in R19.24's free ranges")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"G-seeds: the range {name} overlaps the forbidden {other}")
    for low, high in DEBUG_RANGES.values():
        if low < DEBUG_SEEDS[0] or high > DEBUG_SEEDS[1]:
            raise GuardError("G-seeds: a debug range leaves the debug block")
    for s in DEBUG_MODEL_SEEDS:
        if not DEBUG_SEEDS[0] <= s <= DEBUG_SEEDS[1]:
            raise GuardError("G-seeds: a debug torch seed leaves the debug block")
    if seeds_of("K") and len(seeds_of("K")) != K_RESETS:
        raise GuardError("G-seeds: K has 64 resets")
    if len(seeds_of("D")) != D_RESETS or len(seeds_of("S")) != S_RESETS:
        raise GuardError("G-seeds: D has 16 resets and S 128")
    if len(seeds_of("corpus")) != CORPUS_ROOTS:
        raise GuardError("G-seeds: the corpus has 2 000 roots")
    mine = list(SALTS.values())
    if sorted(mine) != list(range(8405, 8413)) or set(mine) & set(EARLIER_SALTS.values()):
        raise GuardError("G-salts: 8405-8412 are this task's and fresh")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """Plain ints; exactly the role's declared range, once, in order (G-fresh, G-split)."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    want = seeds_of(role, debug=debug)
    if seeds != want:
        where = (DEBUG_RANGES if debug else SEED_RANGES)[role]
        raise GuardError(f"G-fresh: {role} simulates exactly {where}, once, in order")
    return seeds


def check_model_seed(seed: int, *, debug: bool = False) -> int:
    allowed = DEBUG_MODEL_SEEDS if debug else MODEL_SEEDS
    if type(seed) is not int or seed not in allowed:
        raise GuardError(f"G-seeds: model seed {seed!r} is not one of {allowed}")
    return seed


reset_of = lm.reset_of  # v2's own reset: wm_critic_v2.wide_reset_values(seed)

# ----- the condition (§2.1, R20.2) ---------------------------------------------------------------
LAW_NAME = "sat"
LAW = {
    "name": "U-sat",
    "dev_law": LAW_NAME,
    "amplitude_m": 0.08,  # A
    "scale_m": 0.12,  # D
    "swirl_rad": 0.25,  # Theta
    "L": 2,
    "s0": 405,
    "s1": 525,
    "form": "F(d) = -A tanh(|d|/D) R(Theta |d|/D) d/|d|, d = palm(t - L) - palm(405), held "
    "after s1 (plate_law_dev.sat_displacement)",
}
if {k: LAW[k] for k in ("amplitude_m", "scale_m", "swirl_rad")} != pl.LAWS[LAW_NAME] or (
    LAW["L"],
    LAW["s0"],
    LAW["s1"],
) != (pl.LAG, pl.S0, pl.S1):
    raise ContractError("G-law: U-sat is plate_law_dev's sat at its fixed parameters")
H_RULE_LAW = dict(c1.RULE)  # C1-M's written law (kappa = -0.5), a declared prior (R20.4)
if H_RULE_LAW["kappa"] != -0.5 or H_RULE_LAW["L"] != 2:
    raise ContractError("G-law: H-rule keeps C1-M's written kappa = -0.5, L = 2")
COMMIT_STEP = lm.COMMIT_STEP  # 405
MOVE_STEP = lm.MOVE_STEP  # 300
RHO_CM = lm.RHO_CM  # 4 (disc)
FAMILY = lm.FAMILY
A_LO, A_HI, B_HALF_M = lm.A_LO, lm.A_HI, lm.B_HALF_M
READ_STEP = lm.READ_STEP  # r = 465 (kept only if K0's r_K <= 465)
HORIZON = lm.HORIZON  # 60
GRID = lm.GRID  # 147 candidates
TOKEN_GRID = lm.TOKEN_GRID
LATENT_DIM = lm.LATENT_DIM
LOOKAHEAD_TOLERANCE_M = c1.LOOKAHEAD_TOLERANCE_M  # 0.25 cm (H-final(commit), the labels)
LOOKAHEAD_MAX_ITER = c1.MAX_ITERATIONS  # 10
MOVE_REDRAW_RADIUS_CM = lm.MOVE_REDRAW_RADIUS_CM  # 6, R16.3's re-draw rule
MOVE_REDRAW_LIMIT = lm.MOVE_REDRAW_LIMIT


def law_displacement(d) -> np.ndarray:
    """U-sat's displacement of the plate from its 405 position for a palm travel ``d`` (m)."""
    return pl.sat_displacement(d, **pl.LAWS[LAW_NAME])


def law_jacobian(d, eps: float = 1e-6) -> np.ndarray:
    """dF/dd at ``d`` by central differences (the echo slope's reference, §9.4, reported)."""
    d = np.asarray(d, np.float64).reshape(2)
    out = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        out[:, j] = (law_displacement(d + e) - law_displacement(d - e)) / (2.0 * eps)
    return out


def move_uniforms(seed: int) -> dict:
    """R16.3's draw with this task's salt 8405: u, v with k = 0 first; k grows while the plate plus
    the offset at C1-M's largest radius (6 cm) in either family leaves the tabletop."""
    plate = np.asarray(reset_of(seed)["plate_xy"], np.float64)
    largest = MOVE_REDRAW_RADIUS_CM / 100.0
    for k in range(MOVE_REDRAW_LIMIT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["move"], int(seed), k]))
        u, v = float(rng.uniform()), float(rng.uniform())
        if all(c1m._on_table(plate + c1m._offset(u, v, f, largest)) for f in c1m.FAMILIES):
            return {"u": u, "v": v, "k": k}
    raise GuardError(f"G-move: seed {seed} found no on-table draw in {MOVE_REDRAW_LIMIT} re-draws")


def move_offset(seed: int) -> np.ndarray:
    d = move_uniforms(seed)
    return c1m._offset(d["u"], d["v"], FAMILY, RHO_CM / 100.0)


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    """K0's planted error at ``level_cm`` in the reset's uniform direction (salt 8408)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["tau_direction"], int(seed)]))
    angle = float(2.0 * np.pi * rng.uniform())
    r = float(level_cm) / 100.0
    return [r * math.cos(angle), r * math.sin(angle)]


def corpus_aim(seed: int) -> tuple[float, float]:
    """The corpus's (a, b), uniform over the box (salt 8406)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    return float(rng.uniform(A_LO, A_HI)), float(rng.uniform(-B_HALF_M, B_HALF_M))


def l_rand_index(seed: int, feasible) -> int:
    """L-rand: a uniform draw over the feasible grid candidates (salt 8412, sub-key 1)."""
    feasible = np.flatnonzero(np.asarray(feasible, bool))
    if len(feasible) == 0:
        return -1
    key = [SALTS["misc"], MISC_KEYS["l_rand"], int(seed)]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    return int(feasible[int(rng.integers(0, len(feasible)))])


def wrong_permutation(n: int) -> np.ndarray:
    """G4's wrong commands: a derangement of the gate-P roots (salt 8412, sub-key 2)."""
    key = [SALTS["misc"], MISC_KEYS["wrong_commands"]]
    perm = np.random.default_rng(np.random.SeedSequence(key)).permutation(int(n))
    for i in range(int(n)):
        if perm[i] == i:
            j = (i + 1) % int(n)
            perm[i], perm[j] = perm[j], perm[i]
    if int(n) > 1 and np.any(perm == np.arange(int(n))):
        raise ContractError("the wrong-command permutation must move every root")
    return perm


def fold_seed(key: str) -> list[int]:
    """A fold assignment's seed (salt 8410 with its declared sub-key), as ``default_rng`` takes it
    (``RidgeReadout.fit``'s ``seed`` and every other fold draw)."""
    return [SALTS["folds"], FOLD_KEYS[key]]


def outer_folds(n: int) -> np.ndarray:
    """Stage O's and Stage G's outer folds by root (salt 8410, sub-key 1)."""
    order = np.random.default_rng(fold_seed("outer")).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


def learned_tier_folds(n: int) -> np.ndarray:
    """The learned tier's CV folds (salt 8410, sub-key 3)."""
    order = np.random.default_rng(fold_seed("learned_tier")).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


def nested_subsets(fit_index, fold: int) -> dict[float, np.ndarray]:
    """The learning curve's nested fit subsets (salt 8410, sub-key 4)."""
    fit_index = np.asarray(fit_index, np.int64)
    key = [SALTS["folds"], FOLD_KEYS["learning_curve"], int(fold)]
    order = np.random.default_rng(np.random.SeedSequence(key)).permutation(fit_index)
    m = len(order)
    return {f: np.sort(order[: int(math.floor(f * m + 0.5))]) for f in FRACTIONS}


def sampler_seed(model_seed: int) -> np.random.SeedSequence:
    """W and N of a seed draw the same windows in the same order (salt 8409)."""
    return np.random.SeedSequence([SALTS["sampler"], int(model_seed)])


# ----- the corpus apple-ul-v2 (§4.3, R20.6) ------------------------------------------------------
CORPUS_NAME = "apple-ul-v2"
CORPUS_ROOTS = 2000
SPLITS = ("train", "val", "gate_p")
SPLIT_SIZES = {"train": 1500, "val": 250, "gate_p": 250}
DEBUG_SPLIT_SIZES = {"train": 30, "val": 5, "gate_p": 5}
AIM_FROM = {"train": "true", "val": "true", "gate_p": "p_hat"}  # each split's box is built from
FIT_SPLITS = ("train", "val")  # R-S, R-N, R-L (Stage G); Stage O's fits use train only
FRAME_STEPS, COMMAND_STEPS, STOP_STEP = lm.FRAME_STEPS, lm.COMMAND_STEPS, lm.STOP_STEP
N_FRAMES, N_COMMANDS = lm.N_FRAMES, lm.N_COMMANDS
WINDOW_START_STEPS = lm.WINDOW_START_STEPS
CORPUS_EXCLUDED_MAX_FRACTION = 0.02  # overall and in gate-P: above it CORPUS-ESCALATE
UNCONVERGED_GATE_MAX_FRACTION = 0.02  # unconverged labels in gate-P: above it CORPUS-ESCALATE
if sum(SPLIT_SIZES.values()) != CORPUS_ROOTS or tuple(SPLIT_SIZES) != SPLITS:
    raise ContractError("the corpus keeps 2 000 roots: train 1 500, val 250, gate-P 250")


def corpus_split(seeds, *, debug: bool = False) -> dict[str, list[int]]:
    """One permutation of the corpus seeds (salt 8407): the first 1 500 train, then 250 val, then
    250 gate-P (debug: 30, 5, 5). Fixed before collection; split by root."""
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


def decide_corpus(split: dict, excluded, unconverged_gate) -> dict:
    """CORPUS-ESCALATE if more than 2 % of all roots or of gate-P are excluded, or more than 2 % of
    gate-P's labels did not converge; otherwise CORPUS-SEALED (no clause either way)."""
    _require(split=split, excluded=excluded, unconverged_gate=unconverged_gate)
    excluded = {int(s) for s in excluded}
    roots = sum(len(v) for v in split.values())
    gate = [int(s) for s in split["gate_p"]]
    fraction = len(excluded) / roots
    gate_fraction = sum(s in excluded for s in gate) / len(gate)
    label_fraction = int(unconverged_gate) / len(gate)
    stops = {
        "excluded_overall": fraction > CORPUS_EXCLUDED_MAX_FRACTION,
        "excluded_gate_p": gate_fraction > CORPUS_EXCLUDED_MAX_FRACTION,
        "unconverged_labels_gate_p": label_fraction > UNCONVERGED_GATE_MAX_FRACTION,
    }
    row = "CORPUS-ESCALATE" if any(stops.values()) else "CORPUS-SEALED"
    return {
        "row": row,
        "stops": stops,
        "excluded": len(excluded),
        "fraction": fraction,
        "gate_p_fraction": gate_fraction,
        "unconverged_gate_p": int(unconverged_gate),
        "unconverged_gate_p_fraction": label_fraction,
        "clause_fires": False,
    }


# ----- the model and training (§4, R20.5) --------------------------------------------------------
BACKEND = lm.BACKEND  # "leworldmodel": the one-line swap
MODEL_CONFIG = dict(lm.MODEL_CONFIG)  # TASK-077's recipe unchanged
TRAIN_HORIZON = lm.TRAIN_HORIZON  # T = 60
BATCH_SIZE = lm.BATCH_SIZE  # 16
ARMS_TRAINED = lm.ARMS_TRAINED  # W, N
METRIC_FLOOR_STD = lm.METRIC_FLOOR_STD
DEVICE_TRAIN, DEVICE_EVAL = lm.DEVICE_TRAIN, lm.DEVICE_EVAL
SELECTION_TOLERANCE = lm.SELECTION_TOLERANCE  # 0.01: run_tools.select_checkpoint
BUDGET = dict(lm.BUDGET) | {  # TASK-077's budget block (check_budget), with its plan's U carried
    "calibration_runs": (),  # R20.5: no calibration jobs
    "carried_updates": CARRIED_PLAN["updates"],
    "carried_select_every": CARRIED_PLAN["select_every"],
}
if (
    BUDGET["carried_updates"] % BUDGET["selection_points"]
    or BUDGET["carried_updates"] // BUDGET["selection_points"] != BUDGET["carried_select_every"]
    or not BUDGET["min"] <= BUDGET["carried_updates"] <= BUDGET["cap"]
):
    raise ContractError("the carried U = 95 000 divides into 20 selections inside the cap")
DEBUG_TRAIN = {"updates": 300, "select_every": 30}  # Stage 0's GPU smoke (a few hundred updates)

# ----- K0 (§7.2, R20.7) -------------------------------------------------------------------------
K_RESETS = 64
TAU_LEVELS_CM = lm.TAU_LEVELS_CM  # 0, 0.5, 1, 1.5, 2, 3
TAU_BAR = 56  # of 64 (TASK-080's pooled rule at n = 64)
K0_CEILING_MIN = 60  # of 64 (TASK-077's 30/32 fraction)
R_TOLERANCE_CM = lm.R_TOLERANCE_CM  # 0.1
R_FRACTION_MIN = 56  # of 64
R_LATEST = READ_STEP  # r_K later than 465: CAL-ESCALATE
PALM_SPEED_LIMIT_CM = lm.PALM_SPEED_LIMIT_CM  # 0.5 per step at 405
CLIP_OUTSIDE_MAX = 4  # of 64: ceiling aims outside the p-hat box (an allocation)
K0_REPORTED = ("H-rule", "H-now")
K0_STOPS = {
    "level0_below_bar": "level 0 < 56/64 (tau_commit undefined)",
    "tau_zero": "tau_commit = 0 (level 0 reaches 56/64 but 0.5 cm does not)",
    "ceiling_below_60": "N_K(0) < 60/64",
    "r_late": "r_K later than 465 or undefined (the recipe rolls out exactly 60 steps)",
    "palm_fast": "the median palm speed at 405 > 0.5 cm per step",
    "box_binds": "more than 4/64 level-0 ceiling aims outside the box built from p-hat and h",
}
K0_MEASURED: dict | None = None  # written at the freeze from K0's report
TAU_BELOW_1_RULING = (
    "a tau_commit below 1.0 cm makes Stage T's GO a ruling point (proceed, or close TASK-082 as "
    "CAL-ESCALATE without the clause) before any training job (§7.2)"
)


def decide_tau(counts: dict) -> dict:
    """tau_commit: the largest level such that every level up to it reaches 56/64 (None if level 0
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
    return {
        "counts": {str(k): v for k, v in sorted(by_level.items())},
        "tau_commit_cm": tau,
        "bar": TAU_BAR,
        "resets": K_RESETS,
    }


def read_step_k(distances) -> int | None:
    """r_K: the earliest step in [405, 525] at which >= 56/64 level-0 attempts are within 0.1 cm
    of plate(525) (R14.3's form; a refused attempt is never within)."""
    need = math.ceil(R_FRACTION_MIN / K_RESETS * len(distances) - 1e-12)
    for t in range(LAW["s0"], LAW["s1"] + 1):
        within = sum(1 for d in distances if d is not None and d.get(t, math.inf) <= R_TOLERANCE_CM)
        if within >= need:
            return t
    return None


def box_coordinates(g, p_hat, h) -> dict:
    """An aim in the box coordinates (a, b) of the box built from ``p_hat`` and ``h``, and whether
    it lies outside the box (a outside [-0.5, 0.5] or |b| > 3 cm)."""
    a, b = c1.to_box(g, p_hat, h)
    outside = bool(a < A_LO - 1e-12 or a > A_HI + 1e-12 or abs(b) > B_HALF_M + 1e-12)
    return {"a": a, "b_m": b, "outside": outside}


def decide_k0(*, tau: dict, ceiling: int, r_k, palm_speed_median_cm, clip_outside: int) -> dict:
    """K0's stops, all CAL-ESCALATE (escalate, no clause, nothing frozen), else K0-PASS. A
    tau_commit below 1.0 cm is not a stop: it makes Stage T's GO a ruling point."""
    _require(tau=tau, ceiling=ceiling, palm_speed_median_cm=palm_speed_median_cm,
             clip_outside=clip_outside)  # fmt: skip
    t = tau["tau_commit_cm"]
    stops = {
        "level0_below_bar": t is None,
        "tau_zero": t is not None and float(t) == 0.0,
        "ceiling_below_60": int(ceiling) < K0_CEILING_MIN,
        "r_late": r_k is None or int(r_k) > R_LATEST,
        "palm_fast": float(palm_speed_median_cm) > PALM_SPEED_LIMIT_CM,
        "box_binds": int(clip_outside) > CLIP_OUTSIDE_MAX,
    }
    row = "CAL-ESCALATE" if any(stops.values()) else "K0-PASS"
    return {
        "row": row,
        "stops": stops,
        "tau_commit_cm": t,
        "stage_t_go_is_a_ruling_point": bool(t is not None and 0.0 < float(t) < 1.0),
        "clause_fires": False,
    }


def tau_curve_probability(aim_error_cm, counts: dict, resets: int = K_RESETS) -> np.ndarray:
    """K0's tau curve as a success probability of an aim error (TASK-077's piecewise-linear
    mapping; the last level's rate beyond it)."""
    return lm.tau_curve_probability(aim_error_cm, counts, resets)


def predicted_count(aim_error_cm, counts: dict, resets: int = K_RESETS, cohort: int = 128) -> float:
    """The offline predicted count of ``cohort`` (a prediction, never a closed-loop count)."""
    return float(cohort * tau_curve_probability(aim_error_cm, counts, resets).mean())


# ----- Stage O (§9.1) ----------------------------------------------------------------------------
OUTER_FOLDS = lm.OUTER_FOLDS  # 5
INNER_FOLDS = lm.INNER_FOLDS  # 5
LAMBDA_GRID_RELATIVE = lm.LAMBDA_GRID_RELATIVE
FRACTIONS = lm.FRACTIONS
O_ROWS = dict(lm.O_ROWS)
KRR_LENGTHS = pl.KRR_LENGTHS  # the design note's declared grids (standardised units)
KRR_LAMBDAS = pl.KRR_LAMBDAS


def decide_o(*, c_plate: dict, prior_ratio: dict, hidden: dict, tau_commit_cm, falling) -> dict:
    """TASK-077 §8.1 carried: O1 (measured), O3 (definitional), O4 (measured); first match
    O-ARM-KEYED, O-NO-BAR, O-PASS; each escalates without the clause."""
    return lm.decide_o(c_plate=c_plate, prior_ratio=prior_ratio, hidden=hidden,
                       tau_commit_cm=tau_commit_cm, falling=falling)  # fmt: skip


# ----- Stage G (§9.4, R20.8) ---------------------------------------------------------------------
G_THRESHOLDS = {k: v for k, v in lm.G_THRESHOLDS.items() if not k.startswith("G5")}
REPORTED_HORIZONS = lm.REPORTED_HORIZONS  # 16, 30
COMPARATIVE_RESAMPLES = lm.COMPARATIVE_RESAMPLES
COLLAPSE_STD = lm.COLLAPSE_STD
R_ARMS = ("W", "N", "L-shuf", "L-mean", "L-rand")  # A1-A2 (the primary seed)
OFFLINE_REPORTED_ARMS = ("H-rule", "H-sysid", "H-sysid-krr", "P-aim", "H-rule-fit")
A1_BAR = 112  # of 128: G-bar's count, the offline aims mapped through K0's tau curve
A2_MIN_SEPARATION = lm.MIN_SEPARATION  # +7: a necessary floor (R17.15's count)
R2_RATIO_UPPER_EXCLUSIVE = 1.0  # definitional
G_ROWS = {
    "V": "the void rule: one repeat after a recorded fix",
    "H-GATE-FAIL": "any of G1-G4 fails on any seed: escalate, no clause",
    "R-VOID-CEILING": "R0 fails: escalate, no clause",
    "R-COMMAND-KEYED": "R3 fails on any seed: escalate, no clause",
    "R-NO-BAR": "R1 or R2 fails on any seed: escalate, no clause",
    "A-NO-BAR": "A1 fails: escalate, no clause",
    "A-TWIN": "A2 fails for any of N, L-shuf, L-mean, L-rand: escalate, no clause",
    "G-PASS": "Stage D may get its GO",
}


def dynamics_passes(stats: dict) -> dict:
    """G1-G4 of one seed at h = 60 from its statistics (TASK-077 §8.2's definitions; G1's bars
    carried)."""
    t = G_THRESHOLDS
    return {
        "G1": bool(stats["G1"]["parts"]["passes"]),
        "G2": bool(stats["G2"]["ci95"][1] <= t["G2_max_ratio_upper"]),
        "G3": bool(stats["G3"]["ci95"][1] < t["G3_max_ratio_upper_exclusive"]),
        "G4": bool(
            stats["G4_wrong"]["ci95"][0] >= t["G4_min_ratio_lower"]
            and stats["G4_zero"]["ci95"][0] >= t["G4_min_ratio_lower"]
        ),
    }


def g_gates(*, dynamics: dict, r0: dict, readouts: dict, aims: dict, tau_commit_cm, seeds,
            debug: bool = False) -> dict:  # fmt: skip
    """G1-G4 and R1-R3 on every seed in ``seeds``, R0, A1-A2 on the primary seed. ``dynamics[s]``:
    G1-G4 booleans; ``readouts[s]``: ``e_S``, ``e_L`` (median_ci) and ``S_over_N``
    (median_ratio_ci); ``aims``: arm -> predicted count of 128."""
    _require(dynamics=dynamics, r0=r0, readouts=readouts, aims=aims, tau_commit_cm=tau_commit_cm)
    tau = float(tau_commit_cm)
    want = sorted(int(s) for s in seeds)
    if want != sorted(DEBUG_MODEL_SEEDS if debug else MODEL_SEEDS):
        raise ContractError("Stage G gates every model seed of its run")
    if sorted(int(s) for s in dynamics) != want or sorted(int(s) for s in readouts) != want:
        raise ContractError(f"Stage G needs every model seed {want}")
    missing = [a for a in R_ARMS if a not in aims or is_missing(aims[a])]
    if missing:
        raise ContractError(f"A1-A2 need the predicted counts of {missing}")
    per_seed = {}
    for s in want:
        d = {k: dynamics[s][k] for k in ("G1", "G2", "G3", "G4")}
        if not all(isinstance(v, bool) for v in d.values()):
            raise ContractError(f"seed {s}: G1-G4 not evaluated")
        item = readouts[s]
        per_seed[str(s)] = d | {
            "R1": bool(item["e_S"]["ci95"][1] <= tau),
            "R2": bool(item["S_over_N"]["ci95"][1] < R2_RATIO_UPPER_EXCLUSIVE),
            "R3": bool(item["e_L"]["ci95"][0] > tau),
        }
    w = float(aims["W"])
    a2 = {t: bool(w - float(aims[t]) >= A2_MIN_SEPARATION) for t in TWINS}
    gates = {k: all(v[k] for v in per_seed.values()) for k in
             ("G1", "G2", "G3", "G4", "R1", "R2", "R3")}  # fmt: skip
    gates |= {"R0": bool(r0["ci95"][1] <= tau), "A1": bool(w >= A1_BAR), "A2": all(a2.values())}
    return {"gates": gates, "per_seed": per_seed, "a2": a2, "tau_commit_cm": tau}


def decide_g(gates: dict | None, *, void: bool = False) -> dict:
    """§9.4's rows, first match: V, H-GATE-FAIL, R-VOID-CEILING, R-COMMAND-KEYED, R-NO-BAR,
    A-NO-BAR, A-TWIN, G-PASS. None fires the clause."""
    if void:
        return {"row": "V", "meaning": G_ROWS["V"], "clause_fires": False}
    _require(gates=gates)
    g = gates["gates"]
    keys = ("G1", "G2", "G3", "G4", "R0", "R1", "R2", "R3", "A1", "A2")
    missing = [k for k in keys if not isinstance(g.get(k), bool)]
    if missing:
        raise ContractError(f"Stage G's ladder needs {missing}")
    if not (g["G1"] and g["G2"] and g["G3"] and g["G4"]):
        row = "H-GATE-FAIL"
    elif not g["R0"]:
        row = "R-VOID-CEILING"
    elif not g["R3"]:
        row = "R-COMMAND-KEYED"
    elif not (g["R1"] and g["R2"]):
        row = "R-NO-BAR"
    elif not g["A1"]:
        row = "A-NO-BAR"
    elif not g["A2"]:
        row = "A-TWIN"
    else:
        row = "G-PASS"
    return {"row": row, "meaning": G_ROWS[row], "clause_fires": False}


def primary_seed(kept_val: dict) -> int:
    """The W seed with the lowest val criterion at its kept checkpoint (ties to the lower seed)."""
    return int(min(kept_val, key=lambda s: (float(kept_val[s]), int(s))))


# ----- arms (§6, R20.9) --------------------------------------------------------------------------
SPREAD_ARM_PREFIX = "W-"  # the two non-primary seeds' W, each named by its seed: "W-<seed>"
TWINS = ("N", "L-shuf", "L-mean", "L-rand")
COMPARATORS = ("H-rule", "H-sysid")  # the better on S; a tie goes to H-rule
LEARNED_TIER = ("H-sysid-krr", "P-aim")
REPORTED_ONLY_ARMS = ("H-sysid-krr", "P-aim", "H-rule-fit", "H-read", "H-now")
CANDIDATE_ARMS = frozenset({"W", "N", "L-shuf", "L-mean", "L-rand"})
SOLVER_OF = {
    "W": "affine_local",
    "N": "affine_local",
    "L-shuf": "affine_local",
    "L-mean": "affine_local",
    "L-rand": "none",
}
READOUTS = {"W": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}  # never R8
ARMS = {
    "W": {"what": "the primary seed's W from the encoded 405 frame, the p-hat-and-h grid, the "
          "stand-in chunks, read by R-S, committed by affine_local", "privileged": False},
    "N": {"what": "the primary seed's N, zero commands, read by R-N, committed by affine_local",
          "privileged": False},
    "L-shuf": {"what": "W from the logged 405 frame of W's attempt on the next reset that reached "
               "405, this reset's candidate chunks, read by R-S, affine_local",
               "privileged": False},
    "L-mean": {"what": "W from the mean encoded 405 latent of this corpus's train split, read by "
               "R-S, affine_local", "privileged": False},
    "L-rand": {"what": "a uniform feasible grid candidate (salt 8412, sub-key 1), no solver",
               "privileged": False},
    "H-rule": {"what": "RuleCommit with C1-M's written law (kappa = -0.5, L = 2) on p-hat and h, "
               "its fixed point on the kinematic stand-in, clipped: a declared prior that is not "
               "the simulator's law", "privileged": False},
    "H-sysid": {"what": "SysidAim: TASK-077's linear form plate(r) ~ c + A p-hat + B h + C g "
                "refitted on this corpus's train split with R-plate's readings, inverted with "
                "lewm_next_c1.choose_aim, clipped", "privileged": False},
    "H-final": {"what": "H-final(commit): LookaheadAim at 405 in cloned state (branches under "
                "U-sat), unclipped; privileged, not learned", "privileged": True},
    "H-sysid-krr": {"what": "the RBF kernel ridge of plate(r) - p-hat on standardised (p-hat, h, "
                    "g), length and lambda by 5-fold CV (salt 8410, sub-key 3), inverted with "
                    "choose_aim, clipped; reported only (the learned tier)", "privileged": False},
    "P-aim": {"what": "a kernel ridge of the ceiling's logged aim, as g - p-hat, on standardised "
              "(p-hat, h), predicted once at 405 and clipped; no forward model, no solver; "
              "reported only (the learned tier)", "privileged": False},
    "H-rule-fit": {"what": "H-rule's form with kappa fitted on the train split (least squares of "
                   "plate(r) - p-hat on g - h); reported only", "privileged": False},
    "W-<seed>": {"what": "the W of a non-primary model seed with its own R-S, the same "
                 "controller as W; reported only (the seed spread)", "privileged": False},
    "H-read": {"what": "the look-ahead read by R8 from the frame rendered at r in the clone; "
               "privileged, reported only", "privileged": True},
    "H-now": {"what": "the true plate at 405; privileged, reported only", "privileged": True},
}  # fmt: skip
PRIVILEGED_ARMS = frozenset({"H-final", "H-read", "H-now", "H-final-planted", "collect-ul"})
NON_PRIVILEGED_ARMS = frozenset(
    a for a, v in ARMS.items() if not v["privileged"] and a != "W-<seed>"
)


def spread_arms(primary: int, seeds=MODEL_SEEDS) -> tuple[str, ...]:
    """The two non-primary seeds' W arms, named by their seeds."""
    return tuple(f"{SPREAD_ARM_PREFIX}{int(s)}" for s in seeds if int(s) != int(primary))


def spread_seed(arm: str) -> int | None:
    """The model seed of a spread arm ``W-<seed>``; None for any other arm."""
    if arm.startswith(SPREAD_ARM_PREFIX) and arm[len(SPREAD_ARM_PREFIX) :].isdigit():
        return int(arm[len(SPREAD_ARM_PREFIX) :])
    return None


# ----- Stage D (§7.3) ----------------------------------------------------------------------------
D_RESETS = 16
D_ARMS = ("W", "N", "L-shuf", "L-mean", "H-final", "H-rule", "H-sysid")
D_REPORTED = ("H-rule", "H-sysid")
D_BARS = dict(lm.D_BARS)  # W >= 12/16, H-final >= 14/16, W - max(N, L-shuf, L-mean) >= +3/16
D_ROWS = ("L-DEV-STOP", "D-PASS")


def decide_d(counts: dict) -> dict:
    """TASK-080's D stops on W, N, L-shuf, L-mean and H-final(commit); H-rule and H-sysid are
    reported beside them and read by no stop."""
    _require(counts=counts)
    missing = [a for a in D_ARMS if not isinstance(counts.get(a), int)]
    if missing:
        raise ContractError(f"Stage D needs the counts of {missing}")
    out = lm.decide_d({a: counts[a] for a in lm.D_ARMS})
    return out | {"reported_only": {a: counts[a] for a in D_REPORTED}}


# ----- Stage S (§9.3, §9.5) ----------------------------------------------------------------------
S_RESETS = 128
S_GATING_ARMS = ("W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid", "H-final")
S_REPORTED_ARMS = ("H-sysid-krr", "P-aim", "H-rule-fit", "H-read", "H-now")  # + the spread arms
BAR_FRACTION = 0.875  # carried (tau_commit's bar fraction, TASK-076 G1's form)
G_BAR = round(BAR_FRACTION * S_RESETS)  # 112 of 128
DELTA_CANDIDATES = (8, 12, 16)  # of 128 (R20.10)
DELTA_FALLBACK = 16  # TASK-081's allocation
DELTA = 16  # of 128: fixed by R20.10's rule from Stage 0's power simulation (R20.20)
MCNEMAR_P = lm.MCNEMAR_P  # 0.01
MIN_SEPARATION = lm.MIN_SEPARATION  # 7: a count, independent of n
DETERMINISM_RESETS = lm.DETERMINISM_RESETS  # W's re-run on S's first four resets
DETERMINISM_TOLERANCE_M = lm.DETERMINISM_TOLERANCE_M  # 0.1 cm on the R-plate reading
DETERMINISM_TARGET_BOUND_M = lm.DETERMINISM_TARGET_BOUND_M  # 0.6 cm on the commit target
S_ROWS = {
    "V": "one repeat of the stage after a recorded fix",
    "S-VOID-CEILING": "H-final(commit)(S) < 112/128: escalate, no clause, no claim",
    "L-NO-GAIN": "for a twin, the McNemar test fails and W is detectably no better (upper bound of "
    "W - arm < +7 resets): the clause fires",
    "L-INFERIOR": "W is detectably inferior beyond delta (upper bound of W - C < -delta): the "
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
    "H-GATE-FAIL",
    "R-VOID-CEILING",
    "R-COMMAND-KEYED",
    "R-NO-BAR",
    "A-NO-BAR",
    "A-TWIN",
    "L-DEV-STOP",
    "S-VOID-CEILING",
    "L-TWIN-NEAR",
    "L-NEAR",
    "L-BAR",
    "V",
    "INCONCLUSIVE",
)

# ----- statistics (salt 8411; TASK-080's estimators) ---------------------------------------------
BOOTSTRAP_RESAMPLES = pr.BOOTSTRAP_RESAMPLES  # 10 000


def bootstrap_index(n: int, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    return pr.bootstrap_index(n, resamples, salt=SALTS["bootstrap"])


def paired_interval(first, second) -> dict:
    return pr.paired_interval(first, second, salt=SALTS["bootstrap"])


def median_ci(values) -> dict:
    return pr.median_ci(values, salt=SALTS["bootstrap"])


def median_ratio_ci(first, second) -> dict:
    return pr.median_ratio_ci(first, second, salt=SALTS["bootstrap"])


def median_difference_ci(first, second) -> dict:
    return pr.median_difference_ci(first, second, salt=SALTS["bootstrap"])


def ratio_of_sums_ci(numerator, denominator, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """sum(num) / sum(den) over roots, root-bootstrapped (salt 8411; TASK-080's form)."""
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


def comparator(outcomes: dict) -> str:
    """The better of H-rule and H-sysid on S (the larger count; a tie goes to H-rule)."""
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


def both_directions(w, arm, *, delta: int) -> dict:
    """§9.5 items 2-3 (reported): W - arm with its interval and discordant counts, the
    non-inferiority reading at ``delta``, detectable inferiority and the exact one-sided McNemar p
    in both directions."""
    b, c = pt.discordant(w, arm)
    interval = paired_interval(w, arm)
    return interval | {
        "b": b,
        "c": c,
        "p_w_better": pt.mcnemar_one_sided(b, c),
        "p_arm_better": pt.mcnemar_one_sided(c, b),
        "w_better_detectably": bool(pt.mcnemar_one_sided(b, c) < MCNEMAR_P),
        "arm_better_detectably": bool(pt.mcnemar_one_sided(c, b) < MCNEMAR_P),
        "non_inferior_within_delta": bool(interval["ci95"][0] > -int(delta)),
        "w_detectably_inferior_beyond_delta": bool(interval["ci95"][1] < -int(delta)),
    }


def _binom_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1)))


exact_interval = cpv.exact_interval  # Clopper-Pearson, two-sided 95 %


def decide_s(outcomes: dict | None, *, void: bool = False, debug: bool = False,
             delta: int | None = None) -> dict:  # fmt: skip
    """§9.3's ladder, first match (TASK-081's order with this task's delta and salt 8411): V,
    S-VOID-CEILING, L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR, L-NEAR, L-BAR. The secondary claim
    and the learned tier are reported beside the row and read by none."""
    delta = DELTA if delta is None else int(delta)
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
    if any(np.asarray(outcomes[a], bool).shape != w.shape for a in outcomes if outcomes[a]
           is not None):  # fmt: skip
        raise ContractError("every arm is paired on the same resets")
    c_arm = comparator(outcomes)
    ni = paired_interval(w, outcomes[c_arm])
    twins = {a: twin_test(w, outcomes[a]) for a in TWINS}
    g_bar = int(w.sum()) >= G_BAR
    g_ni = bool(ni["ci95"][0] > -delta)
    inferior = bool(ni["ci95"][1] < -delta)
    ceiling = int(np.sum(outcomes["H-final"]))
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
    secondary = {
        "against_C": both_directions(w, outcomes[c_arm], delta=delta) | {"comparator": c_arm},
        **{f"against_{a}": both_directions(w, outcomes[a], delta=delta) for a in COMPARATORS},
        "note": "reported only, never a gate: 'LeWM needed' in R9.8's sense is W detectably "
        "better than the better of H-rule (C1-M's written law) and the linear H-sysid",
    }
    learned = {
        a: both_directions(w, outcomes[a], delta=delta)
        for a in LEARNED_TIER
        if outcomes.get(a) is not None
    }
    return {
        "row": row,
        "meaning": S_ROWS[row],
        "clause_fires": row in CLAUSE_ROWS,
        "delta": delta,
        "counts": {a: int(np.sum(v)) for a, v in outcomes.items() if v is not None},
        "comparator": c_arm,
        "G-bar": g_bar,
        "G-NI": ni | {"passes": g_ni, "detectably_inferior": inferior},
        "twins": twins,
        "ceiling": ceiling,
        "secondary_reported_only": secondary,
        "learned_tier_reported_only": learned,
    }


determinism_check = lm.determinism_check

# ----- the clause (§12, R20.13) ------------------------------------------------------------------
CLAUSE_SCOPE = (
    "LeWM aim selection with a single aim committed at 405 under the declared U-sat plate law "
    "(A = 8 cm, D = 12 cm, Theta = 0.25 rad, L = 2, s0 = 405, s1 = 525) on v2, with the declared "
    "post-pick plate move of radius rho* = 4 cm (disc), from onboard 112 px frozen DINOv2 pooled "
    "tokens, with TASK-066-family predictors trained on a corpus under that law"
)
CLAUSE_NARROWING = lm.CLAUSE_NARROWING  # a 4 x 4 latent on a corpus larger than 1 024 roots
CLAUSE_RESULTS_WORDING = (
    "if the clause fires, the results document states the restored scope plainly: one 8 x 8 "
    "recipe, three seeds, read by a dual ridge fitted on its own stand-in predictions and "
    "committed by affine_local, closes under U-sat every other pooled grid (except that 4 x 4 "
    "case), every readout of the predicted latent and every post-grid solver for the committed "
    "aim from the grid's predictions, not only those that were run"
)
NEVER_CLOSED = (
    "C1-M and every result under it (TASK-081's L-PASS stands)",
    "U-play, U-speed, U-cue, U-hist, a physical plate mechanism and every other law",
    "other law parameters",
    "necessity studies against learned baselines",
    "the full unpooled token grid",
    "history longer than one, action_chunk and predictor_step_embedding",
    "a fine-tuned or another encoder",
    "other views or resolutions",
    "other commit steps, move distributions or radii",
    "TASK-076's results",
    "the LeWM backend",
    "v2",
    "the product goal",
)

# ----- caps, memory, guards (§11) ----------------------------------------------------------------
CAPS_SECONDS = {  # provisional (§11); Stage 0 confirms or raises each at >= 1.5 x its worst case
    "K0": 7_200.0,
    "C": 14_400.0,
    "O_featurisation": 3_600.0,
    "O_readouts": 7_200.0,
    "T_job": 46_800.0,  # TASK-077's, carried
    "G": 14_400.0,
    "D": 7_200.0,
    "S": 21_600.0,
    "per_attempt": 300.0,
    "simulate": 3_600.0,
    "labelcheck": 3_600.0,
}
CAP_FACTOR_MIN = 1.5
MEMORY = dict(lm.MEMORY)  # 12 GiB process-tree PSS; 18 GiB for a Stage T job
DISK_MIN_GIB = lm.DISK_MIN_GIB  # 10 GiB
DISK_MIN_START_GIB = {"C": 15.0, "O": 25.0}
GPU = {"min_free_gib": 8.0, "who": "oej:task082-<stage>", "require_lock": True}
SIM_WORKERS = lm.SIM_WORKERS  # 6
WM_WORKERS = lm.WM_WORKERS  # 4 (workers that load a world model)
WORKER_TORCH_THREADS = lm.WORKER_TORCH_THREADS  # 1
THREAD_ENV = lm.THREAD_ENV
QUIET_MACHINE = lm.QUIET_MACHINE
FEATURE_ANCHOR = lm.FEATURE_ANCHOR
VOID_RULE = {
    "void": lm.VOID_RULE["void"],
    "repeat": "at most one repeat per stage after a committed, pushed and recorded fix, on the "
    "same seeds in a new output directory; a second V of the same stage ends TASK-082 "
    "INCONCLUSIVE; Vs in different stages do not add up",
    "stage_t": "each of Stage T's six jobs is voided and repeated on its own (R17.16); a second V "
    "of the same job ends TASK-082 INCONCLUSIVE",
    "debug": "debug runs only on committed code, only on 74800-74899; nothing in them is read",
}
STAGES = ("K0", "C", "O", "T", "G", "D", "S")
G_LAW = (
    "every attempt and root logs the law it ran under; the runner refuses a record whose law is "
    "not U-sat at the frozen parameters, and H-rule with any law other than C1-M's written "
    "kappa = -0.5"
)
G_SPLIT = (
    "every fit is on train (Stage O) or train + val (Stage G's readouts) only; val only selects; "
    "gate-P is opened only after Stage G's first_outcome_utc and never fitted on; the learned "
    "tier is fitted on train only"
)


# ----- power (§10, R20.12; salt 8412, sub-key 3) ------------------------------------------------
POWER_TRIALS = 20_000
POWER_RESAMPLES = BOOTSTRAP_RESAMPLES
POWER_COUPLINGS = ("overlap", "half", "independent")
POWER_GRID = {
    "p_w": (0.875, 0.906, 0.922, 0.938, 0.953),
    "p_c": (0.80, 0.85, 0.875),
    "rule_gap": 0.10,  # H-rule at C - 0.10, H-sysid at C (the better-of-two comparator)
}
DELTA_RULE = {
    "power_min": 0.80,  # G-NI's power at W = C, every C, the "half" coupling
    "coupling": "half",
    "size_max": 0.05,  # the size at the margin (W = C - delta), every C and every coupling
    "rule": "delta is the smallest of 8, 12 and 16/128 at which G-NI's power at W = C is >= 0.80 "
    "for every C in {0.80, 0.85, 0.875} under the half coupling, and the test's size at the "
    "margin (W = C - delta) is <= 5 % in every coupling; if none qualifies, delta = 16/128",
    "size_reading": "the size at the margin is the larger of the single-comparator (H-sysid at C) "
    "and the better-of-two rates (the stricter reading, declared in R20.17)",
}
TWIN_RATES = {  # TASK-081's S rates, and a stronger N
    "N": 73 / 128,
    "L-shuf": 39 / 128,
    "L-mean": 59 / 128,
    "L-rand": 20 / 128,
    "N_strong": 0.75,
}


def _power_rng(*key) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([SALTS["misc"], MISC_KEYS["power"], *[int(k) for k in key]])
    )


_NI_CACHE: dict = {}


def ni_bounds(kp: int, km: int, *, n: int = S_RESETS, resamples: int = POWER_RESAMPLES):
    """The reset-bootstrap 2.5th and 97.5th percentiles of the summed W - C difference for
    (k+, k-) resets won by W only and by C only (a multinomial draw: the bootstrap depends on the
    resets only through (k+, k-); salt 8412, sub-key 3, stream 0, one draw per cell)."""
    key = (int(kp), int(km), int(n), int(resamples))
    if key not in _NI_CACHE:
        rng = _power_rng(0, *key)
        draws = rng.multinomial(n, [kp / n, km / n, (n - kp - km) / n], size=resamples)
        d = draws[:, 0] - draws[:, 1]
        _NI_CACHE[key] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return _NI_CACHE[key]


def _draw_arm(rng, w, p_w: float, p_arm: float, coupling: str) -> np.ndarray:
    q = pr.discordant_probabilities(p_w, p_arm, {"overlap": "nested"}.get(coupling, coupling))
    return lm._draw_relative(rng, w, p_w, *q)


def ni_power(p_w: float, p_c: float, coupling: str, *, deltas=DELTA_CANDIDATES,
             n: int = S_RESETS, trials: int = POWER_TRIALS, index: int = 0,
             rule_gap: float = POWER_GRID["rule_gap"]) -> dict:  # fmt: skip
    """G-NI's pass rate, L-INFERIOR's rate (upper bound < -delta) and the secondary claim's
    McNemar pass rate (W > C, p < 0.01) by simulation: W at ``p_w``; H-sysid at ``p_c`` and H-rule
    at ``p_c - rule_gap``, each coupled to W by ``coupling`` and conditionally independent of each
    other given W; C the better of the two per trial (a tie to H-rule). The single comparator
    (H-sysid alone) is reported beside it."""
    rng = _power_rng(2, index)
    w = rng.uniform(size=(trials, n)) < p_w
    sysid = _draw_arm(rng, w, p_w, p_c, coupling)
    rule = _draw_arm(rng, w, p_w, max(p_c - rule_gap, 0.0), coupling)
    better = np.where((rule.sum(1) >= sysid.sum(1))[:, None], rule, sysid)
    out = {"p_w": p_w, "p_c": p_c, "p_rule": max(p_c - rule_gap, 0.0), "coupling": coupling,
           "n": n, "trials": trials}  # fmt: skip
    table = pr.mcnemar_pass_table(n)
    for name, comp in (("better_of_two", better), ("single", sysid)):
        kp, km = (w & ~comp).sum(1), (~w & comp).sum(1)
        cells: dict = {}
        for a, b in zip(kp.tolist(), km.tolist(), strict=True):
            cells[(a, b)] = cells.get((a, b), 0) + 1
        bounds = {cell: ni_bounds(*cell, n=n) for cell in cells}
        item = {}
        for delta in deltas:
            g_ni = sum(k for c, k in cells.items() if bounds[c][0] > -delta)
            inferior = sum(k for c, k in cells.items() if bounds[c][1] < -delta)
            item[str(delta)] = {"g_ni_pass_rate": g_ni / trials,
                                "l_inferior_rate": inferior / trials}  # fmt: skip
        item["secondary_pass_rate"] = float(table[kp, km].mean())
        out[name] = item
    return out


def twin_power(p_w: float, p_t: float, coupling: str, *, n: int = S_RESETS) -> float:
    """The exact one-sided McNemar test's pass probability at p < 0.01, exactly (TASK-081's)."""
    return cpv.twin_power(p_w, p_t, coupling, n=n)


def g_bar_power(rate: float, *, n: int = S_RESETS, bar: int = G_BAR) -> float:
    return pr.g_bar_power(rate, n=n, bar=bar)


def delta_rule(power_at_equal: dict, size: dict) -> dict:
    """R20.10: ``power_at_equal[delta][p_c]``: G-NI's pass rate at W = C under the half coupling;
    ``size[delta][p_c][coupling]``: the size at the margin (the stricter of the two readings)."""
    out = {}
    for delta in DELTA_CANDIDATES:
        powers = [float(v) for v in power_at_equal[str(delta)].values()]
        sizes = [float(v) for c in size[str(delta)].values() for v in c.values()]
        out[str(delta)] = {
            "power_min": min(powers),
            "size_max": max(sizes),
            "qualifies": bool(
                min(powers) >= DELTA_RULE["power_min"] and max(sizes) <= DELTA_RULE["size_max"]
            ),  # fmt: skip
        }
    qualifying = [d for d in DELTA_CANDIDATES if out[str(d)]["qualifies"]]
    chosen = qualifying[0] if qualifying else DELTA_FALLBACK
    return {"candidates": out, "delta": int(chosen), "fallback": not qualifying,
            "rule": DELTA_RULE["rule"]}  # fmt: skip


def power_tables(*, trials: int = POWER_TRIALS, grid=None) -> dict:
    """§10 recomputed: G-bar exactly; G-NI, L-INFERIOR and the secondary claim by simulation over
    the W x C grid with the better-of-two comparator and every delta; the size at the margin; the
    twin tests exactly; L-PASS at the planning rates (the product under the overlap coupling); and
    delta by R20.10's rule."""
    grid = POWER_GRID if grid is None else grid
    rows, k = [], 0
    for p_c in grid["p_c"]:
        for p_w in grid["p_w"]:
            for coupling in POWER_COUPLINGS:
                rows.append(ni_power(p_w, p_c, coupling, trials=trials, index=k))
                k += 1
    at_equal = {str(d): {} for d in DELTA_CANDIDATES}
    for p_c in grid["p_c"]:
        row = ni_power(p_c, p_c, DELTA_RULE["coupling"], trials=trials, index=k)
        k += 1
        for d in DELTA_CANDIDATES:
            at_equal[str(d)][str(p_c)] = row["better_of_two"][str(d)]["g_ni_pass_rate"]
    size, size_rows = {str(d): {} for d in DELTA_CANDIDATES}, []
    for d in DELTA_CANDIDATES:
        for p_c in grid["p_c"]:
            size[str(d)][str(p_c)] = {}
            for coupling in POWER_COUPLINGS:
                row = ni_power(p_c - d / S_RESETS, p_c, coupling, deltas=(d,), trials=trials,
                               index=k)  # fmt: skip
                k += 1
                single = row["single"][str(d)]["g_ni_pass_rate"]
                better = row["better_of_two"][str(d)]["g_ni_pass_rate"]
                size[str(d)][str(p_c)][coupling] = max(single, better)
                size_rows.append({
                    "delta": d, "p_c": p_c, "coupling": coupling, "size_single": single,
                    "size_better_of_two": better,
                    "l_inferior_at_margin_single": row["single"][str(d)]["l_inferior_rate"],
                    "l_inferior_at_margin_better_of_two":
                        row["better_of_two"][str(d)]["l_inferior_rate"],
                })  # fmt: skip
    twins = [
        {"p_w": p_w, "twin": t, "p_t": p_t, **{c: twin_power(p_w, p_t, c) for c in POWER_COUPLINGS}}
        for p_w in grid["p_w"]
        for t, p_t in TWIN_RATES.items()
    ]
    chosen = delta_rule(at_equal, size)
    l_pass = []
    for p_w in grid["p_w"]:
        for p_c in grid["p_c"]:
            ni = next(r for r in rows if r["p_w"] == p_w and r["p_c"] == p_c
                      and r["coupling"] == "overlap")  # fmt: skip
            twin_product = 1.0
            for t in ("N", "L-shuf", "L-mean", "L-rand"):
                twin_product *= twin_power(p_w, TWIN_RATES[t], "overlap")
            l_pass.append({
                "p_w": p_w, "p_c": p_c,
                "l_pass_product": g_bar_power(p_w)
                * ni["better_of_two"][str(chosen["delta"])]["g_ni_pass_rate"] * twin_product,
            })  # fmt: skip
    return {
        "g_bar": {str(r): g_bar_power(r) for r in grid["p_w"]},
        "g_ni": rows,
        "power_at_w_equal_c_half": at_equal,
        "size_at_margin": size_rows,
        "twins": twins,
        "l_pass_overlap_product": l_pass,
        "delta_rule": chosen,
        "trials": trials,
        "resamples_per_cell": POWER_RESAMPLES,
        "salt": [SALTS["misc"], MISC_KEYS["power"]],
    }


# ----- helpers ------------------------------------------------------------------------------------
def is_missing(value) -> bool:
    return lm.is_missing(value)


def _require(**inputs) -> None:
    missing = sorted(k for k, v in inputs.items() if is_missing(v))
    if missing:
        raise ContractError(f"G-sentinel: a row ladder was given missing inputs {missing}")


def sentinel_fields(names) -> dict:
    return lm.sentinel_fields(names)


ROW_NAMES = frozenset(
    {*S_ROWS, *G_ROWS, *O_ROWS, *D_ROWS, "CAL-ESCALATE", "K0-PASS", "CORPUS-SEALED",
     "CORPUS-ESCALATE", "T-JOB-DONE", "LABELS-UNCHANGED", "LABELS-CHANGED"}
)  # fmt: skip


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


# ----- Stage 0 (filled by the Stage-0 PR; development only, debug seeds and synthetic data) -------
STAGE0: dict = {  # the record: docs/experiments/apple_lewm_unknown_law_v2_stage0.md (R20.17-R20.22)
    "smoke_root": "outputs/task082-smoke-1 in the task082-s0chain worktree (git-ignored); debug "
    "seeds 74800-74899 only; nothing in them is read",
    "smoke_revisions": {"k0, corpus, labelcheck": "2c861a1", "the rest": "8232523"},
    "smokes": {
        "k0": {"outcome": "CAL-ESCALATE-DEBUG", "seconds": 117.7, "peak_tree_pss_gib": 8.97},
        "corpus": {"outcome": "CORPUS-SEALED-DEBUG", "roots": 40, "excluded": 0,
                   "labels_converged": 40, "seconds": 98.9, "peak_tree_pss_gib": 9.53},
        "labelcheck": {"outcome": "LABELS-UNCHANGED-DEBUG", "roots": 6, "seconds": 61.9,
                       "identical": "frames, commands, plate, palm, hidden render, target, 405 "
                       "state, p-hat, executed steps, termination, outcome: bit-exact on 6 of 6"},
        "featurise": {"outcome": "FEATURISED-DEBUG", "seconds": 10.7,
                      "anchor_max_abs_difference": 8.76e-05},
        "readouts": {"outcome": "O-NO-BAR-DEBUG", "seconds": 0.4},
        "train": {"outcome": "T-JOB-DONE-DEBUG x 6 (W and N of 74890-74892, 300 updates)",
                  "seconds": [53.7, 55.8], "per_update_median_s": 0.167,
                  "gpu_max_allocated_gib": 5.48},
        "gates": {"outcome": "H-GATE-FAIL-DEBUG", "seconds": 131.5, "peak_tree_pss_gib": 9.73},
        "closedD": {"outcome": "L-DEV-STOP-DEBUG", "seconds": 135.5, "peak_tree_pss_gib": 9.36},
        "closedS": {"outcome": "S-VOID-CEILING-DEBUG", "seconds": 1135.5,
                    "peak_tree_pss_gib": 10.76, "determinism_ok": True,
                    "g_tests": "2211 passed, 37 skipped (in-run)"},
    },
    "seconds_per_attempt_max": {"K0_planted": 8.58, "corpus_root_with_label": 8.19,
                                "S_slowest_arm": 15.9},
    "scaled_worst_case_seconds": {"K0": 910, "C": 3100, "O_featurisation": 500,
                                  "O_readouts": 330, "G": 6500, "D": 600, "S": 4600},
    "gscale": {"report_sha256": "199e2704e927f538f9270456ebffc71a635829aa66b78d8586021b3384d08557",
               "fit_seconds": 750.7, "read_seconds": 101.2, "dynamics_seconds": 157.6,
               "offline_aims_scaled_seconds": 5201.6, "worst_case_seconds": 6211.1,
               "peak_tree_gib": 4.05},
    "simulate": {
        "report_sha256": "dd1a22d8171e1daedda614d68ccbfb4a02dd4b2bb8e3595c0a71d32ded125efb",
        "revision": "148a2b8",
        "voided_first_run": "simulate-1 (d9625c7) voided by its 3 600 s cap after a 75-minute "
        "G-quiet wait; its power tables equal simulate-2's exactly (R20.18)",
        "power_at_w_equal_c_half": {"8": [0.387, 0.462, 0.526], "12": [0.709, 0.792, 0.853],
                                    "16": [0.917, 0.960, 0.978]},  # C = 0.80, 0.85, 0.875
        "size_at_margin_max": {"8": 0.0397, "12": 0.0384, "16": 0.0351},
        "l_inferior_at_margin_max": {"8": 0.0249, "12": 0.0222, "16": 0.0237},
    },
    "delta": {"chosen": 16, "rule": "R20.10", "fallback": False},
}  # fmt: skip


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
            "task080_frozen_sha256": TASK080_FROZEN_SHA256,
            "task077_frozen_sha256": TASK077_FROZEN_SHA256,
            "manifests": [TASK081_MANIFEST, TASK080_MANIFEST, TASK077_MANIFEST],
            "pinned_dev_files": PINNED_DEV_FILES,
            "own_files": OWN_FILES,
            "carried": CARRIED,
            "carried_plan": CARRIED_PLAN,
            "g1_bars": G1_BARS,
            "seed_block": SEED_BLOCK,
            "seed_ranges": SEED_RANGES,
            "model_seeds": MODEL_SEEDS,
            "reserved_ranges": RESERVED_RANGES,
            "debug_seeds": DEBUG_SEEDS,
            "debug_ranges": DEBUG_RANGES,
            "debug_model_seeds": DEBUG_MODEL_SEEDS,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "fold_keys": FOLD_KEYS,
            "misc_keys": MISC_KEYS,
            "used_salts": USED_SALTS,
            "condition": {
                "law": LAW,
                "h_rule_law": H_RULE_LAW,
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
                "lookahead_tolerance_m": LOOKAHEAD_TOLERANCE_M,
                "lookahead_max_iter": LOOKAHEAD_MAX_ITER,
            },
            "corpus": {
                "name": CORPUS_NAME,
                "roots": CORPUS_ROOTS,
                "splits": SPLITS,
                "split_sizes": SPLIT_SIZES,
                "debug_split_sizes": DEBUG_SPLIT_SIZES,
                "aim_from": AIM_FROM,
                "fit_splits": FIT_SPLITS,
                "frame_steps": FRAME_STEPS,
                "command_steps": COMMAND_STEPS,
                "stop_step": STOP_STEP,
                "window_start_steps": WINDOW_START_STEPS,
                "excluded_max_fraction": CORPUS_EXCLUDED_MAX_FRACTION,
                "unconverged_gate_max_fraction": UNCONVERGED_GATE_MAX_FRACTION,
            },
            "model": {
                "backend": BACKEND,
                "config": MODEL_CONFIG,
                "train_horizon": TRAIN_HORIZON,
                "batch_size": BATCH_SIZE,
                "device_train": DEVICE_TRAIN,
                "device_eval": DEVICE_EVAL,
                "selection_tolerance": SELECTION_TOLERANCE,
            },
            "budget": BUDGET,
            "debug_train": DEBUG_TRAIN,
            "k0": {
                "resets": K_RESETS,
                "tau_levels_cm": TAU_LEVELS_CM,
                "tau_bar": TAU_BAR,
                "ceiling_min": K0_CEILING_MIN,
                "r_fraction_min": R_FRACTION_MIN,
                "r_latest": R_LATEST,
                "palm_speed_limit_cm": PALM_SPEED_LIMIT_CM,
                "clip_outside_max": CLIP_OUTSIDE_MAX,
                "reported": K0_REPORTED,
                "stops": K0_STOPS,
                "tau_below_1_ruling": TAU_BELOW_1_RULING,
            },
            "k0_measured": K0_MEASURED,
            "stage_o": {
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "lambda_grid_relative": LAMBDA_GRID_RELATIVE,
                "fractions": FRACTIONS,
                "krr_lengths": KRR_LENGTHS,
                "krr_lambdas": KRR_LAMBDAS,
                "rows": O_ROWS,
            },
            "stage_g": {
                "thresholds": G_THRESHOLDS,
                "reported_horizons": REPORTED_HORIZONS,
                "comparative_resamples": COMPARATIVE_RESAMPLES,
                "r_arms": R_ARMS,
                "offline_reported_arms": OFFLINE_REPORTED_ARMS,
                "a1_bar": A1_BAR,
                "a2_min_separation": A2_MIN_SEPARATION,
                "rows": G_ROWS,
            },
            "arms": ARMS,
            "solver": cpv.SOLVER,
            "solver_of": SOLVER_OF,
            "readouts": READOUTS,
            "twins": TWINS,
            "comparators": COMPARATORS,
            "learned_tier": LEARNED_TIER,
            "d": {"resets": D_RESETS, "arms": D_ARMS, "reported": D_REPORTED, "bars": D_BARS},
            "s": {
                "resets": S_RESETS,
                "gating_arms": S_GATING_ARMS,
                "reported_arms": S_REPORTED_ARMS,
                "bar_fraction": BAR_FRACTION,
                "g_bar": G_BAR,
                "delta_candidates": DELTA_CANDIDATES,
                "delta": DELTA,
                "mcnemar_p": MCNEMAR_P,
                "min_separation": MIN_SEPARATION,
                "determinism": [
                    DETERMINISM_RESETS,
                    DETERMINISM_TOLERANCE_M,
                    DETERMINISM_TARGET_BOUND_M,
                ],  # fmt: skip
                "rows": S_ROWS,
            },
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
            "disk_min_start_gib": DISK_MIN_START_GIB,
            "gpu": GPU,
            "sim_workers": SIM_WORKERS,
            "wm_workers": WM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "feature_anchor": FEATURE_ANCHOR,
            "void_rule": VOID_RULE,
            "g_law": G_LAW,
            "g_split": G_SPLIT,
            "stages": STAGES,
            "power": {
                "trials": POWER_TRIALS,
                "resamples_per_cell": POWER_RESAMPLES,
                "couplings": POWER_COUPLINGS,
                "grid": POWER_GRID,
                "delta_rule": DELTA_RULE,
                "twin_rates": TWIN_RATES,
            },
            "stage0": STAGE0,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()


check_seed_ranges()
if G_BAR != 112 or MIN_SEPARATION != 7 or DELTA not in DELTA_CANDIDATES:
    raise ContractError("S: G-bar 112/128, the McNemar minimum 7, delta from {8, 12, 16}/128")
if cpv.frozen_sha256() != TASK081_FROZEN_SHA256:
    raise ContractError("G-frozen: TASK-081's frozen block is not its pin")
