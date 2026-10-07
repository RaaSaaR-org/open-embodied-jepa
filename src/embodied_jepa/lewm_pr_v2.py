"""TASK-080 (``apple_lewm_c1m_v2_pred_readout``): LeWM's committed aim under C1-M, read by a
readout fitted on W's predicted latents, on fresh roots.

Protocol ``docs/experiments/apple_lewm_c1m_v2_pred_readout.md`` (STATUS FROZEN after K0′-PASS;
rulings R18 in ``docs/DECISIONS.md``, decisions 2026-10-07, each decided by Claude under owner
delegation, 2026-09-30). This module is the protocol's **frozen block** as module constants
(FROZEN at the freeze, R18.22; its sha256 is pinned in ``tests/test_lewm_pr_v2.py`` and the
manifest), with its statistics (salt 8206), the K0′, C′, R and S row ladders and the Stage-0
power simulation. TASK-077's modules (``lewm_c1m_v2*``) are imported, never edited: everything
carried is read from them, and G-frozen checks that TASK-077's frozen block and pinned files are
unchanged.

What changes from TASK-077 (protocol §0): W reads its predicted plate with **R-S**, a dual ridge
fitted on W's own predicted latents at r under the kinematic stand-in's chunks (N with **R-N**,
fitted on N's zero-command predictions); the gate is a **fresh** 500-root corpus; nothing is
trained (TASK-077's six Stage T checkpoints are reused).

**G-sentinel**: every report field a stage has not reached holds :data:`NOT_EVALUATED`; every
ladder refuses a missing input.

NumPy only: it imports without torch or MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_c1m_v2_pred_readout"
TASK = "TASK-080"
STATUS = "FROZEN"
DOCUMENT = "docs/experiments/apple_lewm_c1m_v2_pred_readout.md"
MANIFEST = "benchmarks/manifests/apple-lewm-pr-v2.json"
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
RULINGS = "DECISIONS.md decision 2026-10-07, R18.1-R18.14 (Stage 0: R18.15-R18.21; freeze: R18.22)"
GuardError = pt.GuardError
NOT_EVALUATED = lm.NOT_EVALUATED

# ----- TASK-077, carried and checked (§2.1, G-frozen) --------------------------------------------
TASK077_MANIFEST = lm.MANIFEST  # its 13 file pins and its frozen-block pin are checked by G-frozen
TASK077_FROZEN_SHA256 = "f28e5e2cd23d110f40ff043c7308e0bb9b3b71f46a2a4b6940bc536cc5e3548d"
OWN_FILES = (
    "src/embodied_jepa/lewm_pr_v2.py",
    "src/embodied_jepa/lewm_pr_v2_runtime.py",
    "src/embodied_jepa/lewm_pr_v2_offline.py",
    "scripts/run_lewm_pr_v2.py",
)

# ----- the reused artifacts (§2.2, G-hash): content or file sha256, as TASK-077 recorded them ----
OLD_CORPUS_SHA256 = "ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb"
OLD_SPLITS = ("train", "val", "gate")
OLD_SPLIT_SIZES = {"train": 1495, "val": 250, "gate": 250}  # the sealed corpus's kept roots
REUSED = {
    "featurise_report_sha256": "f187c7c8179ffe0d1883739be0f30ae960c6033e4c1d088d800075490ef3e9ee",
    "readouts_report_sha256": "452045d22c21b067c8bfe78cb72f4e842fc61f3c44a9e17b6fae860c244f8b78",
    "plan_report_sha256": None,  # G1's bars (reported G1 only); checked by outcome and corpus
    "moments": "5415eea4b5eda30712176b4f23c0022886a0f9cf1ac60cdf742863393550de6d",
    "r8": "62a5ea8abe348d2e7cbdb3379a22ecddcabd6a3f57c65a6f38f8ae57b8e43b4a",
    "r_plate": "08bde901ca75ab0ee4bc1c74bf4da91117ba57c0491828d81bc9db921cc49eb9",
    "mean_latent": "85da63360c6d3c308d4c760cfc43c2391c0beab1c3fefd3ceb5017c7759d53bd",
    "sysid_file": "671841cd581d86844c0ad14c03839cf965e912f628dd875b89c7adf3b46121fc",
    "models": {
        "W-66800": "891d26640e82ab8fce62c32b7d4a7a9eba99ae1babb8614777c668fe498876b8",
        "W-66801": "27aeadab1476e5097b550f2a17c5de36ef1ea3c75852f4556eab0d54773ec193",
        "W-66802": "ba2614ec271525e0ca9f3a5980afd080e2ae00302e145d96adb0f13cc6ef97b4",
        "N-66800": "51b51035be262e7c0ba2d1bb96beeec2939ee82b5d86217409e90424391ed2c2",
        "N-66801": "15f5dd413aac48aff0eb628b9b3c66d62b97f986111eaec3de4db23887112d37",
        "N-66802": "4f92a8feba04218b21dd289e5749da5abc4c6aef5c5efc08c4f94fd81a01bfef",
    },
    "k0_report_sha256": lm.K0_MEASURED["report_sha256"],
}
MODEL_SEEDS = lm.MODEL_SEEDS  # 66800-66802: reused, nothing is trained (R18.2)
PRIMARY_SEED = 66800  # TASK-077's rule (the lowest kept val criterion, 0.350399), carried
PRIMARY_SEED_FLAG = (
    "W-66800's last_two_triggered flag is true and every W curve was lowest at its last point "
    "(TASK-077 results §3, caveats 1-2): stated beside every seed-66800 comparison"
)

# ----- seeds and salts (§7, R18.9) ---------------------------------------------------------------
SEED_BLOCK = (65000, 65999)
SEED_RANGES = {
    "K": (65000, 65031),  # K0′
    "D": (65100, 65115),  # the development closed loop
    "S": (65200, 65263),  # the gated cohort
    "corpus": (65300, 65799),  # F: the fresh corpus apple-c1m-v2-f (gate-P and contrast-T)
}
DEBUG_SEEDS = (65900, 65999)  # runner mechanics only; nothing in them is read
DEBUG_RANGES = {
    "K": (65900, 65903),
    "D": (65910, 65913),
    "S": (65920, 65923),
    "corpus": (65940, 65943),  # one root per fresh half at least (2 + 2)
}
FORBIDDEN_RANGES = dict(lm.FORBIDDEN_RANGES) | {"task077_block": lm.SEED_BLOCK}
SALTS = {
    "move": 8201,  # default_rng(SeedSequence([8201, seed, k])), k the re-draw index
    "corpus_aim": 8202,  # the fresh corpus's uniform (a, b) per root
    "corpus_split": 8203,  # gate-P / contrast-T, one permutation, fixed before collection
    "outer_folds": 8204,  # the readouts' reported cross-fit over the old roots
    "inner_folds": 8205,  # lambda selection in every ridge
    "bootstrap": 8206,  # every bootstrap and every power or feasibility simulation
    "tau_direction": 8207,  # K0′'s planted-error direction per reset
    "learning_curve": 8208,  # the readouts' nested learning-curve subsample
    "l_rand": 8209,  # L-rand's draw
    "wrong_commands": 8210,  # G4's wrong-command permutation across fresh roots (reported)
}
RESERVED_SALTS = {"reserved_a": 8211, "reserved_b": 8212}  # any use needs its own ruling
EARLIER_SALTS = {**lm.EARLIER_SALTS, **{f"task077_{k}": v for k, v in lm.SALTS.items()}}
EARLIER_SALTS["task077_reserved"] = lm.RESERVED_SALTS["reserved"]


def seeds_of(role: str, *, debug: bool = False) -> tuple[int, ...]:
    low, high = (DEBUG_RANGES if debug else SEED_RANGES)[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """The ranges lie in the block, are disjoint, avoid every forbidden range (TASK-077's whole
    block 66000-68999 included); the salts are 8201-8212 and fresh."""
    spans = dict(SEED_RANGES) | {"debug": DEBUG_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two TASK-080 ranges overlap")
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
    mine = [*SALTS.values(), *RESERVED_SALTS.values()]
    if len(set(mine)) != len(mine) or set(mine) & used:
        raise GuardError("G-seeds: a TASK-080 salt repeats or is an earlier task's salt")
    if sorted(mine) != list(range(8201, 8213)):
        raise GuardError("G-seeds: the salts are 8201-8212")


def check_seeds(role: str, seeds, *, debug: bool = False) -> tuple[int, ...]:
    """G-fresh: plain ints; exactly the role's seeds in order (distinct debug seeds in a debug
    run); never a seed of a forbidden range."""
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


def fresh_root(seed: int) -> bool:
    """A root of the fresh corpus F (or its debug range): G-fresh refuses it in any fit."""
    lo, hi = SEED_RANGES["corpus"]
    d_lo, d_hi = DEBUG_RANGES["corpus"]
    return lo <= int(seed) <= hi or d_lo <= int(seed) <= d_hi


def check_fit_roots(seeds) -> None:
    """G-fresh / G-split: every fit (R-S, R-N, R-L) reads the old corpus's roots only."""
    seeds = [int(s) for s in seeds]
    if any(fresh_root(s) for s in seeds):
        raise GuardError("G-fresh: a readout fit includes a fresh-corpus root")
    if any(SEED_BLOCK[0] <= s <= SEED_BLOCK[1] for s in seeds):
        raise GuardError("G-fresh: a readout fit includes a TASK-080 seed")


# ----- the condition (§2.1): C1-M exactly as TASK-077 records it; only the move salt changes ----
RULE = lm.RULE
COMMIT_STEP = lm.COMMIT_STEP
MOVE_STEP = lm.MOVE_STEP
RHO_CM = lm.RHO_CM
FAMILY = lm.FAMILY
A_LO, A_HI, B_HALF_M = lm.A_LO, lm.A_HI, lm.B_HALF_M
READ_STEP = lm.READ_STEP  # r = 465
HORIZON = lm.HORIZON  # 60
GRID = lm.GRID  # 147 candidates
REFINE_MAX = lm.REFINE_MAX
REFINE_FRACTION_OF_TAU = lm.REFINE_FRACTION_OF_TAU
TOKEN_GRID = lm.TOKEN_GRID
LATENT_DIM = lm.LATENT_DIM
reset_of = lm.reset_of


def move_uniforms(seed: int) -> dict:
    """R16.3's rule (TASK-077's ``move_uniforms``) with salt 8201."""
    plate = np.asarray(reset_of(seed)["plate_xy"], np.float64)
    largest = lm.MOVE_REDRAW_RADIUS_CM / 100.0
    for k in range(lm.MOVE_REDRAW_LIMIT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([SALTS["move"], int(seed), k]))
        u, v = float(rng.uniform()), float(rng.uniform())
        if all(c1m._on_table(plate + c1m._offset(u, v, f, largest)) for f in c1m.FAMILIES):
            return {"u": u, "v": v, "k": k}
    raise GuardError(f"G-move: seed {seed} found no on-table draw in {lm.MOVE_REDRAW_LIMIT}")


def move_offset(seed: int) -> np.ndarray:
    d = move_uniforms(seed)
    return c1m._offset(d["u"], d["v"], FAMILY, RHO_CM / 100.0)


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    """K0′'s planted error at ``level_cm`` in the reset's direction (salt 8207)."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["tau_direction"], int(seed)]))
    angle = float(2.0 * np.pi * rng.uniform())
    r = float(level_cm) / 100.0
    return [r * math.cos(angle), r * math.sin(angle)]


def corpus_aim(seed: int) -> tuple[float, float]:
    """The fresh corpus's (a, b), uniform over the box (salt 8202), for both halves."""
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["corpus_aim"], int(seed)]))
    return float(rng.uniform(A_LO, A_HI)), float(rng.uniform(-B_HALF_M, B_HALF_M))


def l_rand_index(seed: int, feasible) -> int:
    """L-rand: a uniform draw over the feasible grid candidates (salt 8209); -1 if none."""
    feasible = np.flatnonzero(np.asarray(feasible, bool))
    if len(feasible) == 0:
        return -1
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["l_rand"], int(seed)]))
    return int(feasible[int(rng.integers(0, len(feasible)))])


# ----- the fresh corpus F (§5.2, R18.4) ---------------------------------------------------------
CORPUS_ROOTS = 500
HALVES = ("gate_p", "contrast_t")
HALF_SIZES = {"gate_p": 250, "contrast_t": 250}
DEBUG_HALF_SIZES = {"gate_p": 2, "contrast_t": 2}
AIM_FROM = {"gate_p": "p_hat", "contrast_t": "true"}  # what each half's box is built from
CORPUS_EXCLUDED_MAX_FRACTION = lm.CORPUS_EXCLUDED_MAX_FRACTION  # 2 %, overall and per half
if sum(HALF_SIZES.values()) != CORPUS_ROOTS or seeds_of("corpus") != tuple(range(65300, 65800)):
    raise ContractError("F is 500 roots, 65300-65799, in two halves of 250")


def corpus_split(seeds, *, debug: bool = False) -> dict[str, list[int]]:
    """One permutation of F's seeds (salt 8203): the first 250 gate-P, the next 250 contrast-T
    (debug 2 and 2). Fixed before collection; split by root."""
    seeds = [int(s) for s in seeds]
    sizes = DEBUG_HALF_SIZES if debug else HALF_SIZES
    if len(seeds) != sum(sizes.values()) or len(set(seeds)) != len(seeds):
        raise ContractError("the split covers every fresh root once")
    order = np.random.default_rng(SALTS["corpus_split"]).permutation(len(seeds))
    out, lo = {}, 0
    for name, size in sizes.items():
        out[name] = sorted(seeds[i] for i in order[lo : lo + size])
        lo += size
    return out


def decide_corpus(split: dict, excluded) -> dict:
    """CORPUS-ESCALATE when more than 2 % of F, or of either half, is excluded."""
    _require(split=split, excluded=excluded)
    excluded = {int(s) for s in excluded}
    total = sum(len(v) for v in split.values())
    fractions = {"all": len(excluded) / total}
    for half, seeds in split.items():
        fractions[half] = len(excluded & {int(s) for s in seeds}) / len(seeds)
    escalate = any(f > CORPUS_EXCLUDED_MAX_FRACTION for f in fractions.values())
    return {
        "row": "CORPUS-ESCALATE" if escalate else "CORPUS-SEALED",
        "excluded": len(excluded),
        "fractions": fractions,
        "clause_fires": False,
    }


# ----- K0′ (§8 step 2, R18.6) ---------------------------------------------------------------------
K_RESETS = 32
POOLED_RESETS = 64  # K (TASK-077) and K′
TAU_LEVELS_CM = lm.TAU_LEVELS_CM
TAU_BAR_K = lm.TAU_BAR  # 28 of K′'s 32 (level 0)
TAU_BAR_POOLED = 56  # of the pooled 64, every level up to tau_commit
K0_CEILING_MIN = lm.K0_CEILING_MIN  # N_K′(0) >= 30 of 32
R_LATEST = READ_STEP
PALM_SPEED_LIMIT_CM = lm.PALM_SPEED_LIMIT_CM
K_COUNTS = dict(lm.K0_MEASURED["tau"]["counts"])  # TASK-077's K: 32, 29, 29, 18, 11, 2
K0_STOPS = {
    "level0_below_bar": "K′ level 0 < 28/32 (carried; implied by the next stop)",
    "ceiling_below_30": "N_K′(0) < 30/32",
    "r_late": "r_K′ later than 465 or undefined",
    "palm_fast": "the median palm speed at 405 > 0.5 cm per step",
    "pooled_tau_undefined": "the pooled tau_commit is undefined (pooled level 0 < 56/64)",
}
K0_PRIME_MEASURED: dict | None = {
    # Written after K0′ ran once, ending K0′-PASS (protocol §8 step 2 and §8.1; R18.22). The run:
    # ``run_lewm_pr_v2.py k0 --output outputs/task080-k0-1 --log outputs/task080-k0-1.log
    # --evidence <task076-evidence>`` at 931281a (clean tree; DRAFT frozen sha d3ebc26b... before
    # this record was added), CPU only, 6 workers, without the GPU lock, in the worktree
    # /home/huhn/develop/emai/worktrees/task080-k0 on the Linux PC, on the reviewer's reported GO
    # (#154, issuecomment-6047218319). Every value below is copied from that report
    # (outputs/task080-k0-1/report.json, sha256 a0939e3e...4c16).
    "row": "K0′-PASS",
    "report_sha256": "a0939e3eea7b695d1c3029e733970bcd123ea62db76192104c69479ad9314c16",
    "revision": "931281a925e36f6b3e6e9e88f05ba2a0e8892a56",
    "protocol_status_at_run": "DRAFT",
    "frozen_sha256_at_run": "d3ebc26b3ad85006977481e4ab0a660aaf791af0fb8496fea0428bd5788754fe",
    "seeds": (65000, 65031),
    "tau_commit_cm": 1.0,
    "tightened_below_task077": False,
    "counts_k_prime": {"0.0": 32, "0.5": 29, "1.0": 31, "1.5": 14, "2.0": 18, "3.0": 5},
    "failed_seeds": {
        "0.0": [],
        "0.5": [65008, 65014, 65024],
        "1.0": [65001],
        "1.5": [
            65001,
            65002,
            65004,
            65006,
            65008,
            65009,
            65011,
            65013,
            65015,
            65017,
            65020,
            65022,
            65023,
            65024,
            65026,
            65027,
            65028,
            65029,
        ],
        "2.0": [
            65001,
            65002,
            65003,
            65004,
            65008,
            65009,
            65012,
            65019,
            65020,
            65023,
            65026,
            65027,
            65029,
            65031,
        ],
        "3.0": [
            65000,
            65001,
            65003,
            65004,
            65005,
            65006,
            65007,
            65009,
            65011,
            65012,
            65013,
            65014,
            65015,
            65016,
            65017,
            65019,
            65020,
            65021,
            65022,
            65023,
            65024,
            65026,
            65027,
            65028,
            65029,
            65030,
            65031,
        ],
    },
    "landing_miss_cm_median": {
        "0.0": 0.0751,
        "0.5": 0.7464,
        "1.0": 1.489,
        "1.5": 2.2407,
        "2.0": 2.9764,
        "3.0": 4.4764,
    },
    "clip_binding_fraction": 0.0,
    "fallbacks": 0,
    "refused": 0,
    "pooled": {
        "counts_k": {"0.0": 32, "0.5": 29, "1.0": 29, "1.5": 18, "2.0": 11, "3.0": 2},
        "counts_k_prime": {"0.0": 32, "0.5": 29, "1.0": 31, "1.5": 14, "2.0": 18, "3.0": 5},
        "counts_pooled": {"0.0": 64, "0.5": 58, "1.0": 60, "1.5": 32, "2.0": 29, "3.0": 7},
        "tau_commit_cm": 1.0,
        "bar": 56,
        "resets": 64,
    },
    "n_k_prime_0": 32,
    "ceiling_failed_seeds": [],
    "r_k_prime": 459,
    "history": {
        "palm_speed_405_cm_per_step": {"median": 0.0027703868621819895, "max": 0.0141141458940937},
        "m2_405_cm": {"median": 0.002172692072816615, "max": 0.013545317066374676},
        "remaining_405_cm_median": 6.516742886740665,
    },
    "stops": {
        "ceiling_below_30": False,
        "level0_below_bar": False,
        "palm_fast": False,
        "pooled_tau_undefined": False,
        "r_late": False,
    },
    "run": {
        "g_tests": "2066 passed, 37 skipped",
        "g_repro": "8/8",
        "render_disagreements": 0,
        "seconds": 495.8,
        "load_average_at_start": [0.67, 1.76],
        "peak_tree_pss_gib": 8.08,
        "workers": 6,
    },
}


FREEZE_REMAP = {
    # R18.19, R18.22: R18.13's dry run (development, optimistic, sets no bar; primary seed 66800,
    # flagged last_two_triggered) re-mapped through the pooled K and K′ curve by
    # ``run_lewm_pr_v2.py simulate --dryrun <dry run report> --k0-prime outputs/task080-k0-1/...``
    # at 931281a (outputs/task080-simulate-k0p-1/report.json, sha256 5e97ac85...48cb; G-tests
    # 2066 passed, 37 skipped). Predicted counts of 64 are predictions, never closed-loop counts.
    "report_sha256": "5e97ac85daf23763b9c48e8a55a246f44e276eb1f3d3c6596636922ae0ad48cb",
    "dry_run_report_sha256": "e3becdfad070cc96b7080feadd80f0b0496a18bdd88d9728fc636a7db9e1aed5",
    "revision": "931281a925e36f6b3e6e9e88f05ba2a0e8892a56",
    "predicted_counts_of_64": {
        "W": 57.35,
        "N": 28.36,
        "L-shuf": 23.79,
        "L-mean": 28.02,
        "L-rand": 11.62,
        "H-rule": 58.62,
        "H-sysid": 54.49,
    },
    "predicted_counts_of_64_k_curve": {
        "W": 56.88,
        "N": 25.94,
        "L-shuf": 21.24,
        "L-mean": 25.54,
        "L-rand": 8.54,
        "H-rule": 58.5,
        "H-sysid": 54.22,
    },
    "g_bar_power_at_w": 0.783,
    "twin_power_min": 0.99946,  # N, independent coupling; every other cell is higher
}


def pooled_tau(k_counts: dict, k_prime_counts: dict) -> dict:
    """TASK-076 K0's rule on the pooled 64 resets of K and K′: the largest level such that every
    level up to it reaches >= 56/64 (None if pooled level 0 misses)."""
    _require(k_counts=k_counts, k_prime_counts=k_prime_counts)
    a = {float(k): int(v) for k, v in k_counts.items()}
    b = {float(k): int(v) for k, v in k_prime_counts.items()}
    if sorted(a) != sorted(TAU_LEVELS_CM) or sorted(b) != sorted(TAU_LEVELS_CM):
        raise ContractError("the pooled tau needs every declared level of K and K′")
    pooled = {lvl: a[lvl] + b[lvl] for lvl in sorted(a)}
    tau = None
    for level in sorted(pooled):
        if pooled[level] >= TAU_BAR_POOLED:
            tau = level
        else:
            break
    return {
        "counts_pooled": {str(k): v for k, v in pooled.items()},
        "counts_k": {str(k): v for k, v in sorted(a.items())},
        "counts_k_prime": {str(k): v for k, v in sorted(b.items())},
        "tau_commit_cm": tau,
        "bar": TAU_BAR_POOLED,
        "resets": POOLED_RESETS,
    }


def decide_k0_prime(*, k_prime_counts: dict, ceiling: int, r_k, palm_speed_median_cm) -> dict:
    """K0′'s stops, all CAL-ESCALATE (escalate, no clause, nothing frozen), else K0′-PASS."""
    _require(k_prime_counts=k_prime_counts, ceiling=ceiling, palm=palm_speed_median_cm)
    pooled = pooled_tau(K_COUNTS, k_prime_counts)
    level0 = int({float(k): v for k, v in k_prime_counts.items()}[0.0])
    stops = {
        "level0_below_bar": level0 < TAU_BAR_K,
        "ceiling_below_30": int(ceiling) < K0_CEILING_MIN,
        "r_late": r_k is None or int(r_k) > R_LATEST,
        "palm_fast": float(palm_speed_median_cm) > PALM_SPEED_LIMIT_CM,
        "pooled_tau_undefined": pooled["tau_commit_cm"] is None,
    }
    row = "CAL-ESCALATE" if any(stops.values()) else "K0′-PASS"
    tightened = pooled["tau_commit_cm"] is not None and pooled["tau_commit_cm"] < 1.0
    return {
        "row": row,
        "stops": stops,
        "pooled": pooled,
        "tau_commit_cm": pooled["tau_commit_cm"],
        "tightened_below_task077": bool(tightened),
        "clause_fires": False,
    }


def tau_curve_probability(aim_error_cm, counts: dict, resets: int) -> np.ndarray:
    """A tau curve as a success probability of an aim error (TASK-077's mapping): linear between
    the planted levels, the last level's rate beyond it."""
    return lm.tau_curve_probability(aim_error_cm, counts, resets)


def predicted_count(aim_error_cm, counts: dict, resets: int, cohort: int = 64) -> float:
    """The offline predicted count of ``cohort`` (a prediction, never a closed-loop count)."""
    return float(cohort * tau_curve_probability(aim_error_cm, counts, resets).mean())


# ----- Stage R (§9.4, R18.7) ---------------------------------------------------------------------
R_ARMS = ("W", "N", "L-shuf", "L-mean", "L-rand")  # A1-A2 (the primary seed)
OFFLINE_REPORTED_ARMS = ("H-rule", "H-sysid")  # their offline aims are reported only
READOUTS = {"W": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}  # never R8 (§6)
A1_BAR = 56  # of 64: G-bar's count (measured: the pooled tau curve maps the aims)
A2_MIN_SEPARATION = lm.MIN_SEPARATION  # +7/64: a necessary floor (R17.15), not definitional
R2_RATIO_UPPER_EXCLUSIVE = 1.0  # definitional (G5 (b)'s form)
R_ROWS = {
    "V": "the void rule: one repeat after a recorded fix",
    "R-VOID-CEILING": "R0 fails (the fresh encoded frame does not read the plate within tau): "
    "escalate, no clause",
    "R-COMMAND-KEYED": "R3 fails on any seed (the commands alone read the plate within tau): "
    "escalate, no clause",
    "R-NO-BAR": "R1 or R2 fails on any seed: escalate, no clause",
    "A-NO-BAR": "A1 fails: escalate, no clause",
    "A-TWIN": "A2 fails for any of N, L-shuf, L-mean, L-rand: escalate, no clause",
    "R-PASS": "Stage D may get its GO",
}


def r_gates(*, r0: dict, per_seed: dict, aims: dict, tau_commit_cm: float) -> dict:
    """R0-R3 and A1-A2 (§9.4). ``r0``: R8's median_ci on the encoded frame at r. ``per_seed[s]``:
    ``e_S`` (median_ci), ``S_over_N`` (median_ratio_ci), ``e_L`` (median_ci). ``aims``: arm ->
    predicted count of 64 (primary seed)."""
    _require(r0=r0, per_seed=per_seed, aims=aims, tau_commit_cm=tau_commit_cm)
    tau = float(tau_commit_cm)
    if sorted(int(s) for s in per_seed) != sorted(MODEL_SEEDS):
        raise ContractError(f"R1-R3 need every model seed {MODEL_SEEDS}")
    missing = [a for a in R_ARMS if a not in aims or is_missing(aims[a])]
    if missing:
        raise ContractError(f"A1-A2 need the predicted counts of {missing}")
    gates = {"R0": bool(r0["ci95"][1] <= tau)}
    seeds = {}
    for s, item in per_seed.items():
        seeds[str(s)] = {
            "R1": bool(item["e_S"]["ci95"][1] <= tau),
            "R2": bool(item["S_over_N"]["ci95"][1] < R2_RATIO_UPPER_EXCLUSIVE),
            "R3": bool(item["e_L"]["ci95"][0] > tau),
        }
    w = float(aims["W"])
    a2 = {t: bool(w - float(aims[t]) >= A2_MIN_SEPARATION) for t in lm.TWINS}
    gates |= {
        "R1": all(v["R1"] for v in seeds.values()),
        "R2": all(v["R2"] for v in seeds.values()),
        "R3": all(v["R3"] for v in seeds.values()),
        "A1": bool(w >= A1_BAR),
        "A2": all(a2.values()),
    }
    return {"gates": gates, "per_seed": seeds, "a2": a2, "tau_commit_cm": tau}


def decide_r(gates: dict | None, *, void: bool = False) -> dict:
    """§9.4's rows, first match: V, R-VOID-CEILING, R-COMMAND-KEYED, R-NO-BAR, A-NO-BAR, A-TWIN,
    R-PASS. None fires the clause."""
    if void:
        return {"row": "V", "meaning": R_ROWS["V"], "clause_fires": False}
    _require(gates=gates)
    g = gates["gates"]
    missing = [k for k in ("R0", "R1", "R2", "R3", "A1", "A2") if not isinstance(g.get(k), bool)]
    if missing:
        raise ContractError(f"Stage R's ladder needs {missing}")
    if not g["R0"]:
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
        row = "R-PASS"
    return {"row": row, "meaning": R_ROWS[row], "clause_fires": False}


# ----- Stages D and S (§9.3, §9.5): TASK-077's arms, bars and ladders --------------------------
D_RESETS, D_ARMS, D_BARS = lm.D_RESETS, lm.D_ARMS, lm.D_BARS
S_RESETS, S_ARMS, TWINS, COMPARATORS = lm.S_RESETS, lm.S_ARMS, lm.TWINS, lm.COMPARATORS
G_BAR, DELTA, MCNEMAR_P, MIN_SEPARATION = lm.G_BAR, lm.DELTA, lm.MCNEMAR_P, lm.MIN_SEPARATION
S_ROWS = lm.S_ROWS
CLAUSE_ROWS = lm.CLAUSE_ROWS
NO_CLAUSE_ROWS = (
    "CAL-ESCALATE",
    "CORPUS-ESCALATE",
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
CANDIDATE_ARMS = lm.CANDIDATE_ARMS
PRIVILEGED_ARMS = lm.PRIVILEGED_ARMS | {"collect-f"}
ARMS = {
    **{k: dict(v) for k, v in lm.ARMS.items()},
    "W": {
        "what": "the primary seed's W from the encoded 405 frame, read by R-S",
        "privileged": False,
    },
    "N": {"what": "the primary seed's N, zero commands, read by R-N", "privileged": False},
    "L-shuf": {
        "what": "W from the logged 405 frame of W's attempt on the next reset that reached 405, "
        "this reset's candidate commands, read by R-S",
        "privileged": False,
    },
    "L-mean": {"what": "W from L-mean's mean 405 latent, read by R-S", "privileged": False},
    "L-rand": {
        "what": "a uniform feasible grid candidate (salt 8209), no refinement",
        "privileged": False,
    },
}
DETERMINISM_RESETS = lm.DETERMINISM_RESETS
decide_d = lm.decide_d
determinism_check = lm.determinism_check


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
    """TASK-077 §8.4's ladder unchanged (``lewm_c1m_v2.decide_s``) with this task's bootstrap
    salt (8206): V, S-VOID-CEILING, L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR, L-NEAR, L-BAR."""
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
    c_arm = lm.comparator(outcomes)
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


# ----- the clause (§12, R18.10) -------------------------------------------------------------------
CLAUSE_SCOPE = lm.CLAUSE_SCOPE  # TASK-077's R17.10 unchanged
CLAUSE_NARROWING = lm.CLAUSE_NARROWING
CLAUSE_RESULTS_WORDING = (
    "if the clause fires, the results document states plainly that one 8 x 8 run read by a dual "
    "ridge fitted on its own stand-in predictions closes, under this condition, every other "
    "pooled grid (except 4 x 4 on a corpus larger than 1 024 roots) and every readout of the "
    "predicted latent, including trained readout heads and readouts fitted on executed-command "
    "or other predictions, not only the readout that was run"
)
NEVER_CLOSED = tuple(x for x in lm.NEVER_CLOSED if x != "a 4 x 4 latent on a larger corpus")

# ----- caps, memory, guards (§11) -----------------------------------------------------------------
CAPS_SECONDS = {  # each >= 1.5 x its measured or scaled worst case (Stage 0's record)
    "K0": 7_200.0,
    "C": 7_200.0,
    "R_featurisation": 3_600.0,
    "R": 14_400.0,
    "D": 7_200.0,
    "S": 21_600.0,
    "per_attempt": 300.0,
    "dryrun": 14_400.0,
}
CAP_FACTOR_MIN = 1.5
MEMORY = dict(lm.MEMORY)  # 12 GiB process-tree PSS on every CPU stage
DISK_MIN_GIB = lm.DISK_MIN_GIB  # 10 GiB
DISK_MIN_START_GIB = {"C": 15.0, "R_featurisation": 15.0}
GPU = {"min_free_gib": 8.0, "who": "oej:task080-<stage>", "require_lock": True}
SIM_WORKERS = lm.SIM_WORKERS  # 6
WM_WORKERS = lm.WM_WORKERS  # 4 (workers that load a world model)
WORKER_TORCH_THREADS = lm.WORKER_TORCH_THREADS
THREAD_ENV = lm.THREAD_ENV
QUIET_MACHINE = lm.QUIET_MACHINE
FEATURE_ANCHOR = lm.FEATURE_ANCHOR
VOID_RULE = {
    "void": lm.VOID_RULE["void"],
    "repeat": "at most one repeat per stage after a committed, pushed and recorded fix, on the "
    "same seeds in a new output directory; a second V of the same stage ends TASK-080 "
    "INCONCLUSIVE; Vs in different stages do not add up; there is no Stage T",
    "debug": "debug runs only on committed code, only on 65900-65999; nothing in them is read",
}
STAGES = ("K0′", "C′", "R", "D", "S")

# ----- statistics (salt 8206; TASK-077's estimators) ----------------------------------------------
BOOTSTRAP_RESAMPLES = lm.BOOTSTRAP_RESAMPLES
OUTER_FOLDS, INNER_FOLDS = lm.OUTER_FOLDS, lm.INNER_FOLDS
LAMBDA_GRID_RELATIVE = lm.LAMBDA_GRID_RELATIVE
FRACTIONS = lm.FRACTIONS


def bootstrap_index(n: int, resamples: int = BOOTSTRAP_RESAMPLES, *, salt: int | None = None):
    """``lewm_c1m_v2.bootstrap_index``'s draw with salt 8206 (``salt`` only for the equality
    test against TASK-077's estimator)."""
    rng = np.random.default_rng(SALTS["bootstrap"] if salt is None else int(salt))
    return rng.integers(0, int(n), size=(int(resamples), int(n)))


def paired_interval(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES, salt=None) -> dict:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ContractError("paired outcomes need the same resets")
    d = a.astype(np.int64) - b.astype(np.int64)
    boot = d[bootstrap_index(len(d), resamples, salt=salt)].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    only_first, only_second = pt.discordant(a, b)
    return {
        "difference": int(d.sum()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(d)),
        "only_first": only_first,
        "only_second": only_second,
    }


def median_ci(values, *, resamples: int = BOOTSTRAP_RESAMPLES, salt=None) -> dict:
    v = np.asarray(values, np.float64)
    boot = np.median(v[bootstrap_index(len(v), resamples, salt=salt)], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "median": float(np.median(v)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
        "p87_5": float(np.percentile(v, 87.5)),
    }


def median_ratio_ci(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES, salt=None) -> dict:
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.shape != b.shape:
        raise ContractError("a paired median ratio needs the same roots")
    idx = bootstrap_index(len(a), resamples, salt=salt)
    boot = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(np.median(a) / np.median(b)), "ci95": [float(lo), float(hi)]}


def median_difference_ci(first, second, *, resamples=BOOTSTRAP_RESAMPLES, salt=None) -> dict:
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.shape != b.shape:
        raise ContractError("a paired median difference needs the same roots")
    idx = bootstrap_index(len(a), resamples, salt=salt)
    boot = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(np.median(a) - np.median(b)), "ci95": [float(lo), float(hi)]}


def unpaired_median_difference_ci(first, second, *, resamples=BOOTSTRAP_RESAMPLES) -> dict:
    """The confound's size (§5.3 point 2): median(first) - median(second) over two independent
    root sets, each resampled on its own (salt 8206; one stream, first set's draws first)."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    rng = np.random.default_rng(SALTS["bootstrap"])
    ia = rng.integers(0, len(a), size=(int(resamples), len(a)))
    ib = rng.integers(0, len(b), size=(int(resamples), len(b)))
    boot = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(np.median(a) - np.median(b)), "ci95": [float(lo), float(hi)]}


def ratio_of_sums_ci(numerator, denominator, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
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


def outer_folds(n: int) -> np.ndarray:
    order = np.random.default_rng(SALTS["outer_folds"]).permutation(int(n))
    fold = np.empty(int(n), np.int64)
    fold[order] = np.arange(int(n)) % OUTER_FOLDS
    return fold


def nested_subsets(n: int) -> dict[float, np.ndarray]:
    """The learning curve's nested fit subsets of ``n`` fit roots (salt 8208)."""
    order = np.random.default_rng(SALTS["learning_curve"]).permutation(int(n))
    return {f: np.sort(order[: int(math.floor(f * n + 0.5))]) for f in FRACTIONS}


def wrong_permutation(n: int) -> np.ndarray:
    """G4's wrong commands on fresh roots (reported): a derangement (salt 8210)."""
    perm = np.random.default_rng(SALTS["wrong_commands"]).permutation(int(n))
    for i in range(int(n)):
        if perm[i] == i:
            j = (i + 1) % int(n)
            perm[i], perm[j] = perm[j], perm[i]
    if int(n) > 1 and np.any(perm == np.arange(int(n))):
        raise ContractError("the wrong-command permutation must move every root")
    return perm


# ----- power (§10, R18.11) ------------------------------------------------------------------------
POWER_TRIALS = 200_000
POWER_COUPLINGS = ("nested", "half", "independent")
POWER_TABLE_DRAFT = {  # §10's table as the draft states it (the power stage recomputes it)
    (0.875, 0.70): (0.95, 0.61, 0.46),
    (0.875, 0.75): (0.70, 0.32, 0.23),
    (0.906, 0.60): (1.00, 0.99, 0.95),
    (0.906, 0.70): (0.99, 0.80, 0.67),
    (0.906, 0.75): (0.89, 0.54, 0.42),
    (0.906, 0.80): (0.53, 0.25, 0.18),
    (0.9375, 0.70): (1.00, 0.93, 0.86),
    (0.9375, 0.75): (0.97, 0.77, 0.66),
    (0.9375, 0.80): (0.79, 0.49, 0.38),
}


def discordant_probabilities(p_w: float, p_t: float, coupling: str) -> tuple[float, float]:
    """(P(W only), P(twin only)) per reset: "nested" (every twin success is a W success),
    "independent" (independent given the rates) or "half" (halfway between)."""
    nested = (max(p_w - p_t, 0.0), max(p_t - p_w, 0.0))
    independent = (p_w * (1.0 - p_t), p_t * (1.0 - p_w))
    if coupling == "nested":
        return nested
    if coupling == "independent":
        return independent
    if coupling == "half":
        return tuple(0.5 * (a + b) for a, b in zip(nested, independent, strict=True))
    raise ContractError(f"unknown coupling {coupling!r}")


def mcnemar_pass_table(n: int) -> np.ndarray:
    """passes[b, c]: the exact one-sided McNemar p < 0.01 for b W-only and c twin-only pairs."""
    out = np.zeros((n + 1, n + 1), bool)
    for b in range(n + 1):
        for c in range(n + 1 - b):
            out[b, c] = pt.mcnemar_one_sided(b, c) < MCNEMAR_P
    return out


def twin_power(p_w, p_t, coupling, *, n=S_RESETS, trials=POWER_TRIALS, index=0, table=None):
    """The exact one-sided McNemar test's pass rate on ``n`` paired resets (salt 8206)."""
    q_w, q_t = discordant_probabilities(float(p_w), float(p_t), coupling)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["bootstrap"], 3, int(index)]))
    draws = rng.multinomial(int(n), [q_w, q_t, max(1.0 - q_w - q_t, 0.0)], size=int(trials))
    table = mcnemar_pass_table(int(n)) if table is None else table
    return float(table[draws[:, 0], draws[:, 1]].mean())


def g_bar_power(rate: float, *, n: int = S_RESETS, bar: int = G_BAR) -> float:
    """P(X >= bar) for X ~ Binomial(n, rate), exact."""
    return float(sum(math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(bar, n + 1)))


def power_table(*, trials: int = POWER_TRIALS, cells=None, n: int = S_RESETS) -> list[dict]:
    """§10's twin table recomputed (or ``cells`` [(p_w, p_t)] at ``n`` resets)."""
    cells = list(POWER_TABLE_DRAFT) if cells is None else list(cells)
    table = mcnemar_pass_table(int(n))
    out, k = [], 0
    for p_w, p_t in cells:
        row = {"p_w": p_w, "p_t": p_t, "n": int(n)}
        for coupling in POWER_COUPLINGS:
            row[coupling] = twin_power(p_w, p_t, coupling, n=n, trials=trials, index=k, table=table)
            k += 1
        out.append(row)
    return out


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
    {
        *R_ROWS,
        *S_ROWS,
        "CAL-ESCALATE",
        "K0′-PASS",
        "CORPUS-SEALED",
        "CORPUS-ESCALATE",
        "L-DEV-STOP",
        "D-PASS",
    }
)


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
STAGE0: dict = {  # the record: docs/experiments/apple_lewm_c1m_v2_pred_readout_stage0.md
    "smoke_revision": "e4babc1",
    "smoke_root": "outputs/task080-smoke-* in the task080-stage0 worktree (git-ignored); debug "
    "seeds 65900-65999 only; nothing in them is read",
    "dry_run": {  # R18.13: development only, sets no bar (R18.15)
        "revision": "e4babc1c885d354b7a07069402503d6308693548",
        "report": "outputs/task080-dryrun-1/report.json (task080-stage0 worktree)",
        "report_sha256": "e3becdfad070cc96b7080feadd80f0b0496a18bdd88d9728fc636a7db9e1aed5",
        "fit_roots": 1745,
        "eval_roots": "TASK-077's 250 val roots",
        "tau_curve": "TASK-077's K (32 resets), tau_commit 1.0 cm",
        "predicted_counts_of_64": {
            "W": 56.88,
            "N": 25.94,
            "L-shuf": 21.24,
            "L-mean": 25.54,
            "L-rand": 8.54,
            "H-rule": 58.50,
            "H-sysid": 54.22,
        },
        "w_aim_error_cm": {"median": 0.612, "ci95": [0.559, 0.660], "p87_5": 1.044},
        "e_S_val_cm": {"66800": 0.540, "66801": 0.529, "66802": 0.535},
        "e_S_val_upper_cm": {"66800": 0.581, "66801": 0.571, "66802": 0.597},
        "e_N_val_cm": {"66800": 1.583, "66801": 1.421, "66802": 1.581},
        "e_L_val_lower_cm": {"66800": 2.092, "66801": 2.104, "66802": 2.093},
        "encoded_ceiling_val_cm": {"median": 0.382, "ci95": [0.343, 0.424]},
        "seconds": 3312,
        "peak_tree_pss_gib": 9.96,
    },
    "simulate_report_sha256": "f28b708225358f5304aa438e6884cad3567bc4402d5b3534f187d536046698cc",
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
            "task077_frozen_sha256": TASK077_FROZEN_SHA256,
            "task077_manifest": TASK077_MANIFEST,
            "own_files": OWN_FILES,
            "reused": REUSED,
            "old_corpus_sha256": OLD_CORPUS_SHA256,
            "old_split_sizes": OLD_SPLIT_SIZES,
            "model_seeds": MODEL_SEEDS,
            "primary_seed": PRIMARY_SEED,
            "primary_seed_flag": PRIMARY_SEED_FLAG,
            "seed_block": SEED_BLOCK,
            "seed_ranges": SEED_RANGES,
            "debug_seeds": DEBUG_SEEDS,
            "debug_ranges": DEBUG_RANGES,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "reserved_salts": RESERVED_SALTS,
            "condition": {
                "rule": RULE,
                "commit_step": COMMIT_STEP,
                "move_step": MOVE_STEP,
                "rho_cm": RHO_CM,
                "family": FAMILY,
                "a_lo": A_LO,
                "a_hi": A_HI,
                "b_half_m": B_HALF_M,
                "read_step": READ_STEP,
                "horizon": HORIZON,
                "grid": GRID,
                "refine_max": REFINE_MAX,
                "refine_fraction_of_tau": REFINE_FRACTION_OF_TAU,
                "token_grid": TOKEN_GRID,
            },
            "corpus": {
                "roots": CORPUS_ROOTS,
                "halves": HALVES,
                "half_sizes": HALF_SIZES,
                "debug_half_sizes": DEBUG_HALF_SIZES,
                "aim_from": AIM_FROM,
                "excluded_max_fraction": CORPUS_EXCLUDED_MAX_FRACTION,
            },
            "k0_prime": {
                "resets": K_RESETS,
                "pooled_resets": POOLED_RESETS,
                "tau_levels_cm": TAU_LEVELS_CM,
                "tau_bar_k": TAU_BAR_K,
                "tau_bar_pooled": TAU_BAR_POOLED,
                "ceiling_min": K0_CEILING_MIN,
                "r_latest": R_LATEST,
                "palm_speed_limit_cm": PALM_SPEED_LIMIT_CM,
                "k_counts": K_COUNTS,
                "stops": K0_STOPS,
            },
            "k0_prime_measured": K0_PRIME_MEASURED,
            "freeze_remap": FREEZE_REMAP,
            "stage_r": {
                "arms": R_ARMS,
                "reported_arms": OFFLINE_REPORTED_ARMS,
                "readouts": READOUTS,
                "a1_bar": A1_BAR,
                "a2_min_separation": A2_MIN_SEPARATION,
                "r2_ratio_upper_exclusive": R2_RATIO_UPPER_EXCLUSIVE,
                "rows": R_ROWS,
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "lambda_grid_relative": LAMBDA_GRID_RELATIVE,
                "fractions": FRACTIONS,
            },
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
            "disk_min_start_gib": DISK_MIN_START_GIB,
            "gpu": GPU,
            "sim_workers": SIM_WORKERS,
            "wm_workers": WM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "feature_anchor": FEATURE_ANCHOR,
            "void_rule": VOID_RULE,
            "stages": STAGES,
            "power": {"trials": POWER_TRIALS, "couplings": POWER_COUPLINGS},
            "stage0": STAGE0,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()


check_seed_ranges()
if len(GRID) != 147 or HORIZON != 60 or c1.RULE["kappa"] != -0.5:
    raise ContractError("the condition is C1-M's, as TASK-077 records it")
