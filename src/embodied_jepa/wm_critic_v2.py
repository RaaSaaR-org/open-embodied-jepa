"""TASK-073 (``apple_wm_critic_v2``): a LeWM token critic that picks the carried policy's aim on
``apple-to-plate-v2`` under a mid-episode plate shift.

Protocol ``docs/experiments/apple_wm_critic_v2.md``. This module fixes in code, before any
cohort seed is simulated, what the protocol preregisters: the seeds and their guard, the stored
cohort values, the plate-shift condition and its direction rule, the controller's decision grid,
the frozen offset target ``o*``, the K0 calibration rule, the corpus plan, the training budget
rule, the offline gates O1-O5, the development stop rule (D3), the gated rows on cohorts S and
U, the void rule, the abandonment clause and the owner's pre-authorised fallback. NumPy only: it
imports without torch or MuJoCo.

What is carried unchanged: the v2 scene, e9 and its schedule, the counted success (T71-R1/R2),
the attempt length and settle (``first_policy_v2``), and the carried policy P-3 of TASK-072
run-1 with its readout evidence (``first_policy_v2_m2.EVIDENCE``).

**Learned Apple->Plate on the frozen benchmark is still 0 successes.** The plate shift is a
simulation-only diagnostic condition, not a benchmark (owner D1). An H-LeWM success would be
"a learned policy trained on privileged expert labels plus a LeWM critic" (owner D2); it does
not change the v1 headline, and P-3 stays the controller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from math import comb

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_m2 as fm
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_wm_critic_v2"
TASK = "TASK-073"
GuardError = fp2.GuardError

# ----- carried unchanged --------------------------------------------------------------------------
SCENE_VERSION = fp2.SCENE_VERSION
COUNTED_SUCCESS = fp2.COUNTED_SUCCESS
counted_success = fp2.counted_success
EXPERT = fp2.EXPERT  # e9
EXPERT_PHASES = fp2.EXPERT_PHASES
EXPERT_BUDGETS = fp2.EXPERT_BUDGETS  # (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
PICK_PHASES = 4  # orient, descend, close, lift: built from the apple only (a test pins it)
TRANSFER_START = int(sum(EXPERT_BUDGETS[:4]))  # 405
STEADY_END = int(sum(EXPERT_BUDGETS[:7]))  # 585
MAX_POLICY_STEPS = fp2.MAX_POLICY_STEPS  # 740
SETTLE_STEPS = fp2.SETTLE_STEPS  # 60
CARRIED = fm.CARRIED  # P-3
EVIDENCE = fm.EVIDENCE  # TASK-072 run-1: report, P-3 checkpoint, corpus (by sha256)
P3_CHECKPOINT_SHA256 = EVIDENCE["checkpoints"][CARRIED]

# ----- seeds --------------------------------------------------------------------------------------
# The repository search of 2026-09-29 (protocol §4.1) found no seed use in 53000-53999 in src,
# scripts, docs, benchmarks, tests, configs or .mc, and no use of 7300-7313 as a seed constant.
SEED_RANGES = {
    "K0": (53000, 53031),  # condition calibration, development
    "R": (53040, 53071),  # offline ranking and regret (O3, O4); never fitted on
    "D3": (53100, 53115),  # development closed loop, stop rule only
    "corpus": (53400, 53699),  # apple-shift-v2, 240 / 30 / 30 by reset
    "S": (53800, 53863),  # gated, 64 shift resets
    "U": (53900, 53931),  # gated no-harm, 32 undisturbed resets
}
SMOKE_SEEDS = (53950, 53999)  # smokes only; nothing from them is read (protocol §9)
TASK_BLOCK = (53000, 53999)
COHORT_ROLES = ("K0", "R", "D3", "S", "U")
FORBIDDEN_RANGES = {
    **fp2.FORBIDDEN_RANGES,
    "task071_072_block": fp2.TASK_BLOCK,  # 51000-52199, including D2 (52000-52015)
    "cohort_C": (fp2.COHORT_C[0], fp2.COHORT_C[-1]),  # 45300-45339, now simulated once (M2)
}
MODEL_SEEDS = (7300, 7301, 7302)  # W and N, one model each per seed
SEEDS = {
    "shift_direction": 7303,
    "misaim": 7304,
    "corpus_split": 7305,
    "corpus_plan_salt": 7306,
    "h_rand": 7307,
    "random_controller": 7308,
    "bootstrap": 7309,
    "readout_folds": 7310,
    "sampler_salt": 7311,
    "wrong_actions": 7312,
    "comparative_rank": 7313,
}


def seeds_of(role: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[role]
    return tuple(range(low, high + 1))


def smoke_seeds() -> tuple[int, ...]:
    return tuple(range(SMOKE_SEEDS[0], SMOKE_SEEDS[1] + 1))


def check_seed_ranges() -> None:
    """Every TASK-073 range lies in 53000-53999, the ranges are pairwise disjoint, and none
    overlaps a forbidden range, cohort C or D2."""
    spans = dict(SEED_RANGES) | {"smoke": SMOKE_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("TASK-073 seed ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < TASK_BLOCK[0] or high > TASK_BLOCK[1]:
            raise GuardError(f"the TASK-073 range {name} leaves {TASK_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"the TASK-073 range {name} overlaps {other}")
    model_and_rng = (*MODEL_SEEDS, *SEEDS.values())
    if len(set(model_and_rng)) != len(model_and_rng):
        raise GuardError("TASK-073 model and RNG seeds must be distinct")


def check_role_seeds(role: str, seeds, *, smoke: bool = False) -> tuple[int, ...]:
    """G-seeds: plain ints, exactly the role's seeds in order (or smoke seeds only in a smoke).

    Cohort C, D2 and every forbidden range are refused before any coercion."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    if smoke:
        if not set(seeds) <= set(smoke_seeds()) or len(set(seeds)) != len(seeds):
            raise GuardError("G-seeds: a smoke simulates smoke seeds 53950-53999 only")
        return seeds
    if role not in SEED_RANGES:
        raise GuardError(f"G-seeds: unknown role {role!r}")
    if seeds != seeds_of(role):
        raise GuardError(f"G-seeds: {role} simulates exactly {SEED_RANGES[role]}, once, in order")
    return seeds


# ----- resets and the stored cohort values --------------------------------------------------------
RESET_CENTERS = {"object_xy": (0.34, -0.18), "plate_xy": (0.49, -0.09)}  # TASK-047
WIDE_JITTER_M = {"object_xy": 0.03, "plate_xy": 0.02}


def wide_reset_values(seed: int) -> dict:
    """TASK-047's wide-jitter reset (``scripts/evaluate_apple.wide_reset``), in NumPy; a test pins
    equality. Used only to generate the stored values; runners read the stored values."""
    rng = np.random.default_rng(seed)
    obj = np.array(RESET_CENTERS["object_xy"]) + rng.uniform(
        -WIDE_JITTER_M["object_xy"], WIDE_JITTER_M["object_xy"], 2
    )
    plate = np.array(RESET_CENTERS["plate_xy"]) + rng.uniform(
        -WIDE_JITTER_M["plate_xy"], WIDE_JITTER_M["plate_xy"], 2
    )
    return {"object_xy": obj.tolist(), "plate_xy": plate.tolist()}


# ----- the plate-shift condition (owner D1: simulation-only, diagnostic) --------------------------
SHIFT_GRID_CM = (3, 4, 5, 6)  # K0's |d| grid (owner D3)
SHIFT_STEPS = (300, 480)  # 300; 480 is the one allowed retry (owner D3)
BASE_XY = (0.0, 0.0)  # the fixed pelvis in world xy (simulation.py; a MuJoCo test pins it)
E9_RELEASE = {  # RestingPlaceExpert defaults with e9's kwargs: the release-pose inputs
    "release_dx": EXPERT["release_dx"],
    "reach_radius_m": 0.485,
    "floor": 0.10,
    "ceiling": 0.26,
}
ARC_TOLERANCE_M = 0.005
DIRECTION_GRID_DEG = 1
TABLETOP = {"x": (0.18, 0.65), "y": (-0.32, 0.32)}  # simulation.reset's tabletop check
PLATE_APPLE_CLEARANCE_M = 0.071 + 0.027 + 0.005  # simulation.reset's plate/apple rule


def release_xy(plate_xy) -> np.ndarray:
    """Where e9 puts the palm for release (world xy), from its own ``release_pose`` geometry."""
    from embodied_jepa.resting_expert import RestingPlaceExpert  # NumPy only at import

    xy = np.asarray(plate_xy, float) - np.asarray(BASE_XY) + [E9_RELEASE["release_dx"], 0.0]
    pose = RestingPlaceExpert.release_pose(
        xy,
        reach_radius_m=E9_RELEASE["reach_radius_m"],
        floor=E9_RELEASE["floor"],
        ceiling=E9_RELEASE["ceiling"],
    )
    return pose[:2] + np.asarray(BASE_XY)


def eligible_directions(plate_xy, object_xy, magnitude_m: float) -> list[int]:
    """Shift directions (whole degrees) that keep e9's release geometry relative to the plate.

    e9's release pose is pulled back to its reach sphere on every v2 reset (protocol §3.2), so a
    shift along x would move the plate but not the release. A direction is eligible when e9's
    release point relative to the shifted plate stays within ``ARC_TOLERANCE_M`` of its value at
    the reset, the shifted plate stays on the tabletop, and it keeps the reset rule's clearance
    from the apple's reset position."""
    plate = np.asarray(plate_xy, float)
    apple = np.asarray(object_xy, float)
    before = release_xy(plate) - plate
    out = []
    for deg in range(0, 360, DIRECTION_GRID_DEG):
        theta = math.radians(deg)
        moved = plate + magnitude_m * np.array([math.cos(theta), math.sin(theta)])
        if not (
            TABLETOP["x"][0] <= moved[0] <= TABLETOP["x"][1]
            and TABLETOP["y"][0] <= moved[1] <= TABLETOP["y"][1]
        ):
            continue
        if np.linalg.norm(moved - apple) < PLATE_APPLE_CLEARANCE_M:
            continue
        try:
            after = release_xy(moved) - moved
        except ContractError:
            continue
        if np.linalg.norm(after - before) <= ARC_TOLERANCE_M:
            out.append(deg)
    return out


def shift_vector(seed: int, reset: dict, magnitude_cm: int) -> list[float]:
    """The stored shift (world xy, m) of one reset at one |d|: a direction drawn uniformly from
    the eligible set with ``default_rng(SeedSequence([7303, seed]))`` (one draw per reset, shared
    by every |d|)."""
    eligible = eligible_directions(reset["plate_xy"], reset["object_xy"], magnitude_cm / 100.0)
    if not eligible:
        raise GuardError(f"no eligible shift direction for seed {seed} at {magnitude_cm} cm")
    u = np.random.default_rng(np.random.SeedSequence([SEEDS["shift_direction"], int(seed)]))
    pick = eligible[int(math.floor(float(u.uniform()) * len(eligible)))]
    theta = math.radians(pick)
    return [magnitude_cm / 100.0 * math.cos(theta), magnitude_cm / 100.0 * math.sin(theta)]


def cohort_values(role: str) -> dict[str, dict]:
    """The values stored in the manifest for one cohort: per seed its reset and its shift at
    every |d| of the grid (cohort U is never shifted and stores none)."""
    out = {}
    for seed in seeds_of(role):
        reset = wide_reset_values(seed)
        entry = {"seed": seed, **reset}
        if role != "U":
            entry["shift_m"] = {str(m): shift_vector(seed, reset, m) for m in SHIFT_GRID_CM}
        out[str(seed)] = entry
    return out


def cohort_digest(values: dict) -> str:
    return fm.cohort_digest(values)  # sha256 of the sorted-key compact JSON


def stored_cohort(manifest: dict, role: str) -> dict[int, dict]:
    """The stored values of one cohort, checked against their digest and the whitelist."""
    node = manifest["cohorts"][role]
    if cohort_digest(node["values"]) != node["sha256"]:
        raise GuardError(f"G-cohort: the stored {role} values differ from their digest")
    if sorted(int(s) for s in node["values"]) != list(seeds_of(role)):
        raise GuardError(f"G-cohort: the stored {role} cohort is not exactly its seeds")
    out = {}
    for seed in seeds_of(role):
        entry = node["values"][str(seed)]
        want = {"seed", "object_xy", "plate_xy"} | ({"shift_m"} if role != "U" else set())
        if entry["seed"] != seed or set(entry) != want:
            raise GuardError(f"G-cohort: stored {role} entry {seed} is malformed")
        out[seed] = copy.deepcopy(entry)
    return out


# ----- the controller: decision grid, candidates, target, tie rule --------------------------------
DECISION_STEPS = tuple(range(304, 577, 16))  # 304, 320, ..., 576 (18 decisions)
CHUNK = 16  # commands per decision; also W's horizon (TASK-066's validated h = 16)
GRID_OFFSETS_CM = (-2, -1, 0, 1, 2)
CANDIDATE_OFFSETS_M = tuple(
    (dx / 100.0, dy / 100.0) for dx in GRID_OFFSETS_CM for dy in GRID_OFFSETS_CM
)  # 25; index 12 is (0, 0), the incumbent
INCUMBENT_INDEX = CANDIDATE_OFFSETS_M.index((0.0, 0.0))
K = len(CANDIDATE_OFFSETS_M)
TIE_TOLERANCE_CM = 1e-6  # the incumbent wins unless a candidate is lower by more than this
# o*(t): the median apple-minus-plate offset (cm, world xy) at post-look step t over the 112
# counted-success train roots of apple-look-v2-linux run-1 (e9, unshifted; manifest sha256
# 67c342f6...). Frozen now, before any TASK-073 seed exists (protocol §3.5; deviation D-3).
O_STAR_CM = {
    320: (-15.4195, -9.0746),
    336: (-15.4759, -9.0694),
    352: (-15.3417, -9.0633),
    368: (-15.3708, -9.134),
    384: (-15.3102, -9.1261),
    400: (-15.3011, -9.06),
    416: (-12.6455, -6.492),
    432: (-8.5877, -2.4611),
    448: (-4.3337, -0.1759),
    464: (-0.8722, -0.1816),
    480: (0.3462, -0.1463),
    496: (0.5174, -0.1168),
    512: (0.0537, -0.0554),
    528: (-1.3059, -0.1013),
    544: (-1.5759, -0.1414),
    560: (-1.6272, -0.1087),
    576: (-1.5407, -0.0969),
    592: (-1.1686, -0.093),
}
O_STAR_SOURCE = {
    "corpus": EVIDENCE["corpus"],
    "corpus_manifest_sha256": EVIDENCE["corpus_manifest_sha256"],
    "roots": "the 112 train roots with a counted success (run-1's B-replay library)",
    "statistic": "per-step median of apple xy (the stored privileged label) minus the reset "
    "plate xy, in cm, at steps t + 16 for every decision step t",
}


def candidate_aims(incumbent_plate_xy) -> np.ndarray:
    """The 25 plate hypotheses: the incumbent plus the 5 x 5 grid at 1 cm (index 12 = incumbent)."""
    return np.asarray(incumbent_plate_xy, np.float64)[None] + np.asarray(CANDIDATE_OFFSETS_M)


def choose(costs_cm) -> int:
    """The argmin of the critic's costs; the incumbent wins every tie within TIE_TOLERANCE_CM
    (and any non-finite cost loses). With an action-blind critic every cost is equal, so the
    choice is the incumbent and the arm reduces to P-reread."""
    costs = np.asarray(costs_cm, np.float64)
    if costs.shape != (K,):
        raise ContractError(f"a decision needs {K} costs")
    costs = np.where(np.isfinite(costs), costs, np.inf)
    best = int(np.argmin(costs))
    if not np.isfinite(costs[INCUMBENT_INDEX]):
        return best if np.isfinite(costs[best]) else INCUMBENT_INDEX
    if costs[best] < costs[INCUMBENT_INDEX] - TIE_TOLERANCE_CM:
        return best
    return INCUMBENT_INDEX


def critic_cost_cm(predicted_offset_cm, target_step: int) -> np.ndarray:
    """J = || predicted apple-minus-plate offset - o*(t + 16) || in cm."""
    target = np.asarray(O_STAR_CM[int(target_step)], np.float64)
    return np.linalg.norm(np.asarray(predicted_offset_cm, np.float64) - target, axis=-1)


# ----- arms ---------------------------------------------------------------------------------------
ARMS = {
    "H-LeWM": ("L1", "P-3 with R-mid re-reads; W (val-selected seed) ranks the 25 aims"),
    "P-stale": ("L1", "P-3 unchanged: the post-look estimates for the whole attempt"),
    "P-reread": ("L1", "P-3 with R-mid re-reading the plate at every decision step"),
    "H-N": ("L1", "as H-LeWM with the no-action model N (action-blind, run closed loop)"),
    "H-shuf": ("L1", "as H-LeWM with the start latent of reset (i + 1) mod n (scene-blind)"),
    "H-rand": ("L1", "as H-LeWM with a seeded uniform pick among the same 25 candidates"),
    "H-LeWM-s1": ("L1", "H-LeWM with the second W seed (secondary: sign agreement)"),
    "H-LeWM-s2": ("L1", "H-LeWM with the third W seed (secondary: sign agreement)"),
    "P-truth": ("L4", "P-3 given the true post-shift plate at the shift step (perception ceiling)"),
    "H-sim": ("L4", "the same 25 candidates chosen by privileged cloned-state rollouts"),
    "B-oracle-shift": ("L4", "e9 built from the post-shift truth (harness; post-shift oracle)"),
    "B-hold": ("L4", "hold (harness)"),
    "B-random": ("L4", "uniform random within the configured bounds (harness)"),
}
L1_ARMS = tuple(a for a, (rung, _) in ARMS.items() if rung == "L1")
S_ARMS = tuple(ARMS)  # every arm runs once per S reset, paired
U_ARMS = ("P-stale", "P-reread", "H-LeWM")
K0_ARMS = ("B-oracle-shift", "P-stale", "P-truth", "H-sim")  # K0's order of evaluation
D3_ARMS = ("P-stale", "P-reread", "H-LeWM", "H-N", "H-shuf", "H-rand", "H-sim", "B-oracle-shift")
LEARNED_LABEL = (
    "a learned policy trained on privileged expert labels (e9, with a privileged DAgger "
    "labeller at training time) plus a LeWM critic; not LeWM driving the robot alone, and not "
    "the frozen v1 benchmark (owner D2)"
)

# ----- K0: condition calibration (simulator only, no world model) ---------------------------------
K0_RESETS = 32
K0_BARS = {  # evaluated in this order for each |d|; the first failed bar ends that |d|
    "B-oracle-shift_min": 28,
    "P-stale_max": 8,
    "P-truth_min": 20,
    "H-sim_minus_P-truth_min": 4,
}
K0_H_SIM_CENTRE = (
    "in K0 the 25 candidates of H-sim are centred on the true post-shift plate (R-mid does not "
    "exist before the corpus); from D3 on they are centred on R-mid's incumbent"
)


def _k0_bar(arm: str, counts: dict) -> bool:
    if arm == "B-oracle-shift":
        return counts[arm] >= K0_BARS["B-oracle-shift_min"]
    if arm == "P-stale":
        return counts[arm] <= K0_BARS["P-stale_max"]
    if arm == "P-truth":
        return counts[arm] >= K0_BARS["P-truth_min"]
    return counts["H-sim"] - counts["P-truth"] >= K0_BARS["H-sim_minus_P-truth_min"]


def k0_cell_passes(counts: dict) -> dict:
    """One (shift step, |d|) cell: the bars in ``K0_ARMS`` order, stopping at the first failure.
    ``counts`` maps an arm to its counted successes of 32; an arm that did not run fails."""
    for arm in K0_ARMS:
        if counts.get(arm) is None or not _k0_bar(arm, counts):
            return {"passes": False, "failed_bar": arm}
    return {"passes": True, "failed_bar": None}


def k0_next_arm(counts: dict) -> str | None:
    """The next arm K0 runs in a cell, or None once the cell is decided (a bar failed, or all
    four passed). Arms after a failed bar are not run."""
    for arm in K0_ARMS:
        if counts.get(arm) is None:
            return arm
        if not _k0_bar(arm, counts):
            return None
    return None


def k0_select(cells: dict) -> dict:
    """K0's rule. ``cells[(step, cm)]`` holds the counts of the arms that ran. The smallest |d|
    passing all four bars at step 300 is chosen; only if none does, the same at step 480; if
    neither, S-NO-CONDITION (escalates to the owner under D7)."""
    for step in SHIFT_STEPS:
        for cm in SHIFT_GRID_CM:
            if (step, cm) in cells and k0_cell_passes(cells[(step, cm)])["passes"]:
                return {"row": "K0-PASS", "shift_step": step, "shift_cm": cm}
    return {"row": "S-NO-CONDITION", "shift_step": None, "shift_cm": None}


# ----- the corpus: apple-shift-v2 -----------------------------------------------------------------
CORPUS = "apple-shift-v2"
CORPUS_SPLITS = {"train": 240, "val": 30, "test": 30}
READ_SPLITS = ("train", "val")  # the test split is never decoded
SHIFTED_FRACTION = 0.75
MISAIM_FRACTION = 0.50
MISAIM_RADIUS_M = 0.03
NOISE_LEVELS = fp2.NOISE_LEVELS  # TASK-048 Perturber, root i gets level i % 4
EXTRA_EPISODE_ARRAYS = ("plate",)  # float32 [T + 1, 3]: the plate trajectory (privileged label)
W_EXTRA_CORPUS = {
    "corpus": EVIDENCE["corpus"],
    "manifest_sha256": EVIDENCE["corpus_manifest_sha256"],
    "splits": ("train", "val"),  # run-1's 190 train + val roots are added to W's training set
}
FEATURE_BAND = (240, 660)  # frames featurised per root; every decision window lies inside


def corpus_plan() -> list[dict]:
    """One root per corpus seed, fixed before collection: split, shift flag, mis-aim flag and
    vector, noise level and noise seed. The |d| and shift step are K0's choice; the direction is
    the stored rule's (``shift_vector``)."""
    seeds = seeds_of("corpus")
    order = list(seeds)
    np.random.default_rng(SEEDS["corpus_split"]).shuffle(order)
    split = {}
    for i, seed in enumerate(order):
        split[seed] = (
            "val"
            if i < CORPUS_SPLITS["val"]
            else "test"
            if i < CORPUS_SPLITS["val"] + CORPUS_SPLITS["test"]
            else "train"
        )
    rng = np.random.default_rng(SEEDS["misaim"])
    shifted = set(rng.permutation(seeds)[: round(SHIFTED_FRACTION * len(seeds))].tolist())
    misaimed = set(rng.permutation(seeds)[: round(MISAIM_FRACTION * len(seeds))].tolist())
    plan = []
    for index, seed in enumerate(seeds):
        r = np.random.default_rng(np.random.SeedSequence([int(seed), SEEDS["corpus_plan_salt"]]))
        noise_seed = int(r.integers(2**31))
        radius = MISAIM_RADIUS_M * math.sqrt(float(r.uniform()))
        angle = 2 * math.pi * float(r.uniform())
        plan.append(
            {
                "episode_id": f"shift2-{seed}",
                "session_id": f"shift2-reset-{seed}",
                "seed": int(seed),
                "split": split[seed],
                "shifted": seed in shifted,
                "misaimed": seed in misaimed,
                "misaim_m": (
                    [radius * math.cos(angle), radius * math.sin(angle)]
                    if seed in misaimed
                    else [0.0, 0.0]
                ),
                "noise_level": NOISE_LEVELS[index % len(NOISE_LEVELS)],
                "noise_seed": noise_seed,
            }
        )
    return plan


# ----- models, readouts, budget -------------------------------------------------------------------
W_TOKEN_GRID = 4  # token_dynamics.TOKEN_GRID; a test pins equality
W_MODEL = {
    "backend": "leworldmodel",  # the one-line swap
    "latent": "frozen DINOv2 ViT-S/14 final patch tokens pooled to 4 x 4 (TASK-066), 6144-d",
    "config": "token_dynamics.MODEL_CONFIG with the adapter defaults (TASK-066), unchanged",
    "train_horizon": 16,
    "batch_size": 64,
    "arms": ("W", "N"),  # N: every action zero, in training and evaluation
}
BUDGET = {
    "calibration_runs": (("W", 7300), ("N", 7300)),
    "calibration_updates": 60_000,
    "calibration_select_every": 1_000,
    "saturation_tolerance": 0.01,
    "factor": 2.0,
    "step": 5_000,
    "min": 10_000,
    "cap": 60_000,
    "selection_points": 20,
    "last_two_rule": "if any W or N model selects one of its last two selection points, U is "
    "raised once to min(2U, 60 000) and all six are retrained; if that still happens at the cap, "
    "the run escalates to the owner and nothing is frozen",
}


def budget_updates(saturation_updates) -> dict:
    """U = clamp(5000 * ceil(2 * max(u_sat) / 5000), 10 000, 60 000); escalate above the cap."""
    values = [int(u) for u in saturation_updates]
    if not values or min(values) < 1:
        raise ContractError("the budget rule needs positive saturation updates")
    wanted = BUDGET["factor"] * max(values)
    updates = int(BUDGET["step"] * math.ceil(wanted / BUDGET["step"]))
    return {
        "max_saturation_update": max(values),
        "wanted": wanted,
        "updates": min(max(updates, BUDGET["min"]), BUDGET["cap"]),
        "escalate": bool(wanted > BUDGET["cap"]),
    }


READOUTS = {
    "R_off": "linear ridge (primal, intercept) from the raw pooled token latent (6144-d, "
    "train-standardised) to the apple-minus-plate offset xy; rows: apple-shift-v2 train roots at "
    "steps 304-592 every 16",
    "R_mid": "linear ridge (dual, intercept) from the full DINOv2 tokens (256 x 384, the P "
    "readout's feature, train-standardised) to the plate xy; rows: apple-shift-v2 train roots at "
    "the decision steps",
    "lambda_grid_relative": (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    "inner_folds": 5,
    "fold_seed": SEEDS["readout_folds"],
    "fold_unit": "root (session)",
}
W_SEED_SELECTION = (
    "the primary W seed is the one with the lowest val criterion at its selected checkpoint "
    "among the three; fixed before any closed-loop run"
)

# ----- offline gates (every gate on all three W seeds) --------------------------------------------
BOOTSTRAP_RESAMPLES = 10_000
O1 = {
    "horizons": (8, 16),
    "window_sets": ("overall", "post_shift"),
    "rank_ratio_min": 0.16,
    "std_ratio_min": 0.39,
    "collapsed_fraction_max": 0.05,
    "comparative_rank_lower_gt": 0.0,
    "truncation_controls_must_fail": (1, 2, 4),
    "copy_last_ratio_upper_max": 0.8,
    "n_ratio_upper_lt": 1.0,
    "wrong_action_ratio_lower_min": 1.10,
}
O2 = {
    "windows": "every val root of apple-shift-v2; windows starting at the decision steps t with "
    "t >= max(416, shift step) whose true apple-minus-plate offset moves by at least 1 cm over "
    "the 16 commands (TASK-054's moving cohort, world_model_v2.MOVING_THRESHOLD_M)",
    "first_step": 416,
    "moving_threshold_m": 0.01,
    "encoded_median_max_cm": 1.0,
    "predicted_median_max_cm": 1.5,
    "ratio_vs_persistence_upper_max": 0.8,  # (i): TASK-054's G2a bar
    "n_ratio_vs_persistence_upper_min": 0.8,  # (ii): N must fail (i), else the gate is void
    "ratio_vs_clock_prior_upper_max": 0.8,  # (iii), added: W must beat the o*-only predictor
}
R_POINTS = {300: (416, 448, 480, 512, 544), 480: (480, 496, 512, 528, 544)}
O3 = {
    "rho_lower_min": 0.5,
    "margin_lower_min": 0.3,
    "blind_rankers": ("copy-last", "N", "L-shuf", "prior-distance"),
    "void_if_blind_rho_at_least": 0.5,  # median rho of a void ranker at or above it voids O3
    "void_rankers": ("copy-last", "N", "L-shuf"),  # prior-distance enters via the margin only
}
O4 = {"median_regret_max_cm": 1.0, "paired_regret_difference_upper_lt": 0.0}
# O0 (added, protocol §6.3 and §10 D-17): the ranking gates need an incumbent that can be
# improved on. If R-mid's incumbent is already near the best of its 25 aims on cohort R, O3's
# margin and O4's paired bar cannot pass for any critic, and the row is a headroom row (the D7
# fallback), not a critic failure.
O0 = {"median_incumbent_regret_min_cm": 0.5}
O5 = {"median_relative_chunk_error_max": 0.25}


def spearman(a, b) -> float:
    """Spearman's rho with average ranks; 0.0 when either input is constant."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ContractError("spearman needs two equal 1-d vectors")
    if np.ptp(a) == 0 or np.ptp(b) == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=float)
        for value in np.unique(x):
            tie = x == value
            r[tie] = r[tie].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_median_ci(values, clusters, *, seed=SEEDS["bootstrap"], resamples=None) -> dict:
    """The median of per-item values with a cluster (session) bootstrap 95 % percentile CI."""
    v = np.asarray(values, np.float64)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    n = resamples or BOOTSTRAP_RESAMPLES
    boot = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        boot[b] = np.median(v[np.concatenate([groups[i] for i in pick])])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median": float(np.median(v)), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def cluster_median_ratio(num, den, clusters, *, seed=SEEDS["bootstrap"], resamples=None) -> dict:
    """median(num) / median(den) over the same items, with a cluster bootstrap 95 % CI."""
    a, b = np.asarray(num, np.float64), np.asarray(den, np.float64)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    n = resamples or BOOTSTRAP_RESAMPLES
    boot = np.empty(n)
    for k in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d = np.median(b[idx])
        boot[k] = np.median(a[idx]) / d if d > 0 else np.finfo(np.float64).max
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = float(np.median(b))
    return {
        "ratio": float(np.median(a)) / base if base > 0 else None,
        "ci95": [float(lo), float(hi)],
        "n": int(len(a)),
    }


def o1_seed_passes(per: dict) -> dict:
    """O1 for one W seed. ``per[(window_set, h)]`` holds ``collapse`` (td.collapse_statistics),
    ``comparative`` (lower bound), ``truncation_pass`` ({k: bool}), and the ratio dicts
    ``copy``, ``n``, ``shuffled``, ``zero`` (each with ``ci95``)."""
    t = O1
    parts = {}
    for key, v in per.items():
        c = v["collapse"]
        g1 = (
            c["predicted_collapsed_fraction"] <= t["collapsed_fraction_max"]
            and c["effective_rank_ratio"] >= t["rank_ratio_min"]
            and c["std_ratio"] >= t["std_ratio_min"]
            and v["comparative"]["lower"] > t["comparative_rank_lower_gt"]
            and not any(v["truncation_pass"][str(k)] for k in t["truncation_controls_must_fail"])
        )
        parts[f"{key[0]}@h{key[1]}"] = {
            "G1": bool(g1),
            "G2": bool(v["copy"]["ci95"][1] <= t["copy_last_ratio_upper_max"]),
            "G3": bool(v["n"]["ci95"][1] < t["n_ratio_upper_lt"]),
            "G4": bool(
                v["shuffled"]["ci95"][0] >= t["wrong_action_ratio_lower_min"]
                and v["zero"]["ci95"][0] >= t["wrong_action_ratio_lower_min"]
            ),
        }
    need = {f"{w}@h{h}" for w in t["window_sets"] for h in t["horizons"]}
    if set(parts) != need:
        raise ContractError(f"O1 needs exactly {sorted(need)}")
    return {"parts": parts, "passes": all(all(p.values()) for p in parts.values())}


def o2_passes(m: dict) -> dict:
    """O2 for one W seed. Keys: ``encoded_median_cm``, ``predicted_median_cm``,
    ``ratio_vs_persistence`` (W), ``n_ratio_vs_persistence`` (N), ``ratio_vs_clock_prior``."""
    t = O2
    void = m["n_ratio_vs_persistence"]["ci95"][1] < t["n_ratio_vs_persistence_upper_min"]
    passes = (
        m["encoded_median_cm"] <= t["encoded_median_max_cm"]
        and m["predicted_median_cm"] <= t["predicted_median_max_cm"]
        and m["ratio_vs_persistence"]["ci95"][1] <= t["ratio_vs_persistence_upper_max"]
        and m["ratio_vs_clock_prior"]["ci95"][1] <= t["ratio_vs_clock_prior_upper_max"]
    )
    return {"passes": bool(passes and not void), "void": bool(void)}


def o3_passes(m: dict) -> dict:
    """O3 for one W seed: ``rho_w`` (median CI), ``blind`` {ranker: median CI}, ``margin``
    (median of rho_W - max blind, with CI)."""
    t = O3
    void = any(
        m["blind"][r]["median"] >= t["void_if_blind_rho_at_least"] for r in t["void_rankers"]
    )
    passes = m["rho_w"]["ci95"][0] >= t["rho_lower_min"] and (
        m["margin"]["ci95"][0] >= t["margin_lower_min"]
    )
    return {"passes": bool(passes and not void), "void": bool(void)}


def o4_passes(m: dict) -> dict:
    """O4 for one W seed: ``regret_w`` (median CI, cm) and ``regret_difference`` (median of
    regret_W - regret_incumbent, with CI)."""
    t = O4
    passes = (
        m["regret_w"]["median"] <= t["median_regret_max_cm"]
        and m["regret_difference"]["ci95"][1] < t["paired_regret_difference_upper_lt"]
    )
    return {"passes": bool(passes)}


def o5_passes(relative_errors) -> dict:
    e = np.asarray(relative_errors, np.float64)
    if e.size == 0 or not np.isfinite(e).all():
        return {"passes": False, "median": None}
    med = float(np.median(e))
    return {"passes": bool(med <= O5["median_relative_chunk_error_max"]), "median": med}


OFFLINE_ROWS = (
    "OFFLINE-PASS",
    "WMC-NO-DYNAMICS",
    "WMC-G2A",
    "WMC-O2-VOID",
    "R-NO-HEADROOM",
    "WMC-NO-RANK",
    "WMC-O3-VOID",
    "WMC-PROPOSAL",
)


def o0_passes(regret_incumbent: dict) -> dict:
    """O0: the median true regret of R-mid's incumbent on cohort R (cm) is at least 0.5."""
    med = float(regret_incumbent["median"])
    return {"passes": bool(med >= O0["median_incumbent_regret_min_cm"]), "median": med}


def decide_offline(seeds: dict, o5: dict, o0: dict) -> dict:
    """First match over O1, O2, O0, O3, O4, O5. ``seeds[s]`` holds ``O1``, ``O2``, ``O3``,
    ``O4`` results (the dicts returned above) for each W seed; every gate must pass on all three
    seeds. ``o0`` is ``o0_passes`` on cohort R (seed-free: it concerns the incumbent)."""
    if set(seeds) != set(MODEL_SEEDS):
        raise ContractError(f"the offline decision needs every W seed {MODEL_SEEDS}")
    if not all(seeds[s]["O1"]["passes"] for s in MODEL_SEEDS):
        row = "WMC-NO-DYNAMICS"
    elif any(seeds[s]["O2"]["void"] for s in MODEL_SEEDS):
        row = "WMC-O2-VOID"
    elif not all(seeds[s]["O2"]["passes"] for s in MODEL_SEEDS):
        row = "WMC-G2A"
    elif not o0["passes"]:
        row = "R-NO-HEADROOM"
    elif any(seeds[s]["O3"]["void"] for s in MODEL_SEEDS):
        row = "WMC-O3-VOID"
    elif not all(seeds[s]["O3"]["passes"] and seeds[s]["O4"]["passes"] for s in MODEL_SEEDS):
        row = "WMC-NO-RANK"
    elif not o5["passes"]:
        row = "WMC-PROPOSAL"
    else:
        row = "OFFLINE-PASS"
    return {
        "row": row,
        "abandonment_clause_fires": abandonment_fires(row),
        "fallback_authorised": row in FALLBACK_ROWS,
    }


# ----- D3: the development closed loop (non-gating) -----------------------------------------------
D3_RESETS = 16
D3_BARS = {"H-sim_minus_P-reread_min": 2, "H-LeWM_minus_best_blind_min": 2}


def decide_d3(counts: dict) -> dict:
    """D3's stop rule on 16 development resets. Nothing on D3 is refitted."""
    if counts["H-sim"] - counts["P-reread"] < D3_BARS["H-sim_minus_P-reread_min"]:
        row = "NO-HEADROOM"
    elif (
        counts["H-LeWM"] - max(counts["P-reread"], counts["H-shuf"])
        < D3_BARS["H-LeWM_minus_best_blind_min"]
    ):
        row = "WMC-DEV-STOP"
    else:
        row = "D3-GO"
    return {"row": row, "fallback_authorised": row in FALLBACK_ROWS}


# ----- the gated rows on S (64) and U (32) --------------------------------------------------------
S_RESETS = 64
U_RESETS = 32
GATED = {
    "void_b_oracle_shift_min": 58,
    "s_void_condition_p_stale_max": 24,
    "h_no_headroom_min": 12,
    "difference_min": 10,
    "p_max_exclusive": 0.01,
    "g5_min_vs_p_stale": 16,
    "g6_u_tolerance": 2,
    "g7_median_decision_seconds_max": 0.250,
    "determinism_rerun_resets": 4,
}
GATED_ROWS = (
    "VOID",
    "S-VOID-CONDITION",
    "H-NO-HEADROOM",
    "HYB-PASS",
    "HYB-SLOW",
    "HYB-SCENE-BLIND",
    "HYB-HARM",
    "HYB-NO-GAIN",
)


def mcnemar_one_sided(only_first: int, only_second: int) -> float:
    """Exact one-sided McNemar p: P(X >= only_first) for X ~ Bin(n_d, 1/2)."""
    b, c = int(only_first), int(only_second)
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) for i in range(b, n + 1)) / 2**n)


def paired_one_sided(first, second) -> dict:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise GuardError("paired arms need the same resets")
    only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "difference": int(a.sum()) - int(b.sum()),
        "only_first": only_first,
        "only_second": only_second,
        "p_one_sided": mcnemar_one_sided(only_first, only_second),
    }


def decide_gated(s: dict, u: dict, harness: dict) -> dict:
    """First-matching gated row. ``s[arm]`` / ``u[arm]``: per-reset counted successes (64 / 32).
    ``harness``: ``b_hold_grasps``, ``b_random_grasps``, ``privileged_ok`` {L1 arm: bool},
    ``determinism_ok``, ``cuda_allocation_failed``, ``median_decision_seconds``."""
    t = GATED
    for arm in S_ARMS:
        if np.asarray(s[arm]).shape != (S_RESETS,):
            raise GuardError(f"{arm} needs one result per S reset")
    for arm in U_ARMS:
        if np.asarray(u[arm]).shape != (U_RESETS,):
            raise GuardError(f"{arm} needs one result per U reset")
    n = {arm: int(np.asarray(s[arm], bool).sum()) for arm in S_ARMS}
    void = (
        n["B-oracle-shift"] < t["void_b_oracle_shift_min"]
        or harness["b_hold_grasps"] > 0
        or harness["b_random_grasps"] > 0
        or not all(harness["privileged_ok"].get(a, False) for a in L1_ARMS)
        or not harness["determinism_ok"]
        or harness["cuda_allocation_failed"]
    )
    pairs = {
        other: paired_one_sided(s["H-LeWM"], s[other])
        for other in ("P-reread", "H-shuf", "H-N", "H-rand", "P-stale")
    }

    def diff_gate(other):
        p = pairs[other]
        return p["difference"] >= t["difference_min"] and p["p_one_sided"] < t["p_max_exclusive"]

    lat = harness["median_decision_seconds"]
    gates = {
        "G1": diff_gate("P-reread"),
        "G2": diff_gate("H-shuf"),
        "G3": diff_gate("H-N"),
        "G4": diff_gate("H-rand"),
        "G5": pairs["P-stale"]["difference"] >= t["g5_min_vs_p_stale"],
        "G6": int(np.asarray(u["H-LeWM"], bool).sum())
        >= int(np.asarray(u["P-stale"], bool).sum()) - t["g6_u_tolerance"],
        "G7": lat is not None
        and bool(np.isfinite(lat))
        and lat <= t["g7_median_decision_seconds_max"],
    }
    if void:
        row = "VOID"
    elif n["P-stale"] > t["s_void_condition_p_stale_max"]:
        row = "S-VOID-CONDITION"
    elif n["H-sim"] - n["P-reread"] < t["h_no_headroom_min"]:
        row = "H-NO-HEADROOM"
    elif all(gates.values()):
        row = "HYB-PASS"
    elif all(v for k, v in gates.items() if k != "G7"):
        row = "HYB-SLOW"  # every outcome gate passes; only the latency bound fails
    elif gates["G1"] and gates["G3"] and not gates["G2"]:
        row = "HYB-SCENE-BLIND"
    elif not gates["G6"]:
        row = "HYB-HARM"
    else:
        row = "HYB-NO-GAIN"
    secondary = {
        seed_arm: {
            other: int(np.asarray(s[seed_arm], bool).sum() - np.asarray(s[other], bool).sum())
            for other in ("P-reread", "H-shuf", "H-N")
        }
        for seed_arm in ("H-LeWM", "H-LeWM-s1", "H-LeWM-s2")
    }
    signs = {
        other: len({int(np.sign(secondary[a][other])) for a in secondary}) == 1
        for other in ("P-reread", "H-shuf", "H-N")
    }
    return {
        "row": row,
        "gates": gates,
        "counts_of_64": n,
        "u_counts_of_32": {a: int(np.asarray(u[a], bool).sum()) for a in U_ARMS},
        "paired": pairs,
        "secondary_sign_agreement": {"differences": secondary, "same_sign": signs},
        "abandonment_clause_fires": abandonment_fires(row),
        "fallback_authorised": row in FALLBACK_ROWS,
    }


# ----- abandonment clause and the owner's fallback (D7) -------------------------------------------
CLAUSE_ROWS = ("WMC-G2A", "WMC-NO-RANK", "HYB-NO-GAIN")
CLAUSE_SCOPE = (
    "P-3 proposals plus LeWM frozen-DINOv2-token critic selection on apple-to-plate-v2 at 112 px: "
    "no further critic, K, horizon or cost variant is preregistered on apple-shift-v2 or "
    "apple-look-v2-linux without new evidence of a different kind. The LeWM backend, the token "
    "latent, the v2 task and the planner fallback stay open"
)
FALLBACK_ROWS = ("S-NO-CONDITION", "R-NO-HEADROOM", "NO-HEADROOM", "H-NO-HEADROOM")
FALLBACK = (
    "owner D7 (2026-09-29): TASK-074, the LeWM-only planner (lewm-planner), is pre-authorised as "
    "the fallback if TASK-073 ends S-NO-CONDITION, H-NO-HEADROOM or NO-HEADROOM; R-NO-HEADROOM "
    "(added here, the offline form of NO-HEADROOM) is proposed to count with them, pending the "
    "owner's confirmation; TASK-074 needs its own preregistration, which TASK-073 does not design"
)
ALL_ROWS = (
    "K0-PASS",
    "S-NO-CONDITION",
    *OFFLINE_ROWS,
    "NO-HEADROOM",
    "WMC-DEV-STOP",
    "D3-GO",
    *GATED_ROWS,
    "INCONCLUSIVE",
)


def abandonment_fires(row: str) -> bool:
    if row not in ALL_ROWS:
        raise ContractError(f"unknown row {row!r}")
    return row in CLAUSE_ROWS


# ----- platform, devices, caps, void rule ---------------------------------------------------------
PLATFORM = fm.PLATFORM
SIM_WORKERS = 16
H_WORKERS = 8  # workers for arms that make world-model decisions
H_WORKER_TORCH_THREADS = 2
DEVICES = {
    "simulation": "cpu",
    "closed_loop_encoding": "cpu (batch size 1, in the worker)",
    "closed_loop_world_model": "cpu (in the worker)",
    "corpus_featurisation": "cuda, with a CPU anchor check (owner D6)",
    "training": "cuda, strict determinism (devices.require('cuda', strict=True))",
}
FEATURE_ANCHOR = {"frames": 256, "max_abs_pooled_difference": 1e-3}
CAPS_SECONDS = {
    "global": 43_200.0,
    "per_attempt": 300.0,
    "per_model": 5_400.0,
}
VOID_RULE = (
    "a guard, a crash, a cap or a CUDA allocation failure makes the stage V and nothing in it is "
    "read; batches are never shrunk to fit. A V before the first render of a gated (S or U) seed "
    "may be repeated after a reviewed fix; a V after it goes to the owner; a second V closes "
    "TASK-073 as INCONCLUSIVE. Development stages (K0, corpus, training, offline, D3) may be "
    "repeated once from scratch after a reviewed fix. Nothing is re-thresholded, retrained or "
    "re-selected after its numbers are seen"
)
OWNER_DECISIONS = {
    "D7": "Critic first, planner fallback",
    "D1_D2": "Accept, diagnostic only",
    "D3_D4_D6": "Use the defaults",
    "D5": "moot: PR #104 merged as 70f1358; every M2 number cited is checked against "
    "benchmarks/manifests/apple-first-policy-v2-m2-results.json",
    "date": "2026-09-29",
}


def frozen_block() -> dict:
    """Everything this module freezes, as plain JSON types; the manifest's ``frozen`` equals it."""
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "carried_from": [fp2.PROTOCOL, fm.PROTOCOL],
            "scene_version": SCENE_VERSION,
            "counted_success": COUNTED_SUCCESS,
            "expert": {"kwargs": EXPERT, "phases": EXPERT_PHASES, "budgets": EXPERT_BUDGETS},
            "pick_phases": PICK_PHASES,
            "transfer_start": TRANSFER_START,
            "steady_end": STEADY_END,
            "max_policy_steps": MAX_POLICY_STEPS,
            "settle_steps": SETTLE_STEPS,
            "carried": CARRIED,
            "evidence": EVIDENCE,
            "seed_ranges": SEED_RANGES,
            "smoke_seeds": SMOKE_SEEDS,
            "task_block": TASK_BLOCK,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "model_seeds": MODEL_SEEDS,
            "seeds": SEEDS,
            "reset": {"centers": RESET_CENTERS, "jitter_m": WIDE_JITTER_M},
            "shift": {
                "grid_cm": SHIFT_GRID_CM,
                "steps": SHIFT_STEPS,
                "base_xy": BASE_XY,
                "e9_release": E9_RELEASE,
                "arc_tolerance_m": ARC_TOLERANCE_M,
                "direction_grid_deg": DIRECTION_GRID_DEG,
                "tabletop": TABLETOP,
                "plate_apple_clearance_m": PLATE_APPLE_CLEARANCE_M,
            },
            "controller": {
                "decision_steps": DECISION_STEPS,
                "chunk": CHUNK,
                "candidate_offsets_m": CANDIDATE_OFFSETS_M,
                "incumbent_index": INCUMBENT_INDEX,
                "tie_tolerance_cm": TIE_TOLERANCE_CM,
                "o_star_cm": {str(k): v for k, v in O_STAR_CM.items()},
                "o_star_source": O_STAR_SOURCE,
            },
            "arms": ARMS,
            "l1_arms": L1_ARMS,
            "s_arms": S_ARMS,
            "u_arms": U_ARMS,
            "k0_arms": K0_ARMS,
            "d3_arms": D3_ARMS,
            "learned_label": LEARNED_LABEL,
            "k0": {"resets": K0_RESETS, "bars": K0_BARS, "h_sim_centre": K0_H_SIM_CENTRE},
            "corpus": {
                "name": CORPUS,
                "splits": CORPUS_SPLITS,
                "read_splits": READ_SPLITS,
                "shifted_fraction": SHIFTED_FRACTION,
                "misaim_fraction": MISAIM_FRACTION,
                "misaim_radius_m": MISAIM_RADIUS_M,
                "noise_levels": NOISE_LEVELS,
                "extra_episode_arrays": EXTRA_EPISODE_ARRAYS,
                "w_extra_corpus": W_EXTRA_CORPUS,
                "feature_band": FEATURE_BAND,
            },
            "w_model": W_MODEL,
            "budget": BUDGET,
            "readouts": READOUTS,
            "w_seed_selection": W_SEED_SELECTION,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "o1": O1,
            "o2": O2,
            "r_points": {str(k): v for k, v in R_POINTS.items()},
            "o3": O3,
            "o4": O4,
            "o0": O0,
            "o5": O5,
            "offline_rows": OFFLINE_ROWS,
            "d3": {"resets": D3_RESETS, "bars": D3_BARS},
            "gated": GATED,
            "gated_rows": GATED_ROWS,
            "clause_rows": CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "fallback_rows": FALLBACK_ROWS,
            "fallback": FALLBACK,
            "platform": PLATFORM,
            "sim_workers": SIM_WORKERS,
            "h_workers": H_WORKERS,
            "h_worker_torch_threads": H_WORKER_TORCH_THREADS,
            "devices": DEVICES,
            "feature_anchor": FEATURE_ANCHOR,
            "caps_seconds": CAPS_SECONDS,
            "void_rule": VOID_RULE,
            "owner_decisions": OWNER_DECISIONS,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()
