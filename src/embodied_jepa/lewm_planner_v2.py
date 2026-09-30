"""TASK-074 (``apple_lewm_planner_v2``): a LeWM-only planner for the place phase of
``apple-to-plate-v2`` after a far mid-episode plate shift.

Protocol ``docs/experiments/apple_lewm_planner_v2.md``. This module fixes in code, before any
cohort seed is simulated, what the protocol preregisters:
- the seeds and their guard, and the stored cohort values;
- the far-shift condition, its sign convention and its direction rule;
- the planner's decision steps, target grid, cost and tie rule;
- the K1 condition gate and the corpus plan;
- the training budget rule and the data-matched BC control P-far;
- the offline gates, the development stop rule (D3) and the gated rows on cohorts S and U;
- the determinism re-run rule (``run_guards.RerunRule``), the void rule (with
  ``first_outcome_utc``) and the abandonment clause.

NumPy only: it imports without torch or MuJoCo.

What is carried unchanged:
- the v2 scene, e9 and its schedule;
- the counted success (T71-R1/R2), the attempt length and the settle;
- the carried policy P-3 of TASK-072 run-1, with its readout evidence;
- TASK-066's LeWM token predictor configuration;
- TASK-073's direction rule (restricted to one side here) and its renderer rules.

**Learned Apple->Plate on the frozen benchmark is still 0 successes.** The plate shift is a
simulation-only diagnostic condition on v2, not a benchmark. An L-plan success would be
"P-3's learned pick, then a LeWM planner choosing where e9's scripted place primitive puts the
apple". It is not LeWM driving the whole episode, and it does not change the v1 headline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_m2 as fm
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_lewm_planner_v2"
TASK = "TASK-074"
GuardError = fp2.GuardError

# ----- carried unchanged --------------------------------------------------------------------------
SCENE_VERSION = fp2.SCENE_VERSION
COUNTED_SUCCESS = fp2.COUNTED_SUCCESS
counted_success = fp2.counted_success
EXPERT = fp2.EXPERT  # e9: RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)
EXPERT_PHASES = fp2.EXPERT_PHASES
EXPERT_BUDGETS = fp2.EXPERT_BUDGETS  # (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
PICK_PHASES = 4  # orient, descend, close, lift: built from the apple only (TASK-073, tested)
TRANSFER_START = int(sum(EXPERT_BUDGETS[:PICK_PHASES]))  # 405: the handover step
LOWER_START = int(sum(EXPERT_BUDGETS[: PICK_PHASES + 1]))  # 505: the target is frozen from here
PLACE_END = int(sum(EXPERT_BUDGETS))  # 725
MAX_POLICY_STEPS = fp2.MAX_POLICY_STEPS  # 740
SETTLE_STEPS = fp2.SETTLE_STEPS  # 60
CARRIED = fm.CARRIED  # P-3
EVIDENCE = fm.EVIDENCE  # TASK-072 run-1: report, P-3 checkpoint, corpus (by sha256)
P3_CHECKPOINT_SHA256 = EVIDENCE["checkpoints"][CARRIED]

# ----- the owner's rulings (2026-09-30, verbatim) and what was decided under delegation -----------
OWNER_DECISIONS = {
    "date": "2026-09-30",
    "D1": "Beat P-3, tie readout (Recommended)",
    "D2": "Put-down spots (Recommended)",
    "D3_D4": "Right-side move, place only (Recommended)",
    "D5": "Yes, add it (Recommended)",
    "meaning": {
        "D1": "beat P-3 even with the true plate; beat the action-blind, scene-blind and random "
        "twins; stay no worse than the readout arm (H-twin) within 6/64",
        "D2": "a global grid of put-down (release) targets, executed by e9's place phases",
        "D3_D4": "a -y (right-side) plate move at step 300, 9 or 12 cm chosen by K1; P-3 does the "
        "pick and LeWM plans the place",
        "D5": "the 300-root corpus plus a retrained BC policy (P-far) as a reported control",
    },
    "delegation": "the owner delegated every remaining ruling (2026-09-30); each such choice is "
    "the option the design would mark (Recommended) and is labelled DELEGATED",
}
DELEGATED = "decided by Claude under owner delegation, 2026-09-30"
DELEGATED_CHOICES = {
    "cost": "J(g) = || R_off(z_hat_{t+16}) - o_f ||, o_f = o*(501), the terminal release offset "
    "(not TASK-073's per-step o*(t+16), whose unshifted trajectory prior misleads under a 9-12 cm "
    "shift; TASK-073 K0's H-sim fell below P-truth at 6 cm with it)",
    "grid": "coarse 7 x 7 at 3 cm (dx -9..9, dy -15..3 cm around the post-look plate estimate) "
    "then fine 5 x 5 at 0.75 cm around the coarse choice: 74 candidates per decision (the "
    "design's 2 cm coarse grid did not cover the far shifts)",
    "decisions": "every 16 commands from the handover, 405-485 (6); frozen from 505 (lower)",
    "k1_bars": "B-oracle-shift >= 30, H-handover >= 30, P-stale <= 4, "
    "H-handover - P-truth >= +8 (of 32)",
    "gated_bars": "G1 vs P-truth, G2-G4 vs the blind twins at +10/64 and p < 0.01; T at 6/64; "
    "L-NO-HEADROOM at +16/64; G5 no-harm at 2/32; G6 latency 0.8 s",
    "p_far": "BC on the complete, non-mis-aimed corpus train roots plus 3 DAgger iterations "
    "under the condition, with the plate re-read by R-plate at the decision steps",
    "rerun": "L-plan and H-twin on the first four S seeds, compared by run_guards.RerunRule",
    "memory": "the train stage's memory is measured at full corpus scale in the smoke "
    "(stage-smoke 'train-scale'); the gated GO reads that measurement",
}

# ----- seeds --------------------------------------------------------------------------------------
# The repository search of 2026-09-30 (protocol §4) found no use of 54000-54999 before this task's
# design, and none of 7410-7424 as a seed constant. The design probes (protocol §2) used
# 54700-54999 and the direction salt 7401; they are spent and never used again.
SEED_RANGES = {
    "K1": (54000, 54031),  # the condition gate, development
    "R": (54040, 54071),  # offline ranking and regret; never fitted on
    "D3": (54100, 54115),  # development closed loop, stop rule only
    "corpus": (54200, 54499),  # apple-far-shift-v2, 240 / 30 / 30 by reset
    "S": (54500, 54563),  # gated, 64 shifted resets
    "U": (54600, 54631),  # gated no-harm, 32 unshifted resets
}
SMOKE_SEEDS = (54650, 54699)  # smokes only; nothing from them is read
DESIGN_PROBE_SEEDS = (54700, 54999)  # spent by the design probes (protocol §2)
DESIGN_PROBE_SALT = 7401
TASK_BLOCK = (54000, 54999)
COHORT_ROLES = ("K1", "R", "D3", "S", "U")
FORBIDDEN_RANGES = {
    **wc.FORBIDDEN_RANGES,
    "task073_block": wc.TASK_BLOCK,  # 53000-53999
    "task074_design_probes": DESIGN_PROBE_SEEDS,
}
MODEL_SEEDS = (7410, 7411, 7412)  # W and N, one model each per seed
SEEDS = {
    "shift_direction": 7413,
    "misaim": 7414,
    "corpus_split": 7415,
    "corpus_plan_salt": 7416,
    "l_rand": 7417,
    "random_controller": 7418,
    "bootstrap": 7419,
    "readout_folds": 7420,
    "sampler_salt": 7421,
    "wrong_actions": 7422,
    "comparative_rank": 7423,
    "p_far_sampler": 7424,
    "reset_redraw": 7425,
}


def seeds_of(role: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[role]
    return tuple(range(low, high + 1))


def smoke_seeds() -> tuple[int, ...]:
    return tuple(range(SMOKE_SEEDS[0], SMOKE_SEEDS[1] + 1))


def check_seed_ranges() -> None:
    """Every TASK-074 range lies in 54000-54699, the ranges are pairwise disjoint, and none
    overlaps a forbidden range (earlier tasks, TASK-073's block, the spent design probes)."""
    spans = dict(SEED_RANGES) | {"smoke": SMOKE_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("TASK-074 seed ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < TASK_BLOCK[0] or high > TASK_BLOCK[1]:
            raise GuardError(f"the TASK-074 range {name} leaves {TASK_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"the TASK-074 range {name} overlaps {other}")
    model_and_rng = (*MODEL_SEEDS, *SEEDS.values(), DESIGN_PROBE_SALT)
    if len(set(model_and_rng)) != len(model_and_rng):
        raise GuardError("TASK-074 model and RNG seeds must be distinct")


def check_role_seeds(role: str, seeds, *, smoke: bool = False) -> tuple[int, ...]:
    """G-seeds: plain ints, exactly the role's seeds in order (or smoke seeds only in a smoke).
    Every forbidden range, the design probes included, is refused first."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    if smoke:
        if not set(seeds) <= set(smoke_seeds()) or len(set(seeds)) != len(seeds):
            raise GuardError("G-seeds: a smoke simulates smoke seeds 54650-54699 only")
        return seeds
    if role not in SEED_RANGES:
        raise GuardError(f"G-seeds: unknown role {role!r}")
    if seeds != seeds_of(role):
        raise GuardError(f"G-seeds: {role} simulates exactly {SEED_RANGES[role]}, once, in order")
    return seeds


# ----- the condition: a right-side (-y) plate move at step 300 ------------------------------------
SIGN_CONVENTION = (
    "world frame = the fixed pelvis's base frame: +x points forward from the pelvis, +y to the "
    "robot's left, +z up. The right shoulder sits at y = -0.10 m (resting_expert."
    "RIGHT_SHOULDER_BASE) and the reset plate centre at y = -0.09 +- 0.02 m, so -y is the robot's "
    "right, away from the midline y = 0. 'Right-side move' (owner D3+D4) = a shift whose "
    "direction lies in 225-315 degrees (measured from +x towards +y), i.e. with a negative y "
    "component of at least |d| cos(45 deg)"
)
SHIFT_STEP = 300
SHIFT_GRID_CM = (9, 12)  # K1 tries 9 first, then 12
DIRECTION_ARC_DEG = (225, 315)  # inclusive, whole degrees: the -y half (design probe B)
RESET_CENTERS = wc.RESET_CENTERS
WIDE_JITTER_M = wc.WIDE_JITTER_M
wide_reset_values = wc.wide_reset_values


def eligible_right_directions(plate_xy, object_xy, magnitude_m: float) -> list[int]:
    """TASK-073's direction rule (e9's plate-relative release kept within 0.5 cm, the plate on the
    tabletop, the apple clearance kept), restricted to the -y arc of ``DIRECTION_ARC_DEG``."""
    low, high = DIRECTION_ARC_DEG
    return [d for d in wc.eligible_directions(plate_xy, object_xy, magnitude_m) if low <= d <= high]


def shift_vector(seed: int, reset: dict, magnitude_cm: float) -> list[float]:
    """The stored shift (world xy, m) of one reset at one |d|: one uniform draw per reset from
    ``default_rng(SeedSequence([7413, seed]))`` over the eligible -y directions."""
    eligible = eligible_right_directions(
        reset["plate_xy"], reset["object_xy"], magnitude_cm / 100.0
    )
    if not eligible:
        raise GuardError(f"no eligible -y shift direction for seed {seed} at {magnitude_cm} cm")
    u = np.random.default_rng(np.random.SeedSequence([SEEDS["shift_direction"], int(seed)]))
    pick = eligible[int(math.floor(float(u.uniform()) * len(eligible)))]
    theta = math.radians(pick)
    m = magnitude_cm / 100.0
    return [m * math.cos(theta), m * math.sin(theta)]


REDRAW_CM = (3, 6, 9, 12)  # every |d| any stage uses (K1's grid and the corpus's)
REDRAW_LIMIT = 50
REDRAW_RULE = (
    "the reset of every TASK-074 seed (every role, the corpus and U included, so all cohorts share "
    "one reset distribution) is TASK-047's wide-jitter reset of the seed if the plate can move to "
    "the right (-y rule) by every |d| in (3, 6, 9, 12) cm; otherwise the same distribution is "
    "re-drawn from default_rng(SeedSequence([seed, 7425, k])) for k = 1, 2, ... until it can "
    "(about 1-2 % of seeds: the apple's reset position blocks every -y direction); 'redraw' "
    "stores k (0 = the plain reset). " + DELEGATED
)


def _wide_reset_from(rng) -> dict:
    obj = np.array(RESET_CENTERS["object_xy"]) + rng.uniform(
        -WIDE_JITTER_M["object_xy"], WIDE_JITTER_M["object_xy"], 2
    )
    plate = np.array(RESET_CENTERS["plate_xy"]) + rng.uniform(
        -WIDE_JITTER_M["plate_xy"], WIDE_JITTER_M["plate_xy"], 2
    )
    return {"object_xy": obj.tolist(), "plate_xy": plate.tolist()}


def _right_eligible(reset: dict) -> bool:
    return all(
        eligible_right_directions(reset["plate_xy"], reset["object_xy"], cm / 100.0)
        for cm in REDRAW_CM
    )


def condition_reset(seed: int) -> dict:
    """The reset of a TASK-074 seed under ``REDRAW_RULE``; ``redraw`` records k."""
    reset = wide_reset_values(seed)
    k = 0
    while not _right_eligible(reset):
        k += 1
        if k > REDRAW_LIMIT:
            raise GuardError(f"seed {seed}: no right-eligible reset in {REDRAW_LIMIT} re-draws")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), SEEDS["reset_redraw"], k]))
        reset = _wide_reset_from(rng)
    return reset | {"redraw": k}


def cohort_values(role: str) -> dict[str, dict]:
    """The values stored in the manifest for one cohort: per seed its reset (``condition_reset``)
    and, for shifted roles, its shift vector at every |d| of the grid (cohort U stores none)."""
    out = {}
    for seed in seeds_of(role):
        reset = condition_reset(seed)
        entry = {"seed": seed, **reset}
        if role != "U":
            entry["shift_m"] = {str(m): shift_vector(seed, reset, m) for m in SHIFT_GRID_CM}
        out[str(seed)] = entry
    return out


def cohort_digest(values: dict) -> str:
    return fm.cohort_digest(values)


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
        want = {"seed", "object_xy", "plate_xy", "redraw"} | ({"shift_m"} if role != "U" else set())
        if entry["seed"] != seed or set(entry) != want:
            raise GuardError(f"G-cohort: stored {role} entry {seed} is malformed")
        out[seed] = copy.deepcopy(entry)
    return out


# ----- the planner: decisions, target grid, cost, tie rule ---------------------------------------
DECISION_STEPS = tuple(range(TRANSFER_START, LOWER_START - 16, 16))  # 405, ..., 485 (6)
CHUNK = 16  # commands per decision; W's horizon (TASK-066's validated h = 16)
COARSE_DX_CM = (-9, -6, -3, 0, 3, 6, 9)
COARSE_DY_CM = (-15, -12, -9, -6, -3, 0, 3)
COARSE_OFFSETS_M = tuple((dx / 100.0, dy / 100.0) for dx in COARSE_DX_CM for dy in COARSE_DY_CM)
COARSE_INCUMBENT = COARSE_OFFSETS_M.index((0.0, 0.0))  # the post-look plate estimate itself
FINE_STEPS_CM = (-1.5, -0.75, 0.0, 0.75, 1.5)
FINE_OFFSETS_M = tuple((dx / 100.0, dy / 100.0) for dx in FINE_STEPS_CM for dy in FINE_STEPS_CM)
FINE_CENTRE = FINE_OFFSETS_M.index((0.0, 0.0))  # the coarse choice itself
CANDIDATES_PER_DECISION = len(COARSE_OFFSETS_M) + len(FINE_OFFSETS_M)  # 74
TIE_TOLERANCE_CM = 1e-6
# o*(t): the median apple-minus-plate offset (cm, world xy) at post-look step t over the 112
# counted-success train roots of apple-look-v2-linux run-1 (e9, unshifted), TASK-073's source and
# statistic, recomputed at this protocol's steps (it reproduces TASK-073's frozen values at 416,
# 432, ..., 592 exactly). Used by O2's clock-prior baseline. The cost uses its terminal value.
O_STAR_CM = {
    421: (-11.3019, -5.2637),
    437: (-7.1518, -1.1956),
    453: (-3.0801, -0.1662),
    469: (-0.1101, -0.1824),
    485: (0.387, -0.1358),
    501: (0.5731, -0.1121),
}
O_FINAL_STEP = 501
O_FINAL_CM = O_STAR_CM[O_FINAL_STEP]
O_STAR_SOURCE = wc.O_STAR_SOURCE
COST = (
    "J(g) = || R_off(z_hat_{t+16}(g)) - o_f || in cm: R_off (ridge, pooled latent -> apple-minus-"
    "plate offset) read on W's h = 16 prediction for the chunk of target g, against o_f = o*(501) "
    "= (0.5731, -0.1121) cm, e9's median release offset. " + DELEGATED
)


def cost_targets() -> dict[int, tuple]:
    """The critic's target per target step (t + 16 for every decision t): o_f everywhere."""
    return {int(t) + CHUNK: O_FINAL_CM for t in DECISION_STEPS}


def coarse_targets(anchor_xy) -> np.ndarray:
    return np.asarray(anchor_xy, np.float64).reshape(1, 2) + np.asarray(COARSE_OFFSETS_M)


def fine_targets(coarse_xy) -> np.ndarray:
    return np.asarray(coarse_xy, np.float64).reshape(1, 2) + np.asarray(FINE_OFFSETS_M)


def choose(costs_cm, incumbent: int) -> int:
    """The argmin; ``incumbent`` wins every tie within TIE_TOLERANCE_CM and any non-finite cost
    loses. With an action-blind critic every cost is equal, so the choice is the incumbent: the
    coarse (0, 0), which is the post-look (stale) plate estimate, then the fine centre."""
    costs = np.asarray(costs_cm, np.float64).reshape(-1)
    costs = np.where(np.isfinite(costs), costs, np.inf)
    best = int(np.argmin(costs))
    if not np.isfinite(costs[incumbent]):
        return best if np.isfinite(costs[best]) else incumbent
    if costs[best] < costs[incumbent] - TIE_TOLERANCE_CM:
        return best
    return incumbent


def terminal_cost_cm(offset_cm) -> np.ndarray:
    return np.linalg.norm(np.asarray(offset_cm, np.float64) - np.asarray(O_FINAL_CM), axis=-1)


SHIFT_BLOCKED_PREFIX = "the shifted plate touches"
SHIFT_BLOCKED = (
    "if the move at step 300 would leave the plate touching anything other than the table "
    "(plate_shift's guard: the apple not yet lifted clear, or the hand), the plate is not moved; "
    "an evaluated attempt then ends as a counted failure (termination 'shift_blocked'), not a V; "
    "a corpus root is collected again without the move and flagged 'shift_blocked'. Every "
    "blocked move is reported per arm. " + DELEGATED
)
FALLBACK = {
    "decision_wall_seconds": 0.8,
    "rule": "if every candidate is infeasible (stand-in refusal or an unreachable release pose), "
    "the planner keeps its previous target (at 405: the coarse (0, 0), the post-look plate "
    "estimate); a decision is never skipped for time (G6 measures latency instead); every "
    "fallback is logged; no fallback to P-3 after the handover",
}

# ----- arms ---------------------------------------------------------------------------------------
ARMS = {
    "L-plan": ("L1", "P-3 pick; from 405 W (val-selected seed) chooses e9's place target"),
    "L-plan-s1": ("L1", "L-plan with the second W seed (secondary: sign agreement)"),
    "L-plan-s2": ("L1", "L-plan with the third W seed (secondary: sign agreement)"),
    "L-N": ("L1", "as L-plan with the no-action model N (action-blind; ties keep the incumbent)"),
    "L-shuf": ("L1", "as L-plan from H-twin's latent of reset (i + 1) mod n (scene-blind)"),
    "L-rand": ("L1", "a seeded uniform coarse then fine choice among the same 74 targets"),
    "H-twin": ("L1", "the perception twin: e9's place aimed at R-plate's reading of the frame"),
    "P-stale": ("L1", "P-3 unchanged: the post-look estimates for the whole attempt"),
    "P-far": (
        "L1",
        "reported control: P-3's recipe retrained on apple-far-shift-v2 (BC + 3 "
        "DAgger), plate re-read by R-plate at the decision steps",
    ),
    "P-truth": ("L4", "P-3 given the true post-shift plate at the shift step"),
    "H-handover": ("L4", "P-3 pick, then e9's place aimed at the true plate: the family ceiling"),
    "B-oracle-shift": ("L4", "e9 built from the post-shift truth (harness)"),
    "B-hold": ("L4", "hold (harness)"),
    "B-random": ("L4", "uniform random within the configured bounds (harness)"),
}
L1_ARMS = tuple(a for a, (rung, _) in ARMS.items() if rung == "L1")
S_ARMS = tuple(ARMS)
U_ARMS = ("P-stale", "H-twin", "L-plan")
K1_ARMS = ("B-oracle-shift", "H-handover", "P-stale", "P-truth")  # K1's order of evaluation
D3_ARMS = (
    "H-twin",
    "L-plan",
    "L-N",
    "L-shuf",
    "L-rand",
    "P-stale",
    "P-truth",
    "H-handover",
    "B-oracle-shift",
    "P-far",
)
CRITIC_ARMS = {"L-plan": "W0", "L-plan-s1": "W1", "L-plan-s2": "W2", "L-N": "N", "L-shuf": "W0"}
IMAGE_READING_ARMS = ("L-plan", "L-plan-s1", "L-plan-s2", "L-N", "L-shuf", "H-twin", "P-far")
LEARNED_LABEL = (
    "P-3's learned pick (BC/DAgger on privileged e9 labels) followed by a LeWM planner that "
    "chooses where e9's scripted place primitive puts the apple; not LeWM driving the whole "
    "episode, not a real-world disturbance, not the frozen v1 benchmark"
)

# ----- K1: the condition gate (simulator only, no world model) ------------------------------------
K1_RESETS = 32
K1_BARS = {  # in K1_ARMS order for each |d|; the first failed bar ends that |d|
    "B-oracle-shift_min": 30,
    "H-handover_min": 30,
    "P-stale_max": 4,
    "H-handover_minus_P-truth_min": 8,
}


def _k1_bar(arm: str, counts: dict) -> bool:
    if arm == "B-oracle-shift":
        return counts[arm] >= K1_BARS["B-oracle-shift_min"]
    if arm == "H-handover":
        return counts[arm] >= K1_BARS["H-handover_min"]
    if arm == "P-stale":
        return counts[arm] <= K1_BARS["P-stale_max"]
    return counts["H-handover"] - counts["P-truth"] >= K1_BARS["H-handover_minus_P-truth_min"]


def k1_next_arm(counts: dict) -> str | None:
    """The next arm K1 runs in a cell, or None once the cell is decided."""
    for arm in K1_ARMS:
        if counts.get(arm) is None:
            return arm
        if not _k1_bar(arm, counts):
            return None
    return None


def k1_cell_passes(counts: dict) -> dict:
    for arm in K1_ARMS:
        if counts.get(arm) is None or not _k1_bar(arm, counts):
            return {"passes": False, "failed_bar": arm}
    return {"passes": True, "failed_bar": None}


def k1_select(cells: dict) -> dict:
    """``cells[cm]``: the counts of the arms that ran at |d| = cm. The smallest |d| passing every
    bar is chosen; if none, L-NO-CONDITION."""
    for cm in SHIFT_GRID_CM:
        if cm in cells and k1_cell_passes(cells[cm])["passes"]:
            return {"row": "K1-PASS", "shift_step": SHIFT_STEP, "shift_cm": cm}
    return {"row": "L-NO-CONDITION", "shift_step": None, "shift_cm": None}


# ----- the corpus: apple-far-shift-v2 ------------------------------------------------------------
CORPUS = "apple-far-shift-v2"
CORPUS_SPLITS = {"train": 240, "val": 30, "test": 30}
READ_SPLITS = ("train", "val")  # the test split is never decoded
UNSHIFTED_FRACTION = 0.25
CORPUS_SHIFT_CM = (3, 6, 9, 12)  # uniform over the shifted roots, -y rule; independent of K1
MISAIM_FRACTION = 0.50
MISAIM_RADIUS_M = 0.04
NOISE_LEVELS = fp2.NOISE_LEVELS  # TASK-048 Perturber, root i gets level i % 4
EXTRA_EPISODE_ARRAYS = wc.EXTRA_EPISODE_ARRAYS  # ("plate",)
W_EXTRA_CORPUS = wc.W_EXTRA_CORPUS  # apple-look-v2-linux run-1's 190 train + val roots
FEATURE_BAND = (384, 560)  # every decision window (405-501) and O1/O2 window lies inside


def corpus_plan() -> list[dict]:
    """One root per corpus seed, fixed before collection: split, shift (|d| and the -y vector),
    mis-aim flag and vector, noise level and seed. Collected by TASK-073's e9 collector
    (``wm_critic_v2_runtime.run_collect_task``): e9 from the post-shift truth plus the mis-aim."""
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
    unshifted = set(rng.permutation(seeds)[: round(UNSHIFTED_FRACTION * len(seeds))].tolist())
    misaimed = set(rng.permutation(seeds)[: round(MISAIM_FRACTION * len(seeds))].tolist())
    plan = []
    for index, seed in enumerate(seeds):
        r = np.random.default_rng(np.random.SeedSequence([int(seed), SEEDS["corpus_plan_salt"]]))
        noise_seed = int(r.integers(2**31))
        radius = MISAIM_RADIUS_M * math.sqrt(float(r.uniform()))
        angle = 2 * math.pi * float(r.uniform())
        cm = CORPUS_SHIFT_CM[int(r.integers(len(CORPUS_SHIFT_CM)))]
        reset = condition_reset(seed)
        shift = None
        if seed not in unshifted:
            shift = {"step": SHIFT_STEP, "vector": shift_vector(seed, reset, cm), "cm": cm}
        plan.append(
            {
                "episode_id": f"farshift2-{seed}",
                "session_id": f"farshift2-reset-{seed}",
                "seed": int(seed),
                "split": split[seed],
                "reset": reset,
                "shift": shift,
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
W_TOKEN_GRID = wc.W_TOKEN_GRID  # 4
W_MODEL = wc.W_MODEL  # TASK-066's configuration, the LeWM backend, batch 64; N: zero actions
BUDGET = {
    "calibration_runs": (("W", MODEL_SEEDS[0]), ("N", MODEL_SEEDS[0])),
    "calibration_updates": 60_000,
    "calibration_select_every": 1_000,
    "saturation_tolerance": 0.01,
    "factor": 2.0,
    "step": 5_000,
    "min": 10_000,
    "cap": 60_000,
    "selection_points": 20,
    "last_two_rule": wc.BUDGET["last_two_rule"],
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
    "train-standardised) to the apple-minus-plate offset xy; rows: apple-far-shift-v2 train "
    "roots, every 8th band frame from 384",
    "R_plate": "linear ridge (dual, intercept) from the full DINOv2 tokens (256 x 384, the P "
    "readout's feature, train-standardised) to the plate xy; rows: apple-far-shift-v2 train roots "
    "at the decision steps; the perception twin's readout and P-far's re-read",
    "lambda_grid_relative": (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    "inner_folds": 5,
    "fold_seed": SEEDS["readout_folds"],
    "fold_unit": "root (session)",
    "r_off_stride": 8,
}
W_SEED_SELECTION = wc.W_SEED_SELECTION

# ----- P-far: the data-matched BC control (owner D5; reported, not gating) ------------------------
P_FAR = {
    "recipe": "first_policy_v2_model.train with first_policy_v2.TRAINING (P-3's network, "
    "optimiser, loss, selection and model-init seed); the sampler seed is SEEDS['p_far_sampler']",
    "bc_roots": "apple-far-shift-v2 train roots that are complete and not mis-aimed",
    "labels": "e9's clean command (the corpus 'base' array), first_policy_v2_runtime.bc_rows' mask",
    "inputs": "the 132-d input; apple = TASK-072 run-1's P readout of the root's post-look frame; "
    "plate = that readout's post-look plate before 405, then the true plate at the latest "
    "decision step (BC rows) or R-plate's reading there (DAgger rows, as run)",
    "dagger_iterations": 3,
    "dagger_rollouts": (40, 50, 60),  # corpus train seeds, plan order, disjoint slices
    "dagger_labeller": "e9 built from the post-shift truth, advanced on the executed results "
    "(TASK-072's shadow expert); privileged, training time only",
    "run_time": "P-3's controller with the plate re-read by R-plate at the decision steps "
    "(hybrid_selection.HybridController, variant 'reread')",
    "device": "cuda",
    "label": "a data-matched BC control, reported beside L-plan; it gates nothing",
}

# ----- offline gates (every gate on all three W seeds) --------------------------------------------
BOOTSTRAP_RESAMPLES = wc.BOOTSTRAP_RESAMPLES
O1 = dict(wc.O1) | {"window_sets": ("overall", "shifted")}
O2 = {
    "windows": "every val root of apple-far-shift-v2; windows starting at the decision steps "
    "whose true apple-minus-plate offset moves by at least 1 cm over the 16 commands (TASK-054's "
    "moving cohort)",
    "moving_threshold_m": 0.01,
    "encoded_median_max_cm": 1.0,
    "predicted_median_max_cm": 1.5,
    "ratio_vs_persistence_upper_max": 0.8,  # (i): TASK-054's G2a bar
    "n_ratio_vs_persistence_upper_min": 0.8,  # (ii): N must fail (i), else the gate is void
    "ratio_vs_clock_prior_upper_max": 0.8,  # (iii): W must beat the o*(t+16)-only predictor
}
O3 = {
    "groups": "at every decision step of each cohort-R attempt (H-twin runs the attempt): the 49 "
    "coarse targets, and the 25 fine targets around the coarse target with the lowest true cost",
    "rho_lower_min": 0.5,
    "margin_lower_min": 0.3,
    "blind_rankers": ("copy-last", "N", "L-shuf", "prior-distance"),
    "void_if_blind_rho_at_least": 0.5,
    "void_rankers": ("copy-last", "N", "L-shuf"),
    "reported_rankers": ("twin-distance",),  # the readout as a ranker: reported, not a bar (D1)
}
O4 = {
    "median_regret_max_cm": 1.0,
    "paired_regret_difference_upper_lt": 0.0,
    "regret": "the true cost of the chosen target (coarse then fine, as in the closed loop) minus "
    "the best true cost among the 74 candidates; paired against the incumbent's (the coarse "
    "(0, 0), the stale estimate)",
}
O5 = {"median_relative_chunk_error_max": 0.25}
OFFLINE_ROWS = (
    "OFFLINE-PASS",
    "L-NO-DYNAMICS",
    "L-O2-VOID",
    "L-G2A",
    "L-O3-VOID",
    "L-NO-RANK",
    "L-PROPOSAL",
)
spearman = wc.spearman


def cluster_median_ci(values, clusters, *, resamples=None) -> dict:
    return wc.cluster_median_ci(values, clusters, seed=SEEDS["bootstrap"], resamples=resamples)


def cluster_median_ratio(num, den, clusters, *, resamples=None) -> dict:
    return wc.cluster_median_ratio(num, den, clusters, seed=SEEDS["bootstrap"], resamples=resamples)


def o1_seed_passes(per: dict) -> dict:
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
    t = O3
    void = any(
        m["blind"][r]["median"] >= t["void_if_blind_rho_at_least"] for r in t["void_rankers"]
    )
    passes = m["rho_w"]["ci95"][0] >= t["rho_lower_min"] and (
        m["margin"]["ci95"][0] >= t["margin_lower_min"]
    )
    return {"passes": bool(passes and not void), "void": bool(void)}


def o4_passes(m: dict) -> dict:
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


def decide_offline(seeds: dict, o5: dict) -> dict:
    """First match over O1, O2, O3, O4, O5; every gate must pass on all three W seeds."""
    if set(seeds) != set(MODEL_SEEDS):
        raise ContractError(f"the offline decision needs every W seed {MODEL_SEEDS}")
    if not all(seeds[s]["O1"]["passes"] for s in MODEL_SEEDS):
        row = "L-NO-DYNAMICS"
    elif any(seeds[s]["O2"]["void"] for s in MODEL_SEEDS):
        row = "L-O2-VOID"
    elif not all(seeds[s]["O2"]["passes"] for s in MODEL_SEEDS):
        row = "L-G2A"
    elif any(seeds[s]["O3"]["void"] for s in MODEL_SEEDS):
        row = "L-O3-VOID"
    elif not all(seeds[s]["O3"]["passes"] and seeds[s]["O4"]["passes"] for s in MODEL_SEEDS):
        row = "L-NO-RANK"
    elif not o5["passes"]:
        row = "L-PROPOSAL"
    else:
        row = "OFFLINE-PASS"
    return {"row": row, "abandonment_clause_fires": abandonment_fires(row)}


# ----- D3: the development closed loop (non-gating) -----------------------------------------------
D3_RESETS = 16
D3_BARS = {"H-handover_minus_P-truth_min": 4, "L-plan_minus_best_blind_min": 3}


def decide_d3(counts: dict) -> dict:
    if counts["H-handover"] - counts["P-truth"] < D3_BARS["H-handover_minus_P-truth_min"]:
        row = "L-NO-HEADROOM"
    elif (
        counts["L-plan"] - max(counts["L-N"], counts["L-shuf"], counts["L-rand"])
        < D3_BARS["L-plan_minus_best_blind_min"]
    ):
        row = "L-DEV-STOP"
    else:
        row = "D3-GO"
    return {"row": row}


# ----- the gated rows on S (64) and U (32) --------------------------------------------------------
S_RESETS = 64
U_RESETS = 32
GATED = {
    "void_b_oracle_shift_min": 60,
    "s_void_condition_p_stale_max": 8,
    "l_no_headroom_min": 16,  # H-handover - P-truth
    "difference_min": 10,
    "p_max_exclusive": 0.01,
    "g5_u_tolerance": 2,
    "g6_median_decision_seconds_max": 0.8,
    "t_twin_margin": 6,  # owner D1: L-plan >= H-twin - 6 of 64
    "g6_arm": "L-plan",
    "g6_statistic": "median over every decision record of L-plan's 64 S attempts (the wall time "
    "inside act(): encoding, 74 stand-in chunks, the W rollouts and the choice), in the gated "
    "run's H workers",
    "determinism_rerun_seeds": (54500, 54501, 54502, 54503),
    "determinism_rerun_arms": ("L-plan", "H-twin"),
}
# The determinism re-run (run_guards.RerunRule). The planner's candidates are anchored on the
# post-look estimate, which a rare one-level render difference (TASK-073 §7) can move, and
# H-twin reads every decision frame. So both re-run arms are image-reading. Every compared field
# is exact or toleranced, with a command tolerance (the #111 review). TASK-073's thin margin
# (0.87 cm measured against its 1.0 cm incumbent tolerance, H-rand) is handled by construction:
# L-rand's targets depend on no decision-time image, so it is not a re-run arm. The tolerances
# are at least twice the largest difference measured between re-runs on smoke seeds
# (smoke 'rerun'), and the GO reports that margin.
RERUN_FIELDS = (
    "success",
    "grasp",
    "at_rest",
    "termination_reason",
    "executed_steps",
    "final_distance_cm",
    "first_grasp_step",
    "first_place_step",
    "target_cm_max",
)
RERUN_EXACT = (
    "success",
    "grasp",
    "at_rest",
    "termination_reason",
    "executed_steps",
    "chosen_sequence",
)
RERUN_TOLERANCES = {
    "final_distance_cm": 1.0,
    "first_grasp_step": 40,
    "first_place_step": 40,
    "target_cm_max": 1.0,  # the largest per-decision target difference between the two runs
    "max_abs_command_difference": 0.8,
}
RERUN_MARGIN_RULE = (
    "each tolerance is at least twice the largest difference of that field measured between two "
    "runs of the same arm and smoke seed (smoke stage 'rerun'); a gated re-run difference above "
    "half a tolerance is reported in the results; a difference above it is a V"
)


def rerun_rule():
    """The frozen image-arm rule, built and validated by ``run_guards.RerunRule``."""
    from embodied_jepa.run_guards import RerunRule

    return RerunRule(
        image_reading=True,
        exact=RERUN_EXACT,
        tolerances=RERUN_TOLERANCES,
        fields=RERUN_FIELDS,
    )


GATED_ROWS = (
    "VOID",
    "S-VOID-CONDITION",
    "L-NO-HEADROOM",
    "L-HARM",
    "L-PASS",
    "L-PASS-TWIN-BETTER",
    "L-SLOW",
    "L-SCENE-BLIND",
    "L-NO-GAIN",
)
AUTHORISATION_SCOPE = (
    "the gated stage's authorisation PR may add the gated harness (the runner's stage_gated and "
    "its tests), the authorisation record and the manifest's gated_authorization block; it may "
    "not change lewm_planner_v2.py, the frozen block, frozen_sha256 or the stored cohort values"
)
paired_one_sided = wc.paired_one_sided


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
        other: paired_one_sided(s["L-plan"], s[other])
        for other in ("P-truth", "L-shuf", "L-N", "L-rand", "H-twin", "P-far", "P-stale")
    }

    def diff_gate(other):
        p = pairs[other]
        return p["difference"] >= t["difference_min"] and p["p_one_sided"] < t["p_max_exclusive"]

    lat = harness["median_decision_seconds"]
    gates = {
        "G1": diff_gate("P-truth"),
        "G2": diff_gate("L-shuf"),
        "G3": diff_gate("L-N"),
        "G4": diff_gate("L-rand"),
        "G5": int(np.asarray(u["L-plan"], bool).sum())
        >= int(np.asarray(u["P-stale"], bool).sum()) - t["g5_u_tolerance"],
        "G6": lat is not None
        and bool(np.isfinite(lat))
        and lat <= t["g6_median_decision_seconds_max"],
    }
    twin_ok = pairs["H-twin"]["difference"] >= -t["t_twin_margin"]
    outcome = all(gates[g] for g in ("G1", "G2", "G3", "G4", "G5"))
    if void:
        row = "VOID"
    elif n["P-stale"] > t["s_void_condition_p_stale_max"]:
        row = "S-VOID-CONDITION"
    elif n["H-handover"] - n["P-truth"] < t["l_no_headroom_min"]:
        row = "L-NO-HEADROOM"
    elif not gates["G5"]:
        row = "L-HARM"
    elif outcome and gates["G6"] and twin_ok:
        row = "L-PASS"
    elif outcome and gates["G6"]:
        row = "L-PASS-TWIN-BETTER"
    elif outcome:
        row = "L-SLOW"
    elif gates["G1"] and gates["G3"] and not gates["G2"]:
        row = "L-SCENE-BLIND"
    else:
        row = "L-NO-GAIN"
    secondary = {
        seed_arm: {
            other: int(np.asarray(s[seed_arm], bool).sum() - np.asarray(s[other], bool).sum())
            for other in ("P-truth", "L-shuf", "L-N")
        }
        for seed_arm in ("L-plan", "L-plan-s1", "L-plan-s2")
    }
    signs = {
        other: len({int(np.sign(secondary[a][other])) for a in secondary}) == 1
        for other in ("P-truth", "L-shuf", "L-N")
    }
    return {
        "row": row,
        "gates": gates,
        "twin_non_inferior": bool(twin_ok),
        "counts_of_64": n,
        "u_counts_of_32": {a: int(np.asarray(u[a], bool).sum()) for a in U_ARMS},
        "paired": pairs,
        "p_far_reported": pairs["P-far"],
        "secondary_sign_agreement": {"differences": secondary, "same_sign": signs},
        "abandonment_clause_fires": abandonment_fires(row),
    }


SHUF_RULE = (
    "L-shuf's start latent at decision step t on reset i is H-twin's pooled latent at t on reset "
    "(i + 1) mod n (cohort order); if that attempt ended earlier, its latest latent before t; if "
    "it reached no decision, the next reset in order that has one; never the reset's own "
    "(wm_critic_v2.shuf_latents with this protocol's decision steps); every substitution logged"
)


# ----- abandonment clause, rows and consequences --------------------------------------------------
CLAUSE_ROWS = ("L-G2A", "L-NO-RANK", "L-HARM", "L-NO-GAIN")
CLAUSE_SCOPE = (
    "a LeWM frozen-DINOv2-token planner choosing the place target of e9's place primitive after "
    "P-3's pick, on apple-to-plate-v2 at 112 px: no further grid, cost, horizon or proposal "
    "variant is preregistered on apple-far-shift-v2 without new evidence of a different kind. "
    "The LeWM backend, the token latent, the v2 task and the product goal stay open; the next "
    "step is the task owner's choice of a data or hardware change, not another planner variant"
)
ALL_ROWS = (
    "K1-PASS",
    "L-NO-CONDITION",
    *OFFLINE_ROWS,
    "L-DEV-STOP",
    "D3-GO",
    *GATED_ROWS,
    "INCONCLUSIVE",
)
ROW_CONSEQUENCES = {
    "K1-PASS": "next: the corpus",
    "L-NO-CONDITION": "escalate: no far-shift condition leaves room over P-3; nothing is trained",
    "OFFLINE-PASS": "next: D3",
    "L-NO-DYNAMICS": "escalate: the world model fails the dynamics gates; no closed loop runs",
    "L-O2-VOID": "escalate",
    "L-G2A": "clause",
    "L-O3-VOID": "escalate",
    "L-NO-RANK": "clause",
    "L-PROPOSAL": "escalate: the stand-in does not reproduce the place primitive's chunks",
    "L-NO-HEADROOM": "escalate: the family ceiling does not clear P-truth; no clause",
    "L-DEV-STOP": "escalate: the gated stage is not proposed",
    "D3-GO": "next: the gated stage's authorisation record",
    "VOID": "escalate (the void rule)",
    "S-VOID-CONDITION": "escalate: the condition did not hold on S",
    "L-HARM": "clause",
    "L-PASS": "close: report the claim with its labels",
    "L-PASS-TWIN-BETTER": "close: the claim with the label that the readout twin beat LeWM "
    "by more than the margin",
    "L-SLOW": "close: outcome gates pass, latency fails; no clause",
    "L-SCENE-BLIND": "close: no clause; reported as a scene-blind gain",
    "L-NO-GAIN": "clause",
    "INCONCLUSIVE": "close",
    "ESCALATE-BUDGET": "escalate: no freeze of the training budget",
    "ESCALATE-BUDGET-LAST-TWO": "escalate: no freeze of the training budget",
}


def abandonment_fires(row: str) -> bool:
    if row not in ALL_ROWS:
        raise ContractError(f"unknown row {row!r}")
    return row in CLAUSE_ROWS


# ----- platform, devices, caps, memory, void rule -------------------------------------------------
PLATFORM = fm.PLATFORM
SIM_WORKERS = 6
H_WORKERS = 4  # workers for arms that make world-model decisions (and the ranking stage)
H_WORKER_TORCH_THREADS = 4
DEVICES = wc.DEVICES | {"p_far_training": "cuda"}
RENDER = wc.RENDER
THREAD_ENV = wc.THREAD_ENV
QUIET_MACHINE = wc.QUIET_MACHINE
FEATURE_ANCHOR = wc.FEATURE_ANCHOR
MEMORY = {
    "ceiling_gib": 12.0,
    "headroom_gib": 4.0,
    "sample_seconds": 0.5,
    "measure": "summed PSS of the runner and all its descendants (run_guards."
    "process_tree_memory: smaps_rollup, falling back to VmRSS for a process whose PSS cannot be "
    "read, counted in pss_fallback_processes), sampled every 0.5 s; summed VmRSS beside it",
    "measure_key": "pss",
    "train_scale_rule": "the smoke's 'train-scale' stage builds the train stage's feature table "
    "at full corpus scale (270 + 190 read roots x the band) and runs the train stage's code "
    "path; its peak must be within the ceiling before any GO",
}
CAPS_SECONDS = {"global": 43_200.0, "per_attempt": 300.0, "per_model": 5_400.0}
VOID_RULE = (
    "a guard, a crash, a cap or a CUDA allocation failure makes the stage V and nothing in it is "
    "read; batches are never shrunk to fit. A V before a stage's boundary is not a spent "
    "attempt: it may be repeated as-is, recorded, without a fix. The boundary is "
    "cohort_first_render_utc for a simulating stage (K1, corpus, ranking, D3, P-far, gated) and "
    "first_outcome_utc for the train stage (the first number computed on val roots). After "
    "the boundary: a gated (S or U) V goes to the owner and a second V closes TASK-074 as "
    "INCONCLUSIVE; a development stage (K1, corpus, training, P-far, ranking, D3) may be "
    "repeated once from scratch after a reviewed fix. Nothing is re-thresholded, retrained or "
    "re-selected after its numbers are seen"
)
DESIGN_PROBES = {
    "seeds": DESIGN_PROBE_SEEDS,
    "direction_salt": DESIGN_PROBE_SALT,
    "disclosure": "the K1, D3 and gated bars and the -y restriction were set after the design "
    "probes (protocol §2) had been seen: development seeds 54700-54999, now spent. On them, "
    "after a -y shift at step 300: e9 32/32 at 9 and 12 cm; P-3 given the true plate 20/32 and "
    "7/32; P-3 pick then e9 place to the true plate 32/32 and 32/32; the same aimed at a direct "
    "plate readout trained with far data 31/32 and 30/32. One run of 32 resets per cell, no "
    "interval",
}


def frozen_block() -> dict:
    """Everything this module freezes, as plain JSON types; the manifest's ``frozen`` equals it."""
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "carried_from": [fp2.PROTOCOL, fm.PROTOCOL, wc.PROTOCOL],
            "owner_decisions": OWNER_DECISIONS,
            "delegated": DELEGATED,
            "delegated_choices": DELEGATED_CHOICES,
            "scene_version": SCENE_VERSION,
            "counted_success": COUNTED_SUCCESS,
            "expert": {"kwargs": EXPERT, "phases": EXPERT_PHASES, "budgets": EXPERT_BUDGETS},
            "pick_phases": PICK_PHASES,
            "transfer_start": TRANSFER_START,
            "lower_start": LOWER_START,
            "place_end": PLACE_END,
            "max_policy_steps": MAX_POLICY_STEPS,
            "settle_steps": SETTLE_STEPS,
            "carried": CARRIED,
            "evidence": EVIDENCE,
            "seed_ranges": SEED_RANGES,
            "smoke_seeds": SMOKE_SEEDS,
            "design_probes": DESIGN_PROBES,
            "task_block": TASK_BLOCK,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "model_seeds": MODEL_SEEDS,
            "seeds": SEEDS,
            "reset": {
                "centers": RESET_CENTERS,
                "jitter_m": WIDE_JITTER_M,
                "redraw_cm": REDRAW_CM,
                "redraw_rule": REDRAW_RULE,
            },
            "condition": {
                "sign_convention": SIGN_CONVENTION,
                "shift_step": SHIFT_STEP,
                "grid_cm": SHIFT_GRID_CM,
                "direction_arc_deg": DIRECTION_ARC_DEG,
                "direction_rule": eligible_right_directions.__doc__,
                "arc_tolerance_m": wc.ARC_TOLERANCE_M,
                "tabletop": wc.TABLETOP,
                "plate_apple_clearance_m": wc.PLATE_APPLE_CLEARANCE_M,
                "e9_release": wc.E9_RELEASE,
                "shift_blocked": SHIFT_BLOCKED,
            },
            "planner": {
                "decision_steps": DECISION_STEPS,
                "chunk": CHUNK,
                "coarse_offsets_m": COARSE_OFFSETS_M,
                "coarse_incumbent": COARSE_INCUMBENT,
                "fine_offsets_m": FINE_OFFSETS_M,
                "fine_centre": FINE_CENTRE,
                "candidates_per_decision": CANDIDATES_PER_DECISION,
                "tie_tolerance_cm": TIE_TOLERANCE_CM,
                "o_star_cm": {str(k): v for k, v in O_STAR_CM.items()},
                "o_final_step": O_FINAL_STEP,
                "o_star_source": O_STAR_SOURCE,
                "cost": COST,
                "fallback": FALLBACK,
            },
            "arms": ARMS,
            "l1_arms": L1_ARMS,
            "s_arms": S_ARMS,
            "u_arms": U_ARMS,
            "k1_arms": K1_ARMS,
            "d3_arms": D3_ARMS,
            "critic_arms": CRITIC_ARMS,
            "image_reading_arms": IMAGE_READING_ARMS,
            "learned_label": LEARNED_LABEL,
            "k1": {"resets": K1_RESETS, "bars": K1_BARS},
            "corpus": {
                "name": CORPUS,
                "splits": CORPUS_SPLITS,
                "read_splits": READ_SPLITS,
                "unshifted_fraction": UNSHIFTED_FRACTION,
                "shift_cm": CORPUS_SHIFT_CM,
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
            "p_far": P_FAR,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "o1": O1,
            "o2": O2,
            "o3": O3,
            "o4": O4,
            "o5": O5,
            "offline_rows": OFFLINE_ROWS,
            "d3": {"resets": D3_RESETS, "bars": D3_BARS},
            "gated": GATED,
            "rerun": {
                "fields": RERUN_FIELDS,
                "exact": RERUN_EXACT,
                "tolerances": RERUN_TOLERANCES,
                "margin_rule": RERUN_MARGIN_RULE,
            },
            "gated_rows": GATED_ROWS,
            "authorisation_scope": AUTHORISATION_SCOPE,
            "clause_rows": CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "row_consequences": ROW_CONSEQUENCES,
            "shuf_rule": SHUF_RULE,
            "platform": PLATFORM,
            "sim_workers": SIM_WORKERS,
            "h_workers": H_WORKERS,
            "h_worker_torch_threads": H_WORKER_TORCH_THREADS,
            "devices": DEVICES,
            "render": RENDER,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "feature_anchor": FEATURE_ANCHOR,
            "memory": MEMORY,
            "caps_seconds": CAPS_SECONDS,
            "void_rule": VOID_RULE,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()
