"""TASK-076 (``apple_plate_twin_v2``): the plate-readout perception twin and the prediction-headroom
check K-pred, as frozen module constants.

Protocol ``docs/experiments/apple_plate_twin_v2.md`` (FROZEN after K0-PASS; the sha of
:func:`frozen_sha256` is pinned in ``tests/test_plate_twin_v2.py`` and the manifest, and this block
may not change). This module fixes in code what the protocol preregisters:
- the seeds of every cohort, the smoke range, the RNG salts and their guard (§4);
- the arms of the primary closed loop and of K-pred, and who reads what (§3);
- K-pred's plate motions: the constant-velocity cells M-a and M-b and the action-dependent cell A
  (kappa = -0.5, L = 2, s1 = 525), and H-final(A)'s look-ahead tolerance (§3.2);
- the Stage-0 smoke rules and remedies (§3.2, §5), and their measured results once they exist;
- K0's calibration, its stops and its measured values (``K0_MEASURED``, K0-PASS), the offline
  gates O1/O3/O4, G1/G2/G3, the K-pred bars with the noise guard, every row and its consequence,
  the two abandonment clauses (§5-§7);
- ``TRAINING_BUDGET = None`` (no stage trains iteratively), the caps, the memory and GPU plan.

NumPy only: it imports without torch or MuJoCo.

**No arm of TASK-076 runs a world model.** Nothing here can produce, or be reported as, a
LeWM-driven success. H-twin's pass would be a frozen-encoder perception result.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import obs_ceiling_v2 as oc
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_plate_twin_v2"
TASK = "TASK-076"
STATUS = "FROZEN"
GuardError = lp.GuardError

# ----- the owner's delegation -------------------------------------------------------------------
DELEGATED = "decided by Claude under owner delegation (2026-09-30)"
OWNER_DECISIONS = {
    "owner_delegation_verbatim": lp.OWNER_DECISIONS["owner_delegation_verbatim"],
    "owner_delegation_date": lp.OWNER_DECISIONS["owner_delegation_date"],
    "rulings": "DECISIONS.md decision 2026-10-02 (R1-R4, R7) and decision 2026-10-02 (b) (R8, "
    "R8.1-R8.14)",
}

# ----- seeds (§4) -------------------------------------------------------------------------------
# The repository search of 2026-10-02 (re-checked on all 50 refs in the review of #129) found no
# seed use in 56000-56999 and no use of 7601-7620 as a seed constant.
SEED_RANGES = {
    "K": (56000, 56031),  # K0, development, before the freeze
    "M-a": (56040, 56071),  # K-pred, the 12 cm constant-velocity cell (reported)
    "M-b": (56080, 56111),  # K-pred, the 9 cm constant-velocity cell (reported)
    "D": (56120, 56135),  # Stage D, development, stop rule only
    "A": (56160, 56191),  # K-pred, the action-dependent cell (the only admitting cell)
    "S": (56200, 56263),  # gated, 64 shifted resets
    "U": (56300, 56331),  # gated, 32 unshifted resets
}
SMOKE_SEEDS = (56900, 56999)  # Stage 0: mechanics only; nothing in them is read
TASK_BLOCK = (56000, 56999)
ARENA_SEEDS = (50200, 50231)  # #123's Arena seeds (inside TASK-070's block)
FORBIDDEN_RANGES = dict(oc.FORBIDDEN_RANGES) | {
    "task075_block": oc.TASK_BLOCK,
    "arena_123": ARENA_SEEDS,
}
SALTS = {
    "tau_direction": 7601,  # K0's planted-error direction
    "outer_folds": 7602,  # Stage O's cross-fit (R-plate, R-plate-floor, R-plate-pool)
    "inner_folds": 7603,  # lambda selection in every ridge, the closed-loop R-plate included
    "bootstrap": 7604,  # every bootstrap: §6 intervals, m2*, K0's G2 feasibility resampling
    "moving_direction": 7606,  # the M cells' direction (the -y arc rule)
}
RESERVED_SALTS = {"unused_learning_curve": 7605}  # any use needs a reviewed amendment
CARRIED_SALTS = {
    "shift_direction": lp.SEEDS["shift_direction"],  # 7413
    "reset_redraw": lp.SEEDS["reset_redraw"],  # 7425
}
FLOOR_SEED = 0  # pretrained_encoder.random_init (H-floor, R-plate-floor)
COHORT_ROLES = tuple(SEED_RANGES)
SHIFTED_ROLES = ("K", "D", "S")  # TASK-074's step-300 shift; U and the K-pred cells have none
KPRED_ROLES = ("M-a", "M-b", "A")


def seeds_of(role: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[role]
    return tuple(range(low, high + 1))


def smoke_seeds() -> tuple[int, ...]:
    return tuple(range(SMOKE_SEEDS[0], SMOKE_SEEDS[1] + 1))


def check_seed_ranges() -> None:
    """Every range lies in the task block, the ranges are pairwise disjoint, none overlaps a
    forbidden range, and the salts are distinct from each other and from every carried salt."""
    spans = dict(SEED_RANGES) | {"smoke": SMOKE_SEEDS}
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("G-seeds: two TASK-076 ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < TASK_BLOCK[0] or high > TASK_BLOCK[1]:
            raise GuardError(f"G-seeds: the range {name} leaves the task block {TASK_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"G-seeds: the range {name} overlaps the forbidden {other}")
    salts = [*SALTS.values(), *RESERVED_SALTS.values(), *CARRIED_SALTS.values()]
    if len(set(salts)) != len(salts):
        raise GuardError("G-seeds: two RNG salts are equal")
    earlier = {*lp.SEEDS.values(), *oc.SEEDS.values(), *lp.MODEL_SEEDS, lp.DESIGN_PROBE_SALT}
    if set(SALTS.values()) & earlier or set(RESERVED_SALTS.values()) & earlier:
        raise GuardError("G-seeds: a TASK-076 salt is an earlier task's salt")


def check_role_seeds(role: str, seeds, *, smoke: bool = False) -> tuple[int, ...]:
    """G-seeds: plain ints; exactly the role's seeds in order, or distinct smoke seeds only in a
    smoke. Every forbidden range is refused first."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    if smoke:
        low, high = SMOKE_SEEDS
        if any(not low <= s <= high for s in seeds) or len(set(seeds)) != len(seeds):
            raise GuardError(f"G-seeds: a smoke simulates smoke seeds {low}-{high} only")
        return seeds
    if role not in SEED_RANGES:
        raise GuardError(f"G-seeds: unknown role {role!r}")
    if seeds != seeds_of(role):
        raise GuardError(f"G-seeds: {role} simulates exactly {SEED_RANGES[role]}, once, in order")
    return seeds


# ----- carried unchanged (§2) -------------------------------------------------------------------
SCENE_VERSION = lp.SCENE_VERSION
COUNTED_SUCCESS = lp.COUNTED_SUCCESS
EXPERT = lp.EXPERT  # e9
EXPERT_BUDGETS = lp.EXPERT_BUDGETS  # (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
CARRIED = lp.CARRIED  # P-3
P3_CHECKPOINT_SHA256 = lp.P3_CHECKPOINT_SHA256
EVIDENCE = lp.EVIDENCE  # TASK-072 run-1 (G-repro)
TRANSFER_START = lp.TRANSFER_START  # 405
LOWER_START = lp.LOWER_START  # 505
PLACE_END = lp.PLACE_END  # 725
DECISION_STEPS = lp.DECISION_STEPS  # 405, 421, ..., 485
CHUNK = lp.CHUNK  # 16
EVAL_STEPS = tuple(t + CHUNK for t in DECISION_STEPS)  # 421, ..., 501: the plate-hidden frames
CONDITION = {  # TASK-074's condition: cohorts K, D and S only
    "shift_step": lp.SHIFT_STEP,  # 300
    "shift_cm": 9,
    "direction_arc_deg": lp.DIRECTION_ARC_DEG,
    "shift_vector": "lewm_planner_v2.shift_vector(seed, reset, 9) (salt 7413)",
    "reset": "lewm_planner_v2.condition_reset(seed) (salt 7425), every cohort",
    "blocked": lp.SHIFT_BLOCKED,
}
SOURCE_CORPUS = dict(oc.SOURCE_CORPUS)  # apple-far-shift-v2, sealed
VIEWS_STORE = {
    "name": "apple-far-shift-v2-views",
    "manifest_sha256": "ea627a8fb6b8d7d4057a1a02b1d9c698549fe0a1719808c49a3e6d8a86554d77",
    "protocol": oc.PROTOCOL,
    "view": "onboard112",
    "hidden_key": "onboard112__plate_hidden",
    "note": "TASK-075's sealed store; Stage O reads its stored plate-hidden onboard 112 px frames "
    "at the eval steps 421-501; nothing is rendered",
}
READ_ROOTS = {"train": 225, "val": 28, "total": 253}  # roots whose band reaches the decisions

# ----- arms (§3.1) ------------------------------------------------------------------------------
ARMS = {
    "H-twin": {
        "aim": "R-plate's reading of the current onboard 112 px frame (pretrained DINOv2 full "
        "tokens, dual ridge), at every decision",
        "role": "hypothesis",
        "privileged": False,
    },
    "H-handover": {
        "aim": "the true moved plate (lewm_planner_v2 schedule)",
        "role": "the family ceiling",
        "privileged": True,
    },
    "H-clock": {
        "aim": "the per-decision-step median of the true plate position over K0's resets "
        "(fitted once in K0; reads no image at run time)",
        "role": "the image-free control (G2)",
        "privileged": False,
    },
    "H-stale": {
        "aim": "P-3's post-look plate estimate (read before the move; no later image)",
        "role": "the condition's check (S-VOID-CONDITION, K0 stop)",
        "privileged": False,
    },
    "H-floor": {
        "aim": "R-plate refitted on the seed-0 random-init DINOv2's full tokens",
        "role": "the encoder floor (reported)",
        "privileged": False,
    },
    "P-stale": {"aim": "P-3 unchanged", "role": "the incumbent (G3)", "privileged": False},
}
D_ARMS = ("H-twin", "H-handover")
S_ARMS = ("H-twin", "H-handover", "H-clock", "H-stale", "H-floor", "P-stale")
U_ARMS = ("H-twin", "P-stale", "H-handover")
K0_ARMS = ("H-handover", "H-clock", "H-stale")  # H-handover with a planted error for tau
IMAGE_READING_ARMS = ("H-twin", "H-floor", "H-cv", "H-rule")

# ----- K-pred (§3.2) ----------------------------------------------------------------------------
KPRED = {
    "s0": TRANSFER_START,  # 405: the plate is at its reset position until here
    "s1": 525,  # the plate is static after s1 (inside the lower, 505-555)
    "open_phase": (585, 635),
    "lower_phase": (LOWER_START, 555),
    "hook": "a body_pos write plus mj_forward at every step's observation, outside every "
    "controller's act (as plate_shift.PlateShift); a move that would leave the plate touching "
    "anything but the table is refused, and a refused move ends the attempt as a counted failure",
    "no_step300_shift": True,
}
M_CELLS = {
    "M-a": {"distance_cm": 12, "distance_to_go_cm": 4.0},
    "M-b": {"distance_cm": 9, "distance_to_go_cm": 3.0},
}
M_RULE = (
    "the plate moves at constant velocity from s0 to s1 along one direction per reset, drawn by "
    "TASK-074's -y arc rule (lewm_planner_v2.eligible_right_directions at |D|) from "
    "default_rng(SeedSequence([7606, seed])); plate(t) = plate(s0) + D (min(t, s1) - s0) / "
    "(s1 - s0) for t > s0"
)
CELL_A = {
    "kappa": -0.5,
    "L": 2,
    "s1": 525,
    "status": "removed",  # by the Stage-0 removal rule (STAGE0_SMOKES); PRED-INFEASIBLE
    "rule": "for s0 < t <= s1: plate(t) = plate(s0) + kappa (palm(max(t - L, s0)) - palm(s0)), "
    "palm = the right palm's xy from PalmFK on the executed joint state of each step's "
    "observation; palm velocity at or before s0 counts as zero. So plate(s1) - plate(485) = "
    "kappa (palm(s1 - L) - palm(485 - L))",
}
KAPPA_LIMIT = 1.0  # any |kappa| >= 1 is forbidden in this task and in any amendment (R8.10)
KPRED_ARMS = {
    "M": ("H-final", "H-now", "H-twin", "H-cv"),
    "A": ("H-final", "H-now", "H-twin", "H-cv", "H-rule"),
}
KPRED_ARM_SPECS = {
    "H-final": {
        "aim": "M: the scheduled final position; A: the fixed point by privileged look-ahead",
        "privileged": True,
        "role": "the ceiling",
    },
    "H-now": {
        "aim": "the plate's true position at the decision step",
        "privileged": True,
        "role": "the cost of not predicting, with perfect perception",
    },
    "H-twin": {
        "aim": "R-plate's reading of the current frame",
        "privileged": False,
        "role": "the cost of not predicting, with this perception",
    },
    "H-cv": {
        "aim": "a least-squares constant-velocity line through H-twin's readings so far, "
        "extrapolated to s1 (at 405: the reading)",
        "privileged": False,
        "role": "the hand-written action-blind extrapolator (knows the M form and s1)",
    },
    "H-rule": {
        "aim": "the fixed point g = p + kappa (palm_g(s1 - L) - palm(max(d - L, s0))), p = "
        "H-twin's reading, palm(.) from the robot's own executed states, palm_g from the "
        "kinematic stand-in's place-primitive path under aim g; iterated as H-final(A)",
        "privileged": False,
        "role": "the rule-knowing non-world-model arm (A only)",
    },
}
LOOKAHEAD = {
    "start": "g0 = the true current plate",
    "iterate": "simulate the rest of the place in cloned state to s1 under aim g_k (the plate "
    "rule active), g_{k+1} = the simulated plate at s1; a branch whose plate move is refused "
    "ends there and its iterate is the plate where it stopped (logged)",
    "tolerance": "|g_{k+1} - g_k| <= tau_re / 4",
    "tolerance_fraction_of_tau": 0.25,
    "max_iterations": 10,
    "non_convergence": "logged, aimed at the last iterate, an H-final failure if it fails",
}

# ----- the Stage-0 smokes (§3.2, §5, §10) --------------------------------------------------------
STAGE0_RULES = {
    "seeds": "56900-56999 only; mechanics only; nothing in them is read",
    "contact": "no apple-plate contact before s1 on any smoke attempt; if a smoke finds contact, "
    "s1 moves earlier and the cells are rescaled to keep their distance to go",
    "remaining_motion": "under H-final on cell A, the median |plate(s1) - plate(485)| must be >= "
    "2 cm, and at most a quarter of the smoke attempts may be refused",
    "remedy": "if the median is below 2 cm: first L = 1; then s1 later in steps of 10, up to "
    "the contact limit found by the same smoke (s1 strictly before the earliest apple-plate "
    "contact step of any Stage-0 smoke attempt); L is never raised and kappa is not a remedy",
    "removal": "if no allowed setting gives a median >= 2 cm, or more than a quarter of the "
    "smoke attempts are refused, cell A is removed before the freeze (PRED-INFEASIBLE)",
    "m2": "m2 = |kappa| ||palm_xy(485) - palm_xy(483)|| from executed joint states under "
    "H-final on cell A (median and maximum; the pre-freeze estimate, nothing decided from it)",
    "attempt_time": "the wall time of a full-look-ahead H-final(A) attempt; above 60 s, the "
    "per-attempt cap is reviewed before the freeze",
}
STAGE0_MIN_MEDIAN_REMAINING_CM = 2.0
STAGE0_MAX_REFUSED_FRACTION = 0.25
STAGE0_S1_STEP = 10
STAGE0_ATTEMPT_REVIEW_SECONDS = 60.0
# Written after the Stage-0 smokes (docs/experiments/apple_plate_twin_v2_stage0.md has every
# number). Smoke seeds only; mechanics only; P-3's estimates were the reset truth (a smoke
# stand-in: G-repro could not run, see the record), and tau_re was a 1.0 cm placeholder.
STAGE0_SMOKES: dict | None = {
    "revision": "14b23b0734fb2209175d46d1499de72cf7885f57",
    "root": "outputs/task076-smoke-2 in the task076-stage0 worktree (git-ignored)",
    "estimates": "smoke stand-in: the reset truth (G-repro not run: TASK-072 run-1's corpus root "
    "look2-51171.npz is unreadable on the archive disk)",
    "contact": {
        "contact_before_s1_525": 0,
        "attempts": 144,  # A: 5 arms x 16; M-a, M-b: 4 arms x 8 each; 288 remedy attempts too
        "earliest_apple_plate_contact_step": 607,
        "verdict": "s1 = 525 is free of apple-plate contact on every smoke attempt; no change",
    },
    "cell_a_base": {
        "setting": {"kappa": -0.5, "L": 2, "s1": 525},
        "seeds": [56930, 56945],
        "h_final_median_remaining_cm": 0.0601,
        "h_final_max_remaining_cm": 0.0760,
        "h_now_median_remaining_cm": 0.1481,
        "refused": 0,
        "report_sha256": "cdbc046a73960a2bc504ac01d08f77e00790087e15a53546316b445698b287b2",
    },
    "cell_a_remedies": {
        "contact_limit": 607,
        "median_remaining_cm_by_setting": {
            "L1_s1_525": 0.0626,
            "L1_s1_535": 0.0491,
            "L1_s1_545": 0.0315,
            "L1_s1_555": 0.0252,
            "L1_s1_565": 0.0230,
            "L1_s1_575": 0.0222,
            "L1_s1_585": 0.0235,
            "L1_s1_595": 0.0235,
            "L1_s1_605": 0.0233,
        },
        "refused_by_setting": "0 of 16 at every setting",
        "verdict": "no allowed setting gives a median of at least 2 cm: cell A is removed before "
        "the freeze (protocol §3.2), so K-pred's row is PRED-INFEASIBLE if Stage O records "
        "O-PASS and PRED-NOT-RUN otherwise (decide_kpred checks Stage O first); both escalate "
        "with no clause",
    },
    "m2_cm": {"median": 0.0012, "max": 0.0019, "attempts": 16, "setting": "L = 2, s1 = 525"},
    "lookahead_attempt_seconds": {
        "median": 15.0,
        "max": 16.4,
        "max_over_remedies": 25.0,
        "decisions_converged": "96 of 96 (at most 7 iterations)",
        "verdict": "below 60 s: the 300 s per-attempt cap is not reviewed",
    },
}

# ----- K0 (§5) ----------------------------------------------------------------------------------
TAU = {
    "resets": 32,
    "levels_cm": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    "bar": 28,
    "procedure": "TASK-075's TAU_REMEASURE procedure (its §6.3), adopted: H-handover with a "
    "planted error e at every decision, under TASK-074's condition; one direction per reset, "
    "uniform on [0, 2 pi) from default_rng(SeedSequence([7601, seed])), the same at every level",
    "rule": "tau_re = the largest level L such that every level <= L reaches >= 28/32",
}
K0_STOPS = {
    "level0_below_bar": "level 0 < 28/32 (tau_re undefined)",
    "ceiling_below_30": "N_K(0) < 30/32 (G1's 56/64 is not feasible)",
    "stale_above_4": "H-stale(K) > 4/32 (the condition does not need a reading)",
    "clock_near_ceiling": "H-clock(K) >= N_K(0) - 4/32 (G2 is not feasible; G2's minimum "
    "separation)",
}
K0_CEILING_MIN = 30
K0_STALE_MAX = 4
K0_CLOCK_MARGIN = 4
K0_MEASURED: dict | None = {
    # Written after K0 ran once, ending K0-PASS (protocol §5 step 2, §13.2 step 3). The run:
    # ``run_plate_twin_v2.py k0 --output outputs/task076-k0-1 --evidence <task076-evidence>`` at
    # 2d0bdb7 (clean tree; DRAFT frozen sha fe23e917... before this record was added), on the CPU,
    # without the GPU lock (§8), outputs/task076-k0-1/report.json in the main checkout of the
    # Linux PC. Every value below is copied from that report (sha256 ef4b5410...c876).
    "row": "K0-PASS",
    "tau_re_cm": 1.0,
    "counts": {
        "0.0": 32,
        "0.5": 31,
        "1.0": 28,
        "1.5": 23,
        "2.0": 17,
        "2.5": 18,
        "3.0": 16,
        "4.0": 8,
        "5.0": 5,
    },
    "at_rest": {
        "0.0": 32,
        "0.5": 31,
        "1.0": 28,
        "1.5": 23,
        "2.0": 17,
        "2.5": 18,
        "3.0": 16,
        "4.0": 9,
        "5.0": 10,
        "H-clock": 24,
        "H-stale": 0,
    },  # reported only; the curve is the counted successes
    "n_k0": 32,
    "h_clock": 24,
    "h_stale": 0,
    "stops": {
        "level0_below_bar": False,
        "ceiling_below_30": False,
        "stale_above_4": False,
        "clock_near_ceiling": False,
    },
    "level0_failed_seeds": [],
    "h_clock_failed_seeds": [56000, 56007, 56012, 56017, 56018, 56023, 56028, 56029],
    "h_stale_succeeded_seeds": [],
    # H-clock's fitted targets: the per-step median of the true plate xy over K's 32 resets (m).
    # The plate is static after the shift at step 300, so every decision step has the same target.
    "clock_targets": {
        "405": [0.48203604672515477, -0.18195755761417065],
        "421": [0.48203604672515477, -0.18195755761417065],
        "437": [0.48203604672515477, -0.18195755761417065],
        "453": [0.48203604672515477, -0.18195755761417065],
        "469": [0.48203604672515477, -0.18195755761417065],
        "485": [0.48203604672515477, -0.18195755761417065],
    },
    "g2_feasibility": {
        "b": 8,
        "c": 0,
        "point_p_doubled": 1.52587890625e-05,
        "predicted_pass_probability": 0.9983,
        "resamples": 10000,
        "below_half_disclosed": False,
    },
    "seeds": [56000, 56031],
    "report": "outputs/task076-k0-1/report.json (Linux PC, main checkout; git-ignored)",
    "report_sha256": "ef4b541067dc969ee5ebfc9ee78e2c719a4b84e4d3ed0d71a167feb6fcd0c876",
    "revision": "2d0bdb7b9aa5900be40a5b256df0aaa752179151",
    "tracked_tree_dirty": False,
    "frozen_sha256_at_run": "fe23e9177142804e71dfd8f568b5226f276837f4950ce6dcaf43ec28a674112f",
    "protocol_status_at_run": "DRAFT",
    "go": "a reviewer's reported GO for K0 on #134 (issuecomment-5982693642), at the revision",
    "started_utc": "2026-10-04T17:52:07Z",
    "cohort_first_render_utc": "2026-10-04T17:52:52Z",
    "ended_utc": "2026-10-04T17:57:27Z",
    "total_seconds": 319.75,
    "load_average_at_start": [0.21, 0.663, 1.244],
    "peak_tree_pss_gib": 8.2,
    "workers": 6,
    "gpu_lock": "not taken: K0 is a CPU stage (§8); its EGL rendering runs without the GPU lock",
    "g_repro": "every TASK-072 run-1 reproduction check passed (8 of 8)",
    "render_disagreements": 0,
    "evidence_root": "/home/huhn/develop/emai/worktrees/task076-evidence",
}
G2_FEASIBILITY = {
    "point": "the exact one-sided McNemar p on (2b, 2c), K0's H-handover(level 0) vs H-clock "
    "discordant counts doubled to S's 64 resets",
    "probability": "the fraction of 10 000 resamples of 64 pairs drawn with replacement from K0's "
    "32 pairs (salt 7604) whose exact one-sided McNemar p is below 0.01",
    "direction": "H-handover standing in for H-twin is optimistic for G2; H-clock(K) being "
    "in-sample is pessimistic for G2; the two pull in opposite directions",
    "rule": "reported only; it stops nothing; a probability below 0.5 is disclosed in the freeze "
    "PR as a known risk to G2",
    "resamples": 10_000,
}

# ----- Stage O (§6.1) ---------------------------------------------------------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 5
BOOTSTRAP_RESAMPLES = 10_000
LAMBDA_GRID_RELATIVE = lp.READOUTS["lambda_grid_relative"]
READOUTS = {
    "R_plate": "RidgeReadout (dual) from the full pretrained DINOv2 patch tokens (98 304-d, "
    "standardised on the fit rows) of the onboard 112 px decision frames to the plate xy (m); "
    "lambda from LAMBDA_GRID_RELATIVE by 5-fold inner CV grouped by root (salt 7603)",
    "R_plate_floor": "R_plate on the seed-0 random-init DINOv2's full tokens",
    "R_plate_pool": "RidgeReadout (dual; TASK-075's RidgeReadout settings) from the pooled 4 x 4 "
    "pretrained latent (6 144-d) to the plate xy: defines c_plate, gates nothing in TASK-076",
    "clock_prior": "the per-decision-step median plate xy over the outer-training roots",
    "rows": "each read root's decision frames 405-485 that lie inside its band (t <= hi)",
    "hidden_rows": "each read root's stored plate-hidden onboard 112 px frames at 421-501 "
    "(t <= hi), read by the same outer fold's R_plate against the true plate at t",
    "outer_folds": "5, by root: a seeded permutation of the read roots in plan order, position "
    "mod 5 (salt 7602)",
    "closed_loop": "R_plate (and R_plate_floor for H-floor) fitted once after the Stage O "
    "boundary on every read root's decision frames, lambda by inner CV (salt 7603); its sha256 "
    "is recorded and every worker checks it",
    "featurisation": "CUDA through scripts/gpu_run.sh, strict determinism, batch 64, with a CPU "
    "anchor check (lewm_planner_v2.FEATURE_ANCHOR); the fits on the CPU",
}
O_GATES = {
    "O1_precision": "the upper 95 % bound of R_plate's cross-fitted median error <= tau_re",
    "O3_clock_prior": "the upper 95 % bound of median(e_R_plate) / median(e_clock) < 1.0",
    "O4_plate_hidden": "the lower 95 % bound of R_plate's median error on the plate-hidden frames "
    "> tau_re",
}
O_REPORTED = (
    "the 87.5th percentile of R_plate's error (former O2) with its interval",
    "the tau-curve-mapped predicted count with K0's curve, read against 28/32",
    "per-step errors",
    "the signed mean along x and y",
    "c_plate on R_plate_pool (O1's form)",
    "R_plate / R_plate_floor",
)
RATIO_BAR = 1.0
REPORTED_PERCENTILE = 87.5
OFFLINE_ROWS = ("V", "TWIN-OFF-ARM", "TWIN-OFF-FAIL", "O-PASS")

# ----- the gated bars (§6.2) --------------------------------------------------------------------
S_RESETS, U_RESETS, D_RESETS, K_RESETS, KPRED_RESETS = 64, 32, 16, 32, 32
G1_BAR = 56  # of 64: tau's own bar fraction 28/32
G2_P_BAR = 0.01  # exact one-sided McNemar, H-twin > H-clock on S's pairs
G3_MARGIN = 2  # of 32: H-twin(U) >= P-stale(U) - 2/32 (TASK-074's G5)
S_STALE_MAX = 8  # of 64: S-VOID-CONDITION
D_BARS = {"H-twin": 12, "H-handover": 14}  # of 16: TWIN-DEV-STOP below either
RERUN = {
    "arm": "H-twin",
    "seeds": 4,
    "rule": "pp.rerun_compare under lp.rerun_rule(); any difference is a V (TASK-074 §8.3)",
}
GATED_ROWS = (
    "V",
    "S-VOID-CONDITION",
    "U-VOID-CEILING",
    "TWIN-HARM",
    "TWIN-PASS",
    "TWIN-PRIOR",
    "TWIN-NEAR",
    "TWIN-FAIL",
)
G1_POWER = {"0.875": 0.59, "0.89": 0.73, "0.906": 0.86, "0.9375": 0.98, "0.969": "> 0.99"}

# ----- K-pred's bars (§6.3) ---------------------------------------------------------------------
KP1_BAR = 30  # H-final(A) >= 30/32
HEADROOM_BAR = 8  # K-P2..K-P4: >= +8/32
KPRED_ROWS = (
    "PRED-NOT-RUN",
    "V",
    "PRED-INFEASIBLE",
    "PRED-NONE",
    "PRED-NEAR",
    "PRED-NO-BAR",
    "PRED-ADMIT(A)",
)
NOISE_GUARD = {
    "rule": "a failed headroom bar is detectably below +8/32 when its paired interval's upper "
    "bound is below 8/32, and near otherwise; PRED-NONE needs one detectably below",
    "false_fire_at_8_of_32": "about 2-3 % per bar (2.6, 3.0, 2.2 % with 0, 1, 2 reversed pairs "
    "per 32); without the guard 42-45 %; union bound about 9 %",
    "power_at_4_of_32": "PRED-NONE fires about 42 % of the time at a true headroom of 4/32 (36 % "
    "at 5/32 with one reversed pair) and about 87 % at 2/32; at an intermediate headroom "
    "PRED-NEAR is the likely row, an escalation rather than the clause (stated in advance)",
}
M2_STAR = {
    "m2": "|kappa| ||palm_xy(485) - palm_xy(483)|| per H-final(A) attempt, executed joint states",
    "m2_star": "the upper 95 % bound of the median over cell A's 32 H-final attempts (bootstrap, "
    "salt 7604), and its 87.5th percentile",
    "history_rule": "TASK-066's history-one predictor qualifies for TASK-077 only if c_plate + "
    "m2* <= tau_re, or B's calibrated allowance covers m2*; otherwise TASK-077 declares 3 frames "
    "or the last 2 executed commands",
}

# ----- rows, consequences and the clauses (§6, §7, §13.3) ---------------------------------------
K0_ROWS = ("CAL-ESCALATE", "K0-PASS")
D_ROWS = ("TWIN-DEV-STOP", "D-PASS")
CLAUSE_ROWS = ("TWIN-OFF-FAIL", "TWIN-FAIL", "TWIN-HARM")
KPRED_CLAUSE_ROWS = ("PRED-NONE",)
CLAUSE_SCOPE = (
    "aiming e9's place primitive, after P-3's pick, at a single-frame frozen-DINOv2 ridge "
    "readout of the plate from the onboard 112 px camera, under TASK-074's 9 cm condition on "
    "apple-to-plate-v2: no further readout variant of this formulation (pooled or full tokens, "
    "lambda grid, crop, colour) is preregistered without new evidence of a different kind (a "
    "place primitive with its own feedback, temporal aggregation of readings, or a new view or "
    "sensor). The next step is TASK-075's Option 2 (a place servo; R3)"
)
KPRED_CLAUSE_SCOPE = (
    "preregistering a LeWM plate-target place task on v2 under the tested action-dependent rule "
    "(cell A: kappa = -0.5 and the L and s1 it ran with) and under the constant-velocity family "
    "(M-a, M-b), without new evidence of a different kind. It does not close the LeWM backend, "
    "the v2 task or the product goal"
)
NEVER_CLOSED = (
    "the LeWM backend",
    "DINOv2 as an encoder",
    "the v2 task",
    "Arena",
    "the product goal",
)
ROW_CONSEQUENCES = {
    "CAL-ESCALATE": "escalate, no clause; nothing is frozen",
    "K0-PASS": "K0's values enter the frozen block; freeze review",
    "TWIN-OFF-ARM": "escalate, no clause; K-pred is PRED-NOT-RUN",
    "TWIN-OFF-FAIL": "the clause fires; next step Option 2 (R3); K-pred is PRED-NOT-RUN",
    "O-PASS": "Stage D and Stage K-pred may run, each on a GO",
    "TWIN-DEV-STOP": "escalate, no clause",
    "D-PASS": "Stage S/U may run on a GO",
    "V": "one repeat after a reviewed fix; a second V escalates",
    "S-VOID-CONDITION": "escalate, no clause, no claim",
    "U-VOID-CEILING": "escalate, no clause, no claim (R8.13)",
    "TWIN-PRIOR": "escalate, no clause, no claim",
    "TWIN-NEAR": "escalate, no clause, no claim; any repeat needs fresh seeds and its own ruling",
    "TWIN-HARM": "the clause fires; next step Option 2 (R3)",
    "TWIN-FAIL": "the clause fires; next step Option 2 (R3)",
    "TWIN-PASS": "the claim of §1; results PR; the next LeWM task is chosen by K-pred's row",
    "PRED-NOT-RUN": "escalate, no clause; PLAN.md's Branch B",
    "PRED-INFEASIBLE": "escalate, no clause; PLAN.md's Branch B",
    "PRED-NO-BAR": "escalate, no clause; PLAN.md's Branch B",
    "PRED-NONE": "K-pred's clause fires; Branch B",
    "PRED-NEAR": "escalate, no clause, no claim; any repeat needs fresh seeds and its own ruling",
    "PRED-ADMIT(A)": "TASK-077 may be preregistered under cell A, with R8.8's three declarations "
    "and R8.9's history rule (Branch A)",
}

# ----- budgets, caps, platform (§7, §8, §10) ----------------------------------------------------
TRAINING_BUDGET = None  # no stage trains iteratively: every readout is a closed-form ridge
BUDGET_RULE = (
    "if a reviewed amendment ever adds iterative training, its BUDGET block must pass "
    "run_tools.check_budget, select checkpoints with run_tools.select_checkpoint and apply the "
    "last-two rule only through run_tools.last_two_triggered"
)
CAPS_SECONDS = {"invocation": 7_200.0, "per_attempt": 300.0, "featurisation": 1_800.0}
ESTIMATES_SECONDS = {
    "invocation_max": 900.0,
    "attempt_ordinary": 5.5,
    "attempt_lookahead": 42.0,
    "featurisation": 120.0,
}
CAP_FACTOR_MIN = 5.0
PLATFORM = lp.PLATFORM
SIM_WORKERS = lp.SIM_WORKERS  # 6
WORKER_TORCH_THREADS = 1
THREAD_ENV = lp.THREAD_ENV
QUIET_MACHINE = lp.QUIET_MACHINE
FEATURE_ANCHOR = lp.FEATURE_ANCHOR
MEMORY = {
    "ceiling_gib": 12.0,
    "headroom_gib": lp.MEMORY["headroom_gib"],
    "sample_seconds": 0.5,
    "measure": "pss",
    "watch": "run_tools.MemoryWatch (process-tree PSS, run_guards.process_tree_memory)",
    "scale_rule": "the offline smoke's scale probe runs the Stage O core itself "
    "(run_tools.scale_probe) on smoke data presented at the real sizes (240 train + 30 val "
    "slots); its peak must stay SCALE_MARGIN_GIB below the ceiling before any GO",
}
SCALE_MARGIN_GIB = 2.0
GPU = {
    "stage": "Stage O's featurisation only",
    "launcher": "scripts/gpu_run.sh --wait --min-free-gib 4 --board --who oej:task076-O -- uv "
    "run --no-sync python scripts/run_plate_twin_v2.py offline ...",
    "guard": "run_tools.gpu_guard(report, min_free_gib=1.0, process_cap_gib=3.0, "
    "require_lock=True)",
    "min_free_gib": 1.0,
    "process_cap_gib": 3.0,
    "require_lock": True,
    "rule": "the machine-wide flock ~/.local/state/gpu/lock (shared); resident GPU services are "
    "never stopped, killed or reconfigured; everything else runs on the CPU and takes no lock",
}
VOID_RULE = (
    "a stage is V on a stop signal (run_tools.install_guards records the first signal), a cap, "
    "a CUDA allocation failure, a pin or frozen-sha mismatch, a decoded test root, a privileged "
    "read in a non-privileged arm, a failed determinism re-run or G-repro failing. A V after the "
    "stage's outcome boundary (cohort_first_render_utc for a simulating stage, first_outcome_utc "
    "for Stage O) is not read as an outcome; one repeat from scratch after a reviewed fix; a "
    "second V escalates"
)
STAGES = ("stage0", "k0", "offline", "dev", "gated", "kpred", "results")


# ----- K-pred's plate motion ----------------------------------------------------------------------
def moving_vector(seed: int, reset: dict, distance_cm: float) -> list[float]:
    """An M cell's total displacement (world xy, m): one uniform draw per reset from
    ``default_rng(SeedSequence([7606, seed]))`` over TASK-074's eligible -y directions."""
    eligible = lp.eligible_right_directions(
        reset["plate_xy"], reset["object_xy"], distance_cm / 100.0
    )
    if not eligible:
        raise GuardError(f"no eligible -y direction for seed {seed} at {distance_cm} cm")
    u = np.random.default_rng(np.random.SeedSequence([SALTS["moving_direction"], int(seed)]))
    pick = eligible[int(math.floor(float(u.uniform()) * len(eligible)))]
    theta = math.radians(pick)
    m = distance_cm / 100.0
    return [m * math.cos(theta), m * math.sin(theta)]


def constant_velocity_xy(base_xy, vector_m, step: int, s0: int, s1: int) -> np.ndarray:
    """The M cells' plate position at ``step``."""
    frac = (min(max(int(step), s0), s1) - s0) / float(s1 - s0)
    return np.asarray(base_xy, np.float64) + frac * np.asarray(vector_m, np.float64)


def palm_driven_xy(base_xy, palm: dict, step: int, kappa: float, lag: int, s0: int, s1: int):
    """Cell A's plate position at ``step`` from the executed palm xy history ``palm`` (step ->
    xy). Needs ``palm[s0]`` and ``palm[max(min(step, s1) - lag, s0)]``."""
    check_kappa(kappa)
    t = min(int(step), s1)
    if t <= s0:
        return np.asarray(base_xy, np.float64).copy()
    u = max(t - int(lag), s0)
    return np.asarray(base_xy, np.float64) + kappa * (
        np.asarray(palm[u], np.float64) - np.asarray(palm[s0], np.float64)
    )


def check_kappa(kappa: float) -> None:
    if not abs(float(kappa)) < KAPPA_LIMIT:
        raise GuardError("G-kappa: |kappa| >= 1 is forbidden (R8.10)")


def check_lag(lag: int) -> None:
    if type(lag) is not int or lag < 1 or lag > CELL_A["L"]:
        raise GuardError("G-lag: L is 2, or 1 under the remedy; it is never raised")


def m2_cm(palm_483, palm_485, kappa: float = CELL_A["kappa"]) -> float:
    """The 2-step term (cm): |kappa| ||palm_xy(485) - palm_xy(483)||."""
    d = np.asarray(palm_485, np.float64)[:2] - np.asarray(palm_483, np.float64)[:2]
    return 100.0 * abs(float(kappa)) * float(np.linalg.norm(d))


def stage0_cell_a_verdict(remaining_cm, refused: int, attempts: int) -> dict:
    """One Stage-0 setting of cell A: does it meet the 2 cm median and the refusal quarter?
    ``remaining_cm`` holds the H-final attempts' |plate(s1) - plate(485)| (refused ones
    excluded; they count in ``refused``)."""
    values = np.asarray(remaining_cm, np.float64)
    median = float(np.median(values)) if len(values) else None
    fraction = refused / attempts if attempts else 1.0
    ok_motion = median is not None and median >= STAGE0_MIN_MEDIAN_REMAINING_CM
    ok_refused = fraction <= STAGE0_MAX_REFUSED_FRACTION
    return {
        "median_remaining_cm": median,
        "refused": int(refused),
        "attempts": int(attempts),
        "refused_fraction": fraction,
        "motion_ok": bool(ok_motion),
        "refused_ok": bool(ok_refused),
        "feasible": bool(ok_motion and ok_refused),
    }


def remedy_settings(contact_limit: int | None, base_s1: int = CELL_A["s1"]) -> list[dict]:
    """The declared remedy sequence after the base setting (L = 2, s1 = 525) falls short: L = 1,
    then s1 later in steps of 10 (with L = 1) up to the contact limit: the earliest apple-plate
    contact step any Stage-0 smoke attempt showed, s1 staying strictly before it so that no
    smoke attempt has contact before s1 (decided by Claude under owner delegation). With no
    contact at all, s1 stays before the place's end."""
    limit = (contact_limit - 1) if contact_limit is not None else PLACE_END - 1
    out = [{"L": 1, "s1": base_s1}]
    s1 = base_s1 + STAGE0_S1_STEP
    while s1 <= limit:
        out.append({"L": 1, "s1": s1})
        s1 += STAGE0_S1_STEP
    return out


# ----- cohorts (§4) ----------------------------------------------------------------------------
def cohort_values(role: str, seeds=None) -> dict:
    """Per seed: the reset (condition_reset), and the role's step-300 shift or K-pred motion."""
    out = {}
    for seed in seeds or seeds_of(role):
        reset = lp.condition_reset(int(seed))
        entry = {"seed": int(seed), **reset}
        if role in SHIFTED_ROLES:
            entry["shift_m"] = lp.shift_vector(int(seed), reset, CONDITION["shift_cm"])
        if role in M_CELLS:
            entry["moving_m"] = moving_vector(int(seed), reset, M_CELLS[role]["distance_cm"])
        out[str(seed)] = entry
    return out


def cohort_digest(role: str) -> str:
    """sha256 of the role's draws with every float rounded to 10 decimals (lewm_planner_v2.
    plan_digest's rule: the last ulp of a draw is platform-dependent, as TASK-073 found)."""
    return lp.plan_digest(cohort_values(role))


# ----- K0 ---------------------------------------------------------------------------------------
def tau_direction(seed: int) -> float:
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["tau_direction"], int(seed)]))
    return float(2.0 * np.pi * rng.uniform())


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    angle = tau_direction(seed)
    r = float(level_cm) / 100.0
    return [r * float(np.cos(angle)), r * float(np.sin(angle))]


def tau_from_counts(counts: dict) -> dict:
    """tau_re: the largest level L such that every level <= L reaches the bar (None if level 0
    fails)."""
    by_level = {float(k): int(v) for k, v in counts.items()}
    if sorted(by_level) != sorted(TAU["levels_cm"]):
        raise ContractError("tau_re needs every preregistered level")
    tau = None
    for level in sorted(by_level):
        if by_level[level] >= TAU["bar"]:
            tau = level
        else:
            break
    return {"tau_re_cm": tau}


def clock_targets(plate_at_steps: dict) -> dict:
    """H-clock's targets: per decision step, the median of the true plate xy over K0's resets.
    ``plate_at_steps``: seed -> {step: xy}."""
    out = {}
    for t in DECISION_STEPS:
        rows = np.asarray([v[t] for v in plate_at_steps.values()], np.float64)
        out[int(t)] = np.median(rows, axis=0).tolist()
    return out


def decide_k0(counts: dict, n_clock: int, n_stale: int) -> dict:
    """K0's stops (first one that fires is named; every one is recorded)."""
    tau = tau_from_counts(counts)["tau_re_cm"]
    n0 = int(counts[str(0.0)] if str(0.0) in counts else counts[0.0])
    fired = {
        "level0_below_bar": n0 < TAU["bar"],
        "ceiling_below_30": n0 < K0_CEILING_MIN,
        "stale_above_4": int(n_stale) > K0_STALE_MAX,
        "clock_near_ceiling": int(n_clock) >= n0 - K0_CLOCK_MARGIN,
    }
    row = "CAL-ESCALATE" if any(fired.values()) else "K0-PASS"
    return {
        "row": row,
        "tau_re_cm": tau,
        "n_k0": n0,
        "stops": fired,
        "consequence": ROW_CONSEQUENCES[row],
    }


# ----- statistics -------------------------------------------------------------------------------
def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar p: P(X >= b) for X ~ Bin(b + c, 1/2) (b = first-only wins)."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(int(b), n + 1)) / 2.0**n)


def discordant(first, second) -> tuple[int, int]:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ContractError("paired outcomes need the same resets")
    return int((a & ~b).sum()), int((~a & b).sum())


def g2_feasibility(handover, clock, *, resamples: int = G2_FEASIBILITY["resamples"]) -> dict:
    """R8.12: G2's predicted feasibility from K0's paired H-handover (level 0) and H-clock."""
    a, c_ = np.asarray(handover, bool), np.asarray(clock, bool)
    b, c = discordant(a, c_)
    rng = np.random.default_rng(SALTS["bootstrap"])
    passes = 0
    for _ in range(resamples):
        i = rng.integers(0, len(a), S_RESETS)
        bb, cc = discordant(a[i], c_[i])
        passes += mcnemar_one_sided(bb, cc) < G2_P_BAR
    return {
        "b": b,
        "c": c,
        "point_p_doubled": mcnemar_one_sided(2 * b, 2 * c),
        "predicted_pass_probability": passes / resamples,
        "resamples": int(resamples),
        "below_half_disclosed": passes / resamples < 0.5,
    }


def paired_interval(first, second, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The paired difference in successes (first - second) with its reset-clustered bootstrap
    95 % percentile interval (salt 7604) and the discordant counts."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ContractError("paired outcomes need the same resets")
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(SALTS["bootstrap"])
    boot = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    only_first, only_second = discordant(a, b)
    return {
        "difference": int(d.sum()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(d)),
        "only_first": only_first,
        "only_second": only_second,
    }


def _groups(clusters):
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(len(labels))]


def _boot(statistic, groups, resamples):
    rng = np.random.default_rng(SALTS["bootstrap"])
    out = np.empty(resamples)
    for b in range(resamples):
        out[b] = statistic(
            np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        )
    return out


def cluster_median_ci(values, clusters, *, resamples: int | None = None) -> dict:
    v = np.asarray(values, np.float64)
    boot = _boot(lambda i: np.median(v[i]), _groups(clusters), resamples or BOOTSTRAP_RESAMPLES)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "median": float(np.median(v)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
        "clusters": int(len(_groups(clusters))),
    }


def cluster_median_ratio(num, den, clusters, *, resamples: int | None = None) -> dict:
    a, b = np.asarray(num, np.float64), np.asarray(den, np.float64)

    def ratio(i):
        d = np.median(b[i])
        return np.median(a[i]) / d if d > 0 else np.finfo(np.float64).max

    boot = _boot(ratio, _groups(clusters), resamples or BOOTSTRAP_RESAMPLES)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = float(np.median(b))
    return {
        "ratio": float(np.median(a) / base) if base > 0 else None,
        "ci95": [float(lo), float(hi)],
        "n": int(len(a)),
    }


def cluster_quantile_ci(values, clusters, q: float, *, resamples: int | None = None) -> dict:
    v = np.asarray(values, np.float64)
    boot = _boot(
        lambda i: np.percentile(v[i], q), _groups(clusters), resamples or BOOTSTRAP_RESAMPLES
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "quantile": float(q),
        "value": float(np.percentile(v, q)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
    }


def cluster_mean_ci(values, clusters, *, resamples: int | None = None) -> dict:
    v = np.asarray(values, np.float64)
    boot = _boot(lambda i: np.mean(v[i]), _groups(clusters), resamples or BOOTSTRAP_RESAMPLES)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(v)), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def median_upper(values, *, resamples: int | None = None) -> dict:
    """m2*: the median with its bootstrap 95 % interval over attempts (salt 7604), and the 87.5th
    percentile."""
    v = np.asarray(values, np.float64)
    ci = cluster_median_ci(v, np.arange(len(v)), resamples=resamples)
    return ci | {"upper": ci["ci95"][1], "p87_5": float(np.percentile(v, REPORTED_PERCENTILE))}


def tau_curve_fraction(errors_cm, counts: dict) -> np.ndarray:
    """O2 reported: each error mapped to the measured tau curve's success fraction (np.interp;
    beyond 5 cm the 5 cm fraction)."""
    levels = np.asarray(TAU["levels_cm"], np.float64)
    fraction = np.asarray([counts[str(float(x))] for x in levels], np.float64) / TAU["resets"]
    return np.interp(np.asarray(errors_cm, np.float64), levels, fraction)


# ----- decisions --------------------------------------------------------------------------------
def decide_offline(stats: dict, tau_re_cm: float, *, void: bool = False) -> dict:
    """``stats``: ``o1_upper``, ``o3_ratio_upper``, ``o4_lower`` (cm and ratio)."""
    if void:
        return {"row": "V"}
    checks = {
        "O1_precision": bool(stats["o1_upper"] <= tau_re_cm),
        "O3_clock_prior": bool(stats["o3_ratio_upper"] < RATIO_BAR),
        "O4_plate_hidden": bool(stats["o4_lower"] > tau_re_cm),
    }
    if not checks["O4_plate_hidden"]:
        row = "TWIN-OFF-ARM"
    elif not (checks["O1_precision"] and checks["O3_clock_prior"]):
        row = "TWIN-OFF-FAIL"
    else:
        row = "O-PASS"
    return {
        "row": row,
        "checks": checks,
        "tau_re_cm": tau_re_cm,
        "consequence": ROW_CONSEQUENCES[row],
        "clause_fires": row in CLAUSE_ROWS,
    }


def decide_dev(successes: dict) -> dict:
    counts = {arm: int(np.sum(successes[arm])) for arm in D_ARMS}
    stop = any(counts[arm] < D_BARS[arm] for arm in D_ARMS)
    row = "TWIN-DEV-STOP" if stop else "D-PASS"
    return {"row": row, "counts": counts, "bars": D_BARS, "consequence": ROW_CONSEQUENCES[row]}


def decide_gated(s: dict, u: dict, *, void: bool = False, resamples: int | None = None) -> dict:
    """First match over GATED_ROWS. ``s[arm]`` and ``u[arm]``: per-reset successes, in seed
    order."""
    if void:
        return {"row": "V", "consequence": ROW_CONSEQUENCES["V"]}
    n = {arm: int(np.sum(s[arm])) for arm in S_ARMS}
    nu = {arm: int(np.sum(u[arm])) for arm in U_ARMS}
    b, c = discordant(s["H-twin"], s["H-clock"])
    p2 = mcnemar_one_sided(b, c)
    g1 = n["H-twin"] >= G1_BAR
    g2 = n["H-twin"] > n["H-clock"] and p2 < G2_P_BAR
    g3 = nu["H-twin"] >= nu["P-stale"] - G3_MARGIN
    near = paired_interval(s["H-twin"], s["H-handover"], resamples=resamples or BOOTSTRAP_RESAMPLES)
    if n["H-stale"] > S_STALE_MAX or n["H-handover"] < G1_BAR:
        row = "S-VOID-CONDITION"
    elif nu["H-handover"] < nu["P-stale"] - G3_MARGIN:
        row = "U-VOID-CEILING"
    elif not g3:
        row = "TWIN-HARM"
    elif g1 and g2:
        row = "TWIN-PASS"
    elif g1:
        row = "TWIN-PRIOR"
    elif near["ci95"][0] <= 0.0 <= near["ci95"][1]:
        row = "TWIN-NEAR"
    else:
        row = "TWIN-FAIL"
    return {
        "row": row,
        "counts_s": n,
        "counts_u": nu,
        "g1": g1,
        "g2": {"passes": g2, "b": b, "c": c, "p": p2},
        "g3": g3,
        "twin_minus_handover": near,
        "consequence": ROW_CONSEQUENCES[row],
        "clause_fires": row in CLAUSE_ROWS,
        "claim": row == "TWIN-PASS",
    }


def decide_kpred(
    a: dict | None,
    *,
    offline_row: str,
    c_plate_cm: float | None,
    tau_re_cm: float | None,
    void: bool = False,
    resamples: int | None = None,
) -> dict:
    """First match over KPRED_ROWS. ``a[arm]``: cell A's per-reset successes (None when cell A
    was removed at Stage 0)."""
    if offline_row != "O-PASS":
        return {"row": "PRED-NOT-RUN", "consequence": ROW_CONSEQUENCES["PRED-NOT-RUN"]}
    if void:
        return {"row": "V", "consequence": ROW_CONSEQUENCES["V"]}
    if a is None:
        return {
            "row": "PRED-INFEASIBLE",
            "reason": "cell A removed at Stage 0",
            "consequence": ROW_CONSEQUENCES["PRED-INFEASIBLE"],
        }
    final = int(np.sum(a["H-final"]))
    kp1 = final >= KP1_BAR
    headrooms = {}
    for bar, other in (("K-P2", "H-now"), ("K-P3", "H-twin"), ("K-P4", "H-cv")):
        interval = paired_interval(
            a["H-final"], a[other], resamples=resamples or BOOTSTRAP_RESAMPLES
        )
        passes = interval["difference"] >= HEADROOM_BAR
        headrooms[bar] = interval | {
            "passes": passes,
            "detectably_below": (not passes) and interval["ci95"][1] < HEADROOM_BAR,
        }
    kp5 = c_plate_cm is not None and tau_re_cm is not None and c_plate_cm <= tau_re_cm
    failed = [k for k, v in headrooms.items() if not v["passes"]]
    if not kp1:
        row = "PRED-INFEASIBLE"
    elif any(headrooms[k]["detectably_below"] for k in failed):
        row = "PRED-NONE"
    elif failed:
        row = "PRED-NEAR"
    elif not kp5:
        row = "PRED-NO-BAR"
    else:
        row = "PRED-ADMIT(A)"
    return {
        "row": row,
        "h_final": final,
        "kp1": kp1,
        "headrooms": headrooms,
        "kp5": {"passes": kp5, "c_plate_cm": c_plate_cm, "tau_re_cm": tau_re_cm},
        "consequence": ROW_CONSEQUENCES[row],
        "clause_fires": row in KPRED_CLAUSE_ROWS,
        "h_rule_reported_only": True,
    }


def frozen_block() -> dict:
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "carried_from": [lp.PROTOCOL, oc.PROTOCOL],
            "owner_decisions": OWNER_DECISIONS,
            "delegated": DELEGATED,
            "seed_ranges": SEED_RANGES,
            "smoke_seeds": SMOKE_SEEDS,
            "task_block": TASK_BLOCK,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "salts": SALTS,
            "reserved_salts": RESERVED_SALTS,
            "carried_salts": CARRIED_SALTS,
            "floor_seed": FLOOR_SEED,
            "shifted_roles": SHIFTED_ROLES,
            "scene_version": SCENE_VERSION,
            "counted_success": COUNTED_SUCCESS,
            "expert": EXPERT,
            "expert_budgets": EXPERT_BUDGETS,
            "carried": CARRIED,
            "p3_checkpoint_sha256": P3_CHECKPOINT_SHA256,
            "evidence": EVIDENCE,
            "steps": {
                "transfer_start": TRANSFER_START,
                "lower_start": LOWER_START,
                "place_end": PLACE_END,
                "decision": DECISION_STEPS,
                "eval": EVAL_STEPS,
                "chunk": CHUNK,
            },
            "condition": CONDITION,
            "source_corpus": SOURCE_CORPUS,
            "views_store": VIEWS_STORE,
            "read_roots": READ_ROOTS,
            "arms": ARMS,
            "d_arms": D_ARMS,
            "s_arms": S_ARMS,
            "u_arms": U_ARMS,
            "k0_arms": K0_ARMS,
            "kpred": KPRED,
            "m_cells": M_CELLS,
            "m_rule": M_RULE,
            "cell_a": CELL_A,
            "kappa_limit": KAPPA_LIMIT,
            "kpred_arms": KPRED_ARMS,
            "kpred_arm_specs": KPRED_ARM_SPECS,
            "lookahead": LOOKAHEAD,
            "stage0_rules": STAGE0_RULES,
            "stage0_thresholds": {
                "min_median_remaining_cm": STAGE0_MIN_MEDIAN_REMAINING_CM,
                "max_refused_fraction": STAGE0_MAX_REFUSED_FRACTION,
                "s1_step": STAGE0_S1_STEP,
                "attempt_review_seconds": STAGE0_ATTEMPT_REVIEW_SECONDS,
            },
            "stage0_smokes": STAGE0_SMOKES,
            "tau": TAU,
            "k0_stops": K0_STOPS,
            "k0_thresholds": {
                "ceiling_min": K0_CEILING_MIN,
                "stale_max": K0_STALE_MAX,
                "clock_margin": K0_CLOCK_MARGIN,
            },
            "k0_measured": K0_MEASURED,
            "g2_feasibility": G2_FEASIBILITY,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "lambda_grid_relative": LAMBDA_GRID_RELATIVE,
            "readouts": READOUTS,
            "o_gates": O_GATES,
            "o_reported": O_REPORTED,
            "ratio_bar": RATIO_BAR,
            "reported_percentile": REPORTED_PERCENTILE,
            "offline_rows": OFFLINE_ROWS,
            "resets": {
                "S": S_RESETS,
                "U": U_RESETS,
                "D": D_RESETS,
                "K": K_RESETS,
                "kpred": KPRED_RESETS,
            },
            "g1_bar": G1_BAR,
            "g2_p_bar": G2_P_BAR,
            "g3_margin": G3_MARGIN,
            "s_stale_max": S_STALE_MAX,
            "d_bars": D_BARS,
            "rerun": RERUN,
            "gated_rows": GATED_ROWS,
            "g1_power": G1_POWER,
            "kp1_bar": KP1_BAR,
            "headroom_bar": HEADROOM_BAR,
            "kpred_rows": KPRED_ROWS,
            "noise_guard": NOISE_GUARD,
            "m2_star": M2_STAR,
            "k0_rows": K0_ROWS,
            "d_rows": D_ROWS,
            "clause_rows": CLAUSE_ROWS,
            "kpred_clause_rows": KPRED_CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "kpred_clause_scope": KPRED_CLAUSE_SCOPE,
            "never_closed": NEVER_CLOSED,
            "row_consequences": ROW_CONSEQUENCES,
            "training_budget": TRAINING_BUDGET,
            "budget_rule": BUDGET_RULE,
            "caps_seconds": CAPS_SECONDS,
            "estimates_seconds": ESTIMATES_SECONDS,
            "cap_factor_min": CAP_FACTOR_MIN,
            "platform": PLATFORM,
            "sim_workers": SIM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "feature_anchor": FEATURE_ANCHOR,
            "memory": MEMORY,
            "scale_margin_gib": SCALE_MARGIN_GIB,
            "gpu": GPU,
            "void_rule": VOID_RULE,
            "stages": STAGES,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()
