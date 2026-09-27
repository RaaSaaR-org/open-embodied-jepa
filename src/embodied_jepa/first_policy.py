"""TASK-067 (``apple_first_policy_v1``): can an information-matched learned policy reach a first
Apple->Plate success on the development cohort?

Protocol ``docs/experiments/apple_first_policy_v1.md``. This module fixes in code, before any
policy is trained, what the protocol preregisters: the seed ranges, the arms and their labels,
the policy's inputs, the training and DAgger schedule, the calibration rule for the perception
bars (C0), the offline gates (S0), the development milestone rows (M1), the gated-stage gates on
cohort C (M2, a separate authorization), the budget and the abandonment clause. NumPy only: it
imports without torch or MuJoCo.

The controller under test ("P") receives, at run time, what the scripted expert acts on and
nothing privileged: apple and plate xy read once from the frozen DINOv2 tokens of the post-look
frame by a learned readout; its own step counter since the look; proprioception; and the right
palm pose by forward kinematics of that proprioception. It is trained by behaviour cloning on
``apple-look-v1`` train roots and by three DAgger iterations labelled by the privileged expert
on fresh resets, at training time only.

**Learned Apple->Plate is still 0 successes.** A success of P on the development cohort would be
an existence result, labelled "learned policy with a DINOv2 encoder" -- never "LeWM driving the
robot" (owner ruling R4).
"""

from __future__ import annotations

import numpy as np

from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_first_policy_v1"
TASK = "TASK-067"


class GuardError(ContractError):
    """A guard failed: the run is void (row V) and no outcome is read."""


# ----- owner rulings (recorded verbatim in the protocol, §0) ----------------------------------
OWNER_RULINGS_UTC = "2026-09-27T12:43Z"

# ----- data --------------------------------------------------------------------------------------
DATASET = "data/apple-look-v1"
DATASET_MANIFEST_SHA256 = "81d760d1f5a61834b4f37c3daf12ad1a7175e57fe99fe11d5fb4cecbff8edb64"
READ_SPLITS = ("train", "val")  # the test split is never decoded
EXPECTED_ROOTS = {"train": 170, "val": 20}
LOOK_STEPS = 8
DECISION_FRAME = 8  # the post-look frame of every root episode
IMAGE_SIZE = 112
CAMERA = "onboard_rgb"

# ----- seeds -------------------------------------------------------------------------------------
# Every range here lies in 46000-46999, which the repository grep of 2026-09-27 (protocol §4.1)
# found unused by any corpus, cohort, pilot or probe.
SEED_RANGES = {
    "perception_train": (46000, 46255),
    "perception_heldout": (46256, 46383),
    "calibration_C0": (46384, 46415),
    "dagger_1": (46416, 46543),
    "dagger_2": (46544, 46671),
    "dagger_3": (46672, 46799),
}
RESERVED_UNUSED = (46800, 46999)
COHORT_D = tuple(range(45000, 45008)) + tuple(range(45100, 45108))  # development, never gates
COHORT_C = tuple(range(45300, 45340))  # frozen, gating; M2 only, separate authorization
# Ranges this task must never simulate for training, perception or calibration.
FORBIDDEN_RANGES = {
    "mvp_frozen_resets": (20000, 20049),
    "mechanics_probes": (41000, 41101),
    "earlier_train_val_test": (42000, 42031),
    "narrow_development": (43000, 43004),
    "final_cohort": (44000, 44019),
    "cohort_D_and_prior_ceilings": (45000, 45207),
    "cohort_C": (45300, 45339),
    "apple_look_v1_corpus": (47000, 47199),
    "apple_look_pilots": (47900, 47931),
    "apple_wide_v1_corpus_and_pilots": (48000, 48999),
    "grasp_closure_probes": (49000, 49199),
}
MODEL_SEED = 0
READOUT_FOLD_SEED = 6701
C0_DIRECTION_SEED = 6700
TRAIN_SAMPLER_SEED = 6702
RANDOM_CONTROLLER_SEED = 6703


def seeds_of(name: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[name]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    """Every task range is inside 46000-46999, pairwise disjoint, and off every forbidden range."""
    spans = sorted(SEED_RANGES.values())
    for (_, a_high), (b_low, _) in zip(spans, spans[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("TASK-067 seed ranges overlap")
    for low, high in spans:
        if low < 46000 or high > 46999 or low > high:
            raise GuardError("a TASK-067 seed range leaves 46000-46999")
        for name, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"a TASK-067 seed range overlaps {name}")


# ----- the expert's schedule (scripted.apple_collector_policy, after the look) -----------------
PHASES = ("orient", "descend", "close", "lift", "transfer", "release_high", "lower_open", "retreat")
PHASE_BUDGETS = (130, 80, 45, 150, 60, 100, 100, 80)
EXPERT_POLICY_STEPS = sum(PHASE_BUDGETS)  # 745
MAX_POLICY_STEPS = 800  # per attempt, after the look
ATTEMPT_WALL_SECONDS = 300.0


def scheduled_phase(step: int) -> int:
    """The collector's phase index at post-look step ``step`` on its own clock (fallback F only).

    Steps at or beyond the budget map to the last phase."""
    if step < 0:
        raise ContractError("step counts from 0 after the look")
    edges = np.cumsum(PHASE_BUDGETS)
    return int(min(np.searchsorted(edges, step, side="right"), len(PHASES) - 1))


# ----- the policy (fixed now) --------------------------------------------------------------------
FREE_INDICES = (6, 7, 8, 9, 10, 11, 13)
FREE_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "grasp")
CONFIG_BOUNDS = "configs/apple_wm_v4.yaml"  # +-0.5 right arm, left arm and left grasp pinned
CLOCK_FREQUENCIES = 16  # sin/cos pairs at geometric periods from 2 to 2048 steps
CLOCK_SCALE = float(EXPERT_POLICY_STEPS)


def clock_features(step: int) -> np.ndarray:
    """33 features of the post-look step: step / 745, and sin/cos at 16 geometric periods from 2
    to 2048 steps."""
    periods = np.geomspace(2.0, 2048.0, CLOCK_FREQUENCIES)
    angle = 2.0 * np.pi * float(step) / periods
    return np.concatenate(([float(step) / CLOCK_SCALE], np.sin(angle), np.cos(angle))).astype(
        np.float32
    )


POLICY_INPUTS = {
    "estimates": 4,  # apple xy, plate xy (world frame, m), from the readout, fixed per attempt
    "clock": 1 + 2 * CLOCK_FREQUENCIES,
    "proprioception": 86,  # joint positions and velocities
    "palm_pose": 9,  # right palm position (3) and the first two rotation columns (6), by FK
}
POLICY_HEAD = {
    "layers": "LayerNorm -> Linear(132, 512) -> SiLU -> Linear(512, 512) -> SiLU -> "
    "Linear(512, 512) -> SiLU -> Linear(512, 7)",
    "loss": "mean squared error on the 7 free dimensions of the expert label (clipped to "
    "configured bounds), inputs standardised by train moments",
}
TRAINING = {
    "device": "mps",
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "final_learning_rate": 3e-5,  # cosine
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch": 256,
    "updates": 30_000,  # per training, from scratch on the aggregate
    "selection_every": 1_000,
    "selection_min_output_std": 0.02,
}
DAGGER_ITERATIONS = 3
DAGGER_RESETS_PER_ITERATION = 128
SIM_WORKERS = 8

# ----- perception readout ------------------------------------------------------------------------
READOUT_FEATURE = "tokens"  # DINOv2 ViT-S/14 final patch tokens, 256 x 384, TASK-063's P-tok
READOUT_FAMILY = "TASK-063 probe: linear and RBF kernel ridge, family and lambda by inner CV"
READOUT_FOLDS = 10


# ----- C0: the expert's tolerance curve, and the perception bars it sets -------------------------
C0_APPLE_LEVELS_CM = (0.5, 0.8, 1.0, 1.2)
C0_PLATE_LEVELS_CM = (1.0, 1.5, 2.0, 2.5)
C0_RESETS = 32
C0_MIN_SUCCESSES = 28  # of 32, at every level up to the bar
APPLE_MEDIAN_CAP_CM = 0.75
APPLE_P90_CAP_CM = 1.2
PLATE_MEDIAN_CAP_CM = 1.5
PLATE_P90_CAP_CM = 2.5


def _largest_passing(levels, successes) -> float | None:
    bar = None
    for level in levels:
        if successes[level] < C0_MIN_SUCCESSES:
            break
        bar = level
    return bar


def c0_bars(reference_successes: int, apple: dict, plate: dict) -> dict:
    """The S0-P bars from C0. Bars can only tighten below the caps, never loosen.

    ``apple``/``plate`` map each level (cm) to successes of 32. The p90 bar is the largest level
    such that it and every smaller level reach 28/32; the median bar is min(cap, p90 bar).
    Returns ``{"escalate": reason}`` when no bar is feasible or the reference fails."""
    if set(apple) != set(C0_APPLE_LEVELS_CM) or set(plate) != set(C0_PLATE_LEVELS_CM):
        raise ContractError("C0 needs every preregistered level")
    if reference_successes < C0_MIN_SUCCESSES:
        return {"escalate": "C0 reference (no injected error) below 28/32"}
    apple_p90 = _largest_passing(C0_APPLE_LEVELS_CM, apple)
    plate_p90 = _largest_passing(C0_PLATE_LEVELS_CM, plate)
    if apple_p90 is None:
        return {"escalate": "C0: the smallest apple level is below 28/32"}
    if plate_p90 is None:
        return {"escalate": "C0: the smallest plate level is below 28/32"}
    apple_p90 = min(apple_p90, APPLE_P90_CAP_CM)
    plate_p90 = min(plate_p90, PLATE_P90_CAP_CM)
    return {
        "apple_median_cm": min(APPLE_MEDIAN_CAP_CM, apple_p90),
        "apple_p90_cm": apple_p90,
        "plate_median_cm": min(PLATE_MEDIAN_CAP_CM, plate_p90),
        "plate_p90_cm": plate_p90,
    }


def s0_perception(apple_errors_cm, plate_errors_cm, bars: dict) -> dict:
    """S0-P on the held-out perception resets: median and p90 within the C0 bars."""
    apple = np.asarray(apple_errors_cm, float)
    plate = np.asarray(plate_errors_cm, float)
    if apple.size != len(seeds_of("perception_heldout")) or plate.size != apple.size:
        raise ContractError("S0-P is scored on exactly the held-out perception resets")
    if not (np.isfinite(apple).all() and np.isfinite(plate).all()):
        raise GuardError("G-finite: a perception error is not finite")
    values = {
        "apple_median_cm": float(np.median(apple)),
        "apple_p90_cm": float(np.percentile(apple, 90)),
        "plate_median_cm": float(np.median(plate)),
        "plate_p90_cm": float(np.percentile(plate, 90)),
    }
    return {
        **values,
        "apple_passes": values["apple_median_cm"] <= bars["apple_median_cm"]
        and values["apple_p90_cm"] <= bars["apple_p90_cm"],
        "plate_passes": values["plate_median_cm"] <= bars["plate_median_cm"]
        and values["plate_p90_cm"] <= bars["plate_p90_cm"],
    }


# ----- arms, enumerated by name (the lesson of TASK-056 §15) -------------------------------
LADDER = {
    "L1": "learned policy",
    "L2": "partially learned",
    "L3": "learned perception, scripted control (not learned)",
    "L4": "privileged, scripted, replayed or substituted (not learned)",
}
ARMS = {
    "P-0": ("L1", "P after behaviour cloning only"),
    "P-1": ("L1", "P after DAgger iteration 1"),
    "P-2": ("L1", "P after DAgger iteration 2"),
    "P-3": ("L1", "P after DAgger iteration 3"),
    "C-3": ("L1", "no-image control: estimates fixed to the train mean; own DAgger x3"),
    "R-3": ("L1", "random-init DINOv2 floor: same pipeline on seed-0 tokens; own DAgger x3"),
    "F": ("L2", "fallback: per-phase heads, phase from the collector's clock schedule"),
    "A4-look": ("L3", "readout estimates -> scripted apple_collector_policy"),
    "D-oracle-perc": ("L4", "P's carried head fed true apple and plate xy"),
    "B-oracle": ("L4", "scripted_oracle with the look (harness)"),
    "B-hold": ("L4", "hold (harness)"),
    "B-random": ("L4", "uniform random within configured bounds (harness)"),
    "B-replay": ("L4", "open-loop replay of the nearest successful non-aim train root"),
}
P_ARMS = ("P-0", "P-1", "P-2", "P-3")
LEARNED_ARMS = ("P-0", "P-1", "P-2", "P-3", "C-3", "R-3")  # "learned arm" means exactly these
M1_ARMS = (*LEARNED_ARMS, "A4-look", "D-oracle-perc", "B-oracle", "B-hold", "B-random", "B-replay")
A4_MOTOR_THRESHOLD = 8  # of 16
ORACLE_MIN_SUCCESSES = 14  # of 16
D_RESETS = len(COHORT_D)

ROWS = (
    "V",
    "CAL-ESCALATE",
    "S0-APPLE-FAIL",
    "S0-PLATE-FAIL",
    "M1-PASS",
    "M1-MOTOR-F-PARTIAL",
    "M1-MOTOR-F-NONE",
    "M1-PERCEPTION",
)
CLAUSE_ROWS = frozenset({"S0-APPLE-FAIL", "M1-MOTOR-F-NONE", "M1-PERCEPTION"})


def harness_valid(counts: dict) -> bool:
    """B-oracle >= 14/16 successes; B-hold and B-random 0/16 grasps."""
    return (
        counts["B-oracle"]["success"] >= ORACLE_MIN_SUCCESSES
        and counts["B-hold"]["grasp"] == 0
        and counts["B-random"]["grasp"] == 0
    )


def carried_arm(counts: dict) -> str:
    """The P-k carried to M2: most D successes, ties to the later k (a selection on D)."""
    return max(P_ARMS, key=lambda arm: (counts[arm]["success"], P_ARMS.index(arm)))


def decide_m1(counts: dict, *, f_counts: dict | None = None) -> dict:
    """First-matching M1 row from per-arm {"grasp", "success"} counts on the 16 D resets.

    Called only after C0 and S0 passed and the harness is valid (otherwise V or an S0 row)."""
    for arm in M1_ARMS:
        cell = counts.get(arm)
        if cell is None or not (0 <= cell["success"] <= cell["grasp"] <= D_RESETS):
            raise ContractError(f"M1 needs a valid count for {arm}")
    if not harness_valid(counts):
        return {"row": "V", "void_reason": "harness: B-oracle, B-hold or B-random out of bounds"}
    if any(counts[arm]["success"] >= 1 for arm in P_ARMS):
        return {"row": "M1-PASS", "carried": carried_arm(counts), "clause_fires": False}
    if counts["A4-look"]["success"] >= A4_MOTOR_THRESHOLD:
        if f_counts is None:
            return {"row": "M1-MOTOR", "run_F": True, "clause_fires": False}
        if not 0 <= f_counts["success"] <= f_counts["grasp"] <= D_RESETS:
            raise ContractError("F needs a valid count")
        if f_counts["success"] >= 1:
            return {"row": "M1-MOTOR-F-PARTIAL", "clause_fires": False}
        return {"row": "M1-MOTOR-F-NONE", "clause_fires": True}
    return {"row": "M1-PERCEPTION", "clause_fires": True}


# ----- M2 on cohort C (separate authorization; preregistered now, run later) ---------------------
M2 = {
    "G1_min_successes": 17,  # of 40, and strictly more than B-replay on the same resets
    "G2_min_difference_vs_C": 8,  # carried P minus C-3, exact McNemar reported
    "G3_min_difference_vs_R": 8,  # carried P minus R-3
    "G4_min_grasps": 20,  # of 40
    "G5_harness": {"B-hold_grasp": 0, "B-random_grasp": 0, "B-oracle_min_successes": 38},
    "G6_privileged_reads": 0,
    "G7_max_median_control_seconds": 0.100,
    "stop_rule": "an arm with 0/16 grasps on D does not run on C",
    "cohort_values": "stored in benchmarks/manifests/apple-policy-v1.json, never recomputed",
}

# ----- budget and caps ---------------------------------------------------------------------------
CAPS_SECONDS = {
    "global": 36_000.0,
    "per_training": 1_800.0,
    "per_rollout_batch": 3_600.0,
    "c0": 3_600.0,
    "perception_collection": 1_800.0,
    "per_attempt": ATTEMPT_WALL_SECONDS,
}
FEATURE_DEVICE = "cpu"  # DINOv2 forward passes, as TASK-063 to TASK-066
TRAIN_DEVICE = "mps"
ROLLOUT_DEVICE = "cpu"  # policy inference inside the simulation workers


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, frozenset):
        return [_plain(v) for v in sorted(value)]
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def frozen_block() -> dict:
    """Everything this module freezes, as plain JSON types. The manifest's ``frozen`` block must
    equal it exactly; a test pins that."""
    return _plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "owner_rulings_utc": OWNER_RULINGS_UTC,
            "dataset": DATASET,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "read_splits": READ_SPLITS,
            "look_steps": LOOK_STEPS,
            "decision_frame": DECISION_FRAME,
            "image_size": IMAGE_SIZE,
            "seed_ranges": SEED_RANGES,
            "reserved_unused": RESERVED_UNUSED,
            "cohort_D": COHORT_D,
            "cohort_C": COHORT_C,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "seeds": {
                "model": MODEL_SEED,
                "readout_folds": READOUT_FOLD_SEED,
                "c0_directions": C0_DIRECTION_SEED,
                "train_sampler": TRAIN_SAMPLER_SEED,
                "random_controller": RANDOM_CONTROLLER_SEED,
            },
            "phases": PHASES,
            "phase_budgets": PHASE_BUDGETS,
            "max_policy_steps": MAX_POLICY_STEPS,
            "free_indices": FREE_INDICES,
            "config_bounds": CONFIG_BOUNDS,
            "clock_frequencies": CLOCK_FREQUENCIES,
            "policy_inputs": POLICY_INPUTS,
            "policy_head": POLICY_HEAD,
            "training": TRAINING,
            "dagger": {
                "iterations": DAGGER_ITERATIONS,
                "resets_per_iteration": DAGGER_RESETS_PER_ITERATION,
            },
            "sim_workers": SIM_WORKERS,
            "readout": {
                "feature": READOUT_FEATURE,
                "family": READOUT_FAMILY,
                "folds": READOUT_FOLDS,
            },
            "c0": {
                "apple_levels_cm": C0_APPLE_LEVELS_CM,
                "plate_levels_cm": C0_PLATE_LEVELS_CM,
                "resets": C0_RESETS,
                "min_successes": C0_MIN_SUCCESSES,
                "caps_cm": {
                    "apple_median": APPLE_MEDIAN_CAP_CM,
                    "apple_p90": APPLE_P90_CAP_CM,
                    "plate_median": PLATE_MEDIAN_CAP_CM,
                    "plate_p90": PLATE_P90_CAP_CM,
                },
            },
            "ladder": LADDER,
            "arms": ARMS,
            "learned_arms": LEARNED_ARMS,
            "m1_arms": M1_ARMS,
            "a4_motor_threshold": A4_MOTOR_THRESHOLD,
            "oracle_min_successes": ORACLE_MIN_SUCCESSES,
            "rows": ROWS,
            "clause_rows": CLAUSE_ROWS,
            "m2": M2,
            "caps_seconds": CAPS_SECONDS,
            "devices": {
                "features": FEATURE_DEVICE,
                "training": TRAIN_DEVICE,
                "rollouts": ROLLOUT_DEVICE,
            },
        }
    )
