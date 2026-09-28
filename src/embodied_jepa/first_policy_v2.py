"""TASK-071 (``apple_first_policy_v2``): TASK-067's first-learned-policy design, carried to
``apple-to-plate-v2`` with the frozen e9 expert and a fresh e9 demonstration corpus.

Protocol ``docs/experiments/apple_first_policy_v2.md``. This module fixes in code, before any
corpus episode is collected or any policy is trained, what the protocol preregisters: the seeds,
the corpus plan, e9's schedule, the arms and their labels, the policy's inputs, the training and
DAgger schedule, the calibration rule for the perception bars (C0, with the TASK-070 plate
ceiling), the offline gates (S0), the development milestone rows (M1), the gated-stage gates on
cohort C (M2, a separate authorization), the budget and the abandonment clause. NumPy only: it
imports without torch or MuJoCo.

Everything not listed as a change in the protocol's deviation table is v1's
(``first_policy``), imported from there where the value is identical.

Success is ``apple_at_rest_v0`` (``at_rest.py``, TASK-068), evaluated after the task's 60-step
settle; the latched v1 scorer (``task.AppleToPlateTask``) is reported beside it.

**Learned Apple->Plate is still 0 successes.** A success of P on the development cohort would be
an existence result, labelled "learned policy with a DINOv2 encoder" -- never "LeWM driving the
robot" (owner ruling R4, carried).
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_first_policy_v2"
TASK = "TASK-071"
GuardError = fp.GuardError

# ----- task, success, expert (TASK-070, frozen there; carried unchanged) -----------------------
SCENE_VERSION = "apple_to_plate_v2"  # apple_to_plate_v2.SCENE_VERSION; a test pins equality
APPLE_CONTACT = {"condim": 6, "friction": (1.0, 0.01, 0.001)}
SUCCESS_METRIC = "apple_at_rest_v0"  # at_rest.AtRestThresholds().version
REPORTED_BESIDE = "latched v1 scorer: AppleToPlateTask per-step success at any step"
# Owner ruling T71-R1 (2026-09-28): a counted success is at rest AND a latched grasp before
# release. An at-rest attempt without one is recorded and reported per arm, never counted.
COUNTED_SUCCESS = (
    "apple_at_rest_v0 after the settle AND the latched scorer's grasp stage reached during the "
    "attempt's commands, before the settle (owner ruling T71-R1); at rest without that grasp is "
    "reported per arm and never counted"
)


def counted_success(at_rest: bool, grasp_before_settle: bool) -> bool:
    """T71-R1: the success every row, gate, milestone and claim counts."""
    return bool(at_rest) and bool(grasp_before_settle)


EXPERT = {"release_pitch_rad": 0.45, "release_dx": 0.015}  # e9 = apple_to_plate_v2.GATE_EXPERT
EXPERT_CLASS = "resting_expert.RestingPlaceExpert"
EXPERT_PHASES = (
    "orient",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "steady",
    "open",
    "clear",
    "retreat",
)
EXPERT_BUDGETS = (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
EXPERT_POLICY_STEPS = sum(EXPERT_BUDGETS)  # 725
MAX_POLICY_STEPS = 740  # resting_expert.EXPERT_BUDGET: the budget e9 was gated under
SETTLE_STEPS = 60  # at_rest.AtRestThresholds().settle_steps: arm still, hands open
SETTLE_PHASE = len(EXPERT_PHASES)  # the phase index recorded for settle steps
ATTEMPT_STEPS = MAX_POLICY_STEPS + SETTLE_STEPS  # 800, as v1's MAX_POLICY_STEPS
ATTEMPT_WALL_SECONDS = fp.ATTEMPT_WALL_SECONDS

# TASK-070 gated at-rest counts of e9 on 50600-50631 (report sha256 27543757...), used only for
# the plate ceiling below. Applying v1's C0 rule to these two levels gives a p90 bar of 1.5 cm.
TASK070_PLATE_AT_REST = {0.0: 32, 1.0: 30, 1.5: 28}
TASK070_REPORT_SHA256 = "27543757f0f1d7d3a00b2f6b58a7e394d526342d14a407c018a0a27816ad099f"
TASK070_PLATE_CEILING_CM = 1.5

# ----- seeds -------------------------------------------------------------------------------------
# Every gated range lies in 51000-51999, the development cohort D2 and the smoke seeds in
# 52000-52199. The repository search of 2026-09-28 (protocol §4.1) found no seed use there.
SEED_RANGES = {
    "corpus": (51000, 51199),
    "perception_train": (51200, 51455),
    "perception_heldout": (51456, 51583),
    "calibration_C0": (51584, 51615),
    "dagger_1": (51616, 51743),
    "dagger_2": (51744, 51871),
    "dagger_3": (51872, 51999),
}
GATED_BLOCK = (51000, 51999)
COHORT_D2 = tuple(range(52000, 52016))  # the fresh development cohort; never gates
RESERVED_UNUSED = (52016, 52099)
SMOKE_SEEDS = (52100, 52199)  # smoke runs only; nothing from them is read
TASK_BLOCK = (51000, 52199)
COHORT_C = fp.COHORT_C  # frozen, gating; M2 only, separate authorization; stored values
FORBIDDEN_RANGES = {
    **fp.FORBIDDEN_RANGES,
    "task067_block": (46000, 46999),
    "task068_to_070_block": (50000, 50999),  # dev 50000-50299, gate 50600-50631, 50500-50531
}
MODEL_SEED = 7104
READOUT_FOLD_SEED = 7101
C0_DIRECTION_SEED = 7100
TRAIN_SAMPLER_SEED = 7102
RANDOM_CONTROLLER_SEED = 7103
CORPUS_SPLIT_SEED = 7105
CORPUS_PLAN_SALT = 7106


def seeds_of(name: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[name]
    return tuple(range(low, high + 1))


def _spans() -> dict[str, tuple[int, int]]:
    spans = dict(SEED_RANGES)
    spans["cohort_D2"] = (COHORT_D2[0], COHORT_D2[-1])
    spans["reserved_unused"] = RESERVED_UNUSED
    spans["smoke"] = SMOKE_SEEDS
    return spans


def check_seed_ranges() -> None:
    """Every TASK-071 range is inside 51000-52199, pairwise disjoint, off every forbidden range
    and off cohort C; the gated ranges stay inside 51000-51999."""
    spans = _spans()
    ordered = sorted(spans.values())
    for (_, a_high), (b_low, _) in zip(ordered, ordered[1:], strict=False):
        if b_low <= a_high:
            raise GuardError("TASK-071 seed ranges overlap")
    for name, (low, high) in spans.items():
        if low > high or low < TASK_BLOCK[0] or high > TASK_BLOCK[1]:
            raise GuardError(f"the TASK-071 range {name} leaves {TASK_BLOCK}")
        for other, (f_low, f_high) in FORBIDDEN_RANGES.items():
            if low <= f_high and f_low <= high:
                raise GuardError(f"the TASK-071 range {name} overlaps {other}")
        if any(low <= s <= high for s in COHORT_C):
            raise GuardError(f"the TASK-071 range {name} overlaps cohort C")
    for name, (low, high) in SEED_RANGES.items():
        if low < GATED_BLOCK[0] or high > GATED_BLOCK[1]:
            raise GuardError(f"the gated range {name} leaves {GATED_BLOCK}")


# ----- the corpus (fresh; e9 under v2) -----------------------------------------------------------
CORPUS = "apple-look-v2"
CORPUS_SPLIT_FRACTIONS = {"val": 0.10, "test": 0.05}  # whole-reset sessions, as TASK-064
EXPECTED_CORPUS_SPLITS = {"train": 170, "val": 20, "test": 10}
READ_SPLITS = ("train", "val")  # the test split is never decoded
NOISE_LEVELS = (0, 1, 2, 3)  # TASK-048/064: root i gets level i % 4; OU + bursts as there
DECISION_FRAME = 0  # frame 0 of a stored v2 episode is the post-look frame
LOOK_STEPS = fp.LOOK_STEPS
IMAGE_SIZE = fp.IMAGE_SIZE
CAMERA = fp.CAMERA


def corpus_splits(seeds=None) -> dict[int, str]:
    """Whole-reset partition fixed before any episode: 10 % val, 5 % test, the rest train."""
    order = list(seeds_of("corpus") if seeds is None else seeds)
    np.random.default_rng(CORPUS_SPLIT_SEED).shuffle(order)
    n_val = max(1, round(CORPUS_SPLIT_FRACTIONS["val"] * len(order)))
    n_test = max(1, round(CORPUS_SPLIT_FRACTIONS["test"] * len(order)))
    return {
        int(seed): ("val" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, seed in enumerate(order)
    }


def corpus_plan(seeds=None) -> list[dict]:
    """One root per corpus seed: its split, noise level and noise seed (sorted by seed)."""
    seeds = tuple(seeds_of("corpus") if seeds is None else seeds)
    splits = corpus_splits(seeds)
    plan = []
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), CORPUS_PLAN_SALT]))
        plan.append(
            {
                "episode_id": f"look2-{seed}",
                "session_id": f"look2-reset-{seed}",
                "seed": int(seed),
                "split": splits[int(seed)],
                "noise_level": NOISE_LEVELS[index % len(NOISE_LEVELS)],
                "noise_seed": int(rng.integers(2**31)),
            }
        )
    return plan


# ----- e9's schedule (the clock the policy learns, and F's scripted switch) ---------------------
def scheduled_phase(step: int) -> int:
    """e9's phase index at post-look step ``step`` on its own clock (fallback F only).

    Steps at or beyond the budget map to the last phase."""
    if step < 0:
        raise ContractError("step counts from 0 after the look")
    edges = np.cumsum(EXPERT_BUDGETS)
    return int(min(np.searchsorted(edges, step, side="right"), len(EXPERT_PHASES) - 1))


CLOCK_FREQUENCIES = fp.CLOCK_FREQUENCIES
CLOCK_SCALE = float(EXPERT_POLICY_STEPS)


def clock_features(step: int) -> np.ndarray:
    """33 features of the post-look step: step / 725, and sin/cos at 16 geometric periods from 4
    to 2048 steps (v1's features, rescaled to e9's budget)."""
    periods = np.geomspace(4.0, 2048.0, CLOCK_FREQUENCIES)
    angle = 2.0 * np.pi * float(step) / periods
    return np.concatenate(([float(step) / CLOCK_SCALE], np.sin(angle), np.cos(angle))).astype(
        np.float32
    )


# ----- the policy (v1's, unchanged) --------------------------------------------------------------
FREE_INDICES = fp.FREE_INDICES
FREE_NAMES = fp.FREE_NAMES
CONFIG_BOUNDS = fp.CONFIG_BOUNDS
INPUT_STD_FLOOR = fp.INPUT_STD_FLOOR
POLICY_INPUTS = fp.POLICY_INPUTS
POLICY_HEAD = fp.POLICY_HEAD
TRAINING = fp.TRAINING
DAGGER_ITERATIONS = fp.DAGGER_ITERATIONS
DAGGER_RESETS_PER_ITERATION = fp.DAGGER_RESETS_PER_ITERATION
SIM_WORKERS = fp.SIM_WORKERS
WORKER_TORCH_THREADS = fp.WORKER_TORCH_THREADS
READOUT_FEATURE = fp.READOUT_FEATURE
READOUT_FAMILY = fp.READOUT_FAMILY
READOUT_FOLDS = fp.READOUT_FOLDS
PHASE_CLOSE = EXPERT_PHASES.index("close")  # BC mask: drift counts only before close
DISPLACEMENT_LIMIT_M = 0.01  # cloning.DISPLACEMENT_LIMIT_M

# ----- C0 and the perception bars -----------------------------------------------------------------
C0_APPLE_LEVELS_CM = fp.C0_APPLE_LEVELS_CM
C0_PLATE_LEVELS_CM = fp.C0_PLATE_LEVELS_CM
C0_RESETS = fp.C0_RESETS
C0_MIN_SUCCESSES = fp.C0_MIN_SUCCESSES


def task070_plate_ceiling() -> float:
    """v1's C0 rule applied to TASK-070's gated plate levels: the largest level such that it and
    every smaller one reach 28/32 at rest (1.0 cm: 30, 1.5 cm: 28 -> 1.5 cm)."""
    levels = sorted(k for k in TASK070_PLATE_AT_REST if k > 0)
    bar = None
    for level in levels:
        if TASK070_PLATE_AT_REST[level] < C0_MIN_SUCCESSES:
            break
        bar = level
    if bar is None:
        raise ContractError("TASK-070 gives no feasible plate level")
    return float(bar)


def c0_bars(reference_successes: int, apple: dict, plate: dict) -> dict:
    """v1's rule (``first_policy.c0_bars``, at-rest counts of the fresh C0), then the TASK-070
    plate ceiling: the plate p90 bar is at most 1.5 cm, and the median at most the p90. Bars
    only tighten; nothing here can loosen a v1 cap."""
    bars = fp.c0_bars(reference_successes, apple, plate)
    if "escalate" in bars:
        return bars
    ceiling = task070_plate_ceiling()
    plate_p90 = min(bars["plate_p90_cm"], ceiling)
    return {
        **bars,
        "plate_p90_cm": plate_p90,
        "plate_median_cm": min(bars["plate_median_cm"], plate_p90),
        "task070_plate_ceiling_cm": ceiling,
    }


def s0_perception(apple_errors_cm, plate_errors_cm, bars: dict) -> dict:
    """S0-P on the 128 held-out perception resets: median and p90 within the bars (v1's)."""
    return fp.s0_perception(apple_errors_cm, plate_errors_cm, bars)


def a4_threshold(reference_successes: int, apple: dict, plate: dict, apple_err, plate_err) -> int:
    """v1's calibrated A4-look trigger (rate cap 1.0), on the at-rest C0 counts."""
    return fp.a4_threshold(reference_successes, apple, plate, apple_err, plate_err)


# ----- arms, enumerated by name --------------------------------------------------------------
LADDER = fp.LADDER
LADDER_ALLOWANCES_L1 = {
    **fp.LADDER_ALLOWANCES_L1,
    "f": "the task's 60-step settle after the policy's commands (arm still, hands open): part "
    "of the apple_at_rest_v0 scoring procedure, identical for every arm, never learned",
}
ARMS = {
    "P-0": ("L1", "P after behaviour cloning only"),
    "P-1": ("L1", "P after DAgger iteration 1"),
    "P-2": ("L1", "P after DAgger iteration 2"),
    "P-3": ("L1", "P after DAgger iteration 3"),
    "C-3": ("L1", "no-image control: estimates fixed to the train mean; own DAgger x3"),
    "R-3": ("L1", "random-init DINOv2 floor: same pipeline on seed-0 tokens; own DAgger x3"),
    "F": ("L2", "fallback: per-phase heads, phase from e9's clock schedule"),
    "A4-look": ("L3", "readout estimates -> scripted e9"),
    "D-oracle-perc": ("L4", "P's carried head fed true apple and plate xy"),
    "B-oracle": ("L4", "e9 from the reset truth (harness)"),
    "B-hold": ("L4", "hold (harness)"),
    "B-random": ("L4", "uniform random within configured bounds (harness)"),
    "B-replay": ("L4", "open-loop replay of the nearest at-rest train root of apple-look-v2"),
}
P_ARMS = fp.P_ARMS
LEARNED_ARMS = fp.LEARNED_ARMS
M1_ARMS = fp.M1_ARMS
A4_THRESHOLD_FRACTION = fp.A4_THRESHOLD_FRACTION
A4_THRESHOLD_MINIMUM = fp.A4_THRESHOLD_MINIMUM
A4_RATE_CAP = fp.A4_RATE_CAP
D_RESETS = len(COHORT_D2)
ORACLE_MIN_SUCCESSES = fp.ORACLE_MIN_SUCCESSES  # of 16, counted successes (T71-R1)

ROWS = fp.ROWS
INTERMEDIATE_STATES = fp.INTERMEDIATE_STATES
DECLARED_EARLY_STOPS = fp.DECLARED_EARLY_STOPS
VOID_RULE = (
    "a run that stops early other than at a declared early-stop row (CAL-ESCALATE, "
    "S0-APPLE-FAIL, S0-PLATE-FAIL) is V; exactly one from-scratch repeat with the same seeds, "
    "caps and device (a fresh corpus directory, collected again from the same plan); a second V "
    "closes TASK-071 as INCONCLUSIVE"
)
NO_ELIGIBLE_CHECKPOINT = "the arm scores 0/16 on D2, reported as no_eligible_checkpoint"
CLAUSE_ROWS = fp.CLAUSE_ROWS


def _check_cell(arm: str, cell) -> None:
    """v1's consistency check, restored by T71-R1: 0 <= counted success <= grasp <= 16."""
    if cell is None or not 0 <= int(cell["success"]) <= int(cell["grasp"]) <= D_RESETS:
        raise ContractError(f"M1 needs a valid count for {arm}")


def harness_valid(counts: dict) -> bool:
    """B-oracle >= 14/16 counted successes (T71-R1); B-hold and B-random 0/16 grasps."""
    return (
        counts["B-oracle"]["success"] >= ORACLE_MIN_SUCCESSES
        and counts["B-hold"]["grasp"] == 0
        and counts["B-random"]["grasp"] == 0
    )


def carried_arm(counts: dict) -> str:
    """The P-k carried to M2: most D2 at-rest successes, ties to the later k (a selection on D2)."""
    return max(P_ARMS, key=lambda arm: (counts[arm]["success"], P_ARMS.index(arm)))


def decide_m1(counts: dict, a4_threshold_value: int, *, f_counts: dict | None = None) -> dict:
    """v1's first-matching M1 rows (``first_policy.decide_m1``), with ``success`` the counted
    success of T71-R1 (at rest AND a latched grasp before the settle), so success <= grasp as in
    v1. Called only after C0 and S0 passed."""
    for arm in M1_ARMS:
        _check_cell(arm, counts.get(arm))
    if not harness_valid(counts):
        return {"row": "V", "void_reason": "harness: B-oracle, B-hold or B-random out of bounds"}
    if any(counts[arm]["success"] >= 1 for arm in P_ARMS):
        return {"row": "M1-PASS", "carried": carried_arm(counts), "clause_fires": False}
    if not isinstance(a4_threshold_value, int) or a4_threshold_value < A4_THRESHOLD_MINIMUM:
        raise ContractError("the A4-look threshold is calibrated and at least 1")
    if counts["A4-look"]["success"] >= a4_threshold_value:
        if f_counts is None:
            return {"row": "M1-MOTOR", "run_F": True, "clause_fires": False}
        _check_cell("F", f_counts)
        if f_counts["success"] >= 1:
            return {"row": "M1-MOTOR-F-PARTIAL", "clause_fires": False}
        return {"row": "M1-MOTOR-F-NONE", "clause_fires": True}
    return {"row": "M1-PERCEPTION", "clause_fires": True}


# ----- M2 on cohort C (separate authorization; preregistered now, run later) ---------------------
M2 = {
    **fp.M2,
    "success": COUNTED_SUCCESS,
    "G5_harness": {"B-hold_grasp": 0, "B-random_grasp": 0, "B-oracle_min_successes": 38},
    "stop_rule": (
        "the carried P-k does not run on C if it has 0/16 grasps on D2 (then M2 fails); the "
        "controls C-3, R-3, B-replay and the harness always run on C"
    ),
    "G5_oracle": "e9 from the reset truth, counted success (T71-R1)",
    "rows": {
        **fp.M2["rows"],
        "M2-PASS": "every gate passes: a learned policy with a DINOv2 encoder works on this "
        "cohort under apple-to-plate-v2 (not LeWM driving the robot)",
    },
}

# ----- budget and caps ---------------------------------------------------------------------------
CAPS_SECONDS = {
    **fp.CAPS_SECONDS,
    "corpus_collection": 3_600.0,
    "per_attempt": ATTEMPT_WALL_SECONDS,
}
FEATURE_DEVICE = fp.FEATURE_DEVICE
TRAIN_DEVICE = fp.TRAIN_DEVICE
ROLLOUT_DEVICE = fp.ROLLOUT_DEVICE
RENDER_CHECK = (
    "before the pre-run GO, the pre-run reviewer confirms from scripts/check_first_policy_v2_"
    "render.py at the gated worker count (8) that the post-look re-render is bit-identical "
    "(verdict IDENTICAL); otherwise the run does not start and the issue goes to the owner"
)


def frozen_block() -> dict:
    """Everything this module freezes, as plain JSON types. The manifest's ``frozen`` block must
    equal it exactly; a test pins that."""
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "carried_from": fp.PROTOCOL,
            "scene_version": SCENE_VERSION,
            "apple_contact": APPLE_CONTACT,
            "success_metric": SUCCESS_METRIC,
            "counted_success": COUNTED_SUCCESS,
            "reported_beside": REPORTED_BESIDE,
            "expert": {"class": EXPERT_CLASS, "kwargs": EXPERT},
            "expert_phases": EXPERT_PHASES,
            "expert_budgets": EXPERT_BUDGETS,
            "expert_policy_steps": EXPERT_POLICY_STEPS,
            "max_policy_steps": MAX_POLICY_STEPS,
            "settle_steps": SETTLE_STEPS,
            "attempt_steps": ATTEMPT_STEPS,
            "task070_plate_at_rest": {str(k): v for k, v in TASK070_PLATE_AT_REST.items()},
            "task070_report_sha256": TASK070_REPORT_SHA256,
            "task070_plate_ceiling_cm": TASK070_PLATE_CEILING_CM,
            "seed_ranges": SEED_RANGES,
            "gated_block": GATED_BLOCK,
            "cohort_D2": COHORT_D2,
            "reserved_unused": RESERVED_UNUSED,
            "smoke_seeds": SMOKE_SEEDS,
            "task_block": TASK_BLOCK,
            "cohort_C": COHORT_C,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "seeds": {
                "model": MODEL_SEED,
                "readout_folds": READOUT_FOLD_SEED,
                "c0_directions": C0_DIRECTION_SEED,
                "train_sampler": TRAIN_SAMPLER_SEED,
                "random_controller": RANDOM_CONTROLLER_SEED,
                "corpus_split": CORPUS_SPLIT_SEED,
                "corpus_plan_salt": CORPUS_PLAN_SALT,
            },
            "corpus": {
                "name": CORPUS,
                "split_fractions": CORPUS_SPLIT_FRACTIONS,
                "expected_splits": EXPECTED_CORPUS_SPLITS,
                "read_splits": READ_SPLITS,
                "noise_levels": NOISE_LEVELS,
                "decision_frame": DECISION_FRAME,
            },
            "look_steps": LOOK_STEPS,
            "image_size": IMAGE_SIZE,
            "camera": CAMERA,
            "free_indices": FREE_INDICES,
            "config_bounds": CONFIG_BOUNDS,
            "clock_frequencies": CLOCK_FREQUENCIES,
            "clock_scale": CLOCK_SCALE,
            "policy_inputs": POLICY_INPUTS,
            "policy_head": POLICY_HEAD,
            "training": TRAINING,
            "dagger": {
                "iterations": DAGGER_ITERATIONS,
                "resets_per_iteration": DAGGER_RESETS_PER_ITERATION,
            },
            "sim_workers": SIM_WORKERS,
            "worker_torch_threads": WORKER_TORCH_THREADS,
            "readout": {
                "feature": READOUT_FEATURE,
                "family": READOUT_FAMILY,
                "folds": READOUT_FOLDS,
            },
            "bc_mask": {
                "phase_close": PHASE_CLOSE,
                "displacement_limit_m": DISPLACEMENT_LIMIT_M,
            },
            "c0": {
                "apple_levels_cm": C0_APPLE_LEVELS_CM,
                "plate_levels_cm": C0_PLATE_LEVELS_CM,
                "resets": C0_RESETS,
                "min_successes": C0_MIN_SUCCESSES,
                "caps_cm": {
                    "apple_median": fp.APPLE_MEDIAN_CAP_CM,
                    "apple_p90": fp.APPLE_P90_CAP_CM,
                    "plate_median": fp.PLATE_MEDIAN_CAP_CM,
                    "plate_p90": fp.PLATE_P90_CAP_CM,
                },
                "success": COUNTED_SUCCESS,
            },
            "ladder": LADDER,
            "ladder_allowances_L1": LADDER_ALLOWANCES_L1,
            "arms": ARMS,
            "learned_arms": LEARNED_ARMS,
            "m1_arms": M1_ARMS,
            "a4_threshold": {
                "rule": "ceil(0.5 x 16 x mean predicted rate from C0 and S0-P), at least 1",
                "fraction": A4_THRESHOLD_FRACTION,
                "minimum": A4_THRESHOLD_MINIMUM,
                "rate_cap": A4_RATE_CAP,
            },
            "input_std_floor": INPUT_STD_FLOOR,
            "oracle_min_successes": ORACLE_MIN_SUCCESSES,
            "rows": ROWS,
            "intermediate_states": INTERMEDIATE_STATES,
            "declared_early_stops": DECLARED_EARLY_STOPS,
            "clause_rows": CLAUSE_ROWS,
            "void_rule": VOID_RULE,
            "no_eligible_checkpoint": NO_ELIGIBLE_CHECKPOINT,
            "m2": M2,
            "caps_seconds": CAPS_SECONDS,
            "devices": {
                "features": FEATURE_DEVICE,
                "training": TRAIN_DEVICE,
                "rollouts": ROLLOUT_DEVICE,
            },
            "render_check": RENDER_CHECK,
        }
    )
