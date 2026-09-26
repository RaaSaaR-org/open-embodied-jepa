"""TASK-064: the look-prefix 112 px Apple->Plate corpus (``apple_look_corpus_v1``).

Protocol ``docs/experiments/apple_look_corpus_v1.md``. This module fixes, in code and before any
corpus seed is simulated, what the protocol preregisters: the seeds, the whole-reset splits, the
plan (noise, aim offsets, branches), the look prefix, the acceptance checks, the readability
gate and the outcome rows. NumPy only: it imports without torch or MuJoCo. The simulation lives in
``scripts/collect_apple_look.py``.

The corpus is produced by the PRIVILEGED scripted collector (``scripted.apple_collector_policy``,
which reads simulator truth at reset). Nothing in it is a learned-control result. **Learned
Apple->Plate is still 0 successes.** This module preregisters no control formulation and no world
model.

**The look prefix reads nothing about the reset.** Every episode starts with TASK-061's constant
8-command look (``observation_reprobe.look_sequence``), unperturbed, before the collector policy
takes over. Frame ``LOOK_STEPS`` (8) of every root episode is the post-look decision frame.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa import observation_reprobe as orp
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_look_corpus_v1"
TASK = "TASK-064"
COLLECTION = "apple_look_v1"
FROZEN_SEEDS = tuple(range(47000, 47200))
PILOT_SEEDS = tuple(range(47900, 47932))
SPLIT_SEED = 64
PLAN_SALT = 64  # per-root draws: default_rng(SeedSequence([seed, PLAN_SALT]))
# Seeds this corpus must never simulate (the reserved cohorts and every earlier range).
RESERVED_RANGES = {
    "apple_wide_v1_corpus_and_pilots": (48000, 48999),
    "cohort_D_wide_development": (45000, 45107),
    "wide_v3_cohort": (45200, 45207),
    "cohort_C": (45300, 45339),
    "final_cohort": (44000, 44019),
    "narrow_development": (43000, 43004),
    "earlier_train_val_test": (42000, 42031),
    "mechanics_probes": (41000, 41101),
    "mvp_frozen_resets": (20000, 20049),
    "grasp_closure_probes": (49000, 49199),
}

# ----- the TASK-047/048 reset rule and collector settings, unchanged (tests pin equality) -----
RESET_CENTERS = {"object_xy": (0.34, -0.18), "plate_xy": (0.49, -0.09)}
WIDE_JITTER_M = {"object_xy": 0.03, "plate_xy": 0.02}
IMAGE_SIZE = 112
FPS = 20
NOISE_LEVELS = (0, 1, 2, 3)
OU_THETA = 0.85
OU_SIGMA = {"translation": 0.04, "rotation": 0.025, "grasp": 0.1}
BURST_PROBABILITY = 0.01
BURST_COMMANDS = 4
AIM_OFFSET_EVERY = 5
AIM_OFFSET_M = (0.015, 0.03)
BRANCH_KINDS = ("noise_only", "shift_close", "weak_close", "early_lift", "open_during_lift")
BRANCHES_PER_ROOT = 3
BRANCH_PHASE = 2  # the policy's "close" phase: branches start at the pre-grasp state
BRANCH_END_PHASE = 4  # stop after "lift"
POLICY_MAX_COMMANDS = 745  # the collector's full phase budget, after the look
LOWER = np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32)
UPPER = np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32)
MIN_STORED_TRANSITIONS = 8

# ----- the look prefix (TASK-061 §3.1, unchanged) ---------------------------------------------
LOOK_STEPS = orp.LOOK_STEPS  # 8
LOOK_SEQUENCE_SHA256 = orp.LOOK_SEQUENCE_SHA256
DECISION_FRAME = LOOK_STEPS  # frame index of the post-look frame in every root episode
LOOK_PHASE_INDEX = -1  # ``collector__phase_index`` of the look commands
LOOK_STATE_TOLERANCE = 1e-6
LOOK_MOVE_TOLERANCE_M = 1e-6

# ----- stored streams --------------------------------------------------------------------------
CAMERAS = ("onboard_rgb",)  # 112 px only; no hand crop (protocol §4)
FRAME_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 3]

# ----- the readability gate (protocol §7) -------------------------------------------------------
READ_SPLITS = ("train", "val")  # the test split is never decoded
EXPECTED_READ_ROOTS = {"train": 170, "val": 20}
READ_ALPHA_ONE_SIDED = 0.025
READ_ARM = "P-cls"
READ_FLOOR = "R-cls"
REPORTED_SOURCES = ("P_tok", "R_tok", "P_mean", "R_mean", "L_raw")

# ----- budget (protocol §9) ---------------------------------------------------------------------
WORKERS = 12
SUPERVISOR_SECONDS = 5400.0
WORKER_SECONDS = 4500.0
GLOBAL_WALL_SECONDS = 10800.0
DEVICE = "cpu"

ROWS = ("V", "C-ACCEPT", "C-READ-FAIL", "C-DATA-FAIL")


# ----- plan ---------------------------------------------------------------------------------------
def wide_reset(seed: int) -> dict:
    """Identical to ``evaluate_apple.wide_reset`` (TASK-047); a test pins the equality."""
    rng = np.random.default_rng(seed)
    return {
        "object_xy": (
            np.array(RESET_CENTERS["object_xy"])
            + rng.uniform(-WIDE_JITTER_M["object_xy"], WIDE_JITTER_M["object_xy"], 2)
        ).tolist(),
        "plate_xy": (
            np.array(RESET_CENTERS["plate_xy"])
            + rng.uniform(-WIDE_JITTER_M["plate_xy"], WIDE_JITTER_M["plate_xy"], 2)
        ).tolist(),
    }


def split_assignment(seeds) -> dict:
    """Whole-reset (session) partition fixed before any outcome: 10 % val, 5 % test."""
    order = list(seeds)
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    n_val, n_test = max(1, round(0.10 * len(order))), max(1, round(0.05 * len(order)))
    return {
        seed: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, seed in enumerate(order)
    }


def branch_params(kind: str, rng) -> dict:
    if kind == "noise_only":
        return {}
    if kind == "shift_close":
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.025)
        return {"offset_xy_m": [radius * np.cos(angle), radius * np.sin(angle)]}
    if kind == "weak_close":
        return {"grasp_target": float(rng.uniform(-0.2, 0.8))}
    if kind == "early_lift":
        return {"close_commands": int(rng.integers(3, 31))}
    if kind == "open_during_lift":
        return {"open_after_lift_commands": int(rng.integers(2, 41))}
    raise ContractError(f"unknown branch kind {kind}")


def check_seeds(seeds) -> None:
    """Refuse any seed of a reserved range (the cohorts and every earlier corpus)."""
    for seed in seeds:
        for name, (low, high) in RESERVED_RANGES.items():
            if low <= int(seed) <= high:
                raise ContractError(f"seed {seed} lies in the reserved range {name}")


def make_plan(seeds) -> dict:
    check_seeds(seeds)
    splits = split_assignment(seeds)
    roots = []
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, PLAN_SALT]))
        aim = None
        if index % AIM_OFFSET_EVERY == AIM_OFFSET_EVERY - 1:
            angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(*AIM_OFFSET_M)
            aim = [radius * np.cos(angle), radius * np.sin(angle)]
        branches = []
        for b in range(BRANCHES_PER_ROOT):
            # As TASK-048 (review R1, B2): decouple branch kinds from aim-offset roots.
            slot = BRANCHES_PER_ROOT * index + b + index // AIM_OFFSET_EVERY
            kind = BRANCH_KINDS[slot % len(BRANCH_KINDS)]
            branches.append(
                {
                    "episode_id": f"look-{seed}-b{b}-{kind}",
                    "kind": kind,
                    "params": branch_params(kind, rng),
                    "noise_seed": int(rng.integers(2**31)),
                }
            )
        roots.append(
            {
                "episode_id": f"look-{seed}",
                "session_id": f"look-reset-{seed}",
                "seed": seed,
                "split": splits[seed],
                **wide_reset(seed),
                "noise_level": NOISE_LEVELS[index % len(NOISE_LEVELS)],
                "noise_seed": int(rng.integers(2**31)),
                "aim_offset_xy_m": aim,
                "branches": branches,
            }
        )
    return {
        "collection": COLLECTION,
        "task": TASK,
        "protocol": PROTOCOL,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "seeds": list(seeds),
        "reset_rule": "evaluate_apple.wide_reset (TASK-047): apple +-3 cm, plate +-2 cm",
        "reset_centers_xy": {k: list(v) for k, v in RESET_CENTERS.items()},
        "image_size": IMAGE_SIZE,
        "cameras": list(CAMERAS),
        "fps": FPS,
        "look": {
            "steps": LOOK_STEPS,
            "sequence_sha256_float32_bytes": LOOK_SEQUENCE_SHA256,
            "perturbed": False,
            "decision_frame": DECISION_FRAME,
        },
        "noise": {
            "levels": list(NOISE_LEVELS),
            "ou_theta": OU_THETA,
            "ou_sigma_per_level": OU_SIGMA,
            "burst_probability_per_level": BURST_PROBABILITY,
            "burst_commands": BURST_COMMANDS,
        },
        "aim_offset_every": AIM_OFFSET_EVERY,
        "aim_offset_m": list(AIM_OFFSET_M),
        "branch_kinds": list(BRANCH_KINDS),
        "branches_per_root": BRANCHES_PER_ROOT,
        "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "policy_max_commands": POLICY_MAX_COMMANDS,
        "split_seed": SPLIT_SEED,
        "plan_salt": PLAN_SALT,
        "roots": roots,
    }


def plan_bytes(plan: dict) -> bytes:
    """The bytes the collector writes as ``plan.json``."""
    return (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()


def plan_sha256(plan: dict) -> str:
    return hashlib.sha256(plan_bytes(plan)).hexdigest()


def plan_sha256_rounded(plan: dict) -> str:
    """Platform-robust plan hash: every float rounded to 1e-9 (TASK-048's rule; libm's sin/cos
    differ in the last ulp between macOS arm64 and Linux x86-64)."""

    def rounded(value):
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [rounded(v) for v in value]
        if isinstance(value, float):
            return round(value, 9)
        return value

    blob = json.dumps(rounded(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def read_roots(plan: dict) -> list[dict]:
    """The train + val roots the readability gate reads, sorted by seed (the TASK-059 order)."""
    return sorted((r for r in plan["roots"] if r["split"] in READ_SPLITS), key=lambda r: r["seed"])


# ----- the look, checked at collection time on every root -------------------------------------
def look_facts(applied, post_state, apple_xy_moves, plate_moves, contacts) -> dict:
    """Per-root look record. ``applied``: [LOOK_STEPS, 14] applied commands; the moves and
    contacts are one value per executed look command (taken after it)."""
    applied = np.asarray(applied, np.float64)
    requested = orp.look_sequence().astype(np.float64)
    complete = applied.shape == requested.shape
    return {
        "look_commands_executed": int(len(applied)),
        "applied_max_abs_minus_requested": (
            float(np.abs(applied - requested).max()) if complete else None
        ),
        "post_look_state": [float(v) for v in np.asarray(post_state, np.float64)],
        "apple_xy_move_max_m": float(max(apple_xy_moves)) if len(apple_xy_moves) else None,
        "plate_move_max_m": float(max(plate_moves)) if len(plate_moves) else None,
        "hand_contact": bool(any(contacts)),
        "look_steps_checked": int(len(apple_xy_moves)),
    }


def check_look_records(records: list[dict], planned_seeds) -> dict:
    """L1 over every root's collection-time look record. Returns the facts and ``ok``; ``ok``
    needs exactly one record per planned root seed."""
    planned = sorted(int(s) for s in planned_seeds)
    if not records:
        return {"ok": False, "reason": "no look records", "roots": 0}
    complete = [r for r in records if r["look_commands_executed"] == LOOK_STEPS]
    worst_applied = max((r["applied_max_abs_minus_requested"] for r in complete), default=None)
    states = np.array([r["post_look_state"] for r in complete], np.float64)
    spread = float(np.abs(states - states[0]).max()) if len(states) else None
    apple = max((r["apple_xy_move_max_m"] for r in complete), default=None)
    plate = max((r["plate_move_max_m"] for r in complete), default=None)
    touched = sum(bool(r["hand_contact"]) for r in records)
    checked_all = all(r["look_steps_checked"] == LOOK_STEPS for r in records)
    facts = {
        "roots": len(records),
        "roots_with_complete_look": len(complete),
        "applied_max_abs_minus_requested": worst_applied,
        "post_look_state_max_abs_spread": spread,
        "apple_xy_move_max_m": apple,
        "plate_move_max_m": plate,
        "hand_contact_roots": int(touched),
        "every_look_command_checked": bool(checked_all),
    }
    facts["expected_roots"] = len(planned)
    facts["records_match_planned_seeds"] = (
        sorted(int(r.get("seed", -1)) for r in records) == planned
    )
    facts["ok"] = bool(
        facts["records_match_planned_seeds"]
        and len(complete) == len(records)
        and worst_applied == 0.0
        and spread is not None
        and spread <= LOOK_STATE_TOLERANCE
        and apple is not None
        and apple <= LOOK_MOVE_TOLERANCE_M
        and plate is not None
        and plate <= LOOK_MOVE_TOLERANCE_M
        and touched == 0
        and checked_all
    )
    return facts


# ----- acceptance checks (protocol §6) ----------------------------------------------------------
THRESHOLDS = {
    "A1_root_episodes_min": 190,
    "A2_branch_episodes_min": 450,
    "A3_root_full_success_min": 80,
    "A4_grasp_phase_failure_fraction": (0.20, 0.80),
    "A5_grasp_phase_successes_min": 250,
    "A6_right_arm_sign_fraction_min": 0.03,
    "A7_off_script_fraction_min": 0.50,
    "A8_branch_pairs_distinct_fraction_min": 0.90,
}


def checks(result: dict) -> dict:
    """Every preregistered acceptance check; ``all_passed`` iff each holds."""
    t = THRESHOLDS
    cov = result.get("action_coverage") or {}
    audit = result.get("audit") or {}
    pairs = result.get("branch_pair_rgb_distinct_at_16") or {}
    low, high = t["A4_grasp_phase_failure_fraction"]
    items = {
        "A1_root_episodes": result["root_episodes"] >= t["A1_root_episodes_min"],
        "A2_branch_episodes": result["branch_episodes"] >= t["A2_branch_episodes_min"],
        "A3_root_full_success": result["root_full_success"] >= t["A3_root_full_success_min"],
        "A4_grasp_phase_failure_fraction": low <= result["grasp_phase_failure_fraction"] <= high,
        "A5_grasp_phase_successes": (
            result["grasp_phase_successes"] >= t["A5_grasp_phase_successes_min"]
        ),
        "A6_right_arm_both_signs": bool(cov)
        and min(cov["right_arm_positive_fraction_above_0_1"]) >= t["A6_right_arm_sign_fraction_min"]
        and min(cov["right_arm_negative_fraction_below_minus_0_1"])
        >= t["A6_right_arm_sign_fraction_min"],
        "A7_off_script_fraction": bool(cov)
        and cov["off_script_fraction"] >= t["A7_off_script_fraction_min"],
        "A8_branch_pairs_distinct": bool(pairs)
        and pairs["pairs"] > 0
        and pairs["distinct"] >= t["A8_branch_pairs_distinct_fraction_min"] * pairs["pairs"],
        "A9_no_split_leakage": result["sessions_spanning_splits"] == 0
        and result["sessions_off_plan"] == 0
        and bool(result["normalization_is_train_only"])
        and all(result["split_sessions"].get(k, 0) > 0 for k in ("train", "val", "test")),
        "A10_audit": bool(audit)
        and audit.get("episodes_decoded") == audit.get("episodes_expected")
        and audit.get("all_intervals_ok", False),
        "A11_labels_separated": not result["label_errors"]
        and bool(result["privileged_labels_gated"]),
        "A12_resolution": result["camera_shapes"] == {c: FRAME_SHAPE for c in CAMERAS},
        "A13_run_integrity": bool((result.get("integrity") or {}).get("ok", False)),
        "L1_look_prefix": bool((result.get("look") or {}).get("ok", False)),
    }
    return {"checks": items, "all_passed": all(items.values())}


def frame_interval_ok(intervals, fps: int = FPS, tol: float = 1e-6) -> bool:
    intervals = np.asarray(intervals, np.float64)
    return bool(intervals.size and np.all(np.abs(intervals - 1.0 / fps) < tol))


# ----- the readability gate (protocol §7) -------------------------------------------------------
def readability_passes(
    *, succeeds: bool, beats_floor: bool, p_arm: float, spurious: bool, finite: bool
) -> bool:
    """R1-R4 for P-cls on the train + val roots. A non-finite feature fails the gate."""
    return bool(
        finite and succeeds and beats_floor and p_arm <= READ_ALPHA_ONE_SIDED and not spurious
    )


# ----- guards (protocol §8): a failure voids the run --------------------------------------------
class GuardError(ContractError):
    """A §8 guard failed: the run is V and nothing in it is read."""


def check_render(stored_equal: list[bool], what: str) -> None:
    """Q-render: stored frames equal a fresh re-render on every read root."""
    if not stored_equal or not all(stored_equal):
        bad = sum(not x for x in stored_equal)
        raise GuardError(f"Q-render: {what} differs from a fresh re-render on {bad} roots")


def check_expert(recomputed, recorded, aim_offset, *, tol=1e-6) -> dict:
    """Q-expert: the recomputed post-look expert command equals the recorded first policy base
    action on every read root without an aim offset."""
    recomputed, recorded = np.asarray(recomputed, np.float64), np.asarray(recorded, np.float64)
    keep = ~np.asarray(aim_offset, bool)
    if not keep.any():
        raise GuardError("Q-expert: no root without an aim offset")
    worst = float(np.abs(recomputed[keep] - recorded[keep]).max())
    if not worst <= tol:
        raise GuardError(f"Q-expert: recorded post-look expert differs by {worst}")
    return {"roots_compared": int(keep.sum()), "max_abs_difference": worst}


def check_apple_label(labels_xy, truth_xy, *, tol=1e-6) -> float:
    worst = float(np.abs(np.asarray(labels_xy) - np.asarray(truth_xy)).max())
    if not worst <= tol:
        raise GuardError(f"Q-expert: the stored apple label differs from the reset by {worst} m")
    return worst


def check_read_split(roots, splits_by_episode: dict) -> None:
    """Q-split: exactly the planned train + val roots, each in its planned split, none test."""
    counts = {k: 0 for k in READ_SPLITS}
    for root in roots:
        split = root["split"]
        if split not in READ_SPLITS:
            raise GuardError(f"Q-split: {root['episode_id']} is {split}, not train or val")
        if root["episode_id"] not in splits_by_episode:
            # A planned train/val root that errored or never started: the gate cannot be
            # evaluated on the preregistered 190 roots, so the run is V (protocol §9).
            raise GuardError(f"Q-split: planned read root {root['episode_id']} is not stored")
        if splits_by_episode.get(root["episode_id"]) != split:
            raise GuardError(f"Q-split: {root['episode_id']} is not in its planned split")
        counts[split] += 1
    if counts != EXPECTED_READ_ROOTS:
        raise GuardError(f"Q-split: read roots {counts} != {EXPECTED_READ_ROOTS}")


# ----- decision (protocol §10) -----------------------------------------------------------------
def decide(*, void: bool, data_all_passed: bool, readability_passed: bool, look_ok: bool) -> dict:
    """First matching row: V, C-ACCEPT, C-READ-FAIL, C-DATA-FAIL.

    C-READ-FAIL (and its abandonment clause) needs the look check L1 to hold: if the look itself
    failed, a readability failure is confounded by a data defect and the row is C-DATA-FAIL."""
    if void:
        return {"outcome": "V"}
    if data_all_passed and not look_ok:
        raise ContractError("inconsistent inputs: L1 failed but every acceptance check passed")
    if data_all_passed and readability_passed:
        row = "C-ACCEPT"
    elif not readability_passed and look_ok:
        row = "C-READ-FAIL"
    else:
        row = "C-DATA-FAIL"
    also = []
    if row == "C-READ-FAIL" and not data_all_passed:
        also.append("C-DATA-FAIL")
    return {
        "outcome": row,
        "data_all_passed": bool(data_all_passed),
        "readability_passed": bool(readability_passed),
        "look_ok": bool(look_ok),
        "readability_failure_confounded_by_look": bool(not readability_passed and not look_ok),
        "also_matching_rows": also,
        "abandonment_clause_fires": row == "C-READ-FAIL",
        "corpus_accepted": row == "C-ACCEPT",
    }
