"""TASK-068: the privileged scripted expert that rests the apple on the plate, and its harness.

Protocol: ``docs/experiments/apple_resting_expert_v1.md``. This is engineering on a privileged
scripted controller (``scripted.RestingPlaceExpert``); it is not a learned result. NumPy at
import; MuJoCo is imported lazily by the functions that run the simulator.

One attempt is: reset (the TASK-047 wide-jitter distribution), the TASK-064 look, the expert
built from the reset truth with the plate xy shifted by a fixed offset (the TASK-067 C0 plate
perturbation, ``first_policy_runtime.perturbed_truth``), the expert's commands, then a settle of
``AtRestThresholds.settle_steps`` steps (arm still, hands open). Two readings are recorded:

* **at rest** (``task.apple_at_rest``, version ``apple_at_rest_v0``) over the final window;
* the **latched scorer**: ``AppleToPlateTask``'s per-step success at any step of the attempt, as
  C0 and the TASK-067 probes counted it (they stopped at the first success).
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa.task import AtRestThresholds

# ----- seeds ------------------------------------------------------------------------------------
# Fixed here before the first logged development run (protocol §2). The repository search of
# 2026-09-28 found 50000-50999 unused by any corpus, cohort, pilot, probe or calibration. Every
# TASK-068 wide-jitter reset so far, including single-seed exploration, used a seed in this
# range (the descent diagnosis's reach map and rolling check use a fixed scene, not a reset seed).
DEV_SEEDS = tuple(range(50000, 50100))
DEV_DIRECTION_SEED = 6830
PLATE_LEVELS_CM = (0.0, 1.0, 1.5)
EXPERT_BUDGET = 740  # commands after the look; with the settle this is fp.MAX_POLICY_STEPS
SETTLE_STEPS = AtRestThresholds().settle_steps
TASK067_BLOCK = (46000, 46999)  # every TASK-067 range, spent, reserved or smoke


def forbidden_ranges() -> dict[str, tuple[int, int]]:
    """Every seed range a TASK-068 range must avoid (TASK-067's list plus its whole block)."""
    return {**fp.FORBIDDEN_RANGES, "task067_block": TASK067_BLOCK}


def check_seeds(seeds, *, others=()) -> None:
    """``seeds`` avoid every forbidden range and every range in ``others``."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise fp.GuardError("TASK-068 seeds must be non-empty and distinct")
    spans = dict(forbidden_ranges())
    for i, other in enumerate(others):
        spans[f"other_{i}"] = (min(other), max(other))
    for name, (low, high) in spans.items():
        if any(low <= s <= high for s in seeds):
            raise fp.GuardError(f"TASK-068 seeds overlap {name}")
    for s in set(fp.COHORT_C) | set(fp.COHORT_D):
        if s in seeds:
            raise fp.GuardError("TASK-068 seeds overlap a TASK-067 cohort")


def plate_offsets(count: int, level_cm: float, direction_seed: int) -> np.ndarray:
    """Fixed-size plate offsets (metres), one direction per seed index, as the TASK-067 probes."""
    angles = np.random.default_rng(direction_seed).uniform(0, 2 * np.pi, count)
    return level_cm / 100.0 * np.c_[np.cos(angles), np.sin(angles)]


# ----- contacts ---------------------------------------------------------------------------------
PAIR_TYPES = ("apple_hand", "apple_plate_base", "apple_plate_rim", "hand_plate", "apple_table")


def _geom_class(model, geom_id: int) -> str:
    name = model.geom(geom_id).name
    body = model.body(int(model.geom_bodyid[geom_id])).name
    if name == "apple_geom":
        return "apple"
    if name == "plate_base":
        return "plate_base"
    if name.startswith("plate_rim"):
        return "plate_rim"
    if name == "table":
        return "table"
    if "hand_" in body or "wrist_" in body:
        return "hand"
    return "other"


def contact_counts(sim) -> list[int]:
    counts = dict.fromkeys(PAIR_TYPES, 0)
    for c in sim.data.contact[: sim.data.ncon]:
        pair = {_geom_class(sim.model, int(c.geom1)), _geom_class(sim.model, int(c.geom2))}
        if pair == {"apple", "hand"}:
            counts["apple_hand"] += 1
        elif pair == {"apple", "plate_base"}:
            counts["apple_plate_base"] += 1
        elif pair == {"apple", "plate_rim"}:
            counts["apple_plate_rim"] += 1
        elif "hand" in pair and pair & {"plate_base", "plate_rim"}:
            counts["hand_plate"] += 1
        elif pair == {"apple", "table"}:
            counts["apple_table"] += 1
    return [counts[k] for k in PAIR_TYPES]


# ----- one attempt ------------------------------------------------------------------------------
def run_attempt(
    robot,
    bounds,
    *,
    seed: int,
    reset: dict,
    plate_offset,
    make_expert,
    budget: int = EXPERT_BUDGET,
    thresholds: AtRestThresholds | None = None,
) -> tuple[dict, dict]:
    """One attempt. Returns ``(summary, arrays)``; the arrays are per executed step."""
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa.contracts import ContractError
    from embodied_jepa.task import AppleAtRestCheck

    cfg = thresholds or AtRestThresholds()
    lower, upper = bounds
    started = time.monotonic()
    truth, scorer, counter, _obs, _facts = rt.reset_and_look(robot, int(seed), reset)
    counter.remove()  # privileged expert and harness: truth is read by design, for scoring
    perturbed = rt.perturbed_truth(truth, [0.0, 0.0], plate_offset)
    expert = make_expert(perturbed)
    if expert.max_steps > budget:
        raise fp.GuardError(f"the expert's {expert.max_steps} commands exceed the {budget} budget")
    check = AppleAtRestCheck(robot, cfg)
    sim = robot.sim
    names = [p.name for p in expert.phases]
    rows = {
        k: []
        for k in (
            "phase",
            "command",
            "applied",
            "palm",
            "apple_pos",
            "apple_lin",
            "apple_ang",
            "counts",
            "success",
            "distance",
        )
    }
    stop_reason, settle_left = "complete", cfg.settle_steps
    while settle_left > 0:
        robot.observe()
        active = not expert.done
        if active:
            phase = expert.phase_index
            command = np.asarray(expert.action(robot), np.float32)
        else:
            phase = len(names)  # settle
            command = np.zeros(14, np.float32)
            command[12:] = -1.0
        command = np.clip(command, lower, upper).astype(np.float32)
        try:
            projection = robot.project_candidates(command[None, None, None])
        except ContractError as error:
            stop_reason = f"guard: {error}"
            break
        if not bool(projection.feasible[0, 0]):
            stop_reason = "infeasible"
            break
        result = robot.execute(projection.actions[0, 0, 0])
        if result.applied_action is None:
            stop_reason = f"rejected: {result.reason}"
            break
        if active:
            expert.advance(result)
        else:
            settle_left -= 1
        score = scorer.evaluate()
        check.record()
        joint = sim.data.joint("apple_free")
        position, _rotation = robot.ee_pose("right")
        rows["phase"].append(phase)
        rows["command"].append(command)
        rows["applied"].append(np.asarray(result.applied_action, np.float32))
        rows["palm"].append(position.copy())
        rows["apple_pos"].append(sim.data.body("apple").xpos.copy())
        rows["apple_lin"].append(joint.qvel[:3].copy())
        rows["apple_ang"].append(joint.qvel[3:6].copy())
        rows["counts"].append(contact_counts(sim))
        rows["success"].append(bool(score["success"]))
        rows["distance"].append(float(score["object_plate_distance_m"]))
    robot.stop(stop_reason)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    complete = stop_reason == "complete"
    verdict = check.verdict() if complete else None
    plate = np.asarray(truth["plate_position"], float)
    base = np.asarray(truth["base_position_world"], float)
    rest_z = float(truth["container_surface_z"] + truth["object_support_height"])
    summary = {
        "seed": int(seed),
        "stop_reason": stop_reason,
        "complete": complete,
        "steps": len(arrays["phase"]),
        "expert_commands": int(expert.step_count),
        "at_rest": bool(verdict["at_rest"]) if verdict else False,
        "at_rest_detail": verdict,
        "latched_success": bool(arrays["success"].any()) if len(arrays["phase"]) else False,
        "first_success_step": (
            int(np.flatnonzero(arrays["success"])[0]) if arrays["success"].any() else None
        ),
        "final_distance_cm": float(arrays["distance"][-1] * 100) if len(arrays["phase"]) else None,
        "seconds": time.monotonic() - started,
        **release_events(arrays, names, plate, base, rest_z),
    }
    if hasattr(expert, "release_target"):
        summary["release_target_base"] = np.asarray(expert.release_target).round(4).tolist()
    return summary, arrays


def _first(mask) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def release_events(a: dict, names, plate, base, rest_z) -> dict:
    """Release and landing events read from the per-step arrays; none of them is a verdict."""
    if not len(a["phase"]) or "open" not in names:
        return {}
    idx = {k: PAIR_TYPES.index(k) for k in PAIR_TYPES}
    start = _first(a["phase"] == names.index("open"))
    if start is None:
        return {"open_step": None}
    counts = a["counts"][start:]
    hand = np.flatnonzero(counts[:, idx["apple_hand"]] > 0)
    last_hand = start + int(hand[-1]) if hand.size else None
    base_hit = _first(counts[:, idx["apple_plate_base"]] > 0)
    rim = counts[:, idx["apple_plate_rim"]] > 0
    horizontal = np.linalg.norm(a["apple_lin"][:, :2], axis=1)
    palm_world = a["palm"] + base
    out = {
        "open_step": start,
        "palm_at_open_base": a["palm"][start].round(4).tolist(),
        "palm_speed_at_open_m_s": float(
            np.linalg.norm(a["palm"][start] - a["palm"][start - 1]) / 0.05
        )
        if start > 0
        else None,
        "apple_height_above_rest_at_open_cm": float((a["apple_pos"][start, 2] - rest_z) * 100),
        "apple_xy_minus_plate_at_open_cm": ((a["apple_pos"][start, :2] - plate[:2]) * 100)
        .round(2)
        .tolist(),
        "palm_xy_minus_apple_at_open_cm": (
            (palm_world[start, :2] - a["apple_pos"][start, :2]) * 100
        )
        .round(2)
        .tolist(),
        "last_apple_hand_step": last_hand,
        "rim_contact_after_open": bool(rim.any()),
        "hand_plate_contact_steps": int((a["counts"][:, idx["hand_plate"]] > 0).sum()),
    }
    if last_hand is not None and last_hand + 1 < len(horizontal):
        out["apple_horizontal_speed_after_hand_m_s"] = float(horizontal[last_hand + 1])
        out["apple_height_above_rest_at_last_hand_cm"] = float(
            (a["apple_pos"][last_hand, 2] - rest_z) * 100
        )
    if base_hit is not None:
        step = start + base_hit
        out["first_base_contact_step"] = step
        out["landing_xy_minus_plate_cm"] = (
            ((a["apple_pos"][step, :2] - plate[:2]) * 100).round(2).tolist()
        )
        out["horizontal_speed_at_landing_m_s"] = float(horizontal[step])
        out["angular_speed_at_landing_rad_s"] = float(np.linalg.norm(a["apple_ang"][step]))
    return out
