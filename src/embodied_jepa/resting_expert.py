"""TASK-068: the privileged scripted expert that rests the apple on the plate, and its harness.

Protocol: ``docs/experiments/apple_resting_expert_v1.md``. This is engineering on a privileged
scripted controller (``RestingPlaceExpert``, below); it is not a learned result. NumPy at
import; MuJoCo is imported lazily by the functions that run the simulator.

One attempt is: reset (the TASK-047 wide-jitter distribution), the TASK-064 look, the expert
built from the reset truth with the plate xy shifted by a fixed offset (the TASK-067 C0 plate
perturbation, ``first_policy_runtime.perturbed_truth``), the expert's commands, then a settle of
``AtRestThresholds.settle_steps`` steps (arm still, hands open). Two readings are recorded:

* **at rest** (``at_rest.apple_at_rest``, version ``apple_at_rest_v0``) over the final window;
* the **latched scorer**: ``AppleToPlateTask``'s per-step success at any step of the attempt, as
  C0 and the TASK-067 probes counted it (they stopped at the first success).
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa.at_rest import AtRestThresholds
from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import rotation_delta
from embodied_jepa.scripted import OracleManipulationPolicy, OraclePhase

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


# ----- the expert ------------------------------------------------------------------------------
RIGHT_SHOULDER_BASE = np.array([0.0, -0.10, 0.292])  # right_shoulder_pitch_link, fixed pelvis


class RestingPlaceExpert(OracleManipulationPolicy):
    """TASK-068 privileged scripted expert: release the apple still, low and over the plate.

    A scripted initial-truth controller, never a learned result. It keeps the collector's pick
    (orient, descend, close, lift, with the collector's palm offset) and replaces the transfer and
    release with a reachable release pose, a steady hold before opening, a slow opening, a clear
    hold while the apple lands, and a vertical retreat.

    The release pose puts the palm at ``plate + release_dx`` in x and at the plate's y. Its
    height is the lowest the right arm can reach there with the palm down, from a declared reach
    sphere about the right shoulder (``reach_radius_m``, measured with the embodiment's own IK in
    TASK-068's descent diagnosis), and never below ``release_z_floor_m``. If even the top of the
    sphere cannot reach that xy, the palm target moves back toward the shoulder in x until it can.
    """

    def __init__(
        self,
        initial_truth,
        *,
        palm_x_offset=0.015,
        release_dx=-0.015,
        reach_radius_m=0.485,
        release_z_floor_m=0.10,
        release_z_ceiling_m=0.26,
        transfer_commands=100,
        lower_commands=50,
        steady_commands=30,
        opening_ramp=0.04,
        open_commands=50,
        clear_commands=30,
        retreat_commands=60,
        retreat_lift_m=0.08,
        steady_translation_limit=0.4,
        release_pitch_rad=0.0,
        open_dx_m=0.0,
        open_translation_limit=0.4,
        closure=1.0,
    ):
        super().__init__(initial_truth)
        for name, value in (
            ("opening_ramp", opening_ramp),
            ("reach_radius_m", reach_radius_m),
            ("steady_translation_limit", steady_translation_limit),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ContractError(f"{name} must be positive and finite")
        if not release_z_floor_m <= release_z_ceiling_m:
            raise ContractError("release height floor must not exceed its ceiling")
        base = np.asarray(initial_truth["base_position_world"])
        container = np.asarray(initial_truth["plate_position"]) - base
        if not np.isfinite(closure) or not -1 < closure <= 1:
            raise ContractError("closure must lie in (-1, 1]")
        pick = tuple(
            OraclePhase(
                p.name,
                p.target_base + [palm_x_offset, 0, 0],
                closure if p.grasp > 0 else p.grasp,
                p.commands,
            )
            for p in self.phases[:4]
        )
        release = self.release_pose(
            container[:2] + [release_dx, 0.0],
            reach_radius_m=reach_radius_m,
            floor=release_z_floor_m,
            ceiling=release_z_ceiling_m,
        )
        high = pick[3].target_base[2]
        transfer = np.array([release[0], release[1], max(high, release[2])])
        retreat = release + [0.0, 0.0, retreat_lift_m]
        self.release_target = release
        self.opening_ramp = float(opening_ramp)
        self.steady_translation_limit = float(steady_translation_limit)
        self.open_limit = float(open_translation_limit)
        self.accepted_grasp = -1.0
        self.pick_rotation = self.rotation
        self.release_rotation = rotation_delta([0.0, release_pitch_rad, 0.0]) @ self.rotation
        self.phases = (
            *pick,
            OraclePhase("transfer", transfer, closure, transfer_commands),
            OraclePhase("lower", release.copy(), closure, lower_commands),
            OraclePhase("steady", release.copy(), closure, steady_commands),
            OraclePhase("open", release + [open_dx_m, 0.0, 0.0], -1.0, open_commands),
            OraclePhase("clear", release + [open_dx_m, 0.0, 0.0], -1.0, clear_commands),
            OraclePhase("retreat", retreat, -1.0, retreat_commands),
        )

    @staticmethod
    def release_pose(xy, *, reach_radius_m, floor, ceiling, shoulder=RIGHT_SHOULDER_BASE):
        """The lowest palm-down pose at ``xy`` inside the reach sphere, within [floor, ceiling]."""
        xy = np.asarray(xy, float).copy()
        horizontal2 = reach_radius_m**2 - (xy[1] - shoulder[1]) ** 2
        top_dz = shoulder[2] - ceiling
        # pull x back until (x, y, ceiling) is inside the sphere
        limit = horizontal2 - max(top_dz, 0.0) ** 2
        if limit <= 0:
            raise ContractError("release y is outside the reach sphere")
        xy[0] = min(xy[0], shoulder[0] + np.sqrt(limit))
        dz2 = horizontal2 - (xy[0] - shoulder[0]) ** 2
        z = shoulder[2] - np.sqrt(max(dz2, 0.0))
        return np.array([xy[0], xy[1], float(np.clip(z, floor, ceiling))])

    def action(self, robot):
        name = self.phases[self.phase_index].name
        tilted = name in ("lower", "steady", "open", "clear")
        self.rotation = self.release_rotation if tilted else self.pick_rotation
        action = super().action(robot)
        if name in ("steady", "open", "clear"):
            limit = self.steady_translation_limit if name == "steady" else self.open_limit
            action[6:9] = np.clip(action[6:9], -limit, limit)
        if name in ("open", "clear", "retreat"):
            action[13] = max(-1.0, self.accepted_grasp - self.opening_ramp)
        return action

    def advance(self, result):
        super().advance(result)
        if result.applied_action is not None:
            self.accepted_grasp = float(result.applied_action[13])


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
    from embodied_jepa.at_rest import AppleAtRestCheck

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
