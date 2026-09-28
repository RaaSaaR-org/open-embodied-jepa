"""TASK-067 landing diagnosis (owner ruling R9 step 2): why do failures end 4.53-4.62 cm out?

Diagnostic only, with no gate. Scripted and privileged: no learned policy, no corpus read, no test
split. It re-runs the **current** collector (``apple_collector_policy``, ``transfer_x_shift``
-0.035) on the already-probed seeds 46800-46831. Those seeds were spent by the R8 probe; R9
allows reusing them for diagnosis only. It runs at the reference and at a 1.0 cm plate error,
in the R8 probe's directions (``default_rng(6810)``), after the look.

Each attempt runs the collector's full 745 commands, as the probe did. It then holds for 60
settle steps (zero arm motion, hands open) so the apple can come to rest. The attempt does NOT
stop at the scorer's success, so the whole post-grasp trajectory is logged. Every step records:

* the collector phase, its command, and the right palm pose;
* the apple position, orientation (quaternion), and linear and angular velocity;
* the plate body's pose;
* every active contact, classified by pair (apple-plate base, apple-plate rim, apple-hand/finger,
  hand-plate, apple-table, other) with its penetration distance;
* the scorer's reading (stages, apple-plate xy distance, hand contact).

Per-step logs go to ``attempts/<arm>-<level>-<seed>.npz``. Per-attempt event summaries go to
``report.json``. The script does not read an outcome; the probe document's diagnosis section
does.

    uv run --no-sync python scripts/diagnose_first_policy_landing.py \\
        --output outputs/task067-landing-diagnosis/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy as fp  # noqa: E402
from embodied_jepa import first_policy_runtime as rt  # noqa: E402

SEEDS = tuple(range(46800, 46832))  # the spent R8 probe seeds, reused for diagnosis only (R9)
DIRECTION_SEED = 6810  # the R8 probe's directions
PLATE_LEVELS_CM = (0.0, 1.0)
TRANSFER_X_SHIFT = -0.035  # the current collector
SETTLE_STEPS = 60
PAIR_TYPES = (
    "apple_plate_base",
    "apple_plate_rim",
    "apple_hand",
    "hand_plate",
    "apple_table",
    "other_apple",
    "other",
)
PHASE_NAMES = fp.PHASES


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy", ROOT / "scripts" / "run_first_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def init_worker() -> None:
    RUNNER.worker_init()


def geom_class(sim, geom_id: int) -> str:
    model = sim.model
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


def classify(a: str, b: str) -> str:
    pair = {a, b}
    if pair == {"apple", "plate_base"}:
        return "apple_plate_base"
    if pair == {"apple", "plate_rim"}:
        return "apple_plate_rim"
    if pair == {"apple", "hand"}:
        return "apple_hand"
    if "hand" in pair and pair & {"plate_base", "plate_rim"}:
        return "hand_plate"
    if pair == {"apple", "table"}:
        return "apple_table"
    if "apple" in pair:
        return "other_apple"
    return "other"


def contacts(sim) -> dict:
    counts = dict.fromkeys(PAIR_TYPES, 0)
    depth = dict.fromkeys(PAIR_TYPES, 0.0)
    for c in sim.data.contact[: sim.data.ncon]:
        kind = classify(geom_class(sim, int(c.geom1)), geom_class(sim, int(c.geom2)))
        counts[kind] += 1
        depth[kind] = min(depth[kind], float(c.dist))
    return {"counts": counts, "depth": depth}


class Collector:
    def __init__(self, truth, robot):
        from embodied_jepa.scripted import apple_collector_policy

        self.policy = apple_collector_policy(truth, transfer_x_shift=TRANSFER_X_SHIFT)
        self.robot = robot


def run_one(task: dict) -> dict:
    """One diagnostic attempt; per-step arrays are written to ``task['npz']``."""
    try:
        robot = RUNNER._W["robot"]
        lower, upper = RUNNER._W["bounds"]
        truth, scorer, counter, _obs, _facts = rt.reset_and_look(robot, task["seed"], task["reset"])
        counter.remove()  # privileged diagnosis: truth is read every step on purpose
        perturbed = rt.perturbed_truth(truth, [0.0, 0.0], task["plate_offset"])
        collector = Collector(perturbed, robot)
        sim = robot.sim
        rows = {
            k: []
            for k in (
                "phase",
                "command",
                "palm",
                "apple_pos",
                "apple_quat",
                "apple_lin",
                "apple_ang",
                "plate_pos",
                "plate_quat",
                "counts",
                "depth",
                "stage",
                "distance",
                "hand_contact",
            )
        }
        stop_reason = "complete"
        for step in range(fp.EXPERT_POLICY_STEPS + SETTLE_STEPS):
            robot.observe()
            policy = collector.policy
            if step < fp.EXPERT_POLICY_STEPS and not policy.done:
                phase = policy.phase_index
                command = np.asarray(policy.action(robot), np.float32)
            else:
                phase = len(PHASE_NAMES)  # settle
                command = np.zeros(14, np.float32)
                command[12:] = -1.0
            command = np.clip(command, lower, upper).astype(np.float32)
            try:
                projection = robot.project_candidates(command[None, None, None])
            except Exception as error:  # noqa: BLE001 - a guard stop ends the log, recorded
                stop_reason = f"guard: {error}"
                break
            if not bool(projection.feasible[0, 0]):
                stop_reason = "infeasible"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                stop_reason = f"rejected: {result.reason}"
                break
            if step < fp.EXPERT_POLICY_STEPS and not policy.done:
                policy.advance(result)
            score = scorer.evaluate()
            joint = sim.data.joint("apple_free")
            position, rotation = robot.ee_pose("right")
            contact = contacts(sim)
            rows["phase"].append(phase)
            rows["command"].append(command[list(fp.FREE_INDICES)])
            rows["palm"].append(np.r_[position, rotation[:, 0], rotation[:, 1]])
            rows["apple_pos"].append(sim.data.body("apple").xpos.copy())
            rows["apple_quat"].append(sim.data.body("apple").xquat.copy())
            rows["apple_lin"].append(joint.qvel[:3].copy())
            rows["apple_ang"].append(joint.qvel[3:6].copy())
            rows["plate_pos"].append(sim.data.body("plate").xpos.copy())
            rows["plate_quat"].append(sim.data.body("plate").xquat.copy())
            rows["counts"].append([contact["counts"][k] for k in PAIR_TYPES])
            rows["depth"].append([contact["depth"][k] for k in PAIR_TYPES])
            rows["stage"].append(
                [
                    bool(score.get(s))
                    for s in ("reach", "grasp", "transport", "place", "release", "success")
                ]
            )
            rows["distance"].append(score["object_plate_distance_m"])
            rows["hand_contact"].append(bool(score["hand_contact"]))
        robot.stop(stop_reason)
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        np.savez_compressed(task["npz"], **arrays)
        return {
            "ok": True,
            **{k: task[k] for k in ("seed", "plate_cm")},
            "stop_reason": stop_reason,
            "steps": len(rows["phase"]),
            **events(arrays),
        }
    except Exception as error:  # noqa: BLE001 - recorded
        return {
            "ok": False,
            **{k: task[k] for k in ("seed", "plate_cm")},
            "error": f"{type(error).__name__}: {error}",
        }


def _first(mask) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def events(a: dict) -> dict:
    """Per-attempt events read from the logs; none of them is a verdict."""
    idx = {k: PAIR_TYPES.index(k) for k in PAIR_TYPES}
    phase = a["phase"]
    release = _first(phase >= PHASE_NAMES.index("release_high"))
    success_steps = np.flatnonzero(a["stage"][:, 5])
    after = slice(release, None) if release is not None else slice(0, 0)
    counts = a["counts"][after]
    apple = a["apple_pos"][after]
    speed = np.linalg.norm(a["apple_lin"][after], axis=1)
    plate_xy = a["plate_pos"][after][:, :2]
    rel = apple[:, :2] - plate_xy
    first_base = _first(counts[:, idx["apple_plate_base"]] > 0)
    first_rim = _first(counts[:, idx["apple_plate_rim"]] > 0)
    hand_contact_steps = np.flatnonzero(counts[:, idx["apple_hand"]] > 0)
    last_hand = int(hand_contact_steps[-1]) if hand_contact_steps.size else None
    dist = np.linalg.norm(rel, axis=1)
    out = {
        "release_step": release,
        "first_success_step": int(success_steps[0]) if success_steps.size else None,
        "final_success": bool(a["stage"][-1, 5]),
        "ever_success": bool(success_steps.size),
        "final_distance_cm": float(a["distance"][-1] * 100),
        "final_dx_cm": float(rel[-1, 0] * 100) if len(rel) else None,
        "final_dy_cm": float(rel[-1, 1] * 100) if len(rel) else None,
        "final_speed_m_s": float(speed[-1]) if len(speed) else None,
        "final_apple_z_m": float(apple[-1, 2]) if len(apple) else None,
        "final_rim_contact": bool(counts[-1, idx["apple_plate_rim"]] > 0) if len(counts) else None,
        "final_base_contact": bool(counts[-1, idx["apple_plate_base"]] > 0)
        if len(counts)
        else None,
        "final_hand_contact": bool(counts[-1, idx["apple_hand"]] > 0) if len(counts) else None,
        "plate_moved_m": float(np.abs(a["plate_pos"] - a["plate_pos"][0]).max()),
        "plate_rotated": float(np.abs(a["plate_quat"] - a["plate_quat"][0]).max()),
        "hand_plate_contact_steps": int((a["counts"][:, idx["hand_plate"]] > 0).sum()),
        "apple_table_contact_steps_after_release": int((counts[:, idx["apple_table"]] > 0).sum()),
        "last_apple_hand_contact_after_release": last_hand,
        "first_apple_base_contact_after_release": first_base,
        "first_apple_rim_contact_after_release": first_rim,
        "max_speed_after_release_m_s": float(speed.max()) if len(speed) else None,
    }
    if first_base is not None and len(dist):
        out["distance_at_first_base_contact_cm"] = float(dist[first_base] * 100)
        # travel on the plate after the apple first touched its base, split by whether the hand
        # was still touching the apple
        steps = np.arange(len(dist))
        moved = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        on_plate = steps[1:] > first_base
        with_hand = counts[1:, idx["apple_hand"]] > 0
        out["travel_on_plate_with_hand_cm"] = float(moved[on_plate & with_hand].sum() * 100)
        out["travel_on_plate_without_hand_cm"] = float(moved[on_plate & ~with_hand].sum() * 100)
        out["distance_at_last_hand_contact_cm"] = (
            float(dist[last_hand] * 100) if last_hand is not None else None
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    (args.output / "attempts").mkdir(parents=True)
    wide_reset = RUNNER.load_wide_reset()
    angles = np.random.default_rng(DIRECTION_SEED).uniform(0, 2 * np.pi, len(SEEDS))
    unit = np.c_[np.cos(angles), np.sin(angles)]
    tasks = []
    for level in PLATE_LEVELS_CM:
        for i, seed in enumerate(SEEDS):
            r = wide_reset(seed)
            tasks.append(
                {
                    "seed": seed,
                    "plate_cm": level,
                    "reset": {
                        "object_xy": list(map(float, r["object_xy"])),
                        "plate_xy": list(map(float, r["plate_xy"])),
                    },
                    "plate_offset": (level / 100.0 * unit[i]).tolist(),
                    "npz": str(args.output / "attempts" / f"current-{level}-{seed}.npz"),
                }
            )
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(fp.SIM_WORKERS, initializer=init_worker)
    try:
        rows = pool.map(run_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    report = {
        "diagnosis": "TASK-067 R9 step 2: the current collector's post-grasp trajectory",
        "seeds": list(SEEDS),
        "seed_reuse": "spent R8 probe seeds, reused for diagnosis only (R9)",
        "direction_seed": DIRECTION_SEED,
        "transfer_x_shift": TRANSFER_X_SHIFT,
        "plate_levels_cm": list(PLATE_LEVELS_CM),
        "settle_steps": SETTLE_STEPS,
        "pair_types": list(PAIR_TYPES),
        "revision": RUNNER.R65.revision(),
        "tracked_tree_dirty": bool(RUNNER.R65.tracked_tree_dirty()),
        "seconds": time.monotonic() - started,
        "attempts": rows,
    }
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(payload)
    errors = sum(not r["ok"] for r in rows)
    print(
        json.dumps(
            {
                "attempts": len(rows),
                "errors": errors,
                "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
