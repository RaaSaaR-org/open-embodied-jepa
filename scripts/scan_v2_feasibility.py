"""TASK-069 feasibility scan (development only, no gate): candidate v2 task changes.

Privileged scripted engineering, not a learned result. Each cell is a scene (the apple's contact
parameters), a reset distribution (v1 plates, or plates in the set-down region) and an expert
(``resting_expert.RestingPlaceExpert`` keyword arguments), run on the TASK-069 development seeds
at plate exact and at 1.0 cm plate error (TASK-067's C0 perturbation, directions from
``default_rng(6840)``). Every attempt reports the at-rest check (``apple_at_rest_v0``), the
latched scorer, guard stops and the landing speed.

    uv run --no-sync python scripts/scan_v2_feasibility.py --cells all --count 32 \\
        --output outputs/task069-scan/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import resting_expert as rx  # noqa: E402
from embodied_jepa import v2_feasibility as v2  # noqa: E402

V1_CONTACT = {"condim": 3, "torsional": 0.01, "rolling": 0.001}  # the v1 scene, unchanged
B3A = {"condim": 6, "torsional": 0.01, "rolling": 0.001}  # the scene's own declared values
B3B = {"condim": 6, "torsional": 0.02, "rolling": 0.003}  # stronger; plausibility unmeasured
# Plate centres whose set-down palm pose (plate x - 1.5 cm, palm z 0.035 m) is inside the
# palm-down reach region of outputs/task069-reach/run-1 with about 1 cm to spare.
SETDOWN_BOX = ((0.30, 0.37), (-0.25, -0.04))
D12 = {"release_pitch_rad": 0.45}  # TASK-068's best release (0/64 at rest under v1)
SETDOWN = {"release_z_floor_m": 0.0354}  # palm to the set-down height, defaults otherwise
SETDOWN_PITCH = {"release_z_floor_m": 0.0354, "release_pitch_rad": 0.45}
D1 = {}  # RestingPlaceExpert defaults (TASK-068 d1)
COLLECTOR = {"collector": True}  # apple_collector_policy, 745 commands (TASK-067's C0 expert)
CELLS = {
    "v1-d12": {"contact": V1_CONTACT, "reset": "v1", "expert": D12},
    "v1-collector": {"contact": V1_CONTACT, "reset": "v1", "expert": COLLECTOR},
    "b3a-collector": {"contact": B3A, "reset": "v1", "expert": COLLECTOR},
    "b3a-d1": {"contact": B3A, "reset": "v1", "expert": D1},
    "b3a-d12": {"contact": B3A, "reset": "v1", "expert": D12},
    "b3b-d12": {"contact": B3B, "reset": "v1", "expert": D12},
    "b2-setdown": {"contact": V1_CONTACT, "reset": "setdown", "expert": SETDOWN},
    "b2-setdown-pitch": {"contact": V1_CONTACT, "reset": "setdown", "expert": SETDOWN_PITCH},
    "b2b3a-setdown": {"contact": B3A, "reset": "setdown", "expert": SETDOWN},
    "b2b3a-setdown-pitch": {"contact": B3A, "reset": "setdown", "expert": SETDOWN_PITCH},
    "b2b3b-setdown": {"contact": B3B, "reset": "setdown", "expert": SETDOWN},
    "b2b3b-setdown-pitch": {"contact": B3B, "reset": "setdown", "expert": SETDOWN_PITCH},
}
_W: dict = {}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def worker_init() -> None:
    from embodied_jepa import first_policy_runtime as rt

    _W["robot"] = rt.make_robot()
    _W["bounds"] = rt.configured_bounds()


def run_one(task: dict) -> dict:
    cell = CELLS[task["cell"]]
    try:
        change = v2.apply_apple_friction(_W["robot"].model, **cell["contact"])
        params = cell["expert"]
        collector = bool(params.get("collector"))

        def make_expert(truth):
            if collector:
                from embodied_jepa.scripted import apple_collector_policy

                return apple_collector_policy(truth)
            return rx.RestingPlaceExpert(truth, **params)

        summary, arrays = rx.run_attempt(
            _W["robot"],
            _W["bounds"],
            seed=task["seed"],
            reset=task["reset"],
            plate_offset=task["plate_offset"],
            make_expert=make_expert,
            budget=745 if collector else rx.EXPERT_BUDGET,
        )
        np.savez_compressed(task["npz"], **arrays)
        guard = summary["stop_reason"].startswith("guard")
        return {
            "ok": True,
            **{k: task[k] for k in ("cell", "plate_cm", "reset")},
            "contact": change["after"],
            "guard_stop": guard,
            **summary,
        }
    except Exception as error:  # noqa: BLE001 - recorded
        return {
            "ok": False,
            **{k: task[k] for k in ("cell", "seed", "plate_cm")},
            "error": f"{type(error).__name__}: {error}",
        }


def summarise(rows) -> dict:
    ok = [r for r in rows if r["ok"]]
    landing = [
        r["horizontal_speed_at_landing_m_s"] for r in ok if "horizontal_speed_at_landing_m_s" in r
    ]
    window = [r["at_rest_detail"]["max_speed_m_s"] for r in ok if r.get("at_rest_detail")]
    inside = [r["at_rest_detail"]["inside_all"] for r in ok if r.get("at_rest_detail")]
    return {
        "attempts": len(rows),
        "errors": len(rows) - len(ok),
        "complete": sum(r["complete"] for r in ok),
        "guard_stops": sum(r["guard_stop"] for r in ok),
        "other_stops": sum((not r["complete"]) and not r["guard_stop"] for r in ok),
        "at_rest": sum(r["at_rest"] for r in ok),
        "latched": sum(r["latched_success"] for r in ok),
        "inside_window_any_speed": int(sum(inside)),
        "landing_speed_m_s_q10_q50_q90": np.quantile(landing, [0.1, 0.5, 0.9]).round(3).tolist()
        if landing
        else None,
        "landing_measured": len(landing),
        "window_max_speed_m_s_min_median": [float(np.min(window)), float(np.median(window))]
        if window
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", default="all")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--start", type=int, default=0, help="index into DEV_SEEDS")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cells = list(CELLS) if args.cells == "all" else args.cells.split(",")
    seeds = v2.DEV_SEEDS[args.start : args.start + args.count]
    if len(seeds) != args.count:
        raise SystemExit("seed slice leaves the development range")
    revision = git("rev-parse", "HEAD")  # read at start: the code the workers import
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    v2.check_seeds(seeds)
    spec = importlib.util.spec_from_file_location("_eval", ROOT / "scripts" / "evaluate_apple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in seeds:  # the scan's v1 reset is exactly the v1 wide-jitter reset
        wide = module.wide_reset(seed)
        mine = v2.v1_reset(seed)
        if wide["object_xy"] != mine["object_xy"] or wide["plate_xy"] != mine["plate_xy"]:
            raise SystemExit(f"v1 reset mismatch at seed {seed}")
    (args.output / "attempts").mkdir(parents=True)
    tasks = []
    for cell in cells:
        for level in v2.PLATE_LEVELS_CM:
            offsets = rx.plate_offsets(len(v2.DEV_SEEDS), level, v2.DEV_DIRECTION_SEED)
            for seed in seeds:
                reset = (
                    v2.v1_reset(seed)
                    if CELLS[cell]["reset"] == "v1"
                    else v2.setdown_reset(seed, SETDOWN_BOX)
                )
                tasks.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "plate_cm": level,
                        "reset": reset,
                        "plate_offset": offsets[v2.DEV_SEEDS.index(seed)].tolist(),
                        "npz": str(args.output / "attempts" / f"{cell}-{level}-{seed}.npz"),
                    }
                )
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=worker_init)
    try:
        rows = pool.map(run_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    summary = {
        cell: {
            str(level): summarise([r for r in rows if r["cell"] == cell and r["plate_cm"] == level])
            for level in v2.PLATE_LEVELS_CM
        }
        for cell in cells
    }
    report = {
        "task": "TASK-069 feasibility scan (development only, no gate; privileged scripted)",
        "cells": {c: CELLS[c] for c in cells},
        "setdown_box": SETDOWN_BOX,
        "seeds": list(seeds),
        "direction_seed": v2.DEV_DIRECTION_SEED,
        "revision": revision,
        "revision_at_end": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "seconds": time.monotonic() - started,
        "summary": summary,
        "attempts": rows,
    }
    payload = json.dumps(report, indent=1, sort_keys=True, default=float) + "\n"
    (args.output / "report.json").write_text(payload)
    print(json.dumps(summary, indent=1))
    print("report sha256", hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
