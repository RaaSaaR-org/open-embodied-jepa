"""TASK-070 development runs of the v2 expert, under the fixed apple-to-plate-v2 physics.

Engineering on a privileged scripted controller, not a learned result, and not a gated run. The
scene is ``apple_to_plate_v2`` (the v1 scene with the apple's contact at condim 6 and the scene's
own declared friction; owner ruling R12). Every design run here is named in ``DESIGNS`` and
logged, with its counts, in ``docs/experiments/apple_to_plate_v2_expert.md`` (the development
log). Seeds come from ``apple_to_plate_v2.DEV_SEEDS``; plate-error directions from its
``DEV_DIRECTION_SEED``. The physics is never varied here (R12 §3 (iii)).

    uv run --no-sync python scripts/develop_v2_expert.py --design e1 \\
        --count 16 --levels 0,1.0 --output outputs/task070-dev/e1-run-1
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

from embodied_jepa import apple_to_plate_v2 as v2  # noqa: E402
from embodied_jepa import resting_expert as rx  # noqa: E402

# Every design tried, in order. Parameters are RestingPlaceExpert keyword arguments.
DESIGNS: dict[str, dict] = {
    "e1": {"release_pitch_rad": 0.45},  # TASK-068's d12, the R12 starting point
    # e1 on dev seeds: apples reach the far rim and rebound to ~3.3 cm (median). Aim: land
    # slower (pitch), or further back (release_dx), so the roll ends nearer the centre.
    "e2": {"release_pitch_rad": 0.55},
    "e3": {"release_pitch_rad": 0.6},
    "e4": {"release_pitch_rad": 0.45, "release_dx": -0.035},
    "e5": {"release_pitch_rad": 0.45, "release_dx": -0.005},
    "e6": {"release_pitch_rad": 0.45, "opening_ramp": 0.02},
    "e7": {"release_pitch_rad": 0.5},
    # e2-e7: slower landings (e2, e3, e6, e7) end nearer the rim; releasing 1 cm further
    # forward (e5) was best. Continue that direction.
    "e8": {"release_pitch_rad": 0.45, "release_dx": 0.005},
    "e9": {"release_pitch_rad": 0.45, "release_dx": 0.015},
    # Held-back check (50232-50295): e9 59/64 at 1.0 cm, e5 56/64, e1 54/64. One more step.
    "e10": {"release_pitch_rad": 0.45, "release_dx": 0.025},
    "e11": {"release_pitch_rad": 0.4, "release_dx": 0.015},
}

_W: dict = {}


def load_wide_reset():
    spec = importlib.util.spec_from_file_location(
        "_evaluate_apple_reset", ROOT / "scripts" / "evaluate_apple.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wide_reset


def worker_init() -> None:
    from embodied_jepa import first_policy_runtime as rt

    _W["robot"], _W["scene"] = v2.make_v2_robot()
    _W["bounds"] = rt.configured_bounds()


def run_one(task: dict) -> dict:
    params = task["params"]
    try:
        summary, arrays = rx.run_attempt(
            _W["robot"],
            _W["bounds"],
            seed=task["seed"],
            reset=task["reset"],
            plate_offset=task["plate_offset"],
            make_expert=lambda truth: rx.RestingPlaceExpert(truth, **params),
        )
        np.savez_compressed(task["npz"], **arrays)
        return {"ok": True, "plate_cm": task["plate_cm"], "scene": _W["scene"], **summary}
    except Exception as error:  # noqa: BLE001 - recorded
        return {
            "ok": False,
            "seed": task["seed"],
            "plate_cm": task["plate_cm"],
            "error": f"{type(error).__name__}: {error}",
        }


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", required=True, choices=sorted(DESIGNS))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--start", type=int, default=0, help="index into DEV_SEEDS")
    parser.add_argument("--levels", default="0,1.0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    seeds = v2.DEV_SEEDS[args.start : args.start + args.count]
    if len(seeds) != args.count:
        raise SystemExit("seed slice leaves the development range")
    v2.check_seeds(seeds)
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    levels = tuple(float(v) for v in args.levels.split(","))
    params = DESIGNS[args.design]
    (args.output / "attempts").mkdir(parents=True)
    wide_reset = load_wide_reset()
    tasks = []
    for level in levels:
        offsets = rx.plate_offsets(len(v2.DEV_SEEDS), level, v2.DEV_DIRECTION_SEED)
        for seed in seeds:
            r = wide_reset(seed)
            tasks.append(
                {
                    "seed": seed,
                    "plate_cm": level,
                    "params": params,
                    "reset": {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])},
                    "plate_offset": offsets[v2.DEV_SEEDS.index(seed)].tolist(),
                    "npz": str(args.output / "attempts" / f"{level}-{seed}.npz"),
                }
            )
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=worker_init)
    try:
        rows = pool.map(run_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    summary = {}
    for level in levels:
        cell = [r for r in rows if r["plate_cm"] == level]
        ok = [r for r in cell if r["ok"]]
        landing = [
            r["horizontal_speed_at_landing_m_s"]
            for r in ok
            if "horizontal_speed_at_landing_m_s" in r
        ]
        summary[str(level)] = {
            "attempts": len(cell),
            "errors": len(cell) - len(ok),
            "incomplete": sum(not r["complete"] for r in ok),
            "guard_stops": sum(r["stop_reason"].startswith("guard") for r in ok),
            "at_rest": sum(r["at_rest"] for r in ok),
            "latched": sum(r["latched_success"] for r in ok),
            "final_distance_cm_q10_q50_q90": np.quantile(
                [r["final_distance_cm"] for r in ok], [0.1, 0.5, 0.9]
            )
            .round(2)
            .tolist()
            if ok
            else None,
            "landing_speed_m_s_q50": float(np.median(landing)) if landing else None,
        }
    report = {
        "task": "TASK-070 development run (not gated; privileged scripted expert; v2 physics)",
        "scene": v2.SCENE_VERSION,
        "design": args.design,
        "params": params,
        "seeds": list(seeds),
        "direction_seed": v2.DEV_DIRECTION_SEED,
        "levels_cm": list(levels),
        "revision": revision,
        "tracked_tree_dirty": dirty,
        "revision_at_end": git("rev-parse", "HEAD"),
        "seconds": time.monotonic() - started,
        "summary": summary,
        "attempts": rows,
    }
    payload = json.dumps(report, indent=1, sort_keys=True, default=float) + "\n"
    (args.output / "report.json").write_text(payload)
    print(json.dumps({"design": args.design, "summary": summary}, indent=1))
    print("report sha256", hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
