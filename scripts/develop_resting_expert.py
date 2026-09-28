"""TASK-068 development runs of the resting expert, on the declared development seeds only.

Engineering on a privileged scripted controller, not a learned result, and not a gated run.
Every design run here is named in ``DESIGNS`` and logged, with its counts, in
``docs/experiments/apple_resting_expert_v1.md`` §5 (the development log). Seeds must come from
``resting_expert.DEV_SEEDS``; plate-error directions come from ``DEV_DIRECTION_SEED``.

    uv run --no-sync python scripts/develop_resting_expert.py --design d1 \\
        --count 16 --levels 0,1.0 --output outputs/task068-dev/d1-run-1
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

from embodied_jepa import first_policy as fp  # noqa: E402
from embodied_jepa import resting_expert as rx  # noqa: E402

# Every design tried, in order. Parameters are RestingPlaceExpert keyword arguments.
DESIGNS: dict[str, dict] = {
    "d1": {},
    "d2": {"opening_ramp": 1.0},
    "d3": {"opening_ramp": 0.2},
    "d4": {"opening_ramp": 1.0, "release_pitch_rad": -0.3},
    "d5": {"opening_ramp": 1.0, "release_pitch_rad": 0.3},
    "d6": {"opening_ramp": 1.0, "open_dx_m": -0.06, "open_translation_limit": 0.5},
    "d7": {"opening_ramp": 1.0, "open_dx_m": 0.06, "open_translation_limit": 0.5},
    "d8": {"opening_ramp": 0.04, "open_dx_m": -0.06, "open_translation_limit": 0.2},
    "d9": {"opening_ramp": 1.0, "closure": 0.5},
    "d10": {"opening_ramp": 1.0, "palm_x_offset": 0.0},
    "d11": {"opening_ramp": 1.0, "palm_x_offset": 0.03},
    "d12": {"release_pitch_rad": 0.45},
    # Baseline, not a design: the current collector (TASK-067's C0 expert), 745 commands.
    "collector": {"collector": True},
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

    _W["robot"] = rt.make_robot()
    _W["bounds"] = rt.configured_bounds()


def make_factory(params: dict):
    def factory(truth):
        from embodied_jepa.scripted import RestingPlaceExpert, apple_collector_policy

        if params.get("collector"):
            return apple_collector_policy(truth)
        return RestingPlaceExpert(truth, **params)

    return factory


def run_one(task: dict) -> dict:
    try:
        summary, arrays = rx.run_attempt(
            _W["robot"],
            _W["bounds"],
            seed=task["seed"],
            reset=task["reset"],
            plate_offset=task["plate_offset"],
            make_expert=make_factory(task["params"]),
            budget=fp.EXPERT_POLICY_STEPS if task["params"].get("collector") else rx.EXPERT_BUDGET,
        )
        np.savez_compressed(task["npz"], **arrays)
        return {"ok": True, "plate_cm": task["plate_cm"], **summary}
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
    seeds = rx.DEV_SEEDS[args.start : args.start + args.count]
    if len(seeds) != args.count:
        raise SystemExit("seed slice leaves the development range")
    rx.check_seeds(seeds)
    levels = tuple(float(v) for v in args.levels.split(","))
    params = DESIGNS[args.design]
    (args.output / "attempts").mkdir(parents=True)
    wide_reset = load_wide_reset()
    tasks = []
    for level in levels:
        offsets = rx.plate_offsets(len(rx.DEV_SEEDS), level, rx.DEV_DIRECTION_SEED)
        for seed in seeds:
            r = wide_reset(seed)
            tasks.append(
                {
                    "seed": seed,
                    "plate_cm": level,
                    "params": params,
                    "reset": {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])},
                    "plate_offset": offsets[rx.DEV_SEEDS.index(seed)].tolist(),
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
        summary[str(level)] = {
            "attempts": len(cell),
            "errors": len(cell) - len(ok),
            "incomplete": sum(not r["complete"] for r in ok),
            "at_rest": sum(r["at_rest"] for r in ok),
            "latched": sum(r["latched_success"] for r in ok),
            "final_distance_cm_q10_q50_q90": np.quantile(
                [r["final_distance_cm"] for r in ok], [0.1, 0.5, 0.9]
            )
            .round(2)
            .tolist()
            if ok
            else None,
        }
    report = {
        "task": "TASK-068 development run (not gated; privileged scripted expert)",
        "design": args.design,
        "params": params,
        "seeds": list(seeds),
        "direction_seed": rx.DEV_DIRECTION_SEED,
        "levels_cm": list(levels),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
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
