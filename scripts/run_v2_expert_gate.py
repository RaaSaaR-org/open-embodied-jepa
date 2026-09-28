"""TASK-070 gated run: the frozen v2 expert on the frozen gated seeds (apple-to-plate-v2).

Protocol: ``docs/experiments/apple_to_plate_v2_expert.md`` §6. Runs ONLY on the pre-run reviewer's
reported GO. Privileged scripted expert, not a learned result.

Before any episode it refuses: an existing output directory, a dirty tracked tree, any pinned
source whose sha256 differs from ``benchmarks/manifests/apple-to-plate-v2-expert-gate-v1.json``,
and gated seeds that overlap any other declared range. Every attempt: the v2 scene, the TASK-047
wide-jitter reset, the look, the frozen expert with the plate xy perturbed by the level in a fixed
direction per seed (``default_rng(GATE_DIRECTION_SEED)``), then the 60-step settle. The row is
decided by ``apple_to_plate_v2.gate_row``.

    uv run --no-sync python scripts/run_v2_expert_gate.py --output outputs/task070-gate/run-1
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

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-to-plate-v2-expert-gate-v1.json"
_W: dict = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_pins() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    found = {name: sha256(ROOT / name) for name in manifest["source_sha256"]}
    bad = [n for n, h in found.items() if h != manifest["source_sha256"][n]]
    if bad:
        raise SystemExit(f"pinned sources changed: {bad}")
    if manifest["expert"] != v2.GATE_EXPERT or manifest["seeds"] != list(v2.GATE_SEEDS):
        raise SystemExit("manifest and module disagree on the frozen expert or seeds")
    import mujoco

    if mujoco.__version__ != manifest["mujoco_version"]:
        raise SystemExit(
            f"MuJoCo {mujoco.__version__} is not the pinned {manifest['mujoco_version']}"
        )
    return found


def worker_init() -> None:
    from embodied_jepa import first_policy_runtime as rt

    _W["robot"], _W["scene"] = v2.make_v2_robot()
    _W["bounds"] = rt.configured_bounds()


def run_one(task: dict) -> dict:
    try:
        summary, arrays = rx.run_attempt(
            _W["robot"],
            _W["bounds"],
            seed=task["seed"],
            reset=task["reset"],
            plate_offset=task["plate_offset"],
            make_expert=lambda truth: rx.RestingPlaceExpert(truth, **v2.GATE_EXPERT),
        )
        np.savez_compressed(task["npz"], **arrays)
        return {"ok": True, "plate_cm": task["plate_cm"], "scene": _W["scene"], **summary}
    except Exception as error:  # noqa: BLE001 - an exception voids the run
        return {
            "ok": False,
            "seed": task["seed"],
            "plate_cm": task["plate_cm"],
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="2 development seeds; no verdict")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.smoke:
        raise SystemExit("refusing a gated run on a dirty tracked tree")
    pins = check_pins()
    seeds = v2.DEV_SEEDS[-2:] if args.smoke else v2.GATE_SEEDS
    if args.smoke:
        v2.check_seeds(seeds)
    else:
        v2.check_gate_seeds()
    spec = importlib.util.spec_from_file_location("_eval", ROOT / "scripts" / "evaluate_apple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (args.output / "attempts").mkdir(parents=True)
    tasks = []
    for level in v2.GATE_LEVELS_CM:
        offsets = rx.plate_offsets(len(seeds), level, v2.GATE_DIRECTION_SEED)
        for i, seed in enumerate(seeds):
            r = module.wide_reset(seed)
            tasks.append(
                {
                    "seed": seed,
                    "plate_cm": level,
                    "reset": {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])},
                    "plate_offset": offsets[i].tolist(),
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
    cells = {}
    for level in v2.GATE_LEVELS_CM:
        cell = [r for r in rows if r["plate_cm"] == level]
        ok = [r for r in cell if r["ok"]]
        landing = [
            r["horizontal_speed_at_landing_m_s"]
            for r in ok
            if "horizontal_speed_at_landing_m_s" in r
        ]
        cells[str(level)] = {
            "attempts": len(cell),
            "errors": len(cell) - len(ok),
            "guard_stops": sum(r["stop_reason"].startswith("guard") for r in ok),
            "other_stops": sum(
                (not r["complete"]) and not r["stop_reason"].startswith("guard") for r in ok
            ),
            "at_rest": sum(r["at_rest"] for r in ok),
            "latched": sum(r["latched_success"] for r in ok),
            "final_distance_cm_q10_q50_q90": np.quantile(
                [r["final_distance_cm"] for r in ok], [0.1, 0.5, 0.9]
            )
            .round(2)
            .tolist()
            if ok
            else None,
            "landing_speed_m_s_q10_q50_q90": np.quantile(landing, [0.1, 0.5, 0.9]).round(3).tolist()
            if landing
            else None,
        }
    row = None if args.smoke else v2.gate_row(cells)
    report = {
        "task": "TASK-070 v2 expert gate (privileged scripted expert; not a learned result)",
        "smoke": bool(args.smoke),
        "scene": v2.SCENE_VERSION,
        "expert": v2.GATE_EXPERT,
        "seeds": list(seeds),
        "direction_seed": v2.GATE_DIRECTION_SEED,
        "levels_cm": list(v2.GATE_LEVELS_CM),
        "bar": v2.GATE_BAR,
        "revision": revision,
        "tracked_tree_dirty": dirty,
        "revision_at_end": git("rev-parse", "HEAD"),
        "source_sha256": pins,
        "device": "cpu",
        "workers": args.workers,
        "seconds": time.monotonic() - started,
        "cells": cells,
        "row": row,
        "attempts": rows,
    }
    payload = json.dumps(report, indent=1, sort_keys=True, default=float) + "\n"
    (args.output / "report.json").write_text(payload)
    print(json.dumps({"row": row, "cells": cells}, indent=1))
    print("report sha256", hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
