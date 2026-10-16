"""Development measurement for SO-101 S2 (TASK-098; not gated, no claim).

Three subcommands, each writing into a new ``--output`` directory (never overwritten):

- ``reach``: the kinematic reach maps of the scripted followers' grasp and place poses (IK
  residual and collisions on a grid), from which the declared reset regions were chosen.
- ``measure``: the privileged scripted follower on development seeds for
  ``so101_cube_lift_v0`` and ``so101_cube_place_v0`` (success counts with exact 95 % intervals),
  and the K-style placement-tolerance envelope: the follower with rigid grasp-aim offsets
  (lift and place) and world-xy place-aim offsets (place) of several sizes and directions, on
  the same seeds, so cells are paired by reset.
- ``frames``: both cameras at 224 px at a few phases of a few episodes, saved as PNGs and one
  contact sheet per task (EGL rendering: run it through ``scripts/gpu_run.sh --wait --board``).

    uv run --no-sync python scripts/measure_so101_tasks.py reach --output outputs/<new>
    uv run --no-sync python scripts/measure_so101_tasks.py measure --output outputs/<new>
    MUJOCO_GL=egl scripts/gpu_run.sh --wait --board -- \\
        uv run --no-sync python scripts/measure_so101_tasks.py frames --output outputs/<new>

``measure`` and ``reach`` never render: the followers read simulator truth, not images, so the
transport's renderer is replaced by an 8 x 8 black frame. Scripted successes are never learned
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Development seeds of the reported measurement (follower v1). Seeds 0-63 were used by the
# first measurement (follower v0 at 925153f); the followers were developed on seeds 1000-1015.
DEV_SEEDS = tuple(range(100, 164))
GRASP_SIZES_M = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
# In the zero-offset grasp pose's frame: +closing is the fingertip site's +z (the gripper moves
# toward the moving jaw's side, so the fixed jaw moves toward the cube), +along its +y.
GRASP_DIRECTIONS = {
    "+closing": (1, 0, 0),
    "-closing": (-1, 0, 0),
    "+along": (0, 1, 0),
    "-along": (0, -1, 0),
    "+up": (0, 0, 1),
    "-up": (0, 0, -1),
}
PLACE_SIZES_M = (0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05)
PLACE_DIRECTIONS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)


def exact_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval by bisection on the binomial tails (no SciPy)."""
    alpha = 1 - level

    def tail(p, low, high):
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(low, high + 1))

    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: tail(p, k, n), alpha / 2, True)
    upper = 1.0 if k == n else solve(lambda p: tail(p, 0, k), alpha / 2, False)
    return lower, upper


def _robot():
    from embodied_jepa.so101 import SO101Embodiment
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(render=False)
    # No image is read by the followers or the scorers; observe() still needs a frame.
    sim.render = lambda camera="wrist_rgb": np.zeros((8, 8, 3), np.uint8)
    return SO101Embodiment(sim)


def _provenance(argv):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    def sha(path):
        return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

    files = [
        "src/embodied_jepa/so101_tasks.py",
        "src/embodied_jepa/so101_followers.py",
        "src/embodied_jepa/so101_scripted.py",
        "src/embodied_jepa/so101_simulation.py",
        "src/embodied_jepa/so101.py",
        "configs/so101_sim_action.json",
        "assets/so101_manifest.json",
        "scripts/measure_so101_tasks.py",
    ]
    import mujoco

    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "file_sha256": {f: sha(f) for f in files},
        "argv": argv,
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.node(),
        "label": "development only; privileged scripted follower; not gated; not learned",
    }


def _new_output(path: Path) -> Path:
    if path.exists():
        raise SystemExit(f"refusing: {path} exists; use a new --output")
    path.mkdir(parents=True)
    return path


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ----- reach ----------------------------------------------------------------------------------
def reach(args):
    from embodied_jepa.so101_followers import FOLLOWER_V1, reach_map

    out = _new_output(args.output)
    robot = _robot()
    robot.reset(0)
    xs = np.round(np.arange(0.12, 0.295, 0.01), 4)
    started = time.perf_counter()
    grasp = reach_map(
        robot,
        xs=xs,
        ys=np.round(np.arange(-0.16, 0.165, 0.02), 4),
        yaws=np.deg2rad([-45, -30, -15, 0, 15, 30, 45]),
        heights=(0.0, FOLLOWER_V1.pregrasp_height_m),
        mode="grasp",
    )
    place = reach_map(
        robot,
        xs=xs,
        ys=np.round(np.arange(-0.20, 0.205, 0.02), 4),
        yaws=(0.0,),
        heights=(0.0, FOLLOWER_V1.pregrasp_height_m),
        mode="place",
    )
    _write(
        out / "reach.json",
        {
            "provenance": _provenance(sys.argv),
            "seconds": time.perf_counter() - started,
            "grasp": grasp,
            "place": place,
        },
    )
    for name, rows in (("grasp", grasp), ("place", place)):
        print(f"{name}: ok cells {sum(r['ok'] for r in rows)}/{len(rows)}")
        print(_ascii(rows))


def _ascii(rows):
    ys = sorted({r["y"] for r in rows})
    xs = sorted({r["x"] for r in rows})
    cell = {(r["x"], r["y"]): r for r in rows}
    lines = ["  x\\y " + " ".join(f"{y:+.2f}" for y in ys)]
    for x in xs:
        marks = []
        for y in ys:
            r = cell[(x, y)]
            if r["ok"]:
                marks.append("  ok ")
            elif r["poses_in_collision"]:
                marks.append(f"  c{r['poses_in_collision']:<2d}")
            else:
                marks.append(f"r{1000 * r['worst_ik_residual_m']:4.1f}")
        lines.append(f"{x:.2f}  " + " ".join(marks))
    return "\n".join(lines)


# ----- measure --------------------------------------------------------------------------------
def _cells():
    cells = [("lift", "baseline", (0.0, 0.0, 0.0), (0.0, 0.0))]
    cells.append(("place", "baseline", (0.0, 0.0, 0.0), (0.0, 0.0)))
    for task in ("lift", "place"):
        for name, direction in GRASP_DIRECTIONS.items():
            for size in GRASP_SIZES_M:
                offset = tuple(size * np.asarray(direction, float))
                cells.append((task, f"grasp {name} {1000 * size:g} mm", offset, (0.0, 0.0)))
    for degrees in PLACE_DIRECTIONS_DEG:
        for size in PLACE_SIZES_M:
            angle = math.radians(degrees)
            offset = (size * math.cos(angle), size * math.sin(angle))
            cells.append(
                ("place", f"place {degrees}deg {1000 * size:g} mm", (0.0, 0.0, 0.0), offset)
            )
    return cells


def _run_cell(job):
    task_name, label, grasp_offset, place_offset, seeds = job
    from embodied_jepa.so101_followers import follow_lift, follow_place
    from embodied_jepa.so101_tasks import SO101CubeLiftTask, SO101CubePlaceTask

    robot = _robot()
    records = []
    started = time.perf_counter()
    try:
        for seed in seeds:
            if task_name == "lift":
                record = follow_lift(
                    robot, SO101CubeLiftTask(robot), seed, grasp_offset_m=grasp_offset
                )
            else:
                record = follow_place(
                    robot,
                    SO101CubePlaceTask(robot),
                    seed,
                    grasp_offset_m=grasp_offset,
                    place_offset_m=place_offset,
                )
            record.pop("follower")
            records.append(record)
    finally:
        robot.close()
    return {
        "task": task_name,
        "cell": label,
        "grasp_offset_m": list(grasp_offset),
        "place_offset_m": list(place_offset),
        "seconds": time.perf_counter() - started,
        "records": records,
    }


def _summary(cell):
    records = cell["records"]
    n = len(records)
    k = sum(r["success"] for r in records)
    low, high = exact_interval(k, n)
    reasons = {}
    for r in records:
        reasons[r["termination_reason"]] = reasons.get(r["termination_reason"], 0) + 1
    row = {
        "task": cell["task"],
        "cell": cell["cell"],
        "grasp_offset_m": cell["grasp_offset_m"],
        "place_offset_m": cell["place_offset_m"],
        "n": n,
        "successes": k,
        "exact_95": [round(low, 4), round(high, 4)],
        "termination": dict(sorted(reasons.items())),
        "failed_seeds": [r["seed"] for r in records if not r["success"]],
    }
    if cell["task"] == "place":
        row["picked"] = sum(r["score"]["picked"] for r in records)
        offsets = np.array([r["score"]["final_offset_xy_m"] for r in records])
        error = np.linalg.norm(
            offsets - np.asarray(cell["place_offset_m"], float), axis=1
        )  # landing error relative to the (offset) aim
        distance = np.linalg.norm(offsets, axis=1)  # landing distance from the target centre
        row["final_distance_from_target_m"] = {
            "median": round(float(np.median(distance)), 4),
            "max": round(float(distance.max()), 4),
        }
        row["landing_error_from_aim_m"] = {
            "median": round(float(np.median(error)), 4),
            "max": round(float(error.max()), 4),
        }
    return row


def measure(args):
    from embodied_jepa.so101_followers import FOLLOWER_V1
    from embodied_jepa.so101_tasks import LIFT_V0, PLACE_V0, RESETS_V0, sample_reset

    out = _new_output(args.output)
    seeds = tuple(args.seeds)
    cells = _cells()
    if args.baseline_only:
        cells = [c for c in cells if c[1] == "baseline"]
    started = time.perf_counter()
    jobs = [(*cell, seeds) for cell in cells]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_cell, jobs))
    seconds = time.perf_counter() - started
    summary = [_summary(cell) for cell in results]
    _write(
        out / "summary.json",
        {
            "provenance": _provenance(sys.argv),
            "seeds": list(seeds),
            "thresholds": {"lift": asdict(LIFT_V0), "place": asdict(PLACE_V0)},
            "resets": asdict(RESETS_V0),
            "reset_specs": {
                "lift": {s: sample_reset("so101_cube_lift_v0", s) for s in seeds},
                "place": {s: sample_reset("so101_cube_place_v0", s) for s in seeds},
            },
            "follower": asdict(FOLLOWER_V1),
            "wall_seconds": seconds,
            "workers": args.workers,
            "cells": summary,
        },
    )
    with (out / "episodes.jsonl").open("w") as handle:
        for cell in results:
            for record in cell["records"]:
                handle.write(json.dumps({"cell": cell["cell"], **record}, default=float) + "\n")
    for row in summary:
        print(
            f"{row['task']:5s} {row['cell']:24s} {row['successes']:3d}/{row['n']} "
            f"[{row['exact_95'][0]:.3f}, {row['exact_95'][1]:.3f}] {row['termination']}"
        )
    print(f"wall {seconds:.1f} s")


# ----- frames ---------------------------------------------------------------------------------
def frames(args):
    from PIL import Image

    import embodied_jepa.so101_followers as followers
    from embodied_jepa.so101 import SO101Embodiment
    from embodied_jepa.so101_simulation import CAMERAS, SO101Simulation
    from embodied_jepa.so101_tasks import SO101CubeLiftTask, SO101CubePlaceTask

    out = _new_output(args.output)
    shots = {"lift": (0, 45, 70, 90, 125), "place": (0, 70, 120, 170, 200, 230)}
    sheet_rows = {"lift": [], "place": []}
    original_step = followers._Runner.step
    for task_name in ("lift", "place"):
        for seed in args.frame_seeds:
            sim = SO101Simulation(width=args.size, height=args.size)
            robot = SO101Embodiment(sim)
            task = (SO101CubeLiftTask if task_name == "lift" else SO101CubePlaceTask)(robot)
            images = {}

            def step(self, action, images=images, sim=sim, task_name=task_name):
                try:
                    original_step(self, action)
                finally:
                    count = sum(self.statuses.values())
                    if count in shots[task_name]:
                        images[count] = {c: sim.render(c) for c in CAMERAS}

            followers._Runner.step = step
            try:
                task.reset(seed)
                images[0] = {c: sim.render(c) for c in CAMERAS}
                runner = followers.follow_lift if task_name == "lift" else followers.follow_place
                record = runner(robot, task, seed)
            finally:
                followers._Runner.step = original_step
                robot.close()
            row = []
            for count in sorted(images):
                for camera in CAMERAS:
                    name = f"{task_name}_seed{seed}_step{count:03d}_{camera}.png"
                    Image.fromarray(images[count][camera]).save(out / name)
                row.append(np.concatenate([images[count][c] for c in CAMERAS], axis=0))
            sheet_rows[task_name].append(np.concatenate(row, axis=1))
            print(task_name, seed, record["termination_reason"], sorted(images))
    for task_name, rows in sheet_rows.items():
        width = max(r.shape[1] for r in rows)
        padded = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
        Image.fromarray(np.concatenate(padded, axis=0)).save(out / f"sheet_{task_name}.png")
    _write(out / "frames.json", {"provenance": _provenance(sys.argv), "shots": shots})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reach", "measure", "frames"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name == "measure":
            p.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
            p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
            p.add_argument("--baseline-only", action="store_true")
        if name == "frames":
            p.add_argument("--frame-seeds", type=int, nargs="+", default=[0, 1])
            p.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    {"reach": reach, "measure": measure, "frames": frames}[args.command](args)


if __name__ == "__main__":
    main()
