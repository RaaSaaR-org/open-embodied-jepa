"""TASK-072 pre-run check: TASK-071's render-determinism check, on the Linux PC at 16 workers.

This file is ``scripts/check_first_policy_v2_render.py`` with the substitutions listed in
``tests/test_first_policy_v2_linux.py`` (a test pins that) and this docstring. It drives
``run_first_policy_v2_linux.py``'s worker pool (EGL rendering), checks the platform, and gives
``IDENTICAL`` only at the gated worker count, 16. Everything else is TASK-071's check: the smoke
seeds 52100-52131, four renders per seed including after full-length attempts, cross-worker
coverage, the post-look state, and a negative control that must raise G-frame.

    uv run --no-sync python scripts/check_first_policy_v2_linux_render.py \\
        --output outputs/task072-scratch/render-determinism-1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2 as fp  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402

SEEDS = 32
MIN_CROSS_WORKER = 16
PASS3_KINDS = ("hold", "oracle", "random")  # varied scenes before the pass-4 render
NEGATIVE_SEED = fp.SMOKE_SEEDS[0] + SEEDS  # 52132: the G-frame negative control
LONG_STEPS = fp.MAX_POLICY_STEPS  # pass 3: full-length attempts (then the settle), as gated


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy_v2_linux", ROOT / "scripts" / "run_first_policy_v2_linux.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def init_worker() -> None:
    """The runner's worker initialiser, reached through this module (spawn pickles by name)."""
    RUNNER.worker_init()


def run_with_pid(task: dict) -> dict:
    """The runner's ``run_task``, plus the worker's pid; a worker exception is returned."""
    try:
        out = RUNNER.run_task(task)
        out.pop("frame", None)
        return {
            "ok": True,
            "pid": os.getpid(),
            "seed": task["seed"],
            "pass": task["pass"],
            "sha": out["post_look_frame_sha256"],
            "state": out["post_look_state"],
            "executed_steps": out.get("executed_steps"),
            "termination_reason": out.get("termination_reason"),
        }
    except Exception as error:  # noqa: BLE001 - recorded, and it fails the verdict
        return {
            "ok": False,
            "pid": os.getpid(),
            "seed": task["seed"],
            "pass": task["pass"],
            "error": f"{type(error).__name__}: {error}",
        }


def check(output: Path, workers: int) -> dict:
    import multiprocessing as mp

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    low, _high = fp.SMOKE_SEEDS
    seeds = tuple(range(low, low + SEEDS))
    RUNNER.check_simulated_seeds("D", seeds, smoke=True)  # smoke seeds only
    wide_reset = RUNNER.load_wide_reset()
    resets = {
        s: {
            "object_xy": list(map(float, wide_reset(s)["object_xy"])),
            "plate_xy": list(map(float, wide_reset(s)["plate_xy"])),
        }
        for s in seeds
    }
    RUNNER.check_simulated_seeds("D", (NEGATIVE_SEED,), smoke=True)
    negative_reset = {
        "object_xy": list(map(float, wide_reset(NEGATIVE_SEED)["object_xy"])),
        "plate_xy": list(map(float, wide_reset(NEGATIVE_SEED)["plate_xy"])),
    }
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(workers, initializer=init_worker)
    try:
        first = pool.map(
            run_with_pid,
            [{"seed": s, "reset": resets[s], "kind": "frame", "pass": 1} for s in seeds],
            chunksize=1,
        )
        second = pool.map(
            run_with_pid,
            [{"seed": s, "reset": resets[s], "kind": "frame", "pass": 2} for s in reversed(seeds)],
            chunksize=1,
        )
        sha1 = {r["seed"]: r.get("sha") for r in first}
        third = pool.map(
            run_with_pid,
            [
                {
                    "seed": s,
                    "reset": resets[s],
                    "kind": PASS3_KINDS[i % len(PASS3_KINDS)],  # varied scenes before pass 4
                    "pass": 3,
                    "max_steps": LONG_STEPS,
                    "expected_frame_sha256": sha1[s],
                }
                for i, s in enumerate(seeds)
            ],
            chunksize=1,
        )
        # pass 4: render again after every worker has run full-length attempts (pass 3), as the
        # gated run's workers do between DAgger and M1 attempts
        fourth = pool.map(
            run_with_pid,
            [{"seed": s, "reset": resets[s], "kind": "frame", "pass": 4} for s in seeds[::-1]],
            chunksize=1,
        )
        # negative control: a wrong expectation must raise G-frame in the worker
        negative = pool.map(
            run_with_pid,
            [
                {
                    "seed": NEGATIVE_SEED,
                    "reset": negative_reset,
                    "kind": "hold",
                    "pass": "negative",
                    "max_steps": 1,
                    "expected_frame_sha256": "0" * 64,
                }
            ],
            chunksize=1,
        )[0]
    finally:
        pool.terminate()
        pool.join()
    records = first + second + third + fourth
    per_seed = {}
    for s in seeds:
        rows = [r for r in records if r["seed"] == s]
        per_seed[str(s)] = {
            "shas": sorted({r["sha"] for r in rows if r["ok"]}),
            "workers": sorted({r["pid"] for r in rows}),
            "errors": [r["error"] for r in rows if not r["ok"]],
            "pass_3": [
                {
                    "kind": PASS3_KINDS[seeds.index(s) % len(PASS3_KINDS)],
                    "executed_steps": r.get("executed_steps"),
                    "termination_reason": r.get("termination_reason"),
                }
                for r in rows
                if r["pass"] == 3 and r["ok"]
            ],
            "state_max_abs_difference": float(
                max(
                    np.abs(np.asarray(r["state"]) - np.asarray(rows[0]["state"])).max()
                    for r in rows
                    if r["ok"]
                )
                if rows[0]["ok"]
                else np.inf
            ),
        }
    identical = all(
        len(v["shas"]) == 1 and not v["errors"] and v["state_max_abs_difference"] == 0
        for v in per_seed.values()
    )
    cross = sum(len(v["workers"]) >= 2 for v in per_seed.values())
    all_states = [np.asarray(r["state"]) for r in records if r["ok"]]
    states_equal = bool(all_states) and (
        max(float(np.abs(x - all_states[0]).max()) for x in all_states) <= 1e-6
    )
    negative_ok = (not negative["ok"]) and "G-frame" in negative.get("error", "")
    at_gated_count = workers == fpl.SIM_WORKERS
    report = {
        "check": "TASK-072 (TASK-067 R7, on Linux): post-look re-render bit-identity",
        "platform": fpl.check_platform(),
        "workers": workers,
        "gated_worker_count": fpl.SIM_WORKERS,
        "seeds": list(seeds),
        "renders_per_seed": 4,
        "pass_3_steps": LONG_STEPS,
        "distinct_workers_used": len({r["pid"] for r in records}),
        "seeds_rendered_on_two_or_more_workers": cross,
        "min_cross_worker_seeds": MIN_CROSS_WORKER,
        "post_look_state_equal_across_seeds": states_equal,
        "negative_control": {
            "seed": NEGATIVE_SEED,
            "raised_g_frame": negative_ok,
            "error": negative.get("error"),
        },
        "at_gated_worker_count": at_gated_count,
        "pass_3_kinds": list(PASS3_KINDS),
        "per_seed": per_seed,
        "seconds": time.monotonic() - started,
        "revision": RUNNER.R65.revision(),
        "tracked_tree_dirty": bool(RUNNER.R65.tracked_tree_dirty()),
        "verdict": "IDENTICAL"
        if identical
        and cross >= MIN_CROSS_WORKER
        and states_equal
        and negative_ok
        and at_gated_count
        else "NOT_IDENTICAL",
    }
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (output / "report.json").write_text(payload)
    report["report_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=fpl.SIM_WORKERS)
    args = parser.parse_args()
    report = check(args.output, args.workers)
    summary = {
        k: report[k]
        for k in (
            "verdict",
            "workers",
            "distinct_workers_used",
            "seeds_rendered_on_two_or_more_workers",
            "seconds",
            "revision",
            "tracked_tree_dirty",
            "report_sha256",
        )
    }
    print(json.dumps(summary, indent=1))
    return 0 if report["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
