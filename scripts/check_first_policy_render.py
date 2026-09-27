"""TASK-067 pre-run check (owner ruling R7, condition on amendment 1 item 5).

Is the post-look frame bit-identical across renders and across simulation workers, at the gated
worker count? The gated run re-renders every attempt's post-look frame and voids the run on any
sha256 mismatch (G-frame, part of S0-D1). Known render nondeterminism (1-2 px under 12 workers,
TASK-064 §16) could trip it, so this is checked before the pre-run GO, and the check is not
relaxed.

It uses the runner's own worker pool and ``run_task`` (the gated code path), the smoke seeds
46900-46999 only, and CPU only. For each seed:

* pass 1 renders the post-look frame (``frame`` task);
* pass 2 renders it again, with the task order reversed so it lands on other workers;
* pass 3 runs a one-step ``hold`` attempt with pass 1's sha256 as ``expected_frame_sha256``,
  exactly as a DAgger or M1 attempt does (a mismatch raises G-frame in the worker).

Verdict ``IDENTICAL`` only if every seed has one sha256 across all three passes and one post-look
joint state, no pass-3 attempt raised, and at least ``MIN_CROSS_WORKER`` seeds were rendered on
two or more distinct workers. Anything else is ``NOT_IDENTICAL``: the gated run must not start.

    uv run --no-sync python scripts/check_first_policy_render.py \\
        --output outputs/task067-scratch/render-determinism-1
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

from embodied_jepa import first_policy as fp  # noqa: E402

SEEDS = 32
MIN_CROSS_WORKER = 16


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy", ROOT / "scripts" / "run_first_policy.py"
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
                    "kind": "hold",
                    "pass": 3,
                    "max_steps": 1,
                    "expected_frame_sha256": sha1[s],
                }
                for s in seeds
            ],
            chunksize=1,
        )
    finally:
        pool.terminate()
        pool.join()
    records = first + second + third
    per_seed = {}
    for s in seeds:
        rows = [r for r in records if r["seed"] == s]
        per_seed[str(s)] = {
            "shas": sorted({r["sha"] for r in rows if r["ok"]}),
            "workers": sorted({r["pid"] for r in rows}),
            "errors": [r["error"] for r in rows if not r["ok"]],
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
    report = {
        "check": "TASK-067 R7: post-look re-render bit-identity across renders and workers",
        "workers": workers,
        "gated_worker_count": fp.SIM_WORKERS,
        "seeds": list(seeds),
        "renders_per_seed": 3,
        "distinct_workers_used": len({r["pid"] for r in records}),
        "seeds_rendered_on_two_or_more_workers": cross,
        "min_cross_worker_seeds": MIN_CROSS_WORKER,
        "post_look_state_equal_across_seeds": bool(
            max(float(np.abs(s - all_states[0]).max()) for s in all_states) <= 1e-6
        ),
        "per_seed": per_seed,
        "seconds": time.monotonic() - started,
        "revision": RUNNER.R65.revision(),
        "tracked_tree_dirty": bool(RUNNER.R65.tracked_tree_dirty()),
        "verdict": "IDENTICAL" if identical and cross >= MIN_CROSS_WORKER else "NOT_IDENTICAL",
    }
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (output / "report.json").write_text(payload)
    report["report_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=fp.SIM_WORKERS)
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
