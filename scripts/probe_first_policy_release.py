"""TASK-067 release-point probe (owner ruling R8): where does the scripted expert's apple land?

A diagnostic, disclosed, with no gate. Scripted and privileged only: no learned policy, no corpus
read, no test split. Protocol and pre-declared readings:
``docs/experiments/apple_first_policy_v1_release_probe.md``.

Arms (both ``scripted.apple_collector_policy``, built from the reset truth plus a plate error):

* ``current``   -- the collector as used for ``apple-look-v1`` (``transfer_x_shift = -0.035``);
* ``candidate`` -- releases with the held apple over the plate centre (``transfer_x_shift = 0.0``:
  the palm then targets ``container_x - 0.015`` and the apple sits 1.5 cm ahead of the palm).

Conditions: plate error 0 (reference), 1.0 and 1.5 cm, in a fixed direction per seed drawn from
``default_rng(6810)``; the apple is exact. The same 32 seeds (46800-46831) run every condition,
after the look, up to the 800-step cap, through the runner's worker pool (8 workers, CPU).

Recorded per attempt, from simulator truth at the attempt's end: the apple's xy minus the true
plate centre's xy, success, grasp, transport, dropped, termination and steps.

    uv run --no-sync python scripts/probe_first_policy_release.py \\
        --output outputs/task067-release-probe/run-1
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

PROBE_SEEDS = tuple(range(46800, 46832))  # fresh; see the probe document §2
DIRECTION_SEED = 6810
PLATE_LEVELS_CM = (0.0, 1.0, 1.5)
ARMS = {"current": -0.035, "candidate": 0.0}  # transfer_x_shift
PLATE_RADIUS_M = 0.04  # task.TaskThresholds.plate_radius_m


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy", ROOT / "scripts" / "run_first_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def check_seeds() -> None:
    """Fresh seeds only: disjoint from the spent 46000-46415, cohorts C and D, the smoke range
    46900-46999 and the v2 calibration seeds reserved in the probe document (46832-46863)."""
    seeds = set(PROBE_SEEDS)
    forbidden = (
        set(range(46000, 46416))
        | set(fp.COHORT_C)
        | set(fp.COHORT_D)
        | set(range(fp.SMOKE_SEEDS[0], fp.SMOKE_SEEDS[1] + 1))
        | set(range(46832, 46864))
    )
    for low, high in fp.FORBIDDEN_RANGES.values():
        forbidden |= set(range(low, high + 1))
    if seeds & forbidden or len(seeds) != 32:
        raise fp.GuardError("release probe: seeds overlap a spent or reserved range")


def directions() -> np.ndarray:
    angles = np.random.default_rng(DIRECTION_SEED).uniform(0, 2 * np.pi, len(PROBE_SEEDS))
    return np.c_[np.cos(angles), np.sin(angles)]


class ShiftedCollector:
    """``apple_collector_policy`` with a declared ``transfer_x_shift`` (scripted, privileged)."""

    def __init__(self, truth, robot, shift):
        from embodied_jepa.scripted import apple_collector_policy

        self.policy = apple_collector_policy(truth, transfer_x_shift=shift)
        self.robot = robot

    def act(self, observation, step):  # noqa: ARG002 - reads the robot pose, as the collector
        return np.asarray(self.policy.action(self.robot), np.float32)

    def advance(self, result):
        if not self.policy.done:
            self.policy.advance(result)


def init_worker() -> None:
    RUNNER.worker_init()


def run_one(task: dict) -> dict:
    try:
        robot = RUNNER._W["robot"]
        truth, scorer, counter, _obs, _facts = rt.reset_and_look(robot, task["seed"], task["reset"])
        perturbed = rt.perturbed_truth(truth, [0.0, 0.0], task["plate_offset"])
        controller = ShiftedCollector(perturbed, robot, ARMS[task["arm"]])
        record = rt.run_policy_steps(robot, scorer, counter, controller, bounds=RUNNER._W["bounds"])
        end = robot.sim.task_truth()
        landing = np.asarray(end["object_position"])[:2] - np.asarray(end["plate_position"])[:2]
        return {
            "ok": True,
            **{k: task[k] for k in ("seed", "arm", "plate_cm")},
            "landing_dx_m": float(landing[0]),
            "landing_dy_m": float(landing[1]),
            "landing_distance_m": float(np.linalg.norm(landing)),
            "success": record["success"],
            "grasp": record["grasp"],
            "transport": bool(record["final_score"].get("transport", False)),
            "dropped": bool(end["dropped"]),
            "termination_reason": record["termination_reason"],
            "executed_steps": record["executed_steps"],
        }
    except Exception as error:  # noqa: BLE001 - recorded; any error is reported
        return {
            "ok": False,
            **{k: task[k] for k in ("seed", "arm", "plate_cm")},
            "error": f"{type(error).__name__}: {error}",
        }


def summarise(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        for level in PLATE_LEVELS_CM:
            cell = [r for r in rows if r["ok"] and r["arm"] == arm and r["plate_cm"] == level]
            dx = np.array([r["landing_dx_m"] for r in cell]) * 100
            dist = np.array([r["landing_distance_m"] for r in cell]) * 100
            failures = [r for r in cell if not r["success"]]
            placed_out = [
                r for r in failures if r["transport"] and r["landing_distance_m"] > PLATE_RADIUS_M
            ]
            out[f"{arm}@{level}"] = {
                "attempts": len(cell),
                "errors": sum(
                    1 for r in rows if not r["ok"] and r["arm"] == arm and r["plate_cm"] == level
                ),
                "successes": sum(r["success"] for r in cell),
                "grasps": sum(r["grasp"] for r in cell),
                "transports": sum(r["transport"] for r in cell),
                "landing_dx_cm_q10_q50_q90": np.quantile(dx, [0.1, 0.5, 0.9]).tolist()
                if len(dx)
                else None,
                "landing_distance_cm_q10_q50_q90": np.quantile(dist, [0.1, 0.5, 0.9]).tolist()
                if len(dist)
                else None,
                "failures": len(failures),
                "failures_transported_and_landed_outside_radius": len(placed_out),
            }
    return out


def decide(summary: dict) -> dict:
    """The pre-declared reading (probe document §4); the first matching row."""
    ref = summary["current@0.0"]
    cur = summary["current@1.0"]
    cand = summary["candidate@1.0"]
    median_dx = ref["landing_dx_cm_q10_q50_q90"][1]
    short = -4.5 <= median_dx <= -2.5
    placement = cur["failures"] > 0 and (
        cur["failures_transported_and_landed_outside_radius"] >= 0.8 * cur["failures"]
    )
    confirmed = short and placement
    if any(v["errors"] for v in summary.values()):
        row = "P-VOID"
    elif not confirmed:
        row = "P-MECH-REFUTED"
    elif cand["successes"] < 28:
        row = "P-CANDIDATE-FAIL"
    else:
        row = "P-SUPPORTED"
    return {
        "row": row,
        "current_reference_median_dx_cm": median_dx,
        "short_band_ok": short,
        "placement_explains_failures": placement,
        "candidate_successes_at_1cm": cand["successes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    check_seeds()
    wide_reset = RUNNER.load_wide_reset()
    unit = directions()
    tasks = []
    for arm in ARMS:
        for level in PLATE_LEVELS_CM:
            for i, seed in enumerate(PROBE_SEEDS):
                r = wide_reset(seed)
                tasks.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "plate_cm": level,
                        "reset": {
                            "object_xy": list(map(float, r["object_xy"])),
                            "plate_xy": list(map(float, r["plate_xy"])),
                        },
                        "plate_offset": (level / 100.0 * unit[i]).tolist(),
                    }
                )
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(fp.SIM_WORKERS, initializer=init_worker)
    try:
        rows = pool.map(run_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    summary = summarise(rows)
    report = {
        "probe": "TASK-067 R8 release-point probe",
        "seeds": list(PROBE_SEEDS),
        "direction_seed": DIRECTION_SEED,
        "arms_transfer_x_shift": ARMS,
        "plate_levels_cm": list(PLATE_LEVELS_CM),
        "revision": RUNNER.R65.revision(),
        "tracked_tree_dirty": bool(RUNNER.R65.tracked_tree_dirty()),
        "seconds": time.monotonic() - started,
        "summary": summary,
        "decision": decide(summary),
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(payload)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
