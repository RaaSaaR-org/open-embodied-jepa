"""TASK-067 redesign probe (owner ruling R9 step 3): place-then-open at the plate centre.

The one pre-declared expert change. It targets the cause the landing diagnosis identified: a
~15 cm drop, a bounce to the rim, and a roll that comes to rest against the rim outside the 4 cm
radius. Protocol and pass bar:
``docs/experiments/apple_first_policy_v1_landing_diagnosis.md`` §B. Scripted and privileged only:
no learned policy, no corpus read, no test split.

The redesigned expert (``PlaceThenOpen``) is ``apple_collector_policy`` with ``transfer_x_shift``
0.0, so the held apple is over the plate centre. Its phases 5-7 are replaced, within the same
745-command budget:

* ``lower_closed`` (100): the grasp stays closed while the palm lowers to
  ``rest_z + HOLD_OFFSET + CLEARANCE``, which puts the apple about 0.5 cm above its resting
  height on the plate;
* ``open`` (100): at that pose the hand opens on the collector's ramp (0.08 per command);
* ``retreat`` (80): back to the transfer pose, with the hand open.

Seeds 46864-46895 are fresh: they are disjoint from every spent or allocated range, including
46800-46831 (probe and diagnosis) and 46832-46863 (reserved for v2 calibration). The plate
errors are 0, 1.0 and 1.5 cm, in a fixed direction per seed from ``default_rng(6820)``. The
settle period and the logging are the diagnosis script's. **The gate reads the scorer's success
latch**, as in C0 and the gated protocol. The at-rest state after settling is reported only.

    uv run --no-sync python scripts/probe_first_policy_place.py \\
        --output outputs/task067-place-probe/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy as fp  # noqa: E402

SEEDS = tuple(range(46864, 46896))
DIRECTION_SEED = 6820
PLATE_LEVELS_CM = (0.0, 1.0, 1.5)
HOLD_OFFSET_M = 0.0444  # palm-over-apple height while carried (diagnosis logs, see §B)
CLEARANCE_M = 0.005
OPENING_RAMP = 0.08  # the collector's own
PASS = {"successes_at_exact": 28, "successes_at_1cm": 28}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAG = _load("_diagnose_landing", "scripts/diagnose_first_policy_landing.py")
RUNNER = DIAG.RUNNER


def check_seeds() -> None:
    seeds = set(SEEDS)
    forbidden = (
        set(range(46000, 46416))
        | set(range(46800, 46864))
        | set(fp.COHORT_C)
        | set(fp.COHORT_D)
        | set(range(fp.SMOKE_SEEDS[0], fp.SMOKE_SEEDS[1] + 1))
    )
    for low, high in fp.FORBIDDEN_RANGES.values():
        forbidden |= set(range(low, high + 1))
    if seeds & forbidden or len(seeds) != 32:
        raise fp.GuardError("place probe: seeds overlap a spent or reserved range")


class PlaceThenOpen:
    """The one redesigned expert (scripted, privileged). Same 745-command budget."""

    def __init__(self, truth):
        from embodied_jepa.scripted import apple_collector_policy

        policy = apple_collector_policy(truth, transfer_x_shift=0.0)
        base = np.asarray(truth["base_position_world"], float)
        rest_z = truth["container_surface_z"] + truth["object_support_height"] - base[2]
        transfer = policy.phases[4].target_base.copy()
        place = np.array([transfer[0], transfer[1], rest_z + HOLD_OFFSET_M + CLEARANCE_M])
        policy.phases = (
            *policy.phases[:5],
            replace(policy.phases[5], name="lower_closed", target_base=place, grasp=1.0),
            replace(policy.phases[6], name="open", target_base=place.copy(), grasp=-1.0),
            replace(policy.phases[7], name="retreat", target_base=transfer.copy(), grasp=-1.0),
        )
        if tuple(p.commands for p in policy.phases) != fp.PHASE_BUDGETS:
            raise fp.GuardError("PlaceThenOpen must keep the collector's phase budgets")
        self.policy = policy
        self.accepted_grasp = -1.0

    @property
    def done(self):
        return self.policy.done

    @property
    def phase_index(self):
        return self.policy.phase_index

    def action(self, robot):
        from embodied_jepa.scripted import OracleManipulationPolicy

        action = OracleManipulationPolicy.action(self.policy, robot)  # no early-release ramp
        if self.policy.phase_index >= 6:  # open (and retreat) on the collector's ramp
            action[13] = max(-1.0, self.accepted_grasp - OPENING_RAMP)
        return action

    def advance(self, result):
        from embodied_jepa.scripted import OracleManipulationPolicy

        OracleManipulationPolicy.advance(self.policy, result)
        if result.applied_action is not None:
            self.accepted_grasp = float(result.applied_action[13])


class _Wrap:
    def __init__(self, truth, robot):
        self.policy = PlaceThenOpen(truth)
        self.robot = robot


def init_worker() -> None:
    RUNNER.worker_init()


def run_one(task: dict) -> dict:
    # The diagnosis loop, with the redesigned expert in command.
    original = DIAG.Collector
    DIAG.Collector = _Wrap
    try:
        out = DIAG.run_one(task)
    finally:
        DIAG.Collector = original
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    check_seeds()
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
                    "npz": str(args.output / "attempts" / f"place-{level}-{seed}.npz"),
                }
            )
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(fp.SIM_WORKERS, initializer=init_worker)
    try:
        rows = pool.map(run_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    summary = {}
    for level in PLATE_LEVELS_CM:
        cell = [r for r in rows if r["ok"] and r["plate_cm"] == level]
        summary[str(level)] = {
            "attempts": len(cell),
            "errors": sum(1 for r in rows if not r["ok"] and r["plate_cm"] == level),
            "successes_scorer_latch": sum(r["ever_success"] for r in cell),
            "successes_at_rest_after_settle": sum(r["final_success"] for r in cell),
            "final_distance_cm_q10_q50_q90": np.quantile(
                [r["final_distance_cm"] for r in cell], [0.1, 0.5, 0.9]
            ).tolist()
            if cell
            else None,
            "stop_reasons": sorted({r["stop_reason"] for r in cell}),
        }
    errors = sum(v["errors"] for v in summary.values())
    exact = summary["0.0"]["successes_scorer_latch"]
    one = summary["1.0"]["successes_scorer_latch"]
    row = (
        "VOID"
        if errors
        else "PASS"
        if exact >= PASS["successes_at_exact"] and one >= PASS["successes_at_1cm"]
        else "FAIL"
    )
    report = {
        "probe": "TASK-067 R9 step 3: place-then-open at the plate centre",
        "seeds": list(SEEDS),
        "direction_seed": DIRECTION_SEED,
        "hold_offset_m": HOLD_OFFSET_M,
        "clearance_m": CLEARANCE_M,
        "plate_levels_cm": list(PLATE_LEVELS_CM),
        "pass": PASS,
        "revision": RUNNER.R65.revision(),
        "tracked_tree_dirty": bool(RUNNER.R65.tracked_tree_dirty()),
        "seconds": time.monotonic() - started,
        "summary": summary,
        "row": row,
        "attempts": rows,
    }
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(payload)
    print(
        json.dumps(
            {
                "row": row,
                "summary": summary,
                "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
