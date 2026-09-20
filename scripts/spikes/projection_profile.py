"""Bounded projection profile against full-forward reference; no learned-control claims."""

import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import subprocess
import time
from pathlib import Path

import numpy as np

from embodied_jepa.embodiment import G1Embodiment
from embodied_jepa.simulation import MuJoCoSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("profile output must be new")
    args.output.mkdir(parents=True)
    start_cpu = time.process_time()
    start_wall = time.time()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "src/embodied_jepa/embodiment.py"]
    plan = {
        "seed": 341,
        "robot_seed": 41,
        "candidates": 64,
        "horizon": 4,
        "repeats": 3,
        "max_cpu_seconds": 60,
        "max_wall_seconds": 120,
        "states": ["reset", "after_six_physical_commands"],
        "reference": "identical acceptance code, full mj_forward scratch refresh",
        "optimized": "mj_kinematics plus mj_comPos scratch refresh",
        "fixture": "real G1/Dex3 physics with explicit synthetic RGB; cube/target proxies",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "platform": platform.platform(),
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def check_budget():
        if time.process_time() - start_cpu >= 60 or time.time() - start_wall >= 120:
            raise TimeoutError("fixed profiling budget reached")

    rng = np.random.default_rng(341)
    requests = rng.uniform(-0.5, 0.5, (1, 64, 4, 14)).astype(np.float32)
    requests[..., :6] = 0
    requests[..., 12] = -1
    requests[..., 13] = rng.uniform(-1, 1, (1, 64, 4))
    np.save(args.output / "requested.npy", requests)
    sim = MuJoCoSimulation(render=False, object_kind="cube", container_kind="target")
    sim.render = lambda: np.zeros((8, 8, 3), np.uint8)
    robot = G1Embodiment(sim)
    optimized_forward = robot._forward_kinematics
    records = []
    try:
        robot.reset(41)
        robot.observe()
        for state in plan["states"]:
            if state != "reset":
                command = np.zeros(14, np.float32)
                command[2] = command[8] = -0.3
                command[5] = command[11] = 0.25
                command[12:] = -1
                for _ in range(6):
                    check_budget()
                    result = robot.execute(command)
                    if result.applied_action is None:
                        raise RuntimeError(f"fixture rejected motion: {result.reason}")
                    robot.observe()
            outputs = {}
            timings = {}
            for name in ("reference", "optimized"):
                robot._forward_kinematics = (
                    (lambda data: robot.mj.mj_forward(robot.model, data))
                    if name == "reference"
                    else optimized_forward
                )
                values = []
                for _ in range(plan["repeats"]):
                    check_budget()
                    begin = time.perf_counter()
                    outputs[name] = robot.project_candidates(requests)
                    values.append(time.perf_counter() - begin)
                timings[name] = values
                profiler = cProfile.Profile()
                check_budget()
                profiler.enable()
                robot.project_candidates(requests)
                profiler.disable()
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(20)
                (args.output / f"{state}_{name}.txt").write_text(stream.getvalue())
            np.testing.assert_array_equal(
                outputs["reference"].actions, outputs["optimized"].actions
            )
            np.testing.assert_array_equal(
                outputs["reference"].feasible, outputs["optimized"].feasible
            )
            records.append(
                {
                    "state": state,
                    "timings_seconds": timings,
                    "bitwise_equal": True,
                    "feasible_count": int(outputs["optimized"].feasible.sum()),
                    "speedup": float(
                        np.median(timings["reference"]) / np.median(timings["optimized"])
                    ),
                }
            )
            robot._forward_kinematics = optimized_forward
            print(json.dumps(records[-1]), flush=True)
        report = {
            "states": records,
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.time() - start_wall,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
