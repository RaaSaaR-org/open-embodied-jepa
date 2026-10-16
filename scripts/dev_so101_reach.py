"""Development only (SO-101 S1, TASK-097; not gated, no claim): run the privileged scripted
reach, grasp and lift of ``embodied_jepa.so101_scripted`` and print one JSON record per seed.

    MUJOCO_GL=egl uv run --no-sync python scripts/dev_so101_reach.py --seeds 0 1 2 --frames out/

``--frames`` saves both cameras after each phase. Scripted successes are never learned results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_jepa.so101 import SO101Embodiment
from embodied_jepa.so101_scripted import scripted_lift
from embodied_jepa.so101_simulation import SO101Simulation


def run(seed, frames=None):
    sim = SO101Simulation(width=224, height=224)
    robot = SO101Embodiment(sim)

    def save(name):
        from PIL import Image

        for camera, image in sim.read()["rgb"].items():
            Image.fromarray(image).save(Path(frames) / f"seed{seed}_{name}_{camera}.png")

    try:
        return scripted_lift(robot, seed, on_phase=None if frames is None else save)
    finally:
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--frames", type=Path, default=None)
    args = parser.parse_args()
    if args.frames is not None:
        args.frames.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(json.dumps(run(seed, args.frames)))


if __name__ == "__main__":
    main()
