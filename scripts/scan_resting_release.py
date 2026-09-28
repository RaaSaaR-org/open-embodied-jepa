"""TASK-068 single-seed scan of the release orientation and opening ramp (development only).

Engineering on a privileged scripted controller, not a learned result, and not a gated run.
Each grid entry is ``RestingPlaceExpert`` keyword arguments plus an optional ``rpy``: a
rotation (roll, pitch, yaw about the base axes, radians) applied before the collector's
palm-down rotation during lower, steady, open and clear. One attempt per entry, plate exact,
on one development seed. It prints how the apple leaves the hand and lands; the protocol's §5
logs the outputs.

    uv run --no-sync python scripts/scan_resting_release.py --seed 50003 \\
        --grid '[{"opening_ramp": 0.04, "rpy": [0, 0.6, 0]}]'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_runtime as rt  # noqa: E402
from embodied_jepa import resting_expert as rx  # noqa: E402
from embodied_jepa.embodiment import rotation_delta  # noqa: E402
from embodied_jepa.scripted import RestingPlaceExpert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid", required=True, help="JSON list of parameter dicts")
    args = parser.parse_args()
    rx.check_seeds((args.seed,))
    if args.seed not in rx.DEV_SEEDS:
        raise SystemExit("scan seeds must be development seeds")
    spec = importlib.util.spec_from_file_location("_eval", ROOT / "scripts" / "evaluate_apple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = module.wide_reset(args.seed)
    reset = {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])}
    robot = rt.make_robot()
    bounds = rt.configured_bounds()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    print(json.dumps({"seed": args.seed, "revision": revision}))
    for entry in json.loads(args.grid):
        params = dict(entry)
        rpy = params.pop("rpy", None)

        def factory(truth, params=params, rpy=rpy):
            expert = RestingPlaceExpert(truth, **params)
            if rpy is not None:
                expert.release_rotation = rotation_delta(rpy) @ expert.pick_rotation
            return expert

        s, a = rx.run_attempt(
            robot, bounds, seed=args.seed, reset=reset, plate_offset=[0, 0], make_expert=factory
        )
        last = s.get("last_apple_hand_step")
        after = last + 1 if last is not None and last + 1 < len(a["apple_lin"]) else None
        print(
            json.dumps(
                {
                    "entry": entry,
                    "stop": s["stop_reason"],
                    "at_rest": s["at_rest"],
                    "latched": s["latched_success"],
                    "final_distance_cm": round(s["final_distance_cm"], 2),
                    "apple_height_above_rest_at_open_cm": round(
                        s.get("apple_height_above_rest_at_open_cm", float("nan")), 1
                    ),
                    "apple_xy_velocity_after_hand_m_s": None
                    if after is None
                    else np.round(a["apple_lin"][after][:2], 3).tolist(),
                    "apple_spin_after_hand_rad_s": None
                    if after is None
                    else np.round(a["apple_ang"][after], 1).tolist(),
                    "landing_xy_minus_plate_cm": s.get("landing_xy_minus_plate_cm"),
                    "horizontal_speed_at_landing_m_s": s.get("horizontal_speed_at_landing_m_s"),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
