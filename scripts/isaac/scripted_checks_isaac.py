"""Isaac side of the scripted apple physics checks (development, TASK-025).

Runs INSIDE the ``isaaclab_arena`` container via ``scripts/isaac/run_isaac.sh``. Every case in
``scripted_cases.CASES`` starts from ``IsaacTransport.reset`` and runs fixed joint targets; no
learned policy is involved and nothing here is a manipulation or task result. Per interval it
records the apple position and velocity (evaluator-only truth) and the contacts of the last
physics step. ``--physics`` picks the backend (PhysX by default, or Newton / MuJoCo-Warp); the
run record names it (``physics_backend``). ``scripted_checks_mujoco.py`` runs the same cases in
MuJoCo and compares.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Development diagnostic: two of ten Newton runs hung (one CPU core busy, nothing logged)
# before the model was built. Dump every thread's Python stack every 5 min to the log.
faulthandler.dump_traceback_later(300, repeat=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--physics", choices=("physx", "newton"), default="physx")
parser.add_argument(
    "--cases", nargs="*", default=None, help="subset of cases (development; recorded)"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)  # progress lines reach log.txt as they happen
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/oej/src")
from scripted_cases import CASES, targets  # noqa: E402

from embodied_jepa.isaac_scene import canonical_sha256, usd_canonical_hash  # noqa: E402
from embodied_jepa.isaac_transport import IsaacTransport, manifest_sha256, reset_pose  # noqa: E402


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    scene = json.loads(args.scene.read_text())
    tr = IsaacTransport(
        usd_path=args.usd,
        manifest=manifest,
        scene_manifest=scene,
        objects=True,
        render=False,
        record_contacts=True,
        physics=args.physics,
    )
    names = list(tr.joint_names)
    record = {
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "physics_backend": tr.physics,
        "newton_model": tr.newton_model,
        "manifest_sha256": manifest_sha256(manifest),
        "scene_manifest_sha256": canonical_sha256(scene),
        "usd_canonical_tree_sha256": usd_canonical_hash(args.usd),
        "object_properties": tr.object_properties,
        "contact_api_error": tr.contact_api_error,
        # None: every case in scripted_cases.CASES; otherwise a development subset.
        "cases_requested": list(args.cases) if args.cases else None,
        "cases": {},
    }
    for case, spec in CASES.items():
        if args.cases and case not in args.cases:
            continue
        tr.reset(0, object_xy=spec["reset_object_xy"], plate_xy=spec["plate_xy"])
        if spec["start_pos"] is not None:
            tr.set_object_state(spec["start_pos"], spec["lin_vel"], spec["ang_vel"])
        rows = {
            "time": [],
            "pos": [],
            "lin_vel": [],
            "ang_vel": [],
            "contacts": [],
            "qpos": [],
            "plate_pos": [],
        }

        def log(rows=rows):
            truth = tr.task_truth()
            vel = tr._np(tr.apple.data.root_com_vel_w).reshape(-1)
            rows["time"].append(tr.time)
            rows["pos"].append(truth["object_position"].tolist())
            rows["plate_pos"].append(truth["plate_position"].tolist())
            rows["lin_vel"].append(vel[:3].tolist())
            rows["ang_vel"].append(vel[3:6].tolist())
            rows["contacts"].append(tr.contacts()["last_step"])
            rows["qpos"].append(tr.read()["qpos"].tolist())

        log()
        print("START", case, flush=True)
        for k, q in enumerate(targets(case, names, reset_pose(manifest))):
            if k % 10 == 0:
                print("  interval", k, np.round(rows["pos"][-1], 4).tolist(), flush=True)
            ack = tr.send_joint_targets(q, joint_names=names, deadline=tr.time + 0.05)
            if ack["status"] != "applied":
                raise RuntimeError(f"{case}: {ack}")
            log()
        record["cases"][case] = rows
        print("CASE", case, np.round(rows["pos"][-1], 4).tolist(), flush=True)
    tr.close()
    (args.output / "isaac_scripted.json").write_text(json.dumps(record, default=str))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's app.close() can end the process before Python prints the traceback.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()
