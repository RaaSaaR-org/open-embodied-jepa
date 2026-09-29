"""MuJoCo side of the scripted apple physics checks, and the comparison (development, TASK-025).

Run on the host after ``scripted_checks_isaac.py``:

    uv run --no-sync python scripts/isaac/scripted_checks_mujoco.py \\
        --isaac outputs/<isaac-run>/run --output outputs/<isaac-run>/compare

It runs every ``scripted_cases.CASES`` case in the v2 MuJoCo scene (``apply_v2_scene``) from
``MuJoCoSimulation.reset`` with the same apple start state and joint targets, logs the apple
state and contacts per 0.05 s interval, and compares resting position, settle time (speed at
most 0.001 m/s from then on, 0.05 s resolution), path length and apple contact pairs with the
Isaac run. Scripted physics checks only: no learned policy, not a manipulation result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embodied_jepa.apple_to_plate_v2 import apply_v2_scene
from embodied_jepa.isaac_scene import canonical_sha256
from embodied_jepa.isaac_transport import joint_manifest_from_mujoco, manifest_sha256, reset_pose
from embodied_jepa.simulation import MuJoCoSimulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_mujoco import scene_contacts  # noqa: E402
from scripted_cases import CASES, summarise, targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def run_case(sim: MuJoCoSimulation, case: str, spec: dict, reset_q) -> dict:
    sim.reset(0, object_xy=list(spec["reset_object_xy"]), plate_xy=list(spec["plate_xy"]))
    joint = sim.model.joint("apple_free")
    qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
    if spec["start_pos"] is not None:
        sim.data.qpos[qa : qa + 7] = [*spec["start_pos"], 1.0, 0.0, 0.0, 0.0]
        # Free joint: linear velocity in world, angular in the body frame (= world here).
        sim.data.qvel[va : va + 6] = [*spec["lin_vel"], *spec["ang_vel"]]
        import mujoco

        mujoco.mj_forward(sim.model, sim.data)
    rows = {"time": [], "pos": [], "lin_vel": [], "ang_vel": [], "contacts": [], "qpos": []}

    def log():
        rows["time"].append(float(sim.data.time))
        rows["pos"].append(sim.data.body("apple").xpos.tolist())
        rows["lin_vel"].append(sim.data.qvel[va : va + 3].tolist())
        rows["ang_vel"].append(sim.data.qvel[va + 3 : va + 6].tolist())
        rows["contacts"].append(scene_contacts(sim))
        rows["qpos"].append(sim.data.qpos[sim.qadr].tolist())

    log()
    names = list(sim.joint_names)
    for q in targets(case, names, reset_q):
        ack = sim.send_joint_targets(q, joint_names=names, deadline=sim.data.time + 0.05)
        if ack["status"] != "applied":
            raise RuntimeError(f"{case}: {ack}")
        log()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "configs/isaac/apple_to_plate_v2_scene_v1.json",
        help="scene manifest the Isaac run used (e.g. `git show 45e4d55:<path>` for runs "
        "made before its descriptive strings were corrected)",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="compare a run that lacks some cases (development subsets); the report is "
        "then flagged partial",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    isaac = json.loads((args.isaac / "isaac_scripted.json").read_text())
    missing = [case for case in CASES if case not in isaac["cases"]]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"the Isaac run lacks cases {missing} (requested: "
            f"{isaac.get('cases_requested', 'not recorded')}); pass --allow_partial to compare "
            "the rest"
        )
    scene = json.loads(args.scene.read_text())
    if isaac["scene_manifest_sha256"] != canonical_sha256(scene):
        raise SystemExit(
            f"the Isaac run's scene manifest differs from the one given by --scene ({args.scene})"
        )
    sim = MuJoCoSimulation(render=False)
    v2 = apply_v2_scene(sim.model)
    manifest = joint_manifest_from_mujoco(sim)
    committed = json.loads((ROOT / "configs/isaac/g1_dex3_joint_manifest_v1.json").read_text())
    if not manifest_sha256(committed) == manifest_sha256(manifest) == isaac["manifest_sha256"]:
        raise SystemExit("the Isaac run's joint manifest differs from the committed/MuJoCo one")
    reset_q = reset_pose(committed)
    backend = isaac.get("physics_backend", "physx")  # runs before the Newton spike: PhysX
    report = {
        "question": f"Do the v2 apple and plate behave alike in Isaac ({backend}) and MuJoCo "
        "under identical scripted starts and joint targets? (no learned policy)",
        "isaac_run": str(args.isaac),
        "physics_backend": backend,
        "cases_requested": isaac.get("cases_requested"),
        "cases_missing": missing,
        "partial": bool(missing),
        "mujoco_v2_scene": v2,
        "cases": {},
    }
    if backend == "physx":
        report["unmatched_in_physx"] = scene["unmatched_in_physx"]
    else:
        from embodied_jepa.isaac_scene import UNMATCHED_IN_NEWTON

        report["unmatched_in_newton"] = UNMATCHED_IN_NEWTON
    traces = {}
    for case, spec in CASES.items():
        if case not in isaac["cases"]:
            continue
        m = run_case(sim, case, spec, reset_q)
        i = isaac["cases"][case]
        traces[case] = {"mujoco": m, "isaac": i}
        sm = summarise(m["time"], m["pos"], m["lin_vel"], m["contacts"])
        si = summarise(i["time"], i["pos"], i["lin_vel"], i["contacts"])
        mp, ip = np.array(m["pos"]), np.array(i["pos"])
        row = {
            "mujoco": sm,
            "isaac": si,
            "final_xy_diff_m": float(np.linalg.norm(mp[-1, :2] - ip[-1, :2])),
            "final_z_diff_m": float(ip[-1, 2] - mp[-1, 2]),
            "max_pos_diff_m": float(np.linalg.norm(mp - ip, axis=1).max()),
            "settle_time_diff_s": None
            if sm["settle_time_s"] is None or si["settle_time_s"] is None
            else si["settle_time_s"] - sm["settle_time_s"],
            "max_joint_diff_rad": float(np.abs(np.array(m["qpos"]) - np.array(i["qpos"])).max()),
        }
        if i.get("plate_pos"):  # the MuJoCo plate is a static body; Isaac's must not drift
            plate = np.asarray(i["plate_pos"], dtype=float)
            row["isaac_plate_max_displacement_m"] = float(
                np.linalg.norm(plate - plate[0], axis=1).max()
            )
        report["cases"][case] = row
        print(
            f"{case:22s} final xy diff {row['final_xy_diff_m'] * 1e3:7.2f} mm, z diff "
            f"{row['final_z_diff_m'] * 1e3:6.2f} mm, settle mj {sm['settle_time_s']} isaac "
            f"{si['settle_time_s']}, path mj {sm['path_length_xy_m']:.3f} isaac "
            f"{si['path_length_xy_m']:.3f} m"
        )
    sim.close()
    args.output.mkdir(parents=True)
    (args.output / "scripted_report.json").write_text(json.dumps(report, indent=2))
    (args.output / "mujoco_scripted.json").write_text(json.dumps(traces, default=str))


if __name__ == "__main__":
    main()
