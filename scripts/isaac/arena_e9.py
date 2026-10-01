"""Host side of the Arena checks: kinematic agreement (step 1) and e9 in Arena (step 2).

Development only (docs/ARENA.md §7): not a gated run, not a learned or policy result. e9 is
TASK-070's privileged scripted expert; development seeds only (``isaac_e9.check_seeds``).

    # step 1: Arena run (container, see arena_e9_server.py --mode kinematics), then
    uv run --no-sync python scripts/isaac/arena_e9.py kinematics \\
        --run outputs/<kin>/run --output outputs/<kin>/compare
    # step 2: MuJoCo reference on the same seeds (scripts/isaac/e9_replay.py mujoco), the Arena
    # server (arena_e9_server.py --mode serve), then the client:
    uv run --no-sync python scripts/isaac/arena_e9.py arena --socket outputs/<srv>/run/e9.sock \\
        --reference outputs/<ref> --output outputs/<new>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import arena_e9 as ae  # noqa: E402
from embodied_jepa import isaac_e9 as ie  # noqa: E402


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    return {
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "version": ae.VERSION,
        "expert": ie.E9,
    }


def write_report(path: Path, report: dict) -> None:
    payload = json.dumps(report, indent=1, sort_keys=True, default=float) + "\n"
    path.write_text(payload)
    print(path, "sha256", hashlib.sha256(payload.encode()).hexdigest())


def load_e9_replay():
    spec = importlib.util.spec_from_file_location("_e9_replay", ROOT / "scripts/isaac/e9_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- step 1: kinematics ----------------------------------------------------------------------
def stats(values) -> dict:
    v = np.asarray(values, float)
    return {
        "max": float(v.max()),
        "p95": float(np.quantile(v, 0.95)),
        "median": float(np.median(v)),
    }


def cmd_kinematics(args) -> int:
    from embodied_jepa.simulation import MuJoCoSimulation

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    facts = json.loads((args.run / "kinematics.json").read_text())
    data = np.load(args.run / "kinematics.npz")
    arena_names = facts["joint_names"]
    bodies = facts["body_names"]
    sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
    m = sim.model
    mirror_names = list(sim.joint_names)
    report: dict = {"what": "Arena vs MJCF kinematic check (development)", **provenance()}
    report["run"] = str(args.run)
    # names, order, limits
    report["joint_names_equal_as_sets"] = sorted(arena_names) == sorted(mirror_names)
    report["joint_order_equal"] = arena_names == mirror_names
    report["joint_order_first_difference"] = next(
        (i for i, (a, b) in enumerate(zip(arena_names, mirror_names, strict=True)) if a != b), None
    )
    limit_mismatch = {}
    for i, n in enumerate(mirror_names):
        lo, hi = (float(v) for v in m.jnt_range[sim.joint_ids[i]])
        alo, ahi = facts["joint_pos_limits"][n]
        if abs(lo - alo) > 1e-3 or abs(hi - ahi) > 1e-3:
            limit_mismatch[n] = {"mjcf": [lo, hi], "arena": [alo, ahi]}
    report["joint_limit_mismatches"] = limit_mismatch
    # FK at Arena's measured joints, relative to the pelvis, every step
    q = data["q"]
    poses = data["body_pose"]
    pelvis_i = bodies.index("pelvis")
    common = [b for b in bodies if b in {m.body(i).name for i in range(m.nbody)} and b != "pelvis"]
    scratch = sim.mj.MjData(m)
    labels = facts["labels"]
    per_body = {b: {"pos": [], "rot": []} for b in common}
    local = {b: {"pos": [], "rot": []} for b in common}
    ee = {s: [] for s in ("left", "right")}
    palm = {s: [] for s in ("left", "right")}
    parent = {b: m.body(int(m.body_parentid[m.body(b).id])).name for b in common}
    for k in range(len(q)):
        qn = dict(zip(arena_names, q[k].tolist(), strict=True))
        fk = ae.mujoco_fk(m, scratch, sim.qadr, mirror_names, qn, ["pelvis", *common])
        arena = {
            b: ae.relative_pose(poses[k][bodies.index(b)], poses[k][pelvis_i])
            for b in ["pelvis", *common]
        }
        for b in common:
            dp, dr = ae.pose_difference(fk[b], arena[b])
            per_body[b]["pos"].append(dp)
            per_body[b]["rot"].append(dr)
            pa, pf = arena[parent[b]], fk[parent[b]]
            la = (pa[1].T @ (arena[b][0] - pa[0]), pa[1].T @ arena[b][1])
            lf = (pf[1].T @ (fk[b][0] - pf[0]), pf[1].T @ fk[b][1])
            dp, dr = ae.pose_difference(lf, la)
            local[b]["pos"].append(dp)
            local[b]["rot"].append(dr)
        for side in ("left", "right"):
            w = f"{side}_wrist_yaw_link"
            site = scratch.site(f"{side}_ee")
            rp = scratch.body("pelvis").xmat.reshape(3, 3)
            ee_fk = rp.T @ (site.xpos - scratch.body("pelvis").xpos)
            ee_arena = arena[w][0] + arena[w][1] @ np.array([0.12, 0.0, 0.0])
            ee[side].append(float(np.linalg.norm(ee_fk - ee_arena)))
            p = f"{side}_hand_palm_link"
            if p in bodies:
                pp = ae.relative_pose(poses[k][bodies.index(p)], poses[k][pelvis_i])
                palm[side].append(arena[w][1].T @ (pp[0] - arena[w][0]))
    report["body_pose_difference_pelvis_frame"] = {
        b: {"pos_m": stats(v["pos"]), "rot_rad": stats(v["rot"])} for b, v in per_body.items()
    }
    report["link_local_difference"] = {
        b: {"parent": parent[b], "pos_m": stats(v["pos"]), "rot_rad": stats(v["rot"])}
        for b, v in local.items()
    }
    report["ee_site_difference_m"] = {s: stats(v) for s, v in ee.items()}
    report["arena_palm_origin_in_wrist_yaw_frame"] = {
        s: {"mean": np.mean(v, axis=0).tolist(), "spread": float(np.ptp(v, axis=0).max())}
        for s, v in palm.items()
        if v
    }
    report["mjcf_ee_site_in_wrist_yaw_frame"] = [0.12, 0.0, 0.0]
    # tracking: commanded target vs measured joint, at the end of every hold
    action = data["action"][:, :43]
    q_target = data["q_target"]
    hold_end = [
        k
        for k in range(len(labels))
        if labels[k].endswith(":hold") and (k + 1 == len(labels) or labels[k + 1] != labels[k])
    ]
    track = np.abs(action[hold_end] - q[hold_end])
    upper = [
        i
        for i, n in enumerate(arena_names)
        if any(s in n for s in ("shoulder", "elbow", "wrist", "hand_"))
    ]
    report["tracking_at_hold_end_rad"] = {
        arena_names[i]: {"max": float(track[:, i].max()), "median": float(np.median(track[:, i]))}
        for i in upper
    }
    report["wbc_passes_upper_targets_max_abs_rad"] = float(
        np.abs(q_target[:, upper] - action[:, upper]).max()
    )
    pel = data["pelvis_pose"]
    report["pelvis"] = {
        "first": pel[0].tolist(),
        "last": pel[-1].tolist(),
        "xyz_range_m": np.ptp(pel[:, :3], axis=0).tolist(),
        "max_tilt_rad": float(
            max(ae.rotation_angle(ae.at.quat_xyzw_to_matrix(p[3:])) for p in pel)
        ),
    }
    hand = data["apple_hand_force_n"]
    report["apple_hand_sensor"] = {
        "present": bool(np.isfinite(hand).any()),
        "max_force_n": float(np.nanmax(hand)) if np.isfinite(hand).any() else None,
    }
    for key in ("link_prims", "nearby_colliders", "ankle_roll_link_z_last", "settled_pelvis_pose"):
        report[key] = facts.get(key)
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    worst = sorted(per_body.items(), key=lambda kv: -max(kv[1]["pos"]))[:6]
    print(
        json.dumps(
            {
                "order_equal": report["joint_order_equal"],
                "limit_mismatches": len(limit_mismatch),
                "worst_body_pos_mm": {b: round(1000 * max(v["pos"]), 3) for b, v in worst},
                "wrist_yaw_pos_mm": {
                    s: round(1000 * max(per_body[f"{s}_wrist_yaw_link"]["pos"]), 3)
                    for s in ("left", "right")
                },
                "wrist_yaw_rot_deg": {
                    s: round(np.degrees(max(per_body[f"{s}_wrist_yaw_link"]["rot"])), 4)
                    for s in ("left", "right")
                },
                "ee_mm": {s: round(1000 * max(v), 3) for s, v in ee.items()},
                "palm": report["arena_palm_origin_in_wrist_yaw_frame"],
                "wbc_passthrough": report["wbc_passes_upper_targets_max_abs_rad"],
                "pelvis": report["pelvis"],
                "hand_sensor": report["apple_hand_sensor"],
            },
            indent=1,
            default=float,
        )
    )
    return 0


# ----- step 2: e9 in Arena -----------------------------------------------------------------------
class Connection:
    def __init__(self, path: Path, timeout_s: float):
        from multiprocessing.connection import Client

        deadline = time.monotonic() + timeout_s
        key = path.parent / "authkey"
        while not (path.exists() and key.exists()):
            if time.monotonic() > deadline:
                raise SystemExit(f"no server socket at {path} after {timeout_s:.0f} s")
            time.sleep(2.0)
        self.conn = Client(str(path), family="AF_UNIX", authkey=bytes.fromhex(key.read_text()))

    def call(self, cmd: str, **kwargs) -> dict:
        self.conn.send({"cmd": cmd, **kwargs})
        if cmd == "close":
            return self.conn.recv()
        reply = self.conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"server {cmd}: {reply.get('error')}\n{reply.get('traceback')}")
        return reply


def records_arrays(records) -> dict:
    keys = ("pelvis_pose", "apple_com_pose", "apple_com_vel", "plate_pose")
    out = {f"arena_{k}": np.asarray([r[k] for r in records], float) for k in keys}
    for k in ("success", "object_dropped"):
        out[f"arena_{k}"] = np.asarray([bool(r[k]) for r in records])
    for k in ("apple_plate_force_n", "apple_hand_force_n"):
        out[f"arena_{k}"] = np.asarray([np.nan if r[k] is None else r[k] for r in records], float)
    out["arena_command"] = np.asarray([r["command"] for r in records], int)
    return out


def calibrate_rest_height(conn, endpoint, steps: int, plate_height: float, clearance: float):
    """Drop the apple on the plate (after a reset) and read its resting height above the
    plate origin; the Arena-side ``apple_at_rest_v0`` uses it as the expected height."""
    reply = conn.call(
        "calibrate",
        action=endpoint._hold.tolist(),
        steps=steps,
        plate_height=plate_height,
        clearance=clearance,
    )
    rec = reply["records"]
    tail = rec[-20:]
    heights = [r["apple_com_pose"][2] - r["plate_pose"][2] for r in tail]
    speeds = [float(np.linalg.norm(r["apple_com_vel"][:3])) for r in tail]
    dist = [
        float(np.linalg.norm(np.subtract(r["apple_com_pose"][:2], r["plate_pose"][:2])))
        for r in tail
    ]
    return {
        "rest_height_m": float(np.mean(heights)),
        "rest_height_spread_m": float(np.ptp(heights)),
        "max_speed_last20_m_s": float(max(speeds)),
        "distance_from_plate_origin_m": float(dist[-1]),
        "arena_success_fired": bool(any(r["success"] for r in rec)),
        "steps": steps,
    }


def cmd_arena(args) -> int:
    from embodied_jepa import first_policy_runtime as rt

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    prov = provenance()
    e9r = load_e9_replay()
    ref = json.loads((args.reference / "report.json").read_text())
    tasks = [r for r in ref["attempts"] if r["ok"] and r["plate_cm"] in args.levels]
    if args.seeds:
        tasks = [r for r in tasks if r["seed"] in set(args.seeds)]
    ie.check_seeds(sorted({t["seed"] for t in tasks}))
    args.output.mkdir(parents=True)
    conn = Connection(args.socket, args.connect_timeout)
    started = time.monotonic()
    from embodied_jepa.simulation import MuJoCoSimulation

    endpoint = robot = scene = calibration = timing = None
    rows = []
    try:
        names_sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
        mirror_names = list(names_sim.joint_names)  # MJCF actuator order
        names_sim.close()
        endpoint = ae.ArenaEndpoint(
            conn.call, mirror_names, settle_steps=args.settle_steps, place_steps=args.place_steps
        )
        robot, scene = ae.make_arena_mirror_robot(endpoint, render=False)
        recorder = ie.StepRecorder(robot)
        bounds = rt.configured_bounds()
        first = tasks[0]
        robot.reset(first["seed"], **first["reset"])
        robot.sim.task_truth()  # sends the Arena reset
        calibration = calibrate_rest_height(
            conn, endpoint, args.calibrate_steps, args.plate_height, args.clearance
        )
        print("calibration", calibration, flush=True)
        for task in tasks:
            t0 = time.monotonic()
            row = {k: task[k] for k in ("key", "seed", "plate_cm", "reset", "plate_offset")}
            row["mujoco"] = {
                k: task["closed"].get(k)
                for k in ("at_rest", "latched_success", "final_distance_cm", "stop_reason")
            }
            try:
                summary, arrays = e9r.closed_loop(robot, recorder, bounds, task)
                verdicts = ae.arena_verdicts(
                    endpoint.records, rest_height_m=calibration["rest_height_m"]
                )
                pel = np.asarray([r["pelvis_pose"] for r in endpoint.records], float)
                settled = np.asarray(endpoint.reset_info["settled_pelvis_pose"], float)
                summary.update(
                    {
                        "seconds": time.monotonic() - t0,
                        "layout_error_m": robot.sim.layout_error_m,
                        "reset_info": {
                            k: v for k, v in endpoint.reset_info.items() if k != "place_records"
                        },
                        "pelvis_drift_max_m": float(
                            np.linalg.norm(pel[:, :3] - settled[:3], axis=1).max()
                        ),
                        "pelvis_tilt_max_rad": float(
                            max(ae.rotation_angle(ae.at.quat_xyzw_to_matrix(p[3:])) for p in pel)
                        ),
                        **verdicts,
                    }
                )
                np.savez_compressed(
                    args.output / f"arena_{task['key']}.npz",
                    **arrays,
                    **records_arrays(endpoint.records),
                )
                row["arena"] = {"ok": True, **summary}
            except Exception as error:  # noqa: BLE001 - recorded, next attempt
                import traceback

                row["arena"] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                robot.stop("error")
            a = row["arena"]
            print(
                task["key"],
                {
                    k: a.get(k)
                    for k in (
                        "at_rest",
                        "at_rest_arena",
                        "arena_success",
                        "final_distance_cm",
                        "stop_reason",
                        "error",
                    )
                },
                "mujoco",
                row["mujoco"],
                f"{time.monotonic() - t0:.1f}s",
                flush=True,
            )
            rows.append(row)
        timing = conn.call("info")
    finally:
        try:
            conn.call("close")
        except Exception:  # noqa: BLE001
            pass
    ok = [r for r in rows if r["arena"].get("ok")]
    report = {
        "what": "e9 in Isaac Lab-Arena (our v2 layout variant) through the mirror adapter "
        "(development; privileged scripted expert; not a gated run; not a learned result)",
        **prov,
        "reference": str(args.reference),
        "reference_revision": ref.get("revision"),
        "server": str(args.socket),
        "hello": endpoint.hello if endpoint else None,
        "mirror_scene": scene,
        "calibration": calibration,
        "settle_steps": args.settle_steps,
        "place_steps": args.place_steps,
        "server_timing": timing,
        "seconds": time.monotonic() - started,
        "attempts": rows,
        "counts": {
            "attempts": len(rows),
            "errors": len(rows) - len(ok),
            "arena_success": sum(r["arena"]["arena_success"] for r in ok),
            "at_rest_arena": sum(r["arena"]["at_rest_arena"] for r in ok),
            "at_rest_mirror": sum(r["arena"]["at_rest"] for r in ok),
            "mujoco_at_rest": sum(bool(r["mujoco"]["at_rest"]) for r in rows),
            "mujoco_latched": sum(bool(r["mujoco"]["latched_success"]) for r in rows),
        },
    }
    write_report(args.output / "report.json", report)
    print(json.dumps(report["counts"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kinematics")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("arena")
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0])
    p.add_argument("--settle_steps", type=int, default=75)
    p.add_argument("--place_steps", type=int, default=50)
    p.add_argument("--calibrate_steps", type=int, default=100)
    p.add_argument("--plate_height", type=float, default=0.024, help="plate AABB height, m")
    p.add_argument("--clearance", type=float, default=0.01)
    p.add_argument("--connect_timeout", type=float, default=1800.0)
    args = parser.parse_args()
    return {"kinematics": cmd_kinematics, "arena": cmd_arena}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
