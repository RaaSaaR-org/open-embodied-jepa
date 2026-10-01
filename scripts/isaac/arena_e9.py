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
        "e9_arena_version": ae.E9_ARENA_VERSION,
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
    for k in ("hand_min_z", "apple_top_z", "right_hand_net_force_n"):
        if any(k in r for r in records):
            out[f"arena_{k}"] = np.asarray(
                [np.nan if r.get(k) is None else r[k] for r in records], float
            )
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
        "_records": rec,
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
    if args.expert == ae.E9_ARENA:
        if not args.variant:
            raise SystemExit("e9-arena needs at least one --variant (docs/ARENA.md §8)")
        variants = [json.loads(v) for v in args.variant]
        for v in variants:
            if "name" not in v:
                raise SystemExit("every --variant needs a name")
    else:
        if args.variant:
            raise SystemExit("--variant is for e9-arena only")
        variants = [{"name": "e9"}]
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
            conn.call,
            mirror_names,
            settle_steps=args.settle_steps,
            place_steps=args.place_steps,
            zero_leg_velocity=args.zero_leg_velocity,
        )
        robot, scene = ae.make_arena_mirror_robot(
            endpoint, render=False, gravity_offset_targets=args.gravity_offset
        )
        if args.expert == ae.E9_ARENA:
            geometry = conn.call("geometry")
            print("geometry", json.dumps(geometry, default=float)[:2000], flush=True)
        recorder = ie.StepRecorder(robot)
        bounds = rt.configured_bounds()
        first = tasks[0]
        robot.reset(first["seed"], **first["reset"])
        robot.sim.task_truth()  # sends the Arena reset
        calibration = calibrate_rest_height(
            conn, endpoint, args.calibrate_steps, args.plate_height, args.clearance
        )
        cal_records = calibration.pop("_records")
        np.savez_compressed(
            args.output / "calibration.npz",
            **{k: v for k, v in records_arrays([{**r, "command": 0} for r in cal_records]).items()},
        )
        print("calibration", calibration, flush=True)
        for variant, task in [(v, t) for v in variants for t in tasks]:
            t0 = time.monotonic()
            row = {k: task[k] for k in ("key", "seed", "plate_cm", "reset", "plate_offset")}
            row["variant"] = variant["name"]
            row["mujoco"] = {
                k: task["closed"].get(k)
                for k in ("at_rest", "latched_success", "final_distance_cm", "stop_reason")
            }
            try:
                if args.expert == ae.E9_ARENA:
                    summary, arrays, expert_log = adapted_closed_loop(
                        robot, recorder, bounds, task, endpoint, variant
                    )
                else:
                    summary, arrays = e9r.closed_loop(robot, recorder, bounds, task)
                    expert_log = []
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
                        "arena_hand_contact_steps": int(
                            sum((r["apple_hand_force_n"] or 0.0) > 0 for r in endpoint.records)
                        ),
                        "arena_apple_max_rise_m": float(
                            max(r["apple_com_pose"][2] for r in endpoint.records)
                            - endpoint.records[0]["apple_com_pose"][2]
                        ),
                        "leg_speed_max_rad_s": endpoint.leg_speed_max,
                        **grasp_facts(endpoint.records, expert_log),
                    }
                )
                tag = task["key"] if len(variants) == 1 else f"{variant['name']}_{task['key']}"
                np.savez_compressed(
                    args.output / f"arena_{tag}.npz",
                    **arrays,
                    **records_arrays(endpoint.records),
                    arena_target_offsets=np.asarray(endpoint.offsets, float),
                    expert_phase=np.asarray([e["phase"] for e in expert_log]),
                    expert_target_base=np.asarray(
                        [e["target_base"] for e in expert_log], float
                    ).reshape(-1, 3),
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
                variant["name"],
                task["key"],
                {
                    k: a.get(k)
                    for k in (
                        "at_rest",
                        "at_rest_arena",
                        "arena_success",
                        "final_distance_cm",
                        "final_distance_arena_cm",
                        "arena_apple_max_rise_m",
                        "pelvis_drift_max_m",
                        "close_pelvis_step_m",
                        "stop_reason",
                        "error",
                    )
                },
                "placement",
                {
                    k: (a.get("reset_info") or {}).get(k)
                    for k in ("placement_error_xy_m", "placement_apple_hand_force_n")
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
    report = {
        "what": (
            "e9-arena (the Arena-adapted e9 of docs/ARENA.md §8; not e9)"
            if args.expert == ae.E9_ARENA
            else "e9"
        )
        + " in Isaac Lab-Arena (our v2 layout variant) through the mirror adapter "
        "(development; privileged scripted expert; not a gated run; not a learned result)",
        "expert_label": args.expert,
        "variants": variants,
        "zero_leg_velocity": bool(args.zero_leg_velocity),
        **prov,
        "reference": str(args.reference),
        "reference_revision": ref.get("revision"),
        "server": str(args.socket),
        "hello": endpoint.hello if endpoint else None,
        "mirror_scene": scene,
        "calibration": calibration,
        "gravity_offset": bool(args.gravity_offset),
        "settle_steps": args.settle_steps,
        "place_steps": args.place_steps,
        "server_timing": timing,
        "seconds": time.monotonic() - started,
        "attempts": rows,
        "counts": {
            v["name"]: counts([r for r in rows if r["variant"] == v["name"]]) for v in variants
        },
    }
    write_report(args.output / "report.json", report)
    print(json.dumps(report["counts"], indent=1))
    return 0


def counts(rows) -> dict:
    ok = [r for r in rows if r["arena"].get("ok")]
    return {
        "attempts": len(rows),
        "errors": len(rows) - len(ok),
        "arena_success": sum(r["arena"]["arena_success"] for r in ok),
        "at_rest_arena": sum(r["arena"]["at_rest_arena"] for r in ok),
        "at_rest_mirror": sum(r["arena"]["at_rest"] for r in ok),
        "mujoco_at_rest": sum(bool(r["mujoco"]["at_rest"]) for r in rows),
        "mujoco_latched": sum(bool(r["mujoco"]["latched_success"]) for r in rows),
        "placement_hand_contact": sum(
            bool(r["arena"]["reset_info"].get("placement_apple_hand_force_n")) for r in ok
        ),
        "placement_error_over_5mm": sum(
            r["arena"]["reset_info"]["placement_error_xy_m"] > 0.005 for r in ok
        ),
        "apple_hand_contact_any": sum(r["arena"]["arena_hand_contact_steps"] > 0 for r in ok),
        "apple_lifted_2cm": sum(r["arena"]["arena_apple_max_rise_m"] > 0.02 for r in ok),
        "complete": sum(bool(r["arena"].get("complete")) for r in ok),
        "pelvis_drift_over_5cm": sum(r["arena"]["pelvis_drift_max_m"] > 0.05 for r in ok),
    }


def grasp_facts(records, expert_log) -> dict:
    """Per-attempt facts about the grasp: the pelvis step during close, the hand's lowest point
    and the hand-shelf clearance (when the server measures geometry)."""
    out: dict = {}
    if not expert_log:
        return out
    cmd = np.asarray([r["command"] for r in records])
    pel = np.asarray([r["pelvis_pose"][:2] for r in records], float)
    for name in ("descend", "close", "lift"):
        idx = [e["command"] for e in expert_log if e["phase"] == name]
        if not idx:
            continue
        mask = (cmd >= idx[0]) & (cmd <= idx[-1])
        if mask.any():
            k = np.flatnonzero(mask)
            out[f"{name}_pelvis_step_m"] = float(np.linalg.norm(pel[k[-1]] - pel[k[0]]))
            low = [records[i].get("hand_min_z") for i in k]
            if all(v is not None for v in low):
                out[f"{name}_hand_min_above_shelf_m"] = float(min(low) - ae.ARENA_SHELF_TOP_Z)
    return out


def phase_window(records, log, name) -> np.ndarray:
    """Indices of the Arena records (per Arena step) of the expert's phase ``name``."""
    idx = [e["command"] for e in log if e["phase"] == name]
    cmd = np.asarray([r["command"] for r in records])
    if not idx:
        return np.zeros(0, int)
    return np.flatnonzero((cmd >= idx[0]) & (cmd <= idx[-1]))


def probe_facts(records, log) -> dict:
    """The shelf-press probe's measurements of one attempt (docs/ARENA.md §8.1)."""
    pel = np.asarray([r["pelvis_pose"] for r in records], float)
    low = np.asarray([np.nan if r.get("hand_min_z") is None else r["hand_min_z"] for r in records])
    force = np.asarray(
        [
            np.nan if r.get("right_hand_net_force_n") is None else r["right_hand_net_force_n"]
            for r in records
        ]
    )
    orient = [e for e in log if e["phase"] == "orient"][-10:]
    out = {
        "finger_drop_m": float(np.mean([e["ee_world"][2] - e["hand_min_z"] for e in orient])),
        "finger_drop_spread_m": float(np.ptp([e["ee_world"][2] - e["hand_min_z"] for e in orient])),
        "lowest_link_end_orient": orient[-1]["hand_min_link"],
    }
    for name in ("orient", "descend", "close", "hold", "retreat"):
        k = phase_window(records, log, name)
        if not len(k):
            continue
        step = pel[k, :2] - pel[k[0], :2]
        heading = ae.heading(pel[k[0]])[:2]
        out[name] = {
            "pelvis_step_end_m": float(np.linalg.norm(step[-1])),
            "pelvis_step_max_m": float(np.linalg.norm(step, axis=1).max()),
            "pelvis_step_forward_end_m": float(step[-1] @ heading),
            "hand_min_above_shelf_min_m": float(np.nanmin(low[k]) - ae.ARENA_SHELF_TOP_Z),
            "hand_min_above_shelf_end_m": float(low[k[-1]] - ae.ARENA_SHELF_TOP_Z),
            "hand_net_force_max_n": float(np.nanmax(force[k])),
            "hand_net_force_median_n": float(np.nanmedian(force[k])),
            "steps_with_hand_force_over_1n": int(np.nansum(force[k] > 1.0)),
        }
    k = np.r_[phase_window(records, log, "close"), phase_window(records, log, "hold")]
    if len(k):
        step = pel[k, :2] - pel[k[0], :2]
        out["close_hold_pelvis_step_max_m"] = float(np.linalg.norm(step, axis=1).max())
        out["close_hold_pelvis_step_end_m"] = float(np.linalg.norm(step[-1]))
    return out


def cmd_probe(args) -> int:
    """The empty-hand shelf-press probe (docs/ARENA.md §8.1): does the WBC step when the hand
    presses the shelf, against a hover 1 cm above it and a hold at e9's orient height?"""
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.simulation import MuJoCoSimulation

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    for m in args.modes:
        if m not in ae.PROBE_MODES:
            raise SystemExit(f"unknown mode {m}")
    ie.check_seeds([args.seed])
    ref = json.loads((args.reference / "report.json").read_text())
    task = next(
        r for r in ref["attempts"] if r["ok"] and r["seed"] == args.seed and r["plate_cm"] == 0.0
    )
    reset = {**task["reset"], "object_xy": list(args.away_xy)}
    args.output.mkdir(parents=True)
    conn = Connection(args.socket, args.connect_timeout)
    started = time.monotonic()
    rows, geometry = [], None
    try:
        names_sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
        mirror_names = list(names_sim.joint_names)
        names_sim.close()
        endpoint = ae.ArenaEndpoint(
            conn.call,
            mirror_names,
            settle_steps=args.settle_steps,
            place_steps=args.place_steps,
            zero_leg_velocity=True,
        )
        robot, _scene = ae.make_arena_mirror_robot(
            endpoint, render=False, gravity_offset_targets=True
        )
        geometry = conn.call("geometry")
        print("geometry", json.dumps(geometry, default=float)[:3000], flush=True)
        bounds = rt.configured_bounds()
        for i, mode in enumerate(args.modes):
            t0 = time.monotonic()
            holder: dict = {}

            def make(truth, mode=mode, holder=holder):
                settled = endpoint.reset_info["settled_pelvis_pose"]
                lay = ae.layout_in_arena(
                    task["reset"]["object_xy"], task["reset"]["plate_xy"], arena_pelvis_pose=settled
                )
                site = [*lay["apple"][:2], ae.ARENA_SHELF_TOP_Z + args.apple_com_above_shelf]
                holder["site_w"] = site
                holder["p"] = ae.ShelfPressProbe(
                    truth, live=endpoint.live, site_w=site, mode=mode, clearance_m=args.clearance
                )
                return holder["p"]

            row = {"index": i, "mode": mode}
            try:
                summary, arrays = rx.run_attempt(
                    robot,
                    bounds,
                    seed=args.seed,
                    reset=reset,
                    plate_offset=[0.0, 0.0],
                    make_expert=make,
                )
                log = holder["p"].log
                row.update(
                    {
                        "ok": True,
                        "stop_reason": summary["stop_reason"],
                        "site_w": holder["site_w"],
                        "reset_info": {
                            k: v for k, v in endpoint.reset_info.items() if k != "place_records"
                        },
                        "leg_speed_max_rad_s": endpoint.leg_speed_max,
                        "apple_hand_contact_steps": int(
                            sum((r.get("apple_hand_force_n") or 0.0) > 0 for r in endpoint.records)
                        ),
                        **probe_facts(endpoint.records, log),
                        "seconds": time.monotonic() - t0,
                    }
                )
                np.savez_compressed(
                    args.output / f"probe_{i:02d}_{mode}.npz",
                    **{k: v for k, v in arrays.items() if k != "counts"},
                    **records_arrays(endpoint.records),
                    log_phase=np.asarray([e["phase"] for e in log]),
                    log_command=np.asarray([e["command"] for e in log], int),
                    log_ee_world=np.asarray([e["ee_world"] for e in log], float),
                    log_hand_min_z=np.asarray([e["hand_min_z"] for e in log], float),
                    log_target_base=np.asarray([e["target_base"] for e in log], float),
                )
            except Exception as error:  # noqa: BLE001 - recorded, next attempt
                import traceback

                row.update(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                robot.stop("error")
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "index",
                            "mode",
                            "ok",
                            "error",
                            "stop_reason",
                            "finger_drop_m",
                            "close_hold_pelvis_step_max_m",
                            "close_hold_pelvis_step_end_m",
                            "close",
                            "hold",
                        )
                    },
                    default=float,
                ),
                flush=True,
            )
            rows.append(row)
    finally:
        try:
            conn.call("close")
        except Exception:  # noqa: BLE001
            pass
    report = {
        "what": "empty-hand shelf-press probe in Isaac Lab-Arena (development; no apple in "
        "reach; not a gated run; not a learned result)",
        **provenance(),
        "reference": str(args.reference),
        "seed": args.seed,
        "away_xy": args.away_xy,
        "clearance_m": args.clearance,
        "apple_com_above_shelf_m": args.apple_com_above_shelf,
        "geometry": geometry,
        "seconds": time.monotonic() - started,
        "attempts": rows,
    }
    write_report(args.output / "report.json", report)
    return 0


def adapted_closed_loop(robot, recorder, bounds, task, endpoint, variant):
    """One e9-arena attempt (``arena_e9.ArenaAdaptedE9``) through the unchanged harness."""
    from embodied_jepa import resting_expert as rx

    holder: dict = {}
    params = {k: v for k, v in variant.items() if k != "name"}

    def make(truth):
        holder["expert"] = ae.ArenaAdaptedE9(truth, live=endpoint.live, **params)
        return holder["expert"]

    summary, arrays = rx.run_attempt(
        robot,
        bounds,
        seed=task["seed"],
        reset=task["reset"],
        plate_offset=task["plate_offset"],
        make_expert=make,
    )
    rec = recorder.arrays()
    summary["look_steps"] = int(len(rec["rec_time"]) - summary["steps"])
    summary["expert_label"] = ae.E9_ARENA
    return summary, {**arrays, **rec}, holder["expert"].log


def records_from_arrays(d) -> list[dict]:
    """Inverse of ``records_arrays`` (enough for ``arena_e9.arena_verdicts``)."""
    out = []
    for k in range(len(d["arena_command"])):
        hand = float(d["arena_apple_hand_force_n"][k])
        out.append(
            {
                "command": int(d["arena_command"][k]),
                "success": bool(d["arena_success"][k]),
                "object_dropped": bool(d["arena_object_dropped"][k]),
                "success_inputs": None,
                "apple_com_pose": d["arena_apple_com_pose"][k].tolist(),
                "apple_com_vel": d["arena_apple_com_vel"][k].tolist(),
                "plate_pose": d["arena_plate_pose"][k].tolist(),
                "pelvis_pose": d["arena_pelvis_pose"][k].tolist(),
                "apple_plate_force_n": float(d["arena_apple_plate_force_n"][k]),
                "apple_hand_force_n": None if np.isnan(hand) else hand,
            }
        )
    return out


PHASES = (
    "orient",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "steady",
    "open",
    "clear",
    "retreat",
    "settle",
)


def attempt_diagnostics(d) -> dict:
    """What happened in Arena's world frame: base drift, apple motion, contacts, grasp."""
    pel = d["arena_pelvis_pose"][:, :2]
    apple = d["arena_apple_com_pose"][:, :3]
    cmd = d["arena_command"]
    look = int(len(d["rec_time"]) - len(d["phase"]))
    drift = np.linalg.norm(pel - pel[0], axis=1)
    out = {
        "pelvis_drift_max_m": float(drift.max()),
        "pelvis_drift_end_m": float(drift[-1]),
        "apple_world_disp_max_m": float(np.linalg.norm(apple - apple[0], axis=1).max()),
        "apple_rise_max_m": float(apple[:, 2].max() - apple[0, 2]),
        "arena_apple_hand_contact_steps": int(np.nansum(d["arena_apple_hand_force_n"] > 0)),
        "mirror_apple_hand_contact_steps": int(np.sum(d["rec_mirror_hand"])),
    }
    phase_drift = {}
    for i, name in enumerate(PHASES):
        steps = np.flatnonzero(d["phase"] == i) + look
        mask = np.isin(cmd, steps)
        if mask.any():
            k = np.flatnonzero(mask)
            phase_drift[name] = float(np.linalg.norm(pel[k[-1]] - pel[k[0]]))
    out["pelvis_drift_per_phase_m"] = phase_drift
    return out


def cmd_rescore(args) -> int:
    """Recompute the Arena-side verdicts of a finished run with the current ``arena_verdicts``
    (the run's report keeps the verdicts of the code that ran it)."""
    out = args.run / "rescore.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report = json.loads((args.run / "report.json").read_text())
    rest = report["calibration"]["rest_height_m"]
    rows = []
    for a in report["attempts"]:
        if not a["arena"].get("ok"):
            rows.append({"key": a["key"], "ok": False})
            continue
        d = np.load(args.run / f"arena_{a['key']}.npz")
        v = ae.arena_verdicts(records_from_arrays(d), rest_height_m=rest)
        rows.append(
            {
                "key": a["key"],
                "ok": True,
                "complete": a["arena"]["complete"],
                "stop_reason": a["arena"]["stop_reason"],
                "mujoco_at_rest": a["mujoco"]["at_rest"],
                "at_rest_mirror": a["arena"]["at_rest"],
                **{
                    k: v[k]
                    for k in (
                        "arena_success",
                        "arena_first_success_command",
                        "arena_success_with_hand_contact",
                        "at_rest_arena",
                        "at_rest_arena_reported_vel",
                        "final_distance_arena_cm",
                        "dropped",
                    )
                },
                "at_rest_arena_detail": v["at_rest_arena_detail"],
                **attempt_diagnostics(d),
            }
        )
    ok = [r for r in rows if r["ok"]]
    counts = {
        k: sum(bool(r[k]) for r in ok)
        for k in (
            "mujoco_at_rest",
            "arena_success",
            "at_rest_arena",
            "at_rest_arena_reported_vel",
            "at_rest_mirror",
            "complete",
        )
    }
    counts["apple_lifted_2cm"] = sum(r["apple_rise_max_m"] > 0.02 for r in ok)
    counts["arena_hand_contact_any"] = sum(r["arena_apple_hand_contact_steps"] > 0 for r in ok)
    counts["pelvis_drift_over_5cm"] = sum(r["pelvis_drift_max_m"] > 0.05 for r in ok)
    counts["attempts"] = len(rows)
    write_report(out, {**provenance(), "run": str(args.run), "counts": counts, "attempts": rows})
    print(json.dumps(counts, indent=1))
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
    p.add_argument("--calibrate_steps", type=int, default=150)
    p.add_argument("--plate_height", type=float, default=0.024, help="plate AABB height, m")
    p.add_argument("--clearance", type=float, default=0.01)
    p.add_argument("--connect_timeout", type=float, default=1800.0)
    p.add_argument(
        "--gravity_offset",
        action="store_true",
        help="add qfrc_bias / kp_arena to each arm and hand target (arena_e9.gravity_offset)",
    )
    p.add_argument("--expert", choices=("e9", ae.E9_ARENA), default="e9")
    p.add_argument(
        "--variant",
        action="append",
        help='e9-arena parameters as JSON, e.g. {"name": "T1", "stop_height_m": 0.07}; '
        "repeatable (each variant runs every seed, variant by variant)",
    )
    p.add_argument(
        "--zero_leg_velocity",
        action="store_true",
        help="write the WBC's leg velocities as 0 in the mirror (e9-arena; docs/ARENA.md §8)",
    )
    p = sub.add_parser("probe")
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=50200)
    p.add_argument("--modes", nargs="+", default=["hover", "press", "high"] * 2)
    p.add_argument("--away_xy", type=float, nargs=2, default=[0.60, 0.30])
    p.add_argument("--clearance", type=float, default=0.01)
    p.add_argument("--apple_com_above_shelf", type=float, default=0.0263)
    p.add_argument("--settle_steps", type=int, default=75)
    p.add_argument("--place_steps", type=int, default=50)
    p.add_argument("--connect_timeout", type=float, default=1800.0)
    p = sub.add_parser("rescore")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    return {
        "kinematics": cmd_kinematics,
        "arena": cmd_arena,
        "probe": cmd_probe,
        "rescore": cmd_rescore,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
