"""Arena side of the e9 cross-simulator check and of the kinematic check (development).

Container-only, via ``scripts/isaac/run_isaac.sh arena_e9_server.py outputs/<new> <args>``.
Development only (docs/ARENA.md §7): not a gated run, not a learned or policy result. No GR00T
server is started or contacted.

The scene is the tutorial's (``galileo_g1_static_pick_and_place``, ``g1_wbc_agile_joint``)
built through the variant class ``arena_layout_env.OejG1StaticEnvironment`` (Arena's own
``--external_environment_class_path``), cameras off, terminations held (recorded, never
ending the episode). Every ``--oej_*`` argument after ``--`` goes to the variant.

``--mode kinematics`` (step 1): drives ``arena_e9.kinematic_trajectory`` (free-space arm and
Dex3 targets around Arena's default pose) and writes ``kinematics.npz`` (per Arena step:
targets, joints, every link pose) and ``kinematics.json`` (names, limits, defaults, link prim
paths, nearby background colliders, feet heights). The host compares it with FK on our MJCF
(``scripts/isaac/arena_e9.py kinematics``).

``--mode serve`` (step 2): serves ``hello``, ``geometry``, ``settle``, ``place``, ``calibrate``,
``step``, ``read``, ``info`` and ``close`` over a Unix socket (``e9.sock`` in the run directory,
key in ``authkey``) to ``arena_e9.ArenaEndpoint`` on the host (``scripts/isaac/arena_e9.py arena``).
Replies carry evaluator-only truth (apple, plate, pelvis, contact forces): e9 and its scorers
read truth by design.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/oej/src")
sys.path.insert(0, "/oej/scripts/isaac")
from embodied_jepa import arena_e9 as ae  # noqa: E402
from embodied_jepa import arena_transport as at  # noqa: E402

PXR_PIN = at.pin_pxr_work_thread_limit()  # before any Kit import (PR #112)
T0 = time.monotonic()
ENV = "oej_g1_static_pick_and_place"
EXTERNAL = "arena_layout_env:OejG1StaticEnvironment"


def stage(msg: str) -> None:
    print(f"[arena-e9 +{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# ----- collision geometry (the shelf-press probe and e9-arena, docs/ARENA.md §9) ------------------
def _prim_points(prim):
    """Local points of one collision prim: mesh vertices, else its extent's 8 corners."""
    from pxr import Usd, UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts:
            return np.asarray(pts, float), "mesh"
    if prim.IsA(UsdGeom.Boundable):
        ext = UsdGeom.Boundable.ComputeExtentFromPlugins(
            UsdGeom.Boundable(prim), Usd.TimeCode.Default()
        )
        if ext:
            lo, hi = np.asarray(ext[0], float), np.asarray(ext[1], float)
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            return lo + corners * (hi - lo), f"extent:{prim.GetTypeName()}"
    return None, prim.GetTypeName()


def collision_points(body_path: str) -> dict:
    """Every collision point of the rigid body at ``body_path``, in the body's own frame.

    Walks the body's subtree (instance proxies included), stops at nested rigid bodies, and maps
    each collision prim's points through the static prim-to-body transform. Read once from USD;
    per-step poses come from the simulator (``body_link_pose_w`` / ``root_link_pose_w``)."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage_ = omni.usd.get_context().get_stage()
    body = stage_.GetPrimAtPath(body_path)
    if not body.IsValid():
        raise ValueError(f"no prim at {body_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    # The simulator's body pose is rigid; a scale authored on the body prim itself (the apple's
    # 0.009) belongs to the points, so the body frame here is its rigid part only.
    to_body = cache.GetLocalToWorldTransform(body).RemoveScaleShear().GetInverse()
    points, kinds = [], {}
    it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
    for prim in it:
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        local, kind = _prim_points(prim)
        kinds[kind] = kinds.get(kind, 0) + 1
        if local is None:
            continue
        m = cache.GetLocalToWorldTransform(prim) * to_body
        points.extend(np.asarray(m.Transform(Gf.Vec3d(*p)), float) for p in local)
    return {"points": np.asarray(points, float).reshape(-1, 3), "kinds": kinds}


class Geometry:
    """Right-hand and apple collision points; per step: the hand's lowest point and the apple's
    top (env-local z). Measurement only: it changes no physics."""

    def __init__(self, scene: at.ArenaScene):
        import re

        self.scene = scene
        names = list(scene.robot.data.body_names)
        root = "/World/envs/env_0/Robot"
        self.links = [
            n for n in names if n.startswith("right_hand_") or n == "right_wrist_yaw_link"
        ]
        self.link_index = [names.index(n) for n in self.links]
        self.link_points = {n: collision_points(f"{root}/{n}") for n in self.links}
        apple_path = scene.apple.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        apple_path = re.sub(r"env_\.\*", "env_0", apple_path)
        self.apple_path = apple_path
        self.apple_points = collision_points(apple_path)

    def summary(self) -> dict:
        return {
            "links": {
                n: {"n_points": len(v["points"]), "kinds": v["kinds"]}
                for n, v in self.link_points.items()
            },
            "apple_path": self.apple_path,
            "apple": {
                "n_points": len(self.apple_points["points"]),
                "kinds": self.apple_points["kinds"],
                "extent_body_m": (
                    np.ptp(self.apple_points["points"], axis=0).tolist()
                    if len(self.apple_points["points"])
                    else None
                ),
            },
        }

    def measure(self) -> dict:
        s = self.scene
        origin = s.np(s.unwrapped.scene.env_origins)[0]
        poses = s.np(s.robot.data.body_link_pose_w)[0]
        low, low_link = np.inf, None
        for n, i in zip(self.links, self.link_index, strict=True):
            pts = self.link_points[n]["points"]
            if not len(pts):
                continue
            r = at.quat_xyzw_to_matrix(poses[i][3:])
            z = float((pts @ r.T)[:, 2].min() + poses[i][2] - origin[2])
            if z < low:
                low, low_link = z, n
        apple = s.np(s.apple.data.root_link_pose_w)[0]
        pts = self.apple_points["points"]
        top = (
            float((pts @ at.quat_xyzw_to_matrix(apple[3:]).T)[:, 2].max() + apple[2] - origin[2])
            if len(pts)
            else None
        )
        return {
            "hand_min_z": None if low_link is None else low,
            "hand_min_link": low_link,
            "apple_top_z": top,
        }


class Server:
    def __init__(self, scene: at.ArenaScene):
        self.scene = scene
        self.timing: dict[str, list[float]] = {}
        self.geometry: Geometry | None = None

    def extras(self) -> dict:
        return self.geometry.measure() if self.geometry is not None else {}

    def record(self) -> dict:
        s = self.scene
        raw = s.raw_state()
        held = s.held_terms
        return {
            **self.extras(),
            "right_hand_net_force_n": raw.get("right_hand_net_force_n"),
            "success": bool(held.get("success", False)),
            "object_dropped": bool(held.get("object_dropped", False)),
            "success_inputs": dict(s.success_inputs) if held.get("success") else None,
            "pelvis_pose": raw["pelvis_pose"].tolist(),
            "apple_com_pose": raw["apple_com_pose"].tolist(),
            "apple_com_vel": raw["apple_com_vel"].tolist(),
            "plate_pose": raw["plate_pose"].tolist(),
            "apple_plate_force_n": raw["apple_plate_force_n"],
            "apple_hand_force_n": raw.get("apple_hand_force_n"),
        }

    def run(self, action, steps: int, *, keep: bool = True) -> list[dict]:
        a = np.asarray(action, np.float32)
        out = []
        for _ in range(int(steps)):
            self.scene.step(a)
            if keep:
                out.append(self.record())
        return out

    def raw(self) -> dict:
        return {
            **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.scene.raw_state().items()
            },
            **self.extras(),
        }

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        s = self.scene
        if cmd == "hello":
            return {
                "joint_names": list(s.joint_names),
                "physics": "physx (Arena galileo_g1_static_pick_and_place variant)",
                "argv": sys.argv,
            }
        if cmd == "geometry":
            if self.geometry is None:
                self.geometry = Geometry(s)
            return {"geometry": self.geometry.summary(), "now": self.geometry.measure()}
        if cmd == "settle":
            s.reset(seed=0)
            self.run(msg["action"], msg["steps"], keep=False)
            return {"raw": self.raw()}
        if cmd == "place":
            s.set_object_pose("plate", msg["plate"])
            s.set_object_pose("apple", msg["apple"])
            records = self.run(msg["action"], msg["steps"])
            return {"raw": self.raw(), "records": records[-1:]}
        if cmd == "calibrate":
            raw = s.raw_state()
            plate = raw["plate_pose"]
            offset = raw["apple_com_pose"][:2] - raw["apple_pose"][:2]
            z = (
                plate[2]
                + msg["plate_height"]
                + msg["clearance"]
                + ae.ARENA_APPLE_ORIGIN_ABOVE_BOTTOM
            )
            s.set_object_pose("apple", [plate[0] - offset[0], plate[1] - offset[1], z])
            records = self.run(msg["action"], msg["steps"])
            return {"raw": self.raw(), "records": records}
        if cmd == "step":
            records = self.run(msg["action"], msg["steps"])
            return {"raw": self.raw(), "records": records}
        if cmd == "read":
            return {"raw": self.raw()}
        if cmd == "info":
            return {
                k: {
                    "n": len(v),
                    "median_s": float(np.median(v)),
                    "p95_s": float(np.quantile(v, 0.95)),
                }
                for k, v in self.timing.items()
                if v
            }
        raise ValueError(f"unknown command {cmd!r}")


def serve(scene: at.ArenaScene, out: str, idle_timeout: float) -> None:
    from multiprocessing.connection import Listener

    server = Server(scene)
    key = secrets.token_bytes(32)
    key_path = os.path.join(out, "authkey")
    with open(key_path, "w") as f:
        f.write(key.hex())
    # The container writes as the host user (run_isaac.sh passes DOCKER_RUN_USER_ID; the run
    # files are owned by uid 1000), so owner-only permissions suffice for the host client.
    os.chmod(key_path, 0o600)
    sock = os.path.join(out, "e9.sock")
    listener = Listener(sock, family="AF_UNIX", authkey=key)
    os.chmod(sock, 0o600)
    stage(f"listening on {sock}")
    conn = listener.accept()
    stage("client connected")
    try:
        while True:
            if not conn.poll(idle_timeout):
                stage("idle timeout; exiting")
                break
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "close":
                conn.send({"ok": True})
                stage("close requested")
                break
            started = time.monotonic()
            try:
                reply = server.handle(msg)
                server.timing.setdefault(cmd, []).append(time.monotonic() - started)
                conn.send({"ok": True, **jsonable(reply)})
            except Exception as error:  # noqa: BLE001 - returned to the client
                conn.send(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        with open(os.path.join(out, "server_timing.json"), "w") as f:
            json.dump(server.timing, f)
        conn.close()
        listener.close()


# ----- kinematics (step 1) -------------------------------------------------------------------
def link_prims(scene) -> dict:
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage_ = omni.usd.get_context().get_stage()
    root = stage_.GetPrimAtPath("/World/envs/env_0/Robot")
    out = {}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            out[prim.GetName()] = str(prim.GetPath())
    return out


def nearby_colliders(region_min, region_max, max_prims=200000, max_seconds=120.0) -> dict:
    """World AABBs of background collision prims that intersect a box around the robot."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage_ = omni.usd.get_context().get_stage()
    root = stage_.GetPrimAtPath("/World/envs/env_0")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render", "proxy"])
    lo, hi = np.asarray(region_min, float), np.asarray(region_max, float)
    hits, seen, started = [], 0, time.monotonic()
    for prim in Usd.PrimRange(root):
        seen += 1
        if seen > max_prims or time.monotonic() - started > max_seconds:
            return {"hits": hits, "truncated_at": seen}
        path = str(prim.GetPath())
        if "/Robot" in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        bmin, bmax = np.array(box.GetMin()), np.array(box.GetMax())
        if np.all(bmax >= lo) and np.all(bmin <= hi):
            hits.append({"path": path, "min": bmin.tolist(), "max": bmax.tolist()})
    return {"hits": hits, "prims_seen": seen}


def kinematics(scene: at.ArenaScene, out: str) -> None:
    r = scene.robot
    names = list(scene.joint_names)
    default = dict(zip(names, scene.np(r.data.default_joint_pos)[0].tolist(), strict=True))
    lim = scene.np(r.data.joint_pos_limits)[0]
    limits = {n: lim[i].tolist() for i, n in enumerate(names)}
    with open("/oej/configs/g1_sim_action.json") as f:
        action_cfg = json.load(f)
    closed = {
        f"{side}_hand_{finger}_joint": v
        for side in ("left", "right")
        for finger, v in zip(
            action_cfg["grasp_fingers"], action_cfg[f"{side}_closed_rad"], strict=True
        )
    }
    segments = ae.kinematic_segments(default, limits, closed)
    traj_names, targets, labels = ae.kinematic_trajectory(default, segments)
    scene.reset(seed=0)
    rows = {
        k: [] for k in ("q", "qd", "q_target", "pelvis_pose", "body_pose", "apple_hand_force_n")
    }
    actions = []
    t_start = time.monotonic()
    for k, row in enumerate(targets):
        action = at.wbc_action(dict(zip(traj_names, row.tolist(), strict=True)), names)
        actions.append(action)
        scene.step(action)
        raw = scene.raw_state(bodies=True)
        for key in rows:
            rows[key].append(raw.get(key, np.nan))
        if k % 200 == 0:
            stage(f"kinematics step {k}/{len(targets)}")
    stage(f"kinematics: {len(targets)} steps in {time.monotonic() - t_start:.1f} s")
    body_names = list(r.data.body_names)
    np.savez_compressed(
        os.path.join(out, "kinematics.npz"),
        action=np.asarray(actions),
        **{k: np.asarray(v, float) for k, v in rows.items()},
    )
    pelvis = np.asarray(rows["pelvis_pose"][-1])
    facts = {
        "arena_ref": at.ARENA_REF,
        "argv": sys.argv,
        "pxr_pin": PXR_PIN,
        "joint_names": names,
        "body_names": body_names,
        "default_joint_pos": default,
        "joint_pos_limits": limits,
        "segments": segments,
        "labels": labels,
        "settled_pelvis_pose": pelvis.tolist(),
        "env_origin": scene.np(scene.unwrapped.scene.env_origins)[0].tolist(),
        "step_dt": float(scene.unwrapped.step_dt),
    }
    for name, fn in (
        ("link_prims", lambda: link_prims(scene)),
        (
            "nearby_colliders",
            lambda: nearby_colliders(
                [pelvis[0] - 0.6, pelvis[1] - 0.8, pelvis[2] - 0.9],
                [pelvis[0] + 0.7, pelvis[1] + 0.8, pelvis[2] + 0.4],
            ),
        ),
    ):
        t0 = time.monotonic()
        try:
            facts[name] = fn()
        except Exception as exc:  # noqa: BLE001 - diagnostic: record and continue
            facts[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        stage(f"{name} ({time.monotonic() - t0:.1f} s)")
    feet = [body_names.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    facts["ankle_roll_link_z_last"] = [float(rows["body_pose"][-1][i][2]) for i in feet]
    with open(os.path.join(out, "kinematics.json"), "w") as f:
        json.dump(jsonable(facts), f, indent=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("kinematics", "serve"), required=True)
    p.add_argument("--idle_timeout", type=float, default=1800.0)
    own, rest = p.parse_known_args()
    rest = [a for a in rest if a != "--"]
    os.makedirs(own.output, exist_ok=False)
    stage(f"building the Arena scene: {ENV} {rest}")
    scene = at.ArenaScene(
        rest,
        enable_cameras=False,
        environment=ENV,
        external_environment=EXTERNAL,
        hold_terminations=True,
    )
    stage("scene built")
    sys.stdout.reconfigure(line_buffering=True)
    if own.mode == "kinematics":
        kinematics(scene, own.output)
    else:
        serve(scene, own.output, own.idle_timeout)
    stage("done")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit shutdown can hang (as arena_probe.py)
