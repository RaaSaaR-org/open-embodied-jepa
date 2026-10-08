"""Isaac physics and rendering server for the TASK-081 closed loop in Isaac (development only).

Not evidence, not a gated run. Runs INSIDE the ``isaaclab_arena`` container via
``scripts/isaac/run_isaac.sh``. It builds one ``IsaacTransport`` with the ``apple-to-plate-v2``
apple and plate (``configs/isaac/apple_to_plate_v2_scene_v1.json``), the pinned onboard camera
(``configs/isaac/onboard_camera_v1.json``: ``onboard_rgb`` on ``torso_link``, 112 x 112, fovy
75 degrees, the pinned path-traced render settings) and, new here, a world-fixed third-person
camera for the video only (no controller reads it). It serves the e9 server's ``reset`` /
``step`` / ``stop`` protocol (``scripts/isaac/e9_server_isaac.py``) over a Unix socket, plus:

* ``set_plate``: teleport the kinematic plate to a world xyz (the C1-M plate law runs on the host,
  in TASK-081's own ``CorpusMotion``; the host mirror forwards each plate move here before the
  next render or step);
* ``render``: the onboard 112 px RGB frame of the current Isaac state, and with ``third`` also
  the third-person frame, as raw uint8 bytes.

The host side is ``scripts/isaac_lewm_cp_closedloop.py``. Replies carry evaluator-only truth
(apple and plate state, contacts) because the mirror writes Isaac's state into its local MuJoCo
data, as in ``docs/ISAAC_E9_REPLAY.md``; which arm may read what is enforced on the host by
TASK-081's unchanged worker code.

``PXR_WORK_THREAD_LIMIT`` is pinned as in ``e9_server_isaac.py`` (1 for Newton by default).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import secrets
import sys
import time
from pathlib import Path

T0 = time.monotonic()


def stage(message: str) -> None:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[cp-server +{time.monotonic() - T0:8.1f}s {stamp}Z] {message}", flush=True)


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--camera", type=Path, required=True)
parser.add_argument("--physics", choices=("physx", "newton"), default="newton")
parser.add_argument("--third_size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
parser.add_argument("--third_eye", type=float, nargs=3, default=(1.2147, -0.3529, 1.275))
parser.add_argument("--third_target", type=float, nargs=3, default=(0.42, -0.14, 0.80))
parser.add_argument("--third_fovy", type=float, default=45.0)
parser.add_argument("--dump_every", type=float, default=120.0)
parser.add_argument("--idle_timeout", type=float, default=3600.0)
parser.add_argument("--pxr_work_thread_limit", type=int, default=None)
stage("parsing arguments")
_early, _ = parser.parse_known_args()
if _early.pxr_work_thread_limit is not None:
    PXR_PIN = str(_early.pxr_work_thread_limit) if _early.pxr_work_thread_limit > 0 else None
elif "PXR_WORK_THREAD_LIMIT" in os.environ and os.environ["PXR_WORK_THREAD_LIMIT"].isdigit():
    _inherited = os.environ["PXR_WORK_THREAD_LIMIT"]
    PXR_PIN = _inherited if int(_inherited) > 0 else None
else:
    PXR_PIN = "1" if _early.physics == "newton" else None
if PXR_PIN is not None:

    class _PinnedEnviron(type(os.environ)):
        """``os.environ`` whose PXR_WORK_THREAD_LIMIT stays at the pinned value."""

        def __setitem__(self, key, value):
            if key == "PXR_WORK_THREAD_LIMIT" and value != PXR_PIN:
                os.write(2, f"[cp-server] replaced write {key}={value!r} by {PXR_PIN}\n".encode())
                value = PXR_PIN
            super().__setitem__(key, value)

    os.environ.__class__ = _PinnedEnviron
    os.environ["PXR_WORK_THREAD_LIMIT"] = PXR_PIN
stage(f"PXR_WORK_THREAD_LIMIT pinned to {PXR_PIN or '<no pin>'}")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
sys.stdout.reconfigure(line_buffering=True)
faulthandler.dump_traceback_later(args.dump_every, repeat=True)
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
stage("launching the app")
app = AppLauncher(args).app
stage("app launched")

import numpy as np  # noqa: E402

sys.path.insert(0, "/oej/src")
from embodied_jepa.isaac_scene import canonical_sha256, usd_canonical_hash  # noqa: E402
from embodied_jepa.isaac_transport import IsaacTransport, manifest_sha256  # noqa: E402

stage("imports done")


def look_at_xyzw(eye, target) -> tuple[float, float, float, float]:
    """An OpenGL-convention camera (looks along -Z, +Y up) at ``eye`` looking at ``target``,
    world up +Z, as an (x, y, z, w) quaternion."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.stack([right, up, -fwd], axis=1)  # columns: the camera axes in the world
    w = np.sqrt(max(1e-12, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = (m[2, 1] - m[1, 2]) / (4.0 * w)
    y = (m[0, 2] - m[2, 0]) / (4.0 * w)
    z = (m[1, 0] - m[0, 1]) / (4.0 * w)
    return float(x), float(y), float(z), float(w)


class RenderTransport(IsaacTransport):
    """``IsaacTransport`` plus a world-fixed third-person camera (video only) and a plate
    teleport. The onboard camera, lights and render settings are the transport's own."""

    def _spawn_camera_and_lights(self) -> None:
        super()._spawn_camera_and_lights()
        from isaaclab.sensors.camera import Camera, CameraCfg

        from embodied_jepa import isaac_scene as scene

        su = self._sim_utils
        w, h = (int(v) for v in args.third_size)
        focal, h_ap, v_ap = scene.pinhole_from_fovy(float(args.third_fovy), w, h)
        self.third = Camera(
            CameraCfg(
                prim_path="/World/ThirdPersonCamera",
                update_period=0.0,
                height=h,
                width=w,
                data_types=["rgb"],
                spawn=su.PinholeCameraCfg(
                    focal_length=focal,
                    horizontal_aperture=h_ap,
                    vertical_aperture=v_ap,
                    clipping_range=(0.01, 50.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(v) for v in args.third_eye),
                    rot=look_at_xyzw(args.third_eye, args.third_target),
                    convention="opengl",
                ),
            )
        )

    def render_third(self) -> np.ndarray:
        """The third-person frame of the last pumped render (call after ``render()``)."""
        self.third.update(0.0, force_recompute=True)
        return self._np(self.third.data.output["rgb"][0, ..., :3]).astype(np.uint8)

    def set_plate(self, xyz) -> None:
        self._require_open()
        xyz = np.asarray(xyz, float).reshape(3)
        if not np.isfinite(xyz).all():
            raise ValueError("plate position must be finite")
        self._write_object(self.plate, xyz)
        self.plate.update(0.0)


def state(tr, *, after_step: bool = False) -> dict:
    """The mirror's state message (``e9_server_isaac.state``)."""
    q, qd = tr._state()
    snap = tr.last_substep_start if after_step else None
    pre = (
        None
        if snap is None
        else {
            "q": [float(v) for v in snap["q"]],
            "apple_pos": [float(v) for v in snap["apple_pose_xyzw"][:3]],
            "apple_quat_xyzw": [float(v) for v in snap["apple_pose_xyzw"][3:7]],
        }
    )
    pose = tr._np(tr.apple.data.root_link_pose_w).reshape(-1)
    vel = tr._np(tr.apple.data.root_com_vel_w).reshape(-1)
    plate = tr._np(tr.plate.data.root_link_pose_w).reshape(-1)[:3]
    truth = tr.task_truth()
    contacts = tr.contacts()
    return {
        "time": float(tr.time),
        "q": [float(v) for v in q],
        "qd": [float(v) for v in qd],
        "apple_pos": [float(v) for v in pose[:3]],
        "apple_quat_xyzw": [float(v) for v in pose[3:7]],
        "apple_lin_w": [float(v) for v in vel[:3]],
        "apple_ang_w": [float(v) for v in vel[3:6]],
        "plate_pos": [float(v) for v in plate],
        "hand_contact": bool(truth["hand_contact"]),
        "contacts": [
            {"bodies": list(c["bodies"]), "force_n": float(c["force_n"])}
            for c in contacts["last_step"]
        ],
        "interval_pairs": contacts["interval_pairs"],
        "pre": pre,
    }


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    scene = json.loads(args.scene.read_text())
    camera = json.loads(args.camera.read_text())
    stage(f"building the transport (physics={args.physics})")
    tr = RenderTransport(
        usd_path=args.usd,
        manifest=manifest,
        scene_manifest=scene,
        camera_manifest=camera,
        objects=True,
        render=True,
        record_contacts=True,
        physics=args.physics,
    )
    stage("transport built")
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(max(args.dump_every, 1800.0), repeat=True)
    record = {
        "what": "TASK-081 closed loop in Isaac: development only, not evidence",
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "physics_backend": tr.physics,
        "manifest_sha256": manifest_sha256(manifest),
        "scene_manifest_sha256": canonical_sha256(scene),
        "camera_manifest_sha256": canonical_sha256(camera),
        "usd_canonical_tree_sha256": usd_canonical_hash(args.usd),
        "render_settings": tr.render_settings,
        "object_properties": tr.object_properties,
        "joint_names": list(tr.joint_names),
        "third_person": {
            "size": list(args.third_size),
            "eye": list(args.third_eye),
            "target": list(args.third_target),
            "fovy_deg": args.third_fovy,
        },
        "startup_seconds": time.monotonic() - T0,
        "pxr_work_thread_limit_pin": PXR_PIN,
    }
    (args.output / "server.json").write_text(json.dumps(record, default=str))
    names = list(tr.joint_names)
    from multiprocessing.connection import Listener

    key = secrets.token_bytes(32)
    key_path = args.output / "authkey"
    key_path.write_text(key.hex())
    os.chmod(key_path, 0o644)
    sock = args.output / "cp.sock"
    listener = Listener(str(sock), family="AF_UNIX", authkey=key)
    os.chmod(sock, 0o777)
    stage(f"listening on {sock}")
    conn = listener.accept()
    stage("client connected")
    timing = {"reset": [], "step": [], "stop": [], "set_plate": [], "render": []}
    try:
        while True:
            if not conn.poll(args.idle_timeout):
                stage("idle timeout; exiting")
                break
            msg = conn.recv()
            cmd = msg.get("cmd")
            started = time.monotonic()
            try:
                if cmd == "hello":
                    reply = {"joint_names": names, "physics": tr.physics, "server": record}
                elif cmd == "reset":
                    tr.reset(
                        int(msg["seed"]),
                        object_xy=msg["object_xy"],
                        plate_xy=msg["plate_xy"],
                        object_on_container=bool(msg["object_on_container"]),
                        joint_positions=msg["joint_positions"],
                    )
                    reply = {"state": state(tr)}
                elif cmd == "step":
                    ack = tr.send_joint_targets(
                        np.asarray(msg["targets"], float),
                        joint_names=names,
                        deadline=float(msg["deadline"]),
                    )
                    ack = {k: v for k, v in ack.items() if k != "targets"}
                    reply = {"ack": ack, "state": state(tr, after_step=ack["status"] == "applied")}
                elif cmd == "stop":
                    tr.stop(str(msg["reason"]))
                    reply = {"state": state(tr)}
                elif cmd == "set_plate":
                    tr.set_plate(msg["xyz"])
                    plate = tr._np(tr.plate.data.root_link_pose_w).reshape(-1)[:3]
                    reply = {"plate_pos": [float(v) for v in plate]}
                elif cmd == "render":
                    rgb = tr.render()
                    reply = {"rgb": rgb.tobytes(), "shape": list(rgb.shape)}
                    if msg.get("third"):
                        third = tr.render_third()
                        reply |= {"third": third.tobytes(), "third_shape": list(third.shape)}
                elif cmd == "info":
                    reply = {
                        k: {
                            "n": len(v),
                            "median_s": float(np.median(v)) if v else None,
                            "p95_s": float(np.quantile(v, 0.95)) if v else None,
                        }
                        for k, v in timing.items()
                    }
                elif cmd == "close":
                    conn.send({"ok": True})
                    stage("close requested")
                    break
                else:
                    raise ValueError(f"unknown command {cmd!r}")
                if cmd in timing:
                    timing[cmd].append(time.monotonic() - started)
                conn.send({"ok": True, **reply})
            except Exception as error:  # noqa: BLE001 - returned to the client
                import traceback

                conn.send(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        (args.output / "server_timing.json").write_text(json.dumps(timing))
        conn.close()
        listener.close()
        tr.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        stage("closing the app")
        app.close()
