"""Isaac physics server for the e9 cross-simulator replay (development, TASK-025).

Runs INSIDE the ``isaaclab_arena`` container via ``scripts/isaac/run_isaac.sh``. It builds one
``IsaacTransport`` (the v2 apple and plate, no rendering, Newton by default) and serves
``reset`` / ``step`` / ``stop`` requests over a Unix socket in the run directory
(``e9.sock``, key in ``authkey``). The host side, ``scripts/isaac/e9_replay.py isaac``, runs
the unchanged e9 harness on a ``isaac_e9.MirrorSimulation`` against it. The replies carry
evaluator-only truth (apple and plate state, contacts) because the privileged expert and its
scorer read truth by design; nothing here is a learned or policy result.

Every start-up stage is logged with its elapsed time, and every thread's stack is dumped every
``--dump_every`` seconds (``faulthandler``), to diagnose the intermittent start-up hang of
``docs/ISAAC_NEWTON_SPIKE.md`` §6. ``--startup_only`` builds the transport, runs one reset and
three steps, and exits (a start-up probe).

``PXR_WORK_THREAD_LIMIT``: OpenUSD's work pool reads it once, when it starts, so it is set here,
before the app (and with it ``pxr``) is loaded. ``--pxr_work_thread_limit N`` sets it to N
(0 leaves it unset); without the flag an inherited value is kept, and the Newton path defaults
to 1, the workaround Newton documents for the OpenUSD thread-safety bug in
``UsdPhysics.LoadUsdPhysicsFromRange`` (newton#1743, #2216; fixed in OpenUSD 26.05, newer than
this image's USD). The value in effect, ``pxr.Work.GetConcurrencyLimit()`` and the USD version
are logged after the app launches and again right before ``add_usd``.
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
    print(f"[e9-server +{time.monotonic() - T0:8.1f}s {stamp}Z] {message}", flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--physics", choices=("physx", "newton"), default="newton")
parser.add_argument("--startup_only", action="store_true")
parser.add_argument("--dump_every", type=float, default=120.0, help="stack dump period, s")
parser.add_argument("--idle_timeout", type=float, default=1800.0, help="exit after idle, s")
parser.add_argument(
    "--pxr_work_thread_limit",
    type=int,
    default=None,
    help="PXR_WORK_THREAD_LIMIT to set before pxr loads (0: unset; default: inherited, else 1 "
    "for newton)",
)
stage("parsing arguments")
# Before AppLauncher (and pxr) load: the work pool reads PXR_WORK_THREAD_LIMIT when it starts.
_early, _ = parser.parse_known_args()
if _early.pxr_work_thread_limit is not None:
    if _early.pxr_work_thread_limit > 0:
        os.environ["PXR_WORK_THREAD_LIMIT"] = str(_early.pxr_work_thread_limit)
    else:
        os.environ.pop("PXR_WORK_THREAD_LIMIT", None)
elif "PXR_WORK_THREAD_LIMIT" not in os.environ and _early.physics == "newton":
    os.environ["PXR_WORK_THREAD_LIMIT"] = "1"
stage(f"PXR_WORK_THREAD_LIMIT={os.environ.get('PXR_WORK_THREAD_LIMIT', '<unset>')}")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
sys.stdout.reconfigure(line_buffering=True)
faulthandler.dump_traceback_later(args.dump_every, repeat=True)
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
stage("launching the app")
app = AppLauncher(args).app
stage("app launched")


def usd_threads() -> dict:
    """The USD build and work-pool limit actually in effect in this process."""
    import pxr
    from pxr import Usd, UsdPhysics, Work

    return {
        "usd_version": ".".join(str(v) for v in Usd.GetVersion()),
        "pxr_path": str(Path(pxr.__file__).parent),
        "env_PXR_WORK_THREAD_LIMIT": os.environ.get("PXR_WORK_THREAD_LIMIT"),
        "work_concurrency_limit": int(Work.GetConcurrencyLimit()),
        "work_physical_concurrency_limit": int(Work.GetPhysicalConcurrencyLimit()),
        # OpenUSD 26.05 renamed the parser along with the thread-safety fix (OpenUSD PR 4002).
        "has_LoadStageFromPrimRange": hasattr(UsdPhysics, "LoadStageFromPrimRange"),
    }


USD_THREADS = usd_threads()
stage(f"usd threads: {json.dumps(USD_THREADS)}")

import numpy as np  # noqa: E402

sys.path.insert(0, "/oej/src")
from embodied_jepa.isaac_scene import canonical_sha256, usd_canonical_hash  # noqa: E402
from embodied_jepa.isaac_transport import IsaacTransport, manifest_sha256  # noqa: E402

stage("imports done")

if args.physics == "newton":
    from newton import ModelBuilder

    _add_usd = ModelBuilder.add_usd
    ADD_USD_SECONDS: list[float] = []

    def _timed_add_usd(self, *a, **k):
        """Time ``add_usd`` (where the start-up hang sits) and log the limit in effect."""
        stage(f"add_usd: start (work concurrency limit {usd_threads()['work_concurrency_limit']})")
        started = time.monotonic()
        result = _add_usd(self, *a, **k)
        ADD_USD_SECONDS.append(time.monotonic() - started)
        stage(f"add_usd: done in {ADD_USD_SECONDS[-1]:.2f} s")
        return result

    ModelBuilder.add_usd = _timed_add_usd


def state(tr, *, after_step: bool = False) -> dict:
    """The mirror's state message; ``pre`` only right after a step (``_substep_snapshot``)."""
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
    stage(f"building IsaacTransport (physics={args.physics})")
    tr = IsaacTransport(
        usd_path=args.usd,
        manifest=manifest,
        scene_manifest=scene,
        objects=True,
        render=False,
        record_contacts=True,
        physics=args.physics,
    )
    stage("transport built")
    # Start-up is over: keep a sparse dump for a hang while serving (the host waits on us).
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(max(args.dump_every, 1800.0), repeat=True)
    record = {
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "physics_backend": tr.physics,
        "newton_model": tr.newton_model,
        "manifest_sha256": manifest_sha256(manifest),
        "scene_manifest_sha256": canonical_sha256(scene),
        "usd_canonical_tree_sha256": usd_canonical_hash(args.usd),
        "object_properties": tr.object_properties,
        "joint_names": list(tr.joint_names),
        "startup_seconds": time.monotonic() - T0,
        "usd_threads": USD_THREADS,
        "add_usd_seconds": ADD_USD_SECONDS if args.physics == "newton" else None,
    }
    (args.output / "server.json").write_text(json.dumps(record, default=str))
    names = list(tr.joint_names)
    if args.startup_only:
        tr.reset(0)
        first = time.monotonic()
        for _ in range(3):
            tr.send_joint_targets(tr.targets.copy(), joint_names=names, deadline=tr.time + 0.05)
        stage(f"startup probe: 3 steps in {time.monotonic() - first:.1f} s (first includes graph)")
        tr.close()
        return
    from multiprocessing.connection import Listener

    key = secrets.token_bytes(32)
    key_path = args.output / "authkey"
    key_path.write_text(key.hex())
    os.chmod(key_path, 0o644)
    sock = args.output / "e9.sock"
    listener = Listener(str(sock), family="AF_UNIX", authkey=key)
    os.chmod(sock, 0o777)
    stage(f"listening on {sock}")
    conn = listener.accept()
    stage("client connected")
    timing = {"reset": [], "step": [], "stop": []}
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
                    reply = {"joint_names": names, "physics": tr.physics}
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
                elif cmd == "info":
                    reply = {
                        k: {
                            "n": len(v),
                            "median_s": float(np.median(v)) if v else None,
                            "p95_s": float(np.quantile(v, 0.95)) if v else None,
                            "max_s": float(np.max(v)) if v else None,
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
